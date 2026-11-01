"""Success-Guided Sampling (SGS) of terrain levels.

SGS (arXiv 2610.12465, Sec. 3.2 and App. B.2) replaces the promotion ladder:
every (terrain family, level) configuration keeps a ring buffer of its last
``H`` episode outcomes, and an env's next level is drawn from its family's
levels with weight peaking where the success rate ``p`` is near the target
``t``.  An env's family (its terrain column) never changes; SGS only picks the
row.  On the gap board ``pyramid_stairs`` spawns on the summit (stairs down)
and ``pyramid_stairs_inv`` in the pit (stairs up).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ame.mdp.curriculums import terrain_levels_floor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


SGS_PARAMS: dict[str, float] = {
  "H": 100,
  "t": 0.66,
  "kappa": 5.0,
  "T": 2.0,
  "eps": 1e-8,
}
"""Buffer length ``H``, target success rate ``t``, sharpness ``kappa``,
softmax temperature ``T`` and the floor/offset ``eps`` (App. B.2)."""


def sgs_weights(p: torch.Tensor, t: float, kappa: float, eps: float) -> torch.Tensor:
  """``(p+eps)^(kappa t) (1-p+eps)^(kappa (1-t))`` floored at ``eps``; peaks
  at ``p = t``."""
  w = (p + eps) ** (kappa * t) * (1 - p + eps) ** (kappa * (1 - t))
  return w.clamp(min=eps)


def sgs_probs(
  p: torch.Tensor, t: float, kappa: float, T: float, eps: float
) -> torch.Tensor:
  """Sampling probabilities over the last axis: ``softmax(log(w + eps) /
  max(T, 1))``."""
  logit = torch.log(sgs_weights(p, t, kappa, eps) + eps)
  return torch.softmax(logit / max(T, 1.0), dim=-1)


class OutcomeBuffer:
  """Per-configuration ring buffers of the last ``H`` boolean outcomes."""

  def __init__(self, families: int, levels: int, H: int, device: Any) -> None:
    self.families, self.levels, self.H = families, levels, H
    self.outcomes = torch.zeros(families, levels, H, dtype=torch.bool, device=device)
    self.head = torch.zeros(families, levels, dtype=torch.long, device=device)
    self.fill = torch.zeros(families, levels, dtype=torch.long, device=device)

  def record(
    self, family: torch.Tensor, level: torch.Tensor, outcome: torch.Tensor
  ) -> None:
    """Append one outcome per entry; entries sharing a configuration are
    written in order, overwriting the oldest once the buffer is full."""
    if family.numel() == 0:
      return
    cells = self.families * self.levels
    key = family * self.levels + level
    order = torch.argsort(key, stable=True)
    key, outcome = key[order], outcome[order]
    counts = torch.bincount(key, minlength=cells)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(key.numel(), device=key.device) - start[key]
    # More than H at once: only the last H survive, so every slot is unique.
    keep = rank >= counts[key] - self.H
    head = self.head.view(cells)
    slot = (head[key] + rank) % self.H
    self.outcomes.view(cells, self.H)[key[keep], slot[keep]] = outcome[keep]
    head.copy_((head + counts) % self.H)
    self.fill.view(cells).copy_((self.fill.view(cells) + counts).clamp(max=self.H))

  def success_rate(self) -> torch.Tensor:
    """``p`` per configuration, the mean of the filled entries; an empty
    buffer is ``p = 0``.  Unfilled slots are False, so the sum is exact."""
    hits = self.outcomes.sum(dim=-1).float()
    return torch.where(
      self.fill > 0, hits / self.fill.clamp(min=1).float(), torch.zeros_like(hits)
    )


class terrain_levels_sgs(terrain_levels_floor):
  """SGS level selection over the floor term's spawn and success signal.

  Construction is :class:`terrain_levels_floor`'s (rows ``floor`` ..
  ``max_init_terrain_level`` to start).  At each episode end the ended env's
  outcome — the floor term's ``move_up`` under ``rule``, nothing else counts
  as success — is recorded under its (family, level), then its next level is
  drawn from its family's rows ``floor`` .. top by :func:`sgs_probs`.  There is
  no promotion or demotion.  The first call (the construction-pose reset) and
  calls before ``hold_steps`` record and sample nothing.
  """

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.sgs = {**SGS_PARAMS, **cfg.params.get("sgs", {})}
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    self.buffer = OutcomeBuffer(
      families=int(terrain.terrain_origins.shape[1]),
      levels=int(terrain.max_terrain_level),
      H=int(self.sgs["H"]),
      device=terrain.terrain_levels.device,
    )

  def probs(self) -> torch.Tensor:
    """``[families, levels - floor]`` sampling probabilities over rows
    ``floor`` .. top."""
    p = self.buffer.success_rate()[:, self.floor :]
    s = self.sgs
    return sgs_probs(p, s["t"], s["kappa"], s["T"], s["eps"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    floor: int,
    rule: str = "endpoint",
    track_ratio: float = 0.6,
    hold_steps: int = 0,
    sgs: dict[str, float] | None = None,
  ) -> dict[str, Any]:
    del floor, rule, track_ratio, hold_steps, sgs  # resolved at construction
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    generator = terrain.cfg.terrain_generator
    assert generator is not None
    if not self._armed:
      # The construction-pose reset: no outcome to record (see the floor term).
      self._armed = True
      return self._sgs_levels(terrain, generator)
    if int(env.common_step_counter) < self.hold_steps or env_ids.numel() == 0:
      return self._sgs_levels(terrain, generator)
    success, _ = self._judge(env, env_ids, command_name)
    family = terrain.terrain_types[env_ids]
    self.buffer.record(family, terrain.terrain_levels[env_ids], success)
    level = self.floor + torch.multinomial(self.probs()[family], 1).squeeze(1)
    terrain.terrain_levels[env_ids] = level
    terrain.env_origins[env_ids] = terrain.terrain_origins[level, family]
    return self._sgs_levels(terrain, generator)

  def _sgs_levels(self, terrain: Any, generator: Any) -> dict[str, Any]:
    """The floor term's logs plus, per family, ``p`` and the sampling
    probability per level and the smallest fill count."""
    result: dict[str, Any] = self._levels(terrain, generator)
    names = list(generator.sub_terrains.keys())
    if self.buffer.families != len(names):
      return result
    # One host transfer for the whole table rather than one sync per scalar.
    p = self.buffer.success_rate()[:, self.floor :].tolist()
    prob = self.probs().tolist()
    fill_min = self.buffer.fill[:, self.floor :].min(dim=1).values.tolist()
    for f, name in enumerate(names):
      for i in range(len(prob[f])):
        level = self.floor + i
        result[f"{name}/p_{level}"] = p[f][i]
        result[f"{name}/prob_{level}"] = prob[f][i]
      result[f"{name}/fill_min"] = float(fill_min[f])
    return result


__all__ = [
  "SGS_PARAMS",
  "OutcomeBuffer",
  "sgs_probs",
  "sgs_weights",
  "terrain_levels_sgs",
]
