"""One-sided swing-clearance terms and the swing-cost-over-gaps diagnostic.

mjlab's ``foot_clearance`` (``|h - 0.10| * |v_xy|`` per foot, every step) and
``foot_swing_height`` (``(peak / 0.10 - 1)^2`` at touchdown) penalise a foot that is
*higher* than the 0.10 m target as much as one that is lower.  Over a 2 m pit the
foot-height sensor reads the pit (or its miss value), so a swing across a gap pays
for clearance it cannot avoid.  The one-sided variants keep only the "too low" side:

    clearance:    max(0, 0.10 - h) * |v_xy|
    swing height: max(0, 1 - peak / 0.10)^2

``swing_cost_by_gap`` logs (no reward) both variants per swing, split by whether the
swing crossed a gap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import feet_swing_height

from ame.mdp.foothold import (
  FEET_CONTACT_SENSOR,
  FOOT_SITES,
  GAP_DROP,
  SOLE_GRID_NX,
  SOLE_SCAN_SENSOR,
  feet_in_contact,
  sensor_sole_drop,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

SOLE_CENTRE = SOLE_GRID_NX + SOLE_GRID_NX // 2
"""Index of the sole grid's centre point (middle row, middle column)."""
_ROBOT = SceneEntityCfg("robot")
_FEET = SceneEntityCfg("robot", site_names=FOOT_SITES)


def command_active(env, command_name: str | None, threshold: float) -> torch.Tensor:
  """``[B]`` 1.0 where ``|v_xy cmd| + |wz cmd| > threshold`` (mjlab's gate)."""
  if command_name is None:
    return torch.ones(env.num_envs, device=env.device)
  command = env.command_manager.get_command(command_name)
  if command is None:
    return torch.ones(env.num_envs, device=env.device)
  total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (total > threshold).float()


def clearance_cost(
  heights: torch.Tensor, vel_norm: torch.Tensor, target: float, one_sided: bool
) -> torch.Tensor:
  """``[B, F]`` mjlab's per-foot clearance cost, optionally one-sided."""
  err = (
    torch.clamp(target - heights, min=0.0) if one_sided else (heights - target).abs()
  )
  return err * vel_norm


def swing_height_cost(
  peak: torch.Tensor, target: float, one_sided: bool
) -> torch.Tensor:
  """``[B, F]`` mjlab's per-landing swing-height cost, optionally one-sided."""
  err = peak / target - 1.0
  if one_sided:
    err = torch.clamp(err, max=0.0)
  return torch.square(err)


def feet_clearance_one_sided(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """mjlab's ``feet_clearance`` with only the below-target side penalised."""
  asset = env.scene[asset_cfg.name]
  heights = env.scene[height_sensor_name].data.heights
  vel = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  cost = clearance_cost(heights, vel, target_height, one_sided=True).sum(dim=1)
  return cost * command_active(env, command_name, command_threshold)


class feet_swing_height_one_sided(feet_swing_height):
  """mjlab's ``feet_swing_height`` with only the below-target side penalised
  (same peak bookkeeping, same ``Metrics/peak_height_mean`` log)."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact = env.scene[sensor_name]
    heights = env.scene[height_sensor_name].data.heights
    in_air = contact.data.found == 0
    self.peak_heights = torch.where(
      in_air, torch.maximum(self.peak_heights, heights), self.peak_heights
    )
    first = contact.compute_first_contact(dt=self.step_dt)
    active = command_active(env, command_name, command_threshold)
    cost = swing_height_cost(self.peak_heights, target_height, one_sided=True)
    cost = torch.sum(cost * first.float(), dim=1) * active
    landings = torch.sum(first.float())
    env.extras["log"]["Metrics/peak_height_mean"] = torch.sum(
      self.peak_heights * first.float()
    ) / torch.clamp(landings, min=1)
    self.peak_heights = torch.where(
      first, torch.zeros_like(self.peak_heights), self.peak_heights
    )
    return cost


class SwingGapTracker:
  """Per-foot swing bookkeeping (liftoff to touchdown), pure torch.

  Clearance costs are summed over the swing's air steps (raw, unweighted, per policy
  step); the swing-height cost is taken at touchdown from the swing's peak; a swing
  ``crossed`` a gap if the sole centre was over a gap at any air step.  Only swings
  that lifted off from a contact seen since reset are reported (the drop from the
  spawn pose is not a swing)."""

  def __init__(self, num_envs: int, num_feet: int, device, target: float = 0.1):
    z = lambda: torch.zeros(num_envs, num_feet, device=device)  # noqa: E731
    self.target = target
    self.clear2, self.clear1, self.peak = z(), z(), z()
    self.crossed = z().bool()
    self.started = z().bool()

  def reset(self, env_ids=None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    for t in (self.clear2, self.clear1, self.peak):
      t[ids] = 0.0
    self.crossed[ids] = False
    self.started[ids] = False

  def update(
    self,
    contact: torch.Tensor,
    first_contact: torch.Tensor,
    clear2: torch.Tensor,
    clear1: torch.Tensor,
    heights: torch.Tensor,
    over_gap: torch.Tensor,
    active: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Feed one step; returns the swings that landed this step as 1-D tensors
    (``crossed``, ``clear2``, ``clear1``, ``height2``, ``height1``, ``env``)."""
    air = ~contact
    act = active[:, None]
    self.clear2 += clear2 * air * act
    self.clear1 += clear1 * air * act
    self.peak = torch.where(air, torch.maximum(self.peak, heights), self.peak)
    self.crossed |= over_gap & air
    landed = first_contact & self.started
    out = {
      "crossed": self.crossed[landed],
      "clear2": self.clear2[landed],
      "clear1": self.clear1[landed],
      "height2": (swing_height_cost(self.peak, self.target, False) * act)[landed],
      "height1": (swing_height_cost(self.peak, self.target, True) * act)[landed],
      "env": landed.nonzero()[:, 0],
    }
    for t in (self.clear2, self.clear1, self.peak):
      t[first_contact] = 0.0
    self.crossed[first_contact] = False
    self.started |= contact
    return out


def over_gap(sole_drop: torch.Tensor, drop: float = GAP_DROP) -> torch.Tensor:
  """``[B, F]`` the sole centre is over a gap (terrain ``drop`` below it, or none)."""
  return sole_drop[..., SOLE_CENTRE] > drop


class swing_cost_by_gap:
  """Logging-only metric.  Each step, for the swings that landed, writes per-swing
  values to ``extras["log"]`` (the rsl_rl logger concatenates and averages them, so
  each key is a per-swing mean over the iteration):

  ``Swing/clearance_{gap,nogap}``, ``Swing/clearance_1s_{gap,nogap}`` (sum over the
  swing of the per-step clearance cost, two-sided and one-sided; times the
  ``foot_clearance`` weight and dt for reward units), ``Swing/height_*`` and
  ``Swing/height_1s_*`` (swing-height cost at touchdown), ``Swing/gap_share`` (share
  of swings that crossed a gap).  The returned value, averaged per episode by the
  metrics manager, is gap-crossing landings per step."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    num_feet = len(
      env.scene[
        cfg.params.get("contact_sensor_name", FEET_CONTACT_SENSOR)
      ].primary_names
    )
    self.tracker = SwingGapTracker(
      env.num_envs, num_feet, env.device, cfg.params.get("target_height", 0.1)
    )
    self.step_dt = env.step_dt

  def reset(self, env_ids=None) -> None:
    self.tracker.reset(env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    contact_sensor_name: str = FEET_CONTACT_SENSOR,
    height_sensor_name: str = "foot_height_scan",
    sole_sensor_name: str = SOLE_SCAN_SENSOR,
    command_name: str | None = "twist",
    command_threshold: float = 0.05,
    target_height: float = 0.1,
    asset_cfg: SceneEntityCfg = _FEET,
  ) -> torch.Tensor:
    contact_sensor = env.scene[contact_sensor_name]
    heights = env.scene[height_sensor_name].data.heights
    asset = env.scene[asset_cfg.name]
    vel = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    landed = self.tracker.update(
      contact=feet_in_contact(contact_sensor),
      first_contact=contact_sensor.compute_first_contact(dt=self.step_dt),
      clear2=clearance_cost(heights, vel, target_height, one_sided=False),
      clear1=clearance_cost(heights, vel, target_height, one_sided=True),
      heights=heights,
      over_gap=over_gap(sensor_sole_drop(env.scene[sole_sensor_name])),
      active=command_active(env, command_name, command_threshold),
    )
    log_swings(env.extras.setdefault("log", {}), landed)
    per_env = torch.zeros(env.num_envs, device=env.device)
    per_env.index_add_(0, landed["env"], landed["crossed"].float())
    return per_env


def log_swings(log: dict, landed: dict[str, torch.Tensor]) -> None:
  crossed = landed["crossed"]
  if crossed.numel() == 0:
    return
  log["Swing/gap_share"] = crossed.float()
  names = {
    "clear2": "clearance",
    "clear1": "clearance_1s",
    "height2": "height",
    "height1": "height_1s",
  }
  for key, name in names.items():
    for tag, mask in (("gap", crossed), ("nogap", ~crossed)):
      if mask.any():
        log[f"Swing/{name}_{tag}"] = landed[key][mask]


def swing_diagnostics_cfg() -> MetricsTermCfg:
  return MetricsTermCfg(
    func=swing_cost_by_gap,
    params={
      "contact_sensor_name": FEET_CONTACT_SENSOR,
      "height_sensor_name": "foot_height_scan",
      "sole_sensor_name": SOLE_SCAN_SENSOR,
      "command_name": "twist",
      "command_threshold": 0.05,
      "target_height": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
    },
  )


__all__ = [
  "SwingGapTracker",
  "clearance_cost",
  "command_active",
  "feet_clearance_one_sided",
  "feet_swing_height_one_sided",
  "log_swings",
  "over_gap",
  "swing_cost_by_gap",
  "swing_diagnostics_cfg",
  "swing_height_cost",
]
