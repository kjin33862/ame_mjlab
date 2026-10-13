"""Evaluation-only elevation map with configurable noise.  Not used by any
training or play config.

``elevation_map_eval`` produces the same layout as
:func:`ame.mdp.observations.elevation_map` (static x/y grid, z relative to the
sensor, misses at the clamp floor, clamp ``[-1.2, 0]``) and the same noise
model, applied in the same order, with each magnitude a parameter:

* ``drift_cells`` — per-episode lateral shift of the z content by ``dx, dy``
  cells, each uniform in ``{-d, ..., d}``, edge-clamped (training: ``d = 1``);
* ``height_sigma`` — per-cell Gaussian height noise every step (training: 0.03);
* ``height_offset`` — per-episode offset ``U(-o, o)`` (training: 0.05);
* ``delay_steps`` — refresh the map every ``delay_steps`` policy steps and on
  reset; ``<= 1`` is every step (training has no delay).

The per-episode constants are sampled in :meth:`reset`, which the observation
manager also calls on the initial reset, so they apply from the first
episode (as in training, where they are drawn at ``episode_length_buf == 0``).
The ``train`` preset of ``scripts/evaluate.py`` does not use this term: it
keeps the original ``elevation_map(noise=True)`` so it reproduces training
exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ame.mdp.observations import _Z_CLIP, _grid_xy

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


def drift_gather_index(
  dx: torch.Tensor, dy: torch.Tensor, length: int, width: int
) -> torch.Tensor:
  """``[B, R]`` gather indices that shift a z-map by ``(dx, dy)`` cells per env
  with edge clamping; flat index ``iy * length + ix`` (the training layout)."""
  ix = (torch.arange(length, device=dx.device) + dx[:, None]).clamp(0, length - 1)
  iy = (torch.arange(width, device=dy.device) + dy[:, None]).clamp(0, width - 1)
  return (iy[:, :, None] * length + ix[:, None, :]).reshape(dx.shape[0], -1)


class elevation_map_eval:
  """Elevation map with configurable noise (see the module docstring).

  Params: ``sensor_name``, ``height_sigma``, ``height_offset``,
  ``drift_cells``, ``delay_steps``.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._params = cfg.params
    n, device = env.num_envs, env.device
    self._offset = torch.zeros(n, 1, device=device)
    self._dx = torch.zeros(n, dtype=torch.long, device=device)
    self._dy = torch.zeros(n, dtype=torch.long, device=device)
    self._last_map: torch.Tensor | None = None
    self.reset()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    n = len(env_ids)
    device = self._env.device
    o = float(self._params["height_offset"])
    d = int(self._params["drift_cells"])
    self._offset[env_ids] = (torch.rand(n, 1, device=device) * 2 - 1) * o
    self._dx[env_ids] = torch.randint(-d, d + 1, (n,), device=device)
    self._dy[env_ids] = torch.randint(-d, d + 1, (n,), device=device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sigma: float,
    height_offset: float,
    drift_cells: int,
    delay_steps: int,
  ) -> torch.Tensor:
    # Same transform as ame.mdp.observations.elevation_map.
    data = env.scene[sensor_name].data
    sensor_cfg = env.scene[sensor_name].cfg
    xy = _grid_xy(env, sensor_name)  # [R, 2]
    b = data.distances.shape[0]
    z = data.hit_pos_w[..., 2] - data.pos_w[:, 2].unsqueeze(1)  # [B, R]
    missed = (data.distances < 0) | (
      data.distances >= float(sensor_cfg.max_distance) * 0.999
    )
    z = torch.where(missed, torch.full_like(z, _Z_CLIP[0]), z)

    # Noise in the training order: drift, Gaussian, offset.
    if drift_cells > 0:
      res = float(sensor_cfg.pattern.resolution)
      length = round(float(xy[:, 0].max() - xy[:, 0].min()) / res) + 1
      width = z.shape[1] // length
      z = z.gather(1, drift_gather_index(self._dx, self._dy, length, width))
    if height_sigma > 0:
      z = z + torch.randn_like(z) * height_sigma
    if height_offset > 0:
      z = z + self._offset

    z = z.clamp(*_Z_CLIP)
    current = torch.cat(
      [xy.unsqueeze(0).expand(b, -1, -1), z.unsqueeze(-1)], dim=-1
    ).reshape(b, -1)
    if delay_steps <= 1:
      return current
    # Hold the robot-centric map between refreshes; reset envs have
    # episode_length_buf == 0 and always refresh.
    if self._last_map is None or self._last_map.shape != current.shape:
      self._last_map = current.clone()
    refresh = env.episode_length_buf % delay_steps == 0
    self._last_map[refresh] = current[refresh]
    return self._last_map.clone()


__all__ = ["drift_gather_index", "elevation_map_eval"]
