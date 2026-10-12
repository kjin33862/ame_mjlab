"""AME's elevation-map observation on an mjlab raycast grid.

The reference (``ame_locomotion/mdp/observations.py::elevation_map``) feeds the
policy per-point **3-D coordinates in the yaw-aligned sensor frame**, not bare
heights: x/y carry each token's position for the CNN + attention, z is the
terrain height relative to the sensor frame, clamped to ``[-1.2, 0]``.

On an mjlab yaw-aligned grid raycast the x/y of every point are the *static*
grid offsets by construction (the rays are vertical), so they are computed once
from the pattern; z comes from the live hits.  Misses read as the clamp floor —
a deep unknown, the conservative direction for terrain the robot might fall
into.

The reference's noise model is kept: per-point Gaussian height noise (σ 3 cm)
plus a per-env constant offset resampled on reset (±5 cm), emulating elevation
mapping drift.  Noise is a config param so the play/eval variant is clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_Z_CLIP = (-1.2, 0.0)
_NOISE_STD_M = 0.03
_OFFSET_RANGE_M = 0.05
_DRIFT_CELLS = 1
"""Map drift: the z-map is shifted laterally by up to this many grid cells
(+-5 cm at the 0.05 m grid), constant per episode.  The AME tuning guide:
"Adding drift randomization of maps also enforces precise footholds without
reward shaping" — a policy that cannot trust the map's lateral registration
to the centimeter learns to keep margins from edges instead of stepping on
them.  The x/y coordinate channels stay the static grid (the policy's
*belief*); only the z content moves, which is exactly what real elevation
mapping drift does."""

_GRID_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_OFFSET_KEY = "_ame_elevation_offset"
_DRIFT_KEY = "_ame_elevation_drift"


def _grid_xy(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """``[R, 2]`` static x/y offsets of the sensor's ray grid (sensor frame)."""
  key = (id(env), sensor_name)
  cached = _GRID_CACHE.get(key)
  if cached is None:
    pattern = env.scene[sensor_name].cfg.pattern
    offsets, _ = pattern.generate_rays(mj_model=None, device=env.device)
    cached = offsets[:, :2].to(dtype=torch.float32)
    _GRID_CACHE[key] = cached
  return cached


def _drift_index_table(L: int, W: int, device) -> torch.Tensor:
  """``[(2d+1)^2, R]`` gather indices: combo k shifts the z-map by
  (dx, dy) cells with edge clamping (flat index = iy * L + ix)."""
  d = _DRIFT_CELLS
  ix = torch.arange(L, device=device)
  iy = torch.arange(W, device=device)
  tables = []
  for dy in range(-d, d + 1):
    for dx in range(-d, d + 1):
      sx = (ix + dx).clamp(0, L - 1)
      sy = (iy + dy).clamp(0, W - 1)
      tables.append((sy.unsqueeze(1) * L + sx.unsqueeze(0)).reshape(-1))
  return torch.stack(tables)


def elevation_map(
  env: ManagerBasedRlEnv, sensor_name: str, noise: bool = False
) -> torch.Tensor:
  """``[B, R*3]`` yaw-aligned local scan coordinates, AME layout."""
  data = env.scene[sensor_name].data
  sensor_cfg = env.scene[sensor_name].cfg
  xy = _grid_xy(env, sensor_name)  # [R, 2]
  b = data.distances.shape[0]

  # z of each hit relative to the sensor frame; a miss reads the clamp floor.
  # mjlab marks a miss with distance -1 and collapses its hit point onto the
  # sensor origin (z = 0, the clamp ceiling), so the sign test is the real
  # one; the range test covers a sensor that reports the max distance instead.
  z = data.hit_pos_w[..., 2] - data.pos_w[:, 2].unsqueeze(1)  # [B, R]
  missed = (data.distances < 0) | (
    data.distances >= float(sensor_cfg.max_distance) * 0.999
  )
  z = torch.where(missed, torch.full_like(z, _Z_CLIP[0]), z)

  if noise:
    offset = getattr(env, _OFFSET_KEY, None)
    if offset is None or offset.shape[0] != b:
      offset = torch.zeros(b, 1, device=z.device)
      setattr(env, _OFFSET_KEY, offset)
    drift = getattr(env, _DRIFT_KEY, None)
    n_combo = (2 * _DRIFT_CELLS + 1) ** 2
    if drift is None or drift.shape[0] != b:
      drift = torch.full((b,), n_combo // 2, device=z.device, dtype=torch.long)
      setattr(env, _DRIFT_KEY, drift)
    # Resample the per-env constants at episode start (z bias: reference's
    # ±5 cm; xy drift: tuning-guide map-drift randomization).
    fresh = env.episode_length_buf == 0
    if bool(fresh.any()):
      n = int(fresh.sum())
      offset[fresh] = (torch.rand(n, 1, device=z.device) * 2 - 1) * _OFFSET_RANGE_M
      drift[fresh] = torch.randint(0, n_combo, (n,), device=z.device)
    key = (id(env), sensor_name + "/drift")
    table = _GRID_CACHE.get(key)
    if table is None:
      lgt = round(float(xy[:, 0].max() - xy[:, 0].min()) / 0.05) + 1
      wid = z.shape[1] // lgt
      table = _drift_index_table(lgt, wid, z.device)
      _GRID_CACHE[key] = table
    z = z.gather(1, table[drift])
    z = z + torch.randn_like(z) * _NOISE_STD_M + offset

  z = z.clamp(*_Z_CLIP)
  coords = torch.cat([xy.unsqueeze(0).expand(b, -1, -1), z.unsqueeze(-1)], dim=-1)
  return coords.reshape(b, -1)


__all__ = ["elevation_map"]
