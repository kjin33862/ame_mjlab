"""Reset event that spawns the robot on a flat patch of its sub-terrain."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")

LATTICE_YAWS = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
"""Headings along the stone lattice's axes: a G1's stance is 0.30 x 0.24 m,
which fits a 0.40 m stone squarely but not on the diagonal."""
SPAWN_POSE = {
  "x": (-0.03, 0.03),
  "y": (-0.03, 0.03),
  "z": (0.01, 0.05),
  "yaw": (-0.15, 0.15),
}
"""Jitter around a patch: a few centimetres, a few degrees around each heading."""


def _sample_ranges(
  ranges: dict[str, tuple[float, float]] | None, n: int, device: torch.device | str
) -> torch.Tensor:
  out = torch.zeros(n, 6, device=device)
  for i, key in enumerate(_SE3_KEYS):
    if ranges and key in ranges:
      low, high = ranges[key]
      out[:, i] = torch.empty(n, device=device).uniform_(low, high)
  return out


def reset_root_state_on_patches(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  patch_name: str = "spawn",
  pose_range: dict[str, tuple[float, float]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  yaw_choices: tuple[float, ...] = (),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Spawn the asset on a random ``patch_name`` flat patch of its sub-terrain
  and make that patch the env's origin.

  mjlab's ``reset_root_state_from_flat_patches`` leaves ``env_origins`` at
  the tile centre, so anything measured from the origin — the curriculum's
  end-point rule, eval frames, probes — would read the tile rather than the
  start.  Here the origin *is* the spawn.  A sub-terrain without patches
  spawns at its origin (mjlab pads the patch arrays with it).  ``yaw_choices``
  picks a base heading per env before the ``pose_range`` yaw jitter, e.g.
  the four axes of a stone lattice.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  terrain = env.scene.terrain
  if terrain is not None and patch_name in terrain.flat_patches:
    patches = terrain.flat_patches[patch_name]  # (rows, cols, K, 3)
    pick = torch.randint(0, patches.shape[2], (n,), device=env.device)
    env.scene.env_origins[env_ids] = patches[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], pick
    ]
  asset: Any = env.scene[asset_cfg.name]
  root = asset.data.default_root_state[env_ids].clone()
  pose = _sample_ranges(pose_range, n, env.device)
  if yaw_choices:
    choices = torch.tensor(yaw_choices, device=env.device)
    pose[:, 5] += choices[torch.randint(0, len(yaw_choices), (n,), device=env.device)]
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
  )
  velocities = root[:, 7:13] + _sample_ranges(velocity_range, n, env.device)
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


__all__ = ["LATTICE_YAWS", "SPAWN_POSE", "reset_root_state_on_patches"]
