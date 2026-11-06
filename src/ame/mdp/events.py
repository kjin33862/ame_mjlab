"""Reset events: spawn the robot on a flat patch of its sub-terrain, or near
its tile centre."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
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


CENTRE_SPAWN = {
  "offset_radius": 0.5,
  "stone_jitter": 0.03,
  "z": (0.01, 0.05),
  "footprint": 0.14,
  "resolution": 0.02,
}
"""Centre spawn: a uniform point in a ``offset_radius`` disc around the tile
centre; on a stone lattice the stone nearest it, ``stone_jitter`` around its
centre (the patch spawn's jitter); ``z`` above the ground, which is the
highest terrain point within ``footprint`` (a square half-width, about a
G1 stance) of the spawn, looked up on a ``resolution`` grid."""


class reset_root_state_at_centre:
  """Spawn near the tile centre with a uniform heading; the env origin stays
  the tile centre.

  The position is a uniform point in a ``offset_radius`` disc around the
  centre; on a :class:`ame.terrains.BoxStoneLatticeTerrainCfg` tile it is the
  stone nearest that point (``stone_jitter`` around its centre, the stone top
  the ground).  The heading is uniform in ``(-pi, pi)`` on every family.
  Elsewhere the ground is the highest terrain point within ``footprint`` of
  the spawn, so no foot starts inside a bump; that height comes from a grid
  ray-cast once, at construction, against the env's own terrain.

  The env origin is not touched: the curriculum has just set it to the tile
  centre (``terrain_origins[level, type]``), so anything measured from the
  origin reads the tile.  Needs a curriculum board (one family per column,
  difficulty by row), which is what places the lattice's stones.
  """

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
    from ame.terrains import BoxStoneLatticeTerrainCfg

    p = {**CENTRE_SPAWN, **{k: v for k, v in cfg.params.items() if k in CENTRE_SPAWN}}
    self.offset_radius = float(p["offset_radius"])
    self.stone_jitter = float(p["stone_jitter"])
    self.z_range = tuple(p["z"])
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("the centre spawn needs a generated terrain")
    gen = terrain.cfg.terrain_generator
    if gen is None or not gen.curriculum:
      raise ValueError("the centre spawn needs a curriculum board")
    origins = terrain.terrain_origins.cpu().numpy()  # [rows, cols, 3] tile centres
    rows, cols = origins.shape[:2]
    subs = list(gen.sub_terrains.values())
    assert len(subs) == cols, "a curriculum board has one column per family"
    device = terrain.terrain_origins.device

    # Lattice stones in world xy, per tile, padded with +inf.
    lower, upper = gen.difficulty_range
    self.lattice = torch.tensor(
      [isinstance(s, BoxStoneLatticeTerrainCfg) for s in subs], device=device
    )
    tables: dict[tuple[int, int], np.ndarray] = {}
    for c, sub in enumerate(subs):
      if isinstance(sub, BoxStoneLatticeTerrainCfg):
        for r in range(rows):
          difficulty = lower + (upper - lower) * r / max(rows - 1, 1)
          grid, _ = sub.centres(difficulty)
          tables[r, c] = grid - np.asarray(sub.size) / 2 + origins[r, c, :2]
    k = max((len(t) for t in tables.values()), default=1)
    stones = np.full((rows, cols, k, 2), np.inf)
    for (r, c), t in tables.items():
      stones[r, c, : len(t)] = t
    self.stones = torch.tensor(stones, dtype=torch.float32, device=device)

    # Ground height around every tile centre: a ray grid over the disc plus
    # the footprint, then the footprint max.
    res = float(p["resolution"])
    f = round(float(p["footprint"]) / res)
    n_disc = int(np.ceil(self.offset_radius / res))
    half = n_disc + f
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    for j in range(model.njnt):  # move every free body (the robot) out of the way
      if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[model.jnt_qposadr[j] + 2] = -1e3
    mujoco.mj_kinematics(model, data)
    offsets = res * np.arange(-half, half + 1)
    down, gid = np.array([0.0, 0.0, -1.0]), np.zeros(1, dtype=np.int32)
    top = float(origins[..., 2].max()) + 10.0
    heights = np.zeros((rows, cols, 2 * n_disc + 1, 2 * n_disc + 1))
    for r in range(rows):
      for c in range(cols):
        if bool(self.lattice[c]):
          continue
        raw = np.empty((len(offsets), len(offsets)))
        x0, y0 = origins[r, c, :2]
        for i, dx in enumerate(offsets):
          for jj, dy in enumerate(offsets):
            d = mujoco.mj_ray(
              model, data, np.array([x0 + dx, y0 + dy, top]), down, None, 1, -1, gid
            )
            raw[i, jj] = top - d if d >= 0 else -np.inf
        window = np.lib.stride_tricks.sliding_window_view(raw, (2 * f + 1, 2 * f + 1))
        heights[r, c] = window.max(axis=(-2, -1))
    if not np.isfinite(heights).all():
      raise ValueError("a spawn ray missed the terrain")
    self.heights = torch.tensor(heights, dtype=torch.float32, device=device)
    self.resolution, self.n_disc = res, n_disc

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    offset_radius: float | None = None,
    stone_jitter: float | None = None,
    z: tuple[float, float] | None = None,
    footprint: float | None = None,
    resolution: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    # Resolved at construction.
    del offset_radius, stone_jitter, z, footprint, resolution
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    terrain: Any = env.scene.terrain
    level = terrain.terrain_levels[env_ids]
    family = terrain.terrain_types[env_ids]
    centre = env.scene.env_origins[env_ids]
    radius = self.offset_radius * torch.sqrt(torch.rand(n, device=env.device))
    angle = 2 * math.pi * torch.rand(n, device=env.device)
    offset = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
    xy = centre[:, :2] + offset
    # Height-field and stair tiles: the footprint-max ground under the point.
    ij = torch.round(offset / self.resolution).long() + self.n_disc
    ij = ij.clamp(0, self.heights.shape[-1] - 1)
    ground = self.heights[level, family, ij[:, 0], ij[:, 1]]
    # Lattice tiles: the stone nearest the point, the stone top the ground.
    on_lattice = self.lattice[family]
    if bool(on_lattice.any()):
      stones = self.stones[level, family]  # [n, K, 2]
      nearest = torch.linalg.norm(stones - xy[:, None, :], dim=-1).argmin(dim=1)
      stone = stones[torch.arange(n, device=env.device), nearest]
      jitter = (torch.rand(n, 2, device=env.device) * 2 - 1) * self.stone_jitter
      xy = torch.where(on_lattice[:, None], stone + jitter, xy)
      ground = torch.where(on_lattice, centre[:, 2], ground)
    asset: Any = env.scene[asset_cfg.name]
    root = asset.data.default_root_state[env_ids].clone()
    lift = torch.empty(n, device=env.device).uniform_(*self.z_range)
    positions = root[:, 0:3] + torch.stack([xy[:, 0], xy[:, 1], ground + lift], dim=1)
    yaw = (torch.rand(n, device=env.device) * 2 - 1) * math.pi
    zeros = torch.zeros(n, device=env.device)
    orientations = quat_mul(root[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
    asset.write_root_link_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(root[:, 7:13], env_ids=env_ids)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids


__all__ = [
  "CENTRE_SPAWN",
  "LATTICE_YAWS",
  "SPAWN_POSE",
  "reset_root_state_at_centre",
  "reset_root_state_on_patches",
]
