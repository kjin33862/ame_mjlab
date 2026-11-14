"""Can the gap task's actor see the gaps?  CPU probe, no policy, no training.

Builds the play env of Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Velocity-Rough-G1 on one
tile (stepping_stones d0.5 from the AME mix, or the gap board's lattice d0.25),
places the robot kinematically (default joint pose, yaw 0) next to a gap, runs
the sensor pipeline once and dumps:

* the actor's ``height_map`` observation exactly as the policy gets it
  (observation manager output, noise off: the play cfg);
* the same with the training noise (``elevation_map(noise=True)``, the function
  and params of the training cfg), resampled N times;
* the true terrain top under the same 693 ray points (independent mj_ray from
  10 m above, terrain geoms only), unclipped.

Writes PNGs + JSON to notes/gap_perception/.

  uv run python notes/gap_perception_probe.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from mjlab.envs import ManagerBasedRlEnv

from ame.env_cfg import EDGE_DISTANCE, ame_gap_env_cfg
from ame.mdp.observations import _Z_CLIP, elevation_map
from ame.scan import MAP_GROUP, MAP_SHAPE
from ame.terrains import BoxStoneLatticeTerrainCfg, ame_terrains_cfg, gap_board_cfg

OUT = Path(__file__).resolve().parent / "gap_perception"
L, W, _ = MAP_SHAPE
N_NOISE = 500

# Light-surface palette (dataviz reference instance).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list(
  "seq_blue",
  ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
)


def task_kwargs() -> dict:
  return dict(
    rule="progress",
    heading_command=True,
    sgs=True,
    ang_vel_weight=4.0,
    ang_vel_z_only=True,
  )


def describe_cfgs() -> dict:
  """Sensor + map-term settings of the registered train and play cfgs."""
  out = {}
  for name, cfg in (
    (
      "train",
      ame_gap_env_cfg(centre_spawn=True, edge_distance=EDGE_DISTANCE, **task_kwargs()),
    ),
    ("play", ame_gap_env_cfg(play=True, **task_kwargs())),
  ):
    s = next(s for s in cfg.scene.sensors if s.name == "terrain_scan")
    grp = cfg.observations[MAP_GROUP]
    term = grp.terms["elevation"]
    out[name] = {
      "sensor_frame": f"{s.frame.type}:{s.frame.name}",
      "pattern": f"{type(s.pattern).__name__}(size={s.pattern.size}, resolution={s.pattern.resolution})",
      "ray_alignment": s.ray_alignment,
      "max_distance": s.max_distance,
      "include_geom_groups": s.include_geom_groups,
      "exclude_parent_body": s.exclude_parent_body,
      "term_func": term.func.__name__,
      "term_params": {k: v for k, v in term.params.items()},
      "term_scale": term.scale,
      "term_clip": term.clip,
      "term_noise": repr(term.noise),
      "term_delay": getattr(term, "delay_max_lag", None),
      "group_enable_corruption": grp.enable_corruption,
      "group_history_length": grp.history_length,
      "actor_groups_has_map": MAP_GROUP in cfg.observations,
    }
  return out


def make_env(family: str, difficulty: float) -> tuple[ManagerBasedRlEnv, object]:
  cfg = ame_gap_env_cfg(play=True, **task_kwargs())
  cfg.scene.num_envs = 1
  if family == "stone_lattice":
    sub = gap_board_cfg().sub_terrains["stepping_stones"]
    assert isinstance(sub, BoxStoneLatticeTerrainCfg)
    sub = replace(sub, flat_patch_sampling=None)
  else:
    sub = ame_terrains_cfg().sub_terrains[family]
  sub = replace(sub, proportion=1.0)
  gen = cfg.scene.terrain.terrain_generator
  cfg.scene.terrain.terrain_generator = replace(
    gen,
    curriculum=False,
    num_rows=1,
    num_cols=1,
    difficulty_range=(difficulty, difficulty),
    seed=0,
    sub_terrains={family: sub},
  )
  for ev in ("push_robot", "randomize_terrain"):
    cfg.events.pop(ev, None)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  return env, sub


def place(env: ManagerBasedRlEnv, xy_world: np.ndarray, ground_z: float) -> None:
  robot = env.scene["robot"]
  root = robot.data.default_root_state.clone()
  root[:, 0] = float(xy_world[0])
  root[:, 1] = float(xy_world[1])
  root[:, 2] = ground_z + float(robot.data.default_root_state[0, 2])
  root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
  root[:, 7:] = 0.0
  robot.write_root_state_to_sim(root)
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
  )
  env.sim.forward()
  # As in env.step: scene.update invalidates the entity / sensor data caches
  # (mjlab caches sensor data until update or reset), then sense ray-casts.
  env.scene.update(env.physics_dt)
  env.sim.sense()


def terrain_top(
  env: ManagerBasedRlEnv, xy: np.ndarray, start_z: float = 10.0
) -> np.ndarray:
  """True top surface under each (x, y): mj_ray straight down, geom group 0."""
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
  gid = np.zeros(1, dtype=np.int32)
  out = np.empty(len(xy))
  for i, (x, y) in enumerate(xy):
    dist = mujoco.mj_ray(
      m, d, np.array([x, y, start_z]), np.array([0.0, 0, -1]), group, 1, -1, gid
    )
    out[i] = start_z - dist if dist >= 0 else np.nan
  return out


def _robot_bodies(m) -> set[int]:
  pelvis = next(
    b
    for b in range(m.nbody)
    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("pelvis")
  )
  return {b for b in range(m.nbody) if m.body_rootid[b] == m.body_rootid[pelvis]}


def robot_geoms_in_group0(env: ManagerBasedRlEnv) -> int:
  m = env.sim.mj_model
  robot_bodies = _robot_bodies(m)
  return int(
    sum(
      1
      for g in range(m.ngeom)
      if m.geom_group[g] == 0 and m.geom_bodyid[g] in robot_bodies
    )
  )


def tile_geoms(env: ManagerBasedRlEnv) -> dict:
  """Terrain box geoms: top z of the pit floor and of the stones."""
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  tops, floors = [], []
  robot = _robot_bodies(m)
  for g in range(m.ngeom):
    if m.geom_bodyid[g] not in robot and m.geom_group[g] == 0:
      if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
        continue
      z = d.geom_xpos[g, 2]
      top, bottom = z + m.geom_size[g, 2], z - m.geom_size[g, 2]
      half_xy = m.geom_size[g, :2]
      if m.geom_size[g, 2] < 0.1 and half_xy.min() > 1.0:
        floors.append((float(top), float(bottom)))
      else:
        tops.append(
          (float(top), float(bottom), float(2 * half_xy[0]), float(2 * half_xy[1]))
        )
  t = np.array(tops)
  return {
    "pit_floor_boxes_top_bottom": floors,
    "n_column_boxes": len(tops),
    "column_top_z_min_max": [float(t[:, 0].min()), float(t[:, 0].max())],
    "column_bottom_z_min_max": [float(t[:, 1].min()), float(t[:, 1].max())],
    "column_sizes_min_max_m": [float(t[:, 2:].min()), float(t[:, 2:].max())],
  }


def stats(v: np.ndarray) -> dict:
  v = np.asarray(v, dtype=float).ravel()
  if v.size == 0:
    return {"n": 0}
  return {
    "n": int(v.size),
    "mean": round(float(v.mean()), 4),
    "std": round(float(v.std()), 4),
    "min": round(float(v.min()), 4),
    "p01": round(float(np.percentile(v, 1)), 4),
    "p99": round(float(np.percentile(v, 99)), 4),
    "max": round(float(v.max()), 4),
  }


def probe(family: str, difficulty: float, tag: str) -> dict:
  env, sub = make_env(family, difficulty)
  origin = env.scene.terrain.env_origins[0].cpu().numpy()
  if family == "stone_lattice":
    grid, _ = sub.centres(difficulty)
    corner = origin[:2] - np.asarray(sub.size) / 2
    centres = grid + corner
    i = int(np.argmin(np.linalg.norm(centres - origin[:2], axis=1)))
    stone = centres[i]
    # Toes (ankle x + 0.142) about 1 cm short of the stone's front edge.
    pelvis_xy = stone + np.array([sub.stone_size / 2 - 0.16, 0.0])
    ground = 0.0
    where = f"stone centre {np.round(stone - origin[:2], 3).tolist()} m from tile origin, pelvis 0.04 m ahead of it"
  else:
    # Central platform (2 m, top at stone_height); stand at its +x edge.
    ground = float(origin[2])
    pelvis_xy = origin[:2] + np.array([sub.platform_width / 2 - 0.16, 0.0])
    where = "central platform, pelvis 0.84 m ahead of its centre (toes ~1 cm short of its +x edge)"
  place(env, pelvis_xy, ground)

  sensor = env.scene["terrain_scan"]
  data = sensor.data
  pelvis = env.scene["robot"].data.root_link_pos_w[0].cpu().numpy()
  sensor_pos = data.pos_w[0].cpu().numpy()
  torso_id = mujoco.mj_name2id(
    env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link"
  )
  torso_z = (
    float(env.sim.data.xpos[0, torso_id, 2].cpu())
    if hasattr(env.sim.data, "xpos")
    else float("nan")
  )

  obs = env.observation_manager.compute(
    update_history=True
  )  # as env.step; without it the reset-time cache comes back
  hm_clean = obs[MAP_GROUP][0].cpu().numpy().reshape(W, L, 3)
  direct = (
    elevation_map(env, "terrain_scan", noise=False)[0].cpu().numpy().reshape(W, L, 3)
  )
  assert np.allclose(hm_clean, direct), "observation manager output != elevation_map"

  torch.manual_seed(0)
  noisy = []
  for _ in range(N_NOISE):
    env.episode_length_buf[:] = 0  # resample the per-episode offset and drift
    noisy.append(
      elevation_map(env, "terrain_scan", noise=True)[0].cpu().numpy().reshape(W, L, 3)
    )
  noisy = np.stack(noisy)
  # Same height noise with the lateral drift pinned to zero shift (split the
  # two noise sources): resample, then force the centre drift combo.
  torch.manual_seed(1)
  no_drift = []
  for _ in range(N_NOISE):
    env.episode_length_buf[:] = 0
    elevation_map(env, "terrain_scan", noise=True)  # resamples offset + drift
    env._ame_elevation_drift[:] = 4  # (0, 0) shift of the 3 x 3 table
    env.episode_length_buf[:] = 1
    no_drift.append(
      elevation_map(env, "terrain_scan", noise=True)[0].cpu().numpy().reshape(W, L, 3)
    )
  no_drift = np.stack(no_drift)[..., 2]
  env.episode_length_buf[:] = 0

  xy_local = hm_clean[..., :2].reshape(-1, 2)
  xy_world = xy_local + sensor_pos[:2]  # yaw 0
  top = terrain_top(env, xy_world).reshape(W, L)
  true_rel = top - sensor_pos[2]

  hit_z = data.hit_pos_w[0, :, 2].cpu().numpy().reshape(W, L)
  dist = data.distances[0].cpu().numpy().reshape(W, L)
  normal_z = data.normals_w[0, :, 2].cpu().numpy().reshape(W, L)

  stone = top > ground - 0.05
  gap = top < ground - 1.0
  zc, zn = hm_clean[..., 2], noisy[..., 2]
  mid = 0.5 * (zc[stone].mean() + _Z_CLIP[0])
  # Misclassification under noise, judged against the true terrain at the same cell.
  wrong_stone = (zn[:, stone] < mid).mean() if stone.any() else float("nan")
  wrong_gap = (zn[:, gap] > mid).mean() if gap.any() else float("nan")

  nd_wrong_stone = (no_drift[:, stone] < mid).mean() if stone.any() else float("nan")
  nd_wrong_gap = (no_drift[:, gap] > mid).mean() if gap.any() else float("nan")

  # Gap widths along the centre row as the scan sees them (runs of gap cells).
  def runs(mask_row: np.ndarray) -> list[int]:
    r, n = [], 0
    for v in mask_row:
      if v:
        n += 1
      elif n:
        r.append(n)
        n = 0
    if n:
      r.append(n)
    return r

  centre_row = W // 2
  res = {
    "tile": f"{family} d{difficulty}",
    "placement": where,
    "pelvis_z_above_ground_m": round(float(pelvis[2] - ground), 4),
    "sensor_pos_minus_pelvis_m": np.round(sensor_pos - pelvis, 6).tolist(),
    "torso_link_z_minus_pelvis_m": round(torso_z - float(pelvis[2]), 4),
    "robot_geoms_in_group0": robot_geoms_in_group0(env),
    "terrain_geoms": tile_geoms(env),
    "n_points": int(W * L),
    "n_stone_points": int(stone.sum()),
    "n_gap_points": int(gap.sum()),
    "n_other_points": int((~stone & ~gap).sum()),
    "misses": int((dist < 0).sum()),
    "gap_ray_hit_z_world": stats(hit_z[gap]),
    "gap_ray_distance": stats(dist[gap]),
    "gap_ray_normal_z": stats(normal_z[gap]),
    "true_rel_stone": stats(true_rel[stone]),
    "true_rel_gap": stats(true_rel[gap]),
    "obs_clean_stone": stats(zc[stone]),
    "obs_clean_gap": stats(zc[gap]),
    "obs_clean_minus_true_stone_max_abs": round(
      float(np.abs(zc[stone] - true_rel[stone]).max()), 6
    ),
    "obs_noisy_stone": stats(zn[:, stone]),
    "obs_noisy_gap": stats(zn[:, gap]),
    "noisy_threshold_mid_m": round(float(mid), 4),
    "noisy_stone_cells_below_mid": round(float(wrong_stone), 4),
    "noisy_gap_cells_above_mid": round(float(wrong_gap), 4),
    "no_drift_noisy_stone": stats(no_drift[:, stone]),
    "no_drift_noisy_gap": stats(no_drift[:, gap]),
    "no_drift_stone_cells_below_mid": round(float(nd_wrong_stone), 6),
    "no_drift_gap_cells_above_mid": round(float(nd_wrong_gap), 6),
    "noisy_values_at_clip_floor_share_gap": round(
      float((zn[:, gap] <= _Z_CLIP[0] + 1e-6).mean()), 4
    ),
    "noisy_values_at_clip_ceiling_share_all": round(
      float((zn >= _Z_CLIP[1] - 1e-6).mean()), 4
    ),
    "centre_row_gap_runs_cells": runs(gap[centre_row]),
    "xy_channel_equals_grid": bool(np.allclose(noisy[0, ..., :2], hm_clean[..., :2])),
  }
  plot(tag, res["tile"], hm_clean, noisy, true_rel, stone, gap)
  np.savez_compressed(
    OUT / f"{tag}.npz",
    obs_clean=hm_clean,
    obs_noisy=noisy[:20],
    true_rel=true_rel,
    top_world=top,
  )
  return res


def plot(tag, title, hm_clean, noisy, true_rel, stone, gap) -> None:
  ext = (-0.8, 0.8, -0.5, 0.5)
  plt.rcParams.update(
    {
      "font.size": 9,
      "text.color": INK,
      "axes.labelcolor": INK2,
      "xtick.color": INK2,
      "ytick.color": INK2,
      "axes.edgecolor": INK2,
    }
  )
  fig, axes = plt.subplots(
    1, 3, figsize=(15, 3.9), facecolor=SURFACE, constrained_layout=True
  )
  panels = [
    (
      true_rel,
      "True terrain top under each ray,\nrelative to pelvis (unclipped)",
      (-3.0, 0.0),
    ),
    (hm_clean[..., 2], "Actor observation z, noise off\n(play / eval)", (-1.25, 0.0)),
    (noisy[0, ..., 2], "Actor observation z, training noise\n(one draw)", (-1.25, 0.0)),
  ]
  for ax, (img, ttl, (lo, hi)) in zip(axes, panels, strict=True):
    ax.set_facecolor(SURFACE)
    im = ax.imshow(
      img,
      origin="lower",
      extent=ext,
      cmap=SEQ,
      vmin=lo,
      vmax=hi,
      interpolation="nearest",
      aspect="equal",
    )
    ax.plot(0, 0, marker="+", color=ORANGE, ms=10, mew=2)
    ax.set_title(ttl, loc="left")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("z (m)")
  fig.suptitle(
    f"{title}: actor height map vs true terrain (+ = pelvis)", x=0.01, ha="left"
  )
  fig.savefig(OUT / f"{tag}_heatmaps.png", dpi=130, facecolor=SURFACE)
  plt.close(fig)

  # Centre-row profile.
  r = W // 2
  x = hm_clean[r, :, 0]
  fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE, constrained_layout=True)
  ax.set_facecolor(SURFACE)
  zn = noisy[:, r, :, 2]
  ax.fill_between(
    x,
    np.percentile(zn, 5, axis=0),
    np.percentile(zn, 95, axis=0),
    color=ORANGE,
    alpha=0.18,
    lw=0,
    label="training noise, 5-95 % over draws",
  )
  ax.plot(
    x, zn[0], color=ORANGE, lw=1, marker="o", ms=3, label="training noise, one draw"
  )
  ax.plot(
    x,
    true_rel[r],
    color=INK2,
    lw=2,
    drawstyle="steps-mid",
    label="true terrain (unclipped)",
  )
  ax.plot(
    x,
    hm_clean[r, :, 2],
    color=BLUE,
    lw=2,
    marker="o",
    ms=4,
    label="observation, noise off",
  )
  ax.axhline(_Z_CLIP[0], color=INK2, lw=1, ls="--")
  ax.text(
    -0.3, _Z_CLIP[0] - 0.09, "clip floor -1.2 m", color=INK2, ha="left", fontsize=8
  )
  ax.set_xlim(-0.82, 0.82)
  ax.set_ylim(-3.0, 0.15)
  ax.set_xlabel("x forward of pelvis (m), centre row y = 0")
  ax.set_ylabel("z relative to pelvis (m)")
  ax.grid(axis="y", color="#e6e5e1", lw=0.8)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  ax.legend(loc="lower left", frameon=False, fontsize=8)
  ax.set_title(f"{title}: centre-row profile", loc="left")
  fig.savefig(OUT / f"{tag}_profile.png", dpi=130, facecolor=SURFACE)
  plt.close(fig)


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  report = {"cfgs": describe_cfgs()}
  report["stepping_stones_d0.5"] = probe("stepping_stones", 0.5, "stepping_stones_d050")
  report["stone_lattice_d0.25"] = probe("stone_lattice", 0.25, "stone_lattice_d025")
  (OUT / "probe.json").write_text(json.dumps(report, indent=1, default=str))
  print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
  main()
