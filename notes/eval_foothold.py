"""Foothold evaluation per terrain family and level (prepared; GPU).

For one checkpoint: the gap board's curriculum layout (10 levels x 5 families,
difficulty = level / 9), every tile pinned (no curriculum), ``--envs_per_cell``
robots per tile, the task's centre spawn (uniform heading), a fixed forward command
``--vx`` (no heading control, no standing envs), no pushes, no observation noise,
mean actions, one episode of ``--duration_s`` per robot (steps after a robot's first
termination, or after its pelvis leaves its tile's inner square ``--tile_half``, are
dropped).

Per (family, level) and per delta in ``--deltas`` (same functions as the reward,
:mod:`ame.mdp.foothold`):

* unsupported share: mean over in-contact foot-steps of the share of the 15 sole
  points more than delta above the terrain (what the ``foothold`` reward pays);
* touchdown unsupported share: the same at the touchdown step only;
* gap-landing rate: share of touchdowns with any sole point over a gap (terrain
  more than ``GAP_DROP`` = 0.5 m below the sole, or no hit within 1 m);

and the swing-cost diagnostic (:class:`ame.mdp.swing.SwingGapTracker`, as the
training log): per swing, ``foot_clearance`` cost summed over the swing and
``foot_swing_height`` cost at touchdown, two-sided (mjlab) and one-sided, split by
whether the sole centre crossed a gap; plus fall rate.

    uv run python notes/eval_foothold.py --checkpoint <model.pt> --out <dir>
    uv run python notes/eval_foothold.py --checkpoint <model.pt> --task <gap-board task>
    uv run python notes/eval_foothold.py --terrain_table --out <dir>   # geometry only, CPU

``--task`` sets the env and the actor architecture (default: the Foothold task); any
task on the gap board works, the sole scan is added if missing.  Only the actor is
loaded, so single- and double-critic checkpoints both work.  Writes
``foothold_eval.json`` (and ``terrain_table.json``) to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os

FOOTHOLD_TASK = "Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Foothold-Velocity-Rough-G1"


# ------------------------------------------------------------------ env
def eval_env_cfg(task: str, envs_per_cell: int, duration_s: float, vx: float):
  from mjlab.tasks.registry import load_env_cfg

  import ame  # noqa: F401
  from ame.env_cfg import apply_foot_sole_scan

  cfg = load_env_cfg(task)
  gen = cfg.scene.terrain.terrain_generator
  assert gen.curriculum, "needs the curriculum board (one column per family)"
  cells = gen.num_rows * len(gen.sub_terrains)
  cfg.scene.num_envs = cells * envs_per_cell
  cfg.curriculum = {}  # tiles stay where they are put
  cfg.events.pop("push_robot", None)
  cfg.episode_length_s = duration_s
  cfg.observations["actor"].enable_corruption = False
  for group in cfg.observations.values():
    for term in group.terms.values():
      if "noise" in (term.params or {}):
        term.params["noise"] = False
  twist = cfg.commands["twist"]
  twist.heading_command = False
  twist.rel_heading_envs = 0.0
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.ranges.heading = None
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.resampling_time_range = (1.0e9, 1.0e9)
  apply_foot_sole_scan(cfg)
  return cfg


def pin_tiles(env, envs_per_cell: int):
  """Env i on cell i // envs_per_cell, cells level-major; returns (levels, types)."""
  import torch

  terrain = env.scene.terrain
  rows, cols = terrain.terrain_origins.shape[:2]
  cell = torch.arange(env.num_envs, device=env.device) // envs_per_cell
  levels, types = cell // cols, cell % cols
  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  return levels, types, rows, cols


# ------------------------------------------------------------------ rollout
def evaluate(args) -> dict:
  from dataclasses import asdict

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  from ame.mdp.foothold import (
    GAP_DROP,
    SOLE_SCAN_SENSOR,
    feet_in_contact,
    sensor_sole_drop,
    unsupported_fraction,
  )
  from ame.mdp.swing import SwingGapTracker, clearance_cost, command_active, over_gap

  device = args.device
  torch.manual_seed(args.seed)
  cfg = eval_env_cfg(args.task, args.envs_per_cell, args.duration_s, args.vx)
  cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  families = list(cfg.scene.terrain.terrain_generator.sub_terrains)
  levels, types, rows, cols = pin_tiles(env, args.envs_per_cell)
  rl_cfg = load_rl_cfg(args.task)
  venv = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(
    venv, asdict(rl_cfg), device=device
  )
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  obs, _ = venv.reset()

  robot = env.scene["robot"]
  contact_sensor = env.scene["feet_ground_contact"]
  sole = env.scene[SOLE_SCAN_SENSOR]
  height = env.scene["foot_height_scan"]
  sites = [robot.site_names.index(s) for s in ("left_foot", "right_foot")]
  clear_cfg = env.reward_manager.get_term_cfg("foot_clearance").params
  target = float(clear_cfg["target_height"])
  tracker = SwingGapTracker(env.num_envs, 2, device, target)
  deltas = list(args.deltas)
  n_cells = rows * cols
  cell = levels * cols + types
  z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float64)  # noqa: E731
  acc = {
    "contact_foot_steps": z(n_cells),
    "unsupported": z(len(deltas), n_cells),
    "touchdowns": z(n_cells),
    "touchdown_unsupported": z(len(deltas), n_cells),
    "gap_landings": z(n_cells),
    "falls": z(n_cells),
    "left_tile": z(n_cells),
    "robots": torch.bincount(cell, minlength=n_cells).double(),
  }
  for tag in ("gap", "nogap"):
    acc[f"swings_{tag}"] = z(n_cells)
    for k in ("clear2", "clear1", "height2", "height1"):
      acc[f"{k}_{tag}"] = z(n_cells)

  alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
  tile_xy = env.scene.terrain.terrain_origins[levels, types, :2]
  steps = round(args.duration_s / env.step_dt) - 1
  dt = env.step_dt
  for _ in range(steps):
    with torch.inference_mode():
      obs, _, dones, extras = venv.step(policy(obs))
    done = dones.bool()
    fell = done & ~extras.get("time_outs", torch.zeros_like(done)).bool()
    acc["falls"].index_add_(
      0, cell[alive & fell], torch.ones_like(cell[alive & fell], dtype=torch.float64)
    )
    alive &= ~done
    if args.tile_half > 0:
      # Robots walk off their tile (0.8 m/s, 8 m tiles): stop counting a robot
      # once its pelvis leaves the tile's inner square, so every step is
      # attributed to the tile it was spawned on.
      off = (robot.data.root_link_pos_w[:, :2] - tile_xy).abs().amax(
        -1
      ) > args.tile_half
      left = alive & off
      acc["left_tile"].index_add_(
        0, cell[left], torch.ones_like(cell[left], dtype=torch.float64)
      )
      alive &= ~off
    if not alive.any():
      break
    a = alive[:, None]
    contact = feet_in_contact(contact_sensor)
    first = contact_sensor.compute_first_contact(dt=dt)
    drop = sensor_sole_drop(sole)  # [B, 2, 15], sensed at the end of this step
    cf = (contact & a).double()
    acc["contact_foot_steps"].index_add_(0, cell, cf.sum(1))
    td = (first & a).double()
    acc["touchdowns"].index_add_(0, cell, td.sum(1))
    for i, d in enumerate(deltas):
      frac = unsupported_fraction(drop, d).double()
      acc["unsupported"][i].index_add_(0, cell, (frac * cf).sum(1))
      acc["touchdown_unsupported"][i].index_add_(0, cell, (frac * td).sum(1))
    gap_td = ((drop > GAP_DROP).any(-1) & first & a).double()
    acc["gap_landings"].index_add_(0, cell, gap_td.sum(1))
    vel = torch.norm(robot.data.site_lin_vel_w[:, sites, :2], dim=-1)
    h = height.data.heights
    out = tracker.update(
      contact=contact,
      first_contact=first,
      clear2=clearance_cost(h, vel, target, one_sided=False),
      clear1=clearance_cost(h, vel, target, one_sided=True),
      heights=h,
      over_gap=over_gap(drop),
      active=command_active(env, "twist", 0.05),
    )
    keep = alive[out["env"]]
    c = cell[out["env"]][keep]
    crossed = out["crossed"][keep]
    for tag, m in (("gap", crossed), ("nogap", ~crossed)):
      acc[f"swings_{tag}"].index_add_(
        0, c[m], torch.ones_like(c[m], dtype=torch.float64)
      )
      for k in ("clear2", "clear1", "height2", "height1"):
        acc[f"{k}_{tag}"].index_add_(0, c[m], out[k][keep][m].double())
    tracker.reset(torch.nonzero(done).squeeze(-1))

  return summarise(acc, deltas, families, rows, cols, args, env)


def summarise(acc, deltas, families, rows, cols, args, env) -> dict:
  def ratio(num, den):
    return [None if d == 0 else n / d for n, d in zip(num, den, strict=True)]

  a = {k: v.cpu().tolist() for k, v in acc.items()}
  a["unsupported"] = acc["unsupported"].cpu().tolist()
  a["touchdown_unsupported"] = acc["touchdown_unsupported"].cpu().tolist()
  cells = []
  for i in range(rows * cols):
    level, family = i // cols, families[i % cols]
    row = {
      "family": family,
      "level": level,
      "difficulty": level / max(rows - 1, 1),
      "robots": a["robots"][i],
      "fall_rate": ratio([a["falls"][i]], [a["robots"][i]])[0],
      "left_tile_rate": ratio([a["left_tile"][i]], [a["robots"][i]])[0],
      "touchdowns": a["touchdowns"][i],
      "gap_landing_rate": ratio([a["gap_landings"][i]], [a["touchdowns"][i]])[0],
    }
    for j, d in enumerate(deltas):
      row[f"unsupported_share@{d}"] = ratio(
        [a["unsupported"][j][i]], [a["contact_foot_steps"][i]]
      )[0]
      row[f"touchdown_unsupported@{d}"] = ratio(
        [a["touchdown_unsupported"][j][i]], [a["touchdowns"][i]]
      )[0]
    for tag in ("gap", "nogap"):
      n = a[f"swings_{tag}"][i]
      row[f"swings_{tag}"] = n
      for k, name in (
        ("clear2", "clearance"),
        ("clear1", "clearance_1s"),
        ("height2", "swing_height"),
        ("height1", "swing_height_1s"),
      ):
        row[f"{name}_{tag}"] = ratio([a[f"{k}_{tag}"][i]], [n])[0]
    cells.append(row)
  return {
    "checkpoint": os.path.abspath(args.checkpoint),
    "task": args.task,
    "vx": args.vx,
    "duration_s": args.duration_s,
    "tile_half": args.tile_half,
    "envs_per_cell": args.envs_per_cell,
    "deltas": deltas,
    "step_dt": env.step_dt,
    "note": "clearance costs: sum over a swing of the per-step raw cost (mjlab "
    "feet_clearance per foot, unweighted); swing_height: raw cost at touchdown; "
    "reward units = cost * weight * step_dt",
    "cells": cells,
  }


# ------------------------------------------------------------------ terrain table
def terrain_table() -> dict:
  """Stair step heights and stone gap widths / pit depths per level, from the
  compiled gap board (rays on a 1 cm profile from each tile centre), next to the
  config formulas."""
  from dataclasses import replace

  import mujoco
  import numpy as np
  from mjlab.terrains.terrain_generator import TerrainGenerator

  from ame.terrains import gap_board_cfg

  gen = replace(gap_board_cfg(), seed=0)
  tg = TerrainGenerator(gen)
  spec = mujoco.MjSpec()
  tg.compile(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  origins = np.asarray(tg.terrain_origins)
  families = list(gen.sub_terrains)
  half = gen.size[0] / 2

  def heights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gid = np.zeros(1, dtype=np.int32)
    down = np.array([0.0, 0.0, -1.0])
    h = np.empty(len(x))
    for k, (px, py) in enumerate(zip(x, y, strict=True)):
      dist = mujoco.mj_ray(
        model, data, np.array([px, py, 10.0]), down, None, 1, -1, gid
      )
      h[k] = 10.0 - dist if dist >= 0 else np.nan
    return h

  out = []
  for r in range(gen.num_rows):
    d = r / (gen.num_rows - 1)
    for c, fam in enumerate(families):
      sub = gen.sub_terrains[fam]
      cx, cy = origins[r, c, 0], origins[r, c, 1]
      s = np.arange(0.0, half - 0.005, 0.01)
      row = {"family": fam, "level": r, "difficulty": round(d, 4)}
      if "stairs" in fam:
        prof = heights(cx + s, np.full_like(s, cy))
        jumps = np.diff(prof)
        jumps = jumps[np.abs(jumps) > 1e-3]
        row["step_height_cfg"] = sub.step_height_range[0] + d * (
          sub.step_height_range[1] - sub.step_height_range[0]
        )
        row["step_heights_measured"] = sorted({round(float(abs(j)), 4) for j in jumps})
        row["steps_from_centre"] = len(jumps)
        row["direction_from_centre"] = "down" if jumps.mean() < 0 else "up"
        row["step_width"] = sub.step_width
      elif fam == "stepping_stones":
        grid, _ = sub.centres(d)
        y0 = origins[r, c, 1] - gen.size[1] / 2 + grid[0, 1]
        xs = origins[r, c, 0] - half + np.arange(0.0, gen.size[0], 0.005)
        prof = heights(xs, np.full_like(xs, y0))
        pit = prof < -0.5
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, pit.astype(int), 0]) != 0))[::2]
        runs = runs[runs >= 2]  # a lone 5 mm sample is the seam between tiles
        row["gap_cfg"] = sub.gap(d)
        row["gap_measured_mean"] = float(runs.mean() * 0.005) if len(runs) else 0.0
        row["stone_size"] = sub.stone_size
        row["gap_measured_max"] = float(runs.max() * 0.005) if len(runs) else 0.0
        row["pit_depth"] = float(-np.nanmin(prof)) if len(runs) else 0.0
      else:
        g = np.linspace(-half + 0.3, half - 0.3, 60)
        gx, gy = np.meshgrid(cx + g, cy + g)
        h = heights(gx.ravel(), gy.ravel())
        row["height_range"] = float(np.nanmax(h) - np.nanmin(h))
        row["height_std"] = float(np.nanstd(h))
      out.append(row)
  return {"families": families, "rows": out}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--checkpoint")
  p.add_argument("--task", default=FOOTHOLD_TASK)
  p.add_argument("--out", default="logs/eval_results/foothold")
  p.add_argument("--envs_per_cell", type=int, default=8)
  p.add_argument("--duration_s", type=float, default=20.0)
  p.add_argument("--vx", type=float, default=0.8)
  p.add_argument("--deltas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--tile_half",
    type=float,
    default=3.75,
    help="count a robot only while its pelvis is within this Chebyshev distance "
    "of its tile centre (0: no gate)",
  )
  p.add_argument("--terrain_table", action="store_true")
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  if args.terrain_table:
    res = terrain_table()
    path = os.path.join(args.out, "terrain_table.json")
  else:
    if not args.checkpoint:
      p.error("--checkpoint is required unless --terrain_table")
    res = evaluate(args)
    path = os.path.join(args.out, "foothold_eval.json")
  with open(path, "w") as f:
    json.dump(res, f, indent=1)
  print("wrote", path)


if __name__ == "__main__":
  main()
