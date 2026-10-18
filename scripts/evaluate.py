"""Quantitative evaluation of an AME checkpoint on mjlab (headless, one episode per env).

A port of the Isaac Lab evaluation (AME_Locomotion, branch eval-pipeline) with the same outcome rules and output
format. Each env spawns at the origin of its own terrain tile (the center; on stone_lattice the stone nearest
it) with yaw 0 and is commanded vx = 1.0 m/s, vy = 0,
wz = 0 (pure twist; there is no heading controller). Outcome per env, from its FIRST episode only (auto-resets
afterwards are ignored):
    success: forward displacement >= --success_dist before a fail; forward = the base xy displacement from spawn
             projected on the spawn heading (the command direction at spawn: vx in the base frame, yaw 0 = +x).
             --success_rule any restores the old rule (xy displacement in any direction).
    fail:    a termination other than the time-out (illegal_contact -> "contact", bad_orientation ->
             "orientation", physics_blowup -> "blowup"), or a fall (tilt > --tilt_deg, or base height above the
             supporting surface < --fall_height; optionally base z < spawn z - --spawn_drop), held for --fall_hold_s
    stuck:   neither by the end of the episode

Per env, max_forward_m is the largest forward displacement, forward_at_end_m the forward displacement at the last
tracked step (success, fail or time-out), max_xy_disp_m the largest displacement in any direction.

Gait, per env over its tracked steps (feet_ground_contact sensor, sampled once per policy step at 50 Hz): the
fraction of steps with both feet on the terrain (double support), exactly one (single support), or none (flight).

Validity guard: the run is aborted, summary.json is written with "valid": false and the exit code is 2 if the
spawn check fails at step 1 (a robot not at its tile center, or not standing on the terrain), if more than 50% of
envs fail by "height" within the first second, or if any env succeeds faster than 3 m/s on average.

Examples:
    uv run python scripts/evaluate.py --checkpoint logs/rsl_rl/ame_g1/<run>/model_10000.pt --seed 0
    uv run python scripts/evaluate.py --random_policy --num_rows 2 --num_cols 2 --max_steps 50   # pipeline test
    uv run python scripts/evaluate.py --zero_policy --num_rows 2 --num_cols 2 --episode_s 0.1   # time-out path
"""

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PENDING, SUCCESS, FAIL, STUCK = 0, 1, 2, 3
OUTCOME_NAMES = {PENDING: "pending", SUCCESS: "success", FAIL: "fail", STUCK: "stuck"}
FAIL_REASONS = {
  0: "",
  1: "contact",
  2: "tilt",
  3: "height",
  4: "spawn_drop",
  5: "orientation",
  6: "blowup",
}
HEIGHT = 3
TERM_REASON = {"illegal_contact": 1, "bad_orientation": 5, "physics_blowup": 6}
OTHER_TERM_REASON = (
  1  # an unknown termination term counts as "contact", as in the Isaac Lab tool
)
EARLY_WINDOW_S = 1.0  # validity guard: window for early height fails
EARLY_HEIGHT_FRAC = (
  0.5  # validity guard: max fraction of envs failing by height in that window
)
# spawn check: max distance of the reset position from the tile center, max displacement after the first policy
# step, and the plausible range of the median base height above the support after the first step
SPAWN_TOL_M = 0.05
STEP1_DISP_TOL_M = 0.10
STEP1_HEIGHT_RANGE_M = (0.5, 1.1)
# speed check: a success faster than this mean speed is physically implausible (command 1 m/s)
MAX_PLAUSIBLE_SPEED_MPS = 3.0


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Evaluate an AME checkpoint on mjlab.")
  p.add_argument(
    "--task",
    default="Ame-Velocity-Rough-G1",
    help="Registered task whose policy config to use.",
  )
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--checkpoint", help="Checkpoint path (model_<iter>.pt).")
  src.add_argument(
    "--random_policy",
    action="store_true",
    help="Randomly initialized policy, no checkpoint. Pipeline tests only; results are meaningless.",
  )
  src.add_argument(
    "--zero_policy",
    action="store_true",
    help="Zero actions (default joint pose), no checkpoint. Pipeline tests only (with a short --episode_s: the time-out).",
  )
  p.add_argument(
    "--model_name",
    default=None,
    help="Label for this policy in folder names and the sweep table. Default: <run dir>_<ckpt name>.",
  )
  p.add_argument(
    "--terrain",
    default="stepping_stones",
    help="Terrain family of the AME training mix, or stone_lattice (the gap task's lattice, gap = 0.45 * d).",
  )
  p.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Default: num_rows x num_cols (one robot per tile).",
  )
  p.add_argument(
    "--num_rows",
    type=int,
    default=8,
    help="Tile rows. 8x8 needs ~9 GB host RAM to build, 10x10 ~20 GB.",
  )
  p.add_argument("--num_cols", type=int, default=8)
  p.add_argument(
    "--difficulty",
    type=float,
    default=0.5,
    help="Fixed terrain difficulty for all tiles.",
  )
  p.add_argument(
    "--seed", type=int, default=0, help="Seed for the terrain generator and the env."
  )
  p.add_argument("--episode_s", type=float, default=20.0)
  p.add_argument(
    "--stochastic_actions",
    action="store_true",
    help="Sample actions from the policy's Gaussian (as in training rollouts) instead of using its mean. "
    "Sampling is seeded with --seed.",
  )
  p.add_argument(
    "--spawn_asym",
    type=float,
    nargs="?",
    const=0.1,
    default=0.0,
    help="Break the left-right symmetry at spawn: start left_hip_pitch_joint this many rad off the default "
    "pose at every reset (flag alone: +0.1 rad). Default 0 = symmetric spawn.",
  )
  p.add_argument(
    "--actor_obs",
    default="auto",
    choices=["auto", "full", "deploy"],
    help="Actor observation set. full: with base_lin_vel (the AME tasks); deploy: without it (the gap-board "
    "task). auto: from the checkpoint's actor input width (full without a checkpoint).",
  )
  p.add_argument(
    "--com_dr",
    action="store_true",
    help="Keep the CoM randomization of play mode (default off).",
  )
  # noise
  p.add_argument(
    "--noise",
    default="train",
    choices=["train", "off"],
    help="Preset. train: the training map noise (sigma 0.03, offset 0.05, drift 1 cell) and the "
    "actor's training proprio noise. off: no noise.",
  )
  p.add_argument(
    "--height_sigma",
    type=float,
    default=None,
    help="Override: per-cell Gaussian height noise [m].",
  )
  p.add_argument(
    "--height_offset",
    type=float,
    default=None,
    help="Override: per-episode offset U(-x, x) [m].",
  )
  p.add_argument(
    "--drift_cells",
    type=int,
    default=None,
    help="Override: per-episode z-map drift, dx and dy uniform in {-n..n} cells (5 cm).",
  )
  p.add_argument(
    "--map_delay_steps",
    type=int,
    default=None,
    help="Override: refresh the map every N policy steps (5 = 10 Hz). 1 = every step.",
  )
  p.add_argument("--proprio_noise", default="preset", choices=["preset", "on", "off"])
  # outcome rules
  p.add_argument("--success_dist", type=float, default=4.0)
  p.add_argument(
    "--success_rule",
    default="forward",
    choices=["forward", "any"],
    help="forward: displacement along the spawn heading (the command direction) >= --success_dist. "
    "any: xy displacement in any direction (the old rule).",
  )
  p.add_argument("--tilt_deg", type=float, default=60.0)
  p.add_argument(
    "--fall_height",
    type=float,
    default=0.4,
    help="Min base height above supporting surface [m].",
  )
  p.add_argument(
    "--support_radius",
    type=float,
    default=0.3,
    help="XY radius around base for the surface [m].",
  )
  p.add_argument("--fall_hold_s", type=float, default=0.2)
  p.add_argument(
    "--spawn_drop_rule",
    action="store_true",
    help="Also fail if base z drops more than --spawn_drop below spawn (wrong on stairs down).",
  )
  p.add_argument("--spawn_drop", type=float, default=0.5)
  # run control / output
  p.add_argument(
    "--max_steps", type=int, default=None, help="Stop early (smoke / memory tests)."
  )
  p.add_argument(
    "--device", default=None, help="Default: cuda:0 if available, else cpu."
  )
  p.add_argument("--out_root", default=os.path.join(REPO_ROOT, "logs", "eval_results"))
  p.add_argument("--tag", default="", help="Suffix for the output folder name.")
  return p


def resolve_noise(args) -> tuple[dict, bool]:
  from ame.eval import OFF_HEIGHT_NOISE, TRAIN_HEIGHT_NOISE

  train = args.noise == "train"
  height_noise = dict(TRAIN_HEIGHT_NOISE if train else OFF_HEIGHT_NOISE)
  for key, value in (
    ("height_sigma", args.height_sigma),
    ("height_offset", args.height_offset),
    ("drift_cells", args.drift_cells),
    ("delay_steps", args.map_delay_steps),
  ):
    if value is not None:
      height_noise[key] = value
  proprio = train if args.proprio_noise == "preset" else args.proprio_noise == "on"
  return height_noise, proprio


def noise_tag(height_noise: dict, proprio: bool) -> str:
  if (
    not proprio
    and height_noise["height_sigma"] == 0
    and height_noise["height_offset"] == 0
    and height_noise["drift_cells"] == 0
    and height_noise["delay_steps"] <= 1
  ):
    return "noiseoff"
  return (
    f"h{height_noise['height_sigma']:g}-o{height_noise['height_offset']:g}"
    f"-dc{height_noise['drift_cells']}-d{height_noise['delay_steps']}-p{int(proprio)}"
  )


def git_info() -> dict:
  try:
    commit = subprocess.check_output(
      ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
      ["git", "-C", REPO_ROOT, "status", "--porcelain", "-uno"], text=True
    ).strip()
    return {"commit": commit, "dirty_files": dirty.splitlines()}
  except Exception as e:  # git missing or not a repo
    return {"error": str(e)}


def rate(k: int, n: int) -> dict:
  p = k / n if n > 0 else float("nan")
  se = math.sqrt(p * (1.0 - p) / n) if n > 0 else float("nan")
  return {"count": k, "rate": p, "se": se}


def success_progress(disp, heading, rule: str):
  """Per env: (forward, xy distance, the distance the success rule compares with --success_dist). ``disp`` is the
  (N, 2) xy displacement from spawn, ``heading`` the (N,) spawn yaw; forward = ``disp`` along the spawn heading."""
  import torch

  fwd_dir = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
  forward = (disp * fwd_dir).sum(dim=-1)
  xy_dist = torch.linalg.norm(disp, dim=-1)
  return forward, xy_dist, forward if rule == "forward" else xy_dist


def host_max_rss_mb() -> float:
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
  args = build_parser().parse_args()
  if args.num_envs is None:
    args.num_envs = args.num_rows * args.num_cols

  from dataclasses import asdict

  import mujoco
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  from ame.eval import (
    CONTACT_BODY_ATTR,
    CONTACT_PARTNER_ATTR,
    PRE_RESET_POS_ATTR,
    assign_tiles,
    checkpoint_actor_obs,
    eval_env_cfg,
  )
  from ame.reward_cfg import FEET_CONTACT_SENSOR

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cuda = device.startswith("cuda")
  height_noise, proprio = resolve_noise(args)
  checkpoint = None
  if args.checkpoint is not None:
    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
    if not os.path.isfile(checkpoint):
      sys.exit(f"[ERROR] checkpoint not found: {checkpoint}")
  policy_kind = (
    "checkpoint" if checkpoint else ("zero" if args.zero_policy else "random_init")
  )
  if args.model_name is None:
    run_dir = os.path.basename(os.path.dirname(checkpoint)) if checkpoint else ""
    stem = os.path.splitext(os.path.basename(checkpoint))[0] if checkpoint else ""
    args.model_name = f"{run_dir}_{stem}" if checkpoint else policy_kind
  if args.actor_obs == "auto":
    args.actor_obs = checkpoint_actor_obs(checkpoint) if checkpoint else "full"
  print(f"[INFO] actor observations: {args.actor_obs}")

  cuda_used_mb = (
    (lambda: (lambda f, t: (t - f) / 2**20)(*torch.cuda.mem_get_info(device)))
    if cuda
    else (lambda: None)
  )
  cuda_before = cuda_used_mb()
  cuda_peak = cuda_before

  # -- configs
  env_cfg = eval_env_cfg(
    terrain=args.terrain,
    num_rows=args.num_rows,
    num_cols=args.num_cols,
    difficulty=args.difficulty,
    seed=args.seed,
    episode_s=args.episode_s,
    height_noise=height_noise,
    proprio_noise=proprio,
    com_dr=args.com_dr,
    spawn_asym=args.spawn_asym,
    actor_obs=args.actor_obs,
  )
  env_cfg.scene.num_envs = args.num_envs
  rl_cfg = load_rl_cfg(args.task)
  rl_cfg.seed = args.seed

  # -- env and policy (same loading as mjlab's play.py)
  t_build = time.time()
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env_build_s = time.time() - t_build
  assign_tiles(base_env)  # before the wrapper's initial reset
  env = RslRlVecEnvWrapper(base_env, clip_actions=rl_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(rl_cfg), device=device)
  ckpt_iter = None
  if checkpoint is not None:
    print(f"[INFO] Loading model checkpoint from: {checkpoint}")
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    ckpt_iter = torch.load(checkpoint, map_location="cpu", weights_only=False).get(
      "iter"
    )
  else:
    print(f"[WARN] {policy_kind} policy: results are meaningless (pipeline test only).")
  policy = runner.get_inference_policy(device=device)
  if args.stochastic_actions and checkpoint is not None:
    torch.manual_seed(args.seed)
    mean_policy = policy
    policy = lambda obs: mean_policy(obs, stochastic_output=True)  # noqa: E731
  if args.zero_policy:
    num_actions = env.num_actions
    policy = lambda obs: torch.zeros(env.num_envs, num_actions, device=device)  # noqa: E731
  arch = rl_cfg.actor.class_name.split(":")[-1]
  if cuda:
    cuda_peak = max(cuda_peak, cuda_used_mb())

  robot = base_env.scene["robot"]
  scan = base_env.scene["terrain_scan"]
  feet = base_env.scene[FEET_CONTACT_SENSOR]
  terrain = base_env.scene.terrain
  contact_sensor_name = env_cfg.terminations["illegal_contact"].params["sensor_name"]
  contact_body_names = [
    n.split("/")[-1] for n in base_env.scene[contact_sensor_name].primary_names
  ]
  tm = base_env.termination_manager
  fail_terms = [n for n in tm.active_terms if not tm.get_term_cfg(n).time_out]
  num_envs, dt = base_env.num_envs, base_env.step_dt
  mj_model = base_env.sim.mj_model
  terrain_body = next(
    b
    for b in range(mj_model.nbody)
    if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b).split("/")[-1]
    == "terrain"
  )
  terrain_geoms = int((mj_model.geom_bodyid == terrain_body).sum())

  # -- per-env buffers
  spawn_pos = robot.data.root_link_pos_w.clone()
  tile_center = base_env.scene.env_origins.clone()
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  outcome = torch.full((num_envs,), PENDING, dtype=torch.long, device=device)
  fail_reason = torch.zeros(num_envs, dtype=torch.long, device=device)
  term_mask = torch.zeros(
    num_envs, dtype=torch.long, device=device
  )  # bit k = fail_terms[k] fired
  fail_time = torch.full((num_envs,), float("nan"), device=device)
  contact_body = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  contact_partner = [""] * num_envs  # "terrain" / "self:<link>", deepest first
  time_to_success = torch.full((num_envs,), float("nan"), device=device)
  spawn_heading = (
    robot.data.heading_w.clone()
  )  # the command direction at spawn (vx in the base frame)
  max_forward = torch.zeros(num_envs, device=device)
  forward_end = torch.zeros(num_envs, device=device)
  max_xy_disp = torch.zeros(num_envs, device=device)
  err_sum = torch.zeros(num_envs, device=device)
  height_sum = torch.zeros(num_envs, device=device)
  n_height = torch.zeros(num_envs, dtype=torch.long, device=device)
  n_tracked = torch.zeros(num_envs, dtype=torch.long, device=device)
  # steps with 0 / 1 / 2 feet on the terrain: flight / single / double support
  n_feet_down = torch.zeros(num_envs, 3, dtype=torch.long, device=device)
  tilt_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
  low_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
  drop_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
  hold_steps = max(1, round(args.fall_hold_s / dt))
  cos_tilt = math.cos(math.radians(args.tilt_deg))
  episode_steps = base_env.max_episode_length
  step_limit = episode_steps + 5 if args.max_steps is None else args.max_steps
  tile_rows = terrain.terrain_levels.cpu().tolist()
  tile_cols = terrain.terrain_types.cpu().tolist()
  tiles_total = args.num_rows * args.num_cols
  tiles_distinct = len(set(zip(tile_rows, tile_cols, strict=True)))

  # -- output folder (created now so that an aborted run still leaves an INVALID summary)
  tag = noise_tag(height_noise, proprio)
  run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{args.model_name}_{args.terrain}_{tag}"
  if args.spawn_asym:
    run_name += f"_asym{args.spawn_asym:g}"
  if args.stochastic_actions:
    run_name += "_stoch"
  if args.tag:
    run_name += f"_{args.tag}"
  out_dir = os.path.join(os.path.expanduser(args.out_root), run_name)
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, "args.json"), "w") as f:
    json.dump({"argv": sys.argv, **vars(args)}, f, indent=2, default=str)

  run_info = {
    "checkpoint": checkpoint,
    "policy": policy_kind,
    "task": args.task,
    "arch": arch,
    "model_name": args.model_name,
    "terrain": args.terrain,
    "seed": args.seed,
    "success_rule": args.success_rule,
    "success_dist": args.success_dist,
    "num_envs": num_envs,
    "tiles_total": tiles_total,
    "height_noise": height_noise,
    "proprio_noise": proprio,
    "stochastic_actions": args.stochastic_actions,
    "spawn_asym": args.spawn_asym,
    "tag": args.tag,
  }

  # -- validity guard
  early_steps = round(EARLY_WINDOW_S / dt)
  spawn_offset = torch.linalg.norm(spawn_pos[:, :2] - tile_center[:, :2], dim=-1)
  spawn_check = {"spawn_offset_max_m": float(spawn_offset.max())}

  def validity(step: int) -> dict:
    early_height = int(
      ((fail_reason == HEIGHT) & (fail_time <= EARLY_WINDOW_S + 1e-6)).sum()
    )
    min_t = time_to_success[torch.isfinite(time_to_success)]
    min_t = float(min_t.min()) if min_t.numel() else float("nan")
    reason = None
    if spawn_check.get("ok") is False:
      reason = f"spawn check failed: {spawn_check['problem']}"
    elif min_t < args.success_dist / MAX_PLAUSIBLE_SPEED_MPS:
      reason = (
        f"a success after {min_t:.2f} s (> {MAX_PLAUSIBLE_SPEED_MPS:g} m/s mean speed); "
        "physically implausible, probably a teleport or a reset bug"
      )
    elif num_envs <= tiles_total and tiles_distinct != num_envs:
      reason = f"only {tiles_distinct} distinct tiles for {num_envs} envs"
    elif step >= early_steps and early_height > EARLY_HEIGHT_FRAC * num_envs:
      reason = (
        f"{early_height}/{num_envs} envs failed by 'height' within the first {EARLY_WINDOW_S:g} s "
        f"(> {EARLY_HEIGHT_FRAC:.0%}); the robots probably fell through the terrain"
      )
    return {
      "valid": reason is None,
      "invalid_reason": reason,
      "spawn_check": spawn_check,
      "early_height_fails": early_height,
      "blowup_fails": int((fail_reason == 6).sum()),
      "min_time_to_success_s": min_t,
      "checked_at_step": step,
    }

  def abort_if_invalid(step: int) -> dict:
    check = validity(step)
    if check["valid"]:
      return check
    summary = {
      "valid": False,
      "invalid_reason": check["invalid_reason"],
      "validity_check": check,
      **run_info,
      "steps_simulated": step,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
      json.dump(summary, f, indent=2)
    print(f"[ERROR] INVALID RUN: {check['invalid_reason']}", flush=True)
    print(
      f"[ERROR] Marked invalid in: {os.path.join(out_dir, 'summary.json')}", flush=True
    )
    env.close()
    sys.exit(2)

  obs = env.get_observations()
  step = 0
  t_start = time.time()
  with torch.inference_mode():
    while active.any() and step < step_limit:
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      step += 1

      # envs whose episode ended this step: their state is already reset, so use the outcome only
      time_out = base_env.reset_time_outs
      ended = active & (base_env.reset_terminated | time_out)
      failed = ended & ~time_out
      for k, name in enumerate(fail_terms):
        fired = failed & tm.get_term(name).bool()
        term_mask |= fired.long() << k
        first = fired & (fail_reason == 0)
        fail_reason[first] = TERM_REASON.get(name, OTHER_TERM_REASON)
      outcome[failed] = FAIL
      fail_time[failed] = step * dt
      recorded_body = getattr(base_env, CONTACT_BODY_ATTR, None)
      if recorded_body is not None:
        hit = failed & ((term_mask >> fail_terms.index("illegal_contact")) & 1).bool()
        contact_body[hit] = recorded_body[hit]
        recorded_partner = getattr(base_env, CONTACT_PARTNER_ATTR, None)
        if recorded_partner is not None:
          for i in hit.nonzero().flatten().tolist():
            contact_partner[i] = recorded_partner[i]
      outcome[ended & time_out] = STUCK

      # displacement: the terminal (pre-reset) position for envs that just ended
      pos = robot.data.root_link_pos_w
      pre_reset = getattr(base_env, PRE_RESET_POS_ATTR, pos)
      pos_eff = torch.where(ended[:, None], pre_reset, pos)
      disp = pos_eff[:, :2] - spawn_pos[:, :2]
      forward, xy_dist, progress = success_progress(
        disp, spawn_heading, args.success_rule
      )
      ok = active & torch.isfinite(xy_dist)
      max_forward = torch.where(ok, torch.maximum(max_forward, forward), max_forward)
      forward_end = torch.where(ok, forward, forward_end)
      max_xy_disp = torch.where(ok, torch.maximum(max_xy_disp, xy_dist), max_xy_disp)
      active &= ~ended

      # still-running envs
      heading = robot.data.heading_w
      v_w = robot.data.root_link_lin_vel_w
      vx_yaw = torch.cos(heading) * v_w[:, 0] + torch.sin(heading) * v_w[:, 1]
      cmd = base_env.command_manager.get_command("twist")
      err_sum += torch.where(active, (vx_yaw - cmd[:, 0]).abs(), 0.0)

      # base height above the supporting surface: highest ray hit within support_radius of the base
      hits = scan.data.hit_pos_w
      hit_ok = (scan.data.distances >= 0) & torch.isfinite(hits[..., 2])
      hit_z = torch.where(hit_ok, hits[..., 2], -torch.inf)
      near = (
        torch.linalg.norm(hits[..., :2] - pos[:, None, :2], dim=-1)
        <= args.support_radius
      )
      support_z = torch.where(near, hit_z, -torch.inf).max(dim=1).values
      support_z = torch.where(
        torch.isfinite(support_z), support_z, hit_z.max(dim=1).values
      )
      height = pos[:, 2] - support_z  # +inf if no ray hit at all
      finite_h = active & torch.isfinite(height)
      height_sum += torch.where(finite_h, height, 0.0)
      n_height += finite_h.long()
      n_tracked += active.long()
      feet_down = (feet.data.found > 0).sum(dim=1).clamp(max=2)
      n_feet_down += (
        active[:, None] & (feet_down[:, None] == torch.arange(3, device=device))
      ).long()

      if step == 1:
        step1_disp = float(torch.where(active, xy_dist, 0.0).max())
        h_ok = height[finite_h]
        step1_height = float(h_ok.median()) if h_ok.numel() else float("nan")
        spawn_check.update(
          {"step1_disp_max_m": step1_disp, "step1_height_median_m": step1_height}
        )
        problems = []
        if spawn_check["spawn_offset_max_m"] > SPAWN_TOL_M:
          problems.append(
            f"reset position up to {spawn_check['spawn_offset_max_m']:.3f} m from tile center"
          )
        if step1_disp > STEP1_DISP_TOL_M:
          problems.append(f"displacement {step1_disp:.3f} m after the first step")
        if not STEP1_HEIGHT_RANGE_M[0] <= step1_height <= STEP1_HEIGHT_RANGE_M[1]:
          problems.append(
            f"median height above support {step1_height:.3f} m after the first step"
          )
        spawn_check["ok"] = not problems
        spawn_check["problem"] = "; ".join(problems) or None

      # success first, then falls (falls must persist for hold_steps)
      succeeded = active & (progress >= args.success_dist)
      outcome[succeeded] = SUCCESS
      time_to_success[succeeded] = step * dt
      active &= ~succeeded

      tilted = -robot.data.projected_gravity_b[:, 2] < cos_tilt
      tilt_cnt = torch.where(active & tilted, tilt_cnt + 1, 0)
      low_cnt = torch.where(active & (height < args.fall_height), low_cnt + 1, 0)
      if args.spawn_drop_rule:
        dropped = pos[:, 2] < spawn_pos[:, 2] - args.spawn_drop
        drop_cnt = torch.where(active & dropped, drop_cnt + 1, 0)
      for reason, cnt in ((2, tilt_cnt), (3, low_cnt), (4, drop_cnt)):
        fell = active & (cnt >= hold_steps)
        outcome[fell] = FAIL
        fail_reason[fell] = reason
        fail_time[fell] = step * dt
        active &= ~fell

      if step in (1, early_steps):
        abort_if_invalid(step)
      if step % 50 == 0 and cuda:
        cuda_peak = max(cuda_peak, cuda_used_mb())
      if step % 100 == 0:
        print(
          f"[INFO] step {step}/{episode_steps}  active={int(active.sum())}  "
          f"success={int((outcome == SUCCESS).sum())}  fail={int((outcome == FAIL).sum())}",
          flush=True,
        )

  truncated = bool(active.any())  # only if --max_steps cut the run short
  validity_check = abort_if_invalid(max(step, early_steps))
  outcome[active] = STUCK
  wall_time = time.time() - t_start
  if cuda:
    cuda_peak = max(cuda_peak, cuda_used_mb())

  # -- per-env table
  try:
    foot = [
      i
      for i, n in enumerate(robot.geom_names)
      if n.endswith("_collision") and "_foot" in n
    ]
    gid = robot.indexing.geom_ids[foot].long()
    fr = (
      base_env.sim.model.geom_friction
    )  # mjlab TorchArray, [nworld, ngeom, 3] after expansion
    fr = fr[:, gid, 0] if fr.dim() == 3 else fr[gid, 0].expand(num_envs, -1)
    friction = fr.float().mean(dim=-1).cpu().tolist()
  except Exception as e:
    print(f"[WARN] could not read foot friction: {e}")
    friction = [float("nan")] * num_envs
  mean_err = (err_sum / n_tracked.clamp(min=1)).cpu()
  mean_height = torch.where(
    n_height > 0, height_sum / n_height.clamp(min=1), torch.nan
  ).cpu()
  gait_frac = (n_feet_down / n_tracked.clamp(min=1)[:, None]).cpu().tolist()
  outcome_c, reason_c = outcome.cpu().tolist(), fail_reason.cpu().tolist()
  tracked_c, body_c, mask_c = (
    n_tracked.cpu().tolist(),
    contact_body.cpu().tolist(),
    term_mask.cpu().tolist(),
  )
  rows = []
  for i in range(num_envs):
    survivor = outcome_c[i] != FAIL and tracked_c[i] > 0
    rows.append(
      {
        "env_id": i,
        "tile_row": tile_rows[i],
        "tile_col": tile_cols[i],
        "outcome": OUTCOME_NAMES[outcome_c[i]],
        "fail_reason": FAIL_REASONS[reason_c[i]],
        "contact_body": contact_body_names[body_c[i]] if body_c[i] >= 0 else "",
        "contact_partner": contact_partner[i],
        "termination_term": "+".join(
          n for k, n in enumerate(fail_terms) if mask_c[i] >> k & 1
        ),
        "fail_time_s": float(fail_time[i]),
        "time_to_success_s": float(time_to_success[i]),
        "max_forward_m": float(max_forward[i]),
        "forward_at_end_m": float(forward_end[i]),
        "max_xy_disp_m": float(max_xy_disp[i]),
        "mean_abs_vx_err": float(mean_err[i]) if survivor else float("nan"),
        "tracked_steps": tracked_c[i],
        "double_support_frac": gait_frac[i][2] if tracked_c[i] else float("nan"),
        "single_support_frac": gait_frac[i][1] if tracked_c[i] else float("nan"),
        "flight_frac": gait_frac[i][0] if tracked_c[i] else float("nan"),
        "mean_height_above_support_m": float(mean_height[i]),
        "static_friction": float(friction[i]),
      }
    )

  # -- summary
  n = num_envs
  counts = {
    name: sum(r["outcome"] == name for r in rows)
    for name in ("success", "fail", "stuck")
  }
  succ_times = [r["time_to_success_s"] for r in rows if r["outcome"] == "success"]
  surv_errs = [
    r["mean_abs_vx_err"]
    for r in rows
    if r["outcome"] != "fail" and not math.isnan(r["mean_abs_vx_err"])
  ]
  heights = sorted(
    r["mean_height_above_support_m"]
    for r in rows
    if r["tracked_steps"] > 0 and not math.isnan(r["mean_height_above_support_m"])
  )

  def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

  summary = {
    "valid": True,
    "validity_check": validity_check,
    **run_info,
    "checkpoint_iter": ckpt_iter,
    "difficulty": args.difficulty,
    "num_rows": args.num_rows,
    "num_cols": args.num_cols,
    "tiles_distinct_used": tiles_distinct,
    "terrain_geoms": terrain_geoms,
    "com_dr": args.com_dr,
    "success": rate(counts["success"], n),
    "fail": rate(counts["fail"], n),
    "stuck": rate(counts["stuck"], n),
    "fail_reasons": {
      name: sum(row["fail_reason"] == name for row in rows)
      for name in FAIL_REASONS.values()
      if name
    },
    "termination_terms": {
      name: sum(name in row["termination_term"].split("+") for row in rows)
      for name in fail_terms
    },
    "contact_partners": dict(
      Counter(r["contact_partner"] for r in rows if r["contact_partner"])
    ),
    "contact_pairs": dict(
      Counter(
        f"{r['contact_body']} -> {r['contact_partner']}"
        for r in rows
        if r["contact_partner"]
      )
    ),
    "gait_mean": {
      key: mean([r[f"{key}_frac"] for r in rows if r["tracked_steps"] > 0])
      for key in ("double_support", "single_support", "flight")
    },
    "mean_time_to_success_s": mean(succ_times),
    "mean_max_forward_m": mean([r["max_forward_m"] for r in rows]),
    "mean_forward_at_end_m": mean([r["forward_at_end_m"] for r in rows]),
    "mean_max_xy_disp_m": mean([r["max_xy_disp_m"] for r in rows]),
    "mean_abs_vx_err_survivors": mean(surv_errs),
    "n_survivors": len(surv_errs),
    "median_height_above_support_m": heights[len(heights) // 2]
    if heights
    else float("nan"),
    "steps_simulated": step,
    "episode_steps": episode_steps,
    "step_dt": dt,
    "truncated_by_max_steps": truncated,
    "env_build_s": env_build_s,
    "wall_time_s": wall_time,
    "device": device,
    "torch_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20
    if cuda
    else None,
    "cuda_device_used_mb_before": cuda_before,
    "cuda_device_used_mb_peak": cuda_peak,
    "host_max_rss_mb": host_max_rss_mb(),
    "git": git_info(),
  }

  with open(os.path.join(out_dir, "per_env.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
  print(
    f"[INFO] success {summary['success']['rate']:.3f} ± {summary['success']['se']:.3f}  "
    f"fail {summary['fail']['rate']:.3f}  stuck {summary['stuck']['rate']:.3f}  "
    f"(n={n}, distinct tiles={tiles_distinct}, truncated={truncated})"
  )
  print(f"[INFO] Results saved to: {out_dir}")
  env.close()


if __name__ == "__main__":
  main()
