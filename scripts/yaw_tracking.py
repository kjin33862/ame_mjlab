"""Yaw-rate tracking of a checkpoint under fixed twist commands (headless).

One env of --num_per_cmd robots on a --terrain board at a fixed --difficulty (``ame.eval``'s evaluation board: env
``i`` on tile ``i % (num_rows * num_cols)``, spawn at the tile centre with yaw 0). The commands in --commands run one
after another, each from a full reset: the command is pinned for every env inside the twist term's update (as
evaluate.py's goal mode does), so it holds for the whole --hold_s with no resampling. Noise off, mean actions.

Why not every command in one env: with 512 envs on the stone_lattice board the physics silently goes wrong (robots
commanded vx = 0 slide at 1-2.6 m/s with no body rotation, no warning; likely dropped contacts); with 64 it is normal.
One robot per tile, the evaluate.py setup, is the default here. The lattice board is slow (8 x 8 tiles, ~21k geoms:
~1.2 s per step for 64 envs on an RTX 5060 Laptop, ~80 min for the 8 default commands at 10 s).

Per command, over the window [--skip_s, --hold_s] of each robot that did not terminate before --hold_s:
    yaw_mean      mean over robots of the robot's time-mean world-frame yaw rate (root link ang vel z, rad/s;
                  + = counter-clockwise seen from above, i.e. turning left)
    ratio         yaw_mean / commanded wz (empty for wz = 0)
    yaw_std_in    mean over robots of the std of its yaw rate within the window (gait oscillation)
    yaw_std_across std over robots of their time-mean yaw rate (robot-to-robot spread)
    yaw_b_mean    the same mean from the body-frame z rate (what the training reward tracks), as a check
    vx_mean       mean body-frame forward speed (m/s)
    left_frac     fraction of robots whose time-mean yaw rate is > 0 (for wz = 0: how many drift left)
    rp2_mean      mean of roll rate^2 + pitch rate^2 (body frame, (rad/s)^2): the xy term of the yaw reward
    kernel        mean of mjlab's track_angular_velocity kernel exp(-(z_err^2 + xy^2) / std^2) on the body-frame
                  rates, z_err = wz command - body yaw rate, std = --kernel_std
    kernel_z0     the same kernel with z_err = 0: the best the robot could get with this roll / pitch motion
    fell          robots that terminated before --hold_s (excluded from the statistics)
Writes per_command.csv and summary.json under --out_root/<time>_<model_name>/, rewritten after every command (a
stopped run keeps the commands it finished; summary.json's "complete" says whether all ran), and prints the table.

Examples:
    uv run python scripts/yaw_tracking.py --task Ame-Mjlab-Gap-Velocity-Rough-G1 \\
      --checkpoint checkpoints/gap_progress_heading_1024/model_9999.pt --model_name gap_progress_heading_1024
    uv run python scripts/yaw_tracking.py --task Ame-Velocity-Rough-G1 \\
      --checkpoint checkpoints/baseline/model_9999.pt --model_name baseline
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_COMMANDS = "0,0.5 0,-0.5 0,1 0,-1 0.5,0.5 0.5,-0.5 0.5,0 1,0"


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="Yaw-rate tracking under fixed twist commands."
  )
  p.add_argument(
    "--task",
    default="Ame-Velocity-Rough-G1",
    help="Registered task whose policy config to use.",
  )
  p.add_argument(
    "--checkpoint", required=True, help="Checkpoint path (model_<iter>.pt)."
  )
  p.add_argument(
    "--model_name",
    default=None,
    help="Label for the output folder. Default: <run dir>_<ckpt name>.",
  )
  p.add_argument(
    "--commands",
    default=DEFAULT_COMMANDS,
    help="Space-separated vx,wz pairs (vy = 0). Default: the 8 commands of the yaw study.",
  )
  p.add_argument("--num_per_cmd", type=int, default=64, help="Robots per command.")
  p.add_argument(
    "--terrain", default="random_rough", help="Terrain family (ame.eval.EVAL_TERRAINS)."
  )
  p.add_argument("--difficulty", type=float, default=0.0)
  p.add_argument("--num_rows", type=int, default=8)
  p.add_argument("--num_cols", type=int, default=8)
  p.add_argument(
    "--hold_s", type=float, default=10.0, help="How long each command is held [s]."
  )
  p.add_argument(
    "--skip_s",
    type=float,
    default=2.0,
    help="Initial transient left out of the statistics [s].",
  )
  p.add_argument(
    "--kernel_std",
    type=float,
    default=None,
    help="std of the yaw reward kernel. Default: the gap task's track_angular_velocity std.",
  )
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--device", default=None, help="Default: cuda:0 if available, else cpu."
  )
  p.add_argument(
    "--out_root",
    default=os.path.join(REPO_ROOT, "logs", "eval_results", "yaw_tracking"),
  )
  return p


def parse_commands(text: str) -> list[tuple[float, float]]:
  pairs = []
  for item in text.split():
    vx, wz = (float(v) for v in item.split(","))
    pairs.append((vx, wz))
  return pairs


def main():
  args = build_parser().parse_args()
  commands = parse_commands(args.commands)

  from dataclasses import asdict

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  from ame.env_cfg import ame_gap_env_cfg
  from ame.eval import assign_tiles, checkpoint_actor_obs, eval_env_cfg
  from ame.eval.env_cfg import OFF_HEIGHT_NOISE

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
  if not os.path.isfile(checkpoint):
    sys.exit(f"[ERROR] checkpoint not found: {checkpoint}")
  if args.model_name is None:
    run_dir = os.path.basename(os.path.dirname(checkpoint))
    args.model_name = f"{run_dir}_{os.path.splitext(os.path.basename(checkpoint))[0]}"
  actor_obs = checkpoint_actor_obs(checkpoint)
  if args.kernel_std is None:
    args.kernel_std = float(
      ame_gap_env_cfg().rewards["track_angular_velocity"].params["std"]
    )
  num_cmds, per = len(commands), args.num_per_cmd
  num_envs = per
  print(
    f"[INFO] {args.model_name}: task {args.task}, actor obs {actor_obs}, {num_cmds} commands x {per} robots"
  )

  # Episodes a little longer than the hold, so no time-out reset lands inside the window.
  env_cfg = eval_env_cfg(
    terrain=args.terrain,
    num_rows=args.num_rows,
    num_cols=args.num_cols,
    difficulty=args.difficulty,
    seed=args.seed,
    episode_s=args.hold_s + 1.0,
    height_noise=dict(OFF_HEIGHT_NOISE),
    proprio_noise=False,
    command=(0.0, 0.0, 0.0),
    actor_obs=actor_obs,
  )
  env_cfg.scene.num_envs = num_envs
  rl_cfg = load_rl_cfg(args.task)
  rl_cfg.seed = args.seed

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  assign_tiles(base_env)  # env i -> tile i % (rows * cols), before the wrapper's reset

  # Pin the current command after the twist term's own update (also in the first observation after a reset).
  cmd = torch.zeros(2, device=device)  # (vx, wz), set per command below
  term = base_env.command_manager.get_term("twist")
  update = term._update_command

  def pinned_update():
    update()
    term.vel_command_b[:, 0] = cmd[0]
    term.vel_command_b[:, 1] = 0.0
    term.vel_command_b[:, 2] = cmd[1]

  term._update_command = pinned_update
  pinned_update()

  env = RslRlVecEnvWrapper(base_env, clip_actions=rl_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(rl_cfg), device=device)
  print(f"[INFO] Loading model checkpoint from: {checkpoint}")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = base_env.scene["robot"]
  dt = base_env.step_dt
  steps = round(args.hold_s / dt)
  skip = round(args.skip_s / dt)
  var = args.kernel_std**2

  out_dir = os.path.join(
    os.path.expanduser(args.out_root),
    f"{datetime.now():%Y%m%d-%H%M%S}_{args.model_name}",
  )
  os.makedirs(out_dir, exist_ok=True)
  fields = [
    "vx_cmd", "wz_cmd", "robots", "fell", "yaw_mean", "ratio", "yaw_std_in", "yaw_std_across",
    "yaw_b_mean", "vx_mean", "left_frac", "rp2_mean", "kernel", "kernel_z0",
  ]  # fmt: skip

  def write_results(rows: list[dict]) -> None:
    """Rewrite both files with the commands finished so far (a stopped run keeps them)."""
    with open(os.path.join(out_dir, "per_command.csv"), "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fields)
      writer.writeheader()
      writer.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
      json.dump({"argv": sys.argv, "args": vars(args), "actor_obs": actor_obs, "checkpoint": checkpoint,
                 "step_dt": dt, "complete": len(rows) == num_cmds, "rows": rows}, f, indent=2)  # fmt: skip

  rows = []
  for vx, wz in commands:
    cmd[0], cmd[1] = vx, wz
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    yaw_w = torch.zeros(steps - skip, num_envs, device=device)
    yaw_b = torch.zeros_like(yaw_w)
    vx_b = torch.zeros_like(yaw_w)
    rp2 = torch.zeros_like(yaw_w)
    # The reset too: the sensor buffers it clears were made inside inference mode.
    with torch.inference_mode():
      # Every env back to its tile centre, episode length 0, this command pinned.
      obs, _ = env.reset()
      for step in range(steps):
        obs, _, dones, _ = env.step(policy(obs))
        alive &= ~dones.bool()  # a reset env's later samples belong to a new episode
        if step >= skip:
          yaw_w[step - skip] = robot.data.root_link_ang_vel_w[:, 2]
          yaw_b[step - skip] = robot.data.root_link_ang_vel_b[:, 2]
          vx_b[step - skip] = robot.data.root_link_lin_vel_b[:, 0]
          rp2[step - skip] = robot.data.root_link_ang_vel_b[:, :2].square().sum(dim=1)
    held = bool(
      (term.vel_command_b[:, 0] == vx).all() and (term.vel_command_b[:, 2] == wz).all()
    )
    assert held, "command was not held"

    keep = alive.nonzero().flatten()
    n = int(keep.numel())
    row = {"vx_cmd": vx, "wz_cmd": wz, "robots": per, "fell": per - n}
    if n:
      per_env = yaw_w[:, keep].mean(dim=0)
      yaw_mean = float(per_env.mean())
      z_err2 = (wz - yaw_b[:, keep]).square()
      row.update(
        yaw_mean=yaw_mean,
        ratio=yaw_mean / wz if wz != 0 else None,
        yaw_std_in=float(yaw_w[:, keep].std(dim=0).mean()),
        yaw_std_across=float(per_env.std()) if n > 1 else 0.0,
        yaw_b_mean=float(yaw_b[:, keep].mean()),
        vx_mean=float(vx_b[:, keep].mean()),
        left_frac=float((per_env > 0).float().mean()),
        rp2_mean=float(rp2[:, keep].mean()),
        kernel=float(torch.exp(-(z_err2 + rp2[:, keep]) / var).mean()),
        kernel_z0=float(torch.exp(-rp2[:, keep] / var).mean()),
      )
    rows.append(row)
    write_results(rows)
    print(
      f"[INFO] vx {vx:g} wz {wz:+g}: done, {per - n} fell ({len(rows)}/{num_cmds} written)"
    )

  def fmt(v, spec: str) -> str:
    width = int(spec.lstrip("+").split(".")[0])
    return (
      "-".rjust(width)
      if v is None or (isinstance(v, float) and math.isnan(v))
      else format(v, spec)
    )

  print(f"\n{args.model_name}  ({args.terrain} d={args.difficulty:g}, window {args.skip_s:g}-{args.hold_s:g} s, "
        f"kernel std {args.kernel_std:.4f})")  # fmt: skip
  print(f"{'vx':>4} {'wz':>5} | {'yaw_mean':>8} {'ratio':>6} {'std_in':>6} {'std_x':>6} {'yaw_b':>7} {'vx':>5} "
        f"{'left':>5} {'fell':>4} {'rp2':>6} {'kern':>5} {'kern0':>5}")  # fmt: skip
  for r in rows:
    print(
      f"{r['vx_cmd']:>4g} {r['wz_cmd']:>+5g} | {fmt(r.get('yaw_mean'), '+8.3f')} {fmt(r.get('ratio'), '6.2f')} "
      f"{fmt(r.get('yaw_std_in'), '6.3f')} {fmt(r.get('yaw_std_across'), '6.3f')} {fmt(r.get('yaw_b_mean'), '+7.3f')} "
      f"{fmt(r.get('vx_mean'), '5.2f')} {fmt(r.get('left_frac'), '5.2f')} {r['fell']:>4d} "
      f"{fmt(r.get('rp2_mean'), '6.3f')} {fmt(r.get('kernel'), '5.3f')} {fmt(r.get('kernel_z0'), '5.3f')}"
    )
  print(f"\n[INFO] wrote {out_dir}")
  env.close()


if __name__ == "__main__":
  main()
