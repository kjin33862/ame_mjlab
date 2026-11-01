"""Yaw-rate tracking of a checkpoint under fixed twist commands (headless).

Every command in --commands gets --num_per_cmd robots, all in one env: env ``i`` runs command ``i // num_per_cmd``
and stands on tile ``i % (num_rows * num_cols)`` of a --terrain board at a fixed --difficulty (``ame.eval``'s
evaluation board, spawn at the tile centre with yaw 0). The command is pinned per env inside the twist term's update
(as evaluate.py's goal mode does), so it holds for the whole --hold_s with no resampling. Noise off, mean actions.

Per command, over the window [--skip_s, --hold_s] of each robot that did not terminate before --hold_s:
    yaw_mean      mean over robots of the robot's time-mean world-frame yaw rate (root link ang vel z, rad/s;
                  + = counter-clockwise seen from above, i.e. turning left)
    ratio         yaw_mean / commanded wz (empty for wz = 0)
    yaw_std_in    mean over robots of the std of its yaw rate within the window (gait oscillation)
    yaw_std_across std over robots of their time-mean yaw rate (robot-to-robot spread)
    yaw_b_mean    the same mean from the body-frame z rate (what the training reward tracks), as a check
    vx_mean       mean body-frame forward speed (m/s)
    left_frac     fraction of robots whose time-mean yaw rate is > 0 (for wz = 0: how many drift left)
    fell          robots that terminated before --hold_s (excluded from the statistics)
Writes per_command.csv and summary.json under --out_root/<time>_<model_name>/ and prints the table.

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
  p = argparse.ArgumentParser(description="Yaw-rate tracking under fixed twist commands.")
  p.add_argument("--task", default="Ame-Velocity-Rough-G1", help="Registered task whose policy config to use.")
  p.add_argument("--checkpoint", required=True, help="Checkpoint path (model_<iter>.pt).")
  p.add_argument("--model_name", default=None, help="Label for the output folder. Default: <run dir>_<ckpt name>.")
  p.add_argument(
    "--commands",
    default=DEFAULT_COMMANDS,
    help="Space-separated vx,wz pairs (vy = 0). Default: the 8 commands of the yaw study.",
  )
  p.add_argument("--num_per_cmd", type=int, default=64, help="Robots per command.")
  p.add_argument("--terrain", default="random_rough", help="Terrain family (ame.eval.EVAL_TERRAINS).")
  p.add_argument("--difficulty", type=float, default=0.0)
  p.add_argument("--num_rows", type=int, default=8)
  p.add_argument("--num_cols", type=int, default=8)
  p.add_argument("--hold_s", type=float, default=10.0, help="How long each command is held [s].")
  p.add_argument("--skip_s", type=float, default=2.0, help="Initial transient left out of the statistics [s].")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default=None, help="Default: cuda:0 if available, else cpu.")
  p.add_argument("--out_root", default=os.path.join(REPO_ROOT, "logs", "eval_results", "yaw_tracking"))
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
  num_cmds, per = len(commands), args.num_per_cmd
  num_envs = num_cmds * per
  print(f"[INFO] {args.model_name}: task {args.task}, actor obs {actor_obs}, {num_cmds} commands x {per} robots")

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

  # Pin each env's command after the twist term's own update (also in the first observation).
  cmd = torch.tensor(commands, device=device, dtype=torch.float).repeat_interleave(per, dim=0)
  term = base_env.command_manager.get_term("twist")
  update = term._update_command

  def pinned_update():
    update()
    term.vel_command_b[:, 0] = cmd[:, 0]
    term.vel_command_b[:, 1] = 0.0
    term.vel_command_b[:, 2] = cmd[:, 1]

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
  alive = torch.ones(num_envs, dtype=torch.bool, device=device)
  yaw_w = torch.zeros(steps - skip, num_envs, device=device)
  yaw_b = torch.zeros_like(yaw_w)
  vx_b = torch.zeros_like(yaw_w)

  obs = env.get_observations()
  with torch.inference_mode():
    for step in range(steps):
      obs, _, dones, _ = env.step(policy(obs))
      alive &= ~dones.bool()  # a reset env's later samples belong to a new episode
      if step >= skip:
        yaw_w[step - skip] = robot.data.root_link_ang_vel_w[:, 2]
        yaw_b[step - skip] = robot.data.root_link_ang_vel_b[:, 2]
        vx_b[step - skip] = robot.data.root_link_lin_vel_b[:, 0]
  assert bool((term.vel_command_b[:, 2] == cmd[:, 1]).all()), "command was not held"

  rows = []
  for c, (vx, wz) in enumerate(commands):
    group = torch.arange(c * per, (c + 1) * per, device=device)
    keep = group[alive[group]]
    n = int(keep.numel())
    row = {"vx_cmd": vx, "wz_cmd": wz, "robots": per, "fell": per - n}
    if n:
      per_env = yaw_w[:, keep].mean(dim=0)
      yaw_mean = float(per_env.mean())
      row.update(
        yaw_mean=yaw_mean,
        ratio=yaw_mean / wz if wz != 0 else None,
        yaw_std_in=float(yaw_w[:, keep].std(dim=0).mean()),
        yaw_std_across=float(per_env.std()) if n > 1 else 0.0,
        yaw_b_mean=float(yaw_b[:, keep].mean()),
        vx_mean=float(vx_b[:, keep].mean()),
        left_frac=float((per_env > 0).float().mean()),
      )
    rows.append(row)

  out_dir = os.path.join(os.path.expanduser(args.out_root), f"{datetime.now():%Y%m%d-%H%M%S}_{args.model_name}")
  os.makedirs(out_dir, exist_ok=True)
  fields = [
    "vx_cmd", "wz_cmd", "robots", "fell", "yaw_mean", "ratio", "yaw_std_in", "yaw_std_across",
    "yaw_b_mean", "vx_mean", "left_frac",
  ]  # fmt: skip
  with open(os.path.join(out_dir, "per_command.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump({"argv": sys.argv, "args": vars(args), "actor_obs": actor_obs, "checkpoint": checkpoint,
               "step_dt": dt, "rows": rows}, f, indent=2)  # fmt: skip

  def fmt(v, spec: str) -> str:
    width = int(spec.lstrip("+").split(".")[0])
    return "-".rjust(width) if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)

  print(f"\n{args.model_name}  ({args.terrain} d={args.difficulty:g}, window {args.skip_s:g}-{args.hold_s:g} s)")
  print(f"{'vx':>4} {'wz':>5} | {'yaw_mean':>8} {'ratio':>6} {'std_in':>6} {'std_x':>6} {'yaw_b':>7} {'vx':>5} "
        f"{'left':>5} {'fell':>4}")  # fmt: skip
  for r in rows:
    print(
      f"{r['vx_cmd']:>4g} {r['wz_cmd']:>+5g} | {fmt(r.get('yaw_mean'), '+8.3f')} {fmt(r.get('ratio'), '6.2f')} "
      f"{fmt(r.get('yaw_std_in'), '6.3f')} {fmt(r.get('yaw_std_across'), '6.3f')} {fmt(r.get('yaw_b_mean'), '+7.3f')} "
      f"{fmt(r.get('vx_mean'), '5.2f')} {fmt(r.get('left_frac'), '5.2f')} {r['fell']:>4d}"
    )
  print(f"\n[INFO] wrote {out_dir}")
  env.close()


if __name__ == "__main__":
  main()
