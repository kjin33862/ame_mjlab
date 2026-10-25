"""Watch ONE robot on one evaluation tile: the exact env of ``scripts/evaluate.py`` (same terrain family, fixed
difficulty, tile-center spawn with yaw 0, fixed vx = 1 m/s command, noise preset), on a 1 x 1 board.

The episode auto-resets after a termination, so the failure repeats.  Without --video it opens mjlab's viewer
(native window if a display is set, else viser in the browser); with --video it renders offscreen and saves an mp4.

Examples:
    uv run python scripts/view_eval.py --checkpoint checkpoints/baseline/model_9999.pt \
        --terrain stepping_stones --difficulty 0.0 --noise off
    uv run python scripts/view_eval.py --checkpoint checkpoints/baseline/model_9999.pt \
        --terrain stepping_stones --difficulty 0.0 --noise off --video --video_s 3
    uv run python scripts/view_eval.py --checkpoint checkpoints/shuang_gap2048/model_14999.pt \
        --terrain stone_lattice --difficulty 0.5 --noise off --video --out_dir logs/eval_videos/shuang_gap2048
        # the gap task's lattice (gap = 0.45 * d); spawn on the stone nearest the tile centre
    uv run python scripts/view_eval.py --checkpoint checkpoints/gap_1024/model_9999.pt \
        --terrain pyramid_stairs_inv --difficulty 0.25 --noise off --cmd_mode goal --video --video_s 8
        # closed-loop wz toward a goal 10 m ahead (evaluate.py --cmd_mode goal), drawn as a red sphere

Videos go to <out_dir>/<model_name>_<terrain>_d<d>_<noise>_<cmd_mode>[...]/ (model_name: as in evaluate.py).
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import (  # noqa: E402
  goal_points,
  install_goal_commands,
  resolve_noise,
)

GOAL_RADIUS_M = 0.1
GOAL_COLOR = (1.0, 0.2, 0.2, 0.9)


def main():
  p = argparse.ArgumentParser(description="Watch one robot on one evaluation tile.")
  p.add_argument("--task", default="Ame-Velocity-Rough-G1")
  p.add_argument("--checkpoint", required=True)
  p.add_argument(
    "--model_name",
    default=None,
    help="Label in the video folder name. Default: <run dir>_<ckpt name>, as in evaluate.py.",
  )
  p.add_argument(
    "--terrain",
    default="stepping_stones",
    help="As in evaluate.py: a family of the AME mix, or stone_lattice.",
  )
  p.add_argument("--difficulty", type=float, default=0.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--noise", default="off", choices=["train", "off"])
  p.add_argument("--episode_s", type=float, default=20.0)
  p.add_argument(
    "--commands",
    default="fixed",
    choices=["fixed", "train"],
    help="As in evaluate.py: fixed vx = 1 m/s, or the gap task's random training twist commands.",
  )
  p.add_argument(
    "--cmd_mode",
    default="open",
    choices=["open", "goal"],
    help="As in evaluate.py: open (fixed twist) or goal (closed-loop wz toward a goal point).",
  )
  p.add_argument("--goal_dist", type=float, default=10.0)
  p.add_argument("--goal_k", type=float, default=1.0)
  p.add_argument(
    "--spawn_asym",
    type=float,
    nargs="?",
    const=0.1,
    default=0.0,
    help="As in evaluate.py: left_hip_pitch_joint offset at spawn [rad] (flag alone: 0.1).",
  )
  p.add_argument("--viewer", default="auto", choices=["auto", "native", "viser"])
  p.add_argument(
    "--video", action="store_true", help="Save an mp4 instead of opening a viewer."
  )
  p.add_argument(
    "--video_s", type=float, default=4.0, help="Video length [s of sim time]."
  )
  p.add_argument("--video_width", type=int, default=960)
  p.add_argument("--video_height", type=int, default=720)
  p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "logs", "eval_videos"))
  p.add_argument("--device", default=None)
  # the evaluate.py noise overrides, so that resolve_noise() sees the same fields
  for name in ("height_sigma", "height_offset"):
    p.add_argument(f"--{name}", type=float, default=None)
  p.add_argument(
    "--actor_obs",
    default="auto",
    choices=["auto", "full", "deploy"],
    help="Actor observation set. full: with base_lin_vel (the AME tasks); deploy: without it (the gap-board "
    "task). auto: from the checkpoint's actor input width.",
  )
  for name in ("drift_cells", "map_delay_steps"):
    p.add_argument(f"--{name}", type=int, default=None)
  p.add_argument("--proprio_noise", default="preset", choices=["preset", "on", "off"])
  args = p.parse_args()
  if args.cmd_mode == "goal" and args.commands != "fixed":
    p.error("--cmd_mode goal needs --commands fixed")

  from dataclasses import asdict

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.utils.wrappers import VideoRecorder
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  from ame.eval import assign_tiles, checkpoint_actor_obs, eval_env_cfg

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
  if args.model_name is None:
    run_dir = os.path.basename(os.path.dirname(checkpoint))
    args.model_name = f"{run_dir}_{os.path.splitext(os.path.basename(checkpoint))[0]}"
  height_noise, proprio = resolve_noise(args)
  if args.actor_obs == "auto":
    args.actor_obs = checkpoint_actor_obs(checkpoint)
  env_cfg = eval_env_cfg(
    terrain=args.terrain,
    num_rows=1,
    num_cols=1,
    difficulty=args.difficulty,
    seed=args.seed,
    episode_s=args.episode_s,
    height_noise=height_noise,
    proprio_noise=proprio,
    spawn_asym=args.spawn_asym,
    actor_obs=args.actor_obs,
    commands=args.commands,
  )
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = args.video_width
  env_cfg.viewer.height = args.video_height
  rl_cfg = load_rl_cfg(args.task)

  base_env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if args.video else None
  )
  assign_tiles(base_env)
  env = base_env
  if args.video:
    tag = f"{args.model_name}_{args.terrain}_d{args.difficulty:g}_{args.noise}_{args.cmd_mode}"
    if args.spawn_asym:
      tag += f"_asym{args.spawn_asym:g}"
    if args.commands != "fixed":
      tag += f"_cmd{args.commands}"
    env = VideoRecorder(
      base_env,
      video_folder=os.path.join(args.out_dir, tag),
      step_trigger=lambda step: step == 0,
      video_length=round(args.video_s / base_env.step_dt),
      disable_logger=True,
    )
  env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  if args.cmd_mode == "goal":  # after the wrapper's reset: the spawn pose
    robot = base_env.scene["robot"]
    goal = goal_points(robot.data.root_link_pos_w, robot.data.heading_w, args.goal_dist)
    install_goal_commands(base_env, goal, args.goal_k)
    # the goal at spawn height, drawn by every viewer and the video renderer
    goal_np = torch.cat((goal, robot.data.root_link_pos_w[:, 2:]), dim=-1).cpu().numpy()
    update_visualizers = base_env.update_visualizers

    def draw_goal(visualizer):
      update_visualizers(visualizer)
      for i in visualizer.get_env_indices(base_env.num_envs):
        visualizer.add_sphere(goal_np[i], GOAL_RADIUS_M, GOAL_COLOR, label="goal")

    base_env.update_visualizers = draw_goal
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(
    env, asdict(rl_cfg), device=device
  )
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  if args.video:
    obs = env.get_observations()
    with torch.inference_mode():
      for _ in range(round(args.video_s / base_env.step_dt) + 1):
        obs, _, _, _ = env.step(policy(obs))
    env.close()
    print(f"[INFO] Video saved under: {os.path.join(args.out_dir, tag)}")
    return

  viewer = args.viewer
  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  if viewer == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()


if __name__ == "__main__":
  main()
