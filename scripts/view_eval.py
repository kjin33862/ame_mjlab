"""Watch ONE robot on one evaluation tile: the exact env of ``scripts/evaluate.py`` (same terrain family, fixed
difficulty, tile-center spawn with yaw 0, fixed vx = 1 m/s command, noise preset), on a 1 x 1 board.

The episode auto-resets after a termination, so the failure repeats.  Without --video it opens mjlab's viewer
(native window if a display is set, else viser in the browser); with --video it renders offscreen and saves an mp4.

Examples:
    uv run python scripts/view_eval.py --checkpoint checkpoints/baseline/model_9999.pt \
        --terrain stepping_stones --difficulty 0.0 --noise off
    uv run python scripts/view_eval.py --checkpoint checkpoints/baseline/model_9999.pt \
        --terrain stepping_stones --difficulty 0.0 --noise off --video --video_s 3
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import resolve_noise  # noqa: E402


def main():
  p = argparse.ArgumentParser(description="Watch one robot on one evaluation tile.")
  p.add_argument("--task", default="Ame-Velocity-Rough-G1")
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--terrain", default="stepping_stones")
  p.add_argument("--difficulty", type=float, default=0.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--noise", default="off", choices=["train", "off"])
  p.add_argument("--episode_s", type=float, default=20.0)
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
  for name in ("drift_cells", "map_delay_steps"):
    p.add_argument(f"--{name}", type=int, default=None)
  p.add_argument("--proprio_noise", default="preset", choices=["preset", "on", "off"])
  args = p.parse_args()

  from dataclasses import asdict

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.utils.wrappers import VideoRecorder
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  from ame.eval import assign_tiles, eval_env_cfg

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
  height_noise, proprio = resolve_noise(args)
  env_cfg = eval_env_cfg(
    terrain=args.terrain,
    num_rows=1,
    num_cols=1,
    difficulty=args.difficulty,
    seed=args.seed,
    episode_s=args.episode_s,
    height_noise=height_noise,
    proprio_noise=proprio,
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
    tag = f"{args.terrain}_d{args.difficulty:g}_{args.noise}"
    env = VideoRecorder(
      base_env,
      video_folder=os.path.join(args.out_dir, tag),
      step_trigger=lambda step: step == 0,
      video_length=round(args.video_s / base_env.step_dt),
      disable_logger=True,
    )
  env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
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
