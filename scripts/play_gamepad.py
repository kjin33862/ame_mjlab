"""Drive a trained policy with a gamepad in mjlab's play viewer.

The play environment of ``--task`` (as ``uv run play``), or with ``--terrain`` one evaluation tile of
``scripts/view_eval.py`` (one terrain family at one difficulty, noise off).  Every control step the velocity
command of the viewed env is set from the gamepad, after the command term's own update (the hook mjlab's viser
joystick sliders use), so the policy observes it and the command arrows show it.

    left stick up/down     lin_vel_x   (up = forward)
    left stick left/right  lin_vel_y   (left = +y)
    right stick left/right ang_vel_z   (left = counter-clockwise)
    A                      reset the robot
    B                      switch between the gamepad and the task's random commands

Each axis is scaled to the task's command range (stick fully up = the range's max, fully down = its min), with
a dead zone.  The gamepad is read from the Linux joystick API (/dev/input/js*), no extra packages.

Examples:
    uv run python scripts/play_gamepad.py --task Ame-Mjlab-Gap-Velocity-Rough-G1 \
        --checkpoint-file checkpoints/gap_1024/model_9999.pt
    uv run python scripts/play_gamepad.py --task Ame-Mjlab-Gap-Velocity-Rough-G1 \
        --checkpoint-file checkpoints/gap_1024/model_9999.pt --terrain pyramid_stairs_inv --difficulty 0.25
"""

import argparse
import array
import collections
import fcntl
import glob
import os
import struct
import threading
import time

# linux/joystick.h
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12
JSIOCGAXMAP = 0x80406A32
JSIOCGBTNMAP = 0x84006A34
ABS_X, ABS_Y, ABS_RX = 0x00, 0x01, 0x03
BTN_A, BTN_B = 0x130, 0x131
AXIS_MAX = 32767.0


def _js_maps(fd: int) -> tuple[list[int], list[int]]:
  """The input codes of the device's axes and buttons, in joystick-API index order."""
  n = array.array("B", [0])
  fcntl.ioctl(fd, JSIOCGAXES, n)
  num_axes = n[0]
  fcntl.ioctl(fd, JSIOCGBUTTONS, n)
  num_buttons = n[0]
  axmap = array.array("B", [0] * 0x40)
  fcntl.ioctl(fd, JSIOCGAXMAP, axmap)
  btnmap = array.array("H", [0] * 0x200)
  fcntl.ioctl(fd, JSIOCGBTNMAP, btnmap)
  return list(axmap[:num_axes]), list(btnmap[:num_buttons])


def find_gamepad() -> str | None:
  """The first /dev/input/js* with two sticks and an A button (skips keyboards and mice that also
  register as joysticks)."""
  for path in sorted(glob.glob("/dev/input/js*")):
    try:
      fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
      continue
    try:
      axes, buttons = _js_maps(fd)
    finally:
      os.close(fd)
    if {ABS_X, ABS_Y, ABS_RX} <= set(axes) and BTN_A in buttons:
      return path
  return None


class Gamepad:
  """Reads a joystick device on a background thread: stick positions in [-1, 1] (right / down positive, as
  the device reports them) and a queue of button presses.  Reopens the device after an unplug."""

  def __init__(self, path: str):
    self.path = path
    self.sticks = {ABS_X: 0.0, ABS_Y: 0.0, ABS_RX: 0.0}
    self.presses: collections.deque[int] = collections.deque()
    self.connected = False
    threading.Thread(target=self._run, daemon=True).start()

  def _run(self) -> None:
    while True:
      try:
        fd = os.open(self.path, os.O_RDONLY)
      except OSError:
        time.sleep(1.0)
        continue
      try:
        axes, buttons = _js_maps(fd)
        self.connected = True
        print(f"[gamepad] connected: {self.path}")
        while True:
          _, value, kind, index = struct.unpack("IhBB", os.read(fd, 8))
          if kind & JS_EVENT_AXIS and axes[index] in self.sticks:
            self.sticks[axes[index]] = value / AXIS_MAX
          elif kind == JS_EVENT_BUTTON and value == 1:  # a press, not the initial state
            self.presses.append(buttons[index])
      except OSError:
        pass
      finally:
        os.close(fd)
      self.connected = False
      for code in self.sticks:
        self.sticks[code] = 0.0
      print(f"[gamepad] disconnected: {self.path}")


def _dead_zone(value: float, dead_zone: float) -> float:
  """0 inside the dead zone, rescaled to reach +-1 at full deflection outside it."""
  mag = abs(value)
  if mag <= dead_zone:
    return 0.0
  return min(1.0, (mag - dead_zone) / (1.0 - dead_zone)) * (1.0 if value > 0 else -1.0)


def _to_range(value: float, rng: tuple[float, float]) -> float:
  """[-1, 1] -> the range: +1 -> max, -1 -> min, 0 -> 0 (asymmetric ranges keep their two ends)."""
  return value * rng[1] if value >= 0 else -value * rng[0]


def main():
  p = argparse.ArgumentParser(
    description="Drive a trained policy with a gamepad in the play viewer."
  )
  p.add_argument("--task", required=True)
  p.add_argument("--checkpoint-file", required=True)
  p.add_argument("--viewer", default="auto", choices=["auto", "native", "viser"])
  p.add_argument(
    "--terrain",
    default=None,
    help="One evaluation tile (as view_eval.py: a family of the AME mix, or stone_lattice) instead of "
    "the task's play board.",
  )
  p.add_argument("--difficulty", type=float, default=0.0, help="With --terrain.")
  p.add_argument(
    "--js-device",
    default=None,
    help="Joystick device. Default: the first gamepad found.",
  )
  p.add_argument(
    "--dead-zone",
    type=float,
    default=0.1,
    help="Stick dead zone, fraction of full deflection.",
  )
  p.add_argument(
    "--device", default=None, help="Torch device. Default: cuda:0 if available."
  )
  args = p.parse_args()

  js_path = args.js_device or find_gamepad()
  if js_path is None:
    # A pad that drops off USB is gone for about a second; wait for it.
    print("[gamepad] waiting up to 30 s for a gamepad ...")
    deadline = time.monotonic() + 30.0
    while js_path is None and time.monotonic() < deadline:
      time.sleep(0.5)
      js_path = find_gamepad()
  if js_path is None:
    p.error("no gamepad found under /dev/input/js*; pass --js-device")
  pad = Gamepad(js_path)

  from dataclasses import asdict

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  import ame  # noqa: F401  (registers the tasks)

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint_file))
  if not os.path.exists(checkpoint):
    p.error(f"checkpoint not found: {checkpoint}")

  if args.terrain is None:
    env_cfg = load_env_cfg(args.task, play=True)
  else:
    from ame.eval import (
      OFF_HEIGHT_NOISE,
      assign_tiles,
      checkpoint_actor_obs,
      eval_env_cfg,
    )

    # The view_eval.py tile with the gap task's training commands (what B switches back to), no noise and
    # no time limit.
    env_cfg = eval_env_cfg(
      terrain=args.terrain,
      num_rows=1,
      num_cols=1,
      difficulty=args.difficulty,
      episode_s=1e9,
      height_noise=dict(OFF_HEIGHT_NOISE),
      proprio_noise=False,
      actor_obs=checkpoint_actor_obs(checkpoint),
      commands="train",
    )
    env_cfg.commands["twist"].debug_vis = True
  env_cfg.scene.num_envs = 1
  rl_cfg = load_rl_cfg(args.task)

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  if args.terrain is not None:
    assign_tiles(base_env)
  env = RslRlVecEnvWrapper(base_env, clip_actions=rl_cfg.clip_actions)
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(
    env, asdict(rl_cfg), device=device
  )
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  twist = base_env.command_manager.get_term("twist")
  ranges = twist.cfg.ranges
  state = {"gamepad": True, "env_idx": 0}

  # After the term's own update (resampling, heading control, standing, the viser sliders), overwrite the
  # viewed env's command; the observation is computed after this inside env.step.
  term_compute = twist.compute

  def compute(dt: float) -> None:
    term_compute(dt)
    if not state["gamepad"]:
      return
    sticks = {code: _dead_zone(v, args.dead_zone) for code, v in pad.sticks.items()}
    i = state["env_idx"]
    twist.vel_command_b[i, 0] = _to_range(-sticks[ABS_Y], ranges.lin_vel_x)
    twist.vel_command_b[i, 1] = _to_range(-sticks[ABS_X], ranges.lin_vel_y)
    twist.vel_command_b[i, 2] = _to_range(-sticks[ABS_RX], ranges.ang_vel_z)

  twist.compute = compute

  def mode_name() -> str:
    return "gamepad" if state["gamepad"] else "random commands"

  def with_gamepad(viewer_cls):
    class GamepadViewer(viewer_cls):
      def tick(self) -> bool:
        scene = getattr(self, "_scene", None)  # viser keeps the viewed env on its scene
        state["env_idx"] = scene.env_idx if scene is not None else self.env_idx
        while pad.presses:
          button = pad.presses.popleft()
          if button == BTN_A:
            print("[gamepad] reset")
            self.request_reset()
          elif button == BTN_B:
            state["gamepad"] = not state["gamepad"]
            if not state["gamepad"]:
              # The gamepad wrote over the sampled command; resample it next step.
              twist.time_left[state["env_idx"]] = 0.0
            print(f"[gamepad] mode: {mode_name()}")
        return super().tick()

    return GamepadViewer

  print(
    f"[gamepad] {js_path}; ranges vx {ranges.lin_vel_x}, vy {ranges.lin_vel_y}, wz {ranges.ang_vel_z}, "
    f"dead zone {args.dead_zone}\n"
    "[gamepad] left stick: vx / vy, right stick: wz, A: reset, B: gamepad <-> random commands\n"
    f"[gamepad] mode: {mode_name()}"
  )
  viewer = args.viewer
  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  if viewer == "native":
    with_gamepad(NativeMujocoViewer)(env, policy).run()
  else:
    with_gamepad(ViserPlayViewer)(env, policy).run()
  env.close()


if __name__ == "__main__":
  main()
