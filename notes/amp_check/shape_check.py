"""AMP observation layout: expert (MotionData from the npz) vs our simulated robot.

Builds task 1's env (8 envs), writes clip frames into the robot (root pose and
velocity from the pelvis, joint pos / vel), runs sim.forward, and compares the
robot-side observation (AmpDoubleCriticPPO.robot_amp_obs's function on the live
body state) with MotionData's frame for the same clip / index."""
import os
import numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import ame
from ame.amp import AMP_ANCHOR_NAME, AMP_BODY_NAMES, AMP_CLIPS, MotionData, amp_observation, resolve_motion_dir

TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-AMP-Velocity-Rough-G1"
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = 8
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
robot = env.scene["robot"]
names = list(robot.body_names)
motion = MotionData(resolve_motion_dir("~/ref/amp_mjlab_npz/amp"), AMP_CLIPS, names, device="cuda:0")
print("robot bodies", len(names), "| expert frame dim", motion.dim, "| clips", len(motion.names))
ids = [names.index(n) for n in AMP_BODY_NAMES]; a = names.index(AMP_ANCHOR_NAME)
rng = np.random.default_rng(0)
worst = 0.0
for k in range(8):
  c = motion.names[(3 * k) % len(motion.names)]
  z = np.load(resolve_motion_dir("~/ref/amp_mjlab_npz/amp") / f"{c}.npz")
  t = int(rng.integers(1, len(z["joint_pos"]) - 1))
  root = robot.data.default_root_state.clone()
  dev = root.device
  root[:, :3] = torch.tensor(z["body_pos_w"][t, 0], device=dev) + env.scene.env_origins
  root[:, 3:7] = torch.tensor(z["body_quat_w"][t, 0], device=dev)
  root[:, 7:10] = torch.tensor(z["body_lin_vel_w"][t, 0], device=dev)
  root[:, 10:13] = torch.tensor(z["body_ang_vel_w"][t, 0], device=dev)
  robot.write_root_state_to_sim(root)
  robot.write_joint_state_to_sim(
    torch.tensor(z["joint_pos"][t], device=dev).expand(8, -1).contiguous(),
    torch.tensor(z["joint_vel"][t], device=dev).expand(8, -1).contiguous(),
  )
  env.sim.forward()
  env.scene.update(env.sim.mj_model.opt.timestep)
  d = robot.data
  obs = amp_observation(d.body_link_pos_w[:, ids], d.body_link_quat_w[:, ids], d.body_link_lin_vel_w[:, ids],
                        d.body_link_ang_vel_w[:, ids], d.body_link_pos_w[:, a], d.body_link_quat_w[:, a])
  exp = motion.frames[motion.names.index(c)][t]
  err = (obs - exp).abs().max(dim=1).values.max().item()
  seg = [(obs[0, s:e] - exp[s:e]).abs().max().item() for s, e in ((0, 39), (39, 117), (117, 156), (156, 195))]
  worst = max(worst, err)
  print(f"{c} t={t}: robot obs {tuple(obs.shape)}, expert {tuple(exp.shape)}, max |diff| {err:.2e} (pos/ori/lin/ang {', '.join(f'{x:.1e}' for x in seg)})")
print("WORST", worst)
