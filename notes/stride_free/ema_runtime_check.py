"""Live check of the EMA tracking term in the StrideFree env (8 envs, random actions).

A mirror of the average is kept from the pelvis's yaw-frame xy velocity as the
reward manager sees it (captured by wrapping reward_manager.compute: mjlab
computes rewards on derived quantities one physics substep stale, by design),
with the term's own alpha, restarted at the current velocity on the first step
after an env's reset.  The mirror must equal the term's buffer on every env
every step, envs reset in a step must come out flagged, and the next step must
restart them at that step's velocity.

usage: python notes/stride_free/ema_runtime_check.py
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

import ame  # noqa: F401

TASK = (
  "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-StrideFree-Velocity-Rough-G1"
)
N = 8
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = N
cfg.episode_length_s = 2.0  # time-outs every 100 steps, so resets happen
device = "cuda" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg, device=device)
env.reset()
rm = env.reward_manager
term = rm._term_cfgs[rm._term_names.index("track_linear_velocity")].func
print(f"term {type(term).__name__}, alpha {term.alpha:.6f}, step_dt {env.step_dt}")
robot = env.scene["robot"]
dim = env.action_manager.total_action_dim


def v_yaw():
  q, v = robot.data.root_link_quat_w, robot.data.root_link_lin_vel_w
  return quat_apply_inverse(yaw_quat(q), v)[:, :2]


seen = {}
_compute = rm.compute


def compute(dt):
  seen["v"] = v_yaw().clone()
  return _compute(dt)


rm.compute = compute
mirror = torch.zeros(N, 2, device=device)
fresh = torch.ones(N, dtype=torch.bool, device=device)
worst, resets, restarts, checked = 0.0, 0, 0, 0
for _ in range(400):
  _, _, te, tr, _ = env.step(0.3 * torch.randn(N, dim, device=device))
  done = te | tr
  v = seen["v"]
  mirror = torch.where(fresh.unsqueeze(1), v, mirror + term.alpha * (v - mirror))
  worst = max(worst, (term.ema - mirror).abs().max().item())
  checked += N
  restarts += int(fresh.sum())
  assert torch.equal(term.fresh, done), "fresh flag must mark exactly the reset envs"
  resets += int(done.sum())
  fresh = done.clone()
print(f"env-steps compared {checked}, restarts compared {restarts}, resets {resets}")
print(f"max |term.ema - mirror| = {worst:.2e}")
assert worst < 1e-5 and restarts > 0
print("OK")
