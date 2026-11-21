"""Live check of the GoalReach env (64 envs, GPU if available).

1. Goals: per family, the goal's offset from the tile centre (distance,
   bearing, Chebyshev), its offset from the spawn, the stone snap on the
   lattice, the heading range, and the cone at step 0.
2. Observations: the actor's goal block equals the clip of the critic's, the
   critic's time_left decreases by dt / T per step.
3. Zero actions: the robots fall within ~1 s, so for this check the fall
   terminations (illegal_contact, fell_over) are removed and the robots lie
   still.  The official check is a 3-D displacement, and the fall itself
   (root ~0.75 m -> ~0.1 m) counts as moving in the first window (steps
   1-251), so stagnation must fire at step 501 for envs whose goal is > 1 m
   away, and
   the termination penalty must be -500 x dt = -10 per terminated env.
4. Rewards: goal_position / goal_heading are zero before the last 4 s / 2 s.

usage: python notes/goal_reach/goal_runtime_check.py
"""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import ame  # noqa: F401

TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-Contact-FootholdD30-GoalReach-Rough-G1"
N = 64
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = N
cfg.events.pop("push_robot", None)  # a still robot must stay put
cfg.terminations.pop("illegal_contact")
cfg.terminations.pop("fell_over")
device = "cuda" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg, device=device)
env.reset()
goal = env.command_manager.get_term("goal")
terrain = env.scene.terrain
robot = env.scene["robot"]
names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
print(f"step_dt {env.step_dt}  max_episode_length {env.max_episode_length}")
print(f"cone at step 0: {goal.cone_deg():.1f} deg")

centre = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
off = goal.pos_w[:, :2] - centre[:, :2]
spawn_off = goal.pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
for f, name in enumerate(names):
  m = terrain.terrain_types == f
  if not bool(m.any()):
    continue
  d = off[m].norm(dim=1)
  bearing = torch.rad2deg(torch.atan2(off[m, 1], off[m, 0]))
  cheb = off[m].abs().amax(dim=1)
  ds = spawn_off[m].norm(dim=1)
  print(
    f"{name:20s} n={int(m.sum()):3d} d_centre [{d.min():.2f}, {d.max():.2f}]"
    f" bearing [{bearing.min():+.1f}, {bearing.max():+.1f}] deg"
    f" cheb max {cheb.max():.2f} d_spawn [{ds.min():.2f}, {ds.max():.2f}]"
  )
lat = goal.is_lattice[terrain.terrain_types]
if bool(lat.any()):
  stones = goal.stones[terrain.terrain_levels, terrain.terrain_types][lat]
  gap = (stones - goal.pos_w[lat, None, :2]).norm(dim=-1).min(dim=1).values
  print(f"lattice goals: max distance to a stone centre {gap.max():.2e} m")
h = goal.heading_w
print(f"heading_w range [{math.degrees(h.min()):.1f}, {math.degrees(h.max()):.1f}] deg")

obs = env.observation_manager.compute()
actor, critic = obs["actor"], obs["critic"]
a_goal = actor[:, -4:]
lin = 3  # critic leads with base_lin_vel
c_off = lin + 3 + 3 + 29 * 3
c_goal = critic[:, c_off : c_off + 4]
c_time = critic[:, c_off + 4]
dist = c_goal[:, :2].norm(dim=1)
near = dist <= 2.0
clip = torch.where(near[:, None], c_goal[:, :2], c_goal[:, :2] * (2.0 / dist)[:, None])
print(
  f"actor goal xy == clip(critic goal xy): {torch.allclose(a_goal[:, :2], clip, atol=1e-5)}"
)
print(
  f"near envs: actor sin/cos == critic: {torch.allclose(a_goal[near, 2:], c_goal[near, 2:], atol=1e-5)}"
)
print(
  f"critic sin^2+cos^2 = 1: {torch.allclose(c_goal[:, 2:].norm(dim=1), torch.ones(N, device=device))}"
)
print(f"time_left at reset: [{c_time.min():.4f}, {c_time.max():.4f}]")

rm = env.reward_manager
dim = env.action_manager.total_action_dim
zero = torch.zeros(N, dim, device=device)
first_stag = None
far_prev = env.command_manager.get_command("goal")[:, :2].norm(dim=1) > 1.0
penalties = []
early_pos = early_head = 0.0
for step in range(1, 560):
  _, rew, te, tr, extras = env.step(zero)
  tm = env.termination_manager
  stag = tm.get_term("stagnation")
  if bool(stag.any()) and first_stag is None:
    first_stag = step
    far = env.command_manager.get_command("goal")[:, :2].norm(dim=1) > 1.0
    print(
      f"first stagnation at step {step}: {int(stag.sum())} envs"
      f" (envs with goal > 1 m before the step: {int(far_prev.sum())})"
    )
  far_prev = env.command_manager.get_command("goal")[:, :2].norm(dim=1) > 1.0
  if step < 600:  # the episode is 800 steps; the last 4 s start at 600
    early_pos += float(
      rm._step_reward[:, rm._term_names.index("goal_position")].abs().sum()
    )
    early_head += float(
      rm._step_reward[:, rm._term_names.index("goal_heading")].abs().sum()
    )
  term_pen = rm._step_reward[:, rm._term_names.index("termination_penalty")]
  if bool(te.any()):
    penalties.append(term_pen[te].unique().tolist())
  if step == 1:
    t1 = env.observation_manager.compute()["critic"][:, c_off + 4]
  if step == 2:
    t2 = env.observation_manager.compute()["critic"][:, c_off + 4]
    print(
      f"time_left decrement per step: {float((t1 - t2).median()):.6f}"
      f" (expected {1 / env.max_episode_length:.6f})"
    )
print(
  f"terminated-step termination_penalty raw*weight values (x dt=0.02 -> per-step reward): {penalties[:3]}"
)
print(f"goal_position before the last 4 s: {early_pos}, goal_heading: {early_head}")
print(f"stagnation first fired at step {first_stag} (window 250)")
