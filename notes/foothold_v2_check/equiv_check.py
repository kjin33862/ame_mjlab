"""Equivalence + double-critic check of the FootholdV2-Progress task.

The checkpoint of notes/reward_audit (D30 model_8600, actor only) runs in the
NEW task's training env cfg with notes/edge_measure/measure_v2.py's eval edits
(noise off, push / encoder bias / curriculum off, fixed per-robot command,
spawn yaw 0, the policy-stands robots see command 0).  16 robots x 10 s per
condition, seed 42.  Logged per step from the training reward manager:
``_step_reward`` of foothold_v2 and progress_shortfall, the double critic's
``RewardGroups`` split of the env reward, and the foothold_v2 term's
diagnostics (x^2 paid per foot, pay reason, touchdown foot xy), and the
sole scan hits (logged after each step, as measure_v2.py) so the offline
analysis can be recomputed on the same run.

usage: python notes/foothold_v2_check/equiv_check.py CHECKPOINT  -> equiv.npz
"""

import dataclasses
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg

import ame  # noqa: F401

TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdV2-Progress-Velocity-Rough-G1"
SEED = 42
STEPS = 500
PER = 16
OUT = Path(__file__).parent

torch.manual_seed(SEED)
np.random.seed(SEED)

cfg = load_env_cfg(TASK)
agent_cfg = load_rl_cfg(TASK)
gen = cfg.scene.terrain.terrain_generator
train_subs = gen.sub_terrains
stones = train_subs["stepping_stones"]
stairs = train_subs["pyramid_stairs_inv"]
train_rows = gen.num_rows

pyr = train_subs["pyramid_stairs"]
wave = train_subs["wave"]
rough = train_subs["random_rough"]
subs = {"flat": BoxFlatTerrainCfg(proportion=1.0)}
gaps = {}
for L in (1, 2, 3):
  g = stones.gap_range[0] + (stones.gap_range[1] - stones.gap_range[0]) * L / (train_rows - 1)
  gaps[L] = g
  subs[f"stones_L{L}"] = dataclasses.replace(stones, gap_range=(g, g))
for L in (2, 5, 9):
  a_ = wave.amplitude_range[0] + (wave.amplitude_range[1] - wave.amplitude_range[0]) * L / (train_rows - 1)
  subs[f"wave_L{L}"] = dataclasses.replace(wave, amplitude_range=(a_, a_))
subs["random_rough"] = dataclasses.replace(rough)
risers = {}
for L in range(1, 6):
  h = stairs.step_height_range[0] + (stairs.step_height_range[1] - stairs.step_height_range[0]) * L / (train_rows - 1)
  risers[L] = h
  subs[f"pyr_L{L}"] = dataclasses.replace(pyr, step_height_range=(h, h))
  subs[f"inv_L{L}"] = dataclasses.replace(stairs, step_height_range=(h, h))
heights = [round(0.10 + 0.01 * k, 3) for k in range(8)]
for h in heights:
  subs[f"stairs_{int(round(h * 100))}"] = dataclasses.replace(stairs, step_height_range=(h, h))
gen.sub_terrains = subs
gen.num_rows = 4
gen.curriculum = True  # one column per terrain; rows identical
col = {k: i for i, k in enumerate(subs)}

# condition -> (vx, columns, policy?)
CONDS = [
  ("flat_v1.0", 1.0, ["flat"], True),
  ("flat_stand_policy", 1.0, ["flat"], "cmd0"),
  ("stones_L2", 1.0, ["stones_L2"], True),
  ("stones_L3", 1.0, ["stones_L3"], True),
  ("pyr_L3", 1.0, ["pyr_L3"], True),
  ("inv_L3", 1.0, ["inv_L3"], True),
]
N = PER * len(CONDS)
env_cond = np.repeat(np.arange(len(CONDS)), PER)
env_col = np.concatenate([np.array([col[c[2][i % len(c[2])]] for i in range(PER)]) for c in CONDS])
env_vx = np.repeat([c[1] for c in CONDS], PER)
env_policy = np.repeat([c[3] is not False for c in CONDS], PER)
env_cmd0 = np.repeat([c[3] == "cmd0" for c in CONDS], PER)

cfg.scene.num_envs = N
cfg.seed = SEED
cfg.observations["actor"].enable_corruption = False
cfg.observations["height_map"].terms["elevation"].params["noise"] = False
cfg.events.pop("push_robot", None)
cfg.events.pop("encoder_bias", None)
cfg.curriculum = {}
cfg.events["reset_base"].params["yaw_schedule"] = {"steps": 1, "limit": 0.0}
tw = cfg.commands["twist"]
tw.resampling_time_range = (1.0e9, 1.0e9)
tw.heading_command = False
tw.ranges.heading = None
tw.rel_standing_envs = tw.rel_heading_envs = tw.rel_world_envs = tw.rel_forward_envs = 0.0
tw.ranges.lin_vel_x = tw.ranges.lin_vel_y = tw.ranges.ang_vel_z = (0.0, 0.0)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_u = ManagerBasedRlEnv(cfg=cfg, device=device)
target = torch.zeros(N, 3, device=device)
target[:, 0] = torch.tensor(env_vx, dtype=torch.float32, device=device)
term = env_u.command_manager.get_term("twist")
_orig = term._resample_command


def _resample(env_ids):
  _orig(env_ids)
  term.vel_command_b[env_ids] = target[env_ids]
  for flag in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env"):
    getattr(term, flag)[env_ids] = False


term._resample_command = _resample

terrain = env_u.scene.terrain
terrain.terrain_types[:] = torch.tensor(env_col, device=device)
terrain.terrain_levels[:] = 0
terrain.env_origins[:] = terrain.terrain_origins[0, terrain.terrain_types]
env = RslRlVecEnvWrapper(env_u, clip_actions=agent_cfg.clip_actions)  # resets
obs, _ = env.reset()

runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)
use_policy = torch.tensor(env_policy, device=device)[:, None]
cmd0 = torch.tensor(env_cmd0, device=device)
om = env_u.observation_manager
_names = om.active_terms["actor"]
_dims = [int(np.prod(d)) for d in om._group_obs_term_dim["actor"]]
_c0 = sum(_dims[: _names.index("command")])
cmd_slice = slice(_c0, _c0 + _dims[_names.index("command")])
print("actor command slice", cmd_slice)

rm = env_u.reward_manager
robot = env_u.scene["robot"]
feet = env_u.scene["feet_ground_contact"]
foot_ids = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
print("feet primaries:", feet.primary_names, "bodies:", foot_ids)
assert len(feet.primary_names) == 2
dt = env_u.step_dt
from ame.double_critic import RewardGroups

alg = agent_cfg.algorithm
print("runner reward_group_terms", alg.reward_group_terms, "advantage_weights", alg.advantage_weights)
groups = RewardGroups(rm, alg.reward_group_terms, dt)
sole = env_u.scene["foot_sole_scan"]
v2 = rm._term_cfgs[rm._term_names.index("foothold_v2")].func
print("group-2 columns:", [rm._term_names[i] for i in groups.index.tolist()])

log = {k: [] for k in ("rew", "speed", "t_since", "done", "td", "foot", "heading", "root", "cmd", "term", "air", "found", "env_rew", "groups", "v2_paid", "v2_reason", "v2_xy", "sole_hit")}
for step in range(STEPS):
  with torch.inference_mode():
    obs["actor"][cmd0, cmd_slice] = 0.0  # cmd0 robots: the policy is told to stand
    a = policy(obs)
  a = torch.where(use_policy, a, torch.zeros_like(a))
  obs, env_rew, dones, _ = env.step(a)
  log["env_rew"].append(env_rew.cpu())
  log["groups"].append(groups(env_rew).cpu())
  log["v2_paid"].append(v2.paid.cpu())
  log["v2_reason"].append(v2.reason.cpu())
  log["v2_xy"].append(v2.paid_xy.cpu())
  hp = sole.data.hit_pos_w.clone()
  hp[sole.data.distances < 0] = float("nan")
  log["sole_hit"].append(hp.view(N, 2, -1, 3).cpu())
  log["rew"].append(rm._step_reward.clone().cpu())
  log["speed"].append(robot.data.root_link_lin_vel_w[:, :2].norm(dim=1).cpu())
  log["t_since"].append(env_u.episode_length_buf.cpu() * dt)
  log["done"].append(dones.cpu())
  log["td"].append(feet.compute_first_contact(dt).cpu())
  log["air"].append(feet.data.last_air_time.clone().cpu())
  log["found"].append(feet.data.found.clone().cpu())
  log["foot"].append(robot.data.body_link_pos_w[:, foot_ids].cpu())
  log["heading"].append(robot.data.heading_w.cpu())
  log["root"].append(robot.data.root_link_pos_w.cpu())
  log["cmd"].append(term.vel_command_b.clone().cpu())
  tm = env_u.termination_manager
  log["term"].append(torch.stack([tm.get_term(n) for n in tm.active_terms], 1).cpu())

arr = {k: torch.stack(v).numpy() for k, v in log.items()}
np.savez_compressed(
  OUT / "equiv.npz",
  **arr,
  term_names=np.array(rm._term_names),
  env_cond=env_cond, env_col=env_col, env_vx=env_vx, env_policy=env_policy, env_cmd0=env_cmd0,
  cond_names=np.array([c[0] for c in CONDS]),
  col_names=np.array(list(subs)),
  tile_origins=terrain.terrain_origins.cpu().numpy(),
  dt=dt, gaps=np.array([gaps[1], gaps[2], gaps[3]]), stair_heights=np.array(heights),
  risers=np.array([risers[L] for L in range(1, 6)]),
  done_names=np.array(env_u.termination_manager.active_terms),
)
print("saved equiv.npz", {k: v.shape for k, v in arr.items()})
print("dones per condition:", np.bincount(env_cond, weights=arr["done"].sum(0), minlength=len(CONDS)))
