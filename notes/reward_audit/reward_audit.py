"""Reward / stride audit of a trained velocity policy (evaluation only).

One env, 128 robots, 16 per condition, 500 steps (10 s), seed 42:

  flat_v0.5 / flat_v1.0 / flat_v1.2   flat tile, policy, vx command
  flat_stand                          flat tile, vx 1.0, action = 0
                                      (JointPositionAction with
                                      use_default_offset: 0 = default pose)
  flat_stand_policy                   flat tile, rewarded against vx 1.0, but
                                      the policy's observed command is 0:
                                      the policy stands by itself
  stones_L1 / L2 / L3                 stone lattice at the training board's
                                      level 1-3 gap (0.45 * L / 9), policy, vx 1.0
  stairs_up                           inverted pyramid stairs, step height
                                      0.10..0.17 m (2 robots per cm), policy, vx 1.0

The board is built for this audit only (the task registry is untouched): one
column per condition terrain, 4 identical rows along +x so a robot walking
+x for 10 s stays on the same terrain.  Sub-terrain cfgs are the training
ones with only the knob fixed (stone gap, step height); the generator sets
every tile to 8 m as in training.

Changes from the training env cfg, all eval-only: actor corruption and the
height-map noise (bias / drift / jitter) off,
push_robot and encoder_bias removed, curriculum removed (levels fixed),
spawn yaw fixed to 0 (``yaw_schedule`` limit 0), command resampling
disabled and the twist command written per robot (vy = wz = 0, no heading /
standing / world / forward envs).  Rewards, terminations, foot friction and
COM randomisation (startup, seeded) are the training ones.

Logged per step: every reward term from the training reward manager
(``_step_reward`` = weight * raw, i.e. the per-second rate; the per-step
reward is that x dt), pelvis xy speed, time since spawn, terminations, and
per foot touchdowns (``feet_ground_contact.compute_first_contact``, the
preceding swing's ``last_air_time`` and the ``found`` contact state) with
both ankle_roll_link positions and the pelvis heading.

usage: python notes/reward_audit/reward_audit.py CHECKPOINT [--no-illegal-termination]
  -> raw.npz (or raw_noillegal.npz: the illegal_contact termination removed,
  every reward unchanged; at the default pose the wrists touch the hip_roll
  links, so with that termination a held-standing robot is reset ~0.24 s
  after every spawn)
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

TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1"
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

subs = {"flat": BoxFlatTerrainCfg(proportion=1.0)}
gaps = {}
for L in (1, 2, 3):
  g = stones.gap_range[0] + (stones.gap_range[1] - stones.gap_range[0]) * L / (train_rows - 1)
  gaps[L] = g
  subs[f"stones_L{L}"] = dataclasses.replace(stones, gap_range=(g, g))
heights = [round(0.10 + 0.01 * k, 3) for k in range(8)]
for h in heights:
  subs[f"stairs_{int(round(h * 100))}"] = dataclasses.replace(stairs, step_height_range=(h, h))
gen.sub_terrains = subs
gen.num_rows = 4
gen.curriculum = True  # one column per terrain; rows identical
col = {k: i for i, k in enumerate(subs)}

# condition -> (vx, columns, policy?)
CONDS = [
  ("flat_v0.5", 0.5, ["flat"], True),
  ("flat_v1.0", 1.0, ["flat"], True),
  ("flat_v1.2", 1.2, ["flat"], True),
  ("flat_stand", 1.0, ["flat"], False),
  ("flat_stand_policy", 1.0, ["flat"], "cmd0"),
  ("stones_L1", 1.0, ["stones_L1"], True),
  ("stones_L2", 1.0, ["stones_L2"], True),
  ("stones_L3", 1.0, ["stones_L3"], True),
  ("stairs_up", 1.0, [f"stairs_{int(round(h * 100))}" for h in heights], True),
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
NO_ILLEGAL = "--no-illegal-termination" in sys.argv
if NO_ILLEGAL:
  cfg.terminations.pop("illegal_contact")
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

log = {k: [] for k in ("rew", "speed", "t_since", "done", "td", "foot", "heading", "root", "cmd", "term", "air", "found")}
for step in range(STEPS):
  with torch.inference_mode():
    obs["actor"][cmd0, cmd_slice] = 0.0  # cmd0 robots: the policy is told to stand
    a = policy(obs)
  a = torch.where(use_policy, a, torch.zeros_like(a))
  obs, _, dones, _ = env.step(a)
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
  OUT / ("raw_noillegal.npz" if NO_ILLEGAL else "raw.npz"),
  **arr,
  term_names=np.array(rm._term_names),
  env_cond=env_cond, env_col=env_col, env_vx=env_vx, env_policy=env_policy, env_cmd0=env_cmd0,
  cond_names=np.array([c[0] for c in CONDS]),
  col_names=np.array(list(subs)),
  tile_origins=terrain.terrain_origins.cpu().numpy(),
  dt=dt, gaps=np.array([gaps[1], gaps[2], gaps[3]]), stair_heights=np.array(heights),
  done_names=np.array(env_u.termination_manager.active_terms),
)
print("saved", "raw_noillegal.npz" if NO_ILLEGAL else "raw.npz", {k: v.shape for k, v in arr.items()})
print("dones per condition:", np.bincount(env_cond, weights=arr["done"].sum(0), minlength=len(CONDS)))
