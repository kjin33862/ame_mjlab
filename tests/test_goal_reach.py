"""AR025-Stag velocity variant and the AME-2-style GoalReach task (CPU)."""

from __future__ import annotations

import math
import pathlib
import re
from types import SimpleNamespace

import pytest
import torch

D30_TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1"
AR_STAG_TASK = (
  "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-AR025-Stag-Velocity-Rough-G1"
)
GOAL_TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-Contact-FootholdD30-GoalReach-Rough-G1"
REPO = pathlib.Path(__file__).resolve().parents[1]
norm = lambda x: re.sub(r" at 0x[0-9a-f]+", "", repr(x))  # noqa: E731
DT = 0.02


def _cfgs(task: str, play: bool = False):
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import ame  # noqa: F401

  return load_env_cfg(task, play=play), load_rl_cfg(task)


# --------------------------------------------------------------------------
# Existing tasks unchanged.
# --------------------------------------------------------------------------


def test_existing_task_config_hashes_unchanged() -> None:
  """Every task registered at 51b5d6f keeps its env (train, play) and rl
  config hashes (notes/goal_reach/config_hashes.py's normalised repr)."""
  import hashlib
  from dataclasses import asdict

  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import ame  # noqa: F401

  def h(x):
    return hashlib.sha256(norm(x).encode()).hexdigest()[:16]

  baseline = (REPO / "notes/goal_reach/config_hashes_51b5d6f.txt").read_text()
  rows = [line.split() for line in baseline.splitlines() if line.strip()]
  assert len(rows) == 30
  for task, env_h, play_h, rl_h in rows:
    got = (
      h(load_env_cfg(task)),
      h(load_env_cfg(task, play=True)),
      h(asdict(load_rl_cfg(task))),
    )
    assert got == (env_h, play_h, rl_h), task


# --------------------------------------------------------------------------
# Part A: AR025-Stag.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("play", [False, True])
def test_ar025_stag_is_d30_plus_two_changes(play: bool) -> None:
  from ame.mdp.goal import velocity_stagnation

  base, base_rl = _cfgs(D30_TASK, play)
  cfg, rl = _cfgs(AR_STAG_TASK, play)
  assert cfg.rewards["action_rate_l2"].weight == -0.025
  assert base.rewards["action_rate_l2"].weight == -0.1
  stag = cfg.terminations.pop("stagnation")
  assert stag.func is velocity_stagnation and not stag.time_out
  assert stag.params == {
    "command_name": "twist",
    "command_threshold": 0.3,
    "distance_threshold": 0.5,
    "time_window": 5.0,
  }
  # The existing termination_penalty (is_terminated) charges it.
  assert cfg.rewards["termination_penalty"].weight == -50.0
  cfg.rewards["action_rate_l2"].weight = -0.1
  assert norm(cfg) == norm(base)
  assert norm(rl) == norm(base_rl)


class _StagEnv:
  def __init__(self, n: int, cmd_xy: float) -> None:
    self.num_envs = n
    self.device = "cpu"
    self.step_dt = DT
    self.episode_length_buf = torch.zeros(n, dtype=torch.long)
    self.cmd = torch.zeros(n, 4)
    self.cmd[:, 0] = cmd_xy
    self.pos = torch.zeros(n, 3)
    data = SimpleNamespace()
    self.data = data
    self.scene = {"robot": SimpleNamespace(data=data)}
    self.command_manager = SimpleNamespace(get_command=lambda name: self.cmd)
    self.data.root_link_pos_w = self.pos

  def tick(self) -> None:
    self.episode_length_buf += 1


def _velocity_stag(env: _StagEnv):
  from mjlab.managers.termination_manager import TerminationTermCfg

  from ame.mdp.goal import velocity_stagnation

  params = {
    "command_name": "twist",
    "command_threshold": 0.3,
    "distance_threshold": 0.5,
    "time_window": 5.0,
  }
  term = velocity_stagnation(
    TerminationTermCfg(func=velocity_stagnation, params=params), env
  )
  return term, lambda: term(env, **params)


def test_velocity_stagnation_fires_after_a_still_5s_window() -> None:
  env = _StagEnv(3, cmd_xy=0.8)
  env.cmd[2, 0] = 0.2  # a slow command never counts
  _, call = _velocity_stag(env)
  fired = []
  for step in range(1, 300):
    env.tick()
    # env 1 walks at 0.2 m/s: 1 m per 5 s window.
    env.pos[1, 0] = 0.2 * step * DT
    out = call()
    if bool(out.any()):
      fired.append((step, out.nonzero().flatten().tolist()))
  # Step 1 is the episode's first (fresh): counting starts at step 2, and
  # the comparison needs a sample written 250 steps ago under a moving
  # command, so the first possible firing is step 252.
  assert fired and fired[0] == (252, [0])
  assert all(ids == [0] for _, ids in fired)


def test_velocity_stagnation_reset_and_command_drop_restart_the_count() -> None:
  env = _StagEnv(2, cmd_xy=1.0)
  term, call = _velocity_stag(env)
  for _ in range(200):
    env.tick()
    call()
  env.cmd[1, 0] = 0.0  # a standing command for one step
  env.tick()
  call()
  env.cmd[1, 0] = 1.0
  term.reset(torch.tensor([0]))
  for _ in range(200):
    env.tick()
    assert not bool(call().any())


# --------------------------------------------------------------------------
# Part B: GoalReach config.
# --------------------------------------------------------------------------


def _params_with_twist(cfg) -> list[str]:
  found = []
  groups = {
    "rewards": cfg.rewards,
    "metrics": cfg.metrics,
    "terminations": cfg.terminations,
    "curriculum": cfg.curriculum,
    "events": cfg.events,
  }
  for name, terms in groups.items():
    for key, term in terms.items():
      if "twist" in norm(getattr(term, "params", {})):
        found.append(f"{name}.{key}")
  for gname, group in cfg.observations.items():
    for key, term in group.terms.items():
      if "twist" in norm(term.params):
        found.append(f"observations.{gname}.{key}")
  return found


@pytest.mark.parametrize("play", [False, True])
def test_goal_task_config(play: bool) -> None:
  from ame.env_cfg import GOAL_TERMINATION_PENALTY
  from ame.mdp import goal as g

  cfg, rl = _cfgs(GOAL_TASK, play)
  assert set(cfg.commands) == {"goal", "gait"}
  assert isinstance(cfg.commands["goal"], g.GoalCommandCfg)
  assert isinstance(cfg.commands["gait"], g.GoalGaitCommandCfg)
  assert _params_with_twist(cfg) == []
  assert cfg.episode_length_s == 16.0
  # Rewards.
  for name in ("track_linear_velocity", "track_angular_velocity"):
    assert name not in cfg.rewards
  expected = {
    "goal_position": (100.0, g.goal_position_tracking, {"T": 4.0, "std": 2.0}),
    "goal_heading": (50.0, g.goal_heading_tracking, {"T": 2.0, "dist_thr": 0.5}),
    "move_to_goal": (
      5.0,
      g.goal_move_to_goal,
      {"v_min": 0.3, "v_max": 2.0, "cos_thr": 0.5, "dist_thr": 0.5},
    ),
    "stand_at_goal": (5.0, g.goal_stand_at_goal, {"dist_thr": 0.5, "yaw_thr": 0.5}),
  }
  for name, (weight, func, params) in expected.items():
    term = cfg.rewards[name]
    assert term.weight == weight and term.func is func
    assert term.params["command_name"] == "goal"
    for k, v in params.items():
      assert term.params[k] == v, (name, k)
  assert cfg.rewards["move_to_goal"].params["asset_cfg"].body_names == ("torso_link",)
  assert pytest.approx(-500.0) == GOAL_TERMINATION_PENALTY
  assert cfg.rewards["termination_penalty"].weight == pytest.approx(-500.0)
  assert cfg.rewards["action_rate_l2"].weight == -0.025
  # Kept: foothold delta 0.30 and the contact package.
  assert cfg.rewards["foothold"].params["delta"] == 0.3
  assert "illegal_contact" in cfg.terminations
  assert cfg.rewards["undesired_contacts"].weight == -1.0
  # Gated regularisers read the derived gait twist.
  for name in (
    "pose",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
  ):
    assert cfg.rewards[name].params["command_name"] == "gait", name
  # Terminations.
  stag = cfg.terminations["stagnation"]
  assert stag.func is g.goal_stagnation and not stag.time_out
  assert stag.params == {
    "command_name": "goal",
    "distance_threshold": 0.5,
    "goal_distance_threshold": 1.0,
    "time_window": 5.0,
  }
  # Observations: goal block where the command was; critic adds time_left.
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms
  assert list(actor)[-1] == "command" and actor["command"].func is g.goal_obs
  assert actor["command"].params == {"command_name": "goal", "dist_clip": 2.0}
  keys = list(critic)
  assert keys[keys.index("command") + 1] == "time_left"
  assert critic["command"].params == {"command_name": "goal", "dist_clip": None}
  assert "time_left" not in actor
  # Spawn-yaw curriculum.
  assert cfg.events["reset_base"].params["yaw_schedule"] == {
    "steps": 48000,
    "limit": math.pi,
  }
  # Curriculum: SGS with goal success (training only).
  if play:
    assert "terrain_levels" not in cfg.curriculum
  else:
    cur = cfg.curriculum["terrain_levels"]
    assert cur.func is g.terrain_levels_sgs_goal
    assert cur.params["reach_distance"] == 0.5 and cur.params["reach_time_left"] == 4.0
    assert cur.params["sgs"]["uniform_mix"] == 0.2 and cur.params["sgs"]["t"] == 0.5
  # Metrics.
  assert cfg.metrics["goal_distance"].reduce == "last"
  assert "pelvis_speed" in cfg.metrics
  # Runner: double critic on foothold, goal mirror.
  assert rl.algorithm.reward_group_terms == ("foothold",)
  assert rl.algorithm.symmetry_cfg["data_augmentation_func"] == (
    "ame.symmetry:mirror_obs_actions_goal"
  )


def test_goal_task_differs_from_d30_only_where_intended() -> None:
  """Every reward the goal task shares with the base keeps its weight and
  params except the command retarget, termination_penalty and
  action_rate_l2."""
  base, _ = _cfgs(D30_TASK)
  cfg, _ = _cfgs(GOAL_TASK)
  changed = {"termination_penalty", "action_rate_l2"}
  for name, term in base.rewards.items():
    if name in ("track_linear_velocity", "track_angular_velocity"):
      continue
    other = cfg.rewards[name]
    if name in changed:
      assert other.weight != term.weight
      continue
    assert other.weight == term.weight, name
    params = dict(term.params)
    if params.get("command_name") == "twist":
      params["command_name"] = "gait"
    assert norm(other.params) == norm(params), name
  assert set(cfg.rewards) - set(base.rewards) == {
    "goal_position",
    "goal_heading",
    "move_to_goal",
    "stand_at_goal",
  }
  assert set(cfg.terminations) - set(base.terminations) == {"stagnation"}
  assert set(base.terminations) <= set(cfg.terminations)


# --------------------------------------------------------------------------
# Part B: terms.
# --------------------------------------------------------------------------


class _GoalEnv:
  """Fake env exposing a fixed goal command ``(x, y, z, e)``."""

  def __init__(self, cmd, step=0, max_len=800) -> None:
    self.cmd = torch.as_tensor(cmd, dtype=torch.float32)
    n = self.cmd.shape[0]
    self.num_envs = n
    self.device = "cpu"
    self.step_dt = DT
    self.max_episode_length = max_len
    self.episode_length_buf = torch.full((n,), step, dtype=torch.long)
    self.command_manager = SimpleNamespace(get_command=lambda name: self.cmd)


def test_goal_obs_clips_and_randomises_far_goals() -> None:
  from ame.mdp.goal import goal_obs

  env = _GoalEnv([[1.0, 1.0, 0.0, 0.3], [3.0, -4.0, 0.0, 0.3]])
  full = goal_obs(env, "goal", None)
  torch.testing.assert_close(full[:, :2], env.cmd[:, :2])
  torch.testing.assert_close(full[:, 2], torch.sin(env.cmd[:, 3]))
  torch.testing.assert_close(full[:, 3], torch.cos(env.cmd[:, 3]))
  torch.manual_seed(0)
  clip = [goal_obs(env, "goal", 2.0) for _ in range(20)]
  for c in clip:
    torch.testing.assert_close(c[0], full[0])  # within 2 m: exact
    torch.testing.assert_close(c[1, :2], torch.tensor([1.2, -1.6]))
    torch.testing.assert_close(c[1, 2:].norm(), torch.tensor(1.0))
  # The far goal's heading is re-drawn every call.
  assert len({round(float(c[1, 2]), 4) for c in clip}) > 10


def test_position_and_heading_rewards_only_at_the_end() -> None:
  from ame.mdp.goal import goal_heading_tracking, goal_position_tracking

  cmd = [[2.0, 0.0, 0.0, 1.0], [0.3, 0.0, 0.0, 1.0]]
  early = _GoalEnv(cmd, step=599)  # t_left = 4.02 s
  assert float(goal_position_tracking(early, "goal", 4.0, 2.0).abs().sum()) == 0.0
  late = _GoalEnv(cmd, step=601)  # t_left = 3.98 s
  pos = goal_position_tracking(late, "goal", 4.0, 2.0)
  torch.testing.assert_close(pos, torch.tensor([0.5 / 4.0, 1 / (1 + 0.0225) / 4.0]))
  assert float(goal_heading_tracking(late, "goal", 2.0, 0.5).sum()) == 0.0  # t_left > 2
  last = _GoalEnv(cmd, step=760)  # t_left = 0.8 s
  head = goal_heading_tracking(last, "goal", 2.0, 0.5)
  torch.testing.assert_close(head, torch.tensor([0.0, 1 / (1 + 1.0) / 2.0]))


def _robot_env(cmd, vel_w, quat=None, found=None, gz=-1.0, dq=0.0):
  env = _GoalEnv(cmd)
  n = env.num_envs
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n) if quat is None else quat
  data = SimpleNamespace(
    root_link_quat_w=quat,
    root_link_lin_vel_w=torch.tensor(vel_w, dtype=torch.float32),
    body_link_lin_vel_w=torch.tensor(vel_w, dtype=torch.float32)[:, None, :],
    projected_gravity_b=torch.tensor([[0.0, 0.0, gz]] * n),
    joint_pos=torch.full((n, 29), dq),
    default_joint_pos=torch.zeros(n, 29),
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.tensor(found if found is not None else [[1, 1]] * n)
    )
  )
  env.scene = {"robot": SimpleNamespace(data=data), "feet": sensor}
  return env


def test_move_to_goal() -> None:
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  from ame.mdp.goal import goal_move_to_goal

  cmd = [[3.0, 0.0, 0, 0]] * 5 + [[0.3, 0.0, 0, 0]]
  vel = [
    [1.0, 0.0, 0],  # toward, in band
    [0.2, 0.0, 0],  # too slow
    [2.5, 0.0, 0],  # too fast
    [0.5, 1.0, 0],  # cos = 0.447 < 0.5
    [-1.0, 0.0, 0],  # away
    [0.0, 0.0, 0],  # near the goal: always 1
  ]
  env = _robot_env(cmd, vel)
  asset = SceneEntityCfg("robot")
  asset.body_ids = [0]
  out = goal_move_to_goal(env, "goal", 0.3, 2.0, 0.5, 0.5, asset)
  torch.testing.assert_close(out, torch.tensor([1.0, 0, 0, 0, 0, 1.0]))


def test_stand_at_goal() -> None:
  from ame.mdp.goal import goal_stand_at_goal

  cmd = [[0.2, 0.0, 0, 0.1], [0.2, 0.0, 0, 0.1], [0.2, 0.0, 0, 0.6], [0.6, 0.0, 0, 0.0]]
  env = _robot_env(cmd, [[0, 0, 0]] * 4, found=[[1, 1], [1, 0], [1, 1], [1, 1]], dq=0.1)
  out = goal_stand_at_goal(env, "feet", "goal", 0.5, 0.5)
  e0 = math.exp(-(0.0 + 0.0 + 0.1 + 0.2) / 4)
  e1 = math.exp(-(0.5 + 0.0 + 0.1 + 0.2) / 4)
  torch.testing.assert_close(out, torch.tensor([e0, e1, 0.0, 0.0]))


def test_goal_stagnation_windows() -> None:
  from ame.mdp.goal import goal_stagnation

  env = _StagEnv(3, cmd_xy=3.0)
  env.cmd[2, 0] = 0.8  # within 1 m of the goal: never stagnant
  env.max_episode_length = 800
  term = goal_stagnation(None, env)
  fired = []
  for step in range(1, 760):
    env.tick()
    env.pos[1, 0] = 0.2 * step * DT  # 1 m per window
    out = term(env, "goal", 0.5, 1.0, 5.0)
    if bool(out.any()):
      fired.append((step, out.nonzero().flatten().tolist()))
  # Snapshots at steps 1, 251, 501, 751: env 0 is stagnant at each check.
  assert fired == [(251, [0]), (501, [0]), (751, [0])]


def test_gait_twist_from_goal() -> None:
  from ame.mdp.goal import GoalGaitCommand, GoalGaitCommandCfg

  goal_cmd = torch.tensor(
    [[3.0, 4.0, 0.0, 1.0], [0.3, 0.0, 0.0, 1.0], [0.3, 0, 0, 0.01]]
  )
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    command_manager=SimpleNamespace(
      get_term=lambda name: SimpleNamespace(command=goal_cmd)
    ),
  )
  term = GoalGaitCommand(GoalGaitCommandCfg(), env)
  term._update_command()
  torch.testing.assert_close(
    term.command,
    torch.tensor([[0.6, 0.8, 0.0], [0.0, 0.0, 0.4], [0.0, 0.0, 0.01]]),
  )


# --------------------------------------------------------------------------
# Symmetry.
# --------------------------------------------------------------------------


def _goal_groups(n: int, gen: torch.Generator):
  from ame.scan import MAP_SHAPE

  actor = torch.randn(n, 3 + 3 + 87 + 4, generator=gen)
  critic = torch.randn(n, 3 + 3 + 3 + 87 + 4 + 1 + 2 * 5, generator=gen)
  hmap = torch.randn(n, math.prod(MAP_SHAPE), generator=gen)
  return actor, critic, hmap


def test_goal_mirror_flips_goal_y_and_heading() -> None:
  from tensordict import TensorDict

  from ame.symmetry import mirror_obs_actions_goal

  gen = torch.Generator().manual_seed(0)
  actor, critic, hmap = _goal_groups(4, gen)
  obs = TensorDict(
    {"actor": actor, "critic": critic, "height_map": hmap}, batch_size=[4]
  )
  out, _ = mirror_obs_actions_goal(None, obs, None)
  assert out is not None
  ma, mc = out["actor"][4:], out["critic"][4:]
  a0 = 3 + 3 + 87
  torch.testing.assert_close(
    ma[:, a0 : a0 + 4], actor[:, a0 : a0 + 4] * torch.tensor([1, -1, -1, 1.0])
  )
  c0 = 3 + 3 + 3 + 87
  torch.testing.assert_close(
    mc[:, c0 : c0 + 4], critic[:, c0 : c0 + 4] * torch.tensor([1, -1, -1, 1.0])
  )
  torch.testing.assert_close(mc[:, c0 + 4], critic[:, c0 + 4])  # time left
  # Foot pairs after time_left swap.
  tail = critic[:, c0 + 5 :].reshape(4, -1, 2).flip(2).reshape(4, -1)
  torch.testing.assert_close(mc[:, c0 + 5 :], tail)
  # Involution.
  twice, _ = mirror_obs_actions_goal(None, out[4:], None)
  assert twice is not None
  for key in ("actor", "critic", "height_map"):
    torch.testing.assert_close(twice[key][4:], obs[key])
  with pytest.raises(KeyError):
    mirror_obs_actions_goal(None, TensorDict({"policy": actor}, batch_size=[4]), None)


def test_goal_mirror_matches_a_mirrored_world() -> None:
  """goal_obs of the mirrored goal (y -> -y, heading error -> -e) equals the
  mirror of goal_obs, near (exact) and far (clipped position, sin / cos still
  a unit vector)."""
  from ame.mdp.goal import goal_obs
  from ame.symmetry import GOAL_COMMAND_SIGN

  cmd = torch.tensor(
    [[1.2, 0.7, 0.0, 0.9], [-0.5, -1.1, 0.0, -2.8], [3.0, 2.0, 0.0, 0.4]]
  )
  mirrored = cmd * torch.tensor([1.0, -1.0, 1.0, -1.0])
  sign = torch.tensor(GOAL_COMMAND_SIGN)
  a = goal_obs(_GoalEnv(cmd), "goal", None) * sign
  b = goal_obs(_GoalEnv(mirrored), "goal", None)
  torch.testing.assert_close(a, b)
  a = goal_obs(_GoalEnv(cmd), "goal", 2.0) * sign
  b = goal_obs(_GoalEnv(mirrored), "goal", 2.0)
  torch.testing.assert_close(a[:, :2], b[:, :2])
  torch.testing.assert_close(a[:2], b[:2])


def test_velocity_mirror_unchanged() -> None:
  """The refactor behind the goal mirror leaves the velocity mirror as it was:
  (vx, vy, wz) -> (vx, -vy, -wz) and per-foot pairs swapped."""
  from ame.symmetry import _mirror_proprio, _tensors

  gen = torch.Generator().manual_seed(1)
  g = torch.randn(3, 99 + 8, generator=gen)
  out = _mirror_proprio(g, True, _tensors(g.device))
  c = 3 + 3 + 3 + 87
  torch.testing.assert_close(
    out[:, c : c + 3], g[:, c : c + 3] * torch.tensor([1, -1, -1.0])
  )
  torch.testing.assert_close(
    out[:, c + 3 :], g[:, c + 3 :].reshape(3, -1, 2).flip(2).reshape(3, -1)
  )
  torch.testing.assert_close(out[:, 1], -g[:, 1])
  torch.testing.assert_close(out[:, 3], -g[:, 3])
  torch.testing.assert_close(out[:, 4], g[:, 4])


def test_linear_schedule() -> None:
  from ame.mdp.goal import linear_schedule

  assert linear_schedule(0, 48000, 10.0, 45.0) == 10.0
  assert linear_schedule(24000, 48000, 10.0, 45.0) == pytest.approx(27.5)
  assert linear_schedule(10**6, 48000, 10.0, 45.0) == 45.0
