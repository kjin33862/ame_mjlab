"""Edge task: centre spawn, first-passage success, task composition."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

EDGE_TASK = "Ame-Mjlab-Gap-Edge-SGS-Velocity-Rough-G1"
SGS_TASK = "Ame-Mjlab-Gap-ProgressHeading-SGS-Velocity-Rough-G1"
ANGSTD_TASK = "Ame-Mjlab-Gap-Edge-SGS-AngStd-Velocity-Rough-G1"
ROOT_Z = 0.76  # the stub robot's default root height above the ground


class _Scene(SimpleNamespace):
  """A scene stub indexable by entity name, as ``env.scene["robot"]``."""

  def __getitem__(self, key: str):
    return getattr(self, key)


# --------------------------------------------------------------------------- spawn


@pytest.fixture(scope="module")
def board():
  """The gap board's five families on three rows (d = 0, 0.5, 1), compiled
  for real: the spawn ray-casts this geometry."""
  from mjlab.terrains.terrain_generator import TerrainGenerator

  from ame.terrains import gap_board_cfg

  gen = replace(gap_board_cfg(), num_rows=3, seed=0)
  tg = TerrainGenerator(gen)
  spec = mujoco.MjSpec()
  tg.compile(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  return SimpleNamespace(
    cfg=gen, origins=np.asarray(tg.terrain_origins), model=model, data=data
  )


def _spawn_env(board, levels: list[int], types: list[int]):
  n = len(levels)
  origins = torch.tensor(board.origins, dtype=torch.float32)
  terrain = SimpleNamespace(
    terrain_levels=torch.tensor(levels),
    terrain_types=torch.tensor(types),
    terrain_origins=origins,
    cfg=SimpleNamespace(terrain_generator=board.cfg),
  )
  written: dict[str, torch.Tensor] = {}
  default = torch.zeros(n, 13)
  default[:, 2], default[:, 3] = ROOT_Z, 1.0

  def write_pose(pose, env_ids):
    written["pose"] = pose.clone()

  robot = SimpleNamespace(
    data=SimpleNamespace(default_root_state=default),
    write_root_link_pose_to_sim=write_pose,
    write_root_link_velocity_to_sim=lambda vel, env_ids: None,
  )
  env = SimpleNamespace(
    num_envs=n,
    device="cpu",
    sim=SimpleNamespace(mj_model=board.model),
    scene=_Scene(
      terrain=terrain,
      env_origins=origins[terrain.terrain_levels, terrain.terrain_types].clone(),
      robot=robot,
    ),
  )
  return env, written


def _ground(board, x: float, y: float) -> float:
  gid = np.zeros(1, dtype=np.int32)
  d = mujoco.mj_ray(
    board.model,
    board.data,
    np.array([x, y, 20.0]),
    np.array([0, 0, -1.0]),
    None,
    1,
    -1,
    gid,
  )
  return 20.0 - d


def _spawn(board, family: int, n_per_row: int = 200):
  from ame.env_cfg import CENTRE_SPAWN
  from ame.mdp import reset_root_state_at_centre

  levels = [r for r in range(3) for _ in range(n_per_row)]
  env, written = _spawn_env(board, levels, [family] * len(levels))
  term = reset_root_state_at_centre(SimpleNamespace(params=dict(CENTRE_SPAWN)), env)
  torch.manual_seed(0)
  term(env, torch.arange(len(levels)), **CENTRE_SPAWN)
  pose = written["pose"]
  w, z = pose[:, 3], pose[:, 6]
  yaw = torch.atan2(2 * w * z, 1 - 2 * z * z)
  return env, pose[:, :3], yaw


@pytest.mark.parametrize("family", [1, 2, 3, 4])
def test_centre_spawn_is_within_half_a_metre_with_a_uniform_heading(
  board, family
) -> None:
  # pyramid_stairs, pyramid_stairs_inv, wave, random_rough
  env, pos, yaw = _spawn(board, family)
  centre = env.scene.env_origins
  offset = torch.linalg.norm(pos[:, :2] - centre[:, :2], dim=1)
  assert float(offset.max()) <= 0.5 + 1e-6
  assert float(offset.mean()) == pytest.approx(
    0.5 * 2 / 3, abs=0.03
  )  # uniform in the disc
  assert (
    float(yaw.min()) < -2.5 and float(yaw.max()) > 2.5
  )  # heading over (-pi, pi), not lattice axes
  assert torch.histc(yaw, bins=4, min=-math.pi, max=math.pi).min() > 0.15 * len(yaw)
  # Starts above the ground under its pelvis by the lift, at most the footprint's rise more.
  for p in pos[::17]:
    lift = float(p[2]) - ROOT_Z - _ground(board, float(p[0]), float(p[1]))
    assert 0.01 - 0.005 <= lift <= 0.05 + 0.12


def test_centre_spawn_is_on_a_stone_on_the_lattice(board) -> None:
  from ame.terrains import BoxStoneLatticeTerrainCfg

  sub = board.cfg.sub_terrains["stepping_stones"]
  assert isinstance(sub, BoxStoneLatticeTerrainCfg)
  env, pos, yaw = _spawn(board, family=0)
  for i, p in enumerate(pos):
    row = int(env.scene.terrain.terrain_levels[i])
    stones, _ = sub.centres(row / 2)
    stones = stones - np.asarray(sub.size) / 2 + board.origins[row, 0, :2]
    d = np.abs(stones - p[:2].numpy()).max(axis=1)
    assert d.min() <= 0.03 + 1e-6  # on a stone centre, within the jitter
    assert _ground(board, float(p[0]), float(p[1])) == pytest.approx(
      0.0, abs=1e-4
    )  # stone top, not pit
    assert float(p[2]) - ROOT_Z == pytest.approx(0.03, abs=0.02 + 1e-6)
  # The stone nearest a point of the 0.5 m disc: never farther out than half a pitch beyond it.
  offset = torch.linalg.norm(pos[:, :2] - env.scene.env_origins[:, :2], dim=1)
  assert float(offset.max()) <= 0.5 + (0.40 + 0.45) / math.sqrt(2) + 0.03
  assert float(yaw.min()) < -2.5 and float(yaw.max()) > 2.5


# ------------------------------------------------------------------ first passage


def _edge_env(n: int):
  """One row, two families; tile centres at the origin. Positions are set per step."""
  from ame.mdp import edge_distance

  terrain = SimpleNamespace(
    terrain_levels=torch.zeros(n, dtype=torch.long),
    terrain_types=torch.zeros(n, dtype=torch.long),
    terrain_origins=torch.zeros(10, 2, 3),
    env_origins=torch.zeros(n, 3),
    max_terrain_level=10,
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(size=(8.0, 8.0), sub_terrains={"a": 0, "b": 0})
    ),
  )
  pos = torch.zeros(n, 3)
  robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos))
  env = SimpleNamespace(
    num_envs=n,
    device="cpu",
    scene=_Scene(terrain=terrain, env_origins=torch.zeros(n, 3), robot=robot),
    common_step_counter=0,
    command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(n, 3)),
    termination_manager=SimpleNamespace(time_outs=torch.zeros(n, dtype=torch.bool)),
  )
  metric = edge_distance(SimpleNamespace(params={}), env)
  env.metrics_manager = SimpleNamespace(
    active_terms=["edge_distance"], _term_cfgs=[SimpleNamespace(func=metric)]
  )
  return env, pos, metric


def _edge_term(env):
  from ame.env_cfg import EDGE_DISTANCE
  from ame.mdp import terrain_levels_sgs_edge

  params = {
    "command_name": "twist",
    "floor": 0,
    "edge_distance": EDGE_DISTANCE,
    "sgs": {},
  }
  term = terrain_levels_sgs_edge(SimpleNamespace(params=params), env)
  term(env, torch.arange(env.num_envs), **params)  # arming call: nothing recorded
  return term, params


def _episode(env, pos, metric, path: list[list[tuple[float, float, float]]]) -> None:
  """Step every env along its path (one point per step), the metric after each."""
  for step in zip(*path, strict=True):
    pos[:] = torch.tensor(step)
    metric(env)


def test_crossing_then_falling_is_success() -> None:
  env, pos, metric = _edge_env(1)
  term, params = _edge_term(env)
  # walk out along +x to 3.8 m (past 3.75), turn back and fall at 3 m, 0.3 m high
  path = [
    [(0.1 * i, 0.0, 0.76) for i in range(39)] + [(3.4, 0.0, 0.5), (3.0, 0.0, 0.3)]
  ]
  _episode(env, pos, metric, path)
  assert float(pos[0, 0]) < 3.75 and not bool(env.termination_manager.time_outs[0])
  term(env, torch.arange(1), **params)
  assert (
    int(term.buffer.fill[0, 0]) == 1 and float(term.buffer.success_rate()[0, 0]) == 1.0
  )


def test_spinning_in_the_pit_is_failure() -> None:
  env, pos, metric = _edge_env(1)
  term, params = _edge_term(env)
  # dropped into the pit 1.5 m out and circled there until time out
  path = [
    [
      (1.5 + 0.3 * math.cos(0.2 * i), 0.3 * math.sin(0.2 * i), -1.3)
      for i in range(1000)
    ]
  ]
  _episode(env, pos, metric, path)
  env.termination_manager.time_outs[:] = True
  term(env, torch.arange(1), **params)
  assert (
    int(term.buffer.fill[0, 0]) == 1 and float(term.buffer.success_rate()[0, 0]) == 0.0
  )


def test_the_diagonal_needs_chebyshev_distance() -> None:
  env, pos, metric = _edge_env(4)
  term, params = _edge_term(env)
  env.scene.terrain.terrain_types[:] = torch.tensor([0, 1, 1, 1])
  ends = [
    (2.9, 2.9, 0.76),  # 4.1 m out on the diagonal, but only 2.9 m towards either edge
    (3.7, -3.7, 0.76),  # 5.2 m out, max(|dx|, |dy|) = 3.7 < 3.75
    (-3.75, 1.0, 0.76),  # exactly at the threshold
    (0.5, 3.9, 0.76),  # past it along y
  ]
  path = [[(e[0] * s / 20, e[1] * s / 20, e[2]) for s in range(21)] for e in ends]
  _episode(env, pos, metric, path)
  env.termination_manager.time_outs[:] = True
  term(env, torch.arange(4), **params)
  assert metric.max_distance.tolist() == pytest.approx([2.9, 3.7, 3.75, 3.9])
  assert float(term.buffer.success_rate()[0, 0]) == 0.0  # family a: the diagonal
  assert term.buffer.fill[1, 0] == 3 and float(
    term.buffer.success_rate()[1, 0]
  ) == pytest.approx(2 / 3)
  logs = term(env, torch.arange(0), **params)
  assert logs["a/edge_frac"] == 0.0 and logs["b/edge_frac"] == pytest.approx(2 / 3)
  assert {"a/p_0", "a/prob_0", "a/fill_min", "b/p_0", "b/prob_9", "b/fill_min"} <= set(
    logs
  )


def test_edge_metric_resets_per_env_and_uses_the_tile_centre() -> None:
  env, pos, metric = _edge_env(2)
  env.scene.terrain.terrain_origins[3, 1] = torch.tensor([100.0, 50.0, 0.0])
  env.scene.terrain.terrain_levels[1], env.scene.terrain.terrain_types[1] = 3, 1
  pos[:] = torch.tensor([[2.0, -1.0, 0.0], [103.0, 49.0, 0.0]])
  metric(env)
  assert metric.max_distance.tolist() == pytest.approx([2.0, 3.0])
  metric.reset(env_ids=torch.tensor([1]))
  assert metric.max_distance.tolist() == pytest.approx([2.0, 0.0])
  pos[:] = 0.0
  metric(env)  # the running maximum never decreases within an episode
  assert metric.max_distance.tolist() == pytest.approx([2.0, 100.0])


# ------------------------------------------------------------------------- task


def test_edge_task_differs_from_sgs_only_in_spawn_and_success() -> None:
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

  import ame  # noqa: F401
  from ame.env_cfg import EDGE_DISTANCE
  from ame.mdp import (
    edge_distance,
    reset_root_state_at_centre,
    reset_root_state_on_patches,
    terrain_levels_sgs,
    terrain_levels_sgs_edge,
  )
  from ame.mdp.sgs import SGS_PARAMS

  assert EDGE_TASK in list_tasks()
  sgs, edge = load_env_cfg(SGS_TASK), load_env_cfg(EDGE_TASK)
  assert sgs.events["reset_base"].func is reset_root_state_on_patches
  assert edge.events["reset_base"].func is reset_root_state_at_centre
  a, b = sgs.curriculum["terrain_levels"], edge.curriculum["terrain_levels"]
  assert a.func is terrain_levels_sgs and b.func is terrain_levels_sgs_edge
  assert a.params["rule"] == "progress"
  assert "rule" not in b.params and "track_ratio" not in b.params
  assert b.params["edge_distance"] == EDGE_DISTANCE == 3.75
  assert b.params["sgs"] == a.params["sgs"] == SGS_PARAMS
  assert edge.metrics["edge_distance"].func is edge_distance
  assert edge.metrics["edge_distance"].reduce == "last"
  # everything else identical: rewards, commands, terrain, observations, PPO
  for cfg in (sgs, edge):
    cfg.events.pop("reset_base")
    cfg.curriculum.pop("terrain_levels")
  edge.metrics.pop("edge_distance")
  norm = lambda cfg: re.sub(r" at 0x[0-9a-f]+", "", repr(cfg))  # noqa: E731
  assert norm(sgs) == norm(edge)
  assert norm(load_rl_cfg(SGS_TASK)) == norm(load_rl_cfg(EDGE_TASK))
  # play: the SGS task's play config (patch spawn on the play board)
  assert norm(load_env_cfg(SGS_TASK, play=True)) == norm(
    load_env_cfg(EDGE_TASK, play=True)
  )


def test_angstd_task_differs_from_edge_only_in_ang_vel_std() -> None:
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

  import ame  # noqa: F401

  assert ANGSTD_TASK in list_tasks()
  norm = lambda x: re.sub(r" at 0x[0-9a-f]+", "", repr(x))  # noqa: E731
  for play in (False, True):
    edge = load_env_cfg(EDGE_TASK, play=play)
    angstd = load_env_cfg(ANGSTD_TASK, play=play)
    term = angstd.rewards["track_angular_velocity"]
    assert term.params["std"] == math.sqrt(0.25) == 0.5
    assert term.weight == 2.0
    assert edge.rewards["track_angular_velocity"].params["std"] == math.sqrt(0.5)
    # every other reward term identical
    assert edge.rewards.keys() == angstd.rewards.keys()
    for name in edge.rewards:
      if name != "track_angular_velocity":
        assert norm(edge.rewards[name]) == norm(angstd.rewards[name]), name
    # and nothing else in the config differs
    edge.rewards["track_angular_velocity"].params["std"] = 0.5
    assert norm(edge) == norm(angstd)
  assert norm(load_rl_cfg(EDGE_TASK)) == norm(load_rl_cfg(ANGSTD_TASK))


def test_ang_vel_std_default_changes_nothing() -> None:
  from ame.env_cfg import ame_gap_env_cfg

  norm = lambda x: re.sub(r" at 0x[0-9a-f]+", "", repr(x))  # noqa: E731
  assert norm(ame_gap_env_cfg()) == norm(ame_gap_env_cfg(ang_vel_std=None))


def test_edge_distance_needs_sgs() -> None:
  from ame.env_cfg import ame_gap_env_cfg

  with pytest.raises(ValueError, match="sgs=True"):
    ame_gap_env_cfg(rule="progress", edge_distance=3.75)
