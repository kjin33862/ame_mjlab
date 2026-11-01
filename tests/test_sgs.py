"""Success-Guided Sampling: weights, ring buffers, sampling and the task."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from ame.mdp.sgs import SGS_PARAMS, OutcomeBuffer, sgs_probs, sgs_weights

T, KAPPA, EPS = SGS_PARAMS["t"], SGS_PARAMS["kappa"], SGS_PARAMS["eps"]


def test_weight_peaks_at_the_target_rate() -> None:
  p = torch.linspace(0, 1, 10001, dtype=torch.float64)
  w = sgs_weights(p, T, KAPPA, EPS)
  assert float(p[torch.argmax(w)]) == pytest.approx(T, abs=1e-4)


def test_certain_failure_and_success_hit_the_floor() -> None:
  w = sgs_weights(torch.tensor([0.0, 1.0], dtype=torch.float64), T, KAPPA, EPS)
  assert w.tolist() == [EPS, EPS]


def test_empty_buffer_is_p_zero() -> None:
  buffer = OutcomeBuffer(families=2, levels=3, H=100, device="cpu")
  assert torch.equal(buffer.success_rate(), torch.zeros(2, 3))
  buffer.record(torch.tensor([1]), torch.tensor([2]), torch.tensor([True]))
  p = buffer.success_rate()
  assert float(p[1, 2]) == 1.0
  assert float(p.sum()) == 1.0  # every other configuration is still empty


def test_ring_buffer_overwrites_the_oldest_after_h_entries() -> None:
  buffer = OutcomeBuffer(families=1, levels=1, H=100, device="cpu")
  cell = (torch.tensor([0]), torch.tensor([0]))
  for _ in range(100):
    buffer.record(*cell, torch.tensor([True]))
  assert int(buffer.fill[0, 0]) == 100 and float(buffer.success_rate()[0, 0]) == 1.0
  for i in range(30):
    buffer.record(*cell, torch.tensor([False]))
    assert int(buffer.fill[0, 0]) == 100
    assert float(buffer.success_rate()[0, 0]) == pytest.approx((99 - i) / 100)


def test_ring_buffer_takes_simultaneous_episodes_in_order() -> None:
  # 130 envs end on one configuration in one call: only the last 100 count.
  buffer = OutcomeBuffer(families=1, levels=2, H=100, device="cpu")
  outcome = torch.tensor([True] * 30 + [False] * 60 + [True] * 40)
  buffer.record(
    torch.zeros(130, dtype=torch.long), torch.ones(130, dtype=torch.long), outcome
  )
  assert buffer.fill.tolist() == [[0, 100]]
  assert float(buffer.success_rate()[0, 1]) == pytest.approx(0.4)
  assert int(buffer.head[0, 1]) == 30


def test_equal_success_rates_sample_uniformly() -> None:
  s = SGS_PARAMS
  for value in (0.0, 0.3, T, 1.0):
    probs = sgs_probs(torch.full((5, 10), value), s["t"], s["kappa"], s["T"], s["eps"])
    assert torch.allclose(probs, torch.full((5, 10), 0.1))


def test_probabilities_favour_the_target_level() -> None:
  s = SGS_PARAMS
  p = torch.tensor([[1.0, 0.95, T, 0.2, 0.0]])
  probs = sgs_probs(p, s["t"], s["kappa"], s["T"], s["eps"])
  assert float(probs.sum()) == pytest.approx(1.0)
  assert int(torch.argmax(probs)) == 2


class _Scene(SimpleNamespace):
  """A scene stub indexable by entity name, as ``env.scene["robot"]``."""

  def __getitem__(self, key: str):
    return getattr(self, key)


def _stub_env(types: list[int], levels: list[int], distance: list[float]):
  n = len(types)
  terrain = SimpleNamespace(
    terrain_levels=torch.tensor(levels),
    terrain_types=torch.tensor(types),
    terrain_origins=torch.arange(10 * 2 * 3, dtype=torch.float).view(10, 2, 3),
    env_origins=torch.zeros(n, 3),
    max_terrain_level=10,
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(size=(8.0, 8.0), sub_terrains={"a": 0, "b": 0})
    ),
  )
  pos = torch.zeros(n, 3)
  pos[:, 0] = torch.tensor(distance)
  robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos))
  env = SimpleNamespace(
    scene=_Scene(terrain=terrain, env_origins=torch.zeros(n, 3), robot=robot),
    common_step_counter=0,
    command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(n, 3)),
    # Fallen before 10 s: only the 4 m end point can count as success.
    termination_manager=SimpleNamespace(time_outs=torch.zeros(n, dtype=torch.bool)),
    episode_length_buf=torch.full((n,), 100),
    max_episode_length=1000,
    max_episode_length_s=20.0,
    metrics_manager=SimpleNamespace(
      active_terms=["pelvis_progress", "command_speed"],
      _episode_sums={"pelvis_progress": torch.zeros(n), "command_speed": torch.ones(n)},
    ),
  )
  return env


def test_sgs_term_records_move_up_and_samples_within_the_family() -> None:
  from ame.mdp import terrain_levels_sgs

  env = _stub_env(types=[0, 0, 1], levels=[3, 3, 5], distance=[5.0, 1.0, 5.0])
  cfg = SimpleNamespace(params={"floor": 0, "rule": "progress", "sgs": {}})
  term = terrain_levels_sgs(cfg, env)
  ids = torch.arange(3)
  kwargs = {"command_name": "twist", "floor": 0, "rule": "progress"}
  term(env, ids, **kwargs)  # construction-pose reset: nothing recorded
  assert int(term.buffer.fill.sum()) == 0
  assert env.scene.terrain.terrain_levels.tolist() == [3, 3, 5]
  logs = term(env, ids, **kwargs)
  assert term.buffer.fill[0, 3] == 2 and term.buffer.fill[1, 5] == 1
  assert float(term.buffer.success_rate()[0, 3]) == 0.5  # 4 m vs 1 m
  assert float(term.buffer.success_rate()[1, 5]) == 1.0
  terrain = env.scene.terrain
  assert terrain.terrain_types.tolist() == [0, 0, 1]
  assert torch.equal(
    terrain.env_origins,
    terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
  )
  for name in ("a", "b"):
    assert sum(logs[f"{name}/prob_{level}"] for level in range(10)) == pytest.approx(
      1.0
    )
  assert logs["a/p_3"] == 0.5 and logs["a/fill_min"] == 0.0
  assert "a" in logs and "mean" in logs  # the floor term's logs are kept


def test_sgs_task_differs_from_heading_only_in_the_curriculum() -> None:
  from ame.env_cfg import ame_gap_env_cfg
  from ame.mdp import terrain_levels_floor, terrain_levels_sgs

  heading = ame_gap_env_cfg(rule="progress", heading_command=True)
  sgs = ame_gap_env_cfg(rule="progress", heading_command=True, sgs=True)
  a, b = heading.curriculum.pop("terrain_levels"), sgs.curriculum.pop("terrain_levels")
  assert a.func is terrain_levels_floor and b.func is terrain_levels_sgs
  assert {k: v for k, v in b.params.items() if k != "sgs"} == a.params
  assert b.params["sgs"] == SGS_PARAMS
  assert sgs.scene.terrain.max_init_terrain_level == 2
  norm = lambda cfg: re.sub(r" at 0x[0-9a-f]+", "", repr(cfg))  # noqa: E731
  assert norm(heading) == norm(sgs)


def test_sgs_task_is_registered() -> None:
  from mjlab.tasks.registry import list_tasks

  import ame  # noqa: F401

  assert "Ame-Mjlab-Gap-ProgressHeading-SGS-Velocity-Rough-G1" in list_tasks()


def test_sgs_term_returns_logs_when_no_env_ended() -> None:
  from ame.mdp import terrain_levels_sgs

  env = _stub_env(types=[0, 1], levels=[3, 5], distance=[5.0, 5.0])
  cfg = SimpleNamespace(params={"floor": 0, "rule": "progress", "sgs": {}})
  term = terrain_levels_sgs(cfg, env)
  kwargs = {"command_name": "twist", "floor": 0, "rule": "progress"}
  term(env, torch.arange(2), **kwargs)  # arming call
  logs = term(env, torch.arange(0), **kwargs)
  assert int(term.buffer.fill.sum()) == 0
  assert env.scene.terrain.terrain_levels.tolist() == [3, 5]
  assert sum(logs[f"a/prob_{level}"] for level in range(10)) == pytest.approx(1.0)
