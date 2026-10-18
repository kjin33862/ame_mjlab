"""Gap-board task: registration, config composition, curriculum rule."""

from __future__ import annotations

from types import SimpleNamespace

import torch

TASK = "Ame-Mjlab-Gap-Velocity-Rough-G1"


def test_gap_task_is_registered() -> None:
  from mjlab.tasks.registry import list_tasks

  import ame  # noqa: F401

  assert TASK in set(list_tasks())


def test_gap_env_cfg_composes_the_recipe() -> None:
  from ame.env_cfg import GAP_CURRICULUM, ame_gap_env_cfg
  from ame.mdp import reset_root_state_on_patches, terrain_levels_floor

  cfg = ame_gap_env_cfg()
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms
  assert cfg.events["reset_base"].func is reset_root_state_on_patches
  term = cfg.curriculum["terrain_levels"]
  assert term.func is terrain_levels_floor
  assert term.params["rule"] == GAP_CURRICULUM["rule"] == "tracking"
  assert cfg.scene.terrain.max_init_terrain_level == GAP_CURRICULUM["spawn_max"]
  assert {"pelvis_speed", "command_speed"} <= set(cfg.metrics)
  assert cfg.rewards["track_angular_velocity"].weight == 2.0


def test_gap_play_cfg_has_no_curriculum() -> None:
  from ame.env_cfg import ame_gap_env_cfg

  cfg = ame_gap_env_cfg(play=True)
  assert "terrain_levels" not in cfg.curriculum
  assert "base_lin_vel" not in cfg.observations["actor"].terms


class _Scene(SimpleNamespace):
  """A scene stub indexable by entity name, as ``env.scene["robot"]``."""

  def __getitem__(self, key: str):
    return getattr(self, key)


def _stub_env(levels: list[int], moved: list[float], asked: list[float]):
  n = len(levels)
  calls: dict[str, torch.Tensor] = {}

  def update_env_origins(env_ids, move_up, move_down):
    calls["up"], calls["down"] = move_up.clone(), move_down.clone()

  terrain = SimpleNamespace(
    terrain_levels=torch.tensor(levels),
    terrain_types=torch.zeros(n, dtype=torch.long),
    terrain_origins=torch.zeros(10, 1, 3),
    env_origins=torch.zeros(n, 3),
    max_terrain_level=10,
    update_env_origins=update_env_origins,
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(size=(8.0, 8.0), sub_terrains={"s": None})
    ),
  )
  robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(n, 3)))
  env = SimpleNamespace(
    scene=_Scene(terrain=terrain, env_origins=torch.zeros(n, 3), robot=robot),
    common_step_counter=0,
    command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(n, 3)),
    termination_manager=SimpleNamespace(time_outs=torch.ones(n, dtype=torch.bool)),
    episode_length_buf=torch.full((n,), 1000),
    max_episode_length=1000,
    max_episode_length_s=20.0,
    metrics_manager=SimpleNamespace(
      active_terms=["pelvis_speed", "command_speed"],
      _episode_sums={
        "pelvis_speed": torch.tensor(moved),
        "command_speed": torch.tensor(asked),
      },
    ),
  )
  return env, calls


def test_tracking_rule_promotes_progress_and_demotes_a_stall() -> None:
  from ame.mdp import terrain_levels_floor

  # Both episodes timed out at level 3 under the same command; one moved at
  # 80 % of the commanded speed, the other stalled at 10 %.
  env, calls = _stub_env(levels=[3, 3], moved=[0.8, 0.1], asked=[1.0, 1.0])
  cfg = SimpleNamespace(params={"floor": 0, "rule": "tracking", "track_ratio": 0.6})
  term = terrain_levels_floor(cfg, env)
  ids = torch.arange(2)
  term(env, ids, command_name="twist", floor=0, rule="tracking")  # arming call
  term(env, ids, command_name="twist", floor=0, rule="tracking")
  assert calls["up"].tolist() == [True, False]
  assert calls["down"].tolist() == [False, True]


def test_tracking_rule_never_demotes_below_the_floor() -> None:
  from ame.mdp import terrain_levels_floor

  env, calls = _stub_env(levels=[2], moved=[0.0], asked=[1.0])
  cfg = SimpleNamespace(params={"floor": 2, "rule": "tracking"})
  term = terrain_levels_floor(cfg, env)
  ids = torch.arange(1)
  term(env, ids, command_name="twist", floor=2, rule="tracking")
  term(env, ids, command_name="twist", floor=2, rule="tracking")
  assert calls["down"].tolist() == [False]
