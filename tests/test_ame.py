"""AME port: registration, model contract, env/model geometry agreement."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict


def _obs(b=4, p_actor=96, p_critic=120):
  from ame.scan import MAP_SHAPE

  ll, w, c = MAP_SHAPE
  return TensorDict(
    {
      "actor": torch.randn(b, p_actor),
      "critic": torch.randn(b, p_critic),
      "height_map": torch.randn(b, ll * w * c),
    },
    batch_size=[b],
  )


GROUPS = {"actor": ["actor", "height_map"], "critic": ["critic", "height_map"]}


def test_task_is_registered() -> None:
  from mjlab.tasks.registry import list_tasks

  import ame  # noqa: F401

  tasks = set(list_tasks())
  assert {
    "Ame-Velocity-Rough-G1",
    "Delta-Velocity-Rough-G1",
    "Ame-Stage1-Velocity-Rough-G1",
    "Ame-Stage2-Velocity-Rough-G1",
  } <= tasks


def test_env_and_model_agree_on_map_geometry() -> None:
  """The sensor grid and the model's map_shape must describe one scan."""
  from ame.rl_cfg import ame_ppo_runner_cfg
  from ame.scan import MAP_RESOLUTION_M, MAP_SHAPE, MAP_SIZE_M

  ll, w, c = MAP_SHAPE
  assert ll == int(MAP_SIZE_M[0] / MAP_RESOLUTION_M) + 1 == 33
  assert w == int(MAP_SIZE_M[1] / MAP_RESOLUTION_M) + 1 == 21
  assert c == 3
  cfg = ame_ppo_runner_cfg()
  assert tuple(cfg.actor.map_shape) == MAP_SHAPE
  assert tuple(cfg.critic.map_shape) == MAP_SHAPE
  # Both networks must attend the map next to their own proprio group.
  assert cfg.obs_groups == {
    "actor": ("actor", "height_map"),
    "critic": ("critic", "height_map"),
  }


def test_env_cfg_is_twist_conditioned() -> None:
  from ame.env_cfg import ame_rough_env_cfg

  cfg = ame_rough_env_cfg()
  twist = cfg.commands["twist"]
  assert twist.heading_command is False
  assert twist.rel_heading_envs == 0.0
  assert twist.ranges.heading is None
  assert twist.ranges.lin_vel_y == (-0.5, 0.5)  # the deliberate vy opening
  # The base task's command curriculum would overwrite the twist ranges.
  assert "command_vel" not in cfg.curriculum
  # The scalar height_scan must not double-count the terrain.
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "height_scan" not in cfg.observations["critic"].terms
  assert "height_map" in cfg.observations
  # Play mode disables scan noise.
  play = ame_rough_env_cfg(play=True)
  assert play.observations["height_map"].terms["elevation"].params["noise"] is False
  assert cfg.observations["height_map"].terms["elevation"].params["noise"] is True


def test_model_serves_actor_and_critic() -> None:
  from ame.models import AmeModel

  obs = _obs()
  actor = AmeModel(
    obs, GROUPS, "actor", 29,
    distribution_cfg={
      "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar",
    },
  )  # fmt: skip
  critic = AmeModel(obs, GROUPS, "critic", 1)
  a = actor(obs, stochastic_output=True)
  assert a.shape == (4, 29)
  assert actor.get_output_log_prob(a).shape == (4,)
  assert critic(obs).shape == (4, 1)
  # Different proprio widths per side must resolve without tail-slicing tricks.
  assert actor._proprio_dim == 96 and critic._proprio_dim == 120


def test_model_rejects_geometry_mismatch_loudly() -> None:
  from ame.models import AmeModel

  obs = _obs()
  with pytest.raises(ValueError, match="disagree"):
    AmeModel(obs, GROUPS, "actor", 29, map_shape=(17, 11, 3))
  with pytest.raises(ValueError, match="obs_normalization"):
    AmeModel(obs, GROUPS, "actor", 29, obs_normalization=True)
  with pytest.raises(ValueError, match="lacks"):
    AmeModel(obs, {"actor": ["actor"], "critic": ["critic"]}, "actor", 29)


def test_onnx_wrapper_matches_deterministic_forward() -> None:
  from ame.models import AmeModel

  obs = _obs(b=1)
  actor = AmeModel(
    obs, GROUPS, "actor", 29,
    distribution_cfg={
      "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar",
    },
  )  # fmt: skip
  actor.eval()
  wrapper = actor.as_onnx(False)
  with torch.no_grad():
    direct = actor(obs)
    via = wrapper(obs["actor"], obs["height_map"])
  torch.testing.assert_close(direct, via)


def test_elevation_map_layout_and_miss_handling() -> None:
  """x/y are the static grid offsets; z is frame-relative; misses read the
  clamp floor rather than a fake surface at max range."""
  from types import SimpleNamespace

  from ame.mdp.observations import elevation_map

  class _Pattern:
    def generate_rays(self, mj_model, device):
      xy = torch.tensor([[0.1, 0.2, 0.0], [-0.1, 0.0, 0.0]])
      return xy, None

  b, r = 3, 2
  hit = torch.zeros(b, r, 3)
  hit[..., 2] = torch.tensor([[-0.7, -0.1], [-0.7, -0.7], [-0.7, 0.0]])
  # env 1: a miss reported at max range; env 2: mjlab's miss (-1, hit point
  # collapsed onto the sensor origin, which would otherwise read z = 0).
  data = SimpleNamespace(
    distances=torch.tensor([[1.0, 1.0], [1.0, 10.0], [1.0, -1.0]]),
    hit_pos_w=hit,
    pos_w=torch.zeros(b, 3),
  )
  sensor = SimpleNamespace(
    data=data, cfg=SimpleNamespace(pattern=_Pattern(), max_distance=10.0)
  )
  env = SimpleNamespace(
    scene={"terrain_scan": sensor},
    device="cpu",
    episode_length_buf=torch.ones(b, dtype=torch.long),
  )
  out = elevation_map(env, "terrain_scan", noise=False).reshape(b, r, 3)
  assert out[0, 0].tolist() == pytest.approx([0.1, 0.2, -0.7])
  assert out[0, 1, 2] == pytest.approx(-0.1)
  assert out[1, 1, 2] == pytest.approx(-1.2)  # the miss reads the clamp floor
  assert out[2, 1, 2] == pytest.approx(-1.2)  # mjlab's -1 miss, not a wall at z = 0


def test_terrain_mix_is_diverse_and_normalised() -> None:
  """The generator must cover the capability axes the mixer will probe:
  stairs (climb), step-over obstacles (walk on top), foot-placement terrain
  (grid/stones), and continuous ground — with proportions summing to 1."""
  from ame.terrains import ame_terrains_cfg

  gen = ame_terrains_cfg()
  names = set(gen.sub_terrains)
  assert len(names) >= 12
  for required in (
    "pyramid_stairs", "pyramid_stairs_inv", "random_stairs", "open_stairs",
    "discrete_obstacles", "random_grid", "stepping_stones",
    "random_rough", "hf_pyramid_slope", "wave", "flat",
  ):  # fmt: skip
    assert required in names, required
  total = sum(t.proportion for t in gen.sub_terrains.values())
  assert total == pytest.approx(1.0, abs=1e-6)
  assert gen.curriculum is True
  # Gap pressure: the stones sit over a deep floor, not on flat ground.
  assert gen.sub_terrains["stepping_stones"].floor_depth >= 1.0


def test_terrain_context_serves_a_foreign_actor() -> None:
  """The same encoder as a context branch: a [B, mha_dim] feature from obs."""
  from ame.models import TerrainContext

  obs = _obs()
  ctx = TerrainContext(obs, GROUPS, "actor", share_encoder=False)
  assert ctx.out_dim == 64 and ctx(obs).shape == (4, 64)
  assert ctx.query_groups == ["actor"]


def test_legacy_checkpoints_load_into_the_split_model() -> None:
  """Pre-split state dicts kept map_cnn.* / mha.* on the model itself."""
  from ame.models import AmeModel

  obs = _obs()
  a = AmeModel(obs, GROUPS, "actor", 29)
  legacy = {
    (k.replace("terrain.", "", 1) if k.startswith("terrain.") else k): v
    for k, v in a.state_dict().items()
  }
  assert any(k.startswith("map_cnn.") for k in legacy)
  b = AmeModel(obs, GROUPS, "actor", 29)
  b.load_state_dict(legacy)
  for k, v in a.state_dict().items():
    assert torch.equal(b.state_dict()[k], v), k


def test_two_stage_recipe() -> None:
  """SII-FUSC's stage-1 / stage-2 arrangement (terrain shares, noise, DR, spawn)."""
  from ame.env_cfg import ame_stage_env_cfg

  s1, s2 = ame_stage_env_cfg(1), ame_stage_env_cfg(2)
  p1 = {
    k: v.proportion for k, v in s1.scene.terrain.terrain_generator.sub_terrains.items()
  }
  p2 = {
    k: v.proportion for k, v in s2.scene.terrain.terrain_generator.sub_terrains.items()
  }
  assert abs(sum(p1.values()) - 1.0) < 1e-6 and abs(sum(p2.values()) - 1.0) < 1e-6
  assert abs(p1["stepping_stones"] + p1["gaps"] - 0.40) < 1e-9
  assert abs(p2["stakes"] + p2["beams"] + p2["gaps"] - 0.80) < 1e-9
  # Stage 1 is the CLEAN stage: no scan noise, no proprio noise, no push/COM.
  assert s1.observations["height_map"].terms["elevation"].params["noise"] is False
  assert all(t.noise is None for t in s1.observations["actor"].terms.values())
  assert "push_robot" not in s1.events and "base_com" not in s1.events
  assert s2.observations["height_map"].terms["elevation"].params["noise"] is True
  assert "push_robot" in s2.events and "base_com" in s2.events
  pose = s2.events["reset_base"].params["pose_range"]
  assert pose["x"] == (0.0, 0.0) and pose["yaw"] == (0.0, 0.0)
  assert s1.events["reset_base"].params["pose_range"]["yaw"] == (-3.14, 3.14)
  assert (s1.rewards["feet_stumble"].weight, s2.rewards["feet_stumble"].weight) == (
    -2.0,
    -5.0,
  )
  assert (
    s1.rewards["feet_air_time_variance"].weight,
    s2.rewards["feet_air_time_variance"].weight,
  ) == (-0.1, -0.7)
  for cfg in (s1, s2):  # our twist tracking is kept
    assert cfg.rewards["track_lin_vel_xy_exp"].weight == 3.0
    assert cfg.rewards["track_ang_vel_z_exp"].weight == 6.0


def test_deploy_observations_drop_the_privileged_velocity() -> None:
  from ame.env_cfg import ame_rough_env_cfg, apply_deploy_observations

  cfg = ame_rough_env_cfg()
  apply_deploy_observations(cfg)
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms
  with pytest.raises(ValueError, match="base_lin_vel"):
    apply_deploy_observations(cfg)


def test_stones_board_is_a_foothold_task() -> None:
  from ame.terrains import stones_board_cfg

  gen = stones_board_cfg()
  assert abs(sum(t.proportion for t in gen.sub_terrains.values()) - 1.0) < 1e-6
  stones = gen.sub_terrains["stepping_stones"]
  # At full difficulty a 0.45 m gap is wider than a foot is long; regular lattice.
  assert stones.stone_distance_range[1] >= 0.4 and stones.stone_height == 0.0
  assert stones.displacement_range == 0.0 and stones.stone_size_variation == 0.0
  assert gen.sub_terrains["gaps"].height_range == (0.0, 0.0)


def test_gap_board_is_a_gap_curriculum() -> None:
  """The stone gap is the only knob: row 0 is a seamless floor, the top row
  a 0.45 m gap at the same 0.40 m stone; no centre island; spawns are stone
  centres away from the tile edge."""
  import mujoco
  import numpy as np

  from ame.terrains import TILE, BoxStoneLatticeTerrainCfg, gap_board_cfg

  gen = gap_board_cfg()
  subs = gen.sub_terrains
  assert {k: t.proportion for k, t in subs.items()} == {
    "stepping_stones": 0.40,
    "pyramid_stairs": 0.20,
    "pyramid_stairs_inv": 0.20,
    "wave": 0.10,
    "random_rough": 0.10,
  }
  lattice = subs["stepping_stones"]
  assert isinstance(lattice, BoxStoneLatticeTerrainCfg)
  assert lattice.gap(0.0) == 0.0 and abs(lattice.gap(1.0) - 0.45) < 1e-9
  for d in (0.0, 0.5, 1.0):
    grid, ring = lattice.centres(d)
    xs = np.unique(np.round(grid[:, 0], 6))
    pitch = lattice.stone_size + lattice.gap(d)
    assert np.allclose(np.diff(xs), pitch)  # uniform stones, uniform gaps
    # The ring meets the outermost stone across exactly one gap, on both sides.
    assert abs(xs[0] - lattice.stone_size / 2 - lattice.gap(d) - ring) < 1e-9
    assert abs(TILE[0] - xs[-1] - lattice.stone_size / 2 - lattice.gap(d) - ring) < 1e-9
    assert ring >= lattice.border_width
    # No island: the centre is either a stone or a gap like any other cell.
    assert not np.any(np.all(np.abs(grid - np.asarray(TILE) / 2) < 1e-9, axis=1)) or (
      lattice.gap(d) < lattice.stone_size
    )
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    out = lattice.function(d, spec, np.random.default_rng(0))
    assert out.flat_patches is not None and out.flat_patches["spawn"].shape == (64, 3)
    patches = out.flat_patches["spawn"]
    assert np.all(patches[:, 2] == 0.0)
    edge = np.minimum(patches[:, :2], np.asarray(TILE) - patches[:, :2]).min(axis=1)
    assert np.all(edge >= lattice.spawn_margin - 1e-9)
    d_centre = np.abs(patches[:, None, :2] - grid[None]).max(axis=2).min(axis=1)
    assert np.all(d_centre < 1e-9)  # every spawn is a stone centre
    boxes = [
      g for g in spec.body("terrain").geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX
    ]
    assert len(boxes) == len(grid) + 1 + 4  # stones, floor, ring
  assert np.allclose(np.diff(np.unique(lattice.centres(0.0)[0][:, 0])), 0.40)
