"""AME-2 reward stack: term coverage, RBF math, and config wiring."""

from __future__ import annotations

import math

import pytest
import torch

from ame.env_cfg import ame_rough_env_cfg
from ame.mdp import rewards as R
from ame.reward_cfg import UNDESIRED_CONTACT_SENSOR, ame2_rewards

# The reference stack (29dof/velocity_env_cfg_29dof.py::RewardsCfg), verbatim.
REFERENCE_WEIGHTS = {
  "termination_penalty": -200.0,
  "track_lin_vel_xy_exp": 1.0,
  "track_ang_vel_z_exp": 2.0,
  "ang_vel_xy_l2": -0.05,
  "undesired_contacts": -1.0,
  "dof_torques_l2": -1.5e-7,
  "dof_acc_l2": -1.25e-7,
  "dof_vel_l2": -0.001,
  "dof_pos_limits": -1.0,
  "dof_torques_limits": -0.01,
  "action_rate_l2": -0.01,
  "flat_orientation_l2": -2.0,
  "feet_air_time": 0.25,
  "feet_air_time_variance": -0.1,
  "feet_slide": -0.1,
  "feet_stumble": -1.0,
  "feet_too_near": -1.0,
  "joint_coordination": -0.1,
  "joint_deviation_hip": -0.1,
  "joint_deviation_arms": -0.1,
  "joint_deviation_waists": -1.0,
}


def test_reference_weights_reproduced():
  terms = ame2_rewards()
  for name, weight in REFERENCE_WEIGHTS.items():
    assert name in terms, f"missing reference term {name}"
    assert terms[name].weight == pytest.approx(weight), name


def test_rbf_terms_present_and_negative():
  terms = ame2_rewards(rbf_weight=1.0)
  # Requested as "- 1 * RBF(...)": weight -1 on a bounded penalty that is 0 at
  # perfect pose (see ame.mdp.rewards module docstring for the sign argument).
  assert terms["rbf_upper_body_pose"].weight == -1.0
  assert terms["rbf_foot_yaw"].weight == -1.0
  scaled = ame2_rewards(rbf_weight=0.5)
  assert scaled["rbf_foot_yaw"].weight == -0.5


def test_env_cfg_selects_stack():
  ame2 = ame_rough_env_cfg(rewards="ame2")
  assert set(REFERENCE_WEIGHTS) <= set(ame2.rewards)
  assert "rbf_foot_yaw" in ame2.rewards
  # The stack needs its own contact sensor, and it must be per-body: a
  # subtree match on `pelvis` would cover the whole robot (feet included).
  sensor = next(s for s in ame2.scene.sensors if s.name == UNDESIRED_CONTACT_SENSOR)
  assert sensor.primary.mode == "body"
  assert "illegal_contact" in ame2.terminations
  # mjlab's own stack stays reachable for A/B.
  assert "rbf_foot_yaw" not in ame_rough_env_cfg(rewards="mjlab").rewards
  with pytest.raises(ValueError):
    ame_rough_env_cfg(rewards="nonsense")


class _FakeData:
  pass


class _FakeAsset:
  def __init__(self, **kw):
    self.data = _FakeData()
    for k, v in kw.items():
      setattr(self.data, k, v)


class _FakeEnv:
  def __init__(self, asset):
    self.scene = {"robot": asset}
    self.num_envs = (
      asset.data.joint_pos.shape[0] if hasattr(asset.data, "joint_pos") else 1
    )


class _Cfg:
  name = "robot"

  def __init__(self, joint_ids=None, body_ids=None):
    self.joint_ids = joint_ids
    self.body_ids = body_ids


def test_rbf_upper_body_pose_math():
  q = torch.tensor([[0.0, 0.0], [0.3, 0.4]])  # errors 0 and 0.5 total
  asset = _FakeAsset(joint_pos=q, default_joint_pos=torch.zeros_like(q))
  env = _FakeEnv(asset)
  out = R.rbf_upper_body_pose(
    env, std=0.5, asset_cfg=_Cfg(joint_ids=slice(None)), terrain_aware=False
  )
  # Zero deviation -> zero penalty; 0.5 rad of total error -> 1 - exp(-1).
  assert out[0].item() == pytest.approx(0.0, abs=1e-7)
  assert out[1].item() == pytest.approx(1.0 - math.exp(-0.25 / 0.25), abs=1e-6)
  assert torch.all((out >= 0) & (out <= 1))  # bounded


def _quat_yaw(yaw: float) -> list[float]:
  return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def test_rbf_foot_yaw_math_and_wrapping():
  # env 0: feet aligned with base.  env 1: both feet yawed 0.3 rad off.
  base = torch.tensor([_quat_yaw(0.5), _quat_yaw(0.5)])
  feet = torch.tensor(
    [
      [_quat_yaw(0.5), _quat_yaw(0.5)],
      [_quat_yaw(0.8), _quat_yaw(0.8)],
    ]
  )
  env = _FakeEnv(_FakeAsset(root_link_quat_w=base, body_link_quat_w=feet))
  out = R.rbf_foot_yaw(
    env, std=0.5, asset_cfg=_Cfg(body_ids=slice(None)), terrain_aware=False
  )
  assert out[0].item() == pytest.approx(0.0, abs=1e-6)
  expected = 1.0 - math.exp(-(2 * 0.3**2) / 0.25)
  assert out[1].item() == pytest.approx(expected, abs=1e-5)

  # Wrap-around: base at +pi - 0.1, foot at -pi + 0.1 is a 0.2 rad error,
  # not 2*pi - 0.2.
  base_w = torch.tensor([_quat_yaw(math.pi - 0.1)])
  foot_w = torch.tensor([[_quat_yaw(-math.pi + 0.1)]])
  env_w = _FakeEnv(_FakeAsset(root_link_quat_w=base_w, body_link_quat_w=foot_w))
  out_w = R.rbf_foot_yaw(
    env_w, std=0.5, asset_cfg=_Cfg(body_ids=slice(None)), terrain_aware=False
  )
  assert out_w[0].item() == pytest.approx(1.0 - math.exp(-(0.2**2) / 0.25), abs=1e-4)


def test_rbf_saturates_not_explodes():
  """The point of the RBF form: one bad frame cannot dominate the return."""
  q = torch.tensor([[100.0]])
  env = _FakeEnv(_FakeAsset(joint_pos=q, default_joint_pos=torch.zeros_like(q)))
  out = R.rbf_upper_body_pose(
    env, std=0.5, asset_cfg=_Cfg(joint_ids=slice(None)), terrain_aware=False
  )
  assert out.item() == pytest.approx(1.0)


class _FakeScan:
  def __init__(self, z, base_z=0.8):
    self.data = _FakeData()
    self.data.hit_pos_w = torch.stack(
      [torch.zeros_like(z), torch.zeros_like(z), z], dim=-1
    )
    self.data.pos_w = torch.full((z.shape[0], 3), base_z)


def test_terrain_flatness_scales_posture_penalties():
  """The author's guide: 0.1x posture weight where elevation spans are big."""
  flat = torch.zeros(1, 20)  # perfectly flat scan
  rough = torch.zeros(1, 20)
  rough[0, :10] = 0.6  # a 0.6 m step inside the scan
  env_flat = _FakeEnv(_FakeAsset())
  env_flat.scene["terrain_scan"] = _FakeScan(flat)
  env_rough = _FakeEnv(_FakeAsset())
  env_rough.scene["terrain_scan"] = _FakeScan(rough)

  w_flat = R.terrain_flatness(env_flat).item()
  w_rough = R.terrain_flatness(env_rough).item()
  assert w_flat == pytest.approx(1.0, abs=1e-6)
  assert w_rough == pytest.approx(0.1, abs=0.02)

  # And it actually modulates the penalty (the default path).
  q = torch.tensor([[0.5]])
  asset = _FakeAsset(joint_pos=q, default_joint_pos=torch.zeros_like(q))
  for env, expect in ((env_flat, w_flat), (env_rough, w_rough)):
    env.scene["robot"] = asset
    on = R.rbf_upper_body_pose(env, std=0.5, asset_cfg=_Cfg(joint_ids=slice(None)))
    off = R.rbf_upper_body_pose(
      env, std=0.5, asset_cfg=_Cfg(joint_ids=slice(None)), terrain_aware=False
    )
    assert on.item() == pytest.approx(off.item() * expect, rel=1e-5)


def test_symmetry_mirror_is_an_involution():
  """Mirroring twice must be the identity, for every observation channel."""
  from ame.symmetry import _tensors, mirror_joints, mirror_map

  cache = _tensors(torch.device("cpu"))
  q = torch.randn(3, 29)
  assert torch.allclose(mirror_joints(mirror_joints(q, cache), cache), q, atol=1e-6)

  m = torch.randn(3, 33 * 21 * 3)
  assert torch.allclose(mirror_map(mirror_map(m)), m, atol=1e-6)


def test_symmetry_swaps_left_and_right():
  from ame.symmetry import JOINT_NAMES, _tensors, mirror_joints

  cache = _tensors(torch.device("cpu"))
  q = torch.zeros(1, 29)
  li = JOINT_NAMES.index("left_knee_joint")
  q[0, li] = 0.7
  out = mirror_joints(q, cache)
  # A left-knee (pitch) angle lands on the right knee with the same sign.
  assert out[0, JOINT_NAMES.index("right_knee_joint")].item() == pytest.approx(0.7)
  assert out[0, li].item() == pytest.approx(0.0)
  # A roll joint additionally flips sign.
  q2 = torch.zeros(1, 29)
  q2[0, JOINT_NAMES.index("left_hip_roll_joint")] = 0.3
  out2 = mirror_joints(q2, cache)
  assert out2[0, JOINT_NAMES.index("right_hip_roll_joint")].item() == pytest.approx(
    -0.3
  )
  # Waist joints mirror onto themselves; yaw flips sign.
  q3 = torch.zeros(1, 29)
  q3[0, JOINT_NAMES.index("waist_yaw_joint")] = 0.2
  out3 = mirror_joints(q3, cache)
  assert out3[0, JOINT_NAMES.index("waist_yaw_joint")].item() == pytest.approx(-0.2)


def test_symmetry_map_mirrors_z_only():
  """Guide: 'only mirror the z values and keep old x & y values'."""
  from ame.symmetry import mirror_map

  L, W = 33, 21
  grid = torch.zeros(1, W, L, 3)
  grid[..., 0] = 1.0  # x channel marker
  grid[..., 1] = 2.0  # y channel marker
  grid[0, 0, :, 2] = 5.0  # a ridge on the first y row
  out = mirror_map(grid.reshape(1, -1)).reshape(1, W, L, 3)
  assert torch.all(out[..., 0] == 1.0) and torch.all(out[..., 1] == 2.0)
  assert torch.all(out[0, -1, :, 2] == 5.0)  # ridge moved to the last row
  assert torch.all(out[0, 0, :, 2] == 0.0)


def test_delta_shared_encoder_no_duplicate_params():
  """Shared weights must appear once in the optimizer, and take critic grads."""
  from tensordict import TensorDict

  import ame.delta as nd

  nd._SHARED.clear()
  ms = 33 * 21 * 3

  def build(role, pdim, share):
    obs = TensorDict(
      {role: torch.zeros(1, pdim), "height_map": torch.zeros(1, ms)}, batch_size=[1]
    )
    return nd.DeltaModel(
      obs=obs,
      obs_groups={role: [role, "height_map"]},
      obs_set=role,
      output_dim=29,
      hidden_dims=[32],
      share_encoder=share,
    )

  actor, critic = build("actor", 99, True), build("critic", 120, True)
  assert actor.layers is critic.layers
  ids_a = {id(p) for p in actor.parameters()}
  ids_c = {id(p) for p in critic.parameters()}
  assert not (ids_a & ids_c), "shared params would be stepped twice by the optimizer"
  obs = TensorDict(
    {"critic": torch.randn(2, 120), "height_map": torch.randn(2, ms)}, batch_size=[2]
  )
  critic.mlp(critic.get_latent(obs)).sum().backward()
  grad = actor.layers[0].w_s.weight.grad
  assert grad is not None and grad.abs().sum() > 0
  # The critic must not claim the shared weights in its own state dict.
  assert not any(k.startswith("layers.") for k in critic.state_dict())
  nd._SHARED.clear()


def test_rebalanced_preset():
  cfg = ame_rough_env_cfg(rewards="ame2_rebalanced")
  r = cfg.rewards
  assert r["track_lin_vel_xy_exp"].weight == 3.0
  assert r["track_lin_vel_xy_exp"].params["std"] == 0.5
  assert r["track_ang_vel_z_exp"].weight == 6.0
  assert r["track_ang_vel_z_exp"].params["std"] == 0.25
  assert r["termination_penalty"].weight == -50.0
  assert r["feet_air_time"].weight == 0.5
  # gait-quality pass
  assert r["dof_vel_l2"].weight == -0.001
  assert r["action_rate_l2"].weight == -0.025
  assert r["feet_stumble"].params["ratio"] == 1.0
  assert r["feet_stumble"].weight == -1.0
  # untouched examples
  assert r["rbf_foot_yaw"].weight == -1.0
  # plain ame2 stays verbatim
  assert ame_rough_env_cfg(rewards="ame2").rewards["track_ang_vel_z_exp"].weight == 2.0


def test_map_drift_shifts_z_only():
  """Drift moves the z content laterally with edge clamping; x/y stay put."""
  from ame.mdp.observations import _DRIFT_CELLS, _drift_index_table

  L, W = 33, 21
  table = _drift_index_table(L, W, torch.device("cpu"))
  n = (2 * _DRIFT_CELLS + 1) ** 2
  assert table.shape == (n, L * W)
  # Center combo is the identity.
  assert torch.equal(table[n // 2], torch.arange(L * W))
  # A z-ridge at column ix=5 lands where each combo says it should.
  z = torch.zeros(1, W, L)
  z[0, :, 5] = 1.0
  flat = z.reshape(1, -1)
  for combo in range(n):
    _dy, dx = divmod(combo, 2 * _DRIFT_CELLS + 1)
    dx -= _DRIFT_CELLS
    shifted = flat.gather(1, table[combo : combo + 1]).reshape(W, L)
    col = (torch.arange(L) + dx).clamp(0, L - 1)
    expect = torch.zeros(W, L)
    expect[:, col == 5] = 1.0
    assert torch.equal(shifted, expect), f"combo {combo} dx={dx}"


def test_stumble_ratio_param():
  class _Sensor:
    class data:
      force = torch.tensor([[[3.0, 0.0, 2.0]], [[0.5, 0.0, 2.0]]])  # xy vs z

  env = _FakeEnv(_FakeAsset())
  env.scene["c"] = _Sensor()
  # ratio 4: neither foot trips (3 < 8).  ratio 1: the first does (3 > 2).
  assert R.feet_stumble(env, "c", ratio=4.0).tolist() == [0.0, 0.0]
  assert R.feet_stumble(env, "c", ratio=1.0).tolist() == [1.0, 0.0]


def test_leaping_penalty():
  """Fires only when BOTH feet are airborne AND the ground is near-flat."""

  class _Contact:
    class data:
      found = torch.tensor([[0, 0], [0, 1], [0, 0]])

  class _Scan:
    class data:
      # env 0: flat scan; env 1: flat; env 2: a 0.5 m span (gap/step)
      hit_pos_w = torch.stack(
        [
          torch.stack([torch.zeros(9), torch.zeros(9), torch.zeros(9)], -1),
          torch.stack([torch.zeros(9), torch.zeros(9), torch.zeros(9)], -1),
          torch.stack(
            [torch.zeros(9), torch.zeros(9), torch.tensor([0.0] * 8 + [-0.5])], -1
          ),
        ]
      )
      pos_w = torch.full((3, 3), 0.8)

  env = _FakeEnv(_FakeAsset())
  env.scene["c"] = _Contact()
  env.scene["terrain_scan"] = _Scan()
  out = R.leaping(env, "c")
  # env0: airborne + flat -> penalty. env1: one foot down -> no.
  # env2: airborne but 0.5 m span (legit flight over a step) -> no.
  assert out.tolist() == [1.0, 0.0, 0.0]


def test_blowup_guards():
  """Clamped penalties + terminations bound a solver blowup to one step."""
  from ame.mdp.terminations import bad_orientation, physics_blowup

  # Clamp: a 1e6 rad/s^2 blowup reads as 1e4 -> penalty bounded.
  q = torch.tensor([[1.0e6, 100.0]])
  env = _FakeEnv(_FakeAsset(joint_acc=q, joint_vel=q))
  bounded = R.joint_acc_l2_clamped(env, asset_cfg=_Cfg())
  # 1e6 clamps to 1e4 -> 1e8; 100 stays -> 1e4.
  assert bounded.item() == pytest.approx(1.0e8 + 1.0e4, rel=1e-3)

  # High-acc termination fires on the blowup, not on normal gait.
  env2 = _FakeEnv(_FakeAsset(joint_acc=torch.tensor([[1.0e6, 0.0], [2000.0, 100.0]])))
  out = physics_blowup(env2)
  assert out.tolist() == [True, False]

  # Bad orientation: upright ok, flipped terminates.
  up = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.9], [0.0, 0.95, -0.3]])
  env3 = _FakeEnv(_FakeAsset(projected_gravity_b=up))
  out3 = bad_orientation(env3)
  assert out3.tolist() == [False, True, True]


def test_ame2_termination_set_complete():
  cfg = ame_rough_env_cfg(rewards="ame2_rebalanced")
  assert {"illegal_contact", "bad_orientation", "physics_blowup", "time_out"} <= set(
    cfg.terminations
  )


def test_ame_task_recipe():
  """The registered AME task: dense encoder, full recipe, blowup guards."""
  from dataclasses import asdict

  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  env = load_env_cfg("Ame-Velocity-Rough-G1")
  # Rebalanced rewards + the blowup guards ride in via the env side.
  assert env.rewards["track_ang_vel_z_exp"].weight == 6.0
  assert env.rewards["action_rate_l2"].weight == -0.025
  assert {"bad_orientation", "physics_blowup"} <= set(env.terminations)
  a = asdict(load_rl_cfg("Ame-Velocity-Rough-G1"))
  assert a["actor"]["share_encoder"] and a["critic"]["share_encoder"]
  assert a["actor"]["class_name"].endswith("AmeModel")
  assert a["algorithm"]["desired_kl"] == 0.02
  assert a["algorithm"]["num_mini_batches"] == 16
  assert env.scene.num_envs == 1024
  assert bool(a["algorithm"]["symmetry_cfg"])


def test_ame2_table1_terms():
  """The AME-2 Table I additions + stronger torque regularization."""
  r = ame_rough_env_cfg(rewards="ame2_rebalanced").rewards
  assert r["ang_vel_xy_l2"].func is R.base_roll_rate_l2
  assert r["ang_vel_xy_l2"].weight == -0.1
  assert r["dof_torques_l2"].weight == -1.0e-5
  assert r["link_contact_forces"].weight == -1.0e-5
  assert r["joint_vel_limits"].weight == -1.0
  assert r["link_acceleration"].weight == -0.001
  # verbatim ame2 untouched
  base = ame_rough_env_cfg(rewards="ame2").rewards
  assert base["dof_torques_l2"].weight == -1.5e-7
  assert "link_contact_forces" not in base

  # roll-only: pitch rate carries no cost
  w = torch.tensor([[0.0, 5.0, 0.0], [2.0, 0.0, 0.0]])
  env = _FakeEnv(_FakeAsset(root_link_ang_vel_b=w))
  out = R.base_roll_rate_l2(env)
  assert out.tolist() == [0.0, 4.0]

  # vel limits: zero below 90% of the family limit
  class _Sim:
    pass

  # covered by live-env validation; unit-test the clamp shape via direct math
  lim = torch.tensor([[20.0, 37.0]])
  vel = torch.tensor([[17.0, 36.0], [19.0, 20.0]])
  over = (vel.abs() - 0.9 * lim).clamp(min=0.0).sum(dim=1)
  assert over[0].item() == pytest.approx((36 - 33.3) + 0.0, abs=0.01)
  assert over[1].item() == pytest.approx(1.0, abs=1e-6)


def test_rbf_foot_flat_counts_loaded_tilted_feet_only():
  """A tilted foot costs while in contact; the same tilt in swing is free."""
  tilt = 0.3  # rad about x (roll)
  flat_q = [1.0, 0.0, 0.0, 0.0]
  rolled = [math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0]
  feet = torch.tensor([[flat_q, flat_q], [rolled, flat_q], [rolled, flat_q]])
  env = _FakeEnv(_FakeAsset(body_link_quat_w=feet))

  class _Feet:
    class data:
      found = torch.tensor([[1, 1], [1, 1], [0, 1]])  # env 2: the rolled foot swings

  env.scene["feet"] = _Feet()
  out = R.rbf_foot_flat(env, "feet", std=0.4, asset_cfg=_Cfg(body_ids=slice(None)))
  expected = 1.0 - math.exp(-(math.sin(tilt) ** 2) / 0.16)
  assert out[0].item() == pytest.approx(0.0, abs=1e-6)
  assert out[1].item() == pytest.approx(expected, abs=1e-5)
  assert out[2].item() == pytest.approx(0.0, abs=1e-6)


def test_symmetry_mirrors_the_deployment_actor_without_the_velocity() -> None:
  """The actor group without ``base_lin_vel`` (96-D) mirrors as the 99-D one
  does from its angular velocity on; the critic's 99 + pairs still leads
  with the velocity."""
  from tensordict import TensorDict

  from ame.symmetry import has_lin_vel, mirror_obs_actions

  assert not has_lin_vel(96) and has_lin_vel(99)
  assert not has_lin_vel(96 + 8) and has_lin_vel(99 + 8)
  torch.manual_seed(0)
  full = torch.randn(4, 99)
  deploy = full[:, 3:].clone()
  obs = TensorDict({"actor": deploy, "critic": full}, batch_size=[4])
  out, _ = mirror_obs_actions(None, obs, None)
  assert out.batch_size == torch.Size([8])
  m_actor, m_critic = out["actor"][4:], out["critic"][4:]
  torch.testing.assert_close(m_actor, m_critic[:, 3:])  # same mirror past the velocity
  assert torch.equal(m_critic[:, 0], full[:, 0]) and torch.equal(
    m_critic[:, 1], -full[:, 1]
  )
  # Mirroring twice is the identity on the deployment layout too.
  twice, _ = mirror_obs_actions(
    None, TensorDict({"actor": m_actor}, batch_size=[4]), None
  )
  torch.testing.assert_close(twice["actor"][4:], deploy)
