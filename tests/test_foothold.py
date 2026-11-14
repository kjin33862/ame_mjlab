"""Foothold reward, double critic, one-sided clearance, swing diagnostics (CPU)."""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from tensordict import TensorDict

FOOTHOLD_TASK = "Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Foothold-Velocity-Rough-G1"
CONTACT_TASK = "Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Velocity-Rough-G1"
norm = lambda x: re.sub(r" at 0x[0-9a-f]+", "", repr(x))  # noqa: E731


class _Scene(dict):
  pass


# --------------------------------------------------------------------- sole geometry


@pytest.fixture(scope="module")
def g1_model():
  import mjlab

  path = os.path.join(
    os.path.dirname(mjlab.__file__), "asset_zoo/robots/unitree_g1/xmls/g1.xml"
  )
  return mujoco.MjModel.from_xml_path(path)


def _capsules(model, side: str):
  """(a, b, radius) of the foot capsules' axes in the ankle_roll_link frame."""
  out = []
  for i in range(1, 8):
    g = model.geom(f"{side}_foot{i}_collision")
    assert g.type[0] == mujoco.mjtGeom.mjGEOM_CAPSULE
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, g.quat)
    axis = rot.reshape(3, 3)[:, 2] * g.size[1]
    out.append((g.pos - axis, g.pos + axis, g.size[0]))
  return out


@pytest.mark.parametrize("side", ["left", "right"])
def test_sole_points_lie_on_the_sole(g1_model, side) -> None:
  from ame.mdp.foothold import SITE_POS_LINK, SOLE_BOX_LINK, sole_points_link

  caps = _capsules(g1_model, side)
  # the documented bounding box and sole plane are the capsules'
  lo = np.min([np.minimum(a, b) - r for a, b, r in caps], axis=0)
  hi = np.max([np.maximum(a, b) + r for a, b, r in caps], axis=0)
  assert lo[0] == pytest.approx(SOLE_BOX_LINK[0][0], abs=1e-6)
  assert hi[0] == pytest.approx(SOLE_BOX_LINK[0][1], abs=1e-6)
  assert lo[2] == pytest.approx(-0.035, abs=1e-6)
  assert SOLE_BOX_LINK[1][0] <= lo[1] + 1e-6 and hi[1] <= SOLE_BOX_LINK[1][1] + 1e-6
  site = g1_model.site(f"{side}_foot")
  assert np.allclose(site.pos, SITE_POS_LINK)
  pts = sole_points_link().numpy()
  assert pts.shape == (15, 3)
  for p in pts:
    # on the bottom of some capsule: within its radius of the axis, at its bottom
    on = False
    for a, b, r in caps:
      t = np.clip(
        np.dot(p[:2] - a[:2], b[:2] - a[:2]) / np.dot(b[:2] - a[:2], b[:2] - a[:2]),
        0,
        1,
      )
      q = a + t * (b - a)
      on |= bool(np.linalg.norm(p[:2] - q[:2]) < 1e-3 and abs(p[2] - (q[2] - r)) < 1e-6)
    assert on, p
    assert lo[0] < p[0] < hi[0] and lo[1] < p[1] < hi[1]


def test_sole_pattern_rays_start_inside_the_foot() -> None:
  from ame.mdp.foothold import RAY_LIFT, SolePatternCfg, sole_points_site

  off, dirs = SolePatternCfg().generate_rays(None, "cpu")
  assert off.shape == dirs.shape == (15, 3)
  assert torch.allclose(off[:, :2], sole_points_site()[:, :2])
  assert torch.allclose(off[:, 2], torch.full((15,), 0.002 + RAY_LIFT))
  assert torch.all(dirs == torch.tensor([0.0, 0.0, -1.0]))


# --------------------------------------------------------------------- the reward on real rays


@pytest.fixture(scope="module")
def stone():
  """A stone with its top at z = 0 for x < 0, a pit (floor at -2 m) for x > 0."""
  spec = mujoco.MjSpec()
  wb = spec.worldbody
  wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(1, 1, 1), pos=(-1, 0, -1))
  wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(3, 3, 0.05), pos=(0, 0, -2.05))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  return model, data


def _yaw_quat(yaw: float, pitch: float = 0.0) -> np.ndarray:
  q, qy, qp = np.zeros(4), np.zeros(4), np.zeros(4)
  mujoco.mju_axisAngle2Quat(qy, np.array([0, 0, 1.0]), yaw)
  mujoco.mju_axisAngle2Quat(qp, np.array([0, 1.0, 0]), pitch)
  mujoco.mju_mulQuat(q, qy, qp)
  return q


def _sensor(
  stone, feet: list[tuple[tuple[float, float, float], np.ndarray]], max_distance=1.0
):
  """A ``foot_sole_scan`` stand-in: the sensor's ray layout (yaw-aligned pattern
  offsets from the site) cast with ``mj_ray`` on the stone scene, one env."""
  from ame.mdp.foothold import SolePatternCfg, foot_sole_scan_cfg

  model, data = stone
  off, dirs = SolePatternCfg().generate_rays(None, "cpu")
  hits, dists, normals = [], [], []
  for pos, quat in feet:
    yaw = math.atan2(
      2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
    )
    rot = np.array(
      [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )
    for o, d in zip(off.numpy(), dirs.numpy(), strict=True):
      origin = np.asarray(pos) + rot @ o
      gid = np.zeros(1, dtype=np.int32)
      dist = mujoco.mj_ray(model, data, origin, rot @ d, None, 1, -1, gid)
      dist = dist if 0 <= dist <= max_distance else -1.0
      hits.append(origin + rot @ d * max(dist, 0.0))
      dists.append(dist)
      normals.append([0.0, 0.0, 1.0 if dist >= 0 else 0.0])
  data_ = SimpleNamespace(
    frame_pos_w=torch.tensor([[p for p, _ in feet]], dtype=torch.float32),
    frame_quat_w=torch.tensor(np.array([[q for _, q in feet]]), dtype=torch.float32),
    hit_pos_w=torch.tensor(np.array([hits]), dtype=torch.float32),
    distances=torch.tensor([dists], dtype=torch.float32),
    normals_w=torch.tensor([normals], dtype=torch.float32),
  )
  return SimpleNamespace(data=data_, cfg=foot_sole_scan_cfg())


def _reward(sensor, contact: list[bool], delta: float = 0.05) -> float:
  from mjlab.managers.reward_manager import RewardTermCfg

  from ame.mdp.foothold import foothold_unsupported

  contact_sensor = SimpleNamespace(
    data=SimpleNamespace(found=torch.tensor([[int(c) for c in contact]])),
    primary_names=["robot/left_ankle_roll_link", "robot/right_ankle_roll_link"],
  )
  env = SimpleNamespace(
    scene=_Scene(foot_sole_scan=sensor, feet_ground_contact=contact_sensor)
  )
  cfg = RewardTermCfg(func=foothold_unsupported, weight=-1.0, params={"delta": delta})
  term = foothold_unsupported(cfg, env)
  return float(term(env, delta=delta)[0])


SITE_BELOW_SOLE = 0.002
FLAT = -SITE_BELOW_SOLE  # site z of a foot standing flat on the stone top (z = 0)


def _fractions(sensor, delta):
  from ame.mdp.foothold import sensor_sole_drop, unsupported_fraction

  return unsupported_fraction(sensor_sole_drop(sensor), delta)[0].tolist()


@pytest.mark.parametrize("delta", [0.02, 0.05, 0.10])
def test_flat_foot_is_fully_supported(stone, delta) -> None:
  feet = [((-0.5, 0.1, FLAT), _yaw_quat(0.3)), ((-0.5, -0.1, FLAT), _yaw_quat(-2.0))]
  s = _sensor(stone, feet)
  assert _fractions(s, delta) == [0.0, 0.0]
  assert _reward(s, [True, True], delta) == 0.0


def test_foot_half_over_a_gap_is_about_half(stone) -> None:
  # foot along +x, its centre column 1 mm either side of the stone edge (x = 0):
  # 2 or 3 of the 5 columns over the pit -> 0.4 / 0.6, 0.5 on average
  from ame.mdp.foothold import SITE_POS_LINK, sole_points_link

  centre_col = float(sole_points_link()[7, 0]) - SITE_POS_LINK[0]
  res = []
  for shift in (-0.001, 0.001):
    feet = [((shift - centre_col, 0.0, FLAT), _yaw_quat(0.0))] * 2
    s = _sensor(stone, feet)
    for delta in (0.02, 0.05, 0.10):
      fr = _fractions(s, delta)
      assert fr[0] == pytest.approx(0.4 if shift < 0 else 0.6)
    res.append(_reward(s, [True, False]))
  assert res == pytest.approx([0.4, 0.6])
  assert np.mean(res) == pytest.approx(0.5)


def test_foot_in_the_air_scores_zero(stone) -> None:
  feet = [((-0.5, 0.0, 0.20), _yaw_quat(0.0)), ((0.5, 0.0, 0.20), _yaw_quat(0.0))]
  s = _sensor(stone, feet)
  assert _fractions(s, 0.05) == [1.0, 1.0]  # far above the ground / over the pit
  assert _reward(s, [False, False]) == 0.0


def test_misses_are_unsupported_and_backfaces_supported() -> None:
  from ame.mdp.foothold import sole_drop

  B, F, N = 1, 1, 15
  pos = torch.zeros(B, F, 3)
  quat = torch.tensor([[[1.0, 0, 0, 0]]])
  hit = torch.zeros(B, F * N, 3)
  dist = torch.full((B, F * N), 0.05)
  normal = torch.zeros(B, F * N, 3)
  normal[..., 2] = 1.0
  dist[0, 0] = -1.0  # miss
  normal[0, 1, 2] = -1.0  # back face: ray started inside the terrain
  d = sole_drop(pos, quat, hit, dist, normal)[0, 0]
  assert d[0] == float("inf") and d[1] == float("-inf")


def test_pitched_foot_lifts_heel_points(stone) -> None:
  # pitched 10 deg toe-down about the site on the stone: the heel column
  # (8.4 cm behind the site) rises ~1.5 cm, under the 2 cm threshold
  feet = [((-0.5, 0.0, FLAT), _yaw_quat(0.0, math.radians(10)))] * 2
  s = _sensor(stone, feet)
  assert _fractions(s, 0.02) == [0.0, 0.0]
  assert _fractions(s, 0.01)[0] == pytest.approx(3 / 15)


def test_foot_order_mismatch_is_refused() -> None:
  from ame.mdp.foothold import check_foot_order, foot_sole_scan_cfg

  contact = SimpleNamespace(
    primary_names=["robot/right_ankle_roll_link", "robot/left_ankle_roll_link"]
  )
  with pytest.raises(ValueError):
    check_foot_order(contact, foot_sole_scan_cfg())


# --------------------------------------------------------------------- one-sided clearance


def test_one_sided_costs_only_penalise_low_feet() -> None:
  from ame.mdp.swing import clearance_cost, swing_height_cost

  h = torch.tensor([[0.05, 0.10, 0.30]])
  v = torch.ones_like(h)
  assert clearance_cost(h, v, 0.1, False).tolist()[0] == pytest.approx([0.05, 0.0, 0.2])
  assert clearance_cost(h, v, 0.1, True).tolist()[0] == pytest.approx([0.05, 0.0, 0.0])
  assert swing_height_cost(h, 0.1, False).tolist()[0] == pytest.approx([0.25, 0.0, 4.0])
  assert swing_height_cost(h, 0.1, True).tolist()[0] == pytest.approx([0.25, 0.0, 0.0])


def test_one_sided_flag_swaps_exactly_two_functions() -> None:
  from mjlab.tasks.velocity.mdp.rewards import feet_clearance, feet_swing_height

  from ame.env_cfg import ame_gap_env_cfg
  from ame.mdp.swing import feet_clearance_one_sided, feet_swing_height_one_sided

  base, one = ame_gap_env_cfg(), ame_gap_env_cfg(one_sided_clearance=True)
  assert base.rewards["foot_clearance"].func is feet_clearance
  assert base.rewards["foot_swing_height"].func is feet_swing_height
  assert one.rewards["foot_clearance"].func is feet_clearance_one_sided
  assert one.rewards["foot_swing_height"].func is feet_swing_height_one_sided
  one.rewards["foot_clearance"].func = feet_clearance
  one.rewards["foot_swing_height"].func = feet_swing_height
  assert norm(base) == norm(one)


def test_flags_default_off() -> None:
  from ame.env_cfg import ame_gap_env_cfg

  off = dict(foothold=False, one_sided_clearance=False, swing_diagnostics=False)
  assert norm(ame_gap_env_cfg()) == norm(ame_gap_env_cfg(**off))


# --------------------------------------------------------------------- swing diagnostics


def test_swing_tracker_splits_gap_and_no_gap_swings() -> None:
  from ame.mdp.swing import SwingGapTracker

  tr = SwingGapTracker(1, 2, "cpu")
  T, F = True, False
  one = torch.ones(1)

  def step(contact, first, over, h=0.05, c2=1.0, c1=0.5):
    return tr.update(
      contact=torch.tensor([contact]),
      first_contact=torch.tensor([first]),
      clear2=torch.full((1, 2), c2),
      clear1=torch.full((1, 2), c1),
      heights=torch.full((1, 2), h),
      over_gap=torch.tensor([over]),
      active=one,
    )

  # spawn drop: both feet land without a prior contact -> not reported
  assert step([F, F], [F, F], [F, F])["crossed"].numel() == 0
  assert step([T, T], [T, T], [F, F])["crossed"].numel() == 0
  # left swings over a gap for two steps, right swings on solid ground
  step([F, F], [F, F], [T, F])
  step([F, F], [F, F], [F, F], h=0.2)
  out = step([T, T], [T, T], [F, F])
  assert out["crossed"].tolist() == [True, False]
  assert out["clear2"].tolist() == [2.0, 2.0] and out["clear1"].tolist() == [1.0, 1.0]
  assert out["height2"].tolist() == pytest.approx([1.0, 1.0])  # peak 0.2 vs 0.1
  assert out["height1"].tolist() == [0.0, 0.0]
  log: dict = {}
  from ame.mdp.swing import log_swings

  log_swings(log, out)
  assert log["Swing/gap_share"].tolist() == [1.0, 0.0]
  assert log["Swing/height_gap"].tolist() == pytest.approx([1.0])
  assert log["Swing/clearance_1s_nogap"].tolist() == [1.0]


# --------------------------------------------------------------------- double critic


def _models(output_dims=(1, 2), seed=0):
  from rsl_rl.models import MLPModel

  torch.manual_seed(seed)
  obs = TensorDict({"p": torch.randn(6, 5)}, batch_size=[6])
  groups = {"actor": ["p"], "critic": ["p"]}
  dist = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
  actor = MLPModel(obs, groups, "actor", 3, hidden_dims=(8,), distribution_cfg=dist)
  critics = [MLPModel(obs, groups, "critic", d, hidden_dims=(8,)) for d in output_dims]
  return obs, actor, critics


def _fill(storage, T, N, rewards, values, dones):
  storage.rewards.copy_(rewards)
  storage.values.copy_(values)
  storage.dones.copy_(dones)
  storage.step = T


def test_double_critic_reduces_to_single_critic_when_group_two_is_silent() -> None:
  from rsl_rl.algorithms import PPO
  from rsl_rl.storage import RolloutStorage

  from ame.double_critic import DoubleCriticPPO, MultiRewardRolloutStorage

  T, N = 7, 6
  obs, actor, (single, double) = _models()
  # head 1 of the double critic = the single critic, head 2 outputs 0
  sd = single.state_dict()
  dd = double.state_dict()
  for k in dd:
    if dd[k].shape == sd[k].shape:
      dd[k] = sd[k].clone()
  last_w = [k for k in dd if k.endswith("weight")][-1]
  last_b = [k for k in dd if k.endswith("bias")][-1]
  dd[last_w] = torch.cat([sd[last_w], torch.zeros_like(sd[last_w])])
  dd[last_b] = torch.cat([sd[last_b], torch.zeros_like(sd[last_b])])
  double.load_state_dict(dd)
  out = double(obs)
  assert torch.allclose(out[:, 0:1], single(obs)) and torch.all(out[:, 1] == 0)

  torch.manual_seed(1)
  r = torch.randn(T, N, 1)
  v = torch.randn(T, N, 1)
  dones = (torch.rand(T, N, 1) < 0.15).to(torch.uint8)
  ref_storage = RolloutStorage("rl", N, T, obs, [3])
  _fill(ref_storage, T, N, r, v, dones)
  ref = PPO(actor, single, ref_storage)
  ref.compute_returns(obs)

  for w1 in (1.0, 2.0):
    st = MultiRewardRolloutStorage("rl", N, T, obs, [3], num_groups=2)
    _fill(
      st,
      T,
      N,
      torch.cat([r, torch.zeros_like(r)], -1),
      torch.cat([v, torch.zeros_like(v)], -1),
      dones,
    )
    alg = DoubleCriticPPO(actor, double, st, advantage_weights=(w1, 0.25))
    alg.compute_returns(obs)
    # group 2 contributes exactly nothing; group 1 is the single critic's
    # normalised advantage, scaled by w1
    torch.testing.assert_close(st.advantages, w1 * ref_storage.advantages)
    torch.testing.assert_close(st.returns[..., :1], ref_storage.returns)
    assert torch.all(st.returns[..., 1] == 0)


def test_each_group_is_normalised_separately() -> None:
  from ame.double_critic import combine_advantages

  torch.manual_seed(0)
  a = torch.randn(5, 4, 2) * torch.tensor([10.0, 0.01]) + torch.tensor([3.0, -1.0])
  c = combine_advantages(a, (1.0, 0.25))
  n = lambda x: (x - x.mean()) / (x.std() + 1e-8)  # noqa: E731
  torch.testing.assert_close(c[..., 0], n(a[..., 0]) + 0.25 * n(a[..., 1]))


def test_reward_groups_split_by_term() -> None:
  from ame.double_critic import RewardGroups

  step = torch.tensor([[1.0, -2.0, -0.5], [0.5, 0.0, -1.5]])  # weight * value
  dt = 0.02
  mgr = SimpleNamespace(
    active_terms=["track", "foothold", "slip"], _step_reward=step, _scale_by_dt=True
  )
  total = step.sum(1) * dt
  g = RewardGroups(mgr, ("foothold",), dt)(total)
  torch.testing.assert_close(g[:, 1], step[:, 1] * dt)
  torch.testing.assert_close(g.sum(1), total)
  with pytest.raises(ValueError):
    RewardGroups(mgr, ("nope",), dt)


def test_double_critic_update_trains_both_heads() -> None:
  from rsl_rl.storage import RolloutStorage

  from ame.double_critic import DoubleCriticPPO, MultiRewardRolloutStorage

  T, N = 4, 6
  obs, actor, (critic,) = _models(output_dims=(2,))
  st = MultiRewardRolloutStorage("rl", N, T, obs, [3], num_groups=2)
  alg = DoubleCriticPPO(actor, critic, st, num_mini_batches=2, num_learning_epochs=1)
  assert alg.value_loss_coef == 2.0
  rm = SimpleNamespace(
    active_terms=["a", "foothold"], _step_reward=torch.zeros(N, 2), _scale_by_dt=True
  )
  alg.bind_reward_groups(SimpleNamespace(reward_manager=rm, step_dt=0.02))
  before = [p.detach().clone() for p in critic.parameters()]
  with torch.inference_mode():
    for _ in range(T):
      alg.act(obs)
      rm._step_reward = torch.randn(N, 2)
      rew = rm._step_reward.sum(1) * 0.02
      alg.process_env_step(
        obs, rew, torch.zeros(N, dtype=torch.long), {"time_outs": torch.zeros(N)}
      )
    assert torch.allclose(st.rewards.sum(-1), st.rewards[..., 0] + st.rewards[..., 1])
    torch.testing.assert_close(st.rewards[-1, :, 1], rm._step_reward[:, 1] * 0.02)
    alg.compute_returns(obs)
  losses = alg.update()
  assert {
    "value",
    "surrogate",
    "adv_std_g1",
    "adv_std_g2",
    "explained_var_g2",
  } <= losses.keys()
  last_w = list(critic.parameters())[-2]
  moved = (last_w.detach() - before[-2]).abs().sum(dim=1)
  assert torch.all(moved > 0), "both value heads get a gradient"
  assert isinstance(st, RolloutStorage)


# --------------------------------------------------------------------- the task


def test_foothold_task_differs_from_contact_only_in_the_intended_items() -> None:
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity.mdp.rewards import feet_clearance, feet_swing_height

  import ame  # noqa: F401
  from ame.double_critic import DoubleCriticPPO  # noqa: F401
  from ame.mdp.foothold import SOLE_SCAN_SENSOR, foothold_unsupported
  from ame.mdp.swing import (
    feet_clearance_one_sided,
    feet_swing_height_one_sided,
    swing_cost_by_gap,
  )

  assert FOOTHOLD_TASK in list_tasks()
  for play in (False, True):
    base, new = (
      load_env_cfg(CONTACT_TASK, play=play),
      load_env_cfg(FOOTHOLD_TASK, play=play),
    )
    added = [s for s in new.scene.sensors if s.name == SOLE_SCAN_SENSOR]
    assert len(added) == 1 and not any(
      s.name == SOLE_SCAN_SENSOR for s in base.scene.sensors
    )
    term = new.rewards.pop("foothold")
    assert term.func is foothold_unsupported and term.weight == -1.0
    assert term.params["delta"] == 0.10
    assert new.metrics.pop("swing_cost_by_gap").func is swing_cost_by_gap
    # one_sided_clearance on: exactly the two functions are swapped back here.
    for name, one, two in (
      ("foot_clearance", feet_clearance_one_sided, feet_clearance),
      ("foot_swing_height", feet_swing_height_one_sided, feet_swing_height),
    ):
      assert base.rewards[name].func is two and new.rewards[name].func is one
      new.rewards[name].func = two
    new.scene.sensors = tuple(
      s for s in new.scene.sensors if s.name != SOLE_SCAN_SENSOR
    )
    assert norm(base) == norm(new)  # incl. illegal_contact: self-contact still counts
  rl_base, rl_new = (
    asdict(load_rl_cfg(CONTACT_TASK)),
    asdict(load_rl_cfg(FOOTHOLD_TASK)),
  )
  alg = rl_new.pop("algorithm")
  assert alg.pop("class_name") == "ame.double_critic:DoubleCriticPPO"
  assert alg.pop("reward_group_terms") == ("foothold",)
  assert alg.pop("advantage_weights") == (1.0, 0.25)
  base_alg = rl_base.pop("algorithm")
  assert base_alg.pop("class_name") == "PPO"
  assert alg == base_alg and rl_base == rl_new
