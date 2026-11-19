"""StrideFree switches: EMA linear velocity tracking, wider walking posture std (CPU)."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pytest
import torch

D30_TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1"
STRIDE_FREE_TASK = (
  "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-StrideFree-Velocity-Rough-G1"
)
norm = lambda x: re.sub(r" at 0x[0-9a-f]+", "", repr(x))  # noqa: E731
DT = 0.02


def _quat_yaw_pitch(yaw: float, pitch: float) -> list[float]:
  """(w, x, y, z) of R = Rz(yaw) Ry(pitch)."""
  cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
  cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
  return [cy * cp, -sy * sp, cy * sp, sy * cp]


class _Data:
  """The two root fields the terms read, plus mjlab's derived body-frame velocity."""

  def __init__(self, quat, vel):
    self.root_link_quat_w = quat
    self.root_link_lin_vel_w = vel

  @property
  def root_link_lin_vel_b(self):
    from mjlab.utils.lab_api.math import quat_apply_inverse

    return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)


class _Env:
  def __init__(self, cmd, quat, vel):
    self.num_envs = len(cmd)
    self.device = "cpu"
    self.step_dt = DT
    self.cmd = torch.tensor(cmd, dtype=torch.float32)
    self.data = _Data(
      torch.tensor(quat, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)
    )
    self.scene = {"robot": SimpleNamespace(data=self.data)}
    self.command_manager = SimpleNamespace(get_command=lambda name: self.cmd)

  def set_vel(self, vel):
    self.data.root_link_lin_vel_w = torch.tensor(vel, dtype=torch.float32)


def _term(env, tau=1.0):
  from mjlab.managers.reward_manager import RewardTermCfg

  from ame.mdp.rewards import track_lin_vel_xy_ema_exp

  params = {"command_name": "twist", "std": 0.5, "tau": tau}
  cfg = RewardTermCfg(func=track_lin_vel_xy_ema_exp, weight=2.0, params=params)
  term = track_lin_vel_xy_ema_exp(cfg, env)
  return term, lambda: term(env, **params)


def test_ema_alpha() -> None:
  from ame.mdp.rewards import ema_alpha

  assert ema_alpha(DT, 1.0) == pytest.approx(1 - math.exp(-0.02))
  for tau in (0.0, -1.0):
    with pytest.raises(ValueError):
      ema_alpha(DT, tau)


def test_first_step_matches_mjlab_on_an_upright_body() -> None:
  """Right after a reset the average is the current velocity, so on an upright
  body (yaw frame = body frame up to yaw) the reward is mjlab's."""
  from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity

  quat = [_quat_yaw_pitch(0.7, 0.0), _quat_yaw_pitch(-2.0, 0.0)]
  env = _Env(
    [[0.8, 0.1, 0.0], [0.3, -0.4, 0.5]], quat, [[0.2, 0.5, 0.1], [0.0, 0.3, -0.2]]
  )
  _, call = _term(env)
  ref = track_linear_velocity(env, std=0.5, command_name="twist")
  torch.testing.assert_close(call(), ref, atol=1e-6, rtol=0)


def test_ema_update_and_per_env_reset() -> None:
  env = _Env([[1.0, 0.0, 0.0]] * 2, [[1.0, 0, 0, 0]] * 2, [[0.0, 0.0, 0.0]] * 2)
  term, call = _term(env)
  a = 1 - math.exp(-DT / 1.0)
  call()
  torch.testing.assert_close(term.ema, torch.zeros(2, 2))
  env.set_vel([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
  r = call()
  torch.testing.assert_close(term.ema, torch.tensor([[a, 0.5 * a]] * 2))
  err = (1 - a) ** 2 + (0.5 * a) ** 2
  torch.testing.assert_close(r, torch.full((2,), math.exp(-err / 0.25)))
  # Reset env 1 only: it restarts at the current velocity, env 0 keeps averaging.
  term.reset(torch.tensor([1]))
  env.set_vel([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
  call()
  torch.testing.assert_close(
    term.ema[0], torch.tensor([a + a * (2.0 - a), 0.5 * a * (1 - a)])
  )
  torch.testing.assert_close(term.ema[1], torch.tensor([2.0, 0.0]))
  term.reset()  # all envs
  env.set_vel([[0.3, 0.2, 0.0], [0.4, 0.1, 0.0]])
  call()
  torch.testing.assert_close(term.ema, torch.tensor([[0.3, 0.2], [0.4, 0.1]]))


def test_ema_time_constant() -> None:
  """A step in velocity reaches 1 - 1/e of its size after tau seconds."""
  env = _Env([[0.0, 0.0, 0.0]], [[1.0, 0, 0, 0]], [[0.0, 0.0, 0.0]])
  term, call = _term(env, tau=1.0)
  call()
  env.set_vel([[1.0, 0.0, 0.0]])
  for _ in range(round(1.0 / DT)):
    call()
  assert term.ema[0, 0].item() == pytest.approx(1 - math.exp(-1), abs=1e-5)


def test_stride_oscillation_is_averaged_out() -> None:
  """An in-stride speed oscillation around the command costs mjlab's term but
  barely the EMA's once the average has settled."""
  from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity

  env = _Env([[0.6, 0.0, 0.0]], [[1.0, 0, 0, 0]], [[0.6, 0.0, 0.0]])
  _, call = _term(env)
  ema_r, inst_r = [], []
  for k in range(500):  # 10 s, 1 Hz stride, +-0.3 m/s forward and lateral sway
    ph = 2 * math.pi * k * DT
    env.set_vel([[0.6 + 0.3 * math.sin(ph), 0.3 * math.cos(ph), 0.0]])
    ema_r.append(call().item())
    inst_r.append(track_linear_velocity(env, std=0.5, command_name="twist").item())
  late = slice(250, None)
  assert min(ema_r[late]) > 0.97
  assert sum(inst_r[late]) / 250 < 0.75


def test_xy_is_in_the_yaw_frame_and_z_is_body_frame() -> None:
  """A pitched body walking level: xy is the yaw-frame velocity (no pitch
  leakage), z is mjlab's body-frame vertical velocity."""
  yaw, pitch = 0.4, 0.3
  vel_yaw = torch.tensor([0.5, 0.2, 0.0])
  c, s = math.cos(yaw), math.sin(yaw)
  vel_w = [c * 0.5 - s * 0.2, s * 0.5 + c * 0.2, 0.0]
  env = _Env([[0.5, 0.2, 0.0]], [_quat_yaw_pitch(yaw, pitch)], [vel_w])
  term, call = _term(env)
  r = call()
  torch.testing.assert_close(term.ema[0], vel_yaw[:2], atol=1e-6, rtol=0)
  vz_b = 0.5 * math.sin(pitch)  # world-level forward velocity seen from a pitched body
  torch.testing.assert_close(
    r, torch.tensor([math.exp(-(vz_b**2) / 0.25)]), atol=1e-5, rtol=0
  )


def test_switches_default_off() -> None:
  from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
  from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity

  from ame.env_cfg import ame_gap_env_cfg

  cfg = ame_gap_env_cfg()
  assert cfg.rewards["track_linear_velocity"].func is track_linear_velocity
  ref = unitree_g1_rough_env_cfg().rewards["pose"].params
  for k in ("std_standing", "std_walking", "std_running"):
    assert cfg.rewards["pose"].params[k] == ref[k]


def test_walking_std_scale() -> None:
  from ame.env_cfg import (
    STRIDE_FREE_WALKING_STD_SCALE,
    ame_gap_env_cfg,
    apply_walking_posture_std_scale,
  )

  base = ame_gap_env_cfg().rewards["pose"].params
  new = ame_gap_env_cfg(walking_posture_std_scale=STRIDE_FREE_WALKING_STD_SCALE)
  p = new.rewards["pose"].params
  assert p["std_walking"] == {
    **base["std_walking"],
    r".*hip_pitch.*": 0.6,
    r".*hip_roll.*": 0.3,
    r".*knee.*": 0.7,
  }
  assert p["std_standing"] == base["std_standing"]
  assert p["std_running"] == base["std_running"]
  with pytest.raises(ValueError):
    apply_walking_posture_std_scale(new, {r".*hip_pitc.*": 2.0})


def test_walking_std_scale_hits_only_the_six_leg_joints() -> None:
  """Resolved against the G1's joint names: hip pitch / roll and knee of both
  legs, nothing else (hip_yaw, ankles, waist, arms keep their std)."""
  from mjlab.asset_zoo.robots import get_g1_robot_cfg
  from mjlab.entity import Entity
  from mjlab.utils.lab_api.string import resolve_matching_names_values

  from ame.env_cfg import STRIDE_FREE_WALKING_STD_SCALE, ame_gap_env_cfg

  names = Entity(get_g1_robot_cfg()).joint_names
  std = lambda cfg: resolve_matching_names_values(  # noqa: E731
    cfg.rewards["pose"].params["std_walking"], names
  )[2]
  base = std(ame_gap_env_cfg())
  new = std(ame_gap_env_cfg(walking_posture_std_scale=STRIDE_FREE_WALKING_STD_SCALE))
  changed = {n for n, b, x in zip(names, base, new, strict=True) if b != x}
  assert changed == {
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "knee")
  }
  for n, b, x in zip(names, base, new, strict=True):
    if n in changed:
      assert x == pytest.approx(2 * b)


def test_stride_free_task_differs_from_d30_only_in_the_two_switches() -> None:
  from dataclasses import asdict

  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity

  import ame  # noqa: F401
  from ame.mdp.rewards import track_lin_vel_xy_ema_exp

  assert STRIDE_FREE_TASK in list_tasks()
  for play in (False, True):
    base, new = (
      load_env_cfg(D30_TASK, play=play),
      load_env_cfg(STRIDE_FREE_TASK, play=play),
    )
    lb, ln = base.rewards["track_linear_velocity"], new.rewards["track_linear_velocity"]
    assert lb.func is track_linear_velocity and ln.func is track_lin_vel_xy_ema_exp
    assert ln.weight == lb.weight == 2.0
    assert ln.params["std"] == lb.params["std"] == 0.5
    assert ln.params["tau"] == 1.0
    ln.func, ln.params = lb.func, dict(lb.params)
    w = new.rewards["pose"].params["std_walking"]
    for k in (r".*hip_pitch.*", r".*hip_roll.*", r".*knee.*"):
      w[k] = w[k] / 2.0
    assert norm(base) == norm(new)
  assert asdict(load_rl_cfg(D30_TASK)) == asdict(load_rl_cfg(STRIDE_FREE_TASK))
