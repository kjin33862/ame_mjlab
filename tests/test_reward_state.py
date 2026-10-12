"""Reward state ownership and native entity indexing contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from conftest import get_test_device
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ame.mdp import rewards


def test_robot_weight_cache_is_owned_by_environment() -> None:
  device = get_test_device()
  env = SimpleNamespace(
    device=device,
    sim=SimpleNamespace(mj_model=SimpleNamespace(body_mass=np.array([2.0]))),
  )
  rewards._robot_weight(env)
  cached = getattr(env, rewards._WEIGHT_KEY, torch.tensor(-1.0, device=device))
  torch.testing.assert_close(
    cached, torch.tensor(19.62, device=device), atol=1e-5, rtol=1e-6
  )


def test_joint_velocity_limits_follow_entity_joint_order() -> None:
  device = get_test_device()
  asset = SimpleNamespace(
    data=SimpleNamespace(joint_vel=torch.tensor([[40.0, 40.0]], device=device)),
    indexing=SimpleNamespace(joint_ids=torch.tensor([3, 1], device=device)),
  )
  model = SimpleNamespace(
    nu=2,
    actuator_forcerange=np.array([[-25.0, 25.0], [-139.0, 139.0]]),
    actuator_trnid=np.array([[3, -1], [1, -1]]),
  )
  env = SimpleNamespace(scene={"custom": asset}, sim=SimpleNamespace(mj_model=model))
  limits = rewards._joint_vel_limit(env, asset, SceneEntityCfg("custom").name)
  torch.testing.assert_close(
    limits, torch.tensor([[37.0, 20.0]], device=device), atol=1e-6, rtol=0
  )


def test_empty_contact_sensor_set_returns_zero_per_lane() -> None:
  device = get_test_device()
  env = SimpleNamespace(
    device=device,
    num_envs=4,
    sim=SimpleNamespace(mj_model=SimpleNamespace(body_mass=np.array([2.0]))),
  )
  torch.testing.assert_close(
    rewards.link_contact_forces(env, ()),
    torch.zeros(4, device=device),
    atol=1e-6,
    rtol=0,
  )
