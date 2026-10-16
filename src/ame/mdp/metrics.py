"""Per-step tracking metrics the logger averages honestly.

mjlab's ``Metrics/twist/error_vel_*`` accumulate the per-step error over the
episode and divide by the command's maximum duration, so a longer episode
logs a larger "error" at the same tracking quality.  The metrics
manager reports true per-step means; these terms put the pelvis tracking
error and speed there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _pelvis_vel_yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  data = env.scene["robot"].data
  return quat_apply_inverse(
    yaw_quat(data.root_link_quat_w), data.root_link_lin_vel_w[:, :3]
  )


def pelvis_vel_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``|cmd_xy − v_xy|`` of the pelvis in its yaw frame, m/s."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2] - _pelvis_vel_yaw(env)[:, :2], dim=1)


def pelvis_yaw_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``|cmd_wz − w_z|`` of the pelvis, rad/s."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (command[:, 2] - env.scene["robot"].data.root_link_ang_vel_b[:, 2]).abs()


def pelvis_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Planar pelvis speed, m/s — zero is the standing attractor."""
  return torch.norm(_pelvis_vel_yaw(env)[:, :2], dim=1)


def command_speed(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
  """``|cmd_xy|`` per step; with ``pelvis_speed`` it gives an episode's
  progress ratio that holds under arcs, unlike the end-point displacement."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2], dim=1)


__all__ = [
  "command_speed",
  "pelvis_speed",
  "pelvis_vel_error",
  "pelvis_yaw_error",
]
