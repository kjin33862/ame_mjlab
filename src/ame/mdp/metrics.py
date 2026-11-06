"""Per-step tracking metrics the logger averages honestly.

mjlab's ``Metrics/twist/error_vel_*`` accumulate the per-step error over the
episode and divide by the command's maximum duration, so a longer episode
logs a larger "error" at the same tracking quality.  The metrics
manager reports true per-step means; these terms put the pelvis tracking
error and speed there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def pelvis_progress(
  env: ManagerBasedRlEnv, command_name: str = "twist"
) -> torch.Tensor:
  """Pelvis velocity along the commanded direction, ``v_xy · cmd_xy / |cmd_xy|``,
  m/s — circling or walking backwards earns nothing or less.  Zero under a
  standing command, where ``command_speed`` is zero too."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = command[:, :2]
  norm = torch.norm(cmd, dim=1)
  along = (_pelvis_vel_yaw(env)[:, :2] * cmd).sum(dim=1) / norm.clamp(min=1e-6)
  return torch.where(norm > 1e-6, along, torch.zeros_like(along))


def command_speed(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
  """``|cmd_xy|`` per step; with ``pelvis_speed`` it gives an episode's
  progress ratio that holds under arcs, unlike the end-point displacement."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2], dim=1)


class edge_distance:
  """Farthest Chebyshev distance ``max(|dx|, |dy|)`` of the pelvis from its
  tile centre so far this episode, m.

  The tile centre is ``terrain_origins[level, type]``, whatever the spawn.
  Every step's value is the running maximum, so with ``reduce="last"`` the
  episode metric is the episode's farthest excursion.  :attr:`max_distance`
  is read by :class:`ame.mdp.terrain_levels_sgs_edge` at the episode end:
  the metrics manager computes before the curriculum and resets after it, so
  the buffer then holds the whole episode, terminal step included.
  """

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.max_distance = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    terrain: Any = env.scene.terrain
    centre = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    pos = env.scene["robot"].data.root_link_pos_w
    cheb = (pos[:, :2] - centre[:, :2]).abs().amax(dim=1)
    torch.maximum(self.max_distance, cheb, out=self.max_distance)
    return self.max_distance.clone()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.max_distance[slice(None) if env_ids is None else env_ids] = 0.0


__all__ = [
  "command_speed",
  "edge_distance",
  "pelvis_progress",
  "pelvis_speed",
  "pelvis_vel_error",
  "pelvis_yaw_error",
]
