"""AME-2's guard terminations: illegal contact, a flipped base, a solver blowup.

The paper filters "irrecoverable failure cases"; without the last two a warp
contact-solver explosion can feed poisoned rewards for whole episodes and
destroy the value function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0
) -> torch.Tensor:
  """Any non-foot link taking contact force above ``threshold``."""
  force = env.scene[sensor_name].data.force
  assert force is not None
  return torch.any(torch.norm(force, dim=-1) > threshold, dim=1)


def ame2_torso_collision(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 400.0,
  start_time: float = 1.0,
) -> torch.Tensor:
  """AME-2's ``early_illegal_contact`` (ame2_minimal ``mdp/terminations.py``
  ``ame2_illegal_contact``, cfg ``ame2_env_cfg.py`` ``early_illegal_contact``):
  per primary, the contact-force norm averaged over the sensor's substep history
  (one policy step) above ``threshold`` (400 N, AME-2's robot weight), counted
  only once the episode is ``start_time`` s old."""
  hist = env.scene[sensor_name].data.force_history  # [B, N, H, 3]
  assert hist is not None, f"sensor {sensor_name!r} needs history_length > 0"
  avg = torch.norm(hist, dim=-1).mean(dim=-1)  # [B, N]
  active = env.episode_length_buf * env.step_dt >= start_time
  return torch.any(avg > threshold, dim=1) & active


def bad_orientation(
  env: ManagerBasedRlEnv, x_max: float = 0.985, y_max: float = 0.9, z_min: float = 0.3
) -> torch.Tensor:
  """Effectively flipped: gravity-z above ``-z_min`` is a tilt past ~72°.  The
  paper's bound is ``g_z > 0``; a velocity task never needs to climb prone."""
  g = env.scene["robot"].data.projected_gravity_b
  return (g[:, 0].abs() > x_max) | (g[:, 1].abs() > y_max) | (g[:, 2] > -z_min)


def physics_blowup(env: ManagerBasedRlEnv, threshold: float = 2.0e4) -> torch.Tensor:
  """Joint acceleration far beyond gait (~2e3 rad/s²): a contact-solver spike."""
  return torch.any(env.scene["robot"].data.joint_acc.abs() > threshold, dim=1)


__all__ = ["ame2_torso_collision", "bad_orientation", "illegal_contact", "physics_blowup"]
