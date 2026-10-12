"""Left-right symmetry augmentation for the G1 velocity task.

The AME tuning guide (Chong Zhang, 2026-03-28) recommends this as one of the
cheapest wins: *"This can be verified first with MLP before using AME... It
helps improve the motion style a lot."*  For the terrain map it also gives the
exact recipe: *"imagine the robot is walking on a mirrored terrain, so only
mirror the z values and keep old x & y values."*

The mirror is the sagittal plane (y -> -y).  Under it:

* left and right joints swap;
* roll and yaw joints flip sign (they rotate about the mirrored x/z axes),
  pitch joints do not;
* the base's y-velocity, roll rate, yaw rate, and gravity-y flip sign;
* the commanded lateral velocity and yaw rate flip sign;
* the elevation map's *z* values are mirrored across the map's y axis while
  the per-cell x/y coordinate channels are left alone (guide's recipe: the
  coordinates describe the same static grid, only the terrain moves).

``mirror_obs_actions`` is the ``data_augmentation_func`` rsl-rl expects; it
returns the original batch concatenated with its mirror, doubling the
mini-batch.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from tensordict import TensorDict

from ame.scan import MAP_SHAPE

# Joint order of the mjlab G1 (Entity.joint_names).
JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Joints whose sign flips under a sagittal mirror: rotations about the x
# (roll) and z (yaw) axes.  Pitch joints (about y) keep their sign.
_FLIP_TOKENS = ("_roll_", "_yaw_")


def _joint_maps() -> tuple[list[int], list[float]]:
  """Permutation and sign vector taking a joint vector to its mirror."""
  index = {n: i for i, n in enumerate(JOINT_NAMES)}
  perm: list[int] = []
  sign: list[float] = []
  for name in JOINT_NAMES:
    if name.startswith("left_"):
      partner = "right_" + name[len("left_") :]
    elif name.startswith("right_"):
      partner = "left_" + name[len("right_") :]
    else:
      partner = name  # waist joints mirror onto themselves
    perm.append(index[partner])
    sign.append(-1.0 if any(t in name for t in _FLIP_TOKENS) else 1.0)
  return perm, sign


JOINT_PERM, JOINT_SIGN = _joint_maps()

_CACHE: dict[tuple[int, torch.device], dict[str, torch.Tensor]] = {}


def _tensors(device: torch.device) -> dict[str, torch.Tensor]:
  key = (len(JOINT_NAMES), device)
  cached = _CACHE.get(key)
  if cached is None:
    cached = {
      "perm": torch.tensor(JOINT_PERM, dtype=torch.long, device=device),
      "sign": torch.tensor(JOINT_SIGN, dtype=torch.float32, device=device),
    }
    _CACHE[key] = cached
  return cached


def mirror_joints(x: torch.Tensor, device_cache: dict) -> torch.Tensor:
  """Swap left/right and flip roll/yaw signs. ``x``: ``[..., 29]``."""
  return x[..., device_cache["perm"]] * device_cache["sign"]


def mirror_map(map_flat: torch.Tensor) -> torch.Tensor:
  """Mirror the elevation scan across its y axis, z channel only.

  The scan is flattened ``(W, L, 3)`` row-major (rows = y, cols = x), so the
  mirror is a reversal along the W axis of the z channel; the x/y coordinate
  channels describe the same static grid and are left untouched.
  """
  b = map_flat.shape[0]
  lgt, wid, ch = MAP_SHAPE
  grid = map_flat.reshape(b, wid, lgt, ch).clone()
  grid[..., 2] = torch.flip(grid[..., 2], dims=(1,))
  return grid.reshape(b, -1)


PROPRIO_CORE = 3 + 3 + 29 + 29 + 29 + 3
"""ang_vel, gravity, joint_pos, joint_vel, actions, command: the layout with
the privileged linear velocity dropped from the group (the deployment set)."""


def has_lin_vel(width: int) -> bool:
  """Whether a proprio group leads with the 3-D base linear velocity.

  The critic's extras beyond the core come in per-foot pairs, so a group is
  ``96 + 2k`` wide without the velocity and ``99 + 2k`` with it: the parity
  of the width decides.  The actor drops the term for deployment
  (``ame.env_cfg.apply_deploy_observations``); the critic keeps it.
  """
  if width < PROPRIO_CORE:
    raise ValueError(f"proprio group of width {width} is narrower than the core")
  return (width - PROPRIO_CORE) % 2 == 1


def _mirror_proprio(
  group: torch.Tensor, has_lin_vel: bool, cache: dict
) -> torch.Tensor:
  """Mirror one proprio observation group in the task's term order.

  Layout: [base_lin_vel(3)] base_ang_vel(3) projected_gravity(3)
  joint_pos(29) joint_vel(29) actions(29) command(3) [critic extras...].
  """
  out = group.clone()
  i = 0
  if has_lin_vel:  # linear velocity: y flips
    out[:, i + 1] = -out[:, i + 1]
    i += 3
  # angular velocity: x (roll) and z (yaw) flip, y (pitch) does not
  out[:, i + 0] = -out[:, i + 0]
  out[:, i + 2] = -out[:, i + 2]
  i += 3
  out[:, i + 1] = -out[:, i + 1]  # projected gravity: y flips
  i += 3
  for _ in range(3):  # joint_pos, joint_vel, actions
    out[:, i : i + 29] = mirror_joints(group[:, i : i + 29], cache)
    i += 29
  # command (vx, vy, wz): lateral and yaw flip
  out[:, i + 1] = -out[:, i + 1]
  out[:, i + 2] = -out[:, i + 2]
  i += 3
  # Critic extras are all per-foot pairs (height, air time, contact, forces);
  # mirroring swaps the two feet.  Any remainder is a multiple of 2.
  rest = group.shape[1] - i
  if rest > 0:
    tail = group[:, i:].reshape(group.shape[0], -1, 2)
    out[:, i:] = torch.flip(tail, dims=(2,)).reshape(group.shape[0], -1)
  return out


def mirror_obs_actions(
  env: Any, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """rsl-rl ``data_augmentation_func``: append the mirrored half of a batch."""
  del env
  out_obs = None
  if obs is not None:
    mirrored = {}
    for key, value in obs.items():
      if key == "height_map":
        mirrored[key] = mirror_map(value)
      else:
        mirrored[key] = _mirror_proprio(
          value, has_lin_vel=has_lin_vel(value.shape[1]), cache=_tensors(value.device)
        )
    out_obs = cast(
      TensorDict,
      torch.cat([obs, TensorDict(mirrored, batch_size=obs.batch_size)], dim=0),
    )
  out_actions = None
  if actions is not None:
    cache = _tensors(actions.device)
    out_actions = torch.cat([actions, mirror_joints(actions, cache)], dim=0)
  return out_obs, out_actions


__all__ = [
  "JOINT_NAMES",
  "JOINT_PERM",
  "JOINT_SIGN",
  "PROPRIO_CORE",
  "has_lin_vel",
  "mirror_joints",
  "mirror_map",
  "mirror_obs_actions",
]
