"""The AME-2 reward stack, reimplemented on mjlab, plus two RBF posture terms.

mjlab's own G1 velocity rewards (still selectable as ``rewards="mjlab"``) do
not shape the policy toward AME's gait: no stumble or foot-proximity
penalties, no air-time variance, no cross-body arm/leg coordination, no
waist/arm deviation terms.  This module reproduces the reference stack
term-for-term
(``29dof/velocity_env_cfg_29dof.py::RewardsCfg`` + ``mdp/rewards.py``) against
mjlab data, and adds two posture terms:

* ``rbf_upper_body_pose`` — upper body (waist, shoulders, elbows, wrists)
  held at its default pose;
* ``rbf_foot_yaw`` — each foot's yaw aligned with the base yaw.

**Sign convention for the RBF terms.**  Both are written as *bounded
penalties*: ``1 - exp(-e^2 / sigma^2)``, which is 0 at perfect alignment and
saturates at 1 when far off, applied with weight ``-1``.  That is the reading
under which the minus sign is correct (a literal ``-1 * exp(...)``
would be most negative when the pose is perfect, i.e. it would train the
robot to twist).  The two forms differ only by an additive constant — their
gradients w.r.t. pose are identical — but the bounded-penalty form does not
also inject a survival bonus, so it leaves the termination trade-off alone.

Deliberate mjlab adaptations, all noted where they occur: contacts are read
from mjlab ``ContactSensor`` primaries rather than IsaacLab body-indexed
contact forces; feet are the ankle-roll links; torque limits come from the
model's ``actuator_forcerange``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
  """Yaw angle from a (w, x, y, z) quaternion, any leading dims."""
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


##
# Tracking (AME-2 uses the yaw frame and does not penalize vertical velocity,
# unlike mjlab's built-ins).
##


def track_lin_vel_xy_yaw_frame_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Tracking in the yaw frame of ``asset_cfg``'s body (torso by default).

  The AME tuning guide: "Tracking velocity on the torso is recommended over
  pelvis, as the waist can do active stabilization" -- on a 29-DoF G1 the
  three waist joints sit between the two, so a pelvis-referenced command lets
  the waist absorb the tracking error instead of the legs solving it.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  quat, vel_w = _tracking_frame(asset, asset_cfg)
  vel_yaw = quat_apply_inverse(yaw_quat(quat), vel_w)
  error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
  return torch.exp(-error / std**2)


def _tracking_frame(asset: Entity, asset_cfg: SceneEntityCfg):
  """(quat, world linear velocity) of the tracked body; root when unset."""
  ids = asset_cfg.body_ids
  if ids is None or isinstance(ids, slice):
    return asset.data.root_link_quat_w, asset.data.root_link_lin_vel_w[:, :3]
  idx = ids[0] if not isinstance(ids, int) else ids
  return (
    asset.data.body_link_quat_w[:, idx],
    asset.data.body_link_lin_vel_w[:, idx, :3],
  )


def track_ang_vel_z_world_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Yaw-rate tracking on the tracked body (torso by default; see above)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  ids = asset_cfg.body_ids
  if ids is None or isinstance(ids, slice):
    yaw_rate = asset.data.root_link_ang_vel_w[:, 2]
  else:
    idx = ids[0] if not isinstance(ids, int) else ids
    yaw_rate = asset.data.body_link_ang_vel_w[:, idx, 2]
  error = torch.square(command[:, 2] - yaw_rate)
  return torch.exp(-error / std**2)


##
# Penalties.
##


def ang_vel_xy_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT):
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def applied_torque_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Out-of-range applied torque, summed (mjlab: actuator force vs forcerange)."""
  asset: Entity = env.scene[asset_cfg.name]
  force = asset.data.actuator_force
  limit = _actuator_force_limit(env, asset_cfg.name, force)
  return torch.sum((force.abs() - limit).clamp(min=0.0), dim=1)


_LIMIT_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _actuator_force_limit(
  env: ManagerBasedRlEnv, name: str, like: torch.Tensor
) -> torch.Tensor:
  key = (id(env), name)
  cached = _LIMIT_CACHE.get(key)
  if cached is None:
    rng = torch.as_tensor(
      env.sim.mj_model.actuator_forcerange, dtype=like.dtype, device=like.device
    )
    cached = rng[:, 1].abs().clamp(min=1e-6).unsqueeze(0)
    _LIMIT_CACHE[key] = cached
  return cached


def undesired_contacts(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0
) -> torch.Tensor:
  """Count of non-foot primaries in contact above a force threshold.

  IsaacLab reads per-body contact forces; here the same signal comes from a
  ``ContactSensor`` whose primaries are the non-foot links (see
  ``undesired_contact_sensor_cfg``).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force is not None, f"sensor {sensor_name!r} must request 'force'"
  magnitude = torch.norm(data.force, dim=-1)  # [B, N]
  return torch.sum((magnitude > threshold).float(), dim=1)


def joint_acc_l2_clamped(
  env: ManagerBasedRlEnv,
  clamp: float = 1.0e4,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Joint-acceleration L2 with a physical sanity clamp.

  A contact-solver blowup produces joint accelerations of 1e5+ rad/s^2; an
  unbounded quadratic then injects rewards of -1e5 per step, enough to
  destroy the value function (observed: value loss 0.1 -> 1e16 in 200
  iterations).  Normal gait peaks are ~2e3, so the clamp never binds in
  legitimate motion; the paired blowup terminations end the episode the same
  step, bounding the poison to one bounded reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  acc = asset.data.joint_acc.clamp(-clamp, clamp)
  return torch.sum(torch.square(acc), dim=1)


def joint_vel_l2_clamped(
  env: ManagerBasedRlEnv,
  clamp: float = 100.0,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Joint-velocity L2, clamped for the same blowup containment as above."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel.clamp(-clamp, clamp)
  return torch.sum(torch.square(vel), dim=1)


def base_roll_rate_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """AME-2 Table I penalizes ROLL rate only ([w_b]_x^2): pitch motion is what
  stairs and dynamic descents require, and taxing it fights terrain skill.
  Replaces the roll+pitch ``ang_vel_xy_l2`` in the rebalanced preset."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_b[:, 0])


def link_contact_forces(
  env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]
) -> torch.Tensor:
  """AME-2's impact-softening term: ||max(F_con - G, 0)||^2 — contact forces
  above the robot's own weight, squared, summed over links.  Zero for any
  contact the robot could produce statically; grows with landing impacts."""
  weight = _robot_weight(env)
  total = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  for name in sensor_names:
    data = env.scene[name].data
    assert data.force is not None, f"sensor {name!r} must request 'force'"
    excess = (torch.norm(data.force, dim=-1) - weight).clamp(min=0.0)
    cost = torch.sum(torch.square(excess), dim=1)
    total = total + cost
  return total


_WEIGHT_KEY = "_ame_robot_weight"


def _robot_weight(env: ManagerBasedRlEnv) -> torch.Tensor:
  cached = getattr(env, _WEIGHT_KEY, None)
  if cached is None:
    cached = torch.tensor(
      float(env.sim.mj_model.body_mass.sum() * 9.81),
      device=env.device,
      dtype=torch.float32,
    )
    setattr(env, _WEIGHT_KEY, cached)
  return cached


def joint_vel_limits(
  env: ManagerBasedRlEnv, ratio: float = 0.9, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """AME-2: sum of overruns past ``ratio`` of each joint's velocity limit.
  Zero in all normal operation — a pure sim2real barrier (real actuators
  cannot deliver simulator velocities).  Limits derived per joint from the
  actuator family (mjlab's ElectricActuator constants, keyed by effort)."""
  asset: Entity = env.scene[asset_cfg.name]
  limit = _joint_vel_limit(env, asset, asset_cfg.name)
  return torch.sum((asset.data.joint_vel.abs() - ratio * limit).clamp(min=0.0), dim=1)


_VEL_LIMIT_KEY = "_ame_joint_vel_limit"
# mjlab g1_constants ElectricActuator families: effort limit -> velocity limit.
# 50 N·m = dual-5020 parallel joints (waist roll/pitch, ankles): twice the
# torque of one 5020, same 37 rad/s velocity limit.
_EFFORT_TO_VEL = {25.0: 37.0, 50.0: 37.0, 88.0: 32.0, 139.0: 20.0, 5.0: 22.0}


def _joint_vel_limit(
  env: ManagerBasedRlEnv, asset: Entity, asset_name: str
) -> torch.Tensor:
  limits = getattr(env, _VEL_LIMIT_KEY, None)
  if limits is None:
    limits = {}
    setattr(env, _VEL_LIMIT_KEY, limits)
  cached = limits.get(asset_name)
  if cached is None:
    m = env.sim.mj_model
    efforts = [float(m.actuator_forcerange[a, 1]) for a in range(m.nu)]
    # actuator order != joint order; map through the target joint ids.
    per_joint = {}
    for a, e in enumerate(efforts):
      per_joint[int(m.actuator_trnid[a, 0])] = _EFFORT_TO_VEL[round(e, 1)]
    jids = asset.indexing.joint_ids.cpu().tolist()
    like = asset.data.joint_vel
    cached = torch.tensor(
      [per_joint[j] for j in jids], dtype=like.dtype, device=like.device
    ).unsqueeze(0)
    limits[asset_name] = cached
  return cached


_LINK_VEL_KEY = "_ame_prev_link_vel"


def link_acceleration_l1(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """AME-2: sum over links of ||v_dot_l|| — Cartesian smoothness / soft
  impacts.  mjlab exposes no body accelerations, so this finite-differences
  the link velocities across control steps; freshly reset envs read zero."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.body_link_lin_vel_w
  prev = getattr(env, _LINK_VEL_KEY, None)
  if prev is None or prev.shape != vel.shape:
    setattr(env, _LINK_VEL_KEY, vel.clone())
    return torch.zeros(vel.shape[0], device=vel.device)
  acc = torch.norm(vel - prev, dim=-1).sum(dim=1) / env.step_dt
  prev.copy_(vel)
  acc[env.episode_length_buf <= 1] = 0.0
  return acc


def joint_deviation_l1(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  ids = asset_cfg.joint_ids
  return torch.sum(torch.abs(asset.data.joint_pos[:, ids] - default[:, ids]), dim=1)


##
# Feet.
##


def feet_air_time_positive_biped(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold: float = 0.6,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """IsaacLab's biped air-time reward: credit single-stance swing up to
  ``threshold`` seconds, only while a motion command is active."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  air_time = data.current_air_time
  contact_time = data.current_contact_time
  assert air_time is not None and contact_time is not None
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
  )[0]
  reward = torch.clamp(reward, max=threshold)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return reward * (torch.norm(command[:, :2], dim=1) > command_threshold)


def air_time_variance_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize asymmetry between the feet's air/contact durations."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.last_air_time is not None and data.last_contact_time is not None
  return torch.var(torch.clip(data.last_air_time, max=0.5), dim=1) + torch.var(
    torch.clip(data.last_contact_time, max=0.5), dim=1
  )


def feet_stumble(
  env: ManagerBasedRlEnv, sensor_name: str, ratio: float = 4.0
) -> torch.Tensor:
  """Penalize feet hitting vertical surfaces (lateral force > ratio * normal).

  The SII port fired only at 4x — a foot rolling off a stair/stone edge
  produces 1-2x and went unpunished.  AME-2's own definition is 1x
  ("any link having a horizontal contact force larger than the vertical
  force"), which is what the rebalanced preset uses.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force is not None, f"sensor {sensor_name!r} must request 'force'"
  forces = data.force
  forces_z = forces[..., 2].abs()
  forces_xy = torch.norm(forces[..., :2], dim=-1)
  return torch.any(forces_xy > ratio * forces_z, dim=1).float()


def leaping(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scan_name: str = "terrain_scan",
  span_m: float = 0.3,
) -> torch.Tensor:
  """AME-2's anti-hop term: all feet airborne while the ground is near-flat.

  "Leaping on flat terrain: All feet are off the ground when the elevation
  difference is < 30 cm."  Flight phases are legitimate over gaps and big
  steps (hence the scan-span gate) but on walkable ground a synchronized
  two-leg hop is a degenerate gait that tracks velocity while forfeiting
  everything a stride is for.  Binary per-step occurrence, weight -1.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  airborne = torch.all(found == 0, dim=1)
  data = env.scene[scan_name].data
  z = data.hit_pos_w[..., 2] - data.pos_w[:, 2].unsqueeze(1)
  span = z.max(dim=1).values - z.min(dim=1).values
  return (airborne & (span < span_m)).float()


def feet_too_near(
  env: ManagerBasedRlEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  distance = torch.norm(feet[:, 0] - feet[:, 1], dim=-1)
  return (threshold - distance).clamp(min=0.0)


def feet_slide(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """AME-2's slide penalty: |v_xy| (not squared, not command-gated) in contact."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  vel = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  return torch.sum(vel * in_contact, dim=1)


##
# Cross-body coordination.
##


_COORD_CACHE: dict[tuple[int, str], list[tuple[int, int]]] = {}


def joint_coordination_rel(
  env: ManagerBasedRlEnv,
  coord_joints: list[list[str]],
  coord_signs: list[list[float]] | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Penalize deviation from in-phase motion of cross-body joint pairs
  (left hip pitch with right shoulder pitch, and vice versa)."""
  asset: Entity = env.scene[asset_cfg.name]
  key = (id(env), asset_cfg.name)
  pairs = _COORD_CACHE.get(key)
  if pairs is None:
    names = asset.joint_names
    pairs = [(names.index(a), names.index(b)) for a, b in coord_joints]
    _COORD_CACHE[key] = pairs
  signs = coord_signs or [[1.0, 1.0]] * len(pairs)
  default = asset.data.joint_pos - asset.data.default_joint_pos
  cost = torch.zeros(env.num_envs, device=default.device)
  for (i, j), (si, sj) in zip(pairs, signs, strict=True):
    cost += torch.square(default[:, i] * si - default[:, j] * sj)
  return cost / max(len(pairs), 1)


##
# The two requested RBF posture terms (bounded penalties; see module docstring).
##


def terrain_flatness(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  rough_span_m: float = 0.15,
  rough_weight: float = 0.1,
) -> torch.Tensor:
  """``[B]`` posture-penalty scale: 1 on flat ground, ``rough_weight`` on rough.

  The AME author's tuning guide recommends a terrain-aware weight on the
  humanoid posture penalties -- "0.1x weight for terrains with big elevation
  diffs" -- because on stairs and stones the arms and ankles must be free to
  counterbalance, exactly what a uniform pose penalty forbids.  Roughness is
  the elevation span of the robot's own scan, so it needs no privileged
  terrain knowledge and works unchanged at deploy time.
  """
  data = env.scene[sensor_name].data
  z = data.hit_pos_w[..., 2] - data.pos_w[:, 2].unsqueeze(1)
  span = z.max(dim=1).values - z.min(dim=1).values
  return rough_weight + (1.0 - rough_weight) * torch.exp(
    -torch.square(span / rough_span_m)
  )


def rbf_upper_body_pose(
  env: ManagerBasedRlEnv,
  std: float = 0.5,
  asset_cfg: SceneEntityCfg = _ROBOT,
  terrain_aware: bool = True,
  sensor_name: str = "terrain_scan",
) -> torch.Tensor:
  """``1 - exp(-||q_upper - q_default||^2 / std^2)``: hold the upper body still.

  Zero when the waist/arms sit exactly at their nominal pose, saturating at 1
  as the upper body departs from it.  Bounded, so a single bad frame cannot
  dominate the return the way an L2 term can.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  ids = asset_cfg.joint_ids
  error = torch.sum(torch.square(asset.data.joint_pos[:, ids] - default[:, ids]), dim=1)
  penalty = 1.0 - torch.exp(-error / std**2)
  if terrain_aware:
    penalty = penalty * terrain_flatness(env, sensor_name)
  return penalty


def rbf_foot_yaw(
  env: ManagerBasedRlEnv,
  std: float = 0.5,
  asset_cfg: SceneEntityCfg = _ROBOT,
  terrain_aware: bool = True,
  sensor_name: str = "terrain_scan",
) -> torch.Tensor:
  """``1 - exp(-sum_f wrap(yaw_f - yaw_base)^2 / std^2)``: feet point where the
  base points.

  Splayed or counter-rotated feet are the signature of a policy that turns by
  scuffing rather than stepping, so this both cleans up the gait and removes
  the yaw-drift failure mode.
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_yaw = _yaw_of(asset.data.root_link_quat_w)  # [B]
  foot_yaw = _yaw_of(asset.data.body_link_quat_w[:, asset_cfg.body_ids, :])  # [B, F]
  error = torch.sum(torch.square(wrap_to_pi(foot_yaw - base_yaw.unsqueeze(-1))), dim=1)
  penalty = 1.0 - torch.exp(-error / std**2)
  if terrain_aware:
    penalty = penalty * terrain_flatness(env, sensor_name)
  return penalty


def rbf_foot_flat(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  std: float = 0.4,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """``1 - exp(-sum_f c_f |g_xy,f|^2 / std^2)``: a loaded foot lies flat.

  ``g_f`` is gravity in foot ``f``'s frame, so ``|g_xy|`` is the sine of the
  foot's tilt; only feet in contact count (``c_f``), since a swing foot may
  pitch for toe-off and heel-strike.  A foot rolling over a stone or stair
  edge tilts under load — the placement failure that ends most episodes on
  stones — where the stumble term only sees the lateral force spike.  Not
  terrain-scaled: flat contact matters most where the ground is not.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, F, 4]
  b, f = quat.shape[:2]
  down = torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(b * f, 3)
  g_foot = quat_apply_inverse(quat.reshape(-1, 4), down).reshape(b, f, 3)
  tilt = torch.sum(torch.square(g_foot[..., :2]), dim=-1)  # sin^2(tilt) per foot
  in_contact = (sensor.data.found > 0).float()
  error = torch.sum(tilt * in_contact, dim=1)
  return 1.0 - torch.exp(-error / std**2)


__all__ = [
  "air_time_variance_penalty",
  "ang_vel_xy_l2",
  "applied_torque_limits",
  "base_roll_rate_l2",
  "feet_air_time_positive_biped",
  "feet_slide",
  "feet_stumble",
  "feet_too_near",
  "joint_acc_l2_clamped",
  "joint_coordination_rel",
  "joint_deviation_l1",
  "joint_vel_l2_clamped",
  "joint_vel_limits",
  "leaping",
  "link_acceleration_l1",
  "link_contact_forces",
  "rbf_foot_flat",
  "rbf_foot_yaw",
  "rbf_upper_body_pose",
  "terrain_flatness",
  "track_ang_vel_z_world_exp",
  "track_lin_vel_xy_yaw_frame_exp",
  "undesired_contacts",
]
