"""Assemble the AME-2 reward stack (+ RBF posture terms) onto a mjlab env cfg.

Weights are the reference's, verbatim
(``29dof/velocity_env_cfg_29dof.py::RewardsCfg``).  Two notes on how they land
here:

* IsaacLab reward terms are per-step values multiplied by ``dt`` in the
  manager, exactly as mjlab does, so the weights carry over directly.
* The reference's actuator model and PD gains differ from mjlab's G1, so the
  *effort-shaped* terms (torque/acceleration/velocity L2) are the ones most
  likely to need retuning; they are grouped together below for that reason.

``apply_ame2_rewards`` also installs the contact sensor the stack needs
(``undesired_contact`` over the non-foot links) and replaces the termination
set with the reference's illegal-contact rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ame.mdp import rewards as ame_rew
from ame.mdp.terminations import bad_orientation, illegal_contact, physics_blowup

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

TORSO_BODY_RE = r"^torso_link$"
FOOT_BODY_RE = r"^(left|right)_ankle_roll_link$"
FOOT_SITES = ("left_foot", "right_foot")
UPPER_BODY_JOINT_RE = (
  r"^waist_.*_joint$",
  r"^(left|right)_shoulder_.*_joint$",
  r"^(left|right)_elbow_joint$",
  r"^(left|right)_wrist_.*_joint$",
)
NON_FOOT_LINK_RE = (
  r"^torso_link$",
  r"^(left|right)_shoulder_.*_link$",
  r"^(left|right)_hip_.*_link$",
  r"^(left|right)_knee_link$",
  r"^(left|right)_elbow_link$",
  r"^waist_.*_link$",
  r"^pelvis$",
)

UNDESIRED_CONTACT_SENSOR = "undesired_contact"
FEET_CONTACT_SENSOR = "feet_ground_contact"


def undesired_contact_sensor_cfg() -> ContactSensorCfg:
  """Contact sensor over the links that should never touch the ground."""
  return ContactSensorCfg(
    name=UNDESIRED_CONTACT_SENSOR,
    # mode="body", NOT "subtree": a subtree match on `pelvis` covers the
    # whole robot (feet included), so every footstep would read as an
    # illegal contact.  Measured before the fix: 34 % of steps terminated
    # and -68 reward/step for a policy that walks fine.
    primary=ContactMatch(mode="body", pattern=NON_FOOT_LINK_RE, entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def ame2_rewards(rbf_weight: float = 1.0, rbf_std: float = 0.5) -> dict:
  """The reference reward dict, plus the two RBF posture penalties."""
  feet_sites = SceneEntityCfg("robot", site_names=FOOT_SITES)
  feet_bodies = SceneEntityCfg("robot", body_names=(FOOT_BODY_RE,))
  upper_body = SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_RE)
  # Guide: track the torso, not the pelvis -- the waist joints in between can
  # otherwise absorb the tracking error instead of the legs solving it.
  torso = SceneEntityCfg("robot", body_names=(TORSO_BODY_RE,))

  return {
    # -- task ---------------------------------------------------------------
    "termination_penalty": RewTerm(func=envs_mdp.is_terminated, weight=-200.0),
    "track_lin_vel_xy_exp": RewTerm(
      func=ame_rew.track_lin_vel_xy_yaw_frame_exp,
      weight=1.0,
      params={"command_name": "twist", "std": 0.25, "asset_cfg": torso},
    ),
    "track_ang_vel_z_exp": RewTerm(
      func=ame_rew.track_ang_vel_z_world_exp,
      weight=2.0,
      params={"command_name": "twist", "std": 0.25, "asset_cfg": torso},
    ),
    # -- penalties ----------------------------------------------------------
    "ang_vel_xy_l2": RewTerm(func=ame_rew.ang_vel_xy_l2, weight=-0.05),
    "undesired_contacts": RewTerm(
      func=ame_rew.undesired_contacts,
      weight=-1.0,
      params={"sensor_name": UNDESIRED_CONTACT_SENSOR, "threshold": 1.0},
    ),
    "flat_orientation_l2": RewTerm(func=envs_mdp.flat_orientation_l2, weight=-2.0),
    "dof_pos_limits": RewTerm(func=envs_mdp.joint_pos_limits, weight=-1.0),
    "dof_torques_limits": RewTerm(func=ame_rew.applied_torque_limits, weight=-0.01),
    "action_rate_l2": RewTerm(func=envs_mdp.action_rate_l2, weight=-0.01),
    # Effort-shaped: the terms most sensitive to the actuator-model change.
    # Torques are bounded by actuator forcerange; acc/vel are clamped in our
    # variants so a physics blowup cannot inject unbounded negative reward.
    "dof_torques_l2": RewTerm(func=envs_mdp.joint_torques_l2, weight=-1.5e-7),
    "dof_acc_l2": RewTerm(func=ame_rew.joint_acc_l2_clamped, weight=-1.25e-7),
    "dof_vel_l2": RewTerm(func=ame_rew.joint_vel_l2_clamped, weight=-0.001),
    # -- style / feet -------------------------------------------------------
    "feet_air_time": RewTerm(
      func=ame_rew.feet_air_time_positive_biped,
      weight=0.25,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": "twist",
        "threshold": 0.6,
      },
    ),
    "feet_air_time_variance": RewTerm(
      func=ame_rew.air_time_variance_penalty,
      weight=-0.1,
      params={"sensor_name": FEET_CONTACT_SENSOR},
    ),
    "feet_slide": RewTerm(
      func=ame_rew.feet_slide,
      weight=-0.1,
      params={"sensor_name": FEET_CONTACT_SENSOR, "asset_cfg": feet_sites},
    ),
    "feet_stumble": RewTerm(
      func=ame_rew.feet_stumble,
      weight=-1.0,
      params={"sensor_name": FEET_CONTACT_SENSOR},
    ),
    "feet_too_near": RewTerm(
      func=ame_rew.feet_too_near,
      weight=-1.0,
      params={"threshold": 0.2, "asset_cfg": feet_sites},
    ),
    # -- coordination -------------------------------------------------------
    "joint_coordination": RewTerm(
      func=ame_rew.joint_coordination_rel,
      weight=-0.1,
      params={
        "coord_joints": [
          ["left_hip_pitch_joint", "right_shoulder_pitch_joint"],
          ["right_hip_pitch_joint", "left_shoulder_pitch_joint"],
        ],
        "coord_signs": [[1.0, 1.0], [1.0, 1.0]],
      },
    ),
    "joint_deviation_hip": RewTerm(
      func=ame_rew.joint_deviation_l1,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            r"^(left|right)_hip_yaw_joint$",
            r"^(left|right)_hip_roll_joint$",
          ),
        )
      },
    ),
    "joint_deviation_arms": RewTerm(
      func=ame_rew.joint_deviation_l1,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            r"^(left|right)_shoulder_.*_joint$",
            r"^(left|right)_elbow_joint$",
            r"^(left|right)_wrist_.*_joint$",
          ),
        )
      },
    ),
    "joint_deviation_waists": RewTerm(
      func=ame_rew.joint_deviation_l1,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^waist_.*_joint$",))},
    ),
    # -- requested RBF posture terms (bounded penalties) --------------------
    "rbf_upper_body_pose": RewTerm(
      func=ame_rew.rbf_upper_body_pose,
      weight=-rbf_weight,
      params={"std": rbf_std, "asset_cfg": upper_body},
    ),
    "rbf_foot_yaw": RewTerm(
      func=ame_rew.rbf_foot_yaw,
      weight=-rbf_weight,
      params={"std": rbf_std, "asset_cfg": feet_bodies},
    ),
  }


REBALANCE = {
  # Tracking.  The verbatim weights trained policies that turn but do not
  # walk: linear's sigma=0.25 kernel gives a standing robot at cmd 0.5 m/s a
  # reward of exp(-4) ~ 0.02, i.e. no gradient toward walking.  With linear
  # fixed (w3.0, sigma 0.5), angular tracking at w1.0-2.0 then flat-lined —
  # turning never paid for its own action costs next to the walking reward —
  # so yaw is made emphatically worth it at w6.0.
  "track_lin_vel_xy_exp": {"weight": 3.0, "std": 0.5},
  "track_ang_vel_z_exp": {"weight": 6.0},
  "termination_penalty": {"weight": -50.0},
  "feet_air_time": {"weight": 0.5},
  # Gait quality (AME-2 paper + tuning guide; see rewards.feet_stumble and
  # the map drift in observations): edge strikes penalized at the paper's 1x
  # force ratio.
  "feet_stumble": {"ratio": 1.0},
  # Action rate at -0.025 (2.5x the SII value).  A stronger rate penalty
  # locked training into a symmetric two-leg hop: swinging ONE leg is exactly
  # the high-action-derivative move a rate penalty taxes.  This keeps calming
  # pressure without pricing single-leg initiation out.  The AME-2 leaping
  # term is implemented in rewards.py but left out of this preset.
  "action_rate_l2": {"weight": -0.025},
  "dof_vel_l2": {"weight": -0.001},
  # AME-2 Table I (the authors' own weights): roll-only base angular penalty
  # at -0.1 — pitch freedom is what stairs need — and torque regularization
  # raised to their 1e-5 (the SII reproduction shipped 1.5e-7, 70x weaker).
  "ang_vel_xy_l2": {"func": ame_rew.base_roll_rate_l2, "weight": -0.1},
  "dof_torques_l2": {"weight": -1.0e-5},
}

# AME-2 Table I terms with no counterpart in the SII stack (the paper's weights).
EXTRA_TERMS = {
  "link_contact_forces": lambda: RewTerm(
    func=ame_rew.link_contact_forces,
    weight=-1.0e-5,
    params={"sensor_names": (FEET_CONTACT_SENSOR, UNDESIRED_CONTACT_SENSOR)},
  ),
  "joint_vel_limits": lambda: RewTerm(
    func=ame_rew.joint_vel_limits, weight=-1.0, params={"ratio": 0.9}
  ),
  "link_acceleration": lambda: RewTerm(
    func=ame_rew.link_acceleration_l1, weight=-0.001
  ),
}


def rebalance(terms: dict) -> dict:
  """Apply :data:`REBALANCE` and add :data:`EXTRA_TERMS` to an ame2 reward dict."""
  for name, delta in REBALANCE.items():
    term = terms[name]
    for key, value in delta.items():
      if key == "weight":
        term.weight = value
      elif key == "func":
        term.func = value
      else:
        term.params[key] = value
  for name, make in EXTRA_TERMS.items():
    terms[name] = make()
  return terms


def apply_ame2_rewards(
  cfg: ManagerBasedRlEnvCfg, rbf_weight: float = 1.0, rbf_std: float = 0.5
) -> None:
  """Swap mjlab's G1 velocity rewards for the AME-2 stack, in place."""
  sensors = list(cfg.scene.sensors or ())
  if not any(getattr(s, "name", None) == UNDESIRED_CONTACT_SENSOR for s in sensors):
    sensors.append(undesired_contact_sensor_cfg())
    cfg.scene.sensors = tuple(sensors)

  cfg.rewards = ame2_rewards(rbf_weight=rbf_weight, rbf_std=rbf_std)

  # The reference terminates on any illegal (non-foot) contact; keep mjlab's
  # time_out so truncation is still distinguished from failure.
  time_out = cfg.terminations.get("time_out")
  # AME-2's termination list in full: illegal contact, plus the paper's
  # guards for "irrecoverable failure cases" (bad orientation, joint
  # acceleration blowup).  Without the blowup guard a contact-solver
  # explosion feeds poisoned rewards for whole episodes and can destroy the
  # value function.
  cfg.terminations = {
    "illegal_contact": DoneTerm(
      func=illegal_contact,
      params={"sensor_name": UNDESIRED_CONTACT_SENSOR, "threshold": 1.0},
    ),
    "bad_orientation": DoneTerm(func=bad_orientation),
    "physics_blowup": DoneTerm(func=physics_blowup),
  }
  if time_out is not None:
    cfg.terminations["time_out"] = time_out


__all__ = [
  "EXTRA_TERMS",
  "FEET_CONTACT_SENSOR",
  "REBALANCE",
  "UNDESIRED_CONTACT_SENSOR",
  "ame2_rewards",
  "apply_ame2_rewards",
  "undesired_contact_sensor_cfg",
]
