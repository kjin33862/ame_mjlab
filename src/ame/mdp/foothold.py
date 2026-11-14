"""BeamDojo-style foothold reward on the G1 sole (arXiv 2502.10363, Eq. 2), adapted.

BeamDojo penalises every sole sample point of a foot in contact whose global terrain
height is below eps = -0.1 m (its terrain is a flat-topped field over a pit).  Our
board also has stairs and height fields, so a point is *unsupported* when the sole at
that point sits more than ``delta`` above the terrain right under it:

    unsupported_ij = (z_sole_ij - z_terrain_ij) > delta

and the term is the per-foot unsupported *fraction* while the foot is in contact,
summed over the feet (0 to 2).  Used with a negative weight.

The sole.  The G1 has no sole box: each foot carries seven capsules of radius 0.01 m
along x at z = -0.025 m in ``*_ankle_roll_link`` (``foot1`` .. ``foot7``), so the sole
plane is z = -0.035 m and the capsules' bounding box is x in [-0.064, 0.142] m,
y in [-0.037, 0.036] m (0.206 x 0.073 m; the right foot is 0.072 m wide).  The 3 x 5
grid sits on capsule axes, so every point is on the contact surface: y in
{-0.018, 0, 0.018} (the outer full-length capsules and the centre one) and five x
evenly over [-0.044, 0.123] (the y = +-0.018 capsules' extent).

The rays.  One extra ray-cast sensor, ``foot_sole_scan``, attached to the
``left_foot`` / ``right_foot`` sites (sole centre, 2 mm below the capsules), yaw
aligned, a custom pattern holding the 15 points lifted ``RAY_LIFT`` above the sole so
the ray never starts inside the ground.  ``z_sole`` is the exact sole point (full foot
rotation, from the frame pose the sensor recorded with the hits); ``z_terrain`` is the
hit.  A miss (nothing within ``max_distance``) is unsupported; a back-face hit (the ray
started inside terrain, i.e. terrain above the lifted origin) is supported.

Timing: mjlab ray-casts once per policy step, at the end of ``env.step``, so at reward
time the hits and frame pose are one policy step old (consistent with each other),
exactly like mjlab's own ``foot_clearance``; the contact flag is current.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor, RayCastSensor

FOOT_SITES = ("left_foot", "right_foot")
SITE_POS_LINK = (0.04, 0.0, -0.037)
"""The foot site in ``*_ankle_roll_link`` (g1.xml)."""
SOLE_Z_LINK = -0.035
"""Capsule bottoms: axis z = -0.025, radius 0.01."""
SOLE_BOX_LINK = ((-0.064, 0.142), (-0.037, 0.036))
"""Bounding box of the seven foot capsules, x and y, in ``*_ankle_roll_link``."""
SOLE_GRID_X = (-0.044, 0.123)
SOLE_GRID_Y = (-0.018, 0.0, 0.018)
SOLE_GRID_NX = 5
RAY_LIFT = 0.05
"""Ray origins sit this far above the sole (inside the foot), m."""
SOLE_SCAN_SENSOR = "foot_sole_scan"
FEET_CONTACT_SENSOR = "feet_ground_contact"
FOOTHOLD_DELTA = 0.05
"""Default unsupported threshold, m: the lowest stair step on the board is 0.05 m."""
FOOTHOLD_WEIGHT = -1.0
"""BeamDojo's weight 1.0 on ``-sum``; here the term is a cost, so -1.0."""
GAP_DROP = 0.5
"""A sole point is over a gap / pit when the terrain under it is this far below the
sole (stairs are at most 0.20 m; the pits are 2 m)."""


def sole_points_link() -> torch.Tensor:
  """``[15, 3]`` sole sample points in ``*_ankle_roll_link``, x fastest."""
  lo, hi = SOLE_GRID_X
  xs = [lo + (hi - lo) * i / (SOLE_GRID_NX - 1) for i in range(SOLE_GRID_NX)]
  return torch.tensor(
    [(x, y, SOLE_Z_LINK) for y in SOLE_GRID_Y for x in xs], dtype=torch.float32
  )


def sole_points_site() -> torch.Tensor:
  """``[15, 3]`` the same points relative to the foot site (same orientation)."""
  return sole_points_link() - torch.tensor(SITE_POS_LINK)


@dataclass
class SolePatternCfg:
  """Fixed ray origins: the sole grid lifted by ``lift``, straight down.

  Duck-types mjlab's pattern cfgs (``generate_rays`` is all the sensor calls)."""

  lift: float = RAY_LIFT

  def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    del mj_model
    offsets = sole_points_site().to(device)
    offsets[:, 2] += self.lift
    directions = torch.zeros_like(offsets)
    directions[:, 2] = -1.0
    return offsets, directions


def foot_sole_scan_cfg(max_distance: float = 1.0) -> RayCastSensorCfg:
  """2 frames x 15 rays, terrain only (geom group 0)."""
  return RayCastSensorCfg(
    name=SOLE_SCAN_SENSOR,
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
    pattern=SolePatternCfg(),  # pyright: ignore[reportArgumentType]
    ray_alignment="yaw",
    max_distance=max_distance,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )


def sole_drop(
  frame_pos_w: torch.Tensor,
  frame_quat_w: torch.Tensor,
  hit_pos_w: torch.Tensor,
  distances: torch.Tensor,
  normals_w: torch.Tensor,
  points_site: torch.Tensor | None = None,
) -> torch.Tensor:
  """``[B, F, N]`` sole height above the terrain under each point, m.

  ``frame_*``: ``[B, F, 3 | 4]``; ``hit_pos_w`` / ``normals_w``: ``[B, F * N, 3]``;
  ``distances``: ``[B, F * N]`` (-1 = miss).  A miss is ``+inf`` (unsupported at any
  delta), a back-face hit ``-inf`` (terrain above the ray origin: supported)."""
  B, F = frame_pos_w.shape[:2]
  pts = (points_site if points_site is not None else sole_points_site()).to(
    frame_pos_w.device
  )
  N = pts.shape[0]
  quat = frame_quat_w[:, :, None, :].expand(B, F, N, 4)
  sole = frame_pos_w[:, :, None, :] + quat_apply(
    quat.reshape(-1, 4), pts.expand(B, F, N, 3).reshape(-1, 3)
  ).view(B, F, N, 3)
  hit_z = hit_pos_w[..., 2].view(B, F, N)
  dist = distances.view(B, F, N)
  normal_z = normals_w[..., 2].view(B, F, N)
  drop = sole[..., 2] - hit_z
  inf = torch.full_like(drop, float("inf"))
  drop = torch.where(dist < 0, inf, drop)
  drop = torch.where((dist >= 0) & (normal_z < 0), -inf, drop)
  return drop


def sensor_sole_drop(sensor: RayCastSensor) -> torch.Tensor:
  """:func:`sole_drop` from a ``foot_sole_scan`` sensor's data."""
  d = sensor.data
  return sole_drop(d.frame_pos_w, d.frame_quat_w, d.hit_pos_w, d.distances, d.normals_w)


def unsupported_fraction(drop: torch.Tensor, delta: float) -> torch.Tensor:
  """``[B, F]`` share of sole points with ``drop > delta``."""
  return (drop > delta).float().mean(dim=-1)


def feet_in_contact(contact: ContactSensor) -> torch.Tensor:
  """``[B, F]`` bool, the contact flag the air-time bookkeeping uses: any matched
  contact between the foot subtree and the terrain (``found > 0``; no force
  threshold, one slot per foot)."""
  found = contact.data.found
  assert found is not None
  return found > 0


def check_foot_order(contact: ContactSensor, sensor_cfg: RayCastSensorCfg) -> None:
  frames = (
    sensor_cfg.frame if isinstance(sensor_cfg.frame, tuple) else (sensor_cfg.frame,)
  )
  sides = [f.name.split("_")[0] for f in frames]
  prims = [p.split("/")[-1].split("_")[0] for p in contact.primary_names]
  if sides != prims:
    raise ValueError(f"foot order differs: sole scan {sides}, contact {prims}")


class foothold_unsupported:
  """Reward term: sum over feet of contact * unsupported fraction, in [0, 2]."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor = env.scene[cfg.params.get("sensor_name", SOLE_SCAN_SENSOR)]
    contact = env.scene[cfg.params.get("contact_sensor_name", FEET_CONTACT_SENSOR)]
    check_foot_order(contact, sensor.cfg)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str = SOLE_SCAN_SENSOR,
    contact_sensor_name: str = FEET_CONTACT_SENSOR,
    delta: float = FOOTHOLD_DELTA,
  ) -> torch.Tensor:
    frac = unsupported_fraction(sensor_sole_drop(env.scene[sensor_name]), delta)
    contact = feet_in_contact(env.scene[contact_sensor_name]).float()
    return (frac * contact).sum(dim=-1)

  def reset(self, env_ids=None) -> None:
    del env_ids


def foothold_reward_cfg(
  delta: float = FOOTHOLD_DELTA, weight: float = FOOTHOLD_WEIGHT
) -> RewardTermCfg:
  return RewardTermCfg(
    func=foothold_unsupported,
    weight=weight,
    params={
      "sensor_name": SOLE_SCAN_SENSOR,
      "contact_sensor_name": FEET_CONTACT_SENSOR,
      "delta": delta,
    },
  )


__all__ = [
  "FOOTHOLD_DELTA",
  "FOOTHOLD_WEIGHT",
  "GAP_DROP",
  "SOLE_BOX_LINK",
  "SOLE_SCAN_SENSOR",
  "SolePatternCfg",
  "feet_in_contact",
  "foot_sole_scan_cfg",
  "foothold_reward_cfg",
  "foothold_unsupported",
  "sensor_sole_drop",
  "sole_drop",
  "sole_points_link",
  "sole_points_site",
  "unsupported_fraction",
]
