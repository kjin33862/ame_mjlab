"""Evaluation-only env config.  Not used by training or play.

Differences from ``ame_rough_env_cfg(play=True)``:

* a ``num_rows x num_cols`` board of ONE terrain family at a FIXED difficulty,
  generator seeded (a family of the AME mix, or ``stone_lattice``); env ``i``
  spawns at the origin of tile ``(i // num_cols % num_rows, i % num_cols)``
  with yaw 0 (:func:`assign_tiles`): the tile centre, or on ``stone_lattice``
  the stone nearest it;
* no ``randomize_terrain`` reset event, no CoM randomization (foot friction and
  encoder bias stay on, as in play);
* fixed twist command (default ``vx = 1, vy = 0, wz = 0``), no standing envs;
  or, with ``commands="train"``, the gap task's training twist commands
  (random body-frame twist resampled every 3-8 s, 5 % standing, 20 %
  straight forward);
* 20 s episodes;
* map / proprio noise from a preset or explicit values (:func:`apply_eval_noise`);
* the illegal-contact termination also records which body touched (same
  termination result);
* optionally (``spawn_asym``) a left-right symmetry break at spawn: the left
  hip pitch joint starts ``spawn_asym`` rad off the default pose;
* ``actor_obs="deploy"`` drops ``base_lin_vel`` from the actor group (the
  deployment set, as in the gap-board task), for checkpoints whose actor was
  trained without it (:func:`checkpoint_actor_obs` tells which).
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ame.env_cfg import ame_gap_env_cfg, ame_rough_env_cfg, apply_deploy_observations
from ame.eval.observations import elevation_map_eval
from ame.eval.terminations import illegal_contact_recorded
from ame.eval.terrains import stone_lattice_cfg
from ame.mdp import observations as ame_obs
from ame.scan import MAP_GROUP
from ame.symmetry import has_lin_vel
from ame.terrains import ame_terrains_cfg

EVAL_TERRAINS = (*ame_terrains_cfg().sub_terrains, "stone_lattice")
"""Families of the AME training mix (``stepping_stones`` is the old stones
tile, with the centre platform), plus ``stone_lattice``: the gap task's lattice
(gap = 0.45 * difficulty, :mod:`ame.eval.terrains`)."""

TRAIN_HEIGHT_NOISE = {
  "height_sigma": ame_obs._NOISE_STD_M,
  "height_offset": ame_obs._OFFSET_RANGE_M,
  "drift_cells": ame_obs._DRIFT_CELLS,
  "delay_steps": 1,
}
"""The training map noise (``elevation_map(noise=True)``)."""
SPAWN_ASYM_JOINT = "left_hip_pitch_joint"
"""The joint ``spawn_asym`` offsets at every reset."""
OFF_HEIGHT_NOISE = {
  "height_sigma": 0.0,
  "height_offset": 0.0,
  "drift_cells": 0,
  "delay_steps": 1,
}
COMMAND_MODES = ("fixed", "train")
"""``fixed``: the constant ``command`` twist; ``train``: the gap task's training
twist commands."""
ACTOR_OBS = ("full", "deploy")
"""Actor observation sets: ``full`` (with ``base_lin_vel``, the AME tasks) or
``deploy`` (without it)."""


def checkpoint_actor_obs(path: str) -> str:
  """``full`` or ``deploy``, from the width of the actor's proprio input."""
  state = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
  width = state["proprio_embedding.weight"].shape[1]
  return "full" if has_lin_vel(width) else "deploy"


def eval_env_cfg(
  terrain: str = "stepping_stones",
  num_rows: int = 8,
  num_cols: int = 8,
  difficulty: float = 0.5,
  seed: int = 0,
  episode_s: float = 20.0,
  height_noise: dict[str, Any] | None = None,
  proprio_noise: bool = True,
  command: tuple[float, float, float] = (1.0, 0.0, 0.0),
  com_dr: bool = False,
  spawn_asym: float = 0.0,
  actor_obs: str = "full",
  commands: str = "fixed",
) -> ManagerBasedRlEnvCfg:
  """Build the evaluation config.  ``height_noise`` defaults to the training
  map noise; ``proprio_noise`` turns on the actor's training noise."""
  if terrain not in EVAL_TERRAINS:
    raise ValueError(f"terrain must be one of {EVAL_TERRAINS}, got {terrain!r}")
  if commands not in COMMAND_MODES:
    raise ValueError(f"commands must be one of {COMMAND_MODES}, got {commands!r}")
  if actor_obs not in ACTOR_OBS:
    raise ValueError(f"actor_obs must be one of {ACTOR_OBS}, got {actor_obs!r}")
  cfg = ame_rough_env_cfg(play=True)
  if actor_obs == "deploy":
    apply_deploy_observations(cfg)
  cfg.seed = seed
  cfg.scene.num_envs = num_rows * num_cols
  cfg.episode_length_s = episode_s

  # One family, fixed difficulty, seeded; curriculum off = every tile is drawn
  # from the (single) family at a difficulty in difficulty_range.
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None
  if terrain == "stone_lattice":
    family = stone_lattice_cfg()
  else:
    family = copy.deepcopy(ame_terrains_cfg().sub_terrains[terrain])
  family.proportion = 1.0
  cfg.scene.terrain.terrain_generator = replace(
    gen,
    curriculum=False,
    num_rows=num_rows,
    num_cols=num_cols,
    difficulty_range=(difficulty, difficulty),
    seed=seed,
    sub_terrains={terrain: family},
  )

  # Deterministic spawn: tile center, yaw 0 (assign_tiles sets the tile).
  cfg.events.pop("randomize_terrain", None)
  if not com_dr:
    cfg.events.pop("base_com", None)
  pose = dict(cfg.events["reset_base"].params["pose_range"])
  pose.update({"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})
  cfg.events["reset_base"].params["pose_range"] = pose
  if spawn_asym:
    # Appended after reset_robot_joints, so it runs last and overrides only
    # this one joint (velocity stays at the default 0).
    cfg.events["spawn_asym"] = EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (spawn_asym, spawn_asym),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(SPAWN_ASYM_JOINT,)),
      },
    )

  if commands == "train":
    twist = copy.deepcopy(ame_gap_env_cfg().commands["twist"])
    twist.debug_vis = False
    cfg.commands["twist"] = twist
  else:
    _fix_twist(cfg, command)

  apply_eval_noise(cfg, height_noise or dict(TRAIN_HEIGHT_NOISE), proprio_noise)

  # Same contact termination, plus a record of the touching body.
  cfg.terminations["illegal_contact"].func = illegal_contact_recorded
  return cfg


def _fix_twist(cfg: ManagerBasedRlEnvCfg, command: tuple[float, float, float]) -> None:
  """A constant twist command for every env, no standing or forward-only draws."""
  twist: Any = cfg.commands["twist"]
  vx, vy, wz = command
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (vy, vy)
  twist.ranges.ang_vel_z = (wz, wz)
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.init_velocity_prob = 0.0
  twist.debug_vis = False


def apply_eval_noise(
  cfg: ManagerBasedRlEnvCfg, height_noise: dict[str, Any], proprio_noise: bool
) -> None:
  """Map noise: the original training term for the training values (exact
  reproduction), the plain term for no noise, :class:`elevation_map_eval`
  otherwise.  Proprio noise: the actor group's own training noise on/off."""
  term = cfg.observations[MAP_GROUP].terms["elevation"]
  sensor_name = term.params["sensor_name"]
  if height_noise == TRAIN_HEIGHT_NOISE:
    term.func = ame_obs.elevation_map
    term.params = {"sensor_name": sensor_name, "noise": True}
  elif height_noise == OFF_HEIGHT_NOISE:
    term.func = ame_obs.elevation_map
    term.params = {"sensor_name": sensor_name, "noise": False}
  else:
    cfg.observations[MAP_GROUP].terms["elevation"] = ObservationTermCfg(
      func=elevation_map_eval,
      params={
        "sensor_name": sensor_name,
        "height_sigma": float(height_noise["height_sigma"]),
        "height_offset": float(height_noise["height_offset"]),
        "drift_cells": int(height_noise["drift_cells"]),
        "delay_steps": int(height_noise["delay_steps"]),
      },
    )
  cfg.observations["actor"].enable_corruption = proprio_noise


def assign_tiles(env: Any) -> None:
  """Env ``i`` -> tile ``(i // num_cols % num_rows, i % num_cols)``.  Call
  after the env is built and before its first reset (the RSL-RL wrapper
  resets in its constructor); the reset event then spawns at the tile origin."""
  terrain = env.scene.terrain
  origins = terrain.terrain_origins
  num_rows, num_cols = origins.shape[:2]
  ids = torch.arange(env.num_envs, device=origins.device)
  terrain.terrain_levels[:] = (ids // num_cols) % num_rows
  terrain.terrain_types[:] = ids % num_cols
  terrain.env_origins[:] = origins[terrain.terrain_levels, terrain.terrain_types]


__all__ = [
  "ACTOR_OBS",
  "EVAL_TERRAINS",
  "OFF_HEIGHT_NOISE",
  "SPAWN_ASYM_JOINT",
  "TRAIN_HEIGHT_NOISE",
  "apply_eval_noise",
  "assign_tiles",
  "checkpoint_actor_obs",
  "eval_env_cfg",
]
