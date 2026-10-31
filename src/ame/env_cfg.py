"""The AME rough-terrain task as mutators over mjlab's G1 rough velocity task.

mjlab's task already carries a simulator-tuned reward stack, actuator model
and domain randomisation, so the recipe layers only its own pieces:

* ``apply_terrain`` — the reference mix widened with mjlab primitives
  (:mod:`ame.terrains`), spawning on standable rows;
* ``apply_scan`` — the 33 x 21 yaw-aligned elevation scan as its own
  observation group (:mod:`ame.scan`);
* ``apply_twist_commands`` — heading control off, the command is the twist
  ``(vx, vy, wz)`` with ``vy`` open, so a higher-level planner can steer the
  policy anywhere in the twist box;
* ``apply_heading_commands`` — mjlab's heading control back on for 30 % of
  commands, for tasks that want it;
* ``apply_rewards`` — mjlab's stack, the AME-2 stack, or the rebalanced AME-2
  stack the registered tasks train on (:mod:`ame.reward_cfg`);
* ``apply_patch_spawn`` / ``apply_terrain_floor`` / ``add_tracking_metrics`` —
  on-stone spawns, a progress-gated terrain curriculum and honest per-step
  tracking metrics, for the gap-board task.

Each mutator is independent, so they compose onto other tasks; the factories
below are the from-scratch tasks.
"""

from __future__ import annotations

import math
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from ame.mdp.curriculums import terrain_levels_floor
from ame.mdp.events import LATTICE_YAWS, SPAWN_POSE, reset_root_state_on_patches
from ame.mdp.metrics import (
  command_speed,
  pelvis_progress,
  pelvis_speed,
  pelvis_vel_error,
  pelvis_yaw_error,
)
from ame.reward_cfg import apply_ame2_rewards, rebalance
from ame.scan import apply_scan
from ame.terrains import (
  ame_terrains_cfg,
  gap_board_cfg,
  play_board,
  stage_terrains_cfg,
)

REWARD_STACKS = ("mjlab", "ame2", "ame2_rebalanced")

NUM_ENVS = 1024
"""Default training env count (fits a 16 GB GPU; ~9.6 GB measured)."""

TWIST_RANGES = {
  "lin_vel_x": (-0.3, 1.5),
  "lin_vel_y": (-0.5, 0.5),
  "ang_vel_z": (-1.0, 1.0),
}
"""A small reverse band for gate manoeuvring; ``vy`` opened for the twist box."""

GAP_CURRICULUM = {"floor": 0, "rule": "tracking", "spawn_max": 2, "track_ratio": 0.6}
"""Rows 0-2 to start; a timed-out episode promotes only if it covered 60 % of
its commanded speed, a stall or an early fall demotes."""

STAGE_REWARDS = {
  # (stage 1, stage 2) weights, from the reference's FINETUNE conditionals.
  "feet_stumble": (-2.0, -5.0),
  "flat_orientation_l2": (-2.0, -5.0),
  "feet_air_time_variance": (-0.1, -0.7),
}


def apply_terrain(cfg: Any, gen: TerrainGeneratorCfg, play: bool = False) -> None:
  """Install ``gen``; spawn on rows 0-1 (row 5 of the mix is unlearnable
  from scratch and the curriculum promotes fast anyway)."""
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = play_board(gen) if play else gen
  cfg.scene.terrain.max_init_terrain_level = 1


def apply_twist_commands(cfg: Any) -> None:
  twist: Any = cfg.commands["twist"]
  twist.heading_command = False
  twist.rel_heading_envs = 0.0
  twist.rel_standing_envs = 0.05
  for axis, rng in TWIST_RANGES.items():
    setattr(twist.ranges, axis, rng)
  twist.ranges.heading = None  # mjlab refuses a heading range with heading off
  cfg.curriculum.pop("command_vel", None)  # would overwrite the ranges mid-run


def apply_heading_commands(cfg: Any) -> None:
  """Turn heading control back on with mjlab's velocity-task defaults: 30 %
  of commands steer ``wz`` to a world heading instead of a random yaw rate.
  Apply after ``apply_twist_commands``."""
  twist: Any = cfg.commands["twist"]
  twist.heading_command = True
  twist.rel_heading_envs = 0.3
  twist.heading_control_stiffness = 0.5
  twist.ranges.heading = (-math.pi, math.pi)


def apply_deploy_observations(cfg: Any) -> None:
  """Drop ``base_lin_vel`` from the actor group: the reference actor never
  sees it and no onboard estimate matches the simulator's.  The critic keeps
  its privileged view."""
  if cfg.observations["actor"].terms.pop("base_lin_vel", None) is None:
    raise ValueError("the actor group has no base_lin_vel term to drop")


def apply_rewards(cfg: Any, stack: str = "ame2_rebalanced") -> None:
  """``mjlab`` keeps the base task's rewards; ``ame2`` installs the reference
  stack, its contact sensor and guard terminations; ``ame2_rebalanced`` adds
  the walk/turn rebalance and the extra AME-2 regularisers (the default)."""
  if stack not in REWARD_STACKS:
    raise ValueError(f"stack must be one of {REWARD_STACKS}, got {stack!r}")
  if stack == "mjlab":
    return
  apply_ame2_rewards(cfg)
  if stack == "ame2_rebalanced":
    rebalance(cfg.rewards)


def apply_patch_spawn(
  cfg: Any,
  pose_range: dict[str, tuple[float, float]] | None = None,
  yaw_choices: tuple[float, ...] = LATTICE_YAWS,
) -> None:
  """Replace ``reset_base`` with the flat-patch spawn (same slot, same order).

  In play mjlab appends ``randomize_terrain`` after the base reset, which
  moves the origin after the robot is placed; the draw is put first so the
  robot lands on the tile it was assigned."""
  if "reset_base" not in cfg.events:
    raise ValueError("no reset_base event to replace")
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_root_state_on_patches,
    mode="reset",
    params={
      "patch_name": "spawn",
      "pose_range": dict(pose_range or SPAWN_POSE),
      "velocity_range": {},
      "yaw_choices": tuple(yaw_choices),
    },
  )
  if "randomize_terrain" in cfg.events:
    events = {"randomize_terrain": cfg.events.pop("randomize_terrain")}
    events.update(cfg.events)
    cfg.events = events


def apply_terrain_floor(
  cfg: Any,
  floor: int,
  command_name: str = "twist",
  rule: str = "endpoint",
  spawn_max: int | None = None,
  track_ratio: float = 0.6,
  hold_iters: int = 0,
  steps_per_iter: int = 24,
) -> None:
  """Spawn on rows ``floor`` .. ``spawn_max`` and never demote below ``floor``;
  ``rule`` / ``track_ratio`` as in :class:`ame.mdp.terrain_levels_floor`;
  ``hold_iters`` freezes the levels for the first iterations."""
  cfg.scene.terrain.max_init_terrain_level = floor if spawn_max is None else spawn_max
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=terrain_levels_floor,
    params={
      "command_name": command_name,
      "floor": floor,
      "rule": rule,
      "track_ratio": track_ratio,
      "hold_steps": hold_iters * steps_per_iter,
    },
  )


def add_tracking_metrics(cfg: Any, command_name: str = "twist") -> None:
  """Log pelvis tracking error, speed, progress along the command and
  commanded speed as episode means."""
  cfg.metrics["pelvis_vel_error"] = MetricsTermCfg(
    func=pelvis_vel_error, params={"command_name": command_name}
  )
  cfg.metrics["pelvis_yaw_error"] = MetricsTermCfg(
    func=pelvis_yaw_error, params={"command_name": command_name}
  )
  cfg.metrics["pelvis_speed"] = MetricsTermCfg(func=pelvis_speed)
  cfg.metrics["pelvis_progress"] = MetricsTermCfg(
    func=pelvis_progress, params={"command_name": command_name}
  )
  cfg.metrics["command_speed"] = MetricsTermCfg(
    func=command_speed, params={"command_name": command_name}
  )


def ame_rough_env_cfg(
  play: bool = False,
  rewards: str = "ame2_rebalanced",
  scan_noise: bool | None = None,
  terrain_proportions: dict[str, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """The from-scratch AME task.  ``scan_noise`` defaults to on unless
  ``play``; ``False`` is the clean half of the two-stage recipe."""
  cfg = unitree_g1_rough_env_cfg(play=play)
  if not play:
    # mjlab cfgs ship num_envs = 1; a bare `train` would collapse in ~100
    # iterations on 24 samples.  1024 is sized for a 16 GB card: the
    # simulator alone takes ~1 GB + ~5.7 MB per env on this terrain board
    # (2048 envs runs out of memory on 16 GB).  See rl_cfg.NUM_MINI_BATCHES.
    cfg.scene.num_envs = NUM_ENVS
  apply_terrain(cfg, ame_terrains_cfg(terrain_proportions), play)
  apply_scan(cfg, noise=(not play) if scan_noise is None else scan_noise)
  apply_twist_commands(cfg)
  apply_rewards(cfg, rewards)
  return cfg


def ame_stage_env_cfg(stage: int, play: bool = False) -> ManagerBasedRlEnvCfg:
  """SII-FUSC's two-stage recipe: stage 1 is perfect perception and no
  disturbance on a 40 %-gaps board; stage 2 the specialist board under noise
  with a fixed spawn pose and stronger placement penalties."""
  clean = stage == 1
  cfg = ame_rough_env_cfg(play=play, scan_noise=not clean and not play)
  apply_terrain(cfg, stage_terrains_cfg(stage), play)
  if clean:
    for term in cfg.observations["actor"].terms.values():
      term.noise = None
    cfg.events.pop("push_robot", None)
    cfg.events.pop("base_com", None)
  else:
    reset = cfg.events["reset_base"]
    pose = dict(reset.params["pose_range"])
    pose.update({"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})
    reset.params["pose_range"] = pose
  for name, (w1, w2) in STAGE_REWARDS.items():
    if name in cfg.rewards:
      cfg.rewards[name].weight = w1 if clean else w2
  return cfg


def ame_gap_env_cfg(
  play: bool = False,
  rule: str = GAP_CURRICULUM["rule"],
  heading_command: bool = False,
) -> ManagerBasedRlEnvCfg:
  """The plain AME policy on the gap board under mjlab's G1 rewards.

  Uniform 0.40 m stones whose gap is the difficulty (row 0 is a floor),
  on-stone spawns with a lattice-aligned heading, the tracking-gated
  curriculum, mjlab's reward set and domain randomisation, and the deployment
  observation set in the actor (no ``base_lin_vel``).  ``rule`` overrides
  the curriculum's promotion rule (``"progress"`` for the direction-aware
  one); ``heading_command`` turns heading control on
  (``apply_heading_commands``)."""
  cfg = ame_rough_env_cfg(play=play, rewards="mjlab")
  if heading_command:
    apply_heading_commands(cfg)
  apply_terrain(cfg, gap_board_cfg(), play)
  apply_deploy_observations(cfg)
  apply_patch_spawn(cfg)
  add_tracking_metrics(cfg)
  if not play:
    apply_terrain_floor(cfg, command_name="twist", **{**GAP_CURRICULUM, "rule": rule})
  return cfg


__all__ = [
  "GAP_CURRICULUM",
  "NUM_ENVS",
  "REWARD_STACKS",
  "STAGE_REWARDS",
  "TWIST_RANGES",
  "add_tracking_metrics",
  "ame_gap_env_cfg",
  "ame_rough_env_cfg",
  "ame_stage_env_cfg",
  "apply_deploy_observations",
  "apply_heading_commands",
  "apply_patch_spawn",
  "apply_rewards",
  "apply_scan",
  "apply_terrain",
  "apply_terrain_floor",
  "apply_twist_commands",
]
