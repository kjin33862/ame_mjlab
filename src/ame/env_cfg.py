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
* ``apply_rewards`` — mjlab's stack, the AME-2 stack, or the rebalanced AME-2
  stack the registered tasks train on (:mod:`ame.reward_cfg`).

Each mutator is independent, so they compose onto other tasks; the factories
below are the from-scratch tasks.
"""

from __future__ import annotations

from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from ame.reward_cfg import apply_ame2_rewards, rebalance
from ame.scan import apply_scan
from ame.terrains import ame_terrains_cfg, play_board, stage_terrains_cfg

REWARD_STACKS = ("mjlab", "ame2", "ame2_rebalanced")

NUM_ENVS = 1024
"""Default training env count (fits a 16 GB GPU; ~9.6 GB measured)."""

TWIST_RANGES = {
  "lin_vel_x": (-0.3, 1.5),
  "lin_vel_y": (-0.5, 0.5),
  "ang_vel_z": (-1.0, 1.0),
}
"""A small reverse band for gate manoeuvring; ``vy`` opened for the twist box."""

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


__all__ = [
  "NUM_ENVS",
  "REWARD_STACKS",
  "STAGE_REWARDS",
  "TWIST_RANGES",
  "ame_rough_env_cfg",
  "ame_stage_env_cfg",
  "apply_deploy_observations",
  "apply_rewards",
  "apply_scan",
  "apply_terrain",
  "apply_twist_commands",
]
