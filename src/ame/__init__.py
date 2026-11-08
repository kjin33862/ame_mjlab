"""AME perceptive locomotion for the Unitree G1, ported to mjlab.

The recipe — terrain families, the elevation scan, the AME-2 reward stack and
guards, the terrain-attention encoder and the two-stage schedule — is a set of
``apply_*`` mutators over mjlab's G1 rough velocity task, so pieces can be
reused on other tasks.  Importing the package (mjlab does it through the
``mjlab.tasks`` entry point) registers the from-scratch tasks:

* ``Ame-Velocity-Rough-G1`` — dense AME encoder, rebalanced AME-2 rewards;
* ``Delta-Velocity-Rough-G1`` — the same task with the DELTA encoder;
* ``Ame-Stage1-Velocity-Rough-G1`` / ``Ame-Stage2-Velocity-Rough-G1`` —
  SII-FUSC's two-stage schedule (clean perception, then noise and DR);
* ``Ame-Mjlab-Gap-Velocity-Rough-G1`` — the dense AME encoder on the gap
  board under mjlab's G1 rewards, with the deployment actor observations;
* ``Ame-Mjlab-Gap-Progress-Velocity-Rough-G1`` — the same task promoted on
  progress along the command instead of speed (``rule="progress"``);
* ``Ame-Mjlab-Gap-ProgressHeading-Velocity-Rough-G1`` — the progress task
  with mjlab's heading commands back on (30 % of commands steer to a world
  heading);
* ``Ame-Mjlab-Gap-ProgressHeading-SGS-Velocity-Rough-G1`` — the heading task
  with Success-Guided Sampling of terrain levels in place of the promotion
  ladder (``ame.mdp.sgs``);
* ``Ame-Mjlab-Gap-ProgressHeading-AlphaNIST-Velocity-Rough-G1`` — the SGS task
  with AlphaNIST's sampling parameters (``ALPHANIST_SGS_PARAMS``: weight
  ``sqrt(p (1 - p))``, normalized);
* ``Ame-Mjlab-Gap-Edge-SGS-Velocity-Rough-G1`` — the SGS task with a centre
  spawn (within 0.5 m of the tile centre, uniform heading) and first passage
  to the tile edge as SGS's success (``apply_centre_spawn``,
  ``apply_edge_success``, ``EDGE_DISTANCE``);
* ``Ame-Mjlab-Gap-Edge-SGS-AngStd-Velocity-Rough-G1`` — the edge SGS task
  with the ``track_angular_velocity`` reward's std at ``sqrt(0.25)``
  (``ang_vel_std``);
* ``Ame-Mjlab-Gap-Edge-SGS-AngW4-Velocity-Rough-G1`` — the edge SGS task
  with the ``track_angular_velocity`` reward's weight at 4.0 instead of 2.0
  (``ang_vel_weight``; std stays mjlab's ``sqrt(0.5)``);
* ``Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Velocity-Rough-G1`` — the AngW4 task with
  the yaw term made yaw-rate only: ``track_ang_vel_z_world_exp`` on the
  pelvis (world frame, no roll / pitch in the error), std ``sqrt(0.5)``,
  weight 4.0 (``ang_vel_z_only``).

Train / play with mjlab's own CLIs::

    uv run train Ame-Velocity-Rough-G1
    uv run play  Ame-Velocity-Rough-G1 --checkpoint <ckpt>
"""

from __future__ import annotations

import math

from mjlab.tasks.registry import register_mjlab_task

from ame.env_cfg import (
  EDGE_DISTANCE,
  ame_gap_env_cfg,
  ame_rough_env_cfg,
  ame_stage_env_cfg,
)
from ame.mdp.sgs import ALPHANIST_SGS_PARAMS
from ame.rl_cfg import ame_symmetric_ppo_runner_cfg, delta_ppo_runner_cfg

register_mjlab_task(
  task_id="Ame-Velocity-Rough-G1",
  env_cfg=ame_rough_env_cfg(),
  play_env_cfg=ame_rough_env_cfg(play=True),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Delta-Velocity-Rough-G1",
  env_cfg=ame_rough_env_cfg(),
  play_env_cfg=ame_rough_env_cfg(play=True),
  rl_cfg=delta_ppo_runner_cfg(),
)

for _stage in (1, 2):
  register_mjlab_task(
    task_id=f"Ame-Stage{_stage}-Velocity-Rough-G1",
    env_cfg=ame_stage_env_cfg(_stage),
    play_env_cfg=ame_stage_env_cfg(_stage, play=True),
    rl_cfg=ame_symmetric_ppo_runner_cfg(),
  )

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(),
  play_env_cfg=ame_gap_env_cfg(play=True),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-Progress-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(rule="progress"),
  play_env_cfg=ame_gap_env_cfg(play=True, rule="progress"),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-ProgressHeading-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(rule="progress", heading_command=True),
  play_env_cfg=ame_gap_env_cfg(play=True, rule="progress", heading_command=True),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-ProgressHeading-SGS-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(rule="progress", heading_command=True, sgs=True),
  play_env_cfg=ame_gap_env_cfg(
    play=True, rule="progress", heading_command=True, sgs=True
  ),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-ProgressHeading-AlphaNIST-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(
    rule="progress",
    heading_command=True,
    sgs=True,
    sgs_params=ALPHANIST_SGS_PARAMS,
  ),
  play_env_cfg=ame_gap_env_cfg(
    play=True, rule="progress", heading_command=True, sgs=True
  ),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-Edge-SGS-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(
    rule="progress",
    heading_command=True,
    sgs=True,
    centre_spawn=True,
    edge_distance=EDGE_DISTANCE,
  ),
  play_env_cfg=ame_gap_env_cfg(
    play=True, rule="progress", heading_command=True, sgs=True
  ),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-Edge-SGS-AngStd-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(
    rule="progress",
    heading_command=True,
    sgs=True,
    centre_spawn=True,
    edge_distance=EDGE_DISTANCE,
    ang_vel_std=math.sqrt(0.25),
  ),
  play_env_cfg=ame_gap_env_cfg(
    play=True,
    rule="progress",
    heading_command=True,
    sgs=True,
    ang_vel_std=math.sqrt(0.25),
  ),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-Edge-SGS-AngW4-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(
    rule="progress",
    heading_command=True,
    sgs=True,
    centre_spawn=True,
    edge_distance=EDGE_DISTANCE,
    ang_vel_weight=4.0,
  ),
  play_env_cfg=ame_gap_env_cfg(
    play=True,
    rule="progress",
    heading_command=True,
    sgs=True,
    ang_vel_weight=4.0,
  ),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Velocity-Rough-G1",
  env_cfg=ame_gap_env_cfg(
    rule="progress",
    heading_command=True,
    sgs=True,
    centre_spawn=True,
    edge_distance=EDGE_DISTANCE,
    ang_vel_weight=4.0,
    ang_vel_z_only=True,
  ),
  play_env_cfg=ame_gap_env_cfg(
    play=True,
    rule="progress",
    heading_command=True,
    sgs=True,
    ang_vel_weight=4.0,
    ang_vel_z_only=True,
  ),
  rl_cfg=ame_symmetric_ppo_runner_cfg(),
)
