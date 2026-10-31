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
  heading).

Train / play with mjlab's own CLIs::

    uv run train Ame-Velocity-Rough-G1
    uv run play  Ame-Velocity-Rough-G1 --checkpoint <ckpt>
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from ame.env_cfg import ame_gap_env_cfg, ame_rough_env_cfg, ame_stage_env_cfg
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
