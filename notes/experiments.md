# Experiments

One row per training run. All runs below: 1024 envs, 10000 iterations, tensorboard logger.
Results are from `logs/eval_results/` (192 robots = 8x8 tiles x 3 seeds, noise off; success = 4 m along the
command): `matrix_plots/matrix_summary.csv` (open loop) and `goal_plots/goal_summary.csv` (goal-steered).
"stairs_inv" = pyramid_stairs_inv (climbing out of a pit), d = difficulty.

| Run | Branch | Git tag | Task | Command | Result |
|---|---|---|---|---|---|
| gap_1024 | main (upstream `49a85ca`) | none | Ame-Mjlab-Gap-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_1024 --agent.logger tensorboard` | stairs_inv open d0 0.40, d0.25 0.02; goal d0 0.99, d0.25 0.02, d≥0.5 ≤0.06. stone_lattice d0.5 0.55. pyramid_stairs (down) ≥0.98. |
| gap_1024_s43 | main (upstream `49a85ca`) | none | Ame-Mjlab-Gap-Velocity-Rough-G1 | same as gap_1024 with `--agent.seed 43 --agent.run-name gap_1024_s43` | stairs_inv open d0 0.92, d0.25 0.02; goal d0 1.00, d0.25 0.80, d≥0.5 ≤0.09. stone_lattice d0.5 0.38. pyramid_stairs ≥0.95. |
| gap_progress_1024 | progress-rule (`cd93ec0`) | none | Ame-Mjlab-Gap-Progress-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Progress-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_progress_1024 --agent.logger tensorboard` | stairs_inv open d0 0.50, d0.25 0.02; goal d0 0.97, d0.25 0.04, d≥0.5 ≤0.07. stone_lattice d0.5 0.24. pyramid_stairs ≥0.98. The progress rule alone did not teach climbing. |
| gap_progress_world_1024 | world-cmd | run/gap_progress_world_1024 | Ame-Mjlab-Gap-ProgressWorld-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-ProgressWorld-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_progress_world_1024 --agent.logger tensorboard` | |

gap_1024 and gap_1024_s43 have no tag and no recorded commit; the branch is inferred from the task (the gap task
first appeared in `49a85ca`). Their commands are reconstructed from `checkpoints/<run>/params/agent.yaml`.
