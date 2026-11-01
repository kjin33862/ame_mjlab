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
| gap_progress_heading_1024 | world-cmd | run/gap_progress_heading_1024 | Ame-Mjlab-Gap-ProgressHeading-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-ProgressHeading-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_progress_heading_1024 --agent.logger tensorboard` | |
| gap_progress_heading_sgs_1024 | sgs | run/gap_progress_heading_sgs_1024 | Ame-Mjlab-Gap-ProgressHeading-SGS-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-ProgressHeading-SGS-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_progress_heading_sgs_1024 --agent.logger tensorboard` | |

gap_1024 and gap_1024_s43 have no tag and no recorded commit; the branch is inferred from the task (the gap task
first appeared in `49a85ca`). Their commands are reconstructed from `checkpoints/<run>/params/agent.yaml`.

## gap_progress_heading_sgs_1024: Success-Guided Sampling

Same as gap_progress_heading_1024 except how a terrain level is chosen. SGS (arXiv 2610.12465, Sec. 3.2
and App. B.2) replaces the promotion ladder (`ame.mdp.sgs.terrain_levels_sgs`, `apply_sgs`):

- A configuration is (terrain family, level). An env's family (its column) never changes; with
  `curriculum=True` mjlab builds one column per family (`num_cols=20` is ignored), so SGS only picks the row.
- Success = the progress rule's `move_up` for the episode, from the same code (`terrain_levels_floor._judge`):
  end more than 4 m from the spawn, or time out with `Σ pelvis_progress ≥ 0.6 · Σ command_speed`.
  Everything else is failure. Quirk: an episode that reaches 4 m and then falls (at any time) is a success.
- Per configuration a ring buffer of the last H outcomes; p = mean of the filled entries, empty = 0.
  At each episode end the outcome is recorded, then the next level is drawn from the family's levels:
  `w = max((p+eps)^(κt) (1-p+eps)^(κ(1-t)), eps)`, `prob = softmax(log(w+eps) / max(T, 1))`.
- Hyperparameters: H = 100, t = 0.66, κ = 5, T = 2, eps = 1e-8 (`SGS_PARAMS`).
- No promotion or demotion. The first curriculum call (construction-pose reset) records nothing.
  Initial rows 0-2 as before. With every buffer empty, p = 0 everywhere and sampling is uniform over
  rows 0-9 until buffers fill.
- Stairs: `pyramid_stairs` spawns on the summit (stairs down); `pyramid_stairs_inv` spawns in the pit
  (stairs up). Neither has spawn patches, so the spawn is always the tile centre.
- Logs, under `Curriculum/terrain_levels/`: the heading run's `mean`, `max`, `<family>` unchanged, plus
  `<family>/p_<l>`, `<family>/prob_<l>` and `<family>/fill_min`.
