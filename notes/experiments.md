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
| gap_progress_heading_alphanist_1024 | sgs | run/gap_progress_heading_alphanist_1024 | Ame-Mjlab-Gap-ProgressHeading-AlphaNIST-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-ProgressHeading-AlphaNIST-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_progress_heading_alphanist_1024 --agent.logger tensorboard` | |
| gap_edge_sgs_1024 | sgs | run/gap_edge_sgs_1024 | Ame-Mjlab-Gap-Edge-SGS-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Edge-SGS-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_edge_sgs_1024 --agent.logger tensorboard` | |
| gap_edge_sgs_angstd_1024 | sgs | run/gap_edge_sgs_angstd_1024 | Ame-Mjlab-Gap-Edge-SGS-AngStd-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Edge-SGS-AngStd-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_edge_sgs_angstd_1024 --agent.logger tensorboard` | |
| gap_edge_sgs_angw4_1024 | sgs | run/gap_edge_sgs_angw4_1024 | Ame-Mjlab-Gap-Edge-SGS-AngW4-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Edge-SGS-AngW4-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_edge_sgs_angw4_1024 --agent.logger tensorboard` | |

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

## gap_progress_heading_alphanist_1024: AlphaNIST sampling parameters

Same as gap_progress_heading_sgs_1024 except the SGS sampling parameters (`ALPHANIST_SGS_PARAMS`):
t = 0.5, κ = 1, T = 1 (AlphaNIST eq. 4); H = 100 and eps = 1e-8 unchanged. Then
`w = sqrt(p (1-p))` up to eps and `prob = (w+eps) / Σ(w+eps)`, i.e. `sqrt(p (1-p))` normalized: the
weight peaks at p = 0.5 and is much flatter than SGS's (κ = 5, T = 2), so levels far from the target are
drawn more often. Unlike SGS, a level at p = 0 or 1 (including an empty buffer) does not drop to the eps
floor: its weight is (eps)^0.5 = 1e-4 (exponent κt = 0.5), so "up to eps" is up to sqrt(eps) there, about
1e-4 of a mid-range level's weight. A family whose levels are all at 0 or 1 is sampled uniformly.

## gap_edge_sgs_1024: centre spawn, first passage to the tile edge

Same as gap_progress_heading_sgs_1024 (rewards, commands, SGS H = 100, t = 0.66, κ = 5, T = 2, eps = 1e-8,
board, PPO) except the spawn and SGS's success signal:

- Spawn (`reset_root_state_at_centre`, `apply_centre_spawn`): a uniform point in a 0.5 m disc around the tile
  centre, heading uniform in (-π, π) on every family. On the stone lattice, the stone nearest that point
  (±3 cm, as the patch spawn). Elsewhere the ground is the highest terrain point within ±0.14 m of the spawn
  (ray-cast once at construction on a 2 cm grid), plus 1-5 cm. The env origin stays the tile centre. Play
  keeps the SGS task's play config (patch spawn): the play board is not a curriculum board.
- Success (`terrain_levels_sgs_edge`, `apply_edge_success`): the pelvis's Chebyshev distance
  max(|dx|, |dy|) from the tile centre reached D = 3.75 m at any step of the episode (`edge_distance` metric,
  a running max, terminal step included). A fall afterwards still counts; nothing else does (the progress /
  tracking / survival branches are gone from this task).
- D = 3.75 from the offline check on the success audit's centre-spawn rollouts (`notes/success_audit_cheb.py`,
  `logs/eval_results/success_audit/board/cheb_check.csv`): balanced accuracy against "crossed before falling",
  mean over the 5 families 0.987 (worst 0.948, stairs up) vs 0.947 at 3.5 and 0.980 at 4.0; the first passage
  is on the own tile in 100 % of successes (at 4.0 it is on the neighbour tile every time: 4.0 is the edge).
- Logs add `Curriculum/terrain_levels/<family>/edge_frac` (share of the family's last 500 recorded episodes
  that reached the edge) and `Episode_Metrics/edge_distance` (mean farthest excursion of ended episodes).

## gap_edge_sgs_angstd_1024: tighter yaw-rate tracking kernel

Same as gap_edge_sgs_1024 (spawn, edge success D = 3.75 m, SGS H = 100, t = 0.66, κ = 5, T = 2, eps = 1e-8,
commands, board, PPO) except the `track_angular_velocity` reward's std: sqrt(0.25) = 0.5 instead of mjlab's
sqrt(0.5) ≈ 0.707 (`ame_gap_env_cfg(ang_vel_std=...)`; weight 2.0 unchanged, every other reward term
identical). The reward is exp(-e / std²) with e = (ω_z,cmd − ω_z)² + ω_x² + ω_y² (body frame, so roll and
pitch rates count too), so std² halves from 0.5 to 0.25: an error that earned exp(-1) before now earns exp(-2).
The play config gets the same std.

## gap_edge_sgs_angw4_1024: double weight on yaw-rate tracking

Same as gap_edge_sgs_1024 (spawn, edge success D = 3.75 m, SGS H = 100, t = 0.66, κ = 5, T = 2, eps = 1e-8,
commands, board, PPO) except the `track_angular_velocity` reward's weight: 4.0 instead of mjlab's 2.0
(`ame_gap_env_cfg(ang_vel_weight=...)`; std stays sqrt(0.5), every other reward term identical). Motivation: in
goal-mode play on pyramid_stairs_inv d0.5, gap_edge_sgs_1024 (model_9999) carries a large yaw-rate error
(mean (ω_z,cmd − ω_z)² ≈ 1.0 over 192 robots, 97 % of it on steps with |ω_z,cmd| ≥ 0.3; uncommitted
analysis on the eval branch). Unlike the AngStd variant this keeps the kernel's shape and only doubles the term
against the rest of the reward. The play config gets the same weight.
