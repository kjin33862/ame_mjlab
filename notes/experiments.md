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
| gap_edge_sgs_angw4z_1024 | sgs | run/gap_edge_sgs_angw4z_1024 | Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_edge_sgs_angw4z_1024 --agent.logger tensorboard` | |
| gap_edge_alphanist_angw4z_1024 | sgs | run/gap_edge_alphanist_angw4z_1024 | Ame-Mjlab-Gap-Edge-SGS-AngW4Z-AlphaNIST-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Edge-SGS-AngW4Z-AlphaNIST-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_edge_alphanist_angw4z_1024 --agent.logger tensorboard` | |
| gap_edge_sgs_angw4z_contact_1024 | sgs | run/gap_edge_sgs_angw4z_contact_1024 | Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_edge_sgs_angw4z_contact_1024 --agent.logger tensorboard` | Contact package copied as one unit from Ame-Velocity-Rough-G1 (셩's port): illegal_contact termination on 20 bodies (stricter than the AME paper's torso-only rule), undesired_contacts -1, termination_penalty -50 (REBALANCE, reason not recorded; AME paper and Isaac Lab use 200). link_contact_forces left out. Not retuned for the mjlab reward stack. |
| gap_edge_sgs_angw4z_contact_foothold_1024 | sgs | run/gap_edge_sgs_angw4z_contact_foothold_1024 | Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Foothold-Velocity-Rough-G1 | `uv run train Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Foothold-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.seed 42 --agent.run-name gap_edge_sgs_angw4z_contact_foothold_1024 --agent.logger tensorboard` | Not trained yet. Contact task + BeamDojo-style foothold reward (delta 0.10 m, weight -1) in its own reward group under a double critic (A = 1.0 A1 + 0.25 A2), one-sided foot_clearance / foot_swing_height, illegal_contact unchanged (self-contact still terminates), per-swing gap diagnostics. Decisions from the contact run's eval: notes/report_foothold_impl.md sections 11-13. |

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

## gap_edge_sgs_angw4z_1024: yaw-rate-only tracking kernel at weight 4

Same as gap_edge_sgs_angw4_1024 except the `track_angular_velocity` term's function: mjlab's
`track_angular_velocity` (body frame, error (ω_z,cmd − ω_z)² + ω_x² + ω_y²) is replaced by
`ame.mdp.rewards.track_ang_vel_z_world_exp` on the root link (pelvis): exp(−(ω_z,cmd − ω_z,world)² / std²), yaw rate
only, world frame (`ame_gap_env_cfg(ang_vel_z_only=True)`). std stays sqrt(0.5), weight 4.0 as in AngW4; the term
keeps its name, and roll / pitch stay penalised only by the existing `body_ang_vel` (torso xy, −0.05). Every other
reward term, the terminations and the rest of the config are identical to gap_edge_sgs_1024. Compared with AngW4 it
isolates the roll / pitch coupling: in goal-mode play on pyramid_stairs_inv d0.5, gap_edge_sgs_1024 (model_9999)
has mean exp(−(ω_x² + ω_y²)/0.5) = 0.73 (0.80 on flat), the factor by which the coupled kernel scales the yaw reward
and its gradient. The play config gets the same term.

## gap_edge_alphanist_angw4z_1024: AngW4Z with AlphaNIST sampling parameters

Same as gap_edge_sgs_angw4z_1024 except the SGS sampling parameters: `ALPHANIST_SGS_PARAMS` (t = 0.5, κ = 1, T = 1,
AlphaNIST eq. 4; H = 100 and eps = 1e-8 unchanged) instead of `SGS_PARAMS` (t = 0.66, κ = 5, T = 2), as
gap_progress_heading_alphanist_1024 is to gap_progress_heading_sgs_1024. Edge success, centre spawn, the yaw-rate-only
kernel at weight 4.0 and the rest of the config are identical. SGS is training only, so the play config is AngW4Z's.

## gap_edge_sgs_angw4z_contact_foothold_1024: Contact task + foothold reward, double critic, one-sided clearance

Same as gap_edge_sgs_angw4z_contact_1024 (1024 envs, 10k iterations, seed 42; spawn, edge success, SGS, commands,
board, contact package, PPO numbers, symmetry, shared encoder) plus:

- `foothold` reward (`ame.mdp.foothold`), weight -1.0: per foot in contact, the share of 15 sole points (3 x 5 grid
  on the G1 foot capsules) whose sole height above the terrain right below is more than delta = 0.10 m, summed
  over the feet. Read from a new 2 x 15-ray sensor `foot_sole_scan`.
- Double critic (`ame.double_critic.DoubleCriticPPO`, runner cfg `double_critic_ppo_runner_cfg()`): group 2 =
  `foothold`, group 1 = every other term; GAE and batch normalisation per group, A = 1.0 A1 + 0.25 A2, value loss
  L1 + L2.
- One-sided `foot_clearance` / `foot_swing_height` (`ame.mdp.swing`): only a foot below the 0.10 m target is
  penalised; names, weights and params unchanged.
- `swing_cost_by_gap` metric (logging only): per-swing clearance and swing-height costs, two- and one-sided, over
  gaps vs not.

Decisions (from contact model_9999, notes/eval_foothold.py and the stone analysis; numbers in
notes/report_foothold_impl.md sections 11-13):

- delta 0.10: the smallest of 0.02 / 0.05 / 0.10 with touchdown false positives below 2 % on flat, wave and
  random_rough (0.07 %, 0.7 %, 1.2 %; 0.05 gives 3.6 %, 9.3 %, 17.5 %).
- One-sided clearance on: two-sided clearance + swing-height cost is 7.8x higher on gap-crossing swings than on
  other swings (stones levels 1-9).
- illegal_contact kept as is: only 4 of 366 stone contact terminations of the contact run are self-contact.

Differences from BeamDojo (arXiv 2502.10363):

- Relative-height foothold: a sole point is unsupported when it is more than delta above the terrain right below
  it, not when the terrain's global height is below a fixed eps (BeamDojo's terrain is flat-topped over a pit;
  ours has stairs and height fields).
- 3 x 5 sole grid on the G1 foot capsules, and the per-foot fraction instead of the point count.
- Two value heads on one shared critic trunk instead of two critic networks.
- No two-stage training (BeamDojo trains on soft terrain dynamics first, then hard); one stage here, on the gap
  board with SGS.
