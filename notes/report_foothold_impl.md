# Foothold reward, double critic, one-sided clearance: implementation report

Branch `sgs`. Sections 1-9 are the implementation (CPU). Sections 11-15 (2026-11-13/14) add the
analysis of the contact run on stones, the foothold evaluation, the decisions baked into the task,
a GPU smoke train and the commit. Nothing is pushed; no training run was started (the 30-iteration
smoke was deleted).

Summary of the decisions (section 13):

- delta = 0.10 m: the only one of 0.02 / 0.05 / 0.10 whose touchdown false-positive share is below
  2 % on flat, wave and random_rough (0.07 %, 0.7 %, 1.2 % pooled; 0.05 gives 3.6 %, 9.3 %, 17.5 %).
- C (one-sided clearance) ON: on stones, the two-sided clearance + swing-height cost per
  gap-crossing swing is 7.8x that of other swings (levels 1-9 pooled; 1.1x at levels 1-4 alone).
- illegal_contact keeps counting self-contact: 4 of 366 contact terminations on the two stone cells
  are robot-robot (1.1 %), far below the 50 % rule.
- Why the contact run collapsed on stones (section 11): a foot drops into the pit and the same-side
  knee hits the stone edge (H3: 361 of 366 terminations are knee/hip -> terrain, a foot was in the pit
  before every knee hit in the rollout check). The root cause is that its curriculum never left
  levels 0-1 on stones (H4): it never trained on gaps of 0.10 m or more, AngW4Z did from about
  iteration 7500. H2 (stuck) and H1 (self-contact) are rejected.

## 1. Plan and file list (written before implementation)

New task `Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Foothold-Velocity-Rough-G1` on top of
`Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Velocity-Rough-G1`. Every switch is a keyword of
`ame_gap_env_cfg` (env side) or a separate runner-cfg factory (PPO side) that defaults off, so the
27 existing tasks build exactly the configs they built before (checked by hashing, section 7).

Files:

- `src/ame/mdp/foothold.py` (new): sole geometry, the sole-grid ray pattern and sensor cfg, the
  unsupported-fraction function shared by the reward and the eval, the `foothold` reward term.
- `src/ame/mdp/swing.py` (new): one-sided `foot_clearance` / `foot_swing_height` terms, the
  per-swing gap / no-gap tracker and the `swing_cost_by_gap` logging-only metric.
- `src/ame/mdp/__init__.py`: export the two modules.
- `src/ame/double_critic.py` (new): `RewardGroups` (splits the env reward into the two groups),
  `MultiRewardRolloutStorage`, `DoubleCriticPPO` (an rsl_rl `PPO` subclass), GAE / advantage helpers.
- `src/ame/env_cfg.py`: `apply_foothold`, `apply_one_sided_clearance`, `apply_swing_diagnostics`,
  and three keywords on `ame_gap_env_cfg` (`foothold`, `one_sided_clearance`, `swing_diagnostics`).
- `src/ame/rl_cfg.py`: `DoubleCriticPpoCfg` and `double_critic_ppo_runner_cfg()` (new factory; the
  existing factories are untouched).
- `src/ame/__init__.py`: register the new task.
- `tests/test_foothold.py` (new): the CPU tests listed in section 7.
- `notes/eval_foothold.py` (new, prepared, not run): per terrain / level evaluation.
- `notes/report_foothold_impl.md`: this report.

Installed `rsl_rl` and `mjlab` are not edited.

## 2. The new task

`Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Foothold-Velocity-Rough-G1` = the Contact task with:

- A on: `foothold=True` (weight -1.0);
- B on: runner cfg `double_critic_ppo_runner_cfg()` (group 2 = `foothold`, weights 1.0 / 0.25);
- delta 0.10 m (`foothold_delta=0.10`, section 13; the module default `FOOTHOLD_DELTA` stays 0.05
  for other callers);
- C on: `one_sided_clearance=True` (section 13; it was off until the evaluation);
- `illegal_contact` unchanged, self-contact still counts (section 13);
- diagnostics on: `swing_diagnostics=True` (logging only).

The play config gets the same env flags. Everything else (rewards, terminations, curriculum,
SGS, spawn, observations, PPO hyperparameters, symmetry, shared encoder) is the Contact task's.

## 3. A: foothold reward

Definition (`ame.mdp.foothold.foothold_unsupported`): for each foot,
`frac = share of the 15 sole points with (z_sole - z_terrain) > delta` (strict), times the foot's
contact flag, summed over the two feet. Range 0 to 2, weight -1.0, so the reward is BeamDojo's
`-sum_i C_i sum_j 1{...}` with the per-foot count replaced by the per-foot fraction. It stays in
`cfg.rewards` (so `Episode_Reward/foothold` is logged) and the double critic takes it out of group 1.

The G1 sole. There is no sole box. Each foot has seven capsules (`*_foot1..7_collision`, radius
0.01 m, axes along x at z = -0.025 m in `*_ankle_roll_link`). Their union's bounding box is
x -0.064 to 0.142 m, y -0.037 to 0.036 m (0.206 m by 0.073 m; right foot 0.072 m wide), and the
sole plane is z = -0.035 m. The `left_foot` / `right_foot` sites are at (0.04, 0, -0.037), 2 mm
below the sole. The 3 x 5 grid lies on capsule axes at the sole plane: y in {-0.018, 0, +0.018}
(the two outer full-length capsules and the centre one), x at five even steps over
-0.044 to 0.123 m (the extent of the y = +-0.018 capsules). A test checks every point against
the compiled g1.xml.

Sensor. The `foot_height_scan` sensor type (`TerrainHeightSensor`, a `RayCastSensor`) does take
a grid pattern, but `GridPatternCfg` has a single spacing and is centred on the frame, so it
cannot place 3 x 5 points inside a 0.17 x 0.04 m area; and `TerrainHeightSensor.heights` is
`frame_z - hit_z`, which measures from the site, not from each sole point. So I reused the plain
`RayCastSensor` (the parent type) as a new sensor, `foot_sole_scan`, with a 20-line custom
pattern (`SolePatternCfg`, duck-typing `generate_rays`):

- frames: the two foot sites; `ray_alignment="yaw"`; rays straight down; terrain geoms only
  (group 0); parent body excluded; max distance 1 m;
- ray origins are the sole points lifted 5 cm (inside the foot), so a ray never starts below the
  ground;
- `z_sole` is computed exactly from the frame pose the sensor stored with the hits (full foot
  rotation), `z_terrain` is the hit z;
- a miss is unsupported; a back-face hit (ray origin inside terrain, so terrain higher than the
  sole) is supported.

The ray xy positions use the yaw-only rotation while `z_sole` uses the full rotation; at 10 deg
pitch the xy offset is about 1.5 mm at the toe.

Contact source: `feet_ground_contact.found > 0`, the same flag mjlab's air-time bookkeeping
reduces (`ContactSensor._update_air_time_tracking`). Foot subtree vs the terrain body, one slot
per foot, no force threshold (any matched MuJoCo contact counts). The term checks once that the
contact sensor's foot order (left, right) matches the sole scan's frames.

delta. The module default is 0.05 m, the lowest stair step on the board; the task uses 0.10 m
(section 13). Because the test is strict, with 0.05 a sole hanging over a stair edge counts from
level 1 (0.067 m step) and with 0.10 from level 4 (0.117 m step; level 3's step is exactly 0.100 m
and does not count). Any delta flags a pit. The eval (section 12) confirmed the risk noted here
before it ran: on `wave` and `random_rough` a sole in contact leaves points 2-5 cm above the
surface at touchdown, so 0.02 and 0.05 fire on ordinary ground.

Timing. mjlab ray-casts once per policy step, at the end of `env.step` (`sim.sense()`), so at
reward time the hits and the frame pose are one policy step (20 ms) old and consistent with each
other. mjlab's own `foot_clearance` / `foot_swing_height` have the same lag. The contact flag is
current.

Compute cost. Rays per env go from 707 (terrain_scan 693 + foot_height_scan 2 x 7) to 737
(+30, +4.2 %), plus one extra ray-kernel launch per policy step. The torch post-processing
(quaternion rotation of 30 points, compare, mean) took 0.9 ms per step for 1024 envs on CPU,
far less on GPU. The diagnostic metric recomputes the same 30-point drop (a second 0.9 ms CPU
equivalent). That could be cached if it ever shows in the profile.

Eval reuse. The reward and `notes/eval_foothold.py` both call `sensor_sole_drop` and
`unsupported_fraction` (and `feet_in_contact` for the gate). The eval adds the sensor to any
gap-board task via `apply_foot_sole_scan`.

## 4. B: double critic

How the reward manager returns two groups. It doesn't need to: `RewardManager.compute` returns
the summed reward, and it also keeps each term's weighted, non-dt-scaled, NaN-cleaned value of the
step in `_step_reward` `[B, num_terms]`. `ame.double_critic.RewardGroups` reads the group-2
columns right after `env.step` and returns `[r_total - r2, r2]` with `r2 = dt * sum(group-2
columns)`, which is exactly the group-2 share of the sum. mjlab is not modified. The cost is that
this relies on the private `_step_reward` attribute. A test pins the behaviour, and the CPU smoke
run checked `r1 + r2 == r_total` and `r2 == foothold * dt` on the real reward manager.

Algorithm (`DoubleCriticPPO`, subclass of rsl_rl's `PPO`):

- the critic is the same `AmeModel` built with `output_dim=2`: two value heads on the shared
  trunk (and on the actor-shared terrain encoder). BeamDojo describes two critics. Two separate
  networks would be a model-class change; the heads already get separate GAE, normalisation and
  losses;
- `MultiRewardRolloutStorage`: rewards, values and returns are `[T, N, 2]`, advantages
  `[T, N, 1]` (the combined one);
- `process_env_step`: splits the reward, then the base method runs unchanged (the time-out
  bootstrap `r += gamma * V * timeout` is applied per group);
- `compute_returns`: GAE per group (rsl_rl's recursion per channel), each group's raw advantage
  normalised by its own batch mean and std (rsl_rl's formula, eps 1e-8), then
  `A = 1.0 * A1n + 0.25 * A2n`. A group whose advantage is identically zero contributes exactly 0;
- `update`: the base `PPO.update` unchanged. Its clipped value loss is the mean over `[B, 2]`,
  i.e. `(L1 + L2) / 2`; the value-loss coefficient is doubled internally, so the loss is
  `1.0 * (L1 + L2)`. The logged `Loss/value` is `(L1 + L2) / 2`. Extra logs: `adv_std_g1/g2` (raw),
  `return_mean_g1/g2`, `explained_var_g1/g2`;
- symmetry augmentation works unchanged (it repeats `values` / `returns` / `advantages` rows,
  whatever their width). RND and `normalize_advantage_per_mini_batch=True` are refused;
- `construct_algorithm` is rsl_rl's, with critic output 2, the new storage, and the reward
  split bound to the env.

Runner: no change was needed. rsl_rl's runner resolves the algorithm from
`cfg.algorithm.class_name` (`ame.double_critic:DoubleCriticPPO`) and already passes each step's
reward to `alg.process_env_step`; `construct_algorithm` receives the env. `train`, `play` and the
eval use `MjlabOnPolicyRunner` as before.

Checkpoints: the critic's last layer is 2 x 128, so a double-critic checkpoint's critic does not
load into a single-critic run and vice versa (resume across them fails on strict load). Actor-only
loads (play export, eval) work both ways.

Caveat from the method itself: per-group normalisation scales group 2 to unit variance whatever
its size. On a batch where the foothold reward is almost always 0 (early flat levels), A2n is
mostly the value head's own TD noise blown up to unit scale, at weight 0.25. That is how BeamDojo
defines it. `adv_std_g2` and `explained_var_g2` show when this happens.

## 5. C: one-sided clearance (flag `one_sided_clearance`; default off, on in the new task)

- `foot_clearance`: `max(0, 0.10 - h) * |v_xy|` per foot instead of `|h - 0.10| * |v_xy|`;
- `foot_swing_height`: `max(0, 1 - peak / 0.10)^2` at touchdown instead of `(peak / 0.10 - 1)^2`
  (subclass of mjlab's term, same peak bookkeeping and `Metrics/peak_height_mean` log).

Names, weights (-2.0, -0.25) and params are unchanged; only the two functions are swapped
(tested).

## 6. Diagnostics (logging only, `swing_diagnostics`)

Metric term `swing_cost_by_gap` (`ame.mdp.swing`). Per foot it tracks a swing from liftoff to
touchdown (first contact, as mjlab's swing term). A swing "crossed a gap" if at some air step the
terrain under the sole centre was more than 0.5 m below it, or there was no hit. Stairs are at
most 0.20 m and the pits 2 m, so stairs never count. At touchdown it writes per-swing values to
`extras["log"]`; rsl_rl concatenates them over the iteration, so each key is a per-swing mean:

- `Swing/clearance_{gap,nogap}`: per-step `foot_clearance` cost summed over the swing, two-sided;
- `Swing/clearance_1s_{gap,nogap}`: the same, one-sided;
- `Swing/height_{gap,nogap}`, `Swing/height_1s_{gap,nogap}`: `foot_swing_height` cost at touchdown;
- `Swing/gap_share`: share of swings that crossed a gap.

The costs are raw (unweighted, per policy step). Reward units = cost x weight x 0.02. The swing
from the spawn drop is not reported. The metric's episode value is gap-crossing landings per step.

## 7. Verification (all on CPU)

- Config hashes (sha256 of the address-stripped repr of env, play and rl configs, plus the
  runner class) for all 27 tasks before and after: 27 / 27 unchanged; one new task.
- Config diff vs the Contact task (test, train and play): env differs only by the
  `foot_sole_scan` sensor, the `foothold` reward (delta 0.10), the `swing_cost_by_gap` metric and,
  since the decisions of section 13, the functions of `foot_clearance` and `foot_swing_height`
  (one-sided; names, weights and params unchanged). `illegal_contact` is identical. rl differs only
  in `algorithm.class_name` and the two new algorithm fields. The PPO numbers are identical.
- Rechecked on 2026-11-14 after the decisions: 27 / 27 existing task hashes still unchanged; tests
  105 passed, 1 skipped; ruff clean on src, tests and the new notes scripts (`tests/test_edge.py`
  was already unformatted before this work and is left alone).
- Test suite: 105 passed, 1 skipped (baseline 85 passed, 1 skipped; +20 in
  `tests/test_foothold.py`). ruff check and format are clean on the new and changed files.
- New tests:
  - sole points lie on the capsule bottoms of the compiled g1.xml, inside the box, both feet;
    the site offset is as documented;
  - the pattern's ray origins sit 5.2 cm above the sole;
  - flat foot on a stone: fraction 0 at delta 0.02 / 0.05 / 0.10, reward 0 (two yaws);
  - foot half over a pit (real MuJoCo rays): the 5 columns give 0.4 or 0.6 depending on which
    side of the edge the centre column falls (1 mm either way), mean 0.5;
  - foot in the air: fraction 1 (ground far below), reward 0 (no contact);
  - miss = unsupported, back face = supported; 10 deg toe-down pitch lifts only the heel column
    (3/15 at delta 0.01, 0 at 0.02); foot-order mismatch refused;
  - double critic, group-2 reward 0 and V2 = 0: combined advantage equals rsl_rl's single-critic
    advantage exactly (w1 = 1) and 2x it (w1 = 2); returns of head 1 equal rsl_rl's;
  - per-group normalisation; reward split by term name (sum preserved, unknown term refused);
    one `update()` on a toy model trains both value heads (an in-memory unit test, nothing saved);
  - one-sided costs; the C flag swaps exactly two functions; all flags default off; the swing
    tracker splits gap and no-gap swings and skips the spawn drop.
- CPU smoke of the real env (4 envs, 8 policy steps of zero action, no update, nothing saved):
  `foot_sole_scan` is 2 x 15 rays, the critic outputs 2 values, the reward split matches the
  manager, feet resting on flat ground read a drop of -5 mm to +0.4 mm (foothold 0), and
  `compute_returns` fills the `[8, 4, 2]` returns.
- The train script's YAML dump of the new env and agent configs works.

## 8. Eval script (run on 2026-11-14, section 12)

`notes/eval_foothold.py --checkpoint <model.pt> [--task <gap-board task>] --out <dir>`

- Layout: the gap board's curriculum layout, 10 levels x 5 families (difficulty = level / 9).
  Tiles are pinned (curriculum removed), 8 robots per tile by default, the task's centre spawn
  with uniform heading, forward command vx = 0.8 m/s (no heading control, no standing envs). No
  pushes, no observation or height-map noise, mean actions, one 20 s episode per robot; steps after
  a robot's first termination are dropped. Only the actor is loaded, so single- and double-critic
  checkpoints both work.
- Per (family, level) and per delta in {0.02, 0.05, 0.10}:
  - unsupported share (mean over in-contact foot-steps, what the reward pays);
  - unsupported share at touchdown;
  - gap-landing rate (touchdowns with any sole point over a gap);
  - the swing-cost diagnostic over gaps vs not, two- and one-sided;
  - swing counts, touchdowns, fall rate.
- Output: `foothold_eval.json`.
- `--terrain_table` writes the terrain geometry (below).
- Fixed on 2026-11-14 after the first run: robots walk off their 8 m tile (0.8 m/s for 20 s is up
  to 16 m), so the first run mixed neighbouring tiles into each cell (for example 2.8 % "gap
  landings" on the flat level-0 floor). A robot now stops counting once its pelvis is more than
  `--tile_half` 3.75 m (Chebyshev) from its tile centre, and `left_tile_rate` is reported. Only
  the gated run is used in this report.

## 9. Terrain per difficulty (gap board, measured on the compiled board)

Levels 0 to 9, difficulty d = level / 9. Measured = vertical rays on a 1 cm (stairs) / 5 mm
(stones) profile; all match the config formulas within the sampling.

- Stairs (`pyramid_stairs`: down from the summit platform; `pyramid_stairs_inv`: up from the
  centre), step width 0.30 m, step height 0.05 + 0.15 d:
  0.050, 0.067, 0.083, 0.100, 0.117, 0.133, 0.150, 0.167, 0.183, 0.200 m.
- Stone lattice (`stepping_stones`): 0.40 m flush stones, gap 0.45 d:
  0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45 m (level 0 is a continuous floor).
  Pit depth 2.0 m at every level with a gap (stone tops z = 0, floor top z = -2.0).
- `wave` (height field), peak-to-trough: 0, 0.04, 0.08, 0.12, 0.16, 0.22, 0.26, 0.30, 0.34,
  0.40 m.
- `random_rough` (height field): 0.08 m range (std 0.02 m) at every level. The noise does not
  grow with difficulty in this mjlab version.

So with delta = 0.05 a sole over a stair edge counts from level 1 on; with delta = 0.10, from
level 4 on (0.117 m); any delta flags a pit.

## 10. Open points

- Two heads on one trunk vs two critic networks (section 4).
- The SGS level freeze (section 11) is not addressed by this task. If the foothold run also stalls
  at stones levels 1-2, the sampler, not the reward, is the next thing to change.
- The double critic's group 2 is near zero early in training (section 14: explained_var_g2 about
  -6); watch `adv_std_g2` / `explained_var_g2` in the real run.

## 11. Why the contact run collapsed on stones (Part 1)

Runs: contact = `checkpoints/gap_edge_sgs_angw4z_contact_1024/model_9999.pt`, AngW4Z =
`checkpoints/gap_edge_sgs_angw4z_1024/model_9999.pt` (both in `~/ame_mjlab`). Cells: the existing
strict matrix runs (`scripts/evaluate.py`, open mode, vx = 1, noise off, 8 x 8 tiles x seeds 0-2 =
192 robots per cell, a non-foot contact > 1 N ends the robot), read from
`logs/eval_results/*_m9999_{stone_lattice..d0.25,stepping_stones..d0.5}-s{0,1,2}/per_env.csv`.
Nothing was rerun for these counts.

Outcomes per robot (192 per cell):

- stone_lattice d0.25 (gap 0.1125 m, flush 0.40 m stones over a 2 m pit):
  - contact: success 8 (4.2 %), contact-terminated 182 (94.8 %), fall 2 (1.0 %, orientation),
    stuck 0;
  - AngW4Z: success 141 (73.4 %), contact-terminated 51 (26.6 %), fall 0, stuck 0.
- stepping_stones d0.5 (AME-mix stones, 2 m platform, pit 2.2 m):
  - contact: success 7 (3.6 %), contact-terminated 184 (95.8 %), fall 1 (0.5 %), stuck 0;
  - AngW4Z: success 79 (41.1 %), contact-terminated 112 (58.3 %), fall 1 (0.5 %, height), stuck 0.

First non-foot contact of the contact-terminated robots (body -> what it touched):

- contact, stone_lattice d0.25 (182): knee -> terrain 175 (right 104, left 71), hip_yaw -> terrain 5,
  robot's own body 2 (hip_yaw -> own elbow / shoulder).
- contact, stepping_stones d0.5 (184): knee -> terrain 151, hip_yaw -> terrain 30, pelvis -> terrain
  1, own body 2.
- AngW4Z, stone_lattice d0.25 (51): knee -> terrain 49, own body 2.
- AngW4Z, stepping_stones d0.5 (112): knee -> terrain 98, hip_yaw -> terrain 3, own body 11 (mostly
  knee -> the other leg's ankle).
- Self-contact share: contact 4 of 366 (1.1 %), AngW4Z 13 of 163 (8 %).
- When: contact robots end at a median 2.30 s (lattice) / 3.21 s (stones) after covering a median
  1.47 / 1.67 m; AngW4Z robots that fail do so later (3.56 / 3.53 s, 2.74 / 2.23 m).

A knee or hip_yaw link touching the top of flush stones means that leg is down in the pit. To check
the order of events I ran a short rollout (`notes/stone_knee_probe.py`, GPU, one job; same eval env,
seed 0, 64 robots, 8 s, but the fall-only rule so a robot keeps going after the first touch; it
records the state one step before the first non-foot contact):

- contact, stone_lattice d0.25: 61 of 64 robots touch; 60 are knee touches. In all 60 a foot had
  dropped below the stone tops (sole more than 5 cm under the top plane) before the touch; in 56 it
  was the same-side foot at the step before. The knee was a median 0.14 m above the stone tops one
  step before. 59 of 64 robots then fell.
- contact, stepping_stones d0.5: 62 of 64 touch (56 knee); a foot in the pit before 55 of the 56
  knee touches, the same-side foot in 48; 56 fell.
- AngW4Z, stone_lattice d0.25: 27 of 64 touch (all knee, all after a foot dropped); only 2 fell.
  Of the 37 robots that never touched, 36 also dropped a foot into a gap at some point and
  recovered without a knee contact.
- AngW4Z, stepping_stones d0.5: 52 of 64 touch (51 knee, all after a foot dropped); 14 fell.
- Raw: `notes/stone_knee/*.json`.

So both policies put feet into the gaps. The contact policy's knee then lands on the stone edge and
it falls; AngW4Z mostly pulls the foot back out.

Curriculum (tfevents, mean over iterations it +- 100). The gap board has one stone family,
`stepping_stones`, which is the lattice the evaluation calls `stone_lattice` (gap = 0.05 m per level).
The AME-mix `stepping_stones` tile is not on the training board, so it has no curriculum curve.

- Mean stones level, contact: 0.34 (1k), 0.45 (2k), 0.47 (3k), 0.47 (5k), 0.46 (7k), 0.50 (8k),
  0.47 (9k), 0.48 (10k).
- Mean stones level, AngW4Z: 0.27 (1k), 0.44 (2k), 0.61 (3k), 0.49 (5k), 0.51 (7k), 0.89 (8k),
  0.95 (9k), 1.28 (10k).
- Level 2 (0.10 m gaps), SGS success p_2 / sampling prob_2: contact 0 until 5k, then 0.03-0.06 with
  prob_2 at most 0.024 for the rest of the run; AngW4Z p_2 = 0.01 at 7k, 0.53 at 8k, 0.77 at 10k,
  prob_2 0.29-0.36 from 8k on.
- Level 3 (0.15 m): contact p_3 = 0 throughout (prob_3 4e-4); AngW4Z p_3 = 0.03 at 9k, 0.37 at 10k,
  prob_3 0.21 at 10k.
- The evaluation gap at d0.25 is 0.1125 m, between levels 2 and 3. AngW4Z trained there for its last
  ~2.5k iterations; the contact run essentially never did (its stones episodes were levels 0 and 1:
  a floor and 5 cm gaps).

Ranking:

1. H3, knee hits a stone after a foot drops into a gap: confirmed as the failure mechanism. 361 of
   366 (98.6 %) of the contact run's stone terminations are knee or hip_yaw links hitting terrain
   (knee alone 326, 89 %), and the rollout shows
   the foot in the pit first in 60 of 60 (lattice) and 55 of 56 (stones) knee touches.
2. H4, the curriculum kept stones low: confirmed as the reason the contact run fails this way so much
   more often than AngW4Z. It never sampled gaps of 0.10 m or more in meaningful numbers. The
   mechanism is the SGS level freeze described in `~/ame_mjlab/notes/report_contact_eval.md`: with
   the contact termination, a level-2 episode ends at the first knee touch, so p_2 stays near 0 and
   prob_2 on the floor.
3. H2, stuck / refuses: rejected. 0 stuck robots in all four cells; contact robots fail while
   walking (1.5-1.7 m covered).
4. H1, self-contact terminations: rejected. 4 of 366 (1.1 %).

## 12. Foothold evaluation of contact model_9999 (Part 2)

Run: `notes/eval_foothold.py --checkpoint .../gap_edge_sgs_angw4z_contact_1024/model_9999.pt --task
Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Velocity-Rough-G1 --envs_per_cell 16 --duration_s 10` (GPU, one job;
output `notes/foothold_eval/contact_m9999/foothold_eval.json`). Scope reduced as allowed: contact
model only, the whole gap board (5 families x 10 levels, 16 robots per cell).

- Env: the AngW4Z task env, i.e. the contact run's observations without the `illegal_contact`
  termination, so stone robots are measured until they fall rather than until the first knee
  touch. "Flat" is the stones family's level 0 (a continuous floor); the board has no separate flat
  tile. The AME-mix stepping_stones tile is not on this board (section 11 covers it).
- Command speed: a fixed forward command vx = 0.8 m/s (`--vx` default), vy = 0, wz = 0, no heading
  control, no standing envs, from a centre spawn with uniform heading. A robot counts until it
  falls, the 10 s end, or its pelvis leaves its tile (3.75 m from the centre, about 5 s at 0.8 m/s).
- "Touchdown false positive" below = the mean share of the 15 sole points flagged unsupported on the
  touchdown step (what the reward would charge), on tiles with no gaps or edges.

Touchdown unsupported share at delta 0.02 / 0.05 / 0.10:

- flat (stones level 0, 817 touchdowns): 0.400 / 0.036 / 0.0007.
- wave, all levels pooled (9117 touchdowns): 0.370 / 0.093 / 0.0070; worst level 0.430 / 0.130 /
  0.017 (level 7).
- random_rough, pooled (9128): 0.495 / 0.175 / 0.0118; worst level 0.518 / 0.195 / 0.014.
- So at 0.02 almost half the sole is "unsupported" on ordinary ground (heel or toe strike tilts the
  foot by a few cm), and 0.05 is still 4-18 %.
- pyramid_stairs_inv (stairs up), delta 0.10: 0.026, 0.039, 0.063, 0.086 at levels 0-3, then 0.150
  (level 4), 0.220, 0.264, 0.297, 0.219, 0.107 (levels 5-9; at levels 8-9 no robot leaves the
  tile). At 0.05: 0.10 (level 0) to 0.38 (level 7).
- pyramid_stairs (down), delta 0.10: 0.002 at level 0 rising to 0.17-0.18 at levels 7-9.
- stones (lattice), delta 0.10: 0.20 (level 1, 5 cm gaps), 0.39, 0.53, 0.51 (levels 2-4), 0.55-0.61
  (levels 5-8).

Gap-landing rate (touchdowns with any sole point over a pit) and falls, stones: level 1 63 % (falls
0 %), level 2 69 % (63 %), level 3 79 % (94 %), level 4 74 % (100 %), levels 5-9 32-81 % (75-100 %).
Gap landings elsewhere are 0, except 0.4-1.7 % on pyramid_stairs levels 6-9. The in-contact
unsupported share follows the same pattern (flat 0.169 / 0.032 / 0.001).

Swing-cost diagnostic (raw costs per swing; two-sided = mjlab's terms, one-sided = C):

- stones levels 1-9 pooled, gap-crossing swings (n = 1596): clearance 1.50 + swing height 6.23 =
  7.73 two-sided; 1.12 one-sided.
- the same cells, other swings (n = 1870): 0.09 + 0.90 = 0.99 two-sided; 0.88 one-sided.
- Ratio gap / non-gap: 7.8 two-sided, 1.27 one-sided. Whole board: 8.5 and 1.5 (every gap swing is
  on stones).
- Per level, two-sided ratio: 0.64 (1), 1.04 (2), 1.51 (3), 1.67 (4), 7.0 (5), 12.3 (6), 25.8 (7),
  26.9 (8), 27.0 (9). Levels 1-4 pooled: 1.14.
- Why it explodes from level 5: both mjlab terms measure foot height to the terrain straight below.
  Over a 2 m pit that is the pit floor, so a foot crossing a wide gap reads as 1-2 m high, and
  `(peak / 0.10 - 1)^2` and `|h - 0.10| * |v|` become large. At levels 1-4 the gaps are narrow
  enough that the foot is over a stone most of the swing, and the swing-height cost over gaps is
  actually lower than elsewhere (the policy lifts higher there).

The question "with delta 0.05, why does stair-edge overhang only count from level 4?": it doesn't.
With 0.05 a sole over a stair edge counts from level 1 (0.067 m steps); level 4 is the threshold
for delta 0.10 (0.117 m steps; level 3's 0.100 m step is not above 0.10 because the test is strict).
The data agree: at 0.10 the stairs-up share jumps from 0.086 (level 3) to 0.150 (level 4); the
0.03-0.09 below that comes from tilted soles and soles partly over a 0.10 m step by a few mm, not
from overhang.

## 13. Decisions, baked into the task

All in `FOOTHOLD_TASK_FLAGS` (`src/ame/__init__.py`), train and play configs alike.

- delta = 0.10 m. Rule: the smallest of 0.02 / 0.05 / 0.10 with a touchdown false-positive share
  below 2 % on flat, wave and random_rough. 0.02 fails (40-50 %), 0.05 fails (3.6 %, 9.3 %,
  17.5 %), 0.10 passes (0.07 %, 0.7 %, 1.2 % pooled; worst single level 1.7 % on wave). Cost of
  the choice: stair-edge overhang counts only from level 4; pits count at every level.
- C ON (`one_sided_clearance=True`). Rule: on if the two-sided clearance + swing-height cost on
  gap-crossing swings is at least 2x that on non-gap swings. Measured 7.8x (stones levels 1-9
  pooled), 8.5x (whole board). Caveat: levels 1-4 alone give 1.14x; the excess comes from levels
  5-9, where these terms read the 2 m pit as foot height. That is the situation the new task has to
  learn, so the pooled number is the one the rule is applied to. One-sided brings the ratio to 1.27x.
- Self-contact: keep `illegal_contact` as it is (self-contact still terminates). Rule: terrain-only
  if at least half of the stone contact terminations are robot-robot. Measured 4 of 366 (1.1 %).
  `self_collisions` untouched.
- `tests/test_foothold.py` pins the result: the task's env differs from the Contact task only by the
  sole sensor, the foothold reward (delta 0.10), the swing metric and the two one-sided functions.

## 14. GPU smoke train (laptop, nothing kept)

`train Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Foothold-Velocity-Rough-G1 --env.scene.num-envs 64
--agent.max-iterations 30 --agent.run-name smoke_foothold_delete_me --agent.logger tensorboard`,
exit 0, no errors or tracebacks; the run directory was deleted afterwards.

Logged (first -> last of 30 iterations):

- `Loss/adv_std_g1` 0.42 -> 0.074; `Loss/adv_std_g2` 0.18 -> 0.0069.
- `Loss/explained_var_g1` -0.17 -> 0.017; `Loss/explained_var_g2` -6.8 -> -6.3 (min -10.3).
- `Loss/return_mean_g1` -0.98 -> -1.18; `Loss/return_mean_g2` 0.019 -> -0.0004.
- `Loss/value` 0.42 -> 0.0028.
- `Episode_Reward/foothold` -1.7e-4 -> -1.8e-5 (always <= 0, as it must be).
- `Swing/clearance_{gap,nogap}`, `Swing/clearance_1s_*`, `Swing/height_*`, `Swing/height_1s_*` and
  `Swing/gap_share` all present (gap keys in the 10 iterations that had a gap swing; gap_share
  0-0.024). With C on, two-sided and one-sided values coincide except where a foot is above target,
  as expected (for example `Swing/height_nogap` 2.78 vs `height_1s_nogap` 0.85 at iteration 0).

Sane, with one thing that looks bad but is not from this change: the mean episode length falls from
about 12 to 3.5 steps within 30 iterations, almost all `illegal_contact`. The same smoke of the
existing Contact task (`Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Contact-Velocity-Rough-G1`, also deleted)
gives the same 3.5-3.7 steps. With so few steps the foothold reward and group 2 are near zero,
which is why `explained_var_g2` is strongly negative (section 4's caveat). The real contact run
recovered from this (699 steps at 2k iterations with 1024 envs).

## 15. Experiment row, commit

- `notes/experiments.md`: row and section for `gap_edge_sgs_angw4z_contact_foothold_1024`.
- Commit on `sgs`, tag `run/gap_edge_sgs_angw4z_contact_foothold_1024`, not pushed.
