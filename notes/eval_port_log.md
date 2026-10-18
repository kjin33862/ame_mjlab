# Eval port log (Isaac Lab eval pipeline -> ame_mjlab)

Source of the rules: `~/AME_Locomotion` branch `eval-pipeline` (`docs/EVALUATION.md`, `scripts/rsl_rl/evaluate.py`).
Paths below: `mjlab/...` = `.venv/lib/python3.13/site-packages/mjlab/...` (pinned rev 898a700).

## 2026-10-12 — STEP 1: setup and reading

Setup
- Disk: 293 GB free. Host RAM 30 GB (21 GB available), 16 CPUs.
- `uv sync` OK (with approval). `uv run pytest`: 44 passed, 1 skipped. `uv run list-envs`: the four Ame/Delta tasks are listed.
- **GPU unusable right now**: `nvidia-smi` -> "Driver/library version mismatch" (NVML 580.178); torch -> CUDA error 804.
  The driver was updated without a reboot. A reboot is needed before any simulator run.

Findings (file:line)
- Env config: `ame_rough_env_cfg` (src/ame/env_cfg.py:139-158) = mjlab `unitree_g1_rough_env_cfg(play)` +
  `apply_terrain` (:99-104) + `apply_scan(noise=not play)` (:155) + `apply_twist_commands` (:107-115) + `apply_rewards` (:126-136).
- Play mode (mjlab/tasks/velocity/config/g1/env_cfgs.py:164-183): episode 1e9 s, actor corruption off, no pushes,
  curriculum cleared, adds a `randomize_terrain` reset event. AME then installs `play_board` (src/ame/terrains.py:417-419):
  curriculum off, 5x5 tiles, border 10 m. With curriculum off every tile draws a random family and a random difficulty
  in `difficulty_range` (mjlab/terrains/terrain_generator.py:221-254).
- Env origins (mjlab/terrains/terrain_entity.py:325-365): rows random in [0, max_init_terrain_level] (=1, env_cfg.py:104),
  columns by proportion if there is one column per family, else an even split. `randomize_terrain`
  (mjlab/envs/mdp/events.py:58-68) re-randomizes on every reset. `reset_base` adds x,y U(-0.5,0.5) and yaw U(-3.14,3.14)
  (mjlab/tasks/velocity/velocity_env_cfg.py:201-213; events.py:111-157). `scene.env_origins` is the terrain's tensor
  (mjlab/scene/scene.py:104-107), so it can be overwritten in place after construction.
- Stepping stones (src/ame/terrains.py:137-143 + mjlab/terrains/config.py:212-227 defaults,
  mjlab/terrains/primitive_terrains.py:1133-1280). At difficulty d: stone size 0.6-0.3d, gap 0.05+0.2d, size jitter
  +-0.2d, position jitter +-0.1d, top-height jitter +-0.05d; 2 m platform at the tile center (top z 0.2), pit floor at
  -2.0, 0.25 m border ring. Spawn origin = tile center, z 0.2. One numpy RNG for the whole board, seeded by
  `TerrainGeneratorCfg.seed` (random if None; training sets none) (terrain_generator.py:166-170): tiles differ for d > 0;
  at d = 0 all tiles are identical except colour.
- Elevation map (src/ame/mdp/observations.py:80-128): sigma 0.03 per cell per step, per-episode offset U(-0.05,0.05),
  per-episode drift of the z content by dx,dy in {-1,0,1} cells (x/y channels stay static), resampled when
  `episode_length_buf == 0` (so it also applies in the first episode). Off in play (`noise=not play`).
- Actor proprio noise (velocity_env_cfg.py:76-110): lin_vel +-0.5, ang_vel +-0.2, gravity +-0.05, joint_pos +-0.01,
  joint_vel +-1.5 (uniform); off in play (env_cfgs.py:168).
- Terminations (src/ame/reward_cfg.py:279-288; src/ame/mdp/terminations.py): illegal_contact (sensor `undesired_contact`,
  > 1.0 N, current net force, no history), bad_orientation (|gx|>0.985 or |gy|>0.9 or gz>-0.3), physics_blowup
  (|qacc| > 2e4), time_out. Sensor bodies (reward_cfg.py:43-51, 57-70): torso, shoulders, hips, knees, elbows, waist
  links, pelvis (mode body, netforce, 1 slot); index->name via `ContactSensor.primary_names`.
- Per-term values survive the auto-reset (`TerminationManager._term_dones`, mjlab/managers/termination_manager.py:88-113),
  but sensors are re-sensed after the reset (manager_based_rl_env.py:443-466), so the touching body must be captured
  inside a termination term (as in the Isaac Lab tool).
- Commands: twist, heading control off (env_cfg.py:107-115), rel_standing_envs 0.05, base rel_forward_envs 0.2.
  Fix with ranges (1,1),(0,0),(0,0) and rel_standing_envs 0 (velocity_command.py:75-138). No heading controller.
- Checkpoint (mjlab/scripts/play.py:203-208): `MjlabOnPolicyRunner(env, asdict(rl_cfg), device)`,
  `runner.load(path, load_cfg={"actor": True}, strict=True)`, `get_inference_policy()`. Model class from
  `class_name="ame.models:AmeModel"` (src/ame/rl_cfg.py:171-187). Skipping `load` gives a random-init policy.
- Control: 0.005 s x decimation 4 = 0.02 s (50 Hz); episode 20 s = 1000 steps (play: 1e9 s).
- Seeding: `cfg.seed` -> torch/numpy/warp (mjlab/utils/random.py:9-24); MuJoCo Warp is not fully deterministic.

Geom count (the size limit; measured on CPU, `np.triu_indices(ngeom)` in mujoco_warp/_src/io.py:294 is O(ngeom^2))

| board | terrain geoms |
|---|---|
| training board (10 x 12 used) | 5270 |
| stones tile, d = 0 / 0.25-0.5 / 0.75-1 | 134 / 166 / 186 per tile |
| stones 8x8 / 10x10 / 15x15, d = 0.5 | 10.6k / 16.6k / 37.4k |

15x15 needs ~7e8 geom pairs -> tens of GB host RAM: expected out of memory (30 GB). To be measured.

## 2026-10-12 — STEP 2: plan (approved as proposed)

- New files only: `src/ame/eval/` (env cfg, noise term, recorded termination), `scripts/evaluate.py`,
  `scripts/aggregate_sweep.py` (copy), `scripts/run_noise_sweep.sh`, `scripts/eval_memory_test.sh`,
  `tests/test_eval.py`, `docs/EVALUATION.md`. No existing file changed (no hook needed).
- Domain randomization: match the Isaac Lab tool (user's choice): foot friction + encoder bias on, CoM offset off.
- Defaults: difficulty 0.5, noise preset `train`, outputs under `logs/eval_results/` (already git-ignored).

## 2026-10-12 — STEP 3: implementation and tests (CPU only; GPU blocked until reboot)

Done
- Commits on branch `eval` (git had no identity configured; commits use `-c user.name=eojin_kang
  -c user.email=<account email>` per command, global config untouched).
- `uv run pytest`: 61 passed, 1 skipped (17 new). The eval map term is bit-identical to the training term at the
  training values for all 9 drift combinations. `ruff check .` clean.
- mjlab runs on CPU (CUDA graphs are skipped), so every rollout check ran on CPU with `--device cpu`:
  - random-init policy, 2x2 tiles: spawn offset from tile center 0.0 m, displacement after step 1 < 1 mm, median
    height above support 0.78 m, 4 distinct tiles, outputs written, foot friction differs per env (0.43-0.82).
  - every random/zero-policy robot ends by `illegal_contact` within ~0.25 s. Diagnosed with the MuJoCo contact list:
    the hand collision geom (`right_hand_collision`, `right_wrist_*`) touches `right_hip_collision`
    (`right_hip_roll_link`). Even with zero actions (default-pose PD hold), the hands settle onto the hip-roll links.
    Illegal contact counts self-contact (sensor without a secondary filter), so a G1 holding mjlab's default pose
    terminates. Trained policies keep their hands clear, so evaluation of a checkpoint is not affected, but it is
    worth knowing for training (the 5 % standing-command envs). The recorded `contact_body` matched the diagnosis.
  - time-out path: zero policy with `--episode_s 0.1` -> all 4 envs `stuck` at step 5/5, not truncated.
  - custom noise (`--drift_cells 2 --map_delay_steps 5`) runs through `elevation_map_eval` inside the env.
  - aggregator on synthetic summaries (2 models x 9 conditions x 2 seeds, one invalid, one random-init): pooling,
    skipping and all 5 plots OK.
- Board size, CPU build with 4 robots (host peak RSS): 8x8 9.0 GB / 20 s, 10x10 19.7 GB / 51 s,
  12x12 26.3 GB / 121 s. ~160 B per geom pair (mujoco_warp `put_model`). 15x15 (~7e8 pairs) extrapolates to
  > 100 GB: not attempted. Default grid set to 8x8; more samples = more seeds.
- Added beyond the plan: `--zero_policy` (pipeline test), a speed validity check (success faster than 3 m/s mean is
  invalid), `scripts/eval_memory_test.sh`.
- Side check of the Isaac Lab tool: `actions, *_ = policy(obs)` there is correct (the AME policy returns
  `(actions, attention_weights)`); in mjlab the policy returns a tensor and the port uses `actions = policy(obs)`.
- Default board on CPU: 8x8, 64 envs, random policy -> valid, 64/64 distinct tiles, spawn offset 0.0 m, step-1
  displacement 0.5 mm, step-1 median height 0.78 m, host RSS 8.9 GB, build 20 s. Rollout on CPU is impractical
  (13 steps in 25 min with 64 envs), so CPU is for pipeline checks only.

## 2026-10-12 — STEP 3 (continued): GPU checks after the reboot

- `nvidia-smi` OK (driver 580.178.04), torch sees the RTX 5060 Laptop. `uv run pytest`: 61 passed, 1 skipped.
- GPU smoke run (`--random_policy --max_steps 100`, default 8x8, 64 envs): valid, 64/64 distinct tiles, spawn check
  passed, median height above support 0.75 m, build 43 s (cold), GPU used by the run 1.3 GB. As on CPU, all random-
  init robots end by `illegal_contact` (63 on `right_hip_roll_link`, 1 on `right_hip_yaw_link`), 0.20-1.48 s after
  the start, so the episode ends after 74 steps (all envs done).
- Memory test (`bash scripts/eval_memory_test.sh`, 8x8 board), all valid:

  | envs | GPU used by run | torch peak | host RSS | wall time per step |
  |---|---|---|---|---|
  | 64 | 1342 MB | 68 MB | 8.7 GB | 0.31 s |
  | 128 | 1500 MB | 127 MB | 8.7 GB | 0.62 s |
  | 225 | 1862 MB | 212 MB | 8.7 GB | 1.2 s |

  Below the estimate (1.9 / 2.3 / 3 GB). Step time is about linear in the number of envs (~5 ms per env-step), so a
  full 1000-step episode with 64 envs is ~5 min; the sweep's 30 min timeout per run is enough for 8x8.
- Largest board. The laptop has 30 GB of swap, so the earlier CPU numbers for 12x12 may have used swap. Re-measured
  on GPU, one robot per tile, 5 steps, each run in `systemd-run --user --scope -p MemoryMax=27G -p MemorySwapMax=0`:

  | board | geoms | host RSS peak | GPU used by run | build | result |
  |---|---|---|---|---|---|
  | 10x10 | 16,604 | 19.8 GB | 2.8 GB | 49 s | valid |
  | 11x11 | 20,090 | 26.0 GB | 3.9 GB | 77 s | valid |
  | 12x12 | 23,908 | 27.8 GB | 5.4 GB | 115 s | valid |
  | 13x13 | ~28k | 28.7 GB | - | - | Warp CUDA out of memory during the build |

  12x12 is the largest board that loads; it is at the RAM cap (likely under memory pressure), so 10x10 is the
  largest comfortable one. GPU memory also grows ~quadratically with the geom count, so 13x13 fails on the 8 GB GPU
  even with enough host RAM. The 8x8 default stays.
- docs/EVALUATION.md: memory section rewritten with these numbers.

Left to do
1. With a trained checkpoint (from the 2080 Ti run): one full run at default settings; check that successes take
   >= ~4 s (4 m at 1 m/s), `median_height_above_support_m` ~0.75, and that fail bodies make sense.
   Then `MODELS="name=ckpt ..." bash scripts/run_noise_sweep.sh` and `uv run python scripts/aggregate_sweep.py`.
2. Not verified: GPU determinism across reruns, SAP vs NXN broadphase equivalence, families other than stones.
3. Optional: reduce memory of large boards (the O(ngeom^2) pair table in mujoco_warp `put_model`) only by
   patching mujoco_warp; not done.

## 2026-10-14 — Why the baseline fails stepping_stones at difficulty 0

Baseline `checkpoints/baseline/model_9999.pt`, `--terrain stepping_stones --difficulty 0.0 --noise off --seed 0`:
64/64 fail by `illegal_contact` at 1.70-1.92 s, max forward 1.28-1.34 m (mean 1.32), body `left/right_hip_roll_link`.
Flat (same flags): 64/64 success.

### Confirmed (measured or read from code)

- **Terrain** (`ame_terrains_cfg()["stepping_stones"]` -> mjlab `BoxSteppingStonesTerrainCfg`, 8x8 m tile):
  stone size 0.6 -> 0.3 m and nominal gap 0.05 -> 0.25 m (linear in difficulty); stone and platform tops at
  z = 0.2 m, pit floor at z = -2.0 m (`floor_depth`), so holes are 2.2 m deep; size jitter ±0.2·d, position jitter ±0.1·d, height jitter
  ±0.05·d (all 0 at d = 0); platform 2.0 x 2.0 m, border 0.25 m. Spawn = tile center on the platform.
- **Platform edge is 1.00 m from the tile center; the first stone starts at 1.325 m.** Lattice spacing at d = 0
  is 0.65 m, and the stone whose center (4.975 m) falls inside the platform is dropped, so the first gap in front
  of (and behind) the platform is **0.325 m, not 0.05 m**. Built and listed with a scratch script. Over 20 seeds
  this first gap is 0.30-0.33 m at d = 0-0.1, ~0-0.03 m at d = 0.25-0.5, 0.12-0.32 m at d = 0.75, and up to 1.2 m
  at d = 1. So d = 0 is *not* the easiest stones tile at the platform edge.
- **~1.32 m is the pelvis over the gap, at the near edge of the first stone (1.325 m), not the platform edge.**
- **The contact partner is the robot's own hand.** New eval-only column `contact_partner` (raw MuJoCo contacts of
  the flagged body at the termination step): 64/64 `self:<same side>_wrist_yaw_link` (39 also
  `..._wrist_pitch_link`); `terrain` never appears. Summary `contact_pairs`: left hip <- left wrist 29, right hip
  <- right wrist 35. Run: `logs/eval_results/20261014-193546_baseline_stepping_stones_noiseoff_C_ss0_partner`.
  Flat: `20261014-193625_baseline_flat_noiseoff_A_flat_partner`, 64/64 success, no contacts.
- **Self-contact counts as illegal contact.** The `undesired_contact` sensor (`ame/reward_cfg.py`) has primary =
  non-foot links (incl. `hip_*_link`), **no secondary filter**, so any partner counts; the G1 entity uses mjlab's
  `FULL_COLLISION` (self-collisions on). The wrists are not in the primary list, so a hand hitting the hip shows up
  as the hip. The same sensor and termination are used in training (`illegal_contact`, threshold 1 N).
- **Trace of one robot (scratch script, d = 0, noise off):** both ankles stay at x = 0.89-1.02 m (the platform
  edge) from 1.4 s on, while the pelvis keeps going to 1.29 m and speeds up from 1.0 to 1.3 m/s. Hand-to-hip
  distance falls from 0.19 to 0.125 m, mostly sideways (y 0.16 -> 0.07 m), then contact at 1.72 s. On flat the
  hand-hip distance stays at 0.155-0.17 m.
- **Gait:** on flat and stones, both feet lift and land together (ankle x/z equal within a few cm, at 0.02 s
  resolution; left/right indexing checked by y = ±0.12 m): a two-footed hop, ~0.3 s period.
- **Run-to-run:** vs the earlier `*_C_ss0` run, 57/64 envs have the same fail time but only 40/64 the same side
  (GPU non-determinism; symmetric setup).

### Guesses (not verified)

- The policy cannot cross the 0.325 m gap with its two-footed hop: it stops the feet at the edge, the body pitches
  forward over the gap, and the arms close in on the hips. The bad-orientation or a terrain hit would probably
  follow soon after, but the hand-hip self-contact terminates it first.
- Curriculum `terrain_levels/stepping_stones` stayed at 0 in training because the row-0 tiles have the same 0.325 m
  first gap, so robots never got off the platform and never got promoted. Not checked against training logs.
- The wrist-into-hip motion may be how the policy brakes or balances. Whether it would recover without the
  self-contact termination is unknown.

### Watch it

    uv run python scripts/view_eval.py --checkpoint checkpoints/baseline/model_9999.pt \
        --terrain stepping_stones --difficulty 0.0 --noise off            # viewer, episode repeats
    uv run python scripts/view_eval.py --checkpoint checkpoints/baseline/model_9999.pt \
        --terrain stepping_stones --difficulty 0.0 --noise off --video --video_s 3
        # -> logs/eval_videos/stepping_stones_d0_off/rl-video-step-0.mp4 (960x720, failure at ~1.7 s)

## 2026-10-17 — Shuang's gap-board run vs the baseline

Inputs: `checkpoints/shuang_gap2048/` (`model_14999.pt`, `params/`, `git/value_td.diff`, his tfevents) and
`checkpoints/baseline/` (`model_9999.pt`, `params/`, `train_baseline.log`). `upstream/main` = `ae545cc` + `49a85ca`.

### Confirmed (measured or read from files)

**His uncommitted diff (`git/value_td.diff`).** Taken in his own workspace (branch `loco-stack`, commit `d7b1f1b`,
paths `ame/src/ame/...`; that commit is not in our repo). It changes `symmetry.py` only, plus one test: new
`PROPRIO_CORE = 96` and `has_lin_vel(width)`, and `mirror_obs_actions` now decides per observation group whether it
leads with `base_lin_vel` (odd width beyond 96 = yes) instead of assuming it always does. It touches only the
left/right data augmentation in PPO updates: no reward, observation or env change. For a 96-D actor (his) it is
required (without it the mirrored actor batch is shifted by 3 and wrong); for a 99-D actor (baseline) it is a no-op.
**It is already in our code** (`src/ame/symmetry.py:125-140` and `tests/test_rewards.py`, since `ae545cc`).

**Config differences** (`params/agent.yaml`, `params/env.yaml`; baseline -> Shuang):

| | baseline | Shuang |
|---|---|---|
| envs / iterations / minibatches | 1024 / 10000 / 16 | 2048 / 15000 / 8 |
| logger / run name | tensorboard / `baseline` | wandb (`ame_loco`) / `ame_mjlab_gap-2048` |
| rewards | AME-2 rebalanced (`ame.mdp`): `feet_air_time` +0.5 (threshold 0.6), `track_lin_vel_xy_exp` +3, `track_ang_vel_z_exp` +6, `termination_penalty` -50, `flat_orientation_l2` -2, `feet_stumble` -1, `undesired_contacts` -1, `feet_too_near` -1, `joint_deviation_*`, `rbf_*`, `action_rate_l2` -0.025, ... | mjlab G1 velocity stack: `track_linear_velocity` +2, `track_angular_velocity` +2, `upright` +1, `pose` +1 (walking/running stds), `air_time` **0.0**, `foot_clearance` -2 (target 0.1 m), `foot_swing_height` -0.25, `foot_slip` -0.1, `soft_landing` -1e-5, `self_collisions` -1, `body_ang_vel` -0.05, `angular_momentum` -0.02, `action_rate_l2` -0.1 |
| terminations | `illegal_contact` (self-contact counts), `bad_orientation`, `physics_blowup`, time-out | `fell_over` (tilt > 70 deg), `out_of_terrain_bounds`, time-out; **no illegal contact** |
| actor proprio | 99-D (with `base_lin_vel`, noise ±0.5) | 96-D (no `base_lin_vel`); critic identical (111-D) |
| terrain | 12 families: flat 5, pyramid stairs 12+12, random stairs 8, open stairs 5, random grid 10, discrete obstacles 12, rough 10, slopes 6+6, wave 4, stepping stones (`BoxSteppingStonesTerrainCfg`) 10 % | stone lattice (`BoxStoneLatticeTerrainCfg`, 0.40 m stones, gap 0.45·d) 40, pyramid stairs 20 + 20 (same params as baseline), wave 10, rough 10 % |
| spawn (`reset_base`) | tile centre, x/y ±0.5 m, yaw ±pi | random `spawn` flat patch (a stone centre on stones), x/y ±0.03 m, yaw {0, 90, 180, 270 deg} ±0.15 |
| curriculum | `terrain_levels_vel`, start rows 0-1 | `terrain_levels_floor` (rule `tracking`, track_ratio 0.6, floor 0), start rows 0-2 |
| metrics | - | `pelvis_vel_error`, `pelvis_yaw_error`, `pelvis_speed`, `command_speed` |

Same: seed 42, PPO hyperparameters (lr 1e-3, entropy 0.008, desired KL 0.02, gamma 0.99, lam 0.95, 5 epochs,
24 steps/env), AME model (33x21x3 map, 16 heads, mha 64), symmetry augmentation on / mirror loss off, commands,
other actor noise, events other than `reset_base`, board size 10 x 20 tiles of 8 m.

**Training curves** (his tfevents read with `tensorboard` 2.20, already installed; mine from `train_baseline.log`).
His log has no `feet_air_time` and no `illegal_contact` (his task has neither); shown: `air_time` (weight 0) and
`fell_over`. Termination values are as logged (look like counts per iteration, not fractions).

| iter | my feet_air_time | his air_time | my stones lvl | his stones lvl | my mean lvl | his mean lvl | my illegal_contact | his fell_over |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.79 | 0.99 | 0.85 | 0.98 | 76.0 | 0 |
| 1000 | 0.0006 | 0 | 0 | 0.88 | 0.00 | 0.93 | 6.25 | 2.83 |
| 2000 | 0.0010 | 0 | 0 | 1.26 | 0.56 | 4.08 | 2.95 | 1.39 |
| 3000 | 0.0011 | 0 | 0.01 | 1.82 | 1.54 | 4.26 | 2.22 | 1.05 |
| 4000 | 0.0011 | 0 | 0 | 2.32 | 2.13 | 4.57 | 1.95 | 1.17 |
| 5000 | 0.0013 | 0 | 0 | 2.80 | 2.00 | 4.71 | 1.75 | 1.13 |
| 6000 | 0.0016 | 0 | 0 | 2.70 | 2.21 | 4.59 | 1.50 | 0.92 |
| 7000 | 0.0018 | 0 | 0 | 2.93 | 2.78 | 4.83 | 1.59 | 1.55 |
| 8000 | 0.0016 | 0 | 0 | 2.98 | 2.37 | 4.80 | 2.00 | 1.21 |
| 9000 | 0.0019 | 0 | 0.01 | 3.14 | 2.70 | 4.82 | 1.90 | 1.58 |
| 9999 / 10000 | 0.0015 | 0 | 0 | 3.24 | 2.84 | 4.92 | 2.13 | 1.46 |
| 12000 | - | 0 | - | 3.61 | - | 4.97 | - | 1.13 |
| 14999 | - | 0 | - | 3.76 | - | 5.12 | - | 1.14 |

- `feet_air_time` never rises sharply in either run: mine creeps from 0.0006 to 0.002 (every 250 iterations checked),
  his is 0 throughout (weight 0). His foot-lift terms (`foot_clearance` -0.07 -> -0.10, `foot_swing_height` -0.02 ->
  -0.03, `Metrics/peak_height_mean` 0.019 -> 0.033 m) settle by iteration ~2000.
- My stepping-stones level stays 0 for all 10k iterations; his rises steadily to 3.76 (on his board row r = gap
  0.05·r m, so ~0.19 m). His pyramid-stairs level reaches ~6 by iteration 2000.

**Commit `49a85ca`** adds task `Ame-Mjlab-Gap-Velocity-Rough-G1` = `ame_rough_env_cfg(rewards="mjlab")` +
`gap_board_cfg()` + `apply_deploy_observations` + `apply_patch_spawn` + `add_tracking_metrics` + (training only)
`apply_terrain_floor(**GAP_CURRICULUM)`; new terms in `ame/mdp/{curriculums,events,metrics}.py`, tests
`tests/test_gap.py`; same PPO runner config as `Ame-Velocity-Rough-G1`. Terrain: 40 % lattice of 0.40 m flush stones
(top z = 0, pit 2 m), gap = 0.45·d with d = row/9 (row 0 seamless, +5 cm per row); 40 % pyramid stairs up/down
(5-20 cm, 0.30 m treads); 10 % wave; 10 % rough. **No centre platform, so the 0.325 m entry gap does not exist on
his board**: the gap is the same everywhere on a tile. It still exists in our eval: `--terrain stepping_stones`
builds the old `BoxSteppingStonesTerrainCfg` tile (platform + 0.325 m first gap at d = 0).

**Merge (not done; your decision).** `git merge-tree --write-tree eval upstream/main` reports no conflicts, and no file
it touches overlaps the eval work. Proposed: on `eval`, `git merge --no-ff upstream/main`, then `uv run pytest`
(expect 70 passed, 1 skipped: 65 now + 5 in `tests/test_gap.py`) and `uv run list-envs` (five tasks).

**Load check.** With the code as it was, `model_14999.pt` did not load: `proprio_embedding.weight` [64, 96] vs [64, 99]
and `mlp.0.weight` [512, 160] vs [512, 163] — the eval env's actor had `base_lin_vel`. His diff is irrelevant to
this (already present, training only). Fix (eval only): `eval_env_cfg(actor_obs="deploy")` calls
`apply_deploy_observations`; `--actor_obs auto|full|deploy` in `evaluate.py` / `view_eval.py`, `auto` reads the
actor input width from the checkpoint (`checkpoint_actor_obs`). It now loads strictly.

**Evaluation** (8x8 board, 64 robots, one per tile, vx = 1 m/s, noise off, seeds 0 1 2 pooled = 192 robots per row;
same env, rewards and terminations for both, i.e. `Ame-Velocity-Rough-G1` play env incl. illegal contact; his actor
without `base_lin_vel`; all 24 runs valid; folders `logs/eval_results/20261017-03*_cmp`):

| terrain | d | model | success | fail (main reason) | straight successes (forward >= 3.5 m) | mean max forward m | time to success s | double / single / flight |
|---|---|---|---|---|---|---|---|---|
| flat | 0.0 | baseline | 1.00 | 0 | 192 | 3.95 | 4.7 | 0.62 / 0.05 / 0.33 |
| flat | 0.0 | Shuang | 0.99 ± 0.01 | 1 (self-contact hip yaw) | 191 | 3.92 | 4.3 | 0.26 / 0.72 / 0.02 |
| pyramid_stairs | 0.3 | baseline | 1.00 | 0 | 192 | 3.96 | 4.8 | 0.58 / 0.11 / 0.31 |
| pyramid_stairs | 0.3 | Shuang | 1.00 | 0 | 192 | 3.94 | 4.5 | 0.23 / 0.73 / 0.04 |
| stepping_stones | 0.0 | baseline | 0.00 | 192 (self-contact wrist -> hip) | 0 | 1.32 | - | 0.69 / 0.07 / 0.24 |
| stepping_stones | 0.0 | Shuang | 0.98 ± 0.01 | 4 (3 contact: terrain 1, self torso 1; 1 orientation) | **4** | 1.79 | 5.0 | 0.25 / 0.72 / 0.03 |
| stepping_stones | 0.5 | baseline | 0.00 | 192 (self-contact wrist -> hip) | 0 | 1.54 | - | 0.63 / 0.08 / 0.29 |
| stepping_stones | 0.5 | Shuang | 0.98 ± 0.01 | 4 (contact with terrain) | 118 | 3.50 | 4.6 | 0.22 / 0.75 / 0.03 |

- Gait: the baseline hops (both feet down 58-69 % of the time, both feet in the air 24-33 %, one foot ~5-11 %);
  Shuang's policy walks (one foot down ~72-75 %, both 22-26 %, flight 2-4 %).
- Success = 4 m from spawn in **any** direction (`scripts/evaluate.py:601`). On stones d = 0 Shuang's robots mostly
  leave sideways: median forward / total displacement at success 0.44 (0.98 on flat and stairs, 0.89 on stones
  d = 0.5). Video `logs/eval_videos/shuang_gap2048/stepping_stones_d0_off/rl-video-step-0.mp4`: the robot turns
  about 90 deg in the first ~1.5 s, then walks off the platform and across the stones in the new direction.
  It does leave the platform, so it crosses a 0.325 m gap, but not along the command.
- Flat video of his policy: `logs/eval_videos/shuang_gap2048/flat_d0_off/rl-video-step-0.mp4` (6 s).

**Training command for the gap task** (not run; works only after merging `upstream/main`, the task is not registered
on `eval` yet):

    nohup uv run train Ame-Mjlab-Gap-Velocity-Rough-G1 --env.scene.num-envs 1024 --agent.max-iterations 2000 \
      --agent.run-name gap_short --agent.logger tensorboard > train_gap_short.log 2>&1 &

(1024 envs and 16 minibatches are the defaults, seed 42; his run used 2048 envs and 8 minibatches, which need more
than 16 GB of GPU memory per the upstream README.)

### Guesses (not verified)

- The main reason my baseline hops and never learns stones is the reward/terrain recipe, not the symmetry diff:
  `feet_air_time` (threshold 0.6 s) stays tiny all run, so nothing pays for alternating steps, and the old stones tile
  has the 0.325 m entry gap even at row 0, so the curriculum never promotes on stones. His board starts at a gap of 0
  and grows 5 cm per row, and his reward set (pose, foot clearance, swing height) shapes a walking gait without
  any air-time reward.
- The 90 deg turn on stones d = 0 is plausibly learned from his spawn (headings in 90 deg steps on a square lattice)
  and his weak yaw tracking (upstream README: 0.07 rad/s for 0.8 rad/s). Why it turns on d = 0 but mostly not on
  d = 0.5 is unknown (d = 0.5 has a ~0-0.03 m entry gap, d = 0 has 0.325 m).
- His few failures would partly not count in his own task (no illegal-contact termination); the eval is stricter for
  him than his training was.
- Not tested: his own lattice board in the eval (would need the gap task, i.e. the merge), noise on, and turning
  commands.

## 2026-10-18 — Overnight: merge, forward-only success, stone_lattice, full matrix (unattended)

Run unattended (user asleep); choices made without asking are marked **Choice**.

### Part A — eval code

1. **Merge.** `git merge --no-ff upstream/main` on `eval` -> `d2fa505`, no conflicts. `uv run pytest`: 70 passed,
   1 skipped (as predicted). `uv run list-envs`: `Ame-Mjlab-Gap-Velocity-Rough-G1` is listed (with
   `Ame-Stage1/2-Velocity-Rough-G1`, `Ame-Velocity-Rough-G1`).
2. **Success = forward progress only** (`scripts/evaluate.py`). forward = xy displacement from spawn projected on the
   spawn heading (yaw 0 -> +x; the command is vx in the base frame, so the command direction at spawn). Success when
   forward >= `--success_dist` (4.0 m). New per-env column `forward_at_end_m` (forward displacement at the last
   tracked step: success, fail or time-out); `max_xy_disp_m` (any direction) kept; `max_forward_m` is now the
   projection on the spawn heading (identical to before since spawn yaw is always 0). Summary: `success_rule`,
   `success_dist`, `mean_forward_at_end_m`, `mean_max_xy_disp_m`.
   **Choice:** `--success_rule forward|any` (default `forward`) so the old rule can still be reproduced.
   **Consequence:** a robot that walks sideways is now `stuck` at the time-out (or fails later), not `success`.
   Old results (`*_cmp` etc.) used the any-direction rule and are not comparable on stones.
3. **`--terrain stone_lattice`** (`src/ame/eval/terrains.py`): the gap board's `BoxStoneLatticeTerrainCfg` with
   the same fields (0.40 m stones, `gap_range` (0, 0.45) -> gap = 0.45·d, 2 m pit, 8 m tile; copied from
   `gap_board_cfg()` so it follows any change there), proportion 1, no spawn patches. `stepping_stones` unchanged
   (test checks it is the same config as the training mix).
   **Choice / found:** the lattice is centred on the tile, and with an even stone count per side the tile centre is
   a gap crossing: d = 0.25 (14 stones, nearest stone centre 0.26 m away) and d = 1.0 (8 stones, 0.42 m). The eval
   spawns at the tile origin with yaw 0, so the robot would start over the pit. The eval subclass
   `EvalStoneLatticeTerrainCfg` therefore puts the tile origin on the stone nearest the tile centre (ties: lowest x,
   then lowest y = most room ahead). At d = 0.5 / 0.75 there is a stone at the centre; d = 0 is a seamless floor.
   Training code untouched.
4. `view_eval.py` takes `--terrain stone_lattice` (it already passed `--terrain` through; help/doc updated).
   Tests (`tests/test_eval.py`): forward rule (sideways/backward/diagonal/rotated heading, `any` rule), lattice cfg
   equals the gap board's for d in {0, .25, .5, .75, 1}, origin is a stone centre nearest the tile centre, old
   stepping_stones unchanged. `uv run pytest`: 77 passed, 1 skipped. Commit `d63f3b8`.
   Smoke runs on GPU (tag `smoke`, kept): Shuang stone_lattice d = 1.0: valid, spawn offset 0, height 0.79 m after
   step 1, success 0.31 (21 s wall); baseline d = 0.25: valid, success 0, fail 1.0 (42 s wall).

### Part B — setup

5. `scripts/run_policy_matrix.sh <model_name> <checkpoint>`: matrix (7 terrains, d 0/.25/.5/.75/1, flat d = 0
   only, seeds 0 1 2, `--noise off`, 8x8 tiles = 64 robots) = 93 runs, then noise sweep on stone_lattice d = 0.5 =
   36 runs; tags `mx-<terrain>-d<d>-s<seed>` / `nz-<cond>-s<seed>`. Restart-safe (skips a valid summary with the same
   model_name and tag); the list is gone through a second time only if pass 1 had failures (= one retry).
   Gait fractions are always recorded by `evaluate.py` (no flag needed).
   **Choice (noise sweep):** each map parameter is changed alone from "noise off" (no proprio noise, other map
   terms 0); "train" = the training preset (map sigma 0.03, offset 0.05, drift 1 + proprio noise). The previous
   stepping-stone sweep changed parameters from "proprio on, map off" instead, so the two are not the same baseline.
   **Choice:** "off" is run as its own condition (`nz-off-*`) although it equals the matrix cell stone_lattice d = 0.5:
   costs 6 short runs and doubles as a run-to-run (GPU non-determinism) check.
   `scripts/aggregate_matrix.py`: pools seeds per cell, Wilson 95 % CI, CSV + three plots in
   `logs/eval_results/matrix_plots/`. Commit `87f22b9`.
6. **Timing:** one run 14-42 s alone (build ~5 s; short when robots succeed or fail early, up to 1000 steps when
   they survive without progress). ~20 s per run with both models in parallel. Estimate: 129 runs x ~20-40 s
   ≈ 45-90 min per model. **Choice:** both checkpoints run at the same time (two processes, ~3 GB host RAM and
   ~0.4 GB GPU each; 30 GB RAM / 8 GB GPU available). Launched 01:10 with `nohup systemd-inhibit --what=sleep:idle`;
   logs `logs/eval_results/matrix_{baseline,shuang_gap2048}.log`, per run `logs/eval_results/matrix_logs/`.
8. **Videos** (4 s, stone_lattice d = 0.5, noise off, seed 0):
   `logs/eval_videos/baseline/stone_lattice_d0.5_off/rl-video-step-0.mp4` (stays on its start stone, hopping in place)
   and `logs/eval_videos/shuang_gap2048/stone_lattice_d0.5_off/rl-video-step-0.mp4` (walks forward over the stones
   and reaches the tile border at ~3.8 s). Looked at 3 frames each (0.5, 2, 3.8 s).
