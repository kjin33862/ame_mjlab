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
