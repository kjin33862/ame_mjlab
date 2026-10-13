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

Left to do (needs the GPU: reboot first, `nvidia-smi` must work)
1. `uv run pytest` (CUDA path of the tests) and one GPU smoke run:
   `uv run python scripts/evaluate.py --random_policy --max_steps 100`.
2. Memory test: `bash scripts/eval_memory_test.sh` (64 / 128 / 225 envs on the 8x8 board, 200 steps each). Estimate
   before measuring: ~1.9 / 2.3 / 3 GB GPU (README: ~1 GB + 5.7 MB/env on the training board; this board has 2x the
   geoms). Record the result here and in docs/EVALUATION.md.
3. With a trained checkpoint (from the 2080 Ti run): one full run at default settings; check that successes take
   >= ~4 s (4 m at 1 m/s), `median_height_above_support_m` ~0.75, and that fail bodies make sense.
   Then `MODELS="name=ckpt ..." bash scripts/run_noise_sweep.sh` and `uv run python scripts/aggregate_sweep.py`.
4. Not verified: GPU determinism across reruns, SAP vs NXN broadphase equivalence, families other than stones.
5. Optional: reduce host RAM of large boards (the O(ngeom^2) pair table in mujoco_warp `put_model`) only by
   patching mujoco_warp; not done.
