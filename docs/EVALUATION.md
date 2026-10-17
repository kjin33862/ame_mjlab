# Quantitative evaluation of AME checkpoints (mjlab)

A headless evaluation that answers one question per run: *on a grid of terrain tiles, what fraction of robots walk
4 m before falling?* It is a port of the Isaac Lab evaluation in `AME_Locomotion` (branch `eval-pipeline`) with the
same outcome rules and output format, so policies trained here (for example with map drift on/off, or symmetry
augmentation on/off) can be compared the same way.

Nothing here changes training or play. The evaluation builds its own env config from `ame_rough_env_cfg(play=True)`
and uses its own observation and termination terms.

## Files

| file | what it does |
|---|---|
| `src/ame/eval/env_cfg.py` | `eval_env_cfg()`: one terrain family at a fixed difficulty on a seeded `num_rows x num_cols` board, fixed spawn and command, 20 s episodes, eval noise. `assign_tiles()`: env *i* -> its own tile. |
| `src/ame/eval/observations.py` | `elevation_map_eval`: the training elevation map with configurable height noise, offset, drift and map update delay. |
| `src/ame/eval/terminations.py` | `illegal_contact_recorded`: the training illegal-contact termination, plus a record of the touching body. |
| `scripts/evaluate.py` | One evaluation (one policy, one terrain, one noise setting, one seed) -> `per_env.csv`, `summary.json`, `args.json`. |
| `scripts/run_noise_sweep.sh` | The noise-robustness sweep over several checkpoints and seeds. Restart-safe. |
| `scripts/aggregate_sweep.py` | CPU only. Pools seeds per (model, condition), writes `sweep_summary.csv` and plots. |
| `scripts/eval_memory_test.sh` | Short runs with 64 / 128 / 225 envs that report GPU and host memory. |
| `tests/test_eval.py` | Unit tests (no simulator): config, noise term (bit-exact against the training term), recorded termination. |

## Running it

```bash
# Pipeline test without a checkpoint (results are meaningless)
uv run python scripts/evaluate.py --random_policy --num_rows 2 --num_cols 2 --max_steps 50
uv run python scripts/evaluate.py --zero_policy --num_rows 2 --num_cols 2 --episode_s 0.1   # time-out -> stuck

# One evaluation: 8 x 8 stepping-stone tiles, difficulty 0.5, training noise, seed 0
uv run python scripts/evaluate.py --checkpoint logs/rsl_rl/ame_g1/<run>/model_10000.pt --model_name drift_on --seed 0

# The same tiles without any noise
uv run python scripts/evaluate.py --checkpoint <ckpt> --model_name drift_on --seed 0 --noise off

# Sweep over two checkpoints, then the table and plots
MODELS="drift_on=<ckpt1> drift_off=<ckpt2>" bash scripts/run_noise_sweep.sh
uv run python scripts/aggregate_sweep.py
```

Results go to `logs/eval_results/<YYYYmmdd-HHMMSS>_<model_name>_<terrain>_<noise tag>[_<tag>]/` (ignored by git).
The noise tag is `noiseoff` or `h<sigma>-o<offset>-dc<drift cells>-d<delay>-p<proprio 0|1>`. `--task` selects the
policy config (default `Ame-Velocity-Rough-G1`; `Delta-Velocity-Rough-G1` for DELTA checkpoints). The checkpoint is
loaded the same way as `uv run play`. `--actor_obs` (default `auto`) picks the actor observation set from the
checkpoint: `full` (with `base_lin_vel`, 99-D proprio, the AME tasks) or `deploy` (without it, 96-D, e.g. the
gap-board run `checkpoints/shuang_gap2048`). Only the actor's inputs change; the eval env, rewards and terminations
stay those of `Ame-Velocity-Rough-G1` (including illegal contact, which the gap-board task does not train with).

Success is a base displacement of `--success_dist` from spawn in *any* direction; `max_forward_m` in `per_env.csv`
is the progress along the commanded +x. Compare the two when a policy may turn away from the command.

Neither test policy walks: both end by illegal contact within about 0.25 s, because at mjlab's default G1 pose
the hands come to rest against the hip-roll links and illegal contact counts self-contact. The short episode above
is how the time-out path is exercised without a checkpoint.

## Setup of each run

- **Board**: `--num_rows x --num_cols` tiles (default 8 x 8) of one family (`--terrain`, default `stepping_stones`,
  any family of the AME training mix is accepted) at one difficulty (`--difficulty`, default 0.5), generated with
  `--seed`. Same seed = same board, so conditions with the same seed are paired.
- **Spawn**: env *i* spawns at the center of tile `(i // num_cols % num_rows, i % num_cols)`, facing +x (yaw 0).
  With more envs than tiles, env *i* uses tile `i mod tiles`. On stepping stones the tile center is the 2 m start
  platform.
- **Command**: fixed `vx = 1.0 m/s, vy = 0, wz = 0`, no standing envs. There is no heading control in this task, so
  nothing steers a robot back to +x if it turns.
- **Episode**: 20 s = 1000 policy steps of 0.02 s (0.005 s physics x decimation 4).
- **Randomization**: as in play mode, minus the CoM offset: foot friction U(0.3, 1.2) and encoder bias stay on,
  pushes and CoM offset are off (`--com_dr` turns the CoM offset back on).

## Outcome rules

Only the **first episode** of each env counts; after an auto-reset the env is ignored. Every policy step, for envs
that are still active, in this order:

| order | check | result |
|---|---|---|
| 1 | The episode ended this step and it was not a time-out | `fail`, reason from the termination term: `contact` (illegal_contact), `orientation` (bad_orientation), `blowup` (physics_blowup) |
| 1 | The episode ended this step by time-out (20 s) | `stuck` |
| 2 | Base x-y distance from spawn >= `--success_dist` (4.0 m) | `success` |
| 3 | Tilt > `--tilt_deg` (60 deg) for `--fall_hold_s` (0.2 s = 10 steps) | `fail`, reason `tilt` |
| 3 | Base height above the supporting surface < `--fall_height` (0.4 m) for 0.2 s | `fail`, reason `height` |
| 3 | Base z < spawn z - `--spawn_drop` for 0.2 s (only with `--spawn_drop_rule`) | `fail`, reason `spawn_drop` |
| end | No outcome when the loop stops (only with `--max_steps`) | `stuck` |

- **Terminations** (from training): illegal contact = net force > 1 N on torso, shoulders, hips, knees, elbows,
  waist links or pelvis, from **any** contact, self-contact included (for example a hand hitting a hip);
  bad orientation = |g_x| > 0.985 or |g_y| > 0.9 or g_z > -0.3 in the base frame (about 72 deg);
  physics blowup = |joint acceleration| > 2e4. `contact_body` is the sensor body with the largest force when
  illegal contact fired; it is recorded inside the termination because the auto-reset clears the sensors.
- **Supporting surface**: the highest `terrain_scan` ray hit within `--support_radius` (0.3 m, x-y) of the base; if
  no ray hits within that radius, the highest hit of the whole scan. A walking G1 is at about 0.75 m.
- **Success** does not check direction. Its time is `step x 0.02 s`; for tilt/height falls the time is the end of
  the 0.2 s hold.
- **Ties**: a termination is checked before success, success before falls.

## Noise

| option | `train` preset | `off` preset | meaning |
|---|---|---|---|
| `--height_sigma` | 0.03 | 0 | std of per-cell Gaussian height noise, every step [m] |
| `--height_offset` | 0.05 | 0 | per-env, per-episode offset U(-x, x) [m] |
| `--drift_cells` | 1 | 0 | per-env, per-episode shift of the z values by dx, dy uniform in {-n..n} cells (5 cm); x/y channels stay fixed |
| `--map_delay_steps` | 1 | 1 | refresh the map every N policy steps (5 = 10 Hz); 1 = every step |
| `--proprio_noise` | on | off | the actor's training noise: lin vel +/-0.5, ang vel +/-0.2, gravity +/-0.05, joint pos +/-0.01, joint vel +/-1.5 (uniform) |

`--noise train` (the default) with no overrides uses the training observation term itself, so it reproduces the
training noise exactly. Any override switches to `elevation_map_eval`, which applies the same operations in the same
order; with the training values it gives bit-identical maps (unit test). The per-episode offset and drift also apply
in the first episode. `--proprio_noise preset|on|off` follows the preset or forces it.

## Outputs

`args.json`: the full CLI (`argv` included).

`per_env.csv`, one row per env, same columns as the Isaac Lab tool plus `termination_term`:

| column | meaning |
|---|---|
| `env_id`, `tile_row`, `tile_col` | env and its tile |
| `outcome`, `fail_reason` | `success` / `fail` / `stuck`; `contact`, `orientation`, `blowup`, `tilt`, `height`, `spawn_drop` or empty |
| `contact_body` | for illegal-contact fails: the body with the largest force |
| `termination_term` | the termination term(s) that fired, joined by `+` (e.g. `illegal_contact`) |
| `fail_time_s`, `time_to_success_s` | time of the outcome [s], `nan` otherwise |
| `max_forward_m`, `max_xy_disp_m` | largest world-x displacement and x-y distance from spawn, including the terminal step |
| `mean_abs_vx_err` | mean \|v_x - command\| (yaw frame) over active steps; `nan` for failed envs |
| `tracked_steps` | steps the env was active (includes the step of a success or fall, not the step of a termination) |
| `mean_height_above_support_m` | mean base height above the supporting surface while active |
| `static_friction` | the env's foot friction after randomization |

`summary.json` (valid run): the Isaac Lab keys (`valid`, `validity_check`, `checkpoint`, `checkpoint_iter`, `arch`,
`terrain`, `difficulty`, `seed`, `num_envs`, `tiles_total`, `tiles_distinct_used`, `height_noise`, `proprio_noise`,
`tag`, `success` / `fail` / `stuck` as `{count, rate, se}`, `fail_reasons`, `mean_time_to_success_s`,
`mean_max_forward_m`, `mean_abs_vx_err_survivors`, `n_survivors`, `median_height_above_support_m`,
`steps_simulated`, `episode_steps`, `step_dt`, `truncated_by_max_steps`, `wall_time_s`,
`torch_max_memory_allocated_mb`, `git`), plus `task`, `model_name`, `policy` (`checkpoint`, `random_init` or
`zero`), `num_rows`, `num_cols`, `terrain_geoms`, `com_dr`, `termination_terms`, `env_build_s`, `device`,
`cuda_device_used_mb_before` / `cuda_device_used_mb_peak` (whole device, includes Warp), and `host_max_rss_mb`.
`height_noise` here is `{height_sigma, height_offset, drift_cells, delay_steps}`.

## Validity guard

The run stops, writes `summary.json` with `"valid": false` and an `invalid_reason` (no `per_env.csv`), prints
`[ERROR] INVALID RUN` and exits with code 2 if:

- **spawn check** (after the first step): a reset position is more than 5 cm from its tile center, a robot moved
  more than 10 cm, or the median base height above the support is outside 0.5-1.1 m (robots under or inside the
  terrain);
- more than 50 % of envs fail by `height` within the first second (robots fell through the terrain);
- any env succeeds faster than 3 m/s on average (a teleport or reset bug);
- with at most one robot per tile, the number of distinct tiles is not `num_envs`.

Also check valid runs: `median_height_above_support_m` near 0.75, `truncated_by_max_steps` false.
`aggregate_sweep.py` skips invalid, truncated, random-init and zero-policy runs.

## Board size and memory

The board size (number of terrain geoms) is the limit, not the number of envs. MuJoCo-Warp builds a table of all
geom pairs on the host while the env is built, and both host RAM and GPU memory grow roughly with the square of the
geom count. A stepping-stone tile has about 166 boxes at difficulty 0.5. Measured on the RTX 5060 Laptop (8 GB)
with 30 GB host RAM, one robot per tile, build under a 27 GB RAM cap without swap:

| board | terrain geoms | host RAM peak | GPU used by the run | build time |
|---|---|---|---|---|
| 8 x 8 (default) | 10.6 k | 9 GB | 1.3 GB | 20-45 s |
| 10 x 10 | 16.6 k | 20 GB | 2.8 GB | 50 s |
| 11 x 11 | 20.1 k | 26 GB | 3.9 GB | 77 s |
| 12 x 12 | 23.9 k | 28 GB | 5.4 GB | 2 min |
| 13 x 13 | 28 k | 29 GB | out of GPU memory | fails |

12 x 12 is the largest board that loads here, with almost no host RAM left; 10 x 10 is the largest comfortable
one. For more samples, pool seeds (each seed is a new board) rather than enlarging the board.

The number of envs costs little: on the 8 x 8 board (`bash scripts/eval_memory_test.sh`, random-init policy):

| envs | GPU used by the run | torch peak | host RAM peak | wall time per step |
|---|---|---|---|---|
| 64 | 1.3 GB | 68 MB | 8.7 GB | 0.31 s |
| 128 | 1.5 GB | 127 MB | 8.7 GB | 0.62 s |
| 225 | 1.9 GB | 212 MB | 8.7 GB | 1.2 s |

The step time grows with the number of envs (collision checks against the large board), so a full 1000-step
episode takes about 5 min with 64 envs. "GPU used by the run" is the device-level peak minus the memory in use
before the run. The training board (5.3 k geoms) is not affected.

## Differences from the Isaac Lab tool

- **Commands**: pure twist (`vx, vy, wz`), no heading control; robots spawn facing +x.
- **Terminations**: three non-time-out terms instead of one. `fail_reason` has the extra values `orientation` and
  `blowup`, and `termination_term` names the term. Illegal contact uses the current-step net force (no history) and
  counts self-contact.
- **Map noise**: drift moves the z values by whole cells (the training model of map drift) instead of adding an x-y
  offset to the coordinate channels; the option is `--drift_cells` instead of `--xy_shift`.
- **Validity**: no PhysX cooking check (there is no mesh cooking in MuJoCo); a spawn check and a speed check
  instead. Exit code 2 on an invalid run.
- **Displacement** includes the terminal step of a terminated env (Isaac Lab reset the robot before it could be
  read).
- **Board**: 8 x 8 by default instead of 15 x 15 (memory, see above). `--viz` is not ported; use `uv run play`
  to watch a policy.
- **Randomization**: friction U(0.3, 1.2) on the foot geoms and encoder bias, as in mjlab's play mode.

## Limitations

- **Only stepping-stone tiles were tested.** Other families are accepted. Like in the Isaac Lab tool, the
  supporting-surface fallback can misjudge stairs.
- **Tiles differ only for difficulty > 0** (random stone size, position and height scale with difficulty); at
  difficulty 0 all tiles are identical and `n` overstates the independent terrain samples. With more envs than
  tiles, robots on the same tile are not independent samples either.
- **Statistics** treat envs as independent; runs with the same seed share the board, so comparisons between
  conditions are paired and unpaired CIs are conservative.
- **Not fully deterministic**: MuJoCo-Warp is not bit-reproducible on GPU. Treat a run as one sample and compare
  conditions over several seeds.
- **Broadphase**: with this many geoms MuJoCo-Warp switches from the all-pairs broadphase used on the training
  board to sweep-and-prune. Contacts should be the same, but this was not compared directly.
- **`--map_delay_steps` holds the robot-centric map**: between refreshes the policy sees the old map in the old body
  frame, as in the Isaac Lab tool. A real low-rate map would be re-projected to the current pose.
- **Success does not check direction**; a robot that turns and walks 4 m sideways also succeeds.
