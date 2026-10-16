# ame_mjlab — AME perceptive locomotion for the Unitree G1 on mjlab

A port of [SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion)
(a reproduction of *Attention-Based Map Encoding for Learning Generalized
Legged Locomotion*) from Isaac Lab / Isaac Sim to
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp).  It trains a
twist-commanded (`vx, vy, wz`) G1 policy that walks stairs, steps, slopes,
rough ground and stepping stones from an elevation scan.  It also includes
**DELTA**, a deformable-attention alternative to the AME encoder.

## Requirements

- Linux with an NVIDIA GPU, **≥ 12 GB VRAM** for the default settings (see
  [GPU memory](#gpu-memory)), and a driver that supports CUDA 12.8.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).  It installs
  Python 3.13 and every dependency from the pinned `uv.lock`.

## Install

```bash
git clone <this repo> ame_mjlab && cd ame_mjlab
uv sync                 # creates .venv with mjlab, torch (cu128), warp, rsl-rl
uv run pytest           # a few seconds; should report 49 passed, 1 skipped
uv run list-envs        # should list the five tasks below
```

## Train and play

```bash
# Train (logs to Weights & Biases by default; run `uv run wandb login` first)
uv run train Ame-Velocity-Rough-G1

# ...or log to TensorBoard instead
uv run train Ame-Velocity-Rough-G1 --agent.logger tensorboard
uv run tensorboard --logdir logs/rsl_rl

# Watch a checkpoint (saved every 200 iterations)
uv run play Ame-Velocity-Rough-G1 \
  --checkpoint-file logs/rsl_rl/ame_g1/<run-dir>/model_<iter>.pt
```

Any config field can be overridden on the command line, e.g.
`--env.scene.num-envs 2048`, `--agent.algorithm.num-mini-batches 32`,
`--agent.max-iterations 5000`, `--agent.run-name my_try`.  Use
`uv run train <task> --help` for the full list.

### Tasks

| task | what it is |
|---|---|
| `Ame-Velocity-Rough-G1` | **start here** — dense AME encoder, full training recipe |
| `Delta-Velocity-Rough-G1` | same env, DELTA encoder instead of AME (an encoder A/B) |
| `Ame-Stage1-Velocity-Rough-G1` | SII-FUSC two-stage schedule, stage 1: clean perception, no pushes, gap-heavy board |
| `Ame-Stage2-Velocity-Rough-G1` | stage 2: specialist board (stakes, beams, gaps, stairs) with a noisy map; resume from a stage-1 checkpoint |
| `Ame-Mjlab-Gap-Velocity-Rough-G1` | gap board under mjlab's G1 rewards, deployment actor observations (see below) |

### The gap-board task

`Ame-Mjlab-Gap-Velocity-Rough-G1` (`ame.env_cfg.ame_gap_env_cfg`) is the
dense AME encoder trained from scratch on a board built to teach gap
crossing.  Its best checkpoint walks at the commanded speed and crosses
0.20-0.30 m gaps at 0.43-0.51 m/s, but turns poorly (0.07 rad/s for a
0.8 rad/s command).

- **Terrain** (`gap_board_cfg`): 40 % lattice of 0.40 m stones, gap =
  0.45 m × difficulty (row 0 is a seamless floor, +5 cm per row); 40 % stairs
  up and down; 10 % waves; 10 % rough.
- **Spawn**: a random flat patch of the tile (a stone centre on the lattice)
  with a lattice-aligned heading; the patch becomes the env origin.
- **Curriculum** (`GAP_CURRICULUM`): spawn on rows 0-2, never below row 0; a
  timed-out episode promotes only if the pelvis covered 60 % of the commanded
  speed, and a stall or an early fall demotes.
- **Rewards and randomisation**: mjlab's G1 velocity task as shipped.
- **Observations**: the actor drops `base_lin_vel` (96-D proprio + scan); the
  critic keeps it.
- **Metrics**: `Episode_Metrics/pelvis_{vel_error,yaw_error,speed}` are true
  per-step means; judge tracking by these, not by `Metrics/twist/*`.

The reference run used 2048 envs and 8 minibatches (needs more than 16 GB):

```bash
uv run train Ame-Mjlab-Gap-Velocity-Rough-G1 --env.scene.num-envs 2048 \
  --agent.max-iterations 15000 --agent.algorithm.num-mini-batches 8 --agent.seed 42
uv run play Ame-Mjlab-Gap-Velocity-Rough-G1 \
  --checkpoint-file logs/rsl_rl/ame_g1/<run-dir>/model_14999.pt --num-envs 16
```

## GPU memory

Defaults are sized for a 16 GB card: **1024 envs, 16 PPO minibatches**.
Measured on an RTX 5080 (`Ame-Velocity-Rough-G1`, training process only):

| envs | minibatches | GPU memory | s / iteration |
|---|---|---|---|
| 2048 | 8 | out of memory on 16 GB | — |
| 1024 | 8 | 12.3 GB | 4.4 |
| **1024** | **16** | **9.6 GB** (default) | **4.3** |

Two things use the memory:

- **The simulator** takes about 1 GB plus about 5.7 MB per env.  The terrain
  board has ~5300 collision geoms, so contact buffers are large.  Only
  `num_envs` changes this.
- **The PPO update.**  Symmetry augmentation doubles every minibatch, and the
  CNN activations on it dominate.  More minibatches means a smaller footprint
  at almost the same speed.

On a bigger card, raise `--env.scene.num-envs` first (more samples per
iteration means steadier PPO).  Scale `--agent.algorithm.num-mini-batches`
with it: 2048 / 32 keeps the same minibatch size.  If you hit
`Warp CUDA error 2: out of memory`, lower the env count.

## How it works

**Observations.**  The policy sees proprioception (base linear and angular
velocity, gravity, joint positions and velocities, last action, the twist
command) and an **elevation scan**.  Base linear velocity is exact in
simulation but has no reliable onboard estimate.  For a policy meant for
hardware, drop it from the actor with
`ame.env_cfg.apply_deploy_observations(cfg)`; the symmetry mirror handles
both layouts.  The scan is 33 × 21 points over 1.6 m × 1.0 m at
5 cm, yaw-aligned with the robot.  Each point is a 3-D coordinate: x/y are
the static grid offsets, and z is the height relative to the sensor, clamped
to `[-1.2, 0]`.  A ray miss reads as −1.2 m ("deep unknown").  During
training the scan gets per-point noise (σ 3 cm), a per-episode height offset
(±5 cm), and a per-episode lateral drift of up to one cell, so the policy
learns margins from edges instead of trusting the map to the centimetre.  The
critic additionally sees per-foot height, air time, contact and contact
forces.

**AME encoder** (`models.py`).  A small CNN turns the scan into a grid of
64-d tokens.  Proprioception is embedded into one query, which attends over
the tokens with 16-head attention.  The attended feature, concatenated with
the raw proprioception, feeds the actor and critic MLPs (512-256-128).  The
actions are joint-position targets for all 29 DoF.  The encoder is shared
between actor and critic.

**DELTA encoder** (`delta.py`, [arXiv:2608.22033](https://arxiv.org/abs/2608.22033)).
Instead of a dense CNN over all 693 cells, each of 4 heads samples 8
state-conditioned locations: base-grid references plus bounded learned
offsets, a coarse-to-fine scout patch, and a gated centre/context patch.
That is 32 tokens per layer, 3 layers.  The paper reports its biggest gains
on stepping stones, the family AME learns worst here.

**Terrain** (`terrains.py`).  A 10 × 20 board of 8 m tiles with a
difficulty curriculum by row.  It mixes 12 families: flat, pyramid stairs up
and down, random and open stairs, random grid, step-over obstacles, rough
noise, slopes up and down, waves, and stepping stones over a 2 m pit.  Robots
spawn on the two easiest rows and are promoted as they succeed.

**Rewards** (`reward_cfg.py`, `mdp/rewards.py`).  The AME-2 reward stack is
reproduced term for term.  It covers velocity tracking on the torso,
effort/limit penalties, foot style (air time, slide, stumble, feet too
close), arm/leg coordination, and joint deviation.  It adds two bounded
posture penalties `1 − exp(−e²/σ²)`: upper body at its default pose, and foot
yaw aligned with the base.  Both are scaled down on rough terrain, judged
from the robot's own scan.  The default `ame2_rebalanced` stack then changes
a few weights.  Without that, the verbatim weights learn to turn but not to
walk; each change is explained in `REBALANCE`.  Episodes end on illegal
(non-foot) contact, a flipped base, or a physics blowup.

**PPO** (`rl_cfg.py`).  These are the reference hyperparameters: 24
steps/env, 5 epochs, lr 1e-3 adaptive, entropy 0.008, γ 0.99, λ 0.95.  The
registered tasks add left/right **symmetry augmentation** (`symmetry.py`)
with `desired_kl = 0.02`.

## Code layout

A recipe, not a monolithic env: each piece is an `apply_*(cfg)` function
that mutates mjlab's G1 rough-terrain velocity task, so you can mix them
onto other tasks.

| file | contents |
|---|---|
| `src/ame/__init__.py` | registers the five tasks with mjlab |
| `src/ame/env_cfg.py` | `ame_rough_env_cfg` / `ame_stage_env_cfg` and the `apply_*` mutators; `NUM_ENVS` |
| `src/ame/rl_cfg.py` | PPO runner configs and model configs; `NUM_MINI_BATCHES` |
| `src/ame/scan.py` | scan geometry and the `height_map` observation group |
| `src/ame/terrains.py` | terrain mixes, presets, and extra boards (stage boards, a stones board, a gap-curriculum board) |
| `src/ame/models.py` | `TerrainAttention`, `AmeModel` (actor/critic), `TerrainContext` (the encoder as a branch for another actor) |
| `src/ame/delta.py` | `DeltaModel` |
| `src/ame/symmetry.py` | the left/right mirror of observations and actions |
| `src/ame/mdp/` | observation (`elevation_map`), reward, termination, patch-spawn event, floor curriculum and tracking-metric terms |
| `tests/` | unit tests; no simulator rollouts, so they run in seconds |

## Differences from the reference

| what | reference | here | why |
|---|---|---|---|
| simulator | Isaac Sim / PhysX | MuJoCo-Warp (mjlab) | the point of the port |
| commands | heading-driven, `vy = 0`, vx ∈ (0, 1.5) | pure twist: vx ∈ (−0.3, 1.5), vy ∈ ±0.5, wz ∈ ±1 | usable by a planner that outputs a twist |
| gap terrain | `HfGap` | stepping stones over a 2 m pit | mjlab has no gap primitive; a missed stone gives the same pressure |
| extra terrain | — | step-over obstacles, irregular and open stairs, waves | diversity |
| spawn rows | 0–5 of 10 | 0–1 | row 5 of this mix is not standable from scratch |
| reward weights | verbatim | rebalanced (`REBALANCE`) | the verbatim weights did not learn to walk here |
| rsl-rl | custom fork | stock rsl-rl 5.4; the model is loaded by import path (`ame.models:AmeModel`) | no fork needed |
| map plumbing | sliced off the end of the obs vector | its own observation group `height_map` | same tensors, less fragile |

Pretrained checkpoints from the reference are **not** usable here: the
simulator, actuator model and observation normalisation all differ.

## Tips

- Training from scratch takes thousands of iterations (`max_iterations` is
  10 000).  Check the `Curriculum/terrain_levels/*` curves to see which
  terrain families the policy is mastering.
- `ame_rough_env_cfg(scan_noise=False)` gives a clean-perception variant
  (stage 1 of the two-stage recipe that AME, GLAD and TAGA all use).
- `ame_rough_env_cfg(terrain_proportions=STONES_HEAVY)` (or
  `STONES_FINETUNE`) shifts the mix toward stepping stones.
- `rewards="mjlab"` swaps in mjlab's stock G1 velocity rewards for
  comparison.

## References

- Reference implementation: `SII-FUSC/AME_Locomotion` (master, 2026-08-25).
  Files ported: `rsl_rl/modules/actor_critic_encoder.py`,
  `29dof/velocity_env_cfg_29dof.py`, `mdp/observations.py`, `mdp/rewards.py`,
  `agents/ame_rsl_rl_ppo_cfg.py`, `terrains/terrain_cfg.py`.
- Chong Zhang's [AME tuning guide](https://zita-ch.github.io/tech-blog/):
  the source of the shared encoder, symmetry augmentation, torso-frame
  tracking, terrain-aware posture weights, and map drift.
- DELTA: Park, Jung & Hwangbo, [arXiv:2608.22033](https://arxiv.org/abs/2608.22033).
