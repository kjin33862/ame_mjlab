# Can the gap policy see the gaps? (2026-11-13)

Task: `Ame-Mjlab-Gap-Edge-SGS-AngW4Z-Velocity-Rough-G1` (repo `~/ame_mjlab_sgs`, branch `sgs`).
Everything here ran on CPU (`CUDA_VISIBLE_DEVICES=""`). No GPU job, no policy, no training. No training code was changed.

Short answer: **yes.** In the actor's input, a gap reads as a flat −1.20 m; a stone top reads about −0.76 m. That 0.44 m step is about 10 times the per-point noise. Noise never makes a gap look like a stone in our test.

What the map does not give the policy:
- **Depth:** the clip hides how deep a gap is.
- **Exact edges:** the training-time drift shifts gap edges by up to ±5 cm.
- **Gaps under 5 cm:** they can fall between two rays.

Probe script: `notes/gap_perception_probe.py`. Raw numbers: `notes/gap_perception/probe.json`. Figures and arrays: `notes/gap_perception/*.png`, `*.npz`.

## 1. The actor's height-map sensor and every step before the actor

Sensor (mjlab `terrain_scan`, refined by `ame.scan.apply_scan`; the train and play configs match except for noise):
- **Type and frame:** `RayCastSensorCfg` attached to the **pelvis** body (`mjlab/tasks/velocity/config/g1/env_cfgs.py` sets `frame.name = "pelvis"`).
- **Pattern:** `GridPatternCfg(size=(1.6, 1.0), resolution=0.05)`, a grid of **33 × 21 = 693 rays at 5 cm spacing**.
- **Coverage around the pelvis:** x from −0.80 to +0.80 m and y from −0.50 to +0.50 m. That is 0.80 m ahead, 0.80 m behind and 0.50 m to each side, centred on the pelvis.
- **Alignment:** `ray_alignment="yaw"`. The grid turns with the robot's heading, ignores roll and pitch, and the rays point straight down.
- **Ray start height:** rays start at the pelvis origin itself; there is no offset. Measured: sensor `pos_w` minus pelvis position = (0, 0, 0). In the default standing pose the pelvis is 0.76 m above the ground under the feet.
- **Max distance:** 5.0 m. mjlab reports anything farther as a miss.
- **What it can hit:** geometry group 0 only (terrain), with the parent body excluded. The probe counted 0 robot geoms in group 0, so the scan never sees the robot's own legs.
- **What a miss returns:** mjlab sets `distance = −1` and puts the hit point at the ray origin. `elevation_map` then replaces a miss with **−1.2**, the clip floor, so a miss reads like a deep hole. (It also treats `distance ≥ 0.999 × max_distance` as a miss.)

Observation (`ame.mdp.observations.elevation_map`, its own group `height_map`, shape 2079 = 693 × 3), in order:
- **x, y channels:** the fixed grid offsets in the yaw frame, computed once from the pattern. They don't depend on the hits, and noise never touches them; checked in the probe.
- **z:** `hit_z − pelvis_z`, in metres. There is no base-height subtraction other than this pelvis-relative offset, and no scaling. So z is −(pelvis height above the terrain under that ray).
- **Misses:** set to −1.2, as above.
- **Noise (training only; the play config passes `noise=False`):**
  - **Lateral map drift:** the z map is shifted by (dx, dy) ∈ {−1, 0, +1} cells, drawn per episode, with edge cells repeated. That is up to ±5 cm. Only z moves; the x, y channels stay put.
  - **Per-point noise:** Gaussian, σ = 0.03 m, every step.
  - **Per-episode offset:** uniform ±0.05 m added to all heights.
- **Clip:** `clamp(z, −1.2, 0.0)`. This happens **after** the noise.
- **Nothing else:** the term has no scale, clip or noise settings of its own, `enable_corruption` is False, and there is no history or delay. The model has `obs_normalization=False` (`AmeModel` refuses it), so the map reaches the network exactly as above.
- **Inside the model** (for completeness, not preprocessing):
  - The map is reshaped to (W = 21, L = 33, 3).
  - A 5 × 5 convolution with stride 2 runs first, then a 3 × 3 one, each followed by BatchNorm, giving tokens on a 10 cm grid. Multi-head attention runs over those tokens.
  - In training only, symmetry augmentation adds a mirrored copy of the map (`ame.symmetry.mirror_map`).
- **Freshness:** in `env.step` the ray cast (`sim.sense()`) runs right before the observation is computed. In this task nothing reads `terrain_scan` earlier in the step, so the actor gets the current pose's map, with no one-step lag.

Comparison with the original AME (SII-FUSC `AME_Locomotion`, local copy at `~/AME_Locomotion`):
- **Grid:** identical, 1.6 × 1.0 m at 0.05 m (33 × 21), yaw-aligned, x/y/z channels, z clamped to [−1.2, 0].
- **Frame:** **torso_link**, not the pelvis. In the G1 default pose it sits 0.044 m above the pelvis. So in the reference a stone top under a standing robot reads about −0.80 m (ours −0.76 m), and a gap reads −1.2 m in both.
- **Ray start:** an offset of +20 m. Rays start 20 m above the torso and go down with IsaacLab's default max distance of 1e6 m. z is still measured from the torso, since the offset is not part of `pos_w`. So terrain above the torso is visible in the reference; ours cannot see anything above the pelvis. That makes no difference on gap tiles.
- **Misses:** raycast_mesh returns +inf, `nan_to_num` turns it into a huge number, and the clamp gives **0.0** ("terrain at sensor height"). Ours gives **−1.2**. The reference ground mesh is closed, so in practice neither side gets misses on these tiles.
- **Noise:** the same σ = 0.03 m and ±0.05 m offset, also applied before the clip. Differences:
  - **Offset timing:** the reference resamples the offset only when `reset_buf` is set, so it is zero for an env's first episode.
  - **Shift and delay:** the reference's ±3 cm x/y shift and its 10 Hz delay are both commented out.
  - **Drift:** our ±1-cell z-map drift does not exist in the reference. It is ours, from the AME tuning guide.
- **Stage 1** in the reference uses `noise=False` for the actor (see `~/AME_Locomotion/notes/paper_vs_code.md`).
- **AME-2 official config:** not found locally. `pretrained/ame2.pt` in `AME_Locomotion` is the same SII-FUSC environment with the global-context encoder (`attach_global=True`); its map settings are identical to AME-1 above. `paper_vs_code.md` lists the paper's map spec as "to verify". I did not compare against the AME-2 paper text.

## 2. Terrain: the gaps are real pits

Both families are built from MuJoCo box geoms (no height field). Read from the compiled model:
- **`stone_lattice`** (the gap board's `stepping_stones` family, `BoxStoneLatticeTerrainCfg`):
  - Stone tops are all at z = 0.000.
  - Each stone is a column with its bottom at −2.0 m, over a pit-floor box whose top is at **z = −2.000** (bottom −2.1).
  - Pit depth **2.0 m** at every level with a gap.
  - At d = 0.25 the gap is 0.45 × 0.25 = **0.1125 m**.
- **`stepping_stones`** (the AME mix family, mjlab `BoxSteppingStonesTerrainCfg`):
  - Stone tops are at `stone_height` 0.20 m ± 0.025 at d = 0.5. Measured column tops range from 0.000 to 0.225 m; the 0.000 is the outer board border.
  - Pit-floor top at **z = −2.000**, so the pit is about **2.2 m** below the stone tops.
  - At d = 0.5 the nominal gap is 0.15 m, but ±0.05 m stone displacement and size variation leave some cracks only about 5 cm wide.
  - This family is in the AME mix, not on the gap board this task trains on.

What the rays hit over a gap (probe, every gap ray):
- **All gap rays:** they hit the pit-floor top at z = −2.000 m with normal z = +1.
- **Ray length:** 2.76 m on the lattice (pelvis at 0.76 m) and 2.96 m on stepping_stones (pelvis at 0.96 m, standing on a 0.20 m platform).
- **Misses:** 0 of 693 on both tiles; the pit floor is well inside the 5 m max distance.
- **Stone sides:** vertical rays never hit them. Each ray lands either on a stone top or on the pit floor, so no value falls in between and there is no smoothing.

## 3. Robot next to a gap: the observation vs the true terrain

Setup:
- **Env:** the play config of the task on one tile (seed 0).
- **Placement:** the robot is placed directly in the default joint pose, at yaw 0, with the pelvis 0.76 m above the ground under it. The pipeline then runs exactly as in `env.step`: `sim.forward` → `scene.update` → `sim.sense` → `observation_manager.compute(update_history=True)`.
- **Noise off:** the observation-manager output, which is what play and eval feed the actor. It equals a direct `elevation_map(noise=False)` call; the probe asserts this.
- **Training noise:** `elevation_map(noise=True)`, the training config's function and settings, with the per-episode constants redrawn 500 times. I also ran 500 draws with the drift held at zero, to separate the two noise sources.
- **Truth:** `mj_ray` from 10 m above the ground at the same 693 (x, y) points, terrain only, not clipped, given relative to the pelvis.

Placements:
- **`stone_lattice` d0.25:** on the stone nearest the tile centre. The stone's front edge is 0.16–0.18 m ahead of the pelvis, and the gap runs from there to about 0.28 m. The map shows 3 × 3 stones and the crossing gaps.
- **`stepping_stones` d0.5:** at the +x edge of the 2 m central platform, toes about 1 cm short of the edge. Here the pits start 0.18 m ahead at y from +0.2 to +0.4 and from −0.2 to −0.35. Straight ahead there is a 5 cm crack at x ≈ 0.70 m.

Figures (left: true terrain, unclipped; middle: observation with noise off; right: one training-noise draw; + marks the pelvis):
- `gap_perception/stone_lattice_d025_heatmaps.png`
- `gap_perception/stepping_stones_d050_heatmaps.png`
- `gap_perception/stone_lattice_d025_profile.png`: centre row (y = 0), showing the truth, the noise-off observation, one noisy draw and the 5–95 % band.
- `gap_perception/stepping_stones_d050_profile.png`

Measured z (metres, relative to the pelvis):
- **stone_lattice d0.25** (375 stone points, 318 gap points):
  - **True terrain:** stones −0.760, gaps −2.760.
  - **Noise off:** stones −0.760, exactly the truth (largest difference 0). Gaps **−1.200** at every point, clipped.
  - **Training noise, stones:** mean −0.838, std 0.171, 1st–99th percentile −1.20 to −0.67. The −1.20 tail is drift moving gap cells onto stone positions.
  - **Training noise, gaps:** mean −1.109, 1st–99th percentile −1.20 to −0.69.
  - **Height noise only (drift held at zero):** stones mean −0.762, std 0.042, range −0.915 to −0.587. Gaps **exactly −1.200 in all 500 draws.** The pit is 1.56 m below the clip floor, so the clip absorbs the ±0.15 m of noise and offset completely.
  - **Mislabelled cells, split at −0.98 m:** with drift, 17.5 % of stone cells and 20.7 % of gap cells read on the wrong side compared with the true terrain under them. With drift held at zero, **0 %** of both. All of the mixing comes from the 1-cell edge shift, none from height noise.
  - **Gap width in the centre row:** 2–3 samples for the 0.1125 m gap (runs of 3, 2, 2; plus 1 sample cut off at the map's back edge).
- **stepping_stones d0.5** (571 stone points, 122 gap points):
  - **True terrain:** stones −0.743 to −0.779 (the ±2.5 cm stone-height variation), gaps −2.960.
  - **Noise off:** stones the same as the truth; gaps −1.200 at every point.
  - **Training noise, stones:** mean −0.780, std 0.100. **Gaps:** mean −1.106.
  - **Height noise only:** stones std 0.042; gaps exactly −1.200.
  - **Mislabelled cells:** with drift, 4.5 % of stone and 21 % of gap cells; with drift held at zero, 0 %.
  - **The 5 cm crack straight ahead** shows up as a single −1.2 sample in the centre row.

## 4. Conclusion

- **Is a gap distinguishable from a stone?** Yes, clearly:
  - **Contrast:** gap −1.20 vs stone −0.76, a 0.44 m step (in the reference, about −0.80 vs −1.20, 0.40 m).
  - **Noise:** the step is about 10σ of the per-point noise. Even adding the worst offset, no draw pushed a stone below −0.93 m or lifted a pit cell above −1.20 m.
  - **Under noise:** a pit cell has zero variance, a flat −1.2. The noise is added before the clip, and the pit is 1.5–1.8 m below the floor value.
- **Resolution:**
  - **Spacing:** 5 cm along the ground, with the map reaching 0.8 m ahead.
  - **Gaps of 0.10 m or more** (lattice d ≥ 0.22, i.e. curriculum level 2 and up) always land 2 or more samples per row crossing them.
  - **A 5 cm gap** (lattice level 1, d = 0.11, and some stepping_stones cracks) lands 0 or 1 sample, depending on where it falls relative to the grid. It can be invisible in some rows and poses.
  - **Network:** the stride-2 first convolution gives 10 cm tokens, but its 5 × 5 kernel covers every sample, so a 1-sample gap still enters the features.
- **Clipped?** Yes, depth is clipped:
  - **Pits:** the true −2.76 / −2.96 m reads as −1.2. The policy knows "deep" but not how deep. That is irrelevant on these boards, since every pit is 2 m.
  - **Stairs:** the clip floor is only 0.44 m below the standing ground, so a big enough drop also hits −1.2. On `pyramid_stairs` (down) with steps of 0.15 m or more (level 6 and up), cells three steps down, 0.6 m or more ahead, read −1.2 exactly like a pit. This is arithmetic from the stair geometry; I did not probe a stairs tile.
  - **Above:** the 0.0 ceiling (terrain at pelvis height) is never reached on these tiles.
- **Filled?** Nothing is filled or interpolated:
  - Each sample is one vertical ray's hit, on either a stone top or the pit floor.
  - Misses would also read −1.2, but there were none: the pit floor is inside the 5 m range.
- **Where the uncertainty is, by design:** the training drift moves the whole z map by up to one cell (±5 cm). The fixed x, y channels don't move with it, so in about a fifth of draws a gap edge is off by one cell (17–21 % of near-edge cells on the lattice). That is a registration error, not a visibility problem.

So if the policy steps into gaps, it is not because it can't see them. What it does get wrong is edge position, by up to 5 cm under training noise. Gaps under 5 cm can drop out entirely.

## Method notes

- **Pitfalls in the probe itself, not the env.** Two things gave stale values until fixed; neither affects training or eval, because `env.step` does both correctly:
  - mjlab caches sensor data until `scene.update` (or a reset) clears the cache.
  - `ObservationManager.compute()` without `update_history=True` returns its buffer from the last reset.
- **Static pose.** The placement is static (default pose, no physics). Pelvis height while walking varies by a few cm; that moves the stone value, not the −1.2 gap value.
- Rerun with `cd ~/ame_mjlab_sgs && CUDA_VISIBLE_DEVICES="" uv run python notes/gap_perception_probe.py` (about 1 min on CPU).
