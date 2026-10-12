"""Terrain mixes: the AME base mix, its presets, and the SII-FUSC stage boards."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mjlab.terrains as terrain_gen
import mujoco
import numpy as np
from mjlab.terrains.config import (
  discrete_obstacles,
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  open_stairs,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
  random_stairs,
  stepping_stones,
)
from mjlab.terrains.config import wave_terrain as wave
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_border

TILE = (8.0, 8.0)
"""8 m cells on a 10 x 20 board, the reference's board."""

# Proportion presets for ``ame_rough_env_cfg(terrain_proportions=...)``.
# Stepping stones is the family the base mix learns worst, and the one the
# curriculum starves (it stays at low rows while the rest climb).
STONES_HEAVY = {
  # More stones exposure, taken from the families that saturate early.
  "stepping_stones": 0.20,
  "flat": 0.02,
  "wave": 0.02,
  "hf_pyramid_slope": 0.04,
  "hf_pyramid_slope_inv": 0.04,
  "open_stairs": 0.04,
}
STONES_FINETUNE = {
  # A stones-specialist mix for finetuning an already-competent policy.  In
  # one evaluation, falls on stones went from 0.25 at difficulty 0.25 to 0.98
  # at 0.75 while every other family stayed <= 0.19; at 11 k iterations the
  # other families sat at rows 4-6 of 9 and stones at 0.75.  Not stones-only:
  # every other family keeps a real share, because narrow finetuning tends
  # to degrade an all-terrain policy.
  "stepping_stones": 0.45,
  "pyramid_stairs": 0.08,
  "pyramid_stairs_inv": 0.08,
  "random_stairs": 0.05,
  "open_stairs": 0.03,
  "random_grid": 0.07,
  "discrete_obstacles": 0.08,
  "random_rough": 0.06,
  "hf_pyramid_slope": 0.03,
  "hf_pyramid_slope_inv": 0.03,
  "wave": 0.02,
  "flat": 0.02,
}


def _stones(
  proportion: float,
  size: tuple[float, float],
  distance: tuple[float, float],
  height_variation: float,
  regular: bool = False,
) -> terrain_gen.BoxSteppingStonesTerrainCfg:
  """``regular``: the reference's lattice — flush tops, no per-stone size or
  position jitter, so difficulty is the gap and the stone size alone."""
  cfg = terrain_gen.BoxSteppingStonesTerrainCfg(
    proportion=proportion,
    size=TILE,
    stone_size_range=size,
    stone_distance_range=distance,
    stone_height_variation=height_variation,
    platform_width=2.0,
  )
  if regular:
    cfg.stone_height = 0.0
    cfg.stone_size_variation = 0.0
    cfg.displacement_range = 0.0
  return cfg


def ame_terrains_cfg(
  proportions: dict[str, float] | None = None,
) -> TerrainGeneratorCfg:
  """The reference's eight families at its difficulty ranges plus four mjlab
  primitives it lacked (step-over obstacles, irregular and open stairs,
  waves).  The reference's gap terrain has no mjlab primitive; stepping
  stones over a 2 m pit apply the same commitment pressure.  ``proportions``
  overrides per family (see the presets above)."""
  gen = TerrainGeneratorCfg(
    size=TILE,
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
      "flat": flat(proportion=0.05),
      "pyramid_stairs": pyramid_stairs(
        proportion=0.12, step_height_range=(0.05, 0.20), step_width=0.30
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.12, step_height_range=(0.05, 0.20), step_width=0.30
      ),
      "random_stairs": random_stairs(proportion=0.08, step_height_range=(0.05, 0.25)),
      "open_stairs": open_stairs(proportion=0.05, step_height_range=(0.05, 0.20)),
      "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
        proportion=0.10,
        size=TILE,
        grid_width=0.45,
        grid_height_range=(0.05, 0.20),
        platform_width=2.0,
      ),
      "discrete_obstacles": discrete_obstacles(
        proportion=0.12,
        obstacle_height_range=(0.05, 0.30),
        obstacle_width_range=(0.3, 1.0),
      ),
      "random_rough": random_rough(
        proportion=0.10, noise_range=(0.02, 0.10), noise_step=0.02
      ),
      "hf_pyramid_slope": hf_pyramid_slope(proportion=0.06, slope_range=(0.0, 0.4)),
      "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
        proportion=0.06, slope_range=(0.0, 0.4)
      ),
      "wave": wave(proportion=0.04),
      "stepping_stones": stepping_stones(
        proportion=0.10,
        stone_size_range=(0.3, 0.6),
        stone_distance_range=(0.05, 0.25),
        stone_height_variation=0.05,
        platform_width=2.0,
      ),
    },
    add_lights=True,
  )
  for name, proportion in (proportions or {}).items():
    if name not in gen.sub_terrains:
      raise ValueError(f"unknown terrain family {name!r}")
    gen.sub_terrains[name].proportion = proportion
  return gen


def stage_terrains_cfg(stage: int) -> TerrainGeneratorCfg:
  """SII-FUSC's two-stage boards, proportions verbatim: stage 1 is 40 %
  holes and gaps, stage 2 the specialist mix (stakes, beams, gaps, stairs)."""
  if stage == 1:
    sub_terrains = {
      "pyramid_stairs": pyramid_stairs(
        proportion=0.10, step_height_range=(0.05, 0.20), step_width=0.30
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.10, step_height_range=(0.05, 0.20), step_width=0.30
      ),
      "boxes": terrain_gen.BoxRandomGridTerrainCfg(
        proportion=0.10,
        size=TILE,
        grid_width=0.45,
        grid_height_range=(0.05, 0.20),
        platform_width=2.0,
      ),
      "random_rough": random_rough(
        proportion=0.10, noise_range=(0.02, 0.10), noise_step=0.02
      ),
      "hf_pyramid_slope": hf_pyramid_slope(proportion=0.10, slope_range=(0.0, 0.4)),
      "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
        proportion=0.10, slope_range=(0.0, 0.4)
      ),
      "stepping_stones": _stones(0.20, (0.25, 0.50), (0.05, 0.25), 0.05),
      "gaps": _stones(0.20, (0.60, 1.00), (0.10, 0.50), 0.0),
    }
  elif stage == 2:
    sub_terrains = {
      "stakes": _stones(0.50, (0.20, 0.40), (0.10, 0.35), 0.03),
      "beams": terrain_gen.BoxNarrowBeamsTerrainCfg(
        proportion=0.20,
        size=TILE,
        beam_width_range=(0.15, 0.35),
        beam_height=0.03,
        spacing=0.8,
        platform_width=2.0,
      ),
      "gaps": _stones(0.10, (0.60, 1.00), (0.10, 0.50), 0.0),
      "pyramid_stairs": pyramid_stairs(
        proportion=0.10, step_height_range=(0.05, 0.25), step_width=0.30
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.10, step_height_range=(0.05, 0.25), step_width=0.30
      ),
    }
  else:
    raise ValueError(f"stage must be 1 or 2, got {stage}")
  return TerrainGeneratorCfg(
    size=TILE,
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains=sub_terrains,
    add_lights=True,
  )


def stones_board_cfg() -> TerrainGeneratorCfg:
  """A foothold board.  45 % stepping stones whose gaps a foot cannot bridge
  (0.15 m at difficulty 0 to 0.45 m at 1, stones 0.50 to 0.25 m) on the
  reference's regular flush lattice, 20 % ring gaps (the reference's
  trenches), and every other family kept alive against forgetting."""
  return TerrainGeneratorCfg(
    size=TILE,
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
      "stepping_stones": _stones(0.45, (0.25, 0.50), (0.15, 0.45), 0.05, regular=True),
      "gaps": terrain_gen.BoxNestedRingsTerrainCfg(
        proportion=0.20,
        size=TILE,
        num_rings=3,
        ring_width_range=(0.5, 0.5),
        gap_range=(0.1, 0.5),
        height_range=(0.0, 0.0),
        platform_width=2.0,
        floor_depth=2.0,
      ),
      "pyramid_stairs": pyramid_stairs(
        proportion=0.06, step_height_range=(0.05, 0.20), step_width=0.30
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.06, step_height_range=(0.05, 0.20), step_width=0.30
      ),
      "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
        proportion=0.08,
        size=TILE,
        grid_width=0.45,
        grid_height_range=(0.05, 0.20),
        platform_width=2.0,
      ),
      "random_rough": random_rough(
        proportion=0.08, noise_range=(0.02, 0.10), noise_step=0.02
      ),
      "hf_pyramid_slope": hf_pyramid_slope(proportion=0.035, slope_range=(0.0, 0.4)),
      "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
        proportion=0.035, slope_range=(0.0, 0.4)
      ),
    },
    add_lights=True,
  )


@dataclass(kw_only=True)
class BoxStoneLatticeTerrainCfg(SubTerrainCfg):
  """A uniform lattice of flush stones over a pit, the gap the only knob.

  Every stone is ``stone_size`` square with its top at z = 0; the gap between
  neighbours is ``gap_range`` interpolated by difficulty, so difficulty 0 is a
  continuous floor and each row asks for a slightly longer step than the
  last.  There is no centre platform: the lattice covers the whole tile, the
  border ring sits one gap outside the outermost stones, and spawns come
  from the ``spawn`` flat patches — stone centres at least ``spawn_margin``
  from the tile edge — so a robot starts on a stone anywhere on the tile
  rather than on an island at its centre.
  """

  stone_size: float = 0.40
  gap_range: tuple[float, float] = (0.0, 0.45)
  floor_depth: float = 2.0
  border_width: float = 0.25
  spawn_margin: float = 1.0

  def gap(self, difficulty: float) -> float:
    low, high = self.gap_range
    return low + difficulty * (high - low)

  def centres(self, difficulty: float) -> tuple[np.ndarray, float]:
    """Stone centres ``(n_x * n_y, 2)`` in the tile frame and the ring width."""
    gap = self.gap(difficulty)
    pitch = self.stone_size + gap
    # As many stones as leave a ring of at least ``border_width`` plus one gap
    # outside the lattice; the ring then fills what is left, so difficulty 0
    # is seamless from edge to edge.
    counts, offsets = [], []
    for extent in self.size:
      room = extent - 2 * (self.border_width + gap) - self.stone_size
      n = int(np.floor(room / pitch + 1e-9)) + 1
      counts.append(n)
      offsets.append((extent - (n - 1) * pitch) / 2)
    xs = offsets[0] + pitch * np.arange(counts[0])
    ys = offsets[1] + pitch * np.arange(counts[1])
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    ring = offsets[0] - self.stone_size / 2 - gap
    return grid, float(ring)

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,  # pyright: ignore[reportAttributeAccessIssue]
    rng: np.random.Generator,
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    grid, ring = self.centres(difficulty)
    half = self.stone_size / 2
    z_center = -self.floor_depth / 2
    ring_boxes = make_border(
      body,
      self.size,
      (self.size[0] - 2 * ring, self.size[1] - 2 * ring),
      self.floor_depth,
      (0.5 * self.size[0], 0.5 * self.size[1], z_center),
    )
    for geom in ring_boxes:
      geometries.append(TerrainGeometry(geom=geom, color=(0.35, 0.40, 0.30, 1.0)))
    floor_h = 0.1
    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,  # pyright: ignore[reportAttributeAccessIssue]
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor, color=(0.1, 0.1, 0.1, 1.0)))
    for cx, cy in grid:
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,  # pyright: ignore[reportAttributeAccessIssue]
        size=(half, half, self.floor_depth / 2),
        pos=(float(cx), float(cy), z_center),
      )
      shade = 0.45 + 0.15 * rng.uniform()
      geometries.append(
        TerrainGeometry(geom=geom, color=(shade, shade + 0.1, shade, 1.0))
      )
    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
    flat_patches = None
    if self.flat_patch_sampling is not None:
      edge = np.minimum(grid, np.asarray(self.size) - grid).min(axis=1)
      inner = grid[edge >= self.spawn_margin]
      if len(inner) == 0:
        inner = grid[[int(np.argmax(edge))]]
      flat_patches = {}
      for name, patch_cfg in self.flat_patch_sampling.items():
        n = patch_cfg.num_patches
        pick = rng.choice(len(inner), size=n, replace=len(inner) < n)
        flat_patches[name] = np.concatenate([inner[pick], np.zeros((n, 1))], axis=1)
    return TerrainOutput(
      origin=origin, geometries=geometries, flat_patches=flat_patches
    )


SPAWN_PATCHES = 64
"""Spawn candidates stored per sub-terrain (mjlab pads with the origin)."""


def gap_board_cfg(
  stone_size: float = 0.40, gap_range: tuple[float, float] = (0.0, 0.45)
) -> TerrainGeneratorCfg:
  """The gap-curriculum board: four families, the stone gap the difficulty.

  40 % stone lattice (:class:`BoxStoneLatticeTerrainCfg`: ``stone_size``
  stones, gap ``gap_range`` by row — a floor at row 0, 5 cm more per row),
  40 % stairs (up and down, 5-20 cm steps), 10 % waves and 10 % rough for
  variety.  Every family carries ``spawn`` flat patches so a robot starts
  anywhere on its tile, not at the centre: stone centres on the lattice,
  level ground on the height fields, the summit platform on the stairs.
  """
  spawn = {"spawn": FlatPatchSamplingCfg(num_patches=SPAWN_PATCHES)}
  level = lambda diff: {  # noqa: E731
    "spawn": FlatPatchSamplingCfg(
      num_patches=SPAWN_PATCHES,
      patch_radius=0.3,
      max_height_diff=diff,
      x_range=(1.0, TILE[0] - 1.0),
      y_range=(1.0, TILE[1] - 1.0),
    )
  }
  return TerrainGeneratorCfg(
    size=TILE,
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
      "stepping_stones": BoxStoneLatticeTerrainCfg(
        proportion=0.40,
        size=TILE,
        stone_size=stone_size,
        gap_range=gap_range,
        flat_patch_sampling=spawn,
      ),
      "pyramid_stairs": pyramid_stairs(
        proportion=0.20, step_height_range=(0.05, 0.20), step_width=0.30
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.20, step_height_range=(0.05, 0.20), step_width=0.30
      ),
      "wave": wave(proportion=0.10, flat_patch_sampling=level(0.08)),
      "random_rough": random_rough(
        proportion=0.10,
        noise_range=(0.02, 0.10),
        noise_step=0.02,
        flat_patch_sampling=level(0.12),
      ),
    },
    add_lights=True,
  )


def play_board(gen: TerrainGeneratorCfg) -> TerrainGeneratorCfg:
  """The base task's play-mode shrink: no curriculum, a small board."""
  return replace(gen, curriculum=False, num_rows=5, num_cols=5, border_width=10.0)


__all__ = [
  "SPAWN_PATCHES",
  "STONES_FINETUNE",
  "STONES_HEAVY",
  "TILE",
  "BoxStoneLatticeTerrainCfg",
  "ame_terrains_cfg",
  "gap_board_cfg",
  "play_board",
  "stage_terrains_cfg",
  "stones_board_cfg",
]
