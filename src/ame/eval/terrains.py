"""Evaluation-only terrain.  Not used by any training or play config.

``stone_lattice`` is the gap task's stone lattice
(:class:`ame.terrains.BoxStoneLatticeTerrainCfg`, the ``stepping_stones`` family
of :func:`ame.terrains.gap_board_cfg`: 0.40 m flush stones over a 2 m pit,
gap = 0.45 * difficulty) with one change: the spawn origin is the centre of
the stone nearest the tile centre instead of the tile centre itself.  The
lattice is centred on the tile, so with an even number of stones per side
(difficulty 0.25 and 1.0 on the 8 m tile) the tile centre is a gap crossing;
the evaluation spawns every robot at its tile origin with yaw 0, so it would
start over the pit.  Ties (four stones equally near) go to the lowest x, then
the lowest y: the most room ahead of a robot walking along +x.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainOutput

from ame.terrains import BoxStoneLatticeTerrainCfg, gap_board_cfg


@dataclass(kw_only=True)
class EvalStoneLatticeTerrainCfg(BoxStoneLatticeTerrainCfg):
  """The gap-board lattice with the origin on the stone nearest the centre."""

  def spawn_stone(self, difficulty: float) -> np.ndarray:
    """Centre ``(x, y)`` of the stone nearest the tile centre (tile frame)."""
    grid, _ = self.centres(difficulty)
    dist = np.linalg.norm(grid - np.asarray(self.size) / 2, axis=1).round(9)
    near = grid[dist == dist.min()]
    return near[np.lexsort((near[:, 1], near[:, 0]))[0]]

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,  # pyright: ignore[reportAttributeAccessIssue]
    rng: np.random.Generator,
  ) -> TerrainOutput:
    out = super().function(difficulty, spec, rng)
    out.origin = np.array([*self.spawn_stone(difficulty), 0.0])
    return out


def stone_lattice_cfg() -> EvalStoneLatticeTerrainCfg:
  """The gap board's lattice family (same stones, gaps and tile), no spawn
  patches (the evaluation spawns at the origin), proportion 1."""
  src = gap_board_cfg().sub_terrains["stepping_stones"]
  assert isinstance(src, BoxStoneLatticeTerrainCfg)
  kw = {f.name: getattr(src, f.name) for f in fields(src)}
  kw.update(proportion=1.0, flat_patch_sampling=None)
  return EvalStoneLatticeTerrainCfg(**kw)


__all__ = ["EvalStoneLatticeTerrainCfg", "stone_lattice_cfg"]
