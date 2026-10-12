"""The elevation scan: reference geometry and its sensor / observation wiring."""

from __future__ import annotations

from typing import Any

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.sensor import GridPatternCfg

from ame.mdp.observations import elevation_map

MAP_SIZE_M = (1.6, 1.0)
MAP_RESOLUTION_M = 0.05
MAP_SHAPE = (
  int(MAP_SIZE_M[0] / MAP_RESOLUTION_M) + 1,  # L = 33
  int(MAP_SIZE_M[1] / MAP_RESOLUTION_M) + 1,  # W = 21
  3,
)
"""``(L, W, 3)``: per-point yaw-frame coordinates, z relative to the sensor."""

MAP_GROUP = "height_map"


def apply_scan(cfg: Any, noise: bool) -> None:
  """Refine ``terrain_scan`` to the reference grid and expose it as its own
  observation group, replacing the base task's scalar ``height_scan``."""
  sensors = list(cfg.scene.sensors or ())
  for i, sensor in enumerate(sensors):
    if getattr(sensor, "name", None) == "terrain_scan":
      sensor.pattern = GridPatternCfg(size=MAP_SIZE_M, resolution=MAP_RESOLUTION_M)
      sensor.ray_alignment = "yaw"
      sensors[i] = sensor
      break
  else:
    raise ValueError("base task has no 'terrain_scan' sensor to refine")
  cfg.scene.sensors = tuple(sensors)
  for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("height_scan", None)
  cfg.observations[MAP_GROUP] = ObservationGroupCfg(
    terms={
      "elevation": ObservationTermCfg(
        func=elevation_map, params={"sensor_name": "terrain_scan", "noise": noise}
      )
    },
  )


__all__ = ["MAP_GROUP", "MAP_RESOLUTION_M", "MAP_SHAPE", "MAP_SIZE_M", "apply_scan"]
