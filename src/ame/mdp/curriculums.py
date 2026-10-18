"""Terrain curriculum with a floor row and a progress-gated promotion rule.

mjlab's ``terrain_levels_vel`` promotes an env that ends an episode more than
half a tile from its spawn and demotes one that covered less than half the
commanded distance.  Under twist arcs a robot circles, so promotion is rare
and levels degrade into a fall-rate readout.  This term keeps mjlab's rule and
logging, never demotes below ``floor``, and offers rules that judge an
episode by survival and by tracked progress instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class terrain_levels_floor:
  """mjlab's terrain curriculum with a floor row and a choice of signal.

  ``rule="endpoint"`` is mjlab's: an episode ending more than half a tile
  from spawn promotes, one covering less than half the commanded distance
  demotes.  That assumes straight-line commands; under twist arcs (yaw ±1,
  vy ±0.5, resampled every few seconds) a perfect tracker rarely displaces
  4 m, so levels degrade into a fall-rate readout and every family with
  terminal failures settles near the floor.
  ``rule="survival"`` promotes on a timed-out episode as well as on the
  4 m end point, and demotes an episode terminated in its first half.
  ``rule="tracking"`` is survival with a progress gate that holds under
  arcs: a timed-out episode promotes only if the robot's mean speed reached
  ``track_ratio`` of the mean commanded speed (the ``pelvis_speed`` and
  ``command_speed`` metrics), and demotes if it did not — a robot that
  survives by stalling on one stone is sent back down, not up.
  ``hold_steps``: no level changes before this env step, so a frozen actor
  (a critic warm-up) cannot be promoted onto terrain the learning policy
  has never met.
  """

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
    self.floor = int(cfg.params["floor"])
    self.rule = str(cfg.params.get("rule", "endpoint"))
    self.track_ratio = float(cfg.params.get("track_ratio", 0.6))
    self.hold_steps = int(cfg.params.get("hold_steps", 0))
    if self.rule not in ("endpoint", "survival", "tracking"):
      raise ValueError(
        f"rule must be endpoint, survival or tracking, got {self.rule!r}"
      )
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    if self.floor >= int(terrain.max_terrain_level):
      raise ValueError(
        f"terrain floor {self.floor} needs rows above it; the board has "
        f"{int(terrain.max_terrain_level)}"
      )
    # Spawn on the floor row or above: raise every env that drew a lower level.
    terrain.terrain_levels.clamp_(min=self.floor)
    terrain.env_origins[:] = terrain.terrain_origins[
      terrain.terrain_levels, terrain.terrain_types
    ]
    self._armed = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    floor: int,
    rule: str = "endpoint",
    track_ratio: float = 0.6,
    hold_steps: int = 0,
  ) -> dict[str, torch.Tensor]:
    del floor, rule, track_ratio, hold_steps  # resolved at construction
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    generator = terrain.cfg.terrain_generator
    assert generator is not None
    if not self._armed:
      # mjlab evaluates the curriculum at the very first reset, with robots
      # still at their construction poses far from any origin: a spurious +1
      # for the whole fleet.  Skip that one call.
      self._armed = True
      return self._levels(terrain, generator)
    if int(env.common_step_counter) < self.hold_steps:
      return self._levels(terrain, generator)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    robot = env.scene["robot"]
    distance = torch.norm(
      robot.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
      dim=1,
    )
    above_floor = terrain.terrain_levels[env_ids] > self.floor
    move_up = distance > generator.size[0] / 2
    if self.rule in ("survival", "tracking"):
      timed_out = env.termination_manager.time_outs[env_ids]
      length = env.episode_length_buf[env_ids].float()
      early = ~timed_out & (length < 0.5 * float(env.max_episode_length))
      if self.rule == "tracking":
        # The metrics manager resets after the curriculum runs, so the
        # episode's sums are still whole here.
        metrics: Any = env.metrics_manager
        for name in ("pelvis_speed", "command_speed"):
          if name not in metrics.active_terms:
            raise ValueError(
              f"rule='tracking' needs the {name} metric (add_tracking_metrics)"
            )
        moved = metrics._episode_sums["pelvis_speed"][env_ids]
        asked = metrics._episode_sums["command_speed"][env_ids]
        # A standing command asks for nothing; surviving it counts as tracked.
        tracked = moved >= self.track_ratio * asked
        move_up = move_up | (timed_out & tracked)
        move_down = early | (timed_out & ~tracked)
      else:
        move_up = move_up | timed_out
        move_down = early
    else:
      move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
      )
    move_down = move_down & ~move_up & above_floor
    terrain.update_env_origins(env_ids, move_up, move_down)
    return self._levels(terrain, generator)

  @staticmethod
  def _levels(terrain: Any, generator: Any) -> dict[str, torch.Tensor]:
    """mjlab's logging: mean/max plus a per-family mean."""
    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {"mean": levels.mean(), "max": levels.max()}
    names = list(generator.sub_terrains.keys())
    if terrain.terrain_origins.shape[1] == len(names):
      for i, name in enumerate(names):
        mask = terrain.terrain_types == i
        if bool(mask.any()):
          result[name] = levels[mask].mean()
    return result


__all__ = ["terrain_levels_floor"]
