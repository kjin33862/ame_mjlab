"""Evaluation-only curriculum audit.  Not used by any training config.

What the gap task's curriculum (:class:`ame.mdp.terrain_levels_floor` with
``rule="tracking"``) WOULD decide for an episode, from the same quantities:

* promote if the episode ends more than half a tile from its origin
  (``promote_distance``), or if it timed out and the summed planar pelvis speed
  reached ``track_ratio`` x the summed commanded planar speed
  (``promote_speed_only``: speed magnitudes only, the direction is not used);
* demote if it was terminated in its first half, or timed out without that
  speed (``demote``);
* otherwise (terminated in its second half, not far) nothing (``none``).

The floor row is ignored (a demote at row 0 is a no-op in training).
"""

from __future__ import annotations

import torch

DECISIONS = ("none", "promote_distance", "promote_speed_only", "demote")
"""Codes 0..3 of :func:`curriculum_decision`."""


def curriculum_decision(
  distance: torch.Tensor,
  timed_out: torch.Tensor,
  length_steps: torch.Tensor,
  max_steps: int,
  moved_sum: torch.Tensor,
  asked_sum: torch.Tensor,
  half_tile: float,
  track_ratio: float = 0.6,
) -> torch.Tensor:
  """Per episode: an index into :data:`DECISIONS` (same formulas as the
  ``tracking`` rule; ``distance`` is from the spawn, ``*_sum`` are per-step
  sums of ``|v_xy|`` and ``|cmd_xy|`` over the episode)."""
  by_distance = distance > half_tile
  tracked = moved_sum >= track_ratio * asked_sum
  by_speed = timed_out & tracked & ~by_distance
  early = ~timed_out & (length_steps.float() < 0.5 * float(max_steps))
  demote = (early | (timed_out & ~tracked)) & ~by_distance
  out = torch.zeros_like(distance, dtype=torch.long)
  out[by_distance] = 1
  out[by_speed] = 2
  out[demote] = 3
  return out


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
  """To (-pi, pi]."""
  return torch.atan2(torch.sin(a), torch.cos(a))


__all__ = ["DECISIONS", "curriculum_decision", "wrap_angle"]
