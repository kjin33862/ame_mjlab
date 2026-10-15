"""Evaluation-only configs and terms (used by ``scripts/evaluate.py``).

Nothing here is imported by the training tasks.
"""

from __future__ import annotations

from ame.eval.env_cfg import (
  EVAL_TERRAINS,
  OFF_HEIGHT_NOISE,
  TRAIN_HEIGHT_NOISE,
  apply_eval_noise,
  assign_tiles,
  eval_env_cfg,
)
from ame.eval.observations import drift_gather_index, elevation_map_eval
from ame.eval.terminations import (
  CONTACT_BODY_ATTR,
  CONTACT_PARTNER_ATTR,
  PRE_RESET_POS_ATTR,
  illegal_contact_recorded,
)

__all__ = [
  "CONTACT_BODY_ATTR",
  "CONTACT_PARTNER_ATTR",
  "EVAL_TERRAINS",
  "OFF_HEIGHT_NOISE",
  "PRE_RESET_POS_ATTR",
  "TRAIN_HEIGHT_NOISE",
  "apply_eval_noise",
  "assign_tiles",
  "drift_gather_index",
  "elevation_map_eval",
  "eval_env_cfg",
  "illegal_contact_recorded",
]
