"""Evaluation-only configs and terms (used by ``scripts/evaluate.py``).

Nothing here is imported by the training tasks.
"""

from __future__ import annotations

from ame.eval.env_cfg import (
  ACTOR_OBS,
  EVAL_TERRAINS,
  OFF_HEIGHT_NOISE,
  SPAWN_ASYM_JOINT,
  TRAIN_HEIGHT_NOISE,
  apply_eval_noise,
  assign_tiles,
  checkpoint_actor_obs,
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
  "ACTOR_OBS",
  "CONTACT_BODY_ATTR",
  "CONTACT_PARTNER_ATTR",
  "EVAL_TERRAINS",
  "OFF_HEIGHT_NOISE",
  "PRE_RESET_POS_ATTR",
  "SPAWN_ASYM_JOINT",
  "TRAIN_HEIGHT_NOISE",
  "apply_eval_noise",
  "assign_tiles",
  "checkpoint_actor_obs",
  "drift_gather_index",
  "elevation_map_eval",
  "eval_env_cfg",
  "illegal_contact_recorded",
]
