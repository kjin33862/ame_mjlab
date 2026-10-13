"""Evaluation-only termination term.  Not used by any training or play config.

``illegal_contact_recorded`` returns exactly what
:func:`ame.mdp.terminations.illegal_contact` returns and also stores, per env,
which contact-sensor body had the largest force when it fired.  It has to be
recorded here: the env resets terminated envs inside ``step()`` and re-senses
afterwards, so the contact forces are gone by the time ``step()`` returns.

It also snapshots the root position of every env each step (terminations run
before the auto-reset), so the evaluation can use the terminal position of an
env whose episode just ended.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ame.mdp.terminations import illegal_contact

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CONTACT_BODY_ATTR = "_eval_contact_body"
"""Env attribute: contact-sensor body index per env (-1 = none yet).  Index
into ``env.scene[sensor_name].primary_names``."""
PRE_RESET_POS_ATTR = "_eval_pre_reset_root_pos"
"""Env attribute: ``[B, 3]`` root position at the last termination check."""


def illegal_contact_recorded(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0
) -> torch.Tensor:
  """Same result as ``illegal_contact``; records the strongest body."""
  terminated = illegal_contact(env, sensor_name, threshold)

  force = env.scene[sensor_name].data.force
  assert force is not None
  strongest = torch.argmax(torch.norm(force, dim=-1), dim=1)
  record = getattr(env, CONTACT_BODY_ATTR, None)
  if record is None:
    record = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    setattr(env, CONTACT_BODY_ATTR, record)
  record[terminated] = strongest[terminated]

  setattr(env, PRE_RESET_POS_ATTR, env.scene["robot"].data.root_link_pos_w.clone())
  return terminated


__all__ = ["CONTACT_BODY_ATTR", "PRE_RESET_POS_ATTR", "illegal_contact_recorded"]
