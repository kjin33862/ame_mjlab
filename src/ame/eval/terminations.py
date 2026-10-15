"""Evaluation-only termination term.  Not used by any training or play config.

``illegal_contact_recorded`` returns exactly what
:func:`ame.mdp.terminations.illegal_contact` returns and also stores, per env,
which contact-sensor body had the largest force when it fired, and what that
body was touching: the other body of every raw MuJoCo contact involving it
(``terrain`` or ``self:<link>``; the sensor has no secondary filter, so both
count, and the G1 has self-collisions on).  It has to be
recorded here: the env resets terminated envs inside ``step()`` and re-senses
afterwards, so the contact forces are gone by the time ``step()`` returns.

It also snapshots the root position of every env each step (terminations run
before the auto-reset), so the evaluation can use the terminal position of an
env whose episode just ended.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from ame.mdp.terminations import illegal_contact

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CONTACT_BODY_ATTR = "_eval_contact_body"
"""Env attribute: contact-sensor body index per env (-1 = none yet).  Index
into ``env.scene[sensor_name].primary_names``."""
PRE_RESET_POS_ATTR = "_eval_pre_reset_root_pos"
"""Env attribute: ``[B, 3]`` root position at the last termination check."""
CONTACT_PARTNER_ATTR = "_eval_contact_partner"
"""Env attribute: ``list[str]`` per env, what the recorded body touched at its
last termination, deepest contact first, joined by ``+`` (``""`` = no
termination yet, ``"none"`` = no raw contact involved that body)."""
_PARTNER_LOOKUP_ATTR = "_eval_partner_lookup"


def _partner_label(name: str) -> str:
  if name == "terrain":
    return "terrain"
  if name.startswith("robot/"):
    return "self:" + name.removeprefix("robot/")
  return name


def _contact_partners(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  env_ids: torch.Tensor,
  body_idx: torch.Tensor,
) -> list[str]:
  """For each env in ``env_ids``: the partners of sensor body ``body_idx`` in
  the current raw contact list (the same physics state the sensor read)."""
  lookup = getattr(env, _PARTNER_LOOKUP_ATTR, None)
  if lookup is None:
    model = env.sim.mj_model
    names = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
      for b in range(model.nbody)
    ]
    primary = [
      names.index(f"robot/{n.split('/')[-1]}")
      for n in env.scene[sensor_name].primary_names
    ]
    lookup = (names, primary, torch.as_tensor(model.geom_bodyid, dtype=torch.long))
    setattr(env, _PARTNER_LOOKUP_ATTR, lookup)
  names, primary, geom_body = lookup

  data = env.sim.data
  n = int(torch.as_tensor(data.nacon)[0])
  geom = torch.as_tensor(data.contact.geom)[:n].long().cpu()
  world = torch.as_tensor(data.contact.worldid)[:n].long().cpu()
  dist = torch.as_tensor(data.contact.dist)[:n].float().cpu()
  bodies = geom_body[geom]  # [n, 2]

  out = []
  for e, k in zip(env_ids.tolist(), body_idx.tolist(), strict=True):
    body = primary[k]
    sel = (world == e) & (bodies == body).any(dim=1)
    if not sel.any():
      out.append("none")
      continue
    pair, depth = bodies[sel], dist[sel]
    other = torch.where(pair[:, 0] == body, pair[:, 1], pair[:, 0])
    labels: list[str] = []
    for j in torch.argsort(depth).tolist():  # most penetrating first
      label = _partner_label(names[int(other[j])])
      if label not in labels:
        labels.append(label)
    out.append("+".join(labels))
  return out


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

  if terminated.any() and getattr(env, "sim", None) is not None:
    partners = getattr(env, CONTACT_PARTNER_ATTR, None)
    if partners is None:
      partners = [""] * env.num_envs
      setattr(env, CONTACT_PARTNER_ATTR, partners)
    ids = terminated.nonzero().flatten()
    found = _contact_partners(env, sensor_name, ids, strongest[ids])
    for i, label in zip(ids.tolist(), found, strict=True):
      partners[i] = label

  setattr(env, PRE_RESET_POS_ATTR, env.scene["robot"].data.root_link_pos_w.clone())
  return terminated


__all__ = [
  "CONTACT_BODY_ATTR",
  "CONTACT_PARTNER_ATTR",
  "PRE_RESET_POS_ATTR",
  "illegal_contact_recorded",
]
