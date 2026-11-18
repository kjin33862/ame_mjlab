"""Offline prediction: SGS vs AlphaNIST level-sampling probabilities from a run's logged p_k.

For every logged iteration, the per-level success rates p_k of a family are pushed through
ame.mdp.sgs.sgs_probs with SGS_PARAMS and with ALPHANIST_SGS_PARAMS; the per-iteration
probabilities are then averaged over 1000-iteration windows.  The logged prob_k (SGS, what the
run actually sampled) is averaged the same way as a check.  No training; p_k is the SGS run's,
so AlphaNIST's columns (plain and mixed 0.2 uniform) are "AlphaNIST fed the SGS run's success rates" (no feedback loop).

usage: python notes/foothold_d30_alphanist/predict_sampling.py <run_dir> <out.json>
"""
import json
import sys

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ame.mdp.sgs import ALPHANIST_SGS_PARAMS, SGS_PARAMS, UNIFORM_MIX_KEY, sgs_probs

MIX = 0.2
ALPHANIST_MIX = {**ALPHANIST_SGS_PARAMS, UNIFORM_MIX_KEY: MIX}

FAMILIES = ["stepping_stones", "pyramid_stairs", "pyramid_stairs_inv"]
LEVELS = 10
WINDOW = 1000

ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
ea.Reload()
pre = "Curriculum/terrain_levels"


def series(tag):
  ev = ea.Scalars(tag)
  return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def probs(p, s):
  return sgs_probs(
    torch.tensor(p), s["t"], s["kappa"], s["T"], s["eps"], s.get(UNIFORM_MIX_KEY, 0.0)
  ).numpy()


out = {}
for fam in FAMILIES:
  steps, _ = series(f"{pre}/{fam}/p_0")
  p = np.stack([series(f"{pre}/{fam}/p_{k}")[1] for k in range(LEVELS)], 1)
  logged = np.stack([series(f"{pre}/{fam}/prob_{k}")[1] for k in range(LEVELS)], 1)
  sgs = probs(p, SGS_PARAMS)
  alpha = probs(p, ALPHANIST_SGS_PARAMS)
  alpha_mix = probs(p, ALPHANIST_MIX)
  rows = []
  for w0 in range(0, int(steps.max()) + 1, WINDOW):
    m = (steps >= w0) & (steps < w0 + WINDOW)
    rows.append({
      "iters": f"{w0}-{w0 + WINDOW - 1}",
      "p": p[m].mean(0).tolist(),
      "prob_logged_sgs": logged[m].mean(0).tolist(),
      "prob_sgs": sgs[m].mean(0).tolist(),
      "prob_alphanist": alpha[m].mean(0).tolist(),
      "prob_alphanist_mix": alpha_mix[m].mean(0).tolist(),
    })
  out[fam] = rows
json.dump({"run": sys.argv[1], "sgs": SGS_PARAMS, "alphanist": ALPHANIST_SGS_PARAMS, "alphanist_mix": ALPHANIST_MIX,
           "windows": out}, open(sys.argv[2], "w"), indent=1)

for fam, rows in out.items():
  print(f"\n## {fam}")
  for r in rows:
    f = lambda v: " ".join(f"{100 * x:5.1f}" for x in v)  # noqa: E731
    print(r["iters"], "\n  p     ", f(r["p"]), "\n  logged", f(r["prob_logged_sgs"]),
          "\n  SGS   ", f(r["prob_sgs"]), "\n  Alpha ", f(r["prob_alphanist"]),
          "\n  AlphaM", f(r["prob_alphanist_mix"]))
