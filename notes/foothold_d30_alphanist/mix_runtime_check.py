"""Runtime check of the uniform mix on the real task (64 envs, GPU, no training).

Builds the FootholdD30 AlphaNIST env from its registered config, fills the live
terrain_levels term's outcome buffers with the Foothold SGS run's last-window
success rates (prediction.json, 9000-9999), runs the curriculum manager once
with no ended env, and compares the term's logged prob_k (the values written to
TensorBoard as Curriculum/terrain_levels/<family>/prob_k) with
0.8 * AlphaNIST(p) + 0.2 / 10 and with plain AlphaNIST(p).

usage: python notes/foothold_d30_alphanist/mix_runtime_check.py
"""
import json

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import ame  # noqa: F401
from ame.mdp.sgs import ALPHANIST_SGS_PARAMS as A, sgs_probs

TASK = "Ame-Mjlab-Gap-Edge-AlphaNIST-AngW4Z-Contact-FootholdD30-Velocity-Rough-G1"
FAMILIES = ["stepping_stones", "pyramid_stairs", "pyramid_stairs_inv"]

cfg = load_env_cfg(TASK)
cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
term = env.curriculum_manager.get_term_cfg("terrain_levels").func
print("term", type(term).__name__, "sgs params", term.sgs)
names = list(cfg.scene.terrain.terrain_generator.sub_terrains.keys())
last = {f: json.load(open("notes/foothold_d30_alphanist/prediction.json"))["windows"][f][-1]["p"] for f in FAMILIES}
dev = term.buffer.outcomes.device
for f in FAMILIES:
  fi = names.index(f)
  for k, pk in enumerate(last[f]):
    hits = round(pk * term.buffer.H)
    out = torch.tensor([True] * hits + [False] * (term.buffer.H - hits), device=dev)
    term.buffer.record(torch.full((term.buffer.H,), fi, device=dev),
                       torch.full((term.buffer.H,), k, device=dev), out)
env.curriculum_manager.compute(env_ids=torch.zeros(0, dtype=torch.long, device=dev))
logs = env.curriculum_manager._curriculum_state["terrain_levels"]
worst = 0.0
for f in FAMILIES:
  p = torch.tensor([logs[f"{f}/p_{k}"] for k in range(10)], dtype=torch.float64)
  prob = [logs[f"{f}/prob_{k}"] for k in range(10)]
  an = sgs_probs(p, A["t"], A["kappa"], A["T"], A["eps"]).tolist()
  err = max(abs(prob[k] - (0.8 * an[k] + 0.02)) for k in range(10))
  worst = max(worst, err)
  print(f"\n{f}: max|logged prob - (0.8 AN + 0.02)| = {err:.1e}, sum = {sum(prob):.6f}")
  print("  p_k        ", " ".join(f"{100 * x:5.1f}" for x in p.tolist()))
  print("  AlphaNIST  ", " ".join(f"{100 * x:5.1f}" for x in an))
  print("  logged prob", " ".join(f"{100 * x:5.1f}" for x in prob))
print("\nOK" if worst < 1e-5 else "\nMISMATCH")
env.close()
