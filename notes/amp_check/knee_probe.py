"""Task 2 vs D30: does a knee contact end the episode, and is it penalised?

64 envs, random actions (uniform +-1.5) for 400 policy steps, seed 0.  The knee
force is read inside the reward computation (after the termination check, before
the reset), i.e. the same sensor state the termination and the
``undesired_contacts`` term see.  usage: python knee_probe.py TASK"""
import sys
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import ame  # noqa: F401

torch.manual_seed(0)
cfg = load_env_cfg(sys.argv[1]); cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
sens = env.scene["undesired_contact"]
names = list(sens.primary_names)
knees = [i for i, n in enumerate(names) if "knee" in n]
rm, tm = env.reward_manager, env.termination_manager
uc = rm.active_terms.index("undesired_contacts")
seen = {}
_compute = rm.compute


def compute(*a, **k):
  out = _compute(*a, **k)
  seen["knee"] = torch.norm(sens.data.force[:, knees], dim=-1).max(dim=1).values.clone()
  seen["pen"] = rm._step_reward[:, uc].clone()
  return out


rm.compute = compute
n_knee = n_pen = n_done = 0
fired = {t: 0 for t in tm.active_terms}
knee_terms = {t: 0 for t in tm.active_terms}
for _ in range(400):
  a = (torch.rand(64, env.action_manager.total_action_dim, device="cuda:0") * 2 - 1) * 1.5
  _, _, term, trunc, _ = env.step(a)
  k = seen["knee"] > 1.0
  n_knee += int(k.sum()); n_pen += int((k & (seen["pen"] <= -1.0)).sum()); n_done += int((k & term).sum())
  for t in tm.active_terms:
    f = tm.get_term(t)
    fired[t] += int(f.sum()); knee_terms[t] += int((f & k).sum())
print(sys.argv[1].split("AlphaNIST-")[1])
print(f"env-steps with knee force > 1 N: {n_knee}; undesired_contacts <= -1 on those: {n_pen}; terminated on those: {n_done}")
print("terminations (all):", fired)
print("terminations on knee-contact steps:", knee_terms)
