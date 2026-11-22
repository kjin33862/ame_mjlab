"""Offline feasibility formulas (analyze_v2.py) applied to equiv.npz, compared
with the online terms of the same run: separates implementation differences
from run-to-run behaviour.

usage: python notes/foothold_v2_check/same_run.py  -> same_run.md
"""

from pathlib import Path

import numpy as np

D = Path(__file__).parent
E = np.load(D / "equiv.npz")
dt = float(E["dt"])
terms = list(E["term_names"])
conds = list(E["cond_names"])
V2, PR = terms.index("foothold_v2"), terms.index("progress_shortfall")
done, ts = E["done"].astype(bool), E["t_since"]
found = E["found"] > 0
T, B = ts.shape
td = E["td"].astype(bool) & (E["air"] >= 0.1) & (ts >= 0.5)[..., None] & ~done[..., None]
z = E["sole_hit"][..., 2].astype(np.float64)
z = np.where(np.isnan(z), -np.inf, z)

# offline foothold v2 (analyze_v2.py, delta 0.05; a foot with every ray missing
# counts as fraction 1 here, as online; analyze_v2 skipped such windows)
x_off = np.zeros((T, B, 2))  # x^2 attributed to the touchdown step
n_skip = 0
for t, j, k in zip(*np.nonzero(td)):
  steps = [t]
  for s in range(t + 1, min(t + 5, T)):
    if done[s, j] or not found[s, j, k]:
      break
    steps.append(s)
  zz = z[steps, j, k]
  top = zz.max(axis=1, keepdims=True)
  if not np.isfinite(top).all():
    n_skip += 1
  off = ~np.isfinite(zz) | (zz < top - 0.05)
  x_off[t, j, k] = off.mean(axis=1).max() ** 2

# offline progress (analyze_v2.py)
root, head, cmd = E["root"], E["heading"], E["cmd"]
vn = np.linalg.norm(cmd[..., :2], axis=-1)
cw = np.stack([np.cos(head) * cmd[..., 0] - np.sin(head) * cmd[..., 1],
               np.sin(head) * cmd[..., 0] + np.cos(head) * cmd[..., 1]], -1)
u = cw / np.maximum(vn, 1e-9)[..., None]
prog = np.zeros((T, B))
for t in range(50, T):
  ok = (ts[t] >= 1.0) & (vn[t] > 0.3)
  d = ((root[t, :, :2] - root[t - 50, :, :2]) * u[t]).sum(-1)
  prog[t] = np.where(ok, np.maximum(0, 0.8 - d / np.maximum(vn[t], 1e-9)), 0)

out = ["## 같은 run에서 offline 공식(feasibility) vs online term\n",
       "| 조건 | v2 online /s | v2 offline /s | 차이 | touchdown 수 online / offline | progress online /s | progress offline /s | 차이 |",
       "|---|---:|---:|---:|---|---:|---:|---:|"]
for c in conds:
  idx = np.where(E["env_cond"] == conds.index(c))[0]
  vs = float(((ts[:, idx] >= 0.5) & ~done[:, idx]).sum() * dt)
  on = -E["v2_paid"][:, idx].sum() / vs
  off = -x_off[:, idx].sum() / vs
  n_on = int((E["v2_reason"][:, idx] > 0).sum())
  n_off = int(td[:, idx].sum())
  pon = E["rew"][:, idx, PR].mean()
  poff = (-2 * prog[:, idx]).mean()
  rel = lambda a, b: "≈0" if max(abs(a), abs(b)) < 0.02 else f"{(a - b) / abs(b) * 100:+.1f}%"  # noqa: E731
  out.append(f"| {c} | {on:+.3f} | {off:+.3f} | {rel(on, off)} | {n_on} / {n_off} | {pon:+.3f} | {poff:+.3f} | {rel(pon, poff)} |")
out.append(f"\n전부 miss인 window(offline analyze_v2는 건너뜀, 여기서는 online과 같이 x=1): {n_skip}개")
(D / "same_run.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
