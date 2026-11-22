"""Compare equiv.npz (online terms in the new task) with the feasibility tables.

usage: python notes/foothold_v2_check/equiv_analyze.py  -> equiv_tables.md
"""

from pathlib import Path

import numpy as np

D = Path(__file__).parent
E = np.load(D / "equiv.npz")
dt = float(E["dt"])
terms = list(E["term_names"])
conds = list(E["cond_names"])
cols = list(E["col_names"])
orig = E["tile_origins"]
V2, PR = terms.index("foothold_v2"), terms.index("progress_shortfall")
rew, done, ts = E["rew"], E["done"].astype(bool), E["t_since"]
out = []


def p(s=""):
  out.append(s)


def envs(*names):
  return np.concatenate([np.where(E["env_cond"] == conds.index(n))[0] for n in names])


def valid_s(idx):
  return float(((ts[:, idx] >= 0.5) & ~done[:, idx]).sum() * dt)


# Feasibility references (notes/feasibility_v2/tables_v2.md, delta 0.05 squared;
# progress_other_terrains.txt).  v2 per second = -sum x^2 / valid robot-seconds.
REF = {
  "flat vx1.0": (0.0, -0.015),
  "flat 서 있기 (obs cmd 0, vx1.0 기준)": (0.0, -1.440),
  "stones L2": (-0.481, -0.347),
  "stones L3": (-0.726, -0.491),
  "stairs down L3 (pyr+inv, 내리막 구간 touchdown)": (-0.290, None),
  "stairs L3 pyr (progress)": (None, -0.051),
  "stairs L3 inv (progress)": (None, -0.042),
}


def zone_of(xy, col):
  name = cols[col]
  r = np.argmin(np.abs(orig[:, col, 0] - xy[0]))
  lx, ly = xy[0] - orig[r, col, 0], xy[1] - orig[r, col, 1]
  outward = 1.35 <= lx <= 3.15 and abs(ly) < lx
  inward = -3.15 <= lx <= -1.35 and abs(ly) < -lx
  if name.startswith("inv") or name.startswith("stairs_"):
    return "up" if outward else ("down" if inward else "flat")
  if name.startswith("pyr"):
    return "down" if outward else ("up" if inward else "flat")
  return "flat"


paid, reason, pxy = E["v2_paid"], E["v2_reason"], E["v2_xy"]
got = {}
for label, names in (("flat vx1.0", ["flat_v1.0"]), ("flat 서 있기 (obs cmd 0, vx1.0 기준)", ["flat_stand_policy"]),
                     ("stones L2", ["stones_L2"]), ("stones L3", ["stones_L3"])):
  idx = envs(*names)
  got[label] = (-paid[:, idx].sum() / valid_s(idx), rew[:, idx, PR].mean(), rew[:, idx, V2].mean())
idx = envs("pyr_L3", "inv_L3")
down = 0.0
for t, j, k in zip(*np.nonzero(reason[:, idx] > 0)):
  if zone_of(pxy[t, idx[j], k], E["env_col"][idx[j]]) == "down":
    down += paid[t, idx[j], k]
got["stairs down L3 (pyr+inv, 내리막 구간 touchdown)"] = (-down / valid_s(idx), None, rew[:, idx, V2].mean())
got["stairs L3 pyr (progress)"] = (None, rew[:, envs("pyr_L3"), PR].mean(), None)
got["stairs L3 inv (progress)"] = (None, rew[:, envs("inv_L3"), PR].mean(), None)


def cmp(a, b):
  if a is None or b is None:
    return "–"
  if abs(b) < 0.02 and abs(a) < 0.02:
    return "둘 다 ≈0"
  return f"{(a - b) / abs(b) * 100:+.0f}%"


p("## 등가성: 새 태스크 training reward manager vs feasibility 표 (초당)\n")
p("| 조건 | foothold_v2 새 | 표 | 차이 | progress 새 | 표 | 차이 | 종료 수 |")
p("|---|---:|---:|---:|---:|---:|---:|---:|")
for label, ref in REF.items():
  g = got[label]
  f = lambda v: "–" if v is None else f"{v:+.3f}"  # noqa: E731
  idx = {"flat vx1.0": envs("flat_v1.0"), "stones L2": envs("stones_L2"), "stones L3": envs("stones_L3"),
         "stairs L3 pyr (progress)": envs("pyr_L3"), "stairs L3 inv (progress)": envs("inv_L3")}.get(
    label, envs("flat_stand_policy") if "서 있기" in label else envs("pyr_L3", "inv_L3"))
  p(f"| {label} | {f(g[0])} | {f(ref[0])} | {cmp(g[0], ref[0])} | {f(g[1])} | {f(ref[1])} | {cmp(g[1], ref[1])} | {int(done[:, idx].sum())} |")
p()
p("foothold_v2 '새' = −Σx²/(spawn 후 0.5 s 이후·비종료 robot-seconds), feasibility와 같은 분모. reward manager의 전 step 평균(_step_reward 평균)은 아래.")
for label, g in got.items():
  if g[2] is not None:
    p(f"- {label}: _step_reward[foothold_v2] 전 step 평균 {g[2]:+.3f}/s")
p()

# double critic
groups, env_rew = E["groups"], E["env_rew"]
g2_expect = rew[..., V2] * dt
p("## Double critic 분리 (96대 × 500 step)\n")
p(f"- group 2 == foothold_v2 × dt: max |차이| {np.abs(groups[..., 1] - g2_expect).max():.2e}")
p(f"- group 1 + group 2 == env reward: max |차이| {np.abs(groups.sum(-1) - env_rew).max():.2e}")
p(f"- group 1 == env reward − foothold_v2 × dt (progress_shortfall 포함): max |차이| {np.abs(groups[..., 0] - (env_rew - g2_expect)).max():.2e}")
nz = groups[..., 1] != 0
closing = (reason > 0).any(-1)
p(f"- group 2 ≠ 0 인 (step, env): {int(nz.sum())} / {nz.size}; 그중 window 종료 step이 아닌 것: {int((nz & ~closing).sum())}")
p(f"- window 종료 step 수: {int(closing.sum())} (x=0으로 0 지급 {int((closing & ~nz).sum())})")
rs = reason[reason > 0]
p(f"- 종료 사유: full 5 sample {int((rs == 1).sum())}, lift-off cut {int((rs == 2).sum())}, 종료/리셋 cut {int((rs == 3).sum())}")
prog_nz = rew[..., PR] != 0
p(f"- progress_shortfall ≠ 0 인 (step, env): {int(prog_nz.sum())}; 그중 group 2 ≠ 0: {int((prog_nz & nz).sum())} (group 2 값은 foothold_v2 × dt와 일치하므로 progress는 group 2에 없음)")
p(f"- NaN/Inf: rew {int((~np.isfinite(rew)).sum())}, groups {int((~np.isfinite(groups)).sum())}")
(D / "equiv_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
