"""Foothold v2 and progress-shortfall feasibility, from notes/edge_measure/raw_v2.npz.

Foothold v2 (per touchdown).  Touchdown = first contact of a foot whose
preceding swing lasted >= 0.1 s (``last_air_time``), >= 0.5 s after a spawn.
Window = the touchdown step and the next 4 policy steps (0.1 s), cut at the
first step the foot is out of contact or the episode ends.  At each window
step, z_i = terrain height under sole point i (``foot_sole_scan`` hit, foot
tilt ignored; a miss = -inf, i.e. off); point i is off if z_i < max_j z_j -
delta; frac = off share of the 15; x = max over the window.  Penalty, weight
-1, applied once per touchdown: -x (linear) or -x^2 (squared).  Per second =
sum over touchdowns / robot-seconds.

Progress shortfall (per step, weight -2).  For steps >= 1.0 s after a spawn
(so the 1 s window lies inside one episode) and |v_cmd| > 0.3: d = pelvis xy
displacement over the last 50 steps projected on the commanded xy direction
(body-frame command rotated by the current heading), ratio = d / (|v_cmd| *
1 s), penalty = max(0, 0.8 - ratio); otherwise 0.  Per second = -2 * mean.

usage: python notes/feasibility_v2/analyze_v2.py  -> tables_v2.md, fig_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

D = Path(__file__).parent
R = np.load(D.parent / "edge_measure" / "raw_v2.npz")
dt = float(R["dt"])
conds = list(R["cond_names"])
ec, ecol = R["env_cond"], R["env_col"]
colnames = list(R["col_names"])
terms = list(R["term_names"])
orig = R["tile_origins"]
T, B = R["speed"].shape
DELTAS = (0.02, 0.03, 0.05, 0.08)
WARM, MIN_AIR, WIN = 0.5, 0.1, 5
STOP_V, STOP_T = 0.15, 1.0
FH = terms.index("foothold")
out = []


def p(s=""):
  out.append(s)


def envs(name):
  return np.where(ec == conds.index(name))[0]


done = R["done"].astype(bool)
tsin = R["t_since"]
found = R["found"] > 0
td = R["td"].astype(bool) & (R["air"] >= MIN_AIR) & (tsin >= WARM)[..., None] & ~done[..., None]
z = R["sole_hit"][..., 2].astype(np.float64)  # [T, B, 2, 15]
z = np.where(np.isnan(z), -np.inf, z)
xy = R["sole_hit"][..., :2].astype(np.float64)

# ------------------------------------------------------------ touchdown table
rows = []  # env, t, foot, x[delta]..., span, slope, zone, valid_seconds_ignored
for t, j, k in zip(*np.nonzero(td)):
  steps = [t]
  for s in range(t + 1, min(t + WIN, T)):
    if done[s, j] or not found[s, j, k]:
      break
    steps.append(s)
  zz = z[steps, j, k]  # [w, 15]
  top = zz.max(axis=1, keepdims=True)
  if not np.isfinite(top).all():
    continue
  xs = [float(((zz < top - d).mean(axis=1)).max()) for d in DELTAS]
  z0 = zz[0]
  fin = np.isfinite(z0)
  span = float(z0[fin].max() - z0[fin].min()) if fin.all() else np.inf
  slope = np.nan
  if fin.all():
    A = np.c_[xy[t, j, k, :, 0], xy[t, j, k, :, 1], np.ones(15)]
    coef, *_ = np.linalg.lstsq(A, z0, rcond=None)
    slope = float(np.hypot(coef[0], coef[1]))
  f = R["foot"][t, j, k]
  c = ecol[j]
  r_ = np.argmin(np.abs(orig[:, c, 0] - f[0]))
  lx, ly = f[0] - orig[r_, c, 0], f[1] - orig[r_, c, 1]
  outward = 1.35 <= lx <= 3.15 and abs(ly) < lx
  inward = -3.15 <= lx <= -1.35 and abs(ly) < -lx
  name = colnames[c]
  zone = "flat"
  if name.startswith("inv") or name.startswith("stairs_"):
    zone = "up" if outward else ("down" if inward else "flat")
  elif name.startswith("pyr"):
    zone = "down" if outward else ("up" if inward else "flat")
  rows.append((j, t, k, *xs, span, slope, zone))
TDX = np.array([r[:3 + len(DELTAS) + 2] for r in rows], dtype=np.float64)
ZONE = np.array([r[-1] for r in rows])
TJ, TT = TDX[:, 0].astype(int), TDX[:, 1].astype(int)
X = TDX[:, 3:3 + len(DELTAS)]
SPAN, SLOPE = TDX[:, 3 + len(DELTAS)], TDX[:, 4 + len(DELTAS)]


def robot_seconds(idx):
  return float(((tsin[:, idx] >= WARM) & ~done[:, idx]).sum() * dt)


def group(names, zone=None):
  idx = np.concatenate([envs(n) for n in names])
  m = np.isin(TJ, idx)
  if zone is not None:
    m &= ZONE == zone
  return idx, m


GROUPS = [("flat", ["flat_v1.0"], None),
          ("wave L2 (4.4 cm)", ["wave_L2"], None), ("wave L5 (11.1 cm)", ["wave_L5"], None),
          ("wave L9 (20 cm)", ["wave_L9"], None), ("random_rough", ["random_rough"], None)]
risers = R["risers"]
for L in range(1, 6):
  GROUPS.append((f"stairs up L{L} ({risers[L - 1] * 100:.1f} cm)", [f"pyr_L{L}", f"inv_L{L}"], "up"))
for L in range(1, 6):
  GROUPS.append((f"stairs down L{L} ({risers[L - 1] * 100:.1f} cm)", [f"pyr_L{L}", f"inv_L{L}"], "down"))
for L in (1, 2, 3):
  GROUPS.append((f"stones L{L} (gap {R['gaps'][L - 1] * 100:.0f} cm)", [f"stones_L{L}"], None))

p("## B. Foothold v2 — 지형별 (vx 1.0, 16대; 계단은 pyr+inv 32대의 해당 구간 touchdown)\n")
p("per second 값 = weight −1 × Σ(x 또는 x²) / robot-seconds (계단 구간은 그 레벨 32대 전체 robot-seconds로 나눔). 현재 foothold = 학습 term(δ 0.30, stance 매 step)의 초당 값(조건 전체).\n")
summary = {}
for d_i, d in enumerate(DELTAS):
  p(f"### delta = {d:.2f} m\n")
  p("| 지형 | touchdown | x>0 비율 | x≥0.5 비율 | 선형 /s | 제곱 /s | 현재 foothold /s |")
  p("|---|---:|---:|---:|---:|---:|---:|")
  for label, names, zone in GROUPS:
    idx, m = group(names, zone)
    sec = robot_seconds(idx)
    x = X[m, d_i]
    cur = R["rew"][:, idx, FH].mean()
    summary[(label, d)] = ((x > 0).mean() if len(x) else np.nan, (x >= 0.5).mean() if len(x) else np.nan,
                           -x.sum() / sec, -(x ** 2).sum() / sec, cur, len(x))
    s = summary[(label, d)]
    p(f"| {label} | {s[5]} | {s[0]:.3f} | {s[1]:.3f} | {s[2]:+.3f} | {s[3]:+.3f} | {s[4]:+.3f} |")
  p()

# wave false positives vs slope
p("### wave / random_rough / flat: x>0 비율 vs touchdown 시 국소 경사 (15점 평면 fit 기울기)\n")
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 10]
p("| 경사 구간 (m/m) | touchdown | " + " | ".join(f"δ={d:.2f}" for d in DELTAS) + " |")
p("|---|---:|" + "---:|" * len(DELTAS))
idx, mw = group(["wave_L2", "wave_L5", "wave_L9", "random_rough", "flat_v1.0"])
fp_curve = []
for lo, hi in zip(bins[:-1], bins[1:]):
  m = mw & (SLOPE >= lo) & (SLOPE < hi)
  vals = [(X[m, i] > 0).mean() if m.any() else np.nan for i in range(len(DELTAS))]
  fp_curve.append(((lo + min(hi, 0.8)) / 2, vals, int(m.sum())))
  p(f"| {lo:.2f}–{hi if hi < 10 else '∞'} | {int(m.sum())} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
p()
p("15점 길이(x 방향) 0.167 m: 경사 s에서 앞뒤 끝 높이차 ≈ 0.167·s → δ보다 크면 평평한 경사에서도 off. (δ=0.02 → s>0.12, 0.03 → 0.18, 0.05 → 0.30, 0.08 → 0.48)\n")

# stairs placement
p("### 계단: 한 tread에 발 전체가 놓인 touchdown 비율 (touchdown step의 15점 높이 span < 1 cm)\n")
p("| 레벨 | riser (cm) | up: touchdown / 한 tread 비율 | down: touchdown / 한 tread 비율 | up: x>0 (δ0.05) | down: x>0 (δ0.05) |")
p("|---|---:|---|---|---:|---:|")
for L in range(1, 6):
  cells = []
  xs_ = []
  for zone in ("up", "down"):
    _, m = group([f"pyr_L{L}", f"inv_L{L}"], zone)
    cells.append(f"{int(m.sum())} / {(SPAN[m] < 0.01).mean():.2f}")
    xs_.append(f"{(X[m, DELTAS.index(0.05)] > 0).mean():.2f}")
  p(f"| L{L} | {risers[L - 1] * 100:.1f} | {cells[0]} | {cells[1]} | {xs_[0]} | {xs_[1]} |")
p()
p("계단 tread 0.30 m > 발 capsule 길이 0.206 m(15점 0.167 m): 기하적으로 모든 레벨에서 발 전체를 한 tread에 올릴 수 있음(여유 0.094 m).\n")

# ------------------------------------------------------------ C. progress shortfall
root, head, cmd = R["root"], R["heading"], R["cmd"]
vnorm = np.linalg.norm(cmd[..., :2], axis=-1)
cw = np.stack([np.cos(head) * cmd[..., 0] - np.sin(head) * cmd[..., 1],
               np.sin(head) * cmd[..., 0] + np.cos(head) * cmd[..., 1]], -1)
u = cw / np.maximum(vnorm, 1e-9)[..., None]
prog = np.zeros((T, B))
W = int(round(1.0 / dt))
for t in range(W, T):
  ok = (tsin[t] >= 1.0) & (vnorm[t] > 0.3)
  dxy = root[t, :, :2] - root[t - W, :, :2]
  d_ = (dxy * u[t]).sum(-1)
  ratio = d_ / np.maximum(vnorm[t] * 1.0, 1e-9)
  prog[t] = np.where(ok, np.maximum(0.0, 0.8 - ratio), 0.0)
PROG = -2.0 * prog  # per-second rate


def stopped_mask(idx):
  sp = R["speed"][:, idx]
  dn = done[:, idx]
  m = np.zeros(sp.shape, bool)
  need = int(round(STOP_T / dt))
  for j in range(len(idx)):
    start = None
    for t in range(T + 1):
      slow = t < T and sp[t, j] < STOP_V
      if slow and start is None:
        start = t
      if (not slow or (t < T and dn[t, j])) and start is not None:
        end = t + 1 if (t < T and slow) else t
        if end - start > need:
          m[start:end, j] = True
        start = None
  return m


BEST = 0.05  # chosen below from the B tables; recorded in the report
bi = DELTAS.index(BEST)


def v2_per_step(idx):
  """[T, n] v2 squared penalty (weight -1) as a per-second rate placed at the touchdown step."""
  a = np.zeros((T, len(idx)))
  pos = {j: n for n, j in enumerate(idx)}
  for r_, (j, t) in enumerate(zip(TJ, TT)):
    if j in pos:
      a[t, pos[j]] += -(X[r_, bi] ** 2) / dt
  return a


p("## C. Progress shortfall (weight −2, 초당 평균)\n")
p("| 조건 | progress /s | 그 중 penalty>0 인 step 비율 | 평균 ratio (계산된 step) |")
p("|---|---:|---:|---:|")
segs = {}
for name in ("flat_v0.5", "flat_v1.0", "flat_v1.2", "flat_stand_policy"):
  idx = envs(name)
  segs[name] = (idx, np.ones((T, len(idx)), bool))
i23 = np.concatenate([envs("stones_L2"), envs("stones_L3")])
sm = stopped_mask(i23)
segs["stones L2-3 moving"] = (i23, ~sm)
segs["stones L2-3 stopped"] = (i23, sm)
for h in range(10, 18):
  idx = envs(f"stairs_{h}")
  segs[f"stairs up {h} cm"] = (idx, np.ones((T, len(idx)), bool))
for name, (idx, m) in segs.items():
  pr = PROG[:, idx][m]
  raw = prog[:, idx][m]
  calc = ((tsin[:, idx] >= 1.0) & (vnorm[:, idx] > 0.3))[m]
  rr = []
  for t in range(W, T):
    pass
  p(f"| {name} | {pr.mean():+.3f} | {(raw > 0).mean():.2f} | {(0.8 - raw[calc]).mean() if calc.any() else float('nan'):.2f}* |")
p("\n*penalty=max(0,0.8−ratio)라서 penalty>0 구간에서만 ratio가 복원됨; 이 열은 0.8−penalty 평균(=min(ratio,0.8) 평균).\n")

p("## Q1 재계산: (i) 현재 reward, (ii) + progress(−2), (iii) + progress, foothold → v2(δ=%.2f, 제곱, −1/touchdown)\n" % BEST)
p("| 구간 | robot-s | (i) 현재 total /s | (ii) +progress | (iii) +progress, v2 | 현재 foothold /s | v2 /s |")
p("|---|---:|---:|---:|---:|---:|---:|")
tot = {}
for name in ("flat_v1.0", "flat_stand_policy", "stones L2-3 moving", "stones L2-3 stopped", "flat_v0.5", "flat_v1.2"):
  idx, m = segs[name]
  rew = R["rew"][:, idx]  # [T, n, K]
  i_ = rew.sum(-1)[m].mean()
  pr = PROG[:, idx][m].mean()
  fh = rew[..., FH][m].mean()
  v2 = v2_per_step(idx)[m].mean()
  tot[name] = (i_, i_ + pr, i_ + pr - fh + v2)
  p(f"| {name} | {m.sum() * dt:.0f} | {i_:+.3f} | {i_ + pr:+.3f} | {i_ + pr - fh + v2:+.3f} | {fh:+.3f} | {v2:+.3f} |")
p()
for a, b in (("flat_v1.0", "flat_stand_policy"), ("stones L2-3 moving", "stones L2-3 stopped")):
  p(f"- {a} − {b}: (i) {tot[a][0] - tot[b][0]:+.3f}, (ii) {tot[a][1] - tot[b][1]:+.3f}, (iii) {tot[a][2] - tot[b][2]:+.3f} /s")
for L in (2, 3):
  idx = envs(f"stones_L{L}")
  m = stopped_mask(idx)
  p(f"- stones_L{L}: stopped {m.mean() * 100:.1f}% of robot-time")
(D / "tables_v2.md").write_text("\n".join(out) + "\n")
print("\n".join(out))

# ------------------------------------------------------------ figures
KO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(KO).exists():
  font_manager.fontManager.addfont(KO)
  plt.rcParams["font.family"] = [font_manager.FontProperties(fname=KO).get_name(), "DejaVu Sans"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})

# Fig B: x>0 share per terrain, one bar group per delta; and wave FP vs slope
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
labels = [g[0] for g in GROUPS]
yy = np.arange(len(labels))
hgt = 0.2
for i, d in enumerate(DELTAS):
  vals = [summary[(l, d)][0] for l in labels]
  ax.barh(yy + (i - 1.5) * hgt, vals, height=hgt * 0.9, color=SER[i], label=f"δ = {d:.2f} m")
ax.set_yticks(yy, labels)
ax.invert_yaxis()
ax.set_xlabel("x > 0 인 touchdown 비율")
ax.set_title("Foothold v2: 지형별 x>0 비율 (평지·wave·rough는 오탐, 계단·돌은 검출)", loc="left", fontsize=9, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax = axes[1]
for i, d in enumerate(DELTAS):
  xs_ = [c[0] for c in fp_curve]
  ys_ = [c[1][i] for c in fp_curve]
  ax.plot(xs_, ys_, color=SER[i], lw=2, marker="o", ms=5, label=f"δ = {d:.2f}")
ax.set_xlabel("touchdown 국소 경사 (m/m, 구간 중앙)")
ax.set_ylabel("x > 0 비율 (오탐)")
ax.set_title("wave·rough·flat: 경사별 오탐", loc="left", fontsize=9, color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(D / "fig_b_foothold_v2.png", dpi=130)

# Fig C: totals (i)/(ii)/(iii) for the two comparisons
fig, ax = plt.subplots(figsize=(9, 4.4))
names = ["flat_v1.0", "flat_stand_policy", "stones L2-3 moving", "stones L2-3 stopped"]
nice = ["평지 걷기 vx1.0", "평지 서 있기 (b3)", "stones L2–3 이동", "stones L2–3 멈춤"]
w = 0.26
for i, lab in enumerate(["(i) 현재", "(ii) + progress", "(iii) + progress, foothold v2"]):
  vals = [tot[n][i] for n in names]
  bars = ax.bar(np.arange(4) + (i - 1) * w, vals, width=w * 0.92, color=SER[i], label=lab)
  for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
ax.set_xticks(np.arange(4), nice)
ax.set_ylabel("초당 total reward")
ax.set_title("Q1 재계산: 걷기/이동 vs 서 있기/멈춤", loc="left", fontsize=9, color=INK)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
fig.savefig(D / "fig_c_totals.png", dpi=130)
print("figures written")
