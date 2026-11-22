"""Tables and figures from raw.npz / raw_noillegal.npz (see reward_audit.py).

Q1: per-second reward = mean over robots and steps of ``_step_reward``
(weight * raw; the per-step reward is that x dt, so its sum over 10 s / 10 s
is this mean).  All 500 steps are used, resets included.

Q2: a touchdown is ``compute_first_contact`` of one foot; step length /
width are the touching foot's ankle_roll_link minus the other foot's, in the
pelvis heading frame (forward, |lateral|).  Touchdowns within 0.5 s of a
spawn are dropped, and so are contact flickers: a touchdown counts only if
the preceding swing (``last_air_time``) lasted >= MIN_AIR s (the air time at
first contact is bimodal: ~0.01 s chatter during stance vs 0.15-0.3 s
swings).  Cadence = counted touchdowns / valid second per robot.

usage: python notes/reward_audit/analyze.py  -> tables.md, q1_reward_diff.png, q2_stride.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).parent
A = np.load(D / "raw.npz")
B = np.load(D / "raw_noillegal.npz")
terms = list(A["term_names"])
conds = list(A["cond_names"])
dt = float(A["dt"])
ec = A["env_cond"]
STOP_V, STOP_T = 0.15, 1.0
WARM = 0.5
MIN_AIR = 0.1
out = []


def p(s=""):
  out.append(s)


def envs(d, name):
  return np.where(d["env_cond"] == list(d["cond_names"]).index(name))[0]


# ---------------------------------------------------------------- Q1
def stopped_mask(d, idx):
  """[T, n] True where pelvis xy speed < STOP_V for a run > STOP_T s (runs cut at resets)."""
  sp = d["speed"][:, idx]
  done = d["done"][:, idx].astype(bool)
  T, n = sp.shape
  m = np.zeros((T, n), bool)
  need = int(round(STOP_T / dt))
  for j in range(n):
    start = None
    for t in range(T + 1):
      slow = t < T and sp[t, j] < STOP_V
      if slow and start is None:
        start = t
      if (not slow or (t < T and done[t, j])) and start is not None:
        end = t + 1 if (t < T and slow) else t
        if end - start > need:
          m[start:end, j] = True
        start = None
  return m


cols = {}  # label -> (per-term per-second means, robot-seconds, extra)
for label, d, name in [
  ("(a) flat vx0.5", A, "flat_v0.5"),
  ("(a) flat vx1.0", A, "flat_v1.0"),
  ("(a) flat vx1.2", A, "flat_v1.2"),
  ("(b1) default pose, vx1.0", A, "flat_stand"),
  ("(b2) default pose, vx1.0, no illegal-contact term.", B, "flat_stand"),
  ("(b3) policy stands (obs cmd 0), rewarded vs vx1.0", A, "flat_stand_policy"),
]:
  i = envs(d, name)
  r = d["rew"][:, i]  # [T, n, K]
  cols[label] = (r.reshape(-1, r.shape[-1]).mean(0), r.shape[0] * r.shape[1] * dt,
                 d["done"][:, i].sum(), d["speed"][:, i].mean())
i23 = np.concatenate([envs(A, "stones_L2"), envs(A, "stones_L3")])
sm = stopped_mask(A, i23)
r = A["rew"][:, i23]
for label, m in [("(c) stones L2-3 moving", ~sm), ("(c) stones L2-3 stopped", sm)]:
  cols[label] = (r[m].mean(0), m.sum() * dt, None, A["speed"][:, i23][m].mean())

labels = list(cols)
p("## Q1 — reward 항목별 초당 평균 (weight × raw, = 초당 reward)\n")
p("| term | " + " | ".join(labels) + " |")
p("|---|" + "---:|" * len(labels))
for k, t in enumerate(terms):
  p(f"| {t} | " + " | ".join(f"{cols[l][0][k]:+.3f}" for l in labels) + " |")
p("| **total** | " + " | ".join(f"**{cols[l][0].sum():+.3f}**" for l in labels) + " |")
p("| robot-seconds | " + " | ".join(f"{cols[l][1]:.0f}" for l in labels) + " |")
p("| terminations | " + " | ".join("–" if cols[l][2] is None else f"{int(cols[l][2])}" for l in labels) + " |")
p("| mean pelvis xy speed (m/s) | " + " | ".join(f"{cols[l][3]:.2f}" for l in labels) + " |")
p()
for name in ("stones_L2", "stones_L3"):
  i = envs(A, name)
  m = stopped_mask(A, i)
  p(f"- {name}: stopped {m.mean() * 100:.1f}% of robot-time, robots with any stopped segment {int((m.any(0)).sum())}/16, terminations {int(A['done'][:, i].sum())}")
p()

# ---------------------------------------------------------------- Q2
ar_k = terms.index("action_rate_l2")


def touchdowns(d, idx, zone=None):
  """Rows: env, t, length, width, dz, zone (0 flat,1 up,2 down) for each touchdown."""
  rows = []
  orig = d["tile_origins"]
  for j in idx:
    col = d["env_col"][j]
    for t in range(d["td"].shape[0]):
      if d["t_since"][t, j] < WARM or d["done"][t, j]:
        continue
      for k in (0, 1):
        if not d["td"][t, j, k] or d["air"][t, j, k] < MIN_AIR:
          continue
        f, o = d["foot"][t, j, k], d["foot"][t, j, 1 - k]
        h = d["heading"][t, j]
        v = f[:2] - o[:2]
        fwd = np.cos(h) * v[0] + np.sin(h) * v[1]
        lat = -np.sin(h) * v[0] + np.cos(h) * v[1]
        row = np.argmin(np.abs(orig[:, col, 0] - f[0]))
        lx, ly = f[0] - orig[row, col, 0], f[1] - orig[row, col, 1]
        z = 1 if (1.35 <= lx <= 3.15 and abs(ly) < lx) else (2 if (-3.15 <= lx <= -1.35 and abs(ly) < -lx) else 0)
        rows.append((j, t, fwd, abs(lat), f[2] - o[2], z))
  return np.array(rows).reshape(-1, 6)


def valid_seconds(d, j):
  return ((d["t_since"][:, j] >= WARM) & ~d["done"][:, j].astype(bool)).sum() * dt


def stat(x):
  if len(x) == 0:
    return "–"
  return f"{x.mean():.3f} ± {x.std():.3f} [{np.percentile(x, 10):.3f}, {np.percentile(x, 90):.3f}]"


q2 = [
  ("flat vx0.5", "flat_v0.5", None),
  ("flat vx1.0", "flat_v1.0", None),
  ("flat vx1.2", "flat_v1.2", None),
  ("stones L1 vx1.0", "stones_L1", None),
  ("stones L2 vx1.0", "stones_L2", None),
  ("stones L3 vx1.0", "stones_L3", None),
  ("stairs up 10–17 cm vx1.0 (오르막 구간 touchdown)", "stairs_up", 1),
  ("stairs 전체 (평지·오르막·내리막)", "stairs_up", "all"),
  ("(b3) policy stands", "flat_stand_policy", None),
]
p("## Q2 — touchdown 통계 (mean ± std [p10, p90])\n")
p("| 조건 | touchdown 수 (chatter 제외 전 first-contact 수) | step length (m, heading 앞방향) | step width (m, 좌우 거리) | cadence (steps/s, 로봇별) | action_rate (초당, 로봇별) | pelvis 속도 (m/s) |")
p("|---|---:|---|---|---|---|---:|")
per_robot = []  # (cond label, vx, mean length, cadence, action_rate/s)
dist = {}
for label, name, zone in q2:
  idx = envs(A, name)
  td = touchdowns(A, idx)
  sel = td if zone in (None, "all") else td[td[:, 5] == zone]
  dist[label] = sel
  cad = np.array([(td[:, 0] == j).sum() / max(valid_seconds(A, j), 1e-9) for j in idx])
  ar = A["rew"][:, idx, ar_k].mean(0)
  raw_n = int(sum(A["td"][:, j].sum() for j in idx)) if zone is None else "–"
  p(f"| {label} | {len(sel)} ({raw_n}) | {stat(sel[:, 2])} | {stat(sel[:, 3])} | {stat(cad)} | {stat(ar)} | {A['speed'][:, idx].mean():.2f} |")
  if zone is None or zone == "all":
    for n_, j in enumerate(idx):
      tj = td[td[:, 0] == j]
      per_robot.append((label, float(A["env_vx"][j]), tj[:, 2].mean() if len(tj) else np.nan, cad[n_], ar[n_]))
up = dist["stairs up 10–17 cm vx1.0 (오르막 구간 touchdown)"]
p()
p(f"- stairs 오르막 touchdown의 발 높이 차(착지발 − 지지발) dz: {stat(up[:, 4])} m (오르막 확인용)")
allst = touchdowns(A, envs(A, "stairs_up"))
for z, nm in ((0, "평지(웅덩이 바닥·윗면)"), (1, "오르막"), (2, "내리막")):
  s = allst[allst[:, 5] == z]
  p(f"- stairs {nm}: n={len(s)}, length {stat(s[:, 2])}, width {stat(s[:, 3])}")
# stairs by height
p()
p("| 계단 높이 | 오르막 touchdown | step length | step width |")
p("|---:|---:|---|---|")
for c in np.unique(A["env_col"][envs(A, "stairs_up")]):
  idx = [j for j in envs(A, "stairs_up") if A["env_col"][j] == c]
  s = touchdowns(A, idx)
  s = s[s[:, 5] == 1]
  p(f"| {A['col_names'][c]} | {len(s)} | {stat(s[:, 2])} | {stat(s[:, 3])} |")

pr = np.array([(v, l, c, a) for _, v, l, c, a in per_robot if not np.isnan(l)])
walk = [r for r in per_robot if not r[0].startswith("(b3)") and not r[0].startswith("stairs 전체") and not np.isnan(r[2])]
w = np.array([(v, l, c, a) for _, v, l, c, a in walk])
p()
p("action_rate 관계 (보행 조건 로봇 128대 기준, b3 제외; 로봇 1대 = 점 1개):")
p(f"- corr(action_rate/s, step length) = {np.corrcoef(w[:, 3], w[:, 1])[0, 1]:+.2f}")
p(f"- corr(action_rate/s, cadence) = {np.corrcoef(w[:, 3], w[:, 2])[0, 1]:+.2f}")
fl = np.array([(v, l, c, a) for lab, v, l, c, a in walk if lab.startswith("flat")])
p(f"- flat만 (vx 0.5/1.0/1.2, 48대): corr(action_rate/s, vx) = {np.corrcoef(fl[:, 3], fl[:, 0])[0, 1]:+.2f}, corr(action_rate/s, step length) = {np.corrcoef(fl[:, 3], fl[:, 1])[0, 1]:+.2f}")
sl = np.polyfit(fl[:, 1], fl[:, 3], 1)
p(f"- flat 선형근사: action_rate/s ≈ {sl[0]:+.3f} × step length {sl[1]:+.3f}")
(D / "tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))

# ---------------------------------------------------------------- figures
from matplotlib import font_manager

_KO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_KO).exists():
  font_manager.fontManager.addfont(_KO)
  plt.rcParams["font.family"] = [font_manager.FontProperties(fname=_KO).get_name(), "DejaVu Sans"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({
  "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
  "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
  "axes.spines.top": False, "axes.spines.right": False,
})

# Q1: per-term difference (stand/stop minus walk), two panels
walk_ref = cols["(a) flat vx1.0"][0]
pairs = [
  ("평지: 정책이 서 있음(b3) − 걷기 vx1.0(a)", cols["(b3) policy stands (obs cmd 0), rewarded vs vx1.0"][0] - walk_ref),
  ("stones L2–3: 멈춤 구간 − 이동 구간 (c)", cols["(c) stones L2-3 stopped"][0] - cols["(c) stones L2-3 moving"][0]),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
order = np.argsort(-np.abs(pairs[0][1] + pairs[1][1]))
names = [terms[k] for k in order][::-1]
for ax, (title, diff) in zip(axes, pairs):
  v = diff[order][::-1]
  ax.barh(names, v, color=[S1 if x >= 0 else S2 for x in v], height=0.6)
  ax.axvline(0, color=INK2, lw=0.8)
  ax.set_title(title + f"\n합계 {diff.sum():+.2f} /s", fontsize=9, color=INK, loc="left")
  ax.grid(axis="x", color=GRID, lw=0.6)
  ax.set_axisbelow(True)
  ax.set_xlabel("초당 reward 차이 (파랑 = 멈춤 쪽이 더 받음, 주황 = 걷기 쪽이 더 받음)")
  for y, x in enumerate(v):
    if abs(x) > 0.05:
      ax.text(x, y, f" {x:+.2f} ", va="center", ha="left" if x >= 0 else "right", fontsize=7.5, color=INK)
fig.tight_layout()
fig.savefig(D / "q1_reward_diff.png", dpi=130)

# Q2: step length / width per condition, action_rate vs step length per robot
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
show = ["flat vx0.5", "flat vx1.0", "flat vx1.2", "stones L1 vx1.0", "stones L2 vx1.0", "stones L3 vx1.0",
        "stairs up 10–17 cm vx1.0 (오르막 구간 touchdown)"]
short = ["flat\n0.5", "flat\n1.0", "flat\n1.2", "stones\nL1", "stones\nL2", "stones\nL3", "stairs\nup"]
for ax, colk, ttl in ((axes[0], 2, "step length (m)"), (axes[1], 3, "step width (m)")):
  data = [dist[s][:, colk] for s in show]
  bp = ax.boxplot(data, whis=(10, 90), showfliers=False, patch_artist=True, widths=0.55)
  for b, c in zip(bp["boxes"], [S1, S1, S1, S3, S3, S3, S4]):
    b.set_facecolor(c); b.set_alpha(0.85); b.set_edgecolor(SURF)
  for m in bp["medians"]:
    m.set_color(INK)
  ax.set_xticks(range(1, len(short) + 1), short)
  ax.set_title(ttl + " — 상자 p25–p75, 수염 p10–p90", fontsize=9, color=INK, loc="left")
  ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax = axes[2]
groups = [("flat vx0.5", S1, "o"), ("flat vx1.0", S2, "s"), ("flat vx1.2", S3, "^"),
          ("stones L1 vx1.0", S4, "D"), ("stones L2 vx1.0", S5, "v"), ("stones L3 vx1.0", "#6250d6", "P"),
          ("stairs 전체 (평지·오르막·내리막)", "#008300", "X")]
for lab, c, mk in groups:
  rr = np.array([(l, a) for g, _, l, _, a in per_robot if g == lab and not np.isnan(l)])
  ax.scatter(rr[:, 0], rr[:, 1], s=26, color=c, marker=mk, edgecolor=SURF, linewidth=0.8,
             label=lab.replace(" (평지·오르막·내리막)", ""))
ax.set_xlabel("로봇별 평균 step length (m)")
ax.set_ylabel("로봇별 action_rate (초당 reward)")
ax.set_title("action_rate vs step length (로봇 1대 = 점 1개)", fontsize=9, color=INK, loc="left")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(D / "q2_stride.png", dpi=130)
print("figures written")
