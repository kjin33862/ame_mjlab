"""Aggregate the policy matrix and its noise sweep (scripts/run_policy_matrix.sh). CPU only, no simulator.

Reads logs/eval_results/*/summary.json + per_env.csv of the runs tagged "mx-<terrain>-d<d>-s<seed>" (matrix) and
"nz-<cond>-s<seed>" (noise sweep on stone_lattice d = 0.5), keeps valid runs that used the forward success rule (the
latest per model and tag), pools the seeds per cell and writes to logs/eval_results/matrix_plots/:
    matrix_summary.csv           one row per (model, terrain, difficulty) and per (model, noise condition)
    success_vs_difficulty.png    success rate vs difficulty, one panel per terrain, one line per model, 95% CI
    single_support.png           single-support fraction per terrain and model (all difficulties pooled), 95% CI
    noise_sweep.png              success rate per noise condition and model, 95% CI

Pooling: the envs of all seeds are one sample of n = sum(num_envs) episodes; rates are k/n with the Wilson 95% interval.
Gait fractions are the mean over envs (each env's fraction of its own tracked steps; an env that fails early has few
steps), with a normal 95% interval of the mean. success_any_dir = the old rule (xy displacement >= 4 m in any
direction before a fail), computed from max_xy_disp_m. Models are ordered by name unless --models gives an order.

Example:
    uv run python scripts/aggregate_matrix.py
    uv run python scripts/aggregate_matrix.py --models baseline shuang_gap2048 gap_1024
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MX_RE = re.compile(r"^mx-(?P<terrain>.+)-d(?P<d>[0-9.]+)-s(?P<seed>\d+)$")
NZ_RE = re.compile(r"^nz-(?P<cond>.+)-s(?P<seed>\d+)$")
TERRAINS = [
  "flat",
  "pyramid_stairs",
  "pyramid_stairs_inv",
  "random_rough",
  "wave",
  "stone_lattice",
  "stepping_stones",
]
NOISE_CONDS = {
  "off": "off",
  "train": "train preset",
  "sigma0.06": "height sigma 0.06 m",
  "sigma0.10": "height sigma 0.10 m",
  "offset0.10": "height offset ±0.10 m",
  "offset0.15": "height offset ±0.15 m",
  "drift1": "drift 1 cell",
  "drift2": "drift 2 cells",
  "drift3": "drift 3 cells",
  "delay5": "map delay 5 steps",
  "delay10": "map delay 10 steps",
}
FAIL_REASONS = ["contact", "tilt", "height", "spawn_drop", "orientation", "blowup"]
# light chart surface and text tokens; categorical slots 1-3 of the reference palette, each with its own marker and
# line style so a model is not told apart by color alone
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df"}
MODEL_STYLES = [
  {"color": "#2a78d6", "marker": "o", "linestyle": "-", "hatch": None},
  {"color": "#eb6834", "marker": "s", "linestyle": "--", "hatch": "//"},
  {"color": "#1baf7a", "marker": "^", "linestyle": "-.", "hatch": ".."},
  {"color": "#eda100", "marker": "D", "linestyle": ":", "hatch": "xx"},
]
Z95 = 1.959964


def wilson(k: int, n: int) -> tuple[float, float]:
  if n == 0:
    return float("nan"), float("nan")
  p = k / n
  denom = 1 + Z95**2 / n
  center = (p + Z95**2 / (2 * n)) / denom
  half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denom
  return max(0.0, center - half), min(1.0, center + half)


def mean_ci(xs: list[float]) -> tuple[float, float, float]:
  xs = [x for x in xs if not math.isnan(x)]
  if not xs:
    return float("nan"), float("nan"), float("nan")
  m = sum(xs) / len(xs)
  if len(xs) < 2:
    return m, float("nan"), float("nan")
  sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
  half = Z95 * sd / math.sqrt(len(xs))
  return m, m - half, m + half


def load_runs(results_dir: str) -> tuple[dict, list[str]]:
  """{(model, tag): (summary, per-env rows)} of the latest valid forward-rule run, and the skipped folders."""
  latest, skipped = {}, []
  for path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
    folder = os.path.basename(os.path.dirname(path))
    try:
      with open(path) as fh:
        s = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
      skipped.append(f"{folder}: unreadable ({e})")
      continue
    tag = s.get("tag") or ""
    if not (MX_RE.match(tag) or NZ_RE.match(tag)):
      continue
    if s.get("valid") is not True:
      skipped.append(f"{folder}: invalid ({s.get('invalid_reason')})")
      continue
    if s.get("success_rule") != "forward":
      skipped.append(f"{folder}: success rule {s.get('success_rule')!r}, not 'forward'")
      continue
    per_env = os.path.join(os.path.dirname(path), "per_env.csv")
    if not os.path.isfile(per_env):
      skipped.append(f"{folder}: no per_env.csv")
      continue
    key = (s["model_name"], tag)
    if key in latest:
      skipped.append(f"{latest[key][2]}: superseded by {folder}")
    with open(per_env) as fh:
      latest[key] = (s, list(csv.DictReader(fh)), folder)
  return {k: v[:2] for k, v in latest.items()}, skipped


def pool(runs: list[tuple[dict, list[dict]]], success_dist: float = 4.0) -> dict:
  rows = [r for _, env_rows in runs for r in env_rows]
  n = len(rows)
  f = lambda key: [float(r[key]) for r in rows]  # noqa: E731
  k = sum(r["outcome"] == "success" for r in rows)
  k_any = sum(float(r["max_xy_disp_m"]) >= success_dist for r in rows)
  tracked = [r for r in rows if int(r["tracked_steps"]) > 0]
  lo, hi = wilson(k, n)
  out = {
    "n_seeds": len(runs),
    "seeds": " ".join(str(s["seed"]) for s, _ in runs),
    "n": n,
    "success_k": k,
    "success_rate": k / n if n else float("nan"),
    "success_ci_lo": lo,
    "success_ci_hi": hi,
    "success_rate_per_seed": " ".join(
      f"{sum(r['outcome'] == 'success' for r in e) / len(e):.3f}" for _, e in runs
    ),
    "fail_rate": sum(r["outcome"] == "fail" for r in rows) / n if n else float("nan"),
    "stuck_rate": sum(r["outcome"] == "stuck" for r in rows) / n if n else float("nan"),
    "success_any_dir_rate": k_any / n if n else float("nan"),
  }
  for reason in FAIL_REASONS:
    out[f"fail_{reason}"] = sum(r["fail_reason"] == reason for r in rows)
  for key in ("max_forward_m", "forward_at_end_m", "max_xy_disp_m"):
    out[f"mean_{key}"] = mean_ci(f(key))[0]
  succ_t = [float(r["time_to_success_s"]) for r in rows if r["outcome"] == "success"]
  out["mean_time_to_success_s"] = mean_ci(succ_t)[0]
  for key in ("double_support", "single_support", "flight"):
    m, lo, hi = mean_ci([float(r[f"{key}_frac"]) for r in tracked])
    out[f"{key}_frac"] = m
    out[f"{key}_ci_lo"], out[f"{key}_ci_hi"] = lo, hi
  return out


def style_axes(ax) -> None:
  ax.set_facecolor(STYLE["surface"])
  ax.grid(True, color=STYLE["grid"], linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(STYLE["grid"])
  ax.tick_params(colors=STYLE["text2"], labelsize=8)


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--results", default=os.path.join(REPO_ROOT, "logs", "eval_results"))
  p.add_argument("--models", nargs="*", default=None, help="Models and their order.")
  args = p.parse_args()
  out_dir = os.path.join(args.results, "matrix_plots")
  os.makedirs(out_dir, exist_ok=True)

  runs, skipped = load_runs(args.results)
  models = args.models or sorted({m for m, _ in runs})
  mx, nz = defaultdict(list), defaultdict(list)
  for (model, tag), run in sorted(runs.items(), key=lambda kv: kv[1][0]["seed"]):
    if model not in models:
      continue
    if m := MX_RE.match(tag):
      mx[(model, m["terrain"], float(m["d"]))].append(run)
    elif m := NZ_RE.match(tag):
      nz[(model, m["cond"])].append(run)

  table = []
  for (model, terrain, d), cell in sorted(
    mx.items(),
    key=lambda kv: (models.index(kv[0][0]), TERRAINS.index(kv[0][1]), kv[0][2]),
  ):
    table.append(
      {
        "kind": "matrix",
        "model": model,
        "terrain": terrain,
        "difficulty": d,
        "condition": "off",
        **pool(cell),
      }
    )
  for (model, cond), cell in sorted(
    nz.items(),
    key=lambda kv: (models.index(kv[0][0]), list(NOISE_CONDS).index(kv[0][1])),
  ):
    table.append(
      {
        "kind": "noise",
        "model": model,
        "terrain": "stone_lattice",
        "difficulty": 0.5,
        "condition": cond,
        **pool(cell),
      }
    )
  csv_path = os.path.join(out_dir, "matrix_summary.csv")
  with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
    writer.writeheader()
    for row in table:
      writer.writerow(
        {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()}
      )

  style = {m: MODEL_STYLES[i % len(MODEL_STYLES)] for i, m in enumerate(models)}
  plt.rcParams.update(
    {"font.size": 9, "text.color": STYLE["text"], "axes.labelcolor": STYLE["text2"]}
  )
  mx_rows = [r for r in table if r["kind"] == "matrix"]

  # -- success vs difficulty, one panel per terrain
  fig, axes = plt.subplots(
    2, 4, figsize=(14, 6.4), sharey=True, facecolor=STYLE["surface"]
  )
  axes = axes.flatten()
  for ax, terrain in zip(axes, TERRAINS, strict=False):
    style_axes(ax)
    for i, model in enumerate(models):
      rs = [r for r in mx_rows if r["model"] == model and r["terrain"] == terrain]
      if not rs:
        continue
      st, xs = (
        style[model],
        [r["difficulty"] + 0.012 * (i - (len(models) - 1) / 2) for r in rs],
      )
      ys = [r["success_rate"] for r in rs]
      err = [
        [y - r["success_ci_lo"] for y, r in zip(ys, rs, strict=True)],
        [r["success_ci_hi"] - y for y, r in zip(ys, rs, strict=True)],
      ]
      ax.errorbar(
        xs, ys, yerr=err, color=st["color"], marker=st["marker"], markersize=6, linestyle=st["linestyle"],
        linewidth=2, capsize=3, elinewidth=1.2, label=model, markeredgecolor=STYLE["surface"], markeredgewidth=1,
      )  # fmt: skip
    ax.set_title(terrain, color=STYLE["text"], fontsize=10, loc="left")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("difficulty")
  axes[0].set_ylabel("success rate (forward ≥ 4 m)")
  axes[4].set_ylabel("success rate (forward ≥ 4 m)")
  legend_ax = axes[len(TERRAINS)]
  legend_ax.axis("off")
  handles, labels = axes[0].get_legend_handles_labels()
  legend_ax.legend(
    handles, labels, loc="center", frameon=False, title="model", fontsize=10
  )
  legend_ax.text(
    0.5, 0.12,
    "noise off · 8×8 tiles × 3 seeds = 192 robots per point\n95% Wilson CI · stone_lattice gap = 0.45·d m",
    ha="center", va="center", color=STYLE["text2"], fontsize=8, transform=legend_ax.transAxes,
  )  # fmt: skip
  fig.suptitle(
    "Success vs terrain difficulty", x=0.01, ha="left", color=STYLE["text"], fontsize=12
  )
  fig.tight_layout()
  fig.savefig(
    os.path.join(out_dir, "success_vs_difficulty.png"),
    dpi=140,
    facecolor=STYLE["surface"],
  )
  plt.close(fig)

  # -- single-support fraction per terrain (all difficulties pooled)
  fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=STYLE["surface"])
  style_axes(ax)
  width = 0.8 / max(len(models), 1)
  for i, model in enumerate(models):
    st, xs, ys, lo, hi = style[model], [], [], [], []
    for j, terrain in enumerate(TERRAINS):
      cells = [c for (m, t, _), c in mx.items() if m == model and t == terrain]
      if not cells:
        continue
      rows = [
        r
        for c in cells
        for _, env_rows in c
        for r in env_rows
        if int(r["tracked_steps"]) > 0
      ]
      m_, l_, h_ = mean_ci([float(r["single_support_frac"]) for r in rows])
      xs.append(j + (i - (len(models) - 1) / 2) * width)
      ys.append(m_)
      lo.append(m_ - l_)
      hi.append(h_ - m_)
    ax.bar(
      xs, ys, width=width - 0.03, color=st["color"], hatch=st["hatch"], edgecolor=STYLE["surface"], linewidth=0,
      label=model, yerr=[lo, hi], capsize=2, error_kw={"ecolor": STYLE["text2"], "elinewidth": 1},
    )  # fmt: skip
  ax.set_xticks(range(len(TERRAINS)))
  ax.set_xticklabels(TERRAINS, rotation=15)
  ax.set_ylim(0, 1)
  ax.set_ylabel("single-support fraction")
  ax.legend(frameon=False, loc="upper left", ncols=len(models))
  ax.set_title(
    "Fraction of tracked steps with exactly one foot down (all difficulties, seeds pooled; 95% CI of the mean)",
    loc="left", color=STYLE["text"], fontsize=10,
  )  # fmt: skip
  fig.tight_layout()
  fig.savefig(
    os.path.join(out_dir, "single_support.png"), dpi=140, facecolor=STYLE["surface"]
  )
  plt.close(fig)

  # -- noise sweep: success per condition (rows) and model (markers)
  nz_rows = [r for r in table if r["kind"] == "noise"]
  if nz_rows:
    conds = [c for c in NOISE_CONDS if any(r["condition"] == c for r in nz_rows)]
    fig, ax = plt.subplots(
      figsize=(8, 0.42 * len(conds) + 1.4), facecolor=STYLE["surface"]
    )
    style_axes(ax)
    for i, model in enumerate(models):
      st = style[model]
      rs = {r["condition"]: r for r in nz_rows if r["model"] == model}
      ys = [
        k + (i - (len(models) - 1) / 2) * 0.22 for k, c in enumerate(conds) if c in rs
      ]
      xs = [rs[c]["success_rate"] for c in conds if c in rs]
      err = [[rs[c]["success_rate"] - rs[c]["success_ci_lo"] for c in conds if c in rs],
             [rs[c]["success_ci_hi"] - rs[c]["success_rate"] for c in conds if c in rs]]  # fmt: skip
      ax.errorbar(
        xs, ys, xerr=err, color=st["color"], marker=st["marker"], markersize=7, linestyle="none", capsize=3,
        elinewidth=1.2, label=model, markeredgecolor=STYLE["surface"], markeredgewidth=1,
      )  # fmt: skip
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([NOISE_CONDS[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("success rate (forward ≥ 4 m), 95% Wilson CI")
    ax.legend(
      frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(models)
    )
    fig.suptitle(
      "Noise sweep · stone_lattice d = 0.5 · 192 robots per point",
      x=0.01,
      ha="left",
      color=STYLE["text"],
    )
    fig.tight_layout()
    fig.savefig(
      os.path.join(out_dir, "noise_sweep.png"), dpi=140, facecolor=STYLE["surface"]
    )
    plt.close(fig)

  incomplete = [
    f"{m} {t} d{d}: {len(c)} seeds" for (m, t, d), c in mx.items() if len(c) != 3
  ] + [f"{m} noise {c}: {len(v)} seeds" for (m, c), v in nz.items() if len(v) != 3]
  print(
    f"[INFO] {len(runs)} runs, models {models}; wrote {csv_path} and plots in {out_dir}"
  )
  for line in incomplete:
    print(f"[WARN] incomplete cell: {line}")
  for line in skipped:
    print(f"[SKIP] {line}")


if __name__ == "__main__":
  main()
