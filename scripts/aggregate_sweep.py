"""Aggregate the stepping-stone noise sweep (CPU only, no simulator).

Copy of the Isaac Lab version (AME_Locomotion, eval-pipeline) with minimal changes for the mjlab port: models are
grouped by "model_name" (any number of them), the drift axis is in grid cells, the extra fail reasons
"orientation" and "blowup" are counted, and random-init runs are skipped.

Reads logs/eval_results/*sweep*/summary.json, keeps runs with "valid": true, pools seeds per (model, condition) and
writes
    logs/eval_results/sweep_summary.csv
    logs/eval_results/sweep_plots/{height_sigma,height_offset,drift_cells,map_delay_steps,priority1}.png

Pooling: the envs of all seeds are treated as one sample of n = sum(num_envs) episodes. Rates are k/n, the SE is
sqrt(p(1-p)/n) and the 95% CI is the Wilson score interval (stays inside [0, 1] near 0% and 100%). Per-seed success
rates are listed too, so seed-to-seed spread can be compared with the pooled CI.

Example:
    uv run python scripts/aggregate_sweep.py
"""

import argparse
import csv
import glob
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAG_RE = re.compile(r"^sweep-(?P<cond>.+)-s(?P<seed>\d+)$")
MODELS: list[str] = []  # filled from the runs found (model_name)
FAIL_REASONS = ["contact", "tilt", "height", "spawn_drop", "orientation", "blowup"]
P1_CONDITIONS = {
    "C0": "C0: no noise",
    "C1": "C1: proprio only",
    "C2": "C2: train map, no drift",
    "C3": "C3: train (map + drift)",
}
# noise axis -> (condition prefix, x of the C1 baseline, x-axis label)
AXES = {
    "height_sigma": ("sigma", 0.0, "height noise sigma [m]  (0 = C1 baseline)"),
    "height_offset": ("offset", 0.0, "height offset, uniform +/- [m]  (0 = C1 baseline)"),
    "drift_cells": ("drift", 0.0, "z-map drift, uniform +/- [cells of 5 cm]  (0 = C1 baseline)"),
    "map_delay_steps": ("delay", 1.0, "map refresh period [policy steps, 20 ms]  (1 = every step = C1 baseline)"),
}
# light chart surface, text tokens and categorical slots; markers/linestyles so identity is not color-only
STYLE = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df"}
MODEL_STYLES = [
    {"color": "#2a78d6", "marker": "o", "linestyle": "-", "hatch": None},
    {"color": "#eb6834", "marker": "s", "linestyle": "--", "hatch": "//"},
    {"color": "#1f9e74", "marker": "^", "linestyle": "-.", "hatch": ".."},
    {"color": "#8a5cc2", "marker": "D", "linestyle": ":", "hatch": "xx"},
]
Z95 = 1.959964


def model_style(model: str) -> dict:
    return MODEL_STYLES[MODELS.index(model) % len(MODEL_STYLES)]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + Z95**2 / n
    center = (p + Z95**2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_runs(results_dir: str) -> tuple[list[dict], list[str]]:
    """Valid sweep runs (latest per model/tag) and a list of skipped folders with the reason."""
    latest, skipped = {}, []
    for path in sorted(glob.glob(os.path.join(results_dir, "*sweep*", "summary.json"))):
        folder = os.path.basename(os.path.dirname(path))
        try:
            with open(path) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            skipped.append(f"{folder}: unreadable ({e})")
            continue
        if s.get("valid") is not True:
            skipped.append(f"{folder}: invalid ({s.get('invalid_reason', 'no valid flag')})")
            continue
        if s.get("policy") == "random_init":
            skipped.append(f"{folder}: random-init policy (pipeline test)")
            continue
        m = TAG_RE.match(s.get("tag", ""))
        if not m:
            skipped.append(f"{folder}: tag {s.get('tag')!r} is not sweep-<cond>-s<seed>")
            continue
        if s.get("truncated_by_max_steps"):
            skipped.append(f"{folder}: truncated by --max_steps")
            continue
        s["condition"], s["seed_from_tag"], s["folder"] = m["cond"], int(m["seed"]), folder
        s["model"] = s.get("model_name") or s["arch"]
        latest[(s["model"], s["tag"])] = s  # sorted by timestamp prefix -> the last one wins
    return list(latest.values()), skipped


def pool(runs: list[dict]) -> dict:
    n = sum(r["num_envs"] for r in runs)
    k = {o: sum(r[o]["count"] for r in runs) for o in ("success", "fail", "stuck")}
    out = {"n": n, "n_seeds": len(runs), "seeds": " ".join(str(r["seed"]) for r in sorted(runs, key=lambda r: r["seed"]))}
    for o in ("success", "fail", "stuck"):
        p = k[o] / n
        lo, hi = wilson(k[o], n)
        out[f"{o}_rate"] = p
        out[f"{o}_se"] = math.sqrt(p * (1 - p) / n)
        out[f"{o}_ci95_lo"], out[f"{o}_ci95_hi"] = lo, hi
    for reason in FAIL_REASONS:
        out[f"fail_{reason}"] = sum(r["fail_reasons"].get(reason, 0) for r in runs)
    # weighted means of per-run means (weights = number of envs each mean is over)
    w_t = [(r["mean_time_to_success_s"], r["success"]["count"]) for r in runs if r["success"]["count"] > 0]
    w_v = [(r["mean_abs_vx_err_survivors"], r["n_survivors"]) for r in runs if r["n_survivors"] > 0]
    out["mean_time_to_success_s"] = sum(v * w for v, w in w_t) / sum(w for _, w in w_t) if w_t else float("nan")
    out["mean_abs_vx_err_survivors"] = sum(v * w for v, w in w_v) / sum(w for _, w in w_v) if w_v else float("nan")
    out["success_rate_per_seed"] = " ".join(
        f"s{r['seed']}={r['success']['rate']:.3f}" for r in sorted(runs, key=lambda r: r["seed"]))
    out["height_noise"] = json.dumps(runs[0]["height_noise"], sort_keys=True)
    out["proprio_noise"] = runs[0]["proprio_noise"]
    return out


def condition_sort_key(cond: str):
    if cond in P1_CONDITIONS:
        return (0, cond, 0.0)
    m = re.match(r"^([a-z]+)([\d.]+)$", cond)
    return (1, m[1], float(m[2])) if m else (2, cond, 0.0)


def style_axes(ax, ylabel: str):
    ax.set_facecolor(STYLE["surface"])
    ax.grid(axis="y", color=STYLE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(STYLE["text2"])
    ax.tick_params(colors=STYLE["text2"], labelsize=9)
    ax.set_ylabel(ylabel, color=STYLE["text"])
    ax.set_ylim(0, 1.0)


def axis_points(table: dict, model: str, axis: str) -> list[tuple[float, dict]]:
    prefix, x0, _ = AXES[axis]
    pts = [(x0, table[(model, "C1")])] if (model, "C1") in table else []
    for (m, cond), row in table.items():
        mm = re.match(rf"^{prefix}([\d.]+)$", cond)
        if m == model and mm:
            pts.append((float(mm[1]), row))
    return sorted(pts, key=lambda p: p[0])


def plot_axis(table: dict, axis: str, path: str, subtitle: str) -> bool:
    _, _, xlabel = AXES[axis]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=STYLE["surface"])
    plotted = False
    xs_all = sorted({x for model in MODELS for x, _ in axis_points(table, model, axis)})
    span = (xs_all[-1] - xs_all[0]) if len(xs_all) > 1 else 1.0
    for i, model in enumerate(MODELS):
        pts = axis_points(table, model, axis)
        if not pts:
            continue
        dodge = (i - (len(MODELS) - 1) / 2) * 0.012 * span  # keep the models' error bars from overlapping
        xs = [x + dodge for x, _ in pts]
        ys = [r["success_rate"] for _, r in pts]
        err = [[y - r["success_ci95_lo"] for y, (_, r) in zip(ys, pts, strict=True)],
               [r["success_ci95_hi"] - y for y, (_, r) in zip(ys, pts, strict=True)]]
        st = model_style(model)
        ax.errorbar(xs, ys, yerr=err, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                    linewidth=2, markersize=7, capsize=3, elinewidth=1.2,
                    markeredgecolor=STYLE["surface"], markeredgewidth=1.5,
                    label=f"{model} (n/point = {', '.join(sorted({str(r['n']) for _, r in pts}))})")
        ax.annotate(model, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=STYLE["text"])
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xticks(xs_all)
    ax.set_xticklabels([f"{x:g}" for x in xs_all])
    style_axes(ax, "success rate (95% Wilson CI)")
    ax.set_xlabel(xlabel, color=STYLE["text"], fontsize=9)
    ax.set_title(f"Stepping stones: success vs {axis}", color=STYLE["text"], fontsize=11, loc="left", pad=16)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color=STYLE["text2"], va="bottom")
    ax.legend(frameon=False, fontsize=8, labelcolor=STYLE["text"], loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=STYLE["surface"])
    plt.close(fig)
    return True


def plot_priority1(table: dict, path: str, subtitle: str) -> bool:
    conds = [c for c in P1_CONDITIONS if any((m, c) in table for m in MODELS)]
    if not conds:
        return False
    plt.rcParams["hatch.color"] = STYLE["surface"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=STYLE["surface"])
    width = 0.72 / max(1, len(MODELS))
    for i, model in enumerate(MODELS):
        for j, cond in enumerate(conds):
            row = table.get((model, cond))
            if row is None:
                continue
            x = j + (i - (len(MODELS) - 1) / 2) * (width + 0.03)  # 0.03 = surface gap between the bars
            y = row["success_rate"]
            ax.bar(x, y, width=width, color=model_style(model)["color"], edgecolor=STYLE["surface"], linewidth=0,
                   label=model if j == 0 else None, hatch=model_style(model)["hatch"])
            ax.errorbar(x, y, yerr=[[y - row["success_ci95_lo"]], [row["success_ci95_hi"] - y]],
                        color=STYLE["text"], capsize=3, elinewidth=1, linestyle="none")
            ax.text(x, row["success_ci95_hi"] + 0.015, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=8, color=STYLE["text"])
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([P1_CONDITIONS[c] for c in conds], fontsize=9)
    style_axes(ax, "success rate (95% Wilson CI)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Stepping stones: Priority-1 conditions", color=STYLE["text"], fontsize=11, loc="left", pad=16)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color=STYLE["text2"], va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=STYLE["text"], loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=STYLE["surface"])
    plt.close(fig)
    return True


CSV_FIELDS = [
    "model", "condition", "n", "n_seeds", "seeds",
    "success_rate", "success_se", "success_ci95_lo", "success_ci95_hi",
    "fail_rate", "fail_se", "fail_ci95_lo", "fail_ci95_hi",
    "stuck_rate", "stuck_se", "stuck_ci95_lo", "stuck_ci95_hi",
    *[f"fail_{r}" for r in FAIL_REASONS],
    "mean_time_to_success_s", "mean_abs_vx_err_survivors", "success_rate_per_seed", "height_noise", "proprio_noise",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "eval_results"))
    args = parser.parse_args()

    runs, skipped = load_runs(args.results_dir)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["model"], r["condition"]), []).append(r)
    MODELS[:] = sorted({r["model"] for r in runs})
    table = {key: pool(rs) for key, rs in groups.items()}
    keys = sorted(table, key=lambda k: (condition_sort_key(k[1]), k[0]))

    csv_path = os.path.join(args.results_dir, "sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for model, cond in keys:
            row = {"model": model, "condition": cond, **table[(model, cond)]}
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in CSV_FIELDS})

    plot_dir = os.path.join(args.results_dir, "sweep_plots")
    os.makedirs(plot_dir, exist_ok=True)
    envs = ", ".join(str(n) for n in sorted({r["num_envs"] for r in runs}))
    subtitle = f"{len(runs)} valid runs, seeds pooled; {envs or '?'} envs per run; proprio noise on except C0"
    made = [name for name, ok in
            [(f"{axis}.png", plot_axis(table, axis, os.path.join(plot_dir, f"{axis}.png"), subtitle)) for axis in AXES]
            + [("priority1.png", plot_priority1(table, os.path.join(plot_dir, "priority1.png"), subtitle))] if ok]

    # console table
    print(f"valid runs: {len(runs)}   skipped folders: {len(skipped)}")
    for s in skipped:
        print(f"  skipped {s}")
    w = max([5, *(len(m) for m in MODELS)])
    print(f"{'model':{w}} {'condition':10} {'n':>4} {'success [95% CI]':>22} {'fail':>6} {'stuck':>6} "
          f"{'contact/tilt/height':>20} {'t_succ':>6} {'|vx err|':>8}  per-seed success")
    for model, cond in keys:
        r = table[(model, cond)]
        print(f"{model:{w}} {cond:10} {r['n']:4d} {r['success_rate']:7.3f} [{r['success_ci95_lo']:.3f}, {r['success_ci95_hi']:.3f}]"
              f" {r['fail_rate']:6.3f} {r['stuck_rate']:6.3f} "
              f"{r['fail_contact']:>8d}/{r['fail_tilt']}/{r['fail_height']:<6d}    "
              f"{r['mean_time_to_success_s']:6.2f} {r['mean_abs_vx_err_survivors']:8.3f}  {r['success_rate_per_seed']}")
    print(f"wrote {csv_path}")
    print(f"wrote {len(made)} plot(s) to {plot_dir}: {', '.join(made)}")


if __name__ == "__main__":
    main()
