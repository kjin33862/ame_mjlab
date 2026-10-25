"""Aggregate the goal-mode stairs runs (scripts/run_policy_matrix.sh with PARTS=goal). CPU only, no simulator.

Reads logs/eval_results/*/summary.json + per_env.csv of the valid runs tagged "gm-<terrain>-d<d>-s<seed>" (the latest
per model and tag), pools the seeds per (model, terrain, difficulty) and writes logs/eval_results/goal_plots/
goal_summary.csv. The open-mode success rate of the same cell comes from matrix_plots/matrix_summary.csv (read only).
Prints a markdown table: open vs goal success, mean max |heading error to the goal| and mean fraction of tracked steps
with |err| > 45 deg (per env, then the mean over envs).

Example:
    uv run python scripts/aggregate_goal.py --models baseline shuang_gap2048 gap_1024 gap_progress_1024
"""

import argparse
import csv
import glob
import json
import os
import re

from aggregate_matrix import wilson

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GM_RE = re.compile(r"^gm-(?P<terrain>.+)-d(?P<d>[0-9.]+)-s(?P<seed>\d+)$")


def load_goal_runs(results_dir: str) -> dict:
  """{(model, terrain, difficulty): [per-env rows of each seed's latest valid goal-mode run]}."""
  latest = {}
  for path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
    with open(path) as fh:
      s = json.load(fh)
    m = GM_RE.match(s.get("tag") or "")
    if not m or s.get("valid") is not True or s.get("cmd_mode") != "goal":
      continue
    with open(os.path.join(os.path.dirname(path), "per_env.csv")) as fh:
      latest[(s["model_name"], s["tag"])] = (m, list(csv.DictReader(fh)))
  cells = {}
  for (model, _), (m, rows) in latest.items():
    cells.setdefault((model, m["terrain"], float(m["d"])), []).append(rows)
  return cells


def main():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "logs", "eval_results"))
  p.add_argument("--models", nargs="*", default=None, help="Order of the models (default: by name).")
  args = p.parse_args()

  open_rate = {}
  with open(os.path.join(args.results_dir, "matrix_plots", "matrix_summary.csv")) as fh:
    for r in csv.DictReader(fh):
      if r["kind"] == "matrix":
        open_rate[(r["model"], r["terrain"], float(r["difficulty"]))] = (
          float(r["success_rate"]),
          r["success_rate_per_seed"],
        )

  cells = load_goal_runs(args.results_dir)
  order = args.models or sorted({k[0] for k in cells})
  out = []
  for key in sorted(cells, key=lambda k: (order.index(k[0]) if k[0] in order else len(order), k[1], k[2])):
    runs = cells[key]
    rows = [r for env_rows in runs for r in env_rows]
    tracked = [r for r in rows if int(r["tracked_steps"]) > 0]
    k = sum(r["outcome"] == "success" for r in rows)
    lo, hi = wilson(k, len(rows))
    o_rate, o_seeds = open_rate.get(key, (float("nan"), ""))
    out.append(
      {
        "model": key[0],
        "terrain": key[1],
        "difficulty": f"{key[2]:.4f}",
        "n_seeds": len(runs),
        "n": len(rows),
        "open_success_rate": f"{o_rate:.4f}",
        "open_success_rate_per_seed": o_seeds,
        "goal_success_k": k,
        "goal_success_rate": f"{k / len(rows):.4f}",
        "goal_success_ci_lo": f"{lo:.4f}",
        "goal_success_ci_hi": f"{hi:.4f}",
        "goal_success_rate_per_seed": " ".join(
          f"{sum(r['outcome'] == 'success' for r in env_rows) / len(env_rows):.3f}" for env_rows in runs
        ),
        "goal_fail_rate": f"{sum(r['outcome'] == 'fail' for r in rows) / len(rows):.4f}",
        "mean_max_abs_goal_err_deg": f"{sum(float(r['max_abs_goal_err_deg']) for r in tracked) / len(tracked):.1f}",
        "mean_goal_err_gt45_frac": f"{sum(float(r['goal_err_gt45_frac']) for r in tracked) / len(tracked):.3f}",
      }
    )

  out_dir = os.path.join(args.results_dir, "goal_plots")
  os.makedirs(out_dir, exist_ok=True)
  out_csv = os.path.join(out_dir, "goal_summary.csv")
  with open(out_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    writer.writeheader()
    writer.writerows(out)

  print("| model | terrain | d | open success | goal success (95% CI) | mean max abs err [deg] | frac abs err > 45 deg |")
  print("|---|---|---|---|---|---|---|")
  for r in out:
    print(
      f"| {r['model']} | {r['terrain']} | {float(r['difficulty']):g} | {float(r['open_success_rate']):.3f} | "
      f"{float(r['goal_success_rate']):.3f} ({float(r['goal_success_ci_lo']):.2f}-{float(r['goal_success_ci_hi']):.2f}) | "
      f"{r['mean_max_abs_goal_err_deg']} | {r['mean_goal_err_gt45_frac']} |"
    )
  print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
  main()
