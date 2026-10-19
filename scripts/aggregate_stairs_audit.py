"""Aggregate the pyramid_stairs_inv curriculum audit (scripts/run_stairs_audit.sh). CPU only, no simulator.

Reads the valid runs tagged "sb-<commands>-d<d>-s<seed>" (latest per model and tag), pools the seeds per (model,
commands, difficulty) and writes logs/eval_results/stairs_audit/stairs_audit.csv plus a markdown table on stdout:
success (forward >= 4 m before a fail), escaped_pit (max xy distance from spawn in the first episode >= 3 m), the
gap curriculum's would-be decisions, "promoted by speed only but did not escape", mean |heading change| (final -
initial yaw, wrapped) and mean |total turn| (accumulated yaw). Rates: k/n with the Wilson 95% interval.

Example:
    uv run python scripts/aggregate_stairs_audit.py
"""

import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAG_RE = re.compile(r"^sb-(?P<cmd>fixed|train)-d(?P<d>[0-9.]+)-s(?P<seed>\d+)$")
Z95 = 1.959964


def wilson(k: int, n: int) -> tuple[float, float]:
  if n == 0:
    return float("nan"), float("nan")
  p = k / n
  denom = 1 + Z95**2 / n
  center = (p + Z95**2 / (2 * n)) / denom
  half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denom
  return max(0.0, center - half), min(1.0, center + half)


def main():
  results = os.path.join(REPO_ROOT, "logs", "eval_results")
  latest = {}
  for path in sorted(glob.glob(os.path.join(results, "*", "summary.json"))):
    with open(path) as fh:
      s = json.load(fh)
    m = TAG_RE.match(s.get("tag") or "")
    if not m or s.get("valid") is not True or not s.get("curriculum_audit"):
      continue
    with open(os.path.join(os.path.dirname(path), "per_env.csv")) as fh:
      latest[(s["model_name"], s["tag"])] = (m, list(csv.DictReader(fh)))
  cells = defaultdict(list)
  for (model, _), (m, rows) in latest.items():
    cells[(model, m["cmd"], float(m["d"]))].append(rows)

  table = []
  for (model, cmd, d), runs in sorted(cells.items()):
    rows = [r for run in runs for r in run]
    n = len(rows)
    count = lambda pred, rows=rows: sum(1 for r in rows if pred(r))  # noqa: E731
    k_succ = count(lambda r: r["outcome"] == "success")
    k_esc = count(lambda r: r["escaped_pit"] == "True")
    k_spd = count(lambda r: r["curriculum_decision"] == "promote_speed_only")
    k_spd_no = count(
      lambda r: (
        r["curriculum_decision"] == "promote_speed_only" and r["escaped_pit"] == "False"
      )
    )
    row = {
      "model": model,
      "commands": cmd,
      "difficulty": d,
      "n_seeds": len(runs),
      "n": n,
    }
    for name, k in (
      ("success", k_succ),
      ("escaped_pit", k_esc),
      ("promote_speed_only", k_spd),
      ("promote_speed_only_not_escaped", k_spd_no),
    ):
      lo, hi = wilson(k, n)
      row.update({f"{name}_rate": k / n, f"{name}_ci_lo": lo, f"{name}_ci_hi": hi})
    for dec in ("promote_distance", "demote", "none"):
      row[f"{dec}_rate"] = count(lambda r, dec=dec: r["curriculum_decision"] == dec) / n
    row["fail_rate"] = count(lambda r: r["outcome"] == "fail") / n
    for key in ("heading_change_deg", "total_turn_deg"):
      row[f"mean_abs_{key}"] = sum(abs(float(r[key])) for r in rows) / n
    for key in ("speed_ratio", "max_rise_m", "max_xy_episode_m"):
      row[f"mean_{key}"] = sum(float(r[key]) for r in rows) / n
    table.append(row)

  out_dir = os.path.join(results, "stairs_audit")
  os.makedirs(out_dir, exist_ok=True)
  csv_path = os.path.join(out_dir, "stairs_audit.csv")
  with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
    writer.writeheader()
    for row in table:
      writer.writerow(
        {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()}
      )

  print(
    "| model | commands | d | n | success | escaped_pit | promote speed-only | speed-only & not escaped | "
    "demote | mean abs heading change deg | mean abs total turn deg | mean speed ratio | mean max rise m |"
  )
  print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
  for r in table:
    print(
      f"| {r['model']} | {r['commands']} | {r['difficulty']:g} | {r['n']} | {r['success_rate']:.2f} | "
      f"{r['escaped_pit_rate']:.2f} | {r['promote_speed_only_rate']:.2f} | "
      f"{r['promote_speed_only_not_escaped_rate']:.2f} | {r['demote_rate']:.2f} | "
      f"{r['mean_abs_heading_change_deg']:.0f} | {r['mean_abs_total_turn_deg']:.0f} | "
      f"{r['mean_speed_ratio']:.2f} | {r['mean_max_rise_m']:.2f} |"
    )
  print(f"\n[INFO] {len(latest)} runs -> {csv_path}")


if __name__ == "__main__":
  main()
