#!/usr/bin/env bash
# GPU / host memory test of the evaluation: 64, 128 and 225 envs on the 8x8 stepping-stone board (several robots per
# tile above 64), random-init policy, 200 steps each. Prints the measured memory from each summary.json.
#
# Usage (from the repo root, GPU required):  bash scripts/eval_memory_test.sh
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/eval_results/memory_test
mkdir -p "$OUT"
for n in ${ENVS:-64 128 225}; do
    log="$OUT/envs$n.log"
    uv run python -u scripts/evaluate.py --random_policy --num_rows 8 --num_cols 8 --num_envs "$n" \
        --max_steps 200 --out_root "$OUT" --tag "mem$n" > "$log" 2>&1
    echo "envs=$n exit=$? log=$log"
done
for f in "$OUT"/*/summary.json; do
    python3 - "$f" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
if not s.get("valid"):
    print(f"{sys.argv[1]}: INVALID ({s.get('invalid_reason')})")
    sys.exit()
used = (s["cuda_device_used_mb_peak"] or 0) - (s["cuda_device_used_mb_before"] or 0)
print(f"envs={s['num_envs']:4d}  GPU used by run (device-level peak) {used:7.0f} MB  "
      f"torch peak {s['torch_max_memory_allocated_mb'] or 0:6.0f} MB  host RSS {s['host_max_rss_mb'] / 1024:5.1f} GB  "
      f"build {s['env_build_s']:5.1f} s  {s['steps_simulated']} steps in {s['wall_time_s']:5.1f} s")
EOF
done
