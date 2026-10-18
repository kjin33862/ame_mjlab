#!/usr/bin/env bash
# Full evaluation of ONE checkpoint: a fixed terrain x difficulty x seed matrix, then a noise sweep on stone_lattice.
#
# Matrix: terrains flat (d = 0 only), pyramid_stairs, pyramid_stairs_inv, random_rough, wave, stone_lattice,
# stepping_stones; difficulties 0.0 0.25 0.5 0.75 1.0; seeds 0 1 2; noise off. Gait fractions are always recorded by
# evaluate.py. Tags "mx-<terrain>-d<d>-s<seed>".
# Noise sweep: stone_lattice d = 0.5, seeds 0 1 2. Conditions: off, train (training map noise + proprio noise), and
# one map parameter at a time from "off" (no proprio noise): height_sigma 0.06 0.10, height_offset 0.10 0.15,
# drift_cells 1 2 3, map_delay_steps 5 10. Tags "nz-<cond>-s<seed>".
#
# Every run: 8 x 8 tiles, one robot per tile, --model_name <name>. Restart-safe: a run whose folder already has a
# summary.json with "valid": true and the same model_name is skipped. The whole list is gone through twice, so a run
# that failed or was invalid the first time is retried once. Per-run stdout: logs/eval_results/matrix_logs/.
#
# Usage (from the repo root):
#   nohup systemd-inhibit --what=sleep:idle --why="eval" bash scripts/run_policy_matrix.sh \
#     baseline checkpoints/baseline/model_9999.pt > logs/eval_results/matrix_baseline.log 2>&1 &
# Optional: PARTS (default "matrix noise"), SEEDS (default "0 1 2"), NUM_ROWS / NUM_COLS (default 8); TERRAINS and
# DIFFS (default: the matrix above) only for tests: a smaller matrix is not the fixed one.

set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:?usage: run_policy_matrix.sh <model_name> <checkpoint>}"
CKPT="${2:?usage: run_policy_matrix.sh <model_name> <checkpoint>}"
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
PARTS="${PARTS:-matrix noise}"
SEEDS="${SEEDS:-0 1 2}"
NUM_ROWS="${NUM_ROWS:-8}"
NUM_COLS="${NUM_COLS:-8}"
TERRAINS="${TERRAINS:-flat pyramid_stairs pyramid_stairs_inv random_rough wave stone_lattice stepping_stones}"
DIFFS="${DIFFS:-0.0 0.25 0.5 0.75 1.0}"
OUT=logs/eval_results
LOGDIR="$OUT/matrix_logs"
RUN_TIMEOUT=1800  # [s] kill a hung run after 30 min

mkdir -p "$LOGDIR"
n_ok=0; n_skip=0; n_bad=0
t_all=$(date +%s)

log() { echo "[$(date '+%F %T')] $*"; }

# Prints the folder of a valid run of $MODEL with this tag, if any.
valid_run() {
    local tag=$1 f
    for f in "$OUT"/*_"${MODEL}"_*_"${tag}"/summary.json; do
        [ -f "$f" ] || continue
        if python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("valid") is True and s.get("model_name")==sys.argv[2] else 1)' "$f" "$MODEL"; then
            dirname "$f"; return 0
        fi
    done
    return 1
}

# run <tag> [evaluate.py args...]
run() {
    local tag=$1; shift
    local done_dir
    if done_dir=$(valid_run "$tag"); then
        n_skip=$((n_skip + 1)); return
    fi
    local logf="$LOGDIR/${MODEL}_${tag}_$(date +%Y%m%d-%H%M%S).log"
    log "START $MODEL $tag  args: $*"
    local t0=$(date +%s)
    timeout --kill-after=60 "$RUN_TIMEOUT" \
        uv run python -u scripts/evaluate.py --checkpoint "$CKPT" --model_name "$MODEL" \
        --num_rows "$NUM_ROWS" --num_cols "$NUM_COLS" --tag "$tag" "$@" > "$logf" 2>&1
    local rc=$? dt=$(( $(date +%s) - t0 ))
    if done_dir=$(valid_run "$tag"); then
        log "OK    $MODEL $tag  (${dt}s) $(grep -m1 '\[INFO\] success' "$logf")"
        n_ok=$((n_ok + 1))
    else
        log "FAIL  $MODEL $tag  (${dt}s, exit $rc) $(grep -m1 'INVALID RUN' "$logf") -> see $logf"
        n_bad=$((n_bad + 1))
    fi
}

all_runs() {
    local seed terrain d v
    if [[ " $PARTS " == *" matrix "* ]]; then
        for seed in $SEEDS; do
            for terrain in $TERRAINS; do
                for d in $DIFFS; do
                    [ "$terrain" = flat ] && [ "$d" != 0.0 ] && continue
                    run "mx-${terrain}-d${d}-s${seed}" --terrain "$terrain" --difficulty "$d" --seed "$seed" --noise off
                done
            done
        done
    fi
    if [[ " $PARTS " == *" noise "* ]]; then
        local fixed=(--terrain stone_lattice --difficulty 0.5)
        for seed in $SEEDS; do
            run "nz-off-s${seed}" "${fixed[@]}" --seed "$seed" --noise off
            run "nz-train-s${seed}" "${fixed[@]}" --seed "$seed" --noise train
            for v in 0.06 0.10; do run "nz-sigma${v}-s${seed}" "${fixed[@]}" --seed "$seed" --noise off --height_sigma "$v"; done
            for v in 0.10 0.15; do run "nz-offset${v}-s${seed}" "${fixed[@]}" --seed "$seed" --noise off --height_offset "$v"; done
            for v in 1 2 3; do run "nz-drift${v}-s${seed}" "${fixed[@]}" --seed "$seed" --noise off --drift_cells "$v"; done
            for v in 5 10; do run "nz-delay${v}-s${seed}" "${fixed[@]}" --seed "$seed" --noise off --map_delay_steps "$v"; done
        done
    fi
}

log "Start. model: $MODEL  checkpoint: $CKPT  parts: $PARTS  seeds: $SEEDS  grid: ${NUM_ROWS}x${NUM_COLS}"
for pass in 1 2; do
    log "Pass $pass"
    all_runs
    [ "$pass" = 1 ] && [ "$n_bad" = 0 ] && break
done
log "Done in $(( ($(date +%s) - t_all) / 60 )) min: ok=$n_ok skipped=$n_skip failed/invalid=$n_bad (failures of pass 1 were retried once)"
