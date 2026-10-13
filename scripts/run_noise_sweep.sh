#!/usr/bin/env bash
# Noise-robustness sweep on stepping stones (mjlab port of AME_Locomotion's run_noise_sweep.sh).
#
# Every run: one checkpoint, NUM_ROWS x NUM_COLS tiles (one robot per tile), proprio noise on (except C0), one map-
# noise parameter changed at a time from the no-map-noise baseline C1. Output folders end in "_sweep-<cond>-s<seed>";
# per-run stdout goes to logs/eval_results/sweep_logs/. Restart-safe: a run whose folder already has a summary.json
# with "valid": true is skipped. A failed or invalid run is logged and the sweep moves on.
#
# Usage (from the repo root), one "name=checkpoint" pair per model:
#   MODELS="drift_on=logs/rsl_rl/ame_g1/<run>/model_10000.pt drift_off=logs/rsl_rl/ame_g1/<run2>/model_10000.pt" \
#     nohup systemd-inhibit --what=sleep:idle --why="AME sweep" bash scripts/run_noise_sweep.sh \
#       > logs/eval_results/sweep_master.log 2>&1 &
# Optional: P1_SEEDS / P2_SEEDS (default "0 1 2"), NUM_ROWS / NUM_COLS (default 10), DIFFICULTY (default 0.5).

set -u
cd "$(dirname "$0")/.." || exit 1

OUT=logs/eval_results
LOGDIR="$OUT/sweep_logs"
MODELS="${MODELS:?set MODELS=\"name=checkpoint ...\"}"
P1_SEEDS="${P1_SEEDS:-0 1 2}"
P2_SEEDS="${P2_SEEDS:-0 1 2}"
NUM_ROWS="${NUM_ROWS:-10}"
NUM_COLS="${NUM_COLS:-10}"
DIFFICULTY="${DIFFICULTY:-0.5}"
RUN_TIMEOUT=1800  # [s] kill a hung run after 30 min
FIXED=(--terrain stepping_stones --num_rows "$NUM_ROWS" --num_cols "$NUM_COLS" --difficulty "$DIFFICULTY")
PROPRIO_ON=(--proprio_noise on)
MAP_OFF=(--height_sigma 0 --height_offset 0 --drift_cells 0)

mkdir -p "$LOGDIR"
n_ok=0; n_skip=0; n_bad=0
t_sweep=$(date +%s)

log() { echo "[$(date '+%F %T')] $*"; }

# Prints the folder of a valid run for this (model, tag), if any.
valid_run() {
    local model=$1 tag=$2 f
    for f in "$OUT"/*_"${model}"_stepping_stones_*_"${tag}"/summary.json; do
        [ -f "$f" ] || continue
        if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("valid") is True else 1)' "$f"; then
            dirname "$f"; return 0
        fi
    done
    return 1
}

# run <condition> <model> <checkpoint> <seed> [evaluate.py noise args...]
run() {
    local cond=$1 model=$2 ckpt=$3 seed=$4; shift 4
    local tag="sweep-${cond}-s${seed}"
    local done_dir
    if done_dir=$(valid_run "$model" "$tag"); then
        log "SKIP  $model $tag (valid: $done_dir)"; n_skip=$((n_skip + 1)); return
    fi
    local logf="$LOGDIR/${model}_${tag}_$(date +%Y%m%d-%H%M%S).log"
    log "START $model $tag  args: $*"
    local t0=$(date +%s)
    timeout --kill-after=60 "$RUN_TIMEOUT" \
        uv run python -u scripts/evaluate.py --checkpoint "$ckpt" --model_name "$model" "${FIXED[@]}" \
        --seed "$seed" --tag "$tag" "$@" > "$logf" 2>&1
    local rc=$? dt=$(( $(date +%s) - t0 ))
    if done_dir=$(valid_run "$model" "$tag"); then
        log "OK    $model $tag  (${dt}s) $(grep -m1 '\[INFO\] success' "$logf")"
        n_ok=$((n_ok + 1))
    else
        log "FAIL  $model $tag  (${dt}s, exit $rc) $(grep -m1 'INVALID RUN' "$logf") -> see $logf"
        n_bad=$((n_bad + 1))
    fi
}

log "Sweep start. models: $MODELS  P1 seeds: $P1_SEEDS  P2 seeds: $P2_SEEDS  grid: ${NUM_ROWS}x${NUM_COLS}"

# ---- Priority 1: all models x all seeds
for seed in $P1_SEEDS; do
    for pair in $MODELS; do
        model=${pair%%=*}; ckpt=${pair#*=}
        run C0 "$model" "$ckpt" "$seed" --noise off --proprio_noise off
        run C1 "$model" "$ckpt" "$seed" "${PROPRIO_ON[@]}" "${MAP_OFF[@]}"
        run C2 "$model" "$ckpt" "$seed" "${PROPRIO_ON[@]}" --drift_cells 0
        run C3 "$model" "$ckpt" "$seed" "${PROPRIO_ON[@]}"
    done
done

# ---- Priority 2: one parameter at a time from C1 (no map noise), ordered by seed
for seed in $P2_SEEDS; do
    for pair in $MODELS; do
        model=${pair%%=*}; ckpt=${pair#*=}
        for v in 0.06 0.10; do
            run "sigma$v" "$model" "$ckpt" "$seed" "${PROPRIO_ON[@]}" "${MAP_OFF[@]}" --height_sigma "$v"
        done
        for v in 0.05 0.10 0.15; do
            run "offset$v" "$model" "$ckpt" "$seed" "${PROPRIO_ON[@]}" "${MAP_OFF[@]}" --height_offset "$v"
        done
        for v in 1 2 3; do
            run "drift$v" "$model" "$ckpt" "$seed" "${PROPRIO_ON[@]}" "${MAP_OFF[@]}" --drift_cells "$v"
        done
        for v in 5 10; do
            run "delay$v" "$model" "$ckpt" "$seed" "${PROPRIO_ON[@]}" "${MAP_OFF[@]}" --map_delay_steps "$v"
        done
    done
done

log "Sweep done in $(( ($(date +%s) - t_sweep) / 60 )) min: ok=$n_ok skipped=$n_skip failed/invalid=$n_bad"
