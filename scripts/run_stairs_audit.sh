#!/usr/bin/env bash
# Curriculum audit on pyramid_stairs_inv: both checkpoints x commands fixed/train x d 0..1 x seeds 0 1 2, noise off,
# --curriculum_audit, 8x8 tiles. One run at a time; restart-safe (skips a valid summary with the same model and tag).
# Tags "sb-<commands>-d<d>-s<seed>" (the earlier "sa-" runs had an end-distance bug, see the log). Usage: nohup systemd-inhibit --what=sleep:idle --why=eval \
#   bash scripts/run_stairs_audit.sh > logs/eval_results/stairs_audit.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/eval_results
LOGDIR="$OUT/stairs_audit_logs"
mkdir -p "$LOGDIR"
MODELS="${MODELS:-shuang_gap2048=checkpoints/shuang_gap2048/model_14999.pt baseline=checkpoints/baseline/model_9999.pt}"
log() { echo "[$(date '+%F %T')] $*"; }
valid_run() {
    local model=$1 tag=$2 f
    for f in "$OUT"/*_"${model}"_pyramid_stairs_inv_*_"${tag}"/summary.json; do
        [ -f "$f" ] || continue
        python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("valid") is True and s.get("model_name")==sys.argv[2] else 1)' "$f" "$model" && return 0
    done
    return 1
}
for pass in 1 2; do
    for cmd in fixed train; do
        for pair in $MODELS; do
            model=${pair%%=*}; ckpt=${pair#*=}
            for seed in 0 1 2; do
                for d in 0.0 0.25 0.5 0.75 1.0; do
                    tag="sb-${cmd}-d${d}-s${seed}"
                    valid_run "$model" "$tag" && continue
                    logf="$LOGDIR/${model}_${tag}.log"
                    log "START $model $tag"
                    timeout --kill-after=60 1800 uv run python -u scripts/evaluate.py --checkpoint "$ckpt" \
                        --model_name "$model" --terrain pyramid_stairs_inv --difficulty "$d" --seed "$seed" \
                        --noise off --commands "$cmd" --curriculum_audit --tag "$tag" > "$logf" 2>&1
                    if valid_run "$model" "$tag"; then log "OK    $model $tag $(grep -m1 '\[INFO\] success' "$logf")"
                    else log "FAIL  $model $tag -> $logf"; fi
                done
            done
        done
    done
done
log "Done"
