#!/usr/bin/env bash
# Cluster B.3 multi-goal cheat-5 re-eval.
# 5a: full Cluster B (B.1 + B.2 + B.3) baseline, no cross-projections.
# 5b: full Cluster B + patch-matrix cross-projections (density 0.25).
# Each: 3 seeds (42, 43, 44) at 1800 steps, --goal-schedule multi.

set -e
cd "$(dirname "$0")/../../.."

OUT_DIR="research/findings/raw/g11_bg"
LOG_DIR="research/findings/raw/g11_bg/clusterB3_eval_logs"
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "=== Cluster B.3 eval starting at $(date) ==="

for SEED in 42 43 44; do
    # 5a: full Cluster B, no cross-projections
    OUT_5A="$OUT_DIR/g11_seed${SEED}_clusterB3_no_cross.json"
    LOG_5A="$LOG_DIR/seed${SEED}_no_cross.log"
    if [ -f "$OUT_5A" ]; then
        echo "[seed $SEED 5a] SKIP (exists: $OUT_5A)"
    else
        echo "[seed $SEED 5a] starting at $(date '+%H:%M:%S')"
        python -m research.runners.g11_bg_runner --moving-goal --bg-lateral-inhibition \
            --enable-d1-d2-asymmetry --enable-striatal-fsis --enable-tans \
            --goal-schedule multi --seed $SEED --n-steps 1800 \
            --out "$OUT_5A" > "$LOG_5A" 2>&1
        echo "[seed $SEED 5a] done at $(date '+%H:%M:%S')"
    fi

    # 5b: full Cluster B + patch-matrix cross-projections
    OUT_5B="$OUT_DIR/g11_seed${SEED}_clusterB3_patch_matrix.json"
    LOG_5B="$LOG_DIR/seed${SEED}_patch_matrix.log"
    if [ -f "$OUT_5B" ]; then
        echo "[seed $SEED 5b] SKIP (exists: $OUT_5B)"
    else
        echo "[seed $SEED 5b] starting at $(date '+%H:%M:%S')"
        python -m research.runners.g11_bg_runner --moving-goal --bg-lateral-inhibition \
            --enable-d1-d2-asymmetry --enable-striatal-fsis --enable-tans \
            --bg-cross-projections --cross-projection-density 0.25 \
            --goal-schedule multi --seed $SEED --n-steps 1800 \
            --out "$OUT_5B" > "$LOG_5B" 2>&1
        echo "[seed $SEED 5b] done at $(date '+%H:%M:%S')"
    fi
done

ELAPSED=$(( $(date +%s) - START ))
echo "=== Cluster B.3 eval done at $(date) — elapsed ${ELAPSED}s ==="
