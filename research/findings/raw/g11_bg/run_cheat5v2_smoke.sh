#!/usr/bin/env bash
# Cheat #5 v2 closure: 3-seed smoke with cross_projection_weight=0.0
# Plan: docs/plans/2026-04-28-cheat5-v2-zero-init.md
# Estimated wall-clock: ~42 min (14 min × 3 seeds, GPU-serialized).
set -e
LOG=research/findings/raw/g11_bg/cheat5v2_smoke.log
echo "=== Cheat #5 v2 3-seed smoke started: $(date) ===" | tee -a "$LOG"

for SEED in 42 43 44; do
    echo "=== seed $SEED start: $(date) ===" | tee -a "$LOG"
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-cross-projections --cross-projection-weight 0.0 \
        --bg-cross-thaw-step 1200 --bg-cross-phase3-gain 0.5 \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed "$SEED" --n-steps 1800 \
        --out "research/findings/raw/g11_bg/g11_seed${SEED}_cheat5v2.json" 2>&1 | tee -a "$LOG"
    echo "=== seed $SEED done:  $(date) ===" | tee -a "$LOG"
done

echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
