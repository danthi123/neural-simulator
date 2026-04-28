#!/usr/bin/env bash
# Cheat #5 closure: 6-seed validation
# Plan: docs/plans/2026-04-28-cheat5-curriculum-staged-bg-cross.md
# Estimated wall-clock: ~96 min total (16 min × 6 seeds, GPU-serialized).

set -e
LOG=research/findings/raw/g11_bg/cheat5_validation.log
echo "=== Cheat #5 6-seed validation started: $(date) ===" | tee -a "$LOG"

for SEED in 42 43 44 100 101 102; do
    echo "=== seed $SEED start: $(date) ===" | tee -a "$LOG"
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-cross-projections --bg-cross-thaw-step 1200 --bg-cross-phase3-gain 0.5 \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed "$SEED" --n-steps 1800 \
        --out "research/findings/raw/g11_bg/g11_seed${SEED}_cheat5.json" 2>&1 | tee -a "$LOG"
    echo "=== seed $SEED done:  $(date) ===" | tee -a "$LOG"
done

echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
