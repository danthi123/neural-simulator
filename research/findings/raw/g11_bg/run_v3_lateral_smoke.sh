#!/usr/bin/env bash
# v3 (MSN lateral inhibition) — no-regression check vs flagship.
# Plan: docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md
# Estimated wall-clock: ~42 min (14 min × 3 seeds, GPU-serialized).
set -e
LOG=research/findings/raw/g11_bg/v3_lateral_smoke.log
echo "=== v3 lateral inhibition no-regression smoke started: $(date) ===" | tee -a "$LOG"

for SEED in 42 43 44; do
    echo "=== seed $SEED start: $(date) ===" | tee -a "$LOG"
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-lateral-inhibition \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed "$SEED" --n-steps 1800 \
        --out "research/findings/raw/g11_bg/g11_seed${SEED}_v3lateral.json" 2>&1 | tee -a "$LOG"
    echo "=== seed $SEED done:  $(date) ===" | tee -a "$LOG"
done

echo "=== ALL DONE: $(date) ===" | tee -a "$LOG"
