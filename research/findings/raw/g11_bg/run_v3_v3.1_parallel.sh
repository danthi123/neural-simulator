#!/usr/bin/env bash
# Parallel launch: v3 seeds 100/101/102 + all 6 v3.1 seeds.
# Combined with the already-running v3 seeds 43/44, gives full 6-seed
# v3 + 6-seed v3.1 in concurrent execution.
#
# Memory: ~9 new × 1.3 GB = 11.7 GB on top of the 2 already running (2.6 GB).
# Total ~14.3 GB out of 25.8 GB available — comfortable.

mkdir -p research/findings/raw/g11_bg

run_v3() {
    local SEED=$1
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
        --out "research/findings/raw/g11_bg/g11_seed${SEED}_v3lateral.json" \
        > "research/findings/raw/g11_bg/v3_seed${SEED}.log" 2>&1
}

run_v3_1() {
    local SEED=$1
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-lateral-inhibition \
        --bg-cross-projections --cross-projection-weight 0.0 \
        --bg-cross-thaw-step 1200 --bg-cross-phase3-gain 0.5 \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 600 \
        --seed "$SEED" --n-steps 1800 \
        --out "research/findings/raw/g11_bg/g11_seed${SEED}_v3.1cross.json" \
        > "research/findings/raw/g11_bg/v3.1_seed${SEED}.log" 2>&1
}

echo "=== launching parallel batch: $(date) ==="

# v3 seeds: 100, 101, 102 (43, 44 already running separately)
run_v3 100 &
run_v3 101 &
run_v3 102 &

# v3.1 seeds: all 6
run_v3_1 42 &
run_v3_1 43 &
run_v3_1 44 &
run_v3_1 100 &
run_v3_1 101 &
run_v3_1 102 &

echo "=== launched 9 processes; waiting for completion ==="
wait
echo "=== ALL parallel processes done: $(date) ==="
