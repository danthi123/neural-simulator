#!/usr/bin/env bash
# GPU throughput sweep for g11_bg_runner moving-goal config.
# Phase 2/3 of the throughput investigation.
#
# Tests four conditions, capturing step/s for each:
#   A. Baseline:          1 run, no MPS, --progress-print-interval 10
#   B. Per-step sync off: 1 run, no MPS, --progress-print-interval 10  (same as A — placeholder for code fix)
#   C. MPS on:            1 run, MPS daemon, --progress-print-interval 10
#   D. Concurrency sweep: 1, 4, 8 concurrent at MPS+ppi=10
#
# Run AFTER the in-flight batch finishes (currently 10 cheat-5 runs).
# Verify with:
#   curl -s http://localhost:8765/api/runs/launch | python -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for r in d['runs'] if r['running']))"
# Should print 0 before starting this script.

set -euo pipefail

OUT_DIR="research/findings/raw/throughput_test"
mkdir -p "$OUT_DIR"
PYTHON="${PYTHON:-c:/python312/python.exe}"
BASE_ARGS=(
    -m research.runners.g11_bg_runner --moving-goal
    --hippocampus --learned-perception --pfc
    --beacon-perception --beacon-replaces-goal
    --cue-reflex --cue-reflex-replaces-heuristic
    --landmarks --landmarks-replace-place
    --sensed-reward
    --bg-lateral-inhibition
    --adaptive-da --adaptive-da-ema-decay-negative 0.7
    --curriculum --curriculum-warmup-steps 600
    --n-steps 400  # short — measure rate, not result
    --progress-print-interval 10
)

measure_rate() {
    local label="$1"; shift
    local n_concurrent="${1:-1}"; shift || true
    local seeds=(42 43 44 100 101 102 200 201)
    echo
    echo "=== $label (n=$n_concurrent) ==="
    local pids=()
    local logs=()
    local start=$(date +%s)
    for ((i=0; i<n_concurrent; i++)); do
        local seed="${seeds[$i]}"
        local log="$OUT_DIR/${label}_seed${seed}.log"
        logs+=("$log")
        "$PYTHON" "${BASE_ARGS[@]}" --seed "$seed" \
            --out "$OUT_DIR/${label}_seed${seed}.json" > "$log" 2>&1 &
        pids+=($!)
    done
    # Wait for all
    for pid in "${pids[@]}"; do wait "$pid"; done
    local end=$(date +%s)
    local elapsed=$((end - start))

    # Compute step/s per run (from final progress line)
    local total_steps=0
    for log in "${logs[@]}"; do
        local steps=$(grep -oP 'step \K\d+(?=/\d+)' "$log" | tail -1 || echo 0)
        total_steps=$((total_steps + steps))
    done
    local per_run=$((total_steps / n_concurrent))
    local rate_per_run=$(awk "BEGIN{printf \"%.2f\", $per_run / $elapsed}")
    local rate_aggregate=$(awk "BEGIN{printf \"%.2f\", $total_steps / $elapsed}")
    echo "  elapsed=${elapsed}s  per-run-steps=$per_run  rate=${rate_per_run} step/s/run  aggregate=${rate_aggregate} step/s"
    echo "$label,$n_concurrent,$elapsed,$per_run,$rate_per_run,$rate_aggregate" >> "$OUT_DIR/results.csv"
}

# Reset CSV
echo "label,n_concurrent,elapsed_sec,per_run_steps,rate_per_run,rate_aggregate" > "$OUT_DIR/results.csv"

# A: baseline — 1 run, no MPS
measure_rate "A_baseline_1" 1

# C: MPS on — start daemon, 1 run
echo
echo "Starting CUDA MPS daemon..."
nvidia-cuda-mps-control -d || echo "MPS daemon already running or failed to start"
sleep 2
measure_rate "C_mps_1" 1

# D: concurrency sweep with MPS
measure_rate "D_mps_4" 4
measure_rate "D_mps_8" 8

# Stop MPS daemon
echo
echo "Stopping CUDA MPS daemon..."
echo quit | nvidia-cuda-mps-control || echo "MPS daemon stop failed"

echo
echo "=== Summary ==="
column -t -s, "$OUT_DIR/results.csv"
