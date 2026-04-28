#!/usr/bin/env bash
# Profile the g11_bg_runner moving-goal pipeline with nsys (NVIDIA Nsight Systems).
#
# Outputs a .nsys-rep file you can open in Nsight Systems UI, plus a textual
# CUDA kernel summary (top kernels by cumulative time, top kernels by call count,
# memcpy DtoH/HtoD totals).
#
# Usage:
#   ./scripts/profile_g11_runner.sh                          # quick: 200 steps, no pretraining
#   ./scripts/profile_g11_runner.sh --pretraining            # 200 pretrain + 50 eval
#   ./scripts/profile_g11_runner.sh --eval-only --n-steps 500
#
# Prerequisites:
#   - nsys binary on PATH (NVIDIA Nsight Systems; ships with CUDA Toolkit)
#   - GPU idle (no concurrent runs — profiling under contention pollutes results)
#
# Output:
#   - profile_outputs/g11_<timestamp>.nsys-rep  (open in Nsight Systems UI)
#   - profile_outputs/g11_<timestamp>.summary.txt  (kernel summary tables)

set -euo pipefail

OUT_DIR="profile_outputs"
mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
NAME="$OUT_DIR/g11_${TS}"

# Default: short eval-only profile, exercises eval loop's hot path.
PRETRAINING_ARGS=""
N_STEPS=200
ONLY_EVAL_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretraining)
            PRETRAINING_ARGS="--developmental-pretraining --pretraining-n-goals 1 --pretraining-steps-per-goal 200"
            shift ;;
        --n-steps)
            N_STEPS="$2"; shift 2 ;;
        --eval-only)
            ONLY_EVAL_FLAG="1"; shift ;;
        *)
            echo "unknown arg: $1"; exit 1 ;;
    esac
done

# Verify nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys not found on PATH. Install NVIDIA Nsight Systems"
    echo "  (typically bundled with CUDA Toolkit or as a standalone download)."
    exit 1
fi

# Verify no other GPU work is happening
if [ -n "$(curl -s http://localhost:8765/api/runs/launch 2>/dev/null | python -c 'import json,sys; d=json.load(sys.stdin); print("running" if any(r["running"] for r in d["runs"]) else "")' 2>/dev/null)" ]; then
    echo "WARNING: webapp shows in-flight runs. Profiling under concurrent GPU work pollutes results."
    echo "Continue anyway? (y/N)"
    read -n 1 confirm
    echo
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
fi

echo "=== Profiling g11_bg_runner with nsys ==="
echo "Output: ${NAME}.nsys-rep + .summary.txt"
echo "Config: n_steps=${N_STEPS}, pretraining=${PRETRAINING_ARGS:-none}"
echo

# --trace=cuda,cudnn,osrt — capture CUDA kernel launches + memory ops + OS runtime
# --capture-range=cudaProfilerApi — only profile between cuda.profile_start/stop calls
#   (allows skipping warmup)
# --gpu-metrics-device=0 — sample GPU metrics (SM occupancy, mem throughput)
# --sampler=cpu — sample CPU stack traces too
nsys profile \
    --trace=cuda,cudnn,osrt \
    --output="${NAME}" \
    --force-overwrite=true \
    --stats=false \
    python -m research.runners.g11_bg_runner --moving-goal \
        --hippocampus --learned-perception --pfc \
        --beacon-perception --beacon-replaces-goal \
        --cue-reflex --cue-reflex-replaces-heuristic \
        --landmarks --landmarks-replace-place \
        --sensed-reward \
        --bg-lateral-inhibition \
        --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
        --curriculum --curriculum-warmup-steps 100 \
        ${PRETRAINING_ARGS} \
        --seed 42 --n-steps "${N_STEPS}" \
        --out "${NAME}.json"

echo
echo "=== Generating textual summary ==="
nsys stats --report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format=table "${NAME}.nsys-rep" \
    > "${NAME}.summary.txt" 2>&1 || echo "(nsys stats failed — open .nsys-rep in UI for details)"

echo
echo "=== Done ==="
echo "Artifacts:"
echo "  ${NAME}.nsys-rep      — open in Nsight Systems UI for full timeline"
echo "  ${NAME}.summary.txt   — kernel time + memcpy summary"
echo
echo "Quick wins to look for in the summary:"
echo "  - Top 5 kernels by cumulative time. If ANY kernel >40% of total, that's the bottleneck."
echo "  - cudaMemcpy DtoH count + total bytes. High count + small bytes = many .get() calls."
echo "  - cudaMemcpy DtoH total time vs kernel total time. If DtoH > 20%, sync is the bottleneck."
echo "  - Look for 'fused_*' kernels in the top 5 — these are our hot path."
echo "  - Any unexpected kernel (e.g. cupy memset, thrust sort) suggests an inefficient code path."
