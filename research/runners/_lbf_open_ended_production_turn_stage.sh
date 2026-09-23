#!/usr/bin/env bash
# Stage the remaining 5 seeds (43,44,100,101,102) of the open-ended production-turn probe for one MODE, detached
# (setsid) under a kernel memory cap (tools/memcap.sh), logging to the mode's logs/ dir, then aggregating the mode's
# verdicts (seed 42 was run in-session into the SAME out-dir, so the aggregate covers all 6 seeds).
#   bash research/runners/_lbf_open_ended_production_turn_stage.sh <mode> [parallel]
# Pre-registration: docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md (+ amendment 1).
set -euo pipefail
# SUPERSEDED (amendment 3, eefdd666a): runs full brains locally under the withdrawn statistic. Use _stage_a3_pool.sh.
[ "${OEP_A2_LOCAL_ALLOWED:-0}" = 1 ] || { echo "SUPERSEDED: use research/runners/_lbf_open_ended_production_turn_stage_a3_pool.sh" >&2; exit 2; }
cd "$(dirname "$0")/../.."
MODE=${1:?mode}
PAR=${2:-2}
OUT=research/findings/raw/_load_bearing/_oe_production_turn
mkdir -p "$OUT/logs"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
setsid -f bash tools/memcap.sh 8 -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m research.runners._lbf_open_ended_production_turn_probe \
  --mode "$MODE" --seeds 43,44,100,101,102 --parallel "$PAR" --k 40 --out-dir "$OUT/$MODE" \
  --aggregate-out "$OUT/${MODE}_6seed_aggregate.json" > "$OUT/logs/${MODE}_s43-102.log" 2>&1 < /dev/null
echo "staged mode=$MODE -> $OUT/$MODE (log $OUT/logs/${MODE}_s43-102.log, aggregate $OUT/${MODE}_6seed_aggregate.json)"
