#!/bin/bash
# CYCLE-1098 — the DEFINITIVE emergent sparse-pool A->W read-out validation, gated on the FF-WTA + kernel mechanism
# (config E = random + FF-WTA + kernel + synaptic-scaling = 0.688 at 16 words -- the emergent surpass on RANDOM/
# unseparated codes, past the 0.50 ceiling). Two questions:
#  (1) MULTI-SEED: does the winning config hold across dev seeds 42/43/44 (then blind 100/101/102)?
#  (2) SCALING: does it hold at 64 words on ONE bridge (n_shared_pool=8000) -- the single-bridge win (64 words vs
#      the 16-word grandmother cap)?  RANDOM codes (no hand-separation needed -- the read-out mechanism is the fix).
# Runs one GPU process at a time (sequential) -- do NOT parallelize on the GPU (compute-thrash lesson, CYCLE-1098).
# Pass extra flags as $1 (e.g. "--readout-teacher-pA 2000" if the sweep found teacher helps).
set -e
cd "$(dirname "$0")/../../.."
EXTRA=${1:-}
LOG=research/findings/raw/_aw_readout_scaling.log
: > "$LOG"
echo "=== MULTI-SEED (16 words, random + FF-WTA + kernel + synaptic-scaling $EXTRA) ===" | tee -a "$LOG"
for s in 42 43 44; do
  SIM_BACKEND=cupy python -u -m research.runners._sparse_aw_speak_derisk --seeds $s --n-concepts 16 \
    --n-train-events 100 --ff-langout-wta --winner-inactive-ld 0.03 --synaptic-scaling $EXTRA 2>&1 \
    | grep -E "speak_acc=|AGGREGATE" | tee -a "$LOG"
done
echo "=== SCALING (64 words, n_pool=8000, random + FF-WTA + kernel + synaptic-scaling, seed 42) ===" | tee -a "$LOG"
SIM_BACKEND=cupy python -u -m research.runners._sparse_aw_speak_derisk --seeds 42 --n-concepts 64 \
  --n-shared-pool 8000 --n-train-events 100 --ff-langout-wta --winner-inactive-ld 0.03 --synaptic-scaling $EXTRA 2>&1 \
  | grep -E "code overlap|speak_acc=|AGGREGATE" | tee -a "$LOG"
echo "SCALING_DONE" | tee -a "$LOG"
