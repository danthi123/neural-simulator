#!/bin/bash
# A9 gap#4 transport-ceiling GPU run (plan step G5 shape): one gpu_queue job, 4 arm processes in parallel (one per
# arm, each looping replicates 0 1 2 with per-(seed, replicate, arm) checkpoints), then one aggregate. Re-queued after
# `gpu_queue.sh pause --now`, it resumes: completed shards are skipped by the runner (config fingerprint must match).
#
# Pre-registration: research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md.
# EVALUATION seeds (42/43/44/100/101/102) are refused by the runner unless A9_AMENDMENT names a COMMITTED amendment
# whose '## AMENDMENT <n> ... EVALUATION CONFIG' section registers this exact config fingerprint and the seeds
# (prereg AMENDMENT 5; get the fingerprint with --print-fingerprint). As of 2026-09-24 no dev config qualified, so the
# default here is DEV seed 7 at the full size (the transfer + budget check named in the finding's next-lever list).
#
# Usage (from gpu_queue): bash research/queue/_a9_gap4_tc_gpu.sh
#   env: A9_SEED (default 7)  A9_EPOCHS (default 40)  A9_AMENDMENT (path; needed only for evaluation seeds)
#        A9_ARMS (default "frozen fixed_fa micro_inengine transport_ceiling")
#        A9_WT   checkout to run from (default: the repo this script lives in -- never a hardcoded temporary worktree)
#        A9_PIN_SHA  refuse to run unless that checkout's HEAD is this commit
#        A9_PY   interpreter (default: the main checkout's .venv, found through git's common dir, so a worktree works)
set -u
SELF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WT=${A9_WT:-$SELF_ROOT}
PIN=${A9_PIN_SHA:-}
cd "$WT" || { echo "REFUSED: checkout $WT does not exist"; exit 2; }
if [ -n "$PIN" ] && [ "$(git rev-parse HEAD)" != "$(git rev-parse "$PIN^{commit}" 2>/dev/null)" ]; then
  echo "SKIP: $WT HEAD $(git rev-parse --short HEAD) is not the pinned $PIN -- refusing to run moved code"; exit 0
fi
MAIN_ROOT=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${A9_PY:-$MAIN_ROOT/.venv/bin/python}
[ -x "$PY" ] || { echo "REFUSED: no interpreter at $PY (set A9_PY)"; exit 2; }
SEED=${A9_SEED:-7}
EPOCHS=${A9_EPOCHS:-40}
ARMS=${A9_ARMS:-"frozen fixed_fa micro_inengine transport_ceiling"}
D=research/findings/raw/gap4/transport_ceiling_readout/gpu
mkdir -p "$D"
OUT=$D/gpu_s${SEED}_e${EPOCHS}.json
# the best dev operating point (C21, prereg AMENDMENT 4) at the 2026-09-15 net size
CFG="--seeds $SEED --replicates 0 1 2 --hidden 64 --pool-k 16 --train-subsample 400 --epochs $EPOCHS \
 --read-quantity spikes --settle-steps 40 --read-window 30 --read-gain 20 --isi-steps 0 --eval-frozen \
 --spi-silence-outside-credit --no-structural-plasticity --no-ff-stp --ff-w-init 40 --propagation-strength 0.5 \
 --bdsp-w-max 12 --tonic-h-pA 225 --tonic-o-pA 250 --pbar-alpha 0 --lr 5 --hidden-lr-gain 0.2"
AM=""
[ -n "${A9_AMENDMENT:-}" ] && AM="--prereg-amendment $A9_AMENDMENT"
export SIM_BACKEND=cupy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for a in $ARMS; do
  $PY -u -m research.runners._gap4_transport_ceiling_readout_derisk $CFG $AM --arms "$a" --out "$OUT" \
    > "$D/gpu_s${SEED}_e${EPOCHS}_${a}.log" 2>&1 &
done
wait
$PY -u -m research.runners._gap4_transport_ceiling_readout_derisk $CFG $AM --aggregate-only --out "$OUT"
