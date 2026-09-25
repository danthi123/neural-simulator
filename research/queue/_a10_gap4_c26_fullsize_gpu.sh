#!/bin/bash
# A10 gap#4 C26 full-size transfer check (successor to AMENDMENT 5 F's C21 full-size run, research/queue/_a9_gap4_tc_gpu.sh,
# which collapsed every hidden-learning arm to chance -- research/findings/2026-09-24-gap4-transport-ceiling-bound-census-
# clamp-load-bearing-fullsize-UNDEFINED.md). Same shape as that run (4 arm processes in parallel, replicates 0 1 2 looped
# per arm, per-(seed, replicate, arm) checkpoints, then one aggregate), PLUS C26's ratio-baseline flags
# (--pbar-ratio-tau-ms 5000, hidden layers only) on top of the C21 operating point.
#
# Pre-registration: research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md, AMENDMENT 7.
# SCOPE (AMENDMENT 7): this run answers ONLY whether C26 transfers to full size (H64/pool 16), under the SAME rules
# (census rule, Rule B) AMENDMENT 5 F used for C21. It draws NO verdict on the dev-scale C25-C27 AMENDMENT 6 gate (a
# separate, still-queued pool battery) and it is NOT an EVALUATION CONFIG section -- it unlocks NO evaluation seed.
# EVALUATION seeds (42/43/44/100/101/102) stay refused by the runner unless A10_AMENDMENT names a COMMITTED amendment
# whose own '## AMENDMENT <n> ... EVALUATION CONFIG' section registers this exact config fingerprint and the seeds;
# AMENDMENT 7 is not such a section, so the default here stays DEV seed 7.
#
# Usage (from gpu_queue): bash research/queue/_a10_gap4_c26_fullsize_gpu.sh
#   env: A10_SEED (default 7)  A10_EPOCHS (default 40)  A10_AMENDMENT (path; needed only for evaluation seeds)
#        A10_ARMS (default "frozen fixed_fa micro_inengine transport_ceiling")
#        A10_WT   checkout to run from (default: the repo this script lives in -- its own pinned detached worktree,
#                  never a hardcoded temporary worktree)
#        A10_PIN_SHA  refuse to run unless that checkout's HEAD is this commit (set at queue time, per AMENDMENT 7's
#                     own commit, the same convention AMENDMENT 5 F used for A9_PIN_SHA)
#        A10_PY   interpreter (default: the main checkout's .venv, found through git's common dir, so a worktree works)
set -u
SELF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WT=${A10_WT:-$SELF_ROOT}
PIN=${A10_PIN_SHA:-}
cd "$WT" || { echo "REFUSED: checkout $WT does not exist"; exit 2; }
if [ -n "$PIN" ] && [ "$(git rev-parse HEAD)" != "$(git rev-parse "$PIN^{commit}" 2>/dev/null)" ]; then
  echo "SKIP: $WT HEAD $(git rev-parse --short HEAD) is not the pinned $PIN -- refusing to run moved code"; exit 0
fi
MAIN_ROOT=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
PY=${A10_PY:-$MAIN_ROOT/.venv/bin/python}
[ -x "$PY" ] || { echo "REFUSED: no interpreter at $PY (set A10_PY)"; exit 2; }
SEED=${A10_SEED:-7}
EPOCHS=${A10_EPOCHS:-40}
ARMS=${A10_ARMS:-"frozen fixed_fa micro_inengine transport_ceiling"}
D=research/findings/raw/gap4/transport_ceiling_readout/c26_fullsize_gpu
mkdir -p "$D"
OUT=$D/gpu_c26_s${SEED}_e${EPOCHS}.json
# the C21 operating point (AMENDMENT 5 F full-size shape) plus C26's ratio-baseline lever (AMENDMENT 6 / AMENDMENT 7)
CFG="--seeds $SEED --replicates 0 1 2 --hidden 64 --pool-k 16 --train-subsample 400 --epochs $EPOCHS \
 --read-quantity spikes --settle-steps 40 --read-window 30 --read-gain 20 --isi-steps 0 --eval-frozen \
 --spi-silence-outside-credit --no-structural-plasticity --no-ff-stp --ff-w-init 40 --propagation-strength 0.5 \
 --bdsp-w-max 12 --tonic-h-pA 225 --tonic-o-pA 250 --pbar-alpha 0 --lr 5 --hidden-lr-gain 0.2 \
 --pbar-ratio-tau-ms 5000 --pbar-ratio-layers hidden"
AM=""
[ -n "${A10_AMENDMENT:-}" ] && AM="--prereg-amendment $A10_AMENDMENT"
export SIM_BACKEND=cupy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for a in $ARMS; do
  $PY -u -m research.runners._gap4_transport_ceiling_readout_derisk $CFG $AM --arms "$a" --out "$OUT" \
    > "$D/gpu_c26_s${SEED}_e${EPOCHS}_${a}.log" 2>&1 &
done
wait
$PY -u -m research.runners._gap4_transport_ceiling_readout_derisk $CFG $AM --aggregate-only --out "$OUT"
