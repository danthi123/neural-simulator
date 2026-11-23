#!/bin/bash
# K-P-U Step-0 one-night gate (brain/research/2026-11-23-kp-identity-prereg.md section 2), serial, one device.
# Arms per seed in order B, KP, KPU (bank copied + hash-checked from KP), KP0; then drift + probes per arm; then score.
# Device: CUDA fp32 when torch sees a GPU (all arms re-run there), else W CPU fp32 (reuses the existing CPU B and K-P
# after-homes and drift JSONs of s319/s320 from _scratch/kpgate_w, re-runs s321). Every step skips if its JSON exists,
# so the script resumes after a cut. Override: DEVICE=cpu|cuda SEEDS="319 320 321" ARMS="B KP KPU KP0" THREADS=12.
# The GPU goes through tools/gpu_queue.sh + tools/memcap.sh (CLAUDE.md); launch line in the HANDOFF / caller.
set -u
W=/home/dant123/Projects/sim/_scratch/wt-mixday; K=$W/_scratch/kpid; KG=/home/dant123/Projects/sim/_scratch/kpgate_w
PY=/home/dant123/Projects/sim/.venv/bin/python
cd $W || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=${THREADS:-12}
if [ -z "${DEVICE:-}" ]; then
  DEVICE=$($PY -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null | tail -1)
fi
if [ "$DEVICE" = cuda ]; then OUT=$K/cuda_fp32; DEV="--device cuda --cuda-precision fp32"; else OUT=$K/cpu; DEV="--device cpu"; fi
SEEDS=${SEEDS:-"319 320 321"}; ARMS=${ARMS:-"B KP KPU KP0"}
mkdir -p $OUT
src_of() { case $1 in 319) echo $KG/src_s319;; 320) echo $KG/src_s320;; 321) echo $K/src_s321;; esac; }
echo "kpid gate: device $DEVICE -> $OUT, seeds $SEEDS, arms $ARMS, $(date -u +%FT%TZ)"
for S in $SEEDS; do
  SRC=$(src_of $S); [ -f $SRC/meta.json ] || { echo "missing source $SRC"; exit 1; }
  if [ "$DEVICE" = cpu ] && [ "$S" != 321 ]; then   # prereg fallback: reuse the CPU B / K-P after-homes and drift reads
    for A in B KP; do
      [ -e $OUT/home_${A}_s$S ] || ln -s $KG/home_${A}_s$S $OUT/home_${A}_s$S
      [ -e $OUT/home_${A}_s$S.result.json ] || cp $KG/home_${A}_s$S.result.json $OUT/
      [ -e $OUT/drift_${A}_s$S.json ] || cp $KG/drift_${A}_s$S.json $OUT/
    done
  fi
  for A in $ARMS; do
    H=$OUT/home_${A}_s$S
    BF=""; [ $A = KPU ] && BF="--bank-from $OUT/home_KP_s$S"
    [ -f $H.result.json ] || $PY -u $K/gate.py night $A $SRC $H $S $DEV $BF > $OUT/night_${A}_s$S.log 2>&1 \
      || { echo "night $A s$S FAILED (see $OUT/night_${A}_s$S.log)"; exit 1; }
    [ -f $OUT/drift_${A}_s$S.json ] || $PY -u $K/gate.py drift $A $SRC $H $S $OUT/drift_${A}_s$S.json $DEV \
      > $OUT/drift_${A}_s$S.log 2>&1 || { echo "drift $A s$S FAILED"; exit 1; }
    [ -f $OUT/probe_${A}_s$S.json ] || $PY -u $K/gate.py probe $A $H $S $OUT/probe_${A}_s$S.json $DEV \
      > $OUT/probe_${A}_s$S.log 2>&1 || { echo "probe $A s$S FAILED"; exit 1; }
    echo "s$S $A done $(date -u +%FT%TZ)"
  done
done
$PY $K/gate.py score $OUT | tee $OUT/gate_read.txt
echo "kpid gate done $(date -u +%FT%TZ)"
