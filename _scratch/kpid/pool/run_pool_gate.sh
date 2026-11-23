#!/bin/bash
# Runs ON a pool node: K-P identity gate (prereg 2026-11-23-kp-identity-prereg.md section 2, amendment A1), one seed,
# all four arms B, KP, KPU, KP0 re-run on this node's CPU (fp32), then drift + probes. Resumes: each step skips if its
# JSON exists. bash run_pool_gate.sh <seed> <expected cortex_adapters sha256>
set -u
S=$1; WANT=$2
R=~/brain-lane/kpid_repo; K=$R/_scratch/kpid; OUT=$K/pool_cpu; SRC=~/brain-lane/kpgate/src_s$S
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1; mkdir -p $OUT
GOT=$(sha256sum $SRC/cortex_adapters.pt | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo "s$S source sha mismatch $GOT != $WANT" | tee -a $OUT/done.txt; exit 1; }
echo "pool gate s$S start $(hostname) $(date -u +%FT%TZ)" >> $OUT/done.txt
for A in B KP KPU KP0; do
  H=$OUT/home_${A}_s$S; BF=""; [ $A = KPU ] && BF="--bank-from $OUT/home_KP_s$S"
  [ -f $H.result.json ] || $PY -u $K/gate.py night $A $SRC $H $S --device cpu $BF > $OUT/night_${A}_s$S.log 2>&1 \
    || { echo "night $A s$S FAILED" >> $OUT/done.txt; exit 1; }
  [ -f $OUT/drift_${A}_s$S.json ] || $PY -u $K/gate.py drift $A $SRC $H $S $OUT/drift_${A}_s$S.json --device cpu \
    > $OUT/drift_${A}_s$S.log 2>&1 || { echo "drift $A s$S FAILED" >> $OUT/done.txt; exit 1; }
  [ -f $OUT/probe_${A}_s$S.json ] || $PY -u $K/gate.py probe $A $H $S $OUT/probe_${A}_s$S.json --device cpu \
    > $OUT/probe_${A}_s$S.log 2>&1 || { echo "probe $A s$S FAILED" >> $OUT/done.txt; exit 1; }
  echo "s$S $A done $(date -u +%FT%TZ)" >> $OUT/done.txt
done
echo "pool gate s$S done $(date -u +%FT%TZ)" >> $OUT/done.txt
