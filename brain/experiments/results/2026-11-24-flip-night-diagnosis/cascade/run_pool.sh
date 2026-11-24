#!/bin/bash
# Runs ON pool40 (CPU fp32; the node has no GPU). bash run_pool.sh <seed> ...
R=~/brain-lane/flip_repo; C=$R/_scratch/flip/cascade
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
echo "start $(hostname) $(date -u +%FT%TZ) pid $$" >> $C/done.txt
for S in "$@"; do
  $PY -u $C/probe3.py $S $C/homes/s$S/trunk $C/lifeM1_s$S.json.json $C/out_s$S.json > $C/probe_s$S.log 2>&1
  echo "s$S rc $? $(date -u +%FT%TZ)" >> $C/done.txt
done
echo "DONE $(date -u +%FT%TZ)" >> $C/done.txt
