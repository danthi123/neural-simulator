#!/bin/bash
# Runs ON pool42 (CPU fp32): routing screen 3 part X on arm S snapshots, seeds 319 320 321 in turn.
R=~/brain-lane/route_repo; D=$R/_scratch/route/s3
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
for S in 319 320 321; do
  echo "start s$S $(date -u +%FT%TZ) $(hostname) $(cat $R/COMMIT)" >> $D/x_done.txt
  bash $R/tools/memcap.sh 11 -- $PY -u -m brain.experiments.routing_screen3x run --model 1.5b --life $D/lifeS_s$S.json \
    --snaps $D/S_s$S/nightsnap --trunk $D/S_s$S/trunk --out $D/x_s$S.json --work $D/xwork_s$S > $D/x_s$S.log 2>&1
  echo "s$S rc $? $(date -u +%FT%TZ)" >> $D/x_done.txt
done
