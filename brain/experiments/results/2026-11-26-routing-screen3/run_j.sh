#!/bin/bash
# Runs ON a pool node (CPU fp32): routing screen 3 part J (J-space rank ceiling) on KR snapshots. bash run_j.sh <seed...>
R=~/brain-lane/route_repo; K=$R/_scratch/route/kr; D=$R/_scratch/route/s3
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
for S in "$@"; do
  echo "start s$S $(date -u +%FT%TZ) $(hostname) $(cat $R/COMMIT)" >> $D/j_done_s$S.txt
  bash $R/tools/memcap.sh 11 -- $PY -u -m brain.experiments.jspace_ceiling run --model 1.5b --life $K/lifeKR_s$S.json.json \
    --ceil $D/ceil_s$S.json --snaps $K/snaps_s$S --trunk $K/trunk_s$S --J $D/J.pt --out $D/j_s$S.json --work $D/jwork_s$S \
    > $D/j_s$S.log 2>&1
  echo "s$S rc $? $(date -u +%FT%TZ)" >> $D/j_done_s$S.txt
done
