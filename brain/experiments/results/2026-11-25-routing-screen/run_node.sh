#!/bin/bash
# Runs ON a pool node (CPU fp32). bash run_node.sh <seed>. Routing screen replay (prereg 2026-11-24-routing-screen-prereg.md).
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr; S=$1
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
echo "start $(hostname) $(date -u +%FT%TZ) s$S" >> $C/done_s$S.txt
bash $R/tools/memcap.sh 11 -- $PY -u -m brain.experiments.routing_replay replay --life $C/lifeKR_s$S.json.json \
  --snaps $C/snaps_s$S --model 1.5b --device cpu --out $C/replay_s$S.json --work $C/work_s$S > $C/replay_s$S.log 2>&1
echo "s$S rc $? $(date -u +%FT%TZ)" >> $C/done_s$S.txt
