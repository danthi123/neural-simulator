#!/bin/bash
# Runs ON pool40/41/42 (CPU fp32), one seed per node. bash run_node.sh <seed> [extra memory_ceiling run args, e.g. --nights 5-5 --arms OFF,A]
# Prereg brain/research/2026-11-27-memory-ceiling-prereg.md. Needs stage0.json copied next to this script's out dir (see ship below).
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr; S=$1; shift
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export POOL_NODE=1 BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
O=$C/mc_s$S
mkdir -p $O
echo "start $(hostname) $(date -u +%FT%TZ) s$S $(cat $R/COMMIT) $*" >> $C/mc_done_s$S.txt
bash $R/tools/memcap.sh ${MEMCAP_GB:-20} -- $PY -u -m brain.experiments.memory_ceiling run --life $C/lifeKR_s$S.json.json \
  --snaps $C/snaps_s$S --trunk $C/trunk_s$S --stage0 $R/brain/experiments/results/2026-11-27-memory-ceiling/stage0.json \
  --out-dir $O --work $O/work "$@" >> $C/mc_s$S.log 2>&1
echo "s$S rc $? $(date -u +%FT%TZ) $(hostname)" >> $C/mc_done_s$S.txt
