#!/bin/bash
# Runs ON pool40 (CPU fp32). bash run_node.sh <seed> <tag> [extra voice_tag run args]. Prereg brain/research/2026-11-26-source-tag-prereg-DRAFT.md (R1).
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr; S=$1; T=$2; shift 2
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
echo "start $(hostname) $(date -u +%FT%TZ) s$S $T $(cat $R/COMMIT) $*" >> $C/st_done_${T}_s$S.txt
bash $R/tools/memcap.sh ${MEMCAP_GB:-20} -- $PY -u -m brain.experiments.voice_tag run --life $C/lifeKR_s$S.json.json \
  --snaps $C/snaps_s$S --trunk $C/trunk_s$S --out $C/st_${T}_s$S.json --work $C/stwork_${T}_s$S "$@" > $C/st_${T}_s$S.log 2>&1
echo "s$S rc $? $(date -u +%FT%TZ) $(hostname)" >> $C/st_done_${T}_s$S.txt
