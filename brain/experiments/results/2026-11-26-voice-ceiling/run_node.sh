#!/bin/bash
# Runs ON a pool node (CPU fp32). bash run_node.sh <seed> <tag> [extra voice_ceil run args]. Prereg brain/research/2026-11-26-voice-ceiling-prereg.md.
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr; S=$1; T=$2; shift 2
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
echo "start $(hostname) $(date -u +%FT%TZ) s$S $T $(cat $R/COMMIT) $*" >> $C/vc_done_${T}_s$S.txt
bash $R/tools/memcap.sh 20 -- $PY -u -m brain.experiments.voice_ceil run --life $C/lifeKR_s$S.json.json \
  --snaps $C/snaps_s$S --trunk $C/trunk_s$S --out $C/vc_${T}_s$S.json --work $C/cwork_${T}_s$S "$@" > $C/vc_${T}_s$S.log 2>&1
echo "s$S rc $? $(date -u +%FT%TZ) $(hostname)" >> $C/vc_done_${T}_s$S.txt
