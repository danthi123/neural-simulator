#!/bin/bash
# Runs ON a pool node (CPU fp32). bash run_node.sh <seed>. Voice split (prereg brain/research/2026-11-26-voice-split-prereg.md).
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr; S=$1
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
echo "start $(hostname) $(date -u +%FT%TZ) s$S $(cat $R/COMMIT)" >> $C/vs_done_s$S.txt
bash $R/tools/memcap.sh 11 -- $PY -u -m brain.experiments.voice_split run --life $C/lifeKR_s$S.json.json \
  --snaps $C/snaps_s$S --trunk $C/trunk_s$S --out $C/vs_s$S.json --work $C/vwork_s$S > $C/vs_s$S.log 2>&1
echo "s$S rc $? $(date -u +%FT%TZ) $(hostname)" >> $C/vs_done_s$S.txt
