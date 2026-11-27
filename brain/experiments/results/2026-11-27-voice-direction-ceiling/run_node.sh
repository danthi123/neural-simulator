#!/bin/bash
# Runs ON pool40/41/42 (CPU fp32), one seed per node. bash run_node.sh <seed> <tag> [extra voice_ceiling_oracle run args]
# Prereg brain/research/2026-11-27-voice-direction-ceiling-prereg.md. Forward only: expected RSS ~8-10 GB, fits a 15 GB node (one process per node).
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr; S=$1; T=$2; shift 2
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export POOL_NODE=1 BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
# H must reproduce the banked A0snap hippocampus-off rows: the source-tag run's (same code path), then voice_split's (report only)
REF=""; for f in $C/st_full_s$S.json $C/vs_s$S.json; do [ -f "$f" ] && REF="$REF $f"; done
echo "start $(hostname) $(date -u +%FT%TZ) s$S $T $(cat $R/COMMIT) ref:$REF $*" >> $C/vc_done_${T}_s$S.txt
bash $R/tools/memcap.sh ${MEMCAP_GB:-14} -- $PY -u -m brain.experiments.voice_ceiling_oracle run --life $C/lifeKR_s$S.json.json \
  --snaps $C/snaps_s$S --trunk $C/trunk_s$S --out $C/vc_${T}_s$S.json --work $C/vcwork_${T}_s$S --ref $REF "$@" > $C/vc_${T}_s$S.log 2>&1
echo "s$S rc $? $(date -u +%FT%TZ) $(hostname)" >> $C/vc_done_${T}_s$S.txt
