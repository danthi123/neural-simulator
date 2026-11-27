#!/bin/bash
# Runs ON pool40/41/42 (CPU fp32), one seed per node: bash run_node.sh <seed> [extra sims args, e.g. --nights 5-5]
# DRAFT prereg brain/research/2026-11-27-reachable-strength-prereg-DRAFT.md. GATED: do not ship before the memory reply
# ceiling verdict is HEADROOM and the prereg is REGISTERED after its pre-data review.
# On the node: sims, then the TRIPWIRE-ONLY screen (NULL and A; no method or control arm is computed or written).
# After all three seeds, HEADROOM and registration (on any host; the full screen refuses without HEADROOM):
#   for S in 319 320 321; do python -m brain.experiments.strength_screen screen --dir $C/rs --seed $S --ceiling-verdict V; done
#   python -m brain.experiments.strength_screen score --dir $C/rs --seeds 319 320 321 \
#   --ceiling-dir $C/mc_s319 $C/mc_s320 $C/mc_s321 --ceiling-verdict <memory ceiling verdict.json> --out $C/rs/verdict.json
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr; S=$1; shift
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export POOL_NODE=1 BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc)
cd $R || exit 1
O=$C/rs
mkdir -p $O
echo "start $(hostname) $(date -u +%FT%TZ) s$S $(cat $R/COMMIT) $*" >> $C/rs_done_s$S.txt
bash $R/tools/memcap.sh ${MEMCAP_GB:-20} -- $PY -u -m brain.experiments.strength_screen sims --life $C/lifeKR_s$S.json.json \
  --snaps $C/snaps_s$S --out-dir $O --model ~/brain-lane/models/rwkv7-g1l-1.5b-20261031-ctx34304.pth "$@" >> $C/rs_s$S.log 2>&1 \
  && $PY -u -m brain.experiments.strength_screen screen --tripwire-only --dir $O --seed $S >> $C/rs_s$S.log 2>&1
echo "s$S rc $? $(date -u +%FT%TZ) $(hostname)" >> $C/rs_done_s$S.txt
