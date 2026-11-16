#!/bin/bash
# Self-key probe (brain/research/2026-11-15-self-key-question-1.5b.md): this node's slept off-arm intact homes.
#   bash <this> <seed> ...   (pool40: 200 201 306 307, pool41: 202 203 308 309, pool42: 204 205 310 311). Installed as ~/brain-lane/skp.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/skp; mkdir -p $O; rm -f $O/done.txt
H=(); for s in "$@"; do H+=(~/brain-lane/workSC_off_s$s/_weekly_intact); done
$PY -m brain.experiments.self_key_probe --out $O/skp_$(hostname).json "${H[@]}" > $O/skp.log 2>&1
echo "skp rc $?" >> $O/done.txt; echo DONE >> $O/done.txt
