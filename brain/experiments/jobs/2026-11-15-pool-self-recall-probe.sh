#!/bin/bash
# Self-recall-after-night probe (brain/research/2026-11-15-self-recall-after-night-1.5b.md), queued behind the running
# self-completion report-only seeds: waits for ~/brain-lane/outSC/done.txt DONE, then probes this node's slept homes.
#   bash <this> <seed> ...      (pool40: 200 201, pool41: 202 203, pool42: 204 205). Installed as ~/brain-lane/srp.sh.
until grep -q DONE ~/brain-lane/outSC/done.txt 2>/dev/null; do sleep 120; done
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/srp; mkdir -p $O; rm -f $O/done.txt
H=(); for s in "$@"; do H+=(~/brain-lane/workSC_off_s$s/_weekly_intact ~/brain-lane/workSC_on_s$s/_weekly_intact); done
$PY -m brain.experiments.self_recall_probe --out $O/srp_$(hostname).json "${H[@]}" > $O/srp.log 2>&1
echo "srp rc $?" >> $O/done.txt; echo DONE >> $O/done.txt
