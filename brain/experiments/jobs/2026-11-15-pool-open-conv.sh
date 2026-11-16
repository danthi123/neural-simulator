#!/bin/bash
# Open-conversation eval baseline (brain/research/2026-11-15-open-conversation-eval.md), queued behind the own-cue job:
# waits for ~/brain-lane/ocf/done.txt DONE.  bash <this> <seed> ...  (pool40: 1100 1101, pool41: 1102 1103,
# pool42: 1104 1105). Installed as ~/brain-lane/oc.sh.
until grep -q DONE ~/brain-lane/ocf/done.txt 2>/dev/null; do sleep 120; done
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/oc; mkdir -p $O; rm -f $O/done.txt
for s in "$@"; do
  $PY -m brain.eval.open_conv --model 1.5b --seed $s --homes ~/brain-lane/oc/homes_s$s --out $O/oc_s$s.md > $O/oc_s$s.log 2>&1
  echo "oc s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
