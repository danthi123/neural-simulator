#!/bin/bash
# report-only seeds for brain/research/2026-11-15-weekly-1.5b-milestone.md (addendum): bash <this> <seed> ...  (CPU)
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outWK; mkdir -p $O
for s in "$@"; do
  $PY -m brain.eval.weekly --model 1.5b --shared-sleep --seed $s --homes ~/brain-lane/workWK_s$s --out $O/wk15_s$s.md > $O/wk15_s$s.log 2>&1
  echo "wk s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
