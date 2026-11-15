#!/bin/bash
# smoke of weekly --sleep-lesion on weights (report-only, CPU): bash <this> <seed> ...
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outSL; mkdir -p $O
for s in "$@"; do
  $PY -m brain.eval.weekly --model 1.5b --shared-sleep --sleep-lesion --seed $s --homes ~/brain-lane/workSL_s$s --out $O/sl_s$s.md > $O/sl_s$s.log 2>&1
  echo "sl s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
