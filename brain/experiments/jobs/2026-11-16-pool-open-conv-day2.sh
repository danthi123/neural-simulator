#!/bin/bash
# Day-2 control on the open-conv baseline's slept homes (brain/eval/open_conv_day2.py).
#   bash <this> <seed> ...  (pool40: 1100 1101, pool41: 1102 1103, pool42: 1104 1105). Installed as ~/brain-lane/oc2.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
: > ~/brain-lane/oc/done2.txt
for s in "$@"; do
  $PY -m brain.eval.open_conv_day2 --model 1.5b --seed $s --home ~/brain-lane/oc/homes_s$s/hippocampus_off --out ~/brain-lane/oc/d2_s$s.md > ~/brain-lane/oc/d2_s$s.log 2>&1
  echo "d2 s$s rc $?" >> ~/brain-lane/oc/done2.txt
done
echo DONE >> ~/brain-lane/oc/done2.txt
