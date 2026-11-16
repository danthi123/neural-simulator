#!/bin/bash
# Report-only: reply-only rehearsal (Brain(sleep_rehearse="reply"), prereg addendum 3 of
# brain/research/2026-11-16-rehearsal-night-1.5b.md) on the pool seeds whose default arm already ran.
#   bash <this> <seed>   (pool40: 1112, pool41: 1113, pool42: 1114). Installed as ~/brain-lane/hab3.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/hab; mkdir -p $O; : > $O/done3.txt
s=$1
$PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_y_s$s --rehearse-reply --out $O/ocy_s$s.md > $O/ocy_s$s.log 2>&1
echo "ocy s$s rc $?" >> $O/done3.txt
echo DONE >> $O/done3.txt
