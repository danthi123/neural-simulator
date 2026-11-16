#!/bin/bash
# Report-only: reply-only rehearsal (Brain(sleep_rehearse="reply"), prereg addendum 3 of
# brain/research/2026-11-16-rehearsal-night-1.5b.md) on new pool seeds, default and reply-only both (report-only replication of addendum 4).
#   bash <this> <seed>   (pool40: 1121, pool41: 1122, pool42: 1123). Installed as ~/brain-lane/hab4.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/hab; mkdir -p $O; : > $O/done4.txt
s=$1
$PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_d_s$s --out $O/ocd_s$s.md > $O/ocd_s$s.log 2>&1
echo "ocd s$s rc $?" >> $O/done4.txt
$PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_y_s$s --rehearse-reply --out $O/ocy_s$s.md > $O/ocy_s$s.log 2>&1
echo "ocy s$s rc $?" >> $O/done4.txt
echo DONE >> $O/done4.txt
