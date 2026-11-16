#!/bin/bash
# Report-only follow-up to the pool replication (brain/research/2026-11-16-rehearsal-night-1.5b.md addendum 2):
# HR lost day-2 recall on pool seeds 1112-1114 (13 -> 7 of 18). Split it: H alone and R alone on the same seeds.
#   bash <this> <seed>   (pool40: 1112, pool41: 1113, pool42: 1114). Installed as ~/brain-lane/hab2.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/hab; mkdir -p $O; : > $O/done2.txt
s=$1
$PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_h_s$s --habituation 3 --out $O/och_s$s.md > $O/och_s$s.log 2>&1
echo "och s$s rc $?" >> $O/done2.txt
$PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_r_s$s --sleep-rehearse --out $O/ocr_s$s.md > $O/ocr_s$s.log 2>&1
echo "ocr s$s rc $?" >> $O/done2.txt
echo DONE >> $O/done2.txt
