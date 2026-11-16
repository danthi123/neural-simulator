#!/bin/bash
# Report-only pool arms for brain/research/2026-11-16-rehearsal-night-1.5b.md (addendum 2): default vs HR on one seed.
#   bash <this> <seed>   (pool40: 1112, pool41: 1113, pool42: 1114). Installed as ~/brain-lane/hab.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/hab; mkdir -p $O; : > $O/done.txt
s=$1
$PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_d_s$s --out $O/ocd_s$s.md > $O/ocd_s$s.log 2>&1
echo "ocd s$s rc $?" >> $O/done.txt
$PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_hr_s$s --habituation 3 --sleep-rehearse --out $O/ochr_s$s.md > $O/ochr_s$s.log 2>&1
echo "ochr s$s rc $?" >> $O/done.txt
echo DONE >> $O/done.txt
