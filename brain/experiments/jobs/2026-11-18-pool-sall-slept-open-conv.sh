#!/bin/bash
# brain/research/2026-11-18-sall-open-conv-1.5b.md Amendment 1: default (ocd) vs S_all + slept bar (ocs), open conversation, pool CPU.
#   bash <this> <seed> <seed>   (pool40: 1132 1133, pool41: 1134 1135, pool42: 1136 1137).
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/sslept; mkdir -p $O
for s in "$@"; do
  $PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_d_s$s --out $O/ocd_s$s.md > $O/ocd_s$s.log 2>&1
  echo "ocd s$s rc $?" >> $O/done.txt
  $PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_s_s$s --key S_all --key-bar slept --out $O/ocs_s$s.md > $O/ocs_s$s.log 2>&1
  echo "ocs s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
