#!/bin/bash
# brain/research/2026-11-18-sall-open-conv-1.5b.md: default (ocd) vs S_all + size bar (ock), open conversation, pool CPU.
#   bash <this> <seed> <seed>   (pool40: 1126 1127, pool41: 1128 1129, pool42: 1130 1131).
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/salloc; mkdir -p $O
for s in "$@"; do
  $PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_d_s$s --out $O/ocd_s$s.md > $O/ocd_s$s.log 2>&1
  echo "ocd s$s rc $?" >> $O/done.txt; rm -rf $O/homes_d_s$s
  $PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_k_s$s --key S_all --key-bar size --out $O/ock_s$s.md > $O/ock_s$s.log 2>&1
  echo "ock s$s rc $?" >> $O/done.txt; rm -rf $O/homes_k_s$s
done
echo DONE >> $O/done.txt
