#!/bin/bash
# Part B of brain/research/2026-11-17-teacher-night-context-1.5b.md: open conversation with the teacher night, pool CPU,
# against the banked default transcripts of the same seeds (results/2026-11-15-open-conv-baseline-1.5b/).
#   bash <this> <seed> <seed>   (pool40: 1100 1101, pool41: 1102 1103, pool42: 1104 1105).
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/tn; mkdir -p $O
for s in "$@"; do
  $PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_s$s --teacher-night --out $O/octn_s$s.md > $O/octn_s$s.log 2>&1
  echo "octn s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
