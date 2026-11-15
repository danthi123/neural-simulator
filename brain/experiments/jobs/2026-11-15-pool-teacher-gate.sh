#!/bin/bash
# Teacher-night gate (brain/research/2026-11-15-teacher-night-gate-1.5b.md), queued behind the self-recall probe:
# waits for ~/brain-lane/srp/done.txt DONE.  bash <this> <seed> ...   Installed as ~/brain-lane/tg.sh.
until grep -q DONE ~/brain-lane/srp/done.txt 2>/dev/null; do sleep 120; done
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/tg; mkdir -p $O; rm -f $O/done.txt
for s in "$@"; do
  $PY -m brain.experiments.teacher_gate_probe --model 1.5b --seed $s --k 4 --out $O/tg_s$s.json > $O/tg_s$s.log 2>&1
  echo "tg s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
