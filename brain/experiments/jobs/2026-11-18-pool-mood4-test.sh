#!/bin/bash
# brain/research/2026-11-03-mood-state-design.md sections 15-16: 1.5B six-seed test. bash <this> <seed> <seed>
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/mood4; mkdir -p $O; T=$(( $(nproc) - 1 ))
for s in "$@"; do
  $PY -m brain.experiments.mood_probe --model 1.5b --seed $s --threads $T --gain 16 --questions mixed_test --samples 2 \
    --max-tokens 40 --arms pos,neg,off,rand_pos,rand_neg --n-rand 2 --no-embedding \
    --out $O/mood4_1.5b_s$s.json > $O/mood4_s$s.log 2>&1
  echo "test s$s rc $?" >> $O/done.txt
done
echo DONE4 >> $O/done.txt
