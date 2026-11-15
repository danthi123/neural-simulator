#!/bin/bash
# report-only seeds 200-205 for brain/research/2026-11-15-near-miss-told-filter-1.5b.md (M4b): bash <this> <seed> ...
# slept half: even seed -> even, odd seed -> odd. FT only (off = the banked CPU nmoff), settings as the AWS packet (CPU).
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outFT; mkdir -p $O
for s in "$@"; do
  p=$([ $((s % 2)) -eq 0 ] && echo even || echo odd)
  A="--model 1.5b --seed $s --slept $p --steps 480 --lr 5e-4 --kl-weight 3 --arms c,s,r --self-q"
  $PY -m brain.experiments.mixed_day_probe $A --home ~/brain-lane/workFT_s$s --near-miss 4 --near-miss-frame FT --out $O/nmft_s$s.json > $O/nmft_s$s.log 2>&1
  echo "ft s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
