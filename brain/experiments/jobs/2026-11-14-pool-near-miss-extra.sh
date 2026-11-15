#!/bin/bash
# report-only extra seeds (200-205) for brain/research/2026-11-14-near-miss-guard-1.5b.md: bash <this> <seed> ...
# slept half: even seed -> even, odd seed -> odd. Off and on per seed, same settings as the AWS packet (CPU).
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outNM; mkdir -p $O
for s in "$@"; do
  p=$([ $((s % 2)) -eq 0 ] && echo even || echo odd)
  A="--model 1.5b --seed $s --slept $p --steps 480 --lr 5e-4 --kl-weight 3 --arms c,s,r --self-q"
  $PY -m brain.experiments.mixed_day_probe $A --home ~/brain-lane/workNM_off_s$s --out $O/nmoff_s$s.json > $O/nmoff_s$s.log 2>&1
  echo "off s$s rc $?" >> $O/done.txt
  $PY -m brain.experiments.mixed_day_probe $A --home ~/brain-lane/workNM_on_s$s --near-miss 4 --near-miss-frame F --out $O/nmon_s$s.json > $O/nmon_s$s.log 2>&1
  echo "on s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
