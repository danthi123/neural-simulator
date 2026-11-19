#!/bin/bash
# Multi-week life, Stage 0 (design 2026-11-19 section 8): 7 lived days on the DEV split, 1.5B on the pool (CPU).
# bash <this> <seed>   (one life per node: seed 310 on pool40, 311 on pool41, 312 on pool42). Safe to rerun: --resume.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outLIFE; mkdir -p $O
S=$1
$PY -m brain.eval.life --model 1.5b --device cpu --seed $S --days 7 --split dev --full-probe-nights 1,4,7 --nosleep-arm --homes ~/brain-lane/workLIFE_s$S --out ~/brain-lane/outLIFE/life_s$S --resume >> $O/life_s$S.log 2>&1
echo "life s$S rc $?" >> $O/done.txt
