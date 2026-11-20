#!/bin/bash
# Dev lives for M-use / M-scale (prereg brain/research/2026-11-19-life-muse-mscale-prereg.md section 6).
# bash <this> <seed> <arm>   arm: B (baseline, --nosleep-arm) | U (--use-gate) | S (--sleep-downscale 1.0) | US (both)
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outLIFE; mkdir -p $O
S=$1; A=$2
case $A in
  B) X="--nosleep-arm" ;;
  U) X="--use-gate" ;;
  S) X="--sleep-downscale 1.0" ;;
  US) X="--use-gate --sleep-downscale 1.0" ;;
  *) echo "bad arm $A"; exit 2 ;;
esac
$PY -m brain.eval.life --model 1.5b --device cpu --seed $S --days 7 --split dev --full-probe-nights 1,4,7 --frozen-probe $X --homes ~/brain-lane/workLIFE_s${S}_$A --out $O/life_s${S}_$A --resume >> $O/life_s${S}_$A.log 2>&1
echo "life s${S}_$A rc $?" >> $O/done.txt
