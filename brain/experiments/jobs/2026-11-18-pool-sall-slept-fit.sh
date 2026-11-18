#!/bin/bash
# Stage 1 of brain/research/2026-11-15-sall-slept-bar-1.5b.md on pool CPU: one seed per node.
# usage: bash 2026-11-18-pool-sall-slept-fit.sh <seed>
S=$1
cd ~/brain-lane || exit 1
mkdir -p sfit
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
$PY -m brain.experiments.sall_live_bar_fit score --size-sleep --model 1.5b --device cpu --seed "$S" \
  --home sfit/home_s$S --out sfit/score_s$S.json > sfit/score_s$S.log 2>&1
echo "score s$S rc $?" >> sfit/done.txt
echo DONE >> sfit/done.txt
