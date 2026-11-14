#!/bin/bash
# Pool42 lane for brain/research/2026-11-13-gain-refit-1.5b.md. Code + model shipped to ~/brain-lane (rsync of brain/,
# the rwkv package, the 1.5B checkpoint). Runs dev then test serially, one process (15 GB RAM node).
cd ~/brain-lane/src || exit 1
mkdir -p ~/brain-lane/out01
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/derisk-pool/sim/.venv/bin/python
G="0 0.5 1 1.5 2 2.5 3 4 6 8"
$PY -m brain.experiments.hippo_gain_refit --model 0.1b --bank dev --gains $G --out ~/brain-lane/out01/dev_all.json > ~/brain-lane/out01/dev.log 2>&1
$PY -m brain.experiments.hippo_gain_refit --model 0.1b --bank test --gains $G --out ~/brain-lane/out01/test_all.json > ~/brain-lane/out01/test.log 2>&1
echo DONE > ~/brain-lane/out01/DONE
