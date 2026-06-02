#!/bin/bash
cd /e/Documents/Projects/sim
R="research/findings/raw"
rm -f "$R/_flatdist320_codes.npz"   # force fresh 320-code capture from all 5 distinct-seed bridges
echo "===== full-320 flat-distinct chain: DE retrain + 320 structured composition test ====="
echo "[$(date +%H:%M:%S)] retraining spatial@45 + functional@46 ..."
bash "$R/_run_flatdist_DE.sh"
echo "[$(date +%H:%M:%S)] running 320-wide structured SVO composition test ..."
python -m research.findings.raw._insubstrate_flatdistinct320_test 2>&1 \
  | grep -iE "capturing|captured|between-concept|STRUCTURED|seed [0-9]+:|RESULT:|VERDICT:|CANNOT-CONCLUDE"
echo "##### FLATDIST-320-ALL-DONE [$(date +%H:%M:%S)] #####"
