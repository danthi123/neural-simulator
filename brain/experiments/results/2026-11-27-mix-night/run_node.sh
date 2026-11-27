#!/bin/bash
# Runs ON a pool node (CPU fp32). bash run_node.sh <seed> [arm ...]. Mix-night screen: DRAFT, do not launch before the pre-data
# review of brain/research/2026-11-27-sleep-drift-recipe-check.md section 4 and the registered off-frame set.
# Needs OFFFRAME (JSON list of 12 passages) and OFFFRAME_SHA256 (its registered sha256) in the environment.
# One python process per (arm, seed), sequential; units are written atomically, so a rerun resumes (finished units are skipped).
# Step 0 of the note: run `bash run_node.sh <seed> A0r` first (one night) and time it before queueing the rest.
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr; S=$1; shift
[ -n "$S" ] || { echo "usage: run_node.sh <seed> [arm ...]"; exit 2; }
[ -n "$OFFFRAME" ] && [ -n "$OFFFRAME_SHA256" ] || { echo "OFFFRAME and OFFFRAME_SHA256 must be set"; exit 2; }
ARMS="${*:-A0r A0L A0 A0e A0k A1 A2 A3 A4 A5s A5f}"
case $S in 319) F=320;; 320) F=321;; 321) F=319;; *) echo "seed must be 319/320/321"; exit 2;; esac
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=$(nproc) POOL_NODE=1
O=$C/mix_night; mkdir -p $O; cd $R || exit 1
for A in $ARMS; do
  echo "start $(hostname) $(date -u +%FT%TZ) s$S $A" >> $O/done_s$S.txt
  bash $R/tools/memcap.sh 11 -- $PY -u -m brain.experiments.mix_night_screen run --seed $S --arm $A \
    --life $C/lifeKR_s$S.json.json --snaps $C/snaps_s$S --foreign-snaps $C/snaps_s$F \
    --offframe "$OFFFRAME" --offframe-sha256 "$OFFFRAME_SHA256" --model 1.5b --device cpu \
    --out-dir $O --work $O/work_${A}_s$S >> $O/${A}_s$S.log 2>&1
  echo "s$S $A rc $? $(date -u +%FT%TZ)" >> $O/done_s$S.txt
done
