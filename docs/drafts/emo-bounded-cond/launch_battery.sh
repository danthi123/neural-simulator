#!/usr/bin/env bash
# launch_battery.sh <flag 0|1> <prefix> : 6 workers over held-out seeds 301-360 + 1 worker over dev seeds 1-8
cd /home/dant123/Projects/sim-rev/emobc_scratch
flag=$1; pre=$2
PY=/home/dant123/Projects/sim/.venv/bin/python
for w in 0 1 2 3 4 5; do
  lo=$((301+w*10)); seeds=$(seq $lo $((lo+9)) | tr '\n' ' ')
  nohup bash /home/dant123/Projects/sim/tools/memcap.sh 2 -- $PY battery.py $flag ${pre}_w$w $seeds > battery/${pre}_w$w.log 2>&1 &
done
nohup bash /home/dant123/Projects/sim/tools/memcap.sh 2 -- $PY battery.py $flag ${pre}_dev 1 2 3 4 5 6 7 8 > battery/${pre}_dev.log 2>&1 &
sleep 2; echo launched $pre
