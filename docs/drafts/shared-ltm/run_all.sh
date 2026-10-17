#!/usr/bin/env bash
# Review evidence (2026-10-13): seeds 7 and 8, turnset v2, arms on / offiso / off, each under the brain lock.
#   docs/drafts/shared-ltm/run_all.sh
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
for SEED in 7 8; do
  for ARM in on offiso off; do
    flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock "$D/run_arm.sh" "$ARM" 20 "$SEED" v2
  done
done
echo "RUN_ALL DONE"
