#!/bin/bash
# Guarded cupy 6-seed verify for the _spiking_expectation_rpe_derisk read-isolation fix (2026-09-02,
# research/readfix-spiking_expectation_rpe_derisk, commit 1f60ef2c9).
# Skips (does not run the cupy verify) unless the fix marker _EXTRA_RESET_ARRAYS is present in the
# runner ON MAIN -- protects against the daemon dispatching this well after the branch was pushed but
# before it merged (or if it never merges / gets reverted). Runs BOTH configs in one job per the
# audit's re-verify plan: gain=0.4 is the decisive check (numpy: 3/6 -> 4/6, predicted ~5/6, did NOT
# fully confirm -- see research/findings/2026-09-02-spiking-expectation-rpe-read-isolation-fix-PARTIAL-demotion-still-boundary.md);
# gain=0.8 hardens the shipped 6/6 GO (seed 44's thin margin).
set -e
cd /home/dant123/Projects/sim
if ! git show main:research/runners/_spiking_expectation_rpe_derisk.py 2>/dev/null | grep -q "_EXTRA_RESET_ARRAYS"; then
    echo "SKIP: guard failed (read-isolation fix marker _EXTRA_RESET_ARRAYS not found in research/runners/_spiking_expectation_rpe_derisk.py on main) -- refusing to run against stale/unmerged code. Merge research/readfix-spiking_expectation_rpe_derisk (commit 1f60ef2c9) to main first."
    exit 0
fi
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._spiking_expectation_rpe_derisk \
    --seeds 42,43,44,100,101,102 --cue-to-expected-weight 0.4 \
    --out research/findings/raw/_read_isolation_audit/spiking_expectation_rpe_gain0.4_cupy_readfix.json
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._spiking_expectation_rpe_derisk \
    --seeds 42,43,44,100,101,102 --cue-to-expected-weight 0.8 \
    --out research/findings/raw/_read_isolation_audit/spiking_expectation_rpe_gain0.8_cupy_readfix.json
