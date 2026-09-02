#!/bin/bash
# Guarded cupy 6-seed verify for the R3-v2/R3-v3 read-isolation fix (2026-09-02, C2 bug class).
# Skips (does not run the cupy verify) if the runner no longer carries the fix marker
# _EXTRA_RESET_ARRAYS -- protects against running the queued job against stale/reverted/unmerged code.
# R3v3Pool subclasses R3v2Pool and inherits _hard_reset()/_EXTRA_RESET_ARRAYS byte-identically, so ONE
# guard (checked against the r3v2 file, on main) covers both runs below.
set -e
cd /home/dant123/Projects/sim
if ! grep -q "_EXTRA_RESET_ARRAYS" research/runners/_onebrain_integration_r3v2_noncorrupting_dopamine_credit.py; then
    echo "SKIP: guard failed (read-isolation fix marker _EXTRA_RESET_ARRAYS not found in "
    echo "research/runners/_onebrain_integration_r3v2_noncorrupting_dopamine_credit.py on main) -- "
    echo "refusing to run against stale/reverted/unmerged code. Merge "
    echo "research/readfix-onebrain_integration_r3v2_noncorrupting_dopamine_credit to main first."
    exit 0
fi
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._onebrain_integration_r3v2_noncorrupting_dopamine_credit \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_onebrain_integration_r3v2_noncorrupting_readfix_cupy6seed.json
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._onebrain_integration_r3v3_functional_drive \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_onebrain_integration_r3v3_functional_drive_readfix_cupy6seed.json
