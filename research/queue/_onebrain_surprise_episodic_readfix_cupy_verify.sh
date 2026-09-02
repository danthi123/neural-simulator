#!/bin/bash
# Guarded cupy 6-seed verify for the onebrain_integration_surprise_episodic_crossedge read-isolation fix
# (2026-09-02, Port A -- _EXTRA_RESET_ARRAYS, ported from the C2 metacog fix template).
# Skips (does not run the cupy verify) if the runner on MAIN no longer carries the fix marker
# _EXTRA_RESET_ARRAYS -- protects against running the queued job before the fix branch is merged, or
# against stale/reverted code.
set -e
cd /home/dant123/Projects/sim
if ! grep -q "_EXTRA_RESET_ARRAYS" research/runners/_onebrain_integration_surprise_episodic_crossedge.py; then
    echo "SKIP: guard failed (read-isolation fix marker _EXTRA_RESET_ARRAYS not found in research/runners/_onebrain_integration_surprise_episodic_crossedge.py on main) -- refusing to run against stale/pre-merge code"
    exit 0
fi
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._onebrain_integration_surprise_episodic_crossedge \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_readfix_cupy6seed.json
