#!/bin/bash
# Guarded cupy 6-seed verify for the curiosity->d6wm cross-edge read-isolation fix (2026-09-02).
# This fix corrects a LIVE default-ON production faculty's banked "GO 6/6" to a numpy-verified NO-GO 3/6
# (research/findings/2026-09-02-onebrain-crossedge-curiosity-to-d6wm-read-isolation-fix-corrects-GO-to-NOGO-3-6.md).
# Guards on MAIN (not this worktree, which is cleaned up after the build agent finishes): skips the cupy run
# unless research/runners/_onebrain_crossedge_curiosity_to_d6wm.py on the MAIN checkout carries the fix marker
# _selftest_read_isolation -- protects against running the queued job before the fix branch is merged, or
# against a later revert.
set -e
cd /home/dant123/Projects/sim
if ! grep -q "_selftest_read_isolation" research/runners/_onebrain_crossedge_curiosity_to_d6wm.py; then
    echo "SKIP: guard failed (read-isolation fix marker _selftest_read_isolation not found in research/runners/_onebrain_crossedge_curiosity_to_d6wm.py on main) -- refusing to run against stale/pre-fix/reverted code. Merge research/readfix-onebrain_crossedge_curiosity_to_d6wm to main first."
    exit 0
fi
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._onebrain_crossedge_curiosity_to_d6wm \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_readfix_cupy6seed.json
