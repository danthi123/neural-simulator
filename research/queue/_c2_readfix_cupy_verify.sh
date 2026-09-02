#!/bin/bash
# Guarded cupy 6-seed verify for the C2 metacog read-isolation fix (2026-09-02).
# Skips (does not run the cupy verify) if the runner no longer carries the fix marker
# _EXTRA_RESET_ARRAYS -- protects against running the queued job against stale/reverted code.
set -e
cd /home/dant123/Projects/sim/.claude/worktrees/agent-aa24eab2283351266
if ! grep -q "_EXTRA_RESET_ARRAYS" research/runners/_crossedge_surprise_metacog_derisk.py; then
    echo "SKIP: guard failed (read-isolation fix marker _EXTRA_RESET_ARRAYS not found in the runner) -- refusing to run against stale/reverted code"
    exit 0
fi
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._crossedge_surprise_metacog_derisk \
    --ablation plain --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_crossedge_surprise_metacog_readfix_cupy6seed.json
