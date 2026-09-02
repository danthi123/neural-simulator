#!/bin/bash
# Guarded cupy 6-seed verify for the H-2 fswta_drive read-isolation fix (2026-09-02).
# numpy already found a REAL verdict flip for A5 (PARTIAL/NEGATIVE -> GO, best_deeper 0.899667 -> 0.955333 at the
# runner's own default --fs-inh/--fs-settle) -- this is NOT pure hardening, so (unlike H-3/the clean runners) it
# gets the standard cupy re-verify. S3 is included for parity/completeness even though numpy already showed it
# byte-identical (already at the metric ceiling both arms).
# Skips (does not run the cupy verify) if the runner no longer carries the fix marker -- protects against running
# the queued job against stale/reverted code.
set -e
cd /home/dant123/Projects/sim/.claude/worktrees/wf_cafa4862-d16-5
if ! grep -q "_EXTRA_RESET_ARRAYS" research/runners/_d3_spiking_attractor_derisk.py; then
    echo "SKIP: guard failed (read-isolation fix marker _EXTRA_RESET_ARRAYS not found in the runner) -- refusing to run against stale/reverted code"
    exit 0
fi
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._d3_spiking_attractor_derisk \
    --group S3 --seeds 42,43,44 \
    --json research/findings/raw/_read_isolation_audit/h2_fswta_drive_s3_cupy3seed.json
SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._d3_spiking_attractor_derisk \
    --group A5 --seeds 100,101,102 \
    --json research/findings/raw/_read_isolation_audit/h2_fswta_drive_a5_cupy3seed.json
