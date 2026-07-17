#!/usr/bin/env bash
# reboot_prepare.sh — call when the owner gives short notice before a reboot.
# Drops PAUSE sentinels on resumable develop-loop runs, lists running python jobs, and prints the
# resume state so nothing is lost. Safe to run repeatedly. Does NOT kill anything (PAUSE stops the
# develop loop cleanly at the next day boundary; research runners checkpoint per-seed on their own).
#
#   bash tools/reboot_prepare.sh
#
# After reboot, the owner says "continue"; Claude reads GAP_CLOSURE_MISSION.md, removes the PAUSE
# sentinels, re-arms monitors, and resumes from EXACT NEXT ACTION.
set -u
cd "$(dirname "$0")/.." || exit 1

echo "=== resumable develop-loop runs (dropping PAUSE sentinels) ==="
found=0
for d in bridges/developed/*/; do
  # a develop-loop run root has a lineage/ + day bundles; PAUSE stops it at the next day boundary
  if [ -d "${d}lineage" ] || ls "${d}"day_* >/dev/null 2>&1; then
    touch "${d}PAUSE" && echo "  PAUSE -> ${d}PAUSE"
    found=1
  fi
done
[ "$found" = 0 ] && echo "  (none found)"

echo
echo "=== running python jobs (verify they checkpoint per-seed/day before reboot) ==="
pgrep -af "python.*research" 2>/dev/null | grep -v pgrep | sed 's/^/  /' || echo "  (none)"

echo
echo "=== GPU state ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'

echo
echo "=== resume pointer ==="
echo "  On reboot: 'continue' -> read GAP_CLOSURE_MISSION.md (CURRENT STATE) -> rm the PAUSE sentinels -> re-arm monitors."
echo "  Committing the board + state is the owner/Claude's job before power-off; run tools/push_both.sh."
