#!/usr/bin/env bash
# pool_opsweep_keepalive.sh — idempotent self-heal for the detached consolidation opsweep.
# Re-launches the opsweep ONLY when it has gone fully idle on all nodes (e.g. after a node reboot, or a
# transient), so it never oversubscribes a node that's already running. Resume-safe: the dispatcher skips
# cells whose JSON already exists, so a relaunch is a harmless no-op when everything is done or still running.
# Driven by the pool-opsweep.timer (systemd --user; survives reboot because linger is on).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODES=(pool40 pool41 pool42)
running=0
for h in "${NODES[@]}"; do
  c=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "pgrep -fc '[r]un_cell.sh'" 2>/dev/null || echo 0)
  [ "${c:-0}" -gt 0 ] && running=$((running+1))
done
if [ "$running" -eq 0 ]; then
  echo "[opsweep-keepalive] $(date '+%F %T') opsweep idle on all reachable nodes -> relaunching (resume-safe)"
  bash "$ROOT/tools/pool_opsweep_dispatch.sh"
else
  echo "[opsweep-keepalive] $(date '+%F %T') opsweep running on $running/${#NODES[@]} node(s) -> no action"
fi
