#!/usr/bin/env bash
# pool_opsweep_keepalive.sh — idempotent self-heal for the detached consolidation opsweep.
# Relaunches the opsweep ONLY when EVERY node is reachable AND fully idle — so a transient SSH hiccup
# (which reads as "0 procs") can NEVER trigger a spurious relaunch that oversubscribes the nodes. Resume-safe:
# the dispatcher skips cells whose JSON already exists, so a relaunch is a harmless no-op when done.
# A flock makes concurrent keepalives impossible. Driven by pool-opsweep.timer (systemd --user; linger on).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/research/queue/.opsweep_keepalive.lock"
exec 9>"$LOCK"; flock -n 9 || { echo "[opsweep-keepalive] another instance holds the lock; skipping"; exit 0; }

NODES=(pool40 pool41 pool42)
all_reachable=1; any_busy=0; reachable=0
for h in "${NODES[@]}"; do
  out=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "pgrep -fc '[r]un_cell.sh'" 2>/dev/null); rc=$?
  if [ "$rc" -ne 0 ]; then all_reachable=0; continue; fi   # SSH failed => UNKNOWN, never treat as idle
  reachable=$((reachable+1))
  [ "${out:-0}" -gt 0 ] && any_busy=1
done

if [ "$all_reachable" -eq 1 ] && [ "$reachable" -eq "${#NODES[@]}" ] && [ "$any_busy" -eq 0 ]; then
  echo "[opsweep-keepalive] $(date '+%F %T') ALL ${#NODES[@]} nodes reachable + idle -> relaunching (resume-safe)"
  bash "$ROOT/tools/pool_opsweep_dispatch.sh"
else
  echo "[opsweep-keepalive] $(date '+%F %T') reachable=$reachable/${#NODES[@]} any_busy=$any_busy all_reachable=$all_reachable -> NO relaunch"
fi
