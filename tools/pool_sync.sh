#!/bin/bash
# pool_sync.sh — PULL completed result artifacts BACK from the mini-PC pool nodes to the local repo.
#
# WHY (the gap this closes, 2026-08-20): tools/pool_autodispatch.sh dispatches jobs to the pool via
# `ssh -f -n` fire-and-forget — the job runs on the node, writes its result JSON into the node's own
# `~/derisk-pool/sim/research/findings/raw/...`, and appends rc to the node's job_status.log. NOTHING ever
# pulls those result files back. So completed 0-token compute silently STRANDS on the nodes: on 2026-08-20 a
# survey found ~146 finished result JSONs (gap4 credit sweeps, sleep-replay consolidation, episodic-completion,
# spkbind6, GNW dsub-robustness, fm_reservoir scale, stageA integration, perception) sitting unpulled — several
# of them decisive verdicts that were never banked. This is the retrieval half of the pool lane; run it on a
# cadence (the session heartbeat can call it) and after any pool batch so results land where findings + gates see them.
#
#   tools/pool_sync.sh              # pull newer/missing result JSONs from every POOL_NODES node (safe, non-destructive)
#   tools/pool_sync.sh --dry-run    # show what WOULD transfer, change nothing
#   POOL_NODES="pool40 pool41" tools/pool_sync.sh   # restrict to a subset
#
# SAFETY: rsync -au (archive + UPDATE) copies remote->local ONLY when the remote file is newer or absent locally,
# so a committed local result is never clobbered by a stale remote copy. *.log and per-node _provenance/ are
# excluded (merging three nodes' runs.jsonl would clobber); provenance sidecars (*.prov.json) ARE pulled.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
NODES="${POOL_NODES:-pool40 pool41 pool42}"
REMOTE_DIR="${POOL_REMOTE_DIR:-~/derisk-pool/sim/research/findings/raw/}"
LOCAL_DIR="research/findings/raw/"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY="--dry-run"
mkdir -p "$LOCAL_DIR"
total=0
for N in $NODES; do
  # -u protects newer local files; itemize so we can count + show what moved.
  out=$(timeout 180 rsync -au $DRY --itemize-changes \
        --exclude='*.log' --exclude='_provenance/' \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=6" \
        "$N:$REMOTE_DIR" "$LOCAL_DIR" 2>/dev/null) || { echo "  $N: UNREACHABLE (skipped)"; continue; }
  n=$(printf '%s\n' "$out" | grep -cE '^>f' || true)
  echo "  $N: ${DRY:+would pull }$n file(s)"
  printf '%s\n' "$out" | grep -E '^>f' | grep -vE '\.prov\.json$' | awk '{print "      "$2}' | head -30
  total=$((total+n))
done
echo "pool_sync: ${DRY:+(dry-run) }$total file(s) ${DRY:+would be }pulled from [$NODES]"
