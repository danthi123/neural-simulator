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
# AWS-AS-EXTRA-POOL-NODE (2026-09-23) -- same repo-local, gitignored ssh config as pool_autodispatch.sh /
# pool_provision.sh / pool_queue.sh (see pool_autodispatch.sh's header comment for the full rationale). ABSENT
# by default, so every rsync below is unchanged for anyone who hasn't run `aws_pool_node.sh up`. The default
# node list also grows with .pool_extra_nodes, read fresh on every invocation (this script is not a daemon).
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=6"
[ -f "$POOL_SSH_CONFIG" ] && RSYNC_SSH="ssh -F $POOL_SSH_CONFIG -o BatchMode=yes -o ConnectTimeout=6"
EXTRA_NODES_FILE="${POOL_EXTRA_NODES_FILE:-$ROOT/research/queue/.pool_extra_nodes}"
_EXTRA=""
[ -z "${POOL_NODES:-}" ] && [ -f "$EXTRA_NODES_FILE" ] && \
  _EXTRA=$(grep -vE '^[[:space:]]*(#|$)' "$EXTRA_NODES_FILE" 2>/dev/null | tr -s '[:space:]' ' ')
NODES="${POOL_NODES:-pool40 pool41 pool42} $_EXTRA"
REMOTE_DIR="${POOL_REMOTE_DIR:-~/derisk-pool/sim/research/findings/raw/}"
LOCAL_DIR="research/findings/raw/"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY="--dry-run"
mkdir -p "$LOCAL_DIR"
total=0
for N in $NODES; do
  # -u protects newer local files; itemize so we can count + show what moved.
  out=$(timeout 180 rsync -au $DRY --itemize-changes \
        --exclude='*.log' --exclude='_provenance/' \
        -e "$RSYNC_SSH" \
        "$N:$REMOTE_DIR" "$LOCAL_DIR" 2>/dev/null) || { echo "  $N: UNREACHABLE (skipped)"; continue; }
  n=$(printf '%s\n' "$out" | grep -cE '^>f' || true)
  echo "  $N: ${DRY:+would pull }$n file(s)"
  # BUGFIX (2026-09-03): under `set -eo pipefail`, this display-only pipeline dies with exit 1 whenever
  # a node has ZERO new files -- grep '^>f' finds no match, exits 1, and pipefail propagates that through
  # awk/head to kill the WHOLE SCRIPT before later nodes are even reached. Caught wiring this into a
  # systemd timer for unattended runs: the very first automated invocation "failed" after pool41 (0 new
  # files) and never reached pool42. `|| true` makes "nothing new to show" a normal outcome, not a crash.
  printf '%s\n' "$out" | grep -E '^>f' | grep -vE '\.prov\.json$' | awk '{print "      "$2}' | head -30 || true
  total=$((total+n))
  # ISOLATED REVISIONS (2026-09-23). Branch jobs provisioned with `pool_provision.sh --isolated` write to
  # ~/derisk-pool/revisions/<sha>/research/findings/raw/, which the pull above never reached -- every lane's
  # branch verification would have stranded on the nodes. Pull each revision's raw/ into the same local tree
  # (same -u newer-wins + exclusions; paths under raw/ are already lane-namespaced by the runners).
  [ -n "${POOL_REMOTE_DIR:-}" ] && continue
  revs=$(timeout 20 ssh $RSYNC_SSH "$N" 'ls -d derisk-pool/revisions/*/research/findings/raw 2>/dev/null' || true)
  for R in $revs; do
    rout=$(timeout 180 rsync -au $DRY --itemize-changes \
          --exclude='*.log' --exclude='_provenance/' \
          -e "$RSYNC_SSH" \
          "$N:$R/" "$LOCAL_DIR" 2>/dev/null) || continue
    rn=$(printf '%s\n' "$rout" | grep -cE '^>f' || true)
    [ "$rn" -gt 0 ] && echo "  $N:${R#derisk-pool/revisions/}: ${DRY:+would pull }$rn file(s)"
    total=$((total+rn))
  done
done
echo "pool_sync: ${DRY:+(dry-run) }$total file(s) ${DRY:+would be }pulled from [$NODES]"
