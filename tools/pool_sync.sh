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
#   tools/pool_sync.sh --strict --node pool1        # STRICT: non-zero exit if pool1 is unreachable or ANY of
#                                                    # its (main or per-revision) rsync pulls fails. Same as
#                                                    # POOL_SYNC_STRICT=1 POOL_NODES=pool1 tools/pool_sync.sh.
#                                                    # Used by `aws_pool_node.sh down` to VERIFY the final pull
#                                                    # succeeded before it is safe to terminate an instance whose
#                                                    # root volume is DeleteOnTermination=true (2026-09-23 fix
#                                                    # round: the plain default below always exits 0 even when
#                                                    # every node was UNREACHABLE, which let `down` terminate an
#                                                    # unsynced node unattended -- that DEFAULT TIMER BEHAVIOUR
#                                                    # must not change, so strict is opt-in, never the default).
#
# SAFETY: rsync -au (archive + UPDATE) copies remote->local ONLY when the remote file is newer or absent locally,
# so a committed local result is never clobbered by a stale remote copy. *.log and per-node _provenance/ are
# excluded (merging three nodes' runs.jsonl would clobber); provenance sidecars (*.prov.json) ARE pulled.
#
# STALE-HOSTNAME AUTO-REFRESH (2026-09-25): a first pull failure for a node with a research/queue/.aws_<node>
# state file (i.e. one tools/aws_pool_node.sh manages) triggers one cheap `aws_pool_node.sh refresh <node>` --
# rewrites .pool_ssh_config's Host block if the node is actually running under a NEW ip (there is no Elastic
# IP anywhere in this feature) -- then retries the pull once. pool40/41/42 (no such state file) are unaffected.
set -uo pipefail   # NOT -e: a per-node/per-revision rsync failure must be handled explicitly (below), not abort
                    # the whole script -- strict mode needs to see EVERY node's outcome to report all of them.
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
# BUGFIX (fix round): under `set -e`, this whole line's exit status is the exit status of the LAST command in
# the `&&` chain -- and `grep -vE ... | tr ...` exits 1 (pipefail) whenever .pool_extra_nodes exists but is
# EMPTY or holds only comments (grep -v selects nothing). `aws_pool_node.sh down` leaves exactly that file, so
# every pool_sync after an up/down cycle died silently with rc=1 and zero ssh/rsync calls. `|| true` makes
# "nothing extra to add" a normal outcome, matching pool_autodispatch.sh's extra_nodes() (which has no `set -e`
# to trip on this same pattern).
# --strict / --node <n> / --dry-run (2026-09-23 fix round, ANY order, ANY combination). POOL_SYNC_STRICT=1 is the
# env-var equivalent of --strict, for callers (like a `down` that already sets other POOL_* env vars) that would
# rather not touch argv. Unknown args are ignored (this script has never validated argv, and a stray positional
# from an old caller must not start failing now).
STRICT="${POOL_SYNC_STRICT:-0}"; DRY=""; _NODE_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY="--dry-run" ;;
    --strict) STRICT=1 ;;
    --node) shift; _NODE_ARG="${1:-}" ;;
  esac
  shift || true
done
[ -n "$_NODE_ARG" ] && POOL_NODES="$_NODE_ARG"
[ -z "${POOL_NODES:-}" ] && [ -f "$EXTRA_NODES_FILE" ] && \
  _EXTRA=$(grep -vE '^[[:space:]]*(#|$)' "$EXTRA_NODES_FILE" 2>/dev/null | tr -s '[:space:]' ' ') || true
NODES="${POOL_NODES:-pool40 pool41 pool42} $_EXTRA"
REMOTE_DIR="${POOL_REMOTE_DIR:-~/derisk-pool/sim/research/findings/raw/}"
LOCAL_DIR="research/findings/raw/"
mkdir -p "$LOCAL_DIR"
# STALE-HOSTNAME AUTO-REFRESH (2026-09-25, incident-driven -- pool1's public ip changes on every AWS stop/start,
# no Elastic IP anywhere in this feature, so .pool_ssh_config's Host block for it can go stale between cadences;
# the owner had to fix it by hand). AWS_STATE_DIR default matches tools/aws_pool_node.sh's own state-file
# convention (research/queue/.aws_<node-name>) -- override exists so tests never touch the shared production dir.
AWS_STATE_DIR="${POOL_SYNC_AWS_STATE_DIR:-$ROOT/research/queue}"
_rsync_pull() {   # _rsync_pull <node> -- one itemized pull; the ONE place this shape is spelled out, so the
                   # main call and the post-refresh retry below can never drift apart.
  timeout 180 rsync -au $DRY --itemize-changes \
      --exclude='*.log' --exclude='_provenance/' \
      -e "$RSYNC_SSH" \
      "$1:$REMOTE_DIR" "$LOCAL_DIR" 2>/dev/null
}
total=0
FAILED=0
for N in $NODES; do
  # -u protects newer local files; itemize so we can count + show what moved.
  out=$(_rsync_pull "$N") || {
    # Only for nodes tools/aws_pool_node.sh itself manages (a research/queue/.aws_<N> state file exists) --
    # pool40/41/42 (mini-PCs, no such file) are completely unaffected, byte-identical to before this change.
    # `refresh` is cheap and bounded to this ONE retry: it never starts a stopped instance, so a genuinely
    # stopped/gone node still reports UNREACHABLE exactly as before, just after one extra (fast, local-file-only
    # unless the ip actually changed) round trip.
    _aws_state="$AWS_STATE_DIR/.aws_$N"
    _refreshed=0
    # --dry-run MUST CHANGE NOTHING (2026-09-25 review, LOW): `refresh` rewrites .pool_ssh_config (+ its .bak)
    # on disk -- a real, persistent change -- which the old code ran even under --dry-run, breaking that
    # promise. Skip the refresh attempt entirely when $DRY is set; a dry-run node that would have been
    # refreshed just reports UNREACHABLE like any other unreachable node, same as before this feature existed.
    if [ -z "$DRY" ] && [ -f "$_aws_state" ]; then
      # Only retry if the Host block ACTUALLY changed -- `refresh` exits 0 both when it rewrote something and
      # when it correctly declined (node not running, ip already current, etc.), so a before/after content
      # compare is the one signal that distinguishes "worth a retry" from "nothing to gain by retrying."
      _cfg_before=$(cat "$POOL_SSH_CONFIG" 2>/dev/null || true)
      if AWS_POOL_NODE_STATE_FILE="$_aws_state" POOL_SSH_CONFIG="$POOL_SSH_CONFIG" \
           bash "$ROOT/tools/aws_pool_node.sh" refresh "$N" >>"${POOL_SYNC_REFRESH_LOG:-/dev/null}" 2>&1; then
        _cfg_after=$(cat "$POOL_SSH_CONFIG" 2>/dev/null || true)
        [ "$_cfg_before" != "$_cfg_after" ] && _refreshed=1
      fi
    fi
    if [ "$_refreshed" = 1 ]; then
      out=$(_rsync_pull "$N") || { echo "  $N: UNREACHABLE (skipped, even after a Host-block refresh)"; FAILED=1; continue; }
    else
      echo "  $N: UNREACHABLE (skipped)"; FAILED=1; continue
    fi
  }
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
  # BUGFIX (fix round): RSYNC_SSH already STARTS WITH "ssh ..." (it is the whole `-e` argument, e.g.
  # "ssh -F <config> -o BatchMode=yes ..."), so prefixing it with another literal `ssh` ran
  # `ssh ssh -o ... <node> ...` -- real ssh fails with "Could not resolve hostname ssh" (rc=255), and the
  # trailing `|| true` swallowed that, so results under derisk-pool/revisions/*/research/findings/raw were
  # NEVER pulled from any node (reproduced with a stubbed ssh; confirmed real-ssh rc=255).
  # `|| true` is INSIDE the remote command (not wrapping the ssh call): a glob that matches nothing (no isolated
  # revision ever provisioned on this node, the common case) makes the remote `ls` exit non-zero on ITS OWN,
  # which must read as "zero revisions", not a strict-mode failure. The ssh call's own exit status -- reachability
  # -- is what the `||` on the assignment below reacts to.
  revs=$(timeout 20 $RSYNC_SSH "$N" 'ls -d derisk-pool/revisions/*/research/findings/raw 2>/dev/null || true' 2>/dev/null) \
    || { echo "  $N: revision-list probe failed (unreachable)"; FAILED=1; continue; }
  for R in $revs; do
    rout=$(timeout 180 rsync -au $DRY --itemize-changes \
          --exclude='*.log' --exclude='_provenance/' \
          -e "$RSYNC_SSH" \
          "$N:$R/" "$LOCAL_DIR" 2>/dev/null) || { echo "  $N:${R#derisk-pool/revisions/}: rsync FAILED"; FAILED=1; continue; }
    rn=$(printf '%s\n' "$rout" | grep -cE '^>f' || true)
    [ "$rn" -gt 0 ] && echo "  $N:${R#derisk-pool/revisions/}: ${DRY:+would pull }$rn file(s)"
    total=$((total+rn))
  done
done
echo "pool_sync: ${DRY:+(dry-run) }$total file(s) ${DRY:+would be }pulled from [$NODES]"
if [ "$STRICT" = 1 ] && [ "$FAILED" = 1 ]; then
  echo "pool_sync: ⛔ STRICT: at least one node was unreachable or a pull failed -- see above (exit 1)." >&2
  exit 1
fi
exit 0
