#!/usr/bin/env bash
# aws_stop_safety_lib.sh — shared "make it safe to stop this node" helpers (2026-09-25 review, fix round 3),
# used by BOTH tools/aws_idle_stop.sh's opportunistic idle-stop AND tools/aws_budget.sh's hard-cap `enforce`.
# Extracted from aws_idle_stop.sh (where sync_node_before_stop used to live as a private function) so the two
# callers can never drift on what "synced before stop" or "taken out of dispatch for the stop" means — the
# review's own still-open item was precisely that `aws_budget.sh`'s cap enforcement never got this treatment at
# all.
#
# SOURCE it to get the three functions below (the caller must already have ROOT, LOG and POOL_SSH_CFG set —
# every existing caller does, at the top of its own script). RUN it directly for a CLI test seam that needs no
# real AWS/ssh call to reach the guards inside these functions (2026-09-25 review, LOW: "sync_node_before_stop's
# own key/ip guard is unreachable through the main script -- have_ssh=1 already implies both are set"; this seam
# makes it directly testable, and it is now genuinely reachable in production too, from aws_budget.sh, whose own
# node/ip/key resolution can legitimately come up empty):
#   bash tools/aws_stop_safety_lib.sh --sync <node> <ip> <key>
#   bash tools/aws_stop_safety_lib.sh --pause <node>                 # prints 1 if it removed a registration, else 0
#   bash tools/aws_stop_safety_lib.sh --resume <node> <was_registered>

sync_node_before_stop() {   # sync_node_before_stop <node-name> <ip> <key> -- pull this node's results BEFORE
  # it is stopped (2026-09-25, incident-driven: pool1 was idle-stopped at 13:06Z with two finished DA LTM-on
  # seeds' last arm+seed JSON written after the last routine pool_sync -- stranded on the stopped node's disk
  # until the owner restarted it by hand ~55 min later). Returns 0 iff the pull is VERIFIED to have succeeded;
  # the caller must not stop the instance on a non-zero return (the root volume is not what's at risk here --
  # unlike aws_pool_node.sh down's DeleteOnTermination=true case -- but a `stop`+cold node still leaves any
  # result written after this point unreachable until someone notices and restarts it, exactly what happened).
  #
  # Prefers tools/pool_sync.sh restricted to this one node (POOL_SYNC_STRICT=1) when the node has a registered
  # dispatch alias in .pool_ssh_config -- identical exclusions/isolated-revision handling to the routine 15-min
  # cadence sync, so this call can never diverge from what "synced" already means elsewhere in this repo. Falls
  # back to the SAME plain rsync pool_sync.sh performs when this node is not a registered pool-dispatch alias
  # (e.g. the single-instance `.aws_gpu` CPU-verify lane, never wired into .pool_ssh_config) -- using the ip/key
  # this script already resolved for the no-runner SSH check, so a non-pool AWS lane is covered too, not just
  # aws_pool_node.sh-managed nodes.
  local node="$1" ip="$2" key="$3"
  if [ -z "$ip" ] || [ -z "$key" ] || [ ! -f "$key" ]; then
    echo "$(date -u '+%FT%TZ') [aws-stop-safety] $node: no verified ssh key/ip on hand -- cannot sync, NOT stopping this cycle" | tee -a "$LOG"
    return 1
  fi
  if [ -n "$node" ] && [ -f "$POOL_SSH_CFG" ] && grep -q "^Host $node\$" "$POOL_SSH_CFG" 2>/dev/null; then
    # </dev/null (2026-09-25 review, HIGH #2): pool_sync.sh's own ssh calls must never inherit the CALLER's
    # stdin -- see aws_idle_stop.sh's own fd-3 fix for the concrete production incident this class of bug
    # caused elsewhere in this same script family. This call sits inside a per-instance loop body in every
    # caller, so give it its own belt-and-suspenders guard even though each caller's own fd-3 fix already keeps
    # its id list off fd 0.
    if POOL_SSH_CONFIG="$POOL_SSH_CFG" POOL_NODES="$node" POOL_SYNC_STRICT=1 \
         bash "$ROOT/tools/pool_sync.sh" </dev/null >>"$LOG" 2>&1; then
      return 0
    fi
    echo "$(date -u '+%FT%TZ') [aws-stop-safety] $node: pool_sync --strict FAILED -- NOT stopping this cycle (will retry)" | tee -a "$LOG"
    return 1
  fi
  # CANDIDATE REMOTE DIRS (2026-09-25 review, HIGH #1): a node with no .pool_ssh_config alias is NOT necessarily
  # a tools/aws_pool_node.sh-managed pool node -- the single-instance `.aws_gpu`/.aws_cpuN lanes (tools/
  # aws_cpu_launch.sh + tools/aws_cpu_provision.sh / tools/aws_provision.sh) rsync code to ~/sim, never
  # ~/derisk-pool/sim (see those scripts' own `~/sim/` rsync targets). The OLD single-path fallback always tried
  # ~/derisk-pool/sim: on those lanes the source directory does not exist, rsync exits 23, sync_node_before_stop
  # logged "NOT stopping this cycle" on EVERY cycle forever (only the $50/day cap ever stopped them), and their
  # results under ~/sim were never pulled. FIX: `ssh test -d` each known layout's PROJECT ROOT (not the deeper
  # raw/ results dir -- a genuinely fresh node with zero results yet would otherwise read as a sync FAILURE) and
  # pull from whichever roots actually exist on THIS node -- a node provisioned any other way is still coverable
  # by extending this list, with no change to the decision logic below.
  local remote_dirs rd root status found=0
  if [ -n "${POOL_REMOTE_DIR:-}" ]; then
    remote_dirs="$POOL_REMOTE_DIR"
  else
    remote_dirs="derisk-pool/sim/research/findings/raw/ sim/research/findings/raw/"
  fi
  mkdir -p "$ROOT/research/findings/raw"
  for rd in $remote_dirs; do
    root="${rd%/research/findings/raw/}"
    # ONE round trip per candidate: does the project root exist, and (only if so) does its raw/ results dir
    # exist yet. `-n` (2026-09-25 review, HIGH #2): every remote call here must not read the caller's stdin.
    status=$(ssh -n -i "$key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
        ubuntu@"$ip" "if [ -d ~/$root ]; then if [ -d ~/$rd ]; then echo has_raw; else echo root_only; fi; else echo no_root; fi" 2>/dev/null)
    case "$status" in
      no_root|"") continue ;;   # this layout is not this node's -- or the probe itself failed; try the next candidate
      root_only) found=1; continue ;;   # this IS the node's layout, but nothing has been written yet -- nothing to pull
      has_raw)
        found=1
        if ! timeout 180 rsync -au --exclude='*.log' --exclude='_provenance/' \
            -e "ssh -i $key -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes" \
            "ubuntu@$ip:~/$rd" "$ROOT/research/findings/raw/" >>"$LOG" 2>&1; then
          echo "$(date -u '+%FT%TZ') [aws-stop-safety] $node: fallback rsync of ~/$rd FAILED -- NOT stopping this cycle (will retry)" | tee -a "$LOG"
          return 1
        fi
        ;;
    esac
  done
  if [ "$found" = 0 ]; then
    echo "$(date -u '+%FT%TZ') [aws-stop-safety] $node: no known project layout found on this node (checked: $remote_dirs) -- cannot verify sync, NOT stopping this cycle" | tee -a "$LOG"
    return 1
  fi
  return 0
}

# EXTRA_NODES_FILE override exists for tests (POOL_EXTRA_NODES_FILE, matching tools/aws_pool_node.sh's own
# override name and default) -- must never point at the shared production research/queue/.pool_extra_nodes in a
# test. Only set here when the sourcing script has not already defined ROOT (a bare `source` from a script that
# defines ROOT first, as every real caller does, just reuses that value; the CLI mode below sets ROOT itself).
EXTRA_NODES_FILE="${POOL_EXTRA_NODES_FILE:-${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/research/queue/.pool_extra_nodes}"

pause_dispatch_for_node() {   # pause_dispatch_for_node <node> -- take <node> OUT of tools/pool_autodispatch.sh's
  # dispatch pool for the duration of a sync-before-stop window (2026-09-25 review, still-open item: "the node
  # was not taken out of dispatch before the sync" -- the existing re-check-after-sync only DETECTS a job that
  # landed during the sync window; this PREVENTS one from landing there in the first place, closing the race at
  # its source rather than narrowing it). Prints "1" (stdout) iff it actually removed a registration the caller
  # must restore with resume_dispatch_for_node; prints "0" for a no-op (the node was never registered -- e.g.
  # the single-instance `.aws_gpu`/`.aws_cpuN` lanes, which are never dispatch targets to begin with, or a pool
  # node between `up`/`down`'s own bracketing).
  local node="$1"
  if [ -z "$node" ] || [ ! -f "$EXTRA_NODES_FILE" ] || ! grep -qxF "$node" "$EXTRA_NODES_FILE" 2>/dev/null; then
    echo 0; return 0
  fi
  local dir tmp; dir="$(dirname "$EXTRA_NODES_FILE")"
  tmp=$(mktemp "$dir/.extra_nodes.XXXXXX") || { echo 0; return 0; }
  # `grep -v`'s OWN exit status is not the success signal here (bug caught by this fix's own test): it is 1
  # ("no lines selected") whenever <node> was the ONLY registered node -- an entirely successful, empty result
  # -- not an error. Only `mv`'s exit status (the real filesystem operation) decides success, matching
  # aws_pool_node.sh cmd_down's own `grep -vxF ... || true` treatment of this exact same case.
  grep -vxF "$node" "$EXTRA_NODES_FILE" > "$tmp" 2>/dev/null
  if mv "$tmp" "$EXTRA_NODES_FILE"; then
    echo 1
  else
    rm -f "$tmp"
    echo 0
  fi
}

resume_dispatch_for_node() {   # resume_dispatch_for_node <node> <was_registered> -- undo pause_dispatch_for_node,
  # restoring <node>'s registration when (and only when) it was actually removed above; a no-op otherwise.
  # Callers run this UNCONDITIONALLY after the sync+decision, whether or not the node ends up actually stopped
  # this cycle -- registration tracks 'up'/'down' (tools/aws_pool_node.sh), not any one idle-stop/budget-enforce
  # cycle's outcome, so a stopped-but-still-registered node matches the steady state `start`/`refresh` already
  # expect (aws_pool_node.sh's own `down` is the only place that removes a registration for good).
  local node="$1" was="$2"
  [ "$was" = "1" ] || return 0
  [ -n "$node" ] && [ -f "$EXTRA_NODES_FILE" ] || return 0
  grep -qxF "$node" "$EXTRA_NODES_FILE" 2>/dev/null || echo "$node" >> "$EXTRA_NODES_FILE"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  set -uo pipefail
  ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  LOG="${AWS_SYNC_LOG:-${LOG:-/dev/stderr}}"
  POOL_SSH_CFG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
  EXTRA_NODES_FILE="${POOL_EXTRA_NODES_FILE:-$ROOT/research/queue/.pool_extra_nodes}"
  case "${1:-}" in
    --sync)
      [ "$#" -eq 4 ] || { echo "usage: $0 --sync <node> <ip> <key>" >&2; exit 2; }
      sync_node_before_stop "$2" "$3" "$4"; exit $? ;;
    --pause)
      [ "$#" -eq 2 ] || { echo "usage: $0 --pause <node>" >&2; exit 2; }
      pause_dispatch_for_node "$2"; exit 0 ;;
    --resume)
      [ "$#" -eq 3 ] || { echo "usage: $0 --resume <node> <was_registered>" >&2; exit 2; }
      resume_dispatch_for_node "$2" "$3"; exit 0 ;;
    *) echo "usage: $0 {--sync <node> <ip> <key>|--pause <node>|--resume <node> <was_registered>}" >&2; exit 2 ;;
  esac
fi
