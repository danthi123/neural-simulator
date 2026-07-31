#!/usr/bin/env bash
# pool_autodispatch.sh — keep the mini-PC pool fed WITHOUT waiting for me to notice it went idle.
#
# WHY (2026-07-31, owner-flagged after the same lapse recurred seven times in one session). The cluster went from
# idle-all-night to working only after the owner pointed it out, and the check that was meant to catch it was
# first structurally blind to the cluster and then unable to fire at all. Once BOTH were fixed the pattern became:
# heartbeat fires -> I dispatch -> the job finishes in ~12 minutes -> the pool idles -> heartbeat fires again.
# Detection was solved; ANTICIPATION was not, so utilisation was capped by my response latency rather than by the
# hardware. Seven idle-pool alarms in one session, every one of them work that could have been queued in advance.
#
# THE FIX IS A QUEUE, not a faster reaction. Jobs are staged ahead of time; this loop hands the next one to
# whichever node is free. The mechanical part is in tools/workflow_check.sh, which now fails when the QUEUE IS
# EMPTY -- i.e. it alarms on "nothing staged", the actual defect, rather than on "pool idle", the symptom.
#
#   bash tools/pool_queue.sh add '<remote shell command>'     # stage work
#   nohup bash tools/pool_autodispatch.sh > /tmp/pool_dispatch.log 2>&1 &
#
# A job line is a command run on the node, from ~/derisk-pool/sim. Lines starting with # are ignored.
set -uo pipefail
ROOT=/home/dant123/Projects/sim
QUEUE="$ROOT/research/queue/pool.queue"
CLAIMED="$ROOT/research/queue/pool.running"
POLL="${POOL_DISPATCH_POLL:-60}"
NODES="${POOL_NODES:-pool40 pool41 pool42}"

mkdir -p "$(dirname "$QUEUE")"; touch "$QUEUE" "$CLAIMED"

node_is_idle() {
  # Idle = no research runner AND load below a quarter of its cores. Bracket the pattern: an un-bracketed one
  # matches the ssh command carrying it, which is the self-match that made an earlier check unable to ever fire.
  local out
  out=$(timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=6 "$1" \
        "echo \$(nproc) \$(cut -d' ' -f1 /proc/loadavg) \$(pgrep -fc '[r]esearch\.runners' 2>/dev/null | head -1)" 2>/dev/null) || return 1
  set -- $out
  local cores="${1:-0}" load="${2:-99}" procs="${3:-1}"
  [ "${procs:-1}" -eq 0 ] || return 1
  awk -v l="$load" -v c="$cores" 'BEGIN{exit !(l < c/4)}'
}

pop_job() {
  # Atomically take the first non-comment line. flock keeps two dispatcher instances from claiming the same job.
  local job=""
  exec 9>"$QUEUE.lock"
  flock 9 || return 1
  # STALENESS GUARD: an entry older than MAX_AGE is debris, not staged work. Learned immediately -- the first
  # dispatcher run found 69 jobs from an opsweep abandoned days earlier and launched three of them.
  local now cutoff
  now=$(date +%s); cutoff=$(( now - ${POOL_JOB_MAX_AGE:-43200} ))
  job=$(awk -F'\t' -v c="$cutoff" 'NF>1 && $1+0 >= c {print $2; exit}' "$QUEUE")
  local stale
  stale=$(awk -F'\t' -v c="$cutoff" 'NF>1 && $1+0 < c' "$QUEUE" | wc -l)
  [ "${stale:-0}" -gt 0 ] && echo "[pool-dispatch] SKIPPING $stale stale entr(ies) older than $(( ${POOL_JOB_MAX_AGE:-43200} / 3600 ))h"
  if [ -z "$job" ]; then flock -u 9; printf ''; return 0; fi
  if true; then
    grep -vF "	$job" "$QUEUE" > "$QUEUE.tmp" 2>/dev/null || true
    mv "$QUEUE.tmp" "$QUEUE"          # unconditional: grep -v exits 1 when it filters everything, and a
                                       # `&& mv` there once made a single-line queue never clear, relaunching
                                       # the same job nine times.
  fi
  flock -u 9
  printf '%s' "$job"
}

# SINGLETON GUARD (2026-07-31): repeated restarts during testing left THREE dispatchers polling at once. flock
# in pop_job stops two of them claiming the same job, so correctness was safe -- but three pollers triple the ssh
# load on every node each cycle, and make "is the dispatcher up?" ambiguous, which now matters because
# workflow_check gates on exactly that. Refuse to start when one is already running.
# Exempt the systemd-managed run: systemd ALREADY guarantees one instance per service, so applying the guard
# there makes the two fight -- systemd restarts the unit, the fresh instance sees the outgoing one, exits 0, and
# Restart=always loops it forever (the unit sits in "activating" and never reaches active). INVOCATION_ID is set
# by systemd only. A MANUAL start still refuses, which is the case the guard was actually written for.
# Ask systemd, do not sniff the process table. Two earlier attempts got this wrong:
#   (1) a bare pgrep guard fought systemd -- the unit restarts, the fresh instance saw the outgoing one, exited 0,
#       and Restart=always looped it forever, leaving the unit stuck "activating";
#   (2) exempting systemd via INVOCATION_ID silently disabled the guard for EVERY manual start, because that
#       variable is set for any process under a systemd unit and is INHERITED by children -- Claude Code itself
#       runs under one, so the exemption always fired and a manual start happily launched a second dispatcher.
# MainPID is unambiguous: refuse only when the unit is active and this process is not the unit.
_MAIN=$(systemctl --user show -p MainPID --value pool-dispatch.service 2>/dev/null || echo 0)
if systemctl --user is-active --quiet pool-dispatch.service 2>/dev/null && [ "${_MAIN:-0}" != "$$" ]; then
  echo "[pool-dispatch] REFUSING to start: pool-dispatch.service is active (MainPID $_MAIN). Use: systemctl --user restart pool-dispatch.service"
  exit 0
fi

echo "[pool-dispatch] started $(date '+%H:%M:%S') | queue=$QUEUE | poll=${POLL}s | nodes=$NODES"
while true; do
  for NODE in $NODES; do
    if node_is_idle "$NODE"; then
      JOB=$(pop_job)
      [ -z "$JOB" ] && continue
      echo "[pool-dispatch] $(date '+%H:%M:%S') $NODE <- $JOB"
      printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$NODE" "$JOB" >> "$CLAIMED"
      ssh -f -n -o BatchMode=yes "$NODE" \
        "cd ~/derisk-pool/sim && setsid bash -c '$JOB' </dev/null > autodispatch.out 2>&1 & exit 0" 2>/dev/null
      sleep 5     # let the launch register before this node is polled again
    fi
  done
  sleep "$POLL"
done
