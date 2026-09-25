#!/usr/bin/env bash
# b2b_queue_next_wave.sh -- queue the NEXT wave of Battery B2b's base arm (tag b2b0924-base) onto the pool, but only
# when the queue is nearly drained of this tag.
#
# WHY WAVES (prereg research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md, Amendment 1, A1.8):
# tools/pool_autodispatch.sh skips any queue line older than POOL_JOB_MAX_AGE (default 43200 s; the live
# pool-dispatch.service sets no override). Age = now - the epoch tools/pool_queue.sh writes as the line's first field.
# A skipped line stays in the queue and never runs. With ~150 lines ahead and 5-41 dispatches/hour, all 258 lines at
# once would leave the tail past 12 h, so the arm goes in three waves of 86 lines (two whole seeds each, job-file order).
#
# WHAT IT DOES, in order:
#   1. refuses unless the corpus-hash record (A1.6) exists -- it must be committed before the first line is queued;
#   2. counts b2b0924-base lines still FRESH in the pool queue; if >= THRESHOLD (20), queues nothing;
#   3. otherwise takes the next WAVE_SIZE (86) lines of b2b0924_base_jobs.txt, in file order, that are not already
#      queued (any age), not already dispatched (pool.running), not already recorded as added in the ledger, and whose
#      lb.json is not already in the primary checkout -- and adds each with tools/pool_queue.sh add ... --checked;
#   4. appends one ledger row per add (b2b0924_base_waves.tsv: epoch, wave, job-file line, out path, add rc);
#   5. checks every queued b2b0924-base line starts with `cd ~/derisk-pool/revisions/<F> && ` and ends with
#      `/lb.json  #checked:<reason>` (the torn-line class that hit B2a) and reports stale b2b0924-base lines, which
#      the prereg's re-run procedure (A1.4 c) handles -- this script never re-queues or moves a stale line.
#
# Usage:  bash research/coordination/b2b_queue_next_wave.sh [--dry-run|--status]
# Env:    B2B_WAVE_SIZE (86)  B2B_WAVE_THRESHOLD (20)  POOL_JOB_MAX_AGE (43200)  POOL_ROOT (/home/dant123/Projects/sim)
# Exit:   0 = nothing to do or wave queued cleanly; 1 = an add failed or a queued line is malformed; 2 = refused.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POOL_ROOT="${POOL_ROOT:-/home/dant123/Projects/sim}"
QUEUE_TOOL="$POOL_ROOT/tools/pool_queue.sh"          # the primary checkout's copy: it has the AWS nodes' ssh config
Q="${POOL_QUEUE_PATH:-$POOL_ROOT/research/queue/pool.queue}"
RUNNING="${Q%.queue}.running"                        # the dispatcher's cumulative record of every dispatched job
JOBS="$HERE/b2b0924_base_jobs.txt"
LEDGER="$HERE/b2b0924_base_waves.tsv"
CORPUS_RECORD="$HERE/b2b0924_corpus_sha256.tsv"
F=a308f1e09babcc9ed096c3c8046d00040391368c
TAG=b2b0924-base
REASON='prereg research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md (F a308f1e09) AMENDMENT 1; B2b base; mem_gb=8'
WAVE_SIZE="${B2B_WAVE_SIZE:-86}"
THRESHOLD="${B2B_WAVE_THRESHOLD:-20}"
MAX_AGE="${POOL_JOB_MAX_AGE:-43200}"
MODE="${1:-queue}"

case "$MODE" in queue|--dry-run|--status) ;; *) echo "usage: $0 [--dry-run|--status]" >&2; exit 2 ;; esac
[ -f "$JOBS" ] || { echo "⛔ no job file $JOBS" >&2; exit 2; }
[ -x "$QUEUE_TOOL" ] || [ -f "$QUEUE_TOOL" ] || { echo "⛔ no $QUEUE_TOOL" >&2; exit 2; }
touch "$Q" "$RUNNING" 2>/dev/null || true

exec 8>"${XDG_RUNTIME_DIR:-/tmp}/b2b0924_queue_next_wave.lock"
flock -n 8 || { echo "another b2b_queue_next_wave.sh is running; nothing done" >&2; exit 0; }

now=$(date +%s); cutoff=$(( now - MAX_AGE ))
fresh=$(awk -F'\t' -v c="$cutoff" -v t="/_shards/$TAG/" 'NF>1 && index($2,t) && $1+0 >= c' "$Q" | wc -l)
stale=$(awk -F'\t' -v c="$cutoff" -v t="/_shards/$TAG/" 'NF>1 && index($2,t) && $1+0 < c' "$Q" | wc -l)
dispatched=$(grep -cF "/_shards/$TAG/" "$RUNNING" 2>/dev/null); dispatched=${dispatched:-0}
landed=$(ls "$POOL_ROOT"/research/findings/raw/_load_bearing/_shards/$TAG/s*/*/lb.json 2>/dev/null | wc -l)
echo "[b2b-wave] $TAG: fresh-queued=$fresh stale-queued=$stale dispatched-ever=$dispatched lb.json-landed=$landed" \
     "of $(wc -l < "$JOBS") (threshold $THRESHOLD, wave $WAVE_SIZE, max age ${MAX_AGE}s)"
[ "$stale" -gt 0 ] && echo "[b2b-wave] ⚠ $stale stale $TAG line(s) will never dispatch -- apply prereg A1.4 (c)" >&2

check_queued_lines() {
  # Every queued b2b0924-base line must be whole: pinned cd at the head, lb.json + the checked reason at the tail.
  local bad
  bad=$(awk -F'\t' -v t="/_shards/$TAG/" 'NF>1 && index($2,t)' "$Q" | cut -f2- \
        | awk -v head="cd ~/derisk-pool/revisions/$F && " -v tail="/lb.json  #checked:$REASON" \
              'index($0,head)!=1 || substr($0, length($0)-length(tail)+1) != tail' | wc -l)
  if [ "$bad" -gt 0 ]; then
    echo "[b2b-wave] ⛔ $bad queued $TAG line(s) are malformed (torn / wrong pin / wrong reason)" >&2; return 1
  fi
  echo "[b2b-wave] queued $TAG lines well-formed: $(awk -F'\t' -v t="/_shards/$TAG/" 'NF>1 && index($2,t)' "$Q" | wc -l)"
}

if [ "$MODE" = "--status" ]; then check_queued_lines; exit $?; fi
# A wave with fewer than WAVE_SIZE successful adds in the ledger (interrupted, or an add failed) is RESUMED -- filled up
# to WAVE_SIZE -- whatever the queue depth; only a full wave waits for the depth threshold before the next one starts.
wave=1; wave_done=0
if [ -f "$LEDGER" ]; then
  wave=$(awk -F'\t' 'NR>1 {if ($2+0>m) m=$2+0} END {print m+0}' "$LEDGER")
  wave_done=$(awk -F'\t' -v w="$wave" 'NR>1 && $2+0==w && $5=="0"' "$LEDGER" | wc -l)
  if [ "$wave" -eq 0 ]; then wave=1; wave_done=0
  elif [ "$wave_done" -ge "$WAVE_SIZE" ]; then wave=$((wave + 1)); wave_done=0
  fi
fi
todo=$(( WAVE_SIZE - wave_done ))
if [ "$wave_done" -eq 0 ] && [ "$fresh" -ge "$THRESHOLD" ]; then
  echo "[b2b-wave] $fresh fresh $TAG lines still queued (>= $THRESHOLD): nothing queued"; check_queued_lines; exit $?
fi
[ "$wave_done" -gt 0 ] && echo "[b2b-wave] resuming wave $wave: $wave_done of $WAVE_SIZE already added, $todo to go"
if [ "$MODE" = "queue" ] && [ ! -f "$CORPUS_RECORD" ]; then
  echo "⛔ REFUSED: $CORPUS_RECORD missing -- prereg A1.6 requires the per-node corpus hash recorded (and committed)" \
       "before any line is queued" >&2; exit 2
fi

# Out paths already handled: queued (any age), dispatched, or added per the ledger; plus lb.json already landed.
declare -A HANDLED=()
while IFS= read -r p; do HANDLED["$p"]=1; done < <(
  { cat "$Q" "$RUNNING" 2>/dev/null
    [ -f "$LEDGER" ] && awk -F'\t' '$5=="0" {print $4}' "$LEDGER"; } \
  | grep -oE "research/findings/raw/_load_bearing/_shards/$TAG/s[0-9]+/[a-z0-9-]+/lb\.json" | sort -u)

[ "$MODE" = "queue" ] && [ ! -f "$LEDGER" ] && printf 'epoch\twave\tjob_line\tout\tadd_rc\n' > "$LEDGER"

n=0; fail=0; ln=0
while IFS= read -r line <&3; do
  ln=$((ln + 1))
  [ "$n" -ge "$todo" ] && break
  out=$(printf '%s' "$line" | grep -oE -- '--out [^ ]+$' | cut -d' ' -f2)
  [ -n "$out" ] || { echo "⛔ job line $ln has no trailing --out" >&2; fail=1; continue; }
  [ -n "${HANDLED[$out]:-}" ] && continue
  [ -f "$POOL_ROOT/$out" ] && continue
  case "$line" in "cd ~/derisk-pool/revisions/$F && "*) ;; *) echo "⛔ job line $ln is not pinned to F" >&2; fail=1; continue ;; esac
  case "$line" in *BRAIN_*) echo "⛔ job line $ln carries a BRAIN_* token (base arm must carry none)" >&2; fail=1; continue ;; esac
  n=$((n + 1))
  if [ "$MODE" = "--dry-run" ]; then echo "[b2b-wave] would queue (wave $wave, line $ln): $out"; continue; fi
  # </dev/null: pool_queue.sh's ssh probes read stdin; on the loop's stdin they swallowed the rest of the job file
  # (first run, 2026-09-24: one line queued, then the loop ended).
  bash "$QUEUE_TOOL" add "$line" --checked "$REASON" </dev/null >/dev/null 2>"${XDG_RUNTIME_DIR:-/tmp}/b2b0924_add.err"
  rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$wave" "$ln" "$out" "$rc" >> "$LEDGER"
  if [ "$rc" -ne 0 ]; then
    fail=1; echo "[b2b-wave] ⛔ add failed (rc=$rc) for line $ln: $out" >&2
    tail -3 "${XDG_RUNTIME_DIR:-/tmp}/b2b0924_add.err" >&2
  else
    echo "[b2b-wave] queued wave $wave line $ln: $out"
  fi
done 3< "$JOBS"

[ "$n" -eq 0 ] && echo "[b2b-wave] no unqueued $TAG line left in $(basename "$JOBS")"
[ "$MODE" = "--dry-run" ] && { echo "[b2b-wave] dry run: $n line(s) would be queued as wave $wave"; exit "$fail"; }
echo "[b2b-wave] wave $wave: $n line(s) attempted"
check_queued_lines || fail=1
exit "$fail"
