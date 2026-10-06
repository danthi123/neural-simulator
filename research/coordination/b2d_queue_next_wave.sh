#!/usr/bin/env bash
# b2d_queue_next_wave.sh -- queue the NEXT wave of Battery B2d's stub arm (tag b2d0929) onto the pool, only when the
# queue is nearly drained of this tag. Written 2026-09-30 from b2b_queue_next_wave.sh (prereg Step 6, NEEDS-CLAUDE
# item 5 of the B2d HANDOFF), per research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md
# Step 6 and its "Amendment 1" section.
#
# Differences from the B2b script, and nothing else:
#   * the pin P is read from research/findings/raw/_load_bearing/_shards/b2d0929/PIN.txt in THIS checkout (written by
#     research/packets/b2d-mouth-renderer/repin.py apply); every job line must be `cd ~/derisk-pool/revisions/<P> && `;
#   * TAG=b2d0929, JOBS=b2d_jobs.txt, LEDGER=b2d_waves.tsv, CORPUS_RECORD=b2d_corpus_sha256.tsv, its own lock file;
#   * the corpus record must hold a row for P (source_git_sha = P, provisioned_ok yes) for every node in B2D_NODES
#     (default "pool41 pool42"), not merely exist;
#   * WAVE_SIZE 43 (one seed) and THRESHOLD 10 by default (the pool-only lane); B2D_WAVE_SIZE=86 B2D_WAVE_THRESHOLD=20
#     when AWS nodes serve the pool too;
#   * the reason carries `max_age_h=168`, and "fresh" uses the dispatcher's own per-line rule (tools/pool_autodispatch.sh
#     POOL_FRESH_AWK: a line's max_age_h=N, capped at 240 h, else POOL_JOB_MAX_AGE), so a line waiting behind a
#     three-to-four-day queue is neither skipped by the dispatcher nor counted stale here.
#
# Usage:  bash research/coordination/b2d_queue_next_wave.sh [--dry-run|--status]
# Env:    B2D_WAVE_SIZE (43)  B2D_WAVE_THRESHOLD (10)  B2D_NODES ("pool41 pool42")  POOL_JOB_MAX_AGE (43200)
#         POOL_ROOT (/home/dant123/Projects/sim)
# Exit:   0 = nothing to do or wave queued cleanly; 1 = an add failed or a queued line is malformed; 2 = refused.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POOL_ROOT="${POOL_ROOT:-/home/dant123/Projects/neural-simulator}"
QUEUE_TOOL="$POOL_ROOT/tools/pool_queue.sh"          # the primary checkout's copy: it has the AWS nodes' ssh config
Q="${POOL_QUEUE_PATH:-$POOL_ROOT/research/queue/pool.queue}"
RUNNING="${Q%.queue}.running"                        # the dispatcher's cumulative record of every dispatched job
JOBS="$HERE/b2d_jobs.txt"
LEDGER="$HERE/b2d_waves.tsv"
CORPUS_RECORD="$HERE/b2d_corpus_sha256.tsv"
PIN_FILE="$HERE/../findings/raw/_load_bearing/_shards/b2d0929/PIN.txt"
F=$(tr -d '[:space:]' < "$PIN_FILE" 2>/dev/null)
[[ "$F" =~ ^[0-9a-f]{40}$ ]] || { echo "⛔ REFUSED: $PIN_FILE does not hold a full pin" >&2; exit 2; }
TAG=b2d0929
REASON="prereg research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md (P ${F:0:9}) AMENDMENT 1; B2d stub arm; mem_gb=8 max_age_h=168"
WAVE_SIZE="${B2D_WAVE_SIZE:-43}"
THRESHOLD="${B2D_WAVE_THRESHOLD:-10}"
NODES="${B2D_NODES:-pool41 pool42}"
MAX_AGE="${POOL_JOB_MAX_AGE:-43200}"
MODE="${1:-queue}"

case "$MODE" in queue|--dry-run|--status) ;; *) echo "usage: $0 [--dry-run|--status]" >&2; exit 2 ;; esac
[ -f "$JOBS" ] || { echo "⛔ no job file $JOBS" >&2; exit 2; }
[ -x "$QUEUE_TOOL" ] || [ -f "$QUEUE_TOOL" ] || { echo "⛔ no $QUEUE_TOOL" >&2; exit 2; }
touch "$Q" "$RUNNING" 2>/dev/null || true

exec 8>"${XDG_RUNTIME_DIR:-/tmp}/b2d0929_queue_next_wave.lock"
flock -n 8 || { echo "another b2d_queue_next_wave.sh is running; nothing done" >&2; exit 0; }

now=$(date +%s)
# the dispatcher's per-line rule (tools/pool_autodispatch.sh POOL_FRESH_AWK): max_age_h=N in the line (cap 240 h), else MAX_AGE
AGE_AWK='function maxage(l,  m) { if (match(l, /max_age_h=[0-9]+/)) { m = substr(l, RSTART + 10, RLENGTH - 10) + 0; if (m > cap) m = cap; return m * 3600 } return d }'
fresh=$(awk -F'\t' -v now="$now" -v d="$MAX_AGE" -v cap="${POOL_JOB_MAX_AGE_CAP_H:-240}" -v t="/_shards/$TAG/" "$AGE_AWK"' NF>1 && index($2,t) && now - $1 <= maxage($0)' "$Q" | wc -l)
stale=$(awk -F'\t' -v now="$now" -v d="$MAX_AGE" -v cap="${POOL_JOB_MAX_AGE_CAP_H:-240}" -v t="/_shards/$TAG/" "$AGE_AWK"' NF>1 && index($2,t) && now - $1 > maxage($0)' "$Q" | wc -l)
dispatched=$(grep -cF "/_shards/$TAG/" "$RUNNING" 2>/dev/null); dispatched=${dispatched:-0}
landed=$(ls "$POOL_ROOT"/research/findings/raw/_load_bearing/_shards/$TAG/s*/*/lb.json 2>/dev/null | wc -l)
echo "[b2d-wave] $TAG: fresh-queued=$fresh stale-queued=$stale dispatched-ever=$dispatched lb.json-landed=$landed" \
     "of $(wc -l < "$JOBS") (threshold $THRESHOLD, wave $WAVE_SIZE, max age ${MAX_AGE}s)"
[ "$stale" -gt 0 ] && echo "[b2d-wave] ⚠ $stale stale $TAG line(s) will never dispatch -- apply prereg A1.4 (c)" >&2

check_queued_lines() {
  # Every queued b2d0929 line must be whole: pinned cd at the head, lb.json + a recognised checked reason at the
  # tail. Accepts the wave REASON, or a reason beginning with the A1.4 re-run prefix (the HANDOFF's re-run lines name
  # the cell and the first host, so they cannot match one fixed string).
  local bad amendment2_prefix="prereg research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md (P ${F:0:9}) AMENDMENT 1; B2d re-run"
  bad=$(awk -F'\t' -v t="/_shards/$TAG/" 'NF>1 && index($2,t)' "$Q" | cut -f2- \
        | awk -v head="cd ~/derisk-pool/revisions/$F && " -v marker="/lb.json  #checked:" \
              -v reason1="$REASON" -v a2prefix="$amendment2_prefix" '
          {
            ok = 0
            if (index($0, head) == 1) {
              midx = index($0, marker)
              if (midx > 0) {
                reason = substr($0, midx + length(marker))
                if (reason == reason1) ok = 1
                else if (index(reason, a2prefix) == 1) ok = 1
              }
            }
            if (!ok) print
          }' | wc -l)
  if [ "$bad" -gt 0 ]; then
    echo "[b2d-wave] ⛔ $bad queued $TAG line(s) are malformed (torn / wrong pin / wrong reason)" >&2; return 1
  fi
  echo "[b2d-wave] queued $TAG lines well-formed: $(awk -F'\t' -v t="/_shards/$TAG/" 'NF>1 && index($2,t)' "$Q" | wc -l)"
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
  echo "[b2d-wave] $fresh fresh $TAG lines still queued (>= $THRESHOLD): nothing queued"; check_queued_lines; exit $?
fi
[ "$wave_done" -gt 0 ] && echo "[b2d-wave] resuming wave $wave: $wave_done of $WAVE_SIZE already added, $todo to go"
if [ "$MODE" = "queue" ] && [ ! -f "$CORPUS_RECORD" ]; then
  echo "⛔ REFUSED: $CORPUS_RECORD missing -- prereg A1.6 requires the per-node corpus hash recorded (and committed)" \
       "before any line is queued" >&2; exit 2
fi
if [ "$MODE" = "queue" ]; then
  for node in $NODES; do
    awk -F'\t' -v n="$node" -v p="$F" '$1==n && $5==p && $6=="yes" {ok=1} END {exit !ok}' "$CORPUS_RECORD" || {
      echo "⛔ REFUSED: $CORPUS_RECORD has no corpus-hash row for node $node at pin $F (prereg Step 4 / A1.6)" >&2; exit 2; }
  done
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
  case "$line" in "cd ~/derisk-pool/revisions/$F && "*) ;; *) echo "⛔ job line $ln is not pinned to P=$F" >&2; fail=1; continue ;; esac
  case "$line" in *BRAIN_*) echo "⛔ job line $ln carries a BRAIN_* token (base arm must carry none)" >&2; fail=1; continue ;; esac
  n=$((n + 1))
  if [ "$MODE" = "--dry-run" ]; then echo "[b2d-wave] would queue (wave $wave, line $ln): $out"; continue; fi
  # </dev/null: pool_queue.sh's ssh probes read stdin; on the loop's stdin they swallowed the rest of the job file
  # (first run, 2026-09-24: one line queued, then the loop ended).
  bash "$QUEUE_TOOL" add "$line" --checked "$REASON" </dev/null >/dev/null 2>"${XDG_RUNTIME_DIR:-/tmp}/b2d0929_add.err"
  rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$wave" "$ln" "$out" "$rc" >> "$LEDGER"
  if [ "$rc" -ne 0 ]; then
    fail=1; echo "[b2d-wave] ⛔ add failed (rc=$rc) for line $ln: $out" >&2
    tail -3 "${XDG_RUNTIME_DIR:-/tmp}/b2d0929_add.err" >&2
  else
    echo "[b2d-wave] queued wave $wave line $ln: $out"
  fi
done 3< "$JOBS"

[ "$n" -eq 0 ] && echo "[b2d-wave] no unqueued $TAG line left in $(basename "$JOBS")"
[ "$MODE" = "--dry-run" ] && { echo "[b2d-wave] dry run: $n line(s) would be queued as wave $wave"; exit "$fail"; }
echo "[b2d-wave] wave $wave: $n line(s) attempted"
check_queued_lines || fail=1
exit "$fail"
