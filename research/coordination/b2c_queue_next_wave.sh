#!/usr/bin/env bash
# b2c_queue_next_wave.sh -- queue the NEXT wave of Battery B2c (tags b2c0925-base + b2c0925-flipcand) onto the pool,
# only when the queue is nearly drained of this battery. Modelled on b2b_queue_next_wave.sh (B2b Amendment 1, A1.8).
#
# Prereg: research/findings/2026-09-25-production-default-battery-B2c-paired-flip-PREREGISTRATION.md ("Queueing").
# WHY WAVES: tools/pool_autodispatch.sh skips any queue line older than POOL_JOB_MAX_AGE (default 43200 s; the live
# pool-dispatch.service sets no override), and a skipped line stays queued and never runs. 516 lines at once would
# leave the tail past 12 h, so B2c goes in six waves of 86 lines: ONE seed per wave, each (row, seed) base line
# immediately followed by its flipcand twin (b2c0925_jobs.txt order), so the two cells of a pair dispatch together.
#
# WHAT IT DOES, in order:
#   1. refuses (exit 2) unless the start preconditions exist: the per-node corpus-hash record
#      (b2c0925_corpus_sha256.tsv, committed), PIN.txt for both tags + EXPECT_ENV.txt for flipcand only under the
#      primary checkout's shard tree (written by `b2c_make_jobs.sh --record-pin` there), and the registered pair
#      scorer tools/b2c_score.py -- so a heartbeat running this every cycle queues nothing until the battery is
#      deliberately started;
#   2. counts B2c lines still FRESH in the pool queue; if >= THRESHOLD (20), queues nothing;
#   3. otherwise takes the next WAVE_SIZE (86) lines of b2c0925_jobs.txt, in file order, that are not already queued
#      (any age), dispatched (pool.running), recorded as added in the ledger, or landed (lb.json in the primary
#      checkout), checks each (pinned cd + guard; base: no BRAIN_* token; flipcand: exactly the three pair tokens),
#      and adds it with tools/pool_queue.sh add ... --checked;
#   4. appends one ledger row per add (b2c0925_waves.tsv: epoch, wave, job-file line, out path, add rc);
#   5. checks every queued B2c line is whole (pinned cd at the head, `/lb.json  #checked:<REASON>` at the tail) and
#      reports stale B2c lines, which the prereg's re-run procedure handles -- this script never re-queues or moves one.
#
# Usage:  bash research/coordination/b2c_queue_next_wave.sh [--dry-run|--status]
#   --dry-run and --status write nothing (no touch, no ledger row, no add) and skip the start-precondition refusal
#   (they report it instead), so they are safe to run before the battery starts.
# Env:    B2C_WAVE_SIZE (86)  B2C_WAVE_THRESHOLD (20)  POOL_JOB_MAX_AGE (43200)  POOL_ROOT (/home/dant123/Projects/sim)
#         POOL_QUEUE_PATH (default $POOL_ROOT/research/queue/pool.queue)
# Exit:   0 = nothing to do or wave queued cleanly; 1 = an add failed or a queued line is malformed; 2 = refused/usage.
set -uo pipefail
export LC_ALL=C   # byte-order sort and grep: the env-token comparisons below must not depend on the locale

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POOL_ROOT="${POOL_ROOT:-/home/dant123/Projects/sim}"
QUEUE_TOOL="$POOL_ROOT/tools/pool_queue.sh"          # the primary checkout's copy: it has the AWS nodes' ssh config
Q="${POOL_QUEUE_PATH:-$POOL_ROOT/research/queue/pool.queue}"
RUNNING="${Q%.queue}.running"                        # the dispatcher's cumulative record of every dispatched job
JOBS="$HERE/b2c0925_jobs.txt"
LEDGER="$HERE/b2c0925_waves.tsv"
CORPUS_RECORD="$HERE/b2c0925_corpus_sha256.tsv"
SHARDS="$POOL_ROOT/research/findings/raw/_load_bearing/_shards"
F2=fd29040db19987819461693aaf385977e45840ef
PAT="/_shards/b2c0925-"                              # both tags
PAIR_TOKENS="BRAIN_DA_TAG_CAPTURE=1 BRAIN_DA_TAG_CAPTURE_CLOCK=turn BRAIN_SLEEP_REPLAY_CAPTURE=1"
REASON='prereg research/findings/2026-09-25-production-default-battery-B2c-paired-flip-PREREGISTRATION.md (F2 fd29040db); B2c base+flipcand; mem_gb=8'
WAVE_SIZE="${B2C_WAVE_SIZE:-86}"
THRESHOLD="${B2C_WAVE_THRESHOLD:-20}"
MAX_AGE="${POOL_JOB_MAX_AGE:-43200}"
MODE="${1:-queue}"

case "$MODE" in queue|--dry-run|--status) ;; *) echo "usage: $0 [--dry-run|--status]" >&2; exit 2 ;; esac
[ -f "$JOBS" ] || { echo "⛔ no job file $JOBS" >&2; exit 2; }
[ -f "$QUEUE_TOOL" ] || { echo "⛔ no $QUEUE_TOOL" >&2; exit 2; }
[ "$MODE" = "queue" ] && { touch "$Q" "$RUNNING" 2>/dev/null || true; }

exec 8>"${XDG_RUNTIME_DIR:-/tmp}/b2c0925_queue_next_wave.lock"
flock -n 8 || { echo "another b2c_queue_next_wave.sh is running; nothing done" >&2; exit 0; }

# ---- start preconditions (prereg "Queueing", items 1-3) ----
missing=()
[ -f "$CORPUS_RECORD" ] || missing+=("corpus-hash record $CORPUS_RECORD")
for t in b2c0925-base b2c0925-flipcand; do
  pf="$SHARDS/$t/PIN.txt"
  if [ ! -f "$pf" ] || [ "$(tr -d '[:space:]' < "$pf")" != "$F2" ]; then missing+=("$pf holding $F2"); fi
done
ef="$SHARDS/b2c0925-flipcand/EXPECT_ENV.txt"
if [ ! -f "$ef" ] || [ "$(sort "$ef" | tr '\n' ' ' | sed 's/ $//')" != "$PAIR_TOKENS" ]; then
  missing+=("$ef holding exactly: $PAIR_TOKENS")
fi
[ -f "$SHARDS/b2c0925-base/EXPECT_ENV.txt" ] && missing+=("NO EXPECT_ENV.txt for b2c0925-base (the base arm expects none)")
# the pair scorer the prereg specifies ("Scoring code") must be merged to main before any cell exists, so no rule
# is coded after data is seen. A file existing in the working tree is not "merged" (fix round, review of this
# branch): require it to be present, unchanged, in origin/main -- present at THAT ref (cat-file -e) and identical
# to the working tree's copy (diff --quiet), so an uncommitted or unpushed local copy still refuses the queue.
if ! git -C "$POOL_ROOT" cat-file -e origin/main:tools/b2c_score.py 2>/dev/null \
   || ! git -C "$POOL_ROOT" diff --quiet origin/main -- tools/b2c_score.py 2>/dev/null; then
  missing+=("the registered pair scorer tools/b2c_score.py is not merged, unchanged, to origin/main (git -C $POOL_ROOT cat-file -e origin/main:tools/b2c_score.py && git -C $POOL_ROOT diff --quiet origin/main -- tools/b2c_score.py)")
fi
if [ "${#missing[@]}" -gt 0 ]; then
  if [ "$MODE" = "queue" ]; then
    printf '⛔ REFUSED (B2c not started): missing %s\n' "${missing[@]}" >&2; exit 2
  fi
  printf '[b2c-wave] (not started) missing %s\n' "${missing[@]}"
fi

cat_if() { [ -f "$1" ] && cat "$1" || true; }
now=$(date +%s); cutoff=$(( now - MAX_AGE ))
fresh=$(cat_if "$Q" | awk -F'\t' -v c="$cutoff" -v t="$PAT" 'NF>1 && index($2,t) && $1+0 >= c' | wc -l)
stale=$(cat_if "$Q" | awk -F'\t' -v c="$cutoff" -v t="$PAT" 'NF>1 && index($2,t) && $1+0 < c' | wc -l)
dispatched=$(cat_if "$RUNNING" | grep -cF "$PAT"); dispatched=${dispatched:-0}
landed=$(ls "$SHARDS"/b2c0925-*/s*/*/lb.json 2>/dev/null | wc -l)
echo "[b2c-wave] b2c0925-*: fresh-queued=$fresh stale-queued=$stale dispatched-ever=$dispatched lb.json-landed=$landed" \
     "of $(wc -l < "$JOBS") (threshold $THRESHOLD, wave $WAVE_SIZE, max age ${MAX_AGE}s)"
[ "$stale" -gt 0 ] && echo "[b2c-wave] ⚠ $stale stale B2c line(s) will never dispatch -- apply the prereg's re-run rule (c)" >&2

line_ok() {  # the static per-line rule, same at generation, at queueing and on the queue
  local l=$1
  case "$l" in "cd ~/derisk-pool/revisions/$F2 && .venv/bin/python tools/assert_flipped_defaults.py && "*) ;; *) return 1 ;; esac
  local got
  got=$(printf '%s\n' "$l" | grep -oE 'BRAIN_[A-Z0-9_]+=[^ ]+' | sort | tr '\n' ' ' | sed 's/ $//')
  case "$l" in
    *"/_shards/b2c0925-base/"*) [ -z "$got" ] ;;
    *"/_shards/b2c0925-flipcand/"*) [ "$got" = "$PAIR_TOKENS" ] ;;
    *) return 1 ;;
  esac
}

check_queued_lines() {
  local bad=0 n=0 l
  while IFS= read -r l; do
    n=$((n + 1))
    case "$l" in *"/lb.json  #checked:$REASON") ;; *) bad=$((bad + 1)); continue ;; esac
    line_ok "${l%%  #checked:*}" || bad=$((bad + 1))
  done < <(cat_if "$Q" | awk -F'\t' -v t="$PAT" 'NF>1 && index($2,t)' | cut -f2-)
  if [ "$bad" -gt 0 ]; then
    echo "[b2c-wave] ⛔ $bad queued B2c line(s) are malformed (torn / wrong pin / wrong env / wrong reason)" >&2; return 1
  fi
  echo "[b2c-wave] queued B2c lines well-formed: $n"
}

if [ "$MODE" = "--status" ]; then check_queued_lines; exit $?; fi
# A wave with fewer than WAVE_SIZE successful adds in the ledger is RESUMED whatever the queue depth; only a full wave
# waits for the depth threshold before the next one starts (b2b_queue_next_wave.sh's rule).
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
  echo "[b2c-wave] $fresh fresh B2c lines still queued (>= $THRESHOLD): nothing queued"; check_queued_lines; exit $?
fi
[ "$wave_done" -gt 0 ] && echo "[b2c-wave] resuming wave $wave: $wave_done of $WAVE_SIZE already added, $todo to go"

declare -A HANDLED=()
while IFS= read -r p; do HANDLED["$p"]=1; done < <(
  { cat_if "$Q"; cat_if "$RUNNING"
    [ -f "$LEDGER" ] && awk -F'\t' '$5=="0" {print $4}' "$LEDGER"; } \
  | grep -oE "research/findings/raw/_load_bearing/_shards/b2c0925-(base|flipcand)/s[0-9]+/[a-z0-9-]+/lb\.json" | sort -u)

[ "$MODE" = "queue" ] && [ ! -f "$LEDGER" ] && printf 'epoch\twave\tjob_line\tout\tadd_rc\n' > "$LEDGER"

n=0; fail=0; ln=0
while IFS= read -r line <&3; do
  ln=$((ln + 1))
  [ "$n" -ge "$todo" ] && break
  out=$(printf '%s' "$line" | grep -oE -- '--out [^ ]+$' | cut -d' ' -f2)
  [ -n "$out" ] || { echo "⛔ job line $ln has no trailing --out" >&2; fail=1; continue; }
  [ -n "${HANDLED[$out]:-}" ] && continue
  [ -f "$POOL_ROOT/$out" ] && continue
  line_ok "$line" || { echo "⛔ job line $ln fails the static rule (pin / guard / env tokens)" >&2; fail=1; continue; }
  n=$((n + 1))
  if [ "$MODE" = "--dry-run" ]; then echo "[b2c-wave] would queue (wave $wave, line $ln): $out"; continue; fi
  # </dev/null: pool_queue.sh's ssh probes read stdin (b2b_queue_next_wave.sh, first run 2026-09-24).
  bash "$QUEUE_TOOL" add "$line" --checked "$REASON" </dev/null >/dev/null 2>"${XDG_RUNTIME_DIR:-/tmp}/b2c0925_add.err"
  rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$wave" "$ln" "$out" "$rc" >> "$LEDGER"
  if [ "$rc" -ne 0 ]; then
    fail=1; echo "[b2c-wave] ⛔ add failed (rc=$rc) for line $ln: $out" >&2
    tail -3 "${XDG_RUNTIME_DIR:-/tmp}/b2c0925_add.err" >&2
  else
    echo "[b2c-wave] queued wave $wave line $ln: $out"
  fi
done 3< "$JOBS"

[ "$n" -eq 0 ] && echo "[b2c-wave] no unqueued B2c line left in $(basename "$JOBS")"
[ "$MODE" = "--dry-run" ] && { echo "[b2c-wave] dry run: $n line(s) would be queued as wave $wave"; exit "$fail"; }
echo "[b2c-wave] wave $wave: $n line(s) attempted"
check_queued_lines || fail=1
exit "$fail"
