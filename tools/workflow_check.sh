#!/usr/bin/env bash
# workflow_check.sh — the three workflow rules the owner keeps having to enforce BY HAND, made mechanical.
#
# WHY (2026-07-30, owner: "these are the kinds of things I feel I have to keep manually reminding you to do").
# Every rule below ALREADY had a tool. All three were bypassed in a single session:
#   * lane_check.py printed "OK — 2 lanes" while SIX lanes sat unserved  -> threshold too lax to bite.
#   * research_gate.sh worked both times it ran -> but nothing ever MADE it run.
#   * queue_add.sh forces a record-check on enqueue -> bypassed by launching with `setsid nohup` directly.
# The failure was never the tools. It was that acting on a generic alarm ("CPU-LANES-STALE") required
# INVENTING work, so the cheap response was always to read it and move on. It fired for 75 minutes.
#
# THE FIX: this prints LITERAL, READY-TO-RUN COMMANDS for whatever is idle, so acting costs a copy-paste
# instead of a planning session. Run it from the heartbeat every cycle.
#
#   bash tools/workflow_check.sh            # full report + exit 1 if any rule is violated
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

queue_health() {
  local queue="$1" now="$2" max_age="$3"
  awk -F'\t' -v now="$now" -v max_age="$max_age" '
    $0 ~ /^[[:space:]]*(#|$)/ {next}
    !($1 ~ /^[0-9]+$/ && NF > 1) {malformed++; next}
    {total++; if ($1 >= now-max_age) live++; else stale++}
    END {printf "%d\t%d\t%d\t%d\n", live+0, stale+0, malformed+0, total+0}
  ' "$queue" 2>/dev/null
}

classify_pool_status() {
  local status_file="$1" sim_root="$2" now="$3" max_age="$4"
  local f1 f2 f3 f4 extra epoch rc cmd payload age out day_epoch
  [ -f "$status_file" ] || return 0
  while IFS=$'\t' read -r f1 f2 f3 f4 extra; do
    epoch=""; rc=""; cmd=""
    if [ "$f1" = "v2" ]; then
      epoch="$f2"; rc="$f3"; payload="$f4"
      [ -z "$extra" ] || continue
      [[ "$epoch" =~ ^[0-9]+$ && "$rc" =~ ^[0-9]+$ && -n "$payload" ]] || continue
      cmd=$(printf '%s' "$payload" | base64 -d 2>/dev/null) || continue
    else
      # Legacy records had no date and could contain raw newlines. Accept only
      # structurally valid, recent rows during migration; malformed fragments
      # cannot be trusted and are ignored.
      rc="$f1"
      [[ "$rc" =~ ^[0-9]+$ && "$f2" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ && -n "$f3" ]] || continue
      cmd="$f3${f4:+$'\t'$f4}${extra:+$'\t'$extra}"
      day_epoch=$(date -d "$(date -d "@$now" +%F) $f2" +%s 2>/dev/null) || continue
      [ "$day_epoch" -le "$now" ] || day_epoch=$((day_epoch - 86400))
      epoch="$day_epoch"
    fi
    age=$((now - epoch))
    [ "$age" -ge 0 ] && [ "$age" -le "$max_age" ] || continue
    [ "$rc" -ne 0 ] || continue
    out=$(printf '%s' "$cmd" | grep -oE '\-\-out +[^ ]+' | awk '{print $2}' | head -1)
    if [ -n "$out" ] && [ -f "$sim_root/$out" ]; then
      printf 'V\t%s\t%s\n' "$rc" "$out"
    else
      printf 'C\t%s\t%s\n' "$rc" "$(printf '%s' "$cmd" | tr '\n\t' '  ' | cut -c1-90)"
    fi
  done < "$status_file"
}

if [ "${1:-}" = "--queue-health" ]; then
  [ "$#" -eq 4 ] || { echo "usage: $0 --queue-health <queue> <now-epoch> <max-age>" >&2; exit 2; }
  queue_health "$2" "$3" "$4"
  exit $?
fi
if [ "${1:-}" = "--classify-pool-status" ]; then
  [ "$#" -eq 5 ] || { echo "usage: $0 --classify-pool-status <log> <sim-root> <now-epoch> <max-age>" >&2; exit 2; }
  classify_pool_status "$2" "$3" "$4" "$5"
  exit $?
fi

cd "$ROOT" || exit 0
FAIL=0

echo "════ 1. PARALLELISM — is the machine actually being used? ════"
CORES=$(nproc); LOAD=$(cut -d' ' -f1 /proc/loadavg); # `pgrep -c` prints "0" AND exits 1 on no match, so `|| echo 0` appended a SECOND line and PROCS became
# "0\n0" -- which breaks the integer test below. Take the first line and default it instead.
PROCS=$(pgrep -fc 'research\.runners' 2>/dev/null | head -1); PROCS=${PROCS:-0}
IDLE=$(awk -v c="$CORES" -v l="$LOAD" 'BEGIN{printf "%d", c-l}')
echo "  cores=$CORES  load=$LOAD  research-procs=$PROCS  => ~$IDLE idle cores"
# CONTENTION WINDOW: the owner sometimes games on this box (no reboot, runs keep going). During that window a
# job ramp is against their interest, and firing this alarm every ~15 min for hours is the exact false-alarm
# corrosion rule 8 warns about -- it trains the reader to scroll past the alert, which is how a REAL
# under-parallelisation slips through later. So the rule is SUSPENDED, loudly and with an expiry, not deleted.
# Stamp: research/findings/raw/gap4/CONTENTION_WINDOW.txt (auto-expires; delete the file to end it early).
CONT="$ROOT/research/findings/raw/gap4/CONTENTION_WINDOW.txt"
CONT_ACTIVE=0
if [ -f "$CONT" ]; then
  CAGE=$(( $(date +%s) - $(stat -c%Y "$CONT" 2>/dev/null || date +%s) ))
  if [ "$CAGE" -lt 14400 ]; then       # 4h cap: a stale stamp must not silently disable the rule forever
    CONT_ACTIVE=1
    echo "  ⏸  parallelism rule SUSPENDED — owner contention window active ($((CAGE/60)) min in, expires at 240 min)."
    echo "     Existing jobs keep running; do NOT ramp up. Timings measured now are UPPER BOUNDS."
  else
    echo "  ⚠️  contention stamp is $((CAGE/3600))h old (>4h cap) — treating it as STALE and enforcing normally."
  fi
fi
if [ "$CONT_ACTIVE" -eq 0 ] && [ "$IDLE" -gt 6 ] && [ "$PROCS" -lt 8 ]; then
  echo "  ⛔ UNDER-PARALLELISED: >6 cores idle with <8 jobs running."
  echo "     A sweep with independent axes (seeds x params) must be launched as a GRID, not walked one cell"
  echo "     at a time. One session ran ~30 probes serially on one core with 15 idle."
  FAIL=1
fi

echo
echo "════ 2. UNSERVED LANES — with the EXACT command to serve each ════"
# map: lane -> a runner that is READY TO RUN NOW (exists, takes --seeds)
declare -A LANE_RUNNER=(
  ["A · Affect"]="_affect_state_region_derisk"
  ["B · Curiosity"]="_curiosity_seek_learn_onbridge_derisk"
  ["C · Self/Workspace"]="_self_schema_region_derisk"
  ["D · Perception"]="_b1_v1_selforg_onbridge_derisk"
  ["E · Language"]="_grounded_lang_p2_derisk"
)
# NB: match on the MODULE INVOCATION ("-m research.runners.X"), never a bare runner name -- a bare name
# matches the checking shell's OWN command line. That self-match bug appeared SEVEN times in one session
# (six of them killing my own shell via pkill). Never grep a pattern that your own command line contains.
RUNNING=$(ps -eo args 2>/dev/null | grep -oE '\-m research\.runners\.[._a-zA-Z0-9]+' | sed 's/.*runners\.//' | sort -u)
UNSERVED=0
for lane in "${!LANE_RUNNER[@]}"; do
  r="${LANE_RUNNER[$lane]}"
  # Match the RUNNER NAME we already hold in LANE_RUNNER. The first version derived a key from the LANE
  # LABEL instead ("D · Perception" -> "percep"), which appears in NO runner filename -- so four lanes that
  # were genuinely running were all reported idle. Do not re-derive a key you already have exactly.
  # SERVED = running NOW *or* produced a result in the last 2h. The first version asked only "is a process
  # running", but these lane runners finish in 2-15 SECONDS -- so a lane correctly served minutes ago reported
  # IDLE forever, and the alarm cried wolf on work that was already done. "Recently produced a result" is the
  # question that actually matters.
  RECENT=$(find research/findings/raw -maxdepth 1 -newermt '-2 hours' -name '*.json' 2>/dev/null | grep -c "$(echo "$r" | sed 's/^_//; s/_derisk$//; s/_probe$//' | cut -d_ -f1-2)" | head -1)
  RECENT=${RECENT:-0}
  if echo "$RUNNING" | grep -qx "$r" || [ "$RECENT" -gt 0 ]; then
    printf "  ✔ %-22s served%s\n" "$lane" "$([ "$RECENT" -gt 0 ] && echo " (result in last 2h)")"
  else
    UNSERVED=$((UNSERVED+1))
    # A banked GO/verdict artifact means the DE-RISK is done and the lane's next step is INTEGRATION (the
    # roadmap's "wire into the develop-loop teacher hook"), not another identical run. Demanding a re-run of
    # completed work is churn, and an alarm that demands churn is one you learn to ignore.
    # MATCH BY FILENAME, not by a "verdict" key (fixed 2026-07-30). The key-based version required BOTH a
    # '"verdict"' field AND the runner name inside the file CONTENT -- but only some runners write a verdict
    # field, so the check's correctness depended on an unrelated convention. It silently missed lane B and
    # demanded a re-run of a completed 6-seed GO, i.e. it generated exactly the churn its own comment warns
    # about. Runners name their artifact after themselves, so the filename is the reliable key; the old content
    # grep is kept only as a fallback.
    STEM=$(echo "$r" | sed 's/_derisk$//')
    BANKED=$(ls -1 research/findings/raw/${STEM}*.json 2>/dev/null | head -1)
    [ -z "$BANKED" ] && BANKED=$(grep -rl '"verdict"' research/findings/raw/*.json 2>/dev/null | xargs -r grep -l "$(echo "$r" | sed 's/^_//; s/_derisk$//' | cut -d_ -f1-2)" 2>/dev/null | head -1)
    if [ -n "$BANKED" ]; then
      printf "  ◐ %-22s de-risk BANKED (%s) — next step is INTEGRATION, not a re-run\n" "$lane" "$(basename "$BANKED")"
      # A banked artifact is only as good as the backend it ran on. Found the hard way the same day: lane B's
      # "6/6 GO" carried smoke=False but backend=numpy, against its OWN runner docstring ("then GPU 6-seed,
      # SIM_BACKEND=cupy"). numpy is for tiny smoke; a decisive multi-seed arm belongs on cupy.
      if grep -q '"backend": *"numpy"' "$BANKED" 2>/dev/null; then
        echo "     ⚠️  that artifact ran on the NUMPY backend — decisive multi-seed arms belong on cupy."
        echo "        Re-run with SIM_BACKEND=cupy before citing it as a GO."
      fi
      UNSERVED=$((UNSERVED-1))
    elif [ -f "research/runners/$r.py" ]; then
      printf "  ⛔ %-22s IDLE — run:  .venv/bin/python -m research.runners.%s --seeds 42 43 44 100 101 102 &\n" "$lane" "$r"
    else
      printf "  ⛔ %-22s IDLE — no ready runner (%s missing) => this lane needs a BUILD, not a run\n" "$lane" "$r"
    fi
  fi
done
[ "$UNSERVED" -ge 3 ] && { echo "  ⛔ $UNSERVED of 5 CPU-capable lanes idle. They are DISJOINT from GPU work and cost nothing beside it."; FAIL=1; }

echo
echo "════ 3. RESEARCH / PRIMARY SOURCES — was a source read before the last finding? ════"
MARK=research/.last_research_gate
EXT=research/.last_external_search
# An EXTERNAL search is a legitimate source check -- indeed it is the ESCALATION rule 3 itself prescribes when
# the local corpus comes up empty. Treat the newer of the two markers as "the last source check", or the rule
# punishes the exact behaviour it demanded (and a rule that cries wolf on correct action gets ignored).
[ -f "$EXT" ] && [ "$EXT" -nt "$MARK" ] && MARK="$EXT"
# SEAM FIX (2026-08-01): `before_you_build.sh` is the SANCTIONED corpus-first source check -- the
# `corpus_check_required` GATE already treats its log as valid and the door stamps `corpus_check_fresh` from
# it. But this heartbeat rule only ever looked at research_gate.sh's marker, so a whole session of running
# before_you_build.sh (the mandated FIRST move) still read "no source check since the last finding" every
# 15-min cycle -- two source-check tools, one recognizer. Honour the corpus-check log as a third marker.
CORPUS=research/queue/.corpus_checks.jsonl
[ -f "$CORPUS" ] && [ "$CORPUS" -nt "$MARK" ] && MARK="$CORPUS"
# SCOPE: a source check is owed for a BIOLOGICAL claim, not for a finding about our own tooling. On
# 2026-07-31 this fired every 15-min cycle for five hours because the day produced audit/tooling findings
# (gate defects, the sprawl measurement, the walls synthesis) with no primary-source question --
# "run research_gate.sh on why my gate had a bug" is meaningless. A correct rule, mis-scoped, trains the
# reader to skim the block that ALSO carries the real alarms. Findings declaring lane: audit|tooling|
# workflow are skipped; every other finding still owes a source check.
# MTIME IS NOT AUTHORSHIP DATE. The Tier-1 status classification touched 274 LEGACY findings today to add
# frontmatter, so `ls -t` reported a 2026-05 finding as the newest and demanded a fresh source check for
# a document written eleven weeks ago. Sort by the DATE IN THE FILENAME instead, which is what the corpus
# actually encodes, and compare only TODAY's findings against the marker.
TODAY=$(date +%Y-%m-%d)
NEWEST=$(for f in $(ls research/findings/${TODAY}-*.md 2>/dev/null); do
  head -12 "$f" | grep -qE "^lane: *(audit|tooling|workflow)" || { echo "$f"; break; }; done)
if [ ! -f "$MARK" ]; then
  echo "  ⛔ tools/research_gate.sh has NEVER run (no marker). Our findings cite sources in ONE LINE;"
  echo "     that is not reading them. A whole session on place cells never opened O'Keefe-Nadel — when"
  echo "     finally read it produced a mechanism, a confirmed prediction, and TWO corrections."
  echo "     RUN:  bash tools/research_gate.sh \"<your current question>\""
  FAIL=1
elif [ -n "$NEWEST" ] && [ "$NEWEST" -nt "$MARK" ]; then
  echo "  ⛔ A FINDING WAS WRITTEN SINCE THE LAST SOURCE CHECK:"
  echo "     newest finding : $(basename "$NEWEST")"
  echo "     last gate run  : $(date -r "$MARK" '+%Y-%m-%d %H:%M')"
  echo "     RUN:  bash tools/research_gate.sh \"<the question that finding was about>\""
  FAIL=1

else
  echo "  ✔ research_gate.sh ran $(date -r "$MARK" '+%H:%M') — newer than the latest finding."
fi
# EVALUATED UNCONDITIONALLY. As an `elif` this sat after the freshness branch, so a fresh marker
# short-circuited it and rule 3 reported PASS on a search that found ZERO primary sources -- a check that
# LIES, which is worse than no check. A gate that ran and found nothing is a WARNING, never a pass.
EMPTY=research/.last_research_gate_empty
if [ -f "$EMPTY" ] && [ ! "$MARK" -nt "$EMPTY" ] && { [ ! -f "$EXT" ] || [ ! "$EXT" -nt "$EMPTY" ]; }; then
  echo "  ⛔ the last source check RAN but found NO primary source for that question."
  echo "     Informative, not a pass: either our corpus lacks it -> GO EXTERNAL (WebSearch / bio-research MCP),"
  echo "     or the query was wrong. Do NOT build on our own findings alone -- that is the exact failure mode."
  FAIL=1
fi
echo "     canonical sources (single-column, grep clean): ~/Projects/sim-catalog/references/textbooks/<name>/*.txt"
echo "     ⚠️ Kandel's copy is TWO-COLUMN with hyphen-splits: anchor on a short fragment, read a window."

echo
echo
echo "════ 4. CLUSTER — are the mini-PC pool's 36 cores actually working? ════"
# WHY THIS RULE EXISTS (2026-07-30, owner-flagged): rule 1 reads nproc and /proc/loadavg -- LOCAL ONLY. So it
# printed "parallelism satisfied" all evening while three 12-core pool nodes sat at load 0.00. The check was
# STRUCTURALLY INCAPABLE of seeing them, and lane_check.py's prose even referenced "36 idle pool cores" while
# never probing a node, which turned the one mention of the pool into false assurance.
# Compounding it, I then declared the pool OFFLINE after scanning the WRONG SUBNET (192.168.1.x) as the WRONG
# USER (dant123) -- while ~/.ssh/config had working pool40/41/42 aliases (User node) the whole time. A capacity
# claim was made from a failed probe instead of a working one. So this rule uses the ALIASES, and it
# distinguishes IDLE (actionable) from UNREACHABLE (report, do not cry wolf).
# SELF-MATCH, reintroduced by me in the very rule meant to fix a blindness bug (caught 2026-07-30 by testing
# the FAILING direction). `pgrep -fc 'research.runners'` matches the full command line of every process --
# INCLUDING the ssh command carrying that pattern -- so it always returned >=1 and the node always read "busy".
# The rule was structurally incapable of ever firing, exactly like the research gate was structurally
# unsatisfiable. This file's own section-2 comment already warns: "never grep a pattern that your own command
# line contains". Bracket the first char: the regex [r]esearch matches "research", but the literal text
# "[r]esearch" in our own argv does not match it.
POOL_IDLE=0; POOL_UP=0; POOL_DOWN=0; POOL_LINES=""
for H in pool40 pool41 pool42; do
  R=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "$H" \
        "echo \$(nproc) \$(cut -d' ' -f1 /proc/loadavg) \$(pgrep -fc '[r]esearch\\.runners' 2>/dev/null || echo 0)" 2>/dev/null)
  if [ -z "$R" ]; then
    POOL_DOWN=$((POOL_DOWN+1)); POOL_LINES="$POOL_LINES  $(printf '%-8s' "$H") unreachable\n"; continue
  fi
  POOL_UP=$((POOL_UP+1))
  set -- $R; C="${1:-0}"; LD="${2:-0}"; PR="${3:-0}"
  # idle = load below a quarter of its cores AND no runners
  if awk -v l="$LD" -v c="$C" 'BEGIN{exit !(l < c/4)}' && [ "${PR:-0}" -eq 0 ]; then
    POOL_IDLE=$((POOL_IDLE+1)); POOL_LINES="$POOL_LINES  $(printf '%-8s' "$H") ⛔ IDLE  cores=$C load=$LD runners=$PR\n"
  else
    POOL_LINES="$POOL_LINES  $(printf '%-8s' "$H") ✔ busy  cores=$C load=$LD runners=$PR\n"
  fi
done
printf "%b" "$POOL_LINES"
# QUEUE DEPTH is the real check (2026-07-31). "Pool idle" is a SYMPTOM; the defect is "nothing staged", and
# alarming on the symptom caps utilisation at my reaction time -- it fired seven times in one session, each time
# work that could have been queued in advance. Also verify the dispatcher is LIVE: a full queue with a dead
# dispatcher is the same idle pool, and that is exactly what happened after I killed it for a cleanup and did
# not restart it.
# `grep -c` prints "0" AND exits 1 on no match, so `$(... || echo 0)` yields a TWO-LINE "0\n0" and the integer
# test below silently errors instead of firing. This is the SAME defect I fixed for pgrep earlier in this file
# tonight and then reintroduced here verbatim -- take the first line and default it.
QUEUE_PATH="${POOL_QUEUE_PATH:-$ROOT/research/queue/pool.queue}"
IFS=$'\t' read -r QDEPTH QSTALE QMALFORMED QTOTAL <<< "$(queue_health "$QUEUE_PATH" "$(date +%s)" "${POOL_JOB_MAX_AGE:-43200}")"
QDEPTH=${QDEPTH:-0}; QSTALE=${QSTALE:-0}; QMALFORMED=${QMALFORMED:-0}; QTOTAL=${QTOTAL:-0}
DISPATCH=$(pgrep -fc '[p]ool_autodispatch' 2>/dev/null | head -1); DISPATCH=${DISPATCH:-0}
echo "  queue depth=$QDEPTH live, $QSTALE stale  dispatcher=$([ "${DISPATCH:-0}" -gt 0 ] && echo LIVE || echo DEAD)"
if [ "$QSTALE" -gt 0 ]; then
  echo "  · ignoring $QSTALE queue record(s) older than $(( ${POOL_JOB_MAX_AGE:-43200} / 3600 ))h; dispatcher will not launch them."
fi
if [ "${QMALFORMED:-0}" -gt 0 ]; then
  echo "  ⛔ $QMALFORMED MALFORMED POOL QUEUE RECORD(S) — monitoring will not count unexecutable work."
  echo "     They will be preserved in pool.queue.malformed; requeue via tools/pool_queue.sh."
  FAIL=1
fi
if [ "${QDEPTH:-0}" -eq 0 ]; then
  echo "  ⛔ POOL QUEUE EMPTY — nothing is staged, so the next free node will idle by default."
  echo "     Stage work:  bash tools/pool_queue.sh add '<command run from ~/derisk-pool/sim>'"
  FAIL=1
fi
if [ "${DISPATCH:-0}" -eq 0 ] && [ "${QDEPTH:-0}" -gt 0 ]; then
  echo "  ⛔ DISPATCHER DEAD with $QDEPTH job(s) queued — staged work will never launch."
  echo "     nohup bash tools/pool_autodispatch.sh > /tmp/pool_dispatch.log 2>&1 &"
  FAIL=1
fi
# SUPPRESS the idle-node alarm while work is EN ROUTE (2026-07-31). The dispatcher polls every
# POOL_DISPATCH_POLL (default 60 s), so a node that just finished a job is legitimately idle for up to one poll
# interval. Alarming on that is a FALSE POSITIVE, and it fired within minutes of the dispatcher going live:
# "pool41 ⛔ IDLE" at 11:09 while the dispatch log shows pool41 receiving jobs at 11:08:32 AND 11:09:38.
# A false alarm is as corrosive as a missed one -- it trains the reader to skim past the whole block, which is how
# a real failure slips through. The genuine defect is "nothing staged" (queue empty) or "nothing will launch"
# (dispatcher dead); BOTH are already checked above and BOTH still fire. Idle-with-work-queued-and-a-live-
# dispatcher is a transient, not a defect.
if [ "$POOL_IDLE" -gt 0 ] && [ "${QDEPTH:-0}" -gt 0 ] && [ "${DISPATCH:-0}" -gt 0 ]; then
  echo "  · $POOL_IDLE node(s) idle but $QDEPTH job(s) queued and the dispatcher is LIVE — work is en route"
  echo "    (dispatch poll is ${POOL_DISPATCH_POLL:-60}s); not flagging."
  POOL_IDLE=0
fi
if [ "$POOL_IDLE" -gt 0 ]; then
  echo "  ⛔ $POOL_IDLE of $POOL_UP reachable pool node(s) IDLE — that is $((POOL_IDLE*12)) cores doing nothing."
  echo "     They are DISJOINT from the GPU, so they cost nothing to use while the crux runs."
  echo "     The pool copy is an rsync'd tree, NOT a git repo: 'git pull' fails there and it silently predates"
  echo "     any new runner. scp the runner first, then dispatch:"
  echo "       scp research/runners/<runner>.py pool40:~/derisk-pool/sim/research/runners/"
  echo "       ssh -f -n pool40 \"cd ~/derisk-pool/sim && setsid bash <script>.sh </dev/null >out 2>&1 & exit 0\""
  FAIL=1
elif [ "$POOL_UP" -eq 0 ]; then
  echo "  ⚠️  no pool node reachable — report it, do NOT treat 36 cores as available capacity."
else
  echo "  ✔ pool working ($POOL_UP up, $POOL_DOWN unreachable)"
fi

# FAILED POOL JOBS (2026-07-31). The dispatcher records each job's exit status; a non-zero one means compute was
# spent producing nothing, and previously the ONLY evidence was a file on a node nobody reads. Nine jobs died on
# argparse and went unnoticed for an hour. Absence of results is not evidence of failure -- a job can also still
# be running -- so this reads the recorded RC rather than inferring from missing output.
# A non-zero RC alone is NOT "nothing produced" (2026-08-01): the project's runners `return 0 if GO else 1`, so
# rc=1 is an HONEST-NEGATIVE VERDICT that still WROTE its --out artifact (e.g. n_prop=5 floors the oracle and
# correctly exits 1). Flagging every rc!=0 cried wolf on real results. The reliable disambiguator is the
# ARTIFACT: a job whose --out exists on the node PRODUCED a result (verdict) and is not lost compute; only a
# job whose --out is MISSING truly spent compute for nothing (a crash / argparse / module-not-found).
CRASHED=""; VERDICTS=""
for H in pool40 pool41 pool42; do
  R=$({
    declare -f classify_pool_status
    printf '\nclassify_pool_status "$HOME/derisk-pool/sim/job_status.log" "$HOME/derisk-pool/sim" %q %q\n' \
      "$(date +%s)" "${POOL_STATUS_MAX_AGE:-3600}"
  } | timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$H" 'bash -s' 2>/dev/null)
  while IFS=$'\t' read -r kind rc rest; do
    [ -z "$kind" ] && continue
    [ "$kind" = "C" ] && CRASHED="$CRASHED\n  $H rc=$rc: $rest"
    [ "$kind" = "V" ] && VERDICTS="$VERDICTS\n  $H rc=$rc: $rest"
  done <<< "$(printf '%b' "$R")"
done
if [ -n "$CRASHED" ]; then
  echo "  ⛔ POOL JOB(S) CRASHED — compute spent, NO artifact written (argparse / module-not-found / killed):"
  printf "%b\n" "$CRASHED"
  echo "     Read the node's autodispatch.out, fix, and requeue via tools/pool_queue.sh."
  FAIL=1
fi
[ -n "$VERDICTS" ] && { echo "  · pool job(s) exited non-zero but WROTE their artifact (an honest NO-GO exits 1; not lost compute — verify the verdict):"; printf "%b\n" "$VERDICTS"; }

if [ "$FAIL" -eq 0 ]; then echo "✅ workflow_check: all four rules satisfied."; else
  echo "⛔ workflow_check: $FAIL rule-group(s) violated — the commands above are copy-paste ready."; fi
exit $FAIL
