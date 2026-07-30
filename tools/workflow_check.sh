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
ROOT=/home/dant123/Projects/sim
cd "$ROOT" || exit 0
FAIL=0

echo "════ 1. PARALLELISM — is the machine actually being used? ════"
CORES=$(nproc); LOAD=$(cut -d' ' -f1 /proc/loadavg); PROCS=$(pgrep -fc 'research\.runners' 2>/dev/null || echo 0)
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
    BANKED=$(grep -rl '"verdict"' research/findings/raw/*.json 2>/dev/null | xargs -r grep -l "$(echo "$r" | sed 's/^_//; s/_derisk$//' | cut -d_ -f1-2)" 2>/dev/null | head -1)
    if [ -n "$BANKED" ]; then
      printf "  ◐ %-22s de-risk BANKED (%s) — next step is INTEGRATION, not a re-run\n" "$lane" "$(basename "$BANKED")"
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
NEWEST=$(ls -t research/findings/*.md 2>/dev/null | head -1)
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
if [ "$FAIL" -eq 0 ]; then echo "✅ workflow_check: all three rules satisfied."; else
  echo "⛔ workflow_check: $FAIL rule-group(s) violated — the commands above are copy-paste ready."; fi
exit $FAIL
