#!/bin/bash
# Repo copy of /opt/data/scripts/sim-driver.sh in the hermes container (install: tools/hermes/INSTALL.md).
# 2026-11-02: only the per-turn message changed (lean method: CLAUDE.md, HANDOFF.md, lean runbook, one task).
# sim-driver (owner 2026-09-30): the neural-simulator driver as ONE continuing Hermes session ("neural-simulator"),
# not a fresh session per run. Hermes cron runs this every 5 min (no-agent job). sim-driver-gate.py decides whether
# to wake (event-driven since 2026-10-02: new results, a provisioning result, main moved by someone else, an
# unfinished turn, a stall, or a 2-hour heartbeat; see its header); if so, one more turn is sent into the named session
# with `hermes chat -c neural-simulator`, and at the end main and the turn's outcome go to .sim-driver-after.json for
# the gate. Hermes's own turn lease serialises it with anything the owner types into that
# session in the dashboard; Hermes's compression keeps it within context (each compaction continues in a linked
# child session that -c follows). Stdout is what gets delivered: nothing when the turn ends in [SILENT].
set -uo pipefail
SESSION=neural-simulator
WS=/home/dant123/Projects/neural-simulator
S=/opt/data/scripts
gate=$(/opt/hermes/.venv/bin/python "$S/sim-driver-gate.py" 2>>"$S/.sim-driver-gate.log")
reason=$(printf '%s\n' "$gate" | grep '^Woken:' | head -1)
[ -n "$reason" ] || exit 0          # nothing to do (or no answer from the gate): silent, no model call

MSG="$S/.sim-driver-msg.txt"
cat > "$MSG" <<MSGEOF
$reason ($(date '+%Y-%m-%d %H:%M %Z'))

You are the Hermes driver of the neural-simulator project, continuing in this one session. Work only in $WS
(never /workspace). Every turn, re-read (they change): CLAUDE.md (= AGENTS.md), HANDOFF.md, and in
research/coordination/LOCAL_LLM_RUNBOOK.md the sections "1. Every turn", "2. Hard rules", "3. Standing operations",
"4. Roles and how to pick work" and "6. RULINGS", plus only its last lines (tail -n 25). Then do exactly ONE task, picked
as section 4 says (a new RULING for you, else HANDOFF.md if its DRIVER is hermes, else your current list; nothing
ready: end with [SILENT]). Before building, probing or testing anything, run bash tools/before_you_build.sh "<question>"
and read its top hits. Follow the skill file for the task type, read as a plain file: .claude/skills/before-build/SKILL.md
(build, probe, test), .claude/skills/land-result/SKILL.md (commit or report a result), .claude/skills/ops/SKILL.md
(pool, GPU, server, landing data). Progress means the new core's milestones in brain/ARCHITECTURE.md section 4; the
old track is frozen except the Stage-2 retake (RULING pivot-1102). Timebox: 2 days per problem, then a NEEDS STOP.
Commit through the gates (never --no-verify) and write the WORKLOG line. Never push: sim-sync publishes main. Never
run research jobs on this server: anything over ~10 minutes goes to the pool or the workstation (RULING
compute-placement); never relaunch a job Claude moved or stopped. If the task needs an action section 2 reserves for Claude (merge, gates, preregs, scorers, bars,
flipping a default), add a NEEDS-CLAUDE line (python3 tools/needs_line.py ... --write) and stop. Other judgment goes to
one MiMo Pro delegate as RULING models says.
Your reply reaches the owner's phone only if its first line starts with NOTICE: -- use that only when the owner
should hear about it now (you added a NEEDS-CLAUDE line or anything needs the owner, a headline result landed such as
a GO/NO-GO or a stage exit, or you are blocked), then at most 5 plain-language lines on what is needed or what
changed. Otherwise end with exactly [SILENT]. You are woken by events (new results, a provisioning result, new
commits from others) or else every 2 hours; the first line above says why. If you stop mid-task (time or turn limit)
and want to resume in about 30 minutes instead, make the last line [CONTINUE] (after [SILENT] or the NOTICE lines).
MSGEOF

OUT="$S/.sim-driver-last.jsonl"
cd "$WS" || { echo "sim-driver: workspace missing"; exit 1; }
HERMES_CRON_SESSION=1 timeout 3300 /opt/hermes/.venv/bin/hermes chat -c "$SESSION" --create-if-missing \
  --query-file "$MSG" --oneshot -Q --format stream-json --run-budget 3000 --max-turns 200 \
  -t terminal,file,code_execution,todo,memory,session_search,skills,delegation,web > "$OUT" 2>&1
rc=$?
main=$(git -c safe.directory="$WS" -C "$WS" rev-parse -q --verify refs/heads/main 2>/dev/null)
/opt/hermes/.venv/bin/python - "$OUT" "$rc" "$main" "$S/.sim-driver-after.json" <<'PY'
import json, os, sys, time
out, rc, main, after = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
res = None
for ln in open(out, errors="replace"):
    try:
        ev = json.loads(ln)
    except ValueError:
        continue
    if ev.get("type") == "result":
        res = ev
text = ((res or {}).get("text") or "").strip()
lines = [l for l in text.splitlines() if l.strip()]
cont = bool(lines) and lines[-1].strip("*_ `").upper() == "[CONTINUE]"
with open(after + ".tmp", "w") as fh:   # for sim-driver-gate.py: main after this turn, and whether to resume soon
    json.dump({"end": int(time.time()), "rc": rc, "finished": res is not None and rc == 0, "continue": cont,
               "main": main}, fh)
os.replace(after + ".tmp", after)
if cont:
    lines = lines[:-1]
    text = "\n".join(lines)
if res is None or rc not in (0,):
    print("sim-driver: the turn did not finish cleanly (exit %s); last output in /opt/data/scripts/.sim-driver-last.jsonl" % rc)
    sys.exit(1)
# Deliver only an explicit notice: a turn that ends in narration (the local model sometimes stops mid-thought) or
# [SILENT] stays in the session and off the owner's phone.
if lines and lines[0].lstrip("*_# >").upper().startswith("NOTICE:"):
    print(text)
PY
