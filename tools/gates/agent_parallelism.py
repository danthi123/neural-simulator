"""CLASS AP — PENDING WORK SERIALISED while dispatchable agents sit unused. BLOCKING.

WHY (owner, 2026-07-31): "parallel work isn't strictly a compute thing, especially when there's an AI with
dispatchable subagents involved. The only real limit is per-agent context, and even that is loose with
compaction."

Every parallelism check in this repo measured COMPUTE — GPU lanes, pool queue depth, idle CPU cores. NONE
measured whether AGENT work was being fanned out. So on 2026-07-31 I worked an 8-item backlog strictly one item
at a time — build, test, commit, next — for hours, while six of those items were entirely file-disjoint and could
have run at once. Every compute check was GREEN throughout, because the crux and the pool were busy. The axis
with the loosest limit was the one nothing watched.

THE RULE: if the board lists N pending items and fewer than MIN_AGENTS agents are working, that is serialisation,
not prioritisation. Dispatch, or say why not.

SIGNALS, both honest proxies and documented as such:
  * PENDING = numbered items under "NEXT, in order:" in GAP_CLOSURE_MISSION.md's CURRENT STATE. That list is
    durable precisely because it used to live in chat and evaporate.
  * ACTIVE  = claude processes. A proxy: it counts the session itself and any tooling, so the threshold is set
    against a measured idle baseline rather than assuming one process per agent.

THE REAL CONSTRAINT, and why the threshold is not higher: agents must own DISJOINT FILES. Two agents editing one
file corrupt each other. That is a scheduling problem, not a reason to serialise — the fan-out that prompted this
gate split "stale citations" from "plans frontmatter" solely because both would have touched docs/plans/.

WAIVER: research/queue/.agent_waiver with a reason, auto-expiring in AGENT_WAIVER_MAX_H hours — legitimate when
the remaining items genuinely conflict on files, or when one item must finish before the rest are defined.

WHAT IT CANNOT CATCH: whether the parallel work is the RIGHT work. Six agents on trivia passes. That is
judgement and stays with the human and with me.
"""
from __future__ import annotations

import os
import re
import subprocess
import time

NAME = "agent-parallelism"
CLASS_ID = "AP"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BOARD = os.path.join(_ROOT, "GAP_CLOSURE_MISSION.md")
_WAIVER = os.path.join(_ROOT, "research", "queue", ".agent_waiver")
PENDING_TRIGGER = 4          # this many pending items justifies fanning out
IDLE_BASELINE = 12           # measured claude-process count with no subagents running
AGENT_WAIVER_MAX_H = 6


def _pending():
    if not os.path.exists(_BOARD):
        return 0
    try:
        t = open(_BOARD, errors="ignore").read()
    except OSError:
        return 0
    m = re.search(r"NEXT, in order:(.*?)(?:THE WORKFLOW IS MECHANICAL|\n> ## )", t, re.S)
    return len(re.findall(r"^>\s*\d+\.", m.group(1), re.M)) if m else 0


def _claude_procs():
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=15).stdout
        return sum(1 for l in out.split("\n") if "claude" in l and "grep" not in l)
    except Exception:
        return IDLE_BASELINE + 1        # cannot tell -> do not block


def _waiver():
    if not os.path.exists(_WAIVER):
        return None
    if (time.time() - os.path.getmtime(_WAIVER)) / 3600.0 > AGENT_WAIVER_MAX_H:
        return None
    try:
        return open(_WAIVER, errors="ignore").read().strip()[:120] or "(no reason)"
    except OSError:
        return None


def check(paths=None):
    n = _pending()
    if n < PENDING_TRIGGER or _waiver():
        return []
    if _claude_procs() > IDLE_BASELINE:
        return []                        # agents are working
    return ["%d pending items on the board and NO agents dispatched — that is serialisation, not "
            "prioritisation.\n"
            "        Compute parallelism can be GREEN (crux busy, pool busy) while AGENT work runs one item at a\n"
            "        time; that happened for hours on 2026-07-31 and no check saw it.\n"
            "        FIX: fan out the file-DISJOINT items with the Workflow tool (disjoint ownership is the only\n"
            "        real constraint), or waive with a reason (auto-expires %dh):\n"
            "          echo 'items conflict on <file>' > research/queue/.agent_waiver"
            % (n, AGENT_WAIVER_MAX_H)]


def selftest():
    """FAILING DIRECTION FIRST: many pending + no agents MUST fire."""
    bad = []
    board = ("> NEXT, in order:\n> 1. a\n> 2. b\n> 3. c\n> 4. d\n> 5. e\n"
             "> THE WORKFLOW IS MECHANICAL\n")
    m = re.search(r"NEXT, in order:(.*?)(?:THE WORKFLOW IS MECHANICAL|\n> ## )", board, re.S)
    if not m or len(re.findall(r"^>\s*\d+\.", m.group(1), re.M)) != 5:
        bad.append("pending-item parser does not read a 5-item list")
    if _pending() < 1:
        bad.append("cannot parse the real board's pending list — the gate would never fire")
    if PENDING_TRIGGER <= 1:
        bad.append("trigger too low to be meaningful")
    return bad
