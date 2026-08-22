#!/usr/bin/env python3
"""LIVE-STATE generator — the anti-compaction-loss + anti-dilution layer (spec §5).

WHY THIS EXISTS (owner, 2026-08-21). Two root failure modes, both mechanical:
  1. COMPACTION drops load-bearing info (the summariser cannot know what still matters).
  2. A FULL CONTEXT dilutes attention, so in-context info stops driving behaviour.
Heartbeats INFORM but do not ENFORCE. The fix is a small, capped file that is mechanically
RE-INJECTED into context at turn-start AND immediately post-compaction (re-read from the FILE,
never from the lossy summary) — so the frontier, the last decision, the ordered next-actions, the
live runs, and the hard constraints are always present at the *end* of context, where attention is
strongest, regardless of what compaction dropped or how full the window is.

THE CAP IS THE POINT. An over-cap live-state file defeats its own purpose: it becomes just more
dilution. So this generator TRUNCATES/PRIORITISES to stay under `CAP_BYTES`, and the companion gate
`tools/gates/live_state_reinjection.py` REJECTS (blocks a commit) an over-cap file — the dilution
guard. Priority order (least-important dropped first): header+gen line and CONSTRAINTS are never
dropped; FRONTIER next; then as many ordered NEXT-ACTIONS as fit; then LIVE RUNS; then LAST
DECISION; then the footer.

DURABLE SOURCES ONLY (never chat context — chat is exactly what compaction destroys):
  - GAP_CLOSURE_MISSION.md  → CURRENT ARC (frontier), the ordered next-actions, the last landing.
  - research/coordination/backlog.json  → the machine backlog's top items, when present (spec §1;
    another agent owns tools/backlog.py — this reads its output if it exists, degrades gracefully if
    not).
  - tools/gpu_queue.sh status + tools/pool_queue.sh depth  → live runs (best-effort, short timeouts;
    a wedged nvidia-smi must never wedge this — see commit e5e0d83).
  - the hard constraints  → a stable compressed constant (they are non-negotiable and do not drift).

HOOK WIRING (see .claude/settings.json + .claude/hooks/live_state_inject.py):
  - SessionStart (all sources incl. `compact`): regenerate + emit to stdout → survives compaction.
  - UserPromptSubmit (every turn): emit the file to stdout → anti-dilution, keeps it at end-of-context.
  - PostToolUse Write|Edit (findings/board/ledger): regenerate (fast) + nudge → stays fresh on landings.

CLI:
  python tools/live_state.py                 # regenerate + write the file, print a summary
  python tools/live_state.py --stdout        # regenerate + print the content (do not write)
  python tools/live_state.py --emit          # read the file (regen if missing) + print (hook fast path)
  python tools/live_state.py --heartbeat     # ONE-LINE delta carrying the top next-action (noise-cut)
  python tools/live_state.py --check         # print byte size vs cap; exit 1 if over cap

Env overrides (used by the hermetic selftest): LIVE_STATE_ROOT (repo root), LIVE_STATE_FAST=1
(skip the gpu/pool subprocess calls — for the PostToolUse fast path and tests).
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import time

# ~1-2 KB. 2048 bytes is the hard ceiling; the generator aims well under it. The gate blocks anything
# over this, because an over-cap live-state is dilution, which is the very thing it exists to prevent.
CAP_BYTES = 2048

LIVE_STATE_REL = "research/coordination/live_state.md"
BOARD_REL = "GAP_CLOSURE_MISSION.md"
BACKLOG_REL = "research/coordination/backlog.json"
MARKER = "⟦LIVE-STATE⟧"   # ⟦LIVE-STATE⟧ — the injected-block sentinel the gate/tests look for

# The non-negotiables. Stable by definition (they are the mission's invariants), so they are a
# constant here rather than parsed — parsing a constant only adds a way for it to silently rot.
CONSTRAINTS = (
    "CONSTRAINTS (non-negotiable): brain-based-only (host=world/body/clock) · ONE brain · "
    "emergent-not-hand-built · NO-DEFER (a wall defers a METHOD, never a CAPABILITY) · "
    "speed<faithfulness · honesty-boundary (never assert phenomenal experience) · 6-seed "
    "(42/43/44/100/101/102) · gates AUTHORITATIVE · commit BOTH remotes via "
    "tools/push_both.sh, NEVER --no-verify · ONE brain-loading GPU proc at a time (ALL "
    "GPU→gpu_queue.sh) · COST-ROUTING: CPU→pool, GPU→gpu_queue, seeds→--seeds, "
    "AGENTS=genuine builds only"
)


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _root() -> str:
    env = os.environ.get("LIVE_STATE_ROOT")
    if env:
        return env
    # tools/live_state.py -> repo root is two dirs up
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(root: str, rel: str) -> str:
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        return ""
    try:
        return io.open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""


# Strip markdown emphasis (**bold**, `code`) but NOT underscores — underscores are literal in the
# paths and identifiers these docs are full of (research/findings/raw/_d5_separation/…,
# consolidate_used_memory, gpu_queue), and removing them corrupts every one.
_MD = re.compile(r"[*`]+")
_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", _MD.sub("", s)).strip()


def _shorten(s: str, n: int) -> str:
    s = _clean(s)
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)].rstrip() + "…"


def _bytes(s: str) -> int:
    return len(s.encode("utf-8"))


# --------------------------------------------------------------------------------------------------
# durable-source parsers
# --------------------------------------------------------------------------------------------------
def frontier(board: str) -> str:
    """The CURRENT ARC line from the latest STATE header (first match from the top = newest)."""
    lines = board.splitlines()
    for i, line in enumerate(lines):
        if "CURRENT ARC" in line:
            # CURRENT ARC is a wrapped paragraph — join continuation lines so the sentence is whole
            joined = line
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith(
                    ("#", "---", "-", "*", ">")) and len(joined) < 400:
                joined += " " + lines[j].strip()
                j += 1
            return _shorten(joined, 240)
    # fallback: the first STATE-OF-THE-PROJECT line's follow-on, or the NORTH-STAR line
    for line in board.splitlines():
        if "NORTH-STAR" in line or "north-star" in line:
            return _shorten(line, 300)
    return "(frontier: see GAP_CLOSURE_MISSION.md STATE OF THE PROJECT)"


def last_decision(board: str) -> str:
    """The most recent landing — a line marked LANDED, from the latest header."""
    for line in board.splitlines():
        if "LANDED" in line and ("commit" in line or "flip" in line.lower() or "GO" in line):
            return _shorten(line, 260)
    for line in board.splitlines():
        if "LANDED" in line:
            return _shorten(line, 260)
    return ""


def _backlog_items(root: str) -> list[str]:
    """Top ordered items from the machine backlog (spec §1), when tools/backlog.py has produced it."""
    raw = _read(root, BACKLOG_REL)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if isinstance(it, dict):
            ident = it.get("id") or it.get("ref") or ""
            what = it.get("what") or it.get("title") or it.get("summary") or it.get("desc") or ""
            lane = it.get("lane") or it.get("cheapest_lane") or ""
            s = (("%s: " % ident) if ident else "") + _clean(str(what))
            if lane:
                s += " [%s]" % _clean(str(lane))
            if s.strip():
                out.append(s)
        elif isinstance(it, str):
            out.append(_clean(it))
    return out


_NUM = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def _board_next_actions(board: str) -> list[str]:
    """The numbered 'PRE-DECIDED NEXT ACTIONS' list, falling back to 'ORDERED OVERNIGHT BACKLOG'."""
    lines = board.splitlines()

    def _numbered_after(heading_key: str) -> list[str]:
        out, i, n = [], 0, len(lines)
        while i < n and heading_key not in lines[i]:
            i += 1
        if i >= n:
            return out
        i += 1
        cur = None
        while i < n:
            ln = lines[i]
            m = _NUM.match(ln)
            if m:
                if cur is not None:
                    out.append(cur)
                cur = m.group(2)
            elif ln.strip() == "" or ln.lstrip().startswith(("#", "---")) or (
                    ln.startswith("**") and cur is not None):
                # blank / new heading / new bold block ends the list
                if cur is not None:
                    out.append(cur)
                    cur = None
                if ln.lstrip().startswith(("#", "---")):
                    break
                if ln.strip() == "":
                    # allow a single trailing continuation gap? no — a blank ends an item cleanly
                    if not any(_NUM.match(lines[j]) for j in range(i + 1, min(i + 3, n))):
                        break
            elif cur is not None:
                cur += " " + ln.strip()               # continuation line of the current item
            i += 1
        if cur is not None:
            out.append(cur)
        return out

    items = _numbered_after("PRE-DECIDED NEXT ACTIONS")
    if items:
        return items
    # fallback: the dashed ORDERED OVERNIGHT BACKLOG list
    out, i, n = [], 0, len(lines)
    while i < n and "ORDERED OVERNIGHT BACKLOG" not in lines[i]:
        i += 1
    i += 1
    while i < n:
        ln = lines[i]
        if ln.lstrip().startswith(("#", "---")):
            break
        if re.match(r"^\s*[-*]\s+", ln):
            out.append(re.sub(r"^\s*[-*]\s+", "", ln))
        i += 1
    return out


def next_actions(root: str, board: str) -> list[str]:
    """Ordered next-actions: the machine backlog if present, else the board's ordered list."""
    items = _backlog_items(root)
    if not items:
        items = _board_next_actions(board)
    return [_shorten(x, 135) for x in items if x.strip()]


def live_runs(root: str, fast: bool = False) -> str:
    """Best-effort live-run summary. NEVER blocks: short timeouts, all failures degrade to a marker.

    A wedged nvidia-smi once hung the gpu_queue dispatcher (commit e5e0d83); this must not let that,
    or a missing tool, or a slow disk, hang a hook — so every call is timeout-guarded and fail-open.
    """
    if fast:
        return ""
    parts = []

    def _run(cmd: list[str], timeout: float) -> str:
        try:
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
            return (r.stdout or r.stderr or "").strip()
        except Exception:
            return ""

    gq = os.path.join(root, "tools", "gpu_queue.sh")
    if os.path.isfile(gq):
        out = _run(["bash", gq, "status"], 6.0)
        if out:
            parts.append("gpu=[" + _shorten(out.replace("\n", " "), 140) + "]")
    pq = os.path.join(root, "tools", "pool_queue.sh")
    if os.path.isfile(pq):
        out = _run(["bash", pq, "depth"], 6.0)
        if out:
            parts.append("pool-depth=" + _shorten(out.replace("\n", " "), 40))
    return " · ".join(parts)


# --------------------------------------------------------------------------------------------------
# assembly (priority-ordered, hard-capped)
# --------------------------------------------------------------------------------------------------
def build(root: str | None = None, fast: bool = False) -> str:
    """Assemble the LIVE-STATE block, dropping the lowest-priority sections until it fits CAP_BYTES."""
    root = root or _root()
    board = _read(root, BOARD_REL)

    gen = time.strftime("%Y-%m-%d %H:%M")
    header = (
        "%s re-injected each turn + post-compaction — AUTHORITATIVE over older/compacted context; "
        "source of truth is %s, re-read the FILE if this looks stale. gen %s · cap %d B"
        % (MARKER, LIVE_STATE_REL, gen, CAP_BYTES)
    )

    fr = frontier(board)
    na = next_actions(root, board)
    lr = live_runs(root, fast=fast)
    ld = last_decision(board)
    footer = ("SOURCES: %s (CURRENT ARC + ordered next-actions + last landing) · %s (machine "
              "backlog, when present) · gpu_queue/pool status." % (BOARD_REL, BACKLOG_REL))

    # mandatory spine — never dropped
    out = [header, "", CONSTRAINTS, "", "FRONTIER: " + fr]

    def size(extra: str = "") -> int:
        return _bytes("\n".join(out) + ("\n" + extra if extra else ""))

    # NEXT ACTIONS — add items while they fit; this is the ordered backlog, the highest-value payload
    if na:
        out += ["", "NEXT ACTIONS (ordered):"]
        for k, item in enumerate(na, 1):
            line = "%d. %s" % (k, item)
            if size(line) > CAP_BYTES - 120:          # leave headroom for live-runs + footer
                out.append("   …(+%d more — see %s)" % (len(na) - k + 1, BOARD_REL))
                break
            out.append(line)

    # LIVE RUNS
    if lr and size("LIVE RUNS: " + lr) <= CAP_BYTES - 40:
        out += ["", "LIVE RUNS: " + lr]

    # LAST DECISION
    if ld and size("LAST DECISION: " + ld) <= CAP_BYTES - 20:
        out += ["", "LAST DECISION: " + ld]

    # FOOTER
    if size(footer) <= CAP_BYTES:
        out += ["", footer]

    text = "\n".join(out) + "\n"

    # safety net: guarantee <= CAP even if a single section was pathologically long
    if _bytes(text) > CAP_BYTES:
        enc = text.encode("utf-8")[: CAP_BYTES - 16]
        text = enc.decode("utf-8", errors="ignore").rstrip() + "\n…[capped]\n"
    return text


def write(root: str | None = None, fast: bool = False) -> str:
    root = root or _root()
    text = build(root, fast=fast)
    p = os.path.join(root, LIVE_STATE_REL)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8").write(text)
    return text


def read(root: str | None = None) -> str:
    """Read the file; regenerate (fast) if missing so a hook never emits nothing."""
    root = root or _root()
    txt = _read(root, LIVE_STATE_REL)
    if not txt.strip():
        txt = write(root, fast=True)
    return txt


def heartbeat_line(root: str | None = None) -> str:
    """The compressed one-line heartbeat delta (noise-cut §6): the top next-action + the run-state ACT.

    Replaces the verbose multi-line ACT prose the heartbeat echoed every ~15 min (per-cycle dilution)
    with ONE line carrying LIVE-STATE's top next-action. The heartbeat's GPU/procs/new-json STATE
    CHECK stays in the Monitor recipe itself (it is the anti-stall backstop); this supplies the
    action half.
    """
    root = root or _root()
    board = _read(root, BOARD_REL)
    na = next_actions(root, board)
    top = _shorten(na[0], 160) if na else "(no next-action parsed — read GAP_CLOSURE_MISSION.md)"
    return ("NEXT: %s | ACT: FINISHED(new output)→read+act; ALIVE+GPU-idle 2+ beats→STALLED,"
            " check ps + kill/re-scope; IDLE→take the next step. Turn-enders: owner stop / safety."
            % top)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    fast = os.environ.get("LIVE_STATE_FAST") == "1"
    arg = argv[1] if len(argv) > 1 else ""

    if arg == "--heartbeat":
        print(heartbeat_line())
        return 0
    if arg == "--emit":                                # hook fast path: read (regen if missing)
        sys.stdout.write(read())
        return 0
    if arg == "--stdout":
        sys.stdout.write(build(fast=fast))
        return 0
    if arg == "--check":
        text = build(fast=True)
        nb = _bytes(text)
        print("live-state %d B / cap %d B — %s" % (nb, CAP_BYTES, "OK" if nb <= CAP_BYTES else "OVER CAP"))
        return 0 if nb <= CAP_BYTES else 1

    text = write(fast=fast)
    print("wrote %s (%d B / cap %d B)" % (LIVE_STATE_REL, _bytes(text), CAP_BYTES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
