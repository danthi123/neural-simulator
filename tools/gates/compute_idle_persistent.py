"""CLASS UC — DEDICATED compute (a pool node / the GPU) reads idle-with-ready-work, PERSISTENTLY. BLOCKING.

WHY THIS IS BLOCKING AND NOT JUST ANOTHER REPORT (owner, 2026-09-08, on `tools/parallel_audit.py`'s existing
UNDER-PARALLELIZED verdict: "past fixes failed being manual/advisory/passive"). `parallel_audit.py` already
computes this exact condition every heartbeat cycle and PRINTS it — and printing was exactly the failure
mode caught this session: one arc ran all session while 8 roadmap lanes + the mini-PC pool sat idle ~2 days,
with the heartbeat alarming correctly the entire time. `gates/lane_starvation` already solved the identical
shape of problem for idle CPU LANES (owner, 2026-07-31: "what's the point of a gate that doesn't block
non-adherence to our workflow?") by moving the check onto an unavoidable path with a waiver escape. This is
the SAME fix for the SAME resource-waste shape — a genuinely idle, zero-marginal-cost DEDICATED resource
sitting unused next to ready work — just fed by a different signal: a pool node or the GPU instead of a CPU
lane. The owner has already accepted this risk profile for lane_starvation; this gate mirrors it rather than
inventing a new one.

WHY PERSISTENCE, NOT A SINGLE READING. `under_compute` is a SNAPSHOT (is a dedicated lane idle RIGHT NOW)
that can legitimately flicker — a pool job just finished, the next hasn't been dispatched yet. Blocking on a
single snapshot would nag on completely normal gaps between jobs and get bypassed with `--no-verify`, which
disables every OTHER gate on the same commit (the exact "check that gets disabled" failure this project's
gate philosophy warns against). So this reads `research/coordination/parallel_audit_state.json` (written
every heartbeat cycle by `parallel_audit.py` via `tools/parallel_state.py`, which tracks WHEN the current
unbroken "true" streak started) and blocks only once `since_under_compute` shows the condition has held
CONTINUOUSLY past `PERSIST_S`.

THE ESCAPE, because a gate with no legitimate exit gets bypassed and then ignored entirely (the same
rationale `gates/lane_starvation` documents for its own `.lane_waiver`): write
`research/queue/.parallel_compute_waiver` with a REASON. It expires after `WAIVER_MAX_H` hours, so a waiver
cannot silently become permanent. A waiver that excuses the idle lane by PRIORITY/FOCUS ("focused on the
crux") rather than naming a genuine per-lane BLOCKER is REJECTED — reusing lane_starvation's own
rationalisation regex verbatim, because pool/GPU idleness is exactly as zero-cost to fill as an idle CPU
lane, so "I'm prioritising something else" can never excuse leaving it idle either (the 2026-08-01 abuse
lane_starvation's own docstring records).

STALENESS GUARD, AND WHY AN ABSENT/OLD STATE FILE PASSES RATHER THAN BLOCKS. The heartbeat is a Claude-session
habit, not a daemon (CLAUDE.md: "Cross-session continuation is MANUAL ... no watchdog/daemon"). A commit made
before the heartbeat has run this session, or long after the last agent process exited, must not be blocked
on data that no longer describes the world — that would eventually brick every commit in a repo that goes
quiet, a worse failure than the advisory-only gap this closes. So a MISSING file, or one older than
`tools.parallel_state.STALE_S`, is treated as "no signal" and passes silently — the same shape
`gates/tool_health_fresh` uses for an absent smoke.

WHAT IT CANNOT CATCH: whether the launched work is WORTH running (the same limit `lane_starvation` names for
itself), and it is blind between heartbeat cycles — a session that never runs the heartbeat never persists a
streak, so this gate stays silent for it (a coverage gap shared with every heartbeat-fed check here, not a
defect unique to this one). It also does not cover the AGENT-FLOOR half of `parallel_audit`'s verdict — see
`gates/agent_floor_persistent` (report-only) and this session's write-up for why escalating that half to
BLOCKING is left as a proposal for owner review rather than decided here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
import parallel_state    # noqa: E402  (shared writer/reader schema, see tools/parallel_state.py)
import waiver_history as wh  # noqa: E402  (shared CLASS+budget escape hatch, see tools/waiver_history.py)

NAME = "compute-idle-persistent"
CLASS_ID = "UC"
BLOCKING = True

# REVIEW FIX (2026-09-23): pinned to `_ROOT` (this file's OWN worktree) it silently kept a SEPARATE waiver +
# budget per worktree -- `gates/lane_starvation` already solved this for `.lane_waiver` via its own
# `_shared_queue_root()`; `waiver_history.shared_root()` is the identical git-common-dir resolution, reused
# here so `.parallel_compute_waiver` and the shared `.waiver_history.jsonl` both resolve to the SAME physical
# path regardless of which worktree evaluates them.
WAIVER_FILE = os.path.join(wh.shared_root(), "research", "queue", ".parallel_compute_waiver")
PERSIST_S = 45 * 60     # under_compute must hold CONTINUOUSLY this long before it blocks
WAIVER_MAX_H = 6


def _decide(state, now_ts, verdict, persist_s=PERSIST_S):
    """Pure decision (testable without files/clock/waiver disk state). `verdict` is a
    `waiver_history.evaluate(...)` result (or the caller's own dict of the same shape).

    GAME_MODE is handled UPSTREAM in parallel_audit.py (it excludes the local GPU from `under_compute` during a
    game, so the persisted signal this gate reads already reflects pool-only idle) — the mini-PC pool stays
    enforced during a game on purpose (the owner's game-time plan is to keep it busy), so this gate needs no
    game-mode branch of its own."""
    if not parallel_state.is_fresh(state, now_ts):
        return []                       # no signal: absent/stale state never blocks (see module docstring)
    since = state.get("since_under_compute")
    if not (state.get("under_compute") and since is not None and (now_ts - since) >= persist_s):
        return []
    mins = int((now_ts - since) / 60)
    if verdict.get("active"):
        if verdict.get("ok"):
            return []                                              # a valid, in-budget waiver excuses it
        return ["DEDICATED compute (a pool node / the GPU) has read idle-with-ready-work for %d min "
                "straight, and research/queue/.parallel_compute_waiver is REJECTED: %s. Queue something "
                "instead: `tools/sweep_pool.sh` / `tools/gpu_queue.sh add '<cmd>'`, or write a waiver in the "
                "form `CLASS: GAMING|OWNER-PAUSE|RAM-CONTENTION|NO-READY-WORK` (+ avail_gb=/checked= as "
                "required) naming the CURRENT resource constraint, never a plan."
                % (mins, verdict.get("reject_reason", ""))]
    return ["DEDICATED compute (a pool node / the GPU) has read idle-with-ready-work for %d min straight "
            "(budget %d) — this is the same zero-marginal-cost waste `gates/lane_starvation` already blocks "
            "on for CPU lanes, just for GPU/pool idleness. FIX: queue one job — `tools/sweep_pool.sh` (CPU) "
            "or `tools/gpu_queue.sh add '<cmd>'` (GPU) — or waive a genuine blocker (auto-expires in %dh):\n"
            "          printf 'CLASS: RAM-CONTENTION\\navail_gb=%%d\\nreason: <why>\\n' \"$(free -g | "
            "awk '/^Mem:/{print $7}')\" > research/queue/.parallel_compute_waiver\n"
            "        (or CLASS: NO-READY-WORK with checked=<what you searched>, or CLASS: GAMING / "
            "OWNER-PAUSE)"
            % (mins, persist_s // 60, WAIVER_MAX_H)]


def _staged_files():
    """The ACTUAL staged set, any status (mirrors gates/lane_starvation._staged_files -- the hook's own
    --diff-filter=A passthrough misses MODIFY-only commits)."""
    try:
        return subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.split()
    except Exception:
        return []


# .md paths that are THEMSELVES a research-status signal -- exactly the "research/findings, no sim/" carve-out
# the docstring below already named -- so a staged change to one of these must NOT count as infra-only, even
# though it ends in ".md". `research/findings/*.md` is a research artifact; GAP_CLOSURE_MISSION.md/ROADMAP.md
# are the board/roadmap that record whether ready work exists and whether something was "queued" -- exempting
# them would let a commit that only EDITS THE BOARD (e.g. claiming a lane was served) pass as compute-neutral
# without ever having queued anything (2026-09-23 review fix: the original check exempted ANY `.md`, which
# covered exactly these three paths despite the docstring already saying they should not be exempt).
_NON_INFRA_MD = ("GAP_CLOSURE_MISSION.md", "ROADMAP.md")


def _is_research_status_md(p):
    if p in _NON_INFRA_MD:
        return True
    return p.startswith("research/findings/") and p.endswith(".md")


def _is_infra_only(staged):
    """A non-empty staged set that touches ONLY tools/**, tests/** or a GENERIC **/*.md (README, docs/,
    CLAUDE.md, ...) -- no research/runners, no research/findings, no sim/, and no board/roadmap file that
    itself records research status. This gate blocks on a project-wide READY-RESEARCH-WORK signal (idle
    pool/GPU next to a roadmap backlog); a commit that adds no research artifact and queues no research job has
    no bearing on that allocation, exactly the reasoning `gates/lane_starvation._is_doc_only` already
    established for markdown-only commits (2026-08-06/07) -- generalised here to the two other paths a
    compute-lane-neutral commit lives in. Landed 2026-09-23 while closing the waiver loophole itself: this
    gate's OWN infra fix tripped it (a live, GENUINE 14.5-day idle-compute signal, not a test artifact -- see
    the commit message), which is exactly the false-positive shape this exemption removes without weakening
    the real check. NARROWED same day (review fix): the first cut of this exemption accepted ANY `.md`,
    including `research/findings/*.md` and the board/roadmap files -- exactly the two exclusions this
    docstring already named but the code did not enforce."""
    if not staged:
        return False
    for p in staged:
        if p == ".gitignore":
            continue
        if p.endswith(".md"):
            if _is_research_status_md(p):
                return False
            continue
        if p.startswith("tools/") or p.startswith("tests/"):
            continue
        return False
    return True


def check(paths=None):
    del paths
    if _is_infra_only(_staged_files()):
        return []
    now_ts = time.time()
    verdict = wh.evaluate(NAME, WAIVER_FILE, WAIVER_MAX_H, now_ts=now_ts)
    return _decide(parallel_state.load(), now_ts, verdict)


def selftest():
    """FAILING DIRECTION FIRST: a fresh state with under_compute persisted past budget, no waiver, MUST block."""
    bad = []
    now = 2_000_000.0
    fresh_over = {"generated_at": now, "under_compute": True, "since_under_compute": now - PERSIST_S - 60}
    no_waiver = {"active": False}
    if not _decide(fresh_over, now, no_waiver):
        bad.append("did NOT block a persisted-past-budget idle-compute reading with no waiver")
    # NEGATIVE — not yet past budget.
    fresh_under = {"generated_at": now, "under_compute": True, "since_under_compute": now - 60}
    if _decide(fresh_under, now, no_waiver):
        bad.append("FALSE POSITIVE: blocked before the persistence budget was reached")
    # NEGATIVE — currently healthy.
    healthy = {"generated_at": now, "under_compute": False, "since_under_compute": None}
    if _decide(healthy, now, no_waiver):
        bad.append("FALSE POSITIVE: blocked while under_compute currently reads healthy")
    # NEGATIVE — stale state (heartbeat not running recently) must pass SILENTLY, never block.
    stale = {"generated_at": now - parallel_state.STALE_S - 1, "under_compute": True,
             "since_under_compute": now - PERSIST_S - 3600}
    if _decide(stale, now, no_waiver):
        bad.append("FALSE POSITIVE: blocked on a STALE state (heartbeat may not be running)")
    # NEGATIVE — absent state.
    if _decide(None, now, no_waiver):
        bad.append("FALSE POSITIVE: blocked with no state file at all")
    # waiver: a genuine classified+evidenced blocker, in budget, excuses it.
    ok_verdict = {"active": True, "ok": True, "class": "NO-READY-WORK", "age_h": 0.1, "budget_h": 1.0}
    if _decide(fresh_over, now, ok_verdict):
        bad.append("FALSE POSITIVE: a genuine classified in-budget waiver was still blocked")
    # waiver: an INVALID waiver (bad class, promise language, or budget-exhausted) must still block — this is
    # the 2026-09-23 loophole close: `waiver_history.evaluate` is the sole judge of validity here, so this
    # gate's own selftest only needs to prove it RESPECTS that verdict, not re-derive the classification logic
    # (that is `tools/waiver_history.py`'s own selftest's job).
    bad_verdict = {"active": True, "ok": False, "class": None,
                   "reject_reason": "promise/intent language 'will' detected -- REJECTED"}
    if not _decide(fresh_over, now, bad_verdict):
        bad.append("did NOT reject an INVALID waiver verdict (the 2026-09-23 promise-language loophole "
                   "would pass)")
    # _is_infra_only: a tools/tests/docs/.gitignore-only staged set is exempt; research/sim changes are NOT.
    if not _is_infra_only(["tools/waiver_history.py", "tests/test_waiver_history.py", "README.md",
                           ".gitignore"]):
        bad.append("a tools+tests+md+.gitignore staged set was NOT recognised as infra-only")
    if _is_infra_only(["tools/waiver_history.py", "research/runners/some_derisk.py"]):
        bad.append("BROKEN GUARD: a mixed tools+research staged set was treated as infra-only")
    if _is_infra_only([]):
        bad.append("BROKEN GUARD: an EMPTY staged set was treated as infra-only (would exempt every commit)")
    # 2026-09-23 review fix: a research/findings/*.md or board/roadmap .md must NOT be swept into the generic
    # ".md is always infra-only" bucket -- each of these IS a research-status signal.
    if _is_infra_only(["research/findings/2026-09-23-some-result.md"]):
        bad.append("LOOPHOLE STILL OPEN: a research/findings/*.md-only staged set was treated as infra-only "
                   "(a finding is a research artifact, not compute-neutral infra)")
    if _is_infra_only(["GAP_CLOSURE_MISSION.md"]):
        bad.append("LOOPHOLE STILL OPEN: a GAP_CLOSURE_MISSION.md-only staged set was treated as infra-only "
                   "(the board records research status; editing it is not compute-neutral)")
    if _is_infra_only(["ROADMAP.md"]):
        bad.append("LOOPHOLE STILL OPEN: a ROADMAP.md-only staged set was treated as infra-only")
    # NEGATIVE: a genuinely generic doc (README/docs/CLAUDE.md) alongside tools/tests must stay exempt --
    # the narrowing must not regress the original infra-only exemption for ordinary documentation.
    if not _is_infra_only(["tools/waiver_history.py", "docs/FAILURE_GATE_MATRIX.md", "CLAUDE.md"]):
        bad.append("FALSE POSITIVE: narrowing the .md exemption also swept in genuinely generic docs")
    return bad
