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


# FIX ROUND 3 (2026-09-23): an explicit ALLOW-LIST, deny by default. The fix-2 version was a deny-list over
# "any .md is exempt" and "tools/** is exempt", so it still exempted the MASTER ROADMAP (docs/plans/), every
# research/biology/*.md, every other root .md -- and tools/gates/**, i.e. a commit that WEAKENS THIS GATE (or
# waiver_history / parallel_state / parallel_audit / the pre-commit hook it depends on) was exempt from this
# gate. Enforcement machinery is never compute-neutral with respect to the enforcement it changes.
_ENFORCEMENT_PATHS = ("tools/gates/", "tools/githooks/")
_ENFORCEMENT_FILES = ("tools/waiver_history.py", "tools/parallel_state.py", "tools/parallel_audit.py")
_GENERIC_DOCS = ("README.md", "CLAUDE.md", "CONTRIBUTING.md", "CHANGELOG.md", "USER_GUIDE.md", "QUICKSTART.md",
                 "docs/FAILURE_GATE_MATRIX.md", "docs/WRITING.md", "docs/TERMS.md", "docs/ENGINE_REFERENCE.md")


def _is_exempt_path(p):
    if p == ".gitignore" or p in _GENERIC_DOCS:
        return True
    if p in _ENFORCEMENT_FILES or p.startswith(_ENFORCEMENT_PATHS):
        return False
    return p.startswith("tools/") or p.startswith("tests/")


def _is_infra_only(staged):
    """A non-empty staged set in which EVERY path is on this allow-list, and nothing else:
      * `tools/**` and `tests/**` -- EXCEPT the enforcement machinery this gate runs on (`tools/gates/**`,
        `tools/githooks/**`, `tools/waiver_history.py`, `tools/parallel_state.py`, `tools/parallel_audit.py`);
      * the generic, non-status docs named in `_GENERIC_DOCS` (README/CLAUDE/CONTRIBUTING/CHANGELOG/
        USER_GUIDE/QUICKSTART and four process-reference docs under docs/);
      * `.gitignore`.
    Everything else -- research/** (findings, biology, FAILURE_LOG, runners), sim/**, docs/plans/** (the MASTER
    ROADMAP), GAP_CLOSURE_MISSION.md, ROADMAP.md, every other .md -- is NOT exempt. This gate blocks on a
    project-wide READY-RESEARCH-WORK signal (idle pool/GPU next to a roadmap backlog); a commit that adds no
    research artifact, queues no research job and does not touch the enforcement itself has no bearing on it
    (the reasoning `gates/lane_starvation._is_doc_only` established for markdown-only commits, 2026-08-06/07)."""
    return bool(staged) and all(_is_exempt_path(p) for p in staged)


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
    # _is_infra_only (FIX ROUND 3 allow-list). FAILING DIRECTION FIRST: research-status docs and the
    # enforcement machinery itself must NOT be exempt, alone or riding along with an ordinary tools/ change.
    for p in ("research/findings/2026-09-23-some-result.md", "GAP_CLOSURE_MISSION.md", "ROADMAP.md",
              "docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md", "research/biology/btsp.md",
              "tools/gates/compute_idle_persistent.py", "tools/waiver_history.py", "tools/githooks/pre-commit",
              "research/runners/some_derisk.py"):
        if _is_infra_only([p]) or _is_infra_only(["tools/lab.py", p]):
            bad.append("LOOPHOLE: %r was treated as compute-neutral infra" % p)
    if _is_infra_only([]):
        bad.append("BROKEN GUARD: an EMPTY staged set was treated as infra-only (would exempt every commit)")
    # NEGATIVE: genuinely generic infra stays exempt.
    if not _is_infra_only(["tools/lab.py", "tests/test_lab.py", "README.md", "CLAUDE.md",
                           "docs/FAILURE_GATE_MATRIX.md", ".gitignore"]):
        bad.append("FALSE POSITIVE: an ordinary tools+tests+generic-docs staged set was not exempt")
    return bad
