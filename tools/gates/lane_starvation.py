"""CLASS L — CPU LANES STARVED while work continues elsewhere. BLOCKING.

WHY THIS IS BLOCKING AND NOT REPORTING (owner, 2026-07-31: "what's the point of a gate that doesn't block
non-adherence to our workflow?"). On 2026-07-31 five of five CPU lanes sat unserved for 194 MINUTES while the
heartbeat alarmed correctly every 15 minutes and I read past it, building gate infrastructure. A true alarm that
only reports is worth exactly as much as a false one -- nothing.

I first logged this as NOT-GATEABLE on the reasoning that "staging work is judgement about WHAT to run". That
conflated two different things. Choosing what to run IS judgement. **"You may not keep committing while five
disjoint lanes sit idle" is not** -- it is a rule, and rules belong on an unavoidable path.

The five CPU lanes (A Affect · B Curiosity · C Self/Workspace · D Perception · E Language) are explicitly
disjoint per the roadmap's parallelization map: "cleanly concurrent; they share only the bridge + the
stream-cortex codes". They cost nothing beside GPU work. Leaving them unqueued is unused free capacity, not
prioritisation.

THE ESCAPE, because a gate with no legitimate exit gets bypassed with --no-verify and then ignored entirely:
write research/queue/.lane_waiver containing a REASON. It expires after LANE_WAIVER_MAX_H hours, so a waiver
cannot silently become permanent -- the same auto-expiry pattern as the contention window in workflow_check.sh,
which exists because a stale suspension once disabled a rule indefinitely.

WHAT IT CANNOT CATCH: whether the queued work is WORTH running. A lane served by a pointless job passes. That is
the judgement half, and it stays with the human and with me.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

_ROOT_FOR_IMPORT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_ROOT_FOR_IMPORT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT_FOR_IMPORT, "tools"))
import waiver_history as wh  # noqa: E402  (shared CLASS+budget escape hatch, see tools/waiver_history.py)

NAME = "lane-starvation"
CLASS_ID = "L"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_IDLE_LANES = 3          # of the five disjoint CPU lanes
LANE_WAIVER_MAX_H = 6
RECENT_DISPATCH_MIN = 45   # a dispatched pool job runs remotely; it serves its lane for this long

CPU_LANES = {
    "A · Affect":        ["affect", "appraisal", "valence", "liking", "_dr2", "emotion"],
    "B · Curiosity":     ["curiosity", "novelty", "question_gen", "_dr1", "learning_progress"],
    "C · Self/Workspace": ["self_schema", "meta_d", "false_belief", "workspace", "_dr3", "_p1_2", "tom"],
    "D · Perception":    ["v1_selforg", "_b1_", "visual", "retina", "gabor", "v2_", "_it_", "nav_"],
    "E · Language":      ["emerge6", "emerge7", "construction", "morpholog", "lexicon", "grammar",
                          "comprehension", "producer", "confidence_gate"],
}


def _shared_queue_root():
    """Return the checkout whose persistent dispatchers consume the queues."""
    override = os.environ.get("SIM_QUEUE_ROOT")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        if common:
            return os.path.dirname(common)
    except Exception:
        pass
    return _ROOT


def _queue_dir():
    return os.path.join(_shared_queue_root(), "research", "queue")


def _served(text_blobs):
    served = set()
    for ln in text_blobs:
        c = ln.lower()
        for lane, keys in CPU_LANES.items():
            if any(k in c for k in keys):
                served.add(lane)
    return served


def _work_lines():
    lines = []
    queue_dir = _queue_dir()
    for name in ("gpu.queue", "pool.queue"):
        p = os.path.join(queue_dir, name)
        if os.path.exists(p):
            for l in open(p, errors="ignore").read().split("\n"):
                l = l.strip()
                if l and not l.startswith("#"):
                    lines.append(l.split("#checked:")[0])
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=15).stdout
        lines += [l for l in out.split("\n") if "research.runners" in l and "grep" not in l]
    except Exception:
        pass
    # WORK RUNNING ON THE POOL IS INVISIBLE TO A LOCAL `ps`. A job dispatched to a mini-PC leaves the local
    # queue and runs remotely, so counting only local processes reported three lanes STARVED while their jobs
    # were actively running on pool40/41 -- the same blindness that made lane_check read one queue of two.
    # dispatch.log records what went out and when; a recent dispatch counts as serving its lane.
    dl = os.path.join(queue_dir, "dispatch.log")
    if os.path.exists(dl):
        cutoff = time.time() - RECENT_DISPATCH_MIN * 60
        try:
            if os.path.getmtime(dl) >= cutoff:
                tail = open(dl, errors="ignore").read().split("\n")[-80:]
                lines += [l.split("<- ", 1)[1] for l in tail if "<- " in l]
        except OSError:
            pass
    return lines


LANE_BUILD_MAX_H = 6
_LANE_BUILDS = os.path.join(_ROOT, "research", "coordination", "lane_builds.jsonl")


def _build_served(now_ts, path=None):
    """Lanes whose next work is a BUILD in flight (2026-09-24). Every unserved lane's genuine next job sat inside an
    agent build that could not commit while the lane read unserved -- a deadlock (waiver budget exhausted, no honest
    job to queue). A registration names the lane, the workflow/task id and the branch, is committed (auditable), and
    expires after LANE_BUILD_MAX_H hours; malformed or expired entries are ignored."""
    import json
    served = set()
    try:
        rows = open(path or _LANE_BUILDS, errors="ignore").read().splitlines()
    except OSError:
        return served
    for ln in rows:
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if (e.get("lane") in CPU_LANES and e.get("workflow") and e.get("branch")
                and 0 <= now_ts - float(e.get("started", -1)) <= LANE_BUILD_MAX_H * 3600):
            served.add(e["lane"])
    return served


def _waiver_file():
    return os.path.join(_queue_dir(), ".lane_waiver")


def _staged_files():
    """The ACTUAL staged set, ANY status. The pre-commit hook passes only ADDED files (--diff-filter=A), so a
    MODIFY-only commit -- the board/roadmap are ALWAYS modifies -- arrives at check() as an empty list. Read it
    ourselves so the doc-only exemption sees a board edit."""
    try:
        return subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.split()
    except Exception:
        return []


def _is_doc_only(staged):
    """A non-empty staged set that is ALL Markdown -> no compute to parallelise -> exempt."""
    return bool(staged) and all(p.endswith(".md") for p in staged)


def _idle_message(idle, verdict):
    """Pure: the block message (or [] if excused) for a given idle-lane set + waiver_history.evaluate() verdict.
    Factored out of check() so selftest() can exercise the waiver-honoring logic WITHOUT touching
    research/queue/.lane_waiver on disk (a selftest that writes the real waiver file would itself be an
    instance of the exact "gate that has side effects on every commit" failure shape this project avoids)."""
    if verdict.get("active"):
        if verdict.get("ok"):
            return []                                           # a valid, in-budget waiver excuses it
        return ["%d of %d disjoint CPU lanes UNSERVED: %s — and .lane_waiver is REJECTED: %s\n"
                "        A valid waiver declares `CLASS: GAMING|OWNER-PAUSE|RAM-CONTENTION|NO-READY-WORK`\n"
                "        naming the CURRENT resource constraint (RAM-CONTENTION needs avail_gb=, NO-READY-WORK\n"
                "        needs checked=), never a plan or a priority claim. Queue one job per idle lane instead:\n"
                "          bash tools/pool_queue.sh add '<cmd>' --checked '<what the record says>'"
                % (len(idle), len(CPU_LANES), "; ".join(idle), verdict.get("reject_reason", ""))]
    return ["%d of %d disjoint CPU lanes UNSERVED: %s.\n"
            "        They are concurrent with GPU work and cost nothing beside it; leaving them unqueued is\n"
            "        unused capacity, not prioritisation. Five sat idle 194 min on 2026-07-31 while the\n"
            "        heartbeat alarmed correctly and was read past.\n"
            "        FIX: stage one job per idle lane -\n"
            "          bash tools/pool_queue.sh add '<cmd>' --checked '<what the record says>'\n"
            "        Or waive (auto-expires in %dh, and the class/evidence is REQUIRED — free prose is no\n"
            "        longer accepted, the 2026-09-23 promise-language loophole):\n"
            "          printf 'CLASS: NO-READY-WORK\\nchecked=<what you searched>\\n' > research/queue/.lane_waiver"
            % (len(idle), len(CPU_LANES), "; ".join(idle), LANE_WAIVER_MAX_H)]


def check(paths=None):
    # DOC-ONLY EXEMPTION (2026-08-06; HARDENED 2026-08-07). A commit staging ONLY Markdown (board, roadmap,
    # findings, docs, RETRACTED) has no compute to parallelise, so idle CPU lanes cannot be its fault, and
    # blocking it only trains a reflex `--no-verify` (~6 such bypasses in one session; a bypass disables EVERY
    # other gate for that commit). The first version keyed on the passed `paths`, but the hook filters to ADDED
    # files -- and the board is always a MODIFY -- so board commits arrived empty and STILL blocked, the exact
    # case this targeted. Read the real staged set ourselves. Standalone (`python tools/gates/...`, no staged
    # commit) reads empty -> falls through to the corpus scan, unaffected.
    if _is_doc_only(_staged_files()):
        return []
    idle = sorted(set(CPU_LANES) - _served(_work_lines()) - _build_served(time.time()))
    if len(idle) < MAX_IDLE_LANES:
        return []
    verdict = wh.evaluate(NAME, _waiver_file(), LANE_WAIVER_MAX_H, now_ts=time.time())
    return _idle_message(idle, verdict)


def selftest():
    """FAILING DIRECTION FIRST: with nothing queued and no waiver, this MUST fire."""
    bad = []
    if not _served(["python -m research.runners._affect_state_region_derisk --seeds 42"]):
        bad.append("did NOT recognise an affect job as serving lane A")
    if len(set(CPU_LANES) - _served([])) < MAX_IDLE_LANES:
        bad.append("empty work list did NOT read as starvation")
    # a full complement must NOT fire
    full = ["_affect_x", "_curiosity_x", "self_schema_x", "_b1_v1_selforg_x", "construction_x"]
    if set(CPU_LANES) - _served(full):
        bad.append("FALSE POSITIVE: a job per lane still read as unserved")
    # 2026-09-23: an INVALID waiver verdict (bad class, promise language, or budget-exhausted -- classification
    # itself is `tools/waiver_history.py`'s job, see ITS selftest) must still BLOCK here; a valid, in-budget one
    # must excuse it. Exercised via the pure `_idle_message` so this selftest never touches the real
    # research/queue/.lane_waiver file (a selftest with disk side effects on every commit is its own failure
    # shape).
    import json, tempfile
    lane_a = sorted(CPU_LANES)[0]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps({"lane": lane_a, "workflow": "wX", "branch": "research/x", "started": 1000.0}) + "\n")
        fh.write(json.dumps({"lane": lane_a, "workflow": "", "branch": "research/x", "started": 1000.0}) + "\n")
        reg = fh.name
    if _build_served(1000.0 + 60, reg) != {lane_a}:
        bad.append("did NOT count a fresh, well-formed lane-build registration as serving its lane")
    if _build_served(1000.0 + LANE_BUILD_MAX_H * 3600 + 1, reg):
        bad.append("FALSE NEGATIVE-PROOF: an EXPIRED lane-build registration still served its lane")
    os.unlink(reg)
    idle5 = sorted(CPU_LANES)
    rejected = {"active": True, "ok": False, "class": None,
                "reject_reason": "promise/intent language 'will' detected -- REJECTED"}
    if not _idle_message(idle5, rejected):
        bad.append("did NOT block on a REJECTED waiver verdict (the 2026-09-23 promise-language loophole "
                  "would pass)")
    accepted = {"active": True, "ok": True, "class": "NO-READY-WORK", "age_h": 0.1, "budget_h": 1.0}
    if _idle_message(idle5, accepted):
        bad.append("FALSE POSITIVE: a valid in-budget waiver verdict was still blocked")
    if not _idle_message(idle5, {"active": False}):
        bad.append("did NOT block with no waiver at all")
    # DOC-ONLY EXEMPTION (2026-08-07): keys on the ACTUAL staged set (any status) via _is_doc_only.
    if not _is_doc_only(["GAP_CLOSURE_MISSION.md", "docs/RETRACTED.md"]):
        bad.append("a doc-only (.md) staged set was NOT recognised as exempt")
    if _is_doc_only(["GAP_CLOSURE_MISSION.md", "research/runners/x.py"]):
        bad.append("BROKEN GUARD: a mixed doc+code set was treated as doc-only")
    if _is_doc_only([]):
        bad.append("BROKEN GUARD: an EMPTY staged set was treated as doc-only (would exempt every commit)")
    return bad
