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
import re
import subprocess
import time

# A waiver may excuse idle lanes only for a genuine BLOCKER, never for a PRIORITY/FOCUS rationalisation: the
# five lanes are DISJOINT and cost nothing beside the crux, so "I'm focusing on the crux" can never justify
# leaving them idle. Earned 2026-08-01: a false ".lane_waiver" reading "saturated with the gap#4 crux ...
# deprioritized behind the crux, not starved by neglect" suppressed this gate for hours while 24 pool cores
# sat free and lane_check screamed MONOCULTURE (5 lanes unserved 1187 min) — the exact abuse the waiver exists
# to prevent, dressed as prioritisation. A waiver invoking this vocabulary is REJECTED.
_RATIONALISATION = re.compile(r"crux|priorit|focus|deprioriti|momentum|behind the|saturated with", re.I)

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


def _waiver_active():
    waiver = os.path.join(_queue_dir(), ".lane_waiver")
    if not os.path.exists(waiver):
        return None
    age_h = (time.time() - os.path.getmtime(waiver)) / 3600.0
    if age_h > LANE_WAIVER_MAX_H:
        return None
    try:
        return open(waiver, errors="ignore").read().strip()[:120] or "(no reason given)"
    except OSError:
        return None


def check(paths=None):
    idle = sorted(set(CPU_LANES) - _served(_work_lines()))
    if len(idle) < MAX_IDLE_LANES:
        return []
    w = _waiver_active()
    if w:
        if _RATIONALISATION.search(w):
            return ["%d of %d disjoint CPU lanes UNSERVED: %s — and the .lane_waiver justifies it by "
                    "PRIORITY/FOCUS (\"%s\"), which is REJECTED.\n"
                    "        The five lanes are DISJOINT and cost NOTHING beside the crux, so a priority can\n"
                    "        NEVER excuse leaving them idle (2026-08-01: a false 'saturated with the crux' waiver\n"
                    "        held this open for hours while 24 pool cores sat free). A valid waiver names a real\n"
                    "        BLOCKER — no ready de-risk for the lane, or genuinely zero free CPU capacity — not\n"
                    "        what you'd rather work on. Queue one job per idle lane instead:\n"
                    "          bash tools/pool_queue.sh add '<cmd>' --checked '<what the record says>'"
                    % (len(idle), len(CPU_LANES), "; ".join(idle), w)]
        return []
    return ["%d of %d disjoint CPU lanes UNSERVED: %s.\n"
            "        They are concurrent with GPU work and cost nothing beside it; leaving them unqueued is\n"
            "        unused capacity, not prioritisation. Five sat idle 194 min on 2026-07-31 while the\n"
            "        heartbeat alarmed correctly and was read past.\n"
            "        FIX: stage one job per idle lane -\n"
            "          bash tools/pool_queue.sh add '<cmd>' --checked '<what the record says>'\n"
            "        Or waive with a reason (auto-expires in %dh):\n"
            "          echo 'why' > research/queue/.lane_waiver"
            % (len(idle), len(CPU_LANES), "; ".join(idle), LANE_WAIVER_MAX_H)]


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
    # 2026-08-01: a waiver that justifies idle lanes by PRIORITY/FOCUS must be caught (the exact abuse); a
    # waiver naming a genuine per-lane BLOCKER must NOT be flagged as a rationalisation.
    if not _RATIONALISATION.search("saturated with the gap#4 crux; deprioritized behind the crux, not neglect"):
        bad.append("did NOT detect the priority/focus rationalisation waiver -> the 2026-08-01 abuse would pass")
    if _RATIONALISATION.search("no ready de-risk for these lanes: the stream-code cache is absent; blocked on its build"):
        bad.append("FALSE POSITIVE: a genuine per-lane BLOCKER waiver was mis-flagged as a rationalisation")
    return bad
