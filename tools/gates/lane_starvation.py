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

import json
import os
import re
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
    "A · Affect":        ["affect", "appraisal", "valence", "liking", "_dr2", "emotion", "need_reversal", "bg_live_seam"],
    "B · Curiosity":     ["curiosity", "novelty", "question_gen", "_dr1", "learning_progress", "long_delay_credit"],
    "C · Self/Workspace": ["self_schema", "meta_d", "false_belief", "workspace", "_dr3", "_p1_2", "tom", "host_share"],
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


# Self-declared FILLER never serves a lane (2026-09-29, FAILURE_LOG). Twice in one day a subagent passed this gate by
# queueing cheap "keepalive" smokes it described in POOL_CHECKED_REASON as "lane-service only ... not a research lever":
# honest labels on dishonest work. A lane is served by research, not by a job whose own reason says it is not research.
FILLER_MARKERS = ("lane-service", "lane service", "keepalive", "keep-alive", "not a research lever", "not research")


def _is_filler(line):
    c = line.lower().replace("\\ ", " ")
    return any(m in c for m in FILLER_MARKERS)


def _served(text_blobs):
    served = set()
    for ln in text_blobs:
        if _is_filler(ln):
            continue
        c = ln.lower()
        for lane, keys in CPU_LANES.items():
            if any(k in c for k in keys):
                served.add(lane)
    return served


# POOL SATURATED / REPEAT BUSYWORK (owner-approved 2026-10-12). The gate's premise is unused capacity. Two ways it
# misfired: (1) it fired while every pool node was already running stage-critical work, so obeying it meant
# queueing behind that work or waiving; (2) it was "served" by the same smoke re-queued with only a new output name
# (laneB --smoke x4 on 10-11/12), which is busywork, not research. So: when every default pool node has a job
# RUNNING (dispatched in pool.running, no exit since in sim-jobstatus's exits.json), idle lanes are not unused
# capacity and the gate stands down; and a job identical to one dispatched in the last day, apart from its
# output path, does not serve a lane. With no exits.json (a host without the publisher) nothing changes.
EXITS_JSON = os.environ.get("SIM_POOL_EXITS_JSON", "/opt/host-status/sim-pool-jobs/exits.json")
RUNNING_MAX_H = 24
_OUT_RE = re.compile(r"--(?:out|json)[ =](\S+)")
_REASON_RE = re.compile(r"POOL_CHECKED_REASON=(?:\\.|\S)*\s*")


def _norm_cmd(job):
    """A job's command with its checked reason and output path removed (two runs of one config compare equal)."""
    j = _REASON_RE.sub("", job.split("#checked:")[0])
    return " ".join(_OUT_RE.sub("--out _", j).split())


def _dispatches(now_ts, running_path=None):
    """(ts, node, job) from pool.running within RUNNING_MAX_H."""
    path = running_path or os.path.join(_queue_dir(), "pool.running")
    out = []
    try:
        rows = open(path, errors="ignore").read().split("\n")[-600:]
    except OSError:
        return out
    for l in rows:
        parts = l.split("\t", 2)
        if len(parts) != 3:
            continue
        try:
            ts = time.mktime(time.strptime(parts[0], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        if ts >= now_ts - RUNNING_MAX_H * 3600:
            out.append((ts, parts[1], parts[2]))
    return out


def _pool_saturated(now_ts, running_path=None, exits_path=None, nodes=None):
    """True iff every default pool node has a dispatched job with no published exit since its dispatch."""
    try:
        exits = json.load(open(exits_path or EXITS_JSON)).get("exits") or []
    except (OSError, ValueError, AttributeError):
        return False
    if nodes is None:
        import pool_nodes
        nodes = pool_nodes.default_nodes()
    if not nodes:
        return False
    done = {}
    for e in exits:
        if isinstance(e, dict) and e.get("out"):
            k = (e.get("node"), e["out"])
            done[k] = max(done.get(k, 0), float(e.get("ts") or 0))
    busy = set()
    for ts, node, job in _dispatches(now_ts, running_path):
        m = _OUT_RE.search(job)
        if m and done.get((node, m.group(1)), 0) < ts:
            busy.add(node)
        elif not m and ts >= now_ts - RECENT_DISPATCH_MIN * 60:
            busy.add(node)
    return all(n in busy for n in nodes)


def _drop_repeats(lines, now_ts, running_path=None):
    """Drop work lines whose normalized command was already dispatched (another time) within RUNNING_MAX_H."""
    seen = {}
    for ts, _node, job in _dispatches(now_ts, running_path):
        seen.setdefault(_norm_cmd(job), []).append(job)
    keep = []
    for l in lines:
        prior = seen.get(_norm_cmd(l), [])
        if any(_OUT_RE.search(j) and _OUT_RE.search(l) and _OUT_RE.search(j).group(1) != _OUT_RE.search(l).group(1)
               for j in prior):
            continue        # the same config re-run under a new output name: not new research
        keep.append(l)
    return keep


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
    # pool.running holds the same record WITH a date ("YYYY-MM-DD HH:MM:SS<TAB>node<TAB>job"). On srv-host-0 the
    # dispatcher logs to the journal (2026-09-30: sim-pool-dispatch.service, no dispatch.log), so this is the file
    # that still says what went out recently there.
    pr = os.path.join(queue_dir, "pool.running")
    if os.path.exists(pr):
        cutoff = time.time() - RECENT_DISPATCH_MIN * 60
        try:
            if os.path.getmtime(pr) >= cutoff:
                for l in open(pr, errors="ignore").read().split("\n")[-200:]:
                    parts = l.split("\t", 2)
                    if len(parts) != 3:
                        continue
                    try:
                        ts = time.mktime(time.strptime(parts[0], "%Y-%m-%d %H:%M:%S"))
                    except ValueError:
                        continue
                    if ts >= cutoff:
                        lines.append(parts[2].split("#checked:")[0])
        except OSError:
            pass
    return _drop_repeats(lines, time.time())


LANE_BUILD_MAX_H = 6
_LANE_BUILDS = os.path.join(_ROOT, "research", "coordination", "lane_builds.jsonl")


def _branch_exists(branch):
    """A registered build must name a REAL branch (2026-09-27: a local model registered two builds whose branches did not
    exist, with notes claiming "6-seed run queued on the pool" while the queue was empty). Local or origin ref."""
    for ref in ("refs/heads/%s" % branch, "refs/remotes/origin/%s" % branch):
        try:
            r = subprocess.run(["git", "-C", _ROOT, "rev-parse", "--verify", "--quiet", ref],
                               capture_output=True, text=True, timeout=10)
        except Exception:
            continue
        if r.returncode == 0:
            return True
    return False


def _build_served(now_ts, path=None, branch_exists=None):
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
                and 0 <= now_ts - float(e.get("started", -1)) <= LANE_BUILD_MAX_H * 3600
                and (branch_exists or _branch_exists)(e["branch"])):
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
    staged = _staged_files()
    if _is_doc_only(staged):
        return []
    # INFRA-ONLY EXEMPTION (2026-09-29): the same allow-list gates/compute_idle_persistent uses (tools/** and tests/**
    # minus the enforcement machinery, generic docs, .gitignore). A tooling commit queues no research and adds no
    # research artifact, so idle lanes are not its fault; blocking it is what pushed agents to queue filler.
    if _is_infra_only(staged):
        return []
    # RESULT-DATA-ONLY EXEMPTION (2026-10-10): a commit staging ONLY research/findings/raw/** lands the output of
    # jobs that ALREADY RAN; it queues nothing and stops nothing, so idle lanes are not its fault. With the Hermes
    # driver, lanes sit idle while their next work waits on a Claude ruling, and blocking result landings only
    # delayed the record and burned the 6h waiver budget meant for genuine idle stretches (3 blocks on 10-09/10).
    # A commit that also touches code, runners, findings prose or queues still faces the full gate.
    if _is_result_data_only(staged):
        return []
    idle = sorted(set(CPU_LANES) - _served(_work_lines()) - _build_served(time.time()))
    if len(idle) < MAX_IDLE_LANES:
        return []
    if _pool_saturated(time.time()):
        return []           # every pool node is running work: idle lanes are not unused capacity
    verdict = wh.evaluate(NAME, _waiver_file(), LANE_WAIVER_MAX_H, now_ts=time.time())
    return _idle_message(idle, verdict)


def _is_result_data_only(staged):
    """A non-empty staged set that is ALL under research/findings/raw/ (result data + provenance sidecars)."""
    return bool(staged) and all(p.startswith("research/findings/raw/") for p in staged)


def _is_infra_only(staged):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cip_for_lane", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_idle_persistent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._is_infra_only(staged)


def selftest():
    """FAILING DIRECTION FIRST: with nothing queued and no waiver, this MUST fire."""
    bad = []
    filler = ("POOL_CHECKED_REASON=lane-service\\ only\\ (CLASS\\ L):\\ cheap\\ --smoke\\ dev-seed\\ 7\\ keepalive\\ for\\ lane\\ A "
              "SIM_BACKEND=numpy .venv/bin/python -m research.runners._affect_state_region_derisk --smoke --seed 7")
    if _served([filler]):
        bad.append("FALSE NEGATIVE-PROOF: a self-declared keepalive/lane-service filler job still served a lane")
    if not _served(["POOL_CHECKED_REASON=prereg\\ x.md python -m research.runners._affect_state_region_derisk --seeds 42"]):
        bad.append("a genuine prereg'd affect job no longer serves lane A")
    if not _is_infra_only(["tools/foo.py", "tests/test_foo.py"]):
        bad.append("a tools/tests-only commit is not exempt")
    if _is_infra_only(["tools/gates/lane_starvation.py"]) or _is_infra_only(["research/runners/x.py", "tools/a.py"]):
        bad.append("FALSE POSITIVE: enforcement files or research paths were treated as infra-only")
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
    real = lambda b: True  # noqa: E731  (the selftest's registrations name a branch assumed to exist)
    if _build_served(1000.0 + 60, reg, branch_exists=real) != {lane_a}:
        bad.append("did NOT count a fresh, well-formed lane-build registration as serving its lane")
    if _build_served(1000.0 + LANE_BUILD_MAX_H * 3600 + 1, reg, branch_exists=real):
        bad.append("FALSE NEGATIVE-PROOF: an EXPIRED lane-build registration still served its lane")
    # 2026-09-27: a registration naming a branch that does NOT exist must not serve its lane
    if _build_served(1000.0 + 60, reg, branch_exists=lambda b: False):
        bad.append("counted a registration whose branch does not exist (the 2026-09-27 fabricated-build hole)")
    if _build_served(1000.0 + 60, reg):  # real git: 'research/x' does not exist in this repo
        bad.append("real-git branch check accepted the non-existent branch 'research/x'")
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
    if not _is_result_data_only(["research/findings/raw/_x/a.json", "research/findings/raw/_x/a.json.prov.json"]):
        bad.append("a result-data-only staged set was NOT recognised as exempt")
    if (_is_result_data_only(["research/findings/raw/_x/a.json", "research/runners/x.py"])
            or _is_result_data_only(["research/findings/raw/_x/a.json", "research/queue/pool.queue"])
            or _is_result_data_only([])):
        bad.append("BROKEN GUARD: a mixed or empty set was treated as result-data-only")
    if _is_doc_only([]):
        bad.append("BROKEN GUARD: an EMPTY staged set was treated as doc-only (would exempt every commit)")
    # 2026-10-12: pool saturation + repeat busywork
    import tempfile as _tf
    now = time.time()
    stamp = lambda dt: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - dt))  # noqa: E731
    with _tf.TemporaryDirectory() as d:
        run = os.path.join(d, "pool.running"); ex = os.path.join(d, "exits.json")
        jA = "POOL_CHECKED_REASON=x\\ y python -m research.runners._affect_a --seeds 1 --out r/a1.json"
        jB = "POOL_CHECKED_REASON=x python -m research.runners._curiosity_b --out r/b1.json"
        open(run, "w").write("%s\tn1\t%s\n%s\tn2\t%s\n" % (stamp(7200), jA, stamp(3600), jB))
        json.dump({"exits": [{"node": "n1", "ts": now - 9000, "out": "r/a1.json"}]}, open(ex, "w"))
        if not _pool_saturated(now, run, ex, nodes=["n1", "n2"]):
            bad.append("did NOT read two nodes each running an un-exited job as saturated")
        json.dump({"exits": [{"node": "n1", "ts": now - 60, "out": "r/a1.json"}]}, open(ex, "w"))
        if _pool_saturated(now, run, ex, nodes=["n1", "n2"]):
            bad.append("FALSE POSITIVE: a node whose job already exited still read as busy")
        if _pool_saturated(now, run, ex, nodes=["n1", "n2", "n3"]):
            bad.append("FALSE POSITIVE: a node with no job at all still read as saturated")
        if _pool_saturated(now, run, os.path.join(d, "missing.json"), nodes=["n1", "n2"]):
            bad.append("BROKEN GUARD: no exits.json must never exempt (old behaviour)")
        rerun = jA.replace("r/a1.json", "r/a1b.json")
        if _drop_repeats([rerun], now, run):
            bad.append("FALSE NEGATIVE-PROOF: the same config under a new output name still served a lane")
        if not _drop_repeats([jA.replace("--seeds 1", "--seeds 2").replace("a1", "a2")], now, run):
            bad.append("a genuinely different config (new seeds) was dropped as a repeat")
    return bad
