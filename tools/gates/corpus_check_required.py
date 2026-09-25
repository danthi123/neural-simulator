"""CLASS CC — AN EXPENSIVE RUN WHOSE QUESTION WAS NEVER CHECKED AGAINST THE RECORD.

THE DEFECT, and it is the most expensive one measured on 2026-07-31. A nine-hour, eight-cell GPU crux was
launched against "does deep credit train on the on-bridge spiking forward". That question was already
answered:

  2026-07-07  the depth-2 spiking net does NOT train this task — all arms below chance, ALL SIX SEEDS,
              oracle 1.0. The identical signature the crux spent nine hours reproducing.
  2026-07-14  the cause, LOCATED: Izhikevich forward NOISE at full-task scale — not the rule, not epochs
              (300 epochs moved train 0.482 -> 0.497).
  2026-07-08  population coding swept K in {1,8,16}: no crossover.
  2026-07-12  the negative repeated at depth 2.

`tools/before_you_build.sh` returns all four in **0.63 seconds**. It was not run before launch. The 1-seed
replication was then written up as a new localisation, and only a post-hoc corpus check caught it.

WHY REPORTING WAS NOT ENOUGH, which is the whole argument for making this block. The heartbeat printed
"⛔ A FINDING WAS WRITTEN SINCE THE LAST SOURCE CHECK" roughly FIFTEEN times that day and was read past
every single time. That is the same shape as `lane_starvation`, where a true alarm ran for 194 minutes
unheeded and had to be made blocking. An alarm nobody acts on is not coverage.

EVERY OTHER GATE HERE LOOKS FOR A *WRONG* CLAIM. This one looks for a *REDUNDANT* one — and redundancy is
the more expensive failure, because a wrong claim gets caught downstream while a redundant one quietly
burns GPU-hours and produces a finding that reads perfectly well.

WHAT IT ENFORCES, on newly-added artifacts only: a run recording more than `MIN_COST_S` of compute must
carry evidence that the record was consulted — `corpus_check_fresh` in the artifact or its provenance
sidecar, stamped automatically by `research/runners/__init__` from the log that
`tools/before_you_build.sh` now writes. Cheap runs are exempt: the cost of re-deriving a two-minute smoke is
two minutes, and a gate that fires on those gets switched off.

WHAT IT CANNOT CATCH: a corpus check that was RUN and not READ. The check records that the question was
asked, never that the answer was understood — on 2026-07-31 the priors were one command away and the
failure was not looking, but a future failure could equally be looking and not reading. That is judgement,
and it is left as judgement rather than pretended away.

FALLBACK FOR AN UN-STAMPED ARTIFACT (research/corpus-check-shared-log, 2026-09-25). `corpus_check_fresh` is
only ever written by the provenance door for a path named on argv (`--out`/`--output`/`--json`) or registered
via `declare_output()` — a side artifact a runner writes directly (a checkpoint into `--ckpt-dir`, say) gets no
sidecar at all, so `fresh` reads `None`: not "checked and stale", just never given the chance to say. Blocking
that artifact unconditionally would be correct-but-blunt; instead, for `fresh is None` ONLY (an explicit
`corpus_check_fresh: false` still blocks outright — that artifact WAS able to carry a stamp and didn't), this
looks for direct evidence that `before_you_build.sh` ran before this artifact's OWN run started: the newest
matching entry in the shared log (`research/runners._shared_corpus_check_log`) or any surviving PER-WORKTREE
legacy log (`.claude/worktrees/*/research/queue/.corpus_checks.jsonl`, written before that log was made
shared) dated at or before the run's start and inside the freshness window. Like the direct stamp it stands in
for, **this proves a check happened before the run — never that its query was ON-TOPIC for this artifact's
question.** Nothing here (or in the original mechanism) reads the query text; a human still has to.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime

NAME = "corpus-check-required"
CLASS_ID = "CC"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ELAPSED_KEYS = ("elapsed_seconds", "elapsed_s", "elapsed", "runtime_seconds", "wall_seconds")
MIN_COST_S = 3600.0            # an hour. Below this, re-deriving costs less than the check's friction.
FRESHNESS_WINDOW_S = 24.0 * 3600.0   # matches research/runners._corpus_check_state's default max_age_h
# LEGACY, per-worktree logs from before this log was made shared -- a real check that happened to run in a
# worktree that has since been deleted or never wrote to the new shared location. Hardcoded to this one
# machine's checkout, matching the rest of this single-owner repo's tooling (tools/parallel_audit.py,
# tools/pool_queue.sh, ...).
_WORKTREE_LOG_GLOB = "/home/dant123/Projects/sim/.claude/worktrees/*/research/queue/.corpus_checks.jsonl"


def _find(obj, keys, depth=0):
    if not isinstance(obj, dict):
        return None
    for k, v in obj.items():
        if k.lower() in keys and v is not None and not isinstance(v, (dict, list)):
            return v
    if depth < 2:
        for v in obj.values():
            if isinstance(v, dict):
                got = _find(v, keys, depth + 1)
                if got is not None:
                    return got
    return None


def _sidecar(path):
    for sib in (path + ".prov.json", os.path.splitext(path)[0] + ".prov.json"):
        if os.path.exists(sib):
            try:
                return json.load(open(sib, errors="ignore"))
            except (OSError, ValueError):
                return None
    return None


def _shared_log_path():
    """The ONE corpus-check log `tools/before_you_build.sh` writes, at the git COMMON dir root -- the same
    resolution `research/runners._shared_corpus_check_log` uses, so a run's own sidecar and this gate's
    fallback agree on where the evidence lives. `SIM_CORPUS_CHECK_LOG` overrides both, for hermetic tests."""
    override = os.environ.get("SIM_CORPUS_CHECK_LOG")
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "-C", _ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        out = ""
    return os.path.join(out, "corpus_checks_shared.jsonl") if out else None


def _iter_log_entries(path):
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def _has_timezone(value):
    """True only for an ISO string that names its own offset (a trailing Z, or +HH:MM / -HH:MM) -- a naive
    string like `time.strftime(..., time.localtime(x))` produces has neither, and parsing it as if it did is
    exactly the bug this replaces (a hostless/AWS node's wall clock is UTC but the string never says so, and
    re-parsing it as this machine's local zone read +4h late)."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    return v.endswith("Z") or bool(re.search(r"[+-]\d{2}:\d{2}$", v))


def _run_start_epoch(obj, side):
    """WHEN this artifact's producing run started, from the artifact itself or its sidecar (if any), in the
    only order that is safe: an explicit UTC-nanosecond timestamp, then the UTC epoch second embedded in
    `run_id` (`research/runners._record_start` mints it as `"%d-%d" % (int(_START), pid)` -- unambiguous
    because `time.time()` is already UTC), then an ISO `started` string but ONLY if `_has_timezone` accepts
    it. Anything else FAILS CLOSED: no mtime fallback, deliberately -- a file's mtime is when it was last
    WRITTEN, not when the run that produced it began, and a checkpoint written well after its run started
    (mid-training) would understate the run's age, silently admitting a check that ran too late."""
    for src in (obj, side):
        if not isinstance(src, dict):
            continue
        v = src.get("started_utc_ns")
        if isinstance(v, (int, float)) and v > 0:
            return v / 1e9
    for src in (obj, side):
        if not isinstance(src, dict):
            continue
        v = src.get("run_id")
        if isinstance(v, str) and "-" in v:
            try:
                epoch = int(v.split("-", 1)[0])
            except ValueError:
                epoch = None
            if epoch and epoch > 0:
                return float(epoch)
    for src in (obj, side):
        if not isinstance(src, dict):
            continue
        v = src.get("started")
        if _has_timezone(v):
            try:
                return datetime.fromisoformat(v.strip().replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
    return None


def _fallback_evidence(run_start):
    """Search the shared log and any surviving per-worktree legacy log for the NEWEST entry dated at or
    before RUN_START and inside FRESHNESS_WINDOW_S. Returns (evidence_file, entry), or (None, None) if
    RUN_START is unknown or nothing qualifies. A check logged AFTER the run started proves nothing about
    whether it was consulted beforehand, so `when > run_start` entries are excluded outright, not merely
    deprioritized."""
    if run_start is None:
        return None, None
    best_file, best_entry = None, None
    candidates = [_shared_log_path()] + sorted(glob.glob(_WORKTREE_LOG_GLOB))
    for log_path in candidates:
        for entry in _iter_log_entries(log_path):
            try:
                when = float(entry.get("when"))
            except (TypeError, ValueError):
                continue
            if when > run_start or (run_start - when) > FRESHNESS_WINDOW_S:
                continue
            if best_entry is None or when > float(best_entry.get("when", -1)):
                best_file, best_entry = log_path, entry
    return best_file, best_entry


def _check_one(path, rel=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    if rel.endswith(".prov.json") or rel.endswith(".cmd.json"):
        return []
    try:
        obj = json.load(open(path, errors="ignore"))
    except (OSError, ValueError):
        return []
    if not isinstance(obj, dict):
        return []

    elapsed = _find(obj, ELAPSED_KEYS)
    try:
        elapsed = float(elapsed) if elapsed is not None else None
    except (TypeError, ValueError):
        elapsed = None
    if elapsed is None or elapsed <= MIN_COST_S:
        return []                                          # cheap or untimed: out of scope by design

    side = _sidecar(path)
    fresh = obj.get("corpus_check_fresh")
    if fresh is None and isinstance(side, dict):
        fresh = side.get("corpus_check_fresh")
    if fresh:
        return []
    if fresh is None:
        # NO STAMP AT ALL (never "checked and stale" -- never given the chance to say). Try the fallback
        # described in the module docstring before concluding this run had no evidence.
        run_start = _run_start_epoch(obj, side if isinstance(side, dict) else None)
        evidence_file, evidence_entry = _fallback_evidence(run_start)
        if evidence_entry is not None:
            print("[corpus-check-required] %s: no direct stamp, but ACCEPTED via fallback evidence in %s "
                  "(checked %s, query: %r). This proves a check happened before the run, not that it was "
                  "on-topic -- same strength as the direct stamp." % (
                      rel, evidence_file,
                      evidence_entry.get("iso") or evidence_entry.get("when"),
                      str(evidence_entry.get("query", ""))[:80]))
            return []
    return ["%s: records %.1fh of compute with NO recent corpus check (`corpus_check_fresh` absent or false, "
            "in the artifact and its provenance sidecar, and no fallback evidence in the shared or any "
            "worktree corpus-check log before this run started). Run `bash tools/before_you_build.sh \"<the "
            "question>\"` — it returns the priors in under a second. On 2026-07-31 a nine-hour eight-cell "
            "crux re-derived a SIX-SEED result banked three weeks earlier whose root cause was already "
            "located, and the heartbeat's advisory warning was read past ~15 times that day."
            % (rel, elapsed / 3600.0)]


def check(paths):
    if paths is None or len(paths) == 0:
        return []                                          # legacy predates the stamp; audited on touch
    problems = []
    for p in [x for x in paths if x.endswith(".json")]:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full, p)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: the expensive unchecked run, then everything that must NOT fire."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, obj):
            p = os.path.join(d, name)
            json.dump(obj, open(p, "w"))
            return p

        # 1. THE REAL CASE: a 9-hour run with no corpus check.
        if not _check_one(w("a.json", {"elapsed_seconds": 9 * 3600}), "raw/a.json"):
            bad.append("did NOT catch an expensive run with no corpus check")
        # 2. explicitly stale must fire too, not just absent.
        if not _check_one(w("b.json", {"elapsed_seconds": 9 * 3600, "corpus_check_fresh": False}), "raw/b.json"):
            bad.append("did NOT catch an expensive run whose corpus check was STALE")
        # 3. NEGATIVE CONTROL — a checked expensive run passes, or nobody can satisfy the gate.
        if _check_one(w("c.json", {"elapsed_seconds": 9 * 3600, "corpus_check_fresh": True}), "raw/c.json"):
            bad.append("FALSE POSITIVE: flagged an expensive run that DID check the corpus")
        # 4. NEGATIVE CONTROL — a CHEAP run is out of scope; re-deriving a smoke costs a smoke.
        if _check_one(w("d.json", {"elapsed_seconds": 120}), "raw/d.json"):
            bad.append("FALSE POSITIVE: flagged a cheap run")
        # 5. NEGATIVE CONTROL — an untimed artifact cannot be judged expensive.
        if _check_one(w("e.json", {"means": {"acc": 0.5}}), "raw/e.json"):
            bad.append("FALSE POSITIVE: flagged an artifact with no elapsed time")
        # 6. NEGATIVE CONTROL — the sidecar may carry the evidence instead of the artifact.
        p = w("f.json", {"elapsed_seconds": 9 * 3600})
        json.dump({"corpus_check_fresh": True}, open(p + ".prov.json", "w"))
        if _check_one(p, "raw/f.json"):
            bad.append("FALSE POSITIVE: ignored a provenance sidecar carrying the corpus check")
        # 7. NEGATIVE CONTROL — sidecars are evidence, not subjects.
        if _check_one(p + ".prov.json", "raw/f.json.prov.json"):
            bad.append("FALSE POSITIVE: audited a provenance sidecar as a result")
        # 8. SCOPING — standalone/empty scans nothing.
        if check(None) or check([]):
            bad.append("SCOPE LEAK: standalone/empty mode must not scan the legacy corpus")

        # 9-11. THE FALLBACK (research/corpus-check-shared-log, 2026-09-25): an un-stamped artifact whose
        # run_id embeds a run-start epoch, checked against SIM_CORPUS_CHECK_LOG so this stays hermetic
        # regardless of what the real repo's shared/worktree logs happen to hold. A run-start far in 2001
        # keeps the negative case (10) safe from ever colliding with a real 2026 worktree log entry.
        _old_env = os.environ.get("SIM_CORPUS_CHECK_LOG")
        try:
            run_start = 1000000000.0                                          # 2001-09-09, an arbitrary past epoch
            no_stamp = {"elapsed_seconds": 9 * 3600, "run_id": "%d-999" % int(run_start)}
            shared_log = os.path.join(d, "shared.jsonl")
            os.environ["SIM_CORPUS_CHECK_LOG"] = shared_log
            # 9. a qualifying entry logged BEFORE run_start, inside the window -> ACCEPTED.
            with open(shared_log, "w") as fh:
                fh.write(json.dumps({"when": run_start - 3600, "query": "selftest fallback probe"}) + "\n")
            if _check_one(w("g.json", no_stamp), "raw/g.json"):
                bad.append("FALSE POSITIVE: fallback rejected a qualifying pre-run-start shared-log entry")
            # 10. NEGATIVE CONTROL — no evidence anywhere (log absent entirely) must still block.
            os.remove(shared_log)
            if not _check_one(w("h.json", dict(no_stamp)), "raw/h.json"):
                bad.append("did NOT catch an expensive un-stamped run with NO fallback evidence anywhere")
            # 11. NEGATIVE CONTROL — a check logged AFTER run_start proves nothing and must not pass.
            with open(shared_log, "w") as fh:
                fh.write(json.dumps({"when": run_start + 3600, "query": "too late"}) + "\n")
            if not _check_one(w("i.json", dict(no_stamp)), "raw/i.json"):
                bad.append("FALSE POSITIVE: accepted a corpus check logged AFTER the run started")
        finally:
            if _old_env is None:
                os.environ.pop("SIM_CORPUS_CHECK_LOG", None)
            else:
                os.environ["SIM_CORPUS_CHECK_LOG"] = _old_env
    return bad
