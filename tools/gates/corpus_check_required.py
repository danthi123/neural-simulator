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
"""
from __future__ import annotations

import json
import os
import tempfile

NAME = "corpus-check-required"
CLASS_ID = "CC"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ELAPSED_KEYS = ("elapsed_seconds", "elapsed_s", "elapsed", "runtime_seconds", "wall_seconds")
MIN_COST_S = 3600.0            # an hour. Below this, re-deriving costs less than the check's friction.


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

    fresh = obj.get("corpus_check_fresh")
    if fresh is None:
        side = _sidecar(path)
        fresh = side.get("corpus_check_fresh") if isinstance(side, dict) else None
    if fresh:
        return []
    return ["%s: records %.1fh of compute with NO recent corpus check (`corpus_check_fresh` absent or false, "
            "in the artifact and its provenance sidecar). Run `bash tools/before_you_build.sh \"<the "
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
    return bad
