"""CLASS V — AN ARTIFACT ASSERTS A VERDICT WITHOUT CARRYING WHAT EARNED IT.

THE GAP THIS CLOSES, and why it needed a new shape. Every other gate here scans FILES at COMMIT time. Every
miss on 2026-07-31 was a RELATIONSHIP at RUN time, which no file-scanner can see:

  * the task stopped being depth-REQUIRED when the forward changed (gap#4)
  * the runner computed `arm_valid=False` on 3/3 seeds and printed "NO-GO" anyway (affect eviction)
  * the idealised transport ceiling read 0.148 against a chance of 0.200 (the gap#4 crux)
  * the power control zeroed a SYNAPTIC gate while the mechanism lived in per-neuron Izhikevich params (sAHP)
  * `--sweep-weights` was accepted by argparse and silently never reached the code path taken

Five plausible NEGATIVES, each of which would have entered the record clean.

THE BRIDGE. `tools/verdict.Verdict` makes the runtime do the seeing and emit the evidence into the artifact
as a `preconditions` block. This gate then enforces its PRESENCE, which IS a file property and therefore
gateable. Runtime sees; artifact carries; gate enforces.

WHAT IT ENFORCES on an artifact that asserts a verdict (`go`, `GO`, `verdict` or `status` at top level):
  1. a `preconditions` list must be present and non-empty — an unguarded verdict is the defect itself;
  2. no precondition may be unmeasured (`ok: null`) while a GO/NO-GO is asserted;
  3. if any precondition failed (`ok: false`), the status must be UNDEFINED, never GO or NO-GO. This is the
     affect case exactly: the runner HAD the failing value and asserted a negative beside it.

SCOPED TO NEWLY-ADDED ARTIFACTS. 12,000+ banked artifacts predate `Verdict` and carry no preconditions
block; flagging them would emit thousands of hits and get the gate switched off, which is strictly worse
than no gate — the lesson doc-type learned when a Tier-1 classification pulled 192 legacy findings into
scope in one commit. Legacy is retrofitted on next touch.

WHAT IT CANNOT CATCH: whether the preconditions registered were the RIGHT ones. Noticing that a sub-chance
ceiling matters is judgement, and this gate does not pretend otherwise — it enforces that whatever the run
DID check travels with the verdict it produced.
"""
from __future__ import annotations

import json
import os
import tempfile

NAME = "verdict-preconditions"
CLASS_ID = "V"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERDICT_KEYS = ("go", "GO", "verdict", "status")
UNDEFINED_TOKENS = ("undefined", "void", "uninterpretable")


def _asserts_verdict(obj):
    """A top-level GO/NO-GO assertion. A status that is already UNDEFINED asserts nothing to earn.

    STRING VERDICTS ARE READ FIRST, and this ordering is load-bearing. A correctly-behaving artifact says
    BOTH `verdict: "UNDEFINED — ..."` AND `GO: false`, because "undefined" necessarily implies "not a GO".
    Reading the boolean first mistakes that agreement for an asserted negative and flags a run that did
    exactly the right thing. This gate's own first real artifact tripped that — a false positive, which
    this project treats as no less corrosive than a miss, because it trains the reader to skip the line."""
    if not isinstance(obj, dict):
        return False, None
    # pass 1 — an explicit textual verdict is authoritative over any companion boolean
    for k in ("verdict", "status"):
        v = obj.get(k)
        if isinstance(v, str):
            if any(t in v.lower() for t in UNDEFINED_TOKENS):
                return False, None          # already refusing to assert — nothing to enforce
            if any(t in v.upper() for t in ("GO", "NEGATIVE", "PASS", "FAIL", "BOUNDARY")):
                return True, v
    # pass 2 — only a bare boolean, with no textual verdict to qualify it
    for k in ("go", "GO"):
        if isinstance(obj.get(k), bool):
            return True, obj[k]
    return False, None


def _check_one(path, rel=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    try:
        obj = json.load(open(path, errors="ignore"))
    except (OSError, ValueError):
        return []                                   # not our business; artifact_provenance owns readability
    asserts, val = _asserts_verdict(obj)
    if not asserts:
        return []
    pre = obj.get("preconditions")
    if not isinstance(pre, list) or not pre:
        return ["%s: asserts a verdict (%r) but carries NO `preconditions` block. A verdict must travel "
                "with what earned it — use tools.verdict.Verdict, whose to_dict() emits one. An unguarded "
                "verdict is the defect: on 2026-07-31 five plausible negatives came from unchecked "
                "preconditions." % (rel, val)]
    problems = []
    unmeasured = [c.get("name") for c in pre if isinstance(c, dict) and c.get("ok") is None]
    failed = [c.get("name") for c in pre if isinstance(c, dict) and c.get("ok") is False]
    if unmeasured:
        problems.append("%s: asserts a verdict while %d precondition(s) were NEVER MEASURED (%s). "
                        "Unmeasured is not passed." % (rel, len(unmeasured), ", ".join(map(str, unmeasured[:3]))))
    if failed:
        problems.append("%s: asserts %r while %d precondition(s) FAILED (%s). A run whose preconditions do "
                        "not hold yields UNDEFINED, never a negative — this is the affect-eviction case, "
                        "where arm_valid=False sat one key away from the word NO-GO."
                        % (rel, val, len(failed), ", ".join(map(str, failed[:3]))))
    return problems


def check(paths):
    # EMPTY list = staged mode, nothing of mine staged. Only paths=None means standalone, and standalone
    # deliberately checks NOTHING: the 12,000+ legacy artifacts predate Verdict, and a corpus fallback here
    # would undo the hook's --diff-filter=A scoping exactly as doc-type once did.
    if paths is None or len(paths) == 0:
        return []
    problems = []
    for p in [x for x in paths if x.endswith(".json")]:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full, p)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: the three ways a verdict can be unearned, then the negative controls."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, obj):
            p = os.path.join(d, name)
            json.dump(obj, open(p, "w"))
            return p

        # 1. a verdict with NO preconditions at all
        if not _check_one(w("a.json", {"go": True}), "raw/a.json"):
            bad.append("did NOT catch a verdict with no preconditions block")
        # 2. asserting a verdict while a precondition was never measured
        p = w("b.json", {"verdict": "NO-GO", "preconditions": [{"name": "chance", "ok": None}]})
        if not any("NEVER MEASURED" in x for x in _check_one(p, "raw/b.json")):
            bad.append("did NOT catch an unmeasured precondition under an asserted verdict")
        # 3. THE AFFECT CASE: a failing precondition sitting beside an asserted negative
        p = w("c.json", {"verdict": "NO-GO / BOUNDARY",
                         "preconditions": [{"name": "arm_valid", "ok": False}]})
        if not any("FAILED" in x for x in _check_one(p, "raw/c.json")):
            bad.append("did NOT catch an asserted NO-GO whose precondition FAILED")
        # 4. NEGATIVE CONTROL — a properly earned verdict must pass.
        p = w("d.json", {"go": True, "preconditions": [{"name": "chance", "ok": True}]})
        if _check_one(p, "raw/d.json"):
            bad.append("FALSE POSITIVE: flagged a verdict whose preconditions all hold")
        # 5. NEGATIVE CONTROL — a run already reporting UNDEFINED asserts nothing and needs no block.
        p = w("e.json", {"verdict": "UNDEFINED — the arm was crushed"})
        if _check_one(p, "raw/e.json"):
            bad.append("FALSE POSITIVE: flagged an artifact that already refuses to assert a verdict")
        # 6. NEGATIVE CONTROL — an artifact with no verdict at all is out of scope.
        p = w("f.json", {"means": {"acc": 0.5}})
        if _check_one(p, "raw/f.json"):
            bad.append("FALSE POSITIVE: flagged an artifact that asserts no verdict")
        # 7. THE FALSE POSITIVE THIS GATE HIT ON ITS OWN FIRST REAL ARTIFACT: a correct run says BOTH
        #    verdict:"UNDEFINED ..." AND GO:false, because undefined implies not-a-GO. Reading the boolean
        #    first mistook that agreement for an asserted negative and flagged a run that behaved perfectly.
        p = w("g.json", {"verdict": "UNDEFINED (3-seed) — unmet: A5 the arm was CRUSHED", "GO": False,
                         "preconditions": [{"name": "A5", "ok": False}]})
        if _check_one(p, "raw/g.json"):
            bad.append("FALSE POSITIVE: flagged an artifact that correctly reports UNDEFINED alongside GO:false")
        # 8. and the converse must still fire — a bare boolean negative with a FAILED precondition and no
        #    textual verdict to qualify it is the original defect and must not slip through the new ordering.
        p = w("h.json", {"go": False, "preconditions": [{"name": "A5", "ok": False}]})
        if not _check_one(p, "raw/h.json"):
            bad.append("did NOT catch a bare boolean negative asserted over a FAILED precondition")
        # 9. SCOPING — standalone/empty must check nothing, or legacy artifacts flood it.
        if check(None) or check([]):
            bad.append("SCOPE LEAK: standalone/empty mode must not scan the legacy corpus")
    return bad
