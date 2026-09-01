"""CLASS SV — A FINDING STATES A NAMED QUANTITY THAT DISAGREES WITH THE ARTIFACT IT CITES.

THE DEFECT, and it is mine from 2026-07-31. I wrote "chance is 0.200 (k=5)" into a crux verdict and its
commit message. The artifact says `chance=0.167` — the runner PRINTS its own chance and I derived `1/k`
instead of reading it. The conclusion survived (the ceiling at 0.148 is below either value) but two claims
did not: the margin I quoted was 2.7x too generous, and "every arm at or below chance" was FALSE, since a
fixed random reservoir at 0.204 and one kp seed at 0.222 were ABOVE it.

WHY NOTHING CAUGHT IT. `claim_check` verifies that every number in a finding EXISTS somewhere in a cited
artifact. 0.200 existed — as `chance_1_over_k`, a different quantity. Existence is not agreement. The gate
was doing exactly what it promised and the promise was too weak: it checks that a number is real, not that
the number attached to a NAME is the value the run gave that name.

WHAT THIS GATE ENFORCES. For a finding citing artifacts, when the prose states `<name> <number>` and the
artifact carries a top-level key of the same name, the values must agree. Deliberately narrow:

  * only a small set of names that carry verdicts here — chance, floor, baseline, majority, n_seeds — since
    a general "any word near any number" rule would fire constantly on prose;
  * only TOP-LEVEL artifact keys, so a deeply-nested homonym cannot create a false conflict;
  * a mismatch is reported with BOTH values, because which one is right is a judgement the gate must not make.

WHAT IT CANNOT CATCH: a wrong quantity that the artifact never names, and a finding that simply omits the
number. It closes the specific hole where the record already knew the answer and the prose disagreed.
"""
from __future__ import annotations

import json
import os
import re
import tempfile

NAME = "stated-value-mismatch"
CLASS_ID = "SV"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Names that carry a verdict in this project. Kept short on purpose: each added word is a false-positive
# surface, and the ones here are exactly those that decide whether a result is interpretable.
WATCHED_NAMES = ("chance", "floor", "baseline", "majority", "n_seeds")
ARTIFACT_RE = re.compile(r"[\w.\-/]+\.(?:jsonl|json)")
FM_RE = re.compile(r"^type:\s*finding\s*$", re.M)


def _stated(text):
    """Pull `<name> <number>` pairs out of prose. Tolerates `is`, `=`, `:`, `of` and markdown emphasis."""
    out = {}
    for nm in WATCHED_NAMES:
        pat = re.compile(r"\b%s\b[\s*_`]*(?:is|=|:|of|was)?[\s*_`]*([0-9]*\.?[0-9]+)" % nm, re.I)
        for m in pat.finditer(text):
            out.setdefault(nm, set()).add(float(m.group(1)))
    return out


def _artifact_values(paths):
    vals = {}
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if not os.path.exists(full):
            continue
        try:
            obj = json.load(open(full, errors="ignore"))
        except (OSError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        for nm in WATCHED_NAMES:
            for k, v in obj.items():                       # TOP-LEVEL only, deliberately
                if k.lower() == nm and isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.setdefault(nm, set()).add(float(v))
    return vals


def _check_one(path, rel=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return []
    if not FM_RE.search(text):
        return []                                          # only findings declare a verdict-bearing claim
    cited = ARTIFACT_RE.findall(text)
    if not cited:
        return []
    have = _artifact_values(cited)
    problems = []
    for nm, stated in _stated(text).items():
        real = have.get(nm)
        if not real:
            continue                                       # artifact never names it — out of scope
        # agreement means ANY stated value matches ANY artifact value for that name (a finding may quote
        # several runs). A mismatch is only reported when NONE of them line up.
        if not any(abs(s - r) <= 1e-4 * max(1.0, abs(r)) for s in stated for r in real):
            problems.append(
                "%s: states %s = %s but the cited artifact(s) report %s = %s. Existence is not agreement — "
                "`claim_check` passes this because the number exists SOMEWHERE. On 2026-07-31 a crux verdict "
                "said chance 0.200 (derived as 1/k) while the run reported 0.167; the conclusion held but the "
                "quoted margin was 2.7x too generous and a companion claim was false."
                % (rel, nm, sorted(stated), nm, sorted(real)))
    return problems


def check(paths):
    if paths is None or len(paths) == 0:
        return []                                          # legacy corpus predates this; audited on touch
    problems = []
    for p in [x for x in paths if x.endswith(".md")]:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full, p)
    return problems


def selftest():
    """FAILING DIRECTION FIRST — replay the real 2026-07-31 mismatch, then the negative controls."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        art = os.path.join(d, "a.json")
        json.dump({"chance": 0.167, "n_seeds": 6}, open(art, "w"))
        rel_art = os.path.relpath(art, _ROOT)

        def w(body):
            p = os.path.join(d, "f.md")
            open(p, "w").write("---\ntype: finding\n---\n\n" + body + "\n\nArtifact: `%s`\n" % rel_art)
            return p

        # 1. THE REAL CASE: prose says 0.200, the run reported 0.167.
        if not _check_one(w("Chance is 0.200 (k=5) and the ceiling sits below it."), "research/findings/f.md"):
            bad.append("did NOT catch a stated chance disagreeing with the artifact")
        # 2. NEGATIVE CONTROL — agreement must pass, or the gate is unusable.
        if _check_one(w("Chance is 0.167 and the ceiling sits below it."), "research/findings/f.md"):
            bad.append("FALSE POSITIVE: flagged a stated value that AGREES with the artifact")
        # 3. NEGATIVE CONTROL — a name the artifact never carries is out of scope.
        if _check_one(w("The majority rate is 0.333 here."), "research/findings/f.md"):
            bad.append("FALSE POSITIVE: flagged a name the artifact does not report")
        # 4. NEGATIVE CONTROL — a finding may quote several runs; ANY match is agreement.
        if _check_one(w("chance 0.167 in the deep case and chance 0.278 in the shallow one."),
                      "research/findings/f.md"):
            bad.append("FALSE POSITIVE: flagged a finding quoting multiple runs where one matches")
        # 5. NEGATIVE CONTROL — non-findings are out of scope entirely.
        p = os.path.join(d, "plan.md")
        open(p, "w").write("---\ntype: plan\n---\n\nchance is 0.200\n\n`%s`\n" % rel_art)
        if _check_one(p, "docs/plans/plan.md"):
            bad.append("FALSE POSITIVE: flagged a non-finding document")
        # 6. SCOPING — standalone/empty must scan nothing.
        if check(None) or check([]):
            bad.append("SCOPE LEAK: standalone/empty mode must not scan the legacy corpus")
    return bad
