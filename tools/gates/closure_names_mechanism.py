"""CLASS CM — A CLOSURE CLAIM THAT NAMES NO MECHANISM, SO NOTHING CAN ADJUDICATE IT.

THE DEFECT, measured 2026-07-31. The gap#4 record held two LIVE, CONTRADICTORY findings for seventeen days:

    2026-07-14 06:55  "REFUTED — pooling ALREADY works but does NOT lift the accuracy"   (OnBridgeBDSPNet)
    2026-07-14 14:11  "COMPLETE, POSITIVE CLOSURE — K=8 reaches the LIF ceiling"          (OnBridgeEpropNet)

Neither declared a `status:`. Neither declared a `mechanism:`. So nothing in the system could tell they were
about the same question, let alone which was current — and the master roadmap, last synced ten days after
both, went on calling gap#4 "the single load-bearing dependency" while containing zero mentions of the rule
one of them says closed it. A nine-hour, eight-cell GPU crux was then launched into that gap.

THE MACHINERY TO ADJUDICATE THIS ALREADY EXISTS AND WAS SIMPLY UNPOINTED. `biology_check`'s
`check_mechanism_status` enforces one-mechanism-one-`current_finding`, and forces every other live claim on
that mechanism to resolve. It works. It just had no `research/biology/` entry for deep-credit-on-spikes,
because nothing ever required one to exist. Seventeen days of contradiction sat one missing file away from
being mechanically impossible.

WHAT THIS GATE ENFORCES, narrowly and only on newly-added findings: a finding that claims a CLOSURE — CLOSED,
COMPLETE, SURPASSED, SOLVED, or a GO verdict — must declare a `mechanism:` in its frontmatter. That forces
the registry entry into existence exactly where a contradiction is most expensive, without demanding one for
every routine measurement.

DELIBERATELY NOT ENFORCED: that ordinary findings name a mechanism. Most measurements are contributions to a
question rather than claims about it, and requiring the field everywhere would make it noise — declared
reflexively and therefore worthless, which is how a taxonomy dies.

WHAT IT CANNOT CATCH: a closure claim that names the WRONG mechanism, and two closures on the same mechanism
where the registry entry is itself stale. Those are `biology_check`'s job once the entry exists — this gate
only guarantees there is something for it to check.
"""
from __future__ import annotations

import os
import re
import tempfile

NAME = "closure-names-mechanism"
CLASS_ID = "CM"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FM_FINDING_RE = re.compile(r"^type:\s*finding\s*$", re.M)
FM_MECHANISM_RE = re.compile(r"^mechanism:\s*\S", re.M)

# A CLOSURE claim, anchored to verdict-shaped positions so ordinary prose does not trip it. A finding that
# merely DISCUSSES a prior GO is not claiming one; a title or verdict line asserting closure is.
CLOSURE_RE = re.compile(
    r"^#[^\n]*\b(?:CLOSED|COMPLETE|SURPASSED|SOLVED)\b"                    # in the title
    r"|^\*\*Verdict:\*\*[^\n]*\b(?:GO|CLOSED|COMPLETE|SURPASSED|SOLVED)\b"  # in a verdict line
    r"|^\s*\*\*(?:VERDICT|RESULT)[:\s][^\n]*\b(?:GO|CLOSED|COMPLETE|SURPASSED)\b"
    r"|\bPOSITIVE CLOSURE\b|\bCOMPLETE,?\s+POSITIVE\b",
    re.M)
# A withdrawal is not a closure claim, however emphatic.
NEGATED_RE = re.compile(r"\b(?:NOT|NO-GO|UNDEFINED|RETRACTED|VOID|REFUTED)\b", re.I)


def _frontmatter(text):
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    return text[3:end] if end > 0 else None


def _check_one(path, rel=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return []
    fm = _frontmatter(text)
    if fm is None or not FM_FINDING_RE.search(fm):
        return []                                          # legacy or non-finding: out of scope
    if FM_MECHANISM_RE.search(fm):
        return []                                          # already adjudicable
    hit = CLOSURE_RE.search(text)
    if not hit:
        return []
    claim = hit.group(0).strip()[:70]
    if NEGATED_RE.search(claim):
        return []                                          # "NOT CLOSED", "GO ... UNDEFINED" — not a claim
    return ["%s: claims a CLOSURE (%r) but declares no `mechanism:`. Nothing can then adjudicate it against "
            "other live claims on the same question. On 2026-07-31 two contradictory gap#4 findings sat live "
            "for SEVENTEEN days — one 'REFUTED', one 'COMPLETE POSITIVE CLOSURE' — because neither named a "
            "mechanism and no research/biology/ entry existed for biology_check.check_mechanism_status to "
            "adjudicate. A nine-hour GPU crux was launched into that gap." % (rel, claim)]


def check(paths):
    if paths is None or len(paths) == 0:
        return []                                          # legacy audited on touch, per doc_type's lesson
    problems = []
    for p in [x for x in paths if x.endswith(".md")]:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full, p)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: closure claims without a mechanism, then everything that must NOT fire."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, fm, body):
            p = os.path.join(d, name)
            open(p, "w").write("---\n%s---\n\n%s\n" % (fm, body))
            return p

        F = "type: finding\nstatus: live\n"
        FM = F + "mechanism: deep-credit-on-spikes\n"
        # 1. a closure in the TITLE with no mechanism
        if not _check_one(w("a.md", F, "# The arc is COMPLETE on the production substrate"), "research/findings/a.md"):
            bad.append("did NOT catch a title closure claim with no mechanism")
        # 2. a GO verdict line with no mechanism
        if not _check_one(w("b.md", F, "**Verdict:** GO at 6 seeds."), "research/findings/b.md"):
            bad.append("did NOT catch a GO verdict with no mechanism")
        # 3. the real phrasing from 2026-07-14
        if not _check_one(w("c.md", F, "text\n\nTHE ARC — COMPLETE, POSITIVE CLOSURE: e-prop trains it."),
                          "research/findings/c.md"):
            bad.append("did NOT catch the real 'COMPLETE, POSITIVE CLOSURE' phrasing")
        # 4. NEGATIVE CONTROL — declaring a mechanism satisfies it.
        if _check_one(w("d.md", FM, "# The arc is COMPLETE"), "research/findings/d.md"):
            bad.append("FALSE POSITIVE: flagged a closure that DOES name its mechanism")
        # 5. NEGATIVE CONTROL — a withdrawal is not a closure claim.
        if _check_one(w("e.md", F, "**Verdict:** UNDEFINED, NOT a NO-GO."), "research/findings/e.md"):
            bad.append("FALSE POSITIVE: flagged an UNDEFINED verdict as a closure")
        # 6. NEGATIVE CONTROL — a negated title is not a claim.
        if _check_one(w("f.md", F, "# The task is NOT SOLVED on this substrate"), "research/findings/f.md"):
            bad.append("FALSE POSITIVE: flagged a NOT-SOLVED title as a closure")
        # 7. NEGATIVE CONTROL — an ordinary measurement needs no mechanism.
        if _check_one(w("g.md", F, "# Held-out accuracy rises to 0.61 under the expander"),
                      "research/findings/g.md"):
            bad.append("FALSE POSITIVE: flagged an ordinary finding with no closure claim")
        # 8. NEGATIVE CONTROL — legacy (no frontmatter) is out of scope.
        p = os.path.join(d, "h.md")
        open(p, "w").write("# The arc is COMPLETE\n")
        if _check_one(p, "research/findings/h.md"):
            bad.append("FALSE POSITIVE: flagged a legacy no-frontmatter document")
        # 9. SCOPING
        if check(None) or check([]):
            bad.append("SCOPE LEAK: standalone/empty mode must not scan the legacy corpus")
    return bad
