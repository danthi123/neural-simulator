"""refuted-mechanism re-proposal gate (RM).

WHY THIS EXISTS. On 2026-08-02 a finding, the board, and BOTH roadmaps were written naming "two-compartment
dendritic credit" as the *remaining surpass* for the gap#4 on-bridge deep-credit residual — a mechanism the
project has tested and refuted REPEATEDLY (Urbanczik-Senn two-compartment + fixed feedback:
`2026-05-17-dendritic-credit-assignment-NEGATIVE`; BDSP/burstprop/microcircuit 6-seed negatives; coincidence-gated
BDSP on real spikes `2026-08-01`; and a finding literally titled `2026-07-22-gap4-real-issue-NOT-dendrites`). The
owner had caught the identical "keeps coming back to dendrites" reflex at least once before (that is why the
2026-07-22 finding exists). NONE of the existing gates fired:

  - `corpus_check_required` (CC) is out of scope for cheap runs by design — it only fires on artifacts recording
    > 1 h of compute, and the whole gap#4 arc was minutes-long numpy runs.
  - `boundary_verdict_external_check` (BV) fires only on findings whose TITLE/VERDICT shouts a LOUD boundary
    ("fundamental", "impossible", "HONEST NEGATIVE" ...) AND only requires that the finding SHOW *some* external
    touch-point; the re-proposal was framed UPBEAT ("names the residual with a biological surpass", "not a
    characterized limit"), so BV was out of scope, and even in scope it does not check the NAMED mechanism against
    the record.
  - the claim gates check NUMBERS -> artifacts, never a forward-looking mechanism CLAIM.

So a cheap-compute finding that names a "remaining surpass / next mechanism" in a non-loud frame sailed through
every gate even though the named mechanism was already tested-and-refuted in our own corpus.

WHAT THIS GATE ENFORCES. In a newly-added/changed governed doc (a finding, the mission board, or a roadmap), when
a FORWARD-LOOKING PROPOSAL phrase ("remaining surpass", "next mechanism", "the candidate is", "next build" ...)
sits NEAR a term for a mechanism on the refuted register below, the SAME doc must also ACKNOWLEDGE that the
mechanism is already tested — by citing one of its refuting findings (filename stem) or an explicit
already-tested/NOT-the-answer token. Proposing a refuted mechanism as if it were fresh, with no acknowledgement,
BLOCKS. This is the mechanical form of CLAUDE.md's self-check ("the comfortable verdict is the START of the
research, never the end") for the specific, recurring case of re-reaching for a mechanism the record already
closed. It does NOT stop you proposing dendrites — it stops you proposing them while SILENT about the negatives;
naming them alongside their refutation (e.g. "dendritic credit is already 6-seed negative, see 2026-07-22-...,
so the genuinely-untested candidate is instead ...") clears the gate.

WHAT IT CANNOT CATCH. A refuted mechanism proposed under a synonym not in the register, or an acknowledgement token
pasted without the source actually being read. The register is extended by hand as new mechanisms are closed.
"""
from __future__ import annotations

import os
import re
import tempfile

NAME = "refuted-mechanism-reproposal"
CLASS_ID = "RM"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Docs that carry forward-looking mechanism proposals: findings + the live board + the roadmaps.
_GOVERNED_SUFFIXES = ("research/findings/", "GAP_CLOSURE_MISSION.md", "ROADMAP.md",
                      "docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md")

# A FORWARD-LOOKING proposal: "the remaining surpass is", "next mechanism", "the candidate is", "next build" ...
PROPOSAL_RE = re.compile(
    r"(remaining\s+(?:biological\s+|named\s+)?surpass"
    r"|named\s+surpass|the\s+surpass\s+is|standing\s+candidate|the\s+one\s+remaining"
    r"|next\s+mechanism|next\s+candidate|next\s+build|the\s+candidate\s+(?:is|surpass|mechanism)"
    r"|candidate\s+surpass|the\s+fix\s+is\s+(?:a|an|to)|remaining\s+candidate)",
    re.I)

# The refuted register: each entry is a family of terms that, when PROPOSED, must be acknowledged as already-tested
# by citing a refuting finding stem or an explicit token. Extend by hand as mechanisms are closed.
REFUTED = [
    {
        "label": "two-compartment / dendritic / BDSP / burstprop deep-credit rule",
        "terms": re.compile(r"\b(two[- ]compartment|dendritic|dendrite|BDSP|burst[- ]?prop|"
                            r"burst[- ]dependent\s+synaptic)\b", re.I),
        # naming ANY of these findings (stem) in the doc counts as acknowledging the refutation.
        "refuted_by": (
            "2026-07-22-gap4-real-issue-NOT-dendrites",
            "2026-05-17-dendritic-credit-assignment-NEGATIVE",
            "2026-08-01-gap4-coincidence-gated-BDSP",
            "2026-07-12-deep-credit-on-spikes-FA-family-exhausted",
        ),
        "why": ("two-compartment/dendritic/BDSP/burstprop deep credit is already tested-and-NEGATIVE for hidden "
                "credit assignment on spikes (topology is faithful; the frozen fixed-random feedback SIGNAL is the "
                "cause) — do not re-propose it as a fresh surpass without citing the negatives"),
    },
]

# tokens that, near the proposal, explicitly acknowledge the mechanism is already tested (clears the gate).
ACK_RE = re.compile(
    r"(already\s+(?:tested|refuted|negative|closed|6-seed\s+negative|been\s+tested)"
    r"|tested[- ]and[- ]negative|NOT\s+the\s+answer|NOT\s+the\s+dendrite|not\s+to\s+blame"
    r"|topologically\s+faithful|already[- ]tested|has\s+been\s+refuted|prior\s+negative)",
    re.I)

_WINDOW = 320  # chars around a proposal phrase within which a refuted term makes it a re-proposal


def _check_one(path, rel=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    if not any(s in rel for s in _GOVERNED_SUFFIXES):
        return []
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []

    problems = []
    for pm in PROPOSAL_RE.finditer(text):
        lo, hi = max(0, pm.start() - _WINDOW), min(len(text), pm.end() + _WINDOW)
        window = text[lo:hi]
        for entry in REFUTED:
            if not entry["terms"].search(window):
                continue
            # acknowledged if a refuting finding stem appears ANYWHERE in the doc, or an ack token near the proposal.
            cited = any(stem in text for stem in entry["refuted_by"])
            if cited or ACK_RE.search(window):
                continue
            snippet = re.sub(r"\s+", " ", text[pm.start():min(len(text), pm.end() + 80)]).strip()
            problems.append(
                "%s: proposes a REFUTED mechanism (%s) as a next/remaining surpass without acknowledging it is "
                "already tested — \"...%s...\". %s. Cite one of {%s} on the same line, or name the "
                "genuinely-untested candidate instead. (Run `bash tools/before_you_build.sh \"<mechanism>\"` "
                "before naming a surpass.)"
                % (rel, entry["label"], snippet[:90], entry["why"], ", ".join(entry["refuted_by"][:2])))
            break  # one problem per proposal phrase is enough
    return problems


def check(paths):
    if paths is None or len(paths) == 0:
        return []                                          # legacy predates the gate; audited on touch
    problems = []
    for p in [x for x in paths if x.endswith(".md")]:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full, p)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: the silent re-proposal, then everything that must NOT fire."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, body):
            p = os.path.join(d, "research", "findings", name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(body)
            return "research/findings/" + name, p

        # 1. THE REAL CASE: names dendritic credit as the remaining surpass, no acknowledgement of the negatives.
        rel, p = w("x1.md", "---\ntype: finding\n---\nThe one remaining biological surpass is a two-compartment "
                            "dendritic credit with a different fixed-point structure.")
        if not _check_one(p, rel):
            bad.append("did NOT catch a silent re-proposal of two-compartment dendritic credit")

        # 2. THE REAL CASE, board variant: 'next mechanism = BDSP' with no citation.
        relb = "GAP_CLOSURE_MISSION.md"
        pb = os.path.join(d, relb)
        open(pb, "w").write("EXACT NEXT: the next mechanism is a burstprop / BDSP credit rule on the Izhikevich forward.")
        if not _check_one(pb, relb):
            bad.append("did NOT catch a silent BDSP re-proposal on the board")

        # 3. NEGATIVE CONTROL — proposes dendrites BUT cites the refuting finding => acknowledged, must pass.
        rel3, p3 = w("x3.md", "---\ntype: finding\n---\nDendritic two-compartment credit is already tested-and-negative "
                             "(2026-07-22-gap4-real-issue-NOT-dendrites), so the remaining surpass is NOT dendrites but "
                             "the untested BurstCCN STP-demux.")
        if _check_one(p3, rel3):
            bad.append("FALSE POSITIVE: flagged a proposal that DID cite the refutation")

        # 4. NEGATIVE CONTROL — a proposal that names a genuinely-untested mechanism (no refuted term) must pass.
        rel4, p4 = w("x4.md", "---\ntype: finding\n---\nThe remaining surpass is the untested BurstCCN STP-demux of "
                             "the burst/event streams.")
        if _check_one(p4, rel4):
            bad.append("FALSE POSITIVE: flagged a proposal of a non-refuted mechanism")

        # 5. NEGATIVE CONTROL — a finding that MEASURES dendrites (no forward proposal phrase) must pass.
        rel5, p5 = w("x5.md", "---\ntype: finding\n---\nWe measured the two-compartment dendritic credit rule: "
                             "6/6 seeds NEGATIVE, ties the reservoir.")
        if _check_one(p5, rel5):
            bad.append("FALSE POSITIVE: flagged a measurement finding with no next-mechanism proposal")

        # 6. NEGATIVE CONTROL — a non-governed file (e.g. a runner) is out of scope.
        rel6 = "research/runners/foo.md"
        p6 = os.path.join(d, rel6); os.makedirs(os.path.dirname(p6), exist_ok=True)
        open(p6, "w").write("the next mechanism is a two-compartment dendritic credit")
        if _check_one(p6, rel6):
            bad.append("FALSE POSITIVE: fired on a non-governed path")

    return bad
