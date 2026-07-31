"""FAILURE CLASS 11 — TERMINOLOGY OVERCLAIM (3 recorded incidents, all on 2026-07-28).

THE EVIDENCE. Three of nine retractions in one session were pure vocabulary, not measurement: an experiment
called **consolidation** whose replay branch never executed; **compositional** asserted over a code that was
localist by construction; **self-organized** while the host supplied both factors of the learning rule. Every
number underneath was correct and reproducible. `docs/TERMS.md` was written that day and gives each loaded word
a CODE CONDITION plus the fallback wording for when it does not hold.

WHAT THIS GATE CHECKS. In a findings doc dated on/after the day TERMS.md landed: a loaded term in a CLAIM
position — a heading, a `**bold**` span, or the tail of a `Status:`/`Verdict:`/`Result:` line — where the
document NOWHERE engages that term's condition. "Engages" is deliberately generous: any mention of
`replay`/`lesion` clears *consolidation*, any `permuted`/`shuffled`/`null` clears *selective*, and naming
`TERMS.md` clears the file. It catches the headline that shouts a condition it never discusses, not prose.

WHAT IT CANNOT CATCH, and why BLOCKING is False.
  * **GO**, **closed**, **works/solved** are not detected as standalone terms. Measured here: `GO` sits in a
    claim position in 851 of 1841 findings (`GO-gate`, `NO-GO`, `6-seed GO`), `closed` collides with `closed
    loop` and the transmission-gate sense, `works` is ordinary English. String-matching them is a false-positive
    generator. "CONSOLIDATION WORKS" is still caught — via `consolidation` plus the assertion word `WORKS`.
  * A condition the doc *mentions* but does not *satisfy*: only a reader can tell "we ran the permuted control"
    from "the permuted control is future work". This is a prompt to check, never a verdict that it holds.
  * Commit messages and board entries, also governed by TERMS.md — the gate sees files, not the commit message.
  * Anything before the cutoff: the rule is not retroactive, and ~1800 older findings would bury the live ones.
Calibration: 0 hits on the 14 in-scope docs, ~3% of the historical corpus if the date scope were removed, of
which hand-inspection found roughly half to be topic mentions. That residual is why it reports and does not block.
"""
from __future__ import annotations

import os
import re

NAME = "terminology"
CLASS_ID = "11"
BLOCKING = False

# The day docs/TERMS.md landed. Older findings are out of scope: the rule is not retroactive.
CUTOFF = "2026-07-28"

# term -> (pattern in a claim span, "the doc engages the condition" pattern, needs an assertion word, fallback)
# `needs_assert` separates SELF-ASSERTING terms (using them at all is the claim) from DESCRIPTIVE ones, which
# appear constantly as topic words and are a claim only when a verdict word sits in the same span.
LOADED = {
    "byte-identical": (
        r"byte[-\s]identical", r"hash|sha1|sha256|md5|checksum|array_equal|allclose|bit-for-bit|exact|\bdiff\b",
        False, "assert it in the data (hash/exact compare), or say 'expected unchanged (unverified)'"),
    "fully spiking": (
        r"fully[-\s]spiking", r"\bhost\b|read-?out|argmax",
        False, "or say 'spiking with a host read-out'"),
    "consolidation": (
        r"consolidat(?:ion|ed|es)", r"replay|reactivat|lesion",
        True, "verify the replay branch EXECUTES + survives a source lesion, or say 'a cortical write'"),
    "compositional": (
        r"compositional(?:ity)?", r"localist|constituent|per-item|per item|one-unit-per|disjoint",
        True, "the design must distinguish it from a per-item code, or say 'localist'"),
    "self-organized": (
        r"self[-\s]organi[sz]ed", r"\bhost\b|hand-?(?:set|designed|wired|coded)|supervis|teacher|both factors",
        True, "check both factors + target + slot allocation, or say 'host-supervised'"),
    "lesion": (
        r"lesion(?:ed|s)?", r"persist|still hold|re-?grow|verif|drift|held",
        True, "verify the manipulation STILL HOLDS at measurement, or say 'attempted lesion (unverified)'"),
    "selective": (
        r"selectiv(?:e|ity)", r"permut|scrambl|shuffl|random-?set|\bnull\b",
        True, "report the permuted control + raw per-item magnitudes, or say 'ratio X (control not run)'"),
}

_HEAD = re.compile(r"^\s{0,3}#{1,4}\s+(.*)$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_VERDICT = re.compile(r"(?:Status|Verdict|Result|Conclusion|Outcome)\s*:\**\s*(.+)$", re.I)
_ASSERT = re.compile(r"\bGO\b|WORKS?\b|WORKED|CONFIRM|VALIDAT|PROVEN|PROVES|ACHIEV|SUCCESS|SOLVED"
                     r"|\bPASS(?:ES|ED)?\b|✅|🎉", re.I)
# Anything hedged, negated, aspirational or naming-a-thing is NOT a positive claim. Broad on purpose: a missed
# overclaim costs a nudge, a cried wolf costs the gate.
_NEG = re.compile(r"⛔|NO-?GO|NEGATIVE|\bnot\b|\bno\b|never|VOID|RETRACT|withdraw|FALSE|refut|fail|absent"
                  r"|unverified|\?|DESIGN ONLY|obstacle|deferred|tracked|shortcut|standard|toward|goal|target"
                  r"|boundary|wall|ceiling|limit|plan|option|proposal|deserves|would|whether|if\b|TODO"
                  r"|channel|pathway|region|module", re.I)
_DATE_FM = re.compile(r"^date:\s*(\d{4}-\d{2}-\d{2})", re.M)
_STATUS_FM = re.compile(r"^status:\s*(\w+)", re.M)
_BASENAME_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")


def _in_scope(path):
    """A findings markdown file. Nothing else — this gate has no calibration outside that corpus."""
    p = path.replace("\\", "/")
    return p.endswith(".md") and "research/findings/" in p and "/raw/" not in p


def _doc_date(path, text):
    m = _DATE_FM.search(text) or _BASENAME_DATE.match(os.path.basename(path))
    return m.group(1) if m else None


def _spans(line):
    """The claim positions of one line: heading text, bold spans, and the tail of a verdict line."""
    out = []
    m = _HEAD.match(line)
    if m:
        out.append(m.group(1))
    out.extend(_BOLD.findall(line))
    m = _VERDICT.search(line)
    if m and m.group(1).strip():
        out.append(m.group(1))
    return out


def _scan(path, text):
    if "TERMS.md" in text:                                    # the author cited the conditions file
        return []
    st = _STATUS_FM.search(text)
    if st and st.group(1).lower() in ("retracted", "superseded"):
        return []                                             # a retraction records the overclaim on purpose
    d = _doc_date(path, text)
    if d is None or d < CUTOFF:
        return []
    lines = text.splitlines()
    problems = []
    for term, (pat, evidence, needs_assert, fallback) in LOADED.items():
        if re.search(evidence, text, re.I):                    # the doc engages the condition somewhere
            continue
        term_re = re.compile(r"\b" + pat + r"\b", re.I)
        for i, line in enumerate(lines, 1):
            hit = None
            for span in _spans(line):
                if term_re.search(span) and not _NEG.search(span) and (
                        not needs_assert or _ASSERT.search(span)):
                    hit = span.strip()
                    break
            if hit:
                problems.append(
                    "%s:%d class-11 '%s' claimed but its docs/TERMS.md condition is never addressed "
                    "in this doc — %s | %r" % (path, i, term, fallback, hit[:90]))
                break                                          # one report per term per doc
    return problems


def check(paths=None):
    _standalone = paths is None
    paths = list(paths or [])
    # An EMPTY list means "staged mode, nothing of my kind staged" -> nothing to check. Only paths=None means
    # "standalone run, scan the corpus". Without this, the pre-commit driver's --diff-filter=A scoping is undone
    # by this gate's own corpus fallback -- which fired 192 doc-type hits on 2026-04/05 legacy findings the
    # moment the Tier-1 classification gave them frontmatter.
    # _standalone (paths=None) means a full-corpus AUDIT run and must NOT be skipped; an empty
    # LIST means the hook staged nothing of my kind. Collapsing None into [] made both look the
    # same and silently disabled the audit mode -- the mode the project-wide recheck depends on.
    if not _standalone and len(paths) == 0:
        return []
    files = [p for p in paths if _in_scope(p)]
    if not paths:
        import glob
        files = sorted(glob.glob("research/findings/*.md"))
    out = []
    for p in files:
        if not os.path.isfile(p):                              # staged deletion / rename
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            out += _scan(p, fh.read())
    return out


def selftest():
    """FAILING DIRECTION FIRST: the recorded incident must be caught, then three no-cry-wolf directions."""
    import shutil
    import tempfile

    bad = ("---\nstatus: live\ndate: %s\n---\n\n"
           "# 🎉 CONSOLIDATION WORKS — 6-seed 2x2, both ingredients necessary\n\n"
           "The store holds across 18/18 runs and the mean-subtract write is necessary.\n" % CUTOFF)
    good = bad.replace("The store holds",
                       "The replay branch was verified to EXECUTE (breakpoint hit 18/18) and the trace "
                       "survives a lesion of CA3. The store holds")
    # NB: this fixture must contain NO evidence word, or it would be skipped by the evidence path and would
    # silently stop testing the negation filter at all (it did, until a mutation test caught the dead fixture).
    negated = bad.replace("CONSOLIDATION WORKS", "⛔ CONSOLIDATION DOES NOT WORK")
    old = bad.replace("date: %s" % CUTOFF, "date: 2026-05-01")
    retracted = bad.replace("status: live", "status: retracted")
    cited = bad.replace("The store holds", "Terms checked against docs/TERMS.md. The store holds")
    # A bare TOPIC mention in a heading, with no verdict word: must stay silent, or every doc in the lane fires.
    topic = ("---\nstatus: live\ndate: %s\n---\n\n# gap#5 read-density sweep\n\n"
             "## Prior consolidation experiments in this lane\n\n"
             "Earlier runs are listed here so the operating point can be compared.\n" % CUTOFF)
    # Body prose is NOT a claim position. Same sentence, same verdict word, but unemphasised — the gate reads
    # headings, bold spans and verdict lines only, and widening it to whole lines would flag ordinary writing.
    prose = topic.replace("Earlier runs are listed here so the operating point can be compared.",
                          "In the earlier arc the consolidation probe passed its smoke check.")
    # A longer word that merely CONTAINS a loaded term is a different term with its own biology.
    adjacent = topic.replace("## Prior consolidation experiments in this lane",
                             "## Reconsolidation update de-risk = GO")

    root = tempfile.mkdtemp(prefix="gate11_")
    try:
        d = os.path.join(root, "research", "findings")
        os.makedirs(d)
        cases = {"bad": bad, "good": good, "negated": negated, "old": old, "topic": topic,
                 "retracted": retracted, "cited": cited, "prose": prose, "adjacent": adjacent}
        got = {}
        for k, body in cases.items():
            p = os.path.join(d, "%s-%s.md" % (CUTOFF, k))
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body)
            got[k] = check([p])
        problems = []
        if not got["bad"]:
            problems.append("BLIND: the recorded 'CONSOLIDATION WORKS' overclaim was NOT caught")
        elif "consolidation" not in got["bad"][0]:
            problems.append("caught the wrong term: %s" % got["bad"][0][:120])
        for k, why in (("good", "a doc that verifies the replay executes AND lesions the source"),
                       ("negated", "an explicitly negated/⛔ claim"),
                       ("old", "a pre-TERMS.md doc (the rule is not retroactive)"),
                       ("topic", "a bare topic mention in a heading, carrying no verdict word"),
                       ("retracted", "a retracted doc, which records its overclaim on purpose"),
                       ("cited", "a doc that cites docs/TERMS.md"),
                       ("prose", "a body-prose sentence, which is not a claim position"),
                       ("adjacent", "'reconsolidation', a different term that merely contains one")):
            if got[k]:
                problems.append("CRIES WOLF on %s: %s" % (why, got[k][0][:120]))
        if _in_scope("sim/bridge.py") or not _in_scope("research/findings/2026-07-31-x.md"):
            problems.append("scope filter is wrong: it must accept findings .md and nothing else")
        return problems
    finally:
        shutil.rmtree(root, ignore_errors=True)
