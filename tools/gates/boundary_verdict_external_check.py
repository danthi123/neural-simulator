"""CLASS BV — A 'FUNDAMENTAL LIMIT' VERDICT BANKED WITHOUT READING THE FIELD THAT REFUTES IT.

THE DEFECT, measured 2026-08-01. A whole session re-derived the transport-free deep-credit ceiling from a
MEMORY model of the mechanisms (Kolen-Pollack learned feedback, burstprop) instead of reading them, and banked
`b7549514` — "the residual is a FUNDAMENTAL limit of the local transport-free credit class ... the frontier is
a different-paradigm (equilibrium-propagation) question." It cited ZERO external literature. The external SOTA
directly refutes it: WF-Act-PC (arxiv 2607.13380) shows feedback-alignment collapses at depth precisely because
it drops the sigma' factor, and W^T+sigma' matches backprop; our own prior finding
`2026-07-07-D2-rung2-learned-apical-feedback-...` had already BUILT the learned-feedback rule the session
re-invented. A one-hour deep read overturned the banked wall (transport-free graded chained-FA+sigma' clears it,
6-seed 0.935 vs the banked 0.63; KP-learned feedback rescues MNIST depth-4 0.68->0.88). The verdict was wrong
AND redundant, and every existing gate passed it.

WHY THE EXISTING GATES MISSED IT.
  - `corpus_check_required` (CC) fires only on runs recording > 1 h of compute; the re-derivations were cheap
    numpy toys (seconds), exempt by design.
  - `closure_names_mechanism` (CM) forces a `mechanism:` only on POSITIVE closures and its NEGATED_RE
    deliberately whitelists NO-GO / limit / wall verdicts — the negative half was simply absent.
  - `.last_external_search` is a marker with NO WRITER; external-literature consultation was at most a soft,
    ignorable heartbeat nag, never a commit gate.

WHAT THIS GATE ENFORCES, narrowly and only on newly-added findings: a finding whose TITLE or VERDICT asserts a
capability-WALLED verdict — "fundamental limit / wall / barrier", "structural primitive", "characterized limit",
"different-paradigm", "no transport-free point", "cannot be done/closed/surpassed", "HONEST NEGATIVE" shouted as
the verdict — must SHOW it engaged the outside literature: an arxiv/doi/Sources citation, an `(Author et al.,
YEAR)` reference, or an explicit `EXTERNAL-SEARCH-RAN:` / `NO-EXTERNAL-NEEDED: <reason>` line. This is the
mechanical form of CLAUDE.md's own prose self-check ("the moment you write NEGATIVE / BOUNDARY / structural
primitive / honest negative / characterized limit / defensible and your next instinct is to scope the fix, THAT
instinct is the trigger") and drift-mode #9 ("skimming the sources"), neither of which fired on its own.

DELIBERATELY NOT ENFORCED:
  - Ordinary NEGATIVE method verdicts. "This method gave a negative" is a legitimate, common deliverable; only
    a verdict asserting the CAPABILITY is WALLED trips this — that is the class the mission forbids declaring
    without the deep read.
  - Corrections. A finding OVERTURNING a prior boundary (a ⛔/RETRACT/CORRECTION/OVERTURN title, or `status:
    superseded/retracted/void`) is cleared — it is the fix, not the failure.

WHAT IT CANNOT CATCH: an external citation that was pasted but not read, or a boundary verdict phrased so as to
dodge the vocabulary. It guarantees the finding SHOWS an external touch-point at the exact place a comfortable
wall is most expensive; reading it remains judgement.
"""
from __future__ import annotations

import os
import re
import tempfile

NAME = "boundary-verdict-external-check"
CLASS_ID = "BV"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FM_FINDING_RE = re.compile(r"^type:\s*finding\s*$", re.M)

# A capability-WALLED verdict SHOUTED in a title or verdict line (uppercase, verdict-shaped — the same
# convention closure_names_mechanism relies on: titles/verdicts shout their conclusion).
BOUNDARY_LOUD_RE = re.compile(
    r"^#[^\n]*\b(?:FUNDAMENTAL (?:LIMIT|WALL|BOUNDARY|BARRIER)|HARD WALL|STRUCTURAL PRIMITIVE"
    r"|CHARACTERIZED LIMIT|HONEST NEGATIVE|DIFFERENT[- ]PARADIGM|NO TRANSPORT[- ]FREE"
    r"|CANNOT BE (?:DONE|CLOSED|SURPASSED|BEATEN))\b"                                    # in the title
    r"|^\s*\*\*(?:Verdict|VERDICT|RESULT)\b[^\n]*\b(?:FUNDAMENTAL|(?:A |THE )?(?:HARD )?WALL|BOUNDARY"
    r"|STRUCTURAL PRIMITIVE|CHARACTERIZED LIMIT|HONEST NEGATIVE|DIFFERENT[- ]PARADIGM"
    r"|CANNOT BE (?:DONE|CLOSED|SURPASSED|BEATEN))\b",                                    # in a verdict line
    re.M)
# The 'comfortable verdict' capability-walled phrases, dangerous in ANY case, anywhere. Plain "honest negative"
# is EXCLUDED here (it is a common, legitimate method verdict) — it trips only when SHOUTED as the verdict above.
BOUNDARY_SOFT_RE = re.compile(
    r"\bfundamental(?:ly)?\s+(?:a\s+)?(?:limit|limited|wall|boundary|barrier)\b"
    r"|\bstructural primitive\b|\bcharacterized limit\b|\bdifferent[- ]paradigm\b"
    r"|\bno transport[- ]free\s+(?:point|rule|path|solution)\b",
    re.I | re.M)

# The finding SHOWS it engaged the outside literature (any one of these clears the gate).
EXTERNAL_OK_RE = re.compile(
    r"\barxiv\.org\b|\barxiv:\s*\d{3,4}\.\d{4,5}\b|\bdoi\.org\b|\bdoi:\s*10\.\d"
    r"|^#{1,4}\s*Sources\b|^Sources:\s|\bWebSearch\b|\bbio-research\b|\bpubmed\b|\bbiorxiv\b"
    r"|\bEXTERNAL-SEARCH-RAN:|\bNO-EXTERNAL-NEEDED:"
    r"|\([A-Z][A-Za-z.'-]+(?:\s+(?:&|and|et al\.?)[^)]*)?,?\s*20\d\d[a-z]?\)",   # (Author et al., 2024)
    re.I | re.M)

# A withdrawal / correction / overturn is the FIX, not a fresh boundary claim.
CORRECTION_RE = re.compile(
    r"^#[^\n]*(?:⛔|\bRETRACT|\bCORRECTION\b|\bOVERTURN|\bWAS (?:AN? )?(?:ARTIFACT|NOT A)\b"
    r"|\bNOT (?:A |THE )?(?:WALL|LIMIT|BOUNDARY|FUNDAMENTAL)\b)"
    r"|^status:\s*(?:superseded|retracted|void|retract)\b",
    re.M | re.I)


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
    if CORRECTION_RE.search(text):
        return []                                          # a correction/overturn/withdrawal is the fix
    hit = BOUNDARY_LOUD_RE.search(text) or BOUNDARY_SOFT_RE.search(text)
    if not hit:
        return []
    if EXTERNAL_OK_RE.search(text):
        return []                                          # engaged the outside literature — cleared
    claim = hit.group(0).strip()[:70]
    return ["%s: asserts a capability-WALLED verdict (%r) but cites NO external literature and makes no "
            "`EXTERNAL-SEARCH-RAN:` / `NO-EXTERNAL-NEEDED:` declaration. A 'fundamental limit / different-"
            "paradigm' claim is exactly the class the mission forbids banking without the deep read: on "
            "2026-08-01 'the fundamental transport-free ceiling' (b7549514) was banked with zero external "
            "citations and OVERTURNED within the hour by WF-Act-PC (arxiv 2607.13380) + our own adjacent "
            "findings. Add an arxiv/doi/Sources/(Author, YEAR) citation, or an explicit "
            "`NO-EXTERNAL-NEEDED: <reason>` line, before this lands." % (rel, claim)]


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
    """FAILING DIRECTION FIRST: boundary verdicts with no external touch-point, then everything that must NOT fire."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, fm, body):
            p = os.path.join(d, name)
            open(p, "w").write("---\n%s---\n\n%s\n" % (fm, body))
            return p

        F = "type: finding\nstatus: contributing\n"
        # 1. THE REAL CASE — b7549514's phrasing, no external citation.
        if not _check_one(w("a.md", F, "text\n\nThe residual is a FUNDAMENTAL limit of the local transport-free "
                            "credit class; the frontier is a different-paradigm question."), "research/findings/a.md"):
            bad.append("did NOT catch a 'fundamental limit / different-paradigm' verdict with no external cite")
        # 2. a shouted boundary TITLE with no external cite.
        if not _check_one(w("b.md", F, "# The depth wall is a STRUCTURAL PRIMITIVE of the substrate"),
                          "research/findings/b.md"):
            bad.append("did NOT catch a STRUCTURAL PRIMITIVE title with no external cite")
        # 3. a CHARACTERIZED LIMIT verdict line, no external cite.
        if not _check_one(w("c.md", F, "**Verdict:** a CHARACTERIZED LIMIT of the rule family."),
                          "research/findings/c.md"):
            bad.append("did NOT catch a CHARACTERIZED LIMIT verdict with no external cite")
        # 4. NEGATIVE CONTROL — the same boundary claim WITH an arxiv citation clears it.
        if _check_one(w("d.md", F, "The residual is a FUNDAMENTAL limit (but see arxiv.org/abs/2607.13380 — sigma')."),
                      "research/findings/d.md"):
            bad.append("FALSE POSITIVE: flagged a boundary verdict that DID cite external literature (arxiv)")
        # 5. NEGATIVE CONTROL — an explicit NO-EXTERNAL-NEEDED declaration clears it.
        if _check_one(w("e.md", F, "# A FUNDAMENTAL LIMIT of the toy.\n\nNO-EXTERNAL-NEEDED: pure instrument artifact, "
                        "no field claim."), "research/findings/e.md"):
            bad.append("FALSE POSITIVE: flagged a boundary verdict with a NO-EXTERNAL-NEEDED declaration")
        # 6. NEGATIVE CONTROL — a correction/overturn title is the fix, not a fresh claim.
        if _check_one(w("f.md", F, "# OVERTURN: the 'fundamental limit' was an artifact of the binary gate"),
                      "research/findings/f.md"):
            bad.append("FALSE POSITIVE: flagged an OVERTURN/correction title")
        # 7. NEGATIVE CONTROL — a superseded finding is out of scope.
        if _check_one(w("g.md", "type: finding\nstatus: superseded\n",
                        "The residual is a FUNDAMENTAL limit of the class."), "research/findings/g.md"):
            bad.append("FALSE POSITIVE: flagged a superseded finding")
        # 8. NEGATIVE CONTROL — an ordinary NEGATIVE method verdict (not capability-walled) needs no external cite.
        if _check_one(w("h.md", F, "# The dual-route rule does NOT generalize to novel stems (reg_acc 0.19, NO-GO)"),
                      "research/findings/h.md"):
            bad.append("FALSE POSITIVE: flagged an ordinary method NO-GO with no capability-walled claim")
        # 9. NEGATIVE CONTROL — plain 'honest negative' in prose is a legit deliverable, not a shouted wall.
        if _check_one(w("i.md", F, "This is an honest negative for the covariance rule on real spikes."),
                      "research/findings/i.md"):
            bad.append("FALSE POSITIVE: flagged a prose 'honest negative' (legit deliverable, not a walled claim)")
        # 10. NEGATIVE CONTROL — legacy (no frontmatter) is out of scope.
        p = os.path.join(d, "j.md")
        open(p, "w").write("# A FUNDAMENTAL LIMIT of the class\n")
        if _check_one(p, "research/findings/j.md"):
            bad.append("FALSE POSITIVE: flagged a legacy no-frontmatter document")
        # 11. SCOPING — standalone/empty mode must not scan the legacy corpus.
        if check(None) or check([]):
            bad.append("SCOPE LEAK: standalone/empty mode must not scan the legacy corpus")
    return bad
