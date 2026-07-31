#!/usr/bin/env python3
"""Derive a machine-readable STATUS for every finding, so retrieval cannot surface a void result as live.

WHY (2026-07-31, owner question: "are findings/results/memories organised so you pull the right ones and can
recall without hallucinating?"). Measured answer: NO, and the mechanism is precise.

    1841 findings · 101 whose BODY carries a retraction/supersession marker · 1 registered in docs/RETRACTED.md
    · 0 with machine-readable frontmatter

So a RAG hit carries a path and a score and NOTHING about whether the result still stands. The concrete instance,
same day: a search returned `2026-07-24-P0.3-affect-state-region-6seed-GO.md` -- a filename asserting GO -- whose
actual verdict is QUALIFIED-GO/BOUNDARY and whose artifact reads {"GO": false, "n_seeds_go": 2}. That filename
propagated into the board, and from the board into a claim I made to the owner. The finding itself was honest;
nothing carried its honesty to the point of retrieval.

DERIVED, NOT MAINTAINED. Every hand-maintained index in this repo has gone stale -- docs/RETRACTED.md holds ONE
row against 101 marked findings. So this SCANS the corpus and regenerates; there is nothing to keep updated and
therefore nothing to forget.

    .venv/bin/python tools/finding_status.py                 # report the distribution
    .venv/bin/python tools/finding_status.py --write         # regenerate docs/FINDINGS_STATUS.md
    .venv/bin/python tools/finding_status.py --check <path>  # status of one finding (for retrieval to annotate)

STATUS is read from the document's own words, in priority order, because the corpus already speaks this
vocabulary consistently -- it just does so in prose no machine reads:

    retracted  > superseded > corrected > qualified > live
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINDINGS = os.path.join(ROOT, "research", "findings")

# Ordered: first match wins. Patterns are deliberately anchored to how THIS corpus actually writes these words
# (checked against the 101 marked files), not to a generic vocabulary.
RULES = [
    ("retracted", re.compile(
        r"⛔⛔|\bRETRACT(?:ED|ION)?\b|\bWITHDRAWN\b|\bIS VOID\b|\bVOID\b(?!\s*(?:if|_if))|\bINVALID(?:ATES?|ATED)?\b",
        re.I)),
    ("superseded", re.compile(r"\bSUPERSED(?:ED|ES|ING)\b|\bREPLACED BY\b|\bno longer the\b", re.I)),
    ("corrected", re.compile(r"\bCORRECTION\b|\bCORRECTED\b|\bOVERREACH(?:ED)?\b|\bWAS WRONG\b", re.I)),
    ("qualified", re.compile(r"\bQUALIFIED[- ]GO\b|\bBOUNDARY\b|\bNO-GO\b|\bHONEST NEGATIVE\b", re.I)),
]
HEAD_LINES = 40          # status is declared near the top; scanning whole 1000-line files invites false hits


DECL_RE = re.compile(r"^status:\s*([a-z-]+)\s*$", re.M)


def declared_status(path):
    """The AUTHORITY: `status:` in frontmatter. Returns None when absent."""
    try:
        with open(path, errors="ignore") as fh:
            head = "".join(next(fh, "") for _ in range(15))
    except Exception:
        return None
    if not head.startswith("---"):
        return None
    m = DECL_RE.search(head.split("\n---", 1)[0])
    return m.group(1).strip() if m else None


def status_of(path):
    """Declared status if present; otherwise a KEYWORD HINT, which is NOT authoritative.

    ⚠️ PRECISION MEASURED BEFORE SHIPPING, and it is poor: a 6-file sample of the keyword rule's "retracted"
    label was right about 1 time in 6. A finding that PERFORMS a retraction carries the same vocabulary as one
    that IS retracted, and one that merely DISCUSSES the retraction record carries it too --
    `2026-07-31-gap5-stepC-control-void-and-the-fix.md` is LIVE and is the document that did the voiding.
    Keywords cannot separate "this is void" from "this voids something else".

    So the hint is a BACKFILL SUGGESTION, never a verdict. Status has to be DECLARED. That is why the pre-commit
    gate requires it on NEW findings only: 1841 files cannot be backfilled in one pass, and a big-bang migration
    would not happen -- but every new finding declares, and old ones declare when next touched.
    """
    try:
        with open(path, errors="ignore") as fh:
            head = [next(fh, "") for _ in range(HEAD_LINES)]
    except Exception:
        return "unknown", ""
    decl = declared_status(path)
    if decl:
        return decl, "declared"
    text = "".join(head)
    base = os.path.basename(path)
    for name, rx in RULES:
        m = rx.search(base) or rx.search(text)
        if m:
            for ln in head:
                if rx.search(ln):
                    return name, ln.strip()[:150]
            return name, base
    return "live?", ""


def scan():
    rows = []
    for p in sorted(glob.glob(os.path.join(FINDINGS, "*.md"))):
        s, ev = status_of(p)
        rows.append((os.path.relpath(p, ROOT), s, ev))
    return rows


def main():
    args = sys.argv[1:]
    if "--check" in args:
        for p in args[args.index("--check") + 1:]:
            s, ev = status_of(p if os.path.isabs(p) else os.path.join(ROOT, p))
            print("%-11s %s" % (s.upper(), os.path.basename(p)))
            if ev:
                print("            %s" % ev)
        return 0

    rows = scan()
    counts = {}
    for _, s, _ in rows:
        counts[s] = counts.get(s, 0) + 1
    print("finding_status: %d findings" % len(rows))
    declared = sum(1 for p, _, ev in rows if ev == "declared")
    print("  DECLARED (authoritative): %d of %d" % (declared, len(rows)))
    print("  the rest are KEYWORD HINTS, measured ~1-in-6 precise — backfill suggestions, not verdicts:")
    for s in ("live", "live?", "qualified", "corrected", "superseded", "retracted", "unknown"):
        if counts.get(s):
            print("  %-11s %4d" % (s, counts[s]))
    not_live = [r for r in rows if r[1] not in ("live",)]
    print("  => %d of %d flagged non-live by the HINT. Treat as a backfill queue, not a registry." % (len(not_live), len(rows)))

    if "--write" in args:
        out = os.path.join(ROOT, "docs", "FINDINGS_STATUS.md")
        with open(out, "w") as fh:
            fh.write("# Findings status — GENERATED, do not edit\n\n")
            fh.write("Regenerate: `.venv/bin/python tools/finding_status.py --write`\n\n")
            fh.write("Derived by scanning each finding's own head for the vocabulary this corpus already uses.\n")
            fh.write("It exists because `docs/RETRACTED.md` held ONE row against %d non-live findings, so a RAG\n"
                     % len(not_live))
            fh.write("hit could not tell a void result from a live one — and did not, on 2026-07-31.\n\n")
            for s in ("retracted", "superseded", "corrected", "qualified"):
                sel = [r for r in rows if r[1] == s]
                if not sel:
                    continue
                fh.write("\n## %s (%d)\n\n" % (s.upper(), len(sel)))
                for path, _, ev in sel:
                    fh.write("- `%s`%s\n" % (os.path.basename(path), ("  \n  %s" % ev) if ev else ""))
        print("  wrote docs/FINDINGS_STATUS.md (%d non-live entries)" % len(not_live))
    return 0


if __name__ == "__main__":
    sys.exit(main())
