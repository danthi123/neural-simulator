"""CLASS DMS — a substantial findings doc examined (almost) NONE of its own numeric claims, because a
`<!--derived-->` marker suppressed them, and the doc still reads as if every claim traced to a cited artifact.

THE INCIDENT (found 2026-09-25, independent verifier). `tools/claim_check.py`'s standalone `<!--derived-->`
marker opened BLOCK scope until the next `## ` heading. A scorer put a standalone marker right after three
headings, which suppressed EVERY number in those sections: the run reported "0 checked against 336 artifact
values" and still printed "every measurement traces to a cited artifact". Two numbers in the document did not
match the artifact it cited (0.1525 and 0.140; the artifact read 0.17 and 0.1625) and passed the gate.

THE FIX has two independent layers, and this module is the second one:
  1. `tools/claim_check.py` itself now scopes a standalone marker to its own paragraph/table (closes at the next
     blank line AFTER content starts, the next heading, or an explicit `<!--/derived-->`) instead of the whole
     rest of the section. This closes the SPECIFIC mechanism the incident used.
  2. THIS GATE is a second, independent line of defense: regardless of WHICH future mechanism produces the total
     suppression, a SUBSTANTIAL doc (>= `claim_check.LOW_COVERAGE_MIN_TOTAL` numeric claims) that examined
     (almost) none of them is the same shape as the incident and gets flagged, corpus-wide, not just at the
     moment a doc is staged.

WHY THE SIZE FLOOR (not "checked == 0" alone). A 2026-09-25 retro-scan of the 353 `research/findings/*.md` added
since 2026-09-01 found 71 documents that are ENTIRELY, LEGITIMATELY derived -- a short diagnosis note built
purely from already-published ratios/deltas/percentages, correctly all-marked, up to 39 numeric claims. A
blanket "checked==0 fails" rule would have blocked every one of them for doing nothing wrong. The floor sits
just above that observed legitimate maximum and well below the incident's scale (~43-53 claims in the document,
336 candidate values in the artifacts it cited) -- see `tools/claim_check.py`'s `LOW_COVERAGE_MIN_TOTAL` comment
and the 2026-09-25 `research/claimcheck-block-scope` branch notes for the full retro-scan.

REUSE, NOT REIMPLEMENTATION: this module calls `tools.claim_check._scan`, the SAME pure-computation core
`tools/claim_check.py`'s own CLI and `tools/finding_lint.py` use, so the three can never disagree.

WHAT IT CANNOT CATCH: a doc whose numbers are individually WRONG but few enough, or suppressed few enough, to
stay under the coverage floor -- that is `claim_check`'s own per-number check's job (GATE 2 at pre-commit, and
this same module's `check()` return value is a coverage-only, defense-in-depth signal on top of it).
"""
from __future__ import annotations

import glob
import os

import tools.claim_check as claim_check

NAME = "claim-check-coverage"
CLASS_ID = "DMS"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _staged(paths):
    return [p for p in (paths or []) if p.endswith(".md")
            and "research/findings/" in p.replace(os.sep, "/")]


def _audit():
    return [os.path.relpath(p, _ROOT) for p in glob.glob(os.path.join(_ROOT, "research/findings/*.md"))]


def check(paths):
    if paths is not None and len(paths) == 0:
        return []
    files = _staged(paths) if paths else _audit()
    out = []
    for rel in files:
        full = os.path.join(_ROOT, rel)
        if not os.path.isfile(full):
            continue
        try:
            r = claim_check._scan(full)
        except Exception:                      # an unreadable doc is claim_check's own problem, not this gate's
            continue
        if r["low_coverage"]:
            tot = r["total_numeric"]
            supp = r["suppressed"]
            out.append("%s: only %d/%d (%.0f%%) numeric claim(s) checked -- suppressed by <!--derived--> "
                       "(section=%d block=%d inline=%d)%s"
                       % (rel, r["checked"], tot, 100.0 * r["checked"] / tot if tot else 0.0,
                          supp["section"], supp["block"], supp["inline"],
                          ", synthesis=%d" % supp["synthesis"] if supp["synthesis"] else ""))
    return out


def selftest():
    """Both directions, reusing `tools.claim_check._scan` (never a private reimplementation):

      FAILING DIRECTION -- a SUBSTANTIAL doc (>= LOW_COVERAGE_MIN_TOTAL claims) that inline-marks every one of
      its numeric claims as derived, with a real cited artifact and no wrong numbers anywhere, must still be
      flagged: checked==0 of a large total is the incident's own shape regardless of any per-number mismatch.

      PASSING DIRECTION -- a doc of the SAME size with most of its claims normally checked (matching the cited
      artifact) and only a small legitimately-derived table must NOT be flagged.
    """
    import json
    import tempfile
    problems = []
    # NOTE: this exercises `claim_check._scan` directly (the same core `check()` above calls), not `check()`'s
    # own `_staged()` path filter -- that filter only recognises paths already scoped to `research/findings/`
    # (how the pre-commit hook stages files), which a tempdir fixture never is.
    with tempfile.TemporaryDirectory(dir=_ROOT) as d:
        art_abs = os.path.join(d, "art.json")
        json.dump({"accuracy": 0.170000, "baseline": 0.1625}, open(art_abs, "w"))
        art = os.path.relpath(art_abs, _ROOT).replace(os.sep, "/")
        n = claim_check.LOW_COVERAGE_MIN_TOTAL + 5

        bad = os.path.join(d, "bad.md")
        body = "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(n))
        open(bad, "w", encoding="utf-8").write("# Over-marked\n\nArtifact: `%s`\n\n%s\n" % (art, body))
        if not claim_check._scan(bad)["low_coverage"]:
            problems.append("SELFTEST BROKEN: a %d-claim doc that inline-marks EVERY numeric claim as derived "
                            "(checked=0) was NOT flagged -- the incident's own shape ('0 checked against N "
                            "artifact values') would pass silently again" % n)

        good = os.path.join(d, "good.md")
        lines = ["The accuracy is 0.170000 (matches the artifact)."] * (n - 3)
        lines.append("<!--derived-->\n| metric | value |\n|---|---|\n| ratio | 0.104615 |\n| delta | 0.007500 |")
        lines.append("The baseline was 0.162500 here (matches the artifact).")
        open(good, "w", encoding="utf-8").write(
            "# Well-cited finding\n\nArtifact: `%s`\n\n%s\n" % (art, "\n\n".join(lines)))
        if claim_check._scan(good)["low_coverage"]:
            problems.append("SELFTEST BROKEN: a mostly-checked doc with one small, correctly-scoped derived "
                            "table was FLAGGED (false positive)")
    return problems


if __name__ == "__main__":
    hits = check(None)
    print("claim-check-coverage corpus audit: %d finding(s) examined (almost) none of their own numeric claims"
          % len(hits))
    for h in hits[:40]:
        print("  ⛔", h[:200])
