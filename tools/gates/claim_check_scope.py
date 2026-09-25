"""CLASS DMS — a `<!--derived-->` marker exempts more of a findings doc than the numbers it derives.

THE INCIDENT (2026-09-25, independent verifier). `tools/claim_check.py`'s standalone marker exempted everything
up to the next `## ` heading. A scorer put one after each of three headings: the run reported "0 checked
against 336 artifact values" and still printed "every measurement traces to a cited artifact", while two of its
numbers (0.1525, 0.140) did not match the cited artifact (0.17, 0.1625). Three rounds of line-scanner patches
each opened a new hole (see `tools/claim_check.py`'s module docstring and research/FAILURE_LOG.md 2026-09-25);
round 4 rebuilt the scoping on a CommonMark parser.

WHAT THIS GATE DOES, in two parts:
  * `selftest()` runs `tools.claim_check.selftest()` -- every case in `SELFTEST_CASES`, one per hole in the
    history, in both directions -- so the registry refuses to trust the commit gate's own checker whenever a
    scope regression makes any case give the wrong verdict. The registry runs selftests on EVERY commit, so a
    regression is caught before a real finding slips through, with no hook edit.
  * `check()` flags an ADDED findings doc with LOW COVERAGE: >= `LOW_COVERAGE_MIN_TOTAL` numeric claims and
    fewer than `MIN_CHECK_FRACTION` of them checked, whatever mechanism exempted them. GATE 2 already fails
    such a doc; this is the same `_scan()` result, reported under the class that names the failure.

REUSE, NOT REIMPLEMENTATION: both parts call `tools.claim_check` (the same `_scan()` the CLI and
`tools/finding_lint.py` use), so the three can never disagree.

WHAT IT CANNOT CATCH: a doc whose exemptions are individually scoped correctly but whose authors mark numbers
that are not derived at all -- a marker is a claim by the author, and no scope rule can check its truth.
"""
from __future__ import annotations

import glob
import os
import re

import tools.claim_check as claim_check

NAME = "claim-check-scope"
CLASS_ID = "DMS"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FINDING = re.compile(r"^research/findings/[^/]+\.md$")


def _problem(rel, r):
    s = r["suppressed"]
    return ("%s: only %d/%d numeric claim(s) checked (section=%d range=%d block=%d inline=%d exempted) -- "
            "a derived scope should cover the numbers it derives, not the document"
            % (rel, r["checked"], r["total_numeric"], s["section"], s["range"], s["block"], s["inline"]))


def check(paths):
    if paths is None:
        files = [os.path.relpath(p, _ROOT) for p in sorted(glob.glob(os.path.join(_ROOT, "research/findings/*.md")))]
    else:
        files = [p.replace(os.sep, "/") for p in paths if _FINDING.match(p.replace(os.sep, "/"))]
    out = []
    for rel in files:
        full = os.path.join(_ROOT, rel)
        if not os.path.isfile(full):
            continue
        r = claim_check._scan(full)
        if r["low_coverage"]:
            out.append(_problem(rel, r))
    return out


def selftest():
    """Both directions: the full claim_check case registry, plus low coverage through `_scan()` directly."""
    import json
    import tempfile
    problems = list(claim_check.selftest())
    with tempfile.TemporaryDirectory(dir=_ROOT, prefix=".claim_check_scope_selftest_") as d:
        art_abs = os.path.join(d, "art.json")
        json.dump({"accuracy": 0.17}, open(art_abs, "w"))
        art = os.path.relpath(art_abs, _ROOT).replace(os.sep, "/")
        n = claim_check.LOW_COVERAGE_MIN_TOTAL + 5
        bad = os.path.join(d, "bad.md")
        open(bad, "w", encoding="utf-8").write("# Over-marked\n\nArtifact: `%s`\n\n%s\n" % (
            art, "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(n))))
        if not claim_check._scan(bad)["low_coverage"]:
            problems.append("SELFTEST BROKEN: a %d-claim doc exempting every claim was NOT flagged low-coverage" % n)
        good = os.path.join(d, "good.md")
        body = ["The accuracy is 0.170000 (matches the artifact)."] * (n - 1)
        body.append("<!--derived-->\n| metric | value |\n|---|---|\n| ratio | 0.104615 |")
        open(good, "w", encoding="utf-8").write("# Cited\n\nArtifact: `%s`\n\n%s\n" % (art, "\n\n".join(body)))
        if claim_check._scan(good)["low_coverage"]:
            problems.append("SELFTEST BROKEN: a mostly-checked doc with one small derived table was flagged")
    return problems


if __name__ == "__main__":
    hits = check(None)
    print("claim-check-scope corpus audit: %d finding(s) checked (almost) none of their own numeric claims"
          % len(hits))
    for h in hits[:60]:
        print("  ⛔", h[:220])
