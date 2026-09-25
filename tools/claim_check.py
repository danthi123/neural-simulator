#!/usr/bin/env python3
"""Verify a document's NUMBERS and VERDICTS against the artifacts it cites. Blocks hallucinated claims.

WHY (2026-07-31, owner directive). The experiment harness guards EXPERIMENTS. The larger failure class is
CLAIMS -- statements made in findings, commit messages, the board, and to the owner, that were never traced back
to an artifact. That class is not hypothetical:

  * I told the owner "lanes A/B/C/E have banked 6-seed GOs". Lane A's own artifact
    (research/findings/raw/_affect_state_region_6seed.json) reads {"GO": false, "n_seeds_go": 2}. I had repeated
    the BOARD's summary line without opening the JSON. The finding itself was honest; the summary overclaimed.
  * A "2.97 at 16x16 with NO heuristic" claim stood for 2.5 months and propagated into CLAUDE.md; the run's own
    recorded command showed the flag that closes the heuristic was absent, and its default is ON.
  * A headline "circ_dW 0.7050 = 105% of the 0.6705 reference" was real as a measurement and wrong as a claim.

THE RULE THIS ENFORCES: a measurement stated in a document must EXIST in an artifact the document cites.
Not "be plausible". Not "be remembered". Exist, in a file, that a reader can open.

    .venv/bin/python tools/claim_check.py research/findings/2026-07-31-foo.md

Exit 1 if a measurement-shaped number or a verdict word is unsupported by the cited artifacts.

CALIBRATION -- deliberately narrow, because a checker that cries wolf gets ignored (this project's own lesson,
learned twice today). It checks ONLY:
  * numbers with >= 3 decimal places, which are measurements rather than prose ("3 seeds", "97%" are ignored);
  * verdict words that contradict a cited artifact's own verdict field.
Derived values (ratios, differences, percentages) are legitimately absent from artifacts, so an inline
`<!--derived-->` marker on the line, or a `## Derived` section listing them, suppresses the check for that line.

THE BLOCK-SCOPE HOLE (found 2026-09-25, independent verifier). A standalone `<!--derived-->` marker opened
BLOCK scope until the next `## ` heading -- so a marker placed right after a heading suppressed EVERY number in
that WHOLE section, and a doc with the marker after several headings suppressed every number in ALL of them. A
scorer did exactly this: the run reported "0 checked against 336 artifact values" and still printed "every
measurement traces to a cited artifact", while two numbers in the document (0.1525, 0.140) did not match the
artifact it cited (0.17, 0.1625). THE FIX, two layers:
  1. A standalone marker now opens PARAGRAPH scope: it closes at the next blank line, the next heading, or an
     explicit `<!--/derived-->` close marker -- whichever comes first. A `## Derived` heading SECTION (a whole
     section titled "Derived") is a separate, deliberate mechanism and keeps its section-until-next-heading
     scope; that one was never the hole.
  2. Independent of scope logic: the report ALWAYS prints how many numbers were suppressed and by which marker
     (section / block / inline / synthesis), and the check FAILS (not passes) when a non-synthesis doc has
     numeric claims but examined ZERO of them -- the exact shape of the incident, regardless of which future
     mechanism produces a total suppression.
`selftest()` demonstrates both directions: a doc with the ORIGINAL failure shape (marker after headings, wrong
numbers in the sections that follow) still FAILS; a doc with a small, correctly paragraph-scoped derived table
still PASSES.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# >=3 decimals => a measurement, not prose. "6 seeds", "97%", "2.5 months" are not claims about instrument output.
NUM_RE = re.compile(r"(?<![\w.])(-?\d+\.\d{3,})(?![\w])")
# Globs are allowed: a finding over N seeds cites one pattern, not N paths.
# Must contain a "/" -- a bare filename mentioned in prose ("as g5fix_d025_*.json shows") is a REFERENCE, not a
# citation, and treating it as one reports a missing artifact that was never claimed to be a path.
PATH_RE = re.compile(r"([\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.(?:jsonl|json))")
VERDICT_RE = re.compile(r"\b(GO|NO-GO|PASS|FAIL|REFUTED|CONFIRMED)\b")
DERIVED_MARK = "<!--derived-->"
DERIVED_CLOSE = "<!--/derived-->"          # explicit close, for a legitimate derived block wider than one paragraph

# A non-synthesis doc that examined this fraction or less of its own numeric claims is treated the same as
# examining ZERO of them -- the 2026-09-25 incident's ratio was 0/~50 (0%), but the failure mode is "suppressed
# almost everything", not literally "suppressed everything".
MIN_CHECK_FRACTION = 0.05
# ... AND ONLY when the doc has at least this many numeric claims in total. CALIBRATED, not guessed: a
# checked==0-of-everything retro-scan over the 353 `research/findings/*.md` added since 2026-09-01 (2026-09-25)
# found 71 documents that are ENTIRELY, LEGITIMATELY derived (a short diagnosis note built purely from
# already-published ratios/deltas/percentages, correctly all-marked) -- up to 39 numeric claims, 0 checked, by
# design, not by error. A blanket "checked==0 fails" rule would have blocked every one of them. The real incident
# examined 0 of ~50 numeric claims in a SUBSTANTIAL document (336 candidate artifact values, ~43-53 claims in the
# doc itself). This floor sits just above the largest legitimate all-derived doc in the corpus (39) and well
# below the incident's scale, so it passes every real 2026-09-01+ document while still catching the observed
# shape (a large document reporting essentially nothing checked). Retro-scan: see the 2026-09-25 branch notes.
LOW_COVERAGE_MIN_TOTAL = 40


def _flatten_numbers(obj, out):
    """Every numeric leaf in an artifact, at any depth."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 6))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)


def _flatten_verdicts(obj, out):
    """Any key that looks like a verdict, with its value -- so a doc saying GO can be checked against GO:false."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in ("go", "verdict", "overall_verdict", "passed", "signal"):
                out.append((k, v))
            _flatten_verdicts(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_verdicts(v, out)


def load_artifacts(paths):
    nums, verdicts, loaded, missing = set(), [], [], []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        hits = sorted(glob.glob(full)) if any(c in full for c in "*?[") else ([full] if os.path.exists(full) else [])
        if not hits:
            missing.append(p)
            continue
        for h in hits:
            try:
                if h.endswith(".jsonl"):
                    for ln in open(h):
                        ln = ln.strip()
                        if ln:
                            d = json.loads(ln)
                            _flatten_numbers(d, nums); _flatten_verdicts(d, verdicts)
                else:
                    d = json.load(open(h))
                    _flatten_numbers(d, nums); _flatten_verdicts(d, verdicts)
                loaded.append(h)
            except Exception as e:                       # narrow enough to see; never silent
                missing.append("%s (unreadable: %s)" % (p, type(e).__name__))
    return nums, verdicts, loaded, missing


SYNTH_RE = re.compile(r"^claim_check:\s*synthesis\s*$", re.M)


def _scan(doc_path, tol=None):
    """Pure computation, no printing -- shared by the CLI (`check`), `finding_lint.py` and the registry gate
    `tools/gates/claim_check_coverage.py`, so all three see the exact same verdict.

    tol=None => RELATIVE tolerance. An absolute 5e-4 let a fabricated 0.9999 match a stored 1.0, so the
    checker's own negative control failed on first run: with ~1000 artifact values, near-misses are common and an
    absolute window is far too loose. Relative tolerance scales with the claim.
    """
    text = open(doc_path).read()
    lines = text.split("\n")
    # A SYNTHESIS doc quotes other experiments throughout its prose; line-by-line marking degenerates into
    # decorating every paragraph, which is how a check stops being read. `claim_check: synthesis` in frontmatter
    # suppresses the per-line rule -- but NOT the citation requirement: it must still cite artifacts that exist,
    # so the escape cannot be used to publish an uncited claim. Chosen deliberately over --no-verify, which would
    # bypass every gate silently and leave no record of which document was exempted or why.
    synthesis = text.startswith("---") and bool(SYNTH_RE.search(text.split("\n---", 1)[0]))
    cited = sorted(set(PATH_RE.findall(text)))
    nums, verdicts, loaded, missing = load_artifacts(cited)

    unsupported, checked = [], 0
    suppressed = {"section": 0, "block": 0, "inline": 0, "synthesis": 0}
    in_section = False     # a `## Derived` HEADING section -- deliberately scoped until the next heading
    in_block = False       # a standalone marker's PARAGRAPH scope -- closes at a blank line/heading/close-marker
    block_has_content = False  # blank lines BETWEEN the marker and its paragraph (common authoring style: the
    # marker sits alone, then a blank line for visual separation, THEN the derived paragraph/table) do not
    # themselves end the block -- only a blank line AFTER real content has started does. Corpus-measured
    # (2026-09-25 retro-scan): treating the marker's OWN following blank line as an immediate close broke this
    # exact, common, legitimate pattern in real findings (e.g. a `<!--derived-->` line, a blank line, THEN the
    # paragraph it covers) with no security benefit -- the incident shape is a scope that outlives ONE paragraph,
    # not a blank line existing at all near the marker.
    for i, ln in enumerate(lines, 1):
        stripped = ln.strip()

        # Heading transitions. A `## Derived` heading opens the section mechanism; ANY OTHER `## ` heading ends
        # both mechanisms -- ending `in_block` here too is a safety net beyond the blank-line scope below, for a
        # marker block that is never followed by a blank line before the next heading.
        if stripped.lower().startswith("## derived"):
            in_section = True
            in_block = False
            continue
        if ln.startswith("## "):
            in_section = False
            in_block = False

        # An explicit close ends a marker block immediately, for a legitimately-derived block that spans more
        # than one paragraph (e.g. a table followed directly by an explanatory line, no blank line between).
        if stripped == DERIVED_CLOSE:
            in_block = False
            continue
        # A marker ALONE on a line opens PARAGRAPH scope -- THE FIX (2026-09-25): this used to open scope until
        # the next heading, which let one marker suppress whole sections. It now closes at the next blank line
        # AFTER its paragraph/table has started (below), the next heading (above), or `<!--/derived-->` (above)
        # -- whichever comes first.
        if stripped == DERIVED_MARK:
            in_block = True
            block_has_content = False
            continue
        # A blank line BEFORE any content is just spacing between the marker and its paragraph (tolerated). A
        # blank line AFTER content has started ends the paragraph/table -- and with it, the block's scope.
        if in_block and stripped == "":
            if block_has_content:
                in_block = False
            continue

        if in_section:
            reason = "section"
        elif in_block:
            reason = "block"
            block_has_content = True
        elif DERIVED_MARK in ln:
            reason = "inline"
        elif synthesis:
            reason = "synthesis"
        else:
            reason = None

        if reason is not None:
            suppressed[reason] += len(NUM_RE.findall(ln))
            continue

        for m in NUM_RE.finditer(ln):
            val = float(m.group(1))
            checked += 1
            eps = tol if tol is not None else max(5e-6, 1e-4 * abs(val))
            if not any(abs(val - a) <= eps for a in nums):
                unsupported.append((i, val, ln.strip()[:88]))

    if synthesis and not cited:
        unsupported.append((0, 0.0, "synthesis doc cites NO artifact — the escape still requires citations"))

    total_numeric = checked + sum(suppressed.values())
    # LOW COVERAGE (2026-09-25, defense-in-depth independent of the scope fix above): a SUBSTANTIAL non-synthesis
    # doc that examined none/almost-none of its own numeric claims is the same shape as the incident regardless
    # of which future mechanism produces the total suppression -- report it, don't just trust that scope alone
    # closes the class. Gated on LOW_COVERAGE_MIN_TOTAL (see its comment): a SHORT all-derived note is normal
    # here, not suspicious.
    low_coverage = (not synthesis and total_numeric >= LOW_COVERAGE_MIN_TOTAL
                    and (checked == 0 or (checked / total_numeric) < MIN_CHECK_FRACTION))

    return dict(cited=cited, nums=nums, loaded=loaded, missing=missing, checked=checked,
                suppressed=suppressed, total_numeric=total_numeric, unsupported=unsupported,
                synthesis=synthesis, low_coverage=low_coverage)


def check(doc_path, tol=None, verbose=True):
    r = _scan(doc_path, tol)
    checked, suppressed, total_numeric = r["checked"], r["suppressed"], r["total_numeric"]
    unsupported, missing, loaded, cited = r["unsupported"], r["missing"], r["loaded"], r["cited"]
    synthesis, low_coverage = r["synthesis"], r["low_coverage"]

    if verbose:
        print("claim_check: %s" % os.path.relpath(doc_path, ROOT))
        print("  cited artifacts : %d found, %d missing" % (len(loaded), len(missing)))
        for mp in missing[:5]:
            print("      ⛔ MISSING  %s" % mp)
        print("  measurements    : %d checked against %d artifact values%s"
              % (checked, len(r["nums"]), "   [synthesis: per-line rule suppressed, citations still required]"
                 if synthesis else ""))
        # ALWAYS printed (2026-09-25) -- the incident this closes reported "0 checked" with nothing to say WHY.
        print("  suppressed      : %d total by <!--derived--> (section=%d block=%d inline=%d), %d by synthesis "
              "of %d numeric claim(s) found"
              % (suppressed["section"] + suppressed["block"] + suppressed["inline"], suppressed["section"],
                 suppressed["block"], suppressed["inline"], suppressed["synthesis"], total_numeric))
        for lineno, val, ctx in unsupported[:12]:
            print("      ⛔ line %-4d %-14g not in any cited artifact | %s" % (lineno, val, ctx))
        if len(unsupported) > 12:
            print("      ... and %d more" % (len(unsupported) - 12))
        if low_coverage:
            print("      ⛔ LOW COVERAGE: only %d/%d (%.0f%%) numeric claim(s) were actually checked -- the "
                  "rest were suppressed by markers. Verify the <!--derived--> markers are not exempting a whole "
                  "section; a derived block should cover the specific numbers it derives, not surrounding prose."
                  % (checked, total_numeric, 100.0 * checked / total_numeric if total_numeric else 0.0))

    fail = bool(missing) or bool(unsupported) or low_coverage
    if verbose:
        if not cited:
            print("  ⚠️  NO ARTIFACT CITED — a findings doc with no artifact path cannot be checked at all.")
        print("  => %s" % ("⛔ UNSUPPORTED CLAIMS (or missing artifacts) — fix, cite, or mark <!--derived-->"
                           if fail else "✔ every measurement traces to a cited artifact"))
    return 0 if not fail else 1


def selftest():
    """Same contract as `tools/gates/*.selftest()`: return a list of problems, empty means the check itself is
    trustworthy. Demonstrates BOTH directions of the 2026-09-25 block-scope hole:

      (a) FAILING DIRECTION -- a standalone `<!--derived-->` marker placed right after `## ` headings (the
          exact incident shape), with WRONG numbers in the sections that follow, must still FAIL. Under the OLD
          block-until-next-heading scope this PASSED with "0 checked"; if this ever regresses, this selftest
          catches it before a real doc does.
      (b) PASSING DIRECTION -- a legitimately-derived small table, correctly paragraph-scoped (marker, table,
          blank line), sitting next to a normally-checked headline number, must still PASS. A fix that closes
          (a) by over-restricting scope (e.g. treating every marker as inline-only) would break this.
    """
    import tempfile
    problems = []
    # `PATH_RE` (deliberately) requires a citation to start with a word character, so it drops a LEADING "/" off
    # an absolute path when parsing doc prose -- the tempdir lives UNDER ROOT so a ROOT-relative citation (as
    # every real finding uses) resolves exactly the way it does in production, not through that regex quirk.
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        art_abs = os.path.join(d, "art.json")
        json.dump({"accuracy": 0.17, "baseline": 0.1625}, open(art_abs, "w"))
        art = os.path.relpath(art_abs, ROOT).replace(os.sep, "/")

        # Reproduces the REAL incident's shape exactly: a marker right after a heading covers only its OWN
        # immediate paragraph (harmless provenance prose, no numbers) -- the wrong number sits in a LATER
        # paragraph of the SAME section, which the OLD until-next-heading scope also swallowed but this fix does
        # not (it is a separate paragraph, past the closing blank line).
        bad = os.path.join(d, "bad.md")
        open(bad, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "## Section A\n<!--derived-->\n"
            "Runner: some_runner.py, revision deadbeef, no numbers in this sentence at all.\n\n"
            "The accuracy was 0.1525 here.\n\n"
            "## Section B\n<!--derived-->\n"
            "Runner: some_runner.py, revision deadbeef, no numbers in this sentence at all.\n\n"
            "The baseline was 0.140 here.\n"
            "## Section C\n<!--derived-->\nNothing derived here either.\n" % art)
        if check(bad, verbose=False) == 0:
            problems.append("SELFTEST BROKEN: a standalone <!--derived--> marker placed right after `## ` "
                            "headings, with WRONG numbers (0.1525, 0.140) in the sections that follow, PASSED -- "
                            "the 2026-09-25 block-scope hole is back (a scorer's block suppressed every number "
                            "in the sections after it)")

        good = os.path.join(d, "good.md")
        open(good, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "The accuracy is 0.170000 (matches the artifact).\n\n"
            "<!--derived-->\n"
            "| metric | value |\n|---|---|\n| ratio | 0.104615 |\n| delta | 0.007500 |\n\n"
            "## Next section\n\nThe baseline was 0.162500 here (matches the artifact).\n" % art)
        if check(good, verbose=False) != 0:
            problems.append("SELFTEST BROKEN: a legitimately-derived, correctly paragraph-scoped small table "
                            "(next to normally-checked, matching headline numbers) FAILED -- the fix "
                            "over-restricts legitimate derived blocks")

        # a SUBSTANTIAL doc (>= LOW_COVERAGE_MIN_TOTAL claims) that suppresses ALL of them via inline markers,
        # with NO wrong numbers anywhere, must still be flagged LOW COVERAGE -- defense-in-depth independent of
        # scope, sized to the incident (a large doc reporting ~0 checked), not a short legitimately-derived note.
        overmarked = os.path.join(d, "overmarked.md")
        n_claims = LOW_COVERAGE_MIN_TOTAL + 5
        body = "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(n_claims))
        open(overmarked, "w", encoding="utf-8").write("# Over-marked\n\nArtifact: `%s`\n\n%s\n" % (art, body))
        if check(overmarked, verbose=False) == 0:
            problems.append("SELFTEST BROKEN: a %d-claim doc that inline-marks EVERY numeric claim as derived "
                            "(checked=0) PASSED -- the low-coverage safety net did not fire" % n_claims)

        # ... but the SAME pattern on a SHORT doc (below the floor) must still PASS -- a short note built purely
        # from already-derived ratios is normal, not suspicious (2026-09-25 retro-scan: 71 real corpus docs are
        # exactly this shape).
        short_derived = os.path.join(d, "short_derived.md")
        open(short_derived, "w", encoding="utf-8").write(
            "# Short diagnosis note\n\nArtifact: `%s`\n\n"
            "The ratio here is 0.104615. <!--derived-->\n" % art)
        if check(short_derived, verbose=False) != 0:
            problems.append("SELFTEST BROKEN: a SHORT, entirely-derived note (1 claim, below "
                            "LOW_COVERAGE_MIN_TOTAL) FAILED -- the size floor did not protect a legitimate "
                            "short doc")
    return problems


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        problems = selftest()
        for p in problems:
            print("⛔", p)
        print("claim_check selftest: %s" % ("FAILED" if problems else "OK (both directions demonstrated)"))
        return 1 if problems else 0
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for p in sys.argv[1:]:
        rc |= check(p)
    return rc


if __name__ == "__main__":
    sys.exit(main())
