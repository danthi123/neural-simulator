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
    .venv/bin/python tools/claim_check.py --selftest

Exit 1 if a measurement-shaped number or a verdict word is unsupported by the cited artifacts.

CALIBRATION -- deliberately narrow, because a checker that cries wolf gets ignored (this project's own lesson,
learned twice today). It checks ONLY:
  * numbers with >= 3 decimal places, which are measurements rather than prose ("3 seeds", "97%" are ignored);
  * verdict words that contradict a cited artifact's own verdict field.
Derived values (ratios, differences, percentages) are legitimately absent from artifacts, so a `<!--derived-->`
marker exempts a number FROM the check -- but (round 5, 2026-09-25) ONLY when the marker occurs on the SAME
PHYSICAL LINE as the number. There is no other scope.

HISTORY -- why round 5 has no scope rule at all, when four earlier rounds each had one. Main's original rule let
a marker ALONE on its own line open a scope lasting until the next `## ` heading; a scorer used that to hide
three whole sections (0 of 336 artifact values checked, two wrong numbers passed). Every round since has been a
markup-scope rule closing the previous hole and opening a new one of the same shape:
  r1 (a line-scanner over headings/tables/close-markers): a `###` heading did not end an open scope (only `## `
     did); a table followed by a wrong number with no blank line between them was absorbed into the table's
     scope; a close marker sharing a line with other text let the text AFTER it pass too.
  r2 (research/claimcheck-block-scope-round2): a `# derived` comment inside a FENCED CODE BLOCK was read as a
     markdown heading, opening a section around ordinary code; a list's scope covered only its first bullet; a
     fence ended an open list/paragraph early, closing real derived content prematurely.
  r3 (research/claimcheck-block-scope-r3): the fence-open/closed boolean toggle desynced on an unmatched or
     mismatched fence (```` closed by ```, or `~~~`), so a `## Results` heading sitting after a bogus "still
     inside a fence" state was swallowed into the preceding "Derived" section.
  r4 (research/claimcheck-parser-scope @ dde18d55395a2b8e99ac64b61b4939d33a90d132, rebuilt on a real CommonMark
     parser -- markdown-it-py -- specifically to close the whole CLASS of line-scanner desync bugs above; REVIEWED
     UNSOUND): an HTML block or an unclosed HTML comment / `<pre>` could still hide a `## Results` heading inside
     a "Derived" section (the parser treats the swallowed heading text as inert block content, never emitting a
     heading token for it); an inline close marker written on a LATER, unrelated line paired with the MOST
     RECENT unpaired STANDALONE opener rather than anything nearby, hijacking an early opener into a range
     spanning a real heading; an h1 or setext "Derived..." heading, or a `>`-nested `## Derived`, opened a
     section with no container-aware end, so its "same-or-higher heading" boundary could sit arbitrarily far
     away or leak straight out of the blockquote; a table nested inside a blockquote or list item was resolved
     against the WRONG container's next-sibling, leaking scope past the list/blockquote that should have bounded
     it; and markdown-it-py itself was an undeclared transitive dependency (present in the dev venv only because
     someone had `pip install`ed it by hand -- a fresh checkout has no declared reason to have it).

Every one of those is the SAME failure shape: a rule that lets the marker on line N exempt a number on some
line M != N, and a way to make the checker misjudge where N's influence stops. Round 5 removes the concept
instead of patching its boundary again: there is no scope, no heading, no section, no range, no fence-awareness,
no markdown parser. A number is exempt if and only if the literal string `<!--derived-->` occurs somewhere on
ITS OWN physical line -- full stop. A marker alone on a line, a `## Derived`/setext/blockquoted heading, and a
`<!--/derived-->` close marker are all now INERT (they exempt only the line they sit on, which typically holds
no numbers); `check()` prints a non-blocking WARNING wherever it sees one of those spellings, because they meant
something real for a year and an author should not be silently un-exempted.

LOW COVERAGE (defense in depth, independent of the marker rule; unchanged in kind from the r4 draft, values
recalibrated for the line-only rule below MIN_CHECK_FRACTION/LOW_COVERAGE_MIN_TOTAL). A non-synthesis doc with
at least LOW_COVERAGE_MIN_TOTAL numeric claims that checked fewer than MIN_CHECK_FRACTION of them fails,
whatever exempted the rest -- a substantial doc that marks (almost) every claim derived is not "clean", it is
unchecked.
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
DERIVED_CLOSE = "<!--/derived-->"          # no longer scopes anything -- matched only to print a WARNING
SYNTH_RE = re.compile(r"^claim_check:\s*synthesis\s*$", re.M)

# Pre-round-5 scope idioms that are now INERT. Matched loosely (a false-positive here only prints an extra
# WARNING, never blocks), so an author who used one of these on purpose is told, not silently un-exempted.
# A marker alone on its line (optionally inside a blockquote): main's old "open a block" idiom.
_STANDALONE_MARKER_RE = re.compile(r"^\s*(?:>\s*)*" + re.escape(DERIVED_MARK) + r"\s*$")
# An ATX heading (any level, optionally blockquoted, optional emphasis) titled "Derived...": the old "section".
_ATX_DERIVED_RE = re.compile(r"^\s*(?:>\s*)*#{1,6}\s*[*_`]*\s*derived\b", re.I)
# A setext heading is a title line followed immediately by a === / --- underline; checked with 1-line lookahead.
_SETEXT_TITLE_RE = re.compile(r"^\s*(?:>\s*)*[*_`]*\s*derived\b", re.I)
_SETEXT_UNDERLINE_RE = re.compile(r"^\s*(?:=+|-+)\s*$")

# LOW COVERAGE. CALIBRATION (2026-09-25, round 5, stated not guessed, re-derivable with
# `tools/claim_check_retro_scan.py`; raw counts in research/coordination/claimcheck_lineonly_retro_2026-09-25.tsv).
# Under a SAME-LINE-ONLY marker, "checked=0" no longer means "one block marker swept the whole doc" -- it now
# means every single numeric line individually carries the literal marker, which is real, distributed authoring
# effort, not a one-line shortcut. So the population changed shape from r4's calibration and had to be re-scanned,
# not just ported: over the WHOLE research/findings/ corpus (2971 docs), the largest legitimately fully-marked
# non-synthesis doc is 65 numeric claims (0 checked) -- a literature-citation doc where every quoted rate/constant
# already carried its own inline marker (research/findings/2026-08-04-gpi-snr-autonomous-pacemaking-biophysical-
# fallback-RESEARCH.md); the next doc down sits at 40, then a real gap to 27 and below. Restricted to CURRENT
# practice (added since 2026-09-01, 353 docs, matching r4's own reasoning that current authoring practice is the
# relevant population), the ceiling is lower still, at 27. No doc anywhere in the corpus combines >=60 numeric
# claims with <10% checked except that one 65/0 literature doc -- i.e. the "0 of 336 checked" incident shape that
# motivated this check in the first place no longer has a same-scale surviving example under line-only marking,
# because the mechanism that produced it (one marker exempting hundreds of lines) no longer exists. The floor is
# set at 80: comfortably above the observed whole-corpus ceiling (65) so genuinely, laboriously per-line-marked
# docs never trip it, and far below the scale of the original incident, so a doc that reverts to marking
# (almost) everything derived without doing that per-line work still gets caught. Re-scan and move the floor if
# a new legitimate all-derived doc exceeds it.
MIN_CHECK_FRACTION = 0.05
LOW_COVERAGE_MIN_TOTAL = 80


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


def _line_warnings(lines):
    """Lines using a pre-round-5 scope idiom that is now INERT -- for a non-blocking author-facing WARNING.
    Never affects the pass/fail verdict; only what gets printed."""
    warnings = []
    n = len(lines)
    for i, ln in enumerate(lines):
        if _STANDALONE_MARKER_RE.match(ln):
            warnings.append((i + 1, "standalone marker",
                              "a lone <!--derived--> no longer opens a scope -- it exempts only THIS line "
                              "(which holds no numbers of its own). Put the marker on each derived line."))
        if DERIVED_CLOSE in ln:
            warnings.append((i + 1, "close marker",
                              "<!--/derived--> no longer closes a range -- there are no ranges. Remove it, or "
                              "put <!--derived--> on each derived line instead."))
        if _ATX_DERIVED_RE.match(ln):
            warnings.append((i + 1, "'Derived' heading",
                              "a '## Derived'-style heading no longer opens a section. Put the marker on each "
                              "derived line under it."))
        elif i + 1 < n and _SETEXT_TITLE_RE.match(ln) and _SETEXT_UNDERLINE_RE.match(lines[i + 1]):
            warnings.append((i + 1, "'Derived' heading (setext)",
                              "a setext 'Derived' heading no longer opens a section. Put the marker on each "
                              "derived line under it."))
    return warnings


def _scan(doc_path, tol=None):
    """Pure computation, no printing -- shared by the CLI (`check`) and `tools/finding_lint.py`, so both see the
    exact same structured verdict instead of finding_lint re-parsing this module's printed stdout.

    tol=None => RELATIVE tolerance. An absolute 5e-4 let a fabricated 0.9999 match a stored 1.0, so the
    checker's own negative control failed on first run: with ~1000 artifact values, near-misses are common and an
    absolute window is far too loose. Relative tolerance scales with the claim. (Unchanged from main.)
    """
    text = open(doc_path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    # A SYNTHESIS doc quotes other experiments throughout its prose; line-by-line marking degenerates into
    # decorating every paragraph, which is how a check stops being read. `claim_check: synthesis` in frontmatter
    # suppresses the per-line rule -- but NOT the citation requirement: it must still cite artifacts that exist,
    # so the escape cannot be used to publish an uncited claim. Chosen deliberately over --no-verify, which would
    # bypass every gate silently and leave no record of which document was exempted or why. (Unchanged from main.)
    synthesis = text.startswith("---") and bool(SYNTH_RE.search(text.split("\n---", 1)[0]))
    cited = sorted(set(PATH_RE.findall(text)))
    nums, verdicts, loaded, missing = load_artifacts(cited)

    unsupported, checked = [], 0
    suppressed = {"inline": 0, "synthesis": 0}
    marked_lines = []                      # 1-indexed lines carrying a literal <!--derived--> that exempted them

    for i, ln in enumerate(lines, 1):
        matches = list(NUM_RE.finditer(ln))
        if not matches:
            continue
        has_marker = DERIVED_MARK in ln
        if has_marker:
            marked_lines.append(i)
            suppressed["inline"] += len(matches)
            continue
        if synthesis:
            suppressed["synthesis"] += len(matches)
            continue
        for m in matches:
            val = float(m.group(1))
            checked += 1
            eps = tol if tol is not None else max(5e-6, 1e-4 * abs(val))
            if not any(abs(val - a) <= eps for a in nums):
                unsupported.append((i, val, ln.strip()[:88]))

    if synthesis and not cited:
        unsupported.append((0, 0.0, "synthesis doc cites NO artifact — the escape still requires citations"))

    total_numeric = checked + suppressed["inline"] + suppressed["synthesis"]
    low_coverage = (not synthesis and total_numeric >= LOW_COVERAGE_MIN_TOTAL
                    and (checked / total_numeric) < MIN_CHECK_FRACTION)

    return dict(cited=cited, nums=nums, loaded=loaded, missing=missing, checked=checked,
                suppressed=suppressed, total_numeric=total_numeric, unsupported=unsupported,
                synthesis=synthesis, low_coverage=low_coverage, marked_lines=marked_lines,
                warnings=_line_warnings(lines))


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
        # ALWAYS printed (2026-07-31 rule, kept in round 5): how many numbers were exempted and on which lines --
        # the incident this whole file exists to close reported "0 checked" with nothing to say WHY.
        print("  exempted        : %d by inline <!--derived--> on line(s) %s, %d by synthesis, of %d numeric "
              "claim(s) found"
              % (suppressed["inline"], ", ".join(str(n) for n in r["marked_lines"]) if r["marked_lines"] else "-",
                 suppressed["synthesis"], total_numeric))
        for lineno, kind, msg in r["warnings"]:
            print("      ⚠️  WARNING line %-4d %-24s no longer exempts anything — %s" % (lineno, kind, msg))
        for lineno, val, ctx in unsupported[:12]:
            print("      ⛔ line %-4d %-14g not in any cited artifact | %s" % (lineno, val, ctx))
        if len(unsupported) > 12:
            print("      ... and %d more" % (len(unsupported) - 12))
        if low_coverage:
            print("      ⛔ LOW COVERAGE: only %d/%d (%.0f%%) numeric claim(s) were actually checked -- the "
                  "rest were marked <!--derived--> on their own line. A doc this size should not be almost "
                  "entirely derived; mark the specific derived numbers, not the whole document."
                  % (checked, total_numeric, 100.0 * checked / total_numeric if total_numeric else 0.0))

    fail = bool(missing) or bool(unsupported) or low_coverage
    if verbose:
        if not cited:
            print("  ⚠️  NO ARTIFACT CITED — a findings doc with no artifact path cannot be checked at all.")
        print("  => %s" % ("⛔ UNSUPPORTED CLAIMS (or missing artifacts) — fix, cite, or mark <!--derived--> "
                           "on the SAME LINE as the number"
                           if fail else "✔ every measurement traces to a cited artifact"))
    return 0 if not fail else 1


# ---------------------------------------------------------------------------------------------------------------
# SELFTEST REGISTRY. `tests/test_claim_check_line_only.py` re-runs every case here against the historical
# revisions named in `wrong_on` (loaded straight from git, not retyped), so the "this used to pass, now it
# fails" claim is re-derived every run rather than remembered. `%(art)s` is a cited artifact holding
# accuracy=0.17 and baseline=0.1625; 0.1525 / 0.140 / 1.23456 are WRONG numbers (not in the artifact);
# 0.104615 / 0.207531 / 0.311079 are legitimately derived ones.
# ---------------------------------------------------------------------------------------------------------------
_HDR = "# Some finding\n\nArtifact: `%(art)s`\n\n"
SELFTEST_CASES = [
    # --- main's original hole -------------------------------------------------------------------------------
    dict(name="incident_standalone_marker_after_heading", expect="FAIL", wrong_on=("main",),
         why="main's hole: a marker alone right after a `## ` heading exempted the whole section after it",
         doc=_HDR + "## Section A\n<!--derived-->\nRunner: some_runner.py, no numbers in this sentence.\n\n"
                    "The accuracy was 0.1525 here.\n\n## Section B\n<!--derived-->\n"
                    "Runner: some_runner.py, no numbers in this sentence.\n\nThe baseline was 0.140 here.\n"),
    # --- r1: paragraph/line-scanner scope --------------------------------------------------------------------
    dict(name="r1_h3_heading_does_not_end_scope", expect="FAIL", wrong_on=("main", "r1"),
         why="r1's hole: only `## ` ended an open scope, so a `### ` heading right after a marker did not",
         doc=_HDR + "<!--derived-->\n### A subheading, not a level-2 one\nThe accuracy was 0.1525 here.\n"),
    dict(name="r1_table_then_wrong_number_no_blank_line", expect="FAIL", wrong_on=("main", "r1"),
         why="r1's hole: a table right after a marker absorbed a wrong number on the very next line, no blank "
             "line needed",
         doc=_HDR + "<!--derived-->\n| metric | value |\n|---|---|\n| ratio | 0.104615 |\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r1_close_marker_midline_trailing_checked", expect="FAIL", wrong_on=("main", "r1"),
         why="r1's hole: text AFTER a close marker on the same line was swallowed into the range too",
         doc=_HDR + "<!--derived-->\nThe ratio is 0.104615 here. <!--/derived--> The real accuracy is 0.1525 here.\n"),
    # --- r2: fence/list handling -------------------------------------------------------------------------------
    dict(name="r2_hash_derived_comment_in_fence_read_as_heading", expect="FAIL", wrong_on=("r2",),
         why="r2's hole: a `# derived` comment inside a FENCED code block was read as a markdown heading, "
             "opening a Derived section around ordinary code",
         doc=_HDR + "```python\n# derived thresholds below\nvalue = 1.23456\n```\nThe accuracy was 0.1525 here.\n"),
    # --- r3: fence-open/closed boolean toggle --------------------------------------------------------------
    dict(name="r3_mismatched_fence_swallows_heading", expect="FAIL", wrong_on=("r3",),
         why="r3's hole: a `~~~` fence is not closed by a ` ``` ` fence, so the toggle thought the doc was still "
             "inside a fence and the `## Results` heading after the REAL close never ended the Derived section",
         doc=_HDR + "## Derived\nratio 0.104615\n~~~\n```\n~~~\n## Results\nThe accuracy was 0.1525 here.\n"),
    # --- r4 (research/claimcheck-parser-scope @ dde18d55395a2b8e99ac64b61b4939d33a90d132) -- REVIEWED UNSOUND --
    dict(name="r4_unclosed_html_comment_hides_results_heading", expect="FAIL", wrong_on=("r4",),
         why="r4's hole: an unclosed HTML comment right after a Derived section swallows the `## Results` "
             "heading as inert block content, so the parser never emits a heading token to end the section",
         doc=_HDR + "## Derived\nratio 0.104615\n<!-- note, never closed\n## Results\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r4_later_inline_close_hijacks_earlier_standalone_across_heading", expect="FAIL", wrong_on=("r4",),
         why="r4's hole: a close marker embedded in an unrelated LATER paragraph paired with the most recent "
             "unpaired STANDALONE opener rather than anything nearby, stretching an early range across a real "
             "heading and a wrong number in between",
         doc=_HDR + "<!--derived-->\nratio 0.104615\n\n## Results\nThe accuracy was 0.1525 here.\n\n"
                    "A later aside adds a note, value 0.207531 here. <!--/derived-->\n"),
    dict(name="r4_h1_derived_heading_oversized_section", expect="FAIL", wrong_on=("r2", "r3", "r4"),
         why="an h1 'Derived' heading has no same-or-higher heading after it in a short doc, so its section ran "
             "to end of document (r2/r3's simpler heading match also treats it as opening a section here)",
         doc=_HDR + "# Derived\nratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_setext_derived_heading_oversized_section", expect="FAIL", wrong_on=("r4",),
         why="r4's hole: a setext ('Derived\\n=======') heading was never checked for at all, so nothing ever "
             "closed the section it should have opened -- OR (this exact case) opened one no scope rule saw, "
             "leaving a wrong number unchecked by coincidence of a totally different bug; either way the wrong "
             "number here must be caught",
         doc=_HDR + "Derived\n=======\nratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_blockquoted_derived_heading_leaks_scope", expect="FAIL", wrong_on=("r4",),
         why="r4's hole: a `## Derived` heading nested inside a blockquote was still recognised as a top-level "
             "heading token (the flat token stream is not container-aware), so its section boundary leaked "
             "straight out of the blockquote into ordinary top-level prose",
         doc=_HDR + "> ## Derived\n> ratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_list_item_table_leaks_scope_to_sibling_item", expect="FAIL", wrong_on=("main", "r1", "r4"),
         why="r4's hole: a marker's list-scope legitimately covers a table NESTED in its first item, but the "
             "same scope then leaks to a SIBLING item's wrong number too -- a table inside a list/blockquote "
             "item should not license the whole list around it",
         doc=_HDR + "<!--derived-->\n- item one:\n  | metric | value |\n  |---|---|\n  | ratio | 0.104615 |\n"
                    "- item two: accuracy 0.1525\n"),
    # --- round 5's own contract: inline-only marking, both directions -----------------------------------------
    dict(name="line_marked_derived_number_passes", expect="PASS", wrong_on=(),
         why="the ONE thing the new rule allows: the marker on the SAME physical line as the number",
         doc=_HDR + "The ratio is 0.104615 here. <!--derived-->\nThe baseline was 0.162500 here.\n"),
    dict(name="table_with_marker_on_every_derived_row_passes", expect="PASS", wrong_on=(),
         why="a table whose derived rows each carry the marker passes -- no table-awareness needed, the marker "
             "is just text on that row's physical line",
         doc=_HDR + "| metric | value | |\n|---|---|---|\n| ratio | 0.104615 | <!--derived--> |\n"
                    "| gap | 0.207531 | <!--derived--> |\n| accuracy | 0.170000 | |\n"),
    dict(name="marker_on_wrong_line_does_not_reach_over", expect="FAIL", wrong_on=("main", "r1", "r2", "r3", "r4"),
         why="round 5's own rule: a marker one line away from the number it was meant to cover does not reach "
             "it -- every earlier round's whole point was letting a marker reach beyond its own line",
         doc=_HDR + "<!--derived-->\nThe accuracy was 0.1525 here.\n"),
    dict(name="standalone_marker_and_derived_heading_now_inert_but_do_not_crash",
         expect="FAIL", wrong_on=("main", "r1", "r2", "r3", "r4"),
         why="a standalone marker AND a '## Derived' heading both present, neither doing anything under line-"
             "only -- a wrong number right after them is still caught, though every one of the five earlier "
             "revisions treated this doc's tail as an open Derived section with nothing left to close it",
         doc=_HDR + "## Derived\n<!--derived-->\nThe accuracy was 0.1525 here.\n"),
    dict(name="low_coverage_overmarked", expect="FAIL", wrong_on=("main",),
         why="a substantial doc that marks (almost) every claim derived fails on LOW COVERAGE regardless of "
             "what exempted them",
         doc=_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1)
                                for i in range(LOW_COVERAGE_MIN_TOTAL + 5)) + "\n"),
]

_HISTORY_SHAS = {"main": "7e2edc08e", "r1": "d4959ecb0", "r2": "6abb28469", "r3": "662e167e8", "r4": "214e509bf"}


def _write_case(d, case):
    """Write one selftest case (and its artifact) under directory `d`, which must be inside ROOT: PATH_RE drops
    a leading '/', so a real finding's ROOT-relative citation is the only form that resolves the same way."""
    art_abs = os.path.join(d, "art.json")
    if not os.path.exists(art_abs):
        json.dump({"accuracy": 0.17, "baseline": 0.1625}, open(art_abs, "w"))
    art = os.path.relpath(art_abs, ROOT).replace(os.sep, "/")
    path = os.path.join(d, case["name"] + ".md")
    open(path, "w", encoding="utf-8").write(case["doc"] % {"art": art})
    return path


#  the r1-r4 reconstruction cases above use placeholder "derived" numbers (0.104615/0.207531/0.311079) that are
# themselves UNMARKED on their own line in those historical docs -- under line-only they are correctly flagged
# too (that IS the round-5 behaviour: a marker that does not sit on a number's own line no longer reaches it).
# So the FAIL check below only requires the WRONG number to be among the flagged ones, not that flagged == wrong.
WRONG_VALUES = {0.1525, 0.14, 1.23456}


def selftest():
    """Same contract as `tools/gates/*.selftest()`: a list of problems, empty means the check is trustworthy.
    Runs every SELFTEST_CASES entry in both directions: a FAIL case must flag its designated WRONG number (or
    trip LOW COVERAGE), a PASS case must flag nothing at all."""
    import tempfile
    problems = []
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".claim_check_selftest_") as d:
        for case in SELFTEST_CASES:
            p = _write_case(d, case)
            r = _scan(p)
            got = "FAIL" if (bool(r["missing"]) or bool(r["unsupported"]) or r["low_coverage"]) else "PASS"
            if got != case["expect"]:
                problems.append("SELFTEST BROKEN: case %s expected %s, got %s (%s)"
                                % (case["name"], case["expect"], got, case["why"]))
                continue
            flagged = {round(v, 6) for _ln, v, _c in r["unsupported"]}
            if case["expect"] == "FAIL":
                if not (flagged & WRONG_VALUES) and not r["low_coverage"]:
                    problems.append("SELFTEST BROKEN: case %s failed but never flagged its designated wrong "
                                    "number (%s): flagged=%s" % (case["name"], sorted(WRONG_VALUES), flagged))
            elif flagged:
                problems.append("SELFTEST BROKEN: case %s is supposed to PASS clean but flagged: %s"
                                % (case["name"], sorted(flagged)))
    return problems


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        problems = selftest()
        for p in problems:
            print("⛔", p)
        print("claim_check selftest: %s" % ("FAILED" if problems else
                                             "OK (%d cases, both directions)" % len(SELFTEST_CASES)))
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
