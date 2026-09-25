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
  r5 (2026-09-25, this file at 4fda849d4): deleted the scope concept entirely -- exempt iff the literal marker
     sits on the number's own physical line, full stop. REVIEWED SOUND-WITH-ISSUES: the "physical line" boundary
     itself held, but everything ELSE a document can do to a NUMBER (not to the marker's reach) was untested --
     round 6 below closes those.

Every one of rounds 1-4 was the SAME failure shape: a rule that lets the marker on line N exempt a number on some
line M != N, and a way to make the checker misjudge where N's influence stops. Round 5 deleted the concept
instead of patching its boundary again: a number is exempt if and only if the literal string `<!--derived-->`
occurs somewhere on ITS OWN physical line -- full stop. A marker alone on a line, a `## Derived`/setext/
blockquoted heading, and a `<!--/derived-->` close marker are all now INERT (they exempt only the line they sit
on, which typically holds no numbers); `check()` prints a non-blocking WARNING wherever it sees one of those
spellings, because they meant something real for a year and an author should not be silently un-exempted.

ROUND 6 (2026-09-25, this revision) -- round 5's review (SOUND-WITH-ISSUES) found the "same physical line" rule
itself sound, but every adjacent surface unguarded. Ten fixes, each with its own selftest case declaring the
revisions it corrects (`wrong_on`, re-derived from git every run like every case above):

  1. STRICT UTF-8. Reading the doc with `errors="replace"` silently decoded invalid bytes instead of raising --
     a regression relative to main, which crashed (blocking) on the same input. A doc that is not valid UTF-8
     now reports a blocking UNREADABLE result instead of being silently scanned wrong or crashing uncaught.
  2. (fixed in tools/githooks/pre-commit, not here) the hook's own hint text still described the pre-round-5
     block-scope idiom as current practice.
  3. (fixed in tools/gates/claim_check_selftest.py, not here) GATE 2 in the hook shells out to this file without
     ever running `--selftest`, so the registry's "refuse a gate whose selftest cannot fail" rule never covered
     it. A thin gates wrapper closes that.
  4. PER-LINE EXEMPTION IS NOW BOUNDED, not blanket. Two narrowings, because either alone still leaks: (a) a
     marked line/cell exempts at most MAX_EXEMPT_PER_LINE numbers -- past that, the doc is almost certainly using
     the marker as a section-opener again, not marking individual values; (b) inside a markdown TABLE ROW, a
     marker exempts only the CELL it sits in (split on `|`), and inside any line, only the segment before/after
     an HTML `<br>` it sits in (a `<br>` renders as two lines to a reader even though it is one physical line to
     this scanner) -- so a marker in a "delta" column can no longer excuse a wrong MEASURED number in an earlier
     column. Splitting narrows the exemption only, so this fails closed even for a row this scanner misclassifies.
  5. SYNTHESIS now requires (a) a PROPERLY CLOSED frontmatter block (`---` at position 0, and a LATER line that
     is exactly `---`, not just the first `\n---` found anywhere in the doc -- an unclosed frontmatter block
     could hide the flag past the real content), (b) a non-empty `claim_check_reason:` field in that SAME block
     (a stated reason, not just the bare flag), and (c) is BARRED outright when the doc's own H1 title states a
     verdict word (GO/NO-GO/PASS/FAIL/REFUTED/CONFIRMED) -- the exact incident shape this file exists to close
     was a live verdict hiding wrong numbers behind a blanket exemption, and a verdict-bearing title is precisely
     the case where every number must still be checked. A plain literature-survey doc with no such title, that
     states why it is exempt, may still use the escape. Declaring the flag without satisfying (a)+(b) does not
     block on its own -- it silently falls back to the normal per-line + LOW_COVERAGE rules (the strict default),
     with a non-blocking WARNING explaining why the escape did not apply.
  6. NORMALIZE BEFORE MATCHING. A number's own digits were escapable in ways NUM_RE never saw: markdown emphasis
     (`_0.9876_`, `**0.9876**`), a glued unit (`0.9876ms`, `1.9876x`), scientific notation (`9.876e-1`), a
     leading dot (`.9876`), a markdown backslash-escape (`0\\.9876`), an HTML numeric entity (`0&#46;9876`), an
     empty inline tag splitting the digits (`0.9<!---->876`, `0.9<span></span>876`), a zero-width/soft-hyphen/
     word-joiner character among the digits (any Unicode category Cf codepoint), and the two dash glyphs that
     get typed in place of an ASCII minus (U+2212 MINUS SIGN, U+2013 EN DASH -- both were silently DROPPED,
     turning a sign flip into a false match). `_normalize_for_numbers` folds all of these to plain ASCII digits
     before NUM_RE ever runs, and NUM_RE itself now accepts a leading-dot mantissa, an optional exponent, and a
     glued trailing unit. This normalized copy is used ONLY for number-scanning -- PATH_RE still runs against
     the ORIGINAL text, so a real underscore inside a filename citation is untouched.
  7. LOW_COVERAGE's fraction now counts only DISTINCT checked values (rounded to 6dp), not raw occurrences, and
     ignores any number that appears only inside an HTML comment. Eleven literal copies of one real artifact
     value, pasted inside `<!-- -->` purely to pad the "checked" count past the 5% floor, used to count as 11;
     now they count as at most 1 distinct value, and if the comment is otherwise untouched they count as 0.
  8. (fixed in tests/test_claim_check_line_only.py, not here) the historical-revision test depended on
     markdown-it-py through round 4 even when nothing else needs it; missing the package turned 9 skips into
     9 failures. The history fixture now drops round 4 from consideration when the package is not importable.
  9. CITATIONS INSIDE HTML COMMENTS ARE IGNORED (the same comment-stripping as #7, applied before PATH_RE runs),
     and artifact LOADING is capped: at most MAX_GLOB_FILES files per glob, and at most MAX_ARTIFACT_VALUES
     DISTINCT values pooled across every cited artifact. Citing a handful of huge/unrelated raw directories used
     to pool tens of thousands of values, at which point a random 4-decimal float has real odds of landing near
     one of them by pure chance. The caps are calibrated against the corpus's own largest LEGITIMATE per-battery
     citation (research/coordination/claimcheck_caps_calibration_2026-09-25.txt: 17 real glob citations expand
     past 20 files; the biggest SPECIFIC per-battery one pools 8,266 distinct values across 481 files,
     consol_opsweep_gpu; only the one deliberately whole-tree pattern, `raw/**/*.json`, goes further) --
     comfortably below both caps, well below the scale of citing a whole raw/ tree. This is forward-only like
     every other rule here: the gate only checks NEWLY ADDED findings, so an existing committed doc that already
     relies on a huge citation is untouched.
 10. (nit, not reproducible in this checkout -- see the round-6 commit message) a stray count of
     tests/test_doc_rules.py's test functions was off by one in an earlier round's own report; the true count
     (2) is what tests/test_doc_rules.py itself defines and is not restated elsewhere in this repo to drift.

LOW COVERAGE (defense in depth, independent of the marker rule; unchanged in kind from the r4 draft, values
recalibrated for the line-only rule below MIN_CHECK_FRACTION/LOW_COVERAGE_MIN_TOTAL). A non-synthesis doc with
at least LOW_COVERAGE_MIN_TOTAL numeric claims that checked fewer than MIN_CHECK_FRACTION of them fails,
whatever exempted the rest -- a substantial doc that marks (almost) every claim derived is not "clean", it is
unchecked.
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# >=3 decimals => a measurement, not prose. "6 seeds", "97%", "2.5 months" are not claims about instrument output.
# Round 6 widening (issue 6): `\d*` (not `\d+`) admits a leading-dot mantissa (`.9876`); the optional exponent
# group admits scientific notation (`9.876e-1`); the trailing lookahead now forbids only another DIGIT (not a
# dot, and not any word character), so a glued unit (`0.9876ms`, `1.9876x`) no longer hides the number -- the
# unit is simply left out of the captured value. Excluding a trailing '.' here (as an earlier draft of this
# round did) is WRONG, not merely over-cautious: it silently un-matches the single most common shape in this
# corpus's prose, a measurement at the end of a sentence ("...the chance was 0.025."), which is how this exact
# regression surfaced -- `tools/claim_check_retro_scan.py`'s round-5-vs-round-6 comparison found round 6 was
# LOOSER on 5 real findings before this was caught, every one of them a sentence-final number. Applied to a
# NORMALIZED copy of the text (see `_normalize_for_numbers`), never to the raw text used for citation parsing.
NUM_RE = re.compile(r"(?<![\w.])(-?\d*\.\d{3,}(?:[eE][+-]?\d+)?)(?!\d)")
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

# ROUND 6 (issue 4): a marked line/cell exempts at most this many numbers "for free". A doc that marks 9+
# values on one line/cell is almost certainly re-inventing a block marker, not marking individual derived
# numbers -- and past the cap, extra numbers are simply CHECKED like any unmarked claim (fails closed).
MAX_EXEMPT_PER_LINE = 8
# ROUND 6 (issue 9): a single glob citation loads at most this many files (sorted, deterministic), and the
# POOL of distinct artifact values across every cited path in a doc is capped at this size. Calibrated
# 2026-09-25 against the corpus's own largest legitimate per-battery citation (8,266 distinct values across
# 481 files, `research/findings/raw/consol_opsweep_gpu/op*_seed42.json`; the next-largest real citations sit at
# 4,459 and 1,384) -- both caps sit comfortably above every real citation measured, and far below the scale of
# citing a whole `raw/` tree (a single-level `raw/**/*.json` already matches 7,000+ files). Forward-only: the
# gate only checks NEWLY ADDED findings, so no existing committed doc is affected by this cap.
MAX_GLOB_FILES = 1000
MAX_ARTIFACT_VALUES = 15000

# ROUND 6 (issue 6): characters normalized away or mapped before NUM_RE ever sees a line. `<!--.*?-->` also
# closes issues 7 and 9 (a number/citation hidden inside an ordinary HTML comment is invisible to both the
# coverage fraction and PATH_RE) -- DOTALL because an HTML comment's own syntax is unambiguous even when it
# spans a physical newline, unlike the interpretive "scope" markup rounds 1-4 tried and failed to bound.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_EMPTY_SPAN_RE = re.compile(r"<span[^>]*>\s*</span>", re.I)
# Markdown backslash-escapes of punctuation that matter to number syntax (`0\.9876`, `\-0.9876`).
_BACKSLASH_ESCAPE_RE = re.compile(r"\\([.\-_*`])")
# The two dash glyphs typed/pasted in place of an ASCII minus sign; both used to be silently DROPPED by the
# old regex (neither is `-`), turning a claimed sign flip into a false positive match against a positive value.
_DASH_CHARS = ("\u2212", "\u2013")         # U+2212 MINUS SIGN, U+2013 EN DASH

# ROUND 6 (issue 5): synthesis now requires a PROPERLY CLOSED frontmatter block, not just "starts with ---".
# `\A` anchors at the very first character; a later `\n---` found anywhere else in the doc (a horizontal rule,
# or a second, unrelated frontmatter-shaped block near the end) no longer counts as the close.
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---[ \t]*\n", re.S)
_SYNTH_REASON_RE = re.compile(r"^claim_check_reason:\s*(\S.*?)\s*$", re.M)
_TITLE_RE = re.compile(r"^#[ \t]+(.*?)\s*$", re.M)

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
# a new legitimate all-derived doc exceeds it. Round 6 (issue 7) changed WHAT is counted (distinct values outside
# comments, not raw occurrences) but not the threshold itself; a round-5-vs-round-6 corpus retro-scan
# (research/coordination/claimcheck_round6_retro_whole_corpus_2026-09-25.tsv, 2971 docs, and
# claimcheck_round6_retro_since2026-09-01_2026-09-25.tsv, the 353 added since 2026-09-01) found 73 / 20
# documents respectively whose verdict flips from round 5 -- EVERY one via a newly-caught UNSUPPORTED number
# (issues 4/6/9's cap, normalization and citation fixes), never via LOW_COVERAGE alone (the whole-corpus TSV's
# reason column contains the string "low_coverage" zero times). These are candidate real errors in EXISTING
# findings, not a side-effect of the coverage-counting change, and (being pre-existing documents) the
# forward-only gate never retroactively blocks any of them. One further flip runs the OTHER way (round 5 FAIL
# -> round 6 PASS, 2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md): round 5's own
# dropped-sign bug (issue 6) compared a stated `-0.006` as if it were `+0.006` and failed to find a match;
# round 6 parses the sign correctly and finds the artifact's own `"perm": -0.006` -- a correction, not a
# regression (verified by reading the cited artifact directly, not just trusting the flip).
MIN_CHECK_FRACTION = 0.05
LOW_COVERAGE_MIN_TOTAL = 80


def _flatten_numbers(obj, out):
    """Every numeric leaf in an artifact, at any depth. Returns the count of RAW leaves visited (not the size of
    `out`, which deduplicates) -- round 6 uses this to cap how many values a citation can pool (issue 9) without
    needing a second pass over the same structure."""
    if isinstance(obj, bool):
        return 0
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 6))
        return 1
    if isinstance(obj, dict):
        return sum(_flatten_numbers(v, out) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_flatten_numbers(v, out) for v in obj)
    return 0


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
    """Round 6 (issue 9): a glob is capped at MAX_GLOB_FILES files (sorted, so the choice is deterministic), and
    loading stops entirely once the DISTINCT value pool (`nums`) reaches MAX_ARTIFACT_VALUES -- a citation that
    would blow past either cap is truncated, not silently allowed to pool unbounded values. `capped` reports
    every truncation so an author can see it; it never blocks on its own (LOW_COVERAGE / unsupported still do
    the actual gating)."""
    nums, verdicts, loaded, missing, capped = set(), [], [], [], []
    for p in paths:
        if len(nums) >= MAX_ARTIFACT_VALUES:
            capped.append("%s: artifact VALUE pool cap (%d distinct) already reached -- not loaded"
                          % (p, MAX_ARTIFACT_VALUES))
            continue
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        is_glob = any(c in full for c in "*?[")
        hits = sorted(glob.glob(full)) if is_glob else ([full] if os.path.exists(full) else [])
        if not hits:
            missing.append(p)
            continue
        if is_glob and len(hits) > MAX_GLOB_FILES:
            capped.append("%s: glob matched %d file(s), only the first %d loaded (glob cap)"
                          % (p, len(hits), MAX_GLOB_FILES))
            hits = hits[:MAX_GLOB_FILES]
        for h in hits:
            if len(nums) >= MAX_ARTIFACT_VALUES:
                capped.append("%s: artifact VALUE pool cap (%d distinct) reached -- %s and any remaining "
                              "file(s) not loaded" % (p, MAX_ARTIFACT_VALUES, os.path.relpath(h, ROOT)))
                break
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
    return nums, verdicts, loaded, missing, capped


def _strip_comments(s):
    """Remove every `<!--...-->` span (round 6, issues 7+9): a number or citation living ONLY inside an
    ordinary HTML comment is invisible prose, not a real claim -- it must not count toward LOW_COVERAGE's
    numerator, and it must not resolve as a citation either. This runs on text that has ALREADY been checked
    for the derived marker (the marker IS itself a comment), so stripping it afterwards is safe: `has_marker`
    never depends on the stripped copy."""
    return _HTML_COMMENT_RE.sub("", s)


def _normalize_for_numbers(segment):
    """Fold every escape in HISTORY item 6 to plain ASCII before NUM_RE runs. Order matters: entities first (so
    a decoded `&#46;` behaves like a literal '.' for the later steps), then invisible/format characters, then
    markdown escapes and empty decoy tags, then the dash glyphs, then emphasis markers. This is a SCAN-ONLY
    copy -- PATH_RE and the marker/table/segment structure all still see the original text."""
    s = html.unescape(segment)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = _BACKSLASH_ESCAPE_RE.sub(r"\1", s)
    s = _EMPTY_SPAN_RE.sub("", s)
    for ch in _DASH_CHARS:
        s = s.replace(ch, "-")
    s = s.replace("_", "").replace("*", "")
    return s


def _is_table_row(ln):
    stripped = ln.lstrip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _segments(ln):
    """Split a physical line into the independent scopes a <!--derived--> marker can reach (round 6, issue 4):
    markdown table CELLS (split on `|`) and HTML `<br>` sub-lines (an HTML line break renders as two lines to a
    reader even though it is one physical line to this line-based scanner). A marker in one segment must not
    exempt a number in a different segment of the SAME physical line -- round 5's whole-line rule let a marker
    in a trailing "delta" column excuse a wrong MEASURED number in an earlier column. Falls back to the whole
    line as a single segment when neither structure is present, which is round 5's original behaviour."""
    if _is_table_row(ln):
        cells = ln.split("|")
        out = []
        for c in cells:
            out.extend(_BR_RE.split(c))
        return out
    return _BR_RE.split(ln)


def _synthesis_status(text):
    """Round 6 (issue 5). Returns (is_synthesis, reason, barred_warning):
      * is_synthesis=False, barred_warning=None  -- no escape declared at all (the common case).
      * is_synthesis=False, barred_warning=<str> -- the flag was declared but does not apply (no closed
        frontmatter / no reason / a verdict-bearing title); falls back to the strict per-line + LOW_COVERAGE
        rules, with the reason surfaced as a non-blocking WARNING.
      * is_synthesis=True,  reason=<str>          -- the escape applies.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m or not SYNTH_RE.search(m.group(1)):
        return False, None, None
    fm = m.group(1)
    reason_m = _SYNTH_REASON_RE.search(fm)
    reason = reason_m.group(1).strip() if reason_m else ""
    if not reason:
        return False, None, (
            "declares `claim_check: synthesis` but no non-empty `claim_check_reason:` in the SAME frontmatter "
            "block -- falling back to the normal per-line rule (the escape needs a STATED reason)")
    title_m = _TITLE_RE.search(text)
    if title_m:
        vm = VERDICT_RE.search(title_m.group(1))
        if vm:
            return False, None, (
                "declares `claim_check: synthesis` but its title states a verdict (%s) -- a verdict-bearing "
                "document is BARRED from the synthesis escape and every number is checked, synthesis or not"
                % vm.group(0))
    return True, reason, None


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


def _empty_scan_result(unreadable):
    return dict(cited=[], nums=set(), loaded=[], missing=[], capped=[], checked=0, checked_distinct=0,
                suppressed={"inline": 0, "synthesis": 0}, total_numeric=0, unsupported=[],
                synthesis=False, low_coverage=False, marked_lines=[], warnings=[], unreadable=unreadable)


def _scan(doc_path, tol=None):
    """Pure computation, no printing -- shared by the CLI (`check`) and `tools/finding_lint.py`, so both see the
    exact same structured verdict instead of finding_lint re-parsing this module's printed stdout.

    tol=None => RELATIVE tolerance. An absolute 5e-4 let a fabricated 0.9999 match a stored 1.0, so the
    checker's own negative control failed on first run: with ~1000 artifact values, near-misses are common and an
    absolute window is far too loose. Relative tolerance scales with the claim. (Unchanged from main.)
    """
    # ROUND 6 (issue 1): STRICT UTF-8. Round 5 read with errors="replace", which silently decodes invalid bytes
    # instead of raising -- main (and every earlier revision) crashed on the same input, which blocks a commit;
    # round 5 uniquely did not. A doc that fails to decode now returns a blocking UNREADABLE result instead of
    # either a crash or a silent, wrong scan.
    try:
        raw = open(doc_path, "rb").read()
    except OSError as e:
        return _empty_scan_result("cannot read %s: %s: %s" % (doc_path, type(e).__name__, e))
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        return _empty_scan_result("%s is not valid UTF-8 (%s at byte offset %d) -- fix the file's encoding "
                                  "before it can be checked" % (doc_path, e.reason, e.start))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # ROUND 6 (issue 5): synthesis now requires a properly closed frontmatter block + a stated reason, and is
    # barred outright from a verdict-bearing title. See `_synthesis_status`.
    synthesis, _synth_reason, synth_barred = _synthesis_status(text)

    # ROUND 6 (issue 9): citations inside HTML comments are invisible -- strip comments before PATH_RE runs.
    cited = sorted(set(PATH_RE.findall(_strip_comments(text))))
    nums, verdicts, loaded, missing, capped = load_artifacts(cited)

    unsupported, checked = [], 0
    checked_values = set()                 # ROUND 6 (issue 7): DISTINCT values, for the coverage fraction
    suppressed = {"inline": 0, "synthesis": 0}
    marked_lines = []                      # 1-indexed lines carrying a literal <!--derived--> that exempted them
    marked_lines_seen = set()

    for i, ln in enumerate(lines, 1):
        for seg in _segments(ln):
            has_marker = DERIVED_MARK in seg               # checked on the RAW segment -- the marker IS a comment
            scan_text = _normalize_for_numbers(_strip_comments(seg))
            seg_matches = list(NUM_RE.finditer(scan_text))
            if not seg_matches:
                continue
            if has_marker:
                if i not in marked_lines_seen:
                    marked_lines_seen.add(i)
                    marked_lines.append(i)
                # ROUND 6 (issue 4): cap the free exemption; anything past it is CHECKED like an unmarked claim.
                exempt, overflow = seg_matches[:MAX_EXEMPT_PER_LINE], seg_matches[MAX_EXEMPT_PER_LINE:]
                suppressed["inline"] += len(exempt)
                to_check = overflow
                if to_check and synthesis:
                    suppressed["synthesis"] += len(to_check)
                    to_check = []
            elif synthesis:
                suppressed["synthesis"] += len(seg_matches)
                to_check = []
            else:
                to_check = seg_matches
            for m in to_check:
                val = float(m.group(1))
                checked += 1
                checked_values.add(round(val, 6))
                eps = tol if tol is not None else max(5e-6, 1e-4 * abs(val))
                if not any(abs(val - a) <= eps for a in nums):
                    unsupported.append((i, val, ln.strip()[:88]))

    if synthesis and not cited:
        unsupported.append((0, 0.0, "synthesis doc cites NO artifact — the escape still requires citations"))

    total_numeric = checked + suppressed["inline"] + suppressed["synthesis"]
    # ROUND 6 (issue 7): the fraction counts DISTINCT checked values, not raw occurrences -- N copies of one
    # real artifact value pasted in as padding used to count as N; now they count as at most 1.
    low_coverage = (not synthesis and total_numeric >= LOW_COVERAGE_MIN_TOTAL
                    and (len(checked_values) / total_numeric) < MIN_CHECK_FRACTION)

    warnings = _line_warnings(lines)
    if synth_barred:
        warnings.append((1, "synthesis escape not applied", synth_barred))
    for c in capped:
        warnings.append((0, "citation capped", c))

    return dict(cited=cited, nums=nums, loaded=loaded, missing=missing, capped=capped, checked=checked,
                checked_distinct=len(checked_values), suppressed=suppressed, total_numeric=total_numeric,
                unsupported=unsupported, synthesis=synthesis, low_coverage=low_coverage,
                marked_lines=marked_lines, warnings=warnings, unreadable=None)


def _verdict(r):
    """The single FAIL/PASS rule, shared by `check()`, `selftest()` and the test suite, so none of them can
    drift from what the others mean by "this document blocks the commit"."""
    return "FAIL" if (r.get("unreadable") or r["missing"] or r["unsupported"] or r["low_coverage"]) else "PASS"


def check(doc_path, tol=None, verbose=True):
    r = _scan(doc_path, tol)
    if r.get("unreadable"):
        if verbose:
            shown = os.path.relpath(doc_path, ROOT) if os.path.isabs(doc_path) else doc_path
            print("claim_check: %s" % shown)
            print("  ⛔ UNREADABLE: %s" % r["unreadable"])
            print("  => ⛔ UNREADABLE — fix the file's encoding before it can be checked")
        return 1

    checked, suppressed, total_numeric = r["checked"], r["suppressed"], r["total_numeric"]
    unsupported, missing, loaded, cited = r["unsupported"], r["missing"], r["loaded"], r["cited"]
    synthesis, low_coverage = r["synthesis"], r["low_coverage"]

    if verbose:
        print("claim_check: %s" % os.path.relpath(doc_path, ROOT))
        print("  cited artifacts : %d found, %d missing" % (len(loaded), len(missing)))
        for mp in missing[:5]:
            print("      ⛔ MISSING  %s" % mp)
        print("  measurements    : %d checked (%d distinct) against %d artifact values%s"
              % (checked, r["checked_distinct"], len(r["nums"]),
                 "   [synthesis: per-line rule suppressed, citations still required]" if synthesis else ""))
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
            print("      ⛔ LOW COVERAGE: only %d/%d (%.0f%%) DISTINCT numeric value(s) were actually checked "
                  "-- the rest were marked <!--derived--> on their own line/cell (or hidden in a comment). A "
                  "doc this size should not be almost entirely derived; mark the specific derived numbers, not "
                  "the whole document."
                  % (r["checked_distinct"], total_numeric,
                     100.0 * r["checked_distinct"] / total_numeric if total_numeric else 0.0))

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
# accuracy=0.17 and baseline=0.1625; 0.1525 / 0.140 / 1.23456 / -0.1525 are WRONG numbers (not in the artifact);
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
    dict(name="table_with_marker_in_same_cell_as_value_passes", expect="PASS", wrong_on=(),
         why="round 6: a table whose derived rows carry the marker IN THE SAME CELL as the value passes -- the "
             "cell-scoping fix (issue 4) narrows exemption to the marker's own cell, so the marker must now "
             "share a cell with the number it exempts, not merely share the row",
         doc=_HDR + "| metric | value |\n|---|---|\n| ratio | 0.104615 <!--derived--> |\n"
                    "| gap | 0.207531 <!--derived--> |\n| accuracy | 0.170000 |\n"),
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
    # --- round 6, issue 4: bounded per-line/per-cell exemption --------------------------------------------------
    dict(name="cap_exempts_only_first_8_numbers_on_a_marked_line", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 4): a marked line's free exemption is capped at MAX_EXEMPT_PER_LINE=8 -- a 9th "
             "(wrong) number on the same marked line must still be checked, not swept in for free the way a "
             "block-scope marker used to sweep in an entire section",
         doc=_HDR + "The values are 0.100001, 0.100002, 0.100003, 0.100004, 0.100005, 0.100006, 0.100007, "
                    "0.100008, and 0.1525 here. <!--derived-->\n"),
    dict(name="table_row_marker_exempts_only_its_own_cell", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 4), the exact incident repro: a <!--derived--> marker alone in a trailing 'delta' "
             "cell used to exempt the WHOLE row -- a wrong MEASURED number in an earlier, unmarked cell of the "
             "SAME physical line must now still be checked",
         doc=_HDR + "| 42 | 0.1525 | 0.104615 | <!--derived--> |\n"),
    dict(name="br_split_line_marker_does_not_reach_the_other_side", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 4), repro 2: an HTML <br> renders as two lines to a reader even though it is one "
             "PHYSICAL line to this scanner -- a marker before the <br> must not reach a wrong number after it",
         doc=_HDR + "ratio 0.104615 <!--derived--><br>accuracy 0.1525\n"),
    # --- round 6, issue 6: normalize before matching ------------------------------------------------------------
    dict(name="underscore_emphasis_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): NUM_RE's old word-boundary lookaround treated `_0.1525_` as glued to its "
             "emphasis markers and never matched it at all -- a wrong number wrapped in markdown emphasis was "
             "completely invisible, not merely exempt",
         doc=_HDR + "The accuracy was _0.1525_ here.\n"),
    dict(name="glued_unit_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): a unit glued directly onto the number (`0.1525ms`) used to fail the trailing "
             "word-boundary check and vanish entirely",
         doc=_HDR + "The latency was 0.1525ms here.\n"),
    dict(name="leading_dot_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): NUM_RE required at least one leading digit before the dot, so `.1525` (no "
             "leading zero) was invisible",
         doc=_HDR + "The drop was .1525 here.\n"),
    dict(name="markdown_escaped_dot_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): a markdown backslash-escaped decimal point (`0\\.1525`) broke the digit run "
             "and the number was never matched",
         doc=_HDR + "The accuracy was 0\\.1525 here.\n"),
    dict(name="html_entity_dot_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): an HTML numeric entity in place of the decimal point (`0&#46;1525`) broke the "
             "digit run the same way",
         doc=_HDR + "The accuracy was 0&#46;1525 here.\n"),
    dict(name="empty_comment_mid_number_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): an empty HTML comment spliced into the middle of the digits (`0.15<!---->25`) "
             "split the number into two unmatched fragments",
         doc=_HDR + "The accuracy was 0.15<!---->25 here.\n"),
    dict(name="empty_span_mid_number_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): an empty <span></span> spliced into the digits does the same split",
         doc=_HDR + "The accuracy was 0.15<span></span>25 here.\n"),
    dict(name="zero_width_space_mid_number_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): a zero-width space (U+200B, Unicode category Cf) inside the digits is "
             "invisible to a reader but broke the digit run for NUM_RE",
         doc=_HDR + "The accuracy was 0.15\u200b25 here.\n"),
    dict(name="soft_hyphen_mid_number_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): a genuine Unicode soft hyphen (U+00AD, category Cf) inside the digits is the "
             "same class of invisible break (distinct from issue 1's INVALID-UTF-8-byte case, which is about "
             "decode failure, not a valid codepoint that happens to be invisible)",
         doc=_HDR + "The accuracy was 0.15\u00ad25 here.\n"),
    dict(name="scientific_notation_no_longer_hides_a_wrong_number", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): scientific notation (`1.525e-1` = 0.1525) was never matched at all -- the "
             "exponent suffix broke the old trailing word-boundary check",
         doc=_HDR + "The accuracy was 1.525e-1 here.\n"),
    dict(name="typographic_minus_sign_flip_no_longer_passes", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): U+2212 MINUS SIGN is not `-`, so the old regex silently dropped the sign and "
             "matched `0.1625` (POSITIVE, == the cited baseline) out of a claimed `\u22120.1625` (NEGATIVE) -- a "
             "real sign flip against the cited artifact used to read as a clean, wrongly-supported match",
         doc=_HDR + "The delta was \u22120.1625 here.\n"),
    dict(name="en_dash_sign_flip_no_longer_passes", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 6): U+2013 EN DASH is the same class of dropped-sign bug as the typographic minus",
         doc=_HDR + "The delta was \u20130.1625 here.\n"),
    # --- round 6, issue 7: LOW_COVERAGE counts distinct values outside comments ----------------------------------
    dict(name="comment_hidden_decoys_no_longer_pad_coverage", expect="FAIL", wrong_on=("main", "r1", "r2", "r3",
                                                                                        "r4", "r5"),
         why="round 6 (issue 7), the exact incident repro (scaled down): many marked-derived lines plus a "
             "handful of the SAME real artifact value pasted inside HTML comments used to count each hidden "
             "COPY toward the checked fraction, clearing the LOW_COVERAGE floor on padding alone; now a "
             "comment-hidden number counts zero times and repeats of one value count once",
         doc=(_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1)
                                 for i in range(LOW_COVERAGE_MIN_TOTAL))
              + "\n\n" + "\n".join("<!-- padding citation of 0.170000 -->" for _ in range(6)) + "\n"),
         ),
    # issue 9's citation-inside-comment fix needs a CONTROLLED artifact (a real value the hidden citation would
    # spuriously validate) rather than the shared %(art)s fixture every other case uses, so it is a dedicated
    # pytest test (test_citation_inside_html_comment_is_ignored in tests/test_claim_check_line_only.py) instead
    # of a SELFTEST_CASES entry here.
    # --- round 6, issue 5: synthesis needs closed frontmatter + a reason + is barred on a verdict title ----------
    dict(name="synthesis_without_closed_frontmatter_is_not_exempt", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 5): `claim_check: synthesis` appearing after an UNCLOSED frontmatter block (no "
             "later bare `---` line) used to exempt the whole doc via the old `text.split(\"\\n---\", 1)` "
             "first-occurrence split; it no longer does, so the wrong number below is checked normally",
         doc="---\ntitle: not really frontmatter, never closed\n\n# A doc\n\nArtifact: `%(art)s`\n\n"
             "claim_check: synthesis\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_without_reason_is_not_exempt", expect="FAIL",
         wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 5): a properly closed frontmatter declaring the flag but NO `claim_check_reason:` "
             "used to exempt everything on the flag alone; it now falls back to the strict rule",
         doc="---\nclaim_check: synthesis\n---\n\n# A doc\n\nArtifact: `%(art)s`\n\n"
             "The accuracy was 0.1525 here.\n"),
    dict(name="synthesis_barred_by_verdict_title", expect="FAIL", wrong_on=("main", "r1", "r2", "r3", "r4", "r5"),
         why="round 6 (issue 5): a verdict-bearing title (GO) is BARRED from the synthesis escape outright, "
             "even with a closed frontmatter and a stated reason -- the exact incident shape this whole file "
             "exists to close was a live verdict hiding wrong numbers behind a blanket exemption",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes several prior runs\n---\n\n"
             "# Lane A 6-seed GO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_with_closed_frontmatter_and_reason_passes", expect="PASS", wrong_on=(),
         why="round 6 (issue 5): the escape still works for a genuine literature/synthesis doc: closed "
             "frontmatter, a stated reason, a non-verdict title",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes several prior runs, no new measurements\n"
             "---\n\n# A literature summary\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
]

_HISTORY_SHAS = {"main": "7e2edc08e", "r1": "d4959ecb0", "r2": "6abb28469", "r3": "662e167e8", "r4": "214e509bf",
                  "r5": "4fda849d4"}


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
WRONG_VALUES = {0.1525, 0.14, 1.23456, -0.1525, -0.1625}


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
            got = _verdict(r)
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

        # ROUND 6 (issue 1): the UNREADABLE case needs raw, deliberately-invalid bytes -- it cannot go through
        # `_write_case`, which writes valid UTF-8 text. 0xAD is a bare Latin-1 SOFT HYPHEN byte, invalid on its
        # own as UTF-8 (it is not a valid single-byte sequence and does not start a valid multi-byte one).
        bad_path = os.path.join(d, "invalid_utf8.md")
        art_rel = os.path.relpath(os.path.join(d, "art.json"), ROOT).replace(os.sep, "/")
        open(bad_path, "wb").write(
            ("# Some finding\n\nArtifact: `%s`\n\nThe accuracy was 0.98" % art_rel).encode("utf-8")
            + b"\xad" + "76 here.\n".encode("utf-8"))
        r = _scan(bad_path)
        if not r.get("unreadable"):
            problems.append("SELFTEST BROKEN: invalid UTF-8 (a bare Latin-1 soft-hyphen byte) did not report "
                            "UNREADABLE -- round 5's errors=\"replace\" regression (issue 1) is back")
        elif _verdict(r) != "FAIL":
            problems.append("SELFTEST BROKEN: an UNREADABLE result must still be a blocking FAIL")
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
