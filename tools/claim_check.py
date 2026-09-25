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

ROUND 2 (found 2026-09-25, review of the above fix -- SOUND-WITH-ISSUES): the paragraph-scope fix above still
had three exploitable holes, all in the SAME direction (a marker's scope outliving the unit it was meant to
cover). Fixed here, each with its own selftest case:
  3. HEADING LEVEL: the close-on-heading check only matched a literal `## ` (level-2) line. A marker followed,
     with no blank line, by a `### ` (or `#`, `#### ` ... any other level) subheading and then a wrong number
     was never closed by that subheading, so the wrong number stayed suppressed. FIX: `HEADING_RE` now matches
     ANY heading level (`#{1,6}` followed by whitespace or end-of-line); a heading whose text starts with
     "derived" (any level, not only `## `) opens the section mechanism, and every OTHER heading, at any level,
     closes both mechanisms.
  4. UNIT BOUNDARY WITHOUT A BLANK LINE: scope only closed at a blank line, a heading, or an explicit close --
     so a marker, a short derived table, and then a wrong number with NO blank line before it (the table simply
     ends and prose resumes) stayed inside the block, because nothing but a blank line ever ended it. FIX: the
     block now tracks what KIND of unit it opened on its first content line -- a TABLE (a run of `|`-prefixed
     lines) or a PARAGRAPH (everything else) -- and closes at that unit's OWN boundary even with no blank line:
     a table ends at its first line that does not start with `|`; a paragraph ends at a blank line, OR at a
     line that opens a new markdown block on its own (a list item `-`/`*`/`+`/`N.`/`N)`, a blockquote `>`, a
     horizontal rule, a fenced code block delimiter, or a table row `|` -- any of these is unambiguously a NEW
     sentence-block, not a continuation of the paragraph the marker opened. The line that closes the unit is
     itself OUTSIDE it and is checked normally, not suppressed.
  5. CLOSE MARKER SHARING A LINE: `<!--/derived-->` only closed scope when it sat ALONE on its line -- a close
     marker with trailing content on the SAME line (e.g. a derived paragraph's last sentence immediately
     followed by `<!--/derived--> The real number was 0.999.`) never matched the alone-on-a-line check, so the
     block stayed open and the trailing number stayed suppressed. FIX: the close marker now works ANYWHERE on a
     line. Text on the line BEFORE the marker is still accounted under whatever scope was active up to that
     point (it was inside the block); text AFTER the marker on the SAME line is normal content and IS checked
     against the cited artifacts, exactly as if it started a new line right after the close.
`selftest()` carries one adversarial case per hole (3/4/5): each reproduces the exploit shape with a WRONG
number that must be caught (FAIL), verified to actually fail against the pre-round-2 logic before this fix
landed.

ROUND 3 (2026-09-25, re-review of round 2 -- SOUND-WITH-ISSUES). Two more holes, both again in the
"a marker's scope outlives or falls short of the unit it was meant to cover" family:
  6/7. NOT FENCE-AWARE (round 2 REGRESSED this -- main and round 1 did not have the bug because their heading
     check was a literal `## `/`## derived` string, which a bare `#` code comment never matches; round 2's
     `#{1,6}` generalization for hole #3 made ANY `#`-prefixed line a candidate heading, including one sitting
     inside a fenced code block). Two shapes, same root cause:
       6. A `#`-prefixed comment INSIDE a fence (`# derived ...` in a Python/bash snippet -- 230 existing docs
          mix a derived marker with a fenced block) was treated as a real heading; when its text happened to
          start with "derived" it silently opened the phantom SECTION mechanism, which then swallowed every
          later numeric claim until the next GENUINE heading or EOF. Verified: a doc with a value (0.1525)
          contradicting its cited artifact (0.17) passed with exit 0 via this route.
       7. A fence occurring INSIDE an open paragraph/list/table block ended that block unconditionally at the
          fence-OPEN delimiter line, because `_ends_open_paragraph` treated the fence marker like any other
          single-line block-starter (a list item, a table row) even though a fence is a PAIRED, multi-line
          delimiter -- so a legitimately-derived number placed right after an illustrative code snippet, still
          inside the SAME derived paragraph, fell outside protection.
     FIX: an `in_fence` boolean, toggled by `_FENCE_RE`, checked FIRST in the scan loop, before any heading or
     block-scope logic runs. While `in_fence` is true (or on the toggle line itself): never a heading, never a
     derived-heading, never opens/closes `in_block`/`block_kind`, never matches the close marker. Numbers on
     those lines are still run through `_account` with whatever scope was ALREADY active going into the fence
     -- unchanged from every revision before this one, none of which had ANY fence-awareness at all, so a
     number inside a fence was always checked as ordinary prose unless an already-open scope covered it. Fences
     add no new suppression; they are simply transparent to the state machine around them.
  8. LISTS HAD NO PERSISTENT KIND (unlike tables). A standalone marker's block tracked its unit's kind as only
     "table" (a run of `|` rows) or generic "para" -- a list got lumped into "para", whose own continuation
     rule (`_ends_open_paragraph`, pre-round-3) treated ANY list-item line as a block-ender. Two shapes:
       (a) the marker's first content line is ITSELF a list item -- the block opens correctly, but the SECOND
           bullet of the SAME list re-triggers the "list item starts a new block" rule and closes it, so items
           2+ are unprotected;
       (b) the far more common real pattern -- marker, one or two intro/citation sentences (no blank line),
           then the actual derived list -- the intro sentence claims the block's one "para" unit, and the
           list's own FIRST bullet (where the real numbers live) is already past the paragraph's close.
     Corpus-verified against research/findings/*.md (2026-09-25): comparing round 1 vs round 2 found 43
     already-committed, previously-PASSING docs newly FAIL, zero the other way; 3 manually inspected
     (2026-08-01-W5-affective-theory-of-mind-6seed-GO.md:47-52,
     2026-08-06-replay-consolidation-selective-CA1-to-cortex-reinstatement-research-gate.md:19-26,
     2026-09-21-load-bearing-fraction-6seed-adequate-0.85-robust-core-20.md:17-19) are exactly shape (b):
     correctly-marked, legitimately-derived numbers (per-seed re-quotes), not genuine errors.
     FIX: a list gets a THIRD persistent `block_kind`, exactly like a table's. `_unit_starter()` classifies a
     line as starting a "table", "list" or "other" (blockquote/hrule -- still an unambiguous, distinct
     construct) markdown unit, or `None` (an ordinary continuation line, i.e. a CommonMark "lazy continuation"
     of whatever came before). An open "para" block ABSORBS the first table/list unit it meets (transitioning
     `block_kind` to it, persistent from then on -- fixing shape (b)); an open "list"/"table" block continues
     through more of its own unit AND through ordinary continuation lines (fixing shape (a): a list's second,
     third, ... bullet no longer ends it); only a genuinely distinct construct ("other": blockquote/hrule) or
     the existing block-enders (blank line, any heading, an explicit close) still end it.
`selftest()` adds one FAILING-direction and one PASSING-direction case for the fence fix (6/7 share one root
cause and one pair of cases), plus one FAILING-direction and two PASSING-direction cases for the list fix (one
PASSING case per named break shape (a)/(b)) -- each PASSING case verified to actually FAIL on round 2
(commit 6abb28469) before this fix landed, i.e. round 2 wrongly rejected a legitimately-derived document.
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

# A markdown heading at ANY level (round 2, hole #3: only a literal `## ` used to close scope, so a `### `
# subheading -- or any level other than exactly 2 -- right after a marker never closed it).
HEADING_RE = re.compile(r"^#{1,6}(?:\s|$)")
# A heading whose own text starts with "derived", at any level -- opens the SECTION mechanism (until the next
# heading of any kind), same role the old literal `## derived` check played, generalized to any level.
DERIVED_HEADING_RE = re.compile(r"^#{1,6}\s+derived\b", re.I)
# Markdown constructs that start a new block-level unit on their own -- round 2, hole #4: a standalone marker's
# PARAGRAPH ends at any of these even with no blank line before them, because each one is unambiguously a new
# "sentence-block", not a continuation of the paragraph/table the marker opened.
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)])(?:\s|$)")
_BLOCKQUOTE_RE = re.compile(r"^>")
_HRULE_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})\s*$")
_FENCE_RE = re.compile(r"^(?:`{3,}|~{3,})")


def _unit_starter(stripped):
    """Classifies `stripped` as starting a markdown block-level construct of its own: "table" (a `|` row),
    "list" (a list item), "other" (a blockquote or horizontal rule -- unambiguously a NEW, distinct construct,
    never "the paragraph's own list/table") or `None` (ordinary text -- ends nothing; it is either the block's
    own continuing content or, in CommonMark terms, a "lazy continuation" of whatever came before it).

    Round 3, issue #8: table and list are no longer unconditional block-enders for an open "para" unit -- the
    caller (`_scan`) ABSORBS the first one it meets, transitioning `block_kind` into it and keeping it
    persistent from then on (exactly how a table's kind already worked before this fix), because "marker, an
    intro sentence, then its list/table" is the single most common real derived-block shape in this project's
    findings corpus, and the intro sentence must not consume the whole unit's protection and strand the
    list/table it introduces. "other" still ends an open paragraph, unchanged from round 2.

    Fenced code blocks are handled separately and are NOT one of this function's cases -- round 3, issue #7:
    unlike a list item or a table row, a fence is a PAIRED, multi-line delimiter, so `_scan`'s `in_fence` state
    intercepts it before this function is ever consulted (see `_scan`'s fence-awareness comment)."""
    if stripped.startswith("|"):
        return "table"
    if _LIST_ITEM_RE.match(stripped):
        return "list"
    if _BLOCKQUOTE_RE.match(stripped) or _HRULE_RE.match(stripped):
        return "other"
    return None

# A non-synthesis doc that examined this fraction or less of its own numeric claims is treated the same as
# examining ZERO of them -- the 2026-09-25 incident's ratio was 0/~50 (0%), but the failure mode is "suppressed
# almost everything", not literally "suppressed everything".
MIN_CHECK_FRACTION = 0.05
# ... AND ONLY when the doc has at least this many numeric claims in total. CALIBRATED, not guessed -- and
# calibrated on a SPECIFIC, DATED population: a checked==0-of-everything retro-scan over the 353
# `research/findings/*.md` added since 2026-09-01 (2026-09-25) found 71 documents that are ENTIRELY,
# LEGITIMATELY derived (a short diagnosis note built purely from already-published ratios/deltas/percentages,
# correctly all-marked) -- up to 39 numeric claims, 0 checked, by design, not by error. A blanket "checked==0
# fails" rule would have blocked every one of them. The real incident examined 0 of ~50 numeric claims in a
# SUBSTANTIAL document (336 candidate artifact values, ~43-53 claims in the doc itself). This floor (40) sits
# just above the largest legitimate all-derived doc in that corpus (39) and well below the incident's scale, so
# it passes every real 2026-09-01+ document while still catching the observed shape (a large document reporting
# essentially nothing checked). Retro-scan: see the 2026-09-25 branch notes. RECALIBRATE if a future corpus scan
# (a different date cutoff, or the corpus grown much larger) finds a legitimate all-derived doc above 39 claims
# -- this constant is an empirical ceiling over the 2026-09-01..2026-09-25 population, not a law.
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
    in_section = False     # a "derived" HEADING section (any level) -- deliberately scoped until the next heading
    in_block = False       # a standalone marker's PARAGRAPH/LIST/TABLE scope -- closes at ITS OWN unit boundary
    block_kind = None      # None (not yet started) | "table" | "list" | "para" -- set from the block's OWN
    # first content line, and PERSISTENT from then on for table/list (round 3, issue #8: a list used to have no
    # kind of its own and fell into generic "para", whose continuation rule treated its own SECOND bullet as a
    # new block and closed early). A table closes at its first non-`|` line; a list closes at the first line
    # that is a genuinely distinct construct (`_unit_starter(...) == "other"` -- a blockquote or horizontal
    # rule), NOT at its own next bullet or at an ordinary wrapped-text continuation line; a paragraph closes at
    # a blank line OR at a line that is a genuinely distinct construct OF ITS OWN -- except a table/list, which
    # it now ABSORBS instead (transitioning `block_kind` into it, round 3 issue #8 shape (b): "marker, an intro
    # sentence, then its list/table" is the common real shape, and the intro sentence must not steal protection
    # from the list/table it introduces). Blank lines BETWEEN the marker and its paragraph (common authoring
    # style: the marker sits alone, then a blank line for visual separation, THEN the derived paragraph/table)
    # do not themselves end the block -- only a blank line AFTER real content has started does (2026-09-25
    # retro-scan: treating the marker's own following blank line as an immediate close broke this exact, common,
    # legitimate pattern in real findings).
    in_fence = False       # round 3, issues #6/#7: true while inside a ``` or ~~~ fenced code block. Checked
    # FIRST, before any heading/marker/block logic below -- a fence is a PAIRED, multi-line delimiter, so it
    # must be transparent to every piece of state above it: never a heading (a `# derived`-style code comment
    # must never open the phantom section mechanism), never opens/closes `in_block`/`block_kind` (a snippet
    # embedded in an open derived paragraph must not end its protection early), never matches the close marker.
    # Numbers on fenced lines are still run through `_account` below under whatever scope was ALREADY active --
    # unchanged from every revision before this one (none had ANY fence-awareness), so a number inside a fence
    # was always checked as ordinary prose unless an already-open scope covered it; this fix adds no NEW
    # suppression, it only stops fences from being mistaken for headings or block-enders.

    def _reason_for(seg):
        """The suppression reason for `seg` under the CURRENT (already-updated) state -- shared between the
        main per-line path and the mid-line close-marker split below, so both cannot disagree."""
        if in_section:
            return "section"
        if in_block:
            return "block"
        if DERIVED_MARK in seg:
            return "inline"
        if synthesis:
            return "synthesis"
        return None

    def _account(seg, lineno, reason):
        nonlocal checked
        if reason is not None:
            suppressed[reason] += len(NUM_RE.findall(seg))
            return
        for m in NUM_RE.finditer(seg):
            val = float(m.group(1))
            checked += 1
            eps = tol if tol is not None else max(5e-6, 1e-4 * abs(val))
            if not any(abs(val - a) <= eps for a in nums):
                unsupported.append((lineno, val, seg.strip()[:88]))

    for i, ln in enumerate(lines, 1):
        stripped = ln.strip()

        # FENCE-AWARENESS (round 3, issues #6/#7) -- checked before ANYTHING else, so nothing below ever sees a
        # line that is inside, or is itself the delimiter of, a fenced code block as a candidate heading, close
        # marker or block-starter. The delimiter line itself toggles `in_fence` and is then accounted like any
        # other line (it essentially never carries a number, but there is no reason to special-case it). While
        # already inside a fence, every line is accounted under the CURRENT `_reason_for` and nothing else runs.
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
            _account(ln, i, _reason_for(ln))
            continue
        if in_fence:
            _account(ln, i, _reason_for(ln))
            continue

        # Heading transitions, at ANY level (round 2, hole #3: a literal `## ` used to be the only heading that
        # closed scope, so a `### ` subheading -- or any level but 2 -- right after a marker never closed it).
        # A heading whose text starts with "derived" opens the section mechanism; every OTHER heading, at any
        # level, ends both mechanisms -- ending `in_block` here too is a safety net beyond the unit-boundary
        # check below, for a marker block never followed by a blank line before the next heading.
        if DERIVED_HEADING_RE.match(stripped):
            in_section = True
            in_block = False
            block_kind = None
            continue
        if HEADING_RE.match(stripped):
            in_section = False
            in_block = False
            block_kind = None
            # no `continue`: the heading line's own text still falls through below, same as before round 2.

        # An explicit close ends a marker block -- round 2, hole #5: it used to require sitting ALONE on its
        # line, so `<!--/derived--> trailing prose` left the block open and suppressed the trailing content. It
        # now matches ANYWHERE on a line: text before the marker is accounted under the scope active up to this
        # point (it was still inside the block); text after it is normal content, checked like any other line.
        if DERIVED_CLOSE in ln:
            before, _, after = ln.partition(DERIVED_CLOSE)
            _account(before, i, _reason_for(before))
            in_block = False
            block_kind = None
            if after.strip() == "":
                continue
            ln, stripped = after, after.strip()
            # falls through: `after` is processed exactly like a normal line below.

        # A marker ALONE on a line opens the block, with UNDETERMINED kind until its first content line.
        if stripped == DERIVED_MARK:
            in_block = True
            block_kind = None
            continue

        # A blank line BEFORE any content is just spacing between the marker and its paragraph (tolerated). A
        # blank line AFTER content has started ends the block (whichever kind it turned out to be).
        if in_block and stripped == "":
            if block_kind is not None:
                in_block = False
                block_kind = None
            continue

        if in_block:
            if block_kind is None:
                # This is the block's OWN first content line -- it defines the unit, and is unconditionally
                # part of it regardless of its shape. Round 3, issue #8: a list item gets its own persistent
                # "list" kind here too, exactly like a table's "table" -- previously only a table row was
                # recognized this way, so a list's own first bullet fell into generic "para" and lost
                # protection the moment its SECOND bullet arrived (break shape (a) in the module docstring).
                starter = _unit_starter(stripped)
                block_kind = starter if starter in ("table", "list") else "para"
            elif block_kind == "table":
                if not stripped.startswith("|"):
                    # THE FIX (round 2, hole #4): a table ends at ITS OWN boundary (its first non-`|` line),
                    # not only at a blank line. Unchanged by round 3.
                    in_block = False
                    block_kind = None
            elif block_kind == "list":
                starter = _unit_starter(stripped)
                if starter == "other":
                    # A blockquote/horizontal rule is unambiguously a NEW, distinct construct -- ends the list.
                    in_block = False
                    block_kind = None
                elif starter == "table":
                    block_kind = "table"      # a table row mixed into a list (rare) -- absorb it, don't end
                # starter == "list" (another bullet -- round 3 issue #8, break shape (a): this used to close
                # the block here) or `None` (an ordinary wrapped-text continuation line, no blank line before
                # it -- CommonMark's own "lazy continuation" of the bullet above) -- both stay inside the list.
            elif block_kind == "para":
                starter = _unit_starter(stripped)
                if starter in ("table", "list"):
                    # THE FIX (round 3, issue #8, break shape (b)): an intro sentence/citation, then (no blank
                    # line) the list/table it introduces -- ABSORB by transitioning `block_kind`, rather than
                    # ending the paragraph here and leaving the list/table's own first row/bullet -- where the
                    # real numbers usually live -- outside protection.
                    block_kind = starter
                elif starter == "other":
                    # A blockquote or horizontal rule remains a genuinely distinct construct (round 2, hole #4,
                    # unchanged): it still ends an open paragraph, because it is never "the paragraph's own
                    # list/table" the way a list item or table row now is.
                    in_block = False
                    block_kind = None
                # starter is None: an ordinary continuation line -- the paragraph stays open, as always.

        _account(ln, i, _reason_for(ln))

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
    trustworthy. Demonstrates BOTH directions of the 2026-09-25 block-scope hole, ROUND 1:

      (a) FAILING DIRECTION -- a standalone `<!--derived-->` marker placed right after `## ` headings (the
          exact incident shape), with WRONG numbers in the sections that follow, must still FAIL. Under the OLD
          block-until-next-heading scope this PASSED with "0 checked"; if this ever regresses, this selftest
          catches it before a real doc does.
      (b) PASSING DIRECTION -- a legitimately-derived small table, correctly paragraph-scoped (marker, table,
          blank line), sitting next to a normally-checked headline number, must still PASS. A fix that closes
          (a) by over-restricting scope (e.g. treating every marker as inline-only) would break this.

    ROUND 2 (same date, review of the round-1 fix): one FAILING-DIRECTION case per hole (#3/#4/#5 in the module
    docstring), each verified against the round-1 code before this fix landed (it PASSED there, wrongly) and
    against this fix (it must FAIL):
      (c) a marker followed, with NO blank line, by a `### ` subheading (not `## `) and then a wrong number.
      (d) a marker, a short derived table, and a wrong number immediately after it with NO blank line.
      (e) a derived paragraph whose `<!--/derived-->` close shares a line with trailing wrong content.

    ROUND 3 (2026-09-25, re-review of round 2): one FAILING-direction and one PASSING-direction case for the
    fence fix (issues #6/#7 share one root cause), plus one FAILING-direction and two PASSING-direction cases
    for the list fix (issue #8 -- one PASSING case per named break shape). Every PASSING case here is verified
    to actually FAIL on round 2 (commit 6abb28469) before this fix landed -- i.e. round 2 wrongly REJECTED a
    legitimately-derived document, the corpus-verified false-positive class the round-3 review named:
      (f) a `# derived` PYTHON COMMENT sitting inside a fenced code block, with a WRONG number both INSIDE the
          fence and in ordinary prose AFTER it -- must FAIL (the fence must not be mistaken for a heading that
          opens a phantom section, and a number inside a fence must still be checked as ordinary prose, exactly
          as every revision before this one already did with no fence-awareness at all).
      (g) a derived paragraph (marker, intro sentence) containing an illustrative FENCED CODE SNIPPET, followed
          by the paragraph's own legitimately-derived number -- must PASS (the fence must not end the
          paragraph's protection at its open delimiter).
      (h) a WRONG number in ordinary prose well AFTER a derived list has ended (separated by a blank line) --
          must still FAIL: the list's new persistent `block_kind` must not over-broaden past the list's own end.
      (i) break shape (a): a marker's OWN FIRST content line is itself a list item, with legitimately-derived
          numbers on the list's SECOND and THIRD bullets -- must PASS.
      (j) break shape (b): a marker's intro sentence (no blank line) immediately followed by the list it
          introduces, with legitimately-derived numbers on the list's OWN FIRST bullet -- must PASS.
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

        # ROUND 2, hole #3 (heading level): a marker followed, with NO blank line, by a `### ` subheading (not
        # `## `) -- the round-1 fix only closed scope on a literal `## `, so this subheading never closed it and
        # the wrong number below stayed suppressed. Verified against the round-1 code: it PASSED (0 checked).
        bad_heading_level = os.path.join(d, "bad_heading_level.md")
        open(bad_heading_level, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "<!--derived-->\n### A subheading, not a `## ` heading\n"
            "The accuracy was 0.1525 here.\n" % art)
        if check(bad_heading_level, verbose=False) == 0:
            problems.append("SELFTEST BROKEN: a standalone <!--derived--> marker followed (no blank line) by a "
                            "`### ` subheading, with a WRONG number (0.1525) right after it, PASSED -- hole #3 "
                            "(only a literal `## ` closed scope) is back")

        # ROUND 2, hole #4 (unit boundary without a blank line): a marker, a short derived table, and a wrong
        # number immediately after the table's last row with NO blank line -- the round-1 fix only closed a
        # block at a blank line/heading/close-marker, so a table's own end (its first non-`|` line) never closed
        # it and the wrong number stayed suppressed as part of the "block". Verified against the round-1 code:
        # it PASSED (0.1525 suppressed as part of the table's block).
        bad_no_blank = os.path.join(d, "bad_no_blank_after_table.md")
        open(bad_no_blank, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "<!--derived-->\n| metric | value |\n|---|---|\n| ratio | 0.104615 |\n"
            "The accuracy was 0.1525 here.\n" % art)
        if check(bad_no_blank, verbose=False) == 0:
            problems.append("SELFTEST BROKEN: a derived table immediately followed (no blank line) by a WRONG "
                            "number (0.1525) PASSED -- hole #4 (a table's own end never closed the block) is "
                            "back")

        # ROUND 2, hole #5 (close marker sharing a line): a derived paragraph's `<!--/derived-->` close sits on
        # the SAME line as trailing wrong content -- the round-1 fix only closed on a close marker ALONE on its
        # line, so the trailing content stayed suppressed as part of the still-open "block". Verified against
        # the round-1 code: it PASSED (0.1525 suppressed on the same line as the close marker).
        bad_close_shares_line = os.path.join(d, "bad_close_shares_line.md")
        open(bad_close_shares_line, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "<!--derived-->\nThe ratio is 0.104615 here. <!--/derived--> "
            "The real accuracy is 0.1525 here.\n" % art)
        if check(bad_close_shares_line, verbose=False) == 0:
            problems.append("SELFTEST BROKEN: a <!--/derived--> close marker sharing a line with trailing WRONG "
                            "content (0.1525) PASSED -- hole #5 (a close marker only worked ALONE on its line) "
                            "is back")

        # ROUND 3, issues #6/#7 (fence-awareness), FAILING direction: a `# derived` PYTHON COMMENT sitting
        # INSIDE a fenced code block must not be mistaken for a heading -- round 2's `#{1,6}` generalization
        # (for hole #3) made it one, opening a phantom "derived SECTION" that swallowed everything after it
        # (including the WRONG number in ordinary prose past the fence) AND swallowed the in-fence number too
        # (round 2 has zero fence-awareness, so the fence-interior number also fell inside that same phantom
        # section). Two separate wrong numbers here, either one alone is enough to demonstrate the hole: the
        # in-fence 1.23456 (must be CHECKED as ordinary prose, exactly as every revision before round 3 already
        # did with no fence logic at all) and the post-fence 0.1525 (must not be swallowed by the phantom
        # section). Verified against round 2: it PASSED (both suppressed as "section").
        bad_fence_heading = os.path.join(d, "bad_fence_heading.md")
        open(bad_fence_heading, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "```python\n# derived thresholds below\nvalue = 1.23456\n```\n"
            "The accuracy was 0.1525 here.\n" % art)
        if check(bad_fence_heading, verbose=False) == 0:
            problems.append("SELFTEST BROKEN: a '# derived' PYTHON COMMENT inside a fenced code block, with a "
                            "WRONG number both INSIDE the fence (1.23456) and in ordinary prose AFTER it "
                            "(0.1525), PASSED -- hole #6 (a fence-interior '#' comment mistaken for a heading, "
                            "opening a phantom derived SECTION) is back")

        # ROUND 3, issue #7 (fence-awareness), PASSING direction: a derived paragraph (marker, intro sentence)
        # containing an illustrative fenced code snippet, then the paragraph's OWN legitimately-derived number
        # -- the fence must not end the paragraph's protection at its open delimiter. Verified against round 2:
        # it FAILED (`_ends_open_paragraph` treated the fence-open line like any other block-starter and closed
        # the block there, leaving the ratio outside protection).
        good_fence_paragraph = os.path.join(d, "good_fence_paragraph.md")
        open(good_fence_paragraph, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "<!--derived-->\nComputed by the runner shown below:\n"
            "```bash\npython -m research.runners.foo --seed 42\n```\n"
            "The resulting ratio, 0.104615, is a rounded re-quote of the artifact's own accuracy/baseline "
            "pair.\n" % art)
        if check(good_fence_paragraph, verbose=False) != 0:
            problems.append("SELFTEST BROKEN: a legitimately-derived paragraph (marker, intro sentence, an "
                            "illustrative fenced code snippet, then the derived number) FAILED -- hole #7 (a "
                            "fence unconditionally ended the paragraph's protection at its OPEN delimiter) is "
                            "back")

        # ROUND 3, issue #8 (list block_kind), FAILING direction: a WRONG number in ordinary prose well AFTER a
        # derived list has ended (a blank line separates them) must still FAIL -- the list's new persistent
        # `block_kind` must not over-broaden protection past the list's own end.
        bad_list_after_blank = os.path.join(d, "bad_list_after_blank.md")
        open(bad_list_after_blank, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "<!--derived-->\n- bullet one: 0.104615\n- bullet two: 0.207530\n\n"
            "The real accuracy was 0.1525 here, well past the list and not part of it.\n" % art)
        if check(bad_list_after_blank, verbose=False) == 0:
            problems.append("SELFTEST BROKEN: a WRONG number (0.1525) in an ordinary paragraph AFTER a derived "
                            "list has ended (separated by a blank line) PASSED -- the list's persistent "
                            "block_kind is over-broadening protection past the list's own end")

        # ROUND 3, issue #8, break shape (a), PASSING direction: a marker's OWN FIRST content line is itself a
        # list item, with legitimately-derived numbers on the list's SECOND and THIRD bullets. Verified against
        # round 2: it FAILED (the second bullet re-triggered "a list item starts a new block" under the old
        # para-only continuation rule and closed the block, leaving items 2+ unprotected).
        good_list_first_line = os.path.join(d, "good_list_first_line.md")
        open(good_list_first_line, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "<!--derived-->\n"
            "- first bullet, ratio 0.104615\n"
            "- second bullet, ratio 0.207531\n"
            "- third bullet, ratio 0.311079\n\n"
            "The baseline was 0.162500 here (matches the artifact).\n" % art)
        if check(good_list_first_line, verbose=False) != 0:
            problems.append("SELFTEST BROKEN: a standalone marker whose OWN FIRST content line is a list item, "
                            "with legitimately-derived numbers on the SECOND and THIRD bullets of the SAME "
                            "list, FAILED -- break shape (a) (a list had no persistent block_kind, so its "
                            "second bullet re-triggered 'starts a new block' and lost protection) is back")

        # ROUND 3, issue #8, break shape (b), PASSING direction: a marker's intro sentence (no blank line)
        # immediately followed by the list it introduces, with legitimately-derived numbers on the list's OWN
        # FIRST bullet -- the far more common real corpus pattern. Verified against round 2: it FAILED (the
        # intro sentence claimed the block's one paragraph unit, so the list's first bullet -- where the real
        # numbers live -- was already outside it).
        good_list_intro_sentence = os.path.join(d, "good_list_intro_sentence.md")
        open(good_list_intro_sentence, "w", encoding="utf-8").write(
            "# Some finding\n\nArtifact: `%s`\n\n"
            "<!--derived-->\n"
            "All values below are rounded re-quotes from the cited artifact's per-seed breakdown, with no "
            "blank line before the list that follows.\n"
            "- **Lesion collapses** -- mean 0.104615 (per-seed 0.09-0.12).\n"
            "- **Scramble control** -- mean 0.207531 (per-seed 0.19-0.22).\n" % art)
        if check(good_list_intro_sentence, verbose=False) != 0:
            problems.append("SELFTEST BROKEN: a marker's intro sentence (no blank line) immediately followed "
                            "by the list it introduces, with legitimately-derived numbers on the list's OWN "
                            "FIRST bullet, FAILED -- break shape (b) (the intro sentence claimed the block's "
                            "one paragraph unit and the list's first bullet was already outside it) is back")

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
