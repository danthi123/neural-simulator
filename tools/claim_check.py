#!/usr/bin/env python3
"""Verify a document's NUMBERS against the artifacts it cites. Blocks hallucinated claims.

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

    .venv/bin/python tools/claim_check.py research/findings/2026-07-31-foo.md
    .venv/bin/python tools/claim_check.py --selftest

Exit 1 if a measurement-shaped number is unsupported by the cited artifacts, if it matches only a pool so broad a
random number of its shape would match too, if a cited artifact is missing, if the file is unreadable, or if a
substantial document checks (almost) none of its numbers.

ROUND 8 DESIGN PRINCIPLE: FAIL CLOSED. Nothing is ever deleted, hidden or skipped to FIND numbers. Every earlier
round that did so opened a hole (see HISTORY): deleting `*`/`_` glued numbers to letters, a tag regex swallowed
`<FROZEN by 0.1525 ... >`, a comment-span finder that was not code-aware hid whole sections behind a literal
`<!--derived` in a code span, a prefix test on comment text turned `<!--derived-from ...-->` into a marker.

  1. READINGS. Numbers are extracted from the RAW file text, everywhere -- prose, tables, code spans, fences, HTML
     comments, attributes, frontmatter -- by NUM_RE, which accepts a leading dot (`.1525`), an exponent
     (`1.525e-1`), a glued unit (`0.1525ms`) and a scale suffix (`1.088B`, read as the scaled value OR the bare
     mantissa). A number is measurement-shaped when its stated precision is >= 3 decimals (d = fraction digits
     minus the exponent). A dash/minus glyph directly before the digits is a SIGN unless a digit, `.`, `)`, `]`
     or `%` precedes it (then it is a range or a subtraction). Where main and round 5 decide a dash the other way
     EVERY reading of the source reads the number BOTH ways, each a claim: an ASCII hyphen after a word character
     or after `)`, `]`, `%`; a true minus glyph (U+2212, U+2796, U+02D7) after a word character; and ANY other
     dash glyph (en/em dash, hyphen, the small/fullwidth hyphen-minus, ...) in a sign position -- main and round 5
     never read a non-ASCII dash as a sign. Main's side is decided on the ORIGINAL characters at the number's
     offset, never on a normalized copy (where every dash is `-` and a filler is a space): a number holding a
     non-ASCII digit is read only in the copies, and main reads it too (its regex takes a digit of any script).
     The one intended difference: a true minus glyph after a space or punctuation IS a minus sign, read one way.
     Numbers are ALSO extracted from two lightly normalized COPIES in which every
     character maps to exactly one character -- dash/minus variants to `-`, zero-width/format/combining/filler
     characters to a SPACE, any Unicode decimal digit to its ASCII digit, and dot-like characters between digits to
     `.` in one copy and every dot-like character to a SPACE in the other (the `.` copy turns `5`, U+00B7, U+0663,
     `2.5051` into `5.32.5051` and hides the 32.5051 main reads) -- so no reading can glue or drop anything. A
     further, ADDITIVE reading is the text a reader SEES (markdown-it's tokens: emphasis, tags,
     comments, code-span backticks and hidden elements render as NOTHING, entities and escapes are decoded), in
     two variants (elements that carry an attribute shown, and hidden -- a style can hide them). A reader number is
     the SAME claim as a raw one only by POSITION: made of verbatim source characters with no markup glued inside
     it and read the same way from the source characters before it. Every other reader number -- `0.15**25**`,
     `0.15<!---->25`, `0&#46;1525`, `0.15<ZWSP>25`, `<b>-</b>0.1625`, `_.1525_` -- is a claim of its own, checked
     and never exempt. (Matching by VALUE let any same-valued twin vouch for a split number: an exempt copy in an
     attribute, a link title or an image, or a raw number the reader sees glued into another.) A number fails if
     it is unsupported in ANY reading.
  2. EXEMPTION. A number is exempt only if its OWN physical line holds an exact marker `<!--derived-->` or
     `<!--derived: <note>-->` that markdown-it (both CommonMark and GFM-with-tables) parses as an HTML comment,
     i.e. outside code spans, fences, escapes and other HTML. On any line, the line is cut into cells at every `|`
     and at `<br>`/block-level tags, and a marker exempts only numbers in ITS OWN cell, at most
     MAX_EXEMPT_PER_MARKER (8) per marker. Nothing else exempts anything: a marker alone on a line, a `## Derived`
     heading, a `<!--/derived-->` close marker, a marker in code, an escaped marker, and any other spelling
     (`<!-- derived -->`, `<!--derived-from ...-->`) exempt NOTHING and print a WARNING.
  3. MATCHING. A number written with d decimals matches an artifact value v when |x - v| <= 0.5 * 10^-d (v rounds
     to x at the stated precision), or within the legacy relative tolerance max(5e-6, 1e-4 |x|) that main and
     round 5 used. The rule used (exact / rounding / legacy) is reported per number.
  4. DISCRIMINATING POWER, PER CLAIM. For every checked number, ALL 2 * CHANCE_WINDOW decoys x + k * 10^-d
     (k = +-1 .. +-CHANCE_WINDOW) are matched with rule 3; the fraction that match is that CLAIM's own chance-match
     rate -- exact, so it depends on the value, stated precision, scale suffix and tier, never on how the number is
     spelled. A number that matches, but whose own rate exceeds the limit for its stated precision, chance_max(d)
     (0.04 at 3 decimals, 0.15 at 4, CHANCE_MAX 0.20 at 5 or more), is NOT supported: it fails with "cite a
     narrower artifact or state more decimals". (Round 7 averaged the rate over the document, so one wrong coarse
     headline among many precise numbers passed.) The distribution is printed.
  5. KEPT FROM ROUND 5 / REQUIRED: WARNINGs for the inert scope idioms; strict UTF-8 and no bidirectional controls
     (UNREADABLE blocks); `claim_check: synthesis` applies only inside a CLOSED frontmatter block with a non-empty
     `claim_check_reason:` (EVERY such line non-empty, and no other `claim_check:` value in the block: a YAML
     loader keeps the last of duplicate keys), only where main and round 5 also read the flag (before the first
     `\n---` of a file that starts with `---`: no byte-order mark, no quoted value), and never when the filename,
     any frontmatter `title:`/`verdict:` line (any key case, continuation lines included), or any heading -- as
     written, as rendered
     (`G**O**`, `G<!-- -->O`, `&#71;O`), or an HTML `<h1>`-`<h6>` -- carries a verdict word (GO(s), NO-GO, NOGO,
     PASS(ED/ES), FAIL(ED/S), REFUTED, CONFIRMED; case-insensitive, invisible characters removed); the
     LOW_COVERAGE floor on DISTINCT checked values seen outside HTML comments, link-reference lines and hidden
     elements (an HTML comment ends at the FIRST `-->`, as a browser ends it); a citation inside one of those adds
     nothing to the pool (with a WARNING) but is still opened -- a missing or unreadable file fails, as in main and
     round 5; `tools/gates/claim_check_selftest.py` (class CCT) passes this file's selftest problems through
     verbatim.

HISTORY -- every round, and the hole each one left (each hole is a SELFTEST_CASES entry whose `wrong_on` is
re-derived from git by tests/test_claim_check_line_only.py on every run):
  main (7e2edc08e): a marker ALONE on a line opened a scope to the next `## ` heading; a scorer hid three whole
     sections (0 of 336 artifact values checked, two wrong numbers passed).
  r1-r4 (d4959ecb0, 6abb28469, 662e167e8, 214e509bf): four multi-line scope rules (line scanner, fence-aware
     scanner, fence toggle, CommonMark parser), each leaking through a heading level, a fence, an unclosed
     comment, or a container boundary.
  r5 (4fda849d4): SAME-LINE-ONLY exemption; the review found that rule SOUND and everything else a document can
     do to a NUMBER untested (a trailing-cell marker exempting a whole row, `_0.1525_`, `0.1525ms`, `.1525`,
     `0&#46;1525`, split digits, a dropped U+2212 sign, a synthesis escape with no conditions).
  r6 (f2b7db2b4): normalized by DELETING markup -- `gain*0.1525`, `**acc**0.1525`, `&Delta;0.1525` glued the
     number to a letter and hid it; 62% of what it flagged was correct roundings.
  r7 (4ff05b018): replaced markup by separators and treated comments as hidden carriers -- a tag regex swallowed
     `FULL<FROZEN by 0.1525 ... FROZEN>`; a literal `<!--derived` inside a code span or fence opened a "marker"
     span to the next `-->` anywhere and hid whole sections; any comment starting with `derived`
     (`<!--derived-from ...-->`) exempted numbers; the chance rate was a DOC average.
  r8 draft (a960fa231, staged when its session was killed): matched a reader number to a raw one by (line,
     value), so an exempt same-valued twin in an attribute, a link title or destination, an image, or glued into
     another number vouched for a split wrong number; a code span spanning a line break shifted every later reader
     line (16 numbers in 10 findings since 2026-09-01 re-checked as reader-only, 6 failing on nothing else);
     a bold range `**0.170**-**0.1625**` read as a minus sign; an ambiguous ASCII hyphen (`(a)-0.1625`) read one way,
     so a sign error main and r5 catch passed; the synthesis bar missed headings split by markup, a `title: |`
     block across a blank line, and accepted a quoted flag, a flag after an earlier `----` line and a byte-order
     mark that main and r5 refuse.
  r8a (654d95664, round 8 as reviewed): a flat 0.20 chance limit let wrong 3-decimal numbers through more often
     than main (replayed: 12.9% vs 7.6%; in the review's sampling 14.7% vs 5.1%); a non-ASCII dash before a number
     was read only as a minus sign (`lesion`, U+2013, `0.1625`), so a sign error main and r5 catch passed; the `.`
     copy glued a dot-like character and an Arabic-Indic digit into `5.32.5051` and hid the 32.5051 main reads; a
     hidden citation of a corrupt file was only checked for existence; the chance rate was a 100-decoy sample
     seeded by the claim's TEXT (one value rated 0.16 / 0.20 / 0.28 as `.417` / `0.417` / `00.417`); the second
     reader reading dropped a claim as a duplicate while ignoring its scale suffix
     (`0.15<b>2</b>5<span class="u">k</span>`); several scans were quadratic (210 KB of `x <b y` took 198 s, one
     line of `<!--derived: x` never finished); and no registry case pinned the BOTH-parsers marker rule (an
     either-parser mutant passed the whole suite).
  r8b (57b1e5f01, round 8's fix pass as reviewed): only the RAW reading read a number both ways, so a number
     holding a non-ASCII digit -- read only in the normalized copies, one sign each, decided on the copy -- was
     signed unlike main and r5 (`lesion`, U+2013, `0.1`, U+0666, `25` read -0.1625 only; `(a)-0.1`, U+0666, `25`
     +0.1625 only; a Hangul filler before `-` is a letter to main and a space in the copy) and main and r5 failed
     docs it passed; `title:`/`verdict:` were read from their FIRST line only (`title: notes` hid a later
     `title: Lane A GO`); PATH_RE (the same in main) rescanned a run of path characters from every position (50 KB
     of `[` 22.7 s, 100 KB 81 s), which the COST note blamed on markdown-it; the scale suffix in the duplicate
     key was pinned only by a unit test (a mutant that dropped it passed --selftest and the CCT gate); and (found
     by the differential fuzz against main's regex added for this round; r8a had it too) an exponent swallowed the
     first digit of a following number, so `1.8e-0.1525` hid the 0.1525 main reads.

CALIBRATION (2026-09-25; re-derive with `tools/claim_check_retro_compare.py --since 2026-09-01 --calibrate --replay`;
outputs committed as research/coordination/claimcheck_r8_retro_since2026-09-01_2026-09-25.{tsv,txt}):
  * The per-claim limit depends on the stated precision: CHANCE_MAX_BY_DECIMALS {3: 0.04, 4: 0.15}, CHANCE_MAX
    0.20 at 5 or more decimals -- the loosest of the swept limits at which round 8 accepts REPLAYED WRONG NUMBERS
    (every matched claim shifted by 1..10 units of its last decimal) no more often than main at EVERY stated
    precision, in both samplings the script prints. Every distinct matched claim, both directions: 3 decimals
    3.89% vs main 7.59%, 4: 6.85% vs 8.81%, 5: 16.74% vs 17.47%, 6+: 45.17% vs 81.79%. The review's sampling
    (claims the flat limit accepted, +1..+10): 3: 4.88% vs 5.07%, 4 and 5: equal (879 and 341 of 12,420 and
    1,230), 6+: 47.58% vs 82.44%. The flat 0.20 round 8 was reviewed with: 3 decimals 12.9% vs 7.6% (14.7% vs
    5.1%). Rates are measured on the 360 findings since 2026-09-01 with both rules on round 8's own pool.
  * The cost of those limits, same findings: 153 fail (0.20 flat: 119); 45 fail ONLY on numbers matched at
    their written precision but too broad -- each fixed by one more stated decimal or a narrower citation.
  * The per-claim rate over the 4,030 precision-tier matches: p50 0.020, p90 0.160, p95 0.218, p99 0.572. The
    19 legacy-only matches have median rate 0.52: the relative window survives only against sparse pools.
  * CHANCE_WINDOW = 500 units of the claim's last decimal either side; all 1,000 decoys are counted.
  * The rate is taken in the TIER that matched: a claim matched at its stated precision is rated against the
    precision rule, a legacy-only match against precision-or-legacy. (Rating every claim against the union made a
    6-decimal value that EXACTLY matches the artifact read as "too broad" beside a dense sweep -- the relative
    window, not the claim, was broad.)
  * LOW_COVERAGE_MIN_TOTAL = 30: the largest non-synthesis doc since 2026-09-01 under 5% distinct-visible-checked
    has 27 numeric claims.
  * FAILURES on the same 360 findings (none re-gated: the gate checks only NEWLY ADDED findings): round 8 fails
    153 (main 30, r5 138, r6 158, r7 107, r8a 122, r8b 153; r8a -> round 8: 31 newly fail, none newly pass;
    r8b -> round 8: no verdict and no flagged number changes -- no finding holds a number this round's fixes read
    differently, and every number, cause and rate below is re-derived unchanged). Of its 1,668
    flagged numbers, 1,030 (62%) MATCHED at their written precision but are too broad -- by this checker's own
    logic that match does not show them correct, since a random number of their shape matches too (round 8 as
    reviewed called such numbers "correct"; r5: 939 of its 1,395 flags were roundings main's window rejects). By
    cause: 1,044 too broad (14 of them legacy-only matches), 371 unmarked prose numbers (derived, aggregated,
    quoted, or wrong), 75 identifiers, 72 near misses (a truncation or a wrong rounding), 55 with nothing loaded,
    28 in code spans, 20 whose artifact holds the opposite sign, 2 in comments, 1 in a fence. The one finding
    main fails and round 8 passes writes 0.031 for a cited 0.0307 (a rounding at its written precision).

ACCEPTED TRADE vs main and round 5 (required by the round-8 spec): rule 3's precision window 0.5 * 10^-d is WIDER
than main's relative window for a coarse small number (0.477 matches a stored 0.4772; main's window is 4.8e-5), so
a wrong coarse number that lands in the window of an unrelated cited value can pass where main fails it -- bounded
per claim by rule 4, and in aggregate no more often than in main at any stated precision (CALIBRATION, measured
above: at 3 decimals 3.89% / 4.88% of replayed wrong numbers against main's 7.59% / 5.07%).
COST: every scan in this file is linear in the size of the document, and each is timed on adversarial input by the
tests (a claim of linearity is a hypothesis until measured: r8b's said so while its PATH_RE, as main's, took 22.7 s
on 50 KB of `[` and 81 s on 100 KB). markdown-it's html_inline rule, which rescanned to the end of the paragraph
from every unclosed `<!--`, `<?`, `<!X` or `<![CDATA[`, is guarded (`_html_inline_linear`, exact), and every regex
that rescanned a line or the text from each candidate is replaced by an index-and-bisect emulation pinned against
it by a differential test (the regexes stay here as the SPEC) -- PATH_RE included: a run-start lookbehind alone
still rescans a chain `a/a/a/...` from every `/`, so `_path_spans` reads each chain once. 200 KB adversarial
documents scan in 0.3-6 s. What remains is markdown-it-py's own parse, near linear and paid twice (CommonMark and
GFM): about 1.4 s per 100 KB per parser of a paragraph of `[` (plain markdown-it the same; 6.3 s for 400 KB), 0.7 s
per 100 KB of `[a](` -- main has no parser.
CANNOT CATCH (known): a number spelled in words; a decimal comma; homoglyph letters for digits; digit-group
separators (`0.152 5`); an integer mantissa with an exponent (`1525e-4`, as in main and r5); a wrong number within
the matching window of an unrelated cited value whose own chance rate is under its limit; a wrong number within
the legacy relative window of the right one (1e-4 |x|, as in main and r5); a value that IS in the artifact but
belongs to another quantity (existence is not agreement -- gates/stated_value_mismatch); content hidden by CSS
from a stylesheet on an element with no attribute.
BY DESIGN (fail closed): an identifier with >= 3 decimals (an arXiv id, a DOI prefix, a version inside a URL) is
checked like any number -- mark it on its own line (`<!--derived: arXiv id-->`); numbers inside fenced code
cannot be marked (a marker in a fence is code) -- cite an artifact that holds them or move them out of the fence;
a `|` or a `<br>` anywhere on a line (a table row or not) cuts it into cells, which only ever narrows an exemption;
digits either side of an image or an attribute-bearing element are also read glued.
"""
from __future__ import annotations

import bisect
import contextlib
import glob
import hashlib
import html
import io
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:                                               # pinned in requirements-dev.txt: markdown-it-py>=4.2,<5
    from markdown_it import MarkdownIt
    from markdown_it.common import html_re as _mdit_html_re
    from markdown_it.rules_inline.html_inline import html_inline as _mdit_html_inline
    _MD_IMPORT_ERROR = None
except Exception as _e:                            # fail CLOSED: without the parser no marker can be verified
    MarkdownIt = None
    _MD_IMPORT_ERROR = "%s: %s" % (type(_e).__name__, _e)

# =================================================================================================================
# character classes
# =================================================================================================================
# Every Unicode category-Pd character (as shipped with Python 3.11) plus the minus-sign glyphs that are
# not Pd: U+2212 MINUS SIGN, U+2796 HEAVY MINUS, U+02D7 MODIFIER LETTER MINUS, U+2043 HYPHEN BULLET.
_DASH_CHARS = frozenset("-\u058a\u05be\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015\u2e17\u2e1a"
                        "\u2e3a\u2e3b\u2e40\u2e5d\u301c\u3030\u30a0\ufe31\ufe32\ufe58\ufe63\uff0d\U00010ead"
                        "\u2212\u2796\u02d7\u2043")
_DOTLIKE = frozenset("\uff0e\u2024\ufe52\u00b7\u066b\u2027\u2e31\u0387")
_FILLERS = frozenset("\u115f\u1160\u3164\uffa0\u17b4\u17b5\u180e")
_INVISIBLE_CATS = frozenset(("Cf", "Mn", "Me", "Co", "Cn", "Cs"))
_BIDI_RE = re.compile("[\u202a-\u202e\u2066-\u2069]")


def _invisible(c):
    if c in "\n\t":
        return False
    cat = unicodedata.category(c)
    return cat in _INVISIBLE_CATS or cat == "Cc" or c in _FILLERS


# =================================================================================================================
# number syntax
# =================================================================================================================
_SIGN_CLASS = "".join(re.escape(c) for c in sorted(_DASH_CHARS))
# A number that starts with a digit is read unless a digit or a dot precedes it (then it is the tail of a longer
# number, a version or a date: `1.2.345`, `2026.09.25`) -- a LETTER or `_` before it does not hide it (`acc0.1525`,
# `_0.1525_`, `corr0.869` are all read; the whole corpus holds 11 such tokens, 0 since 2026-09-01, half of them run-
# name parameters like `sigma0.001` that must now be cited or marked). A number that starts with its dot is read
# unless a letter, digit, dot or `_` precedes it (`p.347` is a page, `x.125` a field). An exponent is read only when
# no `.digit` follows it: in `1.8e-0.1525` the `0` is not an exponent but the start of the 0.1525 main reads (the
# `-` before it is neither a word character nor `.` to main's regex) -- consuming it hid that number from every
# reading (found by the differential fuzz against main's regex).
_NUM_RE = re.compile(r"(?:(?<![0-9.])([0-9]+)|(?<![A-Za-z0-9._]))\.([0-9]+)"
                     r"(?:[eE]([+" + _SIGN_CLASS + r"]?[0-9]+)(?![.][0-9]))?(?![0-9])")
NUM_RE = _NUM_RE                                   # public alias (tests use it to list a text's numbers)
_MAGNITUDE = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}
_RANGE_LEFT = frozenset("0123456789.)]%")          # a dash after one of these is a range/subtraction, not a sign
MIN_DECIMALS = 3                                   # >= 3 stated decimals => a measurement, not prose

# Globs are allowed: a finding over N seeds cites one pattern, not N paths. Must contain a "/".
# The SPEC; applied by _path_spans. The lookbehind starts a match only at the start of a run of path characters,
# which finds the same matches (a match from inside a run extends to one from the run's start, and finditer tries
# that first); without it each position of a 50 KB run of `[` or `a` rescanned the run (22-26 s).
PATH_RE = re.compile(r"(?<![\w.\-*?\[\]])([\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.(?:jsonl|json))")
_PATH_RUN_RE = re.compile(r"[\w.\-*?\[\]]+")


def _path_spans(text):
    """`[m.span(1) for m in PATH_RE.finditer(text)]` in linear time. The lookbehind alone leaves the regex quadratic
    on a CHAIN of runs joined by single `/` (`a/a/a/...`: every run start rescans the chain to its end). A chain
    R0/R1/.../Rm holds at most one match: from R0's start to the LAST `.json` in the last run Rk (k >= 1) that has
    one after its first character (`.jsonl` when the `l` follows) -- the regex's greedy segments reach the furthest
    such ending first, and a match starting anywhere later in the chain would end later still."""
    out = []
    chain = []

    def flush():
        for k in range(len(chain) - 1, 0, -1):
            a, b = chain[k]
            i = text.rfind(".json", a + 1, b)
            if i >= 0:
                out.append((chain[0][0], i + 6 if text.startswith(".jsonl", i) else i + 5))
                return

    for m in _PATH_RUN_RE.finditer(text):
        a, b = m.span()
        if chain and a == chain[-1][1] + 1 and text[a - 1] == "/":
            chain.append((a, b))
        else:
            flush()
            chain = [(a, b)]
    flush()
    return out
VERDICT_RE = re.compile(r"\b(GO|NO-GO|PASS|FAIL|REFUTED|CONFIRMED)\b")

# =================================================================================================================
# markers
# =================================================================================================================
DERIVED_MARK = "<!--derived-->"
DERIVED_CLOSE = "<!--/derived-->"                  # closes nothing -- matched only to print a WARNING
# The ONLY two spellings that can exempt anything. One physical line; the note may not contain `-->`.
_EXACT_MARKER_RE = re.compile(r"<!--derived(?:-->|:[ \t]+[^\n]*?\S[^\n]*?-->)")   # the SPEC; matched by _marker_end
_MARKER_LIKE_RE = re.compile(r"<!--[ \t]*derived", re.I)   # anything an author may have MEANT as a marker
_STANDALONE_MARKER_RE = re.compile(r"^\s*(?:>\s*)*<!--\s*derived\b[^\n]*?-->\s*$", re.I)
_ATX_DERIVED_RE = re.compile(r"^\s*(?:>\s*)*#{1,6}\s*[*_`]*\s*derived\b", re.I)
_SETEXT_TITLE_RE = re.compile(r"^\s*(?:>\s*)*[*_`]*\s*derived\b", re.I)
_SETEXT_UNDERLINE_RE = re.compile(r"^\s*(?:=+|-+)\s*$")
MAX_EXEMPT_PER_MARKER = 8
MAX_EXEMPT_PER_LINE = MAX_EXEMPT_PER_MARKER        # name kept for callers of earlier rounds
# Cell cuts: every `|` (escaped or not, table row or not -- a cut only NARROWS an exemption), `<br>` and
# block-level tags.
_CELL_TAGS = (r"(?:br|p|div|li|tr|td|th|table|thead|tbody|tfoot|ul|ol|dl|dt|dd|h[1-6]|hr|blockquote|pre|section|"
              r"article|header|footer|details|summary|caption|figure|figcaption)\b")
_CELL_CUT_RE = re.compile(r"\||<\s*/?\s*" + _CELL_TAGS + r"[^>\n]*>", re.I)   # the SPEC; matched by _cell_cuts
_CELL_CUT_HEAD_RE = re.compile(r"\||<\s*/?\s*" + _CELL_TAGS, re.I)


def _cell_cuts(ln):
    """Start offsets of `_CELL_CUT_RE.finditer(ln)` (non-overlapping) in linear time: the regex scanned from every
    `<br` to the end of the line when no `>` followed."""
    cuts = []
    gt = None
    resume = 0
    for m in _CELL_CUT_HEAD_RE.finditer(ln):
        if m.start() < resume:
            continue
        if m.group() == "|":
            cuts.append(m.start())
            continue
        if gt is None:
            gt = _Next(ln, ">")
        g = gt(m.end())
        if g >= 0:
            cuts.append(m.start())
            resume = g + 1
    return cuts

# =================================================================================================================
# matching and discriminating power
# =================================================================================================================
LEGACY_REL_TOL = 1e-4
LEGACY_ABS_FLOOR = 5e-6
CHANCE_WINDOW = 500
CHANCE_DECOYS = 2 * CHANCE_WINDOW                  # every decoy, +-1 .. +-CHANCE_WINDOW: the rate is exact
# CALIBRATED 2026-09-25 -- see the docstring CALIBRATION (tools/claim_check_retro_compare.py --calibrate --replay).
# The limit depends on the stated precision: at 3 decimals the precision window (+-0.0005) is ~10x main's relative
# window for a number near 0.5, so a looser limit let wrong 3-decimal numbers through more often than main does.
CHANCE_MAX = 0.20                                  # 4 or more decimals
CHANCE_MAX_BY_DECIMALS = {3: 0.04, 4: 0.15}

# LOW COVERAGE (defense in depth against marking (almost) everything derived). Distinct checked values seen
# outside hidden regions / all numeric claims.
MIN_CHECK_FRACTION = 0.05
LOW_COVERAGE_MIN_TOTAL = 30

# =================================================================================================================
# synthesis
# =================================================================================================================
SYNTH_RE = re.compile(r"^claim_check:[ \t]*synthesis[ \t]*$", re.M)
# main's and round 5's own test, kept as a NECESSARY condition so the escape is never granted where they checked
# every number: the flag must precede the first `\n---` of a file that starts with `---` (no byte-order mark).
_MAIN_SYNTH_RE = re.compile(r"^claim_check:\s*synthesis\s*$", re.M)
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
_VERDICT_WORD_RE = re.compile(r"(?<![A-Za-z0-9])((?:no[ \t-]*)?gos?|pass(?:ed|es)?|fail(?:ed|s)?|refuted|confirmed)"
                              r"(?![A-Za-z0-9])", re.I)
_HTML_HEADING_RE = re.compile(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]\s*>", re.I | re.S)
_ATX_ANY_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t#]*$")
_SETEXT_ANY_RE = re.compile(r"^[ \t]{0,3}(?:=+|-+)[ \t]*$")

# =================================================================================================================
# hidden regions (for CITATIONS and the COVERAGE numerator only -- numbers there are still checked)
# =================================================================================================================
# markdown-it's own comment pattern -- kept for reference: a hidden comment span follows the BROWSER (_comment_spans)
_COMMENT_RE = re.compile(r"<!---?>|<!--(?:[^-]|-[^-]|--[^>])*-->")
# the SPEC of a hidden element's opening tag; matched in linear time by _hidden_element_spans (differential test)
_HIDDEN_OPEN_RE = re.compile(r"<([A-Za-z][A-Za-z0-9-]*)\b(?=[^>]*(?:\bhidden\b|display\s*:\s*none|"
                             r"visibility\s*:\s*hidden))[^>]*>|<(script|style|template|noscript)\b[^>]*>", re.I)
_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_INLINE_TAGS = frozenset("a abbr b bdi bdo big cite code data del dfn em font i ins kbd mark q s samp small span "
                         "strike strong sub sup time tt u var wbr img".split())
_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7};|#[xX][0-9a-fA-F]{1,6};|[A-Za-z][A-Za-z0-9]{1,31};)")

Claim = namedtuple("Claim", "start end line value decimals unit alts text reading")


# =================================================================================================================
# artifacts
# =================================================================================================================
def _flatten_numbers(obj, out):
    """Every finite numeric leaf in an artifact, at any depth, at full precision."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        if v == v and not math.isinf(v):
            out.add(v)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)


def _flatten_verdicts(obj, out):
    """Any key that looks like a verdict, with its value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in ("go", "verdict", "overall_verdict", "passed", "signal"):
                out.append((k, v))
            _flatten_verdicts(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_verdicts(v, out)


_ART_CACHE = {}


def _load_one(h):
    st = os.stat(h)
    key = (h, st.st_mtime_ns, st.st_size)
    if key in _ART_CACHE:
        return _ART_CACHE[key]
    vals, verdicts = set(), []
    with open(h, encoding="utf-8") as fh:
        if h.endswith(".jsonl"):
            for ln in fh:
                ln = ln.strip()
                if ln:
                    d = json.loads(ln)
                    _flatten_numbers(d, vals)
                    _flatten_verdicts(d, verdicts)
        else:
            d = json.load(fh)
            _flatten_numbers(d, vals)
            _flatten_verdicts(d, verdicts)
    res = (frozenset(vals), tuple(verdicts))
    if len(_ART_CACHE) > 4096:
        _ART_CACHE.clear()
    _ART_CACHE[key] = res
    return res


def load_artifacts(paths):
    """-> (sorted pool of distinct values, verdicts, loaded, missing)."""
    vals, verdicts, loaded, missing = set(), [], [], []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        hits = sorted(glob.glob(full)) if any(c in full for c in "*?[") else ([full] if os.path.exists(full) else [])
        if not hits:
            missing.append(p)
            continue
        for h in hits:
            try:
                v, vd = _load_one(h)
                vals |= v
                verdicts.extend(vd)
                loaded.append(h)
            except Exception as e:                       # narrow enough to see; never silent
                missing.append("%s (unreadable: %s)" % (p, type(e).__name__))
    return sorted(vals), verdicts, loaded, missing


# =================================================================================================================
# 1. readings
# =================================================================================================================
def _n_copy(text, dot_as_space=False):
    """The lightly normalized copy: every character maps to EXACTLY ONE character, so positions are preserved and
    nothing can be glued or dropped. A dot-like character between digits becomes `.`; with `dot_as_space` EVERY
    dot-like character becomes a space instead -- a second copy, because turning `5`, U+00B7, U+0663, `2.5051` into
    `5.32.5051` hides the 32.5051 main reads (the `.` glues it into a version-like run), while the space copy reads
    `5 32.5051`."""
    out = list(text)
    n = len(text)
    for i, c in enumerate(text):
        if c.isascii():
            continue
        if c in _DASH_CHARS:
            out[i] = "-"
        elif _invisible(c):
            out[i] = " "
        elif unicodedata.category(c) == "Nd":
            out[i] = str(unicodedata.decimal(c))
        elif c in _DOTLIKE:
            if dot_as_space:
                out[i] = " "
                continue
            left = text[i - 1] if i > 0 else ""
            right = text[i + 1] if i + 1 < n else ""
            if (left.isdigit() or left == "") and right.isdigit():
                out[i] = "."
    return "".join(out)


_WORDCHAR_RE = re.compile(r"\w")
# The glyphs that ARE minus signs. Every other character of _DASH_CHARS (en/em dash, hyphen, figure dash, the
# small/fullwidth hyphen-minus, ...) is punctuation as often as a sign, and main and round 5 never read it as one.
_MINUS_GLYPHS = frozenset("\u2212\u2796\u02d7")


def _sign_here(s, a):
    """Round 8's sign rule on string `s`: a dash/minus glyph directly before the digits at `a` is a sign unless a
    digit, `.`, `)`, `]` or `%` precedes it (a range or a subtraction)."""
    return a > 0 and s[a - 1] in _DASH_CHARS and (s[a - 2] if a >= 2 else " ") not in _RANGE_LEFT


def _sign_main(orig, a):
    """main's and round 5's sign for the number whose digits start at `a` of the ORIGINAL text. Their regex
    `(?<![\\w.])(-?\\d+\\.\\d{3,})` reads only an ASCII hyphen-minus as a sign, and only when neither a (Unicode) word
    character nor `.` precedes it; it reads digits of any script (`\\d`), so it reads `0.1`, U+0666, `25` too."""
    if a < 1 or orig[a - 1] != "-":
        return False
    prev = orig[a - 2] if a >= 2 else " "
    return not (prev == "." or _WORDCHAR_RE.match(prev))


def _ambiguous_hyphen(s, a, orig=None):
    """True when round 8's sign for the number at `a` of `s` (the raw text or a normalized copy) differs from the
    sign main and round 5 read at the SAME offset of the ORIGINAL text `orig` (default `s`; a normalized copy maps
    one character to one character, so the offsets agree), so the number is read BOTH ways, each reading a claim
    of its own. On the raw text that is:
      * an ASCII hyphen-minus after a letter or `_` (`acc-0.1525`: main/round 5 read no sign) or after `)`, `]`, `%`
        (`(a)-0.1525`: main/round 5 read a sign, round 8 a range);
      * a true minus glyph (U+2212, U+2796, U+02D7) after a word character (`x`, U+2212, `0.1625`): main/round 5
        never read a non-ASCII dash as a sign;
      * any OTHER dash glyph in a sign position (`lesion`, U+2013, `0.1625`; `x `, U+2014, `0.1625`).
    In a normalized copy the dash is always `-` and a filler or an invisible character before it is a space, so
    main's side MUST come from the original (`lesion`, U+2013, `0.1`, U+0666, `25` is read only in the copies:
    decided there alone it read -0.1625 only, and a sign error main catches passed).
    The one intended difference is not ambiguous: a true minus glyph after a space or punctuation IS a minus sign
    (every non-ASCII sign in the findings since 2026-09-01 is U+2212 there; main and round 5 read it unsigned)."""
    orig = s if orig is None else orig
    mine = _sign_here(s, a)
    if mine == _sign_main(orig, a):
        return False
    if mine and orig[a - 1] in _MINUS_GLYPHS and not _WORDCHAR_RE.match(orig[a - 2] if a >= 2 else " "):
        return False
    return True


def _extract(s, both_signs=False, orig=None):
    """Every measurement-shaped number in string `s` -> list of (start, end, value, decimals, unit, alts, text).
    `start` covers a sign when one is read. With `both_signs` (every reading of the source), a number whose sign
    main or round 5 would read the other way (decided on `orig`, the original text `s` is a 1:1 copy of; default
    `s`) is read BOTH signed and unsigned -- each reading is a claim of its own, so a sign error main or round 5
    would catch is never read away."""
    out = []
    for m in _NUM_RE.finditer(s):
        a, b = m.start(), m.end()
        ip, frac, exp = m.group(1), m.group(2), m.group(3)
        e = 0
        if exp:
            e = int(("-" + exp[1:]) if exp[0] in _DASH_CHARS else exp)
        d = len(frac) - e
        if d < MIN_DECIMALS:
            continue
        mag = float((ip or "0") + "." + frac + ("e%d" % e if exp else ""))
        neg = _sign_here(s, a)
        sa = a - 1 if neg else a
        alts = ()
        if b < len(s) and s[b] in _MAGNITUDE and (b + 1 >= len(s) or not (s[b + 1].isascii() and s[b + 1].isalnum())):
            alts = ((_MAGNITUDE[s[b]], s[b]),)
        suffix = alts[0][1] if alts else ""
        out.append((sa, b, -mag if neg else mag, d, 10.0 ** (-d), alts, ("-" if neg else "") + s[a:b] + suffix))
        if both_signs and _ambiguous_hyphen(s, a, orig):
            other = not neg
            out.append((a - 1 if other else a, b, -mag if other else mag, d, 10.0 ** (-d), alts,
                        ("-" if other else "") + s[a:b] + suffix))
    return out


# ---- the READER's reading (additive) ----------------------------------------------------------------------------
# The reader's text of every rendered block is a list of ELEMENTS (char, line, verbatim, raw_last):
#   char      a character the reader sees, or None for a BOUNDARY -- markup, a tag, a comment, a hidden element or an
#             invisible character: anything that renders as NOTHING;
#   verbatim  True when the character is the same character at the same place in the SOURCE (plain text, code);
#             False for a boundary and for a DECODED character (an entity, a backslash escape, a normalized autolink);
#   raw_last  the source character directly before whatever follows this element: a boundary's last markup character
#             (`*`, `_`, `~`, `` ` ``, `>`, `)`, `[`), a decoded character's `;` or escaped character.
# A number the reader sees is the SAME claim as a raw/normalized one only when it is made of verbatim characters with
# no boundary glued inside it and the source characters before it decide its start and sign the same way. Every
# other reader number is a claim of its OWN, checked and never exempt. The decision is by POSITION, never by value:
# a value-based match lets any twin with the same value vouch for the split number -- an exempt copy in an
# attribute, a link title, an image, or a raw number the reader sees glued into another (`0.1525<b></b>9`).
_HIDDEN_ATTR_RE = re.compile(r"\bhidden\b|display\s*:\s*none|visibility\s*:\s*hidden", re.I)
_OPEN_TAG_NAME_RE = re.compile(r"<\s*([A-Za-z][A-Za-z0-9-]*)")
_CLOSE_TAG_NAME_RE = re.compile(r"<\s*/\s*([A-Za-z][A-Za-z0-9-]*)")
_ATTR_TAG_RE = re.compile(r"<\s*[A-Za-z][A-Za-z0-9-]*\s+[^\s/>]")     # an opening tag that carries an attribute
_VOID_TAGS = frozenset("area base br col embed hr img input link meta source track wbr".split())


def _hidden_open(tag, hide_attr=False):
    """The element name when `tag` opens an element a reader never sees (a `hidden` attribute, display:none,
    visibility:hidden, or script/style/template/noscript) -- and, in the SECOND reading (`hide_attr`), any element
    that carries an attribute at all, since a style or a class can hide it (`0.15<span style="font-size:0">9</span>25`
    shows 0.1525). Else None."""
    m = _OPEN_TAG_NAME_RE.match(tag)
    if not m or tag.startswith("</") or tag.rstrip().endswith("/>"):
        return None
    name = m.group(1).lower()
    if name in ("script", "style", "template", "noscript") or _HIDDEN_ATTR_RE.search(tag):
        return name
    if hide_attr and name not in _VOID_TAGS and _ATTR_TAG_RE.match(tag):
        return name
    return None


def _code_span_newlines(src, tok, start):
    """How many line breaks the SOURCE of code span `tok` holds (markdown-it turns them into spaces, which would make
    every later reader line lag), found by searching the block's source from `start`. -> (count, next_start); (0,
    start) when it cannot be found -- only a reported line number depends on it."""
    body = "".join("[ \n]" if c == " " else re.escape(c) for c in tok.content)
    fence = re.escape(tok.markup or "`")
    m = re.compile(fence + "[ \n]?" + body + "[ \n]?" + fence).search(src, start)
    if not m:
        return 0, start
    return src.count("\n", m.start(), m.end()), m.end()


def _reader_text_inline(tok, hide_attr=False):
    """The reader's elements for one inline token: text and code (verbatim), decoded entities/escapes (not verbatim),
    and a BOUNDARY for every tag, comment, emphasis/strike/link marker, image and code-span fence; the content of a
    hidden element is skipped."""
    els = []
    ln = 0
    hide = []                                           # stack of hidden element names currently open
    links = []                                          # stack: is the open link an autolink?
    src = tok.content or ""
    code_at = 0

    def put(s, verbatim=True, raw_last=None):
        for c in s:
            if _invisible(c):
                els.append((None, ln, False, " "))
            else:
                els.append((c, ln, verbatim, c if (verbatim or raw_last is None) else raw_last))

    def gap(raw_last):
        els.append((None, ln, False, raw_last or " "))
    for ch in tok.children or ():
        t = ch.type
        if t in ("softbreak", "hardbreak"):
            if hide:
                gap(" ")
            else:
                put("\n")
            ln += 1
        elif t == "code_inline":
            nl, code_at = _code_span_newlines(src, ch, code_at)
            gap("`")
            if not hide:
                put(ch.content)
            ln += nl
            gap("`")
        elif t in ("text", "text_special"):
            if hide:
                continue
            if t == "text_special":                     # an entity or a backslash escape: DECODED
                put(ch.content, False, (ch.markup or ch.content)[-1:])
            elif links and links[-1]:                   # an autolink's text is its NORMALIZED url
                put(ch.content, ("<" + ch.content + ">") in src)
            else:
                put(ch.content)
        elif t == "html_inline":
            name = _hidden_open(ch.content, hide_attr)
            closed = False                              # this tag closes a hidden element (no box: no break)
            if name:
                hide.append(name)
            elif hide:
                m = _CLOSE_TAG_NAME_RE.match(ch.content)
                if m and m.group(1).lower() == hide[-1]:
                    hide.pop()
                    closed = True
            tm = None if ch.content.startswith("<!") else _OPEN_TAG_NAME_RE.match(ch.content.replace("/", "", 1))
            if tm and tm.group(1).lower() not in _INLINE_TAGS and not hide and not closed:
                els.append((" ", ln, False, ">"))        # `<br>`, `</p><p>`, `</td>`: a break the reader sees
            else:
                gap(ch.content[-1:])
            ln += ch.content.count("\n")
        elif t == "image":                              # alt text, src and title render as no text
            gap(")")
            ln += sum(1 for g in ch.children or () if g.type in ("softbreak", "hardbreak"))
        elif t == "link_open":
            links.append(ch.markup in ("autolink", "linkify"))
            gap("<" if links[-1] else "[")
        elif t == "link_close":
            gap(">" if (links.pop() if links else False) else ")")
        else:                                           # emphasis, strong, strikethrough open/close
            gap((ch.markup or " ")[-1:])
    return els


def _reader_text_html(content, hide_attr=False):
    """Approximate rendering of an HTML block: comments and inline tags render as nothing, other tags as a break, a
    hidden element's content is skipped, entities are decoded (not verbatim)."""
    els = []
    ln, i, n = 0, 0, len(content)
    comment_end = dict(_comment_spans(content))          # linear: no per-`<` rescan to the end of the block
    while i < n:
        c = content[i]
        if c == "<":
            m = _TAG_RE.match(content, i) if i not in comment_end else None
            if i in comment_end or m:
                end = comment_end[i] if i in comment_end else m.end()
                tag = content[i:end]
                name = None if tag.startswith("<!") else _hidden_open(tag, hide_attr)
                if name:
                    close = re.compile(r"</\s*%s\s*>" % re.escape(name), re.I).search(content, end)
                    end = close.end() if close else n
                    tag = content[i:end]
                tm = None if tag.startswith("<!") or name else _OPEN_TAG_NAME_RE.match(tag.replace("/", "", 1))
                if tm and tm.group(1).lower() not in _INLINE_TAGS:
                    els.append((" ", ln, False, ">"))    # a block-level tag breaks the text
                else:
                    els.append((None, ln, False, ">"))
                ln += tag.count("\n")
                i = end
                continue
        if c == "&":
            m = _ENTITY_RE.match(content, i)
            if m:
                for c2 in html.unescape(m.group(0)):
                    els.append((None, ln, False, " ") if _invisible(c2) else (c2, ln, False, ";"))
                i = m.end()
                continue
        els.append((None, ln, False, " ") if _invisible(c) else (c, ln, True, c))
        if c == "\n":
            ln += 1
        i += 1
    return els


def _reader_segments(tokens, hide_attr=False):
    """[(token, first_line, elements)] -- the reader's text of every rendered block."""
    segs = []
    for tok in tokens:
        if not tok.map:
            continue
        if tok.type == "inline":
            els = _reader_text_inline(tok, hide_attr)
        elif tok.type in ("fence", "code_block"):
            els, k = [], (1 if tok.type == "fence" else 0)
            for c in tok.content:
                els.append((None, k, False, " ") if _invisible(c) else (c, k, True, c))
                if c == "\n":
                    k += 1
        elif tok.type == "html_block":
            els = _reader_text_html(tok.content, hide_attr)
        else:
            continue
        segs.append((tok, tok.map[0], els))
    return segs


def _resolve_boundaries(els):
    """Markup renders as NOTHING: a run of boundaries is removed and the reader sees the characters on either side
    side by side -- `0.15**25**`, `0.15<!---->25`, `0.15<ZWSP>25` and `<b>-</b>0.1625` show one number,
    `` `0.956`-`1.013` `` a range, `**acc**0.1525` the number 0.1525 after a word. (This reading is additive, so
    gluing can never hide a number the raw reading holds.) The character right after a run records the run's last
    markup character, which is the SOURCE character directly before it. -> [(char, line, verbatim, raw_last,
    joined)], `joined` None or that markup character."""
    out = []
    pend = None
    for c, ln, vb, raw in els:
        if c is None:
            pend = raw or " "
            continue
        out.append((c, ln, vb, raw, pend))
        pend = None
    return out


def _source_before(keep, s, i):
    """The two SOURCE characters directly before reader character i (normalized; fewer at a block start): a glued
    markup run contributes its last character, a decoded character its own last source character."""
    out = []
    p = i
    while len(out) < 2:
        if keep[p][4]:
            out.append(_n_copy(keep[p][4]))
            if len(out) == 2:
                break
        p -= 1
        if p < 0:
            break
        out.append(s[p] if keep[p][2] else _n_copy(keep[p][3]))
    return "".join(reversed(out))


def _same_claim_as_source(keep, s, raw_s, src, sa, b, v, d):
    """True when the reader's number s[sa:b] is exactly a number of the raw/normalized readings at the same place:
    every character verbatim, no markup glued inside it (between its sign and digits included), the two source
    characters before it (which decide where it starts and whether a dash is its sign) reading it the same way, and
    its text occurring in the block's source."""
    if any(not keep[p][2] for p in range(sa, b)) or any(keep[p][4] for p in range(sa + 1, b)):
        return False
    pre = _source_before(keep, s, sa)
    off = len(pre)
    if not any(x[0] == off and x[1] == off + b - sa and x[3] == d and abs(x[2] - v) <= 1e-12 * max(1.0, abs(v))
               for x in _extract(pre + s[sa:b + 2])):
        return False
    return raw_s[sa:b] in src


def _reader_claims(tokens):
    """Numbers only the READER's reading holds -- split by markup or an invisible character, decoded from an entity or
    an escape, or signed/started differently from the source -- each a claim of its own, checked and never exempt.
    Two readings: elements with attributes shown, and hidden (a style can hide them); the second adds only numbers
    the first does not already hold on that line. Additive: it can only ADD a failure."""
    extra = []
    first = Counter()
    for hide_attr in (False, True):
        count = Counter()
        for tok, l0, els in _reader_segments(tokens, hide_attr):
            if hide_attr and not (tok.type == "html_block" or _ATTR_TAG_RE.search(tok.content or "")):
                continue                                # the second reading differs only where a tag has attributes
            keep = _resolve_boundaries(els)
            if not keep:
                continue
            raw_s = "".join(k[0] for k in keep)
            src = tok.content or ""
            found = {}                                  # both normalized copies (see _n_copy), one entry per claim
            for s in (_n_copy(raw_s), _n_copy(raw_s, True)):
                for (sa, b, v, d, u, alts, txt) in _extract(s):
                    k = (sa, b, round(v, 12), d, alts)
                    if k not in found and not _same_claim_as_source(keep, s, raw_s, src, sa, b, v, d):
                        found[k] = (sa, v, d, u, alts, txt)
            for (sa, v, d, u, alts, txt) in found.values():
                line = l0 + keep[min(sa, len(keep) - 1)][1]
                # The second reading drops a number only when the first already holds a claim that is checked
                # IDENTICALLY -- same value, stated precision AND scale suffix (a hidden `k` changes what is checked:
                # `0.15<b>2</b>5<span class="u">k</span>` is 152.5 in the first reading, 0.1525 in the second).
                key = (line, round(v, 12), d, alts)
                count[key] += 1
                if hide_attr and count[key] <= first[key]:
                    continue
                extra.append(Claim(None, None, line, v, d, u, alts, txt, "reader"))
        first = count
    return extra


# =================================================================================================================
# 2. markers (verified by markdown-it on a LABELED copy)
# =================================================================================================================
_PARSERS = {}


def _parsers():
    # `text_join` is disabled so an entity or a backslash escape stays its own `text_special` token: the reader's
    # reading must know which characters were DECODED (`0&#46;1525` is not the source text `0.1525`).
    if not _PARSERS:
        _PARSERS["cm"] = MarkdownIt("commonmark").disable("text_join")
        _PARSERS["gfm"] = MarkdownIt("commonmark").enable("table").enable("strikethrough").disable("text_join")
        for md in _PARSERS.values():
            md.inline.ruler.at("html_inline", _html_inline_linear)
    return _PARSERS["cm"], _PARSERS["gfm"]


# markdown-it's html_inline rule runs HTML_TAG_RE.search(src[pos:]) at EVERY `<`: an unclosed `<!--`, `<?`, `<!X` or
# `<![CDATA[` rescans to the end of the paragraph each time (80 KB of `x <!-- y` took 21 s; main, which has no parser,
# is instant). `_html_inline_linear` decides, from positions indexed once per paragraph, whether HTML_TAG_RE CAN match
# at `pos`; when it cannot it returns False (exactly what the rule would), and when it can it runs the rule itself,
# whose scan then ends at the match -- consumed text is never rescanned. The decision is exact (a differential test
# pins it against HTML_TAG_RE on random text).
_MDIT_TAG_RE = None


def _mdit_comment_closes(src):
    """Where markdown-it's comment body `(?:[^-]|-[^-]|--[^>])*` stops: its tokens are forced (a dash run is eaten 3
    at a time), so from any dash run entered at a token boundary the body stops at `-->` exactly when the run has
    k >= 2 dashes, k % 3 == 2, and `>` follows. -> sorted positions of those `-->`."""
    out = []
    n = len(src)
    i = src.find("-")
    while i >= 0:
        j = i
        while j < n and src[j] == "-":
            j += 1
        if j - i >= 2 and (j - i) % 3 == 2 and j < n and src[j] == ">":
            out.append(j - 2)
        i = src.find("-", j)
    return out


def _mdit_comment_at(src, pos, closes):
    """True when markdown-it's `comment` pattern matches at `pos` (which starts with `<!--`)."""
    if src.startswith("<!-->", pos) or src.startswith("<!--->", pos):
        return True
    n = len(src)
    j = pos + 4
    while j < n and src[j] == "-":
        j += 1
    k = j - (pos + 4)
    if j >= n:
        return False                                     # the body is stuck at the end of the text
    if src[j] == ">" and k >= 2 and k % 3 == 2:
        return True
    # src[j] ends the leading dash run (as `c`, `-c` or `--c`); every later run is entered at a token boundary
    i = bisect.bisect_left(closes, j + 1)
    return i < len(closes)


def _mdit_html_possible(src, pos, memo):
    """True exactly when markdown-it's HTML_TAG_RE.search(src[pos:]) matches (`src[pos]` is `<`); `memo` caches the
    per-paragraph indexes. Pinned against the regex by a differential test."""
    def last(sub):
        if sub not in memo:
            memo[sub] = src.rfind(sub)
        return memo[sub]
    c = src[pos + 1] if pos + 1 < len(src) else ""
    if src.startswith("<!--", pos):
        if "closes" not in memo:
            memo["closes"] = _mdit_comment_closes(src)
        return _mdit_comment_at(src, pos, memo["closes"])
    if c == "?":
        return last("?>") >= pos + 2                     # processing: `<?` ... the first `?>`
    if src.startswith("<![CDATA[", pos):
        return last("]]>") >= pos + 9
    if c == "!":                                         # declaration: `<!` + a letter ... the first `>`
        return pos + 2 < len(src) and src[pos + 2].isascii() and src[pos + 2].isalpha() and last(">") >= pos + 3
    if c == "/" or (c.isascii() and c.isalpha()):
        global _MDIT_TAG_RE
        if _MDIT_TAG_RE is None:
            _MDIT_TAG_RE = re.compile("(?:" + _mdit_html_re.open_tag + "|" + _mdit_html_re.close_tag + ")")
        return _MDIT_TAG_RE.match(src, pos) is not None
    return False


def _html_inline_linear(state, silent):
    src, pos = state.src, state.pos
    if src.startswith("<", pos) and pos + 2 < state.posMax and state.md.options.get("html", None):
        if not _mdit_html_possible(src, pos, state.__dict__.setdefault("_claim_check_memo", {})):
            return False
    return _mdit_html_inline(state, silent)


def _label_prefix(text):
    for c in "\u2603\u2604\u2605\u2606\u2622\u2623\u262f":
        if c not in text:
            return c
    return None


class _Next:
    """Memoized `s.find(sub, frm)` for non-decreasing `frm`: a run of lookups that share one far (or missing) target
    costs one scan, not one scan each."""

    def __init__(self, s, sub):
        self.s, self.sub, self.frm, self.at = s, sub, -1, -1

    def __call__(self, frm):
        if not (self.frm >= 0 and self.frm <= frm and (self.at < 0 or frm <= self.at)):
            self.frm, self.at = frm, self.s.find(self.sub, frm)
        return self.at


def _marker_end(s, j, nl, close):
    """End of an exact marker whose `<!--derived` (or its 7-character label) ends at `j`, or None -- exactly what
    `_EXACT_MARKER_RE` matches there (`-->`, or `:`, one or more spaces/tabs, then a note holding a non-space character
    and ending at the first `-->` after it, all on one line) in LINEAR time: the regex's two lazy runs rescanned the
    rest of the line for every start, so one long line of `<!--derived: x ` never finished. `nl` and `close` are
    `_Next` finders for `\\n` and `-->` over `s`."""
    if s.startswith("-->", j):
        return j + 3
    if not s.startswith(":", j):
        return None
    line_end = nl(j)
    line_end = len(s) if line_end < 0 else line_end
    p = j + 1
    while p < line_end and s[p] in " \t":
        p += 1
    if p == j + 1:
        return None
    while p < line_end and s[p].isspace():               # `[^\n]*?\S`: the first non-space character of the note
        p += 1
    if p >= line_end:
        return None
    e = close(p + 1)
    return e + 3 if 0 <= e and e + 3 <= line_end else None


def _exact_marker_starts(text):
    """Start offsets of `_EXACT_MARKER_RE.finditer(text)` (non-overlapping), in linear time."""
    starts = []
    nl, close = _Next(text, "\n"), _Next(text, "-->")
    i = 0
    while True:
        a = text.find("<!--derived", i)
        if a < 0:
            return starts
        e = _marker_end(text, a + 11, nl, close)
        if e is None:
            i = a + 1
        else:
            starts.append(a)
            i = e


def _labeled(text, marks):
    """Replace the 7 letters `derived` of each exact marker with a unique 7-character label (same length, same
    markdown meaning), so the parse can say which occurrence became an HTML comment. -> (text, labels, prefix)."""
    pre = _label_prefix(text)
    if pre is None:
        return text, [None] * len(marks), None       # cannot label: no marker is live (fails closed)
    buf = list(text)
    labels = []
    for i, start in enumerate(marks):
        lab = pre + "%06d" % i
        a = start + 4
        buf[a:a + 7] = list(lab)
        labels.append(lab)
    return "".join(buf), labels, pre


def _label_at(s, prefix):
    """The label when `s` STARTS with an exact labeled marker (`<!--` + label + the marker's tail), else None ->
    (label, end)."""
    if not (s.startswith("<!--" + prefix) and len(s) >= 11 and s[5:11].isascii() and s[5:11].isdigit()):
        return None, None
    e = _marker_end(s, 11, _Next(s, "\n"), _Next(s, "-->"))
    return (s[4:11], e) if e is not None else (None, None)


def _live_labels(tokens, prefix):
    """Labels whose marker markdown-it parsed as an HTML comment: an html_inline token that IS the marker, or an
    html_block that STARTS with it. Nothing inside code, escapes, image alt text, or elsewhere inside other HTML
    counts."""
    live = set()
    if prefix is None:
        return live
    for tok in tokens:
        if tok.type == "html_block":
            lab, _e = _label_at(tok.content.lstrip(" "), prefix)
            if lab:
                live.add(lab)
        elif tok.type == "inline":
            for ch in tok.children or ():
                if ch.type == "html_inline":
                    lab, e = _label_at(ch.content, prefix)
                    if lab and e == len(ch.content):
                        live.add(lab)
    return live


def _where_label(tokens, lab):
    for tok in tokens:
        if tok.type in ("fence", "code_block") and lab in tok.content:
            return "inside a code block"
        if tok.type == "html_block" and lab in tok.content:
            return "inside an HTML block (only a marker that STARTS the block is a comment markdown sees)"
        if tok.type == "inline":
            for ch in tok.children or ():
                if lab in (ch.content or ""):
                    if ch.type == "code_inline":
                        return "inside a code span"
                    if ch.type == "text":
                        return "escaped or not a comment to markdown (it renders as visible text)"
                if ch.type == "image" and any(lab in (g.content or "") for g in (ch.children or ())):
                    return "inside an image's alt text"
    return "not parsed as an HTML comment"


# =================================================================================================================
# hidden regions (citations + coverage numerator only)
# =================================================================================================================
def _hidden(text, tokens, line_starts):
    """-> (hidden_lines set, hidden char spans sorted). Over-inclusive on purpose: a region wrongly judged hidden only
    loses its citations and its coverage credit (fails closed); its numbers are still checked."""
    # A line markdown renders nowhere (a link-reference definition such as `[//]: # (...)`) is hidden whole. An HTML
    # block is NOT hidden whole: a browser shows its text (`<!--derived--> see raw/x.json` at a line start is an HTML
    # block whose citation a reader sees); only the comments and hidden elements inside it are hidden, below.
    covered = set()
    hidden_lines = set()
    for tok in tokens:
        if tok.map and tok.type in ("inline", "fence", "code_block", "html_block"):
            covered.update(range(tok.map[0], tok.map[1]))
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip() and i not in covered:
            hidden_lines.add(i)                         # link-reference definitions, or anything not rendered
    spans = _comment_spans(text) + _hidden_element_spans(text)
    return hidden_lines, _merge(spans)


_TAG_START_RE = re.compile(r"<([A-Za-z][A-Za-z0-9-]*)\b")
_CLOSE_ANY_RE = re.compile(r"</\s*([A-Za-z][A-Za-z0-9-]*)\s*>")
_HIDDEN_ELEMENT_RE = re.compile(r"<(script|style|template|noscript)\b", re.I)
_HIDDEN_ATTR_PARTS = [re.compile(p, re.I) for p in (r"\bhidden\b", r"display\s*:\s*none", r"visibility\s*:\s*hidden")]


def _hidden_element_spans(text):
    """The spans `_HIDDEN_OPEN_RE.finditer` + a search for each element's close tag give, in LINEAR time: an opening
    tag from `<name` to the next `>` that carries a hidden attribute (or names script/style/template/noscript) is
    hidden up to the first close tag of ITS name after it, or to the end of the text; matches never overlap. The regex
    scanned from every `<letter` to the next `>` and searched for a close tag from every hidden opener (210 KB of
    `x <b y` took 198 s). Here the next `>` is found once per run of openers, hidden attributes (each alternative
    separately, so one nested in another is still seen) and close tags are indexed once, and every lookup is a
    bisection. The regex may shorten the tag name to any word boundary inside it (`<span-hidden>` is the element
    `span-` with the attribute `hidden`), so the name chosen is the LONGEST boundary prefix with an attribute after
    it, as the regex's backtracking chooses; a differential test pins this against the regex."""
    attr_starts = sorted({m.start() for rx in _HIDDEN_ATTR_PARTS for m in rx.finditer(text)})
    closes = {}
    for m in _CLOSE_ANY_RE.finditer(text):
        closes.setdefault(m.group(1).lower(), []).append((m.start(), m.end()))
    spans = []
    gt = _Next(text, ">")
    resume = 0                                           # finditer never reports a match overlapping the last one
    for m in _TAG_START_RE.finditer(text):
        if m.start() < resume:
            continue
        g = gt(m.start())
        if g < 0:
            break                                        # no `>` after this point: no opening tag can end
        i = bisect.bisect_left(attr_starts, g) - 1
        last_attr = attr_starts[i] if i >= 0 else -1     # the last hidden attribute that starts before the `>`
        name = None
        if last_attr >= m.start() + 2:
            # the longest name ending at a word boundary (inside the name: next to a `-`) at or before last_attr
            e = min(m.end(), last_attr)
            while e > m.start() + 2 and e < m.end() and (text[e - 1] == "-") == (text[e] == "-"):
                e -= 1
            if e <= m.end() and (e == m.end() or (text[e - 1] == "-") != (text[e] == "-")):
                name = text[m.start() + 1:e].lower()
        if name is None:
            h = _HIDDEN_ELEMENT_RE.match(text, m.start())
            if not h:
                continue
            name = h.group(1).lower()
        cl = closes.get(name, [])
        j = bisect.bisect_left(cl, (g + 1, -1))
        spans.append((m.start(), cl[j][1] if j < len(cl) else len(text)))
        resume = g + 1
    return spans


def _comment_spans(text):
    """HTML comments as a BROWSER ends them -- `<!--` to the first `-->` after it (`<!-->` and `<!--->` included) --
    in linear time; an unclosed `<!--` hides nothing. (`_COMMENT_RE.finditer`, markdown-it's older pattern, re-scanned
    to the end of the text from every unclosed `<!--`, and could run past a `-->` (`<!-- a ---> b -->`) that a
    browser stops at -- the text after it is VISIBLE, so its citations count.)"""
    spans = []
    i = 0
    while True:
        a = text.find("<!--", i)
        if a < 0:
            return spans
        e = text.find("-->", a + 2)
        if e < 0:
            return spans                                 # no `-->` after this point closes any later `<!--` either
        spans.append((a, e + 3))
        i = e + 3


def _merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _in_spans(merged, pos):
    i = bisect.bisect_right(merged, (pos, float("inf"))) - 1
    return i >= 0 and merged[i][0] <= pos < merged[i][1]


# =================================================================================================================
# 5. synthesis
# =================================================================================================================
def _fm_values(fm, key):
    """The value of EVERY `key:` line of frontmatter `fm`, in order ([] when there is none). A YAML loader keeps the
    LAST of duplicate keys and a reader may take any of them, so each is read: reading only the first let
    `title: notes` hide a later `title: Lane A GO` from the verdict-word bar.
    The key is matched case-insensitively and quoted or not (`Title:`, `"title":`) -- barring is the fail-closed
    direction, so every spelling a reader would take for the title is read.
    (`^key:[ \t]*(.*?)[ \t]*$` rescanned the trailing blanks for every character of the value: quadratic on a long
    line; the prefix is matched here and the value is the rest of its line with trailing blanks removed. Each
    continuation scan stops at the first unindented line -- every key line is one -- so no line is scanned twice.)"""
    out = []
    n = len(fm)
    for m in re.finditer(r"^[\"']?%s[\"']?[ \t]*:[ \t]*" % re.escape(key), fm, re.M | re.I):
        e = fm.find("\n", m.end())
        v = fm[m.end():n if e < 0 else e].rstrip(" \t")
        # Every indented or blank line after the key continues its value -- a block scalar (`|`, `>`, blank lines
        # included) or a multi-line plain or quoted scalar (`title: 'Lane A` / `  GO'`) -- so a verdict word on a
        # continuation line is still read.
        block = []
        while e >= 0:
            s = e + 1
            e = fm.find("\n", s)
            ln = fm[s:n if e < 0 else e]
            if ln.startswith((" ", "\t")) or not ln.strip():
                block.append(ln.strip())
            else:
                break
        if v in ("|", ">", "|-", ">-", "|+", ">+"):
            v = ""
        v = " ".join(x for x in [v] + block if x)
        v = v.strip().strip("\"'").strip()
        out.append("" if v in ("~", "null", "Null", "NULL") else v)
    return out


def _fm_value(fm, key):
    """The FIRST `key:` value (see _fm_values), or ""."""
    vals = _fm_values(fm, key)
    return vals[0] if vals else ""


def _verdict_word(s):
    """The first verdict word in `s`, case-insensitive, in two readings (invisible characters DELETED and replaced by
    a space) -- barring synthesis is fail-closed, so either reading bars."""
    s = s or ""
    base = "".join("-" if c in _DASH_CHARS else c for c in s).replace("_", " ")
    for reading in ("".join(c for c in base if not _invisible(c)),
                    "".join(" " if _invisible(c) else c for c in base)):
        m = _VERDICT_WORD_RE.search(unicodedata.normalize("NFKC", reading))
        if m:
            return m.group(1)
    return None


_TAG_OR_COMMENT_RE = re.compile(r"<!--.*?-->|<[^<>]*>", re.S)      # the SPEC; applied by _strip_tags_comments
_LT_GT_RE = re.compile(r"[<>]")
_HTML_H_OPEN_RE = re.compile(r"<h[1-6]\b", re.I)
_HTML_H_CLOSE_RE = re.compile(r"</h[1-6]\s*>", re.I)


def _strip_tags_comments(s):
    """`_TAG_OR_COMMENT_RE.sub("", s)` in linear time (its lazy comment scan ran to the end from every unclosed
    `<!--`)."""
    out = []
    close = _Next(s, "-->")
    i, n = 0, len(s)
    while i < n:
        if s[i] == "<":
            if s.startswith("<!--", i):
                e = close(i + 4)
                if e >= 0:
                    i = e + 3
                    continue
            m = _LT_GT_RE.search(s, i + 1)
            if m and m.group() == ">":
                i = m.end()
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _html_headings(text):
    """The content (group 1) of every `_HTML_HEADING_RE.finditer(text)` match, in linear time: an opening `<hN` to
    the first `>` after it, then up to the first `</hN>` after that; matches never overlap."""
    closes = [(m.start(), m.end()) for m in _HTML_H_CLOSE_RE.finditer(text)]
    gt = _Next(text, ">")
    out = []
    resume = 0
    for m in _HTML_H_OPEN_RE.finditer(text):
        if m.start() < resume:
            continue
        g = gt(m.end())
        if g < 0:
            break
        j = bisect.bisect_left(closes, (g + 1, -1))
        if j >= len(closes):
            break                                        # no close after this `>`: no later opener closes either
        out.append(text[g + 1:closes[j][0]])
        resume = closes[j][1]
    return out


def _atx_text(ln):
    """`_ATX_ANY_RE.match(ln)` in linear time -> the heading text (group 2, or "" when absent), or None when the line
    is not an ATX heading. (The lazy text group re-scanned the trailing ` #` run for every character.)"""
    i = 0
    while i < 3 and i < len(ln) and ln[i] in " \t":
        i += 1
    j = i
    while j < len(ln) and j - i < 6 and ln[j] == "#":
        j += 1
    if j == i:
        return None
    rest = ln[j:]
    if not rest.strip(" \t#"):
        return ""                                        # only blanks and `#` after the opening run
    if rest[0] not in " \t":
        return None
    body = rest.lstrip(" \t")
    return body[:len(body.rstrip(" \t#"))]


def _synthesis_status(text, doc_path, tokens=(), bom=False):
    """-> (is_synthesis, reason, warning_or_None). Headings are read as WRITTEN and as a reader SEES them (markdown-it's
    heading text, with emphasis, comments and tags rendering as nothing and entities decoded: `G**O**`, `G<!-- -->O`
    and `&#71;O` all show GO)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        if SYNTH_RE.search(text):
            return False, None, ("`claim_check: synthesis` appears outside a closed frontmatter block -- ignored, "
                                 "every number is checked")
        return False, None, None
    fm = m.group(1)
    if not SYNTH_RE.search(fm):
        return False, None, None
    if bom or not _MAIN_SYNTH_RE.search(text.split("\n---", 1)[0]):
        return False, None, ("`claim_check: synthesis` must sit in the FIRST `---` block, before any other line "
                             "starting with `---`, with no byte-order mark before the file's first `---` -- ignored, "
                             "every number is checked")
    # Duplicate keys: a YAML loader keeps the LAST, a reader may take any -- every one must agree with the escape.
    flags = [v for v in _fm_values(fm, "claim_check") if v != "synthesis"]
    if flags:
        return False, None, ("declares `claim_check: synthesis` but another `claim_check:` line in the SAME "
                             "frontmatter block reads %r -- ignored, every number is checked" % flags[0])
    reasons = _fm_values(fm, "claim_check_reason")
    reason = reasons[-1] if reasons else ""
    if not reasons or not all(reasons):
        return False, None, ("declares `claim_check: synthesis` but no non-empty `claim_check_reason:` on the same "
                             "line in the SAME frontmatter block (every `claim_check_reason:` line must be non-empty)"
                             " -- falling back to the normal rules")
    probes = [("filename", os.path.splitext(os.path.basename(doc_path))[0])]
    for key in ("title", "verdict"):
        probes.extend(("frontmatter %s:" % key, v) for v in _fm_values(fm, key))
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        h = _atx_text(ln)
        if h is not None:
            probes.append(("heading on line %d" % (i + 1), h))
        elif i > 0 and _SETEXT_ANY_RE.match(ln) and lines[i - 1].strip():
            probes.append(("setext heading on line %d" % i, lines[i - 1]))
    for hm in _html_headings(text):
        probes.append(("HTML heading", hm))
        probes.append(("HTML heading", html.unescape(_strip_tags_comments(hm))))
    fm_lines = text.count("\n", 0, m.end())            # the frontmatter itself parses as a setext heading: skipped
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.map and tok.map[0] >= fm_lines and i + 1 < len(tokens):
            parts = [c.content for c in tokens[i + 1].children or ()
                     if c.type in ("text", "text_special", "code_inline")]
            for joiner in ("", " "):
                probes.append(("heading on line %d as rendered" % (tok.map[0] + 1), joiner.join(parts)))
    for where, s in probes:
        w = _verdict_word(s)
        if w:
            return False, None, ("declares `claim_check: synthesis` but its %s carries a verdict word (%s) -- a "
                                 "verdict-bearing document is BARRED from the synthesis escape; every number is "
                                 "checked" % (where, w))
    return True, reason, None


# =================================================================================================================
# 3 + 4. matching and chance
# =================================================================================================================
def _readings(c):
    out = [(c.value, c.unit, "")]
    for scale, suf in c.alts:
        out.append((c.value * scale, c.unit * scale, "+" + suf))
    return out


def _any_within(pool, x, w):
    i = bisect.bisect_left(pool, x - w)
    return i < len(pool) and pool[i] <= x + w


def _slack(x):
    return 8.0 * math.ulp(x) if x else 8.0 * math.ulp(1e-300)


def _match_value(value, unit, alts, pool, tol=None, legacy=True):
    """`_match` for a bare (value, unit, alts) -- the decoy loop calls it ~1000 times per claim."""
    readings = [(value, unit, "")] + [(value * scale, unit * scale, "+" + suf) for scale, suf in alts]
    for x, u, lab in readings:
        s = _slack(x)
        if tol is not None:
            if _any_within(pool, x, tol + s):
                return "tolerance" + lab
            continue
        if _any_within(pool, x, s):
            return "exact" + lab
        if _any_within(pool, x, 0.5 * u * (1 + 1e-9) + s):
            return "rounding" + lab
    if tol is None and legacy:
        for x, u, lab in readings:
            if _any_within(pool, x, max(LEGACY_ABS_FLOOR, LEGACY_REL_TOL * abs(x)) + _slack(x)):
                return "legacy" + lab
    return None


def _match(c, pool, tol=None, legacy=True):
    """-> 'exact' | 'rounding' | 'legacy' | 'tolerance' (+ '+<suffix>' for a scaled reading), or None.
    exact/rounding form the PRECISION tier (|x - v| <= 0.5 * 10^-d); 'legacy' is main's relative window, tried
    only when `legacy` is set."""
    return _match_value(c.value, c.unit, c.alts, pool, tol, legacy)


def chance_max(decimals):
    """The per-claim chance limit for a number stated with `decimals` decimals (rule 4; see CALIBRATION)."""
    return CHANCE_MAX_BY_DECIMALS.get(decimals, CHANCE_MAX)


def _chance(c, pool, tol=None, tier="legacy"):
    """This claim's OWN chance-match rate, computed EXACTLY: the fraction of ALL 2 * CHANCE_WINDOW decoys of the same
    shape -- the written value stepped by k whole units of its last decimal, k = +-1 .. +-CHANCE_WINDOW -- that the
    matching rule accepts. It depends only on (value, stated precision, scale suffix, tier), never on how the number
    is spelled (`.417`, `0.417` and `00.417` rate the same; a seeded sample keyed by the TEXT moved the rate by up to
    0.06 between spellings). The rule is the TIER that accepted the claim: a claim matched at its stated precision is
    rated against the precision rule; a claim matched only by the legacy relative window is rated against
    precision-or-legacy (the wider window, the higher the rate)."""
    if not pool:
        return 0.0
    step = max(c.unit, abs(c.value) * 1e-12)
    legacy = tier == "legacy"
    hits = 0
    for k in range(1, CHANCE_WINDOW + 1):
        if _match_value(c.value + k * step, c.unit, c.alts, pool, tol, legacy):
            hits += 1
        if _match_value(c.value - k * step, c.unit, c.alts, pool, tol, legacy):
            hits += 1
    return hits / float(CHANCE_DECOYS)


def _tier(rule):
    if rule is None or rule.startswith("legacy"):
        return "legacy"
    return "precision"


def _hint(c, pool, tol, text=None):
    if c.reading == "reader":
        return "a reader sees this number but markup or an invisible character splits it in the source"
    if c.value < 0 and _match(c._replace(value=-c.value), pool, tol):
        return ("the artifact holds +%s: a dash directly before a number reads as a MINUS sign -- put a space after "
                "a punctuation dash" % c.text.lstrip("-"))
    if c.value > 0 and _match(c._replace(value=-c.value), pool, tol):
        if text and c.start and text[c.start - 1] in _DASH_CHARS:
            return ("the artifact holds -%s: the dash before it is read both as a minus sign and as punctuation "
                    "(glued to a word, or not a minus glyph) -- write the sign after a space: `x -%s` or `x \u2212%s`"
                    % (c.text, c.text, c.text))
        return "the artifact holds -%s: is the minus sign missing?" % c.text
    i = bisect.bisect_left(pool, c.value)
    near = [pool[j] for j in (i - 1, i) if 0 <= j < len(pool)]
    if near:
        v = min(near, key=lambda a: abs(a - c.value))
        if abs(v - c.value) <= 1.5 * c.unit:
            return "near miss: the artifact holds %r, which rounds to %.*f at the stated precision" % (
                v, max(c.decimals, 0), v)
    return ""


# =================================================================================================================
# warnings
# =================================================================================================================
def _line_warnings(lines):
    """Pre-round-5 scope idioms that are INERT -- non-blocking author-facing WARNINGs, never a verdict change."""
    warnings = []
    n = len(lines)
    for i, ln in enumerate(lines):
        if _STANDALONE_MARKER_RE.match(ln):
            warnings.append((i + 1, "standalone marker",
                             "a marker alone on a line exempts nothing (it opens no scope). Put the marker on the "
                             "SAME line -- in a table, in the SAME cell -- as each derived number."))
        if DERIVED_CLOSE in ln:
            warnings.append((i + 1, "close marker",
                             "<!--/derived--> closes nothing -- there are no ranges. Remove it, and put "
                             "<!--derived--> in the same cell as each derived number."))
        if _ATX_DERIVED_RE.match(ln):
            warnings.append((i + 1, "'Derived' heading",
                             "a '## Derived'-style heading opens no section. Put the marker in the same cell as "
                             "each derived number under it."))
        elif i + 1 < n and _SETEXT_TITLE_RE.match(ln) and _SETEXT_UNDERLINE_RE.match(lines[i + 1]):
            warnings.append((i + 1, "'Derived' heading (setext)",
                             "a setext 'Derived' heading opens no section. Put the marker in the same cell as "
                             "each derived number under it."))
    return warnings


# =================================================================================================================
# the scan
# =================================================================================================================
def _empty(unreadable):
    return dict(cited=[], nums=[], loaded=[], missing=[], ignored_citations=[], checked=0, checked_distinct=0,
                checked_visible_distinct=0, suppressed={"inline": 0, "synthesis": 0}, total_numeric=0,
                unsupported=[], too_broad=[], records=[], matched={}, chance=[], synthesis=False,
                low_coverage=False, marked_lines=[], warnings=[], unreadable=unreadable)


def _read(doc_path):
    """-> (text, error, had_byte_order_mark)."""
    try:
        raw = open(doc_path, "rb").read()
    except OSError as e:
        return None, "cannot read %s: %s: %s" % (doc_path, type(e).__name__, e), False
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        return None, ("%s is not valid UTF-8 (%s at byte offset %d) -- fix the file's encoding before it can be "
                      "checked" % (doc_path, e.reason, e.start)), False
    bom = text.startswith("\ufeff")
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    b = _BIDI_RE.search(text)
    if b:
        return None, ("%s contains a bidirectional control character (U+%04X, line %d) that can reorder digits on "
                      "screen -- remove it" % (doc_path, ord(b.group(0)), text.count("\n", 0, b.start()) + 1)), bom
    return text, None, bom


def _source_claims(text, line_of=lambda pos: 0):
    """The raw reading + both normalized copies of the SOURCE text (union, deduplicated, in source order). Every
    reading reads a number BOTH ways where main or round 5 would read its sign the other way, deciding their side on
    the ORIGINAL characters at the same offset (see _ambiguous_hyphen): a number that holds a non-ASCII digit is
    read only in the copies, where every dash is `-` and a filler is a space."""
    claims = []
    for reading, s in (("raw", text), ("normalized", _n_copy(text)), ("normalized", _n_copy(text, True))):
        for (a, b, v, d, u, alts, txt) in _extract(s, both_signs=True, orig=text):
            claims.append(Claim(a, b, line_of(a), v, d, u, alts, txt, reading))
    # Two readings of the SAME number end at the same offset with the same value, precision and scale suffix; a
    # hash on exactly that is linear (a scan of every earlier claim on the line was quadratic: 11,000 numbers on one
    # line took minutes). The suffix is part of the key: `0.1525k` followed by a non-ASCII digit is scaled in the raw
    # reading and bare in the normalized one, and dropping either would check it more loosely.
    seen, rn = set(), []
    for c in claims:
        key = (c.end, round(c.value, 12), c.decimals, c.alts)
        if key not in seen:
            seen.add(key)
            rn.append(c)
    rn.sort(key=lambda c: c.start)
    return rn


def _scan(doc_path, tol=None):
    """Pure computation, no printing -- shared by the CLI (`check`), tools/finding_lint.py and the retro script.
    `tol` (API only) replaces rule 3 with a fixed absolute tolerance."""
    if MarkdownIt is None:
        return _empty("markdown-it-py is not installed (%s) -- `pip install -r requirements-dev.txt`; without it no "
                      "<!--derived--> marker can be verified" % _MD_IMPORT_ERROR)
    text, err, bom = _read(doc_path)
    if err:
        return _empty(err)
    lines = text.split("\n")
    line_starts, off = [], 0
    for ln in lines:
        line_starts.append(off)
        off += len(ln) + 1

    def line_of(pos):
        return bisect.bisect_right(line_starts, pos) - 1

    warnings = _line_warnings(lines)

    # ---- markers -------------------------------------------------------------------------------------------------
    marks = _exact_marker_starts(text)                   # start offsets of every exact marker
    labeled, labels, prefix = _labeled(text, marks)
    md_cm, md_gfm = _parsers()
    try:
        toks_gfm = md_gfm.parse(labeled)
        toks_cm = md_cm.parse(labeled)
    except Exception as e:                               # fail CLOSED: an unparseable doc is not checked as clean
        return _empty("markdown-it could not parse %s (%s: %s)" % (doc_path, type(e).__name__, e))
    # A marker is live only if BOTH parsers read it as an HTML comment (GFM splits table cells before inline
    # parsing, CommonMark does not; where they disagree about a code span, the marker is dead -- fail closed).
    live = _live_labels(toks_gfm, prefix) & _live_labels(toks_cm, prefix)
    live_markers = []                                    # (line, col)
    exact_starts = set()
    for start, lab in zip(marks, labels):
        exact_starts.add(start)
        li = line_of(start)
        if lab in live:
            live_markers.append((li, start - line_starts[li]))
        else:
            warnings.append((li + 1, "marker not live",
                             "this <!--derived--> is %s, so it exempts NOTHING."
                             % (_where_label(toks_gfm, lab) if lab else "unverifiable (no free label character)")))
    for m in _MARKER_LIKE_RE.finditer(text):
        if m.start() not in exact_starts and not text.startswith(DERIVED_CLOSE, m.start()):
            li = line_of(m.start())
            snippet = text[m.start():m.start() + 40].split("\n")[0]
            warnings.append((li + 1, "not an exact marker",
                             "%r is not `<!--derived-->` or `<!--derived: note-->`, so it exempts NOTHING." % snippet))

    rn = _source_claims(text, line_of)
    hidden_lines, hidden_spans = _hidden(text, toks_gfm, line_starts)

    def is_hidden(pos):
        return line_of(pos) in hidden_lines or _in_spans(hidden_spans, pos)

    extra = _reader_claims(toks_gfm)

    # ---- exemption: live markers, per cell, capped ---------------------------------------------------------------
    def cells(li):
        ln = lines[li]
        return _cell_cuts(ln)

    cut_cache = {}

    def cell_of(li, col):
        if li not in cut_cache:
            cut_cache[li] = cells(li)
        return bisect.bisect_right(cut_cache[li], col)

    markers_in = Counter((li, cell_of(li, col)) for li, col in live_markers)
    by_cell = {}
    for c in rn:
        by_cell.setdefault((c.line, cell_of(c.line, c.start - line_starts[c.line])), []).append(c)
    exempt_ids = set()
    marked_lines = set()
    for key, k in markers_in.items():
        cl = by_cell.get(key, [])
        if not cl:
            if not _STANDALONE_MARKER_RE.match(lines[key[0]]):
                warnings.append((key[0] + 1, "marker exempts nothing",
                                 "this <!--derived--> shares its cell with no number, so it exempts NOTHING -- a "
                                 "marker exempts only numbers on its own line and, in a table, in its own cell (a "
                                 "marker alone in a row's last cell does not reach the row's other cells)."))
            continue
        for c in cl[:MAX_EXEMPT_PER_MARKER * k]:
            exempt_ids.add(id(c))
        marked_lines.add(key[0] + 1)
        if len(cl) > MAX_EXEMPT_PER_MARKER * k:
            warnings.append((key[0] + 1, "marker cap",
                             "%d marker(s) in this cell exempt at most %d numbers; the other %d are checked."
                             % (k, MAX_EXEMPT_PER_MARKER * k, len(cl) - MAX_EXEMPT_PER_MARKER * k)))

    # ---- synthesis + citations ----------------------------------------------------------------------------------
    synthesis, _reason, synth_warn = _synthesis_status(text, doc_path, toks_gfm, bom)
    if synth_warn:
        warnings.append((1, "synthesis escape not applied", synth_warn))

    cited, ignored = set(), set()
    for a, b in _path_spans(text):
        (ignored if is_hidden(a) else cited).add(text[a:b])
    ignored -= cited
    for p in sorted(ignored):
        warnings.append((0, "citation ignored",
                         "%s is cited only inside an HTML comment/block, a link-reference line or a hidden element "
                         "-- a reader cannot see it, so it is not loaded" % p))
    cited = sorted(cited)
    pool, _verdicts, loaded, missing = load_artifacts(cited)
    # A hidden citation adds NOTHING to the pool, but it is still opened: a MISSING or UNREADABLE file fails, as in
    # main and round 5 (which load every citation, hidden or not).
    for p in sorted(ignored):
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        hits = sorted(glob.glob(full)) if any(c in full for c in "*?[") else ([full] if os.path.exists(full) else [])
        if not hits:
            missing.append("%s (cited only inside hidden text)" % p)
        for h in hits:
            try:
                _load_one(h)
            except Exception as e:                       # narrow enough to see; never silent
                missing.append("%s (unreadable: %s; cited only inside hidden text)" % (p, type(e).__name__))

    # ---- check -----------------------------------------------------------------------------------------------------
    records, unsupported, too_broad, chances = [], [], [], []
    suppressed = {"inline": 0, "synthesis": 0}
    chance_cache = {}
    for c in rn + extra:
        rec = dict(line=c.line + 1, value=c.value, text=c.text, decimals=c.decimals, alts=c.alts, reading=c.reading,
                   status=None, rule=None, chance=None,
                   visible=(c.reading != "reader" and not is_hidden(c.start)))
        if c.reading != "reader" and id(c) in exempt_ids:
            rec["status"] = "exempt"
            suppressed["inline"] += 1
        elif synthesis:
            rec["status"] = "synthesis"
            suppressed["synthesis"] += 1
        else:
            rec["status"] = "checked"
            rule = _match(c, pool, tol)
            rec["rule"] = rule
            key = (round(c.value, 12), c.decimals, c.alts, _tier(rule))       # never the spelling (see _chance)
            if key not in chance_cache:
                chance_cache[key] = _chance(c, pool, tol, _tier(rule))
            ch = rec["chance"] = chance_cache[key]
            chances.append(ch)
            ctx = lines[c.line].strip()[:88] if c.line < len(lines) else ""
            if rule is None:
                rec["hint"] = _hint(c, pool, tol, text)
                unsupported.append((c.line + 1, c.value, ctx))
            elif ch > chance_max(c.decimals):
                rec["status"] = "too_broad"
                too_broad.append((c.line + 1, c.value, ch, ctx))
        records.append(rec)

    if synthesis and not cited:
        unsupported.append((0, 0.0, "synthesis doc cites NO artifact — the escape still requires citations"))

    checked_recs = [r for r in records if r["status"] in ("checked", "too_broad")]
    checked_distinct = {round(r["value"], 12) for r in checked_recs}
    visible_distinct = {round(r["value"], 12) for r in checked_recs if r["visible"]}
    total_numeric = len(records)
    low_coverage = (not synthesis and total_numeric >= LOW_COVERAGE_MIN_TOTAL
                    and (len(visible_distinct) / float(total_numeric)) < MIN_CHECK_FRACTION)
    matched = Counter((r["rule"] or "unmatched").split("+")[0] for r in checked_recs)
    return dict(cited=cited, nums=pool, loaded=loaded, missing=missing, ignored_citations=sorted(ignored),
                checked=len(checked_recs), checked_distinct=len(checked_distinct),
                checked_visible_distinct=len(visible_distinct), suppressed=suppressed, total_numeric=total_numeric,
                unsupported=unsupported, too_broad=too_broad, records=records, matched=dict(matched),
                chance=chances, synthesis=synthesis, low_coverage=low_coverage,
                marked_lines=sorted(marked_lines), warnings=sorted(warnings, key=lambda w: w[0]), unreadable=None)


def _verdict(r):
    """The single FAIL/PASS rule, shared by `check()`, `selftest()`, finding_lint and the test suite."""
    if r.get("unreadable") or r["missing"] or r["unsupported"] or r["too_broad"] or r["low_coverage"]:
        return "FAIL"
    return "PASS"


# =================================================================================================================
# author-facing text (the SAME wording is used by tools/githooks/pre-commit, tools/finding_lint.py and both
# SKILL.md copies -- tests/test_claim_check_line_only.py pins it)
# =================================================================================================================
MARKER_RULE = ("mark a derived/quoted value with <!--derived--> or <!--derived: note--> on the SAME physical line "
               "as the number -- in a table, in the SAME cell; at most %d numbers per marker; a marker in code, "
               "escaped, alone on a line, or spelled any other way exempts nothing" % MAX_EXEMPT_PER_MARKER)
def _limits_text():
    lim = dict(CHANCE_MAX_BY_DECIMALS)
    return ", ".join(["%d%% at %d decimals" % (round(100 * v), d) for d, v in sorted(lim.items())]
                     + ["%d%% otherwise" % round(100 * CHANCE_MAX)])


TOO_BROAD_MSG = ("matches only by chance: a random number of this shape would match the cited pool more often than "
                 "the limit for its precision (%s) -- cite a narrower artifact or state more decimals" % _limits_text())
FIX_HINT = ("fix the number, cite the artifact FILE that holds it (a path with a /), or " + MARKER_RULE)


def _chance_distribution(ch):
    if not ch:
        return "n/a (no checked number)"
    s = sorted(ch)

    def q(p):
        return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]
    edges = [(0.0, 0.01), (0.01, 0.05), (0.05, CHANCE_MAX), (CHANCE_MAX, 0.25), (0.25, 0.5), (0.5, 1.01)]
    hist = ", ".join("[%d-%d%%) %d" % (round(100 * a), min(100, round(100 * b)), sum(1 for x in s if a <= x < b))
                     for a, b in edges)
    return "p50 %.0f%%, p90 %.0f%%, max %.0f%% over %d claim(s); %s" % (100 * q(.5), 100 * q(.9), 100 * s[-1],
                                                                         len(s), hist)


def check(doc_path, tol=None, verbose=True):
    r = _scan(doc_path, tol)
    shown = os.path.relpath(doc_path, ROOT) if os.path.isabs(doc_path) else doc_path
    if r.get("unreadable"):
        if verbose:
            print("claim_check: %s" % shown)
            print("  ⛔ UNREADABLE: %s" % r["unreadable"])
            print("  => ⛔ UNREADABLE — the file cannot be checked until this is fixed")
        return 1
    fail = _verdict(r) == "FAIL"
    if verbose:
        m = r["matched"]
        print("claim_check: %s" % shown)
        print("  cited artifacts : %d found, %d missing, %d ignored (hidden)"
              % (len(r["loaded"]), len(r["missing"]), len(r["ignored_citations"])))
        for mp in r["missing"][:5]:
            print("      ⛔ MISSING  %s" % mp)
        print("  measurements    : %d checked (%d distinct, %d distinct outside hidden regions) against %d artifact "
              "values%s" % (r["checked"], r["checked_distinct"], r["checked_visible_distinct"], len(r["nums"]),
                            "   [synthesis: numbers not checked, citations still required]" if r["synthesis"]
                            else ""))
        print("  matched by rule : %d exact, %d rounding at the stated precision, %d legacy relative tolerance%s, "
              "%d unmatched" % (m.get("exact", 0), m.get("rounding", 0), m.get("legacy", 0),
                                (", %d fixed tolerance" % m["tolerance"]) if m.get("tolerance") else "",
                                m.get("unmatched", 0)))
        print("  exempted        : %d by a live <!--derived--> in the same cell (line(s) %s), %d by synthesis, of %d "
              "numeric claim(s)" % (r["suppressed"]["inline"], ", ".join(str(n) for n in r["marked_lines"]) or "-",
                                    r["suppressed"]["synthesis"], r["total_numeric"]))
        print("  chance match    : %s; limit per claim %s" % (_chance_distribution(r["chance"]), _limits_text()))
        for lineno, kind, msg in r["warnings"]:
            print("      ⚠  WARNING line %-4d %-24s %s" % (lineno, kind, msg))
        hints = {(x["line"], x["value"]): x.get("hint", "") for x in r["records"] if x["status"] == "checked"}
        written = {(x["line"], x["value"]): x["text"] for x in r["records"]}   # as WRITTEN, never re-rounded
        for lineno, val, ctx in r["unsupported"][:12]:
            h = hints.get((lineno, val), "")
            print("      ⛔ line %-4d %-14s not in any cited artifact | %s%s"
                  % (lineno, written.get((lineno, val), repr(val)), ctx, ("\n           -> " + h) if h else ""))
        if len(r["unsupported"]) > 12:
            print("      ... and %d more" % (len(r["unsupported"]) - 12))
        dec = {(x["line"], x["value"]): x["decimals"] for x in r["records"]}
        for lineno, val, ch, ctx in r["too_broad"][:12]:
            print("      ⛔ line %-4d %-14s chance %.0f%% > %.0f%%: %s | %s"
                  % (lineno, written.get((lineno, val), repr(val)), 100 * ch,
                     100 * chance_max(dec.get((lineno, val), 99)), TOO_BROAD_MSG, ctx))
        if len(r["too_broad"]) > 12:
            print("      ... and %d more too broad" % (len(r["too_broad"]) - 12))
        if r["low_coverage"]:
            print("      ⛔ LOW COVERAGE: only %d/%d (%.0f%%) DISTINCT numeric value(s) seen outside hidden regions "
                  "were actually checked. A doc this size should not be almost entirely derived; mark the specific "
                  "derived numbers, not the whole document."
                  % (r["checked_visible_distinct"], r["total_numeric"],
                     100.0 * r["checked_visible_distinct"] / r["total_numeric"] if r["total_numeric"] else 0.0))
        if not r["cited"]:
            print("  ⚠  NO ARTIFACT CITED — a findings doc with no artifact path cannot be checked at all.")
        print("  => %s" % (("⛔ UNSUPPORTED CLAIMS, missing artifacts, or matches only by chance — " + FIX_HINT)
                           if fail else "✔ every measurement traces to a cited artifact"))
    return 1 if fail else 0


# =================================================================================================================
# SELFTEST REGISTRY -- see tools/claim_check_cases.py (kept in a separate module so this file stays readable).
# `tests/test_claim_check_line_only.py` re-runs every case against the historical revisions in _HISTORY_SHAS,
# loaded from git, and asserts the recorded `wrong_on` equals the set that ACTUALLY gets it wrong.
# =================================================================================================================
_HISTORY_SHAS = {"main": "7e2edc08e", "r1": "d4959ecb0", "r2": "6abb28469", "r3": "662e167e8", "r4": "214e509bf",
                 "r5": "4fda849d4", "r6": "f2b7db2b4", "r7": "4ff05b018", "r8a": "654d95664",
                 "r8b": "57b1e5f01"}
_DEFAULT_ARTIFACT = {"accuracy": 0.17, "baseline": 0.1625}
WRONG_VALUES = {0.1525, 0.14, 1.23456, -0.1525, -0.1625, 0.153, 0.15925, 10.1525}


def _cases():
    """The selftest registry lives in tools/claim_check_cases.py (data only), loaded by FILE PATH next to this
    module so it works whether this file runs as a script (pre-commit) or is imported as tools.claim_check."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claim_check_cases.py")
    spec = importlib.util.spec_from_file_location("_claim_check_cases", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SELFTEST_CASES


def __getattr__(name):                                  # SELFTEST_CASES is loaded lazily (PEP 562)
    if name == "SELFTEST_CASES":
        return _cases()
    raise AttributeError(name)


def _write_case(d, case):
    """Write one selftest case (and its artifact) under directory `d`, which must be inside ROOT: a finding's
    ROOT-relative citation is the only form that resolves the same way for every revision."""
    art_obj = case.get("artifact", _DEFAULT_ARTIFACT)
    art_name = "art_%s.json" % hashlib.sha1(json.dumps(art_obj, sort_keys=True).encode()).hexdigest()[:12]
    art_abs = os.path.join(d, art_name)
    if not os.path.exists(art_abs):
        with open(art_abs, "w") as fh:
            json.dump(art_obj, fh)
    art = os.path.relpath(art_abs, ROOT).replace(os.sep, "/")
    sub = os.path.join(d, case["name"])
    os.makedirs(sub, exist_ok=True)
    for fname, raw in case.get("raw_files", {}).items():   # verbatim files (e.g. a corrupt artifact) next to the doc
        with open(os.path.join(sub, fname), "w", encoding="utf-8") as fh:
            fh.write(raw)
    path = os.path.join(sub, case.get("filename", "doc.md"))
    subst = {"art": art, "sub": os.path.relpath(sub, ROOT).replace(os.sep, "/")}
    body = case["doc"] % subst if ("%(art)s" in case["doc"] or "%(sub)s" in case["doc"]) else case["doc"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def _case_problems(case, r, printed=None):
    """Why one case's scan result disagrees with the case -- [] when it agrees."""
    got = _verdict(r)
    if got != case["expect"]:
        return ["case %s expected %s, got %s (%s)" % (case["name"], case["expect"], got, case["why"])]
    flagged = {round(v, 6) for _l, v, _c in r["unsupported"]} | {round(v, 6) for _l, v, _ch, _c in r["too_broad"]}
    out = []
    reason = case.get("expect_reason")
    if case["expect"] == "FAIL":
        if reason == "low_coverage" and not r["low_coverage"]:
            out.append("case %s failed, but not on LOW COVERAGE" % case["name"])
        elif reason == "too_broad" and not r["too_broad"]:
            out.append("case %s failed, but not as too broad" % case["name"])
        elif reason == "unreadable" and not r.get("unreadable"):
            out.append("case %s failed, but was not refused as unreadable" % case["name"])
        elif reason == "missing" and not r["missing"]:
            out.append("case %s failed, but not on a missing/unreadable artifact" % case["name"])
        elif reason is None and not case.get("must_flag") and not (flagged & WRONG_VALUES):
            out.append("case %s failed but never flagged a designated wrong number (%s): flagged=%s"
                       % (case["name"], sorted(WRONG_VALUES), sorted(flagged)))
        for v in case.get("must_flag", ()):
            if round(v, 6) not in flagged:
                out.append("case %s did not flag %r: flagged=%s" % (case["name"], v, sorted(flagged)))
    elif flagged:
        out.append("case %s is supposed to PASS clean but flagged %s" % (case["name"], sorted(flagged)))
    for w in case.get("expect_warning", ()):
        if not any(w in (kind + " " + msg) for _l, kind, msg in r["warnings"]):
            out.append("case %s: no WARNING containing %r (got %s)" % (case["name"], w, r["warnings"]))
    if printed is not None:
        for s in case.get("expect_output", ()):
            if s not in printed:
                out.append("case %s: check() output lacks %r" % (case["name"], s))
    return out


def selftest():
    """Same contract as `tools/gates/*.selftest()`: a list of problems, empty means the check is trustworthy.
    Every case runs in its own direction: a FAIL case must flag its designated wrong number (or trip its expected
    reason), a PASS case must flag nothing."""
    import tempfile
    problems = []
    try:
        cases = _cases()
    except Exception as e:
        return ["SELFTEST BROKEN: cannot load tools/claim_check_cases.py: %s: %s" % (type(e).__name__, e)]
    if MarkdownIt is None:
        return ["SELFTEST BROKEN: markdown-it-py is not installed (%s)" % _MD_IMPORT_ERROR]
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".claim_check_selftest_") as d:
        for case in cases:
            p = _write_case(d, case)
            r = _scan(p)
            printed = None
            if case.get("expect_output"):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    check(p, verbose=True)
                printed = buf.getvalue()
            problems.extend("SELFTEST BROKEN: " + x for x in _case_problems(case, r, printed))
    return problems


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        problems = selftest()
        for p in problems:
            print("⛔", p)
        print("claim_check selftest: %s" % ("FAILED" if problems else
                                             "OK (%d cases, both directions)" % len(_cases())))
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
