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
     or `%` precedes it (then it is a range or a subtraction); an ASCII hyphen after a word character or after
     `)`, `]`, `%` is AMBIGUOUS (main and round 5 decide those the other way) and is read BOTH ways, each a claim.
     Numbers are ALSO extracted from a lightly normalized COPY in which every character maps to exactly one
     character -- dash/minus variants to `-`, zero-width/format/combining/filler characters to a SPACE, dot-like
     characters between digits to `.`, any Unicode decimal digit to its ASCII digit -- so no reading can glue or
     drop anything. A third, ADDITIVE reading is the text a reader SEES (markdown-it's tokens: emphasis, tags,
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
  4. DISCRIMINATING POWER, PER CLAIM. For every checked number, CHANCE_DECOYS (100) seeded decoys x + k * 10^-d
     (k drawn without replacement from +-[1, CHANCE_WINDOW]) are matched with rule 3; the fraction that match is
     that CLAIM's own chance-match rate. A number that matches, but whose own rate exceeds CHANCE_MAX, is NOT
     supported: it fails with "cite a narrower artifact or state more decimals". (Round 7 averaged the rate over
     the document, so one wrong coarse headline among many precise numbers passed.) The distribution is printed.
  5. KEPT FROM ROUND 5 / REQUIRED: WARNINGs for the inert scope idioms; strict UTF-8 and no bidirectional controls
     (UNREADABLE blocks); `claim_check: synthesis` applies only inside a CLOSED frontmatter block with a non-empty
     `claim_check_reason:`, only where main and round 5 also read the flag (before the first `\n---` of a file
     that starts with `---`: no byte-order mark, no quoted value), and never when the filename, the frontmatter
     `title:`/`verdict:` (any key case, continuation lines included), or any heading -- as written, as rendered
     (`G**O**`, `G<!-- -->O`, `&#71;O`), or an HTML `<h1>`-`<h6>` -- carries a verdict word (GO(s), NO-GO, NOGO,
     PASS(ED/ES), FAIL(ED/S), REFUTED, CONFIRMED; case-insensitive, invisible characters removed); the
     LOW_COVERAGE floor on DISTINCT checked values seen outside HTML comments, link-reference lines and hidden
     elements; a citation inside one of those is ignored (with a WARNING); `tools/gates/claim_check_selftest.py`
     (class CCT) passes this file's selftest problems through verbatim.

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

CALIBRATION (2026-09-25; re-derive with `tools/claim_check_retro_compare.py --since 2026-09-01 --calibrate`;
outputs committed as research/coordination/claimcheck_r8_retro_since2026-09-01_2026-09-25.{tsv,txt}):
  * CHANCE_MAX = 0.20. The per-claim rate over the 3,821 precision-tier matches in the 353 findings added since
    2026-09-01: p50 0.02, p90 0.16, p95 0.25, p99 0.62. Docs that would fail on breadth ALONE at T = 0.05 / 0.10 /
    0.15 / 0.20 / 0.25 / 0.30: 45 / 32 / 24 / 15 / 11 / 7. At 0.20 a wrong number of a claim's own shape is accepted
    at most 1 time in 5; the 15 docs it fails on breadth alone (13 on numbers correct at their written precision,
    2 on legacy-only matches) are each fixed by one more stated decimal or a narrower citation. The 19 legacy-only
    matches have median rate 0.53: the relative window survives only against sparse pools.
  * CHANCE_WINDOW = 500 units of the claim's last decimal either side, CHANCE_DECOYS = 100 drawn without
    replacement, seeded by the claim's own text (sampling error ~0.04 at the 0.20 bar).
  * The rate is taken in the TIER that matched: a claim matched at its stated precision is rated against the
    precision rule, a legacy-only match against precision-or-legacy. (Rating every claim against the union made a
    6-decimal value that EXACTLY matches the artifact read as "too broad" beside a dense sweep -- the relative
    window, not the claim, was broad.)
  * LOW_COVERAGE_MIN_TOTAL = 30: the largest non-synthesis doc since 2026-09-01 under 5% distinct-visible-checked
    has 27 numeric claims.
  * FALSE POSITIVES on the same 353 findings (none re-gated: the gate checks only NEWLY ADDED findings): round 8
    fails 122 (main 31, r5 135, r6 155, r7 108). Of its 911 flagged numbers, 274 (30%) are correct at their
    written precision and fail only as too broad (r5: 931 of its 1,381 flags, 67%, were correct roundings);
    13 findings fail ONLY on such numbers. The rest by cause: 378 unmarked prose numbers (derived, aggregated,
    quoted, or wrong), 75 identifiers, 73 near misses (a truncation or a wrong rounding), 55 with nothing
    loaded, 28 in code spans, 12 read as a minus sign, 2 in comments, 1 in a fence. The one finding main fails
    and round 8 passes writes 0.031 for a cited 0.0307 (a correct rounding; its own chance rate 11%).

ACCEPTED TRADE vs main and round 5 (required by the round-8 spec): rule 3's precision window 0.5 * 10^-d is WIDER
than main's relative window for a coarse small number (0.477 matches a stored 0.4772; main's window is 4.8e-5), so
a wrong coarse number that lands in the window of an unrelated cited value passes where main fails it -- bounded
per claim by rule 4: such a match is accepted only when a random number of the claim's own shape would match less
than CHANCE_MAX of the time.
CANNOT CATCH (known): a number spelled in words; a decimal comma; homoglyph letters for digits; digit-group
separators (`0.152 5`); an integer mantissa with an exponent (`1525e-4`, as in main and r5); a wrong number within
the matching window of an unrelated cited value whose own chance rate is under CHANCE_MAX; a wrong number within
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
import random
import re
import sys
import unicodedata
from collections import Counter, namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:                                               # pinned in requirements-dev.txt: markdown-it-py>=4.2,<5
    from markdown_it import MarkdownIt
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
# unless a letter, digit, dot or `_` precedes it (`p.347` is a page, `x.125` a field).
_NUM_RE = re.compile(r"(?:(?<![0-9.])([0-9]+)|(?<![A-Za-z0-9._]))\.([0-9]+)"
                     r"(?:[eE]([+" + _SIGN_CLASS + r"]?[0-9]+))?(?![0-9])")
NUM_RE = _NUM_RE                                   # public alias (tests use it to list a text's numbers)
_MAGNITUDE = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}
_RANGE_LEFT = frozenset("0123456789.)]%")          # a dash after one of these is a range/subtraction, not a sign
MIN_DECIMALS = 3                                   # >= 3 stated decimals => a measurement, not prose

# Globs are allowed: a finding over N seeds cites one pattern, not N paths. Must contain a "/".
PATH_RE = re.compile(r"([\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.(?:jsonl|json))")
VERDICT_RE = re.compile(r"\b(GO|NO-GO|PASS|FAIL|REFUTED|CONFIRMED)\b")

# =================================================================================================================
# markers
# =================================================================================================================
DERIVED_MARK = "<!--derived-->"
DERIVED_CLOSE = "<!--/derived-->"                  # closes nothing -- matched only to print a WARNING
# The ONLY two spellings that can exempt anything. One physical line; the note may not contain `-->`.
_EXACT_MARKER_RE = re.compile(r"<!--derived(?:-->|:[ \t]+[^\n]*?\S[^\n]*?-->)")
_MARKER_LIKE_RE = re.compile(r"<!--[ \t]*derived", re.I)   # anything an author may have MEANT as a marker
_STANDALONE_MARKER_RE = re.compile(r"^\s*(?:>\s*)*<!--\s*derived\b[^\n]*?-->\s*$", re.I)
_ATX_DERIVED_RE = re.compile(r"^\s*(?:>\s*)*#{1,6}\s*[*_`]*\s*derived\b", re.I)
_SETEXT_TITLE_RE = re.compile(r"^\s*(?:>\s*)*[*_`]*\s*derived\b", re.I)
_SETEXT_UNDERLINE_RE = re.compile(r"^\s*(?:=+|-+)\s*$")
MAX_EXEMPT_PER_MARKER = 8
MAX_EXEMPT_PER_LINE = MAX_EXEMPT_PER_MARKER        # name kept for callers of earlier rounds
# Cell cuts: every `|` (escaped or not, table row or not -- a cut only NARROWS an exemption), `<br>` and
# block-level tags.
_CELL_CUT_RE = re.compile(r"\||<\s*/?\s*(?:br|p|div|li|tr|td|th|table|thead|tbody|tfoot|ul|ol|dl|dt|dd|h[1-6]|hr|"
                          r"blockquote|pre|section|article|header|footer|details|summary|caption|figure|"
                          r"figcaption)\b[^>\n]*>", re.I)

# =================================================================================================================
# matching and discriminating power
# =================================================================================================================
LEGACY_REL_TOL = 1e-4
LEGACY_ABS_FLOOR = 5e-6
CHANCE_DECOYS = 100
CHANCE_WINDOW = 500
CHANCE_SEED = 20260925
# CALIBRATED 2026-09-25 -- see the docstring CALIBRATION (tools/claim_check_retro_compare.py --calibrate).
CHANCE_MAX = 0.20

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
_COMMENT_RE = re.compile(r"<!---?>|<!--(?:[^-]|-[^-]|--[^>])*-->")      # markdown-it's own comment pattern
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
def _n_copy(text):
    """The lightly normalized copy: every character maps to EXACTLY ONE character, so positions are preserved and
    nothing can be glued or dropped."""
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
            left = text[i - 1] if i > 0 else ""
            right = text[i + 1] if i + 1 < n else ""
            if (left.isdigit() or left == "") and right.isdigit():
                out[i] = "."
    return "".join(out)


_WORDCHAR_RE = re.compile(r"\w")


def _ambiguous_hyphen(s, a):
    """True when an ASCII hyphen-minus directly before the digits at `a` is read as a sign by one of main/round 5
    (sign unless a word character or `.` precedes it) and round 8 (sign unless a digit, `.`, `)`, `]` or `%` precedes
    it) but not the other: after a letter or `_` (`acc-0.1525`) or after `)`, `]`, `%` (`(a)-0.1525`)."""
    if a < 1 or s[a - 1] != "-":
        return False
    prev = s[a - 2] if a >= 2 else " "
    return prev in ")]%" or (bool(_WORDCHAR_RE.match(prev)) and prev not in "0123456789")


def _extract(s, both_signs=False):
    """Every measurement-shaped number in string `s` -> list of (start, end, value, decimals, unit, alts, text).
    `start` covers a sign when one is read. With `both_signs` (the RAW reading), a number behind an ambiguous ASCII
    hyphen is read BOTH signed and unsigned -- each reading is a claim of its own, so a sign error main or round 5
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
        neg, sa = False, a
        if a > 0 and s[a - 1] in _DASH_CHARS:
            prev = s[a - 2] if a >= 2 else " "
            if prev not in _RANGE_LEFT:
                neg, sa = True, a - 1
        alts = ()
        if b < len(s) and s[b] in _MAGNITUDE and (b + 1 >= len(s) or not (s[b + 1].isascii() and s[b + 1].isalnum())):
            alts = ((_MAGNITUDE[s[b]], s[b]),)
        suffix = alts[0][1] if alts else ""
        out.append((sa, b, -mag if neg else mag, d, 10.0 ** (-d), alts, ("-" if neg else "") + s[a:b] + suffix))
        if both_signs and _ambiguous_hyphen(s, a):
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
    while i < n:
        c = content[i]
        if c == "<":
            m = _COMMENT_RE.match(content, i) or _TAG_RE.match(content, i)
            if m:
                tag = m.group(0)
                end = m.end()
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
            s = _n_copy(raw_s)
            src = tok.content or ""
            for (sa, b, v, d, u, alts, txt) in _extract(s):
                if _same_claim_as_source(keep, s, raw_s, src, sa, b, v, d):
                    continue
                line = l0 + keep[min(sa, len(keep) - 1)][1]
                key = (line, round(v, 12), d)
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
    return _PARSERS["cm"], _PARSERS["gfm"]


def _label_prefix(text):
    for c in "\u2603\u2604\u2605\u2606\u2622\u2623\u262f":
        if c not in text:
            return c
    return None


def _labeled(text, marks):
    """Replace the 7 letters `derived` of each exact marker with a unique 7-character label (same length, same
    markdown meaning), so the parse can say which occurrence became an HTML comment. -> (text, labels, prefix)."""
    pre = _label_prefix(text)
    if pre is None:
        return text, [None] * len(marks), None       # cannot label: no marker is live (fails closed)
    buf = list(text)
    labels = []
    for i, m in enumerate(marks):
        lab = pre + "%06d" % i
        a = m.start() + 4
        buf[a:a + 7] = list(lab)
        labels.append(lab)
    return "".join(buf), labels, pre


def _live_labels(tokens, prefix):
    """Labels whose marker markdown-it parsed as an HTML comment: an html_inline token that IS the marker, or an
    html_block that STARTS with it. Nothing inside code, escapes, image alt text, or elsewhere inside other HTML
    counts."""
    live = set()
    if prefix is None:
        return live
    rx = re.compile(r"<!--(" + re.escape(prefix) + r"[0-9]{6})(?:-->|:[ \t]+[^\n]*?\S[^\n]*?-->)")
    for tok in tokens:
        if tok.type == "html_block":
            m = rx.match(tok.content.lstrip(" "))
            if m:
                live.add(m.group(1))
        elif tok.type == "inline":
            for ch in tok.children or ():
                if ch.type == "html_inline":
                    m = rx.fullmatch(ch.content)
                    if m:
                        live.add(m.group(1))
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
    spans = [(m.start(), m.end()) for m in _COMMENT_RE.finditer(text)]
    for m in _HIDDEN_OPEN_RE.finditer(text):
        tag = (m.group(1) or m.group(2)).lower()
        close = re.compile(r"</\s*%s\s*>" % re.escape(tag), re.I).search(text, m.end())
        spans.append((m.start(), close.end() if close else len(text)))
    return hidden_lines, _merge(spans)


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
def _fm_value(fm, key):
    # The key is matched case-insensitively and quoted or not (`Title:`, `"title":`) -- barring is the fail-closed
    # direction, so every spelling a reader would take for the title is read.
    m = re.search(r"^[\"']?%s[\"']?[ \t]*:[ \t]*(.*?)[ \t]*$" % re.escape(key), fm, re.M | re.I)
    if not m:
        return ""
    v = m.group(1)
    # Every indented or blank line after the key continues its value -- a block scalar (`|`, `>`, blank lines
    # included) or a multi-line plain or quoted scalar (`title: 'Lane A` / `  GO'`) -- so a verdict word on a
    # continuation line is still read.
    block = []
    for ln in fm[m.end():].split("\n")[1:]:
        if ln.startswith((" ", "\t")) or not ln.strip():
            block.append(ln.strip())
        else:
            break
    if v in ("|", ">", "|-", ">-", "|+", ">+"):
        v = ""
    v = " ".join(x for x in [v] + block if x)
    v = v.strip().strip("\"'").strip()
    return "" if v in ("~", "null", "Null", "NULL") else v


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


_TAG_OR_COMMENT_RE = re.compile(r"<!--.*?-->|<[^<>]*>", re.S)


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
    reason = _fm_value(fm, "claim_check_reason")
    if not reason:
        return False, None, ("declares `claim_check: synthesis` but no non-empty `claim_check_reason:` on the same "
                             "line in the SAME frontmatter block -- falling back to the normal rules")
    probes = [("filename", os.path.splitext(os.path.basename(doc_path))[0]),
              ("frontmatter title:", _fm_value(fm, "title")),
              ("frontmatter verdict:", _fm_value(fm, "verdict"))]
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        h = _ATX_ANY_RE.match(ln)
        if h:
            probes.append(("heading on line %d" % (i + 1), h.group(2) or ""))
        elif i > 0 and _SETEXT_ANY_RE.match(ln) and lines[i - 1].strip():
            probes.append(("setext heading on line %d" % i, lines[i - 1]))
    for hm in _HTML_HEADING_RE.finditer(text):
        probes.append(("HTML heading", hm.group(1)))
        probes.append(("HTML heading", html.unescape(_TAG_OR_COMMENT_RE.sub("", hm.group(1)))))
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


def _match(c, pool, tol=None, legacy=True):
    """-> 'exact' | 'rounding' | 'legacy' | 'tolerance' (+ '+<suffix>' for a scaled reading), or None.
    exact/rounding form the PRECISION tier (|x - v| <= 0.5 * 10^-d); 'legacy' is main's relative window, tried
    only when `legacy` is set."""
    for x, u, lab in _readings(c):
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
        for x, u, lab in _readings(c):
            if _any_within(pool, x, max(LEGACY_ABS_FLOOR, LEGACY_REL_TOL * abs(x)) + _slack(x)):
                return "legacy" + lab
    return None


def _chance(c, pool, tol=None, tier="legacy"):
    """This claim's OWN chance-match rate: the fraction of CHANCE_DECOYS seeded decoys of the same shape (same
    stated precision, stepped by k whole units of its last decimal from the written value, k drawn without
    replacement from +-[1, CHANCE_WINDOW]) that the matching rule accepts. The rule is the TIER that accepted the
    claim: a claim matched at its stated precision is rated against the precision rule; a claim matched only by
    the legacy relative window is rated against precision-or-legacy (the wider window, the higher the rate)."""
    if not pool:
        return 0.0
    h = hashlib.sha256(("%s|%d|%d" % (c.text, c.decimals, CHANCE_SEED)).encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    ks = rng.sample(range(1, 2 * CHANCE_WINDOW + 1), CHANCE_DECOYS)
    step = max(c.unit, abs(c.value) * 1e-12)
    hits = 0
    for k in ks:
        k = k if k <= CHANCE_WINDOW else CHANCE_WINDOW - k      # 1..W and -1..-W
        if _match(c._replace(value=c.value + k * step), pool, tol, legacy=(tier == "legacy")):
            hits += 1
    return hits / float(CHANCE_DECOYS)


def _tier(rule):
    if rule is None or rule.startswith("legacy"):
        return "legacy"
    return "precision"


def _hint(c, pool, tol):
    if c.reading == "reader":
        return "a reader sees this number but markup or an invisible character splits it in the source"
    if c.value < 0 and _match(c._replace(value=-c.value), pool, tol):
        return ("the artifact holds +%s: a dash directly before a number reads as a MINUS sign -- put a space after "
                "a punctuation dash" % c.text.lstrip("-"))
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
    marks = list(_EXACT_MARKER_RE.finditer(text))
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
    for m, lab in zip(marks, labels):
        exact_starts.add(m.start())
        li = line_of(m.start())
        if lab in live:
            live_markers.append((li, m.start() - line_starts[li]))
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

    # ---- claims: raw reading + normalized copy (union, deduplicated) ---------------------------------------------
    claims = []
    for reading, s in (("raw", text), ("normalized", _n_copy(text))):
        for (a, b, v, d, u, alts, txt) in _extract(s, both_signs=(reading == "raw")):
            claims.append(Claim(a, b, line_of(a), v, d, u, alts, txt, reading))
    seen, rn = {}, []
    for c in claims:
        dup = False
        for o in seen.get(c.line, ()):
            if o.start < c.end and c.start < o.end and abs(o.value - c.value) <= 1e-12 * max(1.0, abs(c.value)) \
                    and o.decimals == c.decimals:
                dup = True
                break
        if not dup:
            seen.setdefault(c.line, []).append(c)
            rn.append(c)
    rn.sort(key=lambda c: c.start)
    hidden_lines, hidden_spans = _hidden(text, toks_gfm, line_starts)

    def is_hidden(pos):
        return line_of(pos) in hidden_lines or _in_spans(hidden_spans, pos)

    extra = _reader_claims(toks_gfm)

    # ---- exemption: live markers, per cell, capped ---------------------------------------------------------------
    def cells(li):
        ln = lines[li]
        return [m.start() for m in _CELL_CUT_RE.finditer(ln)]

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
    for m in PATH_RE.finditer(text):
        (ignored if is_hidden(m.start()) else cited).add(m.group(1))
    ignored -= cited
    for p in sorted(ignored):
        warnings.append((0, "citation ignored",
                         "%s is cited only inside an HTML comment/block, a link-reference line or a hidden element "
                         "-- a reader cannot see it, so it is not loaded" % p))
    cited = sorted(cited)
    pool, _verdicts, loaded, missing = load_artifacts(cited)
    for p in sorted(ignored):          # never loaded, but a hidden citation of a MISSING file still fails (as in r5)
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not (glob.glob(full) if any(c in full for c in "*?[") else os.path.exists(full)):
            missing.append("%s (cited only inside hidden text)" % p)

    # ---- check -----------------------------------------------------------------------------------------------------
    records, unsupported, too_broad, chances = [], [], [], []
    suppressed = {"inline": 0, "synthesis": 0}
    chance_cache = {}
    for c in rn + extra:
        rec = dict(line=c.line + 1, value=c.value, text=c.text, decimals=c.decimals, reading=c.reading,
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
            key = (c.text, c.decimals, _tier(rule))
            if key not in chance_cache:
                chance_cache[key] = _chance(c, pool, tol, _tier(rule))
            ch = rec["chance"] = chance_cache[key]
            chances.append(ch)
            ctx = lines[c.line].strip()[:88] if c.line < len(lines) else ""
            if rule is None:
                rec["hint"] = _hint(c, pool, tol)
                unsupported.append((c.line + 1, c.value, ctx))
            elif ch > CHANCE_MAX:
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
TOO_BROAD_MSG = ("matches only by chance: a random number of this shape would match the cited pool more than %d%% of "
                 "the time -- cite a narrower artifact or state more decimals" % round(100 * CHANCE_MAX))
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
        print("  chance match    : %s; limit %.0f%% per claim" % (_chance_distribution(r["chance"]),
                                                                  100 * CHANCE_MAX))
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
        for lineno, val, ch, ctx in r["too_broad"][:12]:
            print("      ⛔ line %-4d %-14s chance %.0f%%: %s | %s"
                  % (lineno, written.get((lineno, val), repr(val)), 100 * ch, TOO_BROAD_MSG, ctx))
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
                 "r5": "4fda849d4", "r6": "f2b7db2b4", "r7": "4ff05b018"}
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
    path = os.path.join(sub, case.get("filename", "doc.md"))
    body = case["doc"] % {"art": art} if "%(art)s" in case["doc"] else case["doc"]
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
