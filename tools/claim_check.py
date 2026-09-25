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

Exit 1 if a measurement-shaped number is unsupported by the cited artifacts, if the cited artifacts are too broad
for a match to mean anything, or if a substantial document checks (almost) none of its numbers.

WHAT IS CHECKED: numbers with >= 3 decimal places (measurements, not prose: "3 seeds", "97%" are ignored).
A derived value (ratio, difference, mean over seeds, a figure quoted from another finding) legitimately lives in no
artifact, so a `<!--derived-->` marker exempts it -- but ONLY a number in the SAME TABLE CELL, or the same
`<br>`-separated segment of a line, as the marker, and at most MAX_EXEMPT_PER_LINE (8) numbers per cell/segment.
There is no other scope: a marker alone on a line, a `## Derived` heading, a `<!--/derived-->` close marker, and a
marker alone in a table row's last cell all exempt NOTHING, and `check()` prints a WARNING wherever it sees one.
A comment whose text STARTS with `derived` (`<!--derived: 0.499 = 30.240 - 29.741-->`) is the same marker with a
note attached; the note itself is never scanned.

HISTORY -- every round, and the hole each one left. Main (7e2edc08e) let a marker ALONE on its own line open a
scope lasting until the next `## ` heading; a scorer used that to hide three whole sections (0 of 336 artifact
values checked, two wrong numbers passed). Rounds 1-4 each tried a better multi-line scope rule and each leaked:
  r1 (d4959ecb0, line-scanner): a `###` heading did not end a scope; a table followed by a wrong number with no
     blank line absorbed it; text AFTER a mid-line close marker was swallowed too.
  r2 (6abb28469): a `# derived` comment inside a code fence was read as a heading; a list's scope covered only its
     first bullet; a fence ended an open list early.
  r3 (662e167e8): the fence toggle desynced on a mismatched fence (```` vs ``` vs ~~~), swallowing a `## Results`.
  r4 (214e509bf, a real CommonMark parser, REVIEWED UNSOUND): an unclosed HTML comment or `<pre>` hid a heading; a
     late inline close marker hijacked an early opener; h1/setext/blockquoted "Derived" headings had no
     container-aware end; a table in a list/blockquote resolved against the wrong container.
  r5 (4fda849d4): deleted scope entirely -- exempt iff the marker is on the number's own PHYSICAL LINE. The review
     found that boundary SOUND, but everything else a document can do to a NUMBER untested.
  r6 (f2b7db2b4): strict UTF-8; per-cell/per-`<br>` exemption capped at 8; synthesis needs closed frontmatter + a
     `claim_check_reason:` + no verdict title; normalization of emphasis/units/exponents/escapes/entities/Cf
     characters/two dash glyphs; distinct-value coverage; comment citations ignored; glob and value-pool caps.
     REVIEWED SOUND-WITH-ISSUES: 11 issues (research/coordination/claimcheck_r7_review_issues.txt), two of which
     decided round 7's design -- (1) the value-pool cap was enforced only BETWEEN files, and a single broad
     citation (a tracked 26 MB artifact pooling 732,007 distinct values) matches 99.7% of random wrong 4-decimal
     numbers, so a cap on the pool size is the wrong tool; (8) 155 of the 353 findings added since 2026-09-01
     FAIL under r6, and 62% of the numbers it flags are CORRECT roundings (a relative tolerance of 1e-4 rejects
     0.477 for 0.4774).

ROUND 6'S RETRO CLAIM, CORRECTED (issue 7). Round 6 said its 73 whole-corpus verdict flips were "EVERY one via a
newly-caught UNSUPPORTED number (the cap, normalization and citation fixes)" and "candidate real errors". The
round-6 review reverted each change one at a time and measured the causes: synthesis tightening 32, cell
segmentation 24, NUM_RE widening 14, normalization 2, comment citations 1, the per-line cap and the artifact caps 0
each; of the 1,074 numbers newly flagged in those docs, 597 were correct roundings of a cited value and 53 were
arXiv/DOI identifiers, and 29 of the 73 flips had nothing else flagged. The claim was wrong. Round 7's retro is a
committed script with a per-cause column: tools/claim_check_retro_compare.py.

ROUND 7 (this revision) -- one design, seven parts:
  A. PRECISION-AWARE MATCHING. A number written with d decimals (d = mantissa decimals minus the exponent) matches
     an artifact value v when |x - v| <= 0.5 * 10^-d, i.e. v rounds to x at the precision the author stated (both
     half-up and half-even readings are accepted at the exact boundary, so binary float representation cannot
     decide it). Rule reported per number: `exact` or `rounding`. The legacy relative tolerance (1e-4 |x|, floor
     5e-6) is NOT kept as a second rule -- DEVIATION from the task's "or within the existing tolerance", measured,
     see CALIBRATION below: it is LOOSER than the stated precision exactly where it matters (|x| >= 5 at 3
     decimals, |x| >= 0.5 at 4), which is where the review's `12.3456` vs `12.3449` wrong number passed.
     `tol=` (API) still selects a fixed absolute tolerance. A glued magnitude suffix (`1.088B params`) is read as
     the scaled value OR the bare mantissa (never silently as the bare mantissa only). Numbers inside a URL, a
     DOI, an arXiv id or a cited file path are identifiers and are skipped (counted and printed).
  B. DISCRIMINATING POWER, per document (replaces round 6's value-pool cap). After loading the cited pool,
     CHANCE_DECOYS (400) decoys are drawn -- seeded, deterministic -- at the doc's OWN checked numbers' shapes:
     decoy = x + k * 10^-d, k uniform in +-[1, CHANCE_WINDOW] (500), cycling over the checked numbers so each
     claim contributes equally; each decoy is matched with rule A exactly like a real claim. The fraction that
     match is the doc's CHANCE-MATCH RATE: the probability that a nearby WRONG number at this doc's own
     precisions would be accepted. Above CHANCE_MAX the doc FAILS with "citations too broad to verify: cite the
     specific artifact file(s)". The rate is printed for every doc.
  C. NORMALIZATION THAT NEVER GLUES. Markup, emphasis, entities, tags and invisible characters are replaced by a
     SEPARATOR, never deleted, so `gain*0.1525`, `**acc**0.1525`, `&Delta;0.1525`, `x\\*0.1525` all expose the
     number. NFKC; every dash/minus variant (category Pd, U+2212, U+FE63, U+FF0D, U+02D7, U+2796) -> '-'; a '-'
     directly before the digits (markup in between is transparent) is a SIGN unless an ASCII digit, '.', ')',
     ']' or '%' precedes it (a range or subtraction); invisible characters beyond category Cf (Mn, Me, Co, Cn,
     Cc, Hangul fillers U+115F/U+1160/U+3164/U+FFA0) are separators; U+00B7/U+2024/U+FE52/U+FF0E/U+066B between
     digits are decimal points; Unicode decimal digits map to ASCII. Where a separator sits INSIDE a decimal
     literal (`0.15<b>25</b>`, `0.15\\u200b25`, `0.15**25**`), the glued reading -- what a reader sees -- is
     checked IN ADDITION (it can only add failures; it never replaces a number the separated reading found).
  D. TABLES. A GFM table row is detected with or without leading pipes, inside blockquotes and list items (any
     line in the block that follows a delimiter row, plus any line whose content starts with `|`); cells split on
     `|`, and every line also splits on `<br ...>` and block-level HTML tags (`</p><p>`, `<td>`, `<li>` ...).
     Splitting only narrows an exemption, so a misclassified prose line fails closed. The row-trailing form
     `| ... | <!--derived--> |` stays INERT (choice made: no second scope rule); it prints a WARNING, and every
     author-facing message now says "in the SAME table cell or <br>-segment as the number".
  E. SYNTHESIS. `claim_check: synthesis` applies only with a properly closed frontmatter block, a non-empty
     `claim_check_reason:` (whose regex no longer crosses a newline), and no verdict word -- matched
     case-insensitively after stripping invisible characters (GO, NO-GO, NOGO, NO GO, PASS(ED), FAIL(ED),
     REFUTED, CONFIRMED) -- in the doc's TITLE (frontmatter `title:`, else the first ATX/setext H1 outside code
     fences and frontmatter, else the filename), nor (a stricter DEVIATION, same reason as the bar itself) in the
     filename or a frontmatter `verdict:` field when the title came from elsewhere.
  F. COVERAGE counts DISTINCT checked values that occur at least once OUTSIDE every hidden carrier: HTML comments
     (multi-line, and an unclosed `<!--` to the end of the doc), link-reference definitions (`[//]: # (...)`),
     and hidden elements (`hidden` attribute, `display:none`, `<script>`/`<style>`/`<template>`). Numbers inside
     a carrier are still CHECKED (fail closed) -- only a derived-marker comment's own note is not scanned -- and a
     citation inside a carrier does not count.
  G. Every review issue has a SELFTEST_CASES entry whose `wrong_on` is re-derived from git on every test run; the
     CCT registry gate reports this file's own selftest problems VERBATIM, labelled BROKEN INSTRUMENT.

CALIBRATION (2026-09-25; re-derive with `tools/claim_check_retro_compare.py --since 2026-09-01 --legacy-compare
--attacks`; outputs committed as research/coordination/claimcheck_r7_calibration_since2026-09-01_2026-09-25.txt,
claimcheck_r7_retro_since2026-09-01_2026-09-25.tsv and the whole-corpus pair). On the 353 findings added since
2026-09-01, in the tracked tree:
  * FAILS: main 31, r5 135, r6 155, round 7 108. Docs failing ONLY on numbers round 7 shows are correct roundings,
    identifiers or misread signs: main 2, r5 51, r6 57; of r6's 1,651 flagged numbers, 1,093 (66%) are correct
    (1,018 roundings, 67 identifiers). Round 7 flags no correct rounding by construction. Its 108 failures: 94
    unsupported numbers (79 on that alone; 563 numbers, dominated by unmarked means/deltas and figures quoted
    from other findings -- the marker contract working, not the rule misfiring), 17 missing artifacts (19 paths
    absent from the tracked tree; 6 more exist only untracked in the main checkout; 1 is a seed-list shorthand
    `..._seed42/43/44/100/101/102.json` that PATH_RE reads as one path), 13 too broad (7 on that alone). The
    rule-caused residue: those 7 breadth-only docs, 2 docs failing only on an inequality bound (`exceeds
    0.9999998`), and the 1 shorthand -- 10 of 353 (2.8%), against r6's 57 (16%).
  * Rule A vs the legacy tolerance: keeping the old relative window as a second rule rescues 14 of 563 unsupported
    numbers (4 docs) and adds 21 too-broad docs (13 -> 34; chance p90 0.122 -> 0.335, 17 docs >= 0.5). Dropped.
  * CHANCE_MAX: rule B's rate over the 227 docs with a checked number: p50 0.015, p90 0.122, p95 0.203, max 0.645.
    Docs above / failing on breadth alone: 5% 52/28, 10% 28/16, 15% 17/9, 20% 13/7, 25% 7/5. The task's 5% start
    fails single-file citations of 67-129 values (a 3-decimal claim has 1,000 cells per unit, so ~N/1000 is the
    floor for N cited values in range) -- a false positive by the task's own definition. 20% (= p95; every wrong
    number caught at least 4 times in 5) is the calibrated bar. Against the review's scenarios with 40 random wrong
    numbers: the 26 MB artifact accepts 40/40 at 3 and 4 decimals, rate 1.000 -> FAIL; the largest legitimate
    battery glob accepts 20/40 at 3 decimals (rate 0.395 -> FAIL) and 6/40 at 4 decimals (0.145 -> the 6 wrong
    numbers pass rule B, the other 34 are flagged); four raw directories accept 17/40 at 3 decimals (0.230 ->
    FAIL) and 0/40 at 4.
  * LOW_COVERAGE_MIN_TOTAL 80 -> 30: under round 7's counting, the largest non-synthesis doc with zero checked
    visible values is 27 claims (since 2026-09-01 and whole corpus alike); 0 docs since 2026-09-01 newly fail.
  * SYNTHESIS: 17 docs since 2026-09-01 (42 whole corpus) declare it and NONE has a `claim_check_reason:`, so all
    lose the escape; 12 (31) flip from r5 PASS to FAIL. With a reason added, 5 (18) would still be barred by a
    verdict word: 3 state real verdicts (`de-risk GO`, `6-seed GO`, a `no-go` filename), 1 is a noun (`hygiene
    pass`), 1 names a lane (`satdiv-GO`). Words naming a gate (`GO gate`, `PASS criteria`) are not verdicts.

VERIFICATION of round 7's first commit (00aa451da) by adversarial probes against r6 found four fail-open
regressions, all fixed here and pinned as cases: a NON-empty comment inside a number (`0.15<!-- x -->25`; the
reader's reading, with every hidden carrier rendered as nothing, is now scanned too); a `|` inside a comment in a
table row (GFM splits cells before inline parsing, so pipes cut on the RAW line); `10.1525/s` taken for a bare DOI;
and a bidirectional override that displays stored digits in another order (the doc is refused). A seeded
differential fuzz (tests/test_claim_check_fuzz.py; 10,000 contexts per direction during verification) then found
`Δ_0.1525_`, and after that 0 fail-open and 0 false positives.

CANNOT CATCH (known, not chased): a number spelled in words; a decimal COMMA (`0,1525`, ambiguous with thousands
separators); homoglyph letters for digits (`O.1525`, Cyrillic O); a wrong number that happens to lie within
half a unit of an unrelated cited value (bounded per doc by rule B, at most CHANCE_MAX); a wrong claim about a
value that IS in the cited artifact but belongs to a different quantity (existence is not agreement --
gates/stated_value_mismatch's job).
"""
from __future__ import annotations

import bisect
import contextlib
import glob
import html
import io
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- number syntax -------------------------------------------------------------------------------------------
# Applied to the NORMALIZED scan copy of one segment (see `_numbers_in`), never to raw text. The lookbehind is
# ASCII-only (round 7, issue 3): an ASCII letter/digit/underscore/dot glued BEFORE the digits makes the token an
# identifier (`lr0.001`, `foo_0.125`), while a non-ASCII letter or symbol (`Δ0.1525`) does not -- `\w` is
# Unicode-aware, so round 6 read `Δ0.1525` as an identifier and dropped it. The trailing lookahead forbids only
# another digit: a glued unit (`0.9876ms`) or a sentence-final period is not part of the number.
_NUM_CORE_RE = re.compile(r"(?<![A-Za-z0-9_.])([0-9]*)\.([0-9]{3,})(?:[eE]([+-]?[0-9]+))?(?![0-9])")
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])(-?[0-9]*\.[0-9]{3,}(?:[eE][+-]?[0-9]+)?)(?![0-9])")   # public, simple
# Globs are allowed: a finding over N seeds cites one pattern, not N paths. Must contain a "/" -- a bare filename
# in prose is a REFERENCE, not a citation.
PATH_RE = re.compile(r"([\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.(?:jsonl|json))")
VERDICT_RE = re.compile(r"\b(GO|NO-GO|PASS|FAIL|REFUTED|CONFIRMED)\b")
DERIVED_MARK = "<!--derived-->"
DERIVED_CLOSE = "<!--/derived-->"          # no longer scopes anything -- matched only to print a WARNING
SYNTH_RE = re.compile(r"^claim_check:[ \t]*[\"']?synthesis[\"']?[ \t]*$", re.M)

# A comment whose text starts with `derived` is a marker (`<!--derived-->`, `<!-- derived -->`,
# `<!--derived: 0.499 = 30.240 - 29.741-->`); its own text is a NOTE and is never scanned for numbers.
_MARKER_INNER_RE = re.compile(r"\s*derived\b", re.I)
_STANDALONE_MARKER_RE = re.compile(r"^\s*(?:>\s*)*<!--\s*derived\b[^>]*-->\s*$", re.I)
_ATX_DERIVED_RE = re.compile(r"^\s*(?:>\s*)*#{1,6}\s*[*_`]*\s*derived\b", re.I)
_SETEXT_TITLE_RE = re.compile(r"^\s*(?:>\s*)*[*_`]*\s*derived\b", re.I)
_SETEXT_UNDERLINE_RE = re.compile(r"^\s*(?:=+|-+)\s*$")

MAX_EXEMPT_PER_LINE = 8          # per table cell / <br>-segment (the name is kept for API compatibility)
MAX_GLOB_FILES = 1000            # a glob loads at most this many files (sorted, deterministic) -- a runtime bound
MAX_ARTIFACT_BYTES = 200_000_000  # a single artifact larger than this is not loaded (reported as missing)
LEGACY_TOLERANCE = False         # rule A's union with the pre-round-7 relative window: measured and rejected

# ---- B. discriminating power ------------------------------------------------------------------------------------
CHANCE_DECOYS = 400
CHANCE_WINDOW = 500              # decoy = x + k * 10^-d, 1 <= |k| <= CHANCE_WINDOW
CHANCE_SEED = 20260925
CHANCE_MAX = 0.20                # calibrated on the 353 findings since 2026-09-01 -- see docstring CALIBRATION

# ---- C. normalization tables ----------------------------------------------------------------------------------
_DASHES = frozenset("\u2212\u02d7\u2796\ufe63\uff0d\u2010\u2011\u2012–—\u2015\u2e3a\u2e3b\ufe58\u2043")
_DOTLIKE = frozenset("\u00b7\u2024\ufe52\uff0e\u066b\u2027\u2e31\u0387")
_FILLERS = frozenset("\u115f\u1160\u3164\uffa0\u17b4\u17b5\u180e")
_BLANKS = frozenset("\u2800\t")
_INVISIBLE_CATS = frozenset(("Cf", "Mn", "Me", "Co", "Cn", "Cs"))
_ASCII_PUNCT = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_RANGE_LEFT = frozenset(".)]}%")     # a '-' after one of these (or a digit) is a range/subtraction, not a sign
_MAGNITUDE = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}
_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7};?|#[xX][0-9a-fA-F]{1,6};?|[A-Za-z][A-Za-z0-9]{1,31};)")
_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>")
# Identifiers, not measurements: skipped (a URL, a DOI whose suffix has a letter, an arXiv id, a cited path).
_ID_RES = (
    re.compile(r"(?:\b(?:https?|ftp)://|\bwww\.)[^\s<>()\[\]{}\"'`|]+", re.I),
    re.compile(r"\barxiv(?:\s*:\s*|\s+)[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?", re.I),
    # a DOI after `doi:`/`doi.org/`; a BARE one only when its suffix is 6+ characters holding a letter AND a digit
    # (`10.1038/415429a`), so a measurement with a unit (`10.1525/s`, `10.1525/step`) is never skipped
    re.compile(r"(?:\bdoi\s*:?\s*|\bdoi\.org/)10\.[0-9]{4,9}/[^\s<>()\[\]{}\"'`|]+", re.I),
    re.compile(r"\b10\.[0-9]{4,9}/(?=[^\s<>()\[\]{}\"'`|]*[A-Za-z])(?=[^\s<>()\[\]{}\"'`|]*[0-9])"
               r"[^\s<>()\[\]{}\"'`|]{6,}"),
    re.compile(r"[\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.(?:jsonl|json|md|py|npz|npy|pt|txt|tsv|csv|log|ya?ml|sh)\b"),
)

_ID_NUMBERISH_RE = re.compile(r"[0-9]*\.[0-9]{3,}")

# ---- D. segmentation ------------------------------------------------------------------------------------------
_SEG_TAG_RE = re.compile(r"<\s*/?\s*(?:br|p|div|li|tr|td|th|table|thead|tbody|tfoot|ul|ol|dl|dt|dd|h[1-6]|hr|"
                         r"blockquote|pre|section|article|header|footer|details|summary|caption|figure|figcaption)"
                         r"\b[^>]*>", re.I)
_DELIM_ROW_RE = re.compile(r"^\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")
_CONTAINER_RE = re.compile(r"^(?:[ \t]*(?:>[ \t]?|[-*+][ \t]+|[0-9]{1,9}[.)][ \t]+))*[ \t]*")

# ---- F. hidden carriers ---------------------------------------------------------------------------------------
_LINKREF_RE = re.compile(r"^[ ]{0,3}\[(?!\^)[^\]\n]+\]:[ \t]*\S[^\n]*$", re.M)
_HIDDEN_OPEN_RE = re.compile(r"<([A-Za-z][A-Za-z0-9-]*)\b(?=[^>]*(?:\bhidden\b|display\s*:\s*none|"
                             r"visibility\s*:\s*hidden))[^>]*>|<(script|style|template|noscript)\b[^>]*>", re.I)
_ZW = "\u200b"                   # blanking character for the SCAN copy: category Cf, so a soft separator
_BIDI_RE = re.compile("[\u202a-\u202e\u2066-\u2069]")

# ---- E. synthesis ---------------------------------------------------------------------------------------------
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_ATX_H1_RE = re.compile(r"^ {0,3}#(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_SETEXT_H1_RE = re.compile(r"^ {0,3}=+[ \t]*$")
_SETEXT_H2_RE = re.compile(r"^ {0,3}-+[ \t]*$")
_ATX_ANY_RE = re.compile(r"^ {0,3}(#{2,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_VERDICT_WORD_RE = re.compile(r"(?<![A-Za-z0-9])(no[- ]?go|go|pass(?:ed)?|fail(?:ed)?|refuted|confirmed)"
                              r"(?![A-Za-z0-9])", re.I)
_VERDICT_NOUN_RE = re.compile(r"(?:\s*/\s*(?:no[- ]?go|fail|pass)\b)?\s*[-/]?\s*(?:gates?|criteri(?:a|on)|bars?|"
                              r"thresholds?|conditions?|rules?)\b", re.I)
_VERDICT_HEADING_RE = re.compile(r"^\W*(?:verdict|result|results|outcome|conclusion|decision|status)\b", re.I)

# LOW COVERAGE -- calibrated under round 7's counting (see docstring CALIBRATION): lowered from 80 to 30.
MIN_CHECK_FRACTION = 0.05
LOW_COVERAGE_MIN_TOTAL = 30

Num = namedtuple("Num", "value decimals unit alts text start end split truncated")


# =================================================================================================================
# artifacts
# =================================================================================================================
_ART_CACHE = {}
_ART_CACHE_VALUES = [0]
_ART_CACHE_LIMIT = 5_000_000


def _flatten_numbers(obj, out):
    """Every finite numeric leaf in an artifact, at any depth, at FULL precision (round 7: round 6 rounded to 6
    decimals, which made a 7+-decimal claim unmatchable and a 6-decimal one mis-rounded). Returns the leaf count."""
    if isinstance(obj, bool):
        return 0
    if isinstance(obj, (int, float)):
        v = float(obj)
        if v == v and v not in (float("inf"), float("-inf")):
            out.add(v)
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


def _load_one(h):
    st = os.stat(h)
    key = (h, st.st_mtime_ns, st.st_size)
    hit = _ART_CACHE.get(key)
    if hit is not None:
        return hit
    if st.st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("larger than MAX_ARTIFACT_BYTES (%d)" % MAX_ARTIFACT_BYTES)
    vals, verdicts = set(), []
    if h.endswith(".jsonl"):
        with open(h, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    d = json.loads(ln)
                    _flatten_numbers(d, vals)
                    _flatten_verdicts(d, verdicts)
    else:
        with open(h, encoding="utf-8") as fh:
            d = json.load(fh)
        _flatten_numbers(d, vals)
        _flatten_verdicts(d, verdicts)
    res = (frozenset(vals), tuple(verdicts))
    if _ART_CACHE_VALUES[0] + len(vals) > _ART_CACHE_LIMIT:
        _ART_CACHE.clear()
        _ART_CACHE_VALUES[0] = 0
    _ART_CACHE[key] = res
    _ART_CACHE_VALUES[0] += len(vals)
    return res


def load_artifacts(paths):
    """Returns (pool, verdicts, loaded, missing, capped). `pool` is the SORTED list of distinct finite values of
    every cited artifact. There is no value cap (round 7, issue 1): breadth is measured per doc by the
    discriminating-power check instead. A glob loads at most MAX_GLOB_FILES files (a runtime bound, reported)."""
    vals, verdicts, loaded, missing, capped = set(), [], [], [], []
    for p in paths:
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
            try:
                v, vd = _load_one(h)
                vals |= v
                verdicts.extend(vd)
                loaded.append(h)
            except Exception as e:                       # narrow enough to see; never silent
                missing.append("%s (unreadable: %s)" % (p, type(e).__name__))
    return sorted(vals), verdicts, loaded, missing, capped


# =================================================================================================================
# A. matching
# =================================================================================================================
def _any_within(pool, x, w):
    i = bisect.bisect_left(pool, x - w)
    return i < len(pool) and pool[i] <= x + w


def _readings(num):
    """(value, unit, label) for every reading of a written number: the number itself, plus the scaled value when a
    magnitude suffix is glued to it."""
    out = [(num.value, num.unit, "")]
    for scale, suf in num.alts:
        out.append((num.value * scale, num.unit * scale, "+" + suf))
    return out


def _slack(x):
    """Float-representation noise only: a few ULPs of x (a 1e-12 absolute slack swallowed every decoy of a pasted
    17-decimal float, whose stated precision is below 1e-12)."""
    return 8.0 * math.ulp(x) if x else 8.0 * math.ulp(1e-300)


def _match(num, pool, tol=None):
    """Rule A. Returns the rule that matched ('exact' | 'rounding' | 'tolerance'), with a '+<suffix>' label for a
    scaled reading, or None. 'tolerance' only when `tol` is given, or when LEGACY_TOLERANCE re-enables the
    pre-round-7 relative window (switchable so tools/claim_check_retro_compare.py can MEASURE the choice)."""
    for x, u, lab in _readings(num):
        slack = _slack(x)
        if tol is not None:
            if _any_within(pool, x, tol + slack):
                return "tolerance" + lab
            continue
        if _any_within(pool, x, slack):
            return "exact" + lab
        if _any_within(pool, x, 0.5 * u * (1 + 1e-9) + slack):
            return "rounding" + lab
        if num.truncated:                  # `3.490537...`: an explicit ellipsis states truncation toward zero
            c = x + math.copysign(0.5 * u, x) if x else x
            if _any_within(pool, c, 0.5 * u * (1 - 1e-9)):
                return "truncation" + lab
        if LEGACY_TOLERANCE and _any_within(pool, x, max(5e-6, 1e-4 * abs(x)) + slack):
            return "tolerance" + lab
    return None


def _chance_rate(basis, pool, tol=None):
    """Rule B. Fraction of CHANCE_DECOYS decoys at the doc's own claim shapes that rule A would accept."""
    if not basis or not pool:
        return None if not basis else 0.0
    rng = random.Random(CHANCE_SEED)
    hits = 0
    for i in range(CHANCE_DECOYS):
        num = basis[i % len(basis)]
        k = rng.randint(1, CHANCE_WINDOW) * (1 if rng.random() < 0.5 else -1)
        # A decoy steps by the stated unit, but never by less than 12 significant digits: past that a float cannot
        # represent the step at all (a pasted 17-decimal float's "unit" is below its own ULP).
        step = max(num.unit, abs(num.value) * 1e-12)
        decoy = num._replace(value=num.value + k * step)
        if _match(decoy, pool, tol):
            hits += 1
    return hits / float(CHANCE_DECOYS)


# =================================================================================================================
# C. normalization of one segment, and number extraction
# =================================================================================================================
def _invisible(c):
    return unicodedata.category(c) in _INVISIBLE_CATS or c in _FILLERS or (
        unicodedata.category(c) == "Cc" and c not in "\t\n")


def _identifier_spans(seg):
    spans = []
    for rx in _ID_RES:
        spans.extend((m.start(), m.end()) for m in rx.finditer(seg))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _normalize(seg):
    """-> (chars, origins, kinds, n_identifiers). kinds: 'c' ordinary, 'l' literal (escaped / entity-decoded),
    's' soft separator (markup that renders as NOTHING: a tag, an invisible character, a blanked comment),
    'e' emphasis separator (`*`, `~`, a backtick, a non-intraword `_`). A separator is a SPACE in the scan text, so
    it can never glue the characters on either side of it (round 7, issue 3)."""
    ch, org, kd = [], [], []

    def emit(c, o, k):
        ch.append(c)
        org.append(o)
        kd.append(k)

    def emit_char(c, o, literal):
        if _invisible(c):
            emit(" ", o, "s")
            return
        if unicodedata.category(c) == "No":           # superscripts/fractions: never glue them to a number
            emit(" ", o, "c")
            return
        for c2 in unicodedata.normalize("NFKC", c):
            cat = unicodedata.category(c2)
            if _invisible(c2):
                emit(" ", o, "s")
            elif c2 in _DASHES or cat == "Pd":
                emit("-", o, "l" if literal else "c")
            elif cat == "Nd":
                emit(str(unicodedata.decimal(c2)), o, "c")
            elif c2 in _DOTLIKE:
                emit(c2, o, "d")
            elif cat == "Zs" or c2 in _BLANKS:
                emit(" ", o, "c")
            elif not literal and c2 in "*~`":
                emit(" ", o, "e")
            elif not literal and c2 == "_":
                emit("_", o, "u")
            else:
                emit(c2, o, "l" if literal else "c")

    ids = _identifier_spans(seg)
    n_ids = sum(1 for a, b in ids if _ID_NUMBERISH_RE.search(seg, a, b))
    idi, i, n = 0, 0, len(seg)
    while i < n:
        while idi < len(ids) and ids[idi][1] <= i:
            idi += 1
        if idi < len(ids) and ids[idi][0] <= i:
            for j in range(i, ids[idi][1]):
                emit(" ", j, "c")
            i = ids[idi][1]
            continue
        c = seg[i]
        if c == "\\" and i + 1 < n and seg[i + 1] in _ASCII_PUNCT:
            emit_char(seg[i + 1], i + 1, True)
            i += 2
            continue
        if c == "&":
            m = _ENTITY_RE.match(seg, i)
            if m:
                dec = html.unescape(m.group(0))
                if dec != m.group(0):
                    for c2 in dec:
                        emit_char(c2, i, True)
                    i = m.end()
                    continue
        if c == "<":
            m = _TAG_RE.match(seg, i)
            if m:
                emit(" ", i, "s")
                i = m.end()
                continue
        emit_char(c, i, False)
        i += 1

    for p, k in enumerate(kd):                         # resolve context-dependent characters
        prev = ch[p - 1] if p > 0 else ""
        nxt = ch[p + 1] if p + 1 < len(ch) else ""
        if k == "u":                                   # an intraword `_` is literal (CommonMark), else emphasis --
            # but only after an ASCII letter/digit: `foo_0.125` is an identifier, `Δ_0.1525` is a symbol and a number
            if prev.isascii() and prev.isalnum() and nxt.isalnum():
                kd[p] = "c"
            else:
                ch[p], kd[p] = " ", "e"
        elif k == "d":                                 # a dot-like character between digits is a decimal point
            ch[p], kd[p] = ("." if prev.isdigit() and nxt.isdigit() else " "), "c"
    return ch, org, kd, n_ids


def _is_sep(k):
    return k == "s" or k == "e"


def _extract(ch, org, kd):
    s = "".join(ch)
    out = []
    for m in _NUM_CORE_RE.finditer(s):
        a, b = m.start(), m.end()
        exp = int(m.group(3)) if m.group(3) else 0
        d = len(m.group(2)) - exp
        mag = float(s[a:b])
        j = a - 1
        while j >= 0 and _is_sep(kd[j]):
            j -= 1
        neg, sa = False, a
        if j >= 0 and ch[j] == "-":
            k = j - 1
            while k >= 0 and _is_sep(kd[k]):
                k -= 1
            prev = ch[k] if k >= 0 else " "            # start of segment: nothing before the dash, so a sign
            if not (prev.isdigit() or prev in _RANGE_LEFT):
                neg, sa = True, j
        alts = ()
        if b < len(s) and kd[b] == "c" and s[b] in _MAGNITUDE and (b + 1 >= len(s) or not s[b + 1].isalnum()):
            alts = ((_MAGNITUDE[s[b]], s[b]),)
        text = ("-" if neg else "") + s[a:b] + (alts[0][1] if alts else "")
        truncated = s[b:b + 3] == "..."                # U+2026 is NFKC-folded to "..." already
        out.append(Num(-mag if neg else mag, d, 10.0 ** (-d), alts, text, org[sa], org[b - 1] + 1, False,
                       truncated))
    return out


def _glued(ch, org, kd):
    """The reading a reader SEES where a separator sits inside a decimal literal: a soft separator (renders as
    nothing) between [0-9.] and [0-9.] is dropped; an emphasis separator only when its left run already holds a
    '.' or its right neighbour is '.' (so `4*0.03972` -- a multiplication -- is NOT glued into 40.03972)."""
    n = len(ch)
    keep = [True] * n
    for p in range(n):
        if not _is_sep(kd[p]):
            continue
        lft = p - 1
        while lft >= 0 and _is_sep(kd[lft]):
            lft -= 1
        rgt = p + 1
        while rgt < n and _is_sep(kd[rgt]):
            rgt += 1
        if lft < 0 or rgt >= n:
            continue
        L, R = ch[lft], ch[rgt]
        if not ((L.isdigit() or L == ".") and (R.isdigit() or R == ".")):
            continue
        if kd[p] == "e":
            q, run = lft, []
            while q >= 0 and (ch[q].isdigit() or ch[q] == "." or _is_sep(kd[q])):
                if not _is_sep(kd[q]):
                    run.append(ch[q])
                q -= 1
            if "." not in run and R != ".":
                continue
        keep[p] = False
    if all(keep):
        return None
    return ([c for c, k in zip(ch, keep) if k], [o for o, k in zip(org, keep) if k],
            [x for x, k in zip(kd, keep) if k])


def _numbers_in(seg):
    """Every measurement-shaped number in one segment of text -> (list[Num], n_identifiers_skipped)."""
    ch, org, kd, n_ids = _normalize(seg)
    nums = _extract(ch, org, kd)
    g = _glued(ch, org, kd)
    if g is not None:
        seen = {(x.start, x.end) for x in nums}
        for x in _extract(*g):
            if (x.start, x.end) not in seen and not any(y.start <= x.start < y.end for y in nums
                                                          if abs(y.value - x.value) < 1e-15):
                nums.append(x._replace(split=True))
    nums.sort(key=lambda x: x.start)
    return nums, n_ids


# =================================================================================================================
# F. hidden carriers, D. segmentation, E. synthesis
# =================================================================================================================
def _hidden_spans(text):
    """(start, end, kind) for every region a READER cannot see. Over-inclusive on purpose: a region wrongly judged
    hidden only loses its citations and its coverage credit (fails closed); its numbers are still checked."""
    spans = []
    i = 0
    while True:
        a = text.find("<!--", i)
        if a < 0:
            break
        b = text.find("-->", a + 4)
        if b < 0:
            spans.append((a, len(text), "comment-unclosed"))
            break
        kind = "marker" if _MARKER_INNER_RE.match(text, a + 4) else "comment"
        spans.append((a, b + 3, kind))
        i = b + 3
    spans.extend((m.start(), m.end(), "linkref") for m in _LINKREF_RE.finditer(text))
    for m in _HIDDEN_OPEN_RE.finditer(text):
        tag = (m.group(1) or m.group(2)).lower()
        close = re.compile(r"</\s*%s\s*>" % re.escape(tag), re.I).search(text, m.end())
        spans.append((m.start(), close.end() if close else len(text), "hidden-element"))
    spans.sort()
    return spans


def _blank(text, spans, fill, keep_inner=False):
    """Replace the characters of each span with `fill`, keeping newlines so line numbers never move."""
    buf = list(text)
    for a, b, kind in spans:
        if keep_inner and kind in ("comment", "comment-unclosed"):
            rng = list(range(a, min(a + 4, b)))
            if kind == "comment":
                rng += list(range(max(a + 4, b - 3), b))
        elif keep_inner and kind != "marker":
            continue
        else:
            rng = range(a, b)
        for j in rng:
            if buf[j] != "\n":
                buf[j] = fill
    return "".join(buf)


def _merge(spans):
    """Union of (start, end, kind) spans as sorted, disjoint (start, end) intervals."""
    out = []
    for a, b, _k in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _in_spans(merged, pos):
    i = bisect.bisect_right(merged, (pos, float("inf"))) - 1
    return i >= 0 and merged[i][0] <= pos < merged[i][1]


def _table_rows(vis_lines):
    rows = set()
    stripped = [_CONTAINER_RE.sub("", ln, count=1) for ln in vis_lines]
    n = len(stripped)
    for i, s in enumerate(stripped):
        if s.startswith("|"):
            rows.add(i)
        if "|" in s and _DELIM_ROW_RE.match(s):
            if i > 0 and stripped[i - 1].strip():
                rows.add(i - 1)
            j = i
            while j < n and stripped[j].strip():
                rows.add(j)
                j += 1
    return rows


def _segments(vis_line, table_row, raw_line=None):
    """Independent exemption scopes of one line, as (start, end) columns. Line-breaking tags cut where they RENDER
    (on the visible line: a `<br>` or `<div>` inside a hidden carrier renders nothing), while a table row's `|`
    cuts on the RAW line (GFM splits a row into cells before any inline parsing, so a `|` inside a comment still
    separates cells for a reader). Every cut only narrows an exemption."""
    cuts = [(m.start(), m.end()) for m in _SEG_TAG_RE.finditer(vis_line)]
    if table_row:
        cuts += [(m.start(), m.end()) for m in re.finditer(r"\|", raw_line if raw_line is not None else vis_line)]
    cuts.sort()
    segs, pos = [], 0
    for a, b in cuts:
        if a > pos:
            segs.append((pos, a))
        pos = max(pos, b)
    segs.append((pos, len(vis_line)))
    return [(a, b) for a, b in segs if b >= a]


def _strip_invisible(s):
    return "".join(" " if c in _DASHES else c for c in unicodedata.normalize(
        "NFKC", "".join(c for c in (s or "") if not _invisible(c))))


def _fm_value(fm, key):
    m = re.search(r"^%s:[ \t]*(.*?)[ \t]*$" % re.escape(key), fm, re.M)
    if not m:
        return ""
    v = m.group(1)
    if v in ("|", ">", "|-", ">-", "|+", ">+"):
        rest = fm[m.end():].split("\n")[1:]
        block = []
        for ln in rest:
            if ln.startswith((" ", "\t")) or not ln.strip():
                block.append(ln.strip())
            else:
                break
        v = " ".join(x for x in block if x)
    v = v.strip().strip("\"'").strip()
    return "" if v in ("~", "null", "Null", "NULL") else v


def _headings(text, fm_m):
    """Every heading OUTSIDE code fences and the frontmatter, in order: (level, kind, text). ATX headings of any
    level, setext H1 (`===`) and H2 (`---` under a paragraph line)."""
    lines = text.split("\n")
    start = fm_m.group(0).count("\n") if fm_m else 0
    out, fence, prev = [], None, None
    for ln in lines[start:]:
        fm = _FENCE_RE.match(ln)
        if fence:
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= len(fence) and \
                    not ln.strip()[len(fm.group(1)):].strip():
                fence = None
            prev = None
            continue
        if fm:
            fence, prev = fm.group(1), None
            continue
        m = _ATX_H1_RE.match(ln)
        if m:
            out.append((1, "H1", m.group(1) or ""))
            prev = None
            continue
        m = _ATX_ANY_RE.match(ln)
        if m:
            out.append((len(m.group(1)), "heading", m.group(2) or ""))
            prev = None
            continue
        if prev is not None and _SETEXT_H1_RE.match(ln):
            out.append((1, "setext H1", prev))
            prev = None
            continue
        if prev is not None and _SETEXT_H2_RE.match(ln):
            out.append((2, "setext heading", prev))
            prev = None
            continue
        prev = ln.strip() if ln.strip() and not ln.startswith(("    ", "\t")) and not ln.lstrip().startswith(
            (">", "#", "|", "- ", "* ", "+ ")) else None
    return out


def _doc_title(text, fm_m, doc_path):
    """E. The doc's title: frontmatter `title:`, else the first ATX/setext H1 outside code fences and frontmatter,
    else the filename."""
    if fm_m:
        t = _fm_value(fm_m.group(1), "title")
        if t:
            return "frontmatter title", t
    for level, kind, t in _headings(text, fm_m):
        if level == 1:
            return kind, t
    return "filename", os.path.splitext(os.path.basename(doc_path))[0]


def _verdict_word(s):
    """The first verdict word in `s` (case-insensitive, invisible characters stripped), skipping a word that NAMES a
    gate rather than stating a verdict ("a GO gate", "PASS criteria", "the GO/NO-GO bar")."""
    t = _strip_invisible(s).replace("_", " ")
    for m in _VERDICT_WORD_RE.finditer(t):
        if _VERDICT_NOUN_RE.match(t, m.end()):
            continue
        return m.group(1)
    return None


def _first_clause(s):
    return re.split(r"\s(?:--|—|–)\s|[;(.:]\s", s or "", maxsplit=1)[0]


def _synthesis_status(text, doc_path="document.md"):
    """Returns (is_synthesis, reason, barred_warning). See docstring part E."""
    m = _FRONTMATTER_RE.match(text)
    if not m or not SYNTH_RE.search(m.group(1)):
        return False, None, None
    fm = m.group(1)
    reason = _fm_value(fm, "claim_check_reason")
    if not reason:
        return False, None, (
            "declares `claim_check: synthesis` but no non-empty `claim_check_reason:` in the SAME frontmatter "
            "block -- falling back to the normal rules (the escape needs a STATED reason)")
    src, title = _doc_title(text, m, doc_path)
    probes = [("title (%s)" % src, title)]
    if src != "filename":
        probes.append(("filename", os.path.splitext(os.path.basename(doc_path))[0]))
    probes.append(("frontmatter verdict: (first clause)", _first_clause(_fm_value(fm, "verdict"))))
    # Every H1 (a leading `# Notes` must not hide a later title), and any heading that announces a result.
    probes.extend(("%s %r" % (kind, t[:60]), t) for lvl, kind, t in _headings(text, m)
                  if lvl == 1 or _VERDICT_HEADING_RE.match(t))
    for where, s in probes:
        w = _verdict_word(s)
        if w:
            return False, None, (
                "declares `claim_check: synthesis` but its %s states a verdict (%s) -- a verdict-bearing "
                "document is BARRED from the synthesis escape and every number is checked" % (where, w))
    return True, reason, None


def _line_warnings(lines):
    """Pre-round-5 scope idioms that are INERT -- non-blocking author-facing WARNINGs, never a verdict change."""
    warnings = []
    n = len(lines)
    for i, ln in enumerate(lines):
        if _STANDALONE_MARKER_RE.match(ln):
            warnings.append((i + 1, "standalone marker",
                             "a lone <!--derived--> exempts nothing (it opens no scope). Put the marker in the "
                             "SAME table cell or <br>-segment as each derived number."))
        if DERIVED_CLOSE in ln:
            warnings.append((i + 1, "close marker",
                             "<!--/derived--> closes nothing -- there are no ranges. Remove it, and put "
                             "<!--derived--> in the same cell/segment as each derived number."))
        if _ATX_DERIVED_RE.match(ln):
            warnings.append((i + 1, "'Derived' heading",
                             "a '## Derived'-style heading opens no section. Put the marker in the same cell/"
                             "segment as each derived number under it."))
        elif i + 1 < n and _SETEXT_TITLE_RE.match(ln) and _SETEXT_UNDERLINE_RE.match(lines[i + 1]):
            warnings.append((i + 1, "'Derived' heading (setext)",
                             "a setext 'Derived' heading opens no section. Put the marker in the same cell/"
                             "segment as each derived number under it."))
    return warnings


# =================================================================================================================
# the scan
# =================================================================================================================
def _empty_scan_result(unreadable):
    return dict(cited=[], nums=[], loaded=[], missing=[], capped=[], checked=0, checked_distinct=0,
                checked_visible_distinct=0, suppressed={"inline": 0, "synthesis": 0}, total_numeric=0,
                unsupported=[], records=[], matched={}, identifiers=0, chance=None, too_broad=False,
                synthesis=False, low_coverage=False, marked_lines=[], warnings=[], unreadable=unreadable)


def _scan(doc_path, tol=None):
    """Pure computation, no printing -- shared by the CLI (`check`), `tools/finding_lint.py` and the retro script.
    `tol=None` is rule A; a number passes `tol` as a fixed absolute tolerance instead."""
    try:
        raw = open(doc_path, "rb").read()
    except OSError as e:
        return _empty_scan_result("cannot read %s: %s: %s" % (doc_path, type(e).__name__, e))
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        return _empty_scan_result("%s is not valid UTF-8 (%s at byte offset %d) -- fix the file's encoding "
                                  "before it can be checked" % (doc_path, e.reason, e.start))
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    bidi = _BIDI_RE.search(text)
    if bidi:
        # A bidirectional embedding/override/isolate control makes the SCREEN order of digits differ from the
        # stored order (`\u202e5251.0` displays as 0.1525): no reading of the stored text is the one a reader sees.
        return _empty_scan_result("%s contains a bidirectional control character (U+%04X, line %d) that can "
                                  "reorder digits on screen -- remove it" % (
                                      doc_path, ord(bidi.group(0)), text.count("\n", 0, bidi.start()) + 1))
    lines = text.split("\n")
    synthesis, _reason, synth_barred = _synthesis_status(text, doc_path)

    hidden = _hidden_spans(text)
    hidden_merged = _merge(hidden)
    vis = _blank(text, hidden, " ")                       # what a reader sees (citations, table structure)
    scan = _blank(text, hidden, _ZW, keep_inner=True)     # what is scanned for numbers (carrier content kept)
    seen = _blank(text, hidden, _ZW)                       # the reader's reading: carriers render as NOTHING
    cited = sorted(set(PATH_RE.findall(vis)))
    pool, _verdicts, loaded, missing, capped = load_artifacts(cited)

    line_starts, off = [], 0
    for ln in lines:
        line_starts.append(off)
        off += len(ln) + 1
    vis_lines = vis.split("\n")
    seen_lines = seen.split("\n")
    scan_lines = scan.split("\n")
    table_rows = _table_rows(vis_lines)
    markers_by_line = {}
    for a, _b, kind in hidden:
        if kind == "marker":
            li = bisect.bisect_right(line_starts, a) - 1
            markers_by_line.setdefault(li, []).append(a - line_starts[li])

    records, unsupported, basis, warnings = [], [], [], []
    suppressed = {"inline": 0, "synthesis": 0}
    marked_lines, n_ids = [], 0
    for li, vln in enumerate(vis_lines):
        segs = _segments(vln, li in table_rows, lines[li])
        seg_markers = [0] * len(segs)
        for col in markers_by_line.get(li, ()):
            for si, (a, b) in enumerate(segs):
                if a <= col < b or (si == len(segs) - 1 and col >= a):
                    seg_markers[si] += 1
                    break
        sln = scan_lines[li]
        for si, (a, b) in enumerate(segs):
            nums, ids = _numbers_in(sln[a:b])
            # ... plus any number that only exists in the READER's reading, where a hidden carrier inside a
            # decimal literal renders as nothing (`0.15<!-- x -->25` shows 0.1525).
            spans = {(x.start, x.end, x.value) for x in nums}
            extra = [x for x in _numbers_in(seen_lines[li][a:b])[0] if (x.start, x.end, x.value) not in spans]
            if extra:
                nums = sorted(nums + [x._replace(split=True) for x in extra], key=lambda x: x.start)
            n_ids += ids
            if seg_markers[si] and not nums and not ids and not _STANDALONE_MARKER_RE.match(lines[li]):
                warnings.append((li + 1, "marker exempts nothing",
                                 "this <!--derived--> sits in a %s that holds no number, so it exempts NOTHING -- "
                                 "a marker exempts only numbers in its own table cell or <br>-segment (a marker "
                                 "alone in a row's last cell does not reach the row's other cells)."
                                 % ("table cell" if li in table_rows else "segment")))
            if not nums:
                continue
            if seg_markers[si] and li + 1 not in marked_lines:
                marked_lines.append(li + 1)
            for idx, num in enumerate(nums):
                pos = line_starts[li] + a + num.start
                visible = not _in_spans(hidden_merged, pos)
                rec = dict(line=li + 1, value=num.value, text=num.text, decimals=num.decimals, visible=visible,
                           split=num.split, status=None, rule=None)
                if seg_markers[si] and idx < MAX_EXEMPT_PER_LINE:
                    rec["status"] = "exempt"
                    suppressed["inline"] += 1
                elif synthesis:
                    rec["status"] = "synthesis"
                    suppressed["synthesis"] += 1
                else:
                    rec["status"] = "checked"
                    basis.append(num)
                    rule = _match(num, pool, tol)
                    rec["rule"] = rule
                    if rule is None:
                        rec["hint"] = _hint(num, pool, tol)
                        unsupported.append((li + 1, num.value, lines[li].strip()[:88]))
                records.append(rec)

    if synthesis and not cited:
        unsupported.append((0, 0.0, "synthesis doc cites NO artifact — the escape still requires citations"))

    checked_recs = [r for r in records if r["status"] == "checked"]
    checked_distinct = {round(r["value"], 9) for r in checked_recs}
    checked_vis_distinct = {round(r["value"], 9) for r in checked_recs if r["visible"]}
    total_numeric = len(records)
    low_coverage = (not synthesis and total_numeric >= LOW_COVERAGE_MIN_TOTAL
                    and (len(checked_vis_distinct) / float(total_numeric)) < MIN_CHECK_FRACTION)
    matched = {}
    for r in checked_recs:
        k = (r["rule"] or "unmatched").split("+")[0]
        matched[k] = matched.get(k, 0) + 1
    chance = _chance_rate(basis, pool, tol) if basis else None
    too_broad = chance is not None and chance > CHANCE_MAX

    warnings = _line_warnings(lines) + warnings
    if synth_barred:
        warnings.append((1, "synthesis escape not applied", synth_barred))
    for c in capped:
        warnings.append((0, "citation capped", c))

    return dict(cited=cited, nums=pool, loaded=loaded, missing=missing, capped=capped,
                checked=len(checked_recs), checked_distinct=len(checked_distinct),
                checked_visible_distinct=len(checked_vis_distinct), suppressed=suppressed,
                total_numeric=total_numeric, unsupported=unsupported, records=records, matched=matched,
                identifiers=n_ids, chance=chance, too_broad=too_broad, synthesis=synthesis,
                low_coverage=low_coverage, marked_lines=marked_lines, warnings=warnings, unreadable=None)


def _hint(num, pool, tol):
    """Why an unsupported number is unsupported, when a cheap probe can tell: a sign flip, a split number, or a
    near miss (the artifact holds a value one step away at the stated precision -- round it, do not truncate)."""
    if num.split:
        return "number split by markup (read as the glued value a reader sees)"
    if num.value < 0 and _match(num._replace(value=-num.value), pool, tol):
        return ("the artifact holds +%s: a dash directly before a number reads as a MINUS sign -- put a space "
                "after a punctuation dash" % num.text.lstrip("-"))
    if tol is None:
        i = bisect.bisect_left(pool, num.value)
        near = [pool[j] for j in (i - 1, i) if 0 <= j < len(pool)]
        if near:
            v = min(near, key=lambda a: abs(a - num.value))
            if abs(v - num.value) <= 1.5 * num.unit:
                return "near miss: the artifact holds %r, which rounds to %.*f at the stated precision" % (
                    v, max(num.decimals, 0), v)
    return ""


def _verdict(r):
    """The single FAIL/PASS rule, shared by `check()`, `selftest()` and the test suite."""
    return "FAIL" if (r.get("unreadable") or r["missing"] or r["unsupported"] or r["low_coverage"]
                      or r.get("too_broad")) else "PASS"


TOO_BROAD_MSG = ("citations too broad to verify: cite the specific artifact file(s) -- a wrong number at this doc's "
                 "precision would match the cited pool by chance more than %d%% of the time (or state each "
                 "measurement at its full precision: one more decimal makes a match ~10x more specific)"
                 % round(100 * CHANCE_MAX))

FIX_HINT = ("fix the number, cite the artifact FILE that holds it (a path with a /), or mark a derived/quoted value "
            "<!--derived--> in the SAME table cell or <br>-segment as the number (at most %d per cell; a marker "
            "alone on a line or in a row's last cell exempts nothing)" % MAX_EXEMPT_PER_LINE)


def check(doc_path, tol=None, verbose=True):
    r = _scan(doc_path, tol)
    shown = os.path.relpath(doc_path, ROOT) if os.path.isabs(doc_path) else doc_path
    if r.get("unreadable"):
        if verbose:
            print("claim_check: %s" % shown)
            print("  ⛔ UNREADABLE: %s" % r["unreadable"])
            print("  => ⛔ UNREADABLE — fix the file's encoding before it can be checked")
        return 1
    fail = _verdict(r) == "FAIL"
    if verbose:
        m = r["matched"]
        print("claim_check: %s" % shown)
        print("  cited artifacts : %d found, %d missing" % (len(r["loaded"]), len(r["missing"])))
        for mp in r["missing"][:5]:
            print("      ⛔ MISSING  %s" % mp)
        print("  measurements    : %d checked (%d distinct, %d distinct outside hidden carriers) against %d "
              "artifact values%s" % (r["checked"], r["checked_distinct"], r["checked_visible_distinct"],
                                     len(r["nums"]), "   [synthesis: per-number rule suppressed, citations "
                                                     "still required]" if r["synthesis"] else ""))
        print("  matched         : %d exact, %d by rounding at the stated precision%s%s, %d unmatched"
              % (m.get("exact", 0), m.get("rounding", 0),
                 (", %d by stated truncation (...)" % m["truncation"]) if m.get("truncation") else "",
                 (", %d by tolerance" % m["tolerance"]) if m.get("tolerance") else "", m.get("unmatched", 0)))
        print("  exempted        : %d by <!--derived--> in the same cell/segment (line(s) %s), %d by synthesis, "
              "of %d numeric claim(s); %d identifier(s) skipped (URL/DOI/arXiv/path)"
              % (r["suppressed"]["inline"], ", ".join(str(n) for n in r["marked_lines"]) or "-",
                 r["suppressed"]["synthesis"], r["total_numeric"], r["identifiers"]))
        if r["chance"] is None:
            print("  chance match    : n/a (no checked number to draw decoys from)")
        else:
            print("  chance match    : %.1f%% of %d decoys at this doc's own precisions would pass (limit %.0f%%)"
                  % (100 * r["chance"], CHANCE_DECOYS, 100 * CHANCE_MAX))
            if r["too_broad"]:
                print("      ⛔ %s" % TOO_BROAD_MSG)
        for lineno, kind, msg in r["warnings"]:
            print("      ⚠️  WARNING line %-4d %-24s %s" % (lineno, kind, msg))
        hints = {(x["line"], x["value"]): x.get("hint", "") for x in r["records"] if x["status"] == "checked"}
        for lineno, val, ctx in r["unsupported"][:12]:
            h = hints.get((lineno, val), "")
            print("      ⛔ line %-4d %-14g not in any cited artifact | %s%s" % (lineno, val, ctx,
                                                                              ("\n           -> " + h) if h else ""))
        if len(r["unsupported"]) > 12:
            print("      ... and %d more" % (len(r["unsupported"]) - 12))
        if r["low_coverage"]:
            print("      ⛔ LOW COVERAGE: only %d/%d (%.0f%%) DISTINCT numeric value(s) outside hidden carriers were "
                  "actually checked -- the rest were marked <!--derived--> or hidden. A doc this size should not be "
                  "almost entirely derived; mark the specific derived numbers, not the whole document."
                  % (r["checked_visible_distinct"], r["total_numeric"],
                     100.0 * r["checked_visible_distinct"] / r["total_numeric"] if r["total_numeric"] else 0.0))
        if not r["cited"]:
            print("  ⚠️  NO ARTIFACT CITED — a findings doc with no artifact path cannot be checked at all.")
        print("  => %s" % (("⛔ UNSUPPORTED CLAIMS, missing artifacts, or citations too broad — " + FIX_HINT)
                           if fail else "✔ every measurement traces to a cited artifact"))
    return 1 if fail else 0


# ---------------------------------------------------------------------------------------------------------------
# SELFTEST REGISTRY. `tests/test_claim_check_line_only.py` re-runs every case here against the historical
# revisions in _HISTORY_SHAS (loaded straight from git, not retyped) and asserts the recorded `wrong_on` equals the
# set that ACTUALLY gets it wrong, so "this used to pass, now it fails" is re-derived every run. `%(art)s` is the
# cited artifact: {"accuracy": 0.17, "baseline": 0.1625} unless a case supplies its own `artifact`. 0.1525 / 0.140 /
# 1.23456 / -0.1525 / -0.1625 are WRONG numbers; 0.104615 / 0.207531 / 0.311079 are legitimately derived ones.
# Optional keys: `issue` (the round-6 review issue a case pins), `artifact`, `expect_reason` ('too_broad' |
# 'low_coverage'), `expect_output` (a substring check()'s printed report must contain), `kind='gate'`.
# ---------------------------------------------------------------------------------------------------------------
_HDR = "# Some finding\n\nArtifact: `%(art)s`\n\n"
_ALL_BEFORE_R6 = ("main", "r1", "r2", "r3", "r4", "r5")
_ALL = _ALL_BEFORE_R6 + ("r6",)
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
         why="r1's hole: a table right after a marker absorbed a wrong number on the very next line",
         doc=_HDR + "<!--derived-->\n| metric | value |\n|---|---|\n| ratio | 0.104615 |\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r1_close_marker_midline_trailing_checked", expect="FAIL", wrong_on=("main", "r1"),
         why="r1's hole: text AFTER a close marker on the same line was swallowed into the range too",
         doc=_HDR + "<!--derived-->\nThe ratio is 0.104615 here. <!--/derived--> The real accuracy is 0.1525 here.\n"),
    # --- r2: fence/list handling -------------------------------------------------------------------------------
    dict(name="r2_hash_derived_comment_in_fence_read_as_heading", expect="FAIL", wrong_on=("r2",),
         why="r2's hole: a `# derived` comment inside a FENCED code block was read as a markdown heading",
         doc=_HDR + "```python\n# derived thresholds below\nvalue = 1.23456\n```\nThe accuracy was 0.1525 here.\n"),
    # --- r3: fence-open/closed boolean toggle --------------------------------------------------------------
    dict(name="r3_mismatched_fence_swallows_heading", expect="FAIL", wrong_on=("r3",),
         why="r3's hole: a `~~~` fence is not closed by a ` ``` ` fence, so the `## Results` heading after the REAL "
             "close never ended the Derived section",
         doc=_HDR + "## Derived\nratio 0.104615\n~~~\n```\n~~~\n## Results\nThe accuracy was 0.1525 here.\n"),
    # --- r4 (REVIEWED UNSOUND) -----------------------------------------------------------------------------------
    dict(name="r4_unclosed_html_comment_hides_results_heading", expect="FAIL", wrong_on=("r4",),
         why="r4's hole: an unclosed HTML comment swallows the `## Results` heading as inert block content",
         doc=_HDR + "## Derived\nratio 0.104615\n<!-- note, never closed\n## Results\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r4_later_inline_close_hijacks_earlier_standalone_across_heading", expect="FAIL", wrong_on=("r4",),
         why="r4's hole: a late close marker paired with the most recent unpaired STANDALONE opener, stretching a "
             "range across a real heading and a wrong number in between",
         doc=_HDR + "<!--derived-->\nratio 0.104615\n\n## Results\nThe accuracy was 0.1525 here.\n\n"
                    "A later aside adds a note, value 0.207531 here. <!--/derived-->\n"),
    dict(name="r4_h1_derived_heading_oversized_section", expect="FAIL", wrong_on=("r2", "r3", "r4"),
         why="an h1 'Derived' heading has no same-or-higher heading after it in a short doc, so its section ran "
             "to end of document",
         doc=_HDR + "# Derived\nratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_setext_derived_heading_oversized_section", expect="FAIL", wrong_on=("r4",),
         why="r4's hole: a setext ('Derived\\n=======') heading; the wrong number after it must be caught",
         doc=_HDR + "Derived\n=======\nratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_blockquoted_derived_heading_leaks_scope", expect="FAIL", wrong_on=("r4",),
         why="r4's hole: a `## Derived` heading inside a blockquote leaked its section out of the blockquote",
         doc=_HDR + "> ## Derived\n> ratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_list_item_table_leaks_scope_to_sibling_item", expect="FAIL", wrong_on=("main", "r1", "r4"),
         why="r4's hole: a table nested in a list item licensed the whole list, leaking to a sibling item",
         doc=_HDR + "<!--derived-->\n- item one:\n  | metric | value |\n  |---|---|\n  | ratio | 0.104615 |\n"
                    "- item two: accuracy 0.1525\n"),
    # --- round 5's own contract --------------------------------------------------------------------------------
    dict(name="line_marked_derived_number_passes", expect="PASS", wrong_on=(),
         why="the ONE thing the rule allows: the marker in the SAME segment as the number",
         doc=_HDR + "The ratio is 0.104615 here. <!--derived-->\nThe baseline was 0.162500 here.\n"),
    dict(name="table_with_marker_in_same_cell_as_value_passes", expect="PASS", wrong_on=(),
         why="round 6: a table whose derived rows carry the marker IN THE SAME CELL as the value passes",
         doc=_HDR + "| metric | value |\n|---|---|\n| ratio | 0.104615 <!--derived--> |\n"
                    "| gap | 0.207531 <!--derived--> |\n| accuracy | 0.170000 |\n"),
    dict(name="marker_on_wrong_line_does_not_reach_over", expect="FAIL", wrong_on=("main", "r1", "r2", "r3", "r4"),
         why="round 5's own rule: a marker one line away from the number does not reach it",
         doc=_HDR + "<!--derived-->\nThe accuracy was 0.1525 here.\n"),
    dict(name="standalone_marker_and_derived_heading_now_inert_but_do_not_crash",
         expect="FAIL", wrong_on=("main", "r1", "r2", "r3", "r4"),
         why="a standalone marker AND a '## Derived' heading, neither doing anything -- the wrong number is caught",
         doc=_HDR + "## Derived\n<!--derived-->\nThe accuracy was 0.1525 here.\n"),
    dict(name="low_coverage_overmarked", expect="FAIL", wrong_on=("main",), expect_reason="low_coverage",
         why="a substantial doc that marks (almost) every claim derived fails on LOW COVERAGE",
         doc=_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1)
                                for i in range(85)) + "\n"),
    # --- round 6 (round 5's review) --------------------------------------------------------------------------
    dict(name="cap_exempts_only_first_8_numbers_on_a_marked_line", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: a marked line's free exemption is capped at 8 -- a 9th (wrong) number is checked",
         doc=_HDR + "The values are 0.100001, 0.100002, 0.100003, 0.100004, 0.100005, 0.100006, 0.100007, "
                    "0.100008, and 0.1525 here. <!--derived-->\n"),
    dict(name="table_row_marker_exempts_only_its_own_cell", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6, the incident repro: a marker alone in a trailing cell used to exempt the WHOLE row",
         doc=_HDR + "| 42 | 0.1525 | 0.104615 | <!--derived--> |\n"),
    dict(name="br_split_line_marker_does_not_reach_the_other_side", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: a marker before a <br> must not reach a wrong number after it",
         doc=_HDR + "ratio 0.104615 <!--derived--><br>accuracy 0.1525\n"),
    dict(name="underscore_emphasis_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: `_0.1525_` (emphasis) used to be invisible to the number regex",
         doc=_HDR + "The accuracy was _0.1525_ here.\n"),
    dict(name="glued_unit_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: a unit glued onto the number (`0.1525ms`) used to hide it",
         doc=_HDR + "The latency was 0.1525ms here.\n"),
    dict(name="leading_dot_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: `.1525` (no leading zero) used to be invisible",
         doc=_HDR + "The drop was .1525 here.\n"),
    dict(name="markdown_escaped_dot_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: `0\\.1525` broke the digit run",
         doc=_HDR + "The accuracy was 0\\.1525 here.\n"),
    dict(name="html_entity_dot_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: `0&#46;1525` broke the digit run",
         doc=_HDR + "The accuracy was 0&#46;1525 here.\n"),
    dict(name="empty_comment_mid_number_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: `0.15<!---->25` split the number into two unmatched fragments",
         doc=_HDR + "The accuracy was 0.15<!---->25 here.\n"),
    dict(name="empty_span_mid_number_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: an empty <span></span> spliced into the digits",
         doc=_HDR + "The accuracy was 0.15<span></span>25 here.\n"),
    dict(name="zero_width_space_mid_number_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: a zero-width space (U+200B, Cf) inside the digits",
         doc=_HDR + "The accuracy was 0.15\u200b25 here.\n"),
    dict(name="soft_hyphen_mid_number_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: a soft hyphen (U+00AD, Cf) inside the digits",
         doc=_HDR + "The accuracy was 0.15\u00ad25 here.\n"),
    dict(name="scientific_notation_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: `1.525e-1` (= 0.1525) was never matched",
         doc=_HDR + "The accuracy was 1.525e-1 here.\n"),
    dict(name="typographic_minus_sign_flip_no_longer_passes", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: U+2212 was dropped, so `\u22120.1625` matched the POSITIVE baseline",
         doc=_HDR + "The delta was \u22120.1625 here.\n"),
    dict(name="en_dash_sign_flip_no_longer_passes", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: U+2013 EN DASH, the same dropped-sign bug",
         doc=_HDR + "The delta was –0.1625 here.\n"),
    dict(name="comment_hidden_decoys_no_longer_pad_coverage", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         expect_reason="low_coverage",
         why="round 6: copies of a real value pasted inside HTML comments used to pad the checked fraction",
         doc=(_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1)
                                 for i in range(80))
              + "\n\n" + "\n".join("<!-- padding citation of 0.170000 -->" for _ in range(6)) + "\n")),
    dict(name="synthesis_without_closed_frontmatter_is_not_exempt", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: `claim_check: synthesis` after an UNCLOSED frontmatter block no longer exempts",
         doc="---\ntitle: not really frontmatter, never closed\n\n# A doc\n\nArtifact: `%(art)s`\n\n"
             "claim_check: synthesis\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_without_reason_is_not_exempt", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: the flag without a `claim_check_reason:` no longer exempts",
         doc="---\nclaim_check: synthesis\n---\n\n# A doc\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_barred_by_verdict_title", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="round 6: a verdict-bearing title (GO) is BARRED from the synthesis escape",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes several prior runs\n---\n\n"
             "# Lane A 6-seed GO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_with_closed_frontmatter_and_reason_passes", expect="PASS", wrong_on=(),
         why="the escape still works for a genuine literature doc: closed frontmatter, a reason, a neutral title",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes several prior runs, no new measurements\n"
             "---\n\n# A literature summary\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),

    # =========================================================================================================
    # ROUND 7 -- one or more cases per issue of round 6's review (research/coordination/claimcheck_r7_review_
    # issues.json). Each `wrong_on` is re-derived from git by the test suite, r6 included.
    # =========================================================================================================
    # --- issue 1: a single broad citation matches wrong numbers by chance -------------------------------------
    dict(name="broad_single_artifact_fails_as_too_broad", issue=1, expect="FAIL", expect_reason="too_broad",
         wrong_on=_ALL,
         why="issue 1: ONE cited file holding a dense series (10,000 values, one per 0.0001) accepts the wrong "
             "0.1523 by pure chance; round 6's value cap was checked only BETWEEN files (and 10,000 is under it), "
             "and a cap is the wrong tool anyway -- the per-doc chance-match rate (~100% here) fails the doc",
         artifact={"series": [round(i / 10000.0, 4) for i in range(10000)]},
         doc=_HDR + "The accuracy was 0.1523 here.\n"),
    # --- issue 2: table rows in other GFM forms, and other line-break tags ---------------------------------------
    dict(name="pipeless_table_row_marker_exempts_only_its_own_cell", issue=2, expect="FAIL", wrong_on=_ALL,
         why="issue 2: a GFM table row WITHOUT leading pipes (under a `---|---` delimiter row) is still a row; "
             "the trailing marker must not reach the wrong 0.1525 in an earlier cell",
         doc=_HDR + "seed | acc | ratio | note\n---|---|---|---\n42 | 0.1525 | 0.104615 | <!--derived-->\n"),
    dict(name="blockquoted_table_row_marker_exempts_only_its_own_cell", issue=2, expect="FAIL", wrong_on=_ALL,
         why="issue 2: a table row inside a blockquote (`> | ... |`) is still a row",
         doc=_HDR + "> | seed | acc | ratio | note |\n> |---|---|---|---|\n> | 42 | 0.1525 | 0.104615 | "
                    "<!--derived--> |\n"),
    dict(name="br_with_attributes_splits_the_line", issue=2, expect="FAIL", wrong_on=_ALL,
         why="issue 2: `<br class=x>` is a line break too (round 6's regex matched only `<br>`/`<br/>`)",
         doc=_HDR + "ratio 0.104615 <!--derived--><br class=x>accuracy 0.1525\n"),
    dict(name="paragraph_close_open_splits_the_line", issue=2, expect="FAIL", wrong_on=_ALL,
         why="issue 2: `</p><p>` renders as two paragraphs; a marker in one must not reach the other",
         doc=_HDR + "ratio 0.104615 <!--derived--></p><p>accuracy 0.1525\n"),
    # --- issue 3: round 6's normalization GLUED a number to the character before it ------------------------------
    dict(name="star_glued_word_exposes_number", issue=3, expect="FAIL", wrong_on=("r6",),
         why="issue 3: deleting `*` turned `gain*0.1525` into `gain0.1525`, an identifier -- a separator does not",
         doc=_HDR + "The gain*0.1525 here.\n"),
    dict(name="bold_word_glued_number_exposed", issue=3, expect="FAIL", wrong_on=("r6",),
         why="issue 3: `**acc**0.1525` renders as acc0.1525 with the number visible",
         doc=_HDR + "Final **acc**0.1525 here.\n"),
    dict(name="entity_letter_before_number_exposed", issue=3, expect="FAIL", wrong_on=("r6",),
         why="issue 3: `&Delta;0.1525` decodes to a NON-ASCII letter, which the old Unicode \\w lookbehind treated "
             "as an identifier character",
         doc=_HDR + "The shift &Delta;0.1525 here.\n"),
    dict(name="escaped_star_before_number_exposed", issue=3, expect="FAIL", wrong_on=("r6",),
         why="issue 3: `x\\*0.1525` is a LITERAL asterisk; round 6 unescaped then deleted it, gluing x0.1525",
         doc=_HDR + "The product x\\*0.1525 here.\n"),
    dict(name="star_before_signed_number_keeps_sign", issue=3, expect="FAIL", wrong_on=("r6",),
         why="issue 3: `n*-0.1625` glued to `n-0.1625` read as +0.1625 (the positive baseline)",
         doc=_HDR + "The term n*-0.1625 here.\n"),
    dict(name="comment_opener_in_code_span_leaves_number_visible", issue=3, expect="FAIL", wrong_on=("r6",),
         why="issue 3: `` `<!--` 0.1525 `-->` `` is two code spans around VISIBLE text; round 6 stripped it as a "
             "comment",
         doc=_HDR + "Write `<!--` 0.1525 `-->` here.\n"),
    dict(name="escaped_comment_opener_leaves_number_visible", issue=3, expect="FAIL", wrong_on=("r6",),
         why="issue 3: `\\<!-- 0.1525 -->` is literal visible text, not a comment",
         doc=_HDR + "Literal \\<!-- 0.1525 --> here.\n"),
    dict(name="multiplication_star_is_not_glued", issue=3, expect="PASS", wrong_on=("r6",),
         why="issue 3, the corpus instance: `4*0.170` is a multiplication; round 6 glued it into 40.170",
         doc=_HDR + "The product 4*0.170 here.\n"),
    # --- issue 4: dash variants, a letter before a signed number, invisible characters, tags, decimal points ------
] + [
    dict(name="minus_variant_u%04x_keeps_sign" % ord(c), issue=4, expect="FAIL", wrong_on=_ALL,
         why="issue 4: U+%04X before the digits is a minus sign; round 6 read `%s0.1625` as +0.1625" % (ord(c), c),
         doc=_HDR + "The delta was " + c + "0.1625 here.\n")
    for c in ("\u2010", "\u2011", "\u2012", "—", "\ufe63", "\uff0d", "\u02d7", "\u2796")
] + [
    dict(name="letter_before_minus_sign_keeps_sign", issue=4, expect="FAIL", wrong_on=_ALL,
         why="issue 4: `Δ\u22120.1625` -- a letter directly before a signed number; the sign must survive",
         doc=_HDR + "The shift Δ\u22120.1625 here.\n"),
] + [
    dict(name="invisible_u%04x_mid_number" % ord(c), issue=4, expect="FAIL", wrong_on=_ALL,
         why="issue 4: U+%04X is invisible but not category Cf; inside the digits it hid the number" % ord(c),
         doc=_HDR + "The accuracy was 0.15" + c + "25 here.\n")
    for c in ("\u034f", "\ufe0f", "\ufe00", "\u3164", "\u115f")
] + [
    dict(name="tag_%s_mid_number" % tag_name, issue=4, expect="FAIL", wrong_on=_ALL,
         why="issue 4: a non-empty or non-span inline tag (%s) inside the digits hid the number" % tag,
         doc=_HDR + "The accuracy was " + tag + " here.\n")
    for tag_name, tag in (("b", "0.15<b>25</b>"), ("wbr", "0.15<wbr>25"), ("i", "0.15<i></i>25"),
                          ("sup", "0.15<sup></sup>25"), ("a", "0.15<a></a>25"))
] + [
    dict(name="decimal_point_u%04x" % ord(c), issue=4, expect="FAIL", wrong_on=_ALL,
         why="issue 4: U+%04X is read as a decimal point; `0%s1525` was invisible" % (ord(c), c),
         doc=_HDR + "The accuracy was 0" + c + "1525 here.\n")
    for c in ("\uff0e", "\u2024", "\ufe52", "\u00b7")
] + [
    # --- issue 5: the synthesis verdict bar was bypassable -----------------------------------------------------
    dict(name="synthesis_frontmatter_title_beats_leading_notes_h1", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: a leading `# Notes` H1 won over the real (frontmatter) title stating GO",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\ntitle: Lane A 6-seed GO\n---\n\n"
             "# Notes\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_later_heading_verdict_bars", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: a leading `# Notes` H1 hid a later heading stating the verdict",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n# Notes\n\n"
             "## Result: 6-seed GO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_hash_line_in_code_fence_is_not_the_title", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: a `# comment` inside a code fence was taken as the title",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n```bash\n# run it\n```\n\n"
             "# Lane A 6-seed GO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_yaml_comment_is_not_the_title", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: a `# comment` inside the YAML frontmatter was taken as the title",
         doc="---\n# yaml comment\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n"
             "# Lane A 6-seed GO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_setext_title_verdict_bars", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: a setext H1 title was never read",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\nLane A 6-seed GO\n"
             "================\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_lowercase_verdict_bars", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: the bar was case-sensitive (`no-go` passed)",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n# Lane A: no-go at 6 seeds"
             "\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_nogo_without_hyphen_bars", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: `NOGO` (no hyphen) matched neither GO nor NO-GO",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n# Lane A NOGO\n\n"
             "Artifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_zero_width_verdict_bars", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: a zero-width space inside the verdict word (`G\\u200bO`) hid it",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n# Lane A G\u200bO\n\n"
             "Artifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_empty_reason_does_not_capture_next_key", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: `\\s*` crossed the newline, so an EMPTY `claim_check_reason:` captured the next key",
         doc="---\nclaim_check: synthesis\nclaim_check_reason:\nlane: gap#5\n---\n\n# A literature summary\n\n"
             "Artifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_filename_verdict_bars", issue=5, expect="FAIL", wrong_on=_ALL,
         filename="survey-lane-a-6seed-GO.md",
         why="issue 5: the filename (what retrieval surfaces) was never read",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n# A literature summary\n\n"
             "Artifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_frontmatter_verdict_field_bars", issue=5, expect="FAIL", wrong_on=_ALL,
         why="issue 5: a frontmatter `verdict:` field was never read",
         doc="---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\nverdict: GO\n---\n\n"
             "# A literature summary\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    # --- issue 6: the row-trailing marker is inert -- say so, precisely ---------------------------------------------
    dict(name="row_trailing_marker_warns_that_it_exempts_nothing", issue=6, expect="FAIL", wrong_on=_ALL,
         expect_output=("exempts NOTHING", "in the SAME table cell or <br>-segment as the number"),
         why="issue 6: a marker alone in a row's last cell is inert; the author must be TOLD so (WARNING), and the "
             "closing message must name the cell rule, not 'same line'",
         doc=_HDR + "| seed | acc | ratio | note |\n|---|---|---|---|\n| 42 | 0.170 | 0.1525 | <!--derived--> |\n"),
    # --- issue 7: a correct rounding is not a candidate error, and the report says which rule matched ----------
    dict(name="correct_rounding_is_reported_as_rounding", issue=7, expect="PASS", wrong_on=_ALL,
         expect_output="1 by rounding",
         why="issue 7: round 6's retro called flagged numbers like this 'candidate real errors'; 1.235 is the "
             "correct 3-decimal rounding of the cited 1.23456789",
         artifact={"loss": 1.23456789},
         doc=_HDR + "The loss was 1.235 here.\n"),
    # --- issue 8: false positives on legitimate docs -------------------------------------------------------------
    dict(name="three_decimal_rounding_of_four_decimal_value_passes", issue=8, expect="PASS", wrong_on=_ALL,
         why="issue 8: relative tolerance 1e-4 rejected 0.477 for a cited 0.4774",
         artifact={"acc": 0.4774}, doc=_HDR + "The accuracy was 0.477 here.\n"),
    dict(name="arxiv_id_is_not_a_measurement", issue=8, expect="PASS", wrong_on=_ALL,
         why="issue 8: an arXiv id (2403.12345) is an identifier, not a measurement",
         doc=_HDR + "Method from arXiv:2403.12345; accuracy 0.170 here.\n"),
    dict(name="doi_is_not_a_measurement", issue=8, expect="PASS", wrong_on=_ALL,
         why="issue 8: a DOI prefix (10.1038) is an identifier",
         doc=_HDR + "Ernst & Banks 2002, doi:10.1038/415429a; accuracy 0.170 here.\n"),
    dict(name="url_number_is_not_a_measurement", issue=8, expect="PASS", wrong_on=_ALL,
         why="issue 8: a number inside a URL is an identifier",
         doc=_HDR + "See https://arxiv.org/abs/2403.12345 -- accuracy 0.170 here.\n"),
    dict(name="magnitude_suffix_is_scaled_not_stripped", issue=8, expect="PASS", wrong_on=("r6",),
         why="issue 8: `1.088B params` was read as 1.088",
         artifact={"n_params": 1088000000}, doc=_HDR + "A 1.088B params model here.\n"),
    dict(name="precision_aware_is_tighter_at_four_decimals", issue=8, expect="FAIL", wrong_on=_ALL,
         why="issue 8: at |x| >= 0.5 a 4-decimal claim got a relative window WIDER than its stated precision -- "
             "12.3456 passed for a cited 12.3449",
         artifact={"x": 12.3449}, doc=_HDR + "The value was 12.3456 here.\n"),
    # --- issue 9: hidden carriers, coverage --------------------------------------------------------------------
    dict(name="multiline_comment_values_do_not_pad_coverage", issue=9, expect="FAIL", expect_reason="low_coverage",
         wrong_on=_ALL,
         why="issue 9: 11 DISTINCT real values inside a MULTI-LINE comment counted as checked (round 6 stripped "
             "comments per line)",
         artifact={"v": [round(0.3001 + i / 10000.0, 4) for i in range(11)]},
         doc=(_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(80))
              + "\n\n<!--\n" + "\n".join("%.4f" % (0.3001 + i / 10000.0) for i in range(11)) + "\n-->\n")),
    dict(name="linkref_comment_values_do_not_pad_coverage", issue=9, expect="FAIL", expect_reason="low_coverage",
         wrong_on=_ALL,
         why="issue 9: `[//]: # (...)` is an invisible link-reference 'comment'",
         artifact={"v": [round(0.3001 + i / 10000.0, 4) for i in range(11)]},
         doc=(_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(80))
              + "\n\n" + "\n".join("[//]: # (padding %.4f)" % (0.3001 + i / 10000.0) for i in range(11)) + "\n")),
    dict(name="hidden_div_values_do_not_pad_coverage", issue=9, expect="FAIL", expect_reason="low_coverage",
         wrong_on=_ALL,
         why="issue 9: `<div hidden>` content is invisible",
         artifact={"v": [round(0.3001 + i / 10000.0, 4) for i in range(11)]},
         doc=(_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(80))
              + "\n\n<div hidden>\n" + "\n".join("%.4f" % (0.3001 + i / 10000.0) for i in range(11))
              + "\n</div>\n")),
    dict(name="derived_note_comment_is_not_scanned", issue=9, expect="PASS", wrong_on=_ALL,
         why="issue 9: a multi-line `<!--derived: ...-->` note is the marker plus the author's derivation -- its "
             "own numbers are not claims (4 corpus docs were false-flagged by it)",
         doc=_HDR + "The ratio is 0.104615 here. <!--derived: 0.104615 = 0.17 / 1.625, and\n"
                    "the gap 0.207531 below is its double -->\nThe baseline was 0.162500 here.\n"),
    dict(name="hidden_citation_in_hidden_div_is_ignored", issue=9, expect="FAIL", wrong_on=_ALL,
         why="issue 9: a citation inside `<div hidden>` is one no reader can see",
         artifact={"accuracy": 0.17, "baseline": 0.1625},
         must_flag=(0.1625,),
         doc="# Some finding\n\n<div hidden>see `%(art)s`</div>\n\nThe baseline was 0.162500 here.\n"),
    dict(name="seventy_nine_all_marked_claims_fail_coverage", issue=9, expect="FAIL", expect_reason="low_coverage",
         wrong_on=("main", "r5", "r6"),
         why="issue 9: a doc with 79 numeric claims, EVERY one marked derived, passed (the floor was 80); round 7's "
             "floor is 30, above the largest legitimately all-marked doc since 2026-09-01 (27 claims)",
         doc=_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(79)) + "\n"),
    # --- issue 10: pin round 6's fixes in THIS registry ---------------------------------------------------------
    dict(name="visible_copies_of_one_value_count_once", issue=10, expect="FAIL", expect_reason="low_coverage",
         wrong_on=_ALL_BEFORE_R6,
         why="issue 10: round 6's DISTINCT counting (6 VISIBLE copies of 0.170000 count once) was pinned by nothing",
         doc=(_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(80))
              + "\n\n" + "\n".join("The accuracy was 0.170000 here." for _ in range(6)) + "\n")),
    dict(name="hidden_citation_in_html_comment_is_ignored", issue=10, expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="issue 10: round 6's comment-citation fix, moved from pytest into this registry with its own artifact",
         artifact={"accuracy": 0.17, "baseline": 0.1625},
         must_flag=(0.1625,),
         doc="# Some finding\n\n<!-- see `%(art)s` for context -->\n\nThe baseline was 0.162500 here.\n"),
    dict(name="hidden_citation_in_linkref_comment_is_ignored", issue=10, expect="FAIL", wrong_on=_ALL,
         why="issue 10: the same hidden citation in a `[//]: # (...)` link-reference comment",
         artifact={"accuracy": 0.17, "baseline": 0.1625},
         must_flag=(0.1625,),
         doc="# Some finding\n\n[//]: # (see `%(art)s`)\n\nThe baseline was 0.162500 here.\n"),
    # --- issue 11: the CCT gate must report a broken instrument verbatim -----------------------------------------
    dict(name="cct_gate_reports_broken_instrument_verbatim", issue=11, kind="gate", expect="PASS", wrong_on=_ALL,
         why="issue 11: with claim_check's selftest broken, the gate must pass the problems through VERBATIM "
             "labelled BROKEN INSTRUMENT, and must not fail its OWN selftest (which made the registry skip "
             "check() and mislabel the regression a 'false positive')", doc=""),
    # --- round 7's own verification (adversarial probes of the first r7 commit, 00aa451da) ------------------------
    dict(name="nonempty_comment_mid_number_is_read_as_the_reader_sees_it", expect="FAIL",
         wrong_on=_ALL_BEFORE_R6,
         why="r7 regression caught in verification: keeping comment CONTENT for scanning left `0.15<!-- x -->25` "
             "as three fragments, while a reader sees 0.1525 (r6 stripped the comment and caught it); the "
             "reader's reading, with every hidden carrier rendered as nothing, is now scanned too",
         doc=_HDR + "The accuracy was 0.15<!-- x -->25 here.\n"),
    dict(name="hidden_element_mid_number_is_read_as_the_reader_sees_it", expect="FAIL", wrong_on=_ALL,
         why="the same with a hidden element: `0.15<div hidden>9</div>25` shows 0.1525",
         doc=_HDR + "The accuracy was 0.15<div hidden>9</div>25 here.\n"),
    dict(name="pipe_inside_comment_still_splits_a_table_row", expect="FAIL", wrong_on=_ALL_BEFORE_R6,
         why="r7 regression caught in verification: GFM splits a row into cells BEFORE inline parsing, so a `|` "
             "inside a comment separates cells for a reader; cutting on the comment-blanked line merged the wrong "
             "0.1525 into the marker's cell",
         doc=_HDR + "| a | b | c |\n|---|---|---|\n| 42 | 0.1525 <!-- | --> 0.104615 <!--derived--> |\n"),
    dict(name="measurement_with_a_slash_unit_is_not_a_doi", expect="FAIL", wrong_on=(),
         why="r7 regression caught in verification: `10.1525/s` matched the bare-DOI pattern and was skipped; a "
             "bare DOI now needs a 6+ character suffix holding a letter AND a digit",
         doc=_HDR + "Throughput 10.1525/s here.\n", must_flag=(10.1525,)),
    dict(name="nonascii_letter_underscore_number_is_a_number", expect="FAIL", wrong_on=_ALL,
         why="found by the fuzz (tests/test_claim_check_fuzz.py): `\\u0394_0.1525_` -- an intraword `_` is an "
             "identifier character only after an ASCII letter/digit (`foo_0.125`); after a symbol letter it is "
             "emphasis or a separator and the number is a claim",
         doc=_HDR + "The shift Δ_0.1525_ here.\n"),
    dict(name="bidi_override_is_unreadable", expect="FAIL", wrong_on=_ALL,
         why="a bidirectional override (`\\u202e5251.0\\u202c`) DISPLAYS as 0.1525 while storing 5251.0 -- no "
             "reading of the stored text is the reader's; the doc is refused as unreadable",
         doc=_HDR + "The accuracy was \u202e5251.0\u202c here.\n", expect_output="bidirectional control",
         expect_reason="unreadable"),
    # --- round 7 guards: behaviour kept on purpose -----------------------------------------------------------------
    dict(name="hyphen_range_is_not_a_sign", expect="PASS", wrong_on=(),
         why="a '-' after a digit is a range or a subtraction, not a sign",
         doc=_HDR + "Between 0.1625-0.170 here.\n"),
    dict(name="explicit_ellipsis_states_truncation", expect="PASS", wrong_on=(),
         why="`3.490537...` states a TRUNCATION (the correct rounding of 3.4905378 is 3.490538); the ellipsis is "
             "the author saying so, and rule A accepts it only then",
         artifact={"x": 3.4905378}, doc=_HDR + "Seed 42 read `3.490537...` here.\n"),
    dict(name="truncation_without_ellipsis_is_a_misstatement", issue=8, expect="FAIL", must_flag=(3.490537,),
         wrong_on=_ALL,
         why="the same truncation WITHOUT the ellipsis claims 3.4905365-3.4905375, which excludes the cited "
             "3.4905378 -- the relative tolerance used to accept it",
         artifact={"x": 3.4905378}, doc=_HDR + "Seed 42 read 3.490537 here.\n"),
    dict(name="punctuation_dash_glued_to_number_reads_as_minus", expect="FAIL", wrong_on=_ALL,
         expect_output="reads as a MINUS sign",
         why="fail-closed choice: an em dash glued to a number is read as a minus; the report says why",
         doc=_HDR + "The baseline—0.1625—was stable.\n"),
]

_HISTORY_SHAS = {"main": "7e2edc08e", "r1": "d4959ecb0", "r2": "6abb28469", "r3": "662e167e8", "r4": "214e509bf",
                 "r5": "4fda849d4", "r6": "f2b7db2b4"}
_DEFAULT_ARTIFACT = {"accuracy": 0.17, "baseline": 0.1625}
WRONG_VALUES = {0.1525, 0.14, 1.23456, -0.1525, -0.1625, 12.3456}


def _write_case(d, case):
    """Write one selftest case (and its artifact) under directory `d`, which must be inside ROOT: PATH_RE drops
    a leading '/', so a real finding's ROOT-relative citation is the only form that resolves the same way."""
    art_obj = case.get("artifact")
    art_abs = os.path.join(d, ("art_%s.json" % case["name"]) if art_obj is not None else "art.json")
    if not os.path.exists(art_abs):
        with open(art_abs, "w") as fh:
            json.dump(art_obj if art_obj is not None else _DEFAULT_ARTIFACT, fh)
    art = os.path.relpath(art_abs, ROOT).replace(os.sep, "/")
    path = os.path.join(d, case.get("filename", case["name"] + ".md"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(case["doc"] % {"art": art})
    return path


def _gate_case_ok(gate_mod):
    """Issue 11, evaluated against ANY revision of tools/gates/claim_check_selftest.py: with a BROKEN claim_check
    injected, the gate's check() must pass the problems through VERBATIM labelled BROKEN INSTRUMENT, and the
    gate's own selftest() must NOT fail (a real regression is the instrument's problem, not the wrapper's)."""
    import types
    broken = types.SimpleNamespace(
        selftest=lambda: ["SELFTEST BROKEN: case demo expected FAIL, got PASS (demo)"],
        SELFTEST_CASES=[dict(name="demo", expect="FAIL", why="demo", doc="")])
    healthy = types.SimpleNamespace(selftest=lambda: [], SELFTEST_CASES=[dict(name="demo", expect="FAIL",
                                                                              why="demo", doc="")])
    saved = getattr(gate_mod, "claim_check", None)
    try:
        gate_mod.claim_check = broken
        probs = list(gate_mod.check(None))
        st = list(gate_mod.selftest())
        gate_mod.claim_check = healthy
        clean = list(gate_mod.check(None))
    except Exception as e:
        return False, "gate raised %s: %s" % (type(e).__name__, e)
    finally:
        gate_mod.claim_check = saved
    ok = (bool(probs) and all("BROKEN INSTRUMENT" in p for p in probs)
          and any("SELFTEST BROKEN: case demo expected FAIL, got PASS (demo)" in p for p in probs)
          and not st and not clean)
    return ok, "check(broken)=%s selftest(broken)=%s check(healthy)=%s" % (probs, st, clean)


def _expected_outputs(case):
    eo = case.get("expect_output") or ()
    return (eo,) if isinstance(eo, str) else tuple(eo)


def _case_outcome(case, mod, casedir):
    """(verdict, output) for one case against a claim_check MODULE of any revision -- only `check()` is assumed."""
    p = _write_case(casedir, case)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = mod.check(p, verbose=True)
        got = "FAIL" if rc else "PASS"
    except Exception as e:
        # A crash also blocks a commit (an uncaught exception exits non-zero) -- equivalent to a FAIL return.
        got = "FAIL"
        buf.write("CRASH %s: %s" % (type(e).__name__, e))
    return got, buf.getvalue()


def selftest():
    """Same contract as `tools/gates/*.selftest()`: a list of problems, empty means the check is trustworthy.
    Runs every SELFTEST_CASES entry in both directions: a FAIL case must fail for its DESIGNATED reason (its wrong
    number flagged, or LOW COVERAGE, or TOO BROAD), a PASS case must flag nothing at all."""
    import tempfile
    problems = []
    me = sys.modules[__name__]
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".claim_check_selftest_") as d:
        for case in SELFTEST_CASES:
            if case.get("kind") == "gate":
                if ROOT not in sys.path:
                    sys.path.insert(0, ROOT)
                import importlib
                gate = importlib.import_module("tools.gates.claim_check_selftest")
                ok, detail = _gate_case_ok(gate)
                if not ok:
                    problems.append("SELFTEST BROKEN: case %s (%s): %s" % (case["name"], case["why"], detail))
                continue
            got, out = _case_outcome(case, me, d)
            if got != case["expect"]:
                problems.append("SELFTEST BROKEN: case %s expected %s, got %s (%s)"
                                % (case["name"], case["expect"], got, case["why"]))
                continue
            missing_out = [o for o in _expected_outputs(case) if o not in out]
            if missing_out:
                problems.append("SELFTEST BROKEN: case %s: report lacks %r (%s)"
                                % (case["name"], missing_out, case["why"]))
                continue
            r = _scan(_write_case(d, case))
            flagged = {round(v, 6) for _ln, v, _c in r["unsupported"]}
            reason = case.get("expect_reason")
            if case["expect"] == "FAIL":
                if case.get("must_flag"):
                    if not {round(v, 6) for v in case["must_flag"]} <= flagged:
                        problems.append("SELFTEST BROKEN: case %s failed but did not flag %s: flagged=%s"
                                        % (case["name"], case["must_flag"], flagged))
                elif reason == "too_broad":
                    if not r["too_broad"]:
                        problems.append("SELFTEST BROKEN: case %s failed but not as TOO BROAD (chance=%s)"
                                        % (case["name"], r["chance"]))
                elif reason == "low_coverage":
                    if not r["low_coverage"]:
                        problems.append("SELFTEST BROKEN: case %s failed but not on LOW COVERAGE" % case["name"])
                elif reason == "unreadable":
                    if not r.get("unreadable"):
                        problems.append("SELFTEST BROKEN: case %s failed but was not refused as UNREADABLE"
                                        % case["name"])
                elif not (flagged & WRONG_VALUES) and not r["low_coverage"]:
                    problems.append("SELFTEST BROKEN: case %s failed but never flagged its designated wrong "
                                    "number (%s): flagged=%s" % (case["name"], sorted(WRONG_VALUES), flagged))
            elif flagged or r["low_coverage"] or r["too_broad"]:
                problems.append("SELFTEST BROKEN: case %s is supposed to PASS clean but flagged: %s"
                                % (case["name"], sorted(flagged)))

        # ROUND 6 (issue 1 of round 5's review): the UNREADABLE case needs raw, deliberately-invalid bytes.
        bad_path = os.path.join(d, "invalid_utf8.md")
        art_rel = os.path.relpath(os.path.join(d, "art.json"), ROOT).replace(os.sep, "/")
        with open(bad_path, "wb") as fh:
            fh.write(("# Some finding\n\nArtifact: `%s`\n\nThe accuracy was 0.98" % art_rel).encode("utf-8")
                     + b"\xad" + "76 here.\n".encode("utf-8"))
        r = _scan(bad_path)
        if not r.get("unreadable"):
            problems.append("SELFTEST BROKEN: invalid UTF-8 did not report UNREADABLE (round 5's errors=\"replace\" "
                            "regression is back)")
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
