"""SELFTEST REGISTRY for tools/claim_check.py (round 8). DATA ONLY -- no imports from claim_check.

Every entry is one repro. `tests/test_claim_check_line_only.py` re-runs each against every historical revision in
claim_check._HISTORY_SHAS (main, r1-r7, r8a, loaded from git) and asserts the recorded `wrong_on` equals the set that
ACTUALLY gets the case wrong -- so "this used to pass, now it fails" is RE-DERIVED on every run, never remembered.

Keys: name, expect ('FAIL'|'PASS'), wrong_on, why; optional: doc (`%(art)s` = the cited artifact's path),
artifact (default {"accuracy": 0.17, "baseline": 0.1625}), filename, expect_reason ('low_coverage' | 'too_broad' |
'unreadable' | 'missing'), must_flag (values that must be flagged -- replaces the designated-wrong-value requirement),
expect_warning (substrings a WARNING must contain), expect_output (substrings check()'s report must contain),
raw_files ({name: text} written verbatim next to the doc -- a corrupt artifact; `%(sub)s` in doc is that directory).

Designated WRONG numbers (not in the default artifact): 0.1525, 0.14, 1.23456, -0.1525, -0.1625, 0.153, 0.15925,
10.1525. Legitimately derived placeholders: 0.104615, 0.207531, 0.311079.

No `\\u` escapes are written in this file's source on purpose: invisible and bidirectional characters are built
with chr() so the file itself never carries them.
"""
from __future__ import annotations

U = chr
ZWSP, SHY, BIDI_RLO, BIDI_PDF = U(0x200B), U(0x00AD), U(0x202E), U(0x202C)
MINUS, EN_DASH, EM_DASH = U(0x2212), U(0x2013), U(0x2014)
HYPHEN, MIDDLE_DOT, ARABIC_3, ARABIC_DECIMAL_SEP = U(0x2010), U(0x00B7), U(0x0663), U(0x066B)
BOM = U(0xFEFF)
DELTA = U(0x0394)

_HDR = "# Some finding\n\nArtifact: `%(art)s`\n\n"
_BEFORE_R6 = ("main", "r1", "r2", "r3", "r4", "r5")
_THROUGH_R6 = _BEFORE_R6 + ("r6",)
_ALL_BEFORE_R8 = _THROUGH_R6 + ("r7",)
# 210 offsets (units of 1e-4 around 0.1525) whose values make the EXACT chance rate of a written 0.1525 0.21 -- over
# the 4-decimal limit (0.15) -- while round 8 as reviewed, sampling 100 decoys seeded by the TEXT, rated `0.1525` 0.15
# and `.1525` 0.24 (found by search; see the case below).
_SPELLING_KS = (
    -499, -498, -495, -491, -488, -486, -483, -478, -474, -472, -471, -470, -469, -467, -465, -456,
    -443, -436, -428, -415, -412, -407, -404, -398, -396, -390, -380, -377, -368, -363, -333, -328,
    -326, -323, -319, -315, -310, -306, -296, -290, -286, -279, -276, -273, -268, -266, -265, -264,
    -263, -262, -252, -239, -228, -225, -210, -204, -197, -190, -185, -175, -161, -160, -148, -147,
    -146, -139, -135, -127, -125, -121, -117, -112, -110, -108, -101, -98, -97, -87, -86, -77,
    -76, -74, -68, -64, -57, -52, -51, -44, -40, -37, -31, -30, -20, -17, -9, -8,
    -4, -1, 2, 8, 12, 13, 15, 17, 18, 20, 21, 25, 27, 32, 34, 41,
    52, 53, 54, 55, 62, 64, 65, 67, 68, 70, 75, 83, 84, 92, 93, 99,
    102, 104, 106, 108, 115, 123, 124, 128, 130, 139, 145, 151, 158, 163, 166, 168,
    176, 180, 181, 187, 190, 191, 194, 203, 213, 220, 221, 229, 239, 242, 244, 247,
    251, 256, 259, 261, 262, 273, 274, 280, 281, 283, 286, 290, 295, 297, 298, 303,
    306, 308, 317, 322, 329, 350, 355, 356, 357, 358, 361, 367, 368, 369, 374, 381,
    393, 397, 403, 404, 406, 415, 418, 424, 438, 453, 462, 470, 482, 491, 492, 494,
    496, 497,
)
_SPELLING_POOL = {"sweep": sorted({round(0.1525 + k * 1e-4, 6) for k in _SPELLING_KS} | {0.15252})}
# A sweep with one value every 0.01 (offset 0.0032): a wrong 3-decimal 0.153 lands within +-0.0005 of 0.1532, and a
# random 3-decimal number would match about 1 time in 10 -- under the old flat 0.20 limit, over the 3-decimal 0.04.
_SWEEP_3DEC = {"accuracy": 0.17, "baseline": 0.1625, "sweep": [round(0.0032 + i / 100.0, 4) for i in range(100)]}
_NEG = {"accuracy": 0.17, "delta": -0.1625}
_MARKED_80 = "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(80))
_SYN = "---\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n"

SELFTEST_CASES = [
    # =============================================================================================================
    # main's original hole, and rounds 1-4 (multi-line scope rules)
    # =============================================================================================================
    dict(name="incident_standalone_marker_after_heading", expect="FAIL", wrong_on=('main',),
         why="main's hole: a marker alone right after a `## ` heading exempted the whole section after it",
         doc=_HDR + "## Section A\n<!--derived-->\nRunner: some_runner.py, no numbers in this sentence.\n\n"
                    "The accuracy was 0.1525 here.\n\n## Section B\n<!--derived-->\n"
                    "Runner: some_runner.py, no numbers in this sentence.\n\nThe baseline was 0.140 here.\n"),
    dict(name="r1_h3_heading_does_not_end_scope", expect="FAIL", wrong_on=('main', 'r1'),
         why="r1's hole: only `## ` ended an open scope, so a `### ` heading right after a marker did not",
         doc=_HDR + "<!--derived-->\n### A subheading, not a level-2 one\nThe accuracy was 0.1525 here.\n"),
    dict(name="r1_table_then_wrong_number_no_blank_line", expect="FAIL", wrong_on=('main', 'r1'),
         why="r1's hole: a table right after a marker absorbed a wrong number on the very next line",
         doc=_HDR + "<!--derived-->\n| metric | value |\n|---|---|\n| ratio | 0.104615 |\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r1_close_marker_midline_trailing_checked", expect="FAIL", wrong_on=('main', 'r1'),
         why="r1's hole: text AFTER a close marker on the same line was swallowed into the range too",
         doc=_HDR + "<!--derived-->\nThe ratio is 0.104615 here. <!--/derived--> The real accuracy is 0.1525 here.\n"),
    dict(name="r2_hash_derived_comment_in_fence_read_as_heading", expect="FAIL", wrong_on=('r2',),
         why="r2's hole: a `# derived` comment inside a FENCED code block was read as a markdown heading",
         doc=_HDR + "```python\n# derived thresholds below\nvalue = 1.23456\n```\nThe accuracy was 0.1525 here.\n"),
    dict(name="r3_mismatched_fence_swallows_heading", expect="FAIL", wrong_on=('r3',),
         why="r3's hole: a `~~~` fence is not closed by a ``` fence, so `## Results` never ended the Derived section",
         doc=_HDR + "## Derived\nratio 0.104615\n~~~\n```\n~~~\n## Results\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_unclosed_html_comment_hides_results_heading", expect="FAIL", wrong_on=('r4',),
         why="r4's hole: an unclosed HTML comment swallows the `## Results` heading as inert block content",
         doc=_HDR + "## Derived\nratio 0.104615\n<!-- note, never closed\n## Results\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r4_later_inline_close_hijacks_earlier_standalone_across_heading", expect="FAIL", wrong_on=('r4',),
         why="r4's hole: a late inline close marker paired with the most recent unpaired STANDALONE opener",
         doc=_HDR + "<!--derived-->\nratio 0.104615\n\n## Results\nThe accuracy was 0.1525 here.\n\n"
                    "A later aside adds a note, value 0.207531 here. <!--/derived-->\n"),
    dict(name="r4_h1_derived_heading_oversized_section", expect="FAIL", wrong_on=('r2', 'r3', 'r4'),
         why="an h1 'Derived' heading had no same-or-higher heading after it, so its section ran to end of doc",
         doc=_HDR + "# Derived\nratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_setext_derived_heading_oversized_section", expect="FAIL", wrong_on=('r4',),
         why="r4's hole: a setext ('Derived\\n=======') heading was never handled",
         doc=_HDR + "Derived\n=======\nratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_blockquoted_derived_heading_leaks_scope", expect="FAIL", wrong_on=('r4',),
         why="r4's hole: a `## Derived` heading inside a blockquote leaked its section out of the blockquote",
         doc=_HDR + "> ## Derived\n> ratio 0.104615\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r4_list_item_table_leaks_scope_to_sibling_item", expect="FAIL", wrong_on=('main', 'r1', 'r4'),
         why="r4's hole: a table nested in a list item licensed the whole list around it",
         doc=_HDR + "<!--derived-->\n- item one:\n  | metric | value |\n  |---|---|\n  | ratio | 0.104615 |\n"
                    "- item two: accuracy 0.1525\n"),
    # --- the same-line contract (round 5), both directions ---------------------------------------------------
    dict(name="line_marked_derived_number_passes", expect="PASS", wrong_on=(),
         why="the marker on the SAME physical line as the number exempts it",
         doc=_HDR + "The ratio is 0.104615 here. <!--derived-->\nThe baseline was 0.162500 here.\n"),
    dict(name="table_with_marker_in_same_cell_as_value_passes", expect="PASS", wrong_on=(),
         why="a derived table cell carrying its own marker passes",
         doc=_HDR + "| metric | value |\n|---|---|\n| ratio | 0.104615 <!--derived--> |\n"
                    "| gap | 0.207531 <!--derived--> |\n| accuracy | 0.170000 |\n"),
    dict(name="marker_on_wrong_line_does_not_reach_over", expect="FAIL", wrong_on=('main', 'r1', 'r2', 'r3', 'r4'),
         why="a marker one line away from the number does not reach it",
         doc=_HDR + "<!--derived-->\nThe accuracy was 0.1525 here.\n"),
    dict(name="standalone_marker_and_derived_heading_now_inert_but_do_not_crash", expect="FAIL",
         wrong_on=('main', 'r1', 'r2', 'r3', 'r4'),
         why="a standalone marker AND a '## Derived' heading exempt nothing",
         expect_warning=("standalone marker", "'Derived' heading"),
         doc=_HDR + "## Derived\n<!--derived-->\nThe accuracy was 0.1525 here.\n"),
    dict(name="low_coverage_overmarked", expect="FAIL", wrong_on=('main',), expect_reason="low_coverage",
         why="a substantial doc that marks every claim derived fails on LOW COVERAGE",
         doc=_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1)
                                for i in range(85)) + "\n"),

    # =============================================================================================================
    # round 5's review (fixed first in r6): numbers hidden by markup, the cell rule, synthesis conditions
    # =============================================================================================================
    dict(name="cap_exempts_only_first_8_numbers_on_a_marked_line", expect="FAIL", wrong_on=_BEFORE_R6,
         why="a marker exempts at most 8 numbers -- a 9th (wrong) number is checked",
         expect_warning=("marker cap",),
         doc=_HDR + "The values are 0.100001, 0.100002, 0.100003, 0.100004, 0.100005, 0.100006, 0.100007, "
                    "0.100008, and 0.1525 here. <!--derived-->\n"),
    dict(name="table_row_marker_exempts_only_its_own_cell", expect="FAIL", wrong_on=_BEFORE_R6,
         why="the incident repro: a marker alone in a trailing cell used to exempt the WHOLE row",
         doc=_HDR + "| 42 | 0.1525 | 0.104615 | <!--derived--> |\n"),
    dict(name="br_split_line_marker_does_not_reach_the_other_side", expect="FAIL", wrong_on=_BEFORE_R6,
         why="a marker before a <br> must not reach a wrong number after it",
         doc=_HDR + "ratio 0.104615 <!--derived--><br>accuracy 0.1525\n"),
    dict(name="underscore_emphasis_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`_0.1525_` (emphasis) was invisible to main's `(?<![\\w.])` lookbehind",
         doc=_HDR + "The accuracy was _0.1525_ here.\n"),
    dict(name="glued_unit_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="a unit glued onto the number (`0.1525ms`)",
         doc=_HDR + "The latency was 0.1525ms here.\n"),
    dict(name="leading_dot_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`.1525` (no leading zero)",
         doc=_HDR + "The drop was .1525 here.\n"),
    dict(name="markdown_escaped_dot_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`0\\.1525` broke the digit run",
         doc=_HDR + "The accuracy was 0\\.1525 here.\n"),
    dict(name="html_entity_dot_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`0&#46;1525` renders as 0.1525",
         doc=_HDR + "The accuracy was 0&#46;1525 here.\n"),
    dict(name="empty_comment_mid_number_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`0.15<!---->25` renders as 0.1525",
         doc=_HDR + "The accuracy was 0.15<!---->25 here.\n"),
    dict(name="empty_span_mid_number_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="an empty <span></span> spliced into the digits",
         doc=_HDR + "The accuracy was 0.15<span></span>25 here.\n"),
    dict(name="zero_width_space_mid_number_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="a zero-width space (U+200B) inside the digits",
         doc=_HDR + "The accuracy was 0.15" + ZWSP + "25 here.\n"),
    dict(name="soft_hyphen_mid_number_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="a soft hyphen (U+00AD) inside the digits",
         doc=_HDR + "The accuracy was 0.15" + SHY + "25 here.\n"),
    dict(name="scientific_notation_no_longer_hides_a_wrong_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`1.525e-1` (= 0.1525) was never matched",
         doc=_HDR + "The accuracy was 1.525e-1 here.\n"),
    dict(name="typographic_minus_sign_flip_no_longer_passes", expect="FAIL", wrong_on=_BEFORE_R6,
         why="U+2212 was dropped, so a minus 0.1625 matched the POSITIVE baseline",
         doc=_HDR + "The delta was " + MINUS + "0.1625 here.\n"),
    dict(name="en_dash_sign_flip_no_longer_passes", expect="FAIL", wrong_on=_BEFORE_R6,
         why="U+2013 EN DASH, the same dropped-sign bug",
         doc=_HDR + "The delta was " + EN_DASH + "0.1625 here.\n"),
    dict(name="comment_hidden_decoys_no_longer_pad_coverage", expect="FAIL", wrong_on=_BEFORE_R6,
         expect_reason="low_coverage",
         why="copies of a real value pasted inside HTML comments padded the checked fraction",
         doc=_HDR + _MARKED_80 + "\n\n" + "\n".join("<!-- padding citation of 0.170000 -->" for _ in range(6)) + "\n"),
    dict(name="synthesis_without_closed_frontmatter_is_not_exempt", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`claim_check: synthesis` after an UNCLOSED frontmatter block exempts nothing",
         doc="---\ntitle: not really frontmatter, never closed\n\n# A doc\n\nArtifact: `%(art)s`\n\n"
             "claim_check: synthesis\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_without_reason_is_not_exempt", expect="FAIL", wrong_on=_BEFORE_R6,
         why="the flag without a `claim_check_reason:` exempts nothing",
         doc="---\nclaim_check: synthesis\n---\n\n# A doc\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_barred_by_verdict_title", expect="FAIL", wrong_on=_BEFORE_R6,
         why="a verdict-bearing title (GO) is BARRED from the synthesis escape",
         doc=_SYN + "---\n\n# Lane A 6-seed GO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_with_closed_frontmatter_and_reason_passes", expect="PASS", wrong_on=(),
         why="the escape still works for a genuine literature doc: closed frontmatter, a reason, a neutral title",
         doc=_SYN + "---\n\n# A literature summary\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),

    # =============================================================================================================
    # round 6's review (fixed first in r7)
    # =============================================================================================================
    dict(name="broad_single_artifact_fails_as_too_broad", expect="FAIL", expect_reason="too_broad",
         wrong_on=_THROUGH_R6,
         why="ONE cited file holding a dense series (10,000 values, one per 0.0001) accepts a wrong 4-decimal number "
             "by chance; the claim's own chance-match rate is ~100%",
         artifact={"series": [round(i / 10000.0, 4) for i in range(10000)]},
         doc=_HDR + "The accuracy was 0.1523 here.\n"),
    dict(name="pipeless_table_row_marker_exempts_only_its_own_cell", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a GFM row WITHOUT leading pipes is still a row",
         doc=_HDR + "seed | acc | ratio | note\n---|---|---|---\n42 | 0.1525 | 0.104615 | <!--derived-->\n"),
    dict(name="blockquoted_table_row_marker_exempts_only_its_own_cell", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a table row inside a blockquote is still a row",
         doc=_HDR + "> | seed | acc | ratio | note |\n> |---|---|---|---|\n> | 42 | 0.1525 | 0.104615 | "
                    "<!--derived--> |\n"),
    dict(name="br_with_attributes_splits_the_line", expect="FAIL", wrong_on=_THROUGH_R6,
         why="`<br class=x>` is a line break too",
         doc=_HDR + "ratio 0.104615 <!--derived--><br class=x>accuracy 0.1525\n"),
    dict(name="paragraph_close_open_splits_the_line", expect="FAIL", wrong_on=_THROUGH_R6,
         why="`</p><p>` renders as two paragraphs",
         doc=_HDR + "ratio 0.104615 <!--derived--></p><p>accuracy 0.1525\n"),
    dict(name="star_glued_word_exposes_number", expect="FAIL", wrong_on=('r6',),
         why="r6 deleted `*`, turning `gain*0.1525` into the identifier `gain0.1525`",
         doc=_HDR + "The gain*0.1525 here.\n"),
    dict(name="bold_word_glued_number_exposed", expect="FAIL", wrong_on=('r6',),
         why="`**acc**0.1525` renders as acc0.1525 with the number visible",
         doc=_HDR + "Final **acc**0.1525 here.\n"),
    dict(name="entity_letter_before_number_exposed", expect="FAIL", wrong_on=('r6',),
         why="`&Delta;0.1525` decodes to a NON-ASCII letter",
         doc=_HDR + "The shift &Delta;0.1525 here.\n"),
    dict(name="escaped_star_before_number_exposed", expect="FAIL", wrong_on=('r6',),
         why="`x\\*0.1525` is a literal asterisk",
         doc=_HDR + "The product x\\*0.1525 here.\n"),
    dict(name="star_before_signed_number_keeps_sign", expect="FAIL", wrong_on=('r6',),
         why="`n*-0.1625` must read as -0.1625, not the positive baseline",
         doc=_HDR + "The term n*-0.1625 here.\n"),
    dict(name="comment_opener_in_code_span_leaves_number_visible", expect="FAIL", wrong_on=('r6',),
         why="`` `<!--` 0.1525 `-->` `` is two code spans around VISIBLE text",
         doc=_HDR + "Write `<!--` 0.1525 `-->` here.\n"),
    dict(name="escaped_comment_opener_leaves_number_visible", expect="FAIL", wrong_on=('r6',),
         why="`\\<!-- 0.1525 -->` is literal visible text",
         doc=_HDR + "Literal \\<!-- 0.1525 --> here.\n"),
    dict(name="multiplication_star_is_not_glued", expect="PASS", wrong_on=('r6',),
         why="`4*0.170` is a multiplication, not 40.170",
         doc=_HDR + "The product 4*0.170 here.\n"),
] + [
    dict(name="minus_variant_u%04x_keeps_sign" % c, expect="FAIL", wrong_on=_THROUGH_R6,
         why="U+%04X directly before the digits is a minus sign" % c,
         doc=_HDR + "The delta was " + U(c) + "0.1625 here.\n")
    for c in (0x2010, 0x2011, 0x2012, 0x2014, 0xFE63, 0xFF0D, 0x02D7, 0x2796)
] + [
    dict(name="letter_before_minus_sign_keeps_sign", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a letter directly before a signed number; the sign must survive",
         doc=_HDR + "The shift " + DELTA + MINUS + "0.1625 here.\n"),
] + [
    dict(name="invisible_u%04x_mid_number" % c, expect="FAIL", wrong_on=_THROUGH_R6,
         why="U+%04X is invisible but not category Cf; inside the digits it hid the number" % c,
         doc=_HDR + "The accuracy was 0.15" + U(c) + "25 here.\n")
    for c in (0x034F, 0xFE0F, 0xFE00, 0x3164, 0x115F)
] + [
    dict(name="tag_%s_mid_number" % tag_name, expect="FAIL", wrong_on=_THROUGH_R6,
         why="an inline tag (%s) inside the digits hid the number" % tag,
         doc=_HDR + "The accuracy was " + tag + " here.\n")
    for tag_name, tag in (("b", "0.15<b>25</b>"), ("wbr", "0.15<wbr>25"), ("i", "0.15<i></i>25"),
                          ("sup", "0.15<sup></sup>25"), ("a", "0.15<a></a>25"))
] + [
    dict(name="decimal_point_u%04x" % c, expect="FAIL", wrong_on=_THROUGH_R6,
         why="U+%04X between digits is a decimal point to a reader" % c,
         doc=_HDR + "The accuracy was 0" + U(c) + "1525 here.\n")
    for c in (0xFF0E, 0x2024, 0xFE52, 0x00B7)
] + [
    dict(name="synthesis_frontmatter_title_beats_leading_notes_h1", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a leading `# Notes` H1 must not hide a frontmatter title stating GO",
         doc=_SYN + "title: Lane A 6-seed GO\n---\n\n# Notes\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_later_heading_verdict_bars", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a later heading stating the verdict bars the escape",
         doc=_SYN + "---\n\n# Notes\n\n## Result: 6-seed GO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_hash_line_in_code_fence_is_not_the_title", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a `# comment` inside a code fence must not become the only title read",
         doc=_SYN + "---\n\n```bash\n# run it\n```\n\n# Lane A 6-seed GO\n\nArtifact: `%(art)s`\n\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="synthesis_yaml_comment_is_not_the_title", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a `# comment` inside the YAML frontmatter must not become the only title read",
         doc="---\n# yaml comment\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n"
             "# Lane A 6-seed GO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_setext_title_verdict_bars", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a setext H1 title is read",
         doc=_SYN + "---\n\nLane A 6-seed GO\n================\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_lowercase_verdict_bars", expect="FAIL", wrong_on=_THROUGH_R6,
         why="the bar is case-insensitive (`no-go`)",
         doc=_SYN + "---\n\n# Lane A: no-go at 6 seeds\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_nogo_without_hyphen_bars", expect="FAIL", wrong_on=_THROUGH_R6,
         why="`NOGO` (no hyphen) is a verdict word",
         doc=_SYN + "---\n\n# Lane A NOGO\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_zero_width_verdict_bars", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a zero-width space inside the verdict word must not hide it",
         doc=_SYN + "---\n\n# Lane A G" + ZWSP + "O\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_empty_reason_does_not_capture_next_key", expect="FAIL", wrong_on=_THROUGH_R6,
         why="an EMPTY `claim_check_reason:` must not capture the next key as its value",
         doc="---\nclaim_check: synthesis\nclaim_check_reason:\nlane: gap#5\n---\n\n# A literature summary\n\n"
             "Artifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_filename_verdict_bars", expect="FAIL",
         wrong_on=_THROUGH_R6, filename="survey-lane-a-6seed-GO.md",
         why="the filename is read",
         doc=_SYN + "---\n\n# A literature summary\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="synthesis_frontmatter_verdict_field_bars", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a frontmatter `verdict:` field is read",
         doc=_SYN + "verdict: GO\n---\n\n# A literature summary\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_plural_verdict_bars", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="`6-seed GOs` is a verdict word too (a trailing plural `s` hid it)",
         doc=_SYN + "---\n\n# Lanes A and B: 6-seed GOs\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_multiline_title_verdict_bars", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="a quoted frontmatter title continued on an indented line still carries its verdict word",
         doc=_SYN + "title: 'Lane A, six seeds,\n  GO'\n---\n\n# Notes\n\nArtifact: `%(art)s`\n\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_html_heading_verdict_bars", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="an HTML `<h1>` heading is a heading",
         doc=_SYN + "---\n\n<h1>Lane A 6-seed GO</h1>\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="row_trailing_marker_warns_that_it_exempts_nothing", expect="FAIL", wrong_on=_BEFORE_R6,
         expect_warning=("exempts NOTHING",),
         expect_output=("in the SAME cell",),
         why="a marker alone in a row's last cell is inert, and the author is TOLD so",
         doc=_HDR + "| seed | acc | ratio | note |\n|---|---|---|---|\n| 42 | 0.170 | 0.1525 | <!--derived--> |\n"),
    dict(name="correct_rounding_is_reported_as_rounding", expect="PASS", wrong_on=_THROUGH_R6,
         expect_output=("1 rounding at the stated precision",),
         why="1.235 is the correct 3-decimal rounding of the cited 1.23456789",
         artifact={"loss": 1.23456789}, doc=_HDR + "The loss was 1.235 here.\n"),
    dict(name="three_decimal_rounding_of_four_decimal_value_passes", expect="PASS", wrong_on=_THROUGH_R6,
         why="relative tolerance 1e-4 rejected 0.477 for a cited 0.4774 -- a correct rounding",
         artifact={"acc": 0.4774}, doc=_HDR + "The accuracy was 0.477 here.\n"),
    dict(name="magnitude_suffix_is_scaled_not_stripped", expect="PASS", wrong_on=('r6',),
         why="`1.088B params` is read as 1.088e9 (or the bare 1.088)",
         artifact={"n_params": 1088000000}, doc=_HDR + "A 1.088B params model here.\n"),
    dict(name="multiline_comment_values_do_not_pad_coverage", expect="FAIL", expect_reason="low_coverage",
         wrong_on=_THROUGH_R6,
         why="DISTINCT real values inside a multi-line comment do not count toward coverage",
         artifact={"v": [round(0.3001 + i / 10000.0, 4) for i in range(11)]},
         doc=_HDR + _MARKED_80 + "\n\n<!--\n" + "\n".join("%.4f" % (0.3001 + i / 10000.0) for i in range(11))
             + "\n-->\n"),
    dict(name="linkref_comment_values_do_not_pad_coverage", expect="FAIL", expect_reason="low_coverage",
         wrong_on=_THROUGH_R6,
         why="`[//]: # (...)` is an invisible link-reference 'comment'",
         artifact={"v": [round(0.3001 + i / 10000.0, 4) for i in range(11)]},
         doc=_HDR + _MARKED_80 + "\n\n" + "\n".join("[//]: # (padding %.4f)" % (0.3001 + i / 10000.0)
                                                   for i in range(11)) + "\n"),
    dict(name="hidden_div_values_do_not_pad_coverage", expect="FAIL", expect_reason="low_coverage",
         wrong_on=_THROUGH_R6,
         why="`<div hidden>` content is invisible",
         artifact={"v": [round(0.3001 + i / 10000.0, 4) for i in range(11)]},
         doc=_HDR + _MARKED_80 + "\n\n<div hidden>\n" + "\n".join("%.4f" % (0.3001 + i / 10000.0)
                                                                 for i in range(11)) + "\n</div>\n"),
    dict(name="hidden_citation_in_hidden_div_is_ignored", expect="FAIL", wrong_on=_THROUGH_R6, must_flag=(0.1625,),
         why="a citation inside `<div hidden>` is one no reader can see",
         doc="# Some finding\n\n<div hidden>see `%(art)s`</div>\n\nThe baseline was 0.162500 here.\n"),
    dict(name="seventy_nine_all_marked_claims_fail_coverage", expect="FAIL", expect_reason="low_coverage",
         wrong_on=("main", "r5", "r6"),
         why="79 numeric claims, EVERY one marked derived, passed under a floor of 80",
         doc=_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1) for i in range(79)) + "\n"),
    dict(name="visible_copies_of_one_value_count_once", expect="FAIL", expect_reason="low_coverage",
         wrong_on=_BEFORE_R6,
         why="6 VISIBLE copies of one real value count once toward coverage",
         doc=_HDR + _MARKED_80 + "\n\n" + "\n".join("The accuracy was 0.170000 here." for _ in range(6)) + "\n"),
    dict(name="hidden_citation_in_html_comment_is_ignored", expect="FAIL", wrong_on=_BEFORE_R6, must_flag=(0.1625,),
         why="a citation inside an HTML comment is one no reader can see",
         doc="# Some finding\n\n<!-- see `%(art)s` for context -->\n\nThe baseline was 0.162500 here.\n"),
    dict(name="hidden_citation_in_linkref_comment_is_ignored", expect="FAIL",
         wrong_on=_THROUGH_R6, must_flag=(0.1625,),
         why="the same hidden citation in a `[//]: # (...)` link-reference comment",
         doc="# Some finding\n\n[//]: # (see `%(art)s`)\n\nThe baseline was 0.162500 here.\n"),
    dict(name="nonempty_comment_mid_number_is_read_as_the_reader_sees_it", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`0.15<!-- x -->25` shows 0.1525",
         doc=_HDR + "The accuracy was 0.15<!-- x -->25 here.\n"),
    dict(name="hidden_element_mid_number_is_read_as_the_reader_sees_it", expect="FAIL", wrong_on=_THROUGH_R6,
         why="`0.15<div hidden>9</div>25`: the digits a reader sees are not one of the cited values",
         doc=_HDR + "The accuracy was 0.15<div hidden>9</div>25 here.\n"),
    dict(name="pipe_inside_comment_still_splits_a_table_row", expect="FAIL", wrong_on=_BEFORE_R6,
         why="GFM splits a row into cells BEFORE inline parsing, so a `|` inside a comment separates cells",
         doc=_HDR + "| a | b | c |\n|---|---|---|\n| 42 | 0.1525 <!-- | --> 0.104615 <!--derived--> |\n"),
    dict(name="measurement_with_a_slash_unit_is_not_a_doi", expect="FAIL", wrong_on=(), must_flag=(10.1525,),
         why="`10.1525/s` is a measurement with a unit",
         doc=_HDR + "Throughput 10.1525/s here.\n"),
    dict(name="nonascii_letter_underscore_number_is_a_number", expect="FAIL", wrong_on=_THROUGH_R6,
         why="`Δ_0.1525_` -- an intraword `_` is an identifier character only after an ASCII letter/digit",
         doc=_HDR + "The shift " + DELTA + "_0.1525_ here.\n"),
    dict(name="bidi_override_is_unreadable", expect="FAIL", wrong_on=_THROUGH_R6, expect_reason="unreadable",
         expect_output=("bidirectional control",),
         why="a bidirectional override DISPLAYS 5251.0 as 0.1525; no reading of the stored text is the reader's",
         doc=_HDR + "The accuracy was " + BIDI_RLO + "5251.0" + BIDI_PDF + " here.\n"),
    dict(name="hyphen_range_is_not_a_sign", expect="PASS", wrong_on=(),
         why="a '-' after a digit is a range or a subtraction, not a sign",
         doc=_HDR + "Between 0.1625-0.170 here.\n"),
    dict(name="punctuation_dash_glued_to_number_reads_as_minus", expect="FAIL", wrong_on=_THROUGH_R6,
         expect_output=("reads as a MINUS sign",),
         why="fail-closed choice: an em dash glued to a number is read as a minus; the report says why",
         doc=_HDR + "The baseline" + EM_DASH + "0.1625" + EM_DASH + "was stable.\n"),

    # =============================================================================================================
    # ROUND 7's review -- the holes that decided round 8's design (wrong_on must include r7)
    # =============================================================================================================
    dict(name="r7_tag_regex_swallowed_number_between_angle_brackets", expect="FAIL", wrong_on=('r7',),
         why="r7's tag regex read `<FROZEN by 0.1525 here, and FROZEN>` as ONE tag and blanked the number; round 8 "
             "reads the raw text, where the number is plainly there",
         doc=_HDR + "The gap FULL<FROZEN by 0.1525 here, and FROZEN>SHUF by 0.170 too.\n"),
    dict(name="r7_marker_opener_in_code_span_hid_a_section", expect="FAIL", wrong_on=('r7',),
         why="r7 found `<!--` anywhere, even inside a code span, and blanked everything to the next `-->` as a "
             "'marker' span -- a whole section vanished; round 8 never blanks anything and asks markdown-it what "
             "is a comment",
         doc=_HDR + "Mark derived numbers with a `<!--derived` comment.\n\nThe accuracy was 0.1525 here.\n\n"
                    "Close it with `-->` as usual.\n"),
    dict(name="r7_marker_opener_in_fence_hid_a_section", expect="FAIL", wrong_on=('r7',),
         why="the same with the opener inside a fenced code block",
         doc=_HDR + "```\n<!--derived\n```\n\nThe accuracy was 0.1525 here.\n\n```\n-->\n```\n"),
    dict(name="r7_derived_prefixed_comment_exempted_numbers", expect="FAIL", wrong_on=('r7',),
         expect_warning=("not an exact marker",),
         why="r7 accepted ANY comment starting with `derived` (`<!--derived-from the sweep-->`) as a marker",
         doc=_HDR + "The accuracy was 0.1525 here. <!--derived-from the sweep-->\n"),
    dict(name="r7_spaced_marker_spelling_is_not_a_marker", expect="FAIL", wrong_on=('r7',),
         expect_warning=("not an exact marker",),
         why="`<!-- derived -->` is not one of the two exact spellings",
         doc=_HDR + "The accuracy was 0.1525 here. <!-- derived -->\n"),
    dict(name="r7_capitalized_marker_spelling_is_not_a_marker", expect="FAIL", wrong_on=('r7',),
         expect_warning=("not an exact marker",),
         why="`<!--Derived-->` is not one of the two exact spellings",
         doc=_HDR + "The accuracy was 0.1525 here. <!--Derived-->\n"),
    dict(name="r7_multiline_derived_note_is_not_a_marker", expect="FAIL", wrong_on=('r7',),
         expect_warning=("not an exact marker",),
         why="a marker must sit whole on the number's own line; a note that spans lines is not an exact marker",
         doc=_HDR + "The accuracy was 0.1525 here. <!--derived: from the table,\nsee below -->\n"),
    dict(name="r7_doc_averaged_chance_let_a_coarse_wrong_headline_through", expect="FAIL", wrong_on=('r7',),
         expect_reason="too_broad",
         why="r7 averaged the chance-match rate over the DOC: ten precise, correct numbers pulled the average under "
             "its 20% bar, so one coarse wrong headline (0.153, matching an unrelated dense sweep value by chance) "
             "passed; round 8 rates every claim on its own",
         artifact={"precise": [0.123456, 0.234567, 0.345678, 0.456789, 0.567891, 0.678912, 0.789123, 0.891234,
                               0.912345, 0.132465],
                   "sweep": [round(i / 1000.0 + 0.0002, 4) for i in range(1000)]},
         doc=_HDR + "".join("Seed value %s here.\n\n" % v for v in ("0.123456", "0.234567", "0.345678", "0.456789",
                                                                   "0.567891", "0.678912", "0.789123", "0.891234",
                                                                   "0.912345", "0.132465"))
             + "Headline accuracy 0.153 here.\n"),
    # =============================================================================================================
    # ROUND 8's own contract, both directions
    # =============================================================================================================
    dict(name="r8_marker_in_code_span_on_the_same_line_exempts_nothing", expect="FAIL",
         wrong_on=('main', 'r1', 'r2', 'r3', 'r5', 'r6', 'r7'),
         expect_warning=("inside a code span",),
         why="a marker inside a code span is code, not a comment",
         doc=_HDR + "The accuracy was 0.1525 here. `<!--derived-->`\n"),
    dict(name="r8_escaped_marker_exempts_nothing", expect="FAIL",
         wrong_on=('main', 'r1', 'r2', 'r3', 'r5', 'r6', 'r7'),
         expect_warning=("renders as visible text",),
         why="`\\<!--derived-->` renders as visible text",
         doc=_HDR + "The accuracy was 0.1525 here. \\<!--derived-->\n"),
    dict(name="r8_marker_in_fence_exempts_nothing", expect="FAIL",
         wrong_on=('main', 'r1', 'r2', 'r3', 'r5', 'r6', 'r7'),
         expect_warning=("inside a code block",),
         why="a marker inside a fenced code block is code; the fence's numbers are still checked",
         doc=_HDR + "```\nacc 0.1525 <!--derived-->\n```\n"),
    dict(name="r8_number_inside_html_comment_is_checked", expect="FAIL", wrong_on=('r6',),
         why="nothing is removed from checking: a number in a comment must be supported too",
         doc=_HDR + "Some text. <!-- the accuracy was 0.1525 -->\n"),
    dict(name="r8_number_in_fence_is_checked", expect="FAIL", wrong_on=(),
         why="numbers in fenced code are checked (a marker in a fence is code, so cite the artifact or move them)",
         doc=_HDR + "```\nacc = 0.1525\n```\n"),
    dict(name="r8_reader_only_split_number_is_never_exempt", expect="FAIL",
         wrong_on=_ALL_BEFORE_R8, must_flag=(0.104615,),
         why="`0.10**4615**` shows 0.104615, a number only the reader's reading holds; it cannot be exempted",
         doc=_HDR + "The ratio 0.10**4615** here. <!--derived-->\n"),
    dict(name="r8_multiline_comment_mid_number_in_a_paragraph", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="an inline comment spanning a line break inside a paragraph renders as nothing: `0.15<!--`/`-->25` "
             "shows 0.1525",
         doc=_HDR + "The accuracy was 0.15<!--\n-->25 here.\n"),
    dict(name="r8_invisible_before_number_never_glues", expect="FAIL", wrong_on=('r6',),
         why="a zero-width character is a SPACE in the normalized copy, never deleted, so `acc<ZWSP>0.1525` keeps "
             "its number",
         doc=_HDR + "The acc" + ZWSP + "0.1525 here.\n"),
    dict(name="r8_hidden_exempt_twin_cannot_vouch_for_a_split_number", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="the reader sees 0.1525 (split by emphasis); an exempt copy hidden in a comment on the same line must "
             "not vouch for it -- a reader number is matched only to a VISIBLE raw number on its own line",
         doc=_HDR + "The accuracy was 0.15**25** here. <!-- 0.1525 --> <!--derived-->\n"),
    dict(name="r8_exempt_twin_on_another_line_cannot_vouch_for_a_split_number", expect="FAIL", wrong_on=_BEFORE_R6,
         why="the same with the exempt copy on the previous line of the same paragraph",
         doc=_HDR + "Ratio 0.1525 <!--derived-->\nand the accuracy was 0.15**25** here.\n"),
    dict(name="r8_bold_derived_number_is_one_claim", expect="PASS", wrong_on=(),
         why="a bold derived number is ONE claim (raw and reader readings agree on its line), so its marker exempts it",
         doc=_HDR + "The ratio was **0.104615** here. <!--derived-->\n"),
    dict(name="r8_invisible_between_minus_and_digits_keeps_sign", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`<MINUS><ZWSP>0.1625` shows -0.1625: the raw and normalized readings see an unsigned 0.1625 (the "
             "positive baseline), the reader's reading glues the sign",
         doc=_HDR + "The delta was " + MINUS + ZWSP + "0.1625 here.\n"),
    dict(name="r8_bold_minus_before_digits_keeps_sign", expect="FAIL", wrong_on=_THROUGH_R6,
         why="`<b>-</b>0.1625` shows -0.1625 with a bold minus (a markdown `**-**0.1625` is NOT emphasis -- the "
             "closing `**` is not right-flanking -- so it renders its asterisks and is read as written)",
         doc=_HDR + "The delta was <b>" + MINUS + "</b>0.1625 here.\n"),
    dict(name="r8_number_glued_after_a_letter_is_read", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="`corr0.1525` -- every earlier revision's lookbehind treated a letter before the digits as an identifier "
             "and skipped the number (the corpus has one such measurement, `corr0.869`)",
         doc=_HDR + "Best balance corr0.1525 here.\n"),
    dict(name="r8_entity_minus_sign_flip", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`&minus;0.1625` shows -0.1625 while the artifact holds +0.1625",
         doc=_HDR + "The delta was &minus;0.1625 here.\n"),
    dict(name="r8_fullwidth_digit_number_is_read", expect="FAIL", wrong_on=(),
         why="a fullwidth digit maps to its ASCII digit in the normalized copy",
         doc=_HDR + "The accuracy was " + U(0xFF10) + ".1525 here.\n"),
    dict(name="r8_arxiv_id_is_checked_unless_marked", expect="FAIL", wrong_on=('r7',), must_flag=(2403.12345,),
         why="fail closed: an identifier with >= 3 decimals is checked like any number",
         doc=_HDR + "Method from arXiv:2403.12345; accuracy 0.170 here.\n"),
    dict(name="r8_arxiv_id_marked_on_its_line_passes", expect="PASS", wrong_on=_THROUGH_R6,
         why="an identifier marked on its own line with a note passes",
         doc=_HDR + "Method from arXiv:2403.12345 <!--derived: arXiv id-->; accuracy 0.170 here.\n"),
    dict(name="r8_doi_is_checked_unless_marked", expect="FAIL", wrong_on=('r7',), must_flag=(10.1038,),
         why="a DOI prefix is checked like any number",
         doc=_HDR + "Ernst & Banks 2002, doi:10.1038/415429a; accuracy 0.170 here.\n"),
    dict(name="r8_url_number_is_checked_unless_marked", expect="FAIL", wrong_on=('r7',), must_flag=(2403.12345,),
         why="a number inside a URL is checked like any number",
         doc=_HDR + "See https://arxiv.org/abs/2403.12345 -- accuracy 0.170 here.\n"),
    dict(name="r8_derived_note_marker_on_its_line_passes", expect="PASS", wrong_on=_THROUGH_R6,
         why="`<!--derived: note-->` on the number's own line exempts it; the note's own numbers are on that line too",
         doc=_HDR + "The ratio is 0.104615 here. <!--derived: 0.104615 = 0.17 / 1.625-->\n"
                    "The baseline was 0.162500 here.\n"),
    dict(name="r8_two_markers_in_one_cell_exempt_sixteen", expect="PASS", wrong_on=('r6', 'r7'),
         why="the cap is per marker: two markers in one cell exempt up to 16 numbers",
         doc=_HDR + "Values " + ", ".join("0.1000%02d" % i for i in range(1, 17)) + " <!--derived--><!--derived-->\n"),
    dict(name="r8_two_markers_do_not_reach_a_seventeenth", expect="FAIL", wrong_on=_BEFORE_R6,
         why="... and a 17th number in that cell is checked",
         doc=_HDR + "Values " + ", ".join("0.1000%02d" % i for i in range(1, 17)) + ", 0.1525 "
                    "<!--derived--><!--derived-->\n"),
    dict(name="r8_correctly_rounded_negative_with_typographic_minus_passes", expect="PASS", wrong_on=_THROUGH_R6,
         why="a typographic minus before a correct negative value is a sign, not a false positive",
         artifact={"delta": -0.16248}, doc=_HDR + "The delta was " + MINUS + "0.1625 here.\n"),
    dict(name="r8_marker_starting_an_html_block_line_exempts_its_line", expect="PASS", wrong_on=(),
         why="a marker that starts a line is an HTML block to markdown -- still a comment, still its own line",
         doc=_HDR + "<!--derived--> the ratio 0.104615 was derived\n\nThe baseline was 0.162500 here.\n"),
    dict(name="r8_citation_in_a_line_start_html_block_is_visible", expect="PASS", wrong_on=(),
         why="a line that STARTS with a marker is an HTML block, and a browser shows its text: a citation there is "
             "one a reader sees, so it is loaded (an early round-8 draft hid whole HTML-block lines)",
         doc="# Some finding\n\n<!--derived--> the ratio 0.104615 is from `%(art)s`\n\nThe baseline was 0.162500 "
             "here.\n"),
    # =============================================================================================================
    # ROUND 8, second pass (after the 07:51 kill): a reader number is matched to a source number by POSITION, never
    # by value -- the staged draft matched by (line, value) and let any same-valued twin vouch for a split number
    # =============================================================================================================
    dict(name="r8_attribute_twin_cannot_vouch_for_a_split_number", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="an exempt copy of 0.1525 inside a tag attribute is a raw number no reader sees; it must not vouch for "
             "the 0.15**25** the reader does see",
         doc=_HDR + "The accuracy was 0.15**25** here. <a title=\"0.1525\"></a> <!--derived-->\n"),
    dict(name="r8_link_title_twin_cannot_vouch_for_a_split_number", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="the same with the exempt copy in a link title",
         doc=_HDR + "The accuracy was 0.15**25** here, [see](http://x.org \"0.1525\") <!--derived-->\n"),
    dict(name="r8_link_url_twin_cannot_vouch_for_a_split_number", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="the same with the exempt copy in a link destination",
         doc=_HDR + "The accuracy was 0.15**25** here, [see](http://x.org/0.1525) <!--derived-->\n"),
    dict(name="r8_image_alt_twin_cannot_vouch_for_a_split_number", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="the same with the exempt copy in an image's alt text",
         doc=_HDR + "The accuracy was 0.15**25** here ![0.1525](x.png) <!--derived-->\n"),
    dict(name="r8_glued_twin_cannot_vouch_for_a_split_number", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         must_flag=(0.1525,),
         artifact={"accuracy": 0.17, "x": 0.15259},
         why="the raw 0.1525 in `0.1525<b></b>9` is exempt, but a reader sees it as 0.15259 (which the artifact "
             "holds); it must not vouch for the split 0.15**25** later on the line",
         doc=_HDR + "Ratio 0.1525<b></b>9 and 0.15**25** here. <!--derived-->\n"),
    dict(name="r8_sign_flipped_twin_cannot_vouch_for_a_split_number", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         must_flag=(0.1525,),
         artifact={"accuracy": 0.17, "delta": -0.1525},
         why="the raw +0.1525 in `<b>-</b>0.1525` is exempt, but a reader sees -0.1525 (which the artifact holds); "
             "it must not vouch for the split +0.15**25**",
         doc=_HDR + "Delta <b>-</b>0.1525 and 0.15**25** here. <!--derived-->\n"),
    dict(name="r8_style_hidden_digit_is_read_as_hidden", expect="FAIL", wrong_on=_ALL_BEFORE_R8, must_flag=(0.1525,),
         artifact={"accuracy": 0.17, "x": 0.15925},
         why="`0.15<span style=\"font-size:0\">9</span>25` shows 0.1525: a second reading hides the content of every "
             "element that carries an attribute (a style or class can hide it)",
         doc=_HDR + "The accuracy was 0.15<span style=\"font-size:0\">9</span>25 here.\n"),
    dict(name="r8_underscore_emphasis_dot_start_number_is_read", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`_.1525_` shows .1525; the raw lookbehind skips a dot after `_`, the reader's reading does not",
         doc=_HDR + "The drop was _.1525_ here.\n"),
    dict(name="r8_entity_digit_inside_a_number_is_read", expect="FAIL", wrong_on=_BEFORE_R6,
         why="`0.&#49;525` shows 0.1525",
         doc=_HDR + "The accuracy was 0.&#49;525 here.\n"),
    dict(name="r8_percent_decoded_autolink_number_is_read", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="an autolink shows its url percent-DECODED: `<http://x.org/0%%2E1525>` shows 0.1525",
         doc=_HDR + "See <http://x.org/0%%2E1525> here.\n"),
    dict(name="r8_code_span_glued_after_a_digit_is_read", expect="FAIL", wrong_on=_BEFORE_R6,
         why="``0`.1525` `` shows 0.1525 (the backticks render as nothing)",
         doc=_HDR + "The value 0`.1525` here.\n"),
    dict(name="r8_code_span_range_is_read_as_a_range", expect="PASS", wrong_on=(),
         why="`` `0.170`-`0.1625` `` shows the range 0.170-0.1625, not a negative 0.1625 (markup renders as nothing, "
             "so the reader's reading must not put a space where the backticks were)",
         doc=_HDR + "Between `0.170`-`0.1625` here.\n"),
    dict(name="r8_bold_range_is_read_as_a_range", expect="PASS", wrong_on=(),
         why="`**0.170**-**0.1625**` shows the range 0.170-0.1625 (the draft put a space after the bold and read a "
             "minus sign)",
         doc=_HDR + "Between **0.170**-**0.1625** here.\n"),
    dict(name="r8_bold_derived_number_after_a_multiline_code_span_passes", expect="PASS", wrong_on=(),
         why="a code span that spans a line break keeps its newline in the line count, so a bold derived number after "
             "it is still the SAME claim as its raw copy (the draft keyed it one line early and re-checked it)",
         doc=_HDR + "See `foo\nbar` and the ratio **0.104615** here. <!--derived-->\n"),
    dict(name="r8_bold_negative_derived_number_passes", expect="PASS", wrong_on=(),
         why="`**-0.104615**` is one claim: its sign sits inside the bold, so raw and reader read it the same way",
         doc=_HDR + "Delta **-0.104615** here. <!--derived-->\n"),
    dict(name="r8_marked_autolink_identifier_passes", expect="PASS", wrong_on=_THROUGH_R6,
         why="an autolink's text is the source url when nothing was percent-decoded, so its marker exempts it",
         doc=_HDR + "See <https://arxiv.org/abs/2403.12345> <!--derived: arXiv id--> -- accuracy 0.170 here.\n"),
    dict(name="r8_ascii_hyphen_after_a_paren_reads_both_signs", expect="FAIL", wrong_on=('r7',), must_flag=(-0.1625,),
         why="`(lesion)-0.1625`: main and round 5 read a minus sign, the draft read a range; an ambiguous ASCII "
             "hyphen is read BOTH ways, so the sign error main catches is caught",
         doc=_HDR + "The gain (lesion)-0.1625 here.\n"),
    dict(name="r8_ascii_hyphen_after_a_letter_reads_both_signs", expect="FAIL", wrong_on=('r7',), must_flag=(0.1525,),
         artifact={"accuracy": 0.17, "delta": -0.1525},
         why="`acc-0.1525`: main and round 5 read +0.1525 (a hyphen after a word), the draft a minus sign; both "
             "readings are checked",
         doc=_HDR + "The acc-0.1525 here.\n"),
    dict(name="r8_synthesis_heading_split_by_emphasis_bars", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="`# Lane A G**O**` shows GO",
         doc=_SYN + "---\n\n# Lane A G**O**\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_heading_split_by_comment_bars", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="`# Lane A G<!-- -->O` shows GO",
         doc=_SYN + "---\n\n# Lane A G<!-- -->O\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_heading_entity_bars", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="`# Lane A &#71;O` shows GO",
         doc=_SYN + "---\n\n# Lane A &#71;O\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_block_scalar_title_across_a_blank_line_bars", expect="FAIL", wrong_on=_THROUGH_R6,
         why="a `title: |` block scalar continues across a blank line (the draft stopped reading at it)",
         doc=_SYN + "title: |\n  Lane A\n\n  GO\nlane: x\n---\n\n# Notes\n\nArtifact: `%(art)s`\n\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_capitalized_title_key_bars", expect="FAIL", wrong_on=_ALL_BEFORE_R8,
         why="`Title:` is read as the title too",
         doc=_SYN + "Title: Lane A GO\nlane: x\n---\n\n# Notes\n\nArtifact: `%(art)s`\n\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_quoted_flag_is_not_synthesis", expect="FAIL", wrong_on=('r7',),
         why="main and round 5 do not read `claim_check: \"synthesis\"` as the flag, so round 8 does not either",
         doc="---\nclaim_check: \"synthesis\"\nclaim_check_reason: quotes prior runs\n---\n\n# A literature summary\n\n"
             "Artifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_flag_after_an_earlier_dash_line_is_not_synthesis", expect="FAIL", wrong_on=('r6', 'r7'),
         why="main and round 5 read the flag only before the first `\\n---`; a `----` line ends their block",
         doc="---\ntitle: notes\n----\nclaim_check: synthesis\nclaim_check_reason: quotes prior runs\n---\n\n"
             "# A literature summary\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r8_synthesis_after_a_byte_order_mark_is_not_synthesis", expect="FAIL", wrong_on=('r7',),
         why="main and round 5 do not see `---` at the start of a file that begins with a byte-order mark",
         doc=BOM + _SYN + "---\n\n# A literature summary\n\nArtifact: `%(art)s`\n\nThe accuracy was 0.1525 here.\n"),
    dict(name="r8_legacy_tolerance_is_accepted_and_reported", expect="PASS", wrong_on=('r7',),
         expect_output=("1 legacy relative tolerance",),
         why="SPEC: the legacy relative tolerance max(5e-6, 1e-4|x|) is also accepted, as in main and r5 -- "
             "12.3456 for a cited 12.3449 passes (a wider window than its stated precision; its own chance rate "
             "stays low because the pool is sparse). Recorded so the acceptance is a stated decision",
         artifact={"x": 12.3449}, doc=_HDR + "The value was 12.3456 here.\n"),
    # =============================================================================================================
    # round 8 as reviewed (654d95664): the review's repros, each wrong on r8a and caught now
    # =============================================================================================================
    dict(name="r8a_wrong_3_decimal_number_in_a_10pct_pool_is_too_broad", expect="FAIL", wrong_on=('r7', 'r8a'),
         expect_reason="too_broad", artifact=_SWEEP_3DEC,
         why="review 1: a wrong 3-decimal number lands in the +-0.0005 window of an unrelated value; its own chance "
             "rate (~0.10) was under the flat 0.20 limit, so wrong 3-decimal numbers passed more often than in main "
             "(replay: 12.9% vs 7.6%) -- the 3-decimal limit is now 0.04",
         doc=_HDR + "The headline was 0.153 here.\n"),
    dict(name="r8a_en_dash_after_a_word_reads_both_signs", expect="FAIL", wrong_on=('r7', 'r8a'), must_flag=(0.1625,),
         artifact=_NEG,
         why="review 2: main and r5 never read a non-ASCII dash as a sign, so `lesion" + EN_DASH + "0.1625` is +0.1625 "
             "to them; r8a read only -0.1625 and passed a doc main fails -- both readings are checked now",
         doc=_HDR + "The lesion" + EN_DASH + "0.1625 here.\n"),
    dict(name="r8a_em_dash_after_a_space_reads_both_signs", expect="FAIL", wrong_on=('r7', 'r8a'), must_flag=(0.1625,),
         artifact=_NEG, why="review 2: `x " + EM_DASH + "0.1625` -- an em dash in a sign position is read both ways",
         doc=_HDR + "The x " + EM_DASH + "0.1625 here.\n"),
    dict(name="r8a_unicode_hyphen_after_a_word_reads_both_signs", expect="FAIL", wrong_on=('r7', 'r8a'),
         must_flag=(0.1625,), artifact=_NEG,
         why="review 2: `lesion" + HYPHEN + "0.1625` (U+2010) -- a hyphen glyph is read both ways",
         doc=_HDR + "The lesion" + HYPHEN + "0.1625 here.\n"),
    dict(name="r8a_minus_sign_after_a_letter_reads_both_signs", expect="FAIL", wrong_on=('r7', 'r8a'),
         must_flag=(0.1625,),
         artifact=_NEG,
         why="review 2: `x" + MINUS + "0.1625` -- U+2212 glued to a word is read both ways (x minus 0.1625)",
         doc=_HDR + "The x" + MINUS + "0.1625 here.\n"),
    dict(name="r8a_minus_sign_after_a_space_is_a_sign", expect="PASS", wrong_on=('main', 'r1', 'r2', 'r3', 'r4', 'r5'),
         artifact=_NEG,
         why="U+2212 after a space is a minus sign (every non-ASCII sign in the findings since 2026-09-01 is written "
             "so); main and r1-r5 read it as +0.1625",
         doc=_HDR + "The delta " + MINUS + "0.1625 here.\n"),
    dict(name="r8a_middle_dot_before_an_arabic_digit_hides_nothing", expect="FAIL", wrong_on=('r7', 'r8a'),
         must_flag=(32.5051,), artifact={"x": 2.5051},
         why="review 3: in `5" + MIDDLE_DOT + ARABIC_3 + "2.5051` main reads 32.5051; the raw reading stops at the "
             "Arabic-Indic digit (2.5051) and the dot copy reads `5.32.5051` (nothing) -- a copy with the dot as a "
             "space reads `5 32.5051`",
         doc=_HDR + "The value 5" + MIDDLE_DOT + ARABIC_3 + "2.5051 here.\n"),
    dict(name="r8a_arabic_decimal_separator_before_an_arabic_digit_hides_nothing", expect="FAIL",
         wrong_on=('r7', 'r8a'),
         must_flag=(32.5051,), artifact={"x": 2.5051},
         why="review 3: the same with U+066B ARABIC DECIMAL SEPARATOR",
         doc=_HDR + "The value 5" + ARABIC_DECIMAL_SEP + ARABIC_3 + "2.5051 here.\n"),
    dict(name="r8a_hidden_citation_of_a_corrupt_artifact_fails", expect="FAIL", wrong_on=('r6', 'r7', 'r8a'),
         expect_reason="missing", raw_files={"broken.json": "{not json"},
         why="review 4: a citation inside a comment adds nothing to the pool but is still opened -- a file that is "
             "not valid JSON fails as unreadable, as in main and r5 (r8a only checked that it existed)",
         doc=_HDR + "<!-- see also %(sub)s/broken.json -->\n\nThe accuracy was 0.170 here.\n"),
    dict(name="r8a_chance_rate_does_not_depend_on_the_spelling", expect="FAIL", wrong_on=('r8a',),
         expect_reason="too_broad", artifact=_SPELLING_POOL,
         why="review 5: r8a sampled 100 decoys seeded by the claim's TEXT -- `0.1525` rated 0.15 (passed) and "
             "`.1525` 0.24 against this pool; the rate is now exact over all 1000 decoys (0.21 for every spelling)",
         doc=_HDR + "The value was 0.1525 here.\n"),
    dict(name="r8a_hidden_scale_suffix_is_not_the_same_claim", expect="FAIL", wrong_on=_THROUGH_R6 + ('r8a',),
         must_flag=(0.1525,),
         artifact={"x": 152.5},
         why="review 6: the first reader reading sees `0.1525k` (152.5, in the pool); the second hides the "
             "attribute-bearing span and sees 0.1525 -- r8a dropped it as a duplicate of (line, value, decimals) "
             "ignoring the suffix",
         doc=_HDR + 'The value 0.15<b>2</b>5<span class="u">k</span> here.\n'),
    dict(name="r8a_marker_live_in_gfm_only_exempts_nothing", expect="FAIL",
         wrong_on=('main', 'r1', 'r2', 'r3', 'r5', 'r6', 'r7'),
         why="review 7: GFM splits the row into cells before inline parsing, so the marker is a comment in its own "
             "cell; CommonMark reads one code span across the pipes. A marker is live only when BOTH parsers read a "
             "comment (an either-parser rule passes this doc)",
         doc=_HDR + "| a | b | c |\n|---|---|---|\n| `x | 0.1525 <!--derived--> | y` |\n"),
]
