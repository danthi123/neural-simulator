"""tools/claim_check.py -- both directions, one test per hole, every historical claim RE-DERIVED from git.

Round 5 made the derived-marker exemption same-line-only; round 6 narrowed it to the same table cell / <br>-segment
and normalized numbers before matching; round 7 (this file's current contract) adds precision-aware matching,
a per-doc discriminating-power check, normalization that never glues, GFM tables in every form, a hardened
synthesis bar, hidden-carrier-aware coverage, and a CCT gate that reports the instrument's own failures verbatim.
Layers:

  1. every SELFTEST_CASES entry gives its expected verdict for its DESIGNATED reason (claim_check.selftest());
  2. HISTORY: every case is re-run through each earlier checker (main, r1-r6, read straight from git) -- the
     case's recorded `wrong_on` must equal the set of revisions that actually get it wrong, so "this used to pass,
     now it fails" is re-derived every run. A `kind='gate'` case is run against each revision's
     tools/gates/claim_check_selftest.py (absent before r6 => wrong);
  3. spec guards and warnings for the marker rule;
  4. round 7 unit tests: rule A, rule B, normalization, hidden carriers, synthesis title, author-facing text.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tools.claim_check as cc                      # noqa: E402

ODD_FENCE_DOCS = [
    "research/findings/2026-06-11-dual-CLS-cortex-channel-derisk-GO.md",
    "research/findings/2026-06-17-ordered-wm-position-binding-derisk.md",
    "research/findings/2026-06-20-S5-divisive-norm-derisk.md",
    "research/findings/2026-06-26-multibridge-deep-knowledge-design.md",
]
DOC_CASES = [c for c in cc.SELFTEST_CASES if c.get("kind") != "gate"]
GATE_CASES = [c for c in cc.SELFTEST_CASES if c.get("kind") == "gate"]


@pytest.fixture(scope="module")
def casedir():
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test_claim_check_line_only_") as d:
        yield d


# ---- 1. the registry (both directions) -----------------------------------------------------------------------
def test_selftest_is_clean():
    assert cc.selftest() == []


def test_every_registry_case_names_a_real_historical_revision():
    names = [c["name"] for c in cc.SELFTEST_CASES]
    assert len(names) == len(set(names)), "duplicate case name"
    for c in cc.SELFTEST_CASES:
        assert set(c["wrong_on"]) <= set(cc._HISTORY_SHAS), c["name"]


def test_every_round6_review_issue_has_a_case_that_round6_gets_wrong():
    """Round 7's contract (task G): each of the 11 review issues has at least one SELFTEST_CASES entry that fails on
    f2b7db2b4 (recorded in wrong_on, and re-derived from git by test_case_wrong_on_is_re_derived)."""
    covered = {c["issue"] for c in cc.SELFTEST_CASES if "issue" in c and "r6" in c["wrong_on"]}
    assert covered == set(range(1, 12)), "issues without an r6-failing case: %s" % sorted(set(range(1, 12)) - covered)
    with open(os.path.join(ROOT, "research/coordination/claimcheck_r7_review_issues.txt")) as fh:
        assert len(json.load(fh)["issues"]) == 11


# ---- 2. history: each case against every earlier checker, wrong_on RE-DERIVED, not remembered -----------------
def _git_show(sha, path):
    try:
        return subprocess.run(["git", "-C", ROOT, "show", "%s:%s" % (sha, path)],
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _exec_source(src, tag):
    mod_path = os.path.join(ROOT, ".claim_check_hist_%s_%d.py" % (tag, os.getpid()))
    try:
        with open(mod_path, "w", encoding="utf-8") as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location("claim_check_hist_%s" % tag, mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if os.path.exists(mod_path):
            os.remove(mod_path)
    return mod


def _historical(tag, sha):
    src = _git_show(sha, "tools/claim_check.py")
    if src is None:
        return None
    mod = _exec_source(src, tag)
    mod.ROOT = ROOT
    return mod


_MARKDOWN_IT_AVAILABLE = importlib.util.find_spec("markdown_it") is not None


@pytest.fixture(scope="module")
def history():
    mods = {tag: _historical(tag, sha) for tag, sha in cc._HISTORY_SHAS.items()}
    if any(m is None for m in mods.values()):
        pytest.skip("earlier claim_check revisions are not in this clone (%s)"
                    % ", ".join(t for t, m in mods.items() if m is None))
    return mods


def _wrong_set(case, history, casedir):
    wrong = set()
    for tag, mod in history.items():
        # round 4's check() lazily imports markdown-it-py; without it the verdict says nothing about the case.
        if tag == "r4" and not _MARKDOWN_IT_AVAILABLE:
            continue
        got, out = cc._case_outcome(case, mod, casedir)
        if got != case["expect"] or any(o not in out for o in cc._expected_outputs(case)):
            wrong.add(tag)
    return wrong


@pytest.mark.parametrize("case", DOC_CASES, ids=[c["name"] for c in DOC_CASES])
def test_case_wrong_on_is_re_derived(case, casedir, history):
    wrong = _wrong_set(case, history, casedir)
    expected = set(case["wrong_on"]) - ({"r4"} if not _MARKDOWN_IT_AVAILABLE else set())
    assert wrong == expected, ("recorded wrong_on=%s but this revision-set actually gets it wrong: %s (%s)"
                               % (sorted(case["wrong_on"]), sorted(wrong), case["why"]))


@pytest.mark.parametrize("case", GATE_CASES, ids=[c["name"] for c in GATE_CASES])
def test_gate_case_wrong_on_is_re_derived(case):
    wrong = set()
    for tag, sha in cc._HISTORY_SHAS.items():
        src = _git_show(sha, "tools/gates/claim_check_selftest.py")
        if src is None:
            wrong.add(tag)                          # no CCT gate at all: a broken instrument went unreported
            continue
        ok, _detail = cc._gate_case_ok(_exec_source(src, "gate_" + tag))
        if not ok:
            wrong.add(tag)
    assert wrong == set(case["wrong_on"]), (case["wrong_on"], wrong)
    import tools.gates.claim_check_selftest as gate
    assert cc._gate_case_ok(gate)[0], "the CURRENT gate must pass its own case"


# ---- 3. spec guards: the marker rule, both directions ----------------------------------------------------------
_H = "# Some finding\n\nArtifact: `%(art)s`\n\n"


@pytest.mark.parametrize("name,body,flagged_expected", [
    ("inline_marker_same_segment_exempts", _H + "The ratio is 0.104615 here. <!--derived-->\n"
                                                "The baseline was 0.162500 here.\n", set()),
    ("marker_one_line_early_does_not_reach", _H + "<!--derived-->\nThe accuracy was 0.1525 here.\n", {0.1525}),
    ("marker_one_line_late_does_not_reach", _H + "The accuracy was 0.1525 here.\n<!--derived-->\n", {0.1525}),
    ("every_derived_table_row_marked_passes",
     _H + "| metric | value |\n|---|---|\n| ratio | 0.104615 <!--derived--> |\n"
          "| gap | 0.207531 <!--derived--> |\n| accuracy | 0.170000 |\n", set()),
    ("one_unmarked_table_row_is_checked",
     _H + "| metric | value |\n|---|---|\n| ratio | 0.104615 <!--derived--> |\n| accuracy | 0.1525 |\n", {0.1525}),
    ("marker_in_separate_cell_does_not_reach_other_cells",
     _H + "| metric | value | delta |\n|---|---|---|\n| ratio | 0.1525 | 0.104615 | <!--derived--> |\n",
     {0.1525, 0.104615}),
    ("br_split_marker_does_not_reach_other_side",
     _H + "ratio 0.104615 <!--derived--><br>accuracy 0.1525\n", {0.1525}),
    ("cap_bounds_a_marked_segments_free_exemption",
     _H + "vals 0.100001 0.100002 0.100003 0.100004 0.100005 0.100006 0.100007 0.100008 0.1525 <!--derived-->\n",
     {0.1525}),
    ("marker_inside_fence_does_not_leak_out",
     _H + "```\n<!--derived-->\n```\nThe accuracy was 0.1525 here.\n", {0.1525}),
    ("hash_derived_comment_is_not_a_marker",
     _H + "```python\n# derived\n```\nThe accuracy was 0.1525 here.\n", {0.1525}),
    ("derived_note_marker_exempts_its_segment",
     _H + "The gap is 0.207531 here. <!--derived: 2 x 0.104615-->\nThe accuracy was 0.1525 here.\n", {0.1525}),
    ("list_item_table_row_splits_cells",
     _H + "- | seed | acc | ratio |\n  |---|---|---|\n  | 42 | 0.1525 | 0.104615 <!--derived--> |\n", {0.1525}),
    ("negative_number_alone_in_a_cell_keeps_its_sign",
     _H + "| a | b |\n|---|---|\n| delta |-0.1625|\n", {-0.1625}),
])
def test_spec_guard(name, body, flagged_expected, casedir):
    p = cc._write_case(casedir, {"name": name, "doc": body})
    r = cc._scan(p)
    assert {round(v, 6) for _ln, v, _c in r["unsupported"]} == flagged_expected, name


@pytest.mark.parametrize("name,body,warn_kind_substr", [
    ("standalone_marker_warns", _H + "<!--derived-->\nThe accuracy was 0.1525 here.\n", "standalone marker"),
    ("atx_derived_heading_warns", _H + "## Derived\nThe accuracy was 0.1525 here.\n", "heading"),
    ("h1_derived_heading_warns", _H + "# Derived\nThe accuracy was 0.1525 here.\n", "heading"),
    ("blockquoted_derived_heading_warns", _H + "> ## Derived\nThe accuracy was 0.1525 here.\n", "heading"),
    ("setext_derived_heading_warns", _H + "Derived\n=======\nThe accuracy was 0.1525 here.\n", "heading"),
    ("close_marker_warns", _H + "The ratio is 0.104615. <!--/derived-->\nThe accuracy was 0.1525 here.\n",
     "close marker"),
    ("row_trailing_marker_warns", _H + "| a | b |\n|---|---|\n| 0.1525 | <!--derived--> |\n", "exempts nothing"),
])
def test_inert_idiom_warns_but_does_not_exempt(name, body, warn_kind_substr, casedir):
    p = cc._write_case(casedir, {"name": name, "doc": body})
    r = cc._scan(p)
    assert 0.1525 in {round(v, 6) for _ln, v, _c in r["unsupported"]}, "%s: the idiom must not exempt" % name
    assert any(warn_kind_substr in kind.lower() for _ln, kind, _msg in r["warnings"]), r["warnings"]


def test_clean_doc_has_no_warnings(casedir):
    p = cc._write_case(casedir, {"name": "clean_doc_no_warnings", "doc": _H + "The ratio is 0.104615 here. "
                                 "<!--derived-->\nThe baseline was 0.162500 here.\n"})
    assert cc._scan(p)["warnings"] == []


# ---- the four real odd-fence docs: no looser than main in VERDICT --------------------------------------------
@pytest.fixture(scope="module")
def main_module():
    mod = _historical("main_for_odd_fence", cc._HISTORY_SHAS["main"])
    if mod is None:
        pytest.skip("main's pre-round-5 claim_check.py is not in this clone")
    return mod


@pytest.mark.parametrize("rel", ODD_FENCE_DOCS)
def test_odd_fence_doc_behaves_like_main_or_stricter(rel, main_module):
    """Round 5 does no fence-awareness at all, so going simpler must not open a hole main did not have. (Round 7
    is looser than main BY DESIGN on correct roundings -- rule A -- but on these four docs not in verdict.)"""
    full = os.path.join(ROOT, rel)
    if not os.path.exists(full):
        pytest.skip("%s not present in this checkout" % rel)
    assert cc.check(full, verbose=False) >= main_module.check(full, verbose=False)


# ---- round 6 tests that still hold ------------------------------------------------------------------------------
def test_invalid_utf8_is_unreadable_and_blocks(tmp_path):
    doc = tmp_path / "bad_utf8.md"
    doc.write_bytes(b"# f\n\nThe accuracy was 0.98" + b"\xad" + b"76 here.\n")
    r = cc._scan(str(doc))
    assert r["unreadable"] and cc._verdict(r) == "FAIL" and cc.check(str(doc), verbose=False) == 1


def test_valid_utf8_with_real_unicode_prose_is_unaffected(casedir):
    p = cc._write_case(casedir, {"name": "valid_unicode_prose",
                                 "doc": _H + "The accuracy improved — genuinely — to 0.170000 here.\n"})
    r = cc._scan(p)
    assert not r["unreadable"] and cc._verdict(r) == "PASS"


def test_glob_file_count_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "MAX_GLOB_FILES", 3)
    for i in range(6):
        json.dump({"v": i + 0.123456}, open(tmp_path / ("f%d.json" % i), "w"))
    pool, _v, loaded, _m, capped = cc.load_artifacts([os.path.join(str(tmp_path), "f*.json")])
    assert len(loaded) == 3 and any("glob cap" in c for c in capped)


# ---- 4. round 7 units --------------------------------------------------------------------------------------------
def _num(text):
    nums, _ids = cc._numbers_in(text)
    return nums


@pytest.mark.parametrize("written,pool,rule", [
    ("0.477", [0.4774], "rounding"),
    ("0.4774", [0.4774], "exact"),
    ("0.478", [0.4774], None),
    ("0.163", [0.1625], "rounding"),            # a half boundary: half-up reading accepted
    ("0.162", [0.1625], "rounding"),            # ... and half-even
    ("12.3456", [12.3449], None),               # tighter than the old relative window
    ("1.235", [1.23456789], "rounding"),
    ("1.525e-1", [0.1525], "exact"),
    ("9.876e-1", [0.98764], "rounding"),        # d = 3 - (-1) = 4
])
def test_rule_a_precision_aware(written, pool, rule):
    (n,) = _num(written)
    got = cc._match(n, sorted(pool))
    assert (got.split("+")[0] if got else None) == rule, (written, pool, got)


def test_rule_a_magnitude_suffix_reads_scaled_or_bare():
    (n,) = _num("a 1.088B model")
    assert cc._match(n, [1088000000.0]) == "exact+B"
    assert cc._match(n, [1.088]) == "exact"
    assert cc._match(n, [1.2e9]) is None


def test_rule_b_is_deterministic_and_tracks_pool_breadth():
    basis = _num("0.4774 0.5123 0.3350")
    narrow = sorted([0.4774, 0.5123, 0.335])
    broad = sorted(i / 10000.0 for i in range(10000))
    r1, r2 = cc._chance_rate(basis, narrow), cc._chance_rate(basis, narrow)
    assert r1 == r2 and r1 < 0.01
    assert cc._chance_rate(basis, broad) > 0.99
    assert cc._chance_rate([], broad) is None


def test_rule_b_decoys_are_representable_for_pasted_17_decimal_floats():
    """A pasted repr float states a unit below its own ULP; decoys must still move (round 7 bug found in retro:
    every 16-17-decimal doc read chance 1.000)."""
    basis = _num("0.30000000000000004")
    assert cc._chance_rate(basis, [0.30000000000000004]) < 0.01


@pytest.mark.parametrize("text,expected", [
    ("gain*0.1525", [0.1525]), ("**acc**0.1525", [0.1525]), ("&Delta;0.1525", [0.1525]),
    ("x\\*0.1525", [0.1525]), ("n*-0.1625", [-0.1625]), ("Δ\u22120.1625", [-0.1625]),
    ("0.412-0.498", [0.412, 0.498]), ("0.412 – 0.498", [0.412, 0.498]), ("lr0.001", []),
    ("foo_0.125", []), ("4*0.170", [0.17]), ("0.15<b>25</b>", [0.1525]), ("0.15\u034f25", [0.1525]),
    ("0\u00b71525", [0.1525]), ("see arXiv:2403.12345", []), ("doi:10.1038/415429a", []),
    ("raw/sweep_0.125/x.json", []), ("-**0.1625**", [-0.1625]), ("\uff10.\uff11\uff15\uff12\uff15", [0.1525]),
    ("-0.1625", [-0.1625]), ("\u22120.1625", [-0.1625]),       # a sign at SEGMENT START (round 7 bug, caught)
    ("(0.5)-0.1625", [0.1625]), ("1.088B", [1.088]), ("3.490537...", [3.490537]),
])
def test_normalization_never_glues_and_never_hides(text, expected):
    assert [round(n.value, 6) for n in _num(text)] == expected, text


def test_normalization_property_markup_before_a_number_never_hides_it():
    """Property (issue 3/4): whatever markup/invisible character sits between a word and a number, the number is
    still found, with its sign."""
    rng = random.Random(0)
    seps = ["*", "**", "_", "`", "~~", "<b>", "</i>", "<wbr>", "&#8203;", "\u200b", "\u034f", "\ufe0f", "\u3164",
            "&nbsp;", " ", "<span class=x>", "\\*"]
    for _ in range(300):
        word = rng.choice(["gain", "acc", "Δ", "n", "x"])
        sep = "".join(rng.choice(seps) for _ in range(rng.randint(1, 3)))
        sign = rng.choice(["", "-", "\u2212", "—" if word == "Δ" else "-"])
        got = [round(n.value, 6) for n in _num(word + sep + sign + "0.1625")]
        want = -0.1625 if sign else 0.1625
        assert got == [want], (word, sep, sign, got)


def test_hidden_spans_cover_every_carrier():
    text = ("a <!-- c1 -->\n<!--\nmulti\n-->\n[//]: # (x)\n[ref]: http://x \"t\"\n<div hidden>h</div>\n"
            "<span style=\"display:none\">s</span>\n<!-- unclosed")
    kinds = {k for _a, _b, k in cc._hidden_spans(text)}
    assert kinds == {"comment", "linkref", "hidden-element", "comment-unclosed"}
    assert all(k != "marker" for k in kinds)
    assert [k for _a, _b, k in cc._hidden_spans("x <!--derived: note--> y")] == ["marker"]


@pytest.mark.parametrize("doc,src,title", [
    ("---\ntitle: T GO\n---\n# Notes\n", "frontmatter title", "T GO"),
    ("```\n# not a title\n```\n# Real\n", "H1", "Real"),
    ("Real\n====\n", "setext H1", "Real"),
    ("no heading at all\n", "filename", "some-file"),
])
def test_doc_title_chain(doc, src, title):
    m = cc._FRONTMATTER_RE.match(doc)
    assert cc._doc_title(doc, m, "x/some-file.md") == (src, title)


# ---- issue 6: every author-facing message says the SAME thing about marker scope --------------------------------
@pytest.mark.parametrize("rel", ["tools/githooks/pre-commit", "tools/finding_lint.py",
                                 ".claude/skills/neural-simulator/SKILL.md", "tools/claim_check.py"])
def test_author_facing_text_states_the_cell_rule(rel):
    text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    assert "SAME table cell or <br>-segment" in text, rel
    assert "on the SAME LINE as the number" not in text and "on THAT SAME PHYSICAL LINE" not in text, rel
    assert not re.search(r"including every row of a derived table", text), rel
