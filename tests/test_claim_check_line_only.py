"""tools/claim_check.py derived-marker scoping (round 5, SAME-LINE-ONLY) -- both directions, one test per hole.

Round 5 replaces every earlier scope rule (main's standalone-marker-to-next-heading, and rounds 1-4's
progressively more elaborate line-scanner / CommonMark-parser scoping) with NO scope at all: a number is exempt
only when the literal `<!--derived-->` marker sits on ITS OWN physical line. Three layers here, mirroring the
round-4 test file's own methodology:

  1. every SELFTEST_CASES entry gives its expected verdict, and a FAIL case must flag its designated WRONG
     number (0.1525 / 0.14 / 1.23456 -- never the placeholder "derived" numbers 0.104615 / 0.207531 / 0.311079,
     which are legitimate values absent from the artifact, not smuggled wrong ones); a PASS case must flag
     nothing.
  2. HISTORY: every case is re-run through each earlier checker (main before round 5, and rounds 1-4, read
     straight from git) -- the case's recorded `wrong_on` must equal the set of earlier revisions that actually
     get it wrong, so "this used to pass, now it fails" is RE-DERIVED every run, never just remembered.
  3. the four real findings with an odd number of fence lines behave like main or stricter (never looser): round
     5 does no fence-awareness at all, so this is a basic sanity check that going simpler did not silently open
     a hole main did not already have.
"""
from __future__ import annotations

import importlib.util
import json
import os
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


@pytest.fixture(scope="module")
def casedir():
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test_claim_check_line_only_") as d:
        yield d


def _nums_in(text):
    return {round(float(m), 6) for m in cc.NUM_RE.findall(text)}


# ---- 1. the registry (both directions) -----------------------------------------------------------------------
def test_selftest_is_clean():
    assert cc.selftest() == []


def test_every_registry_case_names_a_real_historical_revision():
    names = [c["name"] for c in cc.SELFTEST_CASES]
    assert len(names) == len(set(names)), "duplicate case name"
    for c in cc.SELFTEST_CASES:
        assert set(c["wrong_on"]) <= set(cc._HISTORY_SHAS), c["name"]


@pytest.mark.parametrize("case", cc.SELFTEST_CASES, ids=[c["name"] for c in cc.SELFTEST_CASES])
def test_case_verdict_and_reason(case, casedir):
    p = cc._write_case(casedir, case)
    r = cc._scan(p)
    got = cc._verdict(r)
    assert got == case["expect"], case["why"]
    flagged = {round(v, 6) for _ln, v, _c in r["unsupported"]}
    if case["expect"] == "FAIL":
        # a LOW COVERAGE case (e.g. low_coverage_overmarked) fails without any single "wrong" number -- the
        # defect is the suppression ratio itself, not a specific unsupported value.
        assert (flagged & cc.WRONG_VALUES) or r["low_coverage"], \
            "must flag its designated wrong number or trip LOW COVERAGE: %s" % case["why"]
    else:
        assert not flagged and not r["low_coverage"], "a clean PASS case must flag nothing"


# ---- 2. history: each case against every earlier checker, wrong_on RE-DERIVED, not remembered -----------------
def _historical(tag, sha):
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", "%s:tools/claim_check.py" % sha],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    mod_path = os.path.join(ROOT, ".claim_check_hist_%s.py" % tag)
    try:
        open(mod_path, "w", encoding="utf-8").write(src)
        spec = importlib.util.spec_from_file_location("claim_check_hist_%s" % tag, mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if os.path.exists(mod_path):
            os.remove(mod_path)
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


@pytest.mark.parametrize("case", cc.SELFTEST_CASES, ids=[c["name"] for c in cc.SELFTEST_CASES])
def test_case_wrong_on_is_re_derived(case, casedir, history):
    p = cc._write_case(casedir, case)
    wrong = set()
    for tag, mod in history.items():
        # ROUND 6 (issue 8): round 4's `check()` lazily imports markdown-it-py; without the package installed,
        # calling it raises ModuleNotFoundError for a reason that has NOTHING to do with any case's own bug,
        # which would otherwise mark r4 "wrong" (or "right") by accident of environment rather than by the
        # actual regression under test. Skip r4 specifically when the dependency is absent -- every other
        # historical tag is still fully re-derived.
        if tag == "r4" and not _MARKDOWN_IT_AVAILABLE:
            continue
        try:
            rc = mod.check(p, verbose=False)
            got = "FAIL" if rc else "PASS"
        except Exception:
            # A crash also blocks a commit (an uncaught exception exits non-zero) -- treat it as equivalent to
            # a clean FAIL return, not as "wrong for an unrelated reason". This matters for round 6's own new
            # UTF-8 case: every pre-round-5 revision reads the doc with a bare `open(doc_path).read()` and
            # CRASHES on invalid UTF-8 (which does block), while round 5 uniquely swallows it via
            # errors="replace" and returns a clean, wrong PASS.
            got = "FAIL"
        if got != case["expect"]:
            wrong.add(tag)
    assert wrong == set(case["wrong_on"]) - ({"r4"} if not _MARKDOWN_IT_AVAILABLE else set()), (
        "recorded wrong_on=%s but this revision-set actually gets it wrong: %s (%s)"
        % (sorted(case["wrong_on"]), sorted(wrong), case["why"]))


# ---- 3. spec guards: round 5's own contract, both directions --------------------------------------------------
_H = "# Some finding\n\nArtifact: `%(art)s`\n\n"


@pytest.mark.parametrize("name,body,flagged_expected", [
    # the ONE thing round 5 allows: marker and number on the same physical line
    ("inline_marker_same_line_exempts", _H + "The ratio is 0.104615 here. <!--derived-->\n"
                                             "The baseline was 0.162500 here.\n", set()),
    # a marker on the line BEFORE a number does not reach it -- no exceptions, this is the whole point
    ("marker_one_line_early_does_not_reach", _H + "<!--derived-->\nThe accuracy was 0.1525 here.\n", {0.1525}),
    # a marker on the line AFTER a number does not reach it either (no "look-back" any more than "look-forward")
    ("marker_one_line_late_does_not_reach", _H + "The accuracy was 0.1525 here.\n<!--derived-->\n", {0.1525}),
    # ROUND 6 (issue 4): a table whose derived rows carry the marker IN THE SAME CELL as the value passes --
    # exemption is now scoped to the marker's own CELL, not the whole row, so the marker must share a cell
    # with the number it exempts.
    ("every_derived_table_row_marked_passes",
     _H + "| metric | value |\n|---|---|\n| ratio | 0.104615 <!--derived--> |\n"
          "| gap | 0.207531 <!--derived--> |\n| accuracy | 0.170000 |\n", set()),
    # ... but ONE unmarked row in an otherwise-marked table is still checked (and flagged if wrong)
    ("one_unmarked_table_row_is_checked",
     _H + "| metric | value |\n|---|---|\n| ratio | 0.104615 <!--derived--> |\n| accuracy | 0.1525 |\n", {0.1525}),
    # ROUND 6 (issue 4), the exact incident repro: a marker ALONE in its own trailing cell no longer reaches a
    # DIFFERENT cell's number on the same row -- round 5's whole-line rule let this exempt the entire row.
    ("marker_in_separate_cell_does_not_reach_other_cells",
     _H + "| metric | value | delta |\n|---|---|---|\n| ratio | 0.1525 | 0.104615 | <!--derived--> |\n",
     {0.1525, 0.104615}),
    # ROUND 6 (issue 4), repro 2: an HTML <br> renders as two lines to a reader though it is one physical line
    # here -- a marker before the <br> must not reach a number after it.
    ("br_split_marker_does_not_reach_other_side",
     _H + "ratio 0.104615 <!--derived--><br>accuracy 0.1525\n", {0.1525}),
    # ROUND 6 (issue 4): the cap on a marked line/cell's free exemption -- the 9th number is checked normally.
    ("cap_bounds_a_marked_lines_free_exemption",
     _H + "vals 0.100001 0.100002 0.100003 0.100004 0.100005 0.100006 0.100007 0.100008 0.1525 <!--derived-->\n",
     {0.1525}),
    # a marker inside a fenced code block is still just text on that line -- exempts nothing outside the fence
    ("marker_inside_fence_does_not_leak_out",
     _H + "```\n<!--derived-->\n```\nThe accuracy was 0.1525 here.\n", {0.1525}),
    # a `# derived` code COMMENT is not a marker at all (the literal string must be `<!--derived-->`)
    ("hash_derived_comment_is_not_a_marker",
     _H + "```python\n# derived\n```\nThe accuracy was 0.1525 here.\n", {0.1525}),
])
def test_spec_guard(name, body, flagged_expected, casedir):
    case = {"name": name, "doc": body}
    p = cc._write_case(casedir, case)
    r = cc._scan(p)
    flagged = {round(v, 6) for _ln, v, _c in r["unsupported"]}
    assert flagged == flagged_expected, name


# ---- 4. WARNINGs for the now-inert pre-round-5 idioms: printed, never exempting ---------------------------------
@pytest.mark.parametrize("name,body,warn_kind_substr", [
    ("standalone_marker_warns", _H + "<!--derived-->\nThe accuracy was 0.1525 here.\n", "standalone marker"),
    ("atx_derived_heading_warns", _H + "## Derived\nThe accuracy was 0.1525 here.\n", "heading"),
    ("h1_derived_heading_warns", _H + "# Derived\nThe accuracy was 0.1525 here.\n", "heading"),
    ("blockquoted_derived_heading_warns", _H + "> ## Derived\nThe accuracy was 0.1525 here.\n", "heading"),
    ("setext_derived_heading_warns", _H + "Derived\n=======\nThe accuracy was 0.1525 here.\n", "heading"),
    ("close_marker_warns", _H + "The ratio is 0.104615. <!--/derived-->\nThe accuracy was 0.1525 here.\n",
     "close marker"),
])
def test_inert_idiom_warns_but_does_not_exempt(name, body, warn_kind_substr, casedir):
    case = {"name": name, "doc": body}
    p = cc._write_case(casedir, case)
    r = cc._scan(p)
    flagged = {round(v, 6) for _ln, v, _c in r["unsupported"]}
    assert 0.1525 in flagged, "%s: the idiom must not exempt anything" % name
    assert any(warn_kind_substr in kind.lower() for _ln, kind, _msg in r["warnings"]), \
        "%s: expected a WARNING mentioning %r, got %s" % (name, warn_kind_substr, r["warnings"])


def test_clean_doc_has_no_warnings(casedir):
    case = {"name": "clean_doc_no_warnings",
            "doc": _H + "The ratio is 0.104615 here. <!--derived-->\nThe baseline was 0.162500 here.\n"}
    p = cc._write_case(casedir, case)
    r = cc._scan(p)
    assert r["warnings"] == []


# ---- 5. the four real odd-fence docs: line-only must be at least as strict as main -----------------------------
@pytest.fixture(scope="module")
def main_module():
    mod = _historical("main_for_odd_fence", cc._HISTORY_SHAS["main"])
    if mod is None:
        pytest.skip("main's pre-round-5 claim_check.py is not in this clone")
    return mod


@pytest.mark.parametrize("rel", ODD_FENCE_DOCS)
def test_odd_fence_doc_behaves_like_main_or_stricter(rel, main_module):
    full = os.path.join(ROOT, rel)
    if not os.path.exists(full):
        pytest.skip("%s not present in this checkout" % rel)
    rc_main = main_module.check(full, verbose=False)
    rc_new = cc.check(full, verbose=False)
    assert rc_new >= rc_main, (
        "%s: main verdict=%s but round-5 verdict=%s -- round 5 must never be LOOSER than main"
        % (rel, "FAIL" if rc_main else "PASS", "FAIL" if rc_new else "PASS"))


# ---- 6. round 6 (2026-09-25): fixes from round 5's own SOUND-WITH-ISSUES review ---------------------------------

# issue 1 -- strict UTF-8, a blocking UNREADABLE result instead of a silent errors="replace" false-pass.
def test_invalid_utf8_is_unreadable_and_blocks(tmp_path):
    doc = tmp_path / "bad_utf8.md"
    # a bare 0xAD byte (Latin-1 SOFT HYPHEN) is not valid UTF-8 on its own.
    doc.write_bytes(b"# f\n\nThe accuracy was 0.98" + b"\xad" + b"76 here.\n")
    r = cc._scan(str(doc))
    assert r["unreadable"], "invalid UTF-8 must be reported, not silently decoded"
    assert cc._verdict(r) == "FAIL"
    assert cc.check(str(doc), verbose=False) == 1


def test_valid_utf8_with_real_unicode_prose_is_unaffected(casedir):
    # a legitimate em-dash / accented text doc must decode and scan normally -- strict decoding must not
    # false-block ordinary valid UTF-8 content.
    case = {"name": "valid_unicode_prose",
            "doc": _H + "The accuracy improved — genuinely — to 0.170000 here.\n"}
    p = cc._write_case(casedir, case)
    r = cc._scan(p)
    assert not r["unreadable"]
    assert cc._verdict(r) == "PASS"


# issue 9 -- glob file-count and artifact value-pool caps.
def test_glob_file_count_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "MAX_GLOB_FILES", 3)
    for i in range(6):
        json.dump({"v": i + 0.123456}, open(tmp_path / ("f%d.json" % i), "w"))
    pattern = os.path.join(str(tmp_path), "f*.json")
    nums, verdicts, loaded, missing, capped = cc.load_artifacts([pattern])
    assert len(loaded) == 3, "glob must be truncated to MAX_GLOB_FILES"
    assert any("glob cap" in c for c in capped)


# issue 9 -- a citation hidden inside an HTML comment must be ignored, so it cannot silently validate a claim
# that carries no VISIBLE citation. Needs a controlled artifact (unlike the shared %(art)s fixture), and needs
# the historical modules to demonstrate the regression, so it lives here rather than in SELFTEST_CASES.
def test_citation_inside_html_comment_is_ignored(casedir, history):
    art_abs = os.path.join(casedir, "art.json")
    if not os.path.exists(art_abs):
        json.dump({"accuracy": 0.17, "baseline": 0.1625}, open(art_abs, "w"))
    art_rel = os.path.relpath(art_abs, ROOT).replace(os.sep, "/")
    doc_path = os.path.join(casedir, "hidden_citation.md")
    # NO visible citation anywhere in the doc -- only inside an HTML comment.
    open(doc_path, "w", encoding="utf-8").write(
        "# Some finding\n\n<!-- see `%s` for context -->\n\nThe baseline was 0.162500 here.\n" % art_rel)
    r = cc._scan(doc_path)
    assert r["cited"] == [], "a citation living only inside an HTML comment must not be resolved"
    assert cc._verdict(r) == "FAIL", "with no visible citation, the stated number must be UNSUPPORTED"
    for tag, mod in history.items():
        if tag == "r4" and not _MARKDOWN_IT_AVAILABLE:
            continue
        rc = mod.check(doc_path, verbose=False)
        assert rc == 0, ("%s: a hidden HTML-comment citation used to be resolved, letting a claim with NO "
                         "visible citation pass" % tag)


def test_artifact_value_pool_is_capped(tmp_path, monkeypatch):
    # the cap is checked BETWEEN files (coarse but matches real usage: a citation attack pools many files, not
    # one giant one) -- ten small files, each with a distinct value, cap well below the total.
    monkeypatch.setattr(cc, "MAX_ARTIFACT_VALUES", 5)
    paths = []
    for i in range(10):
        p = tmp_path / ("f%d.json" % i)
        json.dump({"v": i + 0.111111}, open(p, "w"))
        paths.append(str(p))
    nums, verdicts, loaded, missing, capped = cc.load_artifacts(paths)
    assert len(nums) <= 5
    assert len(loaded) < len(paths), "loading must stop once the distinct-value pool cap is reached"
    assert any("VALUE pool cap" in c for c in capped)
