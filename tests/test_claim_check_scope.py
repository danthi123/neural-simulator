"""tools/claim_check.py derived-marker scoping (round 4, CommonMark parser) -- both directions, one test per hole.

Three layers:
  1. every SELFTEST_CASES entry gives its expected verdict, and a FAIL case fails on exactly its WRONG numbers
     (never on a derived one);
  2. HISTORY: every case is re-run through each earlier checker (main before 2026-09-25 and rounds 1-3, read
     from git), and the case's recorded `wrong_on` must equal the rounds that actually get it wrong -- so the
     registry's "fails on an earlier round, passes now" claim is re-derived, not remembered;
  3. spec guards that no earlier round got wrong (so they are not registry cases), plus the four real findings
     with an odd number of fence lines.
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

WRONG = {0.1525, 0.14, 1.23456}                      # numbers the case artifact does NOT hold
DERIVED = {0.104615, 0.207531, 0.311079}             # numbers the cases mark as derived
HISTORY = {"main": "7e2edc08e", "r1": "d4959ecb0", "r2": "6abb28469", "r3": "662e167e8"}
ODD_FENCE_DOCS = [
    "research/findings/2026-06-11-dual-CLS-cortex-channel-derisk-GO.md",
    "research/findings/2026-06-17-ordered-wm-position-binding-derisk.md",
    "research/findings/2026-06-20-S5-divisive-norm-derisk.md",
    "research/findings/2026-06-26-multibridge-deep-knowledge-design.md",
]


@pytest.fixture(scope="module")
def casedir():
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test_claim_check_scope_") as d:
        yield d


def _doc(d, name, body):
    art_abs = os.path.join(d, "art.json")
    if not os.path.exists(art_abs):
        json.dump({"accuracy": 0.17, "baseline": 0.1625}, open(art_abs, "w"))
    art = os.path.relpath(art_abs, ROOT).replace(os.sep, "/")
    p = os.path.join(d, name + ".md")
    open(p, "w", encoding="utf-8").write(body % {"art": art})
    return p


def _flagged(path):
    return {round(v, 6) for _ln, v, _ctx in cc._scan(path)["unsupported"]}


def _nums_in(text):
    return {round(float(m), 6) for m in cc.NUM_RE.findall(text)}


# ---- 1. the registry ---------------------------------------------------------------------------------------
def test_selftest_is_clean():
    assert cc.selftest() == []


def test_every_registry_case_names_an_earlier_round():
    names = [c["name"] for c in cc.SELFTEST_CASES]
    assert len(names) == len(set(names))
    for c in cc.SELFTEST_CASES:
        assert c["wrong_on"], "%s: a registry case must be wrong on at least one earlier round" % c["name"]
        assert set(c["wrong_on"]) <= set(HISTORY), c["name"]


@pytest.mark.parametrize("case", cc.SELFTEST_CASES, ids=[c["name"] for c in cc.SELFTEST_CASES])
def test_case_verdict_and_reason(case, casedir):
    p = cc._write_case(casedir, case)
    r = cc._scan(p)
    got = "FAIL" if cc.check(p, verbose=False) else "PASS"
    assert got == case["expect"], case["why"]
    flagged = {round(v, 6) for _ln, v, _c in r["unsupported"]}
    assert not flagged & DERIVED, "a derived number was flagged: %s" % sorted(flagged & DERIVED)
    if case["name"] == "low_coverage_overmarked":
        assert r["low_coverage"] and not flagged
    elif case["expect"] == "FAIL":
        assert flagged == _nums_in(case["doc"]) & WRONG, "must fail on exactly its wrong numbers"
    else:
        assert not flagged and not r["low_coverage"]


# ---- 2. history: each case against every earlier checker ---------------------------------------------------
def _historical(tag):
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", "%s:tools/claim_check.py" % HISTORY[tag]],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    mod_path = os.path.join(ROOT, ".claim_check_hist_%s.py" % tag)
    try:
        open(mod_path, "w").write(src)
        spec = importlib.util.spec_from_file_location("claim_check_hist_%s" % tag, mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.remove(mod_path)
    mod.ROOT = ROOT
    return mod


@pytest.fixture(scope="module")
def history():
    mods = {tag: _historical(tag) for tag in HISTORY}
    if any(m is None for m in mods.values()):
        pytest.skip("earlier claim_check revisions are not in this clone (%s)"
                    % ", ".join(t for t, m in mods.items() if m is None))
    return mods


@pytest.mark.parametrize("case", cc.SELFTEST_CASES, ids=[c["name"] for c in cc.SELFTEST_CASES])
def test_case_fails_on_the_recorded_earlier_rounds(case, casedir, history):
    p = cc._write_case(casedir, case)
    wrong = tuple(tag for tag, mod in history.items()
                  if ("FAIL" if mod.check(p, verbose=False) else "PASS") != case["expect"])
    assert wrong == tuple(case["wrong_on"])


# ---- 3. spec guards (both directions) that no earlier round got wrong --------------------------------------
_H = "# Some finding\n\nArtifact: `%(art)s`\n\n"


@pytest.mark.parametrize("name,body,flag", [
    # a fence is never a sibling scope on its own (an unclosed one runs to EOF), so its numbers are checked
    ("fence_is_not_a_sibling_scope", _H + "<!--derived-->\n```\nvalue = 0.1525\n```\n", {0.1525}),
    # round 3 case (g): intro, fence, then the number -- CommonMark ends the intro paragraph at the fence, so the
    # number sits in a NEW paragraph outside the standalone marker's scope; `range_across_fence` is the fix
    ("paragraph_after_fence_is_a_new_sibling",
     _H + "<!--derived-->\nComputed below:\n```\nx\n```\nThe ratio 0.104615 and accuracy 0.1525.\n", {0.104615, 0.1525}),
    # a close marker in an inline code span is text: it must not pair with an opener and swallow a heading
    ("close_marker_in_code_span_is_text",
     _H + "<!--derived-->\nratio 0.104615\n\n## Results\nThe accuracy was 0.1525 here.\n\n"
          "Close a range with `<!--/derived-->` when needed.\n", {0.1525}),
    # a marker in an inline code span exempts nothing
    ("marker_in_code_span_is_text", _H + "Mark with `<!--derived-->`: the accuracy was 0.1525.\n", {0.1525}),
    # an inline marker exempts its own line only
    ("inline_marker_own_line_only", _H + "ratio 0.104615 <!--derived-->\nThe accuracy was 0.1525.\n", {0.1525}),
    # a standalone marker in a blockquote scopes inside the blockquote; the lazy line after it is checked
    ("blockquote_lazy_line", _H + "> <!--derived-->\n> ratio 0.104615\nThe accuracy was 0.1525.\n", {0.1525}),
    # frontmatter is blanked for the parser: `derived: ...` + `---` is not a setext "Derived" heading
    ("frontmatter_is_not_a_heading",
     "---\nstatus: live\nderived: nothing\n---\n" + _H + "The accuracy was 0.1525 here.\n", {0.1525}),
    # an unpaired close marker does nothing (a later number is still checked)
    ("unpaired_close_is_inert", _H + "Text. <!--/derived--> The accuracy was 0.1525.\n", {0.1525}),
    # PASS direction
    ("derived_table_then_heading", _H + "The accuracy is 0.170000.\n\n<!--derived-->\n| m | v |\n|---|---|\n"
                                        "| ratio | 0.104615 |\n\n## Next\n\nThe baseline was 0.162500.\n", set()),
    ("blank_line_between_marker_and_paragraph", _H + "<!--derived-->\n\nratio 0.104615\n", set()),
    ("hash_derived_in_fence_with_real_marker",
     _H + "```\n# derived\nx = 1\n```\n<!--derived-->\nratio 0.104615\n", set()),
    ("pipeless_table_header_row_only_rows_with_pipes",
     _H + "<!--derived-->\nm | v\n--|--\nratio | 0.104615\n", set()),
    ("setext_derived_heading", _H + "Derived\n-------\nratio 0.104615\n\nNext\n----\naccuracy 0.170000\n", set()),
    ("short_all_derived_note", _H + "The ratio here is 0.104615. <!--derived-->\n", set()),
])
def test_spec_guard(name, body, flag, casedir):
    p = _doc(casedir, name, body)
    assert _flagged(p) == {round(v, 6) for v in flag}
    assert cc.check(p, verbose=False) == (1 if flag else 0)


def test_synthesis_doc_still_needs_a_citation(casedir):
    ok = _doc(casedir, "synth_ok", "---\nclaim_check: synthesis\n---\n" + _H + "Quoted 0.1525.\n")
    r = cc._scan(ok)
    assert cc.check(ok, verbose=False) == 0 and r["suppressed"]["synthesis"] == 1 and r["checked"] == 0
    bad = _doc(casedir, "synth_uncited", "---\nclaim_check: synthesis\n---\n# x\n\nQuoted 0.1525.\n")
    assert cc.check(bad, verbose=False) == 1


def test_low_coverage_floor_and_fraction(casedir):
    n = cc.LOW_COVERAGE_MIN_TOTAL
    under = _doc(casedir, "floor_under", _H + "\n\n".join(
        "v 0.%06d <!--derived-->" % (i * 7 + 1) for i in range(n - 1)) + "\n")
    at = _doc(casedir, "floor_at", _H + "\n\n".join(
        "v 0.%06d <!--derived-->" % (i * 7 + 1) for i in range(n)) + "\n")
    assert not cc._scan(under)["low_coverage"]
    assert cc._scan(at)["low_coverage"]
    # 3 of 45 checked = 6.7% (>= 5%) passes; 2 of 45 = 4.4% fails
    body = lambda k: _H + "\n\n".join(["accuracy 0.170000"] * k + ["v 0.%06d <!--derived-->" % (i * 7 + 1)
                                                                   for i in range(45 - k)]) + "\n"
    assert not cc._scan(_doc(casedir, "frac3", body(3)))["low_coverage"]
    assert cc._scan(_doc(casedir, "frac2", body(2)))["low_coverage"]


def test_suppressed_counts_are_reported_by_mechanism(casedir, capsys):
    p = _doc(casedir, "mech", _H + "## Derived\nd 0.104615\n\n## Body\n<!--derived-->\n- 0.207531\n\n"
                                   "x 0.311079 <!--derived-->\n\n<!--derived-->\na 0.104615\n\nb 0.207531\n"
                                   "<!--/derived-->\naccuracy 0.170000\n")
    r = cc._scan(p)
    assert r["suppressed"] == {"section": 1, "range": 2, "block": 1, "inline": 1, "synthesis": 0}
    assert r["checked"] == 1
    cc.check(p, verbose=True)
    out = capsys.readouterr().out
    assert "section=1 range=2 block=1 inline=1" in out


# ---- the four real findings with an odd number of fence lines ----------------------------------------------
@pytest.mark.parametrize("rel", ODD_FENCE_DOCS)
def test_odd_fence_docs_neither_smuggle_nor_false_fail(rel, capsys):
    full = os.path.join(ROOT, rel)
    if not os.path.exists(full):
        pytest.skip("%s not in this checkout" % rel)
    r = cc._scan(full)
    sc = cc.derived_scope(open(full, encoding="utf-8").read())
    # the stray last fence line is the one unclosed fence, CommonMark-style, and it opens on the final lines
    assert len(sc["unclosed_fences"]) == 1 and sc["unclosed_fences"][0] >= len(sc["lines"]) - 3
    # nothing is exempted: every numeric claim in the document is checked (no smuggling)
    assert sum(r["suppressed"].values()) == 0 and r["checked"] == r["total_numeric"]
    # and main (no fence handling at all, no markers here) flags exactly the same numbers (no false failure)
    main = _historical("main")
    if main is None:
        pytest.skip("main's claim_check revision is not in this clone")
    capsys.readouterr()
    main_rc = main.check(full, verbose=True)
    main_flags = [ln for ln in capsys.readouterr().out.splitlines() if "not in any cited artifact" in ln]
    assert cc.check(full, verbose=False) == main_rc
    assert len(main_flags) == min(len(r["unsupported"]), 12)


def test_finding_lint_consumes_structured_results(casedir):
    from tools import finding_lint
    p = _doc(casedir, "lint", _H + "The accuracy was 0.1525 here.\n")
    rc, unsupported, missing, low_cov, scan = finding_lint.run_claim_check(p)
    assert rc == 1 and missing == [] and low_cov == []
    assert [(ln, v) for ln, v, _c in unsupported] == [(5, 0.1525)]
    assert scan["checked"] == 1


def test_registry_gate_contract():
    from tools.gates import claim_check_scope as gate
    assert gate.selftest() == []
    assert gate.check([]) == [] and gate.check(["tools/claim_check.py"]) == []
