"""tools/claim_check.py, round 8 (FAIL CLOSED) -- every repro in the history, both directions, re-derived from git.

  1. REGISTRY: every tools/claim_check_cases.py entry gets its expected verdict for its stated reason (a FAIL case
     must flag its designated wrong number or trip its reason; a PASS case flags nothing).
  2. HISTORY: every case is re-run through each earlier checker (main, r1-r7, read straight from git) and its
     recorded `wrong_on` must equal the set of revisions that ACTUALLY get it wrong -- "this used to pass, now it
     fails" is re-derived on every run, never remembered. Every round-7-review repro must be wrong on r7.
  3. SPEC GUARDS for the round-8 contract: nothing deleted/hidden (the normalized copy is 1:1), markers verified by
     markdown-it, exact spellings only, cell scope + cap 8, precision-aware rounding, the per-claim chance rate.
  4. The four real findings with an odd number of fence lines behave like main or stricter.
  5. The CCT registry gate reports a broken instrument verbatim.
  6. A seeded differential fuzz over the whole hole vocabulary: one wrong number in any markup context fails the
     doc; correct or properly marked numbers pass.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import random
import re
import subprocess
import sys
import tempfile
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tools.claim_check as cc                      # noqa: E402

CASES = cc.SELFTEST_CASES
ODD_FENCE_DOCS = [
    "research/findings/2026-06-11-dual-CLS-cortex-channel-derisk-GO.md",
    "research/findings/2026-06-17-ordered-wm-position-binding-derisk.md",
    "research/findings/2026-06-20-S5-divisive-norm-derisk.md",
    "research/findings/2026-06-26-multibridge-deep-knowledge-design.md",
]
_H = "# Some finding\n\nArtifact: `%(art)s`\n\n"


@pytest.fixture(scope="module")
def casedir():
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test_claim_check_r8_") as d:
        yield d


def _scan_doc(casedir, name, body, artifact=None):
    case = {"name": name, "doc": body}
    if artifact is not None:
        case["artifact"] = artifact
    return cc._scan(cc._write_case(casedir, case))


def _flagged(r):
    return {round(v, 6) for _l, v, _c in r["unsupported"]} | {round(v, 6) for _l, v, _ch, _c in r["too_broad"]}


# ---- 1. the registry -------------------------------------------------------------------------------------------
def test_selftest_is_clean():
    assert cc.selftest() == []


def test_registry_is_well_formed():
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names)), "duplicate case name"
    for c in CASES:
        assert set(c["wrong_on"]) <= set(cc._HISTORY_SHAS), c["name"]
        assert c["expect"] in ("PASS", "FAIL"), c["name"]
        assert c.get("why"), c["name"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_verdict_and_reason(case, casedir):
    p = cc._write_case(casedir, case)
    r = cc._scan(p)
    printed = None
    if case.get("expect_output"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cc.check(p)
        printed = buf.getvalue()
    assert cc._case_problems(case, r, printed) == []


# ---- 2. history: each case against every earlier checker, wrong_on RE-DERIVED ------------------------------------
def _historical(tag, sha):
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", "%s:tools/claim_check.py" % sha],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    mod_path = os.path.join(ROOT, ".claim_check_hist_%s_%d.py" % (tag, os.getpid()))
    try:
        with open(mod_path, "w", encoding="utf-8") as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location("claim_check_hist_%s" % tag, mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        return None
    finally:
        if os.path.exists(mod_path):
            os.remove(mod_path)
    mod.ROOT = ROOT
    return mod


@pytest.fixture(scope="module")
def history():
    mods = {tag: _historical(tag, sha) for tag, sha in cc._HISTORY_SHAS.items()}
    missing = [t for t, m in mods.items() if m is None]
    if missing:
        pytest.skip("earlier claim_check revisions are not loadable in this clone: %s" % ", ".join(missing))
    return mods


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_wrong_on_is_re_derived(case, casedir, history):
    p = cc._write_case(casedir, case)
    wrong = set()
    for tag, mod in history.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = "FAIL" if mod.check(p, verbose=False) else "PASS"
        except Exception:
            got = None                              # a crash is also "not the expected verdict"
        if got != case["expect"]:
            wrong.add(tag)
    assert wrong == set(case["wrong_on"]), (
        "recorded wrong_on=%s but the revisions that actually get it wrong are %s (%s)"
        % (sorted(case["wrong_on"]), sorted(wrong), case["why"]))


def test_every_round7_review_repro_is_wrong_on_r7_and_caught_now():
    r7 = [c for c in CASES if c["name"].startswith("r7_")]
    assert len(r7) >= 5
    for c in r7:
        assert c["expect"] == "FAIL" and "r7" in c["wrong_on"], c["name"]


def test_main_and_r5_holes_are_all_registered_and_caught():
    """Every revision in the history contributes at least one repro that it gets wrong and round 8 catches."""
    for tag in cc._HISTORY_SHAS:
        assert any(c["expect"] == "FAIL" and tag in c["wrong_on"] for c in CASES), tag


# ---- 3. spec guards ---------------------------------------------------------------------------------------------
def test_normalized_copy_is_one_to_one():
    zw, minus, fdot, fzero = chr(0x200B), chr(0x2212), chr(0xFF0E), chr(0xFF10)
    s = "a" + zw + "0.15" + zw + "25 " + minus + "0.1625 " + fzero + fdot + "1525 x"
    n = cc._n_copy(s)
    assert len(n) == len(s)
    assert zw not in n and " 0.15 25 -0.1625 0.1525 x" in n


@pytest.mark.parametrize("written,stored,ok", [
    ("0.477", 0.4774, True), ("0.4775", 0.47745, True), ("0.478", 0.4774, False), ("1.235", 1.23456789, True),
    ("-0.1625", -0.16248, True), ("0.170", 0.17, True), ("0.1700", 0.17004, True), ("0.1701", 0.17, False),
])
def test_precision_aware_rounding(written, stored, ok, casedir):
    r = _scan_doc(casedir, "round_%s_%s" % (written, stored), _H + "The value was %s here.\n" % written,
                  artifact={"x": stored})
    assert (cc._verdict(r) == "PASS") is ok, (written, stored, r["unsupported"], r["too_broad"])
    if ok:
        assert {x["rule"] for x in r["records"]} <= {"exact", "rounding", "legacy"}


def test_rule_used_is_reported(casedir):
    r = _scan_doc(casedir, "rules_reported", _H + "Exact 0.170 and rounded 1.235 and loose 12.3456 here.\n",
                  artifact={"a": 0.17, "b": 1.23456789, "c": 12.3449})
    assert r["matched"] == {"exact": 1, "rounding": 1, "legacy": 1}


def test_marked_derived_cell_passes_and_unmarked_cell_is_checked(casedir):
    ok = _scan_doc(casedir, "cell_ok", _H + "| m | v |\n|---|---|\n| ratio | 0.104615 <!--derived--> |\n")
    assert cc._verdict(ok) == "PASS"
    bad = _scan_doc(casedir, "cell_bad", _H + "| m | v | w |\n|---|---|---|\n| ratio | 0.104615 <!--derived--> | "
                                             "0.1525 |\n")
    assert _flagged(bad) == {0.1525}


@pytest.mark.parametrize("spelling,live", [
    ("<!--derived-->", True), ("<!--derived: mean of 3 seeds-->", True), ("<!--derived:-->", False),
    ("<!-- derived -->", False), ("<!--Derived-->", False), ("<!--derived-from x-->", False),
    ("<!--derived:mean-->", False), ("`<!--derived-->`", False), ("\\<!--derived-->", False),
])
def test_only_exact_live_markers_exempt(spelling, live, casedir):
    r = _scan_doc(casedir, "spell_%d" % abs(hash(spelling)), _H + "The ratio was 0.104615 here. %s\n" % spelling)
    assert (cc._verdict(r) == "PASS") is live, (spelling, r["warnings"])
    if not live:
        assert r["warnings"], "a dead marker spelling must print a WARNING: %r" % spelling


def test_cap_is_eight_per_marker(casedir):
    eight = ", ".join("0.1000%02d" % i for i in range(1, 9))
    assert cc._verdict(_scan_doc(casedir, "cap8", _H + "V %s <!--derived-->\n" % eight)) == "PASS"
    nine = eight + ", 0.100009"
    r = _scan_doc(casedir, "cap9", _H + "V %s <!--derived-->\n" % nine)
    assert _flagged(r) == {0.100009}


def test_marker_in_one_parser_only_is_dead(casedir):
    """GFM splits `a | <!--derived--> | b` into cells before inline parsing; CommonMark reads one code span across
    them. Where the two parsers disagree the marker is dead (fail closed)."""
    body = _H + "x | y | z\n---|---|---\n`a | <!--derived--> | b` 0.104615 | 1 | 2\n"
    r = _scan_doc(casedir, "parser_disagree", body)
    assert 0.104615 in _flagged(r)


def test_chance_is_per_claim_and_printed(casedir):
    art = {"precise": [0.123456, 0.234567], "sweep": [round(i / 1000.0 + 0.0002, 4) for i in range(1000)]}
    body = _H + "A 0.123456 here.\n\nB 0.234567 here.\n\nHeadline 0.153 here.\n"
    r = _scan_doc(casedir, "per_claim_chance", body, artifact=art)
    assert [round(v, 6) for _l, v, _ch, _c in r["too_broad"]] == [0.153]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cc.check(cc._write_case(casedir, {"name": "per_claim_chance", "doc": body, "artifact": art}))
    out = buf.getvalue()
    assert "chance match    : p50" in out and "limit %.0f%% per claim" % (100 * cc.CHANCE_MAX) in out
    assert cc.TOO_BROAD_MSG in out


def test_chance_is_deterministic():
    c = cc.Claim(0, 5, 0, 0.153, 3, 1e-3, (), "0.153", "raw")
    pool = [round(i / 1000.0 + 0.0002, 4) for i in range(1000)]
    assert cc._chance(c, pool, None, "precision") == cc._chance(c, pool, None, "precision")


def test_nothing_is_removed_from_checking(casedir):
    """Numbers in comments, code spans, fences, attributes and frontmatter are all checked."""
    body = ("---\nstatus: live\nscore: 0.1525\n---\n\n# T\n\nArtifact: `%(art)s`\n\n<!-- 0.140 -->\n\n"
            "`1.23456`\n\n```\n-0.1525\n```\n\n<a title=\"0.153\">x</a>\n")
    r = _scan_doc(casedir, "everywhere", body)
    assert _flagged(r) >= {0.1525, 0.14, 1.23456, -0.1525, 0.153}


def test_requirements_dev_pins_markdown_it():
    txt = open(os.path.join(ROOT, "requirements-dev.txt"), encoding="utf-8").read()
    assert re.search(r"^markdown-it-py>=4\.2,<5\s*$", txt, re.M)


def test_author_facing_messages_agree():
    """pre-commit, check(), finding_lint and both SKILL.md copies state the SAME marker rule."""
    key = ("SAME physical line", "SAME cell", "at most 8", "exempts nothing")
    hook = open(os.path.join(ROOT, "tools/githooks/pre-commit"), encoding="utf-8").read()
    lint = open(os.path.join(ROOT, "tools/finding_lint.py"), encoding="utf-8").read()
    assert all(k in cc.MARKER_RULE for k in key)
    assert cc.MARKER_RULE in cc.FIX_HINT
    assert "claim_check.MARKER_RULE" in lint and "claim_check.TOO_BROAD_MSG" in lint
    assert all(k in hook for k in key)
    assert "cite a narrower artifact or state more decimals" in hook
    for sk in (".claude/skills/neural-simulator/SKILL.md", ".hermes/skills/neural-simulator/SKILL.md"):
        s = open(os.path.join(ROOT, sk), encoding="utf-8").read()
        assert all(k in s for k in key), sk
        assert "cite a narrower artifact or state more decimals" in s, sk
        assert "alone on a line or for block scope" not in s, sk


def test_unreadable_without_markdown_it(monkeypatch, casedir):
    monkeypatch.setattr(cc, "MarkdownIt", None)
    r = _scan_doc(casedir, "no_mdit", _H + "The accuracy was 0.170 here.\n")
    assert r["unreadable"] and cc._verdict(r) == "FAIL"


def test_invalid_utf8_is_unreadable(casedir):
    p = os.path.join(casedir, "bad.md")
    with open(p, "wb") as fh:
        fh.write(b"# T\n\nThe accuracy was 0.170 \xff here.\n")
    r = cc._scan(p)
    assert r["unreadable"] and cc._verdict(r) == "FAIL"


def test_source_files_carry_no_invisible_or_bidi_characters():
    for rel in ("tools/claim_check.py", "tools/claim_check_cases.py"):
        s = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        bad = [hex(ord(ch)) for ch in s if cc._invisible(ch) or cc._BIDI_RE.match(ch)]
        assert not bad, (rel, bad[:5])


# ---- 4. the four real odd-fence docs: like main or stricter -------------------------------------------------------
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
    with contextlib.redirect_stdout(io.StringIO()):
        rc_main = main_module.check(full, verbose=False)
    rc_new = cc.check(full, verbose=False)
    assert rc_new >= rc_main, "%s: main=%s r8=%s -- round 8 must never be LOOSER than main" % (rel, rc_main, rc_new)
    # and every number main checked on a line OUTSIDE a fence is still checked (fences are checked too, now)
    r = cc._scan(full)
    assert r["checked"] + r["suppressed"]["inline"] >= 1


# ---- 5. the CCT registry gate ------------------------------------------------------------------------------------
def test_cct_gate_passes_a_broken_instrument_through_verbatim():
    from tools.gates import claim_check_selftest as gate
    assert gate.selftest() == []
    msg = "SELFTEST BROKEN: case x expected FAIL, got PASS (y)"
    got = gate.check([], _cc=types.SimpleNamespace(selftest=lambda: [msg]))
    assert got == [gate._LABEL + msg]
    assert gate.check([]) == []                      # the real instrument is healthy


# ---- 6. seeded differential fuzz --------------------------------------------------------------------------------
# Every split below renders as NOTHING to a reader in its context, so the reader genuinely sees 0.1525 (or, for a
# hidden element, the digits around it): a split that renders visibly (a lone `**`, markup inside code) is not a
# hidden number and is not generated. Invisible characters split numbers in EVERY context, code included.
INVISIBLE = [chr(0x200B), chr(0xAD), chr(0x34F), chr(0xFE0F), chr(0x2060), chr(0xFEFF), chr(0x3164)]
PROSE_SPLITS = INVISIBLE + ["<!---->", "<!-- x -->", "<wbr>", "<i></i>", "<b></b>", "<sup></sup>", "<a></a>",
                            "<span hidden>9</span>", "&#8203;"]         # (a multi-line comment: a registry case)
EMPH = {"*", "**", "_", "~~"}
WRAPS = [("", "prose"), ("*", "prose"), ("**", "prose"), ("_", "prose"), ("`", "code"), ("~~", "prose"),
         ("<b>", "prose"), ("<span class=x>", "prose"), ("(", "prose"), ("[", "prose"), ("ms", "prose"),
         (".", "prose"), ("=", "prose"), ("&nbsp;", "prose"), (chr(0xA0), "prose")]
_CLOSE = {"<b>": "</b>", "<span class=x>": "</span>", "(": ")", "[": "](http://x)"}
PREFIX = ["", "gain*", "**acc**", "&Delta;", "x\\*", chr(0x394), "The value ", "| ", "> ", "- ", "1. ", "acc: ",
          "FULL<FROZEN by "]
MARK = "<!--derived-->"


def _wrap(w, n):
    if w in ("ms", "."):
        return n + w
    if w in ("=", "&nbsp;", chr(0xA0)):
        return w + n
    return w + n + _CLOSE.get(w, w)


def _wrong_number(rng, ctx, wrap=""):
    """A wrong number, possibly split by something that renders as nothing in `ctx` ('prose' or 'code')."""
    digits = "1525"
    choice = rng.randint(0, 2)
    if choice == 0:
        return "0." + digits
    if choice == 1 and ctx == "prose" and wrap not in EMPH:        # paired emphasis inside the digits
        k = rng.randint(0, 2)
        return "0." + digits[:k] + "**" + digits[k:k + 2] + "**" + digits[k + 2:]
    pool = PROSE_SPLITS if ctx == "prose" else INVISIBLE
    s = rng.choice(pool)
    k = rng.randint(0, 4)
    return "0." + digits[:k] + s + digits[k:] if k else "0" + s + "." + digits


def _number_in(rng, ctx="prose"):
    w, wctx = rng.choice(WRAPS)
    ctx = "code" if (ctx == "code" or wctx == "code") else "prose"
    return _wrap(w, _wrong_number(rng, ctx, w))


def _ctx_wrong(rng):
    n = _number_in(rng)
    p = rng.choice(PREFIX)
    kind = rng.randint(0, 16)
    if kind == 0:
        return "%s%s here.\n%s\n" % (p, n, MARK if rng.random() < .5 else "")
    if kind == 1:
        lead = rng.choice(["| ", "", "> | ", "- | "])
        end = " |" if lead.endswith("| ") else ""
        return "%sseed | acc | note%s\n%s---|---|---%s\n%s42 | %s | 0.104615 %s%s\n" % (
            lead, end, lead.replace("seed", ""), end, lead, n, MARK, end)
    if kind == 2:
        br = rng.choice(["<br>", "<br/>", "<br class=x>", "</p><p>", "</td><td>", "</li><li>"])
        return "ratio 0.104615 %s%saccuracy %s\n" % (MARK, br, n)
    if kind == 3:
        return "| a | b | c |\n|---|---|---|\n| 42 | %s <!-- | --> 0.104615 %s |\n" % (n, MARK)
    if kind == 4:
        return "vals " + " ".join("0.10000%d" % i for i in range(1, 9)) + " %s %s\n" % (n, MARK)
    if kind == 5:
        return rng.choice(["## Derived\n", MARK + "\n", "Derived\n=======\n"]) + "%s%s here.\n" % (p, n)
    if kind == 6:
        return "%s%s here. <!--/derived-->\n" % (p, n)
    if kind == 7:
        return "```\n%s %s\n```\n" % (_wrong_number(rng, "code"), rng.choice(["", MARK]))
    if kind == 8:
        return "### Result %s\n" % n
    if kind == 9:
        return "[see %s](http://example.org) and more<br>0.104615 %s\n" % (n, MARK)
    if kind == 10:
        return "<table><tr><td>%s</td><td>0.104615 %s</td></tr></table>\n" % (_wrong_number(rng, "prose", "*"), MARK)
    if kind == 11:
        return "<details><summary>more</summary>\n\n%s\n\n</details>\n" % n
    if kind == 12:
        return "Use a `<!--derived` marker.\n\n%s%s here.\n\nThen `-->`.\n" % (p, n)
    if kind == 13:
        return "%s%s here. %s\n" % (p, n, rng.choice(["<!--derived-from x-->", "<!-- derived -->", "`%s`" % MARK,
                                                    "\\" + MARK, "<!--Derived-->"]))
    if kind == 14:
        return "<!-- %s -->\n" % rng.choice(["0.1525", "-0.1625", "1.23456"])
    if kind == 15:
        return "%s%s here.\n" % (p, n)
    return "| a |\n|---|\n| %s |\n| %s |\n" % (n, MARK)


def _ctx_ok(rng):
    kind = rng.randint(0, 7)
    good = rng.choice(["0.170", "0.1625", "0.17000", "0.163", "0.162"])
    w = lambda n: _wrap(rng.choice(WRAPS[:9])[0], n)    # noqa: E731
    if kind == 0:
        return "%s%s here.\n" % (rng.choice(PREFIX[6:12]), w(good))
    if kind == 1:
        return "| metric | value |\n|---|---|\n| ratio | 0.104615 %s |\n| acc | %s |\n" % (MARK, w(good))
    if kind == 2:
        return "ratio 0.104615 %s<br>accuracy %s\n" % (MARK, w(good))
    if kind == 3:
        return "The gap 0.0075 here. <!--derived: 0.170 - 0.1625-->\n"
    if kind == 4:
        return "See arXiv:2403.12345 <!--derived: arXiv id--> -- acc %s.\n" % w(good)
    lead = rng.choice(["", "> ", "- "])
    if rng.random() < .5:
        return lead + "| s | a | r |\n" + lead + "|---|---|---|\n" + lead + "| 42 | %s | 0.104615 %s |\n" % (good, MARK)
    return lead + "s | a | r\n" + lead + "---|---|---\n" + lead + "42 | %s | 0.104615 %s\n" % (good, MARK)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fuzz_one_wrong_number_in_any_markup_context_fails(seed):
    rng = random.Random(seed)
    opened = []
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fuzz_claim_check_") as d:
        for i in range(300):
            body = _ctx_wrong(rng)
            p = cc._write_case(d, dict(name="w%d" % i, doc="# F\n\nArtifact: `%(art)s`\n\n" + body))
            if cc._verdict(cc._scan(p)) != "FAIL":
                opened.append(body)
    assert not opened, "a wrong number passed in: %r" % opened[:5]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fuzz_correct_or_marked_numbers_pass(seed):
    rng = random.Random(1000 + seed)
    fps = []
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fuzz_claim_check_") as d:
        for i in range(300):
            body = _ctx_ok(rng)
            p = cc._write_case(d, dict(name="o%d" % i, doc="# F\n\nArtifact: `%(art)s`\n\n" + body))
            r = cc._scan(p)
            if cc._verdict(r) != "PASS":
                fps.append((body, sorted(_flagged(r))))
    assert not fps, "a clean doc failed: %r" % fps[:5]
