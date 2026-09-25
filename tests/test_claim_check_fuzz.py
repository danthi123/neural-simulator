"""Differential fuzz for tools/claim_check.py (round 7): ONE wrong number in a random markup context must FAIL the
doc; correct or properly-marked numbers in random decorations must PASS it. Seeded, so every failure reproduces.

Every one of rounds 1-6 shipped a hole of the shape "markup X next to a number hides it / carries an exemption
across it". The per-hole cases live in claim_check.SELFTEST_CASES; this file composes the hole vocabulary at
random -- wrappers x digit-splitters x prefixes x containers -- so a new combination cannot slip between the named
cases. Round 7's verification used it (10,000 contexts per direction, 0 fail-open, 0 false positive after fixes)
and it found the `Δ_0.1525_` shape that no named case covered.
"""
from __future__ import annotations

import os
import random
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tools.claim_check as cc                      # noqa: E402

WRAPS = [lambda n: n, lambda n: "*%s*" % n, lambda n: "**%s**" % n, lambda n: "_%s_" % n, lambda n: "`%s`" % n,
         lambda n: "~~%s~~" % n, lambda n: "<b>%s</b>" % n, lambda n: "<span class=x>%s</span>" % n,
         lambda n: "(%s)" % n, lambda n: "[%s](http://x)" % n, lambda n: "%sms" % n, lambda n: "%s." % n,
         lambda n: "=%s" % n, lambda n: "&nbsp;%s" % n, lambda n: "\u00a0%s" % n]
SPLITS = ["", "<!---->", "<!-- x -->", "\u200b", "\u00ad", "\u034f", "\ufe0f", "<wbr>", "<i></i>", "<b></b>", "**",
          "<sup></sup>", "<div hidden>9</div>", "&#8203;"]
PREFIX = ["", "gain*", "**acc**", "&Delta;", "x\\*", "Δ", "The value ", "| ", "> ", "- ", "1. ", "acc: "]
MARK = "<!--derived-->"


def _wrong_number(rng):
    s = rng.choice(SPLITS)
    k = rng.randint(1, 5)
    return "0." + "1525"[:k - 1] + s + "1525"[k - 1:] if s else "0.1525"


def _ctx_wrong(rng):
    n = rng.choice(WRAPS)(_wrong_number(rng))
    p = rng.choice(PREFIX)
    kind = rng.randint(0, 13)
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
        return "```\n%s\n```\n" % n
    if kind == 8:
        return "### Result %s\n" % n
    if kind == 9:
        return "[see %s](http://example.org) and more<br>0.104615 %s\n" % (n, MARK)
    if kind == 10:
        return "<table><tr><td>%s</td><td>0.104615 %s</td></tr></table>\n" % (n, MARK)
    if kind == 11:
        return "<details><summary>more</summary>\n\n%s\n\n</details>\n" % n
    if kind == 12:
        return "Footnote.[^1]\n\n[^1]: measured %s\n" % n
    return "| a |\n|---|\n| %s |\n| %s |\n" % (n, MARK)


def _ctx_ok(rng):
    kind = rng.randint(0, 7)
    good = rng.choice(["0.170", "0.1625", "0.17000", "0.163", "0.162"])
    w = rng.choice(WRAPS[:9])
    if kind == 0:
        return "%s%s here.\n" % (rng.choice(PREFIX[6:]), w(good))
    if kind == 1:
        return "| metric | value |\n|---|---|\n| ratio | 0.104615 %s |\n| acc | %s |\n" % (MARK, w(good))
    if kind == 2:
        return "ratio 0.104615 %s<br>accuracy %s\n" % (MARK, w(good))
    if kind == 3:
        return "The gap 0.0075 here. <!--derived: 0.170 - 0.1625-->\n"
    if kind == 4:
        return "See arXiv:2403.12345 and https://x.org/0.99 -- acc %s.\n" % w(good)
    lead = rng.choice(["", "> ", "- "])
    if rng.random() < .5:
        return lead + "| s | a | r |\n" + lead + "|---|---|---|\n" + lead + "| 42 | %s | 0.104615 %s |\n" % (good, MARK)
    return lead + "s | a | r\n" + lead + "---|---|---\n" + lead + "42 | %s | 0.104615 %s\n" % (good, MARK)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_wrong_number_in_any_markup_context_fails_the_doc(seed):
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
def test_correct_or_marked_numbers_in_any_decoration_pass(seed):
    rng = random.Random(1000 + seed)
    fps = []
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".fuzz_claim_check_") as d:
        for i in range(300):
            body = _ctx_ok(rng)
            p = cc._write_case(d, dict(name="o%d" % i, doc="# F\n\nArtifact: `%(art)s`\n\n" + body))
            r = cc._scan(p)
            if cc._verdict(r) != "PASS":
                fps.append((body, [v for _l, v, _c in r["unsupported"]]))
    assert not fps, "a clean doc failed: %r" % fps[:5]
