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

Exit 1 if a measurement-shaped number or a verdict word is unsupported by the cited artifacts.

CALIBRATION -- deliberately narrow, because a checker that cries wolf gets ignored (this project's own lesson,
learned twice today). It checks ONLY:
  * numbers with >= 3 decimal places, which are measurements rather than prose ("3 seeds", "97%" are ignored);
  * verdict words that contradict a cited artifact's own verdict field.
Derived values (ratios, differences, percentages) are legitimately absent from artifacts, so a `<!--derived-->`
marker exempts them -- with the SCOPE rules below, which are the part of this file that has been exploited.

DERIVED-MARKER SCOPE (rebuilt 2026-09-25 on a CommonMark parser; markdown-it-py, `table` rule enabled).
Every block boundary below comes from the parser's token line maps, never from a regex over lines, so a `#`
inside a code fence can never be a heading and fences pair the CommonMark way (same character, closing run at
least as long as the opening one; an unclosed fence runs to the end of the document, exactly as it renders).
  * INLINE -- a `<!--derived-->` that is not alone on its line exempts that line only.
  * STANDALONE -- a `<!--derived-->` alone on its line (an `html_block` token holding only the marker) exempts
    the NEXT block-level sibling in the same container, and only when that sibling is a paragraph, a whole list
    (every item, nested content included) or a table. If the sibling is a paragraph whose own next sibling is a
    list or a table (nothing between them: no heading, no other marker), the scope is that intro paragraph plus
    that list/table. Any other next sibling (a heading, a fence, a blockquote, another marker) gets no scope.
    A fenced block is never a sibling scope on its own, because an unclosed fence runs to the end of the file.
  * RANGE -- a standalone `<!--derived-->` followed later by `<!--/derived-->` exempts exactly the lines between.
    The close marker may share a line: text before it is inside the range, text after it is checked. A close
    pairs with the most recent unpaired standalone opener; an unpaired close does nothing.
  * SECTION -- a heading whose title starts with "Derived" (any level) exempts its section: up to the next
    heading of the same or a higher level (`#` is higher than `##`), or the end of the document.
  * Markers inside fenced/indented code or inline code spans are text, not markers (the parser says so).
Two deliberate departures from pure render semantics, both in the direction of checking MORE, because the
author and the reviewer read the SOURCE:
  * a lazy-continuation line (CommonMark "paragraph continuation text" that sits outside its list item's or
    blockquote's indentation, e.g. an unindented sentence right after a bullet) and a table row with no
    unescaped `|` (GFM renders a pipe-less line right after a table as a one-cell row) are never inside a
    sibling scope. In the source they read as prose after the list/table, which is how the "no blank line
    before a wrong number" exploit hid a number.
  * an UNCLOSED fence ends a Derived section at the fence's opening line. Rendered, the rest of the document is
    one code block and every later heading is swallowed; read as source, those headings end the section. The
    stricter reading wins.
Numbers inside fenced code: main's behaviour is kept -- main had no fence handling, so a number in a fence was
checked as ordinary prose unless a marker scope covered it. That is still the rule; the only change is that
the parser, not a line regex, decides where fences are.
The report ALWAYS prints how many numbers each mechanism exempted (section / range / block / inline / synthesis).

HISTORY, in one paragraph (details in research/FAILURE_LOG.md, 2026-09-25). Main's standalone marker scoped to
the next `## ` heading; a scorer hid three whole sections that way (0 of 336 artifact values checked, two wrong
numbers passed). Three rounds of line-scanner patches each opened new holes: `###` headings did not close the
scope; a table followed by a wrong number without a blank line; a close marker sharing a line; a `# derived`
code comment inside a fence read as a heading (round 2); a list protected only its first bullet (round 2); a
fence ended an open list/paragraph (round 2); a boolean fence toggle desynced on mismatched fences (round 3);
list-to-table absorption went one way only (round 3). `SELFTEST_CASES` holds one case per hole, each recorded
with the rounds it fails on; `tests/test_claim_check_scope.py` re-runs every case against every earlier round.

LOW COVERAGE (defense in depth, independent of scope). A non-synthesis doc with >= LOW_COVERAGE_MIN_TOTAL
numeric claims that checked fewer than MIN_CHECK_FRACTION of them fails, whatever mechanism exempted them.
"""
from __future__ import annotations

import functools
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
DERIVED_CLOSE = "<!--/derived-->"
SYNTH_RE = re.compile(r"^claim_check:\s*synthesis\s*$", re.M)

# LOW COVERAGE. A non-synthesis doc with at least LOW_COVERAGE_MIN_TOTAL numeric claims that checked fewer than
# MIN_CHECK_FRACTION of them FAILS. CALIBRATION (stated, not guessed): the round-1 retro-scan of the 353
# research/findings/*.md added 2026-09-01..2026-09-25 found 71 legitimately all-derived short notes with up to 39
# claims each, so the floor sits at 40, above that maximum and below the incident's scale (~43-53 claims, 336
# artifact values). Re-measured under THIS checker on the same population: see
# research/coordination/claimcheck_parser_retro_2026-09-25.tsv (header). An empirical ceiling over one dated
# population, not a law: recalibrate if a later scan finds a legitimate all-derived doc above it.
MIN_CHECK_FRACTION = 0.05
LOW_COVERAGE_MIN_TOTAL = 40

_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
_CLOSING_FENCE = re.compile(r"^(`{3,}|~{3,})[ \t]*$")
_DERIVED_TITLE = re.compile(r"^[\s*_`]*derived\b", re.I)
_LISTS = ("bullet_list", "ordered_list")


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


def load_artifacts(paths, root=None):
    root = root or ROOT
    nums, verdicts, loaded, missing = set(), [], [], []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(root, p)
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


# ---------------------------------------------------------------------------------------------------------------
# The parser. Three small hooks on markdown-it-py, each recording something the stock parser computes and drops:
#   * paragraph: which of its lines are LAZY continuation lines (state.sCount < state.blkIndent -- exactly
#     markdown-it's own test for "this line continues the paragraph from outside its container's indentation";
#     blockquote lazy lines carry sCount = -1, list lazy lines sit left of the item's content column).
#   * fence: whether it found its closing fence (an unclosed one runs to the end of its container).
#   * inline: a `derived_marker` token wherever a marker starts at an inline position. Inline code spans are
#     consumed whole by the backtick rule, so a marker inside one never reaches this rule.
# ---------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _parser():
    from markdown_it import MarkdownIt
    from markdown_it.rules_block import fence as _stock_fence
    from markdown_it.rules_block import paragraph as _stock_paragraph

    md = MarkdownIt("commonmark").enable("table")
    alts = {r.name: list(r.alt) for r in md.block.ruler.__rules__}

    def paragraph(state, startLine, endLine, silent=False):
        n0 = len(state.tokens)
        ok = _stock_paragraph(state, startLine, endLine, silent)
        if ok and not silent and len(state.tokens) > n0 and state.tokens[n0].map:
            tok = state.tokens[n0]
            s, e = tok.map
            tok.meta = dict(tok.meta or {}, lazy={ln: state.sCount[ln] for ln in range(s + 1, e)
                                                  if state.sCount[ln] < state.blkIndent})
        return ok

    def fence(state, startLine, endLine, silent=False):
        n0 = len(state.tokens)
        ok = _stock_fence(state, startLine, endLine, silent)
        if ok and not silent and len(state.tokens) > n0 and state.tokens[n0].map:
            tok = state.tokens[n0]
            s, e = tok.map
            closed = False
            if e - s >= 2:
                last = e - 1
                txt = state.src[state.bMarks[last] + state.tShift[last]:state.eMarks[last]]
                m = _CLOSING_FENCE.match(txt)
                closed = bool(m and m.group(1)[0] == tok.markup[0] and len(m.group(1)) >= len(tok.markup)
                              and state.sCount[last] - state.blkIndent < 4)
            tok.meta = dict(tok.meta or {}, unclosed=not closed)
        return ok

    def marker(state, silent):
        for kind, mark in (("close", DERIVED_CLOSE), ("open", DERIVED_MARK)):
            if state.src.startswith(mark, state.pos):
                if not silent:
                    tok = state.push("derived_marker", "", 0)
                    tok.content = mark
                    tok.meta = {"kind": kind, "line_offset": state.src.count("\n", 0, state.pos)}
                state.pos += len(mark)
                return True
        return False

    md.block.ruler.at("paragraph", paragraph, {"alt": alts["paragraph"]})
    md.block.ruler.at("fence", fence, {"alt": alts["fence"]})
    md.inline.ruler.before("html_inline", "derived_marker", marker)
    return md


def _frontmatter_end(lines):
    """Index of the closing `---` of a leading YAML frontmatter block, or -1. The frontmatter is blanked before
    parsing (CommonMark would read `key: value` + `---` as a setext HEADING), but its numbers are still checked."""
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() in ("---", "..."):
                return j
    return -1


def derived_scope(text):
    """Map every line of `text` to the <!--derived--> mechanism that exempts it (or None).

    Returns dict(lines, reason, split, sections, ranges, blocks, inline, unclosed_fences, code_lines) where
    `reason[i]` is None | "section" | "range" | "block" | "inline" and `split[i] = (a, b)` marks a line whose
    range closes mid-line at columns a..b (text before a is inside the range, text after b is not)."""
    from markdown_it.tree import SyntaxTreeNode

    lines = text.split("\n")
    n = len(lines)
    fm_end = _frontmatter_end(lines)
    src_lines = [("" if i <= fm_end else ln) for i, ln in enumerate(lines)]
    tokens = _parser().parse("\n".join(src_lines))

    code_lines, lazy, pipeless, unclosed = set(), set(), set(), []
    headings, closes, inline_lines = [], [], set()
    for i, t in enumerate(tokens):
        if not t.map:
            continue
        s, e = t.map
        if t.type in ("fence", "code_block"):
            code_lines.update(range(s, e))
            if t.type == "fence" and (t.meta or {}).get("unclosed"):
                unclosed.append(s)
        elif t.type == "paragraph_open":
            lazy.update((t.meta or {}).get("lazy", ()))
        elif t.type == "table_open":
            pipeless.update(ln for ln in range(s, e) if not _UNESCAPED_PIPE.search(lines[ln]))
        elif t.type == "heading_open":
            title = tokens[i + 1].content if i + 1 < len(tokens) and tokens[i + 1].type == "inline" else ""
            headings.append((s, int(t.tag[1]), title))
        elif t.type == "html_block":
            body = t.content.strip()
            if body == DERIVED_MARK:
                continue                                  # a standalone opener: resolved from the tree below
            for ln in range(s, e):
                k = lines[ln].find(DERIVED_CLOSE)
                if k >= 0:
                    closes.append((ln, k, k + len(DERIVED_CLOSE)))
                if DERIVED_MARK in lines[ln]:
                    inline_lines.add(ln)
        elif t.type == "inline":
            for c in t.children or ():
                if c.type != "derived_marker":
                    continue
                ln = s + c.meta["line_offset"]
                if c.meta["kind"] == "open":
                    inline_lines.add(ln)
                else:
                    # the FIRST textual close on the line: if an earlier one sat in a code span, splitting there
                    # only shortens the exempt part (checks more), never lengthens it
                    k = lines[ln].find(DERIVED_CLOSE)
                    if k >= 0:
                        closes.append((ln, k, k + len(DERIVED_CLOSE)))

    root = SyntaxTreeNode(tokens)
    openers = [nd for nd in root.walk() if nd.type == "html_block" and nd.map and nd.content.strip() == DERIVED_MARK]

    # RANGES: a close pairs with the most recent unpaired standalone opener before it.
    events = sorted([(nd.map[0], -1, 0, nd) for nd in openers] + [(c[0], c[1], 1, c) for c in closes],
                    key=lambda ev: (ev[0], ev[1]))
    stack, ranges, paired = [], [], set()
    for _ln, _col, kind, obj in events:
        if kind == 0:
            stack.append(obj)
        elif stack:
            opener = stack.pop()
            paired.add(id(opener))
            ranges.append((opener.map[0], obj))
    closes_by_line = {}
    for o_line, (cl, a, b) in ranges:
        closes_by_line.setdefault(cl, (a, b))

    # SIBLING scopes for every unpaired standalone opener.
    excluded = lazy | pipeless

    def _unit(node):
        return {ln for ln in range(*node.map) if ln not in excluded}

    blocks = []
    for nd in openers:
        if id(nd) in paired:
            continue
        sib = nd.next_sibling
        if sib is None or not sib.map or sib.type not in ("paragraph", "table") + _LISTS:
            blocks.append((nd.map[0], sib.type if sib is not None else None, set()))
            continue
        cov = _unit(sib)
        if sib.type == "paragraph":
            nxt = sib.next_sibling
            if nxt is not None and nxt.map and nxt.type in ("table",) + _LISTS:
                cov |= _unit(nxt)
        blocks.append((nd.map[0], sib.type, cov))

    # SECTIONS: a heading titled "Derived..." up to the next heading of the same or higher level; an unclosed
    # fence inside it ends it at the fence (see the module docstring).
    sections = []
    for idx, (hl, lev, title) in enumerate(headings):
        if not _DERIVED_TITLE.match(title):
            continue
        end = n
        for hl2, lev2, _t in headings[idx + 1:]:
            if lev2 <= lev:
                end = hl2
                break
        for u in sorted(unclosed):
            if hl < u < end:
                end = u
                break
        sections.append((hl, end))

    reason = [None] * n
    for s, e in sections:
        for ln in range(s, e):
            reason[ln] = "section"
    for o_line, (cl, _a, _b) in ranges:
        for ln in range(o_line + 1, cl):
            if reason[ln] is None:
                reason[ln] = "range"
    block_lines = set()
    for _o, _kind, cov in blocks:
        block_lines |= cov
    for ln in block_lines:
        if reason[ln] is None:
            reason[ln] = "block"
    for ln in inline_lines:
        if reason[ln] is None:
            reason[ln] = "inline"
    return dict(lines=lines, reason=reason, split=closes_by_line, sections=sections, ranges=ranges,
                blocks=blocks, inline=sorted(inline_lines), unclosed_fences=sorted(unclosed),
                code_lines=code_lines, block_lines=block_lines)


def _scan(doc_path, tol=None, root=None):
    """Pure computation, no printing -- shared by the CLI (`check`), `tools/finding_lint.py` and the registry
    gate `tools/gates/claim_check_scope.py`, so all three see the exact same verdict.

    tol=None => RELATIVE tolerance. An absolute 5e-4 let a fabricated 0.9999 match a stored 1.0, so the
    checker's own negative control failed on first run: with ~1000 artifact values, near-misses are common and an
    absolute window is far too loose. Relative tolerance scales with the claim.
    """
    text = open(doc_path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n").replace("\r", "\n")
    # A SYNTHESIS doc quotes other experiments throughout its prose; line-by-line marking degenerates into
    # decorating every paragraph, which is how a check stops being read. `claim_check: synthesis` in frontmatter
    # suppresses the per-line rule -- but NOT the citation requirement: it must still cite artifacts that exist,
    # so the escape cannot be used to publish an uncited claim. Chosen deliberately over --no-verify, which would
    # bypass every gate silently and leave no record of which document was exempted or why.
    synthesis = text.startswith("---") and bool(SYNTH_RE.search(text.split("\n---", 1)[0]))
    cited = sorted(set(PATH_RE.findall(text)))
    nums, verdicts, loaded, missing = load_artifacts(cited, root)
    sc = derived_scope(text)

    unsupported, checked = [], 0
    suppressed = {"section": 0, "range": 0, "block": 0, "inline": 0, "synthesis": 0}

    def _account(seg, lineno, why):
        nonlocal checked
        if why is None and synthesis:
            why = "synthesis"
        if why is not None:
            suppressed[why] += len(NUM_RE.findall(seg))
            return
        for m in NUM_RE.finditer(seg):
            val = float(m.group(1))
            checked += 1
            eps = tol if tol is not None else max(5e-6, 1e-4 * abs(val))
            if not any(abs(val - a) <= eps for a in nums):
                unsupported.append((lineno, val, seg.strip()[:88]))

    inline_set = set(sc["inline"])
    for i, ln in enumerate(sc["lines"]):
        why = sc["reason"][i]
        if i in sc["split"]:
            a, b = sc["split"][i]
            before, after = ln[:a], ln[b:]
            _account(before, i + 1, "section" if why == "section" else "range")
            if why == "section":
                after_why = "section"
            elif i in sc["block_lines"]:
                after_why = "block"
            elif i in inline_set and DERIVED_MARK in after:
                after_why = "inline"
            else:
                after_why = None
            _account(after, i + 1, after_why)
        else:
            _account(ln, i + 1, why)

    if synthesis and not cited:
        unsupported.append((0, 0.0, "synthesis doc cites NO artifact — the escape still requires citations"))

    total_numeric = checked + sum(suppressed.values())
    low_coverage = (not synthesis and total_numeric >= LOW_COVERAGE_MIN_TOTAL
                    and (checked / total_numeric) < MIN_CHECK_FRACTION)
    return dict(cited=cited, nums=nums, loaded=loaded, missing=missing, checked=checked,
                suppressed=suppressed, total_numeric=total_numeric, unsupported=unsupported,
                synthesis=synthesis, low_coverage=low_coverage, scope=dict(
                    sections=sc["sections"], ranges=[(o, cl) for o, (cl, _a, _b) in sc["ranges"]],
                    blocks=[(o, kind, len(cov)) for o, kind, cov in sc["blocks"]],
                    inline=sc["inline"], unclosed_fences=sc["unclosed_fences"]))


def check(doc_path, tol=None, verbose=True, root=None):
    r = _scan(doc_path, tol, root)
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
        # ALWAYS printed (2026-09-25): the incident this closes reported "0 checked" with nothing to say WHY.
        marked = suppressed["section"] + suppressed["range"] + suppressed["block"] + suppressed["inline"]
        print("  suppressed      : %d total by <!--derived--> (section=%d range=%d block=%d inline=%d), %d by "
              "synthesis of %d numeric claim(s) found"
              % (marked, suppressed["section"], suppressed["range"], suppressed["block"], suppressed["inline"],
                 suppressed["synthesis"], total_numeric))
        sc = r["scope"]
        if sc["sections"] or sc["ranges"] or sc["blocks"]:
            print("  derived scopes  : sections %s | ranges %s | standalone markers %s"
                  % (", ".join("L%d-%d" % (s + 1, e) for s, e in sc["sections"]) or "-",
                     ", ".join("L%d-%d" % (o + 1, c + 1) for o, c in sc["ranges"]) or "-",
                     ", ".join("L%d->%s(%d lines)" % (o + 1, k or "nothing", m) for o, k, m in sc["blocks"])
                     or "-"))
        for lineno, val, ctx in unsupported[:12]:
            print("      ⛔ line %-4d %-14g not in any cited artifact | %s" % (lineno, val, ctx))
        if len(unsupported) > 12:
            print("      ... and %d more" % (len(unsupported) - 12))
        if low_coverage:
            print("      ⛔ LOW COVERAGE: only %d/%d (%.0f%%) numeric claim(s) were actually checked -- the "
                  "rest were suppressed by markers. A derived scope should cover the specific numbers it "
                  "derives, not surrounding prose." % (checked, total_numeric, 100.0 * checked / total_numeric))

    fail = bool(missing) or bool(unsupported) or low_coverage
    if verbose:
        if not cited:
            print("  ⚠️  NO ARTIFACT CITED — a findings doc with no artifact path cannot be checked at all.")
        print("  => %s" % ("⛔ UNSUPPORTED CLAIMS (or missing artifacts) — fix, cite, or mark <!--derived-->"
                           if fail else "✔ every measurement traces to a cited artifact"))
    return 0 if not fail else 1


# ---------------------------------------------------------------------------------------------------------------
# SELFTEST REGISTRY. One case per hole in the history. `wrong_on` names the earlier checkers that give the
# WRONG verdict on the case: "main" = tools/claim_check.py at 7e2edc08e (main before 2026-09-25), "r1" =
# d4959ecb0, "r2" = 6abb28469, "r3" = 662e167e8 (branches research/claimcheck-block-scope[-r3]). Every case is
# wrong on at least one of them and right here; tests/test_claim_check_scope.py re-derives `wrong_on` by running
# each case through each earlier version, so the record cannot drift from the code. `%(art)s` is a cited artifact
# holding accuracy=0.17 and baseline=0.1625; 0.1525 / 0.140 / 1.23456 are WRONG numbers, 0.104615 / 0.207531 /
# 0.311079 are legitimately derived ones.
# ---------------------------------------------------------------------------------------------------------------
_HDR = "# Some finding\n\nArtifact: `%(art)s`\n\n"
SELFTEST_CASES = [
    dict(name="incident_marker_after_headings", expect="FAIL", wrong_on=("main",),
         why="main's hole: a marker right after `## ` headings exempted the whole sections after it",
         doc=_HDR + "## Section A\n<!--derived-->\nRunner: some_runner.py, no numbers in this sentence.\n\n"
                    "The accuracy was 0.1525 here.\n\n## Section B\n<!--derived-->\n"
                    "Runner: some_runner.py, no numbers in this sentence.\n\nThe baseline was 0.140 here.\n"),
    dict(name="h3_heading_ends_scope", expect="FAIL", wrong_on=("main", "r1"),
         why="a `### ` heading right after a marker must end the scope",
         doc=_HDR + "<!--derived-->\n### A subheading, not a level-2 one\nThe accuracy was 0.1525 here.\n"),
    dict(name="table_then_wrong_number_no_blank", expect="FAIL", wrong_on=("main", "r1"),
         why="a pipe-less line straight after a derived table (GFM: a one-cell row) is checked",
         doc=_HDR + "<!--derived-->\n| metric | value |\n|---|---|\n| ratio | 0.104615 |\n"
                    "The accuracy was 0.1525 here.\n"),
    dict(name="list_lazy_line_wrong_number", expect="FAIL", wrong_on=("main", "r1", "r2", "r3"),
         why="an unindented line straight after a derived list (a lazy continuation) is checked",
         doc=_HDR + "<!--derived-->\n- ratio 0.104615\nThe accuracy was 0.1525 here.\n"),
    dict(name="close_marker_midline_trailing_checked", expect="FAIL", wrong_on=("main", "r1"),
         why="text after a close marker on the same line is checked",
         doc=_HDR + "<!--derived-->\nThe ratio is 0.104615 here. <!--/derived--> The real accuracy is 0.1525 here.\n"),
    dict(name="close_marker_midline_leading_covered", expect="PASS", wrong_on=("r1", "r2", "r3"),
         why="text before a mid-line close marker is inside the range, even after a blank line",
         doc=_HDR + "<!--derived-->\nThe ratio is 0.104615 here.\n\n"
                    "The second ratio is 0.207531. <!--/derived--> The accuracy is 0.170000.\n"),
    dict(name="range_spans_paragraphs", expect="PASS", wrong_on=("r1", "r2", "r3"),
         why="an explicit range covers exactly the lines between its markers",
         doc=_HDR + "<!--derived-->\nThe ratio is 0.104615.\n\nThe second ratio is 0.207531.\n<!--/derived-->\n"
                    "The accuracy is 0.170000.\n"),
    dict(name="hash_derived_comment_in_fence", expect="FAIL", wrong_on=("r2",),
         why="a `# derived` comment inside a fence is code, not a Derived heading",
         doc=_HDR + "```python\n# derived thresholds below\nvalue = 1.23456\n```\nThe accuracy was 0.1525 here.\n"),
    dict(name="marker_inside_fence_is_text", expect="FAIL", wrong_on=("main", "r1", "r2"),
         why="a marker line inside a fence is code, not a marker",
         doc=_HDR + "```\n<!--derived-->\nvalue = 0.1525\n```\n"),
    dict(name="list_second_and_third_bullets", expect="PASS", wrong_on=("r2",),
         why="a standalone marker covers every item of the list after it",
         doc=_HDR + "<!--derived-->\n- first bullet, ratio 0.104615\n- second bullet, ratio 0.207531\n"
                    "- third bullet, ratio 0.311079\n\nThe baseline was 0.162500 here.\n"),
    dict(name="intro_paragraph_then_list", expect="PASS", wrong_on=("r2",),
         why="marker, intro sentence, then the list it introduces: both are covered",
         doc=_HDR + "<!--derived-->\nAll values below are rounded re-quotes from the per-seed breakdown.\n"
                    "- Lesion -- mean 0.104615.\n- Scramble -- mean 0.207531.\n"),
    dict(name="intro_paragraph_blank_then_list", expect="PASS", wrong_on=("r1", "r2", "r3"),
         why="the intro paragraph's next sibling is the list even across a blank line",
         doc=_HDR + "<!--derived-->\nAll values below are rounded re-quotes.\n\n- Lesion -- mean 0.104615.\n"
                    "- Scramble -- mean 0.207531.\n"),
    dict(name="intro_paragraph_then_table", expect="PASS", wrong_on=("r2",),
         why="marker, intro sentence, then the table it introduces: both are covered",
         doc=_HDR + "<!--derived-->\nRatios from the artifact above:\n| metric | value |\n|---|---|\n"
                    "| ratio | 0.104615 |\n"),
    dict(name="list_then_blank_then_wrong_number", expect="FAIL", wrong_on=("main",),
         why="a list's scope ends with the list; a later paragraph is checked",
         doc=_HDR + "<!--derived-->\n- bullet one: 0.104615\n- bullet two: 0.207531\n\n"
                    "The real accuracy was 0.1525 here.\n"),
    dict(name="fence_nested_in_derived_list", expect="PASS", wrong_on=("r2",),
         why="a fence nested in a list item is list content; it does not end the list's scope",
         doc=_HDR + "<!--derived-->\n- first item, computed by:\n  ```bash\n  python -m foo --seed 42\n  ```\n"
                    "- second item, ratio 0.104615\n"),
    dict(name="range_across_fence", expect="PASS", wrong_on=("r2",),
         why="an explicit range around an intro, a code snippet and the derived number covers all of it",
         doc=_HDR + "<!--derived-->\nComputed by the runner shown below:\n```bash\npython -m foo --seed 42\n```\n"
                    "The resulting ratio, 0.104615, is a re-quote.\n<!--/derived-->\n"),
    dict(name="list_with_nested_table_then_bullet", expect="PASS", wrong_on=("r2", "r3"),
         why="a table nested in a list item does not end the list's scope",
         doc=_HDR + "<!--derived-->\n- per-seed table:\n  | seed | ratio |\n  |---|---|\n  | 42 | 0.104615 |\n"
                    "- mean ratio 0.207531\n"),
    dict(name="list_then_unindented_table_wrong_number", expect="FAIL", wrong_on=("main", "r1", "r3"),
         why="an unindented table straight after a derived list is lazy text, not list content",
         doc=_HDR + "<!--derived-->\n- ratio 0.104615\n| metric | value |\n|---|---|\n| accuracy | 0.1525 |\n"),
    dict(name="mismatched_fence_does_not_swallow_heading", expect="FAIL", wrong_on=("r3",),
         why="`~~~` is not closed by a backtick fence; the heading after the real close ends the section",
         doc=_HDR + "## Derived\nratio 0.104615\n~~~\n```\n~~~\n## Results\nThe accuracy was 0.1525 here.\n"),
    dict(name="longer_fence_keeps_later_marker_live", expect="PASS", wrong_on=("r3",),
         why="a ```` fence is not closed by ```; the marker after its real close is live",
         doc=_HDR + "````\n```\n````\n<!--derived-->\nThe ratio is 0.104615.\n"),
    dict(name="unclosed_fence_ends_derived_section", expect="FAIL", wrong_on=("r3",),
         why="an unclosed fence ends a Derived section; the heading and number inside it are checked",
         doc=_HDR + "## Derived\nratio 0.104615\n```\ncode\n## Results\nThe accuracy was 0.1525 here.\n"),
    dict(name="derived_h3_section_keeps_h4", expect="PASS", wrong_on=("main", "r1", "r2", "r3"),
         why="a `### Derived` section continues through its `####` subsections",
         doc=_HDR + "### Derived\n#### Per seed\n- ratio 0.104615\n\n### Next\nThe accuracy is 0.170000.\n"),
    dict(name="derived_section_ends_at_h1", expect="FAIL", wrong_on=("main", "r1"),
         why="a `## Derived` section ends at the next heading of the same OR HIGHER level",
         doc=_HDR + "## Derived\nratio 0.104615\n# Results\nThe accuracy was 0.1525 here.\n"),
    dict(name="low_coverage_overmarked", expect="FAIL", wrong_on=("main",),
         why="a substantial doc that exempts (almost) every claim fails whatever exempted them",
         doc=_HDR + "\n\n".join("The value was 0.%06d here. <!--derived-->" % (i * 7 + 1)
                                for i in range(LOW_COVERAGE_MIN_TOTAL + 5)) + "\n"),
]


def _write_case(d, case):
    """Write one selftest case (and its artifact) under directory `d`, which must be inside ROOT: PATH_RE drops
    a leading '/', so a real finding's ROOT-relative citation is the only form that resolves the same way."""
    art_abs = os.path.join(d, "art.json")
    if not os.path.exists(art_abs):
        json.dump({"accuracy": 0.17, "baseline": 0.1625}, open(art_abs, "w"))
    art = os.path.relpath(art_abs, ROOT).replace(os.sep, "/")
    path = os.path.join(d, case["name"] + ".md")
    open(path, "w", encoding="utf-8").write(case["doc"] % {"art": art})
    return path


def selftest():
    """Same contract as `tools/gates/*.selftest()`: a list of problems, empty means the check is trustworthy.
    Runs every SELFTEST_CASES entry in both directions (FAIL cases must fail, PASS cases must pass)."""
    import tempfile
    problems = []
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".claim_check_selftest_") as d:
        for case in SELFTEST_CASES:
            rc = check(_write_case(d, case), verbose=False)
            got = "FAIL" if rc else "PASS"
            if got != case["expect"]:
                problems.append("SELFTEST BROKEN: case %s expected %s, got %s (%s)"
                                % (case["name"], case["expect"], got, case["why"]))
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
