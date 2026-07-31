#!/usr/bin/env python3
"""Derive a machine-readable STATUS for every finding, so retrieval cannot surface a void result as live.

WHY (2026-07-31, owner question: "are findings/results/memories organised so you pull the right ones and can
recall without hallucinating?"). Measured answer: NO, and the mechanism is precise.

    1841 findings · 101 whose BODY carries a retraction/supersession marker · 1 registered in docs/RETRACTED.md
    · 0 with machine-readable frontmatter

So a RAG hit carries a path and a score and NOTHING about whether the result still stands. The concrete instance,
same day: a search returned `2026-07-24-P0.3-affect-state-region-6seed-GO.md` -- a filename asserting GO -- whose
actual verdict is QUALIFIED-GO/BOUNDARY and whose artifact reads {"GO": false, "n_seeds_go": 2}. That filename
propagated into the board, and from the board into a claim I made to the owner. The finding itself was honest;
nothing carried its honesty to the point of retrieval.

DERIVED, NOT MAINTAINED. Every hand-maintained index in this repo has gone stale -- docs/RETRACTED.md holds ONE
row against 101 marked findings. So this SCANS the corpus and regenerates; there is nothing to keep updated and
therefore nothing to forget.

    .venv/bin/python tools/finding_status.py                 # report the distribution
    .venv/bin/python tools/finding_status.py --write         # regenerate docs/FINDINGS_STATUS.md
    .venv/bin/python tools/finding_status.py --check <path>  # status of one finding (for retrieval to annotate)
    .venv/bin/python tools/finding_status.py --selftest      # prove the frontmatter parser fails where it must

STATUS is read from the document's own words, in priority order, because the corpus already speaks this
vocabulary consistently -- it just does so in prose no machine reads:

    retracted  > superseded > corrected > qualified > live

`superseded_by:` -- WHAT REPLACED IT (2026-07-31, FAILURE_LOG row: "a declared `status: superseded` names NO
SUCCESSOR"). A status of `superseded` answers "is this still live?" and stops there; the next question a reader
always has -- "then what do I cite instead?" -- was answered by tracing descendants BY HAND, at 4 of 5 markers in
one fanout pass, unaudited. The status field made the deadness machine-readable and left the REPAIR manual. This
adds the second half of the fact, in the same place, so `tools/gates/stale_pointer.py` can quote it at the point
of the stale citation instead of the fixer re-deriving it.

    ---
    status: superseded
    superseded_by: 2026-07-25-the-rerun-that-replaced-it.md
    ---

Accepted forms: one value; a comma list; a bracketed list; a YAML block list (`- item` lines). A value may be a
bare finding name, a repo-relative path, or prose containing one -- it is quoted VERBATIM and separately
RESOLVED, so an author who writes a sentence still gets the sentence shown to the reader.

NOT accepted, stated rather than silently mishandled: the key must sit at column 0 of the frontmatter (nested
YAML is not walked), and the field is only ever CONSULTED on a document that also declares `status: retracted`
or `status: superseded` -- a successor on a live document answers a question nobody is asking.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINDINGS = os.path.join(ROOT, "research", "findings")

# Ordered: first match wins. Patterns are deliberately anchored to how THIS corpus actually writes these words
# (checked against the 101 marked files), not to a generic vocabulary.
RULES = [
    ("retracted", re.compile(
        r"⛔⛔|\bRETRACT(?:ED|ION)?\b|\bWITHDRAWN\b|\bIS VOID\b|\bVOID\b(?!\s*(?:if|_if))|\bINVALID(?:ATES?|ATED)?\b",
        re.I)),
    ("superseded", re.compile(r"\bSUPERSED(?:ED|ES|ING)\b|\bREPLACED BY\b|\bno longer the\b", re.I)),
    ("corrected", re.compile(r"\bCORRECTION\b|\bCORRECTED\b|\bOVERREACH(?:ED)?\b|\bWAS WRONG\b", re.I)),
    ("qualified", re.compile(r"\bQUALIFIED[- ]GO\b|\bBOUNDARY\b|\bNO-GO\b|\bHONEST NEGATIVE\b", re.I)),
]
HEAD_LINES = 40          # status is declared near the top; scanning whole 1000-line files invites false hits


FRONTMATTER_LINES = 15   # frontmatter opens the file; reading deeper starts reading BODY PROSE as declaration

DECL_RE = re.compile(r"^status:\s*([a-z-]+)\s*$", re.M)
SUCC_KEY_RE = re.compile(r"^superseded_by:[ \t]*(.*?)[ \t]*$")
_ITEM_RE = re.compile(r"^[ \t]*-[ \t]*(.+?)[ \t]*$")            # a YAML block-list entry
_MD_TOKEN_RE = re.compile(r"((?:[A-Za-z0-9._\-]+/)*[A-Za-z0-9._\-]+\.md)")
_NAME_RE = re.compile(r"^[12]\d{3}-\d{2}-\d{2}-[A-Za-z0-9._\-]+$")
_REF_LIKE = re.compile(r"\.md\b|^[12]\d{3}-\d{2}-\d{2}-")


def _frontmatter(path):
    """The YAML frontmatter block as TEXT, or None when the file does not open with one.

    It must OPEN the file. `status:` / `superseded_by:` appearing in the BODY is prose ABOUT some other
    document's status -- the exact confusion that measured `status_of`'s keyword hint at ~1-in-6 precision.

    LIMITS, stated because they are not caught: only the first FRONTMATTER_LINES (15) lines are read, so a
    declaration below that is invisible; and a frontmatter block longer than that has no visible closing `---`,
    so its tail is read as frontmatter. Both inherited from the original 15-line read, left byte-compatible.
    """
    try:
        with open(path, errors="ignore") as fh:
            head = "".join(next(fh, "") for _ in range(FRONTMATTER_LINES))
    except Exception:
        return None
    if not head.startswith("---"):
        return None
    return head.split("\n---", 1)[0]


def declared_status(path):
    """The AUTHORITY: `status:` in frontmatter. Returns None when absent."""
    fm = _frontmatter(path)
    if fm is None:
        return None
    m = DECL_RE.search(fm)
    return m.group(1).strip() if m else None


def _clean(v):
    return v.strip().strip(",").strip().strip("'\"").strip()


def _split_inline(val):
    """Split an inline `superseded_by:` value into entries -- WITHOUT shredding prose on its commas.

    A comma splits only when the value is bracketed, or when EVERY part looks like a document reference. So
    `[a.md, b.md]` and `a.md, b.md` are two successors, while `a.md, which reran it at 6 seeds` stays ONE
    verbatim entry (its `a.md` is still found by `resolve_successor`).
    """
    v = val.strip()
    bracketed = v.startswith("[") and v.endswith("]")
    if bracketed:
        v = v[1:-1]
    parts = [p for p in (_clean(x) for x in v.split(",")) if p]
    if bracketed or (len(parts) > 1 and all(_REF_LIKE.search(p) for p in parts)):
        return parts
    whole = _clean(v)
    return [whole] if whole else []


def declared_successors(path):
    """`superseded_by:` from frontmatter -> the entries VERBATIM (never rewritten). [] when absent.

    VERBATIM matters: this text is quoted back to whoever has to repair a stale citation, and a paraphrase of
    the author's answer is exactly the hand-reconstruction this field exists to remove.
    """
    fm = _frontmatter(path)
    if fm is None:
        return []
    lines = fm.split("\n")
    for i, ln in enumerate(lines):
        m = SUCC_KEY_RE.match(ln)
        if not m:
            continue
        inline = m.group(1).strip()
        if inline:
            return _split_inline(inline)
        out = []                                   # bare key -> a YAML block list on the following lines
        for nxt in lines[i + 1:]:
            im = _ITEM_RE.match(nxt)
            if not im:
                break
            v = _clean(im.group(1))
            if v:
                out.append(v)
        return out
    return []


def resolve_successor(entry, root=ROOT):
    """-> (repo_relative_path, exists). `path` is None when the entry names no document at all (pure prose).

    The two failures are kept DISTINGUISHABLE on purpose: "you wrote a sentence, not a pointer" and "you wrote a
    pointer to a file that is not there" need different repairs, and collapsing them to a single None would hand
    the reader back the same guessing this field removes.
    """
    m = _MD_TOKEN_RE.search(entry or "")
    if m:
        cand = m.group(1)
    else:
        bare = (entry or "").strip()
        if not _NAME_RE.match(bare):
            return None, False
        cand = bare + ".md"
    rel = cand if "/" in cand else "research/findings/" + cand
    return rel, os.path.isfile(os.path.join(root, rel))


def status_of(path):
    """Declared status if present; otherwise a KEYWORD HINT, which is NOT authoritative.

    ⚠️ PRECISION MEASURED BEFORE SHIPPING, and it is poor: a 6-file sample of the keyword rule's "retracted"
    label was right about 1 time in 6. A finding that PERFORMS a retraction carries the same vocabulary as one
    that IS retracted, and one that merely DISCUSSES the retraction record carries it too --
    `2026-07-31-gap5-stepC-control-void-and-the-fix.md` is LIVE and is the document that did the voiding.
    Keywords cannot separate "this is void" from "this voids something else".

    So the hint is a BACKFILL SUGGESTION, never a verdict. Status has to be DECLARED. That is why the pre-commit
    gate requires it on NEW findings only: 1841 files cannot be backfilled in one pass, and a big-bang migration
    would not happen -- but every new finding declares, and old ones declare when next touched.
    """
    try:
        with open(path, errors="ignore") as fh:
            head = [next(fh, "") for _ in range(HEAD_LINES)]
    except Exception:
        return "unknown", ""
    decl = declared_status(path)
    if decl:
        return decl, "declared"
    text = "".join(head)
    base = os.path.basename(path)
    for name, rx in RULES:
        m = rx.search(base) or rx.search(text)
        if m:
            for ln in head:
                if rx.search(ln):
                    return name, ln.strip()[:150]
            return name, base
    return "live?", ""


def successor_parts(path, root=ROOT):
    """[(rendered, usable)] -- one entry per declared successor. [] when the field is absent.

    `usable` is the whole point of resolving rather than just quoting: a successor that does not exist, or that
    is ITSELF retracted/superseded, is not somewhere to send a reader. It is returned as a FLAG, not inferred
    from the rendered prose, so a caller composing its own sentence cannot get it wrong by string-sniffing.

    Shared with `tools/gates/stale_pointer.py` so the CLI and the gate cannot disagree about what a document
    declares -- two parsers for one field is the integration-seam class logged 2026-07-31 (the provenance door
    wrote `.prov.json` while its own gate looked for `.cmd.json`).
    """
    parts = []
    for s in declared_successors(path):
        rel, exists = resolve_successor(s, root=root)
        if rel is None:
            parts.append(("%s [UNRESOLVED: names no document]" % s, False))
        elif not exists:
            parts.append(("%s [UNRESOLVED: %s does not exist]" % (s, rel), False))
        else:
            st = declared_status(os.path.join(root, rel))
            dead = st in ("retracted", "superseded")
            parts.append(("%s%s" % (os.path.basename(rel), " [⚠ itself declared %s]" % st if dead else ""),
                          not dead))
    return parts


def describe_successors(path, root=ROOT):
    """Each declared successor, rendered WITH its resolution. [] when the field is absent."""
    return [r for r, _ in successor_parts(path, root=root)]


def successor_report(path, root=ROOT):
    """One human line describing `superseded_by:` on `path` -- including its ABSENCE. Never returns ''."""
    parts = describe_successors(path, root=root)
    if not parts:
        return "no `superseded_by:` — what replaced it must be reconstructed by hand"
    return "superseded_by: " + "; ".join(parts)


def scan():
    rows = []
    for p in sorted(glob.glob(os.path.join(FINDINGS, "*.md"))):
        s, ev = status_of(p)
        rows.append((os.path.relpath(p, ROOT), s, ev, declared_successors(p)))
    return rows


def _st_fixture(d):
    import io
    os.makedirs(os.path.join(d, "research/findings"))
    for name, text in (
            ("2026-01-01-one.md", "---\nstatus: superseded\nsuperseded_by: 2026-01-09-heir.md\n---\nbody\n"),
            ("2026-01-02-bracket.md", "---\nstatus: superseded\n"
                                      "superseded_by: [2026-01-09-heir.md, 2026-01-10-heir2.md]\n---\n"),
            ("2026-01-03-commas.md", "---\nstatus: superseded\n"
                                     "superseded_by: 2026-01-09-heir.md, 2026-01-10-heir2.md\n---\n"),
            ("2026-01-04-prose.md", "---\nstatus: superseded\n"
                                    "superseded_by: 2026-01-09-heir.md, which reran it at 6 seeds\n---\n"),
            ("2026-01-05-block.md", "---\nstatus: superseded\nsuperseded_by:\n"
                                    "  - 2026-01-09-heir.md\n  - 2026-01-10-heir2.md\nother: x\n---\n"),
            ("2026-01-06-absent.md", "---\nstatus: superseded\n---\nbody\n"),
            # The ~1-in-6 prose trap, for the NEW field: a document DISCUSSING the convention, not using it.
            ("2026-01-07-bodyonly.md", "---\nstatus: superseded\n---\n\nsuperseded_by: 2026-01-09-heir.md\n"),
            ("2026-01-08-nofm.md", "superseded_by: 2026-01-09-heir.md\nstatus: retracted\n"),
            ("2026-01-09-heir.md", "---\nstatus: live\n---\n"),
            ("2026-01-10-heir2.md", "---\nstatus: live\n---\n"),
            ("2026-01-11-dangling.md", "---\nstatus: superseded\nsuperseded_by: 2026-99-99-nope.md\n---\n"),
            ("2026-01-12-chain.md", "---\nstatus: superseded\nsuperseded_by: 2026-01-01-one.md\n---\n"),
            ("2026-01-13-nodoc.md", "---\nstatus: superseded\nsuperseded_by: ask Daniel\n---\n")):
        io.open(os.path.join(d, "research/findings", name), "w", encoding="utf-8").write(text)
    return d


def selftest():
    """FAILING DIRECTION FIRST: each case is one a regressed parser would get WRONG, not one it passes trivially.

    The parser is shared with `tools/gates/stale_pointer.py`, so a silent regression here would not break a gate
    loudly -- it would make the gate quietly stop naming successors. That is failure class 3 (a check that
    cannot fail), which is why this exists at all for a script that is not itself a gate.
    """
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as d:
        _st_fixture(d)
        F = lambda n: os.path.join(d, "research/findings", n)                              # noqa: E731
        S = lambda n: declared_successors(F(n))                                            # noqa: E731

        # --- the traps: a whole-file regex or a naive quoter gets each of these wrong -------------------
        if S("2026-01-07-bodyonly.md"):
            bad.append("read `superseded_by:` from the BODY (prose about the convention): %r"
                       % S("2026-01-07-bodyonly.md"))
        if S("2026-01-08-nofm.md"):
            bad.append("read `superseded_by:` from a file with NO frontmatter: %r" % S("2026-01-08-nofm.md"))
        if S("2026-01-04-prose.md") != ["2026-01-09-heir.md, which reran it at 6 seeds"]:
            bad.append("shredded a prose value on its comma: %r" % S("2026-01-04-prose.md"))
        if resolve_successor("2026-99-99-nope.md", root=d) != ("research/findings/2026-99-99-nope.md", False):
            bad.append("a DANGLING successor did not report exists=False: %r"
                       % (resolve_successor("2026-99-99-nope.md", root=d),))
        if resolve_successor("ask Daniel", root=d) != (None, False):
            bad.append("a prose value that names no document did not report path=None: %r"
                       % (resolve_successor("ask Daniel", root=d),))
        if "reconstructed by hand" not in successor_report(F("2026-01-06-absent.md"), root=d):
            bad.append("stayed SILENT about an ABSENT successor: %r"
                       % successor_report(F("2026-01-06-absent.md"), root=d))
        if "⚠ itself declared superseded" not in successor_report(F("2026-01-12-chain.md"), root=d):
            bad.append("pointed the reader at a successor that is ITSELF superseded, unflagged: %r"
                       % successor_report(F("2026-01-12-chain.md"), root=d))
        for n, why in (("2026-01-11-dangling.md", "successor file does not exist"),
                       ("2026-01-13-nodoc.md", "value names no document")):
            if "UNRESOLVED" not in successor_report(F(n), root=d):
                bad.append("did not mark UNRESOLVED (%s): %r" % (why, successor_report(F(n), root=d)))

        # --- the forms that must all parse (a single-line-only parser fails the block list) -------------
        for n, want in (("2026-01-01-one.md", ["2026-01-09-heir.md"]),
                        ("2026-01-02-bracket.md", ["2026-01-09-heir.md", "2026-01-10-heir2.md"]),
                        ("2026-01-03-commas.md", ["2026-01-09-heir.md", "2026-01-10-heir2.md"]),
                        ("2026-01-05-block.md", ["2026-01-09-heir.md", "2026-01-10-heir2.md"])):
            if S(n) != want:
                bad.append("form %s parsed as %r, wanted %r" % (n, S(n), want))
        if resolve_successor("2026-01-09-heir.md", root=d) != ("research/findings/2026-01-09-heir.md", True):
            bad.append("a RESOLVABLE successor did not resolve: %r"
                       % (resolve_successor("2026-01-09-heir.md", root=d),))
        if resolve_successor("research/findings/2026-01-09-heir.md", root=d)[1] is not True:
            bad.append("a repo-relative successor path did not resolve")
        if resolve_successor("2026-01-09-heir", root=d)[1] is not True:
            bad.append("a bare successor name (no .md) did not resolve")

        # --- regression pin on the PRE-EXISTING behaviour this refactor moved into _frontmatter --------
        if declared_status(F("2026-01-01-one.md")) != "superseded":
            bad.append("declared_status regressed on frontmatter: %r" % declared_status(F("2026-01-01-one.md")))
        if declared_status(F("2026-01-08-nofm.md")) is not None:
            bad.append("declared_status read a status from a file with no frontmatter")
    return bad


def main():
    args = sys.argv[1:]
    if "--selftest" in args:
        bad = selftest()
        for b in bad:
            print("SELFTEST FAIL: %s" % b)
        print("finding_status selftest: %s" % ("FAILED (%d)" % len(bad) if bad else "PASS"))
        return 1 if bad else 0

    if "--check" in args:
        for p in args[args.index("--check") + 1:]:
            full = p if os.path.isabs(p) else os.path.join(ROOT, p)
            s, ev = status_of(full)
            print("%-11s %s" % (s.upper(), os.path.basename(p)))
            if ev:
                print("            %s" % ev)
            # Printed for retracted/superseded ONLY: elsewhere the question "what replaced this?" is not asked,
            # and a line that prints on every check is noise, not an answer.
            if s in ("retracted", "superseded"):
                print("            %s" % successor_report(full))
        return 0

    rows = scan()
    counts = {}
    for _, s, _, _ in rows:
        counts[s] = counts.get(s, 0) + 1
    print("finding_status: %d findings" % len(rows))
    declared = sum(1 for _, _, ev, _ in rows if ev == "declared")
    print("  DECLARED (authoritative): %d of %d" % (declared, len(rows)))
    print("  the rest are KEYWORD HINTS, measured ~1-in-6 precise — backfill suggestions, not verdicts:")
    for s in ("live", "live?", "qualified", "corrected", "superseded", "retracted", "unknown"):
        if counts.get(s):
            print("  %-11s %4d" % (s, counts[s]))
    not_live = [r for r in rows if r[1] not in ("live",)]
    print("  => %d of %d flagged non-live by the HINT. Treat as a backfill queue, not a registry." % (len(not_live), len(rows)))

    # SUCCESSOR COVERAGE, over DECLARED dead findings only -- the keyword hint is ~1-in-6 precise, so measuring
    # this over hints would report a coverage gap that is mostly mislabelled files.
    dead = [r for r in rows if r[2] == "declared" and r[1] in ("retracted", "superseded")]
    named = [r for r in dead if r[3]]
    print("  successor: %d of %d DECLARED retracted/superseded findings name a `superseded_by:`"
          % (len(named), len(dead)))
    if len(named) < len(dead):
        print("     the other %d leave \"what replaced this?\" to hand-reconstruction at every citation."
              % (len(dead) - len(named)))

    if "--write" in args:
        out = os.path.join(ROOT, "docs", "FINDINGS_STATUS.md")
        with open(out, "w") as fh:
            fh.write("# Findings status — GENERATED, do not edit\n\n")
            fh.write("Regenerate: `.venv/bin/python tools/finding_status.py --write`\n\n")
            fh.write("Derived by scanning each finding's own head for the vocabulary this corpus already uses.\n")
            fh.write("It exists because `docs/RETRACTED.md` held ONE row against %d non-live findings, so a RAG\n"
                     % len(not_live))
            fh.write("hit could not tell a void result from a live one — and did not, on 2026-07-31.\n\n")
            for s in ("retracted", "superseded", "corrected", "qualified"):
                sel = [r for r in rows if r[1] == s]
                if not sel:
                    continue
                fh.write("\n## %s (%d)\n\n" % (s.upper(), len(sel)))
                for path, _, ev, succ in sel:
                    fh.write("- `%s`%s\n" % (os.path.basename(path), ("  \n  %s" % ev) if ev else ""))
                    if s in ("retracted", "superseded") and ev == "declared":
                        fh.write("  \n  %s\n" % successor_report(os.path.join(ROOT, path)))
        print("  wrote docs/FINDINGS_STATUS.md (%d non-live entries)" % len(not_live))
    return 0


if __name__ == "__main__":
    sys.exit(main())
