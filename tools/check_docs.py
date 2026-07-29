#!/usr/bin/env python3
"""Check the two document-structure rules in docs/WRITING.md. Exit 1 on violation.

W1  a governed file may not cite a path registered in docs/RETRACTED.md without an
    inline retraction marker on the same line/bullet.
W2  prose lines in governed files are at most 800 characters (table rows + fenced code exempt).

Deliberately NOT checked: truth, hedging, sentence length, voice, vocabulary. Those either belong to
docs/TERMS.md or to .claude/skills/verify-go/SKILL.md, and six of the nine 2026-07-28 retractions were
instrument failures that no writing rule can catch.
"""
import io, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOVERNED = ["CLAUDE.md", "GAP_CLOSURE_MISSION.md", "ROADMAP.md", "README.md", "docs/TERMS.md",
            "docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md"]
REGISTRY = "docs/RETRACTED.md"
MAX_LINE = 800
MARKERS = ("⛔", "RETRACTED", "VOID", "SUPERSEDED", "WITHDRAWN")


def _registered_paths():
    p = os.path.join(ROOT, REGISTRY)
    if not os.path.exists(p):
        return []
    out = []
    for ln in io.open(p, encoding="utf-8"):
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] and not cells[0].startswith("-") and "path or commit" not in cells[0]:
            out.append(os.path.basename(cells[0]))
    return out


def _lines(path, prose_only):
    """Yield (lineno, text). prose_only=True also skips TABLE ROWS (for W2's length rule).

    W1 must see table rows: a stale citation inside the wall-ledger table is MORE harmful than one in
    prose, not less. The first version of this checker exempted tables from BOTH rules and therefore
    reported 0 W1 violations while an unmarked citation sat in a governed table row.
    """
    infence = False
    for i, ln in enumerate(io.open(path, encoding="utf-8"), 1):
        if re.match(r"^[ \t]*```", ln):
            infence = not infence
            continue
        if infence:
            continue
        if prose_only and ln.lstrip().startswith("|"):
            continue
        yield i, ln.rstrip("\n")


def main():
    reg = _registered_paths()
    w1, w2 = [], []
    for rel in GOVERNED:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        for n, text in _lines(p, prose_only=True):
            if len(text) > MAX_LINE:
                w2.append("%s:%d (%d chars)" % (rel, n, len(text)))
        for n, text in _lines(p, prose_only=False):
            for name in reg:
                # Citations are routinely TRUNCATED in our docs ("...-REATTRIBUTED-..." /
                # "...-REATTRIBUTED-dense-"), so an exact-basename match is a FALSE NEGATIVE: the first
                # version of this checker reported 0 violations while 3 unmarked citations existed.
                # Match any hyphen-boundary prefix of >=30 chars instead.
                stem = name[:-3] if name.endswith(".md") else name
                hit = stem in text
                if not hit:
                    parts = stem.split("-")
                    for k in range(len(parts), 0, -1):
                        pref = "-".join(parts[:k])
                        if len(pref) >= 30 and pref in text:
                            hit = True
                            break
                if hit and not any(m in text for m in MARKERS):
                    w1.append("%s:%d cites retracted %s without a marker" % (rel, n, name))
    for label, rows in (("W1 (unmarked citation of a retracted doc)", w1),
                        ("W2 (prose line > %d chars)" % MAX_LINE, w2)):
        print("%s: %d" % (label, len(rows)))
        for r in rows[:40]:
            print("    " + r)
        if len(rows) > 40:
            print("    ... and %d more" % (len(rows) - 40))
    if w1 or w2:
        print("\nFAIL — see docs/WRITING.md")
        return 1
    print("\nOK — both document-structure rules pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
