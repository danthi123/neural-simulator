#!/usr/bin/env python3
"""Split over-long prose lines in the governed docs at sentence boundaries (docs/WRITING.md W2).

SAFETY: markdown joins consecutive lines of a paragraph, so splitting at sentence boundaries does not
change rendering — PROVIDED any line prefix (blockquote '>', list bullet, indent) is carried onto each
continuation line. This script does that, then VERIFIES that the whitespace-normalised text content of
every file is byte-identical before and after. If verification fails it restores the original and exits 1.

Run:  .venv/bin/python tools/split_long_doc_lines.py [--apply]
Without --apply it reports only.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOVERNED = ["CLAUDE.md", "GAP_CLOSURE_MISSION.md", "ROADMAP.md", "README.md", "docs/TERMS.md",
            "docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md"]
MAX = 800

# Sentence end: . ! ? or a closing ** / ) / ` immediately after one, then whitespace, then a capital,
# emoji, digit, or markdown emphasis. Avoids splitting on "e.g." / "vs." / version numbers / file paths.
SENT = re.compile(r'(?<=[.!?])(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bcf)(?<!\bFig)(?<!\bNo)\s+(?=[A-Z0-9*_`⭐⛔✅🎉🔑📍⚠️🔬📊🧭🎯])')


def _prefix(line):
    """The prefix to repeat on continuation lines: blockquote markers and/or indentation."""
    m = re.match(r'^([ \t]*(?:>[ \t]*)*)', line)
    pre = m.group(1) if m else ""
    # a list bullet is NOT repeated (that would create new items); continuation is indented instead
    m2 = re.match(r'^([ \t]*(?:>[ \t]*)*)([-*+] |\d+\. )', line)
    if m2:
        pre = m2.group(1) + " " * len(m2.group(2))
    return pre


def split_line(line):
    if len(line) <= MAX:
        return [line]
    pre = _prefix(line)
    parts = SENT.split(line)
    if len(parts) == 1:
        # This dialect uses ' · ' and '; ' as de-facto sentence separators (measured: 736 semicolons and
        # 154 '·' in GAP_CLOSURE_MISSION.md), so a 1,200-char "sentence" often has no period at all.
        # Fall back to those boundaries — the separator is KEPT, so content is unchanged.
        parts = re.split(r'(?<=[·;])\s+', line)
    if len(parts) == 1:
        return [line]                      # genuinely no boundary — leave it, report it
    out, cur = [], parts[0]
    for p in parts[1:]:
        cand = cur + " " + p
        if len(cand) > MAX and cur.strip():
            out.append(cur)
            cur = pre + p
        else:
            cur = cand
    out.append(cur)
    return out


def norm(text):
    """Content with markdown line-syntax removed, for verifying nothing was LOST.

    Continuation lines carry the original line's blockquote/indent prefix so the paragraph still renders
    as one block. Those repeated '>' markers are SYNTAX, not content, so they must be stripped before
    comparing — otherwise the check reports a false difference and refuses a correct split. (It did
    exactly that on the first run, which is the check working: it refused to write rather than assume.)
    """
    lines = [re.sub(r'^[ \t]*(?:>[ \t]*)*', '', ln) for ln in text.split("\n")]
    return re.sub(r'\s+', ' ', " ".join(lines)).strip()


def main():
    apply = "--apply" in sys.argv
    infence = False
    total_before = total_after = 0
    for rel in GOVERNED:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        original = io.open(p, encoding="utf-8").read()
        out, infence, changed, unsplittable = [], False, 0, 0
        for ln in original.split("\n"):
            if re.match(r'^[ \t]*```', ln):
                infence = not infence
                out.append(ln)
                continue
            if infence or ln.lstrip().startswith("|") or len(ln) <= MAX:
                out.append(ln)
                continue
            total_before += 1
            pieces = split_line(ln)
            if len(pieces) == 1:
                unsplittable += 1
                out.append(ln)
            else:
                changed += 1
                out.extend(pieces)
        new = "\n".join(out)
        still = sum(1 for l in new.split("\n") if len(l) > MAX and not l.lstrip().startswith("|"))
        print("%-58s split %2d line(s), %d unsplittable, %d still over" % (rel, changed, unsplittable, still))
        total_after += still
        if apply and changed:
            if norm(new) != norm(original):
                print("  ⛔ CONTENT CHANGED — refusing to write %s" % rel)
                return 1
            io.open(p, "w", encoding="utf-8").write(new)
    print("\n%d over-long lines found; %d remain after split" % (total_before, total_after))
    print("(content verified whitespace-identical before writing)" if apply else "(dry run — pass --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
