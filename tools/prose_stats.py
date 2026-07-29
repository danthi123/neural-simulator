#!/usr/bin/env python3
"""Prose readability statistics for markdown docs. Reporting only — enforces nothing.

WHY THIS IS A TOOL AND NOT AN AD-HOC ONE-LINER. I derived this metric three times by hand on
2026-07-28 and got two WRONG baselines from subtly different logic:
  * counting TABLE ROWS and directory trees as prose -> a phantom "284-word sentence", median inflated
    24 -> 27;
  * not treating a trailing ':' as a sentence end -> five short code-block captions glued into one
    fake 83-word sentence.
Both would have inflated a reported improvement. One tool, one definition, reproducible.

EXCLUDED from prose: fenced code, table rows, headings, list items, blockquotes, badge/image lines,
directory-tree lines. Sentence ends: . ! ? and : (a caption introducing a code block IS a sentence).

    .venv/bin/python tools/prose_stats.py [paths...]        # default: the public-facing set
    .venv/bin/python tools/prose_stats.py --git-before FILE # compare working tree against HEAD
"""
import io
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = ["README.md", "QUICKSTART.md", "CONTRIBUTING.md", "ROADMAP.md", "CHANGELOG.md",
           "docs/TERMS.md", "docs/WRITING.md"]
SKIP_PREFIX = ("|", "#", "-", "*", ">", "├", "└", "│", "[!", "!", "=", "+")


def prose(text):
    out, fence = [], False
    for ln in text.split("\n"):
        if re.match(r"^[ \t]*```", ln):
            fence = not fence
            continue
        if fence:
            continue
        s = ln.strip()
        if not s or s.startswith(SKIP_PREFIX) or re.match(r"^\d+\.\s", s):
            continue
        out.append(s)
    return " ".join(out)


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?:])\s+", prose(text)) if len(s.split()) > 2]


def stats(text):
    ws = [len(s.split()) for s in sentences(text)]
    if not ws:
        return dict(n=0, median=0, over25=0, pct25=0, over40=0, longest=0, passive=0)
    p = len(re.findall(r"\b(is|are|was|were|been|being)\s+\w+ed\b", prose(text)))
    return dict(n=len(ws), median=statistics.median(ws), over25=sum(1 for w in ws if w > 25),
                pct25=round(100 * sum(1 for w in ws if w > 25) / len(ws)),
                over40=sum(1 for w in ws if w > 40), longest=max(ws), passive=p)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--git-before" in sys.argv:
        for rel in args:
            before = subprocess.run(["git", "show", "HEAD:" + rel], capture_output=True, text=True,
                                    cwd=ROOT).stdout
            after = io.open(os.path.join(ROOT, rel), encoding="utf-8").read()
            b, a = stats(before), stats(after)
            print("%s" % rel)
            print("  before  n=%-4d median=%-3d >25w=%-3d (%d%%)  >40w=%-3d longest=%d"
                  % (b["n"], b["median"], b["over25"], b["pct25"], b["over40"], b["longest"]))
            print("  after   n=%-4d median=%-3d >25w=%-3d (%d%%)  >40w=%-3d longest=%d"
                  % (a["n"], a["median"], a["over25"], a["pct25"], a["over40"], a["longest"]))
        return 0
    paths = args or DEFAULT
    print("%-46s %6s %7s %8s %7s %8s" % ("doc", "sents", "median", ">25w", ">40w", "longest"))
    for rel in paths:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        s = stats(io.open(p, encoding="utf-8").read())
        flag = "  <-- worst" if s["pct25"] >= 40 or s["longest"] >= 60 else ""
        print("%-46s %6d %7d %5d(%2d%%) %7d %8d%s"
              % (rel, s["n"], s["median"], s["over25"], s["pct25"], s["over40"], s["longest"], flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
