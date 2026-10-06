#!/usr/bin/env python3
"""needs_line.py -- compose (and optionally append) one NEEDS-CLAUDE line for the driver agent (Claude, 2026-10-04).

Two failures this prevents, both seen in session 6: NEEDS lines far over the runbook's 300-character format (8 of 25
open lines), and a NEEDS line re-raising a question a section-6 RULING had already answered (T12, closed 10-01, was
re-filed 10-04). So the helper (1) builds the exact format, (2) refuses a line over 300 characters (put the detail
in the packet's WORKLOG and cite it), and (3) prints every RULING in section 6 that mentions the key and refuses to
write until you pass --rulings-read (read them first: the answer may already be there).

    python3 tools/needs_line.py --key T12 --kind STOP --text "tone-bg finding missing on main; blocks: T12"
    python3 tools/needs_line.py --key T12 --kind STOP --text "..." --rulings-read --write
Exit 0 = printed (and appended with --write); 1 = refused (reason printed); 2 = usage.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNBOOK = os.path.join(ROOT, "research", "coordination", "LOCAL_LLM_RUNBOOK.md")
MAX_LEN = 300
KINDS = ("STOP", "ASK", "RESULT", "INFO")


def compose(key: str, kind: str, text: str, n: str = "", when: _dt.datetime | None = None) -> str:
    when = when or _dt.datetime.now()
    tag = f"{kind} {n}".strip()
    return f"- {when:%Y-%m-%d %H:%M} [{key}] {tag}: {' '.join(text.split())}"


def rulings_for(key: str, runbook_text: str) -> list[str]:
    """Every section-6 ruling (a `- <name>: ...` item, with its continuation lines) that mentions `key`."""
    try:
        sec = runbook_text.split("\n## 6.", 1)[1].split("\n## NEEDS CLAUDE", 1)[0]
    except IndexError:
        return []
    items, cur = [], []
    for line in sec.splitlines():
        if re.match(r"^- \S", line):
            if cur:
                items.append("\n".join(cur))
            cur = [line]
        elif cur and line.startswith("  "):
            cur.append(line)
    if cur:
        items.append("\n".join(cur))
    pat = re.compile(r"(?<![\w-])" + re.escape(key) + r"(?![\w-])", re.IGNORECASE)
    return [it for it in items if pat.search(it)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--key", required=True, help="task or packet key, e.g. T12 or s1-c4-void")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--n", default="", help="the HANDOFF's NEEDS-CLAUDE item, e.g. '#2'")
    ap.add_argument("--text", required=True, help="what you saw; tried: what; blocks: what")
    ap.add_argument("--rulings-read", action="store_true", help="you have read the rulings this prints")
    ap.add_argument("--write", action="store_true", help="append the line to the NEEDS CLAUDE section")
    ap.add_argument("--runbook", default=RUNBOOK)
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9§._/ -]{1,60}", a.key):
        print("bad --key", file=sys.stderr)
        return 2
    line = compose(a.key, a.kind, a.text, a.n)
    text = open(a.runbook, encoding="utf-8").read()
    found = rulings_for(a.key, text)
    print(line)
    print(f"({len(line)} characters, limit {MAX_LEN})")
    if found:
        print(f"\n{len(found)} section-6 RULING(s) mention [{a.key}]; read them, the answer may already be there:")
        for it in found:
            print("  " + it.replace("\n", "\n  "))
    if len(line) > MAX_LEN:
        print(f"\n⛔ REFUSED: {len(line)} > {MAX_LEN} characters. Put the detail in the packet WORKLOG and cite it.",
              file=sys.stderr)
        return 1
    if not a.write:
        return 0
    if found and not a.rulings_read:
        print("\n⛔ REFUSED: rulings exist for this key; read them, then re-run with --rulings-read", file=sys.stderr)
        return 1
    if "\n## NEEDS CLAUDE" not in text:
        print("⛔ REFUSED: no '## NEEDS CLAUDE' section in the runbook", file=sys.stderr)
        return 1
    with open(a.runbook, "a", encoding="utf-8") as f:
        f.write(("" if text.endswith("\n") else "\n") + line + "\n")
    print("appended to NEEDS CLAUDE (commit it: `runbook: NEEDS [<key>] ...`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
