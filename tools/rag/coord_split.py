"""Split the coordination runbook into one RAG document per ruling (stdlib only, so tests import it without LlamaIndex).

Why (2026-10-21): the binding Claude rulings live in research/coordination/LOCAL_LLM_RUNBOOK.md section 6, and Hermes's
RESULT lines under "## NEEDS CLAUDE". The corpus did not index research/coordination/ at all, so a search could not find
the 10-16 and 10-20 rulings that constrain #271, and two design rounds started from a reading that never saw them.
Indexing the 500 KB runbook as one document would let it dominate hits and re-embed on every Hermes edit, so:
- sections before the first split section become one document each (a section of standing rules or tasks);
- from the RULINGS section on, every top-level bullet (a line starting "- " at column 0, with its continuation lines)
  is its own document, so one edit re-embeds one bullet.
Each document id is the runbook path plus a fragment hashed from the bullet's FIRST line, so an edit elsewhere in the
file never changes it; a repeated first line gets an occurrence suffix.
"""
from __future__ import annotations

import hashlib
import re

RUNBOOK_BASENAME = "LOCAL_LLM_RUNBOOK.md"
# Headings from which every top-level bullet is its own document (matched as a prefix of the "## " heading text).
SPLIT_FROM_PREFIXES = ("6. RULINGS",)

# The bullet's OWN key: "- <key>: ..." or "- [date [time zone]] RULING <key> ...". A RULING key mentioned later in the
# line belongs to another ruling and must not name this one.
_RULING = re.compile(r"^- (?:\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?(?: [A-Z]{2,4})? )?RULING\s+([A-Za-z0-9][\w.\-]*)")
_KEY = re.compile(r"^- ([A-Za-z0-9][\w.\-]*):\s")
_TAG = re.compile(r"\[([^\]\n]{1,60})\]")


def _label(first_line: str) -> str:
    """A short display name for a bullet: its RULING key, its `<key>:` prefix, its first [tag], or its opening words."""
    for pattern in (_RULING, _KEY):
        match = pattern.match(first_line)
        if match:
            return ("RULING " if pattern is _RULING else "") + match.group(1).rstrip(".")
    match = _TAG.search(first_line)
    if match:
        return "[" + match.group(1).strip() + "]"
    return first_line[2:62].strip()


def _fragment(kind: str, first_line: str, seen: dict[str, int]) -> str:
    digest = hashlib.sha1(first_line.strip().encode("utf-8", "replace")).hexdigest()[:12]
    base = f"{kind}-{digest}"
    seen[base] = seen.get(base, 0) + 1
    return base if seen[base] == 1 else f"{base}-{seen[base]}"


def split_runbook(text: str) -> list[dict]:
    """Return [{fragment, label, section, text}] for the runbook text (no I/O)."""
    out: list[dict] = []
    seen: dict[str, int] = {}
    section = ""
    splitting = False
    block: list[str] = []
    block_kind = "section"

    def flush():
        body = "\n".join(block).strip()
        if not body:
            return
        first = block[0] if block_kind == "bullet" else (section or "preamble")
        label = _label(first) if block_kind == "bullet" else (section or "preamble")
        out.append({
            "fragment": _fragment(block_kind, first if block_kind == "bullet" else "section:" + label, seen),
            "label": label,
            "section": section,
            "text": f"{RUNBOOK_BASENAME}, section '{section or 'preamble'}':\n{body}",
        })

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            block = []
            section = line[3:].strip()
            splitting = splitting or section.startswith(SPLIT_FROM_PREFIXES)
            block_kind = "section"
            if not splitting:
                block = [line]
            continue
        if splitting and line.startswith("- "):
            flush()
            block, block_kind = [line], "bullet"
            continue
        block.append(line)
    flush()
    return out
