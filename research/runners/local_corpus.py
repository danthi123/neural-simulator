"""Zero-download local English corpus (Increment 1).

The Phase-2 generator needs raw local text. Tiny-Shakespeare raw text
is not committed (only the trained .npz). For a fully self-contained,
zero-network foundation we use the repo's OWN English prose — the
research findings docs — concatenated deterministically. This is real
English with real sequential structure (sentences, grammar), which is
all the Increment-1 anti-cheat gate needs (real-text loss reduction
must beat a permuted-character control). No network, ever.
"""
from __future__ import annotations
import glob
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_local_corpus() -> str:
    """Deterministic concatenation of research/findings/**/*.md (sorted by
    path). Pure local file I/O; no network.

    RECURSIVE since 2026-07-31: a flat `*.md` silently omitted 42 findings sitting in
    `research/findings/raw/`, so they were absent from every consumer of this corpus."""
    pattern = os.path.join(_REPO_ROOT, "research", "findings", "**", "*.md")
    paths = sorted(glob.glob(pattern, recursive=True))
    parts = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                parts.append(fh.read())
        except OSError:
            continue
    return "\n".join(parts)
