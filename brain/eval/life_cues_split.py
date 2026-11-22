# Capability: Lifelong learning
# Record: brain/research/2026-11-22-fresh-test-split-design.md
# Expected-fail: none (lookup shim: life_cues.py stays frozen at its registered sha256 2055dc92...)
"""Per-split cue-text lookup for the cue ladder. dev and test (v1) come from brain/eval/life_cues.py, which is left
byte-identical (its sha256 is the registered hash of the 2026-11-19 multiweek design, and every --cue-ladder life
records it), so TEST v2 (design 2026-11-22) is reached through this shim: the cue texts in life_cues_test2.TEST2 and
the bank in life_bank_test2, both imported lazily so a missing v2 file never breaks dev or test."""
from __future__ import annotations

import importlib

from brain.eval import life_cues as LC

LAZY = {"test2": ("brain.eval.life_cues_test2", "TEST2")}
LAZY_BANK = {"test2": "brain.eval.life_bank_test2"}


def _lazy(mod: str):
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name != mod:
            raise
        return None


def cues(split: str):
    """{item: (C2, C3)} of the split, or None when the split has no cue texts (a v2 file not yet written)."""
    if split in LC.CUES:
        return LC.CUES[split]
    if split in LAZY:
        mod, attr = LAZY[split]
        m = _lazy(mod)
        return None if m is None else getattr(m, attr)
    return None


def overlaps(split: str, bank=None) -> dict:
    """life_cues.overlaps for any split: dev/test delegate to it unchanged; test2 uses the same formula on the v2 cue
    texts and (by default) the v2 bank."""
    if split in LC.CUES:
        return LC.overlaps(split, bank)
    cs = cues(split)
    if cs is None:
        raise ValueError(f"no cue texts for split {split!r}")
    if bank is None:
        bank = importlib.import_module(LAZY_BANK[split])
    out = {}
    for iid, (c2, c3) in cs.items():
        tel = LC.content(bank.ALL[iid].telling)
        out[iid] = (len(LC.content(c2) & tel), len(LC.content(c3) & tel),
                    max(len(LC.content(p) & tel) for p in LC.C4_PROMPTS))
    return out
