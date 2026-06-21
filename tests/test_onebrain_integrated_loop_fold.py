"""CI GUARD (shortcut #3 fold): the `integrated_loop` opt-in on OneBrainComposer routes the (agent, action) cue-match
through the validated spiking K-way sequencer (gated-disinhibition match cascade + BG first-match priority WTA) instead
of the host first-match `_scan` loop. The HARD gate: answer-IDENTITY to the host-`_scan` oracle on the full who/what
matrix INCLUDING every `is None`/`"unknown"` abstention, the no-confab moat 0-false-accept, multi-seed, with the
default-OFF path byte-identical.

Plan: docs/plans/2026-06-21-shortcut3-fold-host-scan-to-spiking-sequencer-plan.md (commit 94a5e237).
De-risk op-point: match_thresh=0.06, gain=0.11, sigma=1.0, input_gain=1.0, retreat=divnorm
(2026-06-21-shortcut3-K32-capability-surpass.md).

The small-V answer-identity matrix runs on numpy-CPU (fast); the 320-scale + K=32 gate is the de-risk runner (GPU).
This guard pins (a) the OFF default is byte-identical and `integrated_loop` defaults False (Task 1), then grows with
each routed site.
"""
import os

import numpy as np
import pytest

# the small-V answer-identity matrix is a CPU exact-algebra parity check (the sequencer fabric + the composer run on
# the numpy backend); force numpy so CI does not require a GPU for this guard.
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402

# a small, fully-distinct fact set + vocab so the host `_scan` is unambiguous and the moat cues are clean.
FACTS = [("dog", "go", "north"), ("cat", "run", "river"), ("fox", "see", "tree"), ("bird", "fly", "sun")]
VOCAB = sorted(set(w for (a, x, p) in FACTS for w in (a, x, p)))   # 12 words


def _build(seed, **kw):
    """A composer on the small fact set (numpy-CPU). enable_batched off keeps the per-block oracle path; D small for
    speed."""
    c = OneBrainComposer(seed=seed, D=64, vocab=VOCAB, k_max=8, enable_batched=False,
                         enable_rf_cudagraph=False, **kw)
    for (a, x, p) in FACTS:
        c.store(a, x, p)
    return c


# ----------------------------------------------------------------------------------------------------------------
# Task 1: the flag exists, defaults OFF, and the OFF path is byte-identical to a no-flag build.
# ----------------------------------------------------------------------------------------------------------------
def test_integrated_loop_defaults_off():
    """The opt-in defaults OFF (byte-identical = the host-`_scan` oracle), mirroring enable_spiking_cleanup."""
    c = OneBrainComposer(seed=42, D=64, vocab=VOCAB, k_max=8, enable_batched=False, enable_rf_cudagraph=False)
    assert c.integrated_loop is False, "integrated_loop must default False (byte-identical host-scan oracle)"
    # the lazy sequencer caches are inert when OFF (no sequencer bridge constructed at __init__)
    assert c._seq is None and c._seq_score is None and c._seq_drives is None


def test_off_path_byte_identical():
    """A default (OFF) composer answers the full who/what matrix + abstentions; an explicit integrated_loop=False build
    is IDENTICAL on the same facts -- the flag-OFF path changes nothing (it IS the pre-fold host read)."""
    base = _build(42)
    off = _build(42, integrated_loop=False)
    queries = [(a, x) for (a, x, p) in FACTS] + [("apple", "go"), ("dog", "fly"), ("zzz", "see")]
    for (a, x) in queries:
        assert base.query_patient(a, x) == off.query_patient(a, x)
    for (a, x, p) in FACTS:
        assert base.ask_yes_no(a, x, p) == off.ask_yes_no(a, x, p) == "yes"
    assert base.query_patient("apple", "go") is None and off.query_patient("apple", "go") is None
