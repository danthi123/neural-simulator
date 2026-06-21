"""CI GUARD (shortcut #3 fold): the `integrated_loop` opt-in on OneBrainComposer routes the (agent, action) cue-match
through the validated spiking K-way sequencer (gated-disinhibition match cascade + BG first-match priority WTA) instead
of the host first-match `_scan` loop. The HARD gate: answer-IDENTITY to the host-`_scan` oracle on the full who/what
matrix INCLUDING every `is None`/`"unknown"` abstention, the no-confab moat 0-false-accept, multi-seed, with the
default-OFF path byte-identical.

Plan: docs/plans/2026-06-21-shortcut3-fold-host-scan-to-spiking-sequencer-plan.md (commit 94a5e237).
De-risk op-point: match_thresh=0.06, gain=0.11, sigma=1.0, input_gain=1.0, retreat=divnorm
(2026-06-21-shortcut3-K32-capability-surpass.md). The divisive-norm op-point is calibrated for the PRODUCTION vocab
scale (V~72-320), so the answer-identity matrix uses the de-risk's V=72 vocab at SMALL K (few blocks = fast) instead of
a hand-tuned small-V op-point -- it exercises the EXACT validated op-point, not a special-cased one.

The small-K answer-identity matrix runs on numpy-CPU (fast); the 320-scale + K=32 gate is the de-risk runner (GPU).
This guard pins (a) the OFF default is byte-identical and `integrated_loop` defaults False; then the routed who/what +
abstention matrix == host, multi-seed.
"""
import os

import numpy as np
import pytest

# the small-K answer-identity matrix is a CPU exact-algebra parity check (the sequencer fabric + the composer run on
# the numpy backend); force numpy so CI does not require a GPU for this guard.
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402
# the de-risk's PRODUCTION-scale vocab (V=72) + its 32 distinct facts -- so the divnorm op-point (gain=0.11) is the
# validated one (a small hand-picked vocab over-normalizes; the gate runs this exact vocab at K=32 / V=320).
from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import (  # noqa: E402
    ALL_FACTS as DERISK_FACTS, VOCAB as DERISK_VOCAB, _build_queries as derisk_build_queries,
)

# a small fully-distinct fact set drawn from the de-risk's set -> the host `_scan` is unambiguous + the moat cues clean.
# D small for CPU speed (the gate's K=32 / V=320 is the de-risk runner). K=4 is the answer-identity workhorse.
K_SMALL = 4
FACTS = DERISK_FACTS[:K_SMALL]
VOCAB = DERISK_VOCAB


def _pair(seed, K=K_SMALL, D=128, **kw):
    """Build a HOST-oracle composer (integrated_loop=False) and a SPIKING composer (integrated_loop=True) on the SAME
    facts/codes, both on numpy-CPU (enable_batched off = the per-block oracle path). D=128 is the PRODUCTION dimension
    (the K=32 surpass gate's D) -- the divnorm op-point + the match_thresh=0.06 margin are calibrated there; D=64 codes
    are noisier and a real match can dip below the production threshold (the documented single-low-fidelity-code
    signature). The small-K (few blocks) keeps it fast even at D=128."""
    facts = DERISK_FACTS[:K]
    host = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=max(8, K), enable_batched=False,
                            enable_rf_cudagraph=False, integrated_loop=False, **kw)
    seq = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=max(8, K), enable_batched=False,
                           enable_rf_cudagraph=False, integrated_loop=True, **kw)
    for (a, x, p) in facts:
        host.store(a, x, p); seq.store(a, x, p)
    return host, seq, facts


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
    facts = DERISK_FACTS[:K_SMALL]
    base = OneBrainComposer(seed=42, D=64, vocab=VOCAB, k_max=8, enable_batched=False, enable_rf_cudagraph=False)
    off = OneBrainComposer(seed=42, D=64, vocab=VOCAB, k_max=8, enable_batched=False, enable_rf_cudagraph=False,
                           integrated_loop=False)
    for (a, x, p) in facts:
        base.store(a, x, p); off.store(a, x, p)
    queries = [(a, x) for (a, x, p) in facts] + [("zzz", facts[0][1]), (facts[0][0], "fly")]
    for (a, x) in queries:
        assert base.query_patient(a, x) == off.query_patient(a, x)
    for (a, x, p) in facts:
        assert base.ask_yes_no(a, x, p) == off.ask_yes_no(a, x, p) == "yes"


# ----------------------------------------------------------------------------------------------------------------
# Task 2: the spiking `_seq_block` branch. At small K, an integrated_loop=True composer's _seq_block selects the same
# block index as the host read for a present cue, and abstains (None) on an unstored cue.
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [42, 43])
def test_seq_block_present_and_abstain(seed):
    """integrated_loop=True: `_seq_block` (the spiking K-way sequencer decision) selects the SAME block index as the
    host first-match on a present cue, and returns None (abstain) on an unstored cue -- the moat. Multi-seed."""
    host, seq, facts = _pair(seed)
    for i, (a, x, p) in enumerate(facts):                      # present cues -> the same block index as host
        h = host._seq_block(a, x)
        s = seq._seq_block(a, x)
        assert h == i, f"host first-match must select block {i} for {(a, x)}, got {h}"
        assert s == h, f"spiking _seq_block {s} != host {h} for present cue {(a, x)}"
    a0, x0 = facts[0][0], facts[0][1]
    for (a, x) in [("zzz", x0), (a0, "fly"), (a0, facts[1][1])]:   # absent agent / absent action / cross -> abstain
        assert host._seq_block(a, x) is None
        assert seq._seq_block(a, x) is None, f"spiking _seq_block must abstain (None) on unstored cue {(a, x)}"


@pytest.mark.parametrize("seed", [42, 43])
def test_query_patient_integrated_loop(seed):
    """integrated_loop=True: query_patient present-cue == host (routes through the spiking decision once Task 3 lands;
    Task 2 keeps query_patient on its host loop so this stays green either way), and an unstored cue is None."""
    host, seq, facts = _pair(seed)
    for (a, x, p) in facts:
        assert seq.query_patient(a, x) == host.query_patient(a, x) == p
    a0, x0 = facts[0][0], facts[0][1]
    assert seq.query_patient("zzz", x0) is None and host.query_patient("zzz", x0) is None
    assert seq.query_patient(a0, facts[1][1]) is None, "moat: a cross cue must abstain"
