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


# ----------------------------------------------------------------------------------------------------------------
# Task 3: route query_patient + _find_cued_block through _seq_block. The answer-identity + moat battery
# (query_patient, reason_chain, update_on_mismatch abstain) == host, fa_total 0; the spiking path is exercised.
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [42, 43])
def test_query_patient_routes_through_sequencer(seed):
    """integrated_loop=True query_patient delegates the (agent, action) cue-match to _seq_block (the spiking decision),
    NOT the inlined host loop. Asserted by spying on _seq_block: it is called for every query_patient (present + moat),
    and the answers still == host (== truth) with the moat intact."""
    host, seq, facts = _pair(seed)
    calls = {"n": 0}
    orig = seq._seq_block
    seq._seq_block = lambda a, x, _o=orig, _c=calls: (_c.__setitem__("n", _c["n"] + 1), _o(a, x))[1]
    for (a, x, p) in facts:
        assert seq.query_patient(a, x) == host.query_patient(a, x) == p
    a0, x0 = facts[0][0], facts[0][1]
    assert seq.query_patient("zzz", x0) is None, "moat: absent agent abstains through the routed query_patient"
    assert seq.query_patient(a0, facts[1][1]) is None, "moat: a cross cue abstains through the routed query_patient"
    assert calls["n"] == len(facts) + 2, "query_patient must call _seq_block once per query (the spiking route)"


@pytest.mark.parametrize("seed", [42, 43])
def test_reason_chain_and_reconsolidation_abstain_through_sequencer(seed):
    """reason_chain (iterates the routed query_patient) and update_on_mismatch (via _find_cued_block) abstain through
    the spiking decision == host. A valid 2-hop chain answers; a broken hop abstains; a never-stored reconsolidation
    cue abstains -- the no-confab moat holds at every hop on the spiking route."""
    # a chain store: fact0.patient is fact1.agent so a 2-hop chase resolves (a0 -x0-> p0==a1 -x1-> p1)
    host = OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, enable_batched=False,
                            enable_rf_cudagraph=False, integrated_loop=False)
    seq = OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, enable_batched=False,
                           enable_rf_cudagraph=False, integrated_loop=True)
    chain = [("dog", "go", "cat"), ("cat", "run", "river")]   # all words are in the de-risk vocab
    for (a, x, p) in chain:
        host.store(a, x, p); seq.store(a, x, p)
    # a valid 2-hop chain (dog -go-> cat -run-> river) == host == truth
    assert seq.query_chain("dog", ["go", "run"]) == host.query_chain("dog", ["go", "run"]) == "river"
    # a broken hop (no (cat, go) fact) abstains == host (the moat at hop 2)
    assert seq.query_chain("dog", ["go", "go"]) is None and host.query_chain("dog", ["go", "go"]) is None
    # reconsolidation: a NEVER-stored cue abstains via _find_cued_block (the routed spiking decision)
    rm = seq.update_on_mismatch("bird", "fly", "north")
    rmh = host.update_on_mismatch("bird", "fly", "north")
    assert rm["action"] == rmh["action"] == "abstain", f"never-stored reconsolidation cue must abstain: {rm} vs {rmh}"
    assert seq.count_facts("bird", "fly") == 0, "no fabricated trace on the spiking route"


# ----------------------------------------------------------------------------------------------------------------
# Task 4: route ask_yes_no through _seq_block. affirmative -> yes, negated -> no, unstored -> unknown == host.
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [42, 43])
def test_ask_yes_no_routes_through_sequencer(seed):
    """ask_yes_no routes the (agent, action) cue-match through _seq_block (the spiking decision), then reads the
    selected block's decoded patient + polarity: an affirmative full-SVO -> 'yes', a negated fact -> 'no', a wrong
    patient -> the SVO does not match -> 'unknown', an unstored cue -> 'unknown' (the moat). == host, multi-seed."""
    host = OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, enable_batched=False,
                            enable_rf_cudagraph=False, integrated_loop=False)
    seq = OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, enable_batched=False,
                           enable_rf_cudagraph=False, integrated_loop=True)
    # fact 0 affirmative, fact 1 negated -> yes/no by polarity
    aff = DERISK_FACTS[0]      # (dog, go, north) AFFIRM
    neg = DERISK_FACTS[1]      # (cat, run, river) NEGATE
    host.store(*aff, polarity="AFFIRM"); seq.store(*aff, polarity="AFFIRM")
    host.store(*neg, polarity="NEGATE"); seq.store(*neg, polarity="NEGATE")
    calls = {"n": 0}
    orig = seq._seq_block
    seq._seq_block = lambda a, x, _o=orig, _c=calls: (_c.__setitem__("n", _c["n"] + 1), _o(a, x))[1]
    assert seq.ask_yes_no(*aff) == host.ask_yes_no(*aff) == "yes", "affirmative full-SVO -> yes"
    assert seq.ask_yes_no(*neg) == host.ask_yes_no(*neg) == "no", "negated fact -> no"
    # a wrong patient on a stored (agent, action): the SVO does not match -> unknown (the selected block's patient != )
    assert seq.ask_yes_no(aff[0], aff[1], "river") == host.ask_yes_no(aff[0], aff[1], "river") == "unknown"
    # an unstored cue -> unknown (the moat, through the spiking decision)
    assert seq.ask_yes_no("zzz", aff[1], aff[2]) == host.ask_yes_no("zzz", aff[1], aff[2]) == "unknown"
    assert calls["n"] == 4, "ask_yes_no must route each call through _seq_block (the spiking decision)"


# ----------------------------------------------------------------------------------------------------------------
# audit #2 (enable_seq_vocab_shrink, DEFAULT ON): the integrated_loop=True sequencer built over the REDUCED cue vocab
# (V'_A distinct stored agents / V'_X distinct stored actions) is answer-IDENTICAL to the full-V sequencer on the whole
# who/what + abstain + cross battery, with a STRICTLY SMALLER fabric -- pins the ~34.6x production reduction's parity.
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [42, 43])
def test_seq_vocab_shrink_answer_identical_and_smaller(seed):
    """The reduced-vocab sequencer (default) selects the SAME block as the full-V sequencer on every present + abstain +
    cross cue, and its bridge has fewer neurons (V word-lines -> V'_A + V'_X). The shrink abstains a cue whose agent or
    action is not a stored filler BEFORE the fabric (== no block matches in full-V); a cross cue (both fillers stored but
    never together) runs the fabric and abstains there -- the moat is byte-identical either way."""
    facts = DERISK_FACTS[:K_SMALL]
    full = OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, enable_batched=False,
                            enable_rf_cudagraph=False, integrated_loop=True, enable_seq_vocab_shrink=False)
    shrink = OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=8, enable_batched=False,
                              enable_rf_cudagraph=False, integrated_loop=True, enable_seq_vocab_shrink=True)
    for (a, x, p) in facts:
        full.store(a, x, p); shrink.store(a, x, p)
    a0, x0 = facts[0][0], facts[0][1]
    battery = [(a, x) for (a, x, p) in facts] + [("zzz", x0), (a0, "fly"), (a0, facts[1][1])]
    for (a, x) in battery:
        assert shrink._seq_block(a, x) == full._seq_block(a, x), f"shrink != full-V block selection on cue {(a, x)}"
    full._ensure_sequencer(len(facts)); shrink._ensure_sequencer(len(facts))
    n_full = full._seq[0].core_config.num_neurons
    n_shrink = shrink._seq[0].core_config.num_neurons
    assert n_shrink < n_full, f"reduced sequencer ({n_shrink}) must be smaller than full-V ({n_full})"
    assert shrink._seq_mapA is not None and shrink._seq_mapX is not None, "the reduced cue maps must be populated"
