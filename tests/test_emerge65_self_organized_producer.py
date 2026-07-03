"""CI for EMERGE-65 (THE CAPSTONE) -- COMPOSE the three self-organized pieces (S2 function words + S1a slot inventory +
S1b slot order) into ONE end-to-end pipeline that discovers the WHOLE spiking-Broca producer structure FROM THE CORPUS
ALONE and renders the EMERGE answers ON SPIKES.

CPU/numpy, offline. A small-stream smoke of the composition: the SelfOrganizedProducer built from the corpus stream
discovers the function words (S2), mines the slot inventory (S1a), learns the slot order (S1b), assembles the
FRAMES-equivalent, renders the three canonical EMERGE surfaces EXACT on spikes, and matches the host FRAMES; the
COMPOSED permuted-corpus + no-corpus controls collapse the WHOLE pipeline; the held-out-frame shared det+subj+verb
backbone generalizes; the gate-first moat holds (0 productions on abstains). A smaller n_sentences keeps it fast.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FRAME_NAMES, decision_from_emerge, build_heldout_facts,
)
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, FRAME_FUNCTION_WORDS,
)
from research.runners._emerge63_corpus_taught_slot_order_derisk import split_sentences  # noqa: E402
from research.runners._emerge65_self_organized_producer_derisk import (  # noqa: E402
    SelfOrganizedProducer, assembled_structure_match, end_to_end_render,
    heldout_frame_generalization, permuted_corpus_collapse,
)

_SEED = 42
_N = 6000     # smaller stream for CI speed (the derisk uses 20000)


def _build(seed=_SEED, n=_N):
    """Build the fully-self-organized producer from a (smaller-for-CI) corpus token stream."""
    tokens = build_stream(seed, n_sentences=n)
    prod = SelfOrganizedProducer(seed).build_from_corpus(tokens)
    return tokens, prod


def test_s2_function_words_discovered_from_corpus():
    """(a) S2: the closed-class function words self-organize from the corpus stream (frequency + coverage); ALL the
    frame function words (the/can/does/not) are discovered -- NO host FUNCTION_WORDS list as input."""
    _tokens, prod = _build()
    assert prod.discovered_function_words, "no function words discovered"
    for fw in FRAME_FUNCTION_WORDS:
        assert fw in prod.discovered_function_words, f"frame function word {fw!r} not discovered"


def test_assembled_structure_matches_host_frames():
    """(b) the assembled structure (mined inventory S1a + learned order S1b) MATCHES the host FRAMES per frame -- slot
    set + function-word fillers + order -- with the host FRAMES dict used only as validation ground-truth."""
    _tokens, prod = _build()
    per_frame, struct_match, inv_acc = assembled_structure_match(prod)
    assert inv_acc == pytest.approx(1.0), f"inventory accuracy {inv_acc:.3f} != 1.0"
    assert struct_match == pytest.approx(1.0), f"assembled-structure match {struct_match:.3f} != 1.0"
    for fr in FRAME_NAMES:
        assert per_frame[fr]["inventory_match"], f"{fr} inventory not matched"
        assert per_frame[fr]["order_match"], f"{fr} order not matched"


def test_end_to_end_render_exact_on_spikes_and_moat():
    """(a)+(d) the assembled-from-corpus structure renders the held-out facts EXACT on spikes (right slots + order +
    function words + inflection), and the gate-first no-confab moat holds (0 producer invocations on abstains)."""
    _tokens, prod = _build()
    facts = build_heldout_facts(_SEED, n=4)
    per_frame, moat_calls, answer_produced = end_to_end_render(prod, facts)
    for fr in FRAME_NAMES:
        assert per_frame[fr]["found"], f"{fr} not assembled from corpus"
        assert per_frame[fr]["exact"] == pytest.approx(1.0), f"{fr} end-to-end render not exact"
    assert moat_calls == 0
    assert answer_produced is True


def test_canonical_emerge_frames_render_and_abstain():
    """End-to-end transcript: the fully-self-organized producer renders the three canonical EMERGE surfaces on spikes,
    and abstains (producer NOT invoked) on the moat probe."""
    _tokens, prod = _build()
    p = prod.producer()
    r1 = p.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    r2 = p.speak(decision_from_emerge("ANSWER", subject="penguin", verb="walks", polarity="negate"))
    r3 = p.speak(decision_from_emerge("ANSWER", subject="penguin", verb="fly", negated_modal=True))
    r4 = p.speak(decision_from_emerge("ABSTAIN"))
    assert r1["surface"] == "the owl can fly"
    assert r2["surface"] == "the penguin walks"
    assert r3["surface"] == "the penguin does not fly"
    assert r4["produced"] is False and r4["surface"] is None


def test_composed_permuted_corpus_collapses_whole_pipeline():
    """(c) THE COMPOSED ANTI-CHEAT: building the producer from the SHUFFLED corpus (each exemplar's word order scrambled
    at BOTH the inventory-mining AND order-learning stages) collapses end-to-end render AND assembled-structure match --
    the entire structure is CORPUS-DERIVED, not host-smuggled."""
    tokens, prod = _build()
    per_frame, _mc, _ap = end_to_end_render(prod, build_heldout_facts(_SEED, n=4))
    main_render = float(np.mean([per_frame[f]["exact"] for f in FRAME_NAMES]))
    perm_render, perm_match = permuted_corpus_collapse(tokens, _SEED, n_shuffles=3)
    assert main_render == pytest.approx(1.0)
    assert main_render >= perm_render + 0.30, f"permuted-corpus did not collapse render ({main_render:.3f} vs {perm_render:.3f})"
    assert perm_match <= 0.70, f"permuted-corpus structure-match did not collapse ({perm_match:.3f})"


def test_no_corpus_control_empty():
    """(c) NO-CORPUS: an empty stream -> no discovered function words / no mined inventory -> nothing assembled."""
    prod = SelfOrganizedProducer(_SEED).build_from_corpus([])
    assert len(prod.mined_slots) == 0
    facts = build_heldout_facts(_SEED, n=2)
    per_frame, _mc, _ap = end_to_end_render(prod, facts)
    assert float(np.mean([per_frame[f]["exact"] for f in FRAME_NAMES])) == pytest.approx(0.0)


def test_heldout_frame_generalizes_on_shared_structure():
    """(c) HELD-OUT-FRAME: the shared det+subj+verb backbone + shared type-level order generalize to a fully-held-out
    frame (F_MODAL & F_INTR) from the OTHER two frames -- the gated shared-structure claim (the distinctive
    function-word / inflection slots are the honestly-named residual, reported not gated)."""
    _tokens, prod = _build()
    sents = split_sentences(build_stream(_SEED, n_sentences=_N))
    _result, shared_backbone, shared_order = heldout_frame_generalization(prod, sents, _SEED)
    assert shared_backbone == pytest.approx(1.0), f"held-out shared backbone {shared_backbone:.3f} != 1.0"
    assert shared_order == pytest.approx(1.0), f"held-out shared order {shared_order:.3f} != 1.0"


def test_heldout_negmod_distinctive_residual_is_named():
    """The CARRIED-FORWARD residual (named, NOT hidden): F_INTR's distinctive 3sg inflection is NOT recoverable when
    F_INTR is held out (only F_INTR attests 3sg) -- exactly EMERGE-63/64's shared-vs-distinctive split. Reported."""
    _tokens, prod = _build()
    sents = split_sentences(build_stream(_SEED, n_sentences=_N))
    result, _bb, _ord = heldout_frame_generalization(prod, sents, _SEED)
    assert result["F_INTR"]["distinctive_inflection_recovered"] is False, \
        "held-out F_INTR 3sg inflection should NOT be recoverable from the other two frames (the named residual)"
