"""CI for EMERGE-73 -- CLOSE the adjective boundary EMERGE-72 named by reclassifying the corpus's adjectives OPEN via
their ATTRIBUTIVE PRE-NOMINAL POSITION (the DET _ NOUN slot), so the self-organized spiking-Broca producer ADMITS the
attributive + predicative adjective constructions and BROADENS to >= 7 corpus-mined, router-selected constructions.

CPU/numpy, offline. A small-stream smoke of the broadening: the position cue reclassifies the Goldilocks-mislabelled
adjectives OPEN (F1 1.0, zero true function words promoted), the registry mines >= 7 constructions (the 5 EMERGE-72 + the
2 adjective ones), the spiking producer renders each EXACT; the input-destruction controls collapse (POSITION-SHUFFLE ->
no adjective constructions; FREQUENCY-ONLY -> the EMERGE-72 5; no-corpus empty); the held-out-construction shared
det+subj+verb backbone generalises; the gate-first moat holds (0 productions on abstains). A smaller n_sentences keeps it
fast. NO sim/ edit; the EMERGE-72 constructions + defaults are preserved.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._emerge62_discover_function_words_derisk import build_stream, _ADJS  # noqa: E402
from research.runners._emerge73_adjective_position_cue_derisk import (  # noqa: E402
    AdjConstructionRegistry, CONSTRUCTION_NAMES, ADJ_CONSTRUCTION_NAMES, CONSTRUCTIONS,
    build_heldout_facts_adj, _render_registry, _adjective_reclassification, _heldout_construction,
    _emit_construction, _expected_surface, _verb_for, decision, RegistryBrocaProducer,
    reclassify_adjectives, compute_attributive_stats, TP_ATTRIB, ADJ,
)
from research.runners._emerge62b_function_words_position_cue_derisk import sentences_from_controlled  # noqa: E402

_SEED = 42
_N = 6000     # smaller stream for CI speed (the derisk uses 20000)
_EMERGE72 = ["F_MODAL", "F_INTR", "F_NEGMOD", "C_PPGOAL", "C_PPLOC"]
_ADJ_SET = set(_ADJS)


def _build(seed=_SEED, n=_N, **kw):
    tokens = build_stream(seed, n_sentences=n)
    reg = AdjConstructionRegistry(seed, **kw).build(tokens=tokens)
    return tokens, reg


def test_registry_mines_at_least_7_constructions():
    """The registry mines >= 7 constructions from the corpus (the 5 EMERGE-72 + the 2 adjective ones) once the
    adjectives are reclassified OPEN by their attributive position."""
    _tokens, reg = _build()
    assert reg.n_registered() >= 7, f"only {reg.n_registered()} constructions registered: {sorted(reg.registered)}"
    for name in _EMERGE72 + ADJ_CONSTRUCTION_NAMES:
        assert name in reg.registered, f"{name} not mined/registered from the corpus"


def test_adjectives_reclassified_open_by_position_cue():
    """(b) the position cue reclassifies the Goldilocks-mislabelled adjectives OPEN with perfect precision/recall and
    ZERO true function words promoted (the ASYMMETRIC/SAFE reclassification)."""
    _tokens, reg = _build()
    rc = _adjective_reclassification(reg)
    assert rc["R"] == pytest.approx(1.0), f"not all mislabelled adjectives reclassified OPEN: {rc}"
    assert rc["F1"] == pytest.approx(1.0), f"reclassification F1 {rc['F1']} < 1.0: {rc}"
    assert rc["promoted_true_function_words"] == [], f"a true function word was promoted: {rc}"
    # every reclassified word IS a ground-truth adjective
    assert set(reg.discovered_adjectives).issubset(_ADJ_SET)
    # and it removed them from the closed class
    assert not (reg.discovered_adjectives & reg.corrected_closed)


def test_seven_constructions_render_exact_on_spikes():
    """(a) the producer renders >= 7 DISTINCT constructions EXACT on spikes (surface == ground-truth template), incl. the
    attributive 'the big owl can fly' + predicative 'the owl is big'. Moat 0 on abstains; an answer is produced."""
    _tokens, reg = _build()
    facts = build_heldout_facts_adj(_SEED, n=6)
    per, moat_calls, answer_produced = _render_registry(reg, facts)
    n_exact = sum(1 for n in CONSTRUCTION_NAMES if per[n]["found"] and per[n]["exact"] == pytest.approx(1.0))
    assert n_exact >= 7, f"only {n_exact} rendered exact: {[(n, per[n]['exact']) for n in CONSTRUCTION_NAMES]}"
    assert moat_calls == 0
    assert answer_produced is True


def test_adjective_constructions_render_with_adjective_slot():
    """The attributive construction renders the adjective BETWEEN the determiner and the head noun; the predicative
    construction renders the adjective AFTER the copula (the ADJ slot spelled by the A->W read-out from the fact)."""
    _tokens, reg = _build()
    cq = reg.render_cq()
    attrib = _emit_construction(cq, "C_ATTRIB", {"subject": "owl", "ability_verb": "fly", "adj": "big"})
    assert attrib == ["the", "big", "owl", "can", "fly"], attrib
    pred = _emit_construction(cq, "C_PRED", {"subject": "owl", "adj": "grey"})
    assert pred == ["the", "owl", "is", "grey"], pred


def test_position_shuffle_collapses_the_adjective_constructions():
    """(c1) POSITION-SHUFFLE (scramble word positions before the attributive stat) DESTROYS the cue -> 0 adjectives
    reclassified -> 0 adjective constructions mined (the position cue is LOAD-BEARING, not spurious)."""
    _tokens, reg_s = _build(shuffle_positions=True)
    assert len(reg_s.discovered_adjectives) == 0, f"shuffle still reclassified {reg_s.discovered_adjectives}"
    adj_registered = [n for n in ADJ_CONSTRUCTION_NAMES if n in reg_s.registered]
    assert adj_registered == [], f"position-shuffle still mined adjective constructions: {adj_registered}"


def test_frequency_only_reproduces_emerge72_state():
    """(c2) FREQUENCY-ONLY (the EMERGE-62 2-cue baseline, no position cue) -> the adjectives stay CLOSED -> 0 adjective
    constructions mined = the EMERGE-72 5-construction state (proving the position cue is what ADDS the adjectives)."""
    _tokens, reg_fo = _build(frequency_only=True)
    assert len(reg_fo.discovered_adjectives) == 0
    adj_registered = [n for n in ADJ_CONSTRUCTION_NAMES if n in reg_fo.registered]
    assert adj_registered == [], f"frequency-only mined adjective constructions: {adj_registered}"
    # the EMERGE-72 five still mine
    for name in _EMERGE72:
        assert name in reg_fo.registered, f"frequency-only lost the EMERGE-72 construction {name}"


def test_no_corpus_yields_empty_registry():
    """(c3) no corpus -> no statistics -> no reclassification -> no registry (empty)."""
    reg_empty = AdjConstructionRegistry(_SEED).build(tokens=[])
    assert reg_empty.n_registered() == 0
    assert len(reg_empty.discovered_adjectives) == 0


def test_heldout_construction_shared_backbone_generalizes():
    """(c4) hold ONE construction out of the mining corpus; its SHARED det+subj+verb backbone is recovered from the
    OTHERS (generalisation, not memorisation)."""
    tokens, reg = _build()
    for held in CONSTRUCTION_NAMES:
        bb = _heldout_construction(_SEED, reg.corrected_closed, reg.discovered_adjectives, held)
        assert bb == pytest.approx(1.0), f"held-out {held} shared backbone {bb:.3f} not recovered"


def test_gate_first_moat_never_invokes_producer_on_abstain():
    """(d) the gate-first no-confab moat: an ABSTAIN never invokes the producer (0 productions); an ANSWER does."""
    _tokens, reg = _build()
    cq = reg.render_cq()
    prod = RegistryBrocaProducer(cq)
    for _ in range(5):
        r = prod.speak(decision("ABSTAIN"))
        assert r["produced"] is False
    assert prod.production_count == 0
    r = prod.speak(decision("ANSWER", "F_MODAL", subject="owl", verb="fly"))
    assert r["produced"] is True
    assert prod.production_count == 1


def test_attributive_threshold_separates_adjectives_from_determiners():
    """The attributive-position statistic separates the Goldilocks-CLOSED adjectives (rate ~0.7) from the true closed
    class (rate <= 0.36) with a clear margin around the pre-registered TP_ATTRIB=0.50 -- the mechanism's core claim."""
    from research.runners._emerge62_discover_function_words_derisk import (
        compute_stats, discover_closed_class, GROUND_TRUTH_CLOSED,
    )
    tokens = build_stream(_SEED, n_sentences=_N)
    words, freq, cover, _content = compute_stats(tokens)
    closed, _p, _fp, _cp = discover_closed_class(words, freq, cover)
    sents = sentences_from_controlled(_SEED) if False else None
    # use the same sentence-segmentation the runner uses on a passed-in stream
    from research.runners._emerge63_corpus_taught_slot_order_derisk import split_sentences
    aw, asc, _occ = compute_attributive_stats(split_sentences(tokens), closed)
    smap = {w: s for w, s in zip(aw, asc)}
    adj_closed_rates = [smap.get(w, 0.0) for w in (_ADJ_SET & closed)]
    func_closed_rates = [smap.get(w, 0.0) for w in (GROUND_TRUTH_CLOSED & closed)]
    assert adj_closed_rates, "no adjectives were mislabelled closed in this stream (nothing to separate)"
    assert min(adj_closed_rates) >= TP_ATTRIB, f"an adjective fell below the threshold: {min(adj_closed_rates)}"
    assert max(func_closed_rates) < TP_ATTRIB, f"a function word exceeded the threshold: {max(func_closed_rates)}"


def test_adj_slot_type_and_construction_templates_present():
    """The ADJ slot type + the two adjective construction templates are wired into the inventory (structural check)."""
    assert ADJ == "adj"
    assert "C_ATTRIB" in CONSTRUCTIONS and "C_PRED" in CONSTRUCTIONS
    # C_ATTRIB is DET adj SUBJ FUNC VERB (5 slots); C_PRED is DET SUBJ FUNC adj (4 slots)
    assert any(st == ADJ for (st, p, i) in CONSTRUCTIONS["C_ATTRIB"])
    assert any(st == ADJ for (st, p, i) in CONSTRUCTIONS["C_PRED"])
    assert ADJ_CONSTRUCTION_NAMES == ["C_ATTRIB", "C_PRED"]
