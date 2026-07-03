"""CI for EMERGE-72 -- BROADEN the self-organized spiking producer beyond the 3 EMERGE frames via a SIGNATURE-KEYED
CONSTRUCTION REGISTRY that renders the constructions the producer already MINES but DISCARDS (transitive-motion PP-goal
+ PP-location added to the 3 EMERGE frames).

CPU/numpy, offline. A small-stream smoke of the broadening: the ConstructionRegistry built from the corpus stream mines
>= 5 constructions, routes each to a stable construction id, and the spiking producer renders each EXACT on spikes; the
input-destruction controls collapse (permuted-corpus / cross-construction / no-corpus); the held-out-construction shared
det+subj+verb backbone generalizes; the gate-first moat holds (0 productions on abstains). A smaller n_sentences keeps it
fast. NO sim/ edit; the 3-frame path is preserved.
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

from research.runners._emerge62_discover_function_words_derisk import build_stream  # noqa: E402
from research.runners._emerge72_construction_registry_derisk import (  # noqa: E402
    ConstructionRegistry, CONSTRUCTION_NAMES, CONSTRUCTIONS, OBJ, FUNC, VERB,
    build_heldout_facts_ext, _render_registry, _cross_construction, _heldout_construction,
    _expected_surface, _verb_for, decision, RegistryBrocaProducer,
    label_sentence_ext, _slot_signature_ext,
)

_SEED = 42
_N = 6000     # smaller stream for CI speed (the derisk uses 20000)
_EXPECTED = ["F_MODAL", "F_INTR", "F_NEGMOD", "C_PPGOAL", "C_PPLOC"]


def _build(seed=_SEED, n=_N):
    tokens = build_stream(seed, n_sentences=n)
    reg = ConstructionRegistry(seed).build(tokens)
    return tokens, reg


def test_registry_mines_at_least_5_constructions():
    """The signature-keyed registry MINES >= 5 constructions from the corpus (the 3 EMERGE frames + the two new
    transitive-motion PP constructions) -- the mine is construction-agnostic, the registry stops discarding the new ones."""
    _tokens, reg = _build()
    assert reg.n_registered() >= 5, f"only {reg.n_registered()} constructions registered"
    for name in _EXPECTED:
        assert name in reg.registered, f"{name} not mined/registered from the corpus"


def test_mined_slots_match_ground_truth_constructions():
    """Each registered construction's MINED slot list matches its ground-truth template (validation ground-truth only) --
    including the NEW PP constructions' post-verbal OBJECT slot + PP preposition scaffold."""
    _tokens, reg = _build()
    for name in _EXPECTED:
        mined = reg.registered[name]
        # ground-truth in EMERGE-59 (slot_type, payload) form
        gt = []
        for (st, p, inf) in CONSTRUCTIONS[name]:
            if st == VERB:
                gt.append((VERB, inf))
            elif st in ("subj", OBJ):
                gt.append((st, None))
            else:
                gt.append((st, p))
        assert list(mined) == gt, f"{name} mined slots {mined} != ground truth {gt}"


def test_at_least_5_constructions_render_exact_on_spikes():
    """(a) the producer renders >= 5 DISTINCT constructions EXACT on spikes (surface == ground-truth template), incl. the
    two new transitive-motion constructions 'the owl flies to/on the <obj>'. The EMERGE-61 wash-out keeps each render an
    independent motor plan (position-independent order)."""
    _tokens, reg = _build()
    facts = build_heldout_facts_ext(_SEED, n=6)
    per, moat_calls, answer_produced = _render_registry(reg, facts)
    n_exact = sum(1 for n in _EXPECTED if per[n]["found"] and per[n]["exact"] == pytest.approx(1.0))
    assert n_exact >= 5, f"only {n_exact} constructions rendered exact: {[(n, per[n]['exact']) for n in _EXPECTED]}"
    assert moat_calls == 0
    assert answer_produced is True


def test_new_pp_constructions_render_argument_after_verb():
    """The NEW transitive-motion constructions render the OBJECT filler AFTER the verb (the biggest expressivity jump --
    arguments after the verb), spelled by the A->W read-out from the gated decision."""
    _tokens, reg = _build()
    cq = reg.render_cq()
    prod = RegistryBrocaProducer(cq)
    r = prod.speak(decision("ANSWER", "C_PPGOAL", subject="owl", verb="fly", obj="pond"))
    assert r["produced"]
    assert r["words"] == ["the", "owl", "flies", "to", "the", "pond"], r["words"]
    r2 = prod.speak(decision("ANSWER", "C_PPLOC", subject="owl", verb="fly", obj="rock"))
    assert r2["words"] == ["the", "owl", "flies", "on", "the", "rock"], r2["words"]


def test_permuted_corpus_collapses_the_registry():
    """(b1) input-destruction: shuffling each exemplar's word order before mining COLLAPSES the registry -> the
    constructions are not confidently mined -> nothing renders (the broadening is corpus-derived, not host-smuggled)."""
    tokens, _reg = _build()
    srng = np.random.default_rng(_SEED * 977 + 13)
    reg_p = ConstructionRegistry(_SEED).build(tokens, shuffle_within=True, shuffle_rng=srng)
    facts = build_heldout_facts_ext(_SEED, n=4)
    per_p, _mc, _ap = _render_registry(reg_p, facts)
    names_p = [n for n in CONSTRUCTION_NAMES if n in reg_p.registered]
    render_p = float(np.mean([per_p[n]["exact"] for n in names_p])) if names_p else 0.0
    assert reg_p.n_registered() == 0, f"permuted corpus still registered {reg_p.n_registered()} constructions"
    assert render_p == pytest.approx(0.0)


def test_cross_construction_control_collapses():
    """(b2) rendering construction A's fact through a DIFFERENT construction B's mined structure is WRONG (construction-
    specific form, Dominey-Hinaut). The cross-construction exact-match must be ~0."""
    _tokens, reg = _build()
    facts = build_heldout_facts_ext(_SEED, n=4)
    cross = _cross_construction(reg, facts)
    assert cross <= 0.05, f"cross-construction exact-match {cross:.3f} too high (not form-specific)"


def test_no_corpus_yields_empty_registry():
    """(b4) no corpus -> no signatures -> no registry (empty)."""
    reg_empty = ConstructionRegistry(_SEED).build([])
    assert reg_empty.n_registered() == 0


def test_heldout_construction_shared_backbone_generalizes():
    """(b3) hold ONE construction out of the mining corpus; its SHARED det+subj+verb backbone is recovered from the
    OTHERS (the gated claim). The distinctive PP scaffold is the named residual (reported)."""
    tokens, reg = _build()
    closed = reg.discovered_function_words
    for held in _EXPECTED:
        bb, _scaffold = _heldout_construction(tokens, closed, held)
        assert bb == pytest.approx(1.0), f"held-out {held} shared backbone {bb:.3f} not recovered"


def test_gate_first_moat_never_invokes_producer_on_abstain():
    """(c) the gate-first no-confab moat: an ABSTAIN never invokes the producer (0 productions); an ANSWER does."""
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


def test_label_ext_skips_adjective_ambiguous_constructions():
    """The bounded label extension admits a post-verbal OBJECT but SKIPS the adjective-based copular/existential
    constructions (the honest boundary: an adjective is statistically ambiguous with the closed class in this corpus, so
    it is not forced into a content role). The predicative-adjective 'the owl is big' does not mine as a clean
    construction here -- it is either skipped or mis-labelled, NOT rendered as a named construction."""
    tokens, reg = _build()
    closed = reg.discovered_function_words
    # predicative-adjective + existential are NOT among the registered named constructions (the named boundary).
    assert "C_PREDADJ" not in reg.registered
    assert "C_EXIST" not in reg.registered
    # and the corpus's adjective-based sentence does not label as any of our 5 clean construction signatures
    gt_sigs = {_slot_signature_ext(CONSTRUCTIONS[n]) for n in _EXPECTED}
    lab = label_sentence_ext(["the", "owl", "is", "big"], closed)
    if lab is not None:
        assert _slot_signature_ext(lab) not in gt_sigs
