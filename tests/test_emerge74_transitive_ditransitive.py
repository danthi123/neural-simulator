"""CI for EMERGE-74 -- BROADEN the self-organized spiking-Broca producer to the CORE SVO argument-structure
constructions: TRANSITIVE ("the dog chases the cat") + DITRANSITIVE ("the dog gives the cat a bone"), routing the
project's already-GO argument-structure inventory (argstructure_composer.FRAME_LEXICON + the _bucketB corpus verb-frame
miner) through the EMERGE-72/73 signature-keyed ConstructionRegistry, corpus-driven.

CPU/numpy, offline. A small-stream smoke of the broadening: TRANSITIVE (5 slots -> fits N_SLOT_POOLS=6) is mined from the
corpus + rendered EXACT on spikes; DITRANSITIVE (7 slots > 6 pools) is MINED but capacity-gated at the render (the honest
named boundary); the input-destruction controls collapse (permuted-corpus -> no registry; cross-construction -> wrong;
no-corpus -> empty); the held-out-construction shared det+subj+verb backbone generalises; the provenance cross-checks
against argstructure/_bucketB; the gate-first moat holds (0 productions on abstains). A smaller n_extra keeps it fast.
NO sim/ edit; the EMERGE-72 constructions + defaults are preserved.
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

from research.runners._emerge59_spiking_broca_frame_slots_derisk import N_SLOT_POOLS  # noqa: E402
from research.runners._emerge74_transitive_ditransitive_derisk import (  # noqa: E402
    SVOConstructionRegistry, CONSTRUCTION_NAMES, SVO_CONSTRUCTION_NAMES, CONSTRUCTIONS, _FITS_POOLS,
    build_stream_svo, build_heldout_facts_svo, _render_registry, _heldout_construction,
    _emit_construction, _cross_construction, _provenance_check, decision, RegistryBrocaProducer,
    IOBJ, OBJ, _SVO_SUBJ, _SVO_OBJ, _THEMES, label_sentence_svo, _slot_signature_svo, _gt_signature,
)

_SEED = 42
_N_EXTRA = 3500     # smaller SVO stream for CI speed (the derisk uses 8000)
_EMERGE72 = ["F_MODAL", "F_INTR", "F_NEGMOD", "C_PPGOAL", "C_PPLOC"]


def _build(seed=_SEED, n_extra=_N_EXTRA):
    tokens = build_stream_svo(seed, n_extra=n_extra)
    reg = SVOConstructionRegistry(seed).build(tokens)
    return tokens, reg


def test_registry_mines_transitive_and_ditransitive():
    """The registry mines BOTH new core-SVO constructions from the corpus (transitive + ditransitive), plus the 5
    EMERGE-72 constructions -> 7 routed total."""
    _tokens, reg = _build()
    assert reg.n_registered() == 7, f"registered {reg.n_registered()}: {sorted(reg.registered)}"
    for name in _EMERGE72 + SVO_CONSTRUCTION_NAMES:
        assert name in reg.registered, f"{name} not mined/registered from the corpus"


def test_transitive_fits_ditransitive_over_capacity():
    """TRANSITIVE (5 slots) fits N_SLOT_POOLS=6; DITRANSITIVE (7 slots) exceeds it -> the honest capacity boundary.
    The ditransitive is MINED (registered) but NOT loaded into the spiking producer (capacity-gated)."""
    _tokens, reg = _build()
    assert len(CONSTRUCTIONS["C_TRANS"]) == 5 and _FITS_POOLS["C_TRANS"] is True
    assert len(CONSTRUCTIONS["C_DITRANS"]) == 7 and _FITS_POOLS["C_DITRANS"] is False
    assert 7 > N_SLOT_POOLS  # the wall is real
    fits = reg.registered_fits()
    over = reg.registered_over_capacity()
    assert "C_TRANS" in fits and "C_DITRANS" not in fits
    assert "C_DITRANS" in over          # mined but over the pool count


def test_six_constructions_render_exact_on_spikes():
    """(a) the producer renders 6 DISTINCT constructions EXACT on spikes (the 5 EMERGE-72 + TRANSITIVE), surface ==
    ground-truth template. The ditransitive is capacity-gated (not rendered). Moat 0 on abstains; an answer is produced."""
    _tokens, reg = _build()
    facts = build_heldout_facts_svo(_SEED, n=6)
    per, moat_calls, answer_produced = _render_registry(reg, facts)
    fits = reg.registered_fits()
    n_exact = sum(1 for n in fits if per[n]["exact"] == pytest.approx(1.0))
    assert n_exact >= 6, f"only {n_exact} rendered exact: {[(n, per[n]['exact']) for n in fits]}"
    assert moat_calls == 0
    assert answer_produced is True


def test_transitive_renders_correct_svo_surface():
    """The transitive construction renders 'the [subj] [verb]s the [obj]' on spikes (DET SUBJ VERB:3sg DET OBJ)."""
    _tokens, reg = _build()
    cq = reg.render_cq()
    words = _emit_construction(cq, "C_TRANS", {"svo_subject": "wolf", "trans_verb": "chase", "obj": "ball"})
    assert words == ["the", "wolf", "chases", "the", "ball"], words


def test_ditransitive_labels_with_two_post_verbal_content():
    """The ditransitive is labelled with a SECOND post-verbal CONTENT (IOBJ) slot + its own determiner, and its mined
    signature MATCHES the ground-truth ditransitive template (det subj verb:3sg det iobj det obj)."""
    tokens = build_stream_svo(_SEED, n_extra=_N_EXTRA)
    from research.runners._emerge62_discover_function_words_derisk import compute_stats, discover_closed_class
    from research.runners._emerge63_corpus_taught_slot_order_derisk import split_sentences
    words, freq, cover, _content = compute_stats(tokens)
    closed, _p, _fp, _cp = discover_closed_class(words, freq, cover)
    ditrans = None
    for s in split_sentences(tokens):
        if len(s) == 7 and s[0] == "the" and s[3] == "the" and s[5] == "a":
            ditrans = s
            break
    assert ditrans is not None, "no ditransitive sentence in the stream"
    slots = label_sentence_svo(ditrans, closed)
    assert slots is not None, f"ditransitive not labelled: {ditrans}"
    assert any(st == IOBJ for (st, p, i) in slots), f"no IOBJ slot: {slots}"
    assert any(st == OBJ for (st, p, i) in slots), f"no OBJ slot: {slots}"
    assert _slot_signature_svo(slots) == _gt_signature("C_DITRANS")


def test_permuted_corpus_collapses_the_registry():
    """(b1) PERMUTED-CORPUS (shuffle each exemplar's word order before mining) collapses the registry -> 0 registered
    (the broadening is genuinely corpus-order-derived, not host-smuggled)."""
    tokens = build_stream_svo(_SEED, n_extra=_N_EXTRA)
    srng = np.random.default_rng(1234)
    reg_p = SVOConstructionRegistry(_SEED).build(tokens, shuffle_within=True, shuffle_rng=srng)
    assert reg_p.n_registered() == 0, f"permuted-corpus still registered {sorted(reg_p.registered)}"


def test_cross_construction_is_wrong():
    """(b2) CROSS-CONSTRUCTION: rendering construction A's fact through a DIFFERENT construction B's mined structure is
    WRONG (construction-specific; Dominey-Hinaut form-specificity)."""
    _tokens, reg = _build()
    facts = build_heldout_facts_svo(_SEED, n=4)
    cross = _cross_construction(reg, facts)
    assert cross < 0.30, f"cross-construction render {cross} not collapsed (constructions not form-specific)"


def test_no_corpus_yields_empty_registry():
    """(b4) no corpus -> no signatures -> no registry (empty)."""
    reg_empty = SVOConstructionRegistry(_SEED).build([])
    assert reg_empty.n_registered() == 0


def test_heldout_construction_shared_backbone_generalizes():
    """(b3) hold ONE construction out of the mining corpus; its SHARED det+subj+verb backbone is recovered from the
    OTHERS (generalisation, not memorisation)."""
    tokens, reg = _build()
    closed = reg.discovered_function_words
    for held in CONSTRUCTION_NAMES:
        bb, _dist = _heldout_construction(tokens, closed, held)
        assert bb == pytest.approx(1.0), f"held-out {held} shared backbone {bb:.3f} not recovered"


def test_heldout_ditransitive_distinctive_iobj_is_the_named_residual():
    """The ditransitive's held-out DISTINCTIVE part (the IOBJ -- a SECOND post-verbal content noun) is attested ONLY by
    the ditransitive itself, so it is NOT recovered from the others -- the honest, precisely-named residual (reported,
    not gated)."""
    tokens, reg = _build()
    closed = reg.discovered_function_words
    _bb, distinctive = _heldout_construction(tokens, closed, "C_DITRANS")
    assert distinctive is False, "the ditransitive's IOBJ should NOT be recoverable from other constructions"


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


def test_provenance_matches_argstructure_and_bucketB():
    """The mined transitive/ditransitive role inventories MATCH argstructure_composer.FRAME_LEXICON (transitive
    `_default` agent-action-patient; ditransitive give agent-action-THEME-RECIPIENT) + the _bucketB mined frames
    (chase->transitive; give->ditransitive) -- the two GO inventories this de-risk unifies."""
    prov = _provenance_check()
    assert prov["provenance_consistent"] is True, prov
    assert prov["emerge74_transitive_n_content"] == 2      # SUBJ, OBJ
    assert prov["emerge74_ditransitive_n_content"] == 3    # SUBJ, IOBJ, OBJ
    if prov.get("argstructure_available"):
        assert prov["argstructure_transitive_is_2content"] is True
        assert prov["argstructure_ditransitive_is_3content"] is True
    if prov.get("bucketB_available"):
        assert prov["bucketB_chase_is_transitive"] is True
        assert prov["bucketB_give_is_ditransitive"] is True


def test_svo_vocab_is_base_disjoint_and_content():
    """The SVO content-noun pools are base-disjoint (so no SVO noun concentrates into the closed class) and no SVO noun
    is mislabelled closed by the discovery -- the structural precondition for clean transitive/ditransitive mining."""
    _tokens, reg = _build()
    svo_nouns = set(_SVO_SUBJ) | set(_SVO_OBJ) | set(_THEMES)
    misclassified = svo_nouns & reg.discovered_function_words
    assert not misclassified, f"SVO nouns mislabelled closed: {sorted(misclassified)}"
    # the SVO pools are non-trivially large (so selectional restriction keeps them context-narrow)
    assert len(_SVO_SUBJ) >= 20 and len(_SVO_OBJ) >= 20 and len(_THEMES) >= 20


def test_iobj_slot_type_and_construction_templates_present():
    """The IOBJ slot type + the two core-SVO construction templates are wired into the inventory (structural check)."""
    assert IOBJ == "iobj"
    assert "C_TRANS" in CONSTRUCTIONS and "C_DITRANS" in CONSTRUCTIONS
    assert SVO_CONSTRUCTION_NAMES == ["C_TRANS", "C_DITRANS"]
    # C_TRANS is DET SUBJ VERB DET OBJ (5 slots); C_DITRANS is DET SUBJ VERB DET IOBJ DET OBJ (7 slots)
    assert not any(st == IOBJ for (st, p, i) in CONSTRUCTIONS["C_TRANS"])
    assert any(st == IOBJ for (st, p, i) in CONSTRUCTIONS["C_DITRANS"])
