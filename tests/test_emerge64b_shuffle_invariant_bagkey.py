"""CI for EMERGE-64b -- the SHUFFLE-INVARIANT bag-keying that strengthens the EMERGE-64 slot-inventory mining so its
permuted-corpus anti-cheat GENUINELY collapses ALL constructions (including the shortest, F_INTR), closing the residual
the EMERGE-62..66 adversarial audit surfaced (`2026-07-03-emerge65-self-organized-producer-GO.md`, "Audit remediation").

CPU/numpy, offline. Small-stream smoke of:
  * the additive `shuffle_invariant_bag` flag on EMERGE-64's `mine_inventory` is DEFAULT-OFF == byte-identical (main +
    permuted mining match the default keying exactly);
  * the DEFECT: under the DEFAULT keying the ~1/3 of F_INTR shuffle orderings that keep `the` at onset re-label it det:
    -> the exact F_INTR bag -> F_INTR reconstructed (perm floor = F_INTR alone), while the invariant keying merges all
    orderings into ONE bag (`_bag_key_invariant`);
  * the MAIN (unshuffled) mining with the shuffle-invariant keying STILL recovers all 3 frames exactly + renders on
    spikes (unregressed) -- the multiset distinguishes the frames by closed-token counts + verb inflection;
  * the PERMUTED-CORPUS control now collapses the shortest F_INTR too (perm render/accuracy -> ~0.0, materially below the
    default keying's 0.333 floor);
  * the held-out backbone still generalizes + the gate-first moat holds under the invariant keying;
  * EMERGE-65's SelfOrganizedProducer opt-in (shuffle_invariant_bag=True) makes its composed permuted-corpus control
    collapse to ~0.0 while the default (False) stays byte-identical.
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
    FRAME_NAMES, build_heldout_facts, DET, FUNC, VERB, SUBJ,
)
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, compute_stats, discover_closed_class,
)
from research.runners._emerge63_corpus_taught_slot_order_derisk import split_sentences  # noqa: E402
from research.runners._emerge64_mine_slot_inventory_derisk import (  # noqa: E402
    label_sentence, mine_inventory, inventory_accuracy, match_inventory_to_frames,
    _spiking_render_from_mined, _frame_signature, _slot_signature, _bag_key, _bag_key_invariant,
    heldout_frame_backbone_recovered,
)


def _prep(seed=42, n=6000):
    tokens = build_stream(seed, n_sentences=n)
    sents = split_sentences(tokens)
    words, freq, cover, _ = compute_stats(tokens)
    closed, *_ = discover_closed_class(words, freq, cover)
    return tokens, sents, closed


def _perm_acc(sents, closed, invariant, n=4, base=1000):
    accs = []
    for k in range(n):
        srng = np.random.default_rng(base + k)
        inv, _ = mine_inventory(sents, closed, shuffle_within=True, shuffle_rng=srng,
                                shuffle_invariant_bag=invariant)
        accs.append(inventory_accuracy(inv)[0])
    return float(np.mean(accs))


# ---------------------------------------------------------------------------------------------------------------------
# The DEFECT + the invariant key mechanism (no mining needed -- pure keying logic).
# ---------------------------------------------------------------------------------------------------------------------
def test_default_keying_scatters_fintr_orderings_into_two_bags():
    """The DEFECT: for the F_INTR token multiset {the, subj, verb+s}, the DEFAULT `_bag_key(sig)` puts the onset-`the`
    ordering (det:the) and the non-onset-`the` orderings (func:the) into DIFFERENT bags -- so the 'wrong' orderings never
    dilute the F_INTR (det) bag under shuffle."""
    _, _, closed = _prep()
    bags = set()
    for perm in (["the", "penguin", "walks"], ["penguin", "the", "walks"], ["penguin", "walks", "the"]):
        slots = label_sentence(perm, closed)
        assert slots is not None
        bags.add(_bag_key(_slot_signature(slots)))
    assert len(bags) == 2, "default keying should scatter F_INTR orderings into 2 bags (det: vs func:)"


def test_invariant_keying_merges_fintr_orderings_into_one_bag():
    """THE FIX: `_bag_key_invariant` decides closed-vs-open by the discovered SET identity (position-independent), so
    ALL labellable orderings of the F_INTR token multiset share ONE bag -> under shuffle they dilute the dominant
    fraction (the load-bearing collapse)."""
    _, _, closed = _prep()
    bags = set()
    for perm in (["the", "penguin", "walks"], ["penguin", "the", "walks"], ["penguin", "walks", "the"]):
        slots = label_sentence(perm, closed)
        assert slots is not None
        bags.add(_bag_key_invariant(slots))
    assert len(bags) == 1, "invariant keying should merge F_INTR orderings into 1 bag"


def test_invariant_key_labels_are_position_independent():
    """A DET slot and a FUNC slot with the same lexeme map to the SAME `closed:` token (no det/func POSITION label); a
    VERB keeps its inflection (surface morphology, position-independent); a SUBJ -> `open`."""
    det_slots = [(DET, "the", None), (SUBJ, None, None), (VERB, None, "3sg")]
    func_slots = [(SUBJ, None, None), (FUNC, "the", None), (VERB, None, "3sg")]
    assert _bag_key_invariant(det_slots) == _bag_key_invariant(func_slots)
    assert _bag_key_invariant(det_slots) == ("closed:the", "open", "verb:3sg")


# ---------------------------------------------------------------------------------------------------------------------
# MAIN mining unregressed under the invariant keying (multiset still distinguishes the frames).
# ---------------------------------------------------------------------------------------------------------------------
def test_main_mining_unregressed_under_invariant_keying():
    """MAIN (unshuffled) mining with the shuffle-invariant keying STILL recovers all 3 EMERGE frames' ordered slot lists
    EXACTLY -- the closed-token multiset + verb inflection separate F_MODAL {the,can}+bare / F_INTR {the}+3sg / F_NEGMOD
    {the,does,not}+bare."""
    _, sents, closed = _prep()
    inv_inv, _ = mine_inventory(sents, closed, shuffle_invariant_bag=True)
    acc, m = inventory_accuracy(inv_inv)
    assert acc == pytest.approx(1.0)
    for fr in FRAME_NAMES:
        assert m[fr]["found"] and m[fr]["slots_match"], f"{fr} not recovered under invariant keying"


def test_main_render_unregressed_under_invariant_keying():
    """The producer renders all 3 EMERGE surfaces EXACT on spikes from the invariant-keyed mined inventory (render 1.0)."""
    _, sents, closed = _prep()
    inv_inv, _ = mine_inventory(sents, closed, shuffle_invariant_bag=True)
    _acc, m = inventory_accuracy(inv_inv)
    facts = build_heldout_facts(42, n=4)
    per_frame, moat_calls, answer_produced = _spiking_render_from_mined(m, 42, facts)
    for fr in FRAME_NAMES:
        assert per_frame[fr]["found"] and per_frame[fr]["exact"] == pytest.approx(1.0), f"{fr} render not exact"
    assert moat_calls == 0 and answer_produced is True


# ---------------------------------------------------------------------------------------------------------------------
# The strengthened permuted-corpus control: F_INTR collapses too (before 0.333 -> after ~0.0).
# ---------------------------------------------------------------------------------------------------------------------
def test_default_permuted_floor_is_fintr_alone():
    """Under the DEFAULT keying, the permuted-corpus inventory-accuracy is ~0.333 (F_INTR alone reconstructed) -- the
    audit-named residual this follow-on closes."""
    _, sents, closed = _prep()
    perm_def = _perm_acc(sents, closed, invariant=False)
    assert perm_def == pytest.approx(1.0 / 3.0, abs=0.05), f"default perm floor should be ~0.333, got {perm_def:.3f}"


def test_invariant_permuted_control_collapses_fintr_too():
    """Under the SHUFFLE-INVARIANT keying, the permuted-corpus inventory-accuracy collapses to ~0.0 (F_INTR collapses
    too) -- materially below the default keying's 0.333 floor. The 'permuted-corpus collapses the whole pipeline' claim
    is now literally true."""
    _, sents, closed = _prep()
    perm_def = _perm_acc(sents, closed, invariant=False)
    perm_inv = _perm_acc(sents, closed, invariant=True)
    assert perm_inv <= 0.05, f"invariant perm should collapse to ~0.0, got {perm_inv:.3f}"
    assert perm_inv < perm_def - 0.20, "invariant keying must materially lower perm accuracy vs default"


def test_invariant_permuted_render_collapses_on_spikes():
    """The spiking RENDER (not just inventory-accuracy) collapses to ~0.0 under the invariant permuted keying."""
    _, sents, closed = _prep()
    facts = build_heldout_facts(42, n=4)
    renders = []
    for k in range(3):
        srng = np.random.default_rng(1000 + k)
        inv, _ = mine_inventory(sents, closed, shuffle_within=True, shuffle_rng=srng, shuffle_invariant_bag=True)
        _acc, m = inventory_accuracy(inv)
        per_frame, _mc, _ap = _spiking_render_from_mined(m, 42, facts)
        renders.append(float(np.mean([per_frame[f]["exact"] for f in FRAME_NAMES])))
    assert float(np.mean(renders)) <= 0.05, f"invariant permuted render should collapse, got {np.mean(renders):.3f}"


# ---------------------------------------------------------------------------------------------------------------------
# Held-out generalization + gate-first moat preserved under the invariant keying.
# ---------------------------------------------------------------------------------------------------------------------
def test_heldout_backbone_generalizes_under_invariant_keying():
    """The held-out-frame SHARED det+subj+verb backbone still generalizes (1.0) under the invariant keying."""
    _, sents, closed = _prep()
    for held in FRAME_NAMES:
        held_sig = _frame_signature(held)
        train = [s for s in sents
                 if (lambda sl: sl is not None and _slot_signature(sl) != held_sig)(label_sentence(s, closed))]
        train_inv, _ = mine_inventory(train, closed, shuffle_invariant_bag=True)
        assert heldout_frame_backbone_recovered(train_inv, held) == pytest.approx(1.0), \
            f"held-out {held} backbone did not generalize under invariant keying"


# ---------------------------------------------------------------------------------------------------------------------
# Additive default-off: the flag=False path is byte-identical to EMERGE-64.
# ---------------------------------------------------------------------------------------------------------------------
def test_default_off_is_byte_identical():
    """`shuffle_invariant_bag=False` (the default) == EMERGE-64's committed behaviour: the mined inventory + the
    permuted floor are identical to not passing the flag at all."""
    _, sents, closed = _prep()
    inv_a, _ = mine_inventory(sents, closed)                                   # committed call (no flag)
    inv_b, _ = mine_inventory(sents, closed, shuffle_invariant_bag=False)      # explicit default
    assert inv_a == inv_b, "shuffle_invariant_bag=False must be byte-identical to not passing the flag"


# ---------------------------------------------------------------------------------------------------------------------
# EMERGE-65 opt-in: the composed permuted-corpus control collapses to ~0.0; default stays byte-identical.
# ---------------------------------------------------------------------------------------------------------------------
def test_emerge65_opt_in_collapses_composed_control():
    """EMERGE-65's SelfOrganizedProducer/ permuted_corpus_collapse gain the additive default-off `shuffle_invariant_bag`
    option: opting in drops the composed permuted-corpus render to ~0.0 (F_INTR collapses too), while the default (False)
    reproduces the committed 0.333 F_INTR-alone floor."""
    from research.runners._emerge65_self_organized_producer_derisk import (
        SelfOrganizedProducer, permuted_corpus_collapse, end_to_end_render, assembled_structure_match,
    )
    tokens = build_stream(42, n_sentences=6000)
    facts = build_heldout_facts(42, n=4)

    # default (committed) -- F_INTR floor
    pr_def, pm_def = permuted_corpus_collapse(tokens, 42, n_shuffles=3, shuffle_invariant_bag=False)
    assert pr_def == pytest.approx(1.0 / 3.0, abs=0.05)

    # opt-in -- whole pipeline collapses
    pr_inv, pm_inv = permuted_corpus_collapse(tokens, 42, n_shuffles=3, shuffle_invariant_bag=True)
    assert pr_inv <= 0.05 and pm_inv <= 0.05

    # MAIN unregressed under the opt-in
    prod = SelfOrganizedProducer(42, shuffle_invariant_bag=True).build_from_corpus(tokens)
    per_frame, moat_calls, _ = end_to_end_render(prod, facts)
    main_render = float(np.mean([per_frame[f]["exact"] for f in FRAME_NAMES]))
    _pf, struct_match, inv_acc = assembled_structure_match(prod)
    assert main_render == pytest.approx(1.0) and struct_match == pytest.approx(1.0) and inv_acc == pytest.approx(1.0)
    assert moat_calls == 0


def test_emerge65_default_producer_byte_identical():
    """EMERGE-65's SelfOrganizedProducer default (shuffle_invariant_bag=False) mines the SAME inventory as before -- the
    committed default is byte-preserved."""
    from research.runners._emerge65_self_organized_producer_derisk import SelfOrganizedProducer
    tokens = build_stream(42, n_sentences=6000)
    prod_default = SelfOrganizedProducer(42).build_from_corpus(tokens)
    prod_explicit = SelfOrganizedProducer(42, shuffle_invariant_bag=False).build_from_corpus(tokens)
    assert prod_default.mined_inventory == prod_explicit.mined_inventory
