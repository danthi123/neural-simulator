"""CI for EMERGE-64 -- MINE the spiking-Broca producer's per-construction slot INVENTORY (WHICH ordered role-slots a
construction licenses) from the corpus, using the EMERGE-62 DISCOVERED function words + position, instead of the host
FRAMES dict (S1a of the self-organizing-grammatical-structure research gate).

CPU/numpy, offline. Small-stream smoke of: the mined inventory reproduces the ground-truth frames exactly + renders on
spikes; the permuted-mining + no-corpus input-destruction controls collapse; the held-out-frame role-type backbone
generalizes; the F_INTR 3sg inflection is the honestly-named held-out residual; the gate-first moat holds; and the
mined_slots=None additive default preserves template behavior.
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
    FRAME_NAMES, BrocaProducer, decision_from_emerge, build_heldout_facts,
)
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, compute_stats, discover_closed_class,
)
from research.runners._emerge63_corpus_taught_slot_order_derisk import split_sentences  # noqa: E402
from research.runners._emerge64_mine_slot_inventory_derisk import (  # noqa: E402
    label_sentence, mine_inventory, inventory_accuracy, match_inventory_to_frames,
    _mined_to_emerge59_slots, _spiking_render_from_mined, _frame_signature, _slot_signature,
    heldout_frame_backbone_recovered, heldout_frame_inflection_recovered,
    MinedInventoryFrameSlotCQ,
)


def _mine(seed, n=6000):
    tokens = build_stream(seed, n_sentences=n)
    sents = split_sentences(tokens)
    words, freq, cover, _ = compute_stats(tokens)
    closed, *_ = discover_closed_class(words, freq, cover)
    inventory, _ = mine_inventory(sents, closed)
    return sents, closed, inventory


def test_discovered_closed_class_labels_roles():
    """label_sentence types each token from the discovered closed class + position: the/can are function words, the
    content nouns/verbs are SUBJ/VERB, and the ORDER is preserved. NO host FRAMES dict is consulted."""
    _, closed, _ = _mine(42)
    slots = label_sentence(["the", "owl", "can", "fly"], closed)
    assert slots == [("det", "the", None), ("subj", None, None), ("func", "can", None), ("verb", None, "bare")]
    slots3 = label_sentence(["the", "penguin", "walks"], closed)
    assert slots3 == [("det", "the", None), ("subj", None, None), ("verb", None, "3sg")]


def test_mined_inventory_matches_ground_truth_frames():
    """The mined per-construction inventory recovers all three EMERGE frames' ordered typed-slot lists EXACTLY (the S1a
    claim: WHICH slots each construction licenses, mined from the corpus, == the host FRAMES dict)."""
    _, _, inventory = _mine(42)
    acc, m = inventory_accuracy(inventory)
    assert acc == pytest.approx(1.0)
    for fr in FRAME_NAMES:
        assert m[fr]["found"] and m[fr]["slots_match"], f"{fr} inventory not recovered"


def test_permuted_mining_collapses():
    """PERMUTED-MINING (shuffle each exemplar's word order before labelling) destroys the construction statistics: the
    dominant ordering scatters below the dominance threshold -> the multi-slot frames are NOT confidently mined -> the
    mined-inventory accuracy collapses far below the main (the load-bearing `_bucketB`-style input-destruction control)."""
    sents, closed, inventory = _mine(42)
    main_acc, _ = inventory_accuracy(inventory)
    accs = []
    for k in range(4):
        srng = np.random.default_rng(1000 + k)
        inv_shuf, _ = mine_inventory(sents, closed, shuffle_within=True, shuffle_rng=srng)
        accs.append(inventory_accuracy(inv_shuf)[0])
    perm_acc = float(np.mean(accs))
    assert main_acc == pytest.approx(1.0)
    assert main_acc >= perm_acc + 0.30, f"permuted-mining did not collapse (main {main_acc:.3f} vs {perm_acc:.3f})"


def test_no_corpus_control_empty():
    """NO-CORPUS (no exemplars) -> no signatures -> empty inventory -> accuracy 0 (no data, no inventory)."""
    _, closed, _ = _mine(42)
    inv_empty, _ = mine_inventory([], closed)
    assert len(inv_empty) == 0
    assert inventory_accuracy(inv_empty)[0] == pytest.approx(0.0)


def test_heldout_frame_role_type_backbone_generalizes():
    """HELD-OUT-FRAME: withholding a frame's exemplars, the mine over the OTHER two recovers the held-out frame's SHARED
    det+subj+verb ROLE-TYPE backbone (the claim: det<subj<verb generalizes across constructions)."""
    sents, closed, _ = _mine(42)
    for held in FRAME_NAMES:
        held_sig = _frame_signature(held)
        train = [s for s in sents
                 if (lambda sl: sl is not None and _slot_signature(sl) != held_sig)(label_sentence(s, closed))]
        train_inv, _ = mine_inventory(train, closed)
        assert heldout_frame_backbone_recovered(train_inv, held) == pytest.approx(1.0), \
            f"held-out {held} role-type backbone did not generalize"


def test_heldout_intr_inflection_is_named_residual():
    """The NAMED RESIDUAL: F_INTR's distinctive 3sg VERB inflection is NOT recoverable when F_INTR is held out (only
    F_INTR attests 3sg; the other two frames are VERB:bare) -- precisely-named, NOT a wall (like EMERGE-63's does<not)."""
    sents, closed, _ = _mine(42)
    held = "F_INTR"
    held_sig = _frame_signature(held)
    train = [s for s in sents
             if (lambda sl: sl is not None and _slot_signature(sl) != held_sig)(label_sentence(s, closed))]
    train_inv, _ = mine_inventory(train, closed)
    assert heldout_frame_inflection_recovered(train_inv, held) is False, \
        "held-out F_INTR 3sg inflection should NOT be recoverable from the other two frames (the named residual)"


def test_producer_renders_mined_inventory_on_spikes_and_moat_holds():
    """The mined inventory feeds the EMERGE-59/63 spiking producer: held-out facts render EXACT on spikes from the MINED
    (not host) slot lists, and the gate-first no-confab moat holds (0 producer invocations on abstains)."""
    _, _, inventory = _mine(42)
    _acc, m = inventory_accuracy(inventory)
    facts = build_heldout_facts(42, n=4)
    per_frame, moat_calls, answer_produced = _spiking_render_from_mined(m, 42, facts)
    for fr in FRAME_NAMES:
        assert per_frame[fr]["found"], f"{fr} not mined"
        assert per_frame[fr]["exact"] == pytest.approx(1.0), f"{fr} exact-surface not 1.0"
    assert moat_calls == 0
    assert answer_produced is True


def test_mined_producer_renders_canonical_frames():
    """End-to-end: the MinedInventoryFrameSlotCQ producer renders the three canonical EMERGE surfaces on spikes from the
    mined inventory, and abstains (no production) on the moat."""
    _, _, inventory = _mine(42)
    _acc, m = inventory_accuracy(inventory)
    mined_slots = {fr: _mined_to_emerge59_slots([tuple(x) for x in m[fr]["mined_slots"]])
                   for fr in FRAME_NAMES if m[fr]["found"]}
    cq = MinedInventoryFrameSlotCQ(seed=42, mined_slots=mined_slots)
    cq.learn()
    prod = BrocaProducer(cq)
    r1 = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    r2 = prod.speak(decision_from_emerge("ANSWER", subject="penguin", verb="walks", polarity="negate"))
    r3 = prod.speak(decision_from_emerge("ANSWER", subject="penguin", verb="fly", negated_modal=True))
    r4 = prod.speak(decision_from_emerge("ABSTAIN"))
    assert r1["surface"] == "the owl can fly"
    assert r2["surface"] == "the penguin walks"
    assert r3["surface"] == "the penguin does not fly"
    assert r4["produced"] is False and r4["surface"] is None


def test_mined_slots_none_preserves_template_behavior():
    """MinedInventoryFrameSlotCQ with mined_slots=None is byte-behavior-identical to the template-order producer (the
    additive default-preserving property; EMERGE-59/61/63 untouched): it renders the canonical frames in template order."""
    cq = MinedInventoryFrameSlotCQ(seed=42, mined_slots=None)
    cq.learn()
    prod = BrocaProducer(cq)
    r = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    assert r["surface"] == "the owl can fly"
