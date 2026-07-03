"""CI for EMERGE-63 -- LEARN the spiking-Broca producer's per-frame slot ORDER from the corpus's WORD-ORDER statistics
(pairwise role precedence / bigram order; Dominey-Hinaut: grammar = the statistics of element order, no explicit rules;
catalog G.12 Broca; usage-based construction grammar), instead of the host template order-teacher.

CPU/numpy, offline. Small-stream smoke of: the corpus-taught order reproduces the template ground-truth exactly and
renders on spikes; the shuffled-corpus + no-corpus input-destruction controls collapse; held-out-frame generalizes on
the shared type-level precedence; the does<not held-out residual is honestly present; the gate-first moat holds.
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

from research.runners.song_g1_core import score_order  # noqa: E402
from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FRAME_NAMES, BrocaProducer, decision_from_emerge, build_heldout_facts, _expected_words,
)
from research.runners._emerge63_corpus_taught_slot_order_derisk import (  # noqa: E402
    build_stream, split_sentences, _classify, _template_role_order,
    corpus_precedence, order_from_precedence, learn_corpus_order, order_heldout_frame,
    CorpusOrderFrameSlotCQ, _spiking_render_scores,
)


def _by_frame(seed, n=6000):
    sents = split_sentences(build_stream(seed, n_sentences=n))
    return {fr: [s for s in sents if _classify(s) == fr] for fr in FRAME_NAMES}


def test_corpus_stream_has_all_three_frames():
    """The corpus stream contains exemplars of ALL three EMERGE frames (so per-frame precedence can be accumulated)."""
    by_frame = _by_frame(42, n=2000)
    for fr in FRAME_NAMES:
        assert len(by_frame[fr]) >= 30, f"too few {fr} exemplars: {len(by_frame[fr])}"


def test_corpus_precedence_reproduces_template_order():
    """The pairwise role precedence over corpus exemplars reproduces each frame's template ground-truth order EXACTLY --
    including the negated-modal's does<not (directly attested in F_NEGMOD's own exemplars), with an HONEST random
    tie-break (not the template order)."""
    by_frame = _by_frame(42)
    tie_rng = np.random.default_rng(0)
    for fr in FRAME_NAMES:
        prec, roles, n_used = corpus_precedence(by_frame[fr], fr)
        assert n_used >= 20
        order = order_from_precedence(prec, roles, tie_rng=tie_rng)
        assert order == _template_role_order(fr), f"{fr}: corpus {order} != template {_template_role_order(fr)}"


def test_shuffled_corpus_control_collapses():
    """SHUFFLED-CORPUS (scramble each example sentence's word order) destroys the precedence statistics -> the learned
    order is wrong/chance, far below the main order (the load-bearing input-destruction anti-cheat)."""
    by_frame = _by_frame(42)
    # main order (avg over frames)
    main = np.mean([
        score_order(order_from_precedence(*corpus_precedence(by_frame[fr], fr)[:2],
                                          tie_rng=np.random.default_rng(1)), _template_role_order(fr))
        for fr in FRAME_NAMES
    ])
    # shuffled order (avg over frames + several shuffle/tie seeds)
    shuf_accs = []
    for k in range(6):
        srng = np.random.default_rng(100 + k)
        trng = np.random.default_rng(200 + k)
        accs = []
        for fr in FRAME_NAMES:
            prec, roles, _ = corpus_precedence(by_frame[fr], fr, shuffle_within=True, shuffle_rng=srng)
            accs.append(score_order(order_from_precedence(prec, roles, tie_rng=trng), _template_role_order(fr)))
        shuf_accs.append(np.mean(accs))
    shuffled = float(np.mean(shuf_accs))
    assert main == pytest.approx(1.0)
    assert main >= shuffled + 0.30, f"shuffled-corpus did not collapse (main {main:.3f} vs shuffled {shuffled:.3f})"


def test_no_corpus_control_is_chance():
    """NO-CORPUS (no example sentences) -> no precedence -> chance order (far below the main order)."""
    empty = {fr: [] for fr in FRAME_NAMES}
    accs = []
    for k in range(6):
        trng = np.random.default_rng(300 + k)
        order, _ = learn_corpus_order(empty, tie_rng=trng)
        accs.append(np.mean([score_order(order[fr], _template_role_order(fr)) for fr in FRAME_NAMES]))
    nocorpus = float(np.mean(accs))
    assert nocorpus <= 0.60, f"no-corpus order not at chance ({nocorpus:.3f})"


def test_heldout_frame_shared_order_generalizes():
    """HELD-OUT-FRAME: the SHARED type-level order (det<subj<func<verb) learned from the OTHER two frames recovers a
    fully-held-out single-/no-function-word frame's order (F_MODAL, F_INTR) -- generalization, not memorization."""
    by_frame = _by_frame(42)
    for held in ("F_MODAL", "F_INTR"):
        accs = []
        for k in range(6):
            trng = np.random.default_rng(400 + k)
            order = order_heldout_frame(by_frame, held, trng)
            accs.append(score_order(order, _template_role_order(held)))
        assert np.mean(accs) == pytest.approx(1.0), f"held-out {held} did not generalize"


def test_heldout_negmod_internal_order_is_the_named_residual():
    """The does-vs-not INTERNAL order of a HELD-OUT F_NEGMOD is NOT learnable from the OTHER two frames alone (only
    F_NEGMOD attests two adjacent function words). With an honest tie-break it sits BELOW 1.0 -- the precisely-named
    residual (NOT a wall), while the shared roles still order correctly."""
    by_frame = _by_frame(42)
    accs = []
    for k in range(8):
        trng = np.random.default_rng(500 + k)
        order = order_heldout_frame(by_frame, "F_NEGMOD", trng)
        accs.append(score_order(order, _template_role_order("F_NEGMOD")))
    mean_acc = float(np.mean(accs))
    # honest residual: strictly below perfect (the does<not internal order is not learnable from the other two frames),
    # yet well above pure chance (the shared det<subj<...<verb roles DO order correctly).
    assert mean_acc < 1.0, "held-out F_NEGMOD should NOT be perfect (does<not is unlearnable from the other frames)"
    assert mean_acc >= 0.5, "the shared roles should still order the held-out F_NEGMOD above chance"


def test_producer_renders_corpus_taught_order_on_spikes_and_moat_holds():
    """The corpus-taught order feeds the EMERGE-59 spiking producer (EMERGE-61 wash-out): held-out facts render EXACT on
    spikes in the corpus-learned order, and the gate-first no-confab moat holds (0 producer invocations on abstains)."""
    by_frame = _by_frame(42)
    corpus_order, _ = learn_corpus_order(by_frame, tie_rng=np.random.default_rng(42 * 131 + 3))
    facts = build_heldout_facts(42, n=4)
    per_frame, moat_calls, answer_produced = _spiking_render_scores(corpus_order, 42, facts)
    for fr in FRAME_NAMES:
        assert per_frame[fr]["order"] == pytest.approx(1.0), f"{fr} order not 1.0"
        assert per_frame[fr]["exact"] == pytest.approx(1.0), f"{fr} exact-surface not 1.0"
    assert moat_calls == 0
    assert answer_produced is True


def test_corpus_order_producer_renders_canonical_frames():
    """End-to-end: the CorpusOrderFrameSlotCQ producer renders the three canonical EMERGE surfaces on spikes in the
    corpus-taught order, and abstains (no production) on the moat."""
    by_frame = _by_frame(42)
    corpus_order, _ = learn_corpus_order(by_frame, tie_rng=np.random.default_rng(42 * 131 + 3))
    cq = CorpusOrderFrameSlotCQ(seed=42, corpus_order=corpus_order)
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


def test_corpus_order_none_preserves_template_behavior():
    """CorpusOrderFrameSlotCQ with corpus_order=None is byte-behavior-identical to the template-order producer (the base
    ResetFrameSlotCQ): it renders the canonical frames in template order (the additive default-preserving property)."""
    cq = CorpusOrderFrameSlotCQ(seed=42, corpus_order=None)
    cq.learn()
    prod = BrocaProducer(cq)
    r = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    assert r["surface"] == "the owl can fly"
