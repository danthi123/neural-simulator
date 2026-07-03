"""CI guard for EMERGE-59 RUNG A -- SIMULATE BROCA: render EMERGE's fixed reply frames FLUENTLY on the SPIKING
substrate via a frame-and-slot grammatical encoder (function-word + inflection slots), behind the gate-first
no-confab MOAT. CPU/numpy, offline. Tests:
  1. the three canonical EMERGE frames render CORRECTLY on spikes (owl->fly, penguin->walks, penguin->does-not-fly).
  2. the LOAD-BEARING MOAT: a gate=ABSTAIN produces NOTHING -- the producer is NEVER invoked (production_count 0 on
     abstains), and an ANSWER decision DOES invoke it (the counter is meaningful).
  3. the anti-cheat controls COLLAPSE: PERMUTED-slot-order (0 exact), NO-LEARNING (chance), FUNCTION-WORD-ABLATION
     (function words dropped -> agrammatic), CROSS-FRAME (a frame does not render another frame's surface).
  4. a fast 3-seed de-risk holds the GO invariants (main beats every control; moat 0-productions).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pytest

from research.runners._emerge59_spiking_broca_frame_slots_derisk import (
    FrameSlotCQ, BrocaProducer, decision_from_emerge, build_heldout_facts,
    _expected_words, _frame_scores, _cross_frame_order, _derisk_one, FRAME_NAMES, FUNC, FRAMES,
)


@pytest.fixture(scope="module")
def cq():
    """Train the frame-slot competitive-queuing producer once (learn the frame slot orders from the templates)."""
    c = FrameSlotCQ(seed=42)
    c.learn()
    return c


def test_three_canonical_frames_render_on_spikes(cq):
    """The three EMERGE reply frames render the CORRECT fluent surface on the spiking substrate."""
    prod = BrocaProducer(cq)
    # affirm-modal (inherited ability): "the owl can fly"
    r1 = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    assert r1["produced"] and r1["surface"] == "the owl can fly", r1["surface"]
    # intransitive exception (cancellation): "the penguin walks" (already-3sg, not double-inflected)
    r2 = prod.speak(decision_from_emerge("ANSWER", subject="penguin", verb="walks", polarity="negate"))
    assert r2["produced"] and r2["surface"] == "the penguin walks", r2["surface"]
    # negated modal (deny the class ability): "the penguin does not fly" (function words + bare inflection)
    r3 = prod.speak(decision_from_emerge("ANSWER", subject="penguin", verb="fly", negated_modal=True))
    assert r3["produced"] and r3["surface"] == "the penguin does not fly", r3["surface"]


def test_gate_first_moat_producer_never_invoked_on_abstain(cq):
    """THE LOAD-BEARING PROPERTY: a gate=ABSTAIN produces NOTHING and NEVER invokes the producer (count unchanged);
    a gate=ANSWER DOES invoke it (so the counter is meaningful)."""
    prod = BrocaProducer(cq)
    before = prod.production_count
    for _ in range(5):
        r = prod.speak(decision_from_emerge("ABSTAIN"))
        assert r["produced"] is False and r["surface"] is None
    assert prod.production_count == before, "producer INVOKED on an abstain (MOAT BREACHED)"
    # positive control: an ANSWER DOES invoke the producer
    r = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    assert r["produced"] is True
    assert prod.production_count == before + 1


def test_inflection_correct(cq):
    """Bare inside 'can' / 'does not' (fly stays fly); already-3sg intransitive not double-inflected (walks stays
    walks); a base intransitive verb inflected to 3sg in F_INTR."""
    prod = BrocaProducer(cq)
    # bare inside the modal frame
    assert prod.speak(decision_from_emerge("ANSWER", "owl", "fly", polarity="affirm"))["surface"].endswith("can fly")
    # already-3sg exception verb: never 'walkses'
    assert prod.speak(decision_from_emerge("ANSWER", "penguin", "walks", polarity="negate"))["surface"] == \
        "the penguin walks"
    # a BASE intransitive verb in F_INTR gets 3sg-inflected (run -> runs)
    r = prod.speak(decision_from_emerge("ANSWER", "robin", "run", polarity="negate"))
    assert r["surface"] == "the robin runs", r["surface"]


def test_permuted_and_nolearning_controls_collapse():
    """PERMUTED-slot-order teaches a wrong order (0 exact-slot match); NO-LEARNING gives chance order. Both must be
    far below the main arm."""
    facts = build_heldout_facts(42)
    cq_main = FrameSlotCQ(seed=42); cq_main.learn()
    main = _frame_scores(cq_main, facts)
    cq_perm = FrameSlotCQ(seed=42, permute_order=True); cq_perm.learn()
    perm = _frame_scores(cq_perm, facts)
    cq_nol = FrameSlotCQ(seed=42, no_learning=True); cq_nol.learn()
    nol = _frame_scores(cq_nol, facts)
    main_exact = float(np.mean([main[f]["exact"] for f in FRAME_NAMES]))
    perm_exact = float(np.mean([perm[f]["exact"] for f in FRAME_NAMES]))
    main_order = float(np.mean([main[f]["order"] for f in FRAME_NAMES]))
    nol_order = float(np.mean([nol[f]["order"] for f in FRAME_NAMES]))
    assert main_exact >= 0.9, main_exact
    assert perm_exact <= 0.1, f"permuted-order control did not collapse exact-slot match: {perm_exact}"
    assert main_order >= nol_order + 0.2, f"main order {main_order} vs no-learning {nol_order} (control not collapsed)"


def test_function_word_ablation_load_bearing():
    """Removing the learned FUNCTION-WORD slots yields agrammatic output (the function words are learned-slot-supplied,
    not host-inserted)."""
    facts = build_heldout_facts(42)
    cq_main = FrameSlotCQ(seed=42); cq_main.learn()
    cq_abl = FrameSlotCQ(seed=42, ablate_func=True); cq_abl.learn()
    spell = lambda w: str(w)
    func_frames = [f for f in FRAME_NAMES if any(s[0] == FUNC for s in FRAMES[f])]
    for frame in func_frames:
        needed = [p for (t, p) in FRAMES[frame] if t == FUNC]
        fact = facts[0]
        verb = fact["ability_verb"] if frame != "F_INTR" else fact["intr_verb"]
        w_main = cq_main.emit(frame, fact["subject"], verb, spell)
        w_abl = cq_abl.emit(frame, fact["subject"], verb, spell)
        assert all(fw in w_main for fw in needed), f"{frame} main missing function words {needed}: {w_main}"
        assert not all(fw in w_abl for fw in needed), f"{frame} ablated still has all function words: {w_abl}"


def test_cross_frame_control_collapses():
    """A frame's own true-word rendering beats rendering another frame's surface for the same content
    (frame-conditioned -- the same content is ordered/worded DIFFERENTLY per frame)."""
    facts = build_heldout_facts(42)
    cq = FrameSlotCQ(seed=42); cq.learn()
    main = _frame_scores(cq, facts)
    main_word = float(np.mean([main[f]["word"] for f in FRAME_NAMES]))
    cross = _cross_frame_order(cq, facts)
    assert cross is not None, "cross-frame control must fire"
    assert main_word >= cross + 0.2, f"own-word {main_word} does not beat cross-frame {cross}"


def test_derisk_go_invariants_3seed():
    """A fast 3-seed de-risk holds the GO invariants: main beats every control with margin; moat 0-productions."""
    per = [_derisk_one(s) for s in (42, 43, 44)]
    main_order = float(np.mean([d["main_order"] for d in per]))
    main_exact = float(np.mean([d["main_exact"] for d in per]))
    main_word = float(np.mean([d["main_word"] for d in per]))
    perm_exact = float(np.mean([d["perm_exact"] for d in per]))
    nol_order = float(np.mean([d["nolearn_order"] for d in per]))
    cross = float(np.mean([d["cross_order"] for d in per if d["cross_order"] is not None]))
    moat_calls = sum(d["moat_calls_on_abstain"] for d in per)
    assert main_order >= 0.9 and main_exact >= 0.9
    assert main_exact >= perm_exact + 0.2
    assert main_order >= nol_order + 0.2
    assert main_word >= cross + 0.2
    assert all(d["main_grammatical"] >= 0.99 and d["ablate_grammatical"] <= 0.01 for d in per)
    assert moat_calls == 0, f"MOAT BREACHED: {moat_calls} producer calls on abstains"
    assert all(d["answer_produced"] for d in per)
