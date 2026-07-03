"""CI for EMERGE-68 -- extend the VALIDATED SPIKING A->W read-out to the FUNCTION-WORD (DET/FUNC) slots so the
EMERGE-frame render is 100% produced on spikes (ORDER via EMERGE-59/63, CONTENT words via EMERGE-67, FUNCTION words now).

CPU-safe where possible: the FUNCTION-word vocab structure, the UNIFIED dispatch wire (content->BRIDGE-A,
function->BRIDGE-F), the all-slot ground-truth + scoring, and the gate-first moat are validated with a token-spell
stand-in (no GPU). The genuinely-spiking A->W read-out (drive a concept pool -> decode from language_output spikes) is
GPU-only -> a skip-if-no-cupy smoke (mirror EMERGE-67's process-sticky skip-guard). The token-spell default path is
byte-identical to EMERGE-59..67.
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
    FRAMES, FRAME_NAMES, SUBJ, VERB, DET, FUNC, FrameSlotCQ, BrocaProducer, decision_from_emerge,
)
import research.runners._emerge68_function_word_spell_derisk as m68  # noqa: E402


def _has_gpu():
    """Non-destructive GPU probe: cupy + a device WITHOUT switching the active (numpy) backend."""
    try:
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class _CountingSpell:
    """A token-spell stand-in with a call counter (mirrors UnifiedNeuralSpell's spell contract, no GPU). Knows the full
    21-word producer vocab (16 content + 5 function) as 'spellable'."""
    def __init__(self):
        self.spell_calls = 0
        self.words = set(m68.m67._AW_CONTENT) | set(m68._FUNC_WORDS)

    def spell(self, word):
        self.spell_calls += 1
        return str(word)


# ---------------------------------------------------------------------------------------------------------------------
# CPU-safe: the FUNCTION-word vocab structure + the wire + moat + all-slot scoring with a token-spell stand-in.
# ---------------------------------------------------------------------------------------------------------------------
def test_five_function_words_are_the_frame_residual():
    """The function words are exactly the EMERGE-frame DET/FUNC residual {the,can,does,not} + the DET alternative `a`."""
    assert m68._FUNC_WORDS == ["the", "a", "can", "does", "not"]
    # every DET/FUNC payload the frames emit is in the function-word set
    frame_func = set()
    for fr in FRAME_NAMES:
        for (t, p) in FRAMES[fr]:
            if t in (DET, FUNC):
                frame_func.add(p)
    assert frame_func == {"the", "can", "does", "not"}
    assert frame_func <= set(m68._FUNC_WORDS)


def test_func_vocab16_rebinds_onto_16_pools_five_function_pools():
    """The function engine's 16-word vocab (5 function + 11 filler) rebinds onto the 16 validated concept pools; the 5
    function words occupy 5 DISTINCT pools of train_word_to_pool-supported kinds (motor + noun)."""
    assert len(m68._FUNC_VOCAB16) == 16 and len(set(m68._FUNC_VOCAB16)) == 16
    wp, pw, fp = m68._func_pool_assignment()
    assert len(wp) == 16 and len(pw) == 16
    assert set(fp) == set(m68._FUNC_WORDS)
    assert len(set(fp.values())) == 5           # 5 distinct function pools
    for pool in fp.values():
        assert pool.startswith("motor_") or pool.startswith("noun_pool_")  # kinds train_word_to_pool supports


def test_all_slot_ground_truth_content_and_function():
    """The all-slot ground-truth includes DET + FUNC surfaces (not just content): 'the owl can fly', 'the penguin does
    not fly', 3sg-inflected verb for F_INTR."""
    assert m68._all_slot_surfaces("F_MODAL", "owl", "fly") == ["the", "owl", "can", "fly"]
    assert m68._all_slot_surfaces("F_INTR", "penguin", "walks") == ["the", "penguin", "walks"]
    assert m68._all_slot_surfaces("F_NEGMOD", "penguin", "fly") == ["the", "penguin", "does", "not", "fly"]
    # a bare ability verb inflects to 3sg in the F_INTR frame slot
    assert m68._all_slot_surfaces("F_INTR", "owl", "fly")[-1] == "flies"


def test_wire_all_slots_use_spell_det_func_and_content():
    """The DET/FUNC branch of realize_slot routes through the SAME spell -- so a function-word-aware spell makes those
    slots spell too. Verify the produced frame (via a spell callback) has ALL words in order + the spell saw them."""
    cq = FrameSlotCQ(seed=42)
    cq.learn()
    sp = _CountingSpell()
    words = cq.emit("F_NEGMOD", "penguin", "fly", sp.spell)
    assert words == ["the", "penguin", "does", "not", "fly"]     # the penguin does not fly
    # every slot (det + subj + 2 func + verb = 5) went through spell
    assert sp.spell_calls == 5


def test_token_spell_no_regression_all_slots():
    """The token-spell ALL-slot + function-slot accuracy == ground-truth (the neural spell must reproduce this on a GO)."""
    cq = FrameSlotCQ(seed=42)
    cq.learn()
    facts = m68._facts(42, n=4)
    token_spell = lambda w: str(w)
    all_acc, func_acc, _ = m68._render_and_score_all(cq, token_spell, facts)
    assert all_acc == pytest.approx(1.0), f"token all-slot accuracy {all_acc} != 1.0"
    assert func_acc == pytest.approx(1.0), f"token function-slot accuracy {func_acc} != 1.0"


def test_gate_first_moat_spell_never_called_on_abstain():
    """The gate-first no-confab MOAT: on ABSTAIN the producer -- and hence the (unified) spell callback -- is NEVER
    invoked. Structural, CPU-safe (the same moat property EMERGE-67 asserts, with the 21-word unified spell contract)."""
    cq = FrameSlotCQ(seed=42)
    cq.learn()
    sp = _CountingSpell()
    prod = BrocaProducer(cq, spell=sp.spell)
    calls0 = sp.spell_calls
    for _ in range(3):
        r = prod.speak(decision_from_emerge("ABSTAIN"))
        assert r["produced"] is False and r["surface"] is None
    assert sp.spell_calls == calls0             # 0 spell calls on abstains
    assert prod.production_count == 0
    # positive control: an ANSWER DOES invoke the producer + spell (function words included)
    r = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    assert r["produced"] is True and sp.spell_calls > calls0 and prod.production_count == 1
    assert r["surface"] == "the owl can fly"    # ALL words (det+subj+func+verb) present


def test_facts_drawn_from_content_vocab():
    """The de-risk facts' content words are drawn from the A->W content vocab so every content slot is spike-spellable
    (the function slots are fixed frame furniture, spelled by BRIDGE-F)."""
    facts = m68._facts(42, n=8)
    for f in facts:
        assert f["subject"] in m68.m67._AW_SUBJECTS
        assert f["ability_verb"] in m68.m67._AW_ABILITY
        assert f["intr_verb"] in m68.m67._AW_INTR3SG


# ---------------------------------------------------------------------------------------------------------------------
# GPU-only smoke (skip-if-no-cupy AND cache-present): the FUNCTION-word A->W read-out spells a function word on spikes +
# the moat holds with the real spiking unified spell. Mirrors EMERGE-67's process-sticky skip-guard.
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("SIM_BACKEND", "numpy") != "cupy" or not _has_gpu(),
                    reason="A->W read-out needs the PROCESS to run SIM_BACKEND=cupy (the backend is process-sticky, so "
                           "it cannot flip mid-run when the CPU-safe tests already loaded numpy); run the CI with "
                           "SIM_BACKEND=cupy to exercise this. The full GPU A->W is validated by the 6-seed --derisk.")
@pytest.mark.skipif(not (m68._FUNC_CACHE_BRIDGE.exists() and m68.m67._CACHE_BRIDGE.exists()),
                    reason="A->W engine caches absent (run EMERGE-67 --train + EMERGE-68 --train to warm them)")
def test_gpu_function_word_decodes_from_spikes_and_moat():
    """GPU smoke: the FUNCTION-word A->W spell decodes a function word from language_output SPIKES (self-cos > 0,
    spikes > 0), and the gate-first moat holds (an abstain never invokes the unified spell)."""
    unified = m68.UnifiedNeuralSpell(load=True)
    assert unified._backend_gpu, "expected a GPU backend for the A->W engines"
    # decode a couple of function words: the read-out reads real spikes (spike-total > 0)
    for w in ("the", "can"):
        decoded, self_cos, top_cos, spikes = unified.func._decode(w)
        assert spikes > 0, f"BRIDGE-F produced no language_output spikes for {w!r} (not spiking)"
        assert self_cos > 0.0
    # moat: the unified spell is never called on an abstain
    cq = FrameSlotCQ(seed=42)
    cq.learn()
    prod = BrocaProducer(cq, spell=unified.spell)
    calls0 = unified.spell_calls
    for _ in range(2):
        prod.speak(decision_from_emerge("ABSTAIN"))
    assert unified.spell_calls == calls0 and prod.production_count == 0
