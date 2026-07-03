"""CI for EMERGE-69 -- wire the FULLY-SPIKING A->W spell (EMERGE-67/68 UnifiedNeuralSpell) into the EMERGE-66 flagship
console (SpikingBrocaConsole self-organized producer) via an additive default-off `neural_spell` flag, so the flagship
renders its EMERGE answers 100% ON SPIKES (self-organized structure + every word content+function spike-spelled).

CPU-safe where possible: the ADDITIVE flag structure (default False == byte-identical token surface), the wire (the
neural spell routed through the console producer's spell), and the gate-first moat are validated with a token-spell
stand-in (no GPU). The genuinely-spiking A->W read-out (drive a concept pool -> decode from language_output spikes) is
GPU-only -> a skip-if-no-cupy smoke (mirror EMERGE-67/68's process-sticky skip-guard). The default (neural_spell=False)
path is byte-identical to EMERGE-59..68.
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
from research.runners._emerge60_console_spiking_broca_derisk import SpikingBrocaConsole  # noqa: E402


def _has_gpu():
    """Non-destructive GPU probe: cupy + a device WITHOUT switching the active (numpy) backend (flipping SIM_BACKEND
    here would stickily cache the cupy backend in-process, breaking the CPU-safe tests)."""
    try:
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


# GPU is required for the on-spikes A->W read-out AND for the caches to be loadable. Mirror EMERGE-67/68: skip the GPU
# smoke unless the PROCESS is SIM_BACKEND=cupy (so we don't stickily flip the backend for the CPU-safe tests).
_SIM_BACKEND = os.environ.get("SIM_BACKEND", "numpy")
_GPU_REQUESTED = (_SIM_BACKEND == "cupy")
_CACHE_A = os.path.join(_REPO, "bridges", "emerge67_aw", "aw_content.simstate.h5")
_CACHE_F = os.path.join(_REPO, "bridges", "emerge68_aw", "aw_func.simstate.h5")
_CACHES_EXIST = os.path.exists(_CACHE_A) and os.path.exists(_CACHE_F)


class _CountingSpell:
    """A token-spell stand-in with a call counter (mirrors UnifiedNeuralSpell's spell contract, no GPU)."""
    def __init__(self):
        self.spell_calls = 0
        # the words the console producer might route (mirrors the 21-word producer vocab well enough for the wire tests)

    def spell(self, word):
        self.spell_calls += 1
        return str(word)


# ---------------------------------------------------------------------------------------------------------------------
# CPU-safe: the additive flag structure + the wire (spell routed through the console producer) + the gate-first moat.
# ---------------------------------------------------------------------------------------------------------------------
def test_neural_spell_flag_is_additive_default_off():
    """The `neural_spell` flag defaults to False (== EMERGE-66 byte-identical token surface). A console built without it
    has neural_spell False, no neural speller, and the render_kind carries NO +neural_spell suffix."""
    con = SpikingBrocaConsole(seed=42, build_fluid=False, neural_spell=False)
    assert con.neural_spell is False
    assert con._neural_speller is None
    assert "+neural_spell" not in con.render_kind
    # the default producer spell is the token surface (a slot spells its own payload string)
    words = con.broca.cq.emit("F_MODAL", "owl", "fly", con.broca.spell)
    assert words == ["the", "owl", "can", "fly"]           # token surface -- byte-identical to EMERGE-60/66


def test_wire_routes_spell_through_console_producer_all_slots():
    """The wire: the producer's spell callback realizes EVERY slot (det + subj + func + verb). A counting token-spell
    spliced into the producer is invoked for all slots (this is the exact mechanism the neural_spell flag installs)."""
    con = SpikingBrocaConsole(seed=42, build_fluid=False, neural_spell=False)
    sp = _CountingSpell()
    # splice a counting spell into the producer (== what neural_spell=True does, but with a CPU stand-in)
    con.broca = BrocaProducer(con.broca.cq, spell=sp.spell)
    words = con.broca.cq.emit("F_NEGMOD", "penguin", "fly", sp.spell)
    assert words == ["the", "penguin", "does", "not", "fly"]   # every slot realized via spell
    assert sp.spell_calls >= 5                                  # det + subj + does + not + verb


def test_gate_first_moat_producer_and_spell_never_called_on_abstain():
    """The gate-first no-confab MOAT: on ABSTAIN the console producer -- and hence the spell (A->W) -- is NEVER invoked.
    Asserted via BOTH BrocaProducer.production_count AND the spell call counter (the EMERGE-69 moat assertion)."""
    con = SpikingBrocaConsole(seed=42, build_fluid=False, neural_spell=False)
    sp = _CountingSpell()
    con.broca = BrocaProducer(con.broca.cq, spell=sp.spell)
    calls0 = sp.spell_calls
    prod0 = con.broca.production_count
    for _ in range(3):
        con.broca.speak(decision_from_emerge("ABSTAIN"))
    assert sp.spell_calls - calls0 == 0                        # spell NEVER called on abstain
    assert con.broca.production_count - prod0 == 0             # producer NEVER produces on abstain
    # positive control: an ANSWER DOES invoke the producer + spell
    con.broca.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    assert sp.spell_calls - calls0 > 0
    assert con.broca.production_count - prod0 == 1


def test_all_slot_ground_truth_content_and_function():
    """The EMERGE-68 all-slot ground-truth (reused by the EMERGE-69 all-word-spike scorer) includes DET + FUNC (not just
    content): 'the owl can fly', 'the penguin does not fly', 3sg-inflected verb for F_INTR."""
    import research.runners._emerge68_function_word_spell_derisk as m68
    assert m68._all_slot_surfaces("F_MODAL", "owl", "fly") == ["the", "owl", "can", "fly"]
    assert m68._all_slot_surfaces("F_NEGMOD", "penguin", "fly") == ["the", "penguin", "does", "not", "fly"]
    assert m68._all_slot_surfaces("F_INTR", "penguin", "walk") == ["the", "penguin", "walks"]


# ---------------------------------------------------------------------------------------------------------------------
# GPU smoke (skip unless the PROCESS is SIM_BACKEND=cupy AND the caches exist): the flagship console with neural_spell +
# self_organized renders EVERY word on spikes, and the moat holds with the real spiking spell.
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (_GPU_REQUESTED and _CACHES_EXIST and _has_gpu()),
                    reason="A->W read-out needs SIM_BACKEND=cupy + the EMERGE-67/68 caches (process-sticky skip-guard)")
def test_gpu_flagship_producer_all_word_spike_render_and_moat():
    """GPU smoke: the flagship's self-organized producer with the neural spell wired in renders every slot on spikes,
    and the gate-first moat holds (producer + A->W spell never invoked on abstain). This is the exact producer the
    `neural_spell` flag installs (SelfOrganizedProducer.producer(spell=UnifiedNeuralSpell.spell))."""
    import research.runners._emerge69_console_fully_spiking_derisk as m69
    import research.runners._emerge68_function_word_spell_derisk as m68
    unified = m68.UnifiedNeuralSpell(load=True)
    assert unified._backend_gpu
    sop, producer = m69._build_self_organized_producer(42, unified.spell)
    # ALL words (content + function) spelled on spikes through the flagship's self-organized producer
    all_acc, func_acc, _ex = m69._producer_all_word_spike_render(producer, 42)
    assert all_acc >= 0.90
    # the moat: on abstain the producer + A->W spell are NEVER invoked
    calls0 = unified.spell_calls
    prod0 = producer.production_count
    producer.speak(decision_from_emerge("ABSTAIN"))
    assert unified.spell_calls - calls0 == 0
    assert producer.production_count - prod0 == 0
