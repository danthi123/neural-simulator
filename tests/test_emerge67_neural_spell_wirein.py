"""CI for EMERGE-67 -- wire the VALIDATED SPIKING A->W (concept-pool -> spoken-word) read-out into the spiking-Broca
producer's `spell` callback, so the CONTENT words (subject/verb) of the EMERGE-frame render are produced ON SPIKES.

CPU-safe where possible: the WIRE (neural_spell -> BrocaProducer.spell) + the gate-first moat + the content-slot
scoring are validated with a token-spell stand-in (no GPU). The genuinely-spiking A->W read-out (drive a concept pool ->
decode from language_output spikes) is GPU-only -> a skip-if-no-cupy smoke (build the engine + a few word decodes +
the moat with the real spiking spell). The token-spell default path is byte-identical to EMERGE-59..66.
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
import research.runners._emerge67_neural_spell_wirein_derisk as m67  # noqa: E402


def _has_gpu():
    """Non-destructive GPU probe: check cupy + a device WITHOUT switching the active (numpy) backend, so the CPU-safe
    tests keep running on numpy (flipping SIM_BACKEND here would stickily cache the cupy backend in-process)."""
    try:
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------------------------
# CPU-safe: the WIRE + moat + content scoring with a token-spell stand-in (the spell callback contract).
# ---------------------------------------------------------------------------------------------------------------------
class _CountingSpell:
    """A token-spell stand-in with a call counter (mirrors NeuralSpell's spell contract, no GPU)."""
    def __init__(self):
        self.spell_calls = 0
        self.word_to_pool = {w: w for w in m67._AW_CONTENT}   # content words are "known"

    def spell(self, word):
        self.spell_calls += 1
        return str(word)


def test_content_vocab_16_words_rebind_onto_16_pools():
    """The producer content vocab is 16 words (8 subjects + 8 verbs) rebound onto the 16 validated concept pools."""
    assert len(m67._AW_CONTENT) == 16
    wp, pw = m67._pool_assignment()
    assert len(wp) == 16 and len(pw) == 16
    assert set(wp) == set(m67._AW_CONTENT)
    # every content word maps to a distinct pool
    assert len(set(wp.values())) == 16


def test_wire_content_slots_use_spell_det_func_token():
    """The wire spells CONTENT slots (subject/verb) via the spell callback; DET/FUNC keep their fixed surface. Verify
    the produced frame contains the (token-spelled) content words + the fixed function words in the right order."""
    cq = FrameSlotCQ(seed=42)
    cq.learn()
    sp = _CountingSpell()
    words = cq.emit("F_MODAL", "owl", "fly", sp.spell)
    assert words == ["the", "owl", "can", "fly"]           # the owl can fly
    # the content slots (owl, fly) went through spell; DET/FUNC (the, can) are fixed
    assert sp.spell_calls >= 2


def test_gate_first_moat_spell_never_called_on_abstain():
    """The gate-first no-confab MOAT: on ABSTAIN the producer -- and hence the spell callback -- is NEVER invoked."""
    cq = FrameSlotCQ(seed=42)
    cq.learn()
    sp = _CountingSpell()
    prod = BrocaProducer(cq, spell=sp.spell)
    calls0 = sp.spell_calls
    for _ in range(3):
        r = prod.speak(decision_from_emerge("ABSTAIN"))
        assert r["produced"] is False and r["surface"] is None
    assert sp.spell_calls == calls0                         # 0 spell calls on abstains
    assert prod.production_count == 0
    # a positive control: an ANSWER DOES invoke the producer + spell
    r = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    assert r["produced"] is True and sp.spell_calls > calls0 and prod.production_count == 1


def test_content_surfaces_ground_truth():
    """The content-slot ground-truth: subject verbatim; verb bare in F_MODAL/F_NEGMOD, 3sg-inflected in F_INTR."""
    assert m67._content_surfaces("F_MODAL", "owl", "fly") == {"subject": "owl", "verb": "fly"}
    assert m67._content_surfaces("F_INTR", "penguin", "walks") == {"subject": "penguin", "verb": "walks"}
    # a bare ability verb inflects to 3sg when the frame slot is 3sg (F_INTR)
    assert m67._content_surfaces("F_INTR", "owl", "fly")["verb"] == "flies"


def test_token_spell_no_regression_content_slots():
    """The token-spell content surfaces == ground-truth (the neural spell must reproduce this on a GO). Structural
    no-regression check for the content slots across all frames."""
    cq = FrameSlotCQ(seed=42)
    cq.learn()
    facts = m67._facts_from_content_vocab(42, n=4)
    token_spell = lambda w: str(w)
    acc, _ex = m67._render_and_score(cq, type("S", (), {"spell": staticmethod(token_spell)})(), facts)
    assert acc == pytest.approx(1.0), f"token spell content accuracy {acc} != 1.0 (ground-truth mismatch)"


def test_facts_drawn_from_content_vocab():
    """The de-risk facts are drawn from the A->W content vocab so every content slot is spike-spellable."""
    facts = m67._facts_from_content_vocab(42, n=8)
    for f in facts:
        assert f["subject"] in m67._AW_SUBJECTS
        assert f["ability_verb"] in m67._AW_ABILITY
        assert f["intr_verb"] in m67._AW_INTR3SG


# ---------------------------------------------------------------------------------------------------------------------
# GPU-only smoke (skip-if-no-cupy): the genuinely-spiking A->W read-out spells a few content words on spikes + the moat
# holds with the real spiking spell. Uses the cached engine if present (fast); otherwise trains (slow) -- skip if the
# cache is absent to keep CI fast unless explicitly warmed.
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("SIM_BACKEND", "numpy") != "cupy" or not _has_gpu(),
                    reason="A->W read-out needs the PROCESS to run SIM_BACKEND=cupy (the backend is process-sticky, so "
                           "it cannot flip mid-run when the CPU-safe tests already loaded numpy); run the CI with "
                           "SIM_BACKEND=cupy to exercise this. The full GPU A->W read-out is validated by the 6-seed "
                           "--derisk (content-spell 1.0, lesion-collapse, moat 0).")
@pytest.mark.skipif(not m67._CACHE_BRIDGE.exists(), reason="A->W engine cache absent (run --train to warm it)")
def test_gpu_neural_spell_decodes_from_spikes_and_moat():
    """GPU smoke: the A->W spell decodes a content word from language_output SPIKES (self-cos > 0, spikes > 0), and the
    gate-first moat holds (an abstain never invokes the spell)."""
    sp = m67.NeuralSpell(load=True)
    assert sp._backend_gpu, "expected a GPU backend for the A->W engine"
    # decode a couple of content words: the read-out reads real spikes (spike-total > 0)
    for w in ("owl", "fly"):
        decoded, self_cos, top_cos, spikes = sp._decode(w)
        assert spikes > 0, f"A->W read-out produced no language_output spikes for {w!r} (not spiking)"
        assert self_cos > 0.0
    # moat: the spell is never called on an abstain
    cq = FrameSlotCQ(seed=42)
    cq.learn()
    prod = BrocaProducer(cq, spell=sp.spell)
    calls0 = sp.spell_calls
    for _ in range(2):
        prod.speak(decision_from_emerge("ABSTAIN"))
    assert sp.spell_calls == calls0 and prod.production_count == 0
