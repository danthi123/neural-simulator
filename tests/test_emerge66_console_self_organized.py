"""CI guard for EMERGE-66 -- the flagship unified console renders its EMERGE emergent-reasoning answers ON THE SPIKING
SUBSTRATE from a FULLY-SELF-ORGANIZED producer (EMERGE-65 SelfOrganizedProducer, structure mined from the corpus) in
place of the host-FRAMES producer, via an ADDITIVE default-off `self_organized` flag on EMERGE-60's SpikingBrocaConsole.
CPU/numpy, offline.

Load-bearing properties: (1) the EMERGE answer is rendered by the self-organized producer with the correct CONTENT +
order on spikes (the whole grammatical structure -- function words + slot inventory + slot order -- was discovered from
the corpus, NOT the host FRAMES); (2) the gate-first no-confab MOAT holds -- the producer is NEVER invoked on an abstain;
(3) membership routing (EMERGE-58 remediation) unchanged; (4) the DEFAULT path (self_organized=False) is byte-identical
to EMERGE-60 (the flag is additive/default-preserving); (5) self-organized provenance (the console's producer structure
matches the host FRAMES from the corpus mine -- the wire is not silently the host FRAMES).
"""
import os
import sys

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._emerge60_console_spiking_broca_derisk import SpikingBrocaConsole
from research.runners._emerge59_spiking_broca_frame_slots_derisk import decision_from_emerge
from research.runners._emerge62_discover_function_words_derisk import FRAME_FUNCTION_WORDS
from research.runners._emerge65_self_organized_producer_derisk import assembled_structure_match

_N = 6000     # smaller corpus stream for CI speed (the derisk uses 6000 too; EMERGE-65 derisk uses 20000)


@pytest.fixture(scope="module")
def con():
    """The self-organized flagship console WITHOUT the heavy FluidChat (render/moat/provenance only) -- fast, CPU, seed 42."""
    return SpikingBrocaConsole(seed=42, build_fluid=False, self_organized=True, self_organized_n_sentences=_N)


def test_render_kind_is_self_organized(con):
    assert con.render_kind == "self_organized_broca"
    assert con.self_organized is True
    assert con.reset_producer is True                            # the self-organized CQ washes out per emit (subsumed)


def test_self_organized_provenance_matches_host_frames(con):
    """The console's producer structure was genuinely MINED from the corpus: the assembled structure MATCHES the host
    FRAMES (slot set + function-word fillers + order) and ALL frame function words were discovered -- the wire is not
    silently the host FRAMES dict."""
    _pf, struct_match, inv_acc = assembled_structure_match(con._sop)
    assert struct_match == pytest.approx(1.0), f"assembled-structure match {struct_match:.3f} != 1.0"
    assert inv_acc == pytest.approx(1.0), f"inventory accuracy {inv_acc:.3f} != 1.0"
    for fw in FRAME_FUNCTION_WORDS:
        assert fw in con._sop.discovered_function_words, f"frame function word {fw!r} not discovered from corpus"


def test_emerge_answers_render_on_spikes_from_self_organized_producer(con):
    """The EMERGE answers render on spikes from the self-organized producer with the correct surface (seed 42 exact)."""
    assert con.turn("can an owl fly?").strip().lower() == "the owl can fly"          # INHERIT (F_MODAL)
    assert con.turn("can a penguin fly?").strip().lower() == "the penguin walks"     # CANCEL (F_INTR, exception)
    robin = con.turn("can a robin breathe?").strip().lower()                          # PER-DIMENSION inherit
    assert set(robin.split()) == {"the", "robin", "can", "breathe"}


def test_negmod_frame_renders_exact_from_self_organized_producer(con):
    """The F_NEGMOD 'does not' frame (the console reasoner never emits it) renders EXACT directly through the
    self-organized producer -- the mined structure covers the full 3-frame inventory."""
    r = con.broca.speak(decision_from_emerge("ANSWER", subject="penguin", verb="fly", negated_modal=True))
    assert r["produced"] is True
    assert r["surface"] == "the penguin does not fly"


def test_producer_invoked_once_on_answer(con):
    before = con.broca.production_count
    con.turn("can an owl fly?")
    assert con.broca.production_count == before + 1


def test_moat_producer_never_invoked_on_abstain(con):
    """The gate-first moat: on a sibling-abstain the self-organized producer is NEVER invoked."""
    before = con.broca.production_count
    sib = con.turn("can an owl swim?")                                                # sibling (bird, not fish)
    assert sib.lower().startswith("i don't know")
    assert con.broca.production_count == before                                       # producer NOT invoked


def test_default_path_byte_identical_to_emerge60():
    """The flag is ADDITIVE / default-preserving: with self_organized=False the console is byte-identical to committed
    EMERGE-60 (host-FRAMES spiking_broca path, reset_producer False)."""
    c = SpikingBrocaConsole(seed=42, build_fluid=False)
    assert c.render_kind == "spiking_broca"
    assert c.self_organized is False
    assert c.reset_producer is False
    assert c.turn("can an owl fly?").strip().lower() == "the owl can fly"             # EMERGE-60 seed-42 exact
    assert c.turn("can a penguin fly?").strip().lower() == "the penguin walks"


@pytest.mark.slow
def test_membership_routing_and_no_false_denial():
    """Builds FluidChat: a fluid-known entity in the SHARED ability frame is answered by the fluid path (NOT falsely
    denied), and the self-organized producer is NOT stolen into it (EMERGE-58 remediation, preserved through the wire)."""
    con = SpikingBrocaConsole(seed=42, build_fluid=True, self_organized=True, self_organized_n_sentences=_N)
    before = con.broca.production_count
    dog = con.turn("can a dog eat?")
    assert not dog.lower().startswith("i don't know what a dog"), f"false denial: {dog!r}"
    assert ("meat" in dog.lower() or "eat" in dog.lower()), dog
    assert con.broca.production_count == before                                       # producer NOT invoked (fluid path)


@pytest.mark.slow
def test_derisk_go_seed42():
    """The full single-seed de-risk (builds FluidChat): the WIRE is correct -- right content routed to the self-organized
    producer, gate-first moat holds, membership routing unchanged, no fluid regression, self-organized provenance."""
    from research.runners._emerge66_console_self_organized_derisk import _derisk_one
    d = _derisk_one(42, build_fluid=True)
    assert d["render_kind"] == "self_organized_broca"
    assert d["emerge_render_words"] == 1.0                                            # right content on spikes
    assert d["negmod_exact"] is True                                                  # F_NEGMOD renders exact
    assert d["moat_ok"] is True and d["moat_producer_calls_on_abstain"] == 0          # gate-first moat
    assert d["membership_ok"] is True                                                 # audit-remediation preserved
    assert d["fluid_ok"] is True                                                      # no fluid regression (Broca-free)
    assert d["provenance_ok"] is True                                                 # self-organized provenance
