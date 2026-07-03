"""CI guard for EMERGE-60 -- the flagship unified console renders its EMERGE emergent-reasoning answers ON THE SPIKING
SUBSTRATE (EMERGE-59 frame-slot competitive queuing on a real SimulationBridge) in place of the 21M ANN, the ANN retired
for the EMERGE frame inventory. CPU/numpy, offline.

The load-bearing properties: (1) the EMERGE answer is rendered by the SPIKING producer (not the ANN) with the correct
CONTENT; (2) the gate-first no-confab MOAT holds -- the spiking producer is NEVER invoked on an abstain; (3) the
membership-aware routing (EMERGE-58 audit remediation) is unchanged -- a fluid-known entity in the shared ability frame
is answered by the fluid path, not falsely denied, and the producer is not stolen into it.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners._emerge60_console_spiking_broca_derisk import SpikingBrocaConsole, _derisk_one


@pytest.fixture(scope="module")
def con():
    """A spiking-Broca console WITHOUT the heavy FluidChat (render/moat only) -- fast, CPU, seed 42 (renders exact)."""
    return SpikingBrocaConsole(seed=42, build_fluid=False)


def test_render_kind_is_spiking(con):
    assert con.render_kind == "spiking_broca"


def test_emerge_answers_render_on_spikes(con):
    """The EMERGE answers are rendered by the spiking producer with the correct surface (seed 42 renders exact; content
    is always correct on every seed)."""
    assert con.turn("can an owl fly?").strip().lower() == "the owl can fly"          # INHERIT (F_MODAL)
    assert con.turn("can a penguin fly?").strip().lower() == "the penguin walks"     # CANCEL (F_INTR, exception)
    robin = con.turn("can a robin breathe?").strip().lower()                          # PER-DIMENSION inherit
    assert set(robin.split()) == {"the", "robin", "can", "breathe"}                   # content correct (order exact on 42)


def test_producer_invoked_once_on_answer(con):
    before = con.broca.production_count
    con.turn("can an owl fly?")
    assert con.broca.production_count == before + 1                                   # ANSWER -> spiking producer runs


def test_moat_producer_never_invoked_on_abstain(con):
    """The gate-first moat: on a sibling-abstain OR an unknown-abstain the SPIKING producer is NEVER invoked."""
    before = con.broca.production_count
    sib = con.turn("can an owl swim?")                                                # sibling (bird, not fish)
    unk = con.turn("can a zzz fly?")                                                  # never observed
    assert sib.lower().startswith("i don't know")
    assert unk.lower().startswith("i don't know")
    assert con.broca.production_count == before                                       # producer NOT invoked on either


@pytest.mark.slow
def test_membership_routing_and_no_false_denial():
    """Builds FluidChat: a fluid-known entity in the SHARED ability frame is answered by the fluid path (NOT falsely
    denied), and the SPIKING producer is NOT stolen into it (EMERGE-58 audit remediation, preserved through the wire)."""
    con = SpikingBrocaConsole(seed=42, build_fluid=True)
    before = con.broca.production_count
    dog = con.turn("can a dog eat?")
    assert not dog.lower().startswith("i don't know what a dog"), f"false denial: {dog!r}"
    assert ("meat" in dog.lower() or "eat" in dog.lower()), dog
    assert con.broca.production_count == before                                       # producer NOT invoked (fluid path)


@pytest.mark.slow
def test_derisk_go_seed42():
    """The full single-seed de-risk (builds FluidChat): the WIRE is correct -- right content routed to the spiking
    producer, gate-first moat holds, membership routing unchanged, no fluid regression."""
    d = _derisk_one(42, build_fluid=True)
    assert d["emerge_render_words"] == 1.0                                            # right content on spikes
    assert d["moat_ok"] is True and d["moat_producer_calls_on_abstain"] == 0          # gate-first moat
    assert d["membership_ok"] is True                                                 # audit-remediation preserved
    assert d["fluid_ok"] is True                                                      # no fluid regression
