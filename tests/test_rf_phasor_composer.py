"""FHRR-on-bridge layer (b) b.1: the parallel RF phasor composer does who/what Q&A + abstention (the no-confab moat)
through the bridge's RF complex-synapse bind/bundle/unbind. The de-risk GATE for layer (b). See
docs/plans/2026-06-05-fhrr-layer-b-composer-recode-design.md.
"""
import pytest

from research.runners.rf_phasor_composer import RFPhasorComposer


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_who_what_abstain(seed):
    """b.1 GATE (multi-seed): store 2 SVO facts; who/what queries retrieve the right role; an absent cue ABSTAINS
    (returns None) -- the no-confab moat preserved on the RF phasor substrate."""
    comp = RFPhasorComposer(seed=seed, D=64, period=200)
    comp.store("dog", "go", "north")
    comp.store("cat", "run", "south")

    # who <action> <patient>? -> agent
    assert comp.query_agent("go", "north") == "dog"
    assert comp.query_agent("run", "south") == "cat"
    # what does <agent> <action>? -> patient
    assert comp.query_patient("dog", "go") == "north"
    assert comp.query_patient("cat", "run") == "south"

    # abstention (the no-confab moat): no stored fact matches these cues -> None
    assert comp.query_agent("go", "south") is None        # action=go but patient=south is not a stored pair
    assert comp.query_patient("dog", "run") is None        # agent=dog but action=run is not a stored pair


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_negation_yesno(seed):
    """b.2: a bound AFFIRM/NEGATE polarity tag -> ask_yes_no returns yes/no via the unbound tag; 'unknown'
    (abstention) when no stored fact matches. A 4-role bind (SVO + polarity) through the RF complex synapses."""
    comp = RFPhasorComposer(seed=seed, D=64, period=200)
    comp.store("dog", "go", "north", polarity="AFFIRM")
    comp.store("cat", "go", "south", polarity="NEGATE")

    assert comp.ask_yes_no("dog", "go", "north") == "yes"
    assert comp.ask_yes_no("cat", "go", "south") == "no"
    # abstention: no stored fact matches this full SVO cue
    assert comp.ask_yes_no("dog", "go", "south") == "unknown"


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_one_attribute(seed):
    """b.3a: a 1-attribute entity ('big apple') -- the ATTRIBUTE role-tag binding RESOLVES (adjective + noun both
    decoded from the RF unbind). A 4-role bind (agent/action/patient/attribute) through the RF complex synapses.
    (2-attribute is the documented K=5-load BOUNDARY, not asserted here -- carries over from the +-1 composer.)"""
    comp = RFPhasorComposer(seed=seed, D=64, period=200)
    comp.store("dog", "look", ("big", "apple"))
    comp.store("cat", "go", "river")

    assert comp.query_patient("dog", "look") == "big apple"   # attribute resolves
    assert comp.query_patient("cat", "go") == "river"           # plain patient still works
    assert comp.query_agent("look", "apple") == "dog"           # query on the noun (patient)
    # abstention
    assert comp.query_patient("cat", "look") is None


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_clause(seed):
    """b.3b: a recursive CLAUSE as a filler ('dog look (cat go north)') -- the clause's bound composite is the
    patient filler; the query renders the nested SVO, decoded from the RF unbind (double-nesting through the RF
    complex synapses). D=128 for the nesting SNR."""
    from research.runners.rf_phasor_composer import Clause
    comp = RFPhasorComposer(seed=seed, D=128, period=200)
    comp.store("dog", "look", Clause("cat", "go", "north"))
    comp.store("river", "stop", "apple")

    assert comp.query_patient("dog", "look") == "cat go north"   # the nested clause renders
    assert comp.query_patient("river", "stop") == "apple"          # a flat patient still works


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_dialogue(seed):
    """b.4: dialogue planning -- elaborate(topic) brings up an on-topic associate via the dlPFC spiking
    content-selection over the association graph built from the RF composer's facts; None when unconnected.
    GPU-only: the reused SpikingSpreadingController (dlPFC) has a numpy-backend IndexError in that component."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("dlPFC SpikingSpreadingController is GPU-validated (numpy-backend IndexError in that component)")
    comp = RFPhasorComposer(seed=seed, D=64, period=200)
    comp.store("dog", "go", "north")
    comp.store("dog", "run", "south")

    assoc = comp.elaborate("dog")
    assert assoc in {"go", "north", "run", "south"}, f"elaborate('dog')={assoc} not an on-topic associate"
    assert comp.elaborate("apple") is None    # apple is in no stored fact -> unconnected (abstention)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_full_matrix_at_scale(seed):
    """Layer (c1): the full capability matrix at a larger fact set (5 facts) -- who/what Q&A, abstention, negation,
    one-attribute, generation -- multi-seed, mirroring the rate-composer's capability bar. D=96 for the 5-fact load."""
    comp = RFPhasorComposer(seed=seed, D=96, period=200)
    comp.store("dog", "go", "north")
    comp.store("cat", "run", "south")
    comp.store("river", "look", ("big", "apple"))           # one-attribute
    comp.store("dog", "stop", "east", polarity="AFFIRM")
    comp.store("cat", "look", "west", polarity="NEGATE")

    # who/what (action disambiguates the two dog/cat facts)
    assert comp.query_agent("go", "north") == "dog"
    assert comp.query_patient("cat", "run") == "south"
    assert comp.query_patient("river", "look") == "big apple"      # one-attribute resolves at scale
    # generation (river is a unique agent)
    assert comp.render_fact("river") == "river look big apple"
    # negation / yes-no
    assert comp.ask_yes_no("dog", "stop", "east") == "yes"
    assert comp.ask_yes_no("cat", "look", "west") == "no"
    # abstention (the no-confab moat) at scale
    assert comp.query_agent("go", "south") is None
    assert comp.ask_yes_no("dog", "go", "west") == "unknown"
    assert comp.render_fact("apple") is None                       # apple is not an agent


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_production_scale(seed):
    """Layer (c-scale): a larger fact set (10 facts) -- who/what Q&A retrieves the RIGHT fact and the no-confab moat
    does NOT false-match among 10 facts. The key production-scale risk: spurious matches as the KB grows. D=128."""
    comp = RFPhasorComposer(seed=seed, D=128, period=200)
    facts = [
        ("dog", "go", "north"), ("cat", "run", "south"), ("dog", "look", "east"),
        ("river", "stop", "west"), ("apple", "go", "south"), ("cat", "look", "north"),
        ("dog", "run", "west"), ("river", "go", "east"), ("apple", "stop", "north"),
        ("cat", "go", "west"),
    ]
    for a, v, p in facts:
        comp.store(a, v, p)

    # retrieval: the (action, patient) / (agent, action) cue disambiguates the right fact among 10
    assert comp.query_agent("go", "north") == "dog"
    assert comp.query_agent("run", "south") == "cat"
    assert comp.query_patient("river", "stop") == "west"
    assert comp.query_patient("apple", "go") == "south"
    # abstention (the no-confab moat) at 10 facts: cues that match NO stored fact -> None (no false match)
    assert comp.query_agent("stop", "south") is None       # no fact has action=stop AND patient=south
    assert comp.query_patient("dog", "stop") is None         # dog never stops


def test_brain_agent_with_rf_composer():
    """(c2a) end-to-end: BrainConversationalAgent using the FHRR-on-bridge RFPhasorComposer (opt-in injection) --
    the Hebbian parser comprehends sentences and feeds the RF composer; hear -> store, query, abstain. Validates
    the switch works at the AGENT level (the rate-coded composer stays the default; this is the explicit opt-in).
    GPU-only: the Hebbian BridgeParser is GPU-validated (numpy-backend KeyError in the parser bridge)."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("BridgeParser is GPU-validated (numpy-backend KeyError in the parser bridge)")
    from research.runners.rf_phasor_composer import RFPhasorComposer
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    rf = RFPhasorComposer(seed=42, D=96, period=200)
    agent = BrainConversationalAgent(seed=42, composer=rf)
    agent.hear("dog go north")
    agent.hear("cat run south")

    # the parser comprehended each sentence and stored it in the RF composer
    assert agent.composer.query_agent("go", "north") == "dog"
    assert agent.composer.query_patient("cat", "run") == "south"
    assert agent.composer.query_agent("go", "south") is None    # abstention -- the no-confab moat, end to end


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_two_attribute(seed):
    """2-attribute ('big hot apple') -- the documented K=5-load BOUNDARY of the rate-coded +-1 composer (the noun
    degrades at the 5-binding edge ~0.93). The FHRR phasor substrate has better capacity (SNR ~ 2N/M, a D dial):
    at D=256 it RESOLVES multi-seed -- the boundary is LIFTED by the substrate, no F=3 resonator needed."""
    comp = RFPhasorComposer(seed=seed, D=256, period=200)
    comp.store("dog", "look", (("big", "hot"), "apple"))   # two-attribute entity (5-binding fact)
    comp.store("cat", "go", "river")
    assert set(comp.query_patient("dog", "look").split()) == {"big", "hot", "apple"}   # BOTH adjectives + the noun
    assert comp.query_patient("cat", "go") == "river"                                    # flat patient still works
    assert comp.query_patient("dog", "go") is None                                       # abstention (no-confab moat)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_spiking_cleanup_parity(seed):
    """Cheat-B conversion (opt-in): enable_spiking_cleanup routes cleanup through the FULLY-on-bridge spiking path
    -- the matched FILTER is the complex-synapse matvec (the same op as unbind; |c_k| read off the membrane) and the
    SELECTION is a spiking Izhikevich WTA (argmax-over-firing, the NEF-cleanup structure). It must give the SAME
    answers as the numpy-argmax default, AND preserve the no-confab moat (abstention). D=256 (the cleanup codebook
    is dense; the D-dial clears it, as for two-attribute)."""
    cn = RFPhasorComposer(seed=seed, D=256, period=200, enable_spiking_cleanup=False)
    cs = RFPhasorComposer(seed=seed, D=256, period=200, enable_spiking_cleanup=True)
    for a, v, p in [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")]:
        cn.store(a, v, p); cs.store(a, v, p)
    for v, p, a in [("go", "north", "dog"), ("run", "south", "cat"), ("look", "apple", "river")]:
        assert cs.query_agent(v, p) == cn.query_agent(v, p) == a            # selection in spikes == numpy
        assert cs.query_patient(a, v) == cn.query_patient(a, v) == p
    assert cs.query_agent("go", "river") is None                            # no-confab moat preserved


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_substrate_store_parity(seed):
    """Cheat-C conversion (opt-in): enable_substrate_store holds each fact's bound composite in per-fact SUBSTRATE
    weights (a trigger->readout complex-synapse bridge, the Crawford-Eliasmith weight-store = Hebb memory-in-weights),
    retrieved by firing the trigger -> phase readout -- NOT a numpy kb array. It must give the SAME answers as the
    numpy-kb default AND preserve the no-confab moat (abstention)."""
    cn = RFPhasorComposer(seed=seed, D=128, period=200, enable_substrate_store=False)
    cs = RFPhasorComposer(seed=seed, D=128, period=200, enable_substrate_store=True)
    for a, v, p in [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")]:
        cn.store(a, v, p); cs.store(a, v, p)
    for v, p, a in [("go", "north", "dog"), ("run", "south", "cat"), ("look", "apple", "river")]:
        assert cs.query_agent(v, p) == cn.query_agent(v, p) == a            # memory in the substrate == numpy
        assert cs.query_patient(a, v) == cn.query_patient(a, v) == p
    assert cs.query_agent("go", "river") is None                           # no-confab moat preserved


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_grounded_codes_interface(seed):
    """Cheat-A conversion (opt-in INTERFACE): grounded_codes={word: phases[D]} overrides the random rng.uniform codes
    for those words. The composer must USE the provided codes and still do who/what Q&A + abstention. This guards the
    grounding INTERFACE; the genuine V1-Gabor-grounded validation (codes from REAL sensory features, 6/6 multi-seed)
    is research/findings/raw/_phase3_grounded_codes_derisk.py. The FULL grounding (real object images + abstract-
    concept grounding) is the documented embodied-cognition boundary -- this is the interface, not full semantics."""
    import numpy as np
    words = ["dog", "cat", "go", "run", "north", "south", "river", "apple"]
    g_rng = np.random.default_rng(seed + 4242)
    grounded = {w: g_rng.uniform(0.0, 1.0, 128) for w in words}             # externally-provided codes (de-risk uses V1)
    comp = RFPhasorComposer(seed=seed, D=128, period=200, grounded_codes=grounded)
    for w in words:
        assert np.allclose(comp.concepts[w], grounded[w])                   # the composer USED the provided codes
    comp.store("dog", "go", "north"); comp.store("cat", "run", "south")
    assert comp.query_agent("go", "north") == "dog"                         # Q&A works on the provided codes
    assert comp.query_patient("cat", "run") == "south"
    assert comp.query_agent("go", "south") is None                         # no-confab moat preserved


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
