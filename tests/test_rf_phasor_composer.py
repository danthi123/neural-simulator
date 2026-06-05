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
    comp = RFPhasorComposer(seed=seed, D=64, period=400)
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
    comp = RFPhasorComposer(seed=seed, D=64, period=400)
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
    comp = RFPhasorComposer(seed=seed, D=64, period=400)
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
    comp = RFPhasorComposer(seed=seed, D=128, period=400)
    comp.store("dog", "look", Clause("cat", "go", "north"))
    comp.store("river", "stop", "apple")

    assert comp.query_patient("dog", "look") == "cat go north"   # the nested clause renders
    assert comp.query_patient("river", "stop") == "apple"          # a flat patient still works


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
