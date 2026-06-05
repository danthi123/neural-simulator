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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
