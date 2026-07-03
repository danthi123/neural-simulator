"""CPU tests for EMERGE-90 -- the conversational-turn capstone: HEAR -> comprehend (reservoir) -> store (composer) ->
ASK -> SPEAK the answer ON SPIKES (self-organized producer).

A light structural test (the de-inflection lexicon inverts emerge_v3; a single turn speaks the right transitive) + a
slow single-seed gate over all the anti-cheats. CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import research.runners._emerge90_conversational_turn_capstone_derisk as m90  # noqa: E402


def test_speak_answer_renders_the_transitive_surface():
    from research.runners._emerge74_transitive_ditransitive_derisk import (
        build_stream_svo, SVOConstructionRegistry, emerge_v3, _TRANS_VERBS,
    )
    from research.runners._emerge72_construction_registry_derisk import RegistryBrocaProducer
    seed = 42
    reg = SVOConstructionRegistry(seed).build(build_stream_svo(seed))
    assert "C_TRANS" in reg.registered
    producer = RegistryBrocaProducer(reg.render_cq())
    v = _TRANS_VERBS[0]                                    # "chase"
    surface, produced = m90._speak_answer(producer, "dog", v, "ball")
    assert produced
    assert surface == f"the dog {emerge_v3(v)} the ball"   # "the dog chases the ball" -- spoken on spikes


def test_deinflection_lexicon_inverts_emerge_v3():
    from research.runners._emerge74_transitive_ditransitive_derisk import emerge_v3, _TRANS_VERBS
    bare_of = {emerge_v3(v): v for v in _TRANS_VERBS}
    for v in _TRANS_VERBS:
        assert bare_of[emerge_v3(v)] == v                  # the 3sg -> bare lexicon is a clean inverse


@pytest.mark.slow
def test_seed42_conversational_turn_capstone():
    d = m90._derisk_one(42)
    assert d["parse_acc"] >= 0.90              # the reservoir comprehends the heard sentence
    assert d["recall"] >= 0.90                 # the composer recalls the patient
    assert d["render_exact"] >= 0.90           # the producer SPEAKS the answer sentence on spikes
    assert d["moat_false_accept"] <= 0.05      # no-confab moat holds
    assert d["moat_producer_invoked_on_abstain"] == 0    # gate-first: producer NEVER invoked on abstain
    assert d["lesion_render_exact"] <= 0.30    # comprehension is load-bearing
    assert d["nolearn_render_exact"] <= 0.60   # the learned spiking order is load-bearing


@pytest.mark.slow
def test_seed42_spiking_reservoir_capstone():
    # EMERGE-91: comprehension is the on-bridge SPIKING reservoir (OnBridgeLSM) -> two spiking bridges + spiking producer
    d = m90._derisk_one(42, spiking_reservoir=True)
    assert d["parse_acc"] >= 0.90
    assert d["recall"] >= 0.90
    assert d["render_exact"] >= 0.90
    assert d["moat_false_accept"] <= 0.05
    assert d["moat_producer_invoked_on_abstain"] == 0
    assert d["lesion_render_exact"] <= 0.30
    assert d["nolearn_render_exact"] <= 0.60
