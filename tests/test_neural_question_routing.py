"""CI guard for NEURAL question-comprehension routing: a fronto-striatal reservoir read-out classifies the
question TYPE (property / what / who / yes-no / describe) on the whole sequence, generalizing to NOVEL fillers,
with the closed-class lesion collapsing it -- the neural replacement for the host keyword-matching router.
numpy reference (the spiking OnBridgeLSM confirmation is GPU/slow; validated in the finding). CPU-only, fast.
"""
import numpy as np
import pytest


def test_reservoir_routes_question_type_generalizing():
    from research.runners._realcorpus_neural_question_routing_derisk import run
    r = run(seed=42, n_per=60)
    assert r["heldout"] >= 0.9                              # routes NOVEL-filler questions (generalizes, not memorizes)
    assert r["heldout"] - r["lesion"] > 0.30               # the closed-class function words are load-bearing (lesion collapses)
    assert r["lesion"] <= 0.45                              # lesion near chance (1/5)


def test_routing_multi_seed_holds():
    from research.runners._realcorpus_neural_question_routing_derisk import run
    for s in (43, 100):
        r = run(seed=s, n_per=40)
        assert r["heldout"] >= 0.9 and r["heldout"] - r["lesion"] > 0.30
