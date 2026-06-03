"""Unit + smoke tests for the content-selection / dialogue-control layer.

All controller-logic tests use small hand-built (synthetic) association graphs so
they are fast and deterministic and need no spiking bridge / GPU. Plain ASCII only.
"""

from research.runners.content_selection import (
    ContextBuffer,
    relevance,
)


# --- Task 1: Context buffer -------------------------------------------------

def test_context_buffer_decays_and_adds():
    cb = ContextBuffer(decay=0.5)
    cb.update(["apple"])                 # apple enters at weight 1.0
    cb.update(["big"])                   # apple fades to 0.5, big enters at 1.0
    w = cb.weights()
    assert abs(w["big"] - 1.0) < 1e-9
    assert abs(w["apple"] - 0.5) < 1e-9


def test_context_buffer_reinforces_repeat():
    cb = ContextBuffer(decay=0.5)
    cb.update(["apple"]); cb.update(["apple"])   # 1.0 -> 0.5 then +1.0 = 1.5
    assert abs(cb.weights()["apple"] - 1.5) < 1e-9


# --- Task 2: Relevance score ------------------------------------------------

def test_relevance_sums_association_to_active_context():
    graph = {"apple": {"big": 2.0, "cat": 1.0}, "big": {"apple": 2.0}}
    context = {"apple": 1.0}             # we've been talking about apple
    # 'big' is associated with apple at 2.0 -> relevance 2.0; 'cat' at 1.0 -> 1.0
    assert abs(relevance("big", context, graph) - 2.0) < 1e-9
    assert abs(relevance("cat", context, graph) - 1.0) < 1e-9
    assert abs(relevance("apple", context, graph) - 0.0) < 1e-9   # not associated w/ itself here


def test_relevance_weights_by_context_strength():
    graph = {"apple": {"big": 2.0}, "river": {"big": 1.0}}
    context = {"apple": 0.5, "river": 1.0}   # river is more active
    # big: 0.5*2.0 + 1.0*1.0 = 2.0
    assert abs(relevance("big", context, graph) - 2.0) < 1e-9
