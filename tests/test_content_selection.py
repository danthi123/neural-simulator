"""Unit + smoke tests for the content-selection / dialogue-control layer.

All controller-logic tests use small hand-built (synthetic) association graphs so
they are fast and deterministic and need no spiking bridge / GPU. Plain ASCII only.
"""

from research.runners.content_selection import ContextBuffer


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
