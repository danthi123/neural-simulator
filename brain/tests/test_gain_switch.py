"""Brain(gain=...) switch (default OFF = "default": Hippocampus.GAIN 4.0). gain="size" sets the per-size refit (1.5B: 0.5,
results/2026-11-13-gain-refit-1.5b.md) on the hippocampus that reinstate() uses, for S0 and S_all; an unknown size or
value is an error. Stand-in cortex, no model weights."""
import pytest
import torch

from brain.brain import Brain
from brain.hippocampus import Hippocampus
from brain.tests.test_brain_key_switch import KW, StubCortex
from brain.tests.test_key_bar_switch import Stub15


def test_default_is_unchanged(tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=Stub15(), **KW)
    assert b.gain_mode == "default" and b.hippocampus.GAIN == Hippocampus.GAIN == 4.0


@pytest.mark.parametrize("key", ["S0", "S_all"])
def test_size_gain_at_15b_reaches_reinstate(tmp_path, key):
    b = Brain(str(tmp_path / key), cortex=Stub15(), **KW, key=key, key_bar="size" if key != "S0" else "default",
              gain="size")
    assert b.hippocampus.GAIN == Brain.GAIN_BY_SIZE["1.5b"] == 0.5
    assert Hippocampus.GAIN == 4.0   # the class constant is not touched

    seen = {}

    class Spy:
        def add_delta(self, state, delta, alpha):
            seen["alpha"] = alpha
            return state
    ep = type("E", (), {"delta": [None]})()
    b.hippocampus.reinstate(Spy(), [torch.zeros(1)], ep)
    assert seen["alpha"] == 0.5


def test_errors(tmp_path):
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, gain="size")   # no size in the name
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "b"), cortex=Stub15(), **KW, gain="0.5")
