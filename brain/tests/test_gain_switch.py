"""Brain(gain=...) (default ON since 2026-11-14 = "auto"): the per-size refit (1.5B: 0.5, results/2026-11-13-gain-refit-1.5b.md,
2026-11-13-live-gain-1.5b.md) where GAIN_BY_SIZE has a row, else Hippocampus.GAIN 4.0 (0.1B and unknown sizes unchanged).
"fixed" = always 4.0 (the flag-off arm); "size" = strict. Stand-in cortex, no model weights."""
import pytest
import torch

from brain.brain import Brain
from brain.hippocampus import Hippocampus
from brain.tests.test_brain_key_switch import KW, StubCortex
from brain.tests.test_key_bar_switch import Stub15


class Stub01(StubCortex):
    model_name = "rwkv7-g1d-0.1b-20260129-ctx8192"


def _alpha(b):
    seen = {}

    class Spy:
        def add_delta(self, state, delta, alpha):
            seen["alpha"] = alpha
            return state
    b.hippocampus.reinstate(Spy(), [torch.zeros(1)], type("E", (), {"delta": [None]})())
    return seen["alpha"]


def test_default_is_auto_and_15b_gets_refit(tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=Stub15(), **KW)
    assert b.gain_mode == "auto" and b.hippocampus.GAIN == 0.5 and _alpha(b) == 0.5
    assert Hippocampus.GAIN == 4.0   # the class constant is not touched


@pytest.mark.parametrize("cortex", [Stub01, StubCortex])
def test_other_sizes_unchanged_by_default(tmp_path, cortex):
    b = Brain(str(tmp_path / "a"), cortex=cortex(), **KW)
    assert b.hippocampus.GAIN == 4.0 and _alpha(b) == 4.0


def test_fixed_is_the_old_behaviour(tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=Stub15(), **KW, gain="fixed")
    assert b.hippocampus.GAIN == 4.0


@pytest.mark.parametrize("key", ["S0", "S_all"])
def test_size_gain_with_either_key(tmp_path, key):
    b = Brain(str(tmp_path / key), cortex=Stub15(), **KW, key=key, key_bar="size" if key != "S0" else "default",
              gain="size")
    assert _alpha(b) == Brain.GAIN_BY_SIZE["1.5b"] == 0.5


def test_errors(tmp_path):
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, gain="size")   # no size in the name
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "b"), cortex=Stub15(), **KW, gain="0.5")
