"""Brain(key_bar=...) switch (default OFF = "default"): a non-S0 key keeps its 0.1B bar unless key_bar="size" picks the
per-size refit; S0 and the default Brain are untouched. The live probe adds arms f/g only with --fix-key.
Runs on the stand-in cortex of test_brain_key_switch (no model weights)."""
import pytest
import torch

from brain.brain import Brain
from brain.experiments import live_brain_probe as lbp
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import KeyedHippocampus
from brain.tests.test_brain_key_switch import KW, StubCortex, _talk


class Stub15(StubCortex):
    model_name = "rwkv7-stub-1.5b-20261031"


def test_default_is_unchanged(tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=Stub15(), **KW)
    assert b.key_bar == "default" and type(b.hippocampus) is Hippocampus
    assert (b.hippocampus.FAMILIARITY, b.hippocampus.MARGIN) == (0.12, 0.02)
    s = Brain(str(tmp_path / "b"), cortex=Stub15(), **KW, key="S_all")
    assert (s.hippocampus.FAMILIARITY, s.hippocampus.MARGIN) == Brain.KEY_BARS["S_all"]


def test_size_bar_for_sall_at_15b(tmp_path):
    b = Brain(str(tmp_path / "h"), cortex=Stub15(), **KW, key="S_all", key_bar="size")
    h = b.hippocampus
    assert isinstance(h, KeyedHippocampus) and h.key_name == "S_all"
    assert (h.FAMILIARITY, h.MARGIN) == Brain.KEY_BARS_BY_SIZE[("S_all", "1.5b")] == (0.22, 0.01)
    assert (Hippocampus.FAMILIARITY, Hippocampus.MARGIN) == (0.12, 0.02)   # class defaults untouched
    _talk(b)
    b.save()
    # reopening keeps the switch only if passed again (not stored in the home); the key itself is stored
    r = Brain(str(tmp_path / "h"), cortex=Stub15(), **KW, key_bar="size")
    assert r.key == "S_all" and (r.hippocampus.FAMILIARITY, r.hippocampus.MARGIN) == (0.22, 0.01)
    d = Brain(str(tmp_path / "h"), cortex=Stub15(), **KW)
    assert d.key == "S_all" and (d.hippocampus.FAMILIARITY, d.hippocampus.MARGIN) == Brain.KEY_BARS["S_all"]


def test_size_bar_changes_recall(tmp_path):
    """The switch is not inert: a top score of 0.20 (runner-up 0.0) fires under the 0.1B bar 0.185, not under 0.22."""
    out = {}
    for kb in ("default", "size"):
        b = Brain(str(tmp_path / kb), cortex=Stub15(), **KW, key="S_all", key_bar=kb)
        _talk(b)
        n = len(b.hippocampus.episodes)
        score = torch.zeros(n)
        score[n - 1] = 0.20
        ep, _ = b.hippocampus._compete(score, True, 0.0)
        out[kb] = ep is not None
    assert out == {"default": True, "size": False}


def test_size_bar_errors(tmp_path):
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, key="S_all", key_bar="size")   # no size in the name
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "b"), cortex=Stub15(), **KW, key="S_all", key_bar="bogus")
    s0 = Brain(str(tmp_path / "c"), cortex=Stub15(), **KW, key_bar="size")   # S0 ignores key_bar
    assert type(s0.hippocampus) is Hippocampus


def test_probe_fix_arms_are_opt_in():
    assert set(lbp.FIX_ARMS) == {"f_fix_sleep", "g_fix_nosleep"}
    assert not set(lbp.FIX_ARMS) & set(lbp.ARMS)
    import inspect
    sig = inspect.signature(lbp.ask_arm)
    assert sig.parameters["hip_key"].default is None and sig.parameters["key_bar"].default == "default"
