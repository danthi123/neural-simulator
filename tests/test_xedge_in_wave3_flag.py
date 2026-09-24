"""BRAIN_XEDGE_IN_WAVE3 (default OFF): the flag reader's preconditions and the merged-pool routing point.

No pool is built: the builders are monkeypatched to sentinels, so this pins only WHICH builder each flag state
routes to (the flag-OFF path must call exactly the pre-flag `get_wave3_pool`)."""
import sys

import pytest

import research.runners.onebrain_wave3_pool_production as W3
from research.runners.onebrain_xedge_wave3_flags import xedge_in_wave3_enabled, xedge_in_wave3_flag

_ENV = ("BRAIN_XEDGE_IN_WAVE3", "BRAIN_ONEBRAIN_XEDGE", "BRAIN_ONEBRAIN_XEDGE_LEARN", "BRAIN_ONEBRAIN_WAVE3_POOL",
        "BRAIN_ONEBRAIN_AFFECT_POOL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)


def test_default_off():
    assert xedge_in_wave3_flag() is False
    assert xedge_in_wave3_enabled() is False


def test_on_requires_every_precondition(monkeypatch):
    monkeypatch.setenv("BRAIN_XEDGE_IN_WAVE3", "1")
    assert xedge_in_wave3_enabled() is True                 # xedge, learn and wave3 all default ON
    for k, v in (("BRAIN_ONEBRAIN_XEDGE", "0"), ("BRAIN_ONEBRAIN_XEDGE_LEARN", "0"),
                 ("BRAIN_ONEBRAIN_WAVE3_POOL", "0"), ("BRAIN_ONEBRAIN_AFFECT_POOL", "1")):
        monkeypatch.setenv(k, v)
        assert xedge_in_wave3_enabled() is False, k
        monkeypatch.delenv(k)


def test_flag_off_routes_to_the_unchanged_wave3_builder(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(W3, "get_wave3_pool", lambda seed=42: sentinel)
    sys.modules.pop("research.runners.onebrain_xedge_wave3", None)
    assert W3.get_merged_cortical_pool(7, min_wave=1) is sentinel
    assert W3.get_merged_cortical_pool(7, min_wave=2) is sentinel
    # the flag-OFF path never imports the in-wave3 module
    assert "research.runners.onebrain_xedge_wave3" not in sys.modules


def test_flag_on_routes_to_the_in_wave3_pool(monkeypatch):
    import research.runners.onebrain_xedge_wave3 as XW3
    on, plain = object(), object()
    monkeypatch.setenv("BRAIN_XEDGE_IN_WAVE3", "1")
    monkeypatch.setattr(W3, "get_wave3_pool", lambda seed=42: plain)
    monkeypatch.setattr(XW3, "get_wave3_xedge_pool", lambda seed=42: on)
    assert W3.get_merged_cortical_pool(7, min_wave=1) is on
    assert W3.get_merged_cortical_pool(7, min_wave=2) is on
    monkeypatch.setenv("BRAIN_ONEBRAIN_AFFECT_POOL", "0")
    assert W3.get_merged_cortical_pool(7) is on
