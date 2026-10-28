"""research/runners/stable_mechc_trait_attrib.py: with SIM_TRAIT_CENTRED_HET=1 and `--only build_slot_bridge`, only the
bridge built under that function is trait-centred; every other bridge reads back requested False. Restored afterwards."""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import sim.bridge as SB  # noqa: E402
from research.runners import stable_mechc_trait_attrib as TA  # noqa: E402


def test_chain_names_names_the_callers():
    def inner():
        return TA.chain_names(depth_skip=1)
    names = inner()
    assert names[0].endswith(".inner") and any(n.endswith(".test_chain_names_names_the_callers") for n in names)


def test_only_the_named_builder_is_centred(monkeypatch):
    from research.runners._D_sparse_heteroassoc import build
    from research.runners._emerge59_spiking_broca_frame_slots_derisk import build_slot_bridge
    monkeypatch.setenv("SIM_TRAIT_CENTRED_HET", "1")
    real = TA.restrict("build_slot_bridge")
    try:
        lag = build(7, n_pool=60, n_fs=24)
        slot, _idx = build_slot_bridge(7)
    finally:
        SB.trait_centred_het_requested = real
    assert lag.trait_centred_het_requested is False and lag.trait_centred_het_applied is False
    assert lag.trait_centred_het_readback == {"heterogeneity_ran": True, "trait_centred": False}
    assert slot.trait_centred_het_requested is True and slot.trait_centred_het_applied is True
    assert slot.trait_centred_het_readback["trait_centred"] is True
    assert SB.trait_centred_het_requested is real


def test_restriction_never_turns_the_switch_on_by_itself(monkeypatch):
    from research.runners._emerge59_spiking_broca_frame_slots_derisk import build_slot_bridge
    monkeypatch.delenv("SIM_TRAIT_CENTRED_HET", raising=False)
    real = TA.restrict("build_slot_bridge")
    try:
        slot, _idx = build_slot_bridge(7)
    finally:
        SB.trait_centred_het_requested = real
    assert slot.trait_centred_het_requested is False
