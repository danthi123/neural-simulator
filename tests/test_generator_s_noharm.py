"""LOAD-BEARING no-harm: Generator-S is PURELY ADDITIVE. The validated
deliverable + the project's anti-cheat machinery are byte-untouched.
Generator-S owns its OWN frozen bars and pulls in NOTHING that mutates
or shadows song_g1_core's bars."""
import sys


def test_generator_s_owns_its_bars_and_does_not_touch_g1():
    import research.runners.subword_lm_gate_core as g
    # Generator-S's OWN frozen bars (NOT g1's 0.10/0.5)
    assert g._GS_PPL_MARGIN == 0.20
    assert g._GS_GENERALIZATION_MAX == 1.5
    assert g._GS_DISTINCT_MIN == 0.5
    assert g._GS_COPY_MAX == 0.20
    assert g._GS_MIN_SEEDS == 3
    # it must NOT carry g1's bar names
    assert not hasattr(g, "_G1_MARGIN")
    assert not hasattr(g, "_G1_ABS_FLOOR")


def test_importing_gate_core_does_not_pull_song_g1_core():
    # gate_core is the anti-cheat heart; it must be isolated from
    # song_g1_core (no shared/mutable bar state across the two lines).
    mods = [m for m in sys.modules
            if m == "research.runners.song_g1_core"]
    import research.runners.subword_lm_gate_core  # noqa: F401
    after = [m for m in sys.modules
             if m == "research.runners.song_g1_core"]
    assert mods == after, (
        "Generator-S gate_core must not import song_g1_core")


def test_minseeds_floor_is_unbypassable_pin():
    # the >=3-seed pre-registered floor cannot be weakened by a caller
    from research.runners.subword_lm_gate_core import (
        gs_aggregate_multiseed)
    P = {"GATE": "PASS"}
    assert gs_aggregate_multiseed([P], min_seeds=1)["GATE"] == "FAIL"
    assert gs_aggregate_multiseed([P, P, P])["GATE"] == "PASS"
