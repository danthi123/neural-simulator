"""Direction 5 verdict module - adversarial test matrix (>=12 cases).

Mirrors the Direction Q + Direction 3 + Direction 4 verdict-test
patterns. Tests:
- Frozen-threshold pins (tampering detection)
- Happy-path verdicts (PASS / PARTIAL / NEGATIVE)
- Instrument-validity (VOID_MALFORMED on every kind of malformed input)
- Boundary cases (exactly at threshold; just below)
- Fuzz-style (never raise on garbage input)

Imports are stdlib-only via importlib.util (the verdict module under
test is itself stdlib-only and CPU-only).
"""
from __future__ import annotations
import importlib.util
import math
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERDICT_PATH = os.path.join(
    REPO_ROOT,
    "research/findings/raw/direction_5_verdict.py",
)


def _load_verdict_module():
    spec = importlib.util.spec_from_file_location(
        "direction_5_verdict", _VERDICT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _all_cells(value: float):
    """Build a single seed entry with all 6 cells = value."""
    return {
        "L=2": {"OB": value, "OI": value},
        "L=3": {"OB": value, "OI": value},
        "L=5": {"OB": value, "OI": value},
    }


def _seeds_at(value: float, n_seeds: int = 3):
    return [_all_cells(value) for _ in range(n_seeds)]


# ===================================================================
# 1. Frozen thresholds (tampering detection)
# ===================================================================

def test_thresholds_frozen_at_design_values():
    """All 4 frozen thresholds match design-doc values."""
    mod = _load_verdict_module()
    assert mod._DIRECTION_5_OB_MIN == 0.80
    assert mod._DIRECTION_5_OI_MIN == 0.80
    assert list(mod._DIRECTION_5_LOADS) == [2, 3, 5]
    assert mod._DIRECTION_5_MIN_SEEDS == 3


def test_threshold_tamper_detection():
    """Verdict module exposes thresholds as MODULE-LEVEL constants,
    not as a runtime-mutable config dict. Any future PR that converts
    them to a dict or adds a setter triggers this test."""
    mod = _load_verdict_module()
    # Direct attribute access (not via a setter) must work
    assert isinstance(mod._DIRECTION_5_OB_MIN, float)
    assert isinstance(mod._DIRECTION_5_OI_MIN, float)
    assert isinstance(mod._DIRECTION_5_LOADS, tuple)
    assert isinstance(mod._DIRECTION_5_MIN_SEEDS, int)
    # No runtime override function exists
    assert not hasattr(mod, "set_thresholds")
    assert not hasattr(mod, "override_bar")


# ===================================================================
# 2. Happy-path verdicts
# ===================================================================

def test_pass_when_all_cells_clear_bar():
    """All 6 cells at 0.85 (>= 0.80) across 3 seeds -> PASS."""
    mod = _load_verdict_module()
    per_seed = _seeds_at(0.85, n_seeds=3)
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_PASS


def test_negative_when_no_cell_clears_bar():
    """All 6 cells at 0.30 (< 0.80) -> NEGATIVE."""
    mod = _load_verdict_module()
    per_seed = _seeds_at(0.30, n_seeds=3)
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_NEGATIVE


def test_partial_when_some_cells_clear():
    """OB cells pass, OI cells fail -> PARTIAL."""
    mod = _load_verdict_module()
    entry = {
        "L=2": {"OB": 0.90, "OI": 0.40},
        "L=3": {"OB": 0.85, "OI": 0.50},
        "L=5": {"OB": 0.82, "OI": 0.30},
    }
    per_seed = [entry, entry, entry]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_PARTIAL


def test_partial_when_one_load_fails_oi():
    """Only L=5 OI fails; everything else passes -> PARTIAL."""
    mod = _load_verdict_module()
    entry = {
        "L=2": {"OB": 0.85, "OI": 0.82},
        "L=3": {"OB": 0.85, "OI": 0.82},
        "L=5": {"OB": 0.85, "OI": 0.50},  # only this cell fails
    }
    per_seed = [entry, entry, entry]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_PARTIAL


# ===================================================================
# 3. Instrument-validity (VOID_MALFORMED)
# ===================================================================

def test_void_on_none_input():
    """None input -> VOID_MALFORMED."""
    mod = _load_verdict_module()
    assert mod.compute_verdict(None) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_empty_list():
    """Empty list (0 seeds) -> VOID."""
    mod = _load_verdict_module()
    assert mod.compute_verdict([]) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_fewer_than_min_seeds():
    """1-2 seeds (below MIN_SEEDS=3) -> VOID."""
    mod = _load_verdict_module()
    for n in (1, 2):
        per_seed = _seeds_at(0.99, n_seeds=n)
        assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED, (
            "n_seeds=" + str(n) + " should be VOID (< MIN_SEEDS=3)"
        )


def test_void_on_missing_load_key():
    """Missing L=3 entry -> VOID."""
    mod = _load_verdict_module()
    entry_missing_L3 = {
        "L=2": {"OB": 0.85, "OI": 0.85},
        "L=5": {"OB": 0.85, "OI": 0.85},
    }
    per_seed = [entry_missing_L3, entry_missing_L3, entry_missing_L3]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_missing_ob_in_load_entry():
    """Missing OB key inside a load entry -> VOID."""
    mod = _load_verdict_module()
    entry = {
        "L=2": {"OI": 0.85},  # missing OB
        "L=3": {"OB": 0.85, "OI": 0.85},
        "L=5": {"OB": 0.85, "OI": 0.85},
    }
    per_seed = [entry, entry, entry]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_nan_ob():
    """NaN OB value -> VOID."""
    mod = _load_verdict_module()
    entry = {
        "L=2": {"OB": float("nan"), "OI": 0.85},
        "L=3": {"OB": 0.85, "OI": 0.85},
        "L=5": {"OB": 0.85, "OI": 0.85},
    }
    per_seed = [entry, entry, entry]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_inf_oi():
    """Inf OI value -> VOID."""
    mod = _load_verdict_module()
    entry = {
        "L=2": {"OB": 0.85, "OI": float("inf")},
        "L=3": {"OB": 0.85, "OI": 0.85},
        "L=5": {"OB": 0.85, "OI": 0.85},
    }
    per_seed = [entry, entry, entry]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_negative_inf():
    """Negative Inf -> VOID."""
    mod = _load_verdict_module()
    entry = {
        "L=2": {"OB": 0.85, "OI": -math.inf},
        "L=3": {"OB": 0.85, "OI": 0.85},
        "L=5": {"OB": 0.85, "OI": 0.85},
    }
    per_seed = [entry, entry, entry]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_string_in_value_slot():
    """String in OB slot -> VOID."""
    mod = _load_verdict_module()
    entry = {
        "L=2": {"OB": "0.85", "OI": 0.85},  # string
        "L=3": {"OB": 0.85, "OI": 0.85},
        "L=5": {"OB": 0.85, "OI": 0.85},
    }
    per_seed = [entry, entry, entry]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_bool_in_value_slot():
    """bool in OI slot -> VOID (bool is a subclass of int but accuracy
    values must be true numeric, not 0/1 truthy)."""
    mod = _load_verdict_module()
    entry = {
        "L=2": {"OB": 0.85, "OI": True},  # bool sneaking in
        "L=3": {"OB": 0.85, "OI": 0.85},
        "L=5": {"OB": 0.85, "OI": 0.85},
    }
    per_seed = [entry, entry, entry]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_non_dict_seed_entry():
    """List-of-list (not dict) seed entry -> VOID."""
    mod = _load_verdict_module()
    per_seed = [
        [0.85, 0.85],  # not a dict
        _all_cells(0.85),
        _all_cells(0.85),
    ]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_string_seed_entry():
    """String seed entry -> VOID."""
    mod = _load_verdict_module()
    per_seed = [
        "L=2:OB=0.85 OI=0.85",
        _all_cells(0.85),
        _all_cells(0.85),
    ]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_none_seed_entry_among_valid():
    """One None seed entry among valid seeds -> VOID (defensive)."""
    mod = _load_verdict_module()
    per_seed = [_all_cells(0.85), None, _all_cells(0.85)]
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_string_per_seed_arg():
    """Top-level string (not list) -> VOID."""
    mod = _load_verdict_module()
    assert mod.compute_verdict("bad input") == mod.DIRECTION_5_VOID_MALFORMED


def test_void_on_dict_per_seed_arg():
    """Top-level dict (not list) -> VOID."""
    mod = _load_verdict_module()
    assert mod.compute_verdict({"per_seed": []}) == mod.DIRECTION_5_VOID_MALFORMED


# ===================================================================
# 4. Boundary cases (exact threshold)
# ===================================================================

def test_boundary_exactly_at_threshold():
    """Value == 0.80 counts as PASS (>=, not strict >)."""
    mod = _load_verdict_module()
    per_seed = _seeds_at(0.80, n_seeds=3)
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_PASS


def test_below_threshold_eps_fails():
    """Value 0.79 < 0.80 -> not PASS."""
    mod = _load_verdict_module()
    per_seed = _seeds_at(0.79, n_seeds=3)
    verdict = mod.compute_verdict(per_seed)
    # All 6 cells at 0.79 -> all below; should be NEGATIVE
    assert verdict == mod.DIRECTION_5_NEGATIVE


def test_multi_seed_mean_above_threshold_passes():
    """Per-seed mean comfortably above threshold via {0.71, 0.80, 0.90}
    (mean ~= 0.8033) -> PASS. Demonstrates the verdict averages across
    seeds (single seed at 0.71 would fail; combined > 0.80 passes)."""
    mod = _load_verdict_module()
    s_low = _all_cells(0.71)
    s_mid = _all_cells(0.80)
    s_high = _all_cells(0.90)
    per_seed = [s_low, s_mid, s_high]  # mean per cell ~= 0.8033
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_PASS


def test_multi_seed_mean_below_threshold_negative():
    """Per-seed mean below threshold via {0.70, 0.80, 0.89}
    (mean ~= 0.7966) -> NEGATIVE on all-cell case. Demonstrates the
    verdict averages across seeds (max is 0.89 but mean is below bar)."""
    mod = _load_verdict_module()
    s_low = _all_cells(0.70)
    s_mid = _all_cells(0.80)
    s_high = _all_cells(0.89)
    per_seed = [s_low, s_mid, s_high]  # mean per cell ~= 0.7966
    # Below bar -> NEGATIVE (all cells fail)
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_NEGATIVE


# ===================================================================
# 5. Fuzz-style (never raise on garbage)
# ===================================================================

def test_verdict_never_raises_on_malformed_garbage():
    """A pile of malformed inputs should all return VOID without
    raising."""
    mod = _load_verdict_module()
    garbage_inputs = [
        None,
        0,
        -1,
        3.14,
        True,
        False,
        "",
        "abc",
        [],
        {},
        [None, None, None],
        [True, False, True],
        [1.0, 2.0, 3.0],
        ["bad", "input", "fmt"],
        [[1, 2], [3, 4], [5, 6]],
        [{"only": "key"}, {"L=2": "wrong-type"}, {"L=3": 0.85}],
        [{"L=2": {"OB": None, "OI": None}}] * 3,
        [{"L=2": {"OB": [0.85], "OI": (0.85,)}}] * 3,
    ]
    for g in garbage_inputs:
        # Must not raise; must return VOID
        verdict = mod.compute_verdict(g)
        assert verdict == mod.DIRECTION_5_VOID_MALFORMED, (
            "Garbage input did not return VOID: " + repr(g)
            + " -> " + str(verdict)
        )


def test_more_than_3_seeds_pass_works():
    """5 seeds all at 0.85 -> PASS (the MIN_SEEDS=3 is a LOWER bound,
    not an upper bound)."""
    mod = _load_verdict_module()
    per_seed = _seeds_at(0.85, n_seeds=5)
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_PASS


def test_partial_with_just_oi_failing():
    """All OB pass; one OI fails -> PARTIAL."""
    mod = _load_verdict_module()
    per_seed = [
        {
            "L=2": {"OB": 0.85, "OI": 0.85},
            "L=3": {"OB": 0.85, "OI": 0.85},
            "L=5": {"OB": 0.85, "OI": 0.79},  # only this cell fails
        },
    ] * 3
    assert mod.compute_verdict(per_seed) == mod.DIRECTION_5_PARTIAL
