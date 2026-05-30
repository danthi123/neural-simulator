"""Task 1 test matrix: phase-factored cheap-first falsification probe.

Pins the EXACT model transcribed in the Task 1 spec:
  - single-pass CONTROL must reproduce the encode-order conflict
    (best achievable min(wm, ep) stays below the frozen 0.90 bar),
  - the two-phase treatment with index-UPDATE clears the bar,
  - the two-phase content-NO-update variant stays below it
    (demonstrating the residual coupling is real),
  - the frozen three-state, fail-closed `probe_verdict`.

stdlib + numpy ONLY. No project/protected import. Plain ASCII.
Deterministic given seed.
"""
from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHEAP_PROBE_PATH = os.path.join(
    REPO_ROOT,
    "research/findings/raw/phase_factored_cheap_probe.py",
)


def _load_probe():
    """Load the probe module by absolute path (same importlib pattern the
    grounding pin uses)."""
    spec = importlib.util.spec_from_file_location(
        "phase_factored_cheap_probe", _CHEAP_PROBE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()

SEEDS = (42, 43, 44)
_REQUIRED_FIELDS = (
    "single_pass_best",
    "two_phase_pointer",
    "two_phase_content_noupdate",
    "two_phase_content_update",
    "wm_at_sep07",
    "ep_pointer",
)


def _is_finite_float(x):
    return isinstance(x, float) and math.isfinite(x)


# ---------------------------------------------------------------------------
# 1. run_probe(42) returns all six finite floats in [0, 1].
# ---------------------------------------------------------------------------
def test_run_probe_returns_six_finite_floats_in_unit_range():
    out = probe.run_probe(42)
    assert isinstance(out, dict)
    for field in _REQUIRED_FIELDS:
        assert field in out, "missing field: " + field
        val = out[field]
        assert _is_finite_float(val), field + " is not a finite float"
        assert 0.0 <= val <= 1.0, field + " out of [0, 1]"
    # Exactly the six advertised fields (no surprise extras break the contract).
    assert set(out.keys()) == set(_REQUIRED_FIELDS)


# ---------------------------------------------------------------------------
# 2. determinism: run_probe(42) == run_probe(42).
# ---------------------------------------------------------------------------
def test_run_probe_is_deterministic():
    a = probe.run_probe(42)
    b = probe.run_probe(42)
    assert a == b


def test_run_probe_differs_across_seeds():
    # Per-seed deterministic noise should make seeds differ at least somewhere.
    a = probe.run_probe(42)
    b = probe.run_probe(43)
    assert a != b


# ---------------------------------------------------------------------------
# 3. single_pass_best < 0.90 for each seed in SEEDS (conflict reproduced).
# ---------------------------------------------------------------------------
def test_single_pass_best_below_bar_each_seed():
    for s in SEEDS:
        out = probe.run_probe(s)
        assert out["single_pass_best"] < 0.90, (
            "single-pass control did not reproduce the conflict at seed %d" % s
        )


# ---------------------------------------------------------------------------
# 4. two_phase_content_update >= 0.90 for each seed.
# ---------------------------------------------------------------------------
def test_two_phase_content_update_clears_bar_each_seed():
    for s in SEEDS:
        out = probe.run_probe(s)
        assert out["two_phase_content_update"] >= 0.90, (
            "content-update variant failed to clear the bar at seed %d" % s
        )


# ---------------------------------------------------------------------------
# 5. two_phase_content_noupdate < two_phase_content_update for each seed.
# ---------------------------------------------------------------------------
def test_noupdate_below_update_each_seed():
    for s in SEEDS:
        out = probe.run_probe(s)
        assert (
            out["two_phase_content_noupdate"] < out["two_phase_content_update"]
        ), "coupling not demonstrated at seed %d" % s


def test_two_phase_content_noupdate_below_bar_each_seed():
    # The no-update variant must fall below the bar (coupling is real).
    for s in SEEDS:
        out = probe.run_probe(s)
        assert out["two_phase_content_noupdate"] < 0.90, (
            "no-update variant unexpectedly cleared the bar at seed %d" % s
        )


# ---------------------------------------------------------------------------
# 6. two_phase_pointer >= 0.90 for each seed.
# ---------------------------------------------------------------------------
def test_two_phase_pointer_clears_bar_each_seed():
    for s in SEEDS:
        out = probe.run_probe(s)
        assert out["two_phase_pointer"] >= 0.90, (
            "pointer variant failed to clear the bar at seed %d" % s
        )


# ---------------------------------------------------------------------------
# 7. probe_verdict on the real per-seed runs -> RESOLVES.
# ---------------------------------------------------------------------------
def test_probe_verdict_resolves_on_real_runs():
    per_seed = [probe.run_probe(s) for s in SEEDS]
    v = probe.probe_verdict(per_seed)
    assert v["verdict"] == "RESOLVES", v
    assert v["frozen_bar"] == 0.90
    assert v.get("coupling_demonstrated") is True


# ---------------------------------------------------------------------------
# 8. instrument-validity: single_pass_best=0.95 -> CANNOT_CONCLUDE.
# ---------------------------------------------------------------------------
def _valid_entry(single_pass=0.6, tp_update=0.93, tp_noupdate=0.79):
    return {
        "single_pass_best": single_pass,
        "two_phase_pointer": 1.0,
        "two_phase_content_noupdate": tp_noupdate,
        "two_phase_content_update": tp_update,
        "wm_at_sep07": 1.0,
        "ep_pointer": 1.0,
    }


def test_instrument_validity_control_failed_to_reproduce_conflict():
    bad = [_valid_entry(single_pass=0.95) for _ in range(3)]
    v = probe.probe_verdict(bad)
    assert v["verdict"] == "CANNOT_CONCLUDE", v
    assert v["frozen_bar"] == 0.90


# ---------------------------------------------------------------------------
# 9. malformed inputs -> CANNOT_CONCLUDE (no raise).
# ---------------------------------------------------------------------------
def test_malformed_none_empty_str():
    for bad in (None, [], "x"):
        v = probe.probe_verdict(bad)
        assert v["verdict"] == "CANNOT_CONCLUDE", repr(bad)
        assert "reason" in v
        assert v["frozen_bar"] == 0.90


# ---------------------------------------------------------------------------
# 10. < 3 seeds -> CANNOT_CONCLUDE.
# ---------------------------------------------------------------------------
def test_too_few_seeds():
    v = probe.probe_verdict([_valid_entry(), _valid_entry()])
    assert v["verdict"] == "CANNOT_CONCLUDE", v
    v1 = probe.probe_verdict([_valid_entry()])
    assert v1["verdict"] == "CANNOT_CONCLUDE", v1


# ---------------------------------------------------------------------------
# 11. non-finite field (NaN / inf / "str" / True) -> CANNOT_CONCLUDE.
# ---------------------------------------------------------------------------
def test_non_finite_field_rejected():
    for bad_val in (float("nan"), float("inf"), float("-inf"), "str", True):
        entries = [_valid_entry() for _ in range(3)]
        entries[1]["two_phase_content_update"] = bad_val
        v = probe.probe_verdict(entries)
        assert v["verdict"] == "CANNOT_CONCLUDE", repr(bad_val)
        assert v["frozen_bar"] == 0.90


def test_missing_field_rejected():
    entries = [_valid_entry() for _ in range(3)]
    del entries[0]["two_phase_content_update"]
    v = probe.probe_verdict(entries)
    assert v["verdict"] == "CANNOT_CONCLUDE", v


# ---------------------------------------------------------------------------
# 12. bar-edge pins: RESOLVES / BOUNDARY / DOES_NOT_RESOLVE.
# ---------------------------------------------------------------------------
def test_bar_edge_resolves():
    entries = [_valid_entry(single_pass=0.6, tp_update=0.93) for _ in range(3)]
    v = probe.probe_verdict(entries)
    assert v["verdict"] == "RESOLVES", v


def test_bar_edge_boundary():
    entries = [_valid_entry(single_pass=0.6, tp_update=0.85) for _ in range(3)]
    v = probe.probe_verdict(entries)
    assert v["verdict"] == "BOUNDARY", v


def test_bar_edge_does_not_resolve():
    entries = [_valid_entry(single_pass=0.6, tp_update=0.70) for _ in range(3)]
    v = probe.probe_verdict(entries)
    assert v["verdict"] == "DOES_NOT_RESOLVE", v


# ---------------------------------------------------------------------------
# 13. _PROBE_BAR == 0.90 and is module-level.
# ---------------------------------------------------------------------------
def test_probe_bar_frozen_module_level():
    assert hasattr(probe, "_PROBE_BAR")
    assert probe._PROBE_BAR == 0.90
    assert hasattr(probe, "_PROBE_MIN_SEEDS")
    assert probe._PROBE_MIN_SEEDS == 3


# ---------------------------------------------------------------------------
# 14. verdict dict always contains keys: verdict, reason, frozen_bar.
# ---------------------------------------------------------------------------
def test_verdict_dict_has_core_keys_all_paths():
    cases = [
        None,
        [],
        "x",
        [_valid_entry()],  # too few
        [_valid_entry(single_pass=0.95) for _ in range(3)],  # instrument fail
        [_valid_entry(tp_update=0.93) for _ in range(3)],  # resolves
        [_valid_entry(tp_update=0.85) for _ in range(3)],  # boundary
        [_valid_entry(tp_update=0.70) for _ in range(3)],  # does not resolve
        [probe.run_probe(s) for s in SEEDS],  # real
    ]
    for c in cases:
        v = probe.probe_verdict(c)
        for key in ("verdict", "reason", "frozen_bar"):
            assert key in v, "missing %s for case %r" % (key, c)
        assert v["frozen_bar"] == 0.90


# ---------------------------------------------------------------------------
# Extra coverage: the exposed scalar diagnostics match the model exactly.
# ---------------------------------------------------------------------------
def test_wm_at_sep07_and_ep_pointer_are_model_calibrated():
    # wm(0.70) = min(1, 0.5 + 0.75*0.70) = 1.0 (before +-0.01 noise).
    # ep_pointer maps idx_fidelity = 1.0 -> ep = 1.0 (before noise).
    # With +-0.01 clipped noise both land within [0.99, 1.0].
    for s in SEEDS:
        out = probe.run_probe(s)
        assert 0.99 <= out["wm_at_sep07"] <= 1.0
        assert 0.99 <= out["ep_pointer"] <= 1.0


def test_module_constants_present():
    assert probe.N_CONCEPTS == 16
    assert probe.D == 64
    assert probe.N == 2
    assert probe.N_TRIALS == 400
    assert probe.N_DISTRACTORS == 4
    assert tuple(probe.SEEDS) == (42, 43, 44)
