"""Task 1 test matrix: phase-factored cheap-first falsification probe.

Pins the EXACT model transcribed in the Task 1 spec, as STRENGTHENED so
the residual-coupling outcome is GENUINELY MEASURED (not closed-form):

  - single-pass CONTROL must reproduce the encode-order conflict
    (best achievable min(wm, ep) stays below the frozen 0.90 bar) -- this
    half is a CLOSED-FORM assumption grounded in the already-validated
    selectivity-needs-shuffle finding;
  - the two-phase index-resolution (idx_fidelity for pointer /
    content-no-update / content-update) is MEASURED from real vectors via
    the common-mode-removal separation transform + nearest-match
    resolution. Its outcome CAN surprise -- the test asserts the measured
    RELATIONS that must hold by construction (update never hurts; the
    transform moves the reps; an identity pointer always resolves), NOT a
    predetermined verdict. If the measured content-no-update stays high
    (residual coupling weaker than the old closed form assumed) that is a
    LEGITIMATE outcome, and the verdict will report coupling_demonstrated
    accordingly.
  - the frozen three-state, fail-closed `probe_verdict` is exercised with
    HAND-BUILT synthetic dicts (unchanged) so the verdict logic is pinned
    independently of whatever the real measurement yields.

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
_MEASURE_FIELDS = (
    "idx_pointer",
    "idx_content_noupdate",
    "idx_content_update",
    "mean_move",
    "sep_gain",
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
#    This half stays a CLOSED-FORM assumption (selectivity needs shuffle).
# ---------------------------------------------------------------------------
def test_single_pass_best_below_bar_each_seed():
    for s in SEEDS:
        out = probe.run_probe(s)
        assert out["single_pass_best"] < 0.90, (
            "single-pass control did not reproduce the conflict at seed %d" % s
        )


# ---------------------------------------------------------------------------
# 4. two_phase_content_update >= 0.90 for each seed.
#    (Updating the index to the moved rep must resolve perfectly.)
# ---------------------------------------------------------------------------
def test_two_phase_content_update_clears_bar_each_seed():
    for s in SEEDS:
        out = probe.run_probe(s)
        assert out["two_phase_content_update"] >= 0.90, (
            "content-update variant failed to clear the bar at seed %d" % s
        )


# ---------------------------------------------------------------------------
# 5. update never hurts no-update. The clean MEASURED RELATION is asserted
#    on measure_index_resolution (test_measure_update_never_hurts_noupdate);
#    here at the run_probe level the two fields each carry INDEPENDENT
#    +-0.01 jitter, so when the underlying measured idx values tie they can
#    flip order by up to the noise envelope (2 * 0.01). The assertion
#    therefore tolerates that envelope -- it only fails if the GAP exceeds
#    what noise can explain (i.e. a genuine update-hurts-resolution event).
# ---------------------------------------------------------------------------
def test_noupdate_at_most_update_within_noise_each_seed():
    noise_envelope = 0.02 + 1e-9  # two independent +-0.01 draws.
    for s in SEEDS:
        out = probe.run_probe(s)
        assert (
            out["two_phase_content_noupdate"]
            <= out["two_phase_content_update"] + noise_envelope
        ), "update genuinely HURT resolution at seed %d (beyond noise)" % s


# ---------------------------------------------------------------------------
# 6. two_phase_pointer >= 0.90 for each seed.
#    An identity pointer is immune to representational drift by construction.
# ---------------------------------------------------------------------------
def test_two_phase_pointer_clears_bar_each_seed():
    for s in SEEDS:
        out = probe.run_probe(s)
        assert out["two_phase_pointer"] >= 0.90, (
            "pointer variant failed to clear the bar at seed %d" % s
        )


# ---------------------------------------------------------------------------
# 7. measure_index_resolution: genuine measurement contract.
# ---------------------------------------------------------------------------
def test_measure_index_resolution_returns_finite_floats():
    m = probe.measure_index_resolution(42)
    assert isinstance(m, dict)
    for field in _MEASURE_FIELDS:
        assert field in m, "missing measurement field: " + field
        assert _is_finite_float(m[field]), field + " is not a finite float"


def test_measure_index_resolution_is_deterministic():
    assert probe.measure_index_resolution(42) == probe.measure_index_resolution(42)


def test_measure_pointer_resolution_is_perfect():
    # An identity pointer resolves trivially -> idx_pointer == 1.0 exactly.
    for s in SEEDS:
        m = probe.measure_index_resolution(s)
        assert m["idx_pointer"] == 1.0, (
            "identity pointer must resolve perfectly at seed %d" % s
        )


def test_measure_update_never_hurts_noupdate():
    # MEASURED RELATION: re-pointing the index at the moved rep can only
    # help -- idx_content_update >= idx_content_noupdate at every seed.
    for s in SEEDS:
        m = probe.measure_index_resolution(s)
        assert m["idx_content_update"] >= m["idx_content_noupdate"] - 1e-9, (
            "update HURT index fidelity at seed %d" % s
        )


def test_measure_transform_moves_reps():
    # The common-mode-removal separation transform must genuinely MOVE the
    # reps -- mean_move > 0 at every seed (else it would be a no-op).
    for s in SEEDS:
        m = probe.measure_index_resolution(s)
        assert m["mean_move"] > 0.0, (
            "separation transform did not move reps at seed %d" % s
        )


def test_measure_idx_fidelities_in_unit_range():
    for s in SEEDS:
        m = probe.measure_index_resolution(s)
        for f in ("idx_pointer", "idx_content_noupdate", "idx_content_update"):
            assert 0.0 <= m[f] <= 1.0, "%s out of [0, 1] at seed %d" % (f, s)


# ---------------------------------------------------------------------------
# 8. run_probe consumes the MEASURED idx_fidelity (not a constant).
#    Pin the wiring: the two-phase variants equal min(wm(0.70), ep(measured))
#    BEFORE noise, i.e. they track the measurement within the +-0.01 jitter.
# ---------------------------------------------------------------------------
def test_run_probe_two_phase_tracks_measurement():
    for s in SEEDS:
        m = probe.measure_index_resolution(s)
        out = probe.run_probe(s)
        wm07 = probe._wm(probe._SEP_PHASE2)
        for variant, idx_key in (
            ("two_phase_pointer", "idx_pointer"),
            ("two_phase_content_noupdate", "idx_content_noupdate"),
            ("two_phase_content_update", "idx_content_update"),
        ):
            expected = min(wm07, probe._ep(m[idx_key]))
            # within the +-0.01 per-seed noise (allow a hair for float).
            assert abs(out[variant] - expected) <= 0.011 + 1e-9, (
                "%s did not track measured %s at seed %d (got %.4f, "
                "expected ~%.4f)" % (variant, idx_key, s, out[variant], expected)
            )


# ---------------------------------------------------------------------------
# 9. probe_verdict on the real per-seed runs: assert the OUTCOME that the
#    real measurement actually produces -- do NOT force RESOLVES if the
#    measurement disagrees. With single_pass_best < 0.90 (instrument valid)
#    and measured content_update >= 0.90, the verdict is RESOLVES; whether
#    coupling_demonstrated is True depends on the measured no-update value.
# ---------------------------------------------------------------------------
def test_probe_verdict_on_real_runs_is_consistent_with_measurement():
    per_seed = [probe.run_probe(s) for s in SEEDS]
    v = probe.probe_verdict(per_seed)
    assert v["frozen_bar"] == 0.90
    # Instrument is valid (control reproduced the conflict).
    assert v["verdict"] != "CANNOT_CONCLUDE", v
    means = v["means"]
    # The verdict band must match the measured content_update mean.
    tp = means["two_phase_content_update"]
    if tp >= 0.90:
        assert v["verdict"] == "RESOLVES", v
    elif tp >= 0.80:
        assert v["verdict"] == "BOUNDARY", v
    else:
        assert v["verdict"] == "DOES_NOT_RESOLVE", v
    # coupling_demonstrated must equal (mean no-update < bar) -- the verdict
    # reports the real coupling strength, not a forced True.
    expected_coupling = means["two_phase_content_noupdate"] < 0.90
    assert v.get("coupling_demonstrated") == expected_coupling, v


# ---------------------------------------------------------------------------
# 10. instrument-validity: single_pass_best=0.95 -> CANNOT_CONCLUDE.
#     Hand-built synthetic dicts -> verdict logic pinned independently.
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
# 11. malformed inputs -> CANNOT_CONCLUDE (no raise).
# ---------------------------------------------------------------------------
def test_malformed_none_empty_str():
    for bad in (None, [], "x"):
        v = probe.probe_verdict(bad)
        assert v["verdict"] == "CANNOT_CONCLUDE", repr(bad)
        assert "reason" in v
        assert v["frozen_bar"] == 0.90


# ---------------------------------------------------------------------------
# 12. < 3 seeds -> CANNOT_CONCLUDE.
# ---------------------------------------------------------------------------
def test_too_few_seeds():
    v = probe.probe_verdict([_valid_entry(), _valid_entry()])
    assert v["verdict"] == "CANNOT_CONCLUDE", v
    v1 = probe.probe_verdict([_valid_entry()])
    assert v1["verdict"] == "CANNOT_CONCLUDE", v1


# ---------------------------------------------------------------------------
# 13. non-finite field (NaN / inf / "str" / True) -> CANNOT_CONCLUDE.
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
# 14. bar-edge pins: RESOLVES / BOUNDARY / DOES_NOT_RESOLVE.
#     Synthetic hand-built inputs -> unchanged verdict logic.
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


def test_coupling_demonstrated_flag_tracks_noupdate():
    # Hand-built: a LOW no-update mean -> coupling_demonstrated True.
    low = [_valid_entry(tp_update=0.93, tp_noupdate=0.60) for _ in range(3)]
    assert probe.probe_verdict(low).get("coupling_demonstrated") is True
    # A HIGH no-update mean (coupling weak) -> coupling_demonstrated False.
    high = [_valid_entry(tp_update=0.93, tp_noupdate=0.99) for _ in range(3)]
    assert probe.probe_verdict(high).get("coupling_demonstrated") is False


# ---------------------------------------------------------------------------
# 15. _PROBE_BAR == 0.90 and is module-level.
# ---------------------------------------------------------------------------
def test_probe_bar_frozen_module_level():
    assert hasattr(probe, "_PROBE_BAR")
    assert probe._PROBE_BAR == 0.90
    assert hasattr(probe, "_PROBE_MIN_SEEDS")
    assert probe._PROBE_MIN_SEEDS == 3


# ---------------------------------------------------------------------------
# 16. verdict dict always contains keys: verdict, reason, frozen_bar.
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
# 17. The exposed scalar diagnostics match the model exactly.
# ---------------------------------------------------------------------------
def test_wm_at_sep07_and_ep_pointer_are_model_calibrated():
    # wm(0.70) = min(1, 0.5 + 0.75*0.70) = 1.0 (before +-0.01 noise).
    # ep_pointer maps measured idx_pointer = 1.0 -> ep = 1.0 (before noise).
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
    # New measurement constants.
    assert probe.BETA == 0.6
    assert probe.N_EPISODES == 200
