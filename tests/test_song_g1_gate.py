"""Pure-CPU TDD for the pre-registered G1 held-out gate's PURE logic.

`aggregate_verdict` and `_check_sidecar_usable` are pure (no file IO, no
heavy import) so they are unit-testable on CPU without a checkpoint or a
GPU. The gate's REAL validation is the post-training run (Task 10) -- but
the aggregate rule (g1_verdict on the means), the EXCLUDED handling, and
the smoke/missing-sidecar refusal are pure and pre-registered, so they
ARE testable now (the only thing testable pre-run).

Anti-cheat invariants exercised here:
  * AGGREGATE = the UNMODIFIED g1_verdict on the MEANS (no new rule).
  * any held-out prop that did NOT clear its gate -> all_gate_cleared
    False -> aggregate FAIL.
  * EXCLUDED props (no permuted-ORDER control existed) are NEVER counted
    as PASS (consistent with g1_verdict's best_perm_score>0 guard).
  * a smoke-tagged OR missing sidecar is REJECTED (must never gate the
    real verdict; never fall back to 650 / recompute).
"""
import numpy as np

from research.runners.song_g1_gate import (
    _check_sidecar_usable, _sidecar_readout, aggregate_verdict,
)
from research.runners.song_g1_core import g1_verdict
from research.runners.song_g1_train import traj_top_rate


# --- aggregate_verdict: g1_verdict on the means -----------------------

def test_aggregate_all_pass_props_is_pass():
    # every counted prop clears its gate, strong true vs weak permuted;
    # the AGGREGATE must equal g1_verdict on the means and PASS.
    per = [
        {"intention": 0, "true_score": 0.95, "best_perm_score": 0.40,
         "gate_cleared": True, "excluded": False},
        {"intention": 1, "true_score": 0.90, "best_perm_score": 0.45,
         "gate_cleared": True, "excluded": False},
    ]
    agg = aggregate_verdict(per)
    # aggregate == g1_verdict on the means (NOT a new rule)
    ref = g1_verdict((0.95 + 0.90) / 2.0, (0.40 + 0.45) / 2.0, True)
    assert agg["GATE"] == "PASS"
    assert agg["gate"] == ref["gate"]
    assert abs(agg["true_score"] - ref["true_score"]) < 1e-9
    assert abs(agg["best_perm_score"] - ref["best_perm_score"]) < 1e-9
    assert agg["n_counted"] == 2 and agg["n_excluded"] == 0
    assert agg["all_gate_cleared"] is True
    assert agg["n_prop_pass"] == 2


def test_aggregate_any_gate_not_cleared_is_fail():
    # one counted prop did NOT clear its gate -> all_gate_cleared False
    # -> aggregate FAIL even though the score gap looks great.
    per = [
        {"intention": 0, "true_score": 0.99, "best_perm_score": 0.30,
         "gate_cleared": True, "excluded": False},
        {"intention": 1, "true_score": 0.99, "best_perm_score": 0.30,
         "gate_cleared": False, "excluded": False},
    ]
    agg = aggregate_verdict(per)
    assert agg["all_gate_cleared"] is False
    assert agg["GATE"] == "FAIL"
    # mirrors g1_verdict(mean_true, mean_perm, all_gate_cleared=False)
    ref = g1_verdict(0.99, 0.30, False)
    assert agg["gate"] == ref["gate"] and ref["gate"] is False


def test_aggregate_excluded_props_not_counted_as_pass():
    # one strong PASS-able prop + one EXCLUDED (no permuted control).
    # The excluded prop must NOT be counted; aggregate runs on the
    # single counted prop only.
    per = [
        {"intention": 0, "true_score": 0.95, "best_perm_score": 0.40,
         "gate_cleared": True, "excluded": False},
        {"intention": 1, "true_score": 1.0, "best_perm_score": 0.0,
         "gate_cleared": True, "excluded": True},
    ]
    agg = aggregate_verdict(per)
    assert agg["n_props"] == 2
    assert agg["n_excluded"] == 1
    assert agg["n_counted"] == 1
    # means computed over the 1 counted prop only (excluded ignored)
    assert abs(agg["mean_true_score"] - 0.95) < 1e-9
    assert abs(agg["mean_best_perm_score"] - 0.40) < 1e-9
    assert 1 in agg["excluded_props"]
    # the excluded prop's perfect-but-uncontrolled score did NOT leak
    # in as a PASS contribution.
    assert agg["GATE"] == "PASS"  # from the genuine counted prop only


def test_aggregate_all_excluded_is_fail_no_evidence():
    # every held-out prop excluded (degenerate multisets) -> NO
    # ORDER-learning evidence -> aggregate FAIL with all-zero means
    # (reuses g1_verdict's own ts=ps=0 / gate=False FAIL).
    per = [
        {"intention": 0, "true_score": 1.0, "best_perm_score": 0.0,
         "gate_cleared": True, "excluded": True},
        {"intention": 1, "true_score": 1.0, "best_perm_score": 0.0,
         "gate_cleared": True, "excluded": True},
    ]
    agg = aggregate_verdict(per)
    assert agg["n_counted"] == 0
    assert agg["GATE"] == "FAIL"
    assert agg["mean_true_score"] == 0.0
    assert agg["mean_best_perm_score"] == 0.0
    assert agg["all_gate_cleared"] is False
    ref = g1_verdict(0.0, 0.0, False)
    assert agg["gate"] == ref["gate"] and ref["gate"] is False


def test_aggregate_empty_is_fail():
    agg = aggregate_verdict([])
    assert agg["GATE"] == "FAIL"
    assert agg["n_props"] == 0 and agg["n_counted"] == 0


def test_aggregate_below_abs_floor_fails_even_with_relative_edge():
    # both means tiny but true technically +>=10% over permuted; the
    # FIXED _G1_ABS_FLOOR=0.5 (inside the UNMODIFIED g1_verdict) must
    # still block it -> aggregate FAIL (not a real generative claim).
    per = [
        {"intention": 0, "true_score": 0.12, "best_perm_score": 0.10,
         "gate_cleared": True, "excluded": False},
        {"intention": 1, "true_score": 0.12, "best_perm_score": 0.10,
         "gate_cleared": True, "excluded": False},
    ]
    agg = aggregate_verdict(per)
    assert agg["GATE"] == "FAIL"
    # exactly g1_verdict on the means -- bar not touched here.
    ref = g1_verdict(0.12, 0.10, True)
    assert agg["gate"] == ref["gate"] and ref["gate"] is False


def test_aggregate_zero_mean_permuted_is_fail_not_pass():
    # counted props all clear the gate, perfect true, but mean permuted
    # == 0 -> NO ORDER contrast -> g1_verdict-on-means FAILs (the
    # best_perm_score>0 guard) -- aggregate must NOT spuriously PASS.
    per = [
        {"intention": 0, "true_score": 1.0, "best_perm_score": 0.0,
         "gate_cleared": True, "excluded": False},
        {"intention": 1, "true_score": 1.0, "best_perm_score": 0.0,
         "gate_cleared": True, "excluded": False},
    ]
    agg = aggregate_verdict(per)
    assert agg["n_counted"] == 2          # NOT excluded -- counted
    assert agg["mean_best_perm_score"] == 0.0
    assert agg["GATE"] == "FAIL"
    assert agg["pct_over_permuted"] == 0.0


# --- _check_sidecar_usable: smoke / missing refusal -------------------

def test_sidecar_none_is_rejected():
    ok, reason = _check_sidecar_usable(None)
    assert ok is False
    assert "missing" in reason.lower()


def test_sidecar_smoke_true_is_rejected():
    meta = {"smoke": True,
            "calibration": {"g1_abstain": 72.0}}
    ok, reason = _check_sidecar_usable(meta)
    assert ok is False
    assert "smoke" in reason.lower()


def test_sidecar_full_with_g1_abstain_is_ok():
    meta = {"smoke": False,
            "calibration": {"g1_abstain": 72.0,
                            "operating_criterion": "control_max"}}
    ok, reason = _check_sidecar_usable(meta)
    assert ok is True
    assert reason == "ok"


def test_sidecar_missing_calibration_is_rejected():
    # no calibration.g1_abstain -> Step-0 floor not frozen -> reject
    # (must NOT fall back to 650 / recompute).
    assert _check_sidecar_usable({"smoke": False})[0] is False
    assert _check_sidecar_usable(
        {"smoke": False, "calibration": {}})[0] is False
    assert _check_sidecar_usable(
        {"smoke": False, "calibration": {"foo": 1}})[0] is False


def test_sidecar_absent_smoke_key_treated_as_full():
    # the trainer records smoke as additive JSON; an absent key means
    # full=False (trainer's own contract) -> usable if g1_abstain set.
    meta = {"calibration": {"g1_abstain": 80.0}}
    ok, _reason = _check_sidecar_usable(meta)
    assert ok is True


def test_sidecar_malformed_non_dict_is_rejected():
    ok, reason = _check_sidecar_usable(["not", "a", "dict"])
    assert ok is False
    assert "malformed" in reason.lower()


# --- G1.5: _check_sidecar_usable readout cross-mode refusal -----------

def test_sidecar_final_regime_rejected_for_trajectory_gate():
    # a final-regime frozen floor must NEVER gate a trajectory run
    # (different decode magnitude regime -- same HARD-refusal class as
    # the smoke-tag rejection).
    meta = {"smoke": False, "readout": "final",
            "calibration": {"g1_abstain": 72.0}}
    ok, reason = _check_sidecar_usable(meta, readout="trajectory")
    assert ok is False
    assert "readout" in reason.lower()
    # still OK when the readout regimes MATCH (final<->final).
    ok2, reason2 = _check_sidecar_usable(meta, readout="final")
    assert ok2 is True and reason2 == "ok"


def test_sidecar_trajectory_regime_rejected_for_final_gate():
    # the inverse: a trajectory-regime floor must NEVER gate a final
    # run. The trainer mirrors readout inside calibration{} (the
    # single source of truth _sidecar_readout reads).
    meta = {"smoke": False, "readout": "trajectory",
            "calibration": {"g1_abstain": 41.0,
                            "readout": "trajectory",
                            "traj_rate_rule": "min"}}
    ok, reason = _check_sidecar_usable(meta, readout="final")
    assert ok is False
    assert "readout" in reason.lower()
    # matches when both trajectory.
    ok2, reason2 = _check_sidecar_usable(meta, readout="trajectory")
    assert ok2 is True and reason2 == "ok"


def test_sidecar_calibration_readout_is_source_of_truth():
    # _step0_calibrate writes readout INSIDE calibration{}; that is the
    # single source of truth (top-level meta['readout'] is a mirror).
    # A calibration.readout=trajectory must be honored even if a stale
    # top-level key disagreed.
    meta = {"smoke": False, "readout": "final",
            "calibration": {"g1_abstain": 41.0,
                            "readout": "trajectory",
                            "traj_rate_rule": "min"}}
    assert _sidecar_readout(meta) == "trajectory"
    ok, _ = _check_sidecar_usable(meta, readout="trajectory")
    assert ok is True
    ok2, reason2 = _check_sidecar_usable(meta, readout="final")
    assert ok2 is False and "readout" in reason2.lower()


def test_sidecar_default_readout_is_final_legacy():
    # legacy G1 sidecars predate the readout key -> "final" regime
    # (the trainer's additive-JSON contract). Default gate --readout
    # is also final, so a legacy sidecar still gates a final run.
    meta = {"smoke": False, "calibration": {"g1_abstain": 72.0}}
    assert _sidecar_readout(meta) == "final"
    # default readout arg == "final" -> usable (back-compat preserved).
    ok, reason = _check_sidecar_usable(meta)
    assert ok is True and reason == "ok"
    # but a trajectory gate must REFUSE that legacy final sidecar.
    ok2, reason2 = _check_sidecar_usable(meta, readout="trajectory")
    assert ok2 is False and "readout" in reason2.lower()


def test_sidecar_smoke_still_rejected_regardless_of_readout():
    # smoke rejection is checked BEFORE readout -- a smoke trajectory
    # sidecar is rejected for the smoke reason (never gates the real
    # verdict), not silently accepted because readout happens to match.
    meta = {"smoke": True, "readout": "trajectory",
            "calibration": {"g1_abstain": 41.0,
                            "readout": "trajectory",
                            "traj_rate_rule": "min"}}
    ok, reason = _check_sidecar_usable(meta, readout="trajectory")
    assert ok is False
    assert "smoke" in reason.lower()


def test_sidecar_none_rejected_for_trajectory_too():
    ok, reason = _check_sidecar_usable(None, readout="trajectory")
    assert ok is False
    assert "missing" in reason.lower()


# --- G1.5: traj_top_rate -- the pre-registered MIN-per-slot rule -----

def test_traj_top_rate_empty_is_zero():
    # empty rates_list (no slots) -> 0.0 (cannot clear any floor).
    assert traj_top_rate([]) == 0.0


def test_traj_top_rate_is_min_per_slot():
    # the documented rule is the MINIMUM per-slot accumulated rate:
    # the production is "confident" only if EVERY slot cleared the
    # floor (mirrors compose_reward's no-confabulation moat).
    assert traj_top_rate([100.0, 50.0, 80.0]) == 50.0
    assert traj_top_rate([5.0]) == 5.0
    assert traj_top_rate([7.0, 7.0, 7.0]) == 7.0
    # NOT max, NOT mean.
    assert traj_top_rate([10.0, 20.0, 30.0]) != 30.0  # not max
    assert traj_top_rate([10.0, 20.0, 30.0]) != 20.0  # not mean


def test_traj_top_rate_deterministic_and_float():
    rates = [12.0, 3.0, 9.0, 3.0]
    a = traj_top_rate(rates)
    b = traj_top_rate(rates)
    assert a == b == 3.0
    assert isinstance(a, float)


def test_traj_top_rate_accepts_ints_returns_float():
    # rates may arrive as ints; the rule coerces to float (a single
    # scalar compared against the frozen float g1_abstain).
    out = traj_top_rate([4, 2, 9])
    assert out == 2.0
    assert isinstance(out, float)


# --- import / signature surface is smoke-able WITHOUT a checkpoint ----

def test_module_imports_clean_and_exposes_pure_surface():
    import research.runners.song_g1_gate as g
    for name in ("main", "aggregate_verdict", "_check_sidecar_usable"):
        assert hasattr(g, name), name
    # numpy is only needed by the pure-test fixtures, not the module
    # top -- a no-op presence check keeping the import deterministic.
    assert np.array([1]).sum() == 1
