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
    _check_sidecar_usable, _sidecar_mode, _sidecar_readout,
    aggregate_verdict,
)
from research.runners.song_g1_core import g1_verdict
from research.runners.song_g1_train import (
    _canonical_candidates, _p_ckpt_path, traj_top_rate,
)
from research.runners.song_g1_train import (
    _sidecar_mode as _train_sidecar_mode,
)


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


# --- Generator-P (--mode p): _sidecar_mode / _check_sidecar_usable ----
#     mode cross-mode refusal (Task 8). PURE: no IO, no bridge, CPU.

def test_sidecar_mode_default_is_songbird_legacy():
    # legacy G1/G1.5 sidecars predate the mode key -> "songbird"
    # regime (the trainer's additive-JSON back-compat contract,
    # exactly mirroring legacy-absent readout -> "final").
    meta = {"smoke": False, "calibration": {"g1_abstain": 72.0}}
    assert _sidecar_mode(meta) == "songbird"
    # default gate --mode is also songbird -> a legacy sidecar still
    # gates a songbird run (back-compat preserved).
    ok, reason = _check_sidecar_usable(meta)            # mode default
    assert ok is True and reason == "ok"
    # the gate's _sidecar_mode mirrors the trainer's EXACTLY.
    assert _sidecar_mode(meta) == _train_sidecar_mode(meta)


def test_sidecar_mode_top_level_then_calibration_source_of_truth():
    # top-level meta["mode"] is read when calibration.mode absent...
    assert _sidecar_mode({"mode": "p", "calibration": {}}) == "p"
    # ...but calibration.mode is the SINGLE source of truth (the
    # trainer mirrors it there; a stale top-level key must NOT win).
    meta = {"mode": "songbird",
            "calibration": {"g1_abstain": 41.0, "mode": "p"}}
    assert _sidecar_mode(meta) == "p"
    assert _train_sidecar_mode(meta) == "p"


def test_sidecar_p_regime_rejected_for_songbird_gate():
    # a P-regime frozen floor must NEVER gate a songbird run
    # (different decode regime -- same HARD-refusal class as the
    # smoke-tag rejection).
    meta = {"smoke": False, "mode": "p",
            "calibration": {"g1_abstain": 33.0, "mode": "p"}}
    ok, reason = _check_sidecar_usable(meta, mode="songbird")
    assert ok is False
    assert "mode" in reason.lower()
    # OK when the mode regimes MATCH (p<->p).
    ok2, reason2 = _check_sidecar_usable(meta, mode="p")
    assert ok2 is True and reason2 == "ok"


def test_sidecar_songbird_regime_rejected_for_p_gate():
    # the inverse: a songbird-regime floor must NEVER gate a P run.
    meta = {"smoke": False, "mode": "songbird",
            "calibration": {"g1_abstain": 72.0, "mode": "songbird"}}
    ok, reason = _check_sidecar_usable(meta, mode="p")
    assert ok is False
    assert "mode" in reason.lower()
    # matches when both songbird.
    ok2, reason2 = _check_sidecar_usable(meta, mode="songbird")
    assert ok2 is True and reason2 == "ok"


def test_sidecar_legacy_absent_mode_rejected_for_p_gate():
    # a legacy sidecar (no mode key -> "songbird") must be REFUSED by
    # a P gate (a songbird-regime floor can never gate a P run); but
    # still usable by a songbird gate (back-compat).
    meta = {"smoke": False, "calibration": {"g1_abstain": 72.0}}
    ok, reason = _check_sidecar_usable(meta, mode="p")
    assert ok is False and "mode" in reason.lower()
    ok2, _ = _check_sidecar_usable(meta, mode="songbird")
    assert ok2 is True


def test_sidecar_smoke_rejected_before_mode_check():
    # smoke rejection is checked BEFORE mode -- a smoke P sidecar is
    # rejected for the SMOKE reason (never gates the real verdict),
    # not silently accepted because the mode happens to match.
    meta = {"smoke": True, "mode": "p",
            "calibration": {"g1_abstain": 33.0, "mode": "p"}}
    ok, reason = _check_sidecar_usable(meta, mode="p")
    assert ok is False
    assert "smoke" in reason.lower()


def test_sidecar_mode_and_readout_both_checked_independently():
    # a P sidecar with readout=final gates a P run with --readout
    # final (mode matches, readout matches). The same P sidecar must
    # be refused by a songbird run (mode mismatch) regardless of
    # readout. mode is checked before readout (both are HARD refusals).
    meta = {"smoke": False, "mode": "p", "readout": "final",
            "calibration": {"g1_abstain": 33.0, "mode": "p",
                            "readout": "final"}}
    assert _check_sidecar_usable(meta, mode="p", readout="final")[0]
    ok, reason = _check_sidecar_usable(meta, mode="songbird",
                                       readout="final")
    assert ok is False and "mode" in reason.lower()


def test_sidecar_none_and_nondict_rejected_for_p_too():
    ok, reason = _check_sidecar_usable(None, mode="p")
    assert ok is False and "missing" in reason.lower()
    ok2, reason2 = _check_sidecar_usable(["x"], mode="p")
    assert ok2 is False and "malformed" in reason2.lower()


# --- Generator-P: _canonical_candidates is NOT target-ordered --------
#     LOAD-BEARING anti-cheat (carry-forward Minor #4): PredictiveCoder
#     .select_next tie-breaks to the FIRST candidate. The builder MUST
#     return a FIXED range(n)-style order independent of any target.

def test_canonical_candidates_is_fixed_range_not_target_ordered():
    # the builder is exactly list(range(n)) -- the natural vocab index
    # order, INDEPENDENT of any intended/target sequence.
    assert _canonical_candidates(5) == [0, 1, 2, 3, 4]
    assert _canonical_candidates(1) == [0]
    assert _canonical_candidates(64) == list(range(64))
    # it does NOT depend on / echo any target sequence: for several
    # distinct "targets" the candidate list is the SAME fixed order
    # (so PredictiveCoder.select_next's first-candidate tie-break can
    # NEVER correlate with the target for a degenerate predictor).
    n = 8
    fixed = _canonical_candidates(n)
    for target in ([7, 3], [3, 7], [0, 1], [5, 5], [6, 2, 4]):
        # whatever the target order, the builder ignores it entirely.
        assert _canonical_candidates(n) == fixed == list(range(n))
        assert _canonical_candidates(n) != list(target)  # not target


def test_canonical_candidates_deterministic_and_int_list():
    a = _canonical_candidates(10)
    b = _canonical_candidates(10)
    assert a == b == list(range(10))
    assert all(isinstance(x, int) for x in a)
    # accepts an int-like and coerces (range(int(n))).
    assert _canonical_candidates(3) == [0, 1, 2]


# --- Generator-P: _p_ckpt_path isolation + PRECEDENCE ----------------

def test_p_ckpt_path_inserts_pc_infix():
    # '.pc' infix before the '.ckpt.npz' (or '.npz') suffix -- the
    # EXACT idiom as _smoke_ckpt_path / _traj_ckpt_path.
    assert _p_ckpt_path("a/b/song_g1.ckpt.npz") == \
        "a/b/song_g1.pc.ckpt.npz"
    assert _p_ckpt_path("x.npz") == "x.pc.npz"
    assert _p_ckpt_path("noext") == "noext.pc"


def test_p_ckpt_path_composes_with_smoke_and_supersedes_traj():
    # PRECEDENCE: --mode p SUPERSEDES --readout for the path infix.
    # The trainer/gate apply '.pc' FIRST (NOT '.traj'), then '.smoke'.
    # Re-derive the documented composition purely.
    from research.runners.song_g1_train import (
        _smoke_ckpt_path, _traj_ckpt_path, _CKPT_DEFAULT,
    )
    base = _CKPT_DEFAULT
    # --mode p -> '.pc' (NOT '.traj')
    p_path = _p_ckpt_path(base)
    assert ".pc." in p_path and ".traj." not in p_path
    # --mode p --smoke -> '.pc.smoke'
    p_smoke = _smoke_ckpt_path(_p_ckpt_path(base))
    assert p_smoke.endswith(".pc.smoke.ckpt.npz")
    # songbird --readout trajectory is UNCHANGED ('.traj', no '.pc')
    traj = _traj_ckpt_path(base)
    assert ".traj." in traj and ".pc." not in traj
    # the three are all DISTINCT namespaces (no collision)
    assert len({base, p_path, p_smoke, traj}) == 4


# --- import / signature surface is smoke-able WITHOUT a checkpoint ----

def test_module_imports_clean_and_exposes_pure_surface():
    import research.runners.song_g1_gate as g
    for name in ("main", "aggregate_verdict", "_check_sidecar_usable",
                 "_sidecar_mode", "_sidecar_readout"):
        assert hasattr(g, name), name
    # numpy is only needed by the pure-test fixtures, not the module
    # top -- a no-op presence check keeping the import deterministic.
    assert np.array([1]).sum() == 1
