"""Fix-round tests for the prospective-memory live cliff detector (v2).

Each test encodes one adversarial-review issue and is written to FAIL on the v1 design (commit f30ab4666) and PASS
on v2. The before/after runs are recorded in research/findings/raw/_pmem_live_cliff_detector_v2/fix_proofs_*.txt.
No brain build: every test runs the controller or detector against a synthetic plant, or reads committed files.
"""
import json
import os
import re

import pytest

import research.runners._pmem_live_cliff_detector_derisk as M

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
V2 = os.path.join(REPO, "research", "findings", "raw", "_pmem_live_cliff_detector_v2")


def _run(plant, g_init=6000.0):
    """Drive whichever controller the module ships against a synthetic plant (v1 has no `measure=` hook, so its
    module-level read is monkeypatched for the same effect)."""
    if hasattr(M, "cusum_step"):
        return M.run_live_cliff_homeostat(0, g_init, 0.001, 0.01, measure=lambda s, g: plant(g))
    orig = M._measure_rel
    M._measure_rel = lambda s, g: plant(g)
    try:
        return M.run_live_cliff_homeostat(0, g_init)
    finally:
        M._measure_rel = orig


# ---- issue 1: the reported quantity must be the controller's settled state, not an arg-max ----
def test_reported_state_is_the_settled_iterate_not_the_running_best():
    # rises to a peak at 7000, then declines gently with 1-step recoveries (no cliff, no 2-step decline) to the
    # cap: an arg-max read-out reports 7000; a settled controller sits where the loop actually stopped.
    table = {6000: 0.250, 6500: 0.262, 7000: 0.270, 7500: 0.266, 8000: 0.268, 8500: 0.264, 9000: 0.266,
             9500: 0.262, 10000: 0.264, 10500: 0.260, 11000: 0.262}
    r = _run(lambda g: table[int(round(g / 500.0) * 500)])
    last = r["trajectory"][-1]
    assert r["final_fac_g"] == last["fac_g"], "reported gain is not the last iterate (arg-max read-out)"
    assert r["final_rel"] == last["rel"], "reported rel is not the one measured at the settled gain"


# ---- issue 2: detector constants must come from seeds outside the evaluation set ----
def test_detector_constants_are_sourced_outside_every_evaluation_seed():
    src = set(getattr(M, "CONSTANT_SOURCE_SEEDS", ()))
    assert src, "no declared source seeds for the detector constants"
    evals = set(getattr(M, "SEEDS_CANONICAL", getattr(M, "SEEDS", ()))) | set(getattr(M, "SEEDS_HELDOUT", ()))
    assert not (src & evals), f"detector constants sourced from evaluation seeds {sorted(src & evals)}"
    assert set(getattr(M, "SEEDS_HELDOUT", ())), "no held-out evaluation seeds"


def test_freeze_is_robust_to_a_cliff_in_the_calibration_scans():
    import random
    rng = random.Random(1)
    n = len(M.LATTICE)
    base = {s: [0.25 + 0.001 * i + rng.gauss(0, 0.004) for i in range(n)] for s in M.SEEDS_CALIB}
    cliffy = {s: list(v) for s, v in base.items()}
    cliffy[M.SEEDS_CALIB[0]][7:] = [x - 0.08 for x in cliffy[M.SEEDS_CALIB[0]][7:]]
    a, b = M.freeze_constants(base)["sigma"], M.freeze_constants(cliffy)["sigma"]
    assert abs(a - b) / a < 0.25, f"one cliff moved the jitter scale {a} -> {b}"


def test_committed_frozen_constants_reproduce_from_committed_calibration_scans():
    fr_path = os.path.join(V2, "frozen_constants.json")
    if not os.path.exists(fr_path):
        pytest.skip("constants not frozen yet (calibration scans still running)")
    fr = json.load(open(fr_path))
    scans = {s: [json.load(open(os.path.join(V2, f"calib_s{s}.json")))["scan"][str(g)] for g in M.LATTICE]
             for s in M.SEEDS_CALIB}
    again = M.freeze_constants(scans)
    for key in ("sigma", "cusum_k", "cusum_h"):
        assert again[key] == fr[key], key
    assert fr["seeds"] == sorted(M.SEEDS_CALIB)


# ---- issue 3: a null for the detector, with the seed as the exchangeable unit ----
def test_permutation_null_separates_real_cliffs_from_chance_and_is_undefined_without_alarms():
    import random
    rng = random.Random(7)
    n = len(M.LATTICE)
    cliffs = [[0.22 + 0.002 * i + rng.gauss(0, 0.001) - (0.05 if i >= 6 + (j % 3) else 0) for i in range(n)]
              for j in range(8)]
    res = M.null_test(cliffs, 0.0015, 0.015, n_perm=500, rng_seed=3)
    assert res["defined"] and res["p"] < 0.05, res
    rising = [[0.22 + 0.002 * i for i in range(n)] for _ in range(6)]
    assert M.null_test(rising, 0.0015, 0.015, n_perm=50)["defined"] is False


# ---- gate: can fail, UNDEFINED is never a pass ----
def _row(**kw):
    base = {"converged_both": True, "same_setpoint": True, "margin_positive": True, "margin_ge_default": True,
            "load_bearing": True, "frozen_clause_fails": [], "domain_bounded": True, "rerun_identical": True,
            "climber": True}
    base.update(kw)
    return base


def test_gate_goes_only_when_everything_holds_and_can_fail():
    held = {s: _row() for s in M.SEEDS_HELDOUT}
    canon = {s: _row() for s in M.SEEDS_CANONICAL}
    null_ok = {"defined": True, "p": 0.01}
    dof = {"exact_equal_all": True, "negative_control_differs": True}
    assert M.decide_gate(held, canon, null_ok, dof)["status"] == "GO"
    bad = dict(held)
    bad[M.SEEDS_HELDOUT[0]] = _row(margin_ge_default=False)
    assert M.decide_gate(bad, canon, null_ok, dof)["status"] == "NO-GO"
    assert M.decide_gate(held, canon, {"defined": True, "p": 0.2}, dof)["status"] == "NO-GO"
    assert M.decide_gate(held, canon, {"defined": False, "p": None}, dof)["status"] == "UNDEFINED"
    assert M.decide_gate(held, canon, null_ok, {"exact_equal_all": True,
                                                "negative_control_differs": False})["status"] == "UNDEFINED"
    no_climb = {s: _row(climber=False) for s in M.SEEDS_HELDOUT}
    assert M.decide_gate(no_climb, canon, null_ok, dof)["status"] == "UNDEFINED"
    assert M.decide_gate({}, {}, None, None)["status"] == "UNDEFINED"


# ---- issue 4: byte-identity asserted in DATA against a pinned SHA, with a compare that can fail ----
def test_default_off_byte_identity_is_asserted_in_data_against_a_pinned_sha():
    p = os.path.join(V2, "default_off_compare.json")
    assert os.path.exists(p), "no default-off exact-compare artifact"
    d = json.load(open(p))
    assert re.fullmatch(r"[0-9a-f]{40}", d["pinned_pre_change_sha"])
    assert d["pinned_pre_change_sha"] == M.PINNED_PRE_CHANGE_SHA
    assert d["pinned_organ_has_cliff_flag"] is False and d["head_organ_has_cliff_flag"] is True
    for name in ("shipped_default", "facilitation_on"):
        c = d["configs"][name]
        assert c["exact_equal"] is True and c["head_sha256"] == c["pinned_sha256"], name
    assert d["negative_control_differs"] is True


# ---- issue 5: prereg integrity ----
def test_v2_preregistration_has_an_amendment_log_and_matches_the_code():
    p = os.path.join(REPO, "research", "findings", "2026-09-23-pmem-live-cliff-detector-v2-PREREGISTRATION.md")
    assert os.path.exists(p), "no v2 pre-registration"
    t = open(p).read()
    assert "AMENDMENT LOG" in t
    for s in list(M.SEEDS_CALIB) + list(M.SEEDS_HELDOUT):
        assert str(s) in t
    assert f"k = {M.CUSUM_K_SIGMA}" in t and f"h = {M.CUSUM_H_SIGMA}" in t
    assert M.PINNED_PRE_CHANGE_SHA in t


# ---- issue 6: the pool isolated-revision gap is logged with its closing gate ----
def test_failure_log_records_the_pool_queue_isolated_revision_gap():
    lines = open(os.path.join(REPO, "research", "FAILURE_LOG.md")).read().splitlines()
    assert any("pool_queue.sh" in ln and "REMOTE_DIR" in ln and re.search(r"isolated.revision", ln, re.I)
               for ln in lines), "the pool_queue.sh isolated-revision probe gap is not logged with its gate"
