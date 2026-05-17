"""Pure CPU adversarial tests for the FIXED-bar Dendritic verdict.
Mirrors the generator_h_core adversarial discipline. Bars are immutable
to results; the LOAD-BEARING 2026-05-03 permuted-label control + the
biologically-local assertion + the oracle-alignment cosine are
fail-closed."""
import pytest
from research.runners import dendritic_core as c


def test_frozen_bars_exact():
    assert c._DEND_GRAD_COSINE_MIN == 0.30
    assert c._DEND_HIDDEN_CREDIT_MIN == 0.90
    assert c._DEND_NOHIDDEN_FLOOR_MAX == 0.70
    assert c._DEND_PERMUTED_MAX == 0.70
    assert c._DEND_MIN_SEEDS == 3


def _good(**kw):
    base = dict(hidden_credit=0.95, nohidden_floor=0.55,
                permuted=0.55, grad_cosine=0.59,
                biologically_local=True, has_permuted_control=True)
    base.update(kw)
    return c.dend_verdict(**base)


def test_verdict_pass_when_all_bars_met():
    assert _good()["GATE"] == "PASS"


def test_no_permuted_control_fails_closed():
    assert _good(has_permuted_control=False)["GATE"] == "FAIL"


def test_not_biologically_local_fails_closed():
    assert _good(biologically_local=False)["GATE"] == "FAIL"


def test_permuted_control_clears_bar_fails():
    # permuted >  _DEND_PERMUTED_MAX => the result is NOT real
    assert _good(permuted=0.95)["GATE"] == "FAIL"


def test_nohidden_floor_too_high_fails():
    # task does not genuinely require hidden credit
    assert _good(nohidden_floor=0.95)["GATE"] == "FAIL"


def test_did_not_learn_fails():
    assert _good(hidden_credit=0.60)["GATE"] == "FAIL"


def test_no_gradient_alignment_fails():
    assert _good(grad_cosine=0.10)["GATE"] == "FAIL"


def test_non_finite_fails_closed():
    assert _good(grad_cosine=float("nan"))["GATE"] == "FAIL"
    assert _good(hidden_credit=float("inf"))["GATE"] == "FAIL"


def test_results_cannot_move_fixed_bars():
    c.dend_verdict(9.9, 9.9, 9.9, 9.9, True, True)
    assert c._DEND_GRAD_COSINE_MIN == 0.30
    assert c._DEND_HIDDEN_CREDIT_MIN == 0.90
    assert c._DEND_PERMUTED_MAX == 0.70


def test_aggregate_requires_three_seeds():
    one = [c.dend_verdict(0.95, 0.55, 0.55, 0.59, True, True)]
    one[0]["n_seed_probes"] = 1
    assert c.dend_aggregate_multiseed(one)["GATE"] == "FAIL"


def test_aggregate_pass_three_good_seeds():
    seeds = []
    for _ in range(3):
        v = c.dend_verdict(0.95, 0.55, 0.55, 0.59, True, True)
        v["n_seed_probes"] = 1
        seeds.append(v)
    assert c.dend_aggregate_multiseed(seeds)["GATE"] == "PASS"
