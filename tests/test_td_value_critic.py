import numpy as np
import pytest
from sim.td_value_critic import (
    analytic_vstar, csc_features, run_pavlovian,
    GAMMA, TRACE, CS_ONSETS, scale_free_transfer)


def test_analytic_vstar_is_exact_discounted_return():
    v = analytic_vstar()
    assert v.shape == (TRACE + 1,)
    np.testing.assert_allclose(v, [GAMMA ** (TRACE - k)
                                   for k in range(TRACE + 1)], rtol=1e-12)


def test_csc_features_bias_plus_cue_anchored_taps():
    X = csc_features(onset=4, T=20)
    assert X.shape == (20, 21)
    assert np.allclose(X[:, 20], 1.0)            # bias always on
    assert X[3].sum() == 1.0                     # pre-cue: bias only
    assert X[4, 0] == 1.0                         # tap-0 at onset
    assert X[5, 1] == 1.0                         # tap-1 next step


def test_scale_free_transfer_is_one_when_us_predicted():
    assert scale_free_transfer(0.46, 0.0) == pytest.approx(1.0, abs=1e-9)
    assert scale_free_transfer(0.1, 0.9) < 0.2


def test_V1_td_critic_converges_to_analytic_vstar():
    vr, tr, ud = run_pavlovian("td", seed=42)
    assert vr <= 0.05
    assert tr >= 0.90
    assert ud <= 0.15


def test_controls_genuinely_fail():
    for mode in ("no_bootstrap", "permuted", "wrongsign"):
        vr, tr, ud = run_pavlovian(mode, seed=42)
        passes = (np.isfinite(vr) and np.isfinite(tr) and np.isfinite(ud)
                  and tr >= 0.90 and ud <= 0.15)
        assert not passes, mode


def test_no_autograd_imported():
    import sim.td_value_critic as m
    src = open(m.__file__).read()
    assert "autograd" not in src and "torch" not in src


def test_reuses_eligibility_kernel_unmodified():
    import sim.td_value_critic as m
    src = open(m.__file__).read()
    assert "fused_eligibility_trace_decay" in src
