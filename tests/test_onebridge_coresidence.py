"""CI guard: the WKV read-out (cp_ssm_state + cp_ssm_readout_w) and the composer RF phasor (cp_rf_* + rf_kick/
resonate/read) CO-RESIDE on ONE bridge byte-identically to isolated, neither corrupting the other (2026-07-20).

The single-shared-substrate consolidation crux. GPU-only (RF ops + cupy); skips on numpy/CPU."""
import numpy as np
import pytest

from sim.backend import get_backend, is_gpu_backend, to_host


pytestmark = pytest.mark.skipif(not is_gpu_backend(), reason="RF co-residence de-risk is GPU-only")


def _run(seed=42, rounds=3, D=48):
    from research.runners._emerge_wkv_onbridge_derisk import _build_ssm_state_bridge
    from research.runners._gap_onebridge_coresidence_derisk import (
        _install_readout, _install_rf, _charge_read, _composer_op)
    xp, _ = get_backend()
    rng = np.random.default_rng(seed)
    decay = 0.9
    bA, *_ = _build_ssm_state_bridge(D, seed, decay, pop_k=1)
    n = bA.core_config.num_neurons
    WA = _install_readout(bA, xp, seed)
    bB, *_ = _build_ssm_state_bridge(D, seed, decay, pop_k=1)
    slice_idx = list(range(n - 16, n))
    _install_rf(bB, slice_idx, seed)
    bC, *_ = _build_ssm_state_bridge(D, seed, decay, pop_k=1)
    _install_readout(bC, xp, seed); _install_rf(bC, slice_idx, seed)
    injects = [rng.standard_normal(n) * 0.5 for _ in range(rounds)]
    ks = [seed * 1000 + r for r in range(rounds)]
    iso_r = [_charge_read(bA, xp, injects[r]) for r in range(rounds)]
    iso_p = [_composer_op(bB, xp, slice_idx, ks[r], 30, do_kick=True) for r in range(rounds)]
    co_r, co_p = [], []
    for r in range(rounds):
        co_r.append(_charge_read(bC, xp, injects[r]))
        co_p.append(_composer_op(bC, xp, slice_idx, ks[r], 30, do_kick=True))
    read_err = max(float(np.max(np.abs(co_r[r] - iso_r[r]))) for r in range(rounds))
    phase_err = max(float(np.max(np.abs(co_p[r] - iso_p[r]))) for r in range(rounds))
    # DISCRIMINATING v/u-sharing control: two no-rekick arms differing ONLY by whether the WKV step runs between
    # resonates (same round-0 kick). Shared v/u => arm A (with WKV step) diverges from arm B (without). Disjoint =>
    # identical. (Replaces the over-determined no-rekick-vs-isolated arm, which diverged from a kick-seed mismatch.)
    bD, *_ = _build_ssm_state_bridge(D, seed, decay, pop_k=1)
    _install_readout(bD, xp, seed); _install_rf(bD, slice_idx, seed)
    armA = [_composer_op(bD, xp, slice_idx, ks[0], 30, do_kick=True)]
    for r in range(1, rounds):
        _charge_read(bD, xp, injects[r])                     # WKV Izhikevich step
        armA.append(_composer_op(bD, xp, slice_idx, 0, 30, do_kick=False))
    bE, *_ = _build_ssm_state_bridge(D, seed, decay, pop_k=1)
    _install_rf(bE, slice_idx, seed)
    armB = [_composer_op(bE, xp, slice_idx, ks[0], 30, do_kick=True)]
    for r in range(1, rounds):
        armB.append(_composer_op(bE, xp, slice_idx, 0, 30, do_kick=False))   # NO WKV step
    vu_shared_err = max(float(np.max(np.abs(armA[r] - armB[r]))) for r in range(1, rounds))
    return read_err, phase_err, vu_shared_err


def test_wkv_readout_byte_clean_co_resident():
    read_err, _, _ = _run()
    assert read_err < 1e-5, f"WKV read-out diverged co-resident: {read_err}"


def test_composer_phase_byte_clean_co_resident():
    _, phase_err, _ = _run()
    assert phase_err < 1e-5, f"composer phase diverged co-resident: {phase_err}"


def test_vu_genuinely_shared_discriminating():
    # discriminating: with-WKV-step vs without-WKV-step (both no-rekick) must DIVERGE -> the WKV step corrupts the
    # SHARED v/u. If v/u were disjoint the two arms would be identical.
    _, _, vu_shared_err = _run()
    assert vu_shared_err > 1e-3, f"v/u-sharing control did NOT diverge ({vu_shared_err}) -- v/u not genuinely shared"
