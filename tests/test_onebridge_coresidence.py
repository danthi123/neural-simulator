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
    # no-rekick divergence
    bD, *_ = _build_ssm_state_bridge(D, seed, decay, pop_k=1)
    _install_readout(bD, xp, seed); _install_rf(bD, slice_idx, seed)
    nk = []
    for r in range(rounds):
        _charge_read(bD, xp, injects[r])
        nk.append(_composer_op(bD, xp, slice_idx, ks[r], 30, do_kick=(r == 0)))
    nk_err = max(float(np.max(np.abs(nk[r] - iso_p[r]))) for r in range(1, rounds))
    return read_err, phase_err, nk_err


def test_wkv_readout_byte_clean_co_resident():
    read_err, _, _ = _run()
    assert read_err < 1e-5, f"WKV read-out diverged co-resident: {read_err}"


def test_composer_phase_byte_clean_co_resident():
    _, phase_err, _ = _run()
    assert phase_err < 1e-5, f"composer phase diverged co-resident: {phase_err}"


def test_shared_vu_is_real_no_rekick_diverges():
    # anti-cheat: without the composer re-kick, the WKV's Izhikevich step corrupts the shared v/u -> divergence.
    _, _, nk_err = _run()
    assert nk_err > 1e-3, f"no-rekick did NOT diverge ({nk_err}) -- v/u not genuinely shared (suspect)"
