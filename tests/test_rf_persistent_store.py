"""CI guard: the PERSISTENT RF store (cp_rf_store_re/im + rf_set_store_weights, 2026-07-20) -- a fact-store that lives
IN device synapses, DISTINCT from the per-op cp_rf_w_* so a per-op bind/kick never wipes it, summed additively into
the RF matvec (disjoint rows => no corruption), with the megakernel bailing to the loop when a store is present.
Byte-identical when off. GPU-only; skips on numpy."""
import numpy as np
import pytest

from sim.backend import is_gpu_backend, to_host

pytestmark = pytest.mark.skipif(not is_gpu_backend(), reason="RF persistent store is GPU-path")


def _bridge():
    from research.runners._emerge_wkv_onbridge_derisk import _build_ssm_state_bridge
    b, *_ = _build_ssm_state_bridge(16, 42, 0.9, pop_k=1)     # n = 32
    return b


def test_store_survives_per_op_bind_and_kick():
    b = _bridge(); n = b.core_config.num_neurons
    b.rf_set_store_weights([(16 + i, 16 + i, complex(0.9, 0.0)) for i in range(8)])   # store rows [16..24)
    h0 = hash(np.asarray(to_host(b.cp_rf_store_re.data)).tobytes())
    b.rf_set_complex_weights([(i, i, complex(0.5, 0.1)) for i in range(8)])           # per-op bind rows [0..8)
    kick = np.zeros(n, np.complex128); kick[:8] = 1.0 + 0.0j; kick[16:24] = np.exp(1j * np.linspace(0, 2, 8))
    b.rf_kick(kick, period=100)
    assert b.cp_rf_store_re is not None, "store wiped by rf_set_complex_weights/rf_kick"
    assert hash(np.asarray(to_host(b.cp_rf_store_re.data)).tobytes()) == h0, "store data changed by a per-op op"


def test_store_matvec_is_applied_in_loop():
    b = _bridge(); n = b.core_config.num_neurons
    b.rf_set_store_weights([(16 + i, 16 + i, complex(0.9, 0.0)) for i in range(8)])
    b.rf_set_complex_weights([(i, i, complex(0.5, 0.1)) for i in range(8)])
    kick = np.zeros(n, np.complex128); kick[:8] = 1.0; kick[16:24] = np.exp(1j * np.linspace(0, 2, 8))
    b.rf_kick(kick, period=100)
    b._rf_advance_one()
    re = np.asarray(to_host(b.cp_membrane_potential_v))
    assert np.any(np.abs(re[16:24]) > 1e-6), "store-readout neurons not driven by Store@z (store matvec not applied)"


def test_store_off_is_deterministic_byte_identical():
    # with NO store the additive block is skipped -> two identical runs are byte-identical (the block is inert).
    def run():
        b = _bridge(); n = b.core_config.num_neurons
        b.rf_set_complex_weights([(i, i, complex(0.5, 0.1)) for i in range(8)])
        kick = np.zeros(n, np.complex128); kick[:8] = np.exp(1j * np.linspace(0, 1, 8))
        b.rf_kick(kick, period=100)
        b.rf_resonate_steps(40)
        return np.asarray(to_host(b.cp_membrane_potential_v)).tobytes()
    assert run() == run(), "store-off RF path is not deterministic (additive block leaked when store None)"


def test_megakernel_bails_when_store_present():
    # with a store installed + cudagraph enabled, rf_resonate_steps must BAIL to the loop (which applies BOTH matvecs),
    # NOT the store-blind megakernel -> the store-readout neurons are driven (the megakernel would leave them at kick).
    b = _bridge(); n = b.core_config.num_neurons
    b.core_config.enable_rf_cudagraph = True
    b.rf_set_store_weights([(16 + i, 16 + i, complex(0.9, 0.0)) for i in range(8)])
    b.rf_set_complex_weights([(i, i, complex(0.5, 0.1)) for i in range(8)])
    kick = np.zeros(n, np.complex128); kick[:8] = 1.0; kick[16:24] = np.exp(1j * np.linspace(0, 2, 8))
    b.rf_kick(kick, period=100)
    called = {"mega": False}
    _orig = type(b)._rf_resonate_steps_megakernel

    def _spy(self, ns):
        called["mega"] = True
        return _orig(self, ns)
    type(b)._rf_resonate_steps_megakernel = _spy
    try:
        b.rf_resonate_steps(5)
    finally:
        type(b)._rf_resonate_steps_megakernel = _orig
    assert not called["mega"], "megakernel was used despite a persistent store (store term would be dropped)"


def test_persistent_store_read_fidelity_matches_staged():
    # Phase-2 de-risk: a PERSISTENT store (kept driving the readout via the additive term through the unbind window)
    # reads the SAME filler as the STAGED store swapped out -- the RF read is phase-based + magnitude-invariant.
    from research.runners._gap_persistent_store_readfidelity_derisk import (
        _build_rf_encoder, _phasor, _circ_absdiff, _read_Q)
    D = 48; n = 1 + 2 * D; per = 200
    rng = np.random.default_rng(7)
    r = rng.uniform(0, 1, D); f = rng.uniform(0, 1, D); composite = (r + f) % 1.0
    store = [(1 + k, 0, _phasor(composite[k])) for k in range(D)]
    unbind = [(1 + D + k, 1 + k, _phasor(-r[k])) for k in range(D)]
    kick = np.zeros(n, np.complex128); kick[0] = 1.0
    bS = _build_rf_encoder(n, seed=7)
    bS.rf_set_complex_weights(store); bS.rf_kick(kick, period=per, lam=0.0); bS.rf_resonate_steps(per + 8)
    bS.rf_set_complex_weights(unbind); bS.rf_resonate_steps(per + 8)
    Qs = _read_Q(bS, D)
    bP = _build_rf_encoder(n, seed=7)
    bP.rf_set_store_weights(store); bP.rf_set_complex_weights(unbind)
    bP.rf_kick(kick, period=per, lam=0.0); bP.rf_resonate_steps(per + 8)
    Qp = _read_Q(bP, D)
    assert _circ_absdiff(Qs, f) < 0.08, "staged did not decode the filler"
    assert _circ_absdiff(Qp, f) < 0.08, "persistent did not decode the filler"
    assert _circ_absdiff(Qp, Qs) < 0.05, "persistent store read DIVERGES from staged (read-fidelity risk real)"
