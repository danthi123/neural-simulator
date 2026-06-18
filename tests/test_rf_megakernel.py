"""RF resonate megakernel (cfg.enable_rf_cudagraph) == the per-step loop. The megakernel does the whole resonate
step (complex sparse matvec + dynamics) in one custom CUDA kernel/step; it must produce the same phase read as the
loop (float32 + the crossing-step is an int, so we allow a tiny tolerance for boundary neurons). GPU-only.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import is_gpu_backend
from research.runners.rf_phasor_composer import _build_rf_bridge

pytestmark = pytest.mark.skipif(not is_gpu_backend(), reason="RF megakernel is GPU-only (numpy falls back to loop)")


def _phases_both_ways(b, conns, kick, period=200):
    b.rf_set_complex_weights(conns)
    b.core_config.enable_rf_cudagraph = False
    b.rf_kick(kick, period=period)
    b.rf_resonate_steps(period + 8)
    ph_loop = b.rf_read_phases().copy()
    b.core_config.enable_rf_cudagraph = True
    b.rf_kick(kick, period=period)
    b.rf_resonate_steps(period + 8)
    ph_mega = b.rf_read_phases().copy()
    b.core_config.enable_rf_cudagraph = False
    return ph_loop, ph_mega


def test_megakernel_matches_loop_bind():
    """Bind = a permutation (post D+k <- pre k x phasor) -> 1 nonzero/row, so the megakernel matvec is exact."""
    D, rng = 128, np.random.default_rng(42)
    b = _build_rf_bridge(2 * D, 42)
    z = np.exp(2j * np.pi * rng.uniform(0, 1, D))
    conns = [(D + k, k, complex(z[k])) for k in range(D)]
    kick = np.zeros(2 * D, dtype=np.complex128)
    kick[:D] = np.exp(2j * np.pi * rng.uniform(0, 1, D))
    ph_loop, ph_mega = _phases_both_ways(b, conns, kick)
    assert np.max(np.abs(ph_loop - ph_mega)) < 1e-2, f"max phase diff {np.max(np.abs(ph_loop - ph_mega))}"


def test_megakernel_matches_loop_bundle():
    """Bundle = a strided sum (post k <- sum_l pre l*D+k) -> L nonzeros/row, exercising the matvec accumulation."""
    D, L, rng = 64, 3, np.random.default_rng(7)
    n = (L + 1) * D
    b = _build_rf_bridge(n, 7)
    conns = [(L * D + k, l * D + k, 1.0 + 0j) for l in range(L) for k in range(D)]
    kick = np.zeros(n, dtype=np.complex128)
    for l in range(L):
        kick[l * D:(l + 1) * D] = np.exp(2j * np.pi * rng.uniform(0, 1, D))
    ph_loop, ph_mega = _phases_both_ways(b, conns, kick)
    assert np.max(np.abs(ph_loop - ph_mega)) < 1e-2, f"max phase diff {np.max(np.abs(ph_loop - ph_mega))}"


def test_megakernel_matches_loop_masked():
    """A5 lever 3 (the masked-megakernel): with a co-residence `neuron_mask`, the megakernel writes ONLY the masked
    (RF) neurons (== the masked `_rf_advance_one` loop) and leaves the non-masked (co-resident Izhikevich stand-in)
    neurons' state UNCHANGED. Gate: the masked RF phases match the loop AND the non-masked neurons keep a sentinel."""
    from sim.backend import to_host
    D, rng = 64, np.random.default_rng(11)
    n = 3 * D                                                  # 2*D RF (masked bind) + D non-masked co-resident stand-in
    mask = np.zeros(n, dtype=bool); mask[:2 * D] = True
    z = np.exp(2j * np.pi * rng.uniform(0, 1, D))
    conns = [(D + k, k, complex(z[k])) for k in range(D)]       # bind WITHIN the RF slice [0, 2D)
    kick = np.zeros(n, dtype=np.complex128); kick[:D] = np.exp(2j * np.pi * rng.uniform(0, 1, D))
    SENTINEL = 1.2345                                           # a known non-masked value that must survive the resonate

    def run(mega):
        b = _build_rf_bridge(n, 11); b.rf_set_complex_weights(conns)
        b.core_config.enable_rf_cudagraph = mega
        b.rf_kick(kick, period=200, neuron_mask=mask)
        b.cp_membrane_potential_v[2 * D:] = SENTINEL           # set the non-masked slice AFTER the masked kick
        b.rf_resonate_steps(208)
        ph = b.rf_read_phases().copy()
        v_nonmasked = np.asarray(to_host(b.cp_membrane_potential_v))[2 * D:].copy()
        b.core_config.enable_rf_cudagraph = False
        return ph, v_nonmasked

    ph_loop, v_loop = run(False)
    ph_mega, v_mega = run(True)
    assert np.max(np.abs(ph_loop[:2 * D] - ph_mega[:2 * D])) < 1e-2, \
        f"masked RF phases must match the loop, max diff {np.max(np.abs(ph_loop[:2 * D] - ph_mega[:2 * D]))}"
    assert np.allclose(v_mega, SENTINEL), "the masked megakernel must leave non-masked neurons UNCHANGED"
    assert np.allclose(v_loop, SENTINEL), "the masked loop must leave non-masked neurons unchanged (sanity)"


def test_default_off_uses_loop():
    """With the flag off (default), rf_resonate_steps must use the loop (byte-identical to before this feature)."""
    D = 64
    b = _build_rf_bridge(2 * D, 42)
    assert b.core_config.enable_rf_cudagraph is False
    z = np.exp(2j * np.pi * np.random.default_rng(1).uniform(0, 1, D))
    b.rf_set_complex_weights([(D + k, k, complex(z[k])) for k in range(D)])
    kick = np.zeros(2 * D, dtype=np.complex128)
    kick[:D] = z
    b.rf_kick(kick, period=64)
    b.rf_resonate_steps(72)            # must not raise; uses the loop path
    assert b.rf_read_phases().shape[0] == 2 * D
