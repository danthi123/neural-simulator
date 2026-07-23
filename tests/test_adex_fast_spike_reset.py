"""Byte-identity guard for the AdEx `fast_spike_reset` path (added 2026-07-23).

The AdEx branch of `_run_one_simulation_step` (sim/bridge.py) now has a fast-reset path mirroring the Izhikevich one:
`cp.where` masked-update (V->V_r, w->w+b, refractory off-by-one) instead of the legacy `cp.where(fired)[0]` index
materialization + fancy-index scatter + boolean-mask decrement -> removes the per-step device->host syncs. It must be
byte-identical to the legacy AdEx reset.

Scope note: at the small scale here the AdEx step has a tiny run-to-run non-determinism (~1 ULP by ~150 steps; two
IDENTICAL-config bridges show it too -- it is the sparse-matvec atomic reorder, NOT the reset). So this test compares
fast-off vs fast-on over a SHORT window (before that background drift accumulates), where the reset's byte-identity is
exact -- that isolates the reset from the substrate's own non-determinism. GPU-only (the sync removal is a cupy concern;
on numpy `cp.where` is not a sync but the path is still exercised).
"""
import numpy as np
import pytest

from sim.backend import is_gpu_backend, to_host


def _build_adex(fast_spike_reset):
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.enums import NeuronModel
    cfg = CoreSimConfig(
        num_neurons=500, connections_per_neuron=50, seed=42, dt_ms=0.5,
        neuron_model_type=NeuronModel.ADEX.name,
        fast_spike_reset=fast_spike_reset,
        ou_std_current_pA=0.0,          # OU off -> the two bridges share the same (zero) noise
        enable_hebbian_learning=False, enable_short_term_plasticity=False, enable_homeostasis=False,
        enable_stdp=False, enable_structural_plasticity=False, enable_reward_modulation=False,
    )
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig(enable_profiling=False))
    b._initialize_simulation_data()
    return b


def _run_capture(b, n_steps):
    fired = []
    for _ in range(n_steps):
        b.cp_external_input_current[:] = 2000.0     # drive AdEx past V_peak so the reset path actually fires
        b._run_one_simulation_step()
        fired.append(to_host(b.cp_firing_states).copy())
    return (np.stack(fired), to_host(b.cp_membrane_potential_v).copy(),
            to_host(b.cp_adex_w).copy(), to_host(b.cp_refractory_timers).copy())


@pytest.mark.skipif(not is_gpu_backend(), reason="fast_spike_reset sync-removal is a cupy/GPU concern")
def test_adex_fast_spike_reset_byte_identical():
    n_steps = 10                                    # short window: before the substrate's ~1-ULP background drift
    fired_off, v_off, w_off, refr_off = _run_capture(_build_adex(False), n_steps)
    fired_on, v_on, w_on, refr_on = _run_capture(_build_adex(True), n_steps)

    assert fired_off.sum() > 0, "no AdEx spikes fired -- test is vacuous, raise the drive"
    assert np.array_equal(fired_off, fired_on), (
        "AdEx fired raster DIFFERS fast-reset off vs on -- not byte-identical "
        f"(first diff at step {int(np.argmax(np.any(fired_off != fired_on, axis=1)))})")
    assert np.array_equal(v_off, v_on), f"AdEx membrane v differs (max|dv|={np.abs(v_off - v_on).max():.2e})"
    assert np.array_equal(w_off, w_on), f"AdEx adaptation w differs (max|dw|={np.abs(w_off - w_on).max():.2e})"
    assert np.array_equal(refr_off, refr_on), "AdEx refractory timers differ"
