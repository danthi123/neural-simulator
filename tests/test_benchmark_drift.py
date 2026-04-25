"""RNG drift-detector tests.

These tests are the infrastructure counterpart to run_benchmarks.py:
they lock down the numerical outputs of a small seeded simulation so
silent RNG refactors, sequence reordering, or kernel changes get
caught at CI time instead of surfacing as intermittent benchmark
failures (see findings/rng-drift/ for context on the gamma benchmark
29-135 Hz variance that motivated this work).

Three layers of protection:

1. test_stdp_kernel_exact
   - Fully analytical, no RNG. If this ever fails the fused STDP kernel
     formula changed.

2. test_tiny_seeded_sim_spike_count
   - 100 neurons, 200 steps, seed=42. Asserts total spike count is the
     exact value observed on cc71207 (baseline commit). Catches any RNG
     stream drift: reordered cupy.random calls, new per-step RNG draws,
     OU noise changes, heterogeneity changes.

3. test_tiny_seeded_sim_reproducible
   - Same seed, two runs -> identical spike trains. Catches non-determinism
     introduced by new async GPU ops or uninitialized state.

test_gamma_peak_locked runs the full gamma benchmark at seed=42 and
asserts the peak lands in classic gamma [25, 55] Hz. It is opt-in via
the RUN_SLOW_TESTS env var because it takes ~15s on RTX 3090.

Run:
    pytest tests/test_benchmark_drift.py -v                       # fast only
    RUN_SLOW_TESTS=1 pytest tests/test_benchmark_drift.py -v      # all
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------
# Layer 1: STDP kernel is deterministic-analytical
# ---------------------------------------------------------------

def test_stdp_kernel_exact():
    """Fused STDP kernel must match Bi & Poo soft-bound formula exactly."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.kernels import fused_stdp_weight_update

    A_plus, A_minus = 0.012, 0.01
    tau_plus, tau_minus = 20.0, 20.0
    w_min, w_max = 0.0, 2.0
    w_init = 0.5

    # Test at four anchor points: symmetric LTP/LTD at |dt|=1ms and 50ms.
    cases = [
        (1.0, +A_plus * (w_max - w_init) * np.exp(-1.0 / tau_plus)),
        (50.0, +A_plus * (w_max - w_init) * np.exp(-50.0 / tau_plus)),
        (-1.0, -A_minus * (w_init - w_min) * np.exp(-1.0 / tau_minus)),
        (-50.0, -A_minus * (w_init - w_min) * np.exp(-50.0 / tau_minus)),
    ]

    for dt, expected_dw in cases:
        dt_gpu = cp.array([float(dt)], dtype=cp.float32)
        w_gpu = cp.array([w_init], dtype=cp.float32)
        w_new = fused_stdp_weight_update(
            dt_gpu, w_gpu, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max
        )
        dw = float((w_new - w_gpu).get()[0])
        assert abs(dw - expected_dw) < 1e-5, (
            f"STDP kernel drift at dt={dt}: got {dw:.6f}, "
            f"expected {expected_dw:.6f} (diff {dw - expected_dw:.2e})"
        )


# ---------------------------------------------------------------
# Layer 2/3: Tiny seeded simulation
# ---------------------------------------------------------------

def _build_tiny_sim(seed: int = 42):
    """Build a minimal RNG-sensitive sim: 100 neurons, OU noise on, no plasticity.

    Chosen to exercise the main RNG stream (OU draws per step) without the
    cost of a full benchmark. Returns an initialized SimulationBridge.
    """
    pytest.importorskip("cupy")
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig,
        RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 100
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    # Disable learning to isolate RNG effects from weight-change feedback
    cfg.enable_hebbian_learning = False
    cfg.enable_stdp = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_structural_plasticity = False
    cfg.enable_reward_modulation = False
    # Keep OU on — this is the main RNG-sensitive path we want to lock.
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 50.0

    sb = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    assert sb.is_initialized, "tiny sim failed to initialize"
    return sb, cfg


def _run_and_count(sb, cfg, n_steps: int = 200):
    """Run n_steps and return (total_spikes, per_step_spikes)."""
    import cupy as cp
    per_step = np.zeros(n_steps, dtype=np.int32)
    for i in range(n_steps):
        sb._run_one_simulation_step()
        sb.runtime_state.current_time_step += 1
        sb.runtime_state.current_time_ms = (
            sb.runtime_state.current_time_step * cfg.dt_ms
        )
        per_step[i] = int(cp.sum(sb.cp_firing_states).get())
    return int(per_step.sum()), per_step


def test_tiny_seeded_sim_reproducible():
    """Same seed -> identical spike trajectory. Catches non-determinism."""
    pytest.importorskip("cupy")

    sb1, cfg1 = _build_tiny_sim(seed=42)
    total1, trajectory1 = _run_and_count(sb1, cfg1, n_steps=200)
    sb1.clear_simulation_state_and_gpu_memory()

    sb2, cfg2 = _build_tiny_sim(seed=42)
    total2, trajectory2 = _run_and_count(sb2, cfg2, n_steps=200)
    sb2.clear_simulation_state_and_gpu_memory()

    assert total1 == total2, (
        f"Non-deterministic: seed=42 run 1 has {total1} spikes, "
        f"run 2 has {total2}"
    )
    assert np.array_equal(trajectory1, trajectory2), (
        f"Spike trajectory diverged: first diff at step "
        f"{np.argmax(trajectory1 != trajectory2)}"
    )


def test_tiny_seeded_sim_spike_count_in_range():
    """Seed=42 tiny sim spike count must fall in a locked range.

    This is the primary drift detector. If this test fails, the RNG
    stream has shifted — investigate before merging.

    Baseline history:
      - 170 spikes on commit cc71207 (2026-04-24, Session A baseline) —
        before Phase A→B added 8 IZH2007 BG/thalamus/HC/DA presets.
      - 149 spikes from commit 5fc92c8 onward (2026-04-25) — current value.
        Cause: bridge.py:917-921 builds `defined_izh2007_types` by
        iterating NeuronType, so adding new IZH2007 enum entries grew the
        list from 2 → 10. The trait-to-preset mapping at
        bridge.py:958 (`np_traits_host % num_defined_izh_variants`) then
        reassigns existing populations from {RS, FS, RS, FS, RS} to
        {RS, FS, MSN, TC_relay, TRN}. This drift test (default
        cfg.num_traits=5, GENERIC_UNSTRUCTURED profile) hits that path,
        producing fewer FS interneurons and a slower-firing mix → 149.
        Commit a16d45f added an opt-out (cfg.num_traits=1 keeps a single
        type) but doesn't help default-num_traits callers.
        See research/findings/2026-04-25-rng-drift-from-izh-presets.md.

    The ±10 tolerance catches real drift (which typically moves by
    dozens-to-hundreds of spikes when RNG streams shift) while
    tolerating minor numerical variance across CuPy versions or
    driver updates. If a legitimate refactor shifts the baseline,
    update EXPECTED_SPIKES + tolerance and note in findings/rng-drift/.
    """
    pytest.importorskip("cupy")

    sb, cfg = _build_tiny_sim(seed=42)
    total, _ = _run_and_count(sb, cfg, n_steps=200)
    sb.clear_simulation_state_and_gpu_memory()

    EXPECTED_SPIKES = 149  # Locked on 5fc92c8 (2026-04-25). Was 170 before.
    TOLERANCE = 10
    assert EXPECTED_SPIKES - TOLERANCE <= total <= EXPECTED_SPIKES + TOLERANCE, (
        f"RNG drift detected: tiny seeded sim produced {total} spikes, "
        f"expected {EXPECTED_SPIKES} +- {TOLERANCE}. Investigate recent "
        f"changes to cp.random usage, OU noise, heterogeneity, or the "
        f"neuron dynamics kernel. If intentional, update this test + "
        f"findings/rng-drift/."
    )


# ---------------------------------------------------------------
# Layer 4 (slow): Gamma benchmark peak is seed-locked
# ---------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Slow (~15s). Set RUN_SLOW_TESTS=1 to enable.",
)
def test_gamma_peak_locked():
    """At seed=42, gamma peak frequency must fall in classic gamma band.

    This is the regression test for the 29-135 Hz variance observed in
    pre-seeding reruns. With seed=42 the expected peak is ~38.7 Hz
    (measured on commit cc71207).

    Slow (~15s on RTX 3090). Run with: pytest -m slow
    """
    pytest.importorskip("cupy")

    from run_benchmarks import benchmark_gamma_oscillations

    r = benchmark_gamma_oscillations(seed_override=42)
    peak = r["peak_freq_hz"]
    assert 25.0 <= peak <= 55.0, (
        f"Gamma peak drifted: seed=42 produced {peak:.1f} Hz, "
        f"expected in [25, 55] Hz (classic gamma band). "
        f"Either RNG stream shifted or oscillatory regime moved. "
        f"See findings/rng-drift/ for context."
    )
    assert r["passed"], (
        f"Gamma benchmark failed at seed=42: {r}"
    )
