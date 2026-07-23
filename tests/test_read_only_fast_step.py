"""Byte-identity guard for the opt-in `read_only_fast_step` perf flag (CoreSimConfig).

The flag skips the TWO per-step device->host sync stalls in `_run_one_simulation_step` -- the
`bool(cp_prev_firing_states.any())` and `bool(spike_count_gpu > 0)` reductions -- by forcing the two cached flags
(`_prev_any`, `_fired_any`) True. Those flags gate skip-fast-paths that produce ZERO contribution on a genuinely
zero-spike step, so forcing them True only does redundant zero-work; it must NEVER change the result.

This test is the PROOF: build two otherwise-identical bridges (same seed, same everything), one with the flag OFF and
one ON, run them in lockstep, and assert the per-step firing states + the final membrane state are BIT-IDENTICAL. If a
flag-gated block consumed RNG (which would diverge the stream when forced-True runs it on a zero-spike step), this test
would catch it. Runs on the numpy backend (CPU, CI-friendly); the byte-identity property is backend-independent.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel  # noqa: E402
from sim.backend import to_host  # noqa: E402


def _build(read_only_fast_step, **overrides):
    cfg = CoreSimConfig(
        num_neurons=120,
        connections_per_neuron=40,
        seed=42,
        neuron_model_type=NeuronModel.IZHIKEVICH.name,
        dt_ms=1.0,
        read_only_fast_step=read_only_fast_step,
        **overrides,
    )
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig(enable_profiling=False))
    b._initialize_simulation_data()
    return b


def _run_capture(b, n_steps):
    fired = []
    for _ in range(n_steps):
        b._run_one_simulation_step()
        fired.append(to_host(b.cp_firing_states).copy())
    v = to_host(b.cp_membrane_potential_v).copy()
    u = to_host(b.cp_recovery_variable_u).copy()
    return np.stack(fired), v, u


def _assert_bit_identical(off_overrides, on_overrides, n_steps=150, label=""):
    b_off = _build(False, **off_overrides)
    fired_off, v_off, u_off = _run_capture(b_off, n_steps)
    b_off.clear_simulation_state_and_gpu_memory()

    b_on = _build(True, **on_overrides)
    fired_on, v_on, u_on = _run_capture(b_on, n_steps)
    b_on.clear_simulation_state_and_gpu_memory()

    # some activity must actually occur, or the test is vacuous
    assert fired_off.sum() > 0, f"{label}: no spikes fired -- test is vacuous, raise the drive"
    assert np.array_equal(fired_off, fired_on), (
        f"{label}: firing states DIFFER with read_only_fast_step on vs off -- flag is NOT byte-identical "
        f"(first diff at step {int(np.argmax(np.any(fired_off != fired_on, axis=1)))})")
    assert np.array_equal(v_off, v_on), f"{label}: final membrane v differs"
    assert np.array_equal(u_off, u_on), f"{label}: final recovery u differs"


def test_byte_identical_inference_plasticity_off():
    # the intended use case: read-only inference (no weight learning). This is the load-bearing guarantee.
    common = dict(enable_hebbian_learning=False, enable_short_term_plasticity=False, enable_homeostasis=False)
    _assert_bit_identical(common, common, label="inference/plasticity-off")


def test_byte_identical_with_stp_and_hebbian():
    # with learning ON the flag is GUARDED INERT (a plasticity-gated block consumes RNG, so forcing the flags True on a
    # zero-spike step would diverge the stream -- this test originally CAUGHT that). The guard makes the flag
    # byte-identical unconditionally: with plasticity on, requesting read_only_fast_step has NO effect, so state is
    # bit-identical because the fast path never activates. This documents that the perf win is scoped to read-only
    # inference (its intended use) while the flag can never silently corrupt a learning run.
    common = dict(enable_hebbian_learning=True, enable_short_term_plasticity=True, enable_homeostasis=False)
    _assert_bit_identical(common, common, label="stp+hebbian-on (flag guarded inert)")


def test_default_is_on_and_field_exists():
    # DEFAULT-ON (owner directive: performance improvements on by default). Still byte-identical to prior behavior:
    # the flag is guarded-inert unless the step is genuinely read-only, and in that regime it is bit-identical (the
    # equivalence tests above assert this). So default-on speeds read-only inference ~3x with IDENTICAL results.
    cfg = CoreSimConfig(num_neurons=10)
    assert cfg.read_only_fast_step is True
