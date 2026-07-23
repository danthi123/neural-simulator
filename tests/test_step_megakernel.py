"""Tests for the opt-in GENERAL-STEP MEGAKERNEL perf flag (CoreSimConfig.enable_step_megakernel).

The flag fuses the per-neuron ELEMENT-WISE inference chain of `_run_one_simulation_step` -- conductance decay +
I_syn + (pre-computed) E/I matvec increment + total-input assembly + Izhikevich-2007 dynamics + threshold-select +
fast_spike_reset -- into ONE @cp.fuse launch (fused_readonly_izh_step), keeping the cuSPARSE E/I matvec + the OU-noise
draw OUTSIDE. It is GPU-only, IZHIKEVICH-only, and guarded to the fully READ-ONLY inference regime; any guard failure
(or the numpy backend) falls through to the UNCHANGED Python step.

Two guarantees:
 (A) BYTE-IDENTICAL-WHEN-OFF / WHEN-GUARDS-FAIL -- runs on the numpy backend (CPU/CI). With the flag ON but the numpy
     backend, the dispatch guard fails at is_gpu_backend() so the Python path runs => state must equal the flag-OFF
     baseline bit-for-bit. This is the load-bearing "purely additive" guarantee.
 (B) ON-PATH EQUIVALENCE -- GPU only (skipped on numpy). Flag off vs on, same cfg.seed, identical per-step input:
     the fired boolean raster must be IDENTICAL every step (spikes are load-bearing) and max|Δv|,|Δu| within a tight
     FMA/summation residual. The controller runs this on the real GPU.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel  # noqa: E402
from sim.backend import to_host, is_gpu_backend, get_backend  # noqa: E402
from sim.kernels import (  # noqa: E402
    fused_readonly_izh_step,
    fused_conductance_decay_and_current,
    fused_izhikevich2007_dynamics_update,
)


# The fully read-only inference regime the megakernel targets: all learning OFF, fast_spike_reset +
# read_only_fast_step ON, Izhikevich. (enable_stdp / enable_structural_plasticity / enable_reward_modulation
# default True, so they must be explicitly disabled to enter the regime.)
_READONLY = dict(
    neuron_model_type=NeuronModel.IZHIKEVICH.name,
    read_only_fast_step=True,
    fast_spike_reset=True,
    enable_hebbian_learning=False,
    enable_short_term_plasticity=False,
    enable_homeostasis=False,
    enable_stdp=False,
    enable_structural_plasticity=False,
    enable_reward_modulation=False,
)


def _build(**overrides):
    cfg = CoreSimConfig(
        num_neurons=200,
        connections_per_neuron=40,
        seed=42,
        dt_ms=1.0,
        **{**_READONLY, **overrides},
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


# --------------------------------------------------------------------------------------------------
# (A) byte-identical when the flag is off / when guards fail (runs on numpy CPU backend)
# --------------------------------------------------------------------------------------------------
def test_default_is_off_and_fields_exist():
    cfg = CoreSimConfig(num_neurons=10)
    assert cfg.enable_step_megakernel is False   # default OFF -> byte-identical by construction
    assert cfg.enable_step_cudagraph is False     # alias also default OFF
    assert cfg.enable_step_megakernel_v2 is False  # v2 (in-kernel matvec) also default OFF


@pytest.mark.parametrize("n_steps", [150])
def test_byte_identical_flag_on_guard_fails_numpy(n_steps):
    # Flag ON but numpy backend => dispatch guard fails at is_gpu_backend() => the UNCHANGED Python path runs.
    # State must be bit-identical to the flag-OFF baseline. This proves the addition is inert unless the GPU
    # fused path actually activates (which it never does on numpy).
    b_off = _build(enable_step_megakernel=False)
    fired_off, v_off, u_off = _run_capture(b_off, n_steps)
    b_off.clear_simulation_state_and_gpu_memory()

    b_on = _build(enable_step_megakernel=True)
    fired_on, v_on, u_on = _run_capture(b_on, n_steps)
    b_on.clear_simulation_state_and_gpu_memory()

    assert fired_off.sum() > 0, "no spikes fired -- test is vacuous, raise the drive"
    assert np.array_equal(fired_off, fired_on), (
        "firing states DIFFER with enable_step_megakernel on vs off on numpy -- flag is NOT inert when guards fail "
        f"(first diff at step {int(np.argmax(np.any(fired_off != fired_on, axis=1)))})")
    assert np.array_equal(v_off, v_on), "final membrane v differs (numpy, flag should be inert)"
    assert np.array_equal(u_off, u_on), "final recovery u differs (numpy, flag should be inert)"


def test_alias_flag_also_inert_numpy():
    # The requested alias enable_step_cudagraph enables the same path; on numpy it is likewise inert.
    b_off = _build(enable_step_megakernel=False)
    fired_off, v_off, u_off = _run_capture(b_off, 120)
    b_off.clear_simulation_state_and_gpu_memory()

    b_on = _build(enable_step_cudagraph=True)
    fired_on, v_on, u_on = _run_capture(b_on, 120)
    b_on.clear_simulation_state_and_gpu_memory()

    assert np.array_equal(fired_off, fired_on)
    assert np.array_equal(v_off, v_on)
    assert np.array_equal(u_off, u_on)


def test_v2_byte_identical_flag_on_guard_fails_numpy():
    # v2 (enable_step_megakernel_v2, the in-kernel-matvec RawKernel path) ON but numpy backend => the dispatch guard
    # fails at is_gpu_backend() => the UNCHANGED Python step runs. State must be bit-identical to the flag-OFF
    # baseline -- the load-bearing "purely additive / inert unless the GPU RawKernel path activates" guarantee.
    b_off = _build(enable_step_megakernel_v2=False)
    fired_off, v_off, u_off = _run_capture(b_off, 150)
    b_off.clear_simulation_state_and_gpu_memory()

    b_on = _build(enable_step_megakernel_v2=True)
    fired_on, v_on, u_on = _run_capture(b_on, 150)
    b_on.clear_simulation_state_and_gpu_memory()

    assert fired_off.sum() > 0, "no spikes fired -- test is vacuous, raise the drive"
    assert np.array_equal(fired_off, fired_on), (
        "firing states DIFFER with enable_step_megakernel_v2 on vs off on numpy -- flag is NOT inert when guards fail "
        f"(first diff at step {int(np.argmax(np.any(fired_off != fired_on, axis=1)))})")
    assert np.array_equal(v_off, v_on), "final membrane v differs (numpy, v2 flag should be inert)"
    assert np.array_equal(u_off, u_on), "final recovery u differs (numpy, v2 flag should be inert)"


# --------------------------------------------------------------------------------------------------
# (A2) fused-kernel MATH equivalence, exercised directly on the CPU backend (@fuse is plain numpy there).
# This validates fused_readonly_izh_step reproduces the reference per-neuron chain -- decoupled from the
# GPU dispatch (which the guard blocks on numpy). On CPU the two share identical expressions => bit-exact.
# --------------------------------------------------------------------------------------------------
def test_fused_kernel_math_matches_reference_chain_cpu():
    xp, _ = get_backend()
    rng = np.random.default_rng(7)
    n = 256

    def _f32(a):
        return xp.asarray(a, dtype=xp.float32)

    g_e = _f32(rng.uniform(0.0, 0.5, n))
    g_i = _f32(rng.uniform(0.0, 0.5, n))
    g_e_inc = _f32(rng.uniform(0.0, 0.2, n))
    g_i_inc = _f32(rng.uniform(0.0, 0.2, n))
    v_np = rng.uniform(-70.0, -40.0, n)
    v_np[:64] = 35.0                          # force a subset above vpeak so the fired/reset branch is exercised
    v = _f32(v_np)
    u = _f32(rng.uniform(-10.0, 10.0, n))
    ext = _f32(rng.uniform(0.0, 300.0, n))
    ou = _f32(rng.uniform(-20.0, 20.0, n))
    C = _f32(rng.uniform(50.0, 150.0, n))
    k = _f32(rng.uniform(0.5, 1.5, n))
    vr = _f32(np.full(n, -60.0))
    vt = _f32(np.full(n, -40.0))
    a = _f32(rng.uniform(0.01, 0.05, n))
    b = _f32(rng.uniform(-2.0, 2.0, n))
    vpeak = _f32(np.full(n, 30.0))
    c_reset = _f32(np.full(n, -50.0))
    d_inc = _f32(np.full(n, 100.0))
    refr_np = rng.integers(0, 3, n)           # mix of in/out of refractory
    refr_np[:64] = 0                           # the forced-high head is out of refractory -> it fires
    refr = xp.asarray(refr_np, dtype=xp.int32)
    decay_e, decay_i, E_e, E_i, dt = 0.9, 0.8, 0.0, -75.0, 1.0
    refr_reset = xp.int32(max(0, 5 - 1))

    # reference chain, using the SAME reference kernels + the same inline reset the step uses
    g_e_dec, g_i_dec, I_syn = fused_conductance_decay_and_current(
        g_e.copy(), g_i.copy(), decay_e, decay_i, v, E_e, E_i)
    g_e_ref = g_e_dec + g_e_inc
    g_i_ref = g_i_dec + g_i_inc
    total = I_syn + ext
    total = total + ou
    v_dyn, u_dyn = fused_izhikevich2007_dynamics_update(v, u, C, k, vr, vt, a, b, total, dt)
    fired_ref = (v_dyn >= vpeak) & (refr <= 0)
    v_ref = xp.where(fired_ref, c_reset, v_dyn)
    u_ref = xp.where(fired_ref, u_dyn + d_inc, u_dyn)
    refr_ref = xp.where(fired_ref, refr_reset, xp.maximum(refr - 1, 0))

    ge2, gi2, v2, u2, fired2, refr2 = fused_readonly_izh_step(
        g_e, g_i, g_e_inc, g_i_inc, decay_e, decay_i, E_e, E_i, v, u, ext, ou,
        C, k, vr, vt, a, b, vpeak, c_reset, d_inc, refr, refr_reset, dt)

    assert bool(to_host(fired_ref).any()), "no neuron crossed threshold -- test is vacuous"
    assert np.array_equal(to_host(fired2), to_host(fired_ref)), "fused fired raster != reference"
    assert np.array_equal(to_host(refr2), to_host(refr_ref)), "fused refractory != reference"
    for name, ref, got in (("g_e", g_e_ref, ge2), ("g_i", g_i_ref, gi2),
                           ("v", v_ref, v2), ("u", u_ref, u2)):
        assert np.array_equal(to_host(got), to_host(ref)), f"fused {name} != reference (CPU should be bit-exact)"


# --------------------------------------------------------------------------------------------------
# (B) on-path equivalence: flag off vs on -- GPU ONLY (the fused path is is_gpu_backend()-gated)
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not is_gpu_backend(), reason="megakernel is GPU-only; on-path equivalence needs the CuPy backend")
def test_onpath_equivalence_raster_identical_gpu():
    n_steps = 200
    b_off = _build(enable_step_megakernel=False)
    fired_off, v_off, u_off = _run_capture(b_off, n_steps)
    b_off.clear_simulation_state_and_gpu_memory()

    b_on = _build(enable_step_megakernel=True)
    # sanity: the fused path must actually be dispatching on this build
    assert b_on._step_megakernel_can_dispatch(), "megakernel did NOT dispatch on a read-only GPU build -- guard too strict"
    fired_on, v_on, u_on = _run_capture(b_on, n_steps)
    b_on.clear_simulation_state_and_gpu_memory()

    assert fired_off.sum() > 0, "no spikes fired -- test is vacuous, raise the drive"
    # spikes are load-bearing: the raster must match bit-for-bit every step
    assert np.array_equal(fired_off, fired_on), (
        "fired raster DIFFERS off vs on -- a neuron flipped across threshold "
        f"(first diff at step {int(np.argmax(np.any(fired_off != fired_on, axis=1)))})")
    # v/u may differ by an FMA/summation residual (same class as the transpose-SpMV atomic scatter)
    dv = float(np.max(np.abs(v_off - v_on)))
    du = float(np.max(np.abs(u_off - u_on)))
    assert dv < 1e-4, f"max|Δv|={dv} exceeds the FMA residual tolerance"
    assert du < 1e-4, f"max|Δu|={du} exceeds the FMA residual tolerance"


# --------------------------------------------------------------------------------------------------
# (B2) v2 on-path equivalence: the RawKernel path that ALSO folds the E/I-split matvec into the single launch.
# GPU ONLY. The in-kernel double-accum matvec is NOT bit-guaranteed identical to the cuSPARSE float32 SpMV, so the
# RASTER staying bit-identical is the ACCEPTANCE GATE (a neuron on threshold could flip under the summation
# residual). THIS is the pass/fail check the controller must run on the real GPU -- the CPU cannot verify it.
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not is_gpu_backend(), reason="megakernel is GPU-only; on-path equivalence needs the CuPy backend")
def test_v2_onpath_equivalence_raster_identical_gpu():
    n_steps = 200
    b_off = _build(enable_step_megakernel_v2=False)   # pure Python step (all megakernel flags off)
    fired_off, v_off, u_off = _run_capture(b_off, n_steps)
    b_off.clear_simulation_state_and_gpu_memory()

    b_on = _build(enable_step_megakernel_v2=True)
    # sanity: the fused RawKernel path must actually be dispatching on this build
    assert b_on._step_megakernel_can_dispatch(), "v2 megakernel did NOT dispatch on a read-only GPU build -- guard too strict"
    fired_on, v_on, u_on = _run_capture(b_on, n_steps)
    # the RawKernel must actually have compiled + run (not silently fallen through)
    assert getattr(type(b_on), "_step_megastep_kernel", None) is not None, "v2 RawKernel never compiled -- path not taken"
    b_on.clear_simulation_state_and_gpu_memory()

    assert fired_off.sum() > 0, "no spikes fired -- test is vacuous, raise the drive"
    # spikes are load-bearing: the raster must match bit-for-bit every step (the ABSOLUTE correctness bar)
    assert np.array_equal(fired_off, fired_on), (
        "fired raster DIFFERS off vs on (v2) -- a neuron flipped across threshold under the in-kernel matvec residual "
        f"(first diff at step {int(np.argmax(np.any(fired_off != fired_on, axis=1)))})")
    # v/u may differ by an FMA/summation residual (double-accum in-kernel matvec vs cuSPARSE float32 SpMV)
    dv = float(np.max(np.abs(v_off - v_on)))
    du = float(np.max(np.abs(u_off - u_on)))
    assert dv < 1e-4, f"max|Δv|={dv} exceeds the FMA residual tolerance"
    assert du < 1e-4, f"max|Δu|={du} exceeds the FMA residual tolerance"


@pytest.mark.skipif(not is_gpu_backend(), reason="megakernel is GPU-only")
def test_v2_matches_v1_raster_gpu():
    # v1 (@cp.fuse + separate cuSPARSE matvec) and v2 (in-kernel matvec) must produce the SAME raster (both are
    # equivalence targets against the Python step). If they agree, the in-kernel matvec reproduces the cuSPARSE one.
    n_steps = 200
    b1 = _build(enable_step_megakernel=True)
    fired_v1, v_v1, u_v1 = _run_capture(b1, n_steps)
    b1.clear_simulation_state_and_gpu_memory()

    b2 = _build(enable_step_megakernel_v2=True)
    fired_v2, v_v2, u_v2 = _run_capture(b2, n_steps)
    b2.clear_simulation_state_and_gpu_memory()

    assert fired_v1.sum() > 0
    assert np.array_equal(fired_v1, fired_v2), (
        "v1 vs v2 raster DIFFERS "
        f"(first diff at step {int(np.argmax(np.any(fired_v1 != fired_v2, axis=1)))})")
    assert float(np.max(np.abs(v_v1 - v_v2))) < 1e-4
    assert float(np.max(np.abs(u_v1 - u_v2))) < 1e-4
