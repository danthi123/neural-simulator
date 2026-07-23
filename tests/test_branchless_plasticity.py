"""Byte-identity guard for `enable_branchless_plasticity` (branchless / compaction-free STDP).

The branchless STDP path (_apply_branchless_stdp) applies the weight update over ALL nnz with a masked SELECT instead of
the compacting `cp.where(candidate_mask)[0]` + `cp.where(within_window_mask)[0]` (which force two per-step device->host
syncs -> the ~15-45x learning-path win at 100K-1M nnz). It MUST be byte-identical to the compacting path.

The delicate part is the CLIP HAZARD: `fused_stdp_weight_update` clips internally (w_min..w_max), and today it runs only
on the compacted active subset, so a frozen out-of-bounds weight (e.g. a conversational read-out at 50 under
stdp_w_max=2) is never clipped. The branchless path runs the kernel over ALL nnz but DISCARDS its output off-active via
`cp.where(active_mask, updated_all, w_all)`, so the frozen weight is preserved verbatim. test_..._clip_hazard is the guard.

Non-vacuity: with strong uniform drive, connected neurons fire the SAME step -> delta_t=0 -> STDP returns the weight
unchanged (a vacuous test). To force REAL updates we set a varied recent spike history each step and fire only a subset,
so cross-synapse delta_t != 0 -> genuine LTP/LTD. Runs on the numpy backend (byte-identity is backend-independent).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402

N = 400
W_INIT = 1.0


def _build(branchless, frozen_oob=False):
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.enums import NeuronModel
    xp, _ = get_backend()
    cfg = CoreSimConfig(
        num_neurons=N, connections_per_neuron=60, seed=42, dt_ms=1.0,
        neuron_model_type=NeuronModel.IZHIKEVICH.name, ou_std_current_pA=0.0,
        enable_stdp=True, enable_reward_modulation=True, enable_hebbian_learning=False,
        enable_short_term_plasticity=False, enable_homeostasis=False, enable_structural_plasticity=False,
        stdp_a_plus=0.05, stdp_a_minus=0.05, stdp_w_max=5.0,
        enable_branchless_plasticity=branchless,
    )
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig(enable_profiling=False))
    b._initialize_simulation_data()
    b.cp_connections.data[:] = W_INIT              # mid-range so STDP has room to move weights
    if frozen_oob:
        # CLIP HAZARD: a frozen (gain-0) synapse parked at 50 with stdp_w_max=5 must be preserved, never clipped to 5.
        b.cp_plasticity_rate_gain = xp.ones(b.cp_connections.nnz, dtype=xp.float32)
        b.cp_plasticity_rate_gain[0] = 0.0
        b.cp_connections.data[0] = 50.0
    return b


def _run(b, steps=8):
    xp, _ = get_backend()
    traj = []
    for _ in range(steps):
        # varied recent spike history (within the STDP window) + fire only the first half NOW -> cross-synapse delta_t != 0
        b.cp_last_spike_time = xp.asarray(np.linspace(-2.0, -12.0, N).astype(np.float32))
        drive = xp.zeros(N, dtype=xp.float32)
        drive[: N // 2] = 2500.0
        b.cp_external_input_current[:] = drive
        b._run_one_simulation_step()
        traj.append(to_host(b.cp_connections.data).copy())
    return np.array(traj), to_host(b.cp_eligibility_trace).copy()


def test_branchless_stdp_byte_identical():
    traj_off, elig_off = _run(_build(False))
    traj_on, elig_on = _run(_build(True))
    n_changed = int((np.abs(traj_off[-1] - W_INIT) > 1e-6).sum())
    assert n_changed > 0, "vacuous: STDP moved no weights -- the drive/spike-history is not producing updates"
    assert np.array_equal(traj_off, traj_on), (
        "branchless STDP weight TRAJECTORY differs from the compacting path -- not byte-identical")
    m = traj_off.shape[1]
    assert np.array_equal(elig_off[:m], elig_on[:m]), "branchless eligibility trace differs from compacting"


def test_branchless_stdp_clip_hazard():
    traj_off, _ = _run(_build(False, frozen_oob=True))
    traj_on, _ = _run(_build(True, frozen_oob=True))
    # the frozen (gain-0) out-of-bounds weight must be preserved verbatim by BOTH paths (the clip must be a no-op for it)
    assert abs(traj_off[-1][0] - 50.0) < 1e-6, "compacting path clipped the frozen weight (setup broken)"
    assert abs(traj_on[-1][0] - 50.0) < 1e-6, (
        "branchless CLIP HAZARD: the frozen out-of-bounds weight was clipped -- the active-mask select is not discarding "
        "the kernel's clipped output off-active")
    assert np.array_equal(traj_off, traj_on), "clip-hazard case: branchless differs from compacting"


def _build_hebbian(branchless):
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.enums import NeuronModel
    cfg = CoreSimConfig(
        num_neurons=N, connections_per_neuron=60, seed=42, dt_ms=1.0,
        neuron_model_type=NeuronModel.IZHIKEVICH.name, ou_std_current_pA=0.0,
        enable_stdp=False, enable_reward_modulation=False,
        enable_hebbian_learning=True, hebbian_symmetric=True,          # symmetric -> synchronous co-firing potentiates
        enable_short_term_plasticity=False, enable_homeostasis=False, enable_structural_plasticity=False,
        hebbian_max_weight=5.0, enable_branchless_plasticity=branchless,
    )
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig(enable_profiling=False))
    b._initialize_simulation_data()
    b.cp_connections.data[:] = W_INIT
    return b


def test_branchless_hebbian_byte_identical():
    xp, _ = get_backend()

    def _run_heb(b, steps=8):
        traj = []
        for _ in range(steps):
            b.cp_external_input_current[:] = xp.full(N, 1000.0, dtype=xp.float32)  # synchronous -> symmetric Hebbian fires
            b._run_one_simulation_step()
            traj.append(to_host(b.cp_connections.data).copy())
        return np.array(traj)

    traj_off = _run_heb(_build_hebbian(False))
    traj_on = _run_heb(_build_hebbian(True))
    assert int((np.abs(traj_off[-1] - W_INIT) > 1e-6).sum()) > 0, "vacuous: Hebbian moved no weights"
    assert np.array_equal(traj_off, traj_on), (
        "branchless Hebbian potentiation weight TRAJECTORY differs from the compacting path -- not byte-identical")
