"""Byte-identity guard for the vectorized activity-driven gate-coupling path (perf, burndown #3 K=32 speedup).

`cfg.enable_vectorized_gate_couplings` batches the per-coupling control-pool firing means
(`cp_firing_states[control_idx].mean()`) into ONE segment-sum (`cp.add.reduceat`) instead of a Python
`.mean()`-per-coupling loop. Each control region is a contiguous DISJOINT index block and the firing states
are boolean, so a pool's mean is an EXACT integer sum / integer count -- the segment-sum reproduces each
per-coupling mean with no float reassociation. These tests pin that the vectorized path is BIT-IDENTICAL to
the scalar reference (the EMA trajectories AND the gate-write decisions), the load-bearing guard for using the
vectorized path on the #3 K=32 moat battery (whose committed result was produced by the scalar path).

CPU-only (SIM_BACKEND=numpy); no GPU required.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import from_host


def _build_coupled_bridge(seed=42, n_ctrl=12, ctrl_sizes=(20, 20, 30, 5, 1, 7)):
    """A tiny Izhikevich bridge with `n_ctrl` control pools of MIXED sizes, each coupled (couple_gate_to_pool)
    to a transmission gate on a 1-1 route. Mixed sizes exercise the per-segment-count divisor; size-1 pools
    exercise the reduceat edge. Returns the bridge."""
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    regions, pathways = [], []
    sizes = [ctrl_sizes[i % len(ctrl_sizes)] for i in range(n_ctrl)]
    for i, sz in enumerate(sizes):
        regions.append(BrainRegion(name=f"ctrl{i}", n_neurons=sz, exc_fraction=1.0, internal_density=0.0))
        regions.append(BrainRegion(name=f"src{i}", n_neurons=4, exc_fraction=1.0, internal_density=0.0))
        regions.append(BrainRegion(name=f"dst{i}", n_neurons=4, exc_fraction=1.0, internal_density=0.0))
        pathways.append(RegionPathway(from_region=f"src{i}", to_region=f"dst{i}", density=1.0,
                                      weight_mean=200.0, weight_jitter=0.0, plastic=False,
                                      transmission_gate=f"g{i}"))
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    # couple each control pool to its gate (varied thresholds/alphas so the gate-write decisions differ per coupling)
    for i in range(n_ctrl):
        sb.couple_gate_to_pool(f"g{i}", f"ctrl{i}", threshold=0.05 + 0.01 * (i % 5),
                               alpha=0.2 + 0.05 * (i % 3))
    return sb


def _drive_and_trace(sb, vectorized, n_steps=300, rng_seed=123, fire_p=0.15):
    """Drive the bridge's gate-coupling hook with random boolean firing states for n_steps; snapshot every
    coupling's EMA and last_value each step. Returns (ema_trace[n_steps, n_coupl], gate_trace[list of lists])."""
    sb.core_config.enable_vectorized_gate_couplings = bool(vectorized)
    sb._gate_coupling_flat = None
    # reset coupling state so both runs start identically
    for c in sb._gate_couplings:
        c["ema"] = 0.0
        c["last_value"] = None
    rng = np.random.default_rng(rng_seed)
    n = sb.core_config.num_neurons
    ema_trace, gate_trace = [], []
    for _ in range(n_steps):
        sb.cp_firing_states[:] = (rng.random(n) < fire_p)
        sb._apply_gate_couplings()
        ema_trace.append([c["ema"] for c in sb._gate_couplings])
        gate_trace.append([c["last_value"] for c in sb._gate_couplings])
    return np.asarray(ema_trace, dtype=np.float64), gate_trace


def test_vectorized_gate_couplings_bit_identical_emas():
    """The vectorized EMA trajectory is BIT-identical to the scalar reference (max|diff| exactly 0.0)."""
    sb = _build_coupled_bridge()
    ema_scalar, _ = _drive_and_trace(sb, vectorized=False)
    ema_vec, _ = _drive_and_trace(sb, vectorized=True)
    assert ema_scalar.shape == ema_vec.shape
    assert np.array_equal(ema_scalar, ema_vec), \
        f"vectorized EMAs differ from scalar: max|diff|={np.max(np.abs(ema_scalar - ema_vec)):.3e}"
    assert float(np.max(np.abs(ema_scalar - ema_vec))) == 0.0


def test_vectorized_gate_couplings_identical_gate_writes():
    """The vectorized gate-write decision trace (last_value per coupling per step) is identical to scalar."""
    sb = _build_coupled_bridge()
    _, gate_scalar = _drive_and_trace(sb, vectorized=False)
    _, gate_vec = _drive_and_trace(sb, vectorized=True)
    assert gate_scalar == gate_vec


def test_default_is_scalar_path():
    """The flag defaults False (the scalar reference path) so every existing caller is byte-unchanged."""
    assert CoreSimConfig().enable_vectorized_gate_couplings is False


def test_vectorized_rates_match_per_coupling_means():
    """The vectorized per-coupling rates equal float(cp_firing_states[control_idx].mean()) for each coupling,
    across several random firing states (the core byte-identity claim, isolated from the EMA/gate logic)."""
    sb = _build_coupled_bridge()
    sb.core_config.enable_vectorized_gate_couplings = True
    sb._gate_coupling_flat = None
    rng = np.random.default_rng(7)
    n = sb.core_config.num_neurons
    for _ in range(50):
        sb.cp_firing_states[:] = (rng.random(n) < 0.2)
        vec_rates = sb._gate_coupling_rates_vectorized()
        scalar_rates = [float(sb.cp_firing_states[c["control_idx"]].mean()) for c in sb._gate_couplings]
        assert len(vec_rates) == len(scalar_rates)
        for vr, sr in zip(vec_rates, scalar_rates):
            assert vr == sr   # bit-exact (boolean exact-integer sum / count)


def test_cache_invalidates_on_new_coupling():
    """couple_gate_to_pool nulls the flat cache (the coupling count changed); the next vectorized call rebuilds
    for the new count and still matches the scalar per-coupling means -- the cache-invalidation contract."""
    sb = _build_coupled_bridge(n_ctrl=4)
    sb.core_config.enable_vectorized_gate_couplings = True
    sb._gate_coupling_flat = None
    n = sb.core_config.num_neurons
    sb.cp_firing_states[:] = (np.arange(n) % 3 == 0)
    r1 = sb._gate_coupling_rates_vectorized()
    assert len(r1) == 4 and sb._gate_coupling_flat is not None and sb._gate_coupling_flat["n"] == 4
    # append a 5th coupling (a distinct existing index block as a fresh control) exactly as couple_gate_to_pool
    # would (which sets _gate_coupling_flat = None) -> the next vectorized call must rebuild for the new count.
    new_idx = from_host(np.asarray(sb.region_manager.indices("dst0"), dtype=np.int64))
    sb._gate_couplings.append({"gate_name": "g0", "control_idx": new_idx,
                               "threshold": 0.05, "alpha": 0.3, "open_value": 1.0, "ema": 0.0, "last_value": None})
    sb._gate_coupling_flat = None                            # exactly what couple_gate_to_pool's hook does on append
    r2 = sb._gate_coupling_rates_vectorized()
    assert len(r2) == 5 and sb._gate_coupling_flat["n"] == 5
    scalar = [float(sb.cp_firing_states[c["control_idx"]].mean()) for c in sb._gate_couplings]
    assert r2 == scalar
