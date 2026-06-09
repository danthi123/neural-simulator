"""N9 convergent-up-state CALIBRATION (CuPy): find the A1 (vs_place_drive) non-plastic
weight that puts the MSN-D1 critic into a ~10-20 Hz up-state AT THE GOAL and ~0 Hz FAR.

Two-region afferent (the design's wiring-collision fix):
  A1  vs_place_drive   -> striosome_value  : dense (0.8), NON-plastic, weight ~W (sweep)
  A2  vs_place_context -> striosome_value  : sparse (0.4), PLASTIC init 0.2
Both rendered with the SAME grid-32 Gaussian place code each step (NEAR or FAR).

MUST be SIM_BACKEND=cupy. Short run. READ-ONLY on sim/.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "cupy")

import numpy as np
from sim.backend import get_backend
xp, _bk = get_backend()
print("backend:", _bk, flush=True)
assert _bk == "cupy", f"this calibration MUST run on cupy (got {_bk})"


def _host(a):
    g = getattr(a, "get", None)
    return g() if g is not None else np.array(a)


def _idx(bridge, name):
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _grid_prefs(n_cells, grid_size):
    side = int(round(np.sqrt(n_cells)))
    xs = np.linspace(0.0, grid_size - 1.0, side, dtype=np.float64)
    ys = np.linspace(0.0, grid_size - 1.0, side, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    px = gx.ravel(); py = gy.ravel()
    if px.size < n_cells:
        reps = int(np.ceil(n_cells / px.size))
        px = np.tile(px, reps)[:n_cells]; py = np.tile(py, reps)[:n_cells]
    return px[:n_cells].copy(), py[:n_cells].copy()


def place_code(pos_xy, prefs_xy, max_pA, sigma):
    px, py = prefs_xy
    x, y = float(pos_xy[0]), float(pos_xy[1])
    dsq = (px - x) ** 2 + (py - y) ** 2
    return (max_pA * np.exp(-dsq / (2.0 * sigma ** 2))).astype(np.float32)


def build(seed, a1_weight, a1_density=0.8, a2_weight=0.2, a2_density=0.4,
          n_vs=200, n_strio=80, grid_size=32):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.12
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 40.0
    # deterministic-nav regime
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False

    cfg.brain_regions = [
        # A1 up-state drive (dense, NON-plastic)
        BrainRegion(name="vs_place_drive", n_neurons=n_vs, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        # A2 plastic value learner
        BrainRegion(name="vs_place_context", n_neurons=n_vs, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="striosome_value", n_neurons=n_strio, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="vs_place_drive", to_region="striosome_value",
                      density=float(a1_density), weight_mean=float(a1_weight),
                      weight_jitter=0.5, plastic=False),
        RegionPathway(from_region="vs_place_context", to_region="striosome_value",
                      density=float(a2_density), weight_mean=float(a2_weight),
                      weight_jitter=0.1, plastic=True, plasticity_gate="value_input"),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def critic_rate(bridge, drive_idx_list, drive_vecs, crit_idx, n_steps=120, warmup=40):
    """Drive A1 + A2 regions with their place-code vecs, measure critic firing rate (Hz)."""
    crit_cp = xp.asarray(crit_idx)
    n_crit = len(crit_idx)
    spk = 0; m = 0
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        for didx, dvec in zip(drive_idx_list, drive_vecs):
            bridge.cp_external_input_current[xp.asarray(didx)] = xp.asarray(dvec, dtype=xp.float32)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * bridge.core_config.dt_ms
        if t >= warmup:
            spk += int(bridge.cp_firing_states[crit_cp].sum()); m += 1
    return spk / max(n_crit, 1) / max(m * 1e-3, 1e-9)


if __name__ == "__main__":
    SEED = 42
    GRID = 32
    SIGMA = 4.0
    DRIVE_MAX = 800.0
    NEAR = (26.571, 26.571)
    FAR = (4.429, 4.429)

    a1_idx = None
    print("\n=== A1 weight sweep (critic rate near vs far) ===", flush=True)
    for w in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        bridge, cfg = build(SEED, a1_weight=w)
        drive_idx = _idx(bridge, "vs_place_drive")
        ctx_idx = _idx(bridge, "vs_place_context")
        crit_idx = _idx(bridge, "striosome_value")
        prefs = _grid_prefs(len(drive_idx), GRID)
        near_vec = place_code(NEAR, prefs, DRIVE_MAX, SIGMA)
        far_vec = place_code(FAR, prefs, DRIVE_MAX, SIGMA)
        n_active_near = int((near_vec > 1.0).sum())
        # Drive BOTH A1 and A2 with the same code (A2 starts weak, ~5 pA, negligible for firing)
        r_near = critic_rate(bridge, [drive_idx, ctx_idx], [near_vec, near_vec], crit_idx)
        r_far = critic_rate(bridge, [drive_idx, ctx_idx], [far_vec, far_vec], crit_idx)
        print(f"  A1 w={w:5.1f} (density 0.8, n_active_near={n_active_near}): "
              f"critic NEAR={r_near:6.2f} Hz  FAR={r_far:6.2f} Hz  ratio={r_near/max(r_far,1e-3):.1f}",
              flush=True)
        del bridge
