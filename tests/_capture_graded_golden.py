"""Capture a byte-exact golden snapshot of a NON-graded 2-region bridge run.

Run this ONCE on the UNEDITED bridge (before the graded-inhibition edit), then the
regression test `test_graded_inhibition_pathway.py::test_byte_identical_when_no_graded_pathway`
re-builds the identical bridge, runs the identical steps, and asserts the arrays match this
golden EXACTLY. This is the load-bearing guard that the default-off graded edit is byte-identical.

Usage:  SIM_BACKEND=numpy python -u tests/_capture_graded_golden.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def build_nongraded_bridge(seed=42):
    """A small inhibitory-feedback bridge that exercises the E/I-split matvec + g_i:
    src (exc) -> mid (inh) -> dst (exc), and src -> dst (exc). No graded pathway anywhere.
    This is the same shape the graded edit will touch (an inhibitory between-region pathway)."""
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    cfg.enable_brain_region_framework = True
    cfg.enable_ou_process = False
    cfg.ou_std_current_pA = 0.0
    cfg.enable_short_term_plasticity = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_structural_plasticity = False
    cfg.enable_reward_modulation = False
    cfg.brain_regions = [
        BrainRegion(name="src", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="mid", n_neurons=20, exc_fraction=0.0, internal_density=0.0,
                    plastic_internal=False),
        BrainRegion(name="dst", n_neurons=30, exc_fraction=1.0, internal_density=0.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="src", to_region="dst", density=1.0, weight_mean=120.0,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="src", to_region="mid", density=1.0, weight_mean=300.0,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="mid", to_region="dst", density=1.0, weight_mean=80.0,
                      weight_jitter=0.0, plastic=False),
    ]
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def run_and_snapshot(sb, n_steps=80, drive_pA=1400.0):
    from sim.backend import to_host
    src = np.asarray(sb.region_manager.indices("src"))
    sb.cp_external_input_current[:] = 0.0
    sb.cp_external_input_current[src] = drive_pA
    spk_acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(n_steps):
        sb._run_one_simulation_step()
        spk_acc += to_host(sb.cp_firing_states).astype(np.float64)
    return {
        "g_e": to_host(sb.cp_conductance_g_e).astype(np.float64),
        "g_i": to_host(sb.cp_conductance_g_i).astype(np.float64),
        "v": to_host(sb.cp_membrane_potential_v).astype(np.float64),
        "u": to_host(sb.cp_recovery_variable_u).astype(np.float64),
        "w": to_host(sb.cp_connections.data).astype(np.float64),
        "spk_acc": spk_acc,
    }


if __name__ == "__main__":
    sb = build_nongraded_bridge()
    # confirm the new attribute is absent / None on the unedited build (sanity for the edit later)
    has_mask = getattr(sb, "cp_graded_synapse_mask", "MISSING")
    snap = run_and_snapshot(sb)
    out = os.path.join(_HERE, "_graded_golden.npz")
    np.savez(out, **snap)
    print(f"[golden] cp_graded_synapse_mask attr on unedited bridge = {has_mask!r}")
    print(f"[golden] saved {out}")
    print(f"[golden] g_e sum={snap['g_e'].sum():.6f}  g_i sum={snap['g_i'].sum():.6f}  "
          f"v sum={snap['v'].sum():.6f}  spikes total={snap['spk_acc'].sum():.0f}")
