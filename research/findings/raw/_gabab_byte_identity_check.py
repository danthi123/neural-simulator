"""Deterministic byte-identity harness for the GABA_B protected edit.

Builds a minimal 3-region bridge with cfg.seed PINNED (so connectivity, heterogeneity,
and OU noise are reproducible across processes), steps it deterministically, and prints a
SHA-256 of the full membrane-potential + conductance trajectory.

Run BEFORE the edit (git stash) and AFTER: with enable_gabab=False the new GABA_B block is
skipped and the trajectory MUST be bit-identical (same hash). The harness avoids referencing
cfg.enable_gabab unless --on is passed, so it also runs cleanly on the pre-edit baseline.
Run under SIM_BACKEND=numpy.

    SIM_BACKEND=numpy python research/findings/raw/_gabab_byte_identity_check.py        # off (default)
    SIM_BACKEND=numpy python research/findings/raw/_gabab_byte_identity_check.py --on    # gabab on (edit only)
"""
from __future__ import annotations
import os, sys, hashlib, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronModel, NeuronType


def _h(arr):
    return to_host(arr).astype(np.float64)


def build(seed, gabab=False):
    cfg = CoreSimConfig()
    cfg.seed = seed                 # PIN — without this the bridge time-seeds and is nondeterministic.
    cfg.heterogeneity_seed = seed
    cfg.ou_seed = seed
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_neuromodulator_subsystem = False
    if gabab:
        cfg.enable_gabab = True
    cfg.brain_regions = [
        BrainRegion(name="cue", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="striosome_value", n_neurons=60, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
        BrainRegion(name="snc", n_neurons=30, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
                    syn_reversal_potential_i_override=-55.0),
    ]
    # Only pass receptor= when gabab is on, so the harness also runs on the PRE-EDIT
    # baseline (whose RegionPathway has no `receptor` field). With gabab off, default
    # "gaba_a" routing applies either way — the byte-identity comparison condition.
    strio_to_snc_kwargs = dict(from_region="striosome_value", to_region="snc",
                               density=0.5, weight_mean=10.0, weight_jitter=0.2, plastic=False)
    if gabab:
        strio_to_snc_kwargs["receptor"] = "gaba_b"
    cfg.region_pathways = [
        RegionPathway(from_region="cue", to_region="striosome_value",
                      density=0.6, weight_mean=20.0, weight_jitter=0.5, plastic=True),
        RegionPathway(**strio_to_snc_kwargs),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def run(seed=42, n_steps=200, gabab=False):
    xp, _ = get_backend()
    bridge, cfg = build(seed, gabab=gabab)
    idx_cue = xp.asarray(np.asarray(bridge.region_manager.indices("cue"), dtype=np.int64))
    idx_snc = xp.asarray(np.asarray(bridge.region_manager.indices("snc"), dtype=np.int64))
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[idx_cue] = xp.float32(600.0)
    bridge.cp_external_input_current[idx_snc] = xp.float32(220.0)
    digest = hashlib.sha256()
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        digest.update(np.ascontiguousarray(_h(bridge.cp_membrane_potential_v)).tobytes())
        digest.update(np.ascontiguousarray(_h(bridge.cp_conductance_g_i)).tobytes())
        digest.update(np.ascontiguousarray(_h(bridge.cp_conductance_g_e)).tobytes())
    v = _h(bridge.cp_membrane_potential_v)
    mask_state = ("None" if getattr(bridge, "cp_conductance_g_gabab", None) is None else "ALLOC")
    print(f"gabab_requested={gabab}  nnz={bridge.cp_connections.nnz}  gabab_conductance={mask_state}")
    print(f"trajectory_sha256={digest.hexdigest()}")
    print(f"final_v_sum={v.sum():.6f}  final_v_mean={v.mean():.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", action="store_true", help="enable_gabab=True + gaba_b receptor")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=200)
    a = ap.parse_args()
    run(seed=a.seed, n_steps=a.n_steps, gabab=a.on)
