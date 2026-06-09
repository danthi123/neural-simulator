"""Deterministic byte-identity harness for the per-region homeostasis protected edit.

Mirrors _gabab_byte_identity_check.py, but tailored to the per-region homeostasis edit,
which touches the spike-threshold SELECTION used in BOTH global-homeostasis states (OFF and
ON). So byte-identity must be verified for BOTH global states when NO region opts into
per-region homeostasis (the default — cp_homeostasis_neuron_mask is None):

  1. Global homeostasis OFF (the deterministic-nav case): the 3-branch refactor must take
     branch 3 (== legacy else -> cp_izh_vpeak). Trajectory bit-identical pre vs post edit.
  2. Global homeostasis ON: the 3-branch refactor must take branch 1 (== legacy
     cp_neuron_firing_thresholds). Trajectory bit-identical pre vs post edit. This proves the
     3-branch is identical to the old 2-branch when the mask is None.

The hash captures membrane V + g_e + g_i + the firing-thresholds array (the array the edit
affects when homeostasis runs). With global homeostasis ON, the EMA threshold update moves
cp_neuron_firing_thresholds over the run, so the homeostasis path is meaningfully exercised.

The harness builds ONLY plain BrainRegions (no per-region enable_homeostasis), so it runs
cleanly on the PRE-EDIT baseline (whose BrainRegion lacks the enable_homeostasis field) and
on the edited tree identically. Strong drive ensures neurons fire (so homeostasis EMA moves).
Run under SIM_BACKEND=numpy.

    SIM_BACKEND=numpy python research/findings/raw/_homeostasis_byte_identity_check.py             # global homeostasis OFF (default)
    SIM_BACKEND=numpy python research/findings/raw/_homeostasis_byte_identity_check.py --global-on  # global homeostasis ON
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


def build(seed, global_homeostasis=False):
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
    cfg.enable_synaptic_scaling = False
    # The variable under test: global homeostasis flag. With NO per-region flags set,
    # both states must be byte-identical pre vs post edit.
    cfg.enable_homeostasis = bool(global_homeostasis)
    # NOTE: deliberately NO BrainRegion(enable_homeostasis=True) anywhere here, so the
    # PRE-EDIT baseline (whose BrainRegion has no such field) builds identically and the
    # per-region mask stays None on the edited tree (the byte-identity comparison condition).
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
    cfg.region_pathways = [
        RegionPathway(from_region="cue", to_region="striosome_value",
                      density=0.6, weight_mean=20.0, weight_jitter=0.5, plastic=True),
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=10.0, weight_jitter=0.2, plastic=False),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def run(seed=42, n_steps=200, global_homeostasis=False):
    xp, _ = get_backend()
    bridge, cfg = build(seed, global_homeostasis=global_homeostasis)
    idx_cue = xp.asarray(np.asarray(bridge.region_manager.indices("cue"), dtype=np.int64))
    idx_strio = xp.asarray(np.asarray(bridge.region_manager.indices("striosome_value"), dtype=np.int64))
    idx_snc = xp.asarray(np.asarray(bridge.region_manager.indices("snc"), dtype=np.int64))
    bridge.cp_external_input_current[:] = 0.0
    # Strong drive on all three regions so neurons fire and the homeostasis EMA / threshold
    # update actually moves cp_neuron_firing_thresholds when global homeostasis is ON.
    bridge.cp_external_input_current[idx_cue] = xp.float32(600.0)
    bridge.cp_external_input_current[idx_strio] = xp.float32(400.0)
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
        # Capture the firing-thresholds array — the array the edit affects when homeostasis runs.
        if bridge.cp_neuron_firing_thresholds is not None:
            digest.update(np.ascontiguousarray(_h(bridge.cp_neuron_firing_thresholds)).tobytes())
    v = _h(bridge.cp_membrane_potential_v)
    mask_state = ("None" if getattr(bridge, "cp_homeostasis_neuron_mask", None) is None else "ALLOC")
    thr = bridge.cp_neuron_firing_thresholds
    thr_sum = (float(_h(thr).sum()) if thr is not None else float("nan"))
    print(f"global_homeostasis={global_homeostasis}  nnz={bridge.cp_connections.nnz}  per_region_mask={mask_state}")
    print(f"trajectory_sha256={digest.hexdigest()}")
    print(f"final_v_sum={v.sum():.6f}  final_v_mean={v.mean():.6f}  thresholds_sum={thr_sum:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-on", action="store_true", help="cfg.enable_homeostasis=True (no per-region flags)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=200)
    a = ap.parse_args()
    run(seed=a.seed, n_steps=a.n_steps, global_homeostasis=a.global_on)
