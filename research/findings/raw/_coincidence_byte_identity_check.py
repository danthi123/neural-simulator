"""Deterministic byte-identity harness for the dendritic-COINCIDENCE protected edit (Route D).

The COINCIDENCE sibling of `_nmda_recurrent_byte_identity_check.py`. Builds a minimal bridge with a
RECURRENT excitatory pathway (the CA3-like topology the edit targets: detonator -> pool + pool -> pool
recurrent), pins cfg.seed (so connectivity / heterogeneity / OU noise are reproducible across
processes), steps it deterministically, and prints a SHA-256 of the full membrane-potential +
conductance trajectory.

Run on a CLEAN-BASELINE tree (off only) and on the EDITED tree (off + on). With
enable_coincidence_detection=False AND no pathway setting coincidence_detector=True, the new
coincidence block is unreached, fused_coincidence_plateau is never called, the g_e matvec is unmasked,
and the trajectory MUST be bit-identical (same hash) to the baseline. The harness avoids referencing
the new fields unless --on is passed, so it ALSO runs cleanly on the pre-edit baseline (whose
RegionPathway has no `coincidence_detector` field and whose CoreSimConfig has no
enable_coincidence_detection field).

    # baseline tree (off):
    SIM_BACKEND=numpy python research/findings/raw/_coincidence_byte_identity_check.py
    # edited tree (off -- MUST equal baseline) and (on -- exercises the new path):
    SIM_BACKEND=numpy python research/findings/raw/_coincidence_byte_identity_check.py
    SIM_BACKEND=numpy python research/findings/raw/_coincidence_byte_identity_check.py --on
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


def build(seed, coincidence=False):
    cfg = CoreSimConfig()
    cfg.seed = seed                 # PIN -- without this the bridge time-seeds and is nondeterministic.
    cfg.heterogeneity_seed = seed
    cfg.ou_seed = seed
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True                       # the recurrent is plastic -- STDP path active
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_neuromodulator_subsystem = False
    if coincidence:
        cfg.enable_coincidence_detection = True
        cfg.coincidence_k_threshold = 4.0  # exercise the new block at a real threshold
    # CA3-like topology: a detonator (mossy) drives a pool; the pool has a recurrent loop.
    cfg.brain_regions = [
        BrainRegion(name="mossy", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="ca3", n_neurons=80, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name),
    ]
    # Only pass coincidence_detector= when coincidence is on, so the harness ALSO runs on the PRE-EDIT
    # baseline (whose RegionPathway has no `coincidence_detector` field). With coincidence off, default
    # routing applies either way -- the byte-identity comparison condition.
    rec_kwargs = dict(from_region="ca3", to_region="ca3",
                      density=0.30, weight_mean=2.0, weight_jitter=0.2, plastic=True)
    if coincidence:
        rec_kwargs["coincidence_detector"] = True
    cfg.region_pathways = [
        RegionPathway(from_region="mossy", to_region="ca3",
                      density=0.40, weight_mean=8.0, weight_jitter=0.3, plastic=True),
        RegionPathway(**rec_kwargs),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def run(seed=42, n_steps=200, coincidence=False):
    xp, _ = get_backend()
    bridge, cfg = build(seed, coincidence=coincidence)
    idx_mossy = xp.asarray(np.asarray(bridge.region_manager.indices("mossy"), dtype=np.int64))
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[idx_mossy] = xp.float32(700.0)  # detonator drive
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
    co_state = ("None" if getattr(bridge, "cp_conductance_g_coincidence", None) is None else "ALLOC")
    mask_state = ("None" if getattr(bridge, "cp_coincidence_synapse_mask", None) is None else "ALLOC")
    print(f"coincidence_requested={coincidence}  nnz={bridge.cp_connections.nnz}  "
          f"g_coincidence={co_state}  routing_mask={mask_state}")
    print(f"trajectory_sha256={digest.hexdigest()}")
    print(f"final_v_sum={v.sum():.6f}  final_v_mean={v.mean():.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", action="store_true", help="enable_coincidence_detection=True + coincidence_detector")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=200)
    a = ap.parse_args()
    run(seed=a.seed, n_steps=a.n_steps, coincidence=a.on)
