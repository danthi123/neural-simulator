"""THROWAWAY: fast MSN-D1 (striosome_value) f-I probe — does the merged nav critic's MSN-D1 fire
at ANY direct drive, with vs without the per-region homeostasis mask? This decides whether the
value-train can ever learn a V the GABA_B can subtract.

Builds a TINY 2-region bridge (just an MSN-D1 striosome_value pool, merged-critic config:
IZH2007_STRIATAL_MSN_D1, syn_reversal -60, enable_nmda=True, optional homeostasis) so it's fast
(no 2-min merged build). Sweeps direct external drive + reports the firing rate + the actual
membrane peak reached. Run under SIM_BACKEND=numpy.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def build_msn(seed, homeostasis):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.enums import NeuronModel, NeuronType
    from sim.regions import BrainRegion
    regions = [BrainRegion(
        name="striosome_value", n_neurons=80, exc_fraction=0.0, internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
        syn_reversal_potential_i_override=-60.0, enable_nmda=True,
        enable_homeostasis=bool(homeostasis))]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions; cfg.region_pathways = []
    cfg.enable_homeostasis = False; cfg.enable_synaptic_scaling = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_ou_process = True; cfg.ou_std_current_pA = 100.0
    cfg.homeostasis_threshold_adapt_rate = 0.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def main():
    from sim.backend import get_backend
    import numpy as np
    xp, backend = get_backend()
    print(f"[msn-fi probe] backend={backend}")
    for homeo in (False, True):
        b = build_msn(42, homeo)
        idx = xp.asarray(np.asarray(b.region_manager.indices("striosome_value"), dtype=np.int64))
        thr = getattr(b, "cp_neuron_firing_thresholds", None)
        thr_v = float(_host(thr[idx]).mean()) if thr is not None else float("nan")
        print(f"\n  homeostasis={homeo}  (adapted-thr mean={thr_v:.1f} mV; vpeak used when no mask)")
        for pa in (200, 400, 600, 800, 1200, 1800, 2500):
            # settle
            b.cp_external_input_current[:] = 0.0
            for _ in range(40):
                b._run_one_simulation_step(); b.runtime_state.current_time_step += 1
                b.runtime_state.current_time_ms = b.runtime_state.current_time_step * b.core_config.dt_ms
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[idx] = xp.float32(pa)
            spk = 0; vpk = -100.0
            for _ in range(60):
                b._run_one_simulation_step(); b.runtime_state.current_time_step += 1
                b.runtime_state.current_time_ms = b.runtime_state.current_time_step * b.core_config.dt_ms
                spk += int(_host(b.cp_firing_states[idx]).sum())
                vpk = max(vpk, float(_host(b.cp_membrane_potential_v[idx]).max()))
            hz = spk / 80 / (60e-3)
            print(f"    drive={pa:5d}pA  rate={hz:6.1f}Hz  v_peak={vpk:7.1f}mV")


if __name__ == "__main__":
    main()
