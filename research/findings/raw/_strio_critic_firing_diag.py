"""READ-ONLY diagnostic for the striatal value-critic firing research (2026-06-08).

NO sim/ edits. Builds a minimal 3-region bridge (place -> striosome_value(MSN-D1) -> snc),
exactly the de-risk topology, and measures the FIRING LAYER under conditions that
replicate the deterministic-nav regime vs the probe regime:

  (A) MSN-D1 rheobase: constant current sweep, OU OFF (deterministic nav) -> the steady
      pA needed for the critic to fire at all. Grounds the "~700 pA" claim.
  (B) The probe regime: OU ON (default CoreSimConfig) -> does a modest steady drive fire it?
  (C) The key question: with OU OFF, can a STEADY EXCITATORY afferent (an up-state surrogate
      = convergent cortical/thalamic drive, NOT noise) drive the MSN-D1 to graded firing?
      This tests the B.02 biology: the up-state is excitation-driven, not noise-driven.
  (D) Does a dense place code (the probe's 2500 pA over a narrow-sigma bump) reach the critic
      with OU OFF? (the deterministic-nav firing wall)

Run: SIM_BACKEND=numpy python research/findings/raw/_strio_critic_firing_diag.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np


def build(seed=42, ou=False, n_strio=60, n_place=40, n_snc=30, strio_type="IZH2007_STRIATAL_MSN_D1"):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    cfg = CoreSimConfig()
    cfg.seed = seed; cfg.heterogeneity_seed = seed; cfg.ou_seed = seed
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    # The two background-depolarization knobs the nav runner disables (lines 3342-3343):
    cfg.enable_ou_process = bool(ou)
    cfg.enable_conductance_noise = False  # off in both (HH-only feature; izh here)
    cfg.enable_parameter_heterogeneity = False
    cfg.brain_regions = [
        BrainRegion(name="place", n_neurons=n_place, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="striosome_value", n_neurons=n_strio, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=getattr(NeuronType, strio_type).name,
                    syn_reversal_potential_i_override=-60.0),
        BrainRegion(name="snc", n_neurons=n_snc, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name),
    ]
    # place -> striosome excitatory afferent (the up-state-surrogate route in test C).
    cfg.region_pathways = [
        RegionPathway(from_region="place", to_region="striosome_value",
                      density=0.6, weight_mean=3.0, weight_jitter=0.0, plastic=False),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def idx(bridge, name):
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def measure_strio_rate(bridge, xp, *, strio_pa=0.0, place_pa=0.0, n_steps=300, warmup=100):
    """Drive striosome directly with strio_pa (constant) AND/OR place region with place_pa,
    step, return striosome mean firing Hz over the post-warmup window."""
    si = xp.asarray(idx(bridge, "striosome_value")); pi = xp.asarray(idx(bridge, "place"))
    n_s = len(host(si))
    spk = 0; m = 0
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = 0.0
        if strio_pa:
            bridge.cp_external_input_current[si] = xp.float32(strio_pa)
        if place_pa:
            bridge.cp_external_input_current[pi] = xp.float32(place_pa)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * bridge.core_config.dt_ms
        if t >= warmup:
            spk += int(host(bridge.cp_firing_states[si]).sum()); m += 1
    return spk / max(n_s, 1) / (m * 1e-3)


def main():
    from sim.backend import get_backend
    xp, bname = get_backend()
    print(f"backend={bname}\n")

    # ---- (A) MSN-D1 rheobase, OU OFF (deterministic nav) ----
    print("=== (A) MSN-D1 rheobase, OU OFF (deterministic nav regime) ===")
    b, _ = build(ou=False)
    for pa in [0, 200, 400, 600, 700, 800, 1000, 1200, 1500]:
        r = measure_strio_rate(b, xp, strio_pa=pa, n_steps=250, warmup=80)
        print(f"  strio_drive={pa:5d} pA  ->  striosome rate {r:7.2f} Hz")

    # ---- (B) Probe regime: OU ON, modest steady drive ----
    print("\n=== (B) MSN-D1 firing, OU ON (probe default CoreSimConfig) ===")
    b2, _ = build(ou=True)
    for pa in [0, 200, 400, 600]:
        r = measure_strio_rate(b2, xp, strio_pa=pa, n_steps=250, warmup=80)
        print(f"  strio_drive={pa:5d} pA + OU(100pA,15ms)  ->  striosome rate {r:7.2f} Hz")

    # ---- (C) Up-state surrogate = STEADY EXCITATORY afferent, OU OFF ----
    # The B.02 biology: the up-state is reached by CONVERGENT EXCITATION (E/I 2-5x exc-dominant),
    # not noise. Here a steady place-region drive (40 cells at place_pa, w=3.0, density 0.6)
    # delivers a steady excitatory synaptic current to the critic. Does it fire it gradedly?
    print("\n=== (C) up-state via STEADY EXCITATORY afferent (place region driven), OU OFF ===")
    b3, _ = build(ou=False)
    for ppa in [0, 200, 400, 800, 1200, 2000, 2500, 3500]:
        r = measure_strio_rate(b3, xp, place_pa=ppa, n_steps=250, warmup=80)
        # also report how hard the place region itself fires
        pi = xp.asarray(idx(b3, "place")); n_p = len(host(pi))
        # quick place-rate probe
        spk = 0
        for _ in range(60):
            b3.cp_external_input_current[:] = 0.0
            b3.cp_external_input_current[pi] = xp.float32(ppa)
            b3._run_one_simulation_step()
            b3.runtime_state.current_time_step += 1
            b3.runtime_state.current_time_ms = b3.runtime_state.current_time_step * b3.core_config.dt_ms
            spk += int(host(b3.cp_firing_states[pi]).sum())
        place_rate = spk / max(n_p, 1) / (60e-3)
        print(f"  place_drive={ppa:5d} pA (place fires {place_rate:6.1f} Hz)  ->  striosome {r:7.2f} Hz")

    # ---- (D) Compare a more-excitable critic type, OU OFF, steady direct drive ----
    print("\n=== (D) Alt critic neuron types, OU OFF, constant drive (excitability comparison) ===")
    for nt in ["IZH2007_STRIATAL_MSN_D1", "IZH2007_HIPPO_PYRAMIDAL", "IZH2007_RS_CORTICAL_PYRAMIDAL",
               "IZH2007_THALAMIC_RELAY", "IZH2007_STRIATAL_TAN"]:
        try:
            bb, _ = build(ou=False, strio_type=nt)
        except Exception as e:
            print(f"  {nt:34s}: build failed ({e})"); continue
        rates = []
        for pa in [200, 400, 700]:
            rates.append(measure_strio_rate(bb, xp, strio_pa=pa, n_steps=200, warmup=60))
        print(f"  {nt:34s}: 200pA={rates[0]:6.1f}  400pA={rates[1]:6.1f}  700pA={rates[2]:6.1f} Hz")


if __name__ == "__main__":
    main()
