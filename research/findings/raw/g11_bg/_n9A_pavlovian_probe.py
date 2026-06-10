"""(A) Pavlovian de-risk: does the spiking reward_us afferent FIRE and BURST the SNc?
Minimal bridge: reward_us (PPN) --exc--> snc (DA) + striosome_value (critic) --gaba_b--> snc.
Drive reward_us, measure (1) reward_us firing, (2) SNc rate vs tonic, (3) the r-V subtraction
(SNc burst shrinks when the critic V fires). Sweep the reward_us->snc weight to TUNE it.

    python research/findings/raw/g11_bg/_n9A_pavlovian_probe.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from sim.backend import get_backend
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronModel, NeuronType

xp, backend = get_backend()
SNC_TONIC = 220.0
US_DRIVE = 1200.0


def build(us_to_snc_w, seed=42):
    cfg = CoreSimConfig()
    cfg.seed = seed; cfg.heterogeneity_seed = seed; cfg.ou_seed = seed
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = False; cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False; cfg.enable_neuromodulator_subsystem = False
    cfg.enable_gabab = True; cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0; cfg.gabab_propagation_strength = 0.02
    cfg.brain_regions = [
        BrainRegion(name="reward_us", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="snc", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
                    syn_reversal_potential_i_override=-55.0),
        BrainRegion(name="striosome_value", n_neurons=80, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="reward_us", to_region="snc",
                      density=0.6, weight_mean=float(us_to_snc_w), weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=10.0, weight_jitter=0.2, plastic=False, receptor="gaba_b"),
    ]
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def idx(b, name):
    return xp.asarray(np.asarray(b.region_manager.indices(name), dtype=np.int64))


def measure(b, us_drive, crit_drive, n=120, warmup=30):
    ius, isn, icr = idx(b, "reward_us"), idx(b, "snc"), idx(b, "striosome_value")
    b.cp_external_input_current[:] = 0.0
    b.cp_external_input_current[ius] = xp.float32(us_drive)
    b.cp_external_input_current[isn] = xp.float32(SNC_TONIC)
    if crit_drive > 0:
        b.cp_external_input_current[icr] = xp.float32(crit_drive)
    us_sp = sn_sp = m = 0
    for t in range(n):
        b._run_one_simulation_step()
        if t >= warmup:
            us_sp += int(b.cp_firing_states[ius].sum())
            sn_sp += int(b.cp_firing_states[isn].sum()); m += 1
    hz = lambda s, npool: s / max(npool, 1) / max(m * 1e-3, 1e-9)
    return hz(us_sp, int(ius.size)), hz(sn_sp, int(isn.size))


print(f"backend={backend}  SNC_TONIC={SNC_TONIC}pA  US_DRIVE={US_DRIVE}pA")
print(f"{'us->snc w':>10} | {'reward_us Hz':>12} | {'SNc tonic(noUS)':>15} | {'SNc +US(r)':>11} | {'SNc +US +V':>11} | burst? r-V?")
print("-" * 90)
for w in (20, 50, 100, 200, 400):
    b = build(w)
    _, snc_tonic = measure(b, 0.0, 0.0)          # no US -> tonic
    us_hz, snc_r = measure(build(w), US_DRIVE, 0.0)   # US on -> reward burst
    _, snc_rv = measure(build(w), US_DRIVE, 600.0)    # US on + critic V -> r-V (should shrink)
    burst = "YES" if snc_r > snc_tonic * 1.4 else "no"
    rv = "YES" if snc_rv < snc_r * 0.8 else "no"
    print(f"{w:>10} | {us_hz:12.1f} | {snc_tonic:15.1f} | {snc_r:11.1f} | {snc_rv:11.1f} | {burst:>5} {rv:>5}")
