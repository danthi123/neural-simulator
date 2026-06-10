"""N5 Option-C CORE de-risk: does a slow-channel TEMPORAL-DIFFERENCE fire on a RISING
signal (= the SC bump moving toward the foveal/rostral pole = the goal getting closer)
and stay quiet on a FALLING one? — the spiking neural-derivative the perceived-approach
reward (N5) needs, replacing the host sign(delta eccentricity).

Mechanism (building on the committed spiking SC's bump):
  sc_rostral (foveal-pole readout; here driven by a controlled ramp standing in for the
      bump's centrality) --nmda_slow (tau ~100ms)--> sc_rostral_slow (a LAGGED copy)
  approach <-- sc_rostral (AMPA, excitatory)  +  sc_rostral_slow (gaba_b/GIRK, inhibitory)
  => approach fires when rostral_now > rostral_slow (RISING = approaching), and is
     suppressed when rostral_now < rostral_slow (FALLING = receding).

This isolates the Option-C derivative from the full SC so it can be de-risked cheaply on
CPU. If approach cleanly separates rising vs falling, wiring it onto the real sc_map
rostral pool (then approach -> reward_us) is the N5 build.

    SIM_BACKEND=numpy python research/runners/sc_approach_td_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sim.backend import get_backend
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronModel, NeuronType

xp, BACKEND = get_backend()


def build(seed=42, w_ros_slow=8.0, w_ros_app=14.0, w_slow_app=10.0,
          nmda_recurrent=True, gabab=True):
    cfg = CoreSimConfig()
    cfg.seed = seed; cfg.heterogeneity_seed = seed; cfg.ou_seed = seed
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    for k in ("enable_stdp", "enable_hebbian_learning", "enable_reward_modulation",
              "enable_short_term_plasticity", "enable_structural_plasticity",
              "enable_neuromodulator_subsystem"):
        setattr(cfg, k, False)
    cfg.ou_std_current_pA = 6.0
    # slow channels (the Option-C substrate; already merged + runner-enabled)
    cfg.enable_nmda_recurrent = bool(nmda_recurrent)
    cfg.enable_gabab = bool(gabab)
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.02
    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    cfg.brain_regions = [
        BrainRegion(name="sc_rostral", n_neurons=24, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="sc_rostral_slow", n_neurons=24, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        # approach: exc drive from rostral, inhibited (gaba_b) by the lagged slow copy.
        # exc_fraction 1.0 — it is an excitatory output cell; the inhibition comes from the
        # PREsynaptic sc_rostral_slow being made inhibitory (a 100%-inhibitory relay pool).
        BrainRegion(name="approach", n_neurons=24, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        # a dedicated inhibitory relay that carries the LAGGED rostral signal as GABA_B onto
        # approach (so the subtraction is rostral_now - rostral_slow). Driven by sc_rostral_slow.
        BrainRegion(name="slow_inh", n_neurons=24, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name),
    ]
    cfg.region_pathways = [
        # rostral -> rostral_slow via SLOW NMDA (the lagged trace)
        RegionPathway(from_region="sc_rostral", to_region="sc_rostral_slow",
                      density=0.6, weight_mean=w_ros_slow, weight_jitter=0.1, plastic=False,
                      exc_receptor="nmda_slow"),
        # rostral -> approach (fast AMPA excitation = the "now" term)
        RegionPathway(from_region="sc_rostral", to_region="approach",
                      density=0.6, weight_mean=w_ros_app, weight_jitter=0.1, plastic=False),
        # rostral_slow -> slow_inh (make the lagged copy inhibitory) ...
        RegionPathway(from_region="sc_rostral_slow", to_region="slow_inh",
                      density=0.6, weight_mean=12.0, weight_jitter=0.1, plastic=False),
        # ... slow_inh -> approach via GABA_B (the subtractive "minus lagged" term)
        RegionPathway(from_region="slow_inh", to_region="approach",
                      density=0.8, weight_mean=w_slow_app, weight_jitter=0.1, plastic=False,
                      receptor="gaba_b"),
    ]
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def idx(b, name):
    return xp.asarray(np.asarray(list(b.region_manager.indices(name)), dtype=np.int64))


def run_trajectory(b, drive_profile, ros_drive_pa=600.0):
    """Drive sc_rostral with drive_profile[t] (a 0..1 centrality ramp) and read the approach
    firing rate per step. Returns (approach_rate_per_step, rostral_rate_per_step)."""
    iros, iapp = idx(b, "sc_rostral"), idx(b, "approach")
    app_rate, ros_rate = [], []
    for c in drive_profile:
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[iros] = xp.float32(ros_drive_pa * float(c))
        b._run_one_simulation_step()
        app_rate.append(int(b.cp_firing_states[iapp].sum()))
        ros_rate.append(int(b.cp_firing_states[iros].sum()))
    return np.asarray(app_rate), np.asarray(ros_rate)


def main():
    print(f"N5 Option-C CORE de-risk (temporal-difference fires on RISING, quiet on FALLING) "
          f"— backend {BACKEND}")
    b = build(seed=42)
    # A trajectory: baseline, RAMP UP (approaching: bump -> centre), hold, RAMP DOWN (receding).
    up = np.concatenate([
        np.full(20, 0.1),                 # baseline (far, low centrality)
        np.linspace(0.1, 1.0, 40),        # APPROACH: centrality rising
        np.full(20, 1.0),                 # at goal (foveated)
        np.linspace(1.0, 0.1, 40),        # RECEDE: centrality falling
        np.full(20, 0.1),                 # far again
    ])
    app, ros = run_trajectory(b, up)
    # windows
    base = slice(0, 20); rise = slice(20, 60); hold = slice(60, 80); fall = slice(80, 120)
    a_base, a_rise, a_hold, a_fall = app[base].mean(), app[rise].mean(), app[hold].mean(), app[fall].mean()
    r_rise, r_fall = ros[rise].mean(), ros[fall].mean()
    print(f"sc_rostral firing: rise={r_rise:.2f} fall={r_fall:.2f} (driven similarly both ways)")
    print(f"approach firing:   baseline={a_base:.2f}  RISE={a_rise:.2f}  hold={a_hold:.2f}  FALL={a_fall:.2f}")
    print(f"  -> approach should be HIGH on RISE (approaching) and LOW on FALL (receding):")
    sep = a_rise - a_fall
    ok_rise = a_rise > a_base * 1.3 and a_rise > a_fall * 1.5
    ok_fall = a_fall <= a_rise * 0.7
    print(f"  RISE>FALL separation = {sep:.2f}  (rise>fall x1.5? {ok_rise};  fall suppressed? {ok_fall})")
    if ok_rise and ok_fall:
        print("VERDICT: RESOLVES — the slow-channel TD fires on the rising (approaching) phase and is "
              "suppressed on the falling (receding) phase = a clean spiking neural-derivative. Wire it "
              "onto the SC rostral pool + approach->reward_us to close N5 (Option C).")
    elif sep > 0:
        print(f"VERDICT: PARTIAL — directionally correct (rise>fall by {sep:.2f}) but not cleanly "
              f"separated; tune nmda_slow weight / gaba_b strength / the lag, or add a dead-band.")
    else:
        print("VERDICT: NOT YET — the TD does not separate rise from fall; the nmda_slow lag may be too "
              "fast/slow vs the gaba_b subtraction. Tune the tau / weights.")


if __name__ == "__main__":
    main()
