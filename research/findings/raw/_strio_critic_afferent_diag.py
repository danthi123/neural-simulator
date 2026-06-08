"""READ-ONLY diagnostic #2: is the WALL the afferent conduit (weight x density x #active
presynaptic cells), and can a DEDICATED DENSE place afferent fire the MSN-D1 critic with
OU OFF (deterministic nav)? NO sim/ edits.

Tests, all OU OFF (deterministic nav regime):
  (E) Afferent weight sweep: place region pinned firing (strong), vary place->strio weight.
  (F) #active-presynaptic-cells sweep: hold per-cell drive, vary how many place cells are
      active (sparse nav code ~1-3 cells vs dense dedicated input ~all 40). The nav
      sensor_place_readout fires ~0.57 Hz with ~1-3 cells active; a dense dedicated input
      would have many more.
  (G) Combined: a realistic DEDICATED DENSE afferent (N=200 place cells, density 0.5,
      weight 6.0) at moderate per-cell firing -> does the MSN-D1 critic reach a graded
      up-state-like firing (20-60 Hz) WITHOUT OU and WITHOUT a 1500 pA actor-perturbing drive?

Run: SIM_BACKEND=numpy PYTHONPATH=<root> python research/findings/raw/_strio_critic_afferent_diag.py
"""
import os, sys
sys.path.insert(0, os.environ.get("REPO_ROOT", os.getcwd()))
import numpy as np


def build(seed=42, n_place=40, n_strio=60, place_density=0.6, place_weight=3.0,
          strio_type="IZH2007_STRIATAL_MSN_D1"):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    cfg = CoreSimConfig()
    cfg.seed = seed; cfg.heterogeneity_seed = seed; cfg.ou_seed = seed
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = False; cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = False          # deterministic nav
    cfg.enable_conductance_noise = False
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
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="place", to_region="striosome_value",
                      density=float(place_density), weight_mean=float(place_weight),
                      weight_jitter=0.0, plastic=False),
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
    try: return to_host(a)
    except Exception: return a


def strio_rate(bridge, xp, place_pa_vec, n_steps=250, warmup=80):
    """place_pa_vec: per-cell pA vector over the place region. Returns striosome mean Hz."""
    si = xp.asarray(idx(bridge, "striosome_value")); pi = xp.asarray(idx(bridge, "place"))
    n_s = len(host(si))
    spk = 0; m = 0
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[pi] = xp.asarray(place_pa_vec, dtype=xp.float32)
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

    # (E) afferent WEIGHT sweep, all 40 place cells strongly driven (1500 pA each), OU off
    print("=== (E) place->strio WEIGHT sweep (all 40 place cells @1500pA, density 0.6), OU OFF ===")
    for w in [3.0, 6.0, 10.0, 15.0, 25.0, 40.0]:
        b, _ = build(place_weight=w)
        vec = np.full(40, 1500.0, dtype=np.float32)
        r = strio_rate(b, xp, vec)
        print(f"  weight={w:5.1f}  ->  striosome {r:7.2f} Hz")

    # (F) #ACTIVE-presynaptic-cells sweep (sparse nav code vs dense), w=10, each active @1500pA
    print("\n=== (F) #active place cells sweep (each active @1500pA, w=10, density 0.6), OU OFF ===")
    b2, _ = build(place_weight=10.0)
    for k in [1, 2, 3, 5, 10, 20, 40]:
        vec = np.zeros(40, dtype=np.float32); vec[:k] = 1500.0
        r = strio_rate(b2, xp, vec)
        print(f"  active_cells={k:3d}/40  ->  striosome {r:7.2f} Hz   "
              f"(nav sensor_place_readout has ~1-3 active)")

    # (G) DEDICATED DENSE afferent: N=200 place cells, density 0.5, weight 6.0; a realistic
    #     dense place input firing at a moderate ~30-40% bump. OU OFF. Does the critic reach
    #     a graded up-state firing without a 1500pA actor-perturbing direct drive?
    print("\n=== (G) DEDICATED DENSE afferent (N=200, density 0.5, w=6.0), OU OFF ===")
    for active_frac, per_pa in [(0.10, 800.0), (0.25, 800.0), (0.40, 800.0), (0.40, 1200.0)]:
        b3, _ = build(n_place=200, place_density=0.5, place_weight=6.0)
        n_act = int(0.5 + active_frac * 200)
        vec = np.zeros(200, dtype=np.float32); vec[:n_act] = per_pa
        r = strio_rate(b3, xp, vec)
        # place rate
        pi = xp.asarray(idx(b3, "place")); n_p = len(host(pi)); spk = 0
        for _ in range(60):
            b3.cp_external_input_current[:] = 0.0
            b3.cp_external_input_current[pi] = xp.asarray(vec, dtype=xp.float32)
            b3._run_one_simulation_step(); b3.runtime_state.current_time_step += 1
            b3.runtime_state.current_time_ms = b3.runtime_state.current_time_step * b3.core_config.dt_ms
            spk += int(host(b3.cp_firing_states[pi]).sum())
        place_rate = spk / max(n_p, 1) / 60e-3
        print(f"  active={n_act:3d}/200 @{per_pa:.0f}pA (place {place_rate:6.1f}Hz)  ->  striosome {r:7.2f} Hz")


if __name__ == "__main__":
    main()
