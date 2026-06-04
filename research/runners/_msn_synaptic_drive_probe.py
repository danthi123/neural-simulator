"""#3 close, cheap-first: can a SCALED presynaptic pool fire the high-rheobase striatal MSN-D1 SYNAPTICALLY,
so the genuine #2 disinhibition cascade completes from a learned cue (no direct current)?

#3's learned cortico-striatal STDP works (correct verb->D1 grows to ~16) but the learned weight with a 30-neuron
verb pool did NOT fire the MSN-D1 at inference (#2 sidestepped this with direct current). The hypothesis: it's a
DRIVE-MAGNITUDE gap -- summed synaptic drive ~ (n_presynaptic x rate x weight), and the validated Tier-1
word->action recipe uses 500-1000 neuron pools. This probe sweeps the verb pool size at a FIXED functional weight
(skipping the learning, to isolate the drive question) and checks the full cascade verb -> d1 -> (silence) gpi ->
(release) thal -> open gate -> motor. If a Tier-1-scale pool fires the MSN and routes, the #3 end-to-end closes by
scale-up; a full retrain then confirms the LEARNED weight does the same.

  SIM_BACKEND=numpy python -m research.runners._msn_synaptic_drive_probe
"""
import numpy as np

from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronType
from research.runners.gated_compose_bg_genuine_demo import THAL_TONIC_PA, GPI_TONIC_PA


def build(seed=42, n_verb=500, n=30, w_vd1=16.0, d1_w=15.0, gpi_w=8.0, route_w=40.0):
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation"):
        setattr(cfg, flag, False)
    cfg.brain_regions = [
        BrainRegion(name="verb_GO", n_neurons=n_verb, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="motor_N", n_neurons=n, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="d1_GO_N", n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name),
        BrainRegion(name="gpi_GO_N", n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                    izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name),
        BrainRegion(name="thal_GO_N", n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                    izh_neuron_type=NeuronType.IZH2007_THALAMIC_RELAY.name),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="verb_GO", to_region="d1_GO_N", density=1.0, weight_mean=w_vd1,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="d1_GO_N", to_region="gpi_GO_N", density=1.0, weight_mean=d1_w,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="gpi_GO_N", to_region="thal_GO_N", density=1.0, weight_mean=gpi_w,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="verb_GO", to_region="motor_N", density=1.0, weight_mean=route_w,
                      weight_jitter=0.0, plastic=False, transmission_gate="g_GO_N"),
    ]
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    sb.couple_gate_to_pool("g_GO_N", "thal_GO_N", threshold=0.03)
    return sb


def _rates(sb, drive_verb, settle=80):
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    sb.cp_external_input_current[np.asarray(sb.region_manager.indices("gpi_GO_N"))] = GPI_TONIC_PA
    sb.cp_external_input_current[np.asarray(sb.region_manager.indices("thal_GO_N"))] = THAL_TONIC_PA
    if drive_verb:
        sb.cp_external_input_current[np.asarray(sb.region_manager.indices("verb_GO"))] = 1500.0
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb._run_one_simulation_step()
        sb.runtime_state.current_time_ms += sb.core_config.dt_ms
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    g = lambda nm: acc[np.asarray(sb.region_manager.indices(nm))].mean() / settle
    return g("d1_GO_N"), g("gpi_GO_N"), g("thal_GO_N"), g("motor_N")


def main():
    print("=== #3 close cheap-first: does a SCALED verb pool fire the MSN-D1 cascade synaptically? ===\n", flush=True)
    print("  (drive verb_GO at 1500 pA; verb->d1 fixed weight 16; check d1 fires -> gpi silenced -> thal released"
          " -> motor_N)\n", flush=True)
    for n_verb in (30, 100, 300, 500, 1000):
        sb = build(seed=42, n_verb=n_verb)
        d1b, gpib, thb, mnb = _rates(sb, drive_verb=False)
        d1, gpi, th, mn = _rates(sb, drive_verb=True)
        fires = d1 > 0.03 and th > thb + 0.02 and mn > 0.05
        print(f"  verb pool={n_verb:>4}: d1={d1:.3f} gpi={gpi:.3f}(base {gpib:.3f}) thal={th:.3f}(base {thb:.3f}) "
              f"motor_N={mn:.3f}  -> {'CASCADE FIRES' if fires else 'too weak'}", flush=True)


if __name__ == "__main__":
    main()
