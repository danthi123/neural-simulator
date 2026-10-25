"""Fix-round probe (scratch): the target cells' own f-I in this engine config (direct current, no synapse), and their
per-neuron Izhikevich parameters after heterogeneity, to locate the rheobase the synaptic arithmetic must reach."""
import os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np, logging
logging.disable(logging.CRITICAL)
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel
from sim.bridge import SimulationBridge
def build(seed, n=12):
    cfg = CoreSimConfig(); cfg.num_neurons = n; cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.seed = seed; cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_watts_strogatz", "enable_ou_process", "enable_hebbian_learning"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    return b
b = build(7)
for nm in ("cp_izh_C", "cp_izh_k", "cp_izh_vr", "cp_izh_vt", "cp_izh_a", "cp_izh_b", "cp_izh_d"):
    a = getattr(b, nm, None)
    if a is not None: a = np.asarray(a); print(nm, "min %.3f med %.3f max %.3f" % (a.min(), np.median(a), a.max()))
for I in (40, 60, 80, 100, 150, 200, 300, 400):
    sp = []; lat = []
    for s in range(1, 6):
        b = build(s); b.cp_external_input_current[:] = 0.0
        for _ in range(60): b._run_one_simulation_step()
        b.cp_external_input_current[:] = float(I); first = np.full(12, -1); cnt = np.zeros(12, int)
        for t in range(40):
            b._run_one_simulation_step(); fs = np.asarray(b.cp_firing_states).astype(bool)
            cnt += fs; first[(first < 0) & fs] = t
        sp.append(cnt.sum()); lat.extend([x for x in first if x >= 0])
    print("I=%4d pA: spikes/12 cells/40 ms med %d (seeds 1-5: %s); cells firing first-spike latency med %s ms" % (
        I, int(np.median(sp)), sp, (int(np.median(lat)) if lat else None)))
