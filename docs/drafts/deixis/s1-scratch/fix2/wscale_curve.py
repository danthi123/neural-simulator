"""Fix-round probe (scratch; read-only on the repo): the engine's transfer curve for one learned route.
A token assembly of NT neurons driven at DRIVE pA (the word-role front-end convention, word_role_parser_organ.py:127)
projects all-to-all at weight w onto a target pool of NP neurons; default propagation_strength 0.05 (sim/config.py:871),
syn_tau_g_e 5 ms (config.py:394), E_e 0 mV (config.py:392), Izhikevich-2007 RS cells (config.py:20-23) with the
organ's per-neuron heterogeneity on. Reports, per (NT, w), over seeds: TOK spikes, target-pool spikes in a 40-step
token, and the fraction of target neurons that fired at least once. Also the conductance arithmetic check: the
steady-state g_e and I_syn the target sees, measured from the bridge."""
import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np, logging
logging.disable(logging.WARNING)
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel
from sim.bridge import SimulationBridge

def run(w, NT=8, NP=12, steps=40, seed=7, drive=1500.0):
    n = NT + NP
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
    tok = np.arange(NT); P = np.arange(NT, n)
    pre = np.repeat(tok, NP); post = np.tile(P, NT)
    plan = {"tok_p": {"pre_indices": pre, "post_indices": post, "initial_weights": np.full(pre.size, float(w), np.float32),
                      "plastic": False, "plasticity_gate": "fixed", "conn_type": "E_TO_E", "count": int(pre.size)}}
    b.inject_explicit_wiring(plan, output_inhibitory_indices=np.array([], dtype=np.int64))
    b.cp_external_input_current[:] = 0.0
    for _ in range(60): b._run_one_simulation_step()
    cur = np.zeros(n, dtype=np.float32); cur[tok] = drive
    b.cp_external_input_current[:] = cur
    tsp = psp = 0; fired = np.zeros(NP, bool); ge = []
    for _ in range(steps):
        b._run_one_simulation_step()
        fs = np.asarray(b.cp_firing_states)
        tsp += int(fs[tok].sum()); psp += int(fs[P].sum()); fired |= fs[P].astype(bool)
        ge.append(float(np.asarray(b.cp_conductance_g_e)[P].mean()))
    return tsp, psp, float(fired.mean()), float(np.mean(ge[10:]))

if __name__ == "__main__":
    out = {}
    seeds = list(range(1, 11))
    for NT in (8, 12):
        for w in (2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30):
            rows = [run(w, NT=NT, seed=s) for s in seeds]
            t = [r[0] for r in rows]; p = [r[1] for r in rows]; f = [r[2] for r in rows]; g = [r[3] for r in rows]
            out["%d_%g" % (NT, w)] = rows
            print("NT=%2d w=%5.1f  TOK spikes med %4d | P spikes min %3d med %3d max %3d | frac P fired min %.2f med %.2f | mean g_e(P) %.3f nS"
                  % (NT, w, int(np.median(t)), min(p), int(np.median(p)), max(p), min(f), float(np.median(f)), float(np.median(g))), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wscale_curve.json"), "w"))
