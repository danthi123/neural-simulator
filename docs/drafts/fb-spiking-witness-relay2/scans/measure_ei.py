"""Measurement (not a lever): during a relay-driven placing write, per belief slot, the mean membrane voltage and the
inhibitory/excitatory conductances, against the GABA_A reversal the profile uses. Dev seed only."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scan import O
from sim.backend import to_host

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
wb = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
g = vars(O)
for k, v in O._BASE_PARAMS.items(): g[k] = v
g.update(dict(O._SHARED, MODE="relay_nmda", RELAY_BIAS_PA=0.0, W_WORLD_RELAY=0.25, W_PRES_RELAY=3.0,
              W_RELAY_BELIEF=wb, W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0))
o = O.SpikingWitnessFalseBeliefOrgan(seed)
o.perceive_arrive()
o._ensure()
b, idx = o._bridge, o._idx
cfg = b.core_config
print("E_i (GABA_A reversal) =", [getattr(cfg, n) for n in dir(cfg) if n.lower() in ("e_i", "e_i_mv", "inhibitory_reversal_mv", "reversal_inhibitory_mv")])
slots = {k: np.asarray(to_host(idx["belief_dev"][k])) for k in range(O.K_LOC)}
acc = {k: {"v": [], "gi": [], "ge": [], "spk": 0} for k in slots}
orig = b._run_one_simulation_step
def step():
    orig()
    v = np.asarray(to_host(b.cp_membrane_potential_v)); gi = np.asarray(to_host(b.cp_conductance_g_i)); ge = np.asarray(to_host(b.cp_conductance_g_e))
    f = np.asarray(to_host(b.cp_firing_states))
    for k, ix in slots.items():
        acc[k]["v"].append(float(v[ix].mean())); acc[k]["gi"].append(float(gi[ix].mean())); acc[k]["ge"].append(float(ge[ix].mean())); acc[k]["spk"] += int(f[ix].sum())
b._run_one_simulation_step = step
o.observe_event(0)
for k, a in acc.items():
    print(f"slot {k}: mean v {np.mean(a['v']):7.2f} mV (min {np.min(a['v']):7.2f}, max {np.max(a['v']):7.2f}) | mean g_i {np.mean(a['gi']):.3f} g_e {np.mean(a['ge']):.3f} | spikes {a['spk']}")
