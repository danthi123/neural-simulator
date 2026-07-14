"""PAST-RESERVOIR Rung 4b-i (the on-`SimulationBridge` realization, step 1): does an INPUT-MODULATED SHUNTING CONDUCTANCE
realize the selective SSM's hold/release leak ON A REAL BRIDGE NEURON? This is the core primitive of the on-bridge Rung 4
— a sub-threshold neuron's membrane V is a graded leaky integrator toward V_rest; an inhibitory conductance g_i with its
reversal E_i = V_rest is a PURE SHUNT (a leak term g_i*(V_rest - V), no hyperpolarization below rest), so a LOW g_i HOLDS
the integrated value and a HIGH g_i RELEASES it (leaks to rest). That input-modulated leak = the selective lambda, on the
bridge, using ONLY existing conductance mechanics (the a0 confirmed: `fused_conductance_decay_and_current` computes
g_i*(E_i - V), and E_i is settable per-neuron). NO `sim/` edit.

TEST: drive a small population's excitatory conductance g_e for a few steps (INJECT -> V rises above rest), then FILLER
steps, then vary g_i:
  - HOLD arm  (low g_i during filler):    V should stay ELEVATED across the filler (slow leak)
  - RELEASE arm (high g_i during filler): V should DECAY back toward V_rest (fast shunt-driven leak)
GO (6-seed): the mean membrane V after the filler is HIGHER under HOLD than under RELEASE by a clear margin, and HOLD
stays above rest while RELEASE returns near rest -> the input-modulated shunt realizes the hold/release leak on-bridge,
green-lighting the full on-bridge selective SSM (learned gate) as Rung 4b-ii.

Run: SIM_BACKEND=numpy python -m research.runners._reslm_rung4b_onbridge_shunt_primitive_derisk --seeds 42
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np

from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.bridge import SimulationBridge
from sim.regions import BrainRegion
from sim.enums import NeuronModel, NeuronType
from sim.backend import to_host as _host

N = 16
INJECT_STEPS = 4
FILLER_STEPS = 12
G_E_INJECT = 3.0             # excitatory conductance during injection (sub-threshold -> V rises but does not spike)
G_I_HOLD = 0.02             # low shunt -> HOLD
G_I_RELEASE = 0.8           # high shunt -> RELEASE (leak to rest)


def _build(seed):
    region = BrainRegion(name="ssm", n_neurons=N, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                         inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                         izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [region]; cfg.region_pathways = []
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
    cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
              "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity",
              "enable_coincidence_detection", "enable_two_compartment_dap"):
        setattr(cfg, f, False)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    return b, cfg


def _v_rest(b, cfg):
    # Izhikevich resting potential vr (per-neuron array or the config default)
    vr = getattr(b, "cp_izh_vr", None)
    if vr is not None:
        return float(np.mean(_host(vr)))
    return float(getattr(cfg, "izh_vr", -60.0))


def _set_shunt_reversal_to_rest(b, vr):
    import sim.backend as _bk
    xp = b.xp
    b.cp_syn_reversal_potential_i_per_neuron = xp.asarray(np.full(N, vr, np.float32))


def _run_arm(seed, g_i_filler):
    b, cfg = _build(seed)
    xp = b.xp
    vr = _v_rest(b, cfg)
    _set_shunt_reversal_to_rest(b, vr)
    n = cfg.num_neurons
    ge = np.zeros(n, np.float32); gi = np.zeros(n, np.float32)
    for step in range(INJECT_STEPS + FILLER_STEPS):
        inj = G_E_INJECT if step < INJECT_STEPS else 0.0
        shunt = 0.0 if step < INJECT_STEPS else g_i_filler
        b.cp_conductance_g_e[:] = xp.asarray(np.full(n, inj, np.float32))
        b.cp_conductance_g_i[:] = xp.asarray(np.full(n, shunt, np.float32))
        b._run_one_simulation_step()
    v = _host(b.cp_membrane_potential_v)
    return float(np.mean(v) - vr)                              # mean membrane ABOVE rest after the filler


def run(seed):
    hold = _run_arm(seed, G_I_HOLD)
    release = _run_arm(seed, G_I_RELEASE)
    go = bool(hold > release + 1.0 and hold > 1.0 and release < hold * 0.6 + 0.5)
    print(f"[rung4b seed={seed}] V-above-rest after filler: HOLD={hold:.2f} mV  RELEASE={release:.2f} mV "
          f"| hold>release by {hold-release:.2f} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "hold_mv": hold, "release_mv": release, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    print(f"[rung4b] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
