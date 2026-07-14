"""PAST-RESERVOIR Rung 4b-ii (on-`SimulationBridge`): the additive SLOW SSM-state mechanism (`enable_selective_ssm_state`)
realizes the input-modulated-leak HOLD/RELEASE that the raw Izhikevich membrane could NOT (Rung 4b-i). The slow per-neuron
state s = lam_eff*s + (1-lam_eff)*inject, lam_eff = clip(1 - ssm_k_leak*(1+shunt), 0, 1), runs IN the bridge step loop; the
runner sets inject/shunt each step (the world/body interface writing the drive + gate) and reads cp_ssm_state. A LOW shunt
HOLDS the injected value across the filler; a HIGH shunt RELEASES it (leaks to 0). This is the on-bridge realization of the
selective-diagonal-SSM lambda (Rung 1-4a, transport-free, real-text-validated). ONE additive `sim/` edit
(`enable_selective_ssm_state`, default-off, byte-identical-when-off, verified).

TEST: inject a value (steps 0-3: inject=1, shunt HIGH so s rises fast to ~1), then FILLER (steps 4-15: inject=0), varying
the filler shunt:
  - HOLD arm  (low shunt):  s should stay ELEVATED across the filler (slow leak)
  - RELEASE arm (high shunt): s should DECAY to ~0 (fast shunt-driven leak)
GO (6-seed): mean s after the filler is HIGHER under HOLD than RELEASE by a clear margin (>0.3), HOLD stays >0.3, RELEASE
returns <0.15 -> the additive slow-state mechanism realizes the input-modulated leak on-bridge, unblocking the full
on-bridge selective SSM (gate + eligibility learning) as Rung 4b-iii.

Run: SIM_BACKEND=numpy python -m research.runners._reslm_rung4b_ii_onbridge_slow_ssm_state_derisk --seeds 42
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
INJECT_VAL = 1.0
SHUNT_INJECT = 4.0          # high shunt during injection -> lam_eff low -> s moves fast toward inject
SHUNT_HOLD = 0.0           # low shunt during filler -> lam_eff ~0.94 -> HOLD
SHUNT_RELEASE = 6.0        # high shunt during filler -> lam_eff ~0 -> RELEASE (leak to 0)
K_LEAK = 0.06


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
    cfg.enable_selective_ssm_state = True             # THE additive mechanism under test
    cfg.ssm_k_leak = K_LEAK
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    return b, cfg


def _run_arm(seed, shunt_filler):
    b, cfg = _build(seed)
    xp = b.xp
    n = cfg.num_neurons
    for step in range(INJECT_STEPS + FILLER_STEPS):
        if step < INJECT_STEPS:
            inj, sh = INJECT_VAL, SHUNT_INJECT
        else:
            inj, sh = 0.0, shunt_filler
        b.cp_ssm_inject[:] = xp.asarray(np.full(n, inj, np.float32))
        b.cp_ssm_shunt[:] = xp.asarray(np.full(n, sh, np.float32))
        b._run_one_simulation_step()
    return float(np.mean(_host(b.cp_ssm_state)))


def run(seed):
    hold = _run_arm(seed, SHUNT_HOLD)
    release = _run_arm(seed, SHUNT_RELEASE)
    go = bool(hold - release > 0.30 and hold > 0.30 and release < 0.15)
    print(f"[rung4b-ii seed={seed}] slow-state s after filler: HOLD={hold:.3f}  RELEASE={release:.3f} "
          f"| hold-release={hold-release:.3f} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "hold": hold, "release": release, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    print(f"[rung4b-ii] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
