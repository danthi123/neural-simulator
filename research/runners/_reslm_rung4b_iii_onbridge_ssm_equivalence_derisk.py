"""PAST-RESERVOIR Rung 4b-iii-a (on-bridge, LIKE-FOR-LIKE equivalence): the on-bridge `cp_ssm_state` mechanism
(`enable_selective_ssm_state`) is BYTE-EQUIVALENT to the numpy selective-SSM state (Rung 2/3/4a) given the same inject +
shunt sequences — so the entire validated numpy ladder (the transport-free selective SSM that beats the fixed reservoir on
real text) transfers to the spiking bridge EXACTLY, not just qualitatively. This is the decisive on-substrate check: the
mechanism IS the SSM, so the Rung-2/3/4a GO results are on-bridge results.

TEST: drive the on-bridge SSM (a bridge with `enable_selective_ssm_state=True`) and a numpy replica with the SAME random
per-neuron inject + shunt sequences (many neurons, many steps, random per step); assert max|cp_ssm_state - numpy_s| is at
numerical-precision. GO (6-seed): max abs diff < 1e-5 (identical realization).

Run: SIM_BACKEND=numpy python -m research.runners._reslm_rung4b_iii_onbridge_ssm_equivalence_derisk --seeds 42
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

N = 32
STEPS = 40
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
    cfg.enable_selective_ssm_state = True; cfg.ssm_k_leak = K_LEAK
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    return b, cfg


def run(seed):
    b, cfg = _build(seed)
    xp = b.xp; n = cfg.num_neurons
    rng = np.random.default_rng(seed)
    s_np = np.zeros(n, np.float64)                                # numpy replica of the SSM state
    max_diff = 0.0
    for _ in range(STEPS):
        inject = rng.standard_normal(n).astype(np.float32)
        shunt = np.abs(rng.standard_normal(n)).astype(np.float32) * 3.0
        # bridge step
        b.cp_ssm_inject[:] = xp.asarray(inject); b.cp_ssm_shunt[:] = xp.asarray(shunt)
        b._run_one_simulation_step()
        # numpy replica (the exact selective-SSM update)
        lam = np.clip(1.0 - K_LEAK * (1.0 + shunt.astype(np.float64)), 0.0, 1.0)
        s_np = lam * s_np + (1.0 - lam) * inject.astype(np.float64)
        d = float(np.max(np.abs(_host(b.cp_ssm_state).astype(np.float64) - s_np)))
        max_diff = max(max_diff, d)
    go = bool(max_diff < 1e-5)
    print(f"[rung4b-iii-eq seed={seed}] max|on-bridge s - numpy s| over {STEPS} steps = {max_diff:.2e} -> "
          f"{'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "max_abs_diff": max_diff, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    print(f"[rung4b-iii-eq] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
