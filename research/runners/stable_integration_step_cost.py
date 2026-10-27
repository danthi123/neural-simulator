"""Per-step cost of SIM_STABLE_INTEGRATION on a fixed network (research/stable-integration; numpy CPU).

The census wall times compare whole conversations whose organ paths differ ON vs OFF (an organ the ON brain never
reaches is not stepped), so they are not a per-step cost. This times the SAME bridge build, OFF and ON, for a fixed
number of steps: an Izhikevich network with AMPA/GABA_A, GABA_B and NMDA pathways and OU noise (the shape of the
bounded-conductance reference network, scaled up). Interleaved repeats; median ms per step per arm.

  .venv/bin/python -m research.runners.stable_integration_step_cost --out research/findings/raw/_stable_integration/step_cost.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

ENV = "SIM_STABLE_INTEGRATION"


def _build(seed, scale):
    from sim import CoreSimConfig, GPUConfig, NeuronModel, RuntimeState, SimulationBridge, VisualizationConfig
    from sim.backend import get_backend
    from sim.regions import BrainRegion, RegionPathway

    def region(name, n, exc, ntype, nmda=False):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=exc, internal_density=0.05, exc_weight_mean=1.0,
                           inh_weight_mean=1.0, weight_jitter=0.1, plastic_internal=False, izh_neuron_type=ntype,
                           enable_homeostasis=False, enable_nmda=nmda)

    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 0.0
    cfg.ou_std_current_pA = 40.0
    cfg.ou_tau_ms = 15.0
    cfg.enable_nmda = True
    cfg.enable_gabab = True
    rs, fs = "IZH2007_RS_CORTICAL_PYRAMIDAL", "IZH2007_FS_CORTICAL_INTERNEURON"
    cfg.brain_regions = [region("ctx", 8 * scale, 1.0, rs, nmda=True), region("inh", 2 * scale, 0.0, fs),
                         region("gb", scale, 0.0, fs), region("drive", 2 * scale, 1.0, rs)]
    cfg.region_pathways = [
        RegionPathway(from_region="drive", to_region="ctx", density=0.1, weight_mean=20.0, weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="ctx", to_region="inh", density=0.1, weight_mean=10.0, weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="inh", to_region="ctx", density=0.2, weight_mean=30.0, weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="gb", to_region="ctx", density=0.1, weight_mean=5.0, weight_jitter=0.1, plastic=False,
                      receptor="gaba_b")]
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()
    for name, pA in (("drive", 900.0), ("gb", 900.0)):
        idx = np.asarray(b.region_manager.indices(name), dtype=np.int64)
        b.cp_external_input_current[xp.asarray(idx)] = xp.float32(pA)
    return b


def _time_arm(on, seed, scale, n_steps, warmup):
    if on:
        os.environ[ENV] = "1"
    else:
        os.environ.pop(ENV, None)
    b = _build(seed, scale)
    for _ in range(warmup):
        b._run_one_simulation_step()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        b._run_one_simulation_step()
    dt = time.perf_counter() - t0
    cfg = b.core_config
    return {"ms_per_step": 1000.0 * dt / n_steps, "n_neurons": int(cfg.num_neurons),
            "applied": bool(b.stable_integration_applied), "substep_calls": int(b.substep_calls),
            "effective": [bool(cfg.enable_bounded_conductance_integration), cfg.izh_membrane_substeps,
                          cfg.izh_substep_refractory_rule]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scales", default="25,250")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if os.environ.get("SIM_BACKEND", "numpy") != "numpy":
        sys.exit("numpy CPU only")
    saved = os.environ.get(ENV)
    res = {"runner": "research/runners/stable_integration_step_cost.py", "seed": a.seed, "steps": a.steps,
           "warmup": a.warmup, "reps": a.reps, "backend": "numpy", "scales": {}}
    try:
        for scale in [int(s) for s in a.scales.split(",")]:
            off, on = [], []
            for _ in range(a.reps):                   # interleaved, so drift hits both arms alike
                off.append(_time_arm(False, a.seed, scale, a.steps, a.warmup))
                on.append(_time_arm(True, a.seed, scale, a.steps, a.warmup))
            m_off = statistics.median(r["ms_per_step"] for r in off)
            m_on = statistics.median(r["ms_per_step"] for r in on)
            res["scales"][str(scale)] = {
                "n_neurons": off[0]["n_neurons"], "ms_per_step_off": [round(r["ms_per_step"], 4) for r in off],
                "ms_per_step_on": [round(r["ms_per_step"], 4) for r in on], "median_ms_off": round(m_off, 4),
                "median_ms_on": round(m_on, 4), "ratio_on_over_off": round(m_on / m_off, 3),
                "on_readback": on[0]["effective"], "on_applied": on[0]["applied"],
                "on_substep_calls": on[0]["substep_calls"], "off_readback": off[0]["effective"],
                "off_substep_calls": off[0]["substep_calls"]}
            print("[step cost] n=%d: off %.3f ms/step, on %.3f ms/step, ratio %.3f" % (
                off[0]["n_neurons"], m_off, m_on, m_on / m_off), flush=True)
    finally:
        if saved is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = saved
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("[step cost] -> %s" % a.out)


if __name__ == "__main__":
    main()
