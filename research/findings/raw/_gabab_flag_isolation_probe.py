"""READ-ONLY blocker-1 isolation probe (2026-06-08, deep-research subagent).

Question: does enabling cfg.enable_gabab=True ALONE (critic OFF, so NO pathway is
tagged receptor="gaba_b" => cp_gabab_synapse_mask is None / all-False) destabilize
the FULL flagship nav network — or does the silence require the critic region +
its GABA_B-tagged synapse?

The neural-critic NEGATIVE diag established: with the critic ON, flag ON silences
the whole net (snc/motors/it all 0) while flag forced-off at runtime navigates
fine — yet g_gabab is provably 0 in both. This probe decouples the FLAG from the
CRITIC. Three conditions, each a SHORT free-running flagship build (no reward loop,
no training — just step and watch whether the network keeps firing):

  (P0) flagship, critic OFF, enable_gabab OFF   (positive control: net active)
  (P1) flagship, critic OFF, enable_gabab ON  forced post-build (mask None/empty)
  (P2) flagship, critic ON,  enable_gabab ON  (reproduces the failing build)

If P1 goes silent like P2 => blocker 1 is the FLAG/empty-mask block path itself
(independent of the critic synapse). If P1 stays active and only P2 silences =>
the silence needs the critic's GABA_B synapse construction.

NO sim/ edits. Builds via g11.build_bg_brain_regions + a direct SimulationBridge.
Drives the cortex pools with a small tonic current so there is baseline activity
to observe (we are testing network STABILITY under the flag, not task learning).
SERIAL, GPU, ~150 steps/condition.
"""
import os
import sys
import json

os.environ.setdefault("SIM_BACKEND", "cupy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import research.runners.g11_bg_runner as g11
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.enums import NeuronModel

_ACTIONS = ["N", "E", "S", "W"]


def _build(seed, enable_neural_critic, force_gabab):
    """Build the flagship A+E+G v2.5 region set (critic optionally on) as a bare
    free-running bridge. We bypass run_moving_goal_episode and construct the cfg
    the way the runner does (mirrors lines ~3240-3280), so we can flip
    enable_gabab independently of the critic."""
    regions, pathways = g11.build_bg_brain_regions(
        enable_bg_lateral_inhibition=True,
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        enable_visual_cortex=True,
        enable_neural_critic=enable_neural_critic,
    )
    cfg = CoreSimConfig()
    cfg.seed = int(seed)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.05
    cfg.stdp_w_max = 150.0
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False
    # PFC NMDA (flagship): global NMDA on, masked to PFC/cortex via per-region mask.
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    # The independent variable:
    if force_gabab:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = 150.0
        cfg.gabab_propagation_strength = 0.105
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _idx(bridge, name):
    try:
        idx = bridge.region_manager.indices(name)
    except Exception:
        return None
    idx = np.asarray(idx.get() if hasattr(idx, "get") else idx)
    return idx if idx.size else None


def run_condition(label, seed, enable_neural_critic, force_gabab, n_steps=150):
    import cupy as cp
    print(f"\n{'='*70}\n  {label}: critic={enable_neural_critic} gabab={force_gabab}\n{'='*70}", flush=True)
    bridge, cfg = _build(seed, enable_neural_critic, force_gabab)

    snc_idx = _idx(bridge, "snc")
    it_idx = _idx(bridge, "cortex_it")
    striov_idx = _idx(bridge, "striosome_value")
    motor_idx = {a: _idx(bridge, f"motor_{a}") for a in _ACTIONS}
    cortex_idx = {a: _idx(bridge, f"cortex_{a}") for a in _ACTIONS}

    # Report whether the GABA_B mask got built and how many synapses it tags.
    gb_mask = getattr(bridge, "cp_gabab_synapse_mask", None)
    g_gabab = getattr(bridge, "cp_conductance_g_gabab", None)
    n_gb = int((gb_mask.get() if hasattr(gb_mask, "get") else np.asarray(gb_mask)).sum()) if gb_mask is not None else -1
    print(f"  cp_gabab_synapse_mask: {'None' if gb_mask is None else f'{n_gb} synapses tagged'}; "
          f"cp_conductance_g_gabab: {'None' if g_gabab is None else 'ALLOC'}; nnz={bridge.cp_connections.nnz}", flush=True)

    # Drive the 4 cortex pools with a steady current so there's baseline activity
    # to observe network stability (the SNc gets its own tonic via the cascade /
    # SNr disinhibition; we add cortex drive so the actor pathway is live).
    def _set_drive():
        bridge.cp_external_input_current[:] = 0.0
        for a in _ACTIONS:
            if cortex_idx[a] is not None:
                bridge.cp_external_input_current[cp.asarray(cortex_idx[a])] = cp.float32(400.0)
        # SNc tonic (Stage-A operating point) so the DA cell has a drive to fire from.
        if snc_idx is not None:
            bridge.cp_external_input_current[cp.asarray(snc_idx)] = cp.float32(220.0)

    def _rate(fs_h, idx):
        if idx is None:
            return float("nan")
        return float(fs_h[idx].mean())

    rows = []
    for step in range(n_steps):
        _set_drive()
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        fs = bridge.cp_firing_states
        fs_h = fs.get() if hasattr(fs, "get") else np.asarray(fs)
        v = bridge.cp_membrane_potential_v
        v_h = v.get() if hasattr(v, "get") else np.asarray(v)
        g_h = None
        if g_gabab is not None:
            g_h = g_gabab.get() if hasattr(g_gabab, "get") else np.asarray(g_gabab)
        rows.append({
            "snc": _rate(fs_h, snc_idx),
            "it": _rate(fs_h, it_idx),
            "striov": _rate(fs_h, striov_idx),
            "cortexN": _rate(fs_h, cortex_idx["N"]),
            "motorN": _rate(fs_h, motor_idx["N"]),
            "n_fired_total": int(fs_h.sum()),
            "v_nan": bool(np.isnan(v_h).any()),
            "v_snc": float(v_h[snc_idx].mean()) if snc_idx is not None else float("nan"),
            "gabab_all_max": float(g_h.max()) if g_h is not None else float("nan"),
        })

    def _m(key):
        vals = np.asarray([r[key] for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        return float(finite.mean()) if finite.size else float("nan")

    out = {
        "label": label, "critic": enable_neural_critic, "gabab": force_gabab,
        "n_gb_synapses": n_gb, "nnz": int(bridge.cp_connections.nnz),
        "snc_mean": _m("snc"), "snc_max": float(max(r["snc"] for r in rows)),
        "it_mean": _m("it"), "striov_mean": _m("striov"),
        "cortexN_mean": _m("cortexN"), "motorN_mean": _m("motorN"),
        "n_fired_total_mean": _m("n_fired_total"),
        "n_fired_total_last": rows[-1]["n_fired_total"],
        "any_v_nan": bool(any(r["v_nan"] for r in rows)),
        "v_snc_mean": _m("v_snc"),
        "gabab_all_max": _m("gabab_all_max"),
        "n_steps_snc_fired": int(sum(1 for r in rows if r["snc"] > 0)),
        "n_steps_any_fired": int(sum(1 for r in rows if r["n_fired_total"] > 0)),
    }
    print(f"  -> snc_mean={out['snc_mean']:.4f}  it_mean={out['it_mean']:.4f}  "
          f"cortexN_mean={out['cortexN_mean']:.4f}  n_fired_total_mean={out['n_fired_total_mean']:.1f}  "
          f"snc_fired={out['n_steps_snc_fired']}/{n_steps}  any_fired={out['n_steps_any_fired']}/{n_steps}  "
          f"v_nan={out['any_v_nan']}  gabab_all_max={out['gabab_all_max']:.4f}", flush=True)
    del bridge
    cp.get_default_memory_pool().free_all_blocks()
    return out


def main():
    seed = 42
    n_steps = 150
    results = {}
    results["P0"] = run_condition("P0 critic-OFF gabab-OFF", seed, False, False, n_steps)
    results["P1"] = run_condition("P1 critic-OFF gabab-ON", seed, False, True, n_steps)
    results["P2"] = run_condition("P2 critic-ON  gabab-ON", seed, True, True, n_steps)
    out_path = os.path.join(_HERE, "_gabab_flag_isolation_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWROTE {out_path}")
    print(f"\n{'='*70}\n  VERDICT\n{'='*70}")
    print(f"{'cond':<26} {'snc_mean':>9} {'n_fired_mean':>13} {'any_fired/N':>12} {'v_nan':>6} {'n_gb':>5}")
    for c in ("P0", "P1", "P2"):
        o = results[c]
        print(f"{o['label']:<26} {o['snc_mean']:>9.4f} {o['n_fired_total_mean']:>13.1f} "
              f"{str(o['n_steps_any_fired'])+'/'+str(n_steps):>12} {str(o['any_v_nan']):>6} {o['n_gb']:>5}"
              if 'n_gb' in o else "")
    # robust print (n_gb key)
    for c in ("P0", "P1", "P2"):
        o = results[c]
        print(f"  {o['label']:<26} snc={o['snc_mean']:.4f} n_fired_mean={o['n_fired_total_mean']:.1f} "
              f"any_fired={o['n_steps_any_fired']}/{n_steps} v_nan={o['any_v_nan']} n_gb_syn={o['n_gb_synapses']}")


if __name__ == "__main__":
    main()
