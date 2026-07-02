"""EMERGE-11 / rung-4 Stage B1 — within-column WTA + the two-compartment dAP -> a dAP-PRIMED SUBSET wins the spiking
competition (sparse, context-specific firing) on the real substrate. This composes the two proven pieces
(Stage-A' two-compartment dAP GO + the nav FS-interneuron WTA recipe) and tests their integration = the core of
EMERGE-9c's spiking inference: predicted/dAP-primed cells fire first, drive the column's inhibitory neuron, and it
suppresses the not-yet-fired cells -> sparse; an UNPREDICTED column (no primed cells) BURSTS.

WIRING: one column of nE two-compartment Izhikevich cells + 1 FS (inhibitory) interneuron; column->FS (excitatory,
so the first spikers drive the FS) + FS->column (inhibitory, so the FS suppresses the rest). A distal
`coincidence_detector` context->column pathway primes a SUBSET (the "predicted" cells). Feedforward drives the whole
column. GO: with dAP + WTA, only the PRIMED subset fires (sparse); WTA-LESION (no FS) -> the whole column fires
(burst); dAP-LESION (no coincidence) -> no primed subset -> burst or silence (no context-specific selection).
Multi-seed. NO new sim/ edit (reuses the Stage-A' two-compartment dAP + the region/pathway framework).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

OUT = Path("research/findings/raw/_emerge11_stageB_wta_dap.json")


def _host(x):
    try:
        return x.get()
    except AttributeError:
        return np.asarray(x)


def build_bridge(seed, n_ctx=40, n_col=12, n_primed=4, k_threshold=8.0, ctx_col_weight=0.1,
                 col_fs_weight=60.0, fs_col_weight=400.0, two_compartment=True, wta=True, coincidence=True,
                 apical_g_couple=1.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    # the "column" is split into a PRIMED subset (context primes these) + the OTHER cells; both share one FS-WTA.
    n_other = int(n_col) - int(n_primed)
    _rs = dict(exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
               plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    regions = [
        BrainRegion(name="context", n_neurons=int(n_ctx), **_rs),
        BrainRegion(name="col_primed", n_neurons=int(n_primed), **_rs),
        BrainRegion(name="col_other", n_neurons=int(n_other), **_rs),
        BrainRegion(name="fs", n_neurons=1, exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0,
                    inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name),
    ]
    pathways = [
        # distal dAP: context -> the PRIMED subset only (a clustered coincidence projection)
        RegionPathway(from_region="context", to_region="col_primed", density=1.0, weight_mean=float(ctx_col_weight),
                      weight_jitter=0.0, plastic=False, coincidence_detector=bool(coincidence)),
    ]
    if wta:
        # within-column WTA: both subsets drive the FS; the FS inhibits both (first spikers escape, the rest are caught)
        for src in ("col_primed", "col_other"):
            pathways.append(RegionPathway(from_region=src, to_region="fs", density=1.0, weight_mean=float(col_fs_weight),
                                          weight_jitter=0.0, plastic=False))
            pathways.append(RegionPathway(from_region="fs", to_region=src, density=1.0, weight_mean=float(fs_col_weight),
                                          weight_jitter=0.0, plastic=False))
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
    cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
              "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_coincidence_detection = bool(coincidence)
    cfg.coincidence_k_threshold = float(k_threshold)
    cfg.enable_two_compartment_dap = bool(two_compartment)
    cfg.apical_g_couple = float(apical_g_couple)
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def run_trial(bridge, ctx_idx, primed_idx, other_idx, prime, i_ff, n_steps=12, ctx_drive=600.0):
    """Drive context (if prime) + feedforward on both column subsets; return (primed fired-frac, other fired-frac)."""
    xp = bridge.xp if hasattr(bridge, "xp") else np
    cp = np.zeros(len(primed_idx)); co = np.zeros(len(other_idx))
    for t in range(n_steps):
        ext = _host(bridge.cp_external_input_current) * 0.0
        if prime:
            ext[ctx_idx] = ctx_drive
        ext[primed_idx] = i_ff; ext[other_idx] = i_ff
        bridge.cp_external_input_current[:] = xp.asarray(ext)
        bridge._run_one_simulation_step()
        f = _host(bridge.cp_firing_states)
        cp += f[primed_idx].astype(float); co += f[other_idx].astype(float)
    return float((cp > 0).mean()), float((co > 0).mean())


def _run_seed(seed, i_grid, n_col, n_primed):
    """Sweep the feedforward drive; for each arm report the BEST selection point (max primed - other over the grid)
    -- the transition zone where the dAP-primed subset fires but the others (without the plateau) do not."""
    res = {}
    for arm, (tc, wta, coin) in {"dap_wta": (True, True, True), "wta_only": (False, True, True),
                                 "dap_nowta": (True, False, True), "lesion_nodap": (False, True, False)}.items():
        best = {"primed_rate": 0.0, "other_rate": 0.0, "gap": -1.0, "i_ff": 0.0, "max_sparsity": 0.0}
        for i_ff in i_grid:
            b, _ = build_bridge(seed, n_col=n_col, n_primed=n_primed, two_compartment=tc, wta=wta, coincidence=coin)
            rm = b.region_manager
            ctx = np.asarray(rm.indices("context"), int)
            pri = np.asarray(rm.indices("col_primed"), int); oth = np.asarray(rm.indices("col_other"), int)
            pr, ot = run_trial(b, ctx, pri, oth, prime=True, i_ff=i_ff)
            sp = (pr * len(pri) + ot * len(oth)) / max(1, len(pri) + len(oth))
            best["max_sparsity"] = max(best["max_sparsity"], sp)
            if pr - ot > best["gap"]:
                best.update({"primed_rate": pr, "other_rate": ot, "gap": pr - ot, "i_ff": float(i_ff)})
        res[arm] = best
    return seed, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--i-min", type=float, default=300.0)
    ap.add_argument("--i-max", type=float, default=700.0)
    ap.add_argument("--i-steps", type=int, default=21)
    ap.add_argument("--n-col", type=int, default=12)
    ap.add_argument("--n-primed", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    i_grid = np.linspace(a.i_min, a.i_max, a.i_steps)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            seed, res = _run_seed(s, i_grid, a.n_col, a.n_primed); res["seed"] = seed; per.append(res)
        for d in per:
            print(f"  [seed {d['seed']}] DAP+WTA primed {d['dap_wta']['primed_rate']:.2f}/other {d['dap_wta']['other_rate']:.2f} @{d['dap_wta']['i_ff']:.0f} "
                  f"| wta_only {d['wta_only']['primed_rate']:.2f}/{d['wta_only']['other_rate']:.2f} "
                  f"| dap_noWTA maxsparsity {d['dap_nowta']['max_sparsity']:.2f} | noDAP {d['lesion_nodap']['primed_rate']:.2f}/{d['lesion_nodap']['other_rate']:.2f}",
                  flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, key):
            return float(np.mean([p[arm][key] for p in per]))
        dw_primed, dw_other = m("dap_wta", "primed_rate"), m("dap_wta", "other_rate")
        wo_primed, wo_other = m("wta_only", "primed_rate"), m("wta_only", "other_rate")
        dnw_sparsity = m("dap_nowta", "max_sparsity")
        # GO: DAP+WTA -> the PRIMED subset fires while the OTHERS are suppressed (sparse, context-specific selection).
        selects = dw_primed >= 0.8 and dw_other <= 0.3
        wta_loadbearing = dnw_sparsity >= dw_other + 0.3         # removing WTA lets the non-primed fire (burst)
        dap_loadbearing = (dw_primed - dw_other) >= (wo_primed - wo_other) + 0.3  # DAP creates the primed-vs-other gap
        go = bool(selects and wta_loadbearing and dap_loadbearing)
        if go:
            verdict = (f"GO -- the two-compartment dAP + within-column WTA select the PRIMED subset on the real "
                       f"substrate: DAP+WTA primed-fire {dw_primed:.2f} vs other {dw_other:.2f} (sparse, context-"
                       f"specific); removing the WTA lets the non-primed fire (dap_noWTA sparsity {dnw_sparsity:.2f}); "
                       f"removing the dAP loses the primed-vs-other gap (wta_only {wo_primed:.2f}/{wo_other:.2f}); "
                       f"multi-seed. => EMERGE-9c's spiking-inference selection (dAP-primed wins the WTA) works on the "
                       f"bridge -> Stage B2 (load frozen EMERGE-9b permanences into a recurrent distal pathway + drive "
                       f"an overlapping-sequence -> branch-prediction parity).")
        else:
            miss = []
            if not selects: miss.append(f"DAP+WTA didn't select the primed subset (primed {dw_primed:.2f}, other {dw_other:.2f})")
            if not wta_loadbearing: miss.append(f"WTA not load-bearing (dap_noWTA sparsity {dnw_sparsity:.2f} vs other {dw_other:.2f})")
            if not dap_loadbearing: miss.append(f"dAP not load-bearing (gap {dw_primed-dw_other:.2f} vs wta_only {wo_primed-wo_other:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f". Tune the WTA operating point "
                       f"(col_fs_weight / fs_col_weight so the FS fires AFTER the primed cells and suppresses the rest) "
                       f"jointly with i_ff + apical_g_couple (primed fire first). NOT a wall; the WTA timing is next.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge11_stageB_wta_dap", "verdict": verdict,
               "mechanism": "within-column FS-interneuron WTA + the two-compartment dAP (Stage-A') -> dAP-primed cells "
                            "fire first, drive the FS, and it suppresses the rest -> sparse context-specific firing "
                            "(the core of EMERGE-9c spiking inference) on the real SimulationBridge",
               "seeds": a.seeds, "config": {"i_grid": i_grid.tolist(), "n_col": a.n_col, "n_primed": a.n_primed},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "rung-4 Stage B1: composes the two proven pieces (two-compartment dAP GO + nav FS-WTA). "
                              "B2 loads the frozen EMERGE-9b permanences into a recurrent distal pathway + drives the "
                              "overlapping-sequence -> branch-prediction parity with EMERGE-9c. Then Stage C (three-term "
                              "kernel -> EMERGE-9d parity) = rung-4 complete."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge11] VERDICT: {verdict}", flush=True)
    print(f"[emerge11] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
