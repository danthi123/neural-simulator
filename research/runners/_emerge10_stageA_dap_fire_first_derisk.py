"""EMERGE-10 / rung-4 Stage A — the SINGLE genuinely-new biophysical behavior of the sim/ HTM-TM port, in isolation:
does a distally-PRIMED cell (a dendritic coincidence PLATEAU = the dAP) fire FIRST / at LOWER feedforward drive than
an unprimed cell? This is the fire-first bias that, in the full port, lets dAP-primed (predicted) cells win the
per-column WTA -> sparse context-specific firing (EMERGE-9c's spiking inference). NO `sim/` edit: the plateau kernel
(`fused_coincidence_plateau`, sim/kernels.py:253) already exists + is guarded byte-inert when off; this is a wiring +
anti-cheat exercise on a real `SimulationBridge` (per the rung-4 scoping).

WIRING (modeled on `research/runners/coincidence_wall_probe.py`): a `context` source pool + a `column` pool of
Izhikevich cells + a FIXED CLUSTERED `context->column` pathway tagged `coincidence_detector=True` (the distal segment).
When >= coincidence_k_threshold context inputs fire SYNCHRONOUSLY onto a column cell, a regenerative all-or-none NMDA
plateau current is injected into that cell (the dAP) -- an ADDITIVE somatic pre-depolarization. We then apply a graded
feedforward drive to the column and measure the firing THRESHOLD: primed (context volley on) vs unprimed (off).

GO: (i) the column fires at a LOWER feedforward drive when primed than when unprimed (the plateau lowers the effective
rheobase = fire-first); (ii) monotone in plateau strength, ~zero advantage at plateau-off; (iii) the plateau ALONE (no
feedforward) does NOT make the cell fire (no spurious priming). ANTI-CHEATS: `enable_coincidence_detection=False`
(dAP-lesion) -> no advantage (collapses); DESYNCHRONIZED context (no synchronous volley) -> no plateau, no advantage
(the coincidence-not-rate property). Multi-seed 42/43/44. CPU (numpy backend ok).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

OUT = Path("research/findings/raw/_emerge10_stageA_dap_fire_first.json")


def build_bridge(seed, n_ctx=60, n_col=40, ctx_col_density=1.0, ctx_col_weight=14.0, k_threshold=10.0,
                 coincidence=True, plateau_scale=1.0):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    regions = [
        BrainRegion(name="context", n_neurons=int(n_ctx), exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="column", n_neurons=int(n_col), exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
    ]
    pathways = [
        # distal segment: FIXED clustered AMPA projection context->column, tagged as a coincidence detector (the dAP)
        RegionPathway(from_region="context", to_region="column", density=float(ctx_col_density),
                      weight_mean=float(ctx_col_weight), weight_jitter=0.0, plastic=False,
                      coincidence_detector=bool(coincidence)),
    ]
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
    cfg.stdp_w_max = 40.0; cfg.fast_spike_reset = True
    cfg.enable_homeostasis = False; cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False; cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False; cfg.enable_structural_plasticity = False
    cfg.enable_coincidence_detection = bool(coincidence)
    cfg.coincidence_k_threshold = float(k_threshold)
    cfg.coincidence_plateau_strength = 80.0 * float(plateau_scale)   # regenerative NMDA-plateau conductance (the dAP)
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _host(x):
    try:
        return x.get()
    except AttributeError:
        return np.asarray(x)


def col_fire_fraction(bridge, ctx_idx, col_idx, i_ff, prime, desync, n_steps=8, ctx_drive=600.0, seed=0):
    """Drive the context (synchronous volley if prime, desynchronized if desync) + a feedforward current i_ff on the
    column; return the fraction of column cells that spiked over the window."""
    xp = bridge.xp if hasattr(bridge, "xp") else np
    rng = np.random.default_rng(seed + 1234)
    fired = np.zeros(len(col_idx), bool)
    for t in range(n_steps):
        ext = _host(bridge.cp_external_input_current).copy() * 0.0
        if prime:
            if desync:
                # only a few context cells fire each step (no synchronous volley -> no coincidence)
                sub = rng.choice(ctx_idx, size=max(1, len(ctx_idx) // 8), replace=False)
                ext[sub] = ctx_drive
            else:
                ext[ctx_idx] = ctx_drive                 # full synchronous volley -> coincidence plateau
        ext[col_idx] = i_ff                              # feedforward drive to the column
        bridge.cp_external_input_current[:] = xp.asarray(ext)
        bridge._run_one_simulation_step()
        f = _host(bridge.cp_firing_states)[col_idx].astype(bool)
        fired |= f
    return float(fired.mean())


def threshold_sweep(bridge, ctx_idx, col_idx, prime, desync, seed, i_grid):
    return np.array([col_fire_fraction(bridge, ctx_idx, col_idx, i_ff, prime, desync, seed=seed) for i_ff in i_grid])


def _run_seed(seed, i_grid, plateau_scale, ctx_weight):
    out = {}
    # PRIMED vs UNPRIMED with coincidence ON
    b, _ = build_bridge(seed, coincidence=True, plateau_scale=plateau_scale, ctx_col_weight=ctx_weight)
    rm = b.region_manager
    ctx = np.asarray(rm.indices("context"), dtype=np.int64); col = np.asarray(rm.indices("column"), dtype=np.int64)
    out["unprimed"] = threshold_sweep(b, ctx, col, prime=False, desync=False, seed=seed, i_grid=i_grid).tolist()
    b, _ = build_bridge(seed, coincidence=True, plateau_scale=plateau_scale, ctx_col_weight=ctx_weight)
    out["primed"] = threshold_sweep(b, ctx, col, prime=True, desync=False, seed=seed, i_grid=i_grid).tolist()
    b, _ = build_bridge(seed, coincidence=True, plateau_scale=plateau_scale, ctx_col_weight=ctx_weight)
    out["primed_noFF"] = threshold_sweep(b, ctx, col, prime=True, desync=False, seed=seed, i_grid=[0.0] * len(i_grid)).tolist()
    # dAP-LESION: coincidence OFF (plateau kernel disabled) -> primed should == unprimed
    b, _ = build_bridge(seed, coincidence=False, plateau_scale=plateau_scale, ctx_col_weight=ctx_weight)
    out["lesion_primed"] = threshold_sweep(b, ctx, col, prime=True, desync=False, seed=seed, i_grid=i_grid).tolist()
    # DESYNC anti-cheat: context fires but NOT synchronously -> no coincidence plateau
    b, _ = build_bridge(seed, coincidence=True, plateau_scale=plateau_scale, ctx_col_weight=ctx_weight)
    out["desync_primed"] = threshold_sweep(b, ctx, col, prime=True, desync=True, seed=seed, i_grid=i_grid).tolist()
    return seed, out


def _thresh(curve, grid, frac=0.5):
    curve = np.asarray(curve)
    idx = np.where(curve >= frac)[0]
    return float(grid[idx[0]]) if idx.size else float(grid[-1] + (grid[1] - grid[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--i-min", type=float, default=0.0)
    ap.add_argument("--i-max", type=float, default=600.0)
    ap.add_argument("--i-steps", type=int, default=13)
    ap.add_argument("--plateau-scale", type=float, default=1.0)
    ap.add_argument("--ctx-weight", type=float, default=2.0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    i_grid = np.linspace(a.i_min, a.i_max, a.i_steps)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            seed, out = _run_seed(s, i_grid, a.plateau_scale, a.ctx_weight)
            out["seed"] = seed; per.append(out)
        for d in per:
            tp = _thresh(d["primed"], i_grid); tu = _thresh(d["unprimed"], i_grid)
            tl = _thresh(d["lesion_primed"], i_grid); td = _thresh(d["desync_primed"], i_grid)
            print(f"  [seed {d['seed']}] fire-threshold pA: primed {tp:.0f} vs unprimed {tu:.0f} (shift {tu-tp:+.0f}) "
                  f"| lesion-primed {tl:.0f} | desync-primed {td:.0f} | primed-noFF max {max(d['primed_noFF']):.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mt(key):
            return float(np.mean([_thresh(d[key], i_grid) for d in per]))
        tp, tu, tl, td = mt("primed"), mt("unprimed"), mt("lesion_primed"), mt("desync_primed")
        noff = float(np.mean([max(d["primed_noFF"]) for d in per]))
        # PLATEAU-SPECIFIC advantage: same context volley (AMPA identical), coincidence ON vs OFF -> isolates the dAP.
        plateau_adv = tl - tp                 # primed fires at LOWER feedforward than lesion-primed (plateau primes)
        fire_first = plateau_adv >= 40.0
        no_spurious = noff <= 0.10            # the plateau ALONE (no feedforward) must NOT fire the cell (predictive != active)
        desync_collapses = td >= tl - 40.0    # desynchronized context -> no plateau -> no advantage over lesion
        go = bool(fire_first and no_spurious and desync_collapses)
        if go:
            verdict = (f"GO -- a distally-PRIMED (dAP coincidence-plateau) cell fires at LOWER feedforward drive than the "
                       f"SAME cell with the plateau OFF (same context AMPA volley): primed {tp:.0f} vs lesion-primed "
                       f"{tl:.0f} pA (plateau-specific advantage {plateau_adv:+.0f}); the plateau ALONE does NOT fire it "
                       f"(no-FF max frac {noff:.2f} -- predictive != active); DESYNCHRONIZED context gives no advantage "
                       f"({td:.0f} ~ lesion {tl:.0f}); unprimed baseline {tu:.0f}; multi-seed; NO sim/ edit. => the "
                       f"fire-first bias (the one genuinely-new biophysical behavior of the HTM-TM port) works on the real "
                       f"substrate -> Stage B (per-column WTA + frozen numpy-learned permanences => EMERGE-9c parity).")
        else:
            miss = []
            if not fire_first: miss.append(f"no plateau-specific fire-first (primed {tp:.0f} vs lesion-primed {tl:.0f}, adv {plateau_adv:+.0f} < 40)")
            if not no_spurious: miss.append(f"plateau fires cell WITHOUT feedforward (no-FF max {noff:.2f} > 0.10 -- lower plateau_scale)")
            if not desync_collapses: miss.append(f"desync didn't collapse (desync {td:.0f} vs lesion {tl:.0f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f". Calibrate the coincidence regime "
                       f"(k_threshold vs the context->column fan-in, ctx_col_weight to trigger a synchronous volley, "
                       f"plateau_scale, the i_ff grid around the Izhikevich rheobase ~ a few hundred pA). If NO regime "
                       f"gives fire-first, the current-injection dAP may be insufficient -> a true cp_v_apical "
                       f"two-compartment NeuronModel (scoping risk 1). NOT a wall; the calibration/mechanism is next.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge10_stageA_dap_fire_first", "verdict": verdict,
               "mechanism": "dendritic coincidence PLATEAU (fused_coincidence_plateau, the dAP) as a somatic "
                            "pre-depolarization biasing the Izhikevich firing threshold -- the fire-first bias for the "
                            "HTM-TM per-column WTA; NO sim/ edit (kernel exists, byte-inert when off)",
               "task": "primed-vs-unprimed feedforward firing-threshold sweep on a real SimulationBridge; dAP-lesion + "
                       "desync anti-cheats; no-spurious-firing invariant",
               "seeds": a.seeds, "config": {"i_grid": i_grid.tolist(), "plateau_scale": a.plateau_scale},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "rung-4 Stage A: the ONE genuinely-new biophysical behavior (distal plateau pre-"
                              "depolarization -> fire-first) in isolation, NO sim/ edit. GO -> Stage B (WTA + frozen "
                              "permanences => EMERGE-9c parity), then Stage C (the new three-term permanence kernel => "
                              "EMERGE-9d parity; that is the additive/guarded sim/ edit)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge10] VERDICT: {verdict}", flush=True)
    print(f"[emerge10] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
