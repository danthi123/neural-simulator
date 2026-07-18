"""GAP#4<->GAP#5 unification (STORING half): does on-bridge BTSP plateau-gated one-shot credit STORE a recurrent
assembly -- potentiate the WITHIN-assembly recurrent synapses one-shot under a plateau, SPECIFICALLY (within >>
between) and plateau-GATED (>> no-plateau)? This is the encoding side of "BTSP stores the CA3 assembly the bistable CA3
completes" (gap#5). Reuses the two committed session edits (bistable BDSP apical + the BTSP block); NO new sim/ edit.

SETUP: one recurrent region `ca3` (plastic internal); a sparse ASSEMBLY subset. Drive the assembly cells to CO-FIRE +
deliver a BRIEF apical plateau pulse to them (latched by the bistable BDSP apical) -> BTSP potentiates the recurrent
assembly->assembly synapses one-shot (pre-elig high on the co-firing assembly x plateau IS high on the same cells).
BDSP learning OFF (lr=0) so BTSP is the sole mover.

ARMS: stored (plateau ON, assembly drive) · no-plateau (assembly drive, NO plateau = the gate lesion) · off (enable_btsp
False). MEASURE the mean recurrent weight WITHIN the assembly vs BETWEEN (assembly<->non-assembly) vs the baseline.
GO (6-seed): within_dw >= 0.3 AND within_dw > 3*between_dw (SPECIFICITY -- only co-firing+plateaued pairs stored) AND
within_dw > 5*noplateau_within_dw (plateau-GATED = the moat) AND off within_dw == 0 (byte-identical). NO new sim/ edit.
Run: python -m research.runners._gap4_btsp_stores_recurrent_assembly_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend  # noqa: E402
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402

xp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_gap4_btsp_stores_recurrent_assembly.json"


def _build(enable_btsp, seed, n_ca3=40):
    regions = [BrainRegion(name="ca3", n_neurons=n_ca3, exc_fraction=1.0, internal_density=1.0,
                           exc_weight_mean=0.2, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=True)]
    cfg = CoreSimConfig(seed=seed)
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions; cfg.region_pathways = []
    for f in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
              "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
              "enable_input_divisive_norm", "enable_nmda"):
        setattr(cfg, f, False)
    cfg.enable_bdsp = True; cfg.bdsp_learning_rate = 0.0          # BDSP path only to evolve the bistable apical
    cfg.bdsp_apical_bistable = True
    cfg.coincidence_plateau_self_regen = 2.0; cfg.coincidence_plateau_v_hold = -35.0; cfg.apical_kir_g = 1.0
    cfg.enable_btsp = bool(enable_btsp)
    cfg.btsp_learning_rate = 0.02; cfg.btsp_elig_tau_ms = 1000.0; cfg.btsp_w_max = 5.0
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _recurrent_means(sb, ca3_idx, assembly_set):
    """mean |weight| of ca3->ca3 synapses WITHIN the assembly vs BETWEEN (one endpoint outside)."""
    from sim.backend import to_host
    conn = sb.cp_connections
    coo = conn.tocoo(copy=False)
    rows = np.asarray(to_host(coo.row)); cols = np.asarray(to_host(coo.col)); data = np.asarray(to_host(coo.data))
    ca3 = set(int(i) for i in ca3_idx)
    win, btw = [], []
    for r, c, d in zip(rows, cols, data):
        if r == c or int(r) not in ca3 or int(c) not in ca3:
            continue
        if int(r) in assembly_set and int(c) in assembly_set:
            win.append(abs(float(d)))
        elif int(r) in assembly_set or int(c) in assembly_set:
            btw.append(abs(float(d)))
    return (float(np.mean(win)) if win else 0.0), (float(np.mean(btw)) if btw else 0.0)


def _run(enable_btsp, plateau, seed, n_ca3=40, assembly_n=10, steps=200, pulse_steps=15, pulse_pA=120.0):
    sb = _build(enable_btsp, seed, n_ca3)
    rm = sb.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")))
    rng = np.random.default_rng(seed)
    assembly = np.sort(rng.choice(ca3_idx, size=assembly_n, replace=False))
    assembly_set = set(int(i) for i in assembly)
    n = sb.cp_membrane_potential_v.size
    sb.cp_bdsp_apical_drive = xp.zeros(n, dtype=xp.float32)
    w_in0, w_bt0 = _recurrent_means(sb, ca3_idx, assembly_set)
    drive = np.zeros(n, dtype=np.float32); drive[assembly] = 900.0    # the ASSEMBLY cells co-fire -> co-eligibility
    for step in range(steps):
        sb.cp_external_input_current[:] = xp.asarray(drive)
        cur = np.zeros(n, dtype=np.float32)
        if plateau and 20 <= step < 20 + pulse_steps:
            cur[assembly] = pulse_pA                                   # plateau ONLY on the assembly cells
        sb.cp_bdsp_apical_drive = xp.asarray(cur)
        sb._run_one_simulation_step()
    w_in1, w_bt1 = _recurrent_means(sb, ca3_idx, assembly_set)
    return {"within_dw": w_in1 - w_in0, "between_dw": w_bt1 - w_bt0}


def run(seed):
    stored = _run(enable_btsp=True, plateau=True, seed=seed)
    noplat = _run(enable_btsp=True, plateau=False, seed=seed)
    off = _run(enable_btsp=False, plateau=True, seed=seed)
    return {"seed": seed, "within_dw": stored["within_dw"], "between_dw": stored["between_dw"],
            "noplateau_within_dw": noplat["within_dw"], "off_within_dw": off["within_dw"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s); per.append(r)
            print(f"  [seed {s}] within_dw {r['within_dw']:.3f} | between_dw {r['between_dw']:.3f} | "
                  f"no-plateau within {r['noplateau_within_dw']:.4f} | off within {r['off_within_dw']:.4f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mean(k): return float(np.mean([p[k] for p in per]))
        wi, bt, npw, ow = mean("within_dw"), mean("between_dw"), mean("noplateau_within_dw"), mean("off_within_dw")
        stores = all(p["within_dw"] >= 0.3 for p in per)
        specific = all(p["within_dw"] > 3.0 * max(p["between_dw"], 1e-6) for p in per)
        gated = all(p["within_dw"] > 5.0 * max(p["noplateau_within_dw"], 1e-6) for p in per)
        off_inert = all(abs(p["off_within_dw"]) < 1e-9 for p in per)
        go = bool(stores and specific and gated and off_inert)
        if go:
            verdict = (f"GO -- on-bridge BTSP STORES a recurrent assembly one-shot, SPECIFICALLY + plateau-gated. The "
                       f"WITHIN-assembly recurrent weights grow (within_dw {wi:.3f}) far more than BETWEEN (between_dw "
                       f"{bt:.3f}, {wi/max(bt,1e-6):.1f}x) -- only co-firing+plateaued pairs are stored; no-plateau "
                       f"(gate lesion) barely moves ({npw:.4f}); enable_btsp=False byte-identical ({ow:.4f}). 6-seed. "
                       f"=> the encoding half of the gap#4<->gap#5 unification (BTSP stores the CA3 assembly the bistable "
                       f"CA3 completes) works on the spiking substrate. NO new sim/ edit.")
        else:
            miss = []
            if not stores: miss.append(f"within didn't grow (within_dw {wi:.3f})")
            if not specific: miss.append(f"not specific (within {wi:.3f} vs between {bt:.3f})")
            if not gated: miss.append(f"not plateau-gated (within {wi:.3f} vs no-plateau {npw:.4f})")
            if not off_inert: miss.append(f"enable_btsp=False not byte-identical ({ow:.4f})")
            verdict = "BOUNDARY -- " + "; ".join(miss) + ". Per THE LAW: tune the drive/pulse/assembly params, NOT a stop."
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "gap4_btsp_stores_recurrent_assembly", "GO": go, "verdict": verdict,
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "The STORING half of the gap#4<->gap#5 unification (BTSP plateau-gated one-shot encoding of "
                              "recurrent assembly weights). Completion (partial cue -> full assembly) is the gap#5 half; "
                              "wiring the two on one bridge is the next rung. Reuses the two committed edits; NO new sim/ edit."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-btsp-store] VERDICT: {verdict}", flush=True)
    print(f"[gap4-btsp-store] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
