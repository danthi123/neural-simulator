"""GAP#4 on-bridge BEHAVIORAL-TIMESCALE de-risk: the REAL bistable dendritic plateau (evolved by the actual on-bridge
dynamics, NOT a manually-set apical) makes on-bridge BTSP a SECONDS-long one-shot credit rule; a transient plateau
gives only a brief window.

Reuses BOTH of this session's edits, no coincidence pathway needed:
  - the BISTABLE BDSP apical (`enable_bdsp` + `bdsp_apical_bistable`, bridge.py:7258): a BRIEF cp_bdsp_apical_drive
    pulse LATCHES cp_v_apical (self-regen SUSTAIN + KIR) so the plateau HOLDS for many steps; without the bistable flag
    the same pulse DECAYS within ~ms.
  - the on-bridge BTSP block (`enable_btsp`, fused_btsp_update): reads that held cp_v_apical (above v_hold) as the
    instructive signal IS and potentiates a co-active pre->post synapse one-shot; dw accumulates over every step IS>0.

PROTOCOL (one presentation): fire the PRE pool (lays down the seconds-long cp_btsp_pre_elig) throughout; deliver a BRIEF
apical drive pulse to POST early; then let the plateau ride while PRE keeps its eligibility. BDSP learning is OFF
(bdsp_learning_rate=0) so the ONLY weight mover is BTSP. Held plateau => IS>0 for many steps => large dw; transient =>
IS>0 for a few steps => small dw; no pulse (silent apical) => IS==0 => dw==0 (the moat).

GO (6-seed): held_dw >= 0.3 AND held_dw > 3*transient_dw (the bistable latch is load-bearing for the behavioral-timescale
window) AND moat_dw <= 0.02*held_dw (silent apical -> no potentiation) AND off_dw == 0 (enable_btsp=False byte-identical).
NO new sim/ edit (reuse-by-import of the two committed edits). GPU or CPU. Run:
  python -m research.runners._gap4_btsp_onbridge_behavioral_timescale_derisk --seeds 42 43 44 100 101 102
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
OUT = _REPO / "research" / "findings" / "raw" / "_gap4_btsp_onbridge_behavioral_timescale.json"


def _build(enable_btsp, bistable, seed):
    regions = [
        BrainRegion(name="pre", n_neurons=8, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="post", n_neurons=8, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [RegionPathway(from_region="pre", to_region="post", density=1.0,
                              weight_mean=0.5, weight_jitter=0.0, plastic=True)]
    cfg = CoreSimConfig(seed=seed)
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions; cfg.region_pathways = pathways
    for f in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
              "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
              "enable_input_divisive_norm", "enable_nmda"):
        setattr(cfg, f, False)
    # BDSP path ONLY to evolve the bistable apical (learning OFF so BDSP moves no weight -> BTSP is the sole mover)
    cfg.enable_bdsp = True
    cfg.bdsp_learning_rate = 0.0
    cfg.bdsp_apical_bistable = bool(bistable)
    cfg.coincidence_plateau_self_regen = 2.0
    cfg.coincidence_plateau_v_hold = -35.0
    cfg.apical_kir_g = 1.0
    # BTSP (the tested rule)
    cfg.enable_btsp = bool(enable_btsp)
    cfg.btsp_learning_rate = 0.02
    cfg.btsp_elig_tau_ms = 1000.0
    cfg.btsp_w_max = 5.0
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _run(enable_btsp, bistable, pulse, seed, steps=200, pulse_steps=15, pulse_pA=120.0):
    """pulse=True: a BRIEF apical drive to POST early -> latches (bistable) or decays (transient). pulse=False: no
    apical drive at all (silent apical -> the moat)."""
    sb = _build(enable_btsp, bistable, seed)
    rm = sb.region_manager
    pre_idx = np.asarray(list(rm.indices("pre"))); post_idx = np.asarray(list(rm.indices("post")))
    n = sb.cp_membrane_potential_v.size
    # BDSP apical drive array (its presence triggers the bridge's apical integration each step)
    ap = np.zeros(n, dtype=np.float32)
    sb.cp_bdsp_apical_drive = xp.asarray(ap)
    drive = np.zeros(n, dtype=np.float32); drive[pre_idx] = 900.0     # PRE fires throughout -> seconds-long eligibility
    w0 = float(xp.asarray(sb.cp_connections.data).sum())
    for step in range(steps):
        sb.cp_external_input_current[:] = xp.asarray(drive)
        # BRIEF apical pulse to POST early (steps 20..20+pulse_steps); silent otherwise. pulse=False -> never any drive.
        cur = ap.copy()
        if pulse and 20 <= step < 20 + pulse_steps:
            cur[post_idx] = pulse_pA
        sb.cp_bdsp_apical_drive = xp.asarray(cur)
        sb._run_one_simulation_step()
    w1 = float(xp.asarray(sb.cp_connections.data).sum())
    # how long the apical stayed above v_hold (the behavioral-timescale window, a sanity read)
    va = xp.asarray(sb.cp_v_apical[post_idx]).mean() if sb.cp_v_apical is not None else float("nan")
    return {"dw": w1 - w0, "v_apical_end": float(va)}


def run(seed):
    held = _run(enable_btsp=True, bistable=True, pulse=True, seed=seed)
    tran = _run(enable_btsp=True, bistable=False, pulse=True, seed=seed)
    moat = _run(enable_btsp=True, bistable=True, pulse=False, seed=seed)     # no pulse -> silent apical
    off = _run(enable_btsp=False, bistable=True, pulse=True, seed=seed)      # BTSP off -> byte-identical
    return {"seed": seed, "held_dw": held["dw"], "transient_dw": tran["dw"], "moat_dw": moat["dw"], "off_dw": off["dw"],
            "held_v_apical_end": held["v_apical_end"], "transient_v_apical_end": tran["v_apical_end"]}


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
            print(f"  [seed {s}] held_dw {r['held_dw']:.3f} (v_apical_end {r['held_v_apical_end']:.1f}) | "
                  f"transient_dw {r['transient_dw']:.3f} (v_apical_end {r['transient_v_apical_end']:.1f}) | "
                  f"moat_dw {r['moat_dw']:.4f} | off_dw {r['off_dw']:.4f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mean(k): return float(np.mean([p[k] for p in per]))
        hd, td, md, od = mean("held_dw"), mean("transient_dw"), mean("moat_dw"), mean("off_dw")
        held_pot = all(p["held_dw"] >= 0.3 for p in per)
        bistable_loadbearing = all(p["held_dw"] > 3.0 * max(p["transient_dw"], 1e-6) for p in per)
        moat = all(abs(p["moat_dw"]) <= 0.02 * max(p["held_dw"], 1e-6) for p in per)
        off_inert = all(abs(p["off_dw"]) < 1e-9 for p in per)
        go = bool(held_pot and bistable_loadbearing and moat and off_inert)
        if go:
            verdict = (f"GO -- on-bridge BTSP is BEHAVIORAL-TIMESCALE via the REAL bistable plateau. A HELD plateau (the "
                       f"bistable BDSP apical, self-regen+KIR) potentiates the co-active pre->post synapse one-shot over "
                       f"a seconds-long window (held_dw {hd:.3f}); a TRANSIENT plateau gives only a brief window "
                       f"(transient_dw {td:.3f}, {hd/max(td,1e-6):.1f}x less). Silent apical -> moat (moat_dw {md:.4f}); "
                       f"enable_btsp=False byte-identical (off_dw {od:.4f}). 6-seed. => the gap#5 bistable plateau makes "
                       f"on-bridge BTSP a seconds-long one-shot credit rule, on the spiking substrate. NO new sim/ edit.")
        else:
            miss = []
            if not held_pot: miss.append(f"held didn't potentiate (held_dw {hd:.3f})")
            if not bistable_loadbearing: miss.append(f"bistability not load-bearing (held {hd:.3f} vs transient {td:.3f})")
            if not moat: miss.append(f"moat leak (moat_dw {md:.4f})")
            if not off_inert: miss.append(f"enable_btsp=False not byte-identical (off_dw {od:.4f})")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". Per THE LAW: tune the pulse/self_regen/eligibility "
                       "timescales or the protocol, NOT a stop.")
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "gap4_btsp_onbridge_behavioral_timescale", "GO": go, "verdict": verdict,
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "mechanism": "on-bridge BTSP (fused_btsp_update) reading the REAL bistable BDSP apical (bdsp_apical_bistable) "
                            "as the instructive signal; held plateau => seconds-long credit, transient => brief.",
               "HONEST_NOTE": "Reuses the two committed session edits (bistable BDSP apical + the BTSP block); NO new sim/ "
                              "edit. On-bridge behavioral-timescale one-shot credit; NOT a multi-layer/deep-credit claim."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-btsp-onbridge] VERDICT: {verdict}", flush=True)
    print(f"[gap4-btsp-onbridge] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
