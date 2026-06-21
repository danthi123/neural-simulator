"""#6 FIX A — ARM 3: multi-seed verdict (grid-32, full 1800 steps).

Per seed, runs FOUR arms with the correct per-arm flags and writes one summary JSON:
  - FIX1+A : sc_popvector + --fix1 + --fixA (the build: divisive norm at the sel_X input).
  - FIX1   : sc_popvector + --fix1 (NO sel divnorm -- the no-divnorm baseline).
  - HOST   : the host-heuristic argmax oracle (the orienting ceiling FIX-A must reach to retire).
  - SCRAM  : sc_popvector_scr + --fix1 + --fixA (the retinotopy scramble LESION; MUST collapse).

The DECISIVE checks (per the de-risk):
  (1) surplus-shrink : FIX1+A sel_counts N-S surplus materially smaller than FIX1's.
  (2) re-orient/ceiling : does FIX1+A's spiking orienting read-out reach <= the HOST ceiling
        (post_change_finalQ_sum approaches host) -> #6 accumulator residual CLOSED (host heuristic retires);
        or not -> NEGATIVE (-> SURPASS: FIX-B opponent-pair).
  (3) nav not regressed : FIX1+A grid-32 score holds (vs FIX1).
  (4) SCRAM collapses : the read-out is load-bearing (scramble worse / does not track).
  (5) moat untouched : the composer RF synapses (cp_rf_w_*) are never built in this standalone rig
        (no --with-conv) -> array-disjoint from the nav cascade by construction.

Writes per-seed summaries incrementally + a roll-up. Commit after EACH seed (anti-rest).

  SIM_BACKEND=cupy python -m research.runners._nav_sc_fixA_arm3_multiseed --seed 42 \
      --grid-size 32 --n-steps 1800 --warmup-steps 600 \
      --sel-divnorm-sigma <S> --sel-divnorm-gain <G> \
      --out research/findings/raw/nav_gate_2a/scpv_FIXA_arm3_seed42.json
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SC_RET_SC", "160")
os.environ.setdefault("SC_REC", "12")
os.environ.setdefault("SC_RET_DRIVE", "3500")
os.environ.setdefault("SC_ROS_US", "40")

import argparse
import json

import numpy as np


def _sel_ns_pct(summary):
    ss = (summary or {}).get("stage_surplus") or {}
    sel = ss.get("sel_counts")
    return (sel["NS_pct"] if sel else None)


def _sel_ns_abs(summary):
    ss = (summary or {}).get("stage_surplus") or {}
    sel = ss.get("sel_counts")
    return (sel["NS_surplus"] if sel else None)


def main():
    ap = argparse.ArgumentParser(description="#6 FIX A ARM 3 multi-seed verdict (single seed per call)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--warmup-steps", type=int, default=600)
    ap.add_argument("--sc-cortex-w", type=float, default=18.0)
    ap.add_argument("--divnorm-sigma", type=float, default=5.0,   # cortex_X pool-1 popvector divnorm
                    help="cortex_X (pool-1) popvector divisive sigma (fixed).")
    ap.add_argument("--divnorm-gain", type=float, default=0.02,
                    help="cortex_X (pool-1) popvector divisive gain (fixed).")
    ap.add_argument("--sel-divnorm-sigma", type=float, required=True,
                    help="FIX-A sel_X divisive sigma (the ARM-2 sweet spot).")
    ap.add_argument("--sel-divnorm-gain", type=float, required=True,
                    help="FIX-A sel_X divisive gain (the ARM-2 sweet spot).")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    from research.runners._nav_sc_popvector_readout_derisk import run_arm

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    point_dir = os.path.join(out_dir, f"arm3_seed{args.seed}")
    os.makedirs(point_dir, exist_ok=True)

    common = dict(seed=args.seed, n_steps=args.n_steps, grid_size=args.grid_size,
                  warmup_steps=args.warmup_steps, out_dir=point_dir, with_conv=False,
                  sc_cortex_w=args.sc_cortex_w, divnorm_sigma=args.divnorm_sigma,
                  divnorm_gain=args.divnorm_gain)

    results = {}

    # --- HOST (orienting ceiling oracle) ---
    print(f"\n[arm3 seed={args.seed}] ===== HOST (orienting ceiling) =====", flush=True)
    results["HOST"] = run_arm("host", **common)
    _dump(args, results, point_dir)

    # --- FIX1 (no sel divnorm baseline) ---
    print(f"\n[arm3 seed={args.seed}] ===== FIX1 (no sel divnorm) =====", flush=True)
    results["FIX1"] = run_arm("sc_popvector", fix1=True, fixA=False, **common)
    _dump(args, results, point_dir)

    # --- FIX1+A (the build) ---
    print(f"\n[arm3 seed={args.seed}] ===== FIX1+A (sel divnorm sigma={args.sel_divnorm_sigma} "
          f"gain={args.sel_divnorm_gain}) =====", flush=True)
    results["FIX1A"] = run_arm("sc_popvector", fix1=True, fixA=True,
                               sel_divnorm_sigma=args.sel_divnorm_sigma,
                               sel_divnorm_gain=args.sel_divnorm_gain, **common)
    _dump(args, results, point_dir)

    # --- SCRAM (retinotopy lesion of FIX1+A) ---
    print(f"\n[arm3 seed={args.seed}] ===== SCRAM (scramble lesion of FIX1+A) =====", flush=True)
    results["SCRAM"] = run_arm("sc_popvector_scr", fix1=True, fixA=True,
                               sel_divnorm_sigma=args.sel_divnorm_sigma,
                               sel_divnorm_gain=args.sel_divnorm_gain, **common)
    _dump(args, results, point_dir, final=True)


def _dump(args, results, point_dir, final=False):
    host = results.get("HOST")
    fix1 = results.get("FIX1")
    fix1a = results.get("FIX1A")
    scram = results.get("SCRAM")

    def _ps(s):  # post_change_finalQ_sum
        return (s.get("post_change_finalQ_sum") if s else None)

    def _trk(s):
        if not s:
            return None
        doms = s.get("per_phase_dominant_cardinal", [])
        nd = len(set(d for d in doms if d is not None))
        return {"dominant_per_phase": doms, "n_distinct": nd,
                "has_W": any(d == "W" for d in doms), "has_E": any(d == "E" for d in doms),
                "tracks_goal": (nd >= 2 and (any(d == "W" for d in doms) or any(d == "E" for d in doms)))}

    verdict = {
        "seed": args.seed, "grid_size": args.grid_size, "n_steps": args.n_steps,
        "sel_divnorm_sigma": args.sel_divnorm_sigma, "sel_divnorm_gain": args.sel_divnorm_gain,
        "HOST_post_change_finalQ_sum": _ps(host),
        "FIX1_post_change_finalQ_sum": _ps(fix1),
        "FIX1A_post_change_finalQ_sum": _ps(fix1a),
        "SCRAM_post_change_finalQ_sum": _ps(scram),
        "HOST_gate_score": (host.get("gate_score") if host else None),
        "FIX1_gate_score": (fix1.get("gate_score") if fix1 else None),
        "FIX1A_gate_score": (fix1a.get("gate_score") if fix1a else None),
        "SCRAM_gate_score": (scram.get("gate_score") if scram else None),
        # (1) surplus-shrink
        "FIX1_sel_NS_pct": _sel_ns_pct(fix1),
        "FIX1A_sel_NS_pct": _sel_ns_pct(fix1a),
        "FIX1_sel_NS_abs": _sel_ns_abs(fix1),
        "FIX1A_sel_NS_abs": _sel_ns_abs(fix1a),
        # tracking
        "HOST_tracking": _trk(host),
        "FIX1_tracking": _trk(fix1),
        "FIX1A_tracking": _trk(fix1a),
        "SCRAM_tracking": _trk(scram),
        "FIX1A_late_sustain": (fix1a.get("late_motor_sustain_frac") if fix1a else None),
        "FIX1A_stage_surplus": (fix1a.get("stage_surplus") if fix1a else None),
        "FIX1_stage_surplus": (fix1.get("stage_surplus") if fix1 else None),
        "moat_untouched": "by construction (no --with-conv: cp_rf_w_* never allocated; nav cascade is "
                          "cp_connections/cp_membrane_potential_v/cp_firing_states, array-disjoint)",
    }
    # (2) ceiling ratio: host / fix1a on post-change re-orient (>=~1 => fix1a reaches host).
    h, fa = _ps(host), _ps(fix1a)
    verdict["host_over_FIX1A_post_ratio"] = (float(h / fa) if (h and fa and fa > 0) else None)
    # SCRAM must be WORSE than FIX1A (collapse) => FIX1A/SCRAM < 1.
    sc = _ps(scram)
    verdict["FIX1A_over_SCRAM_post_ratio"] = (float(fa / sc) if (fa and sc and sc > 0) else None)

    out = {"verdict": verdict, "arms": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if final:
        print(f"\n[arm3 seed={args.seed}] ===== ARM-3 VERDICT (seed {args.seed}) =====", flush=True)
        for k, v in verdict.items():
            print(f"  {k}: {v}", flush=True)
        print(f"[arm3 seed={args.seed}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
