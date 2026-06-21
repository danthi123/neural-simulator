"""Cascade-accumulator FIX A (sel_X divisive-norm) sigma/gain SWEEP (2026-06-20).

Runs the FIX-1 popvector arm of the #6 rig (`_nav_sc_popvector_readout_derisk.run_arm`) across a grid of
(sigma, gain) divisive-norm op-points and tabulates, per setting:
  - the per-stage N-S surplus (thal / sel / commit / motor) -- the DECISIVE surplus-shrink check
  - the per-phase dominant cardinal + whether it TRACKS the moving goal
  - the post-change finalQ sum (the re-orient metric)

The success op-point: the sel/commit N-S surplus SHRINKS toward 0 AND the per-phase dom still TRACKS the
goal (NOT over-flattened to all-random, NOT re-biased to a fixed cardinal) AND post-change finalQ drops
toward HOST. grid-32 IS the verdict.

NO sim/ edit here (this only DRIVES the rig, which sets the existing sc_sel_divnorm kwarg -> the second
divisive pool added 2026-06-20). GPU (SIM_BACKEND=cupy).
"""
import argparse
import json
import os

import numpy as np

from research.runners._nav_sc_popvector_readout_derisk import run_arm


def _track(doms):
    n_distinct = len(set(d for d in doms if d is not None))
    has_W = any(d == "W" for d in doms)
    has_E = any(d == "E" for d in doms)
    return {"n_distinct": n_distinct, "has_W": has_W, "has_E": has_E,
            "tracks": (n_distinct >= 2 and (has_W or has_E))}


def main():
    ap = argparse.ArgumentParser(description="FIX A sel_X divisive-norm sigma/gain sweep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--warmup-steps", type=int, default=600)
    ap.add_argument("--sc-cortex-w", type=float, default=18.0)
    ap.add_argument("--divnorm-sigma", type=float, default=1.0, help="cortex_X pool-1 (popvector) sigma")
    ap.add_argument("--divnorm-gain", type=float, default=1.0, help="cortex_X pool-1 (popvector) gain")
    ap.add_argument("--sigmas", type=str, default="1,10,50,200",
                    help="comma list of FIX A sel_X divnorm sigmas to sweep")
    ap.add_argument("--gains", type=str, default="1.0",
                    help="comma list of FIX A sel_X divnorm gains to sweep")
    ap.add_argument("--with-conv", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/nav_gate_2a/fixA_sweep.json")
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    gains = [float(g) for g in args.gains.split(",") if g.strip()]
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for gain in gains:
        for sigma in sigmas:
            s = run_arm("sc_popvector", args.seed, args.n_steps, args.grid_size, args.warmup_steps,
                        out_dir, args.with_conv, args.sc_cortex_w,
                        args.divnorm_sigma, args.divnorm_gain,
                        fix1=True, fixA=True,
                        sel_divnorm_sigma=sigma, sel_divnorm_gain=gain)
            ss = s.get("stage_surplus") or {}
            doms = s.get("per_phase_dominant_cardinal", [])
            tr = _track(doms)
            row = {
                "sigma": sigma, "gain": gain,
                "thal_NS": (ss.get("thal_counts") or {}).get("NS_surplus"),
                "thal_NS_pct": (ss.get("thal_counts") or {}).get("NS_pct"),
                "sel_NS": (ss.get("sel_counts") or {}).get("NS_surplus"),
                "sel_NS_pct": (ss.get("sel_counts") or {}).get("NS_pct"),
                "commit_NS": (ss.get("commit_counts") or {}).get("NS_surplus"),
                "commit_NS_pct": (ss.get("commit_counts") or {}).get("NS_pct"),
                "motor_NS": (ss.get("motor_counts") or {}).get("NS_surplus"),
                "dom_per_phase": doms,
                "n_distinct_dom": tr["n_distinct"],
                "tracks_goal": tr["tracks"],
                "post_change_finalQ_sum": s.get("post_change_finalQ_sum"),
                "phase0_finalQ": s.get("phase0_finalQ"),
                "tie_break_fraction": s.get("tie_break_fraction"),
                "summary_json": s.get("episode_json"),
            }
            rows.append(row)
            print(f"[fixA-sweep] sigma={sigma} gain={gain} | sel_NS={row['sel_NS']} "
                  f"({row['sel_NS_pct']}%) commit_NS={row['commit_NS']} ({row['commit_NS_pct']}%) "
                  f"dom={doms} tracks={tr['tracks']} postΣ={row['post_change_finalQ_sum']}", flush=True)

    out = {"seed": args.seed, "grid_size": args.grid_size, "n_steps": args.n_steps,
           "warmup_steps": args.warmup_steps, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[fixA-sweep] wrote {args.out}", flush=True)
    print("[fixA-sweep] ===== SWEEP TABLE (sel_NS surplus shrink + dom tracking) =====", flush=True)
    print(f"  {'sigma':>7} {'gain':>5} {'sel_NS':>9} {'sel%':>7} {'commit_NS':>10} {'dom_per_phase':>22} "
          f"{'tracks':>7} {'postΣ':>8}", flush=True)
    for r in rows:
        print(f"  {r['sigma']:>7.1f} {r['gain']:>5.1f} {str(r['sel_NS']):>9} {str(r['sel_NS_pct']):>7} "
              f"{str(r['commit_NS']):>10} {str(r['dom_per_phase']):>22} {str(r['tracks_goal']):>7} "
              f"{str(r['post_change_finalQ_sum']):>8}", flush=True)


if __name__ == "__main__":
    main()
