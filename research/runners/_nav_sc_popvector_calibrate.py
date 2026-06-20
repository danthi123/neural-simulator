"""#6 SC orienting read-out — divnorm CALIBRATION micro-sweep (the step the build abandoned).

The population-VECTOR geometry (install_spiking_sc_wiring popvector=True) is BUILT + verified
correct, but inert unless the bump-mass divisive normalizer (cortex_X input_divisive_norm) is
CALIBRATED to the nav SC drive scale. The build doc's grid-8 smoke ran at the DEFAULT divisive
op-point (sigma=1, gain=1) and the pop-vector arm was STILL stuck-N -- and the doc's own
mechanistic read says default gain=1 (tuned for the conversational cortex's O(1) drives)
OVER-ATTENUATES the SC drive (the nav SC drive is O(tens-hundreds pA), so out_i ~ drive/(gain*mean)
~ O(few) pA, crushing the SC contribution so the cascade N-bias + OU win regardless of the
now-correct cosine geometry). The responsive band is gain << 1.

This is the CALIBRATION SCREEN ONLY (grid-8/480, standalone nav SC = fast). It runs ONLY the
sc_popvector arm across a (divnorm_sigma, divnorm_gain) grid and reports, per cell:
  - per-phase dominant cardinal + action fractions
  - tracks_goal (the dominant cardinal SHIFTS toward the goal across phases, vs stuck-N)
  - phase0/post-change finalQ

PASS-to-proceed = at least one (sigma, gain) makes the grid-8 phase-1 (far-WEST goal) action
distribution shift OFF stuck-N toward W (the host's signature). grid-8 is a WEAK read (only ~2
goal phases complete in 480; the cascade N-bias + OU dominate at small scale) so the VERDICT is
NOT here -- the verdict is the faithful grid-32 confirm (run separately, with --with-conv + host
+ ramp + scramble). The grid-8 false-GO is the explicit cautionary tale; this screen finds the
calibration band, nothing more.

Reuses research.runners._nav_sc_popvector_readout_derisk.run_arm (the built probe). NO sim/ edit.
GPU (SIM_BACKEND=cupy). Default standalone (no --with-conv) for speed -- the SC read-out dynamics
are array-disjoint from the conversational moat, which the calibration does not need.
"""
import os

# MUST precede any CuPy import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SC_RET_SC", "160")
os.environ.setdefault("SC_REC", "12")
os.environ.setdefault("SC_RET_DRIVE", "3500")
os.environ.setdefault("SC_ROS_US", "40")

import argparse
import json

import numpy as np


def _track(s):
    """Mirror of the probe's _track: does the dominant cardinal SHIFT toward the goal?"""
    if not s:
        return None
    doms = s.get("per_phase_dominant_cardinal", [])
    n_distinct = len(set(d for d in doms if d is not None))
    has_W = any(d == "W" for d in doms)
    has_E = any(d == "E" for d in doms)
    return {"dominant_per_phase": doms, "n_distinct_dominant": n_distinct,
            "has_W_dominant_phase": has_W, "has_E_dominant_phase": has_E,
            "tracks_goal": (n_distinct >= 2 and (has_W or has_E))}


def main():
    ap = argparse.ArgumentParser(description="#6 SC pop-vector divnorm calibration micro-sweep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=480)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--sc-cortex-w", type=float, default=18.0,
                    help="MATCHED sc_map->cortex drive (default 18 = the deployed/NEGATIVE level).")
    ap.add_argument("--sigmas", type=str, default="1,5,20",
                    help="comma list of divnorm_sigma values.")
    ap.add_argument("--gains", type=str, default="0.0,0.02,0.05,0.1,0.2",
                    help="comma list of divnorm_gain values (the responsive band is gain << 1).")
    ap.add_argument("--with-conv", action="store_true",
                    help="merged bridge (the NEGATIVE config). Off = standalone nav SC (fast screen).")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/nav_gate_2a/scpv_calibrate.json")
    args = ap.parse_args()

    from research.runners._nav_sc_popvector_readout_derisk import run_arm

    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    gains = [float(x) for x in args.gains.split(",") if x.strip()]
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    cells = []
    for sg in sigmas:
        for gn in gains:
            print(f"\n[cal] ===== sigma={sg} gain={gn} =====", flush=True)
            s = run_arm("sc_popvector", args.seed, args.n_steps, args.grid_size,
                        args.warmup_steps, out_dir, args.with_conv, args.sc_cortex_w, sg, gn)
            tr = _track(s)
            cell = {
                "divnorm_sigma": sg, "divnorm_gain": gn,
                "per_phase_dominant_cardinal": s.get("per_phase_dominant_cardinal"),
                "per_phase_action_frac": s.get("per_phase_action_frac"),
                "per_phase_goal": s.get("per_phase_goal"),
                "phase0_finalQ": s.get("phase0_finalQ"),
                "post_change_finalQ": s.get("post_change_finalQ"),
                "post_change_finalQ_sum": s.get("post_change_finalQ_sum"),
                "late_motor_sustain_frac": s.get("late_motor_sustain_frac"),
                "tracks_goal": (tr or {}).get("tracks_goal"),
                "n_distinct_dominant": (tr or {}).get("n_distinct_dominant"),
                "has_W_dominant_phase": (tr or {}).get("has_W_dominant_phase"),
                "has_E_dominant_phase": (tr or {}).get("has_E_dominant_phase"),
            }
            cells.append(cell)
            print(f"[cal] sigma={sg} gain={gn}: dom={cell['per_phase_dominant_cardinal']} "
                  f"tracks_goal={cell['tracks_goal']} "
                  f"phase0Q={cell['phase0_finalQ']:.3f} "
                  f"postQ_sum={cell['post_change_finalQ_sum']:.3f}", flush=True)

    # rank: cells that track the goal first, then by lowest post_change_finalQ_sum.
    def _key(c):
        track = 0 if c.get("tracks_goal") else 1
        pq = c.get("post_change_finalQ_sum")
        pq = pq if (pq is not None and pq == pq) else 1e9
        return (track, pq)

    ranked = sorted(cells, key=_key)
    best = ranked[0] if ranked else None

    out = {
        "seed": args.seed, "grid_size": args.grid_size, "n_steps": args.n_steps,
        "warmup_steps": args.warmup_steps, "sc_cortex_w": args.sc_cortex_w,
        "with_conv": args.with_conv,
        "cells": cells,
        "best_cell": best,
        "any_tracks_goal": any(c.get("tracks_goal") for c in cells),
        "NOTE": ("CALIBRATION SCREEN (grid-8, standalone). PASS-to-proceed = any_tracks_goal True "
                 "(a (sigma,gain) shifts the dominant cardinal off stuck-N toward the goal). The "
                 "VERDICT is the faithful grid-32 confirm, NOT this screen (grid-8 false-GO is the "
                 "cautionary tale)."),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n[cal] ===== CALIBRATION SWEEP SUMMARY =====", flush=True)
    print(f"  {'sigma':>6} {'gain':>6} | {'tracks':>6} | {'dom_per_phase':<28} | "
          f"{'phase0Q':>8} {'postQ_sum':>9}", flush=True)
    for c in cells:
        print(f"  {c['divnorm_sigma']:>6} {c['divnorm_gain']:>6} | "
              f"{str(c['tracks_goal']):>6} | {str(c['per_phase_dominant_cardinal']):<28} | "
              f"{(c['phase0_finalQ'] or float('nan')):>8.3f} "
              f"{(c['post_change_finalQ_sum'] or float('nan')):>9.3f}", flush=True)
    print(f"\n[cal] any_tracks_goal={out['any_tracks_goal']}", flush=True)
    print(f"[cal] best_cell: {best}", flush=True)
    print(f"[cal] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
