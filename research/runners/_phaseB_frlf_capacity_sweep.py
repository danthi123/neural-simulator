"""CYCLE 146 cheap localization — does the fixed-role + learned-filler BUNDLING (held-out 0.603 @ D_h=64)
lift toward the fixed-algebra ceiling (0.993) when given more bind-space capacity?

The 6-seed A/B (`_phaseB_fixed_role_learned_filler_bundling_derisk.py`) resolved the headline question GO (a
FIXED self-inverse role + LEARNED filler codes recovers bundled superposition where a learned LINEAR inverse
cannot), but the LEARNED-filler version landed at 0.603 -- ~0.39 BELOW the fully-fixed FHRR algebra's 0.993.
Per that doc's pre-registered BOUNDARY clause -- "the LEARNED fillers cost accuracy vs the fully-fixed algebra;
localize (more capacity / a multiplicative cleanup) before committing the on-bridge build" -- this sweep is the
cheap localization that GATES the deferred weeks-scale build:

  * if FR+LF bundling lifts 0.603 -> ~0.9 PARITY as D_h grows, the on-bridge build RE-OPENS as justified
    (route the learned fillers through the guarded `fused_coincidence_plateau` self-inverse primitive);
  * if it plateaus well below the matched ceiling, the fixed FHRR algebra stays the load-bearing bundler and the
    learned frontier is confirmed to be the GENERALIZATION axis (separate PPMI/cross-modal arc), not bundling.

REUSE-BY-IMPORT (zero A/B drift): we set the binder module's global `D_H` per sweep value and call its EXACT
`run_arm1` (the FR+LF main arm + lesion control) and `run_fixed_pm1` (the matched ceiling at the SAME D_h), so
every number is produced by the same validated eval path -- only the bind-space dimension changes. The gap
(ceiling - FR+LF) at each D_h is the localization signal. NO sim/ edit; CPU/numpy.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_frlf_capacity_sweep
      [--seeds 42,43,44,100,101,102] [--dims 64,128,256]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import research.runners._phaseB_fixed_role_learned_filler_bundling_derisk as frlf  # noqa: E402


def run_dim(codes, seed, D_h):
    """FR+LF main arm + lesion control + matched fixed-pm1 ceiling at an explicit bind-space dim D_h.

    Sets the binder module's global D_H so run_arm1 / run_fixed_pm1 build everything (role_pm1 projection,
    W_F/W_O, the +-1 ceiling projections) at this dimension -- the only thing that changes vs the A/B."""
    frlf.D_H = int(D_h)                                        # global read at call-time by run_arm1/run_fixed_pm1
    a1 = frlf.run_arm1(codes, seed)
    a1_lesion = frlf.run_arm1(codes, seed, lesion_sum=True)
    fx = frlf.run_fixed_pm1(codes, seed)
    return {
        "seed": seed, "D_h": int(D_h),
        "frlf_bheld": a1["bundle_held"], "frlf_single": a1["single_held"], "frlf_btrain": a1["bundle_train"],
        "frlf_lesion_bheld": a1_lesion["bundle_held"],
        "fixed_pm1_bheld": fx["bundled"],
        "gap_to_ceiling": fx["bundled"] - a1["bundle_held"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--dims", type=str, default="64,128,256")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw", "_phaseB_frlf_capacity_sweep.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = [int(s) for s in args.seeds.split(",")]
    dims = [int(d) for d in args.dims.split(",")]

    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)

    t0 = time.time()
    print(f"[FR+LF capacity localization] does bundling lift 0.603 -> ~0.9 parity as D_h grows?  "
          f"seeds={seeds} dims={dims}  (A/B point: D_h=64 -> 0.603 vs ceiling 0.993)", flush=True)
    by_dim = {}
    rows = []
    for D_h in dims:
        drows = []
        for seed in seeds:
            r = run_dim(codes, seed, D_h)
            drows.append(r); rows.append(r)
            print(f"  [D_h={D_h:>3} seed {seed}] FR+LF bundled {r['frlf_bheld']:.3f} "
                  f"(single {r['frlf_single']:.3f}, train {r['frlf_btrain']:.3f}) | "
                  f"lesion {r['frlf_lesion_bheld']:.3f} | ceiling {r['fixed_pm1_bheld']:.3f} | "
                  f"gap {r['gap_to_ceiling']:+.3f}", flush=True)
        frlf_m = float(np.mean([r["frlf_bheld"] for r in drows]))
        ceil_m = float(np.mean([r["fixed_pm1_bheld"] for r in drows]))
        single_m = float(np.mean([r["frlf_single"] for r in drows]))
        gap_m = float(np.mean([r["gap_to_ceiling"] for r in drows]))
        n_par = sum(int(r["frlf_bheld"] >= 0.85 * r["fixed_pm1_bheld"] and r["frlf_bheld"] >= 0.80)
                    for r in drows)
        by_dim[D_h] = {"frlf_bheld": frlf_m, "fixed_pm1_bheld": ceil_m, "frlf_single": single_m,
                       "gap_to_ceiling": gap_m, "n_near_parity": n_par,
                       "per_seed": [r["frlf_bheld"] for r in drows]}
        print(f"    -> D_h={D_h}: FR+LF {frlf_m:.3f} | ceiling {ceil_m:.3f} | gap {gap_m:+.3f} | "
              f"single {single_m:.3f} | near-parity {n_par}/{len(seeds)}", flush=True)

    # Localization verdict: did the gap to the matched ceiling SHRINK materially with capacity, toward parity?
    g0 = by_dim[dims[0]]["gap_to_ceiling"]
    gN = by_dim[dims[-1]]["gap_to_ceiling"]
    f0 = by_dim[dims[0]]["frlf_bheld"]
    fN = by_dim[dims[-1]]["frlf_bheld"]
    par_N = by_dim[dims[-1]]["n_near_parity"]
    bar = int(np.ceil(5 / 6 * len(seeds)))
    print(f"\n{'='*100}", flush=True)
    print(f"  TRAJECTORY  FR+LF: {f0:.3f} (D_h={dims[0]}) -> {fN:.3f} (D_h={dims[-1]})   "
          f"gap-to-ceiling: {g0:+.3f} -> {gN:+.3f}   near-parity@{dims[-1]}: {par_N}/{len(seeds)}", flush=True)
    if par_N >= bar and fN >= 0.85:
        verdict = "RE_OPEN_BUILD"
        print(f"  RE-OPEN BUILD: at D_h={dims[-1]} the learned-filler bundling reaches near-parity with the matched "
              f"ceiling ({fN:.3f}) in {par_N}/{len(seeds)} seeds -- the 0.603 was a CAPACITY artifact, not a "
              f"ceiling. The weeks-scale on-bridge build is JUSTIFIED: route the learned fillers through the "
              f"guarded coincidence-plateau self-inverse primitive at the larger bind-space.", flush=True)
    elif fN >= f0 + 0.10:
        verdict = "PARTIAL_CAPACITY"
        print(f"  PARTIAL: capacity helps (FR+LF {f0:.3f} -> {fN:.3f}, +{fN-f0:.3f}) but does NOT reach parity "
              f"(gap {gN:+.3f} at D_h={dims[-1]}). The learned fillers still cost accuracy vs the fixed algebra; "
              f"a sharpened/attractor cleanup or more training is the next cheap lever before any build.", flush=True)
    else:
        verdict = "PLATEAU"
        print(f"  PLATEAU: more bind-space does NOT lift the learned-filler bundling (FR+LF {f0:.3f} -> {fN:.3f}, "
              f"gap stays {gN:+.3f}). The 0.603 is a real ceiling for LEARNED fillers in superposition -- the "
              f"fixed FHRR algebra stays the load-bearing bundler, and the learned-bind frontier is the "
              f"GENERALIZATION axis (PPMI / cross-modal arc), NOT bundling. The on-bridge build stays DEFERRED.",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)

    out = {"verdict": verdict, "seeds": seeds, "dims": dims, "pass_bar": bar,
           "by_dim": {str(k): v for k, v in by_dim.items()},
           "trajectory": {"frlf_first": f0, "frlf_last": fN, "gap_first": g0, "gap_last": gN,
                          "near_parity_last": par_N},
           "ab_reference": {"D_h": 64, "frlf_bheld": 0.603, "fixed_pm1": 0.993},
           "per_seed_rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
