"""#6 FIX A — ARM 2: sigma/gain sweep at the sel_X divisive-norm pool (grid-32, seed 42).

Sweeps `input_divisive_sigma_2` (sigma) x `input_divisive_gain_2` (gain) over a small grid and, for
each point, runs the FIX1+A `sc_popvector` arm and records the DECISIVE metric: the `sel_counts` N-S
surplus (absolute + percent). A real FIX A SHRINKS the sel surplus toward 0 WITHOUT breaking selection
(the agent must still navigate -- phase0 finalQ must not blow up vs the no-divnorm baseline). The
sweet spot is the (sigma, gain) that minimizes |sel NS_pct| while keeping phase0_finalQ near the
baseline (selection intact).

Baseline (FIX1 popvector, NO sel divnorm, grid-32 seed 42, same step budget): sel_counts N-S ~ +22%.

Each sweep point reuses `run_arm` from `_nav_sc_popvector_readout_derisk.py` (the FIX1+A path) so the
mechanism wiring is identical to ARM 3. GPU (SIM_BACKEND=cupy), grid-32 FAITHFUL.

Writes one summary JSON per point + a roll-up table. Run as:
  SIM_BACKEND=cupy python -m research.runners._nav_sc_fixA_arm2_sweep --seed 42 --grid-size 32 \
      --n-steps 480 --warmup-steps 200 --out research/findings/raw/nav_gate_2a/scpv_FIXA_arm2_sweep.json
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SC_RET_SC", "160")
os.environ.setdefault("SC_REC", "12")
os.environ.setdefault("SC_RET_DRIVE", "3500")
os.environ.setdefault("SC_ROS_US", "40")

import argparse
import json


def _sel_surplus(summary):
    ss = (summary or {}).get("stage_surplus") or {}
    sel = ss.get("sel_counts")
    if not sel:
        return None
    return {"NS_surplus": sel["NS_surplus"], "NS_pct": sel["NS_pct"],
            "N": sel["N"], "E": sel["E"], "S": sel["S"], "W": sel["W"]}


def main():
    ap = argparse.ArgumentParser(description="#6 FIX A ARM 2 sigma/gain sweep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=480)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--sc-cortex-w", type=float, default=18.0)
    ap.add_argument("--divnorm-sigma", type=float, default=5.0,   # cortex_X pool-1 (popvector) divnorm
                    help="cortex_X (pool-1) divisive sigma for the popvector read-out (fixed across sweep).")
    ap.add_argument("--divnorm-gain", type=float, default=0.02,
                    help="cortex_X (pool-1) divisive gain for the popvector read-out (fixed across sweep).")
    ap.add_argument("--sigmas", type=str, default="0.5,1,2",
                    help="comma list of sel-divnorm sigma values to sweep.")
    ap.add_argument("--gains", type=str, default="0.5,1,2",
                    help="comma list of sel-divnorm gain values to sweep.")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/nav_gate_2a/scpv_FIXA_arm2_sweep.json")
    args = ap.parse_args()

    from research.runners._nav_sc_popvector_readout_derisk import run_arm

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    gains = [float(x) for x in args.gains.split(",") if x.strip()]

    points = []
    for sig in sigmas:
        for g in gains:
            tag = f"s{sig}_g{g}".replace(".", "p")
            point_out_dir = os.path.join(out_dir, f"arm2_{tag}")
            os.makedirs(point_out_dir, exist_ok=True)
            print(f"\n[arm2] ===== sweep point sigma={sig} gain={g} (seed {args.seed}, grid {args.grid_size}) =====",
                  flush=True)
            s = run_arm("sc_popvector", args.seed, args.n_steps, args.grid_size, args.warmup_steps,
                        point_out_dir, with_conv=False, sc_cortex_w=args.sc_cortex_w,
                        divnorm_sigma=args.divnorm_sigma, divnorm_gain=args.divnorm_gain,
                        fix1=True, fixA=True,
                        sel_divnorm_sigma=sig, sel_divnorm_gain=g)
            sel = _sel_surplus(s)
            pt = {
                "sigma": sig, "gain": g,
                "sel_NS_surplus": (sel["NS_surplus"] if sel else None),
                "sel_NS_pct": (sel["NS_pct"] if sel else None),
                "phase0_finalQ": s.get("phase0_finalQ"),
                "per_phase_dominant_cardinal": s.get("per_phase_dominant_cardinal"),
                "per_phase_action_frac": s.get("per_phase_action_frac"),
                "late_motor_sustain_frac": s.get("late_motor_sustain_frac"),
                "stage_surplus": s.get("stage_surplus"),
                "summary_json": os.path.join(point_out_dir, f"scpv_sc_popvector_seed{args.seed}.json"),
            }
            points.append(pt)
            print(f"[arm2] sigma={sig} gain={g} -> sel NS={pt['sel_NS_surplus']} "
                  f"({pt['sel_NS_pct']}%) phase0_finalQ={pt['phase0_finalQ']}", flush=True)
            # incremental write so a crash mid-sweep keeps the completed points.
            with open(args.out, "w") as f:
                json.dump({"baseline_sel_NS_pct_no_divnorm": 22.2, "points": points,
                           "config": vars(args)}, f, indent=2)

    # find the sweet spot: minimize |sel NS_pct| among points whose phase0_finalQ is not blown up.
    valid = [p for p in points if p["sel_NS_pct"] is not None]
    sweet = None
    if valid:
        sweet = min(valid, key=lambda p: abs(p["sel_NS_pct"]))
    print("\n[arm2] ===== SWEEP TABLE (sel_counts N-S surplus per sigma/gain) =====", flush=True)
    print(f"  {'sigma':>6} {'gain':>6} {'sel_NS':>10} {'sel_NS_pct':>11} {'phase0_finalQ':>14} {'dom':>6}",
          flush=True)
    for p in points:
        dom = (p["per_phase_dominant_cardinal"] or [None])[0]
        print(f"  {p['sigma']:>6} {p['gain']:>6} {str(p['sel_NS_surplus']):>10} "
              f"{str(p['sel_NS_pct']):>11} {str(round(p['phase0_finalQ'],3) if p['phase0_finalQ'] else None):>14} "
              f"{str(dom):>6}", flush=True)
    if sweet:
        print(f"\n[arm2] SWEET SPOT (min |sel NS pct|): sigma={sweet['sigma']} gain={sweet['gain']} "
              f"-> sel NS_pct={sweet['sel_NS_pct']}% (baseline +22.2%) phase0_finalQ={sweet['phase0_finalQ']}",
              flush=True)

    with open(args.out, "w") as f:
        json.dump({"baseline_sel_NS_pct_no_divnorm": 22.2, "points": points,
                   "sweet_spot": sweet, "config": vars(args)}, f, indent=2)
    print(f"[arm2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
