"""Cascade-accumulator FIX B (sel opponent-pair) weight SWEEP (2026-06-20).

Runs the FIX-1 popvector arm of the #6 rig with FIX B (opponent-pair the sel accumulators: N<->S, E<->W
integrate the DIFFERENCE via balanced sel_FS axis-partner inhibition) across a grid of opponent weights,
and tabulates the per-stage N-S surplus + per-phase dom tracking + post-change finalQ.

The success op-point: the sel/commit N-S surplus SHRINKS toward 0 (here the common-mode N-S cancels in the
difference, so the shrink should be DECOUPLED from absolute decisiveness -- the FIX-A failure mode) AND the
per-phase dom still TRACKS the goal. grid-32 IS the verdict.

NO sim/ edit (runner-only opponent-pair re-weighting). GPU (SIM_BACKEND=cupy).
"""
import argparse
import json
import os

from research.runners._nav_sc_popvector_readout_derisk import run_arm


def _track(doms):
    n_distinct = len(set(d for d in doms if d is not None))
    has_W = any(d == "W" for d in doms)
    has_E = any(d == "E" for d in doms)
    return {"n_distinct": n_distinct, "has_W": has_W, "has_E": has_E,
            "tracks": (n_distinct >= 2 and (has_W or has_E))}


def main():
    ap = argparse.ArgumentParser(description="FIX B sel opponent-pair weight sweep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--warmup-steps", type=int, default=600)
    ap.add_argument("--sc-cortex-w", type=float, default=18.0)
    ap.add_argument("--divnorm-sigma", type=float, default=1.0)
    ap.add_argument("--divnorm-gain", type=float, default=1.0)
    ap.add_argument("--opp-weights", type=str, default="5,12,25,50",
                    help="comma list of FIX B opponent weights to sweep")
    ap.add_argument("--crossaxis-weight", type=float, default=0.0)
    ap.add_argument("--with-conv", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/nav_gate_2a/fixB_sweep.json")
    args = ap.parse_args()

    opp_weights = [float(w) for w in args.opp_weights.split(",") if w.strip()]
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for ow in opp_weights:
        s = run_arm("sc_popvector", args.seed, args.n_steps, args.grid_size, args.warmup_steps,
                    out_dir, args.with_conv, args.sc_cortex_w,
                    args.divnorm_sigma, args.divnorm_gain,
                    fix1=True, fixB=True,
                    sel_opponent_weight=ow, sel_crossaxis_weight=args.crossaxis_weight)
        ss = s.get("stage_surplus") or {}
        doms = s.get("per_phase_dominant_cardinal", [])
        tr = _track(doms)
        row = {
            "opp_weight": ow, "crossaxis_weight": args.crossaxis_weight,
            "thal_NS": (ss.get("thal_counts") or {}).get("NS_surplus"),
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
        print(f"[fixB-sweep] opp_w={ow} | sel_NS={row['sel_NS']} ({row['sel_NS_pct']}%) "
              f"commit_NS={row['commit_NS']} ({row['commit_NS_pct']}%) dom={doms} "
              f"tracks={tr['tracks']} postΣ={row['post_change_finalQ_sum']}", flush=True)

    out = {"seed": args.seed, "grid_size": args.grid_size, "n_steps": args.n_steps,
           "warmup_steps": args.warmup_steps, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[fixB-sweep] wrote {args.out}", flush=True)
    print("[fixB-sweep] ===== SWEEP TABLE (sel_NS surplus shrink + dom tracking) =====", flush=True)
    print(f"  {'opp_w':>6} {'sel_NS':>9} {'sel%':>7} {'commit_NS':>10} {'dom_per_phase':>22} "
          f"{'tracks':>7} {'postΣ':>8}", flush=True)
    for r in rows:
        print(f"  {r['opp_weight']:>6.1f} {str(r['sel_NS']):>9} {str(r['sel_NS_pct']):>7} "
              f"{str(r['commit_NS']):>10} {str(r['dom_per_phase']):>22} {str(r['tracks_goal']):>7} "
              f"{str(r['post_change_finalQ_sum']):>8}", flush=True)


if __name__ == "__main__":
    main()
