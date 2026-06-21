"""#6 FIX B — the SURPASS step after FIX A's characterized NEGATIVE: opponent-pair the sel accumulators.

FIX A (divisive norm at the sel input) was the wrong operator for the residual: it removes ABSOLUTE
common-mode amplitude but a divisive COMMON SCALAR cannot remove a RELATIVE (ratio) sel bias, and it
HURT re-orienting (FIX1 tracked 3/4 phases; FIX1+A was stuck-E). The SURPASS ranking (Bogacz 2006) puts
FIX B rank-1: re-weight the sel ring into balanced AXIS-PARTNER opponent inhibition (N<->S, E<->W) so each
axis integrates the DIFFERENCE -> a shared offset AND a relative ratio bias cancel structurally, and the
(already-present) goal-direction SC margin becomes the decisive axis winner.

Per seed, four arms at the FIX-B operating point, mirroring the FIX-A ARM-3 rig exactly:
  - FIX1+B : sc_popvector + --fix1 + --fixB (the SURPASS build).
  - FIX1   : sc_popvector + --fix1 (the no-opponent baseline; tracks 3/4 at seed 42).
  - HOST   : the host-heuristic argmax oracle (the orienting ceiling FIX-B must reach to retire the host).
  - SCRAM  : sc_popvector_scr + --fix1 + --fixB (the retinotopy lesion; MUST collapse if FIX1+B tracks).

Decisive checks: (1) does FIX1+B's sel/commit N-S surplus shrink in BOTH abs AND percent (the ratio
collapses, unlike FIX-A); (2) does FIX1+B RE-ORIENT (tracks_goal True, >=2 distinct dominants incl the
W/E phases) and approach the HOST ceiling; (3) nav not regressed; (4) SCRAM collapses (read-out is now
load-bearing); (5) moat untouched (no --with-conv -> cp_rf_w_* never built).

  SIM_BACKEND=cupy python -m research.runners._nav_sc_fixB_arm3_multiseed --seed 42 \
      --grid-size 32 --n-steps 1800 --warmup-steps 600 \
      --sel-opponent-weight 12.0 --sel-crossaxis-weight 0.0 \
      --out research/findings/raw/nav_gate_2a/scpv_FIXB_arm3_seed42.json
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SC_RET_SC", "160")
os.environ.setdefault("SC_REC", "12")
os.environ.setdefault("SC_RET_DRIVE", "3500")
os.environ.setdefault("SC_ROS_US", "40")

import argparse
import json


def _sel(summary, key):
    ss = (summary or {}).get("stage_surplus") or {}
    sel = ss.get("sel_counts")
    return (sel.get(key) if sel else None)


def main():
    ap = argparse.ArgumentParser(description="#6 FIX B ARM 3 SURPASS (single seed per call)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--warmup-steps", type=int, default=600)
    ap.add_argument("--sc-cortex-w", type=float, default=18.0)
    ap.add_argument("--divnorm-sigma", type=float, default=5.0)   # cortex_X pool-1 popvector divnorm
    ap.add_argument("--divnorm-gain", type=float, default=0.02)
    ap.add_argument("--sel-opponent-weight", type=float, default=12.0,
                    help="FIX-B balanced sel_FS_X -> axis-partner inhibitory weight (the opponent strength).")
    ap.add_argument("--sel-crossaxis-weight", type=float, default=0.0,
                    help="FIX-B weak/zero cross-axis sel_FS_X -> non-partner inhibitory weight.")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    from research.runners._nav_sc_popvector_readout_derisk import run_arm

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    point_dir = os.path.join(out_dir, f"fixB_seed{args.seed}")
    os.makedirs(point_dir, exist_ok=True)

    common = dict(seed=args.seed, n_steps=args.n_steps, grid_size=args.grid_size,
                  warmup_steps=args.warmup_steps, out_dir=point_dir, with_conv=False,
                  sc_cortex_w=args.sc_cortex_w, divnorm_sigma=args.divnorm_sigma,
                  divnorm_gain=args.divnorm_gain)

    results = {}
    print(f"\n[fixB seed={args.seed}] ===== HOST (orienting ceiling) =====", flush=True)
    results["HOST"] = run_arm("host", **common)
    _dump(args, results, final=False)
    print(f"\n[fixB seed={args.seed}] ===== FIX1 (no opponent baseline) =====", flush=True)
    results["FIX1"] = run_arm("sc_popvector", fix1=True, **common)
    _dump(args, results, final=False)
    print(f"\n[fixB seed={args.seed}] ===== FIX1+B (opponent-pair sel, w={args.sel_opponent_weight}) =====",
          flush=True)
    results["FIX1B"] = run_arm("sc_popvector", fix1=True, fixB=True,
                               sel_opponent_weight=args.sel_opponent_weight,
                               sel_crossaxis_weight=args.sel_crossaxis_weight, **common)
    _dump(args, results, final=False)
    print(f"\n[fixB seed={args.seed}] ===== SCRAM (lesion of FIX1+B) =====", flush=True)
    results["SCRAM"] = run_arm("sc_popvector_scr", fix1=True, fixB=True,
                               sel_opponent_weight=args.sel_opponent_weight,
                               sel_crossaxis_weight=args.sel_crossaxis_weight, **common)
    _dump(args, results, final=True)


def _dump(args, results, final):
    host = results.get("HOST"); fix1 = results.get("FIX1")
    fix1b = results.get("FIX1B"); scram = results.get("SCRAM")

    def _ps(s):
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
        "sel_opponent_weight": args.sel_opponent_weight, "sel_crossaxis_weight": args.sel_crossaxis_weight,
        "HOST_post_change_finalQ_sum": _ps(host), "FIX1_post_change_finalQ_sum": _ps(fix1),
        "FIX1B_post_change_finalQ_sum": _ps(fix1b), "SCRAM_post_change_finalQ_sum": _ps(scram),
        "HOST_gate_score": (host.get("gate_score") if host else None),
        "FIX1_gate_score": (fix1.get("gate_score") if fix1 else None),
        "FIX1B_gate_score": (fix1b.get("gate_score") if fix1b else None),
        "SCRAM_gate_score": (scram.get("gate_score") if scram else None),
        "FIX1_sel_NS_pct": _sel(fix1, "NS_pct"), "FIX1B_sel_NS_pct": _sel(fix1b, "NS_pct"),
        "FIX1_sel_NS_abs": _sel(fix1, "NS_surplus"), "FIX1B_sel_NS_abs": _sel(fix1b, "NS_surplus"),
        "HOST_tracking": _trk(host), "FIX1_tracking": _trk(fix1),
        "FIX1B_tracking": _trk(fix1b), "SCRAM_tracking": _trk(scram),
        "FIX1B_late_sustain": (fix1b.get("late_motor_sustain_frac") if fix1b else None),
        "FIX1B_stage_surplus": (fix1b.get("stage_surplus") if fix1b else None),
        "FIX1_stage_surplus": (fix1.get("stage_surplus") if fix1 else None),
        "moat_untouched": "by construction (no --with-conv: cp_rf_w_* never allocated; nav cascade arrays "
                          "are array-disjoint)",
    }
    h, fb, sc = _ps(host), _ps(fix1b), _ps(scram)
    verdict["host_over_FIX1B_post_ratio"] = (float(h / fb) if (h and fb and fb > 0) else None)
    verdict["FIX1B_over_SCRAM_post_ratio"] = (float(fb / sc) if (fb and sc and sc > 0) else None)

    with open(args.out, "w") as f:
        json.dump({"verdict": verdict, "arms": results}, f, indent=2)
    if final:
        print(f"\n[fixB seed={args.seed}] ===== FIX-B VERDICT (seed {args.seed}) =====", flush=True)
        for k, v in verdict.items():
            print(f"  {k}: {v}", flush=True)
        print(f"[fixB seed={args.seed}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
