"""STEP-1 diagnostic sweep for the cued theta-disinhibition sweep readout: does relaxing sel_inhib_spare (the basket->
assembly-member synapse weight the DECOUPLED store zeroes) let THETA-ON-BASKET reach + disinhibit the assembly cells
WITHOUT over-inhibiting them out of completion? For a given --spare, per seed: (a) does the encode's forward-asymmetry
SURVIVE (adj_fwd>>adj_rev, ratio~7x -- verify the store didn't break; sel_inhib_spare touches only basket->member
inhibition, NOT the ca3->ca3 chain weights), (b) does theta become NON-inert (depth=0 vs depth>0 change the dynamics),
(c) does an assembly COMPLETE (per_asm_active>0) forward-ordered (forward_frac>reverse_frac)? Reuses ONE frozen store
per (spare,seed) across the theta_depth sweep (weights frozen). numpy CPU; run 3 in parallel (one per spare). NO sim/ edit.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence, _detect_sequence_events  # noqa: E402
from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG  # noqa: E402
from research.runners._gap5_theta_sweep_replay_derisk import _rest_theta_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spare", type=float, required=True, help="sel_inhib_spare (basket->member weight); 0=GO store")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--n-ca3", type=int, default=2000)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.0, 80.0, 160.0, 300.0])
    ap.add_argument("--det-pa", type=float, default=3000.0)
    ap.add_argument("--theta-period", type=int, default=220)
    ap.add_argument("--rest-steps", type=int, default=880)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = dict(DECOUPLED_CFG); cfg["n_ca3"] = int(a.n_ca3); cfg["n_mem"] = 3
    cfg["sel_inhib_spare"] = float(a.spare)
    det = dict(W=5, ev_floor=0.4, ev_k=4.0, active_frac=0.12, onset_frac=0.08)
    base = dict(theta_period=a.theta_period, basket_baseline=0.0, theta_exc_pa=800.0, det_frac=0.15, det_pa=a.det_pa,
                det_dur=12, det_settle=60, self_regen_read=0.0, d_abs=40.0, a_abs=0.008, adapt=True)
    out_path = a.out or f"research/findings/raw/gap5_r4/spare_sweep_spare{a.spare:g}.json"
    t0 = time.time()
    print(f"[spare-sweep] sel_inhib_spare={a.spare} n_ca3={a.n_ca3} depths={a.depths} seeds={a.seeds}", flush=True)
    results = {"sel_inhib_spare": a.spare, "n_ca3": a.n_ca3, "depths": a.depths, "det_pa": a.det_pa,
               "theta_period": a.theta_period, "rest_steps": a.rest_steps, "per_seed": []}
    for seed in a.seeds:
        prep = _prepare_sequence(seed, cfg, do_encode=True)
        al = prep["assemblies_local"]
        asy = dict(within=float(prep["w_within"]), adj_fwd=float(prep["w_adj_fwd"]), adj_rev=float(prep["w_adj_rev"]),
                   ratio=float(prep["w_adj_fwd"]) / max(abs(float(prep["w_adj_rev"])), 1e-6),
                   n_fwd=int(prep["n_between_fwd"]), n_rev=int(prep["n_between_rev"]),
                   sizes=[int(len(x)) for x in prep["assemblies"]])
        print(f"  [spare={a.spare:g} seed {seed}] (a) ASYMMETRY: within={asy['within']:.1f} adj_fwd={asy['adj_fwd']:.2f} "
              f"adj_rev={asy['adj_rev']:.2f} ratio={asy['ratio']:.2f}x (n_fwd={asy['n_fwd']} n_rev={asy['n_rev']}) "
              f"({time.time()-t0:.0f}s)", flush=True)
        rows = []
        for depth in a.depths:
            r = _rest_theta_sweep(prep, a.rest_steps, seed, theta_target="basket", cue=True, theta_depth=depth, **base)
            s = _detect_sequence_events(r["F"], al, **det)
            row = dict(depth=depth, pop_rate=float(s["pop_rate"]), n_events=int(s["n_events"]),
                       n_multi=int(s["n_multi"]), forward_frac=float(s["forward_frac"]),
                       reverse_frac=float(s["reverse_frac"]), chance=float(s["chance_forward"]),
                       per_asm_active=[int(x) for x in s["per_asm_active"]], duty=float(s["duty_cycle"]),
                       n_cues=int(r["n_cues"]), frozen=bool(r["weights_frozen"]))
            rows.append(row)
            print(f"  [spare={a.spare:g} seed {seed}] (c) depth={depth:>6g}: pop={row['pop_rate']:.4f} "
                  f"ev={row['n_events']:>2} multi={row['n_multi']:>2} FWD={row['forward_frac']:.3f} "
                  f"REV={row['reverse_frac']:.3f} chance={row['chance']:.3f} act={row['per_asm_active']} "
                  f"duty={row['duty']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        # (b) theta non-inert? compare depth=0 vs the max-depth dynamics
        pop0 = rows[0]["pop_rate"]; popm = rows[-1]["pop_rate"]
        act0 = sum(rows[0]["per_asm_active"]); actm = sum(rows[-1]["per_asm_active"])
        theta_reaches = bool(abs(popm - pop0) > 0.005 or actm != act0)
        best = max(rows, key=lambda x: (x["forward_frac"] - x["reverse_frac"], sum(x["per_asm_active"]), -x["duty"]))
        completes = bool(sum(best["per_asm_active"]) > 0)
        forward_ordered = bool(best["forward_frac"] > best["reverse_frac"] and best["n_multi"] >= 1)
        print(f"  [spare={a.spare:g} seed {seed}] SUMMARY: asym_survives={asy['ratio']>=4.0} theta_reaches={theta_reaches} "
              f"completes={completes} forward_ordered={forward_ordered} (best depth={best['depth']:g} "
              f"FWD={best['forward_frac']:.3f} REV={best['reverse_frac']:.3f} act={best['per_asm_active']})", flush=True)
        results["per_seed"].append(dict(seed=seed, asymmetry=asy, rows=rows, theta_reaches=theta_reaches,
                                        completes=completes, forward_ordered=forward_ordered, best=best))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"[spare-sweep] DONE spare={a.spare} -> {out_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
