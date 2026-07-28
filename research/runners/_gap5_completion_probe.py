"""DECISIVE probe after STEP-1 (sel_inhib_spare sweep) showed theta reaches the members but NO assembly COMPLETES
(per_asm_active=[0,0,0] at every spare x depth). Two decisive axes, run in parallel:

  AXIS A (completion): on the PLAIN GO store (sel_inhib_spare=0, no ff_basket), does the sparse per-theta CUE ignite a
  FULL assembly-0 when the plateau SUSTAINS the completion (self_regen_read>0 = bistable) vs the de-latch (self_regen=0)?
  The STEP-1 readout used self_regen_read=0 (de-latch) + cranked adaptation d_abs=40 -> the cue cells fire once + fatigue
  before the within-attractor completes. Hypothesis: the completion needs a SUSTAINING plateau (self_regen>0); the
  de-latch is for the TRANSITION, but you can't transition what never ignited.

  AXIS B (STEP-2 ff_basket): build the RANK-2 E-pct-max ca3_ff_basket (a SEPARATE region NOT subject to the ca3_pv_basket
  ->member sparing), target theta onto it, same self_regen x depth sweep. Decisive re: does the ff_basket enable a
  discrete completing + forward-ordered sweep where the ca3_pv_basket could not?

Direct completion metric: a0_peak = max over time of the SMOOTHED assembly-0 active fraction (independent of the event
detector's threshold) -> unambiguous "did assembly-0 ignite to a full burst?". Also the full _detect_sequence_events
order stats. numpy CPU; reuse ONE frozen store per build across the self_regen x depth grid. NO sim/ edit.
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


def _smooth(x, W=5):
    return np.convolve(x.astype(float), np.ones(W), mode="same") / W if W > 1 else x.astype(float)


def _asm_peaks(F, assemblies_local, W=5):
    """Direct per-assembly MAX smoothed active fraction over the whole run (completion, no event-detector threshold)."""
    return [float(_smooth(F[:, a].sum(1), W).max()) / max(1, len(a)) for a in assemblies_local]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff-inhib", type=float, default=0.0, help="0 = plain store (AXIS A); >0 = build ca3_ff_basket (AXIS B)")
    ap.add_argument("--region", type=str, default="ca3_pv_basket", choices=["ca3_pv_basket", "ca3_ff_basket"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000)
    ap.add_argument("--self-regens", type=float, nargs="+", default=[0.0, 0.05, 0.15])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.0, 120.0])
    ap.add_argument("--d-abs", type=float, default=40.0)
    ap.add_argument("--det-pa", type=float, default=3000.0)
    ap.add_argument("--theta-period", type=int, default=220)
    ap.add_argument("--rest-steps", type=int, default=880)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = dict(DECOUPLED_CFG); cfg["n_ca3"] = int(a.n_ca3); cfg["n_mem"] = 3
    if a.ff_inhib > 0:
        cfg["ca3_ff_inhib"] = float(a.ff_inhib)      # build the RANK-2 ff_basket (threaded to _build; else absent)
    det = dict(W=5, ev_floor=0.4, ev_k=4.0, active_frac=0.12, onset_frac=0.08)
    axis = "B(ff_basket)" if a.ff_inhib > 0 else "A(plain)"
    tag = f"ff{a.ff_inhib:g}_{a.region}"
    out_path = a.out or f"research/findings/raw/gap5_r4/completion_probe_{tag}.json"
    t0 = time.time()
    print(f"[compl-probe] AXIS {axis} ff_inhib={a.ff_inhib} region={a.region} n_ca3={a.n_ca3} "
          f"self_regens={a.self_regens} depths={a.depths} d_abs={a.d_abs} seeds={a.seeds}", flush=True)
    results = {"axis": axis, "ff_inhib": a.ff_inhib, "region": a.region, "self_regens": a.self_regens,
               "depths": a.depths, "d_abs": a.d_abs, "det_pa": a.det_pa, "per_seed": []}
    theta_region = a.region if a.ff_inhib > 0 else None
    for seed in a.seeds:
        prep = _prepare_sequence(seed, cfg, do_encode=True)
        al = prep["assemblies_local"]
        asy = dict(within=float(prep["w_within"]), adj_fwd=float(prep["w_adj_fwd"]), adj_rev=float(prep["w_adj_rev"]),
                   ratio=float(prep["w_adj_fwd"]) / max(abs(float(prep["w_adj_rev"])), 1e-6))
        # confirm the ff_basket region got built (AXIS B)
        try:
            ffn = len(list(prep["bridge"].region_manager.indices("ca3_ff_basket")))
        except Exception:
            ffn = 0
        print(f"  [{axis} seed {seed}] store within={asy['within']:.1f} adj_fwd={asy['adj_fwd']:.2f} "
              f"adj_rev={asy['adj_rev']:.2f} ratio={asy['ratio']:.2f}x ff_basket_n={ffn} ({time.time()-t0:.0f}s)", flush=True)
        rows = []
        for sr in a.self_regens:
            for depth in a.depths:
                r = _rest_theta_sweep(prep, a.rest_steps, seed, theta_target="basket", cue=True, theta_period=a.theta_period,
                                      theta_depth=depth, basket_baseline=0.0, theta_exc_pa=800.0, det_frac=0.15,
                                      det_pa=a.det_pa, det_dur=12, det_settle=60, self_regen_read=sr, d_abs=a.d_abs,
                                      a_abs=0.008, adapt=True, theta_region=theta_region)
                peaks = _asm_peaks(r["F"], al)
                s = _detect_sequence_events(r["F"], al, **det)
                row = dict(self_regen=sr, depth=depth, a0_peak=peaks[0], asm_peaks=[round(p, 3) for p in peaks],
                           completes_a0=bool(peaks[0] >= 0.30), pop_rate=float(s["pop_rate"]),
                           n_events=int(s["n_events"]), n_multi=int(s["n_multi"]),
                           forward_frac=float(s["forward_frac"]), reverse_frac=float(s["reverse_frac"]),
                           per_asm_active=[int(x) for x in s["per_asm_active"]], duty=float(s["duty_cycle"]),
                           basket_n=int(r["basket_n"]), n_cues=int(r["n_cues"]))
                rows.append(row)
                flag = "  <== a0 IGNITES" if row["completes_a0"] else ""
                fflag = "  +FWD>REV" if (s["forward_frac"] > s["reverse_frac"] and s["n_multi"] >= 1) else ""
                print(f"  [{axis} seed {seed}] sr={sr:.2f} depth={depth:>5g}: a0_peak={peaks[0]:.3f} "
                      f"peaks={row['asm_peaks']} pop={row['pop_rate']:.4f} ev={row['n_events']:>2} multi={row['n_multi']:>2} "
                      f"FWD={row['forward_frac']:.3f} REV={row['reverse_frac']:.3f} act={row['per_asm_active']} "
                      f"({time.time()-t0:.0f}s){flag}{fflag}", flush=True)
        any_ignite = any(x["completes_a0"] for x in rows)
        any_fwd = any(x["forward_frac"] > x["reverse_frac"] and x["n_multi"] >= 1 and x["completes_a0"] for x in rows)
        print(f"  [{axis} seed {seed}] SUMMARY: a0_ever_ignites={any_ignite} forward_with_completion={any_fwd} "
              f"(max a0_peak={max(x['a0_peak'] for x in rows):.3f})", flush=True)
        results["per_seed"].append(dict(seed=seed, asymmetry=asy, ff_basket_n=ffn, rows=rows,
                                        a0_ever_ignites=any_ignite, forward_with_completion=any_fwd))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"[compl-probe] DONE AXIS {axis} -> {out_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
