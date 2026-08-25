#!/usr/bin/env python3
"""gap#5 branch-B RANK-2 OPERATING-POINT sweep (2026-08-25): find the recall-time ca3_pv_basket FEEDBACK-inhibition gain
(fb_read) x detonator strength (det_pa) at which the DECOUPLED forward-asymmetric store IGNITES a SINGLE assembly
DISCRETELY (competitive WTA) and hands off forward -- the mechanism the DG-detonator honest-negative's own ranked
next-rung named.

WHY (read the substrate, 2026-08-25): the store builds a ca3_pv_basket E->I->E FEEDBACK pool (ca3_fb_inhib=20) but
ENCODE spares ALL members from it (sel_inhib_spare=0 zeroes basket->member), so at readout there is NO cross-assembly
lateral inhibition -> no competition. The on-disk ignition_sweep (n_ca3=2000) reads member_frac=0.000 EVERYWHERE (even
the symmetric positive control, det_pa up to 48000): the detonator lights ~k_det cells but NOTHING crosses the event
floor (ev_floor*asize ~= 103 co-firing cells) -> the assembly does not complete OR the always-on spared basket clamps
it. This sweep DISAMBIGUATES: fb_read = -1 (untouched, the on-disk baseline), 0 (basket OFF -> pure completion test),
and a RANGE (re-armed global lateral inhibition -> WTA). It records the per-assembly COMPLETION breakdown (peak a0/a1/a2
active fraction, independent of the event detector) so 'under-completion' vs 'diffuse co-fire needing WTA' is legible.

NO sim/ edit. Reuses the (modified) _rest_and_detonate (fb_read knob) + _score + _prepare_sequence + DECOUPLED_CFG.
This is a PROBE: the operating point it finds feeds the full 6-seed control suite (_gap5_dg_detonator_ignition_derisk
--fb-read <winner> --det-pa <winner>). A clean single-assembly discrete ignition here is a POINTER, not the GO.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
from sim.backend import get_backend  # noqa: E402
from research.runners._gap5_dg_detonator_ignition_derisk import _rest_and_detonate, _score  # noqa: E402
from research.runners._gap5_sequence_replay_derisk import _prepare_sequence  # noqa: E402
from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_r4" / "ignition_fbinhib_sweep.json"


def _peaks(F, assemblies_local, W=5):
    """Direct per-assembly MAX smoothed active fraction (completion, independent of the event-detector threshold)."""
    out = []
    for a in assemblies_local:
        v = F[:, a].sum(1).astype(float)
        s = np.convolve(v, np.ones(W), mode="same") / W
        out.append(float(s.max() / max(1, len(a))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000)
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--assembly-idx", type=int, default=0)
    ap.add_argument("--freeze-between", type=int, default=1, help="1 = DECOUPLED store (freeze between-refresh, forward-asym); 0 = SYMMETRIC positive control")
    ap.add_argument("--chain-btsp-lr", type=float, nargs="+", default=[None],
                    help="forward-chain learning-rate sweep (adj_fwd strength); None = store default (0.5 -> adj_fwd~38). Rebuilds the store per value.")
    ap.add_argument("--fb-read", type=float, nargs="+", default=[-1.0, 0.0, 5.0, 10.0, 20.0, 40.0],
                    help="recall ca3_pv_basket->ca3 weight sweep; -1 = untouched (on-disk baseline), 0 = basket off, >0 = re-armed WTA")
    ap.add_argument("--fb-drive", type=float, nargs="+", default=[-1.0],
                    help="E->I drive (ca3->ca3_pv_basket) multiplier sweep so a sparse burst actually fires the basket; -1 = untouched")
    ap.add_argument("--det-pa", type=float, nargs="+", default=[3000.0, 6000.0, 12000.0])
    ap.add_argument("--det-frac", type=float, default=0.15)
    ap.add_argument("--det-dur", type=int, default=15)
    ap.add_argument("--det-period", type=int, default=150)
    ap.add_argument("--det-settle", type=int, default=50)
    ap.add_argument("--self-regen-read", type=float, default=0.0)
    ap.add_argument("--d-abs", type=float, default=40.0)
    ap.add_argument("--a-abs", type=float, default=0.008)
    ap.add_argument("--apical-gc-read", type=float, default=None)
    ap.add_argument("--rest-steps", type=int, default=700)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.5)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--min-frac", type=float, default=0.30)
    ap.add_argument("--active-frac", type=float, default=0.12)
    ap.add_argument("--onset-frac", type=float, default=0.08)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    store = "DECOUPLED" if a.freeze_between else "SYMMETRIC"
    aidx = int(a.assembly_idx)
    print(f"[fb-sweep] {store} store | n_ca3={a.n_ca3} fb_read={a.fb_read} det_pa={a.det_pa} det_frac={a.det_frac} "
          f"self_regen_read={a.self_regen_read} rest_steps={a.rest_steps} seeds={a.seeds} backend={backend}", flush=True)

    all_rows = []
    t_all = time.time()
    for seed in a.seeds:
      for chain_lr in a.chain_btsp_lr:
        t0 = time.time()
        cfg = {**DECOUPLED_CFG, "n_ca3": int(a.n_ca3), "n_mem": int(a.n_mem),
               "freeze_between_refresh": bool(a.freeze_between)}
        if chain_lr is not None:
            cfg["chain_btsp_lr"] = float(chain_lr)
        prep = _prepare_sequence(seed, cfg, do_encode=True)   # ONE frozen store, reused across the fb x det_pa grid
        al = prep["assemblies_local"]
        print(f"  [seed {seed}] {store} store chain_lr={chain_lr}: within={prep['w_within']:.1f} adj_fwd={prep['w_adj_fwd']:.2f} "
              f"adj_rev={prep['w_adj_rev']:.2f} asize={[len(x) for x in al]} ({time.time()-t0:.0f}s)", flush=True)
        for fb in a.fb_read:
          fb_arg = None if fb < 0 else float(fb)
          for fbd in a.fb_drive:
            fbd_arg = None if fbd < 0 else float(fbd)
            for det_pa in a.det_pa:
                r = _rest_and_detonate(prep, ("assembly", aidx, a.det_frac, det_pa, a.det_dur), a.rest_steps, seed,
                                       float(a.self_regen_read), adapt=True, d_abs=a.d_abs, a_abs=a.a_abs,
                                       det_period=a.det_period, det_settle=a.det_settle,
                                       apical_gc_read=a.apical_gc_read, verbose=False, fb_read=fb_arg, fb_drive=fbd_arg)
                ev, seq = _score(r["F"], al, al, aidx, seed, a.window, a.ev_floor, a.ev_k, a.min_frac,
                                 a.active_frac, a.onset_frac)
                pk = _peaks(r["F"], al, a.window)
                # WTA discreteness proxy: winner (a0) peak vs the strongest competitor peak
                comp = max(pk[1:]) if len(pk) > 1 else 0.0
                row = dict(seed=seed, chain_lr=chain_lr, adj_fwd=round(float(prep["w_adj_fwd"]), 2),
                           adj_rev=round(float(prep["w_adj_rev"]), 2), fb_read=fb_arg, fb_drive=fbd_arg, det_pa=det_pa,
                           k_det=int(r["k_det"]),
                           n_detonations=int(r["n_detonations"]), n_fb_set=int(r.get("n_fb_set") or 0),
                           n_fb_drive=int(r.get("n_fb_drive") or 0),
                           basket_mean=(round(r["basket_mean"], 5) if r.get("basket_mean") is not None else None),
                           n_events=int(ev["n_events"]), n_specific=int(ev["n_specific"]),
                           member_frac=round(float(ev["member_frac"]), 4), random_frac=round(float(ev["random_frac"]), 4),
                           cross_frac=round(float(ev["cross_frac"]), 4), specificity=round(float(ev["specificity"]), 4),
                           duty_cycle=round(float(ev["duty_cycle"]), 4), pop_rate=round(float(ev["pop_rate"]), 5),
                           a0_peak=round(pk[0], 3), comp_peak=round(comp, 3), asm_peaks=[round(x, 3) for x in pk],
                           win_margin=round(pk[0] - comp, 3),
                           forward_frac=round(float(seq["forward_frac"]), 3), reverse_frac=round(float(seq["reverse_frac"]), 3),
                           n_multi=int(seq["n_multi"]), chance_forward=round(float(seq["chance_forward"]), 3),
                           per_asm_active=[int(x) for x in seq["per_asm_active"]], weights_frozen=bool(r["weights_frozen"]))
                all_rows.append(row)
                flag = ""
                if pk[0] >= 0.30 and (pk[0] - comp) >= 0.15:
                    flag = "  <== a0 WINS discretely"
                elif pk[0] >= 0.30:
                    flag = "  <== a0 ignites (co-fire)"
                print(f"  [seed {seed}] fb={str(fb_arg):>5} fbd={str(fbd_arg):>5} pa={det_pa:>7g} k={r['k_det']:>3} nDet={r['n_detonations']:>2} "
                      f"bask={row['basket_mean']} | ev={ev['n_events']:>2} spec={ev['n_specific']:>2} "
                      f"memb={ev['member_frac']:.3f} cross={ev['cross_frac']:.3f} duty={ev['duty_cycle']:.3f} | "
                      f"a0pk={pk[0]:.3f} peaks={row['asm_peaks']} margin={row['win_margin']:+.3f} "
                      f"FWD={seq['forward_frac']:.3f} REV={seq['reverse_frac']:.3f} multi={seq['n_multi']} "
                      f"frozen={r['weights_frozen']} ({time.time()-t0:.0f}s){flag}", flush=True)
        print(f"  [seed {seed}] done ({time.time()-t0:.0f}s)", flush=True)

    # summary: best discrete-WTA config = highest win_margin among rows that ignite a0 (a0_peak>=0.30) and register an event
    igniters = [r for r in all_rows if r["a0_peak"] >= 0.30]
    discrete = [r for r in igniters if r["win_margin"] >= 0.15]
    evrows = [r for r in all_rows if r["n_events"] >= 1 and r["n_specific"] >= 1]
    best = max(all_rows, key=lambda r: (r["a0_peak"] >= 0.30, r["win_margin"], r["a0_peak"])) if all_rows else None
    max_a0 = max((r["a0_peak"] for r in all_rows), default=0.0)
    if discrete:
        b = max(discrete, key=lambda r: (r["n_specific"], r["win_margin"], r["forward_frac"]))
        verdict = (f"WTA-DISCRETE-YES: {len(discrete)}/{len(all_rows)} configs ignite a0 discretely (a0_peak>=0.30 & "
                   f"win_margin>=0.15); {len(evrows)} register a specific event. BEST: fb_read={b['fb_read']} "
                   f"det_pa={b['det_pa']} -> a0_peak={b['a0_peak']} peaks={b['asm_peaks']} margin={b['win_margin']:+.3f} "
                   f"ev={b['n_events']} memb={b['member_frac']} FWD={b['forward_frac']} REV={b['reverse_frac']}. "
                   f"=> feed the full 6-seed control suite at this (fb_read, det_pa).")
    elif igniters:
        b = max(igniters, key=lambda r: r["a0_peak"])
        verdict = (f"IGNITES-BUT-NOT-DISCRETE: a0 completes (max a0_peak={max_a0:.3f}) at fb_read={b['fb_read']} "
                   f"det_pa={b['det_pa']} peaks={b['asm_peaks']} but win_margin<0.15 (co-fire). WTA gain needs tuning "
                   f"(raise fb_read for sharper competition, or the completion is diffuse across assemblies).")
    else:
        verdict = (f"NO-COMPLETION: max a0_peak={max_a0:.3f} across ALL {len(all_rows)} configs (fb_read {a.fb_read}, "
                   f"det_pa up to {max(a.det_pa):g}) -- the detonator does NOT complete a full assembly even with the "
                   f"basket OFF (fb_read=0). This is UNDER-completion, not diffuse over-fire: adding inhibition is the "
                   f"WRONG lever. Residual = within-attractor completion strength; pivot per THE LAW.")

    summary = dict(probe="gap5_ignition_fbinhib_sweep", store=store, question="recall fb-inhib gain for discrete WTA ignition?",
                   n_configs=len(all_rows), n_igniters=len(igniters), n_discrete=len(discrete), n_event_rows=len(evrows),
                   max_a0_peak=round(max_a0, 3), best=best, seeds=a.seeds, n_ca3=a.n_ca3, fb_read=a.fb_read,
                   det_pa=a.det_pa, det_frac=a.det_frac, self_regen_read=a.self_regen_read, verdict=verdict,
                   elapsed_seconds=round(time.time() - t_all, 1), rows=all_rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118 + f"\n[fb-sweep] VERDICT: {verdict}\n[fb-sweep] wrote {a.out}\n" + "=" * 118, flush=True)
    return 0 if discrete else 1


if __name__ == "__main__":
    sys.exit(main())
