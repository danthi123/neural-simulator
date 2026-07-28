#!/usr/bin/env python3
"""
gap#5 branch-B DECISIVE IGNITION SWEEP (focused; reuse-by-import of the DG-detonator de-risk internals).

THE SINGLE QUESTION (coordinator, 2026-07-23): the DG-detonator did NOT ignite even the strong-between
SYMMETRIC positive control at the default det_pa (smoke: sym forward 0.000, but that smoke was n_ca3=600).
Does ANY config make the SYMMETRIC positive control ignite DISCRETELY (ev>=1) at the completion scale
(n_ca3=2000) with escalated drive? This is the readout-can-ignite gate:
  - YES -> the readout can ignite -> proceed to the decoupled store + full 6-seed.
  - NO even at strong drive -> CLEAN branch-B honest negative (targeted ignition also can't ignite this
    store on this substrate) -> pivot to candidate #3 (theta-gamma phase-precession timing).

This probe drives the SYMMETRIC store (freeze_between_refresh=False) DIRECTLY across a
det_pa x self_regen_read x det_frac grid (the existing runner only drives the symmetric control at the
decoupled store's best_pa, so it cannot sweep the symmetric control's own drive). It reuses the exact
validated functions from _gap5_dg_detonator_ignition_derisk (_rest_and_detonate, _score) +
_gap5_sequence_replay_derisk (_prepare_sequence) + the DECOUPLED_CFG. NO new mechanism. NO sim/ edit.

ev>=1 (n_events, via _detect_events, min_frac=0.30) means the assembly COMPLETED beyond the sparse
det_frac driven cells (not just the driven cells firing) -> genuine ignition. det_frac=0.15 is the clean
ignition test (needs recurrent completion); det_frac=0.30 is borderline (driven cells alone ~= min_frac).
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

OUT = _REPO / "research" / "findings" / "raw" / "gap5_r4" / "ignition_sweep.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000, help="completion scale (RANK-1: the store completes at 2000)")
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--assembly-idx", type=int, default=0)
    ap.add_argument("--det-pa", type=float, nargs="+", default=[6000.0, 12000.0, 24000.0, 48000.0],
                    help="STRONG detonator sweep (base default was 1500..6000; this is up to 32x the base)")
    ap.add_argument("--self-regen-read", type=float, nargs="+", default=[0.0, 0.15],
                    help="0 = pure de-latch (transient); >0 adds plateau sustain so an ignition REGISTERS as an event")
    ap.add_argument("--det-frac", type=float, nargs="+", default=[0.15, 0.30])
    ap.add_argument("--det-dur", type=int, default=15)
    ap.add_argument("--det-period", type=int, default=150)
    ap.add_argument("--det-settle", type=int, default=50)
    ap.add_argument("--d-abs", type=float, default=40.0)
    ap.add_argument("--a-abs", type=float, default=0.008)
    ap.add_argument("--apical-gc-read", type=float, default=None)
    ap.add_argument("--rest-steps", type=int, default=600, help="fast: det_settle + ~3-4 det_periods is enough to detect ev>=1")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.5)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--min-frac", type=float, default=0.30)
    ap.add_argument("--active-frac", type=float, default=0.12)
    ap.add_argument("--onset-frac", type=float, default=0.08)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    print(f"[ignition-sweep] SYMMETRIC positive-control ignition sweep | n_ca3={a.n_ca3} "
          f"det_pa={a.det_pa} self_regen_read={a.self_regen_read} det_frac={a.det_frac} "
          f"det_dur={a.det_dur} rest_steps={a.rest_steps} seeds={a.seeds} backend={backend}", flush=True)

    aidx = int(a.assembly_idx)
    all_rows = []
    any_ignite = False
    t_all = time.time()
    for seed in a.seeds:
        t0 = time.time()
        cfg = {**DECOUPLED_CFG, "n_ca3": int(a.n_ca3), "n_mem": int(a.n_mem), "freeze_between_refresh": False}
        prep = _prepare_sequence(seed, cfg, do_encode=True)   # SYMMETRIC store (strong within + between)
        al = prep["assemblies_local"]
        print(f"  [seed {seed}] SYM store built: within={prep['w_within']:.1f} adj_fwd={prep['w_adj_fwd']:.2f} "
              f"adj_rev={prep['w_adj_rev']:.2f} n_ca3={a.n_ca3} ({time.time()-t0:.0f}s)", flush=True)
        for det_pa in a.det_pa:
            for sr in a.self_regen_read:
                for df in a.det_frac:
                    r = _rest_and_detonate(prep, ("assembly", aidx, df, det_pa, a.det_dur), a.rest_steps, seed,
                                           float(sr), adapt=True, d_abs=a.d_abs, a_abs=a.a_abs,
                                           det_period=a.det_period, det_settle=a.det_settle,
                                           apical_gc_read=a.apical_gc_read, verbose=False)
                    ev, seq = _score(r["F"], al, al, aidx, seed, a.window, a.ev_floor, a.ev_k, a.min_frac,
                                     a.active_frac, a.onset_frac)
                    ign = int(ev["n_events"]) >= 1
                    any_ignite = any_ignite or ign
                    row = dict(seed=seed, det_pa=det_pa, self_regen_read=sr, det_frac=df, k_det=r["k_det"],
                               n_detonations=r["n_detonations"], n_events=int(ev["n_events"]),
                               n_specific=int(ev["n_specific"]), member_frac=float(ev["member_frac"]),
                               random_frac=float(ev["random_frac"]), cross_frac=float(ev["cross_frac"]),
                               duty_cycle=float(ev["duty_cycle"]), pop_rate=float(ev["pop_rate"]),
                               forward_frac=float(seq["forward_frac"]), reverse_frac=float(seq["reverse_frac"]),
                               weights_frozen=bool(r["weights_frozen"]), apical_n_latched=int(r["apical_n_latched"]))
                    all_rows.append(row)
                    print(f"  [seed {seed}] det_pa={det_pa:>7g} sr={sr:<4g} df={df:<4g} k={r['k_det']:>3} "
                          f"nDet={r['n_detonations']:>2} | ev={ev['n_events']:>2} spec={ev['n_specific']:>2} "
                          f"memb={ev['member_frac']:.3f} cross={ev['cross_frac']:.3f} duty={ev['duty_cycle']:.3f} "
                          f"pop={ev['pop_rate']:.5f} FWD={seq['forward_frac']:.3f} frozen={r['weights_frozen']} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  [seed {seed}] done ({time.time()-t0:.0f}s)", flush=True)

    # verdict: did ANY config ignite the symmetric positive control (ev>=1)?
    max_ev = max((row["n_events"] for row in all_rows), default=0)
    max_spec = max((row["n_specific"] for row in all_rows), default=0)
    igniters = [r for r in all_rows if r["n_events"] >= 1]
    # a "clean" igniter: det_frac=0.15 (needs completion), specific (n_specific>=1)
    clean = [r for r in igniters if r["det_frac"] <= 0.15 and r["n_specific"] >= 1]
    if any_ignite:
        verdict = (f"IGNITES-YES max_ev={max_ev} max_spec={max_spec} -- the SYMMETRIC positive control DOES "
                   f"ignite discretely at strong drive ({len(igniters)}/{len(all_rows)} configs ev>=1; "
                   f"{len(clean)} clean [det_frac<=0.15 & specific]). The readout CAN ignite on this substrate "
                   f"=> proceed to the DECOUPLED store + full 6-seed at the igniting config.")
    else:
        verdict = (f"IGNITES-NO max_ev={max_ev} across ALL {len(all_rows)} configs (det_pa up to "
                   f"{max(a.det_pa):g}, self_regen_read up to {max(a.self_regen_read):g}, det_frac up to "
                   f"{max(a.det_frac):g}) -- even the strong-between SYMMETRIC positive control does NOT ignite "
                   f"discretely. The targeted DG-detonator readout itself cannot ignite this store on this "
                   f"substrate => CLEAN branch-B honest negative; pivot to candidate #3 (theta-gamma "
                   f"phase-precession timing: order the chain by TIMING, not ignition).")

    summary = dict(probe="gap5_ignition_sweep", question="does the SYMMETRIC positive control ignite (ev>=1)?",
                   any_ignite=bool(any_ignite), max_ev=max_ev, max_n_specific=max_spec,
                   n_igniting_configs=len(igniters), n_clean_igniting=len(clean), n_configs=len(all_rows),
                   seeds=a.seeds, n_ca3=a.n_ca3, det_pa=a.det_pa, self_regen_read=a.self_regen_read,
                   det_frac=a.det_frac, igniters=igniters, verdict=verdict,
                   elapsed_seconds=round(time.time() - t_all, 1), rows=all_rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118 + f"\n[ignition-sweep] VERDICT: {verdict}\n[ignition-sweep] wrote {a.out}\n" + "=" * 118, flush=True)
    return 0 if any_ignite else 1


if __name__ == "__main__":
    sys.exit(main())
