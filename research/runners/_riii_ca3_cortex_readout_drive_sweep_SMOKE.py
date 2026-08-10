"""SMOKE sweep: strengthen the CA3->cortex READOUT DRIVE (episodic-recall residual).

Frontier (2026-08-10): the 6-seed GO recall is 0.646 but the ceiling probe named the WEAKEST link as the
CA3->cortex heteroassociative readout drive. This runner sweeps the RUNNER-SIDE readout levers
(--ca3-cortex-w initial weight_mean, --ca3-cortex-density fan-in, --assembly-frac assembly size) on the
SAME machinery as the committed GO runner and asks: does full recall rise above the 0.646 baseline as the
readout drive increases, WHILE permuted-cue stays specific (near chance) and untrained stays at chance?

SKEPTICAL CONTROL (mandatory): specificity (permute) must NOT rise with recall. If full recall and permute
rise together, that is a NEGATIVE (non-specific over-drive), not a win. The in-sweep w=4 point reproduces
the known baseline as the control arm.

NO sim/ edit. NEW file (does not touch the committed runner). Single-seed by design (cheap decisive smoke).
Reuses research.runners._riii_ca3_cortical_episodic_wta_derisk.run() verbatim.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._riii_ca3_cortical_episodic_wta_derisk import run, _verify_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ca3", type=int, default=1500)
    ap.add_argument("--k-items", type=int, default=4)
    ap.add_argument("--train-events", type=int, default=30)   # matches the 6-seed GO command
    ap.add_argument("--ca3-cue-frac", type=float, default=0.5)
    ap.add_argument("--recall-k-thresh", type=float, default=40.0)
    ap.add_argument("--ca3-density", type=float, default=0.05)
    # sweep points: (ca3_cortex_w, ca3_cortex_density, assembly_frac)
    ap.add_argument("--grid", default=None,
                    help="semicolon list of w,density,assembly triples; default = the frontier sweep")
    ap.add_argument("--verify-seed", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.verify_seed:
        _verify_seed(a.seed)
        return

    if a.grid:
        grid = []
        for tok in a.grid.split(";"):
            w, d, af = tok.split(",")
            grid.append((float(w), float(d), float(af)))
    else:
        # baseline (the GO config) first, then raise the readout drive three ways
        grid = [
            (4.0, 1.0, 0.10),    # BASELINE / control arm -> should reproduce ~0.646, permute/untrained at chance
            (8.0, 1.0, 0.10),    # 2x weight
            (16.0, 1.0, 0.10),   # 4x weight
            (32.0, 1.0, 0.10),   # 8x weight
            (16.0, 1.0, 0.15),   # 4x weight + larger assembly (more fan-in cells)
        ]

    chance = 1.0 / a.k_items
    common = dict(n_ca3=a.n_ca3, k_items=a.k_items, train_events=a.train_events,
                  ca3_cue_frac=a.ca3_cue_frac, recall_k_thresh=a.recall_k_thresh,
                  ca3_density=a.ca3_density, verbose=False)

    t0 = time.time()
    rows = []
    print(f"chance={chance:.3f}  seed={a.seed}  n_ca3={a.n_ca3}  train_events={a.train_events}", flush=True)
    print(f"{'w':>6} {'dens':>5} {'assy':>5} | {'FULL':>6} {'maxcx':>7} {'sep':>7} | "
          f"{'PERM':>6} {'UNTR':>6} | reads-as", flush=True)
    for (w, dens, af) in grid:
        kw = dict(common, ca3_cortex_w=w, ca3_cortex_density=dens, assembly_frac=af)
        full = run(a.seed, **kw)
        perm = run(a.seed, permute_cue=True, **kw)
        untr = run(a.seed, untrained=True, **kw)
        fw = full["winner_overall"]; pw = perm["winner_overall"]; uw = untr["winner_overall"]
        mcx = full["max_cortex_rate"]; sep = full["sep_overall"]
        from tools.lab import attributable_to
        attributable_to(f"[w={w} dens={dens}] readout drive: full recall vs untrained-CA3 (engram/attractor)", fw, uw)
        # verdict per point: recall must beat chance AND beat permute AND beat untrained (specific), and
        # permute must NOT rise materially above chance (else non-specific over-drive)
        specific = (fw > chance + 1e-6) and (fw > pw + 1e-6) and (fw > uw + 1e-6)
        perm_ok = pw <= chance + 0.08
        reads = ("SPECIFIC" if (specific and perm_ok) else
                 "NON-SPECIFIC-OVERDRIVE" if (fw > chance and pw > chance + 0.08) else
                 "at/below chance")
        rows.append({"w": w, "density": dens, "assembly_frac": af,
                     "full_winner": fw, "full_max_cortex": mcx, "full_sep": sep,
                     "permute_winner": pw, "untrained_winner": uw,
                     "specific": bool(specific), "perm_ok": bool(perm_ok), "reads_as": reads})
        print(f"{w:>6.1f} {dens:>5.2f} {af:>5.2f} | {fw:>6.3f} {mcx:>7.4f} {sep:>+7.4f} | "
              f"{pw:>6.3f} {uw:>6.3f} | {reads}   ({time.time()-t0:.0f}s)", flush=True)

    baseline = rows[0]["full_winner"]
    best = max(rows, key=lambda r: r["full_winner"] if r["specific"] and r["perm_ok"] else -1)
    print("\n=== SUMMARY ===", flush=True)
    print(f"baseline (w=4) full recall = {baseline:.3f} (headline GO = 0.646)", flush=True)
    if best["full_winner"] > baseline + 1e-6 and best["specific"] and best["perm_ok"]:
        print(f"POSITIVE: best specific point full={best['full_winner']:.3f} at "
              f"w={best['w']} dens={best['density']} assy={best['assembly_frac']} "
              f"(+{best['full_winner']-baseline:.3f} over baseline), permute={best['permute_winner']:.3f} stays specific",
              flush=True)
    else:
        print(f"NEGATIVE: no specific point beats baseline {baseline:.3f}; "
              f"best specific full={best['full_winner']:.3f} -> readout drive is capped runner-side "
              f"(consistent with the sim-internals current-delivery wall)", flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"seed": a.seed, "chance": chance, "baseline": baseline, "rows": rows}, f, indent=2)
        print(f"[wrote] {a.out}", flush=True)


if __name__ == "__main__":
    main()
