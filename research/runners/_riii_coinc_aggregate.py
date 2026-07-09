"""Aggregate per-seed JSONs from _riii_ca3_coincidence_completion_derisk (one file per seed, fanned across CPU
cores) into the 6-seed GO verdict. GO = coincidence-ON completes the held-out CA3 neurons (>0.30) FAR above the
LINEAR baseline (gain>0.15) and the NO-TRAIN control (gain>0.15), specifically (non-stored<0.20), every seed."""
from __future__ import annotations
import argparse, glob, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob of per-seed JSON files")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(a.glob)):
        try:
            rows.extend(json.load(open(f)))
        except Exception as e:
            print(f"  [skip {f}] {e}")
    if not rows:
        print("NO ROWS"); return
    rows.sort(key=lambda r: r["seed"])
    for r in rows:
        cd = f"c_drive[h={r.get('held_cdrive'):.1f} ns={r.get('nonstored_cdrive'):.1f}]" if r.get("held_cdrive") is not None else ""
        print(f"  seed {r['seed']:>3}: ON={r['on_held']:.3f} LINEAR={r['off_held']:.3f} NO-TRAIN={r['notrain_held']:.3f} "
              f"non-stored={r['on_nonstored']:.3f} {cd} (vs-lin={r['gain_vs_linear']:+.3f} vs-notr={r['gain_vs_notrain']:+.3f})")
    on = [r["on_held"] for r in rows]; gl = [r["gain_vs_linear"] for r in rows]
    gn = [r["gain_vs_notrain"] for r in rows]; ns = [r["on_nonstored"] for r in rows]
    go = (all(h > 0.30 for h in on) and all(g > 0.15 for g in gl) and all(g > 0.15 for g in gn) and all(n < 0.20 for n in ns))
    print(f"\n  N={len(rows)} | ON held-out={np.mean(on):.3f} | gain vs LINEAR={np.mean(gl):+.3f} vs NO-TRAIN={np.mean(gn):+.3f} | non-stored={np.mean(ns):.3f}")
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} (all-seed gates: ON>0.30, vs-linear>0.15, vs-notrain>0.15, non-stored<0.20)")


if __name__ == "__main__":
    main()
