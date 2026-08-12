"""Aggregate the per-seed gap#4 Forward-Forward artifacts into a per-depth 6-seed table + the GO/NEGATIVE verdict.
Reads every research/findings/raw/_gap4_ff/ff_xor_seed*.json (one process per seed, each sweeping --n-list), groups by
depth N, and reports the ENTER-THE-REGIME headline: FF held-out vs majority, per-arm means, min-over-seeds, and the GO
gate (6-seed FF >= chance+0.20, min clearly above majority, beats reservoir by >=0.10, permuted collapses, BPTT confirms).
"""
from __future__ import annotations
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _min(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.min(xs)) if xs else float("nan")


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "research/findings/raw/_gap4_ff/ff_xor_seed*.json"
    files = sorted(glob.glob(pat))
    by_N = defaultdict(list)
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        for r in d.get("results", []):
            if "error" in r:
                print(f"[ERROR] {fp} seed {r.get('seed')} N={r.get('N')}: {r['error']}")
                continue
            by_N[r["N"]].append(r)

    print(f"\n{'='*118}\ngap#4 FORWARD-FORWARD local contrastive -- 6-seed aggregate ({len(files)} seed-files)\n{'='*118}")
    hdr = (f"{'N':>2} {'seeds':>5} {'maj':>6} {'FF_inh':>7} {'FF_min':>7} {'+maj':>6} {'topLyr+maj':>9} "
           f"{'resRidge':>8} {'resFF':>6} {'BPTT':>6} {'perm':>6} {'beatRes':>7} {'>perm':>6} {'GO/6':>5}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for N in sorted(by_N):
        rows = by_N[N]
        maj = _mean([r["majority"] for r in rows])
        ff = [r["ff_inherit"] for r in rows]
        ff_mean = _mean(ff); ff_min = _min(ff)
        top = _mean([r["top_layer_above_majority"] for r in rows])
        resr = _mean([r["reservoir_ridge_inherit"] for r in rows])
        resff = _mean([r["ff_reservoir_inherit"] for r in rows])
        bptt = _mean([r["bptt_ceiling_inherit"] for r in rows])
        perm = _mean([r["permuted_inherit"] for r in rows])
        beatres = _mean([r["beats_reservoir_ridge_by"] for r in rows])
        overperm = _mean([r["directed_over_permuted"] for r in rows])
        n_go = sum(1 for r in rows if r.get("GO"))
        # 6-seed GO: mean FF >= chance+0.20, min above majority, beats reservoir, permuted collapses, bptt confirms
        six_go = bool(
            ff_mean >= maj + 0.20 and ff_min > maj + 0.05 and beatres >= 0.10 and
            overperm >= 0.10 and perm <= maj + 0.08 and bptt > maj + 0.15
        )
        enters = sum(1 for r in rows if r.get("enters_learning_regime"))
        weak = sum(1 for r in rows if r.get("weak_coupling_suspected"))
        print(f"{N:>2} {len(rows):>5} {maj:>6.3f} {ff_mean:>7.3f} {ff_min:>7.3f} {ff_mean-maj:>+6.3f} {top:>+9.3f} "
              f"{resr:>8.3f} {resff:>6.3f} {bptt:>6.3f} {perm:>6.3f} {beatres:>+7.3f} {overperm:>+6.3f} "
              f"{str(n_go)+'/'+str(len(rows)):>5}")
        summary[N] = {"n_seeds": len(rows), "majority": maj, "ff_mean": ff_mean, "ff_min": ff_min,
                      "ff_above_majority": ff_mean - maj, "top_layer_above_majority": top,
                      "reservoir_ridge": resr, "reservoir_ff": resff, "bptt_ceiling": bptt,
                      "permuted": perm, "beats_reservoir_by": beatres, "directed_over_permuted": overperm,
                      "per_seed_GO": n_go, "six_seed_GO": six_go, "n_enters_regime": enters,
                      "n_weak_coupling": weak,
                      "per_layer_acc_mean": [_mean([r["ff_per_layer_acc"][i] for r in rows
                                                    if len(r.get("ff_per_layer_acc", [])) > i])
                                             for i in range(N)]}
    print(f"\nPer-depth 6-seed GO: " + ", ".join(f"N={N}:{'GO' if summary[N]['six_seed_GO'] else 'no'}"
                                                 for N in sorted(summary)))
    print("per-layer mean held-out acc (index 0 = first hidden -> last = top hidden):")
    for N in sorted(summary):
        print(f"  N={N}: {['%.3f'%x for x in summary[N]['per_layer_acc_mean']]}  (majority {summary[N]['majority']:.3f})")
    outp = Path(pat).parent / "aggregate_gap4_ff.json"
    with open(outp, "w") as f:
        json.dump({"n_files": len(files), "by_depth": summary}, f, indent=2)
    print(f"\n[wrote {outp}]")


if __name__ == "__main__":
    main()
