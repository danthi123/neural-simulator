"""(de-risk A) Aggregate the per-seed divnorm sweeps (42/43/44) to find the ROBUST operating point — the
(w_match, bias, w_cfs, w_fs, einh) that maximizes the WORST-CASE recovery across seeds (max-of-min). The
seed-42-only sweep overfits (its best point gave seed-43 0.507 in the multiseed); the honest de-risk question
is whether ANY single fixed operating point reaches numpy parity on ALL seeds.

GO if the robust point's min-across-seeds reaches numpy parity (>= ~0.95). NEGATIVE if even the best worst-case
leaves a seed well below numpy 1.000 (the fixed-threshold spiking cleanup cannot be made seed-robust here).

  python -m research.findings.raw._divnorm_robust_agg
"""
from __future__ import annotations
import json


SEEDS = [42, 43, 44]
KEYS = ("w_match", "bias", "w_cfs", "w_fs", "einh")


def opkey(row):
    return tuple(row[k] for k in KEYS)


def main():
    per_seed = {}
    numpy_by_seed = {}
    nodiv_best_by_seed = {}
    for s in SEEDS:
        d = json.load(open(f"research/findings/raw/_divnorm_sweep_seed{s}.json"))
        per_seed[s] = {opkey(r): r["spiking"] for r in d["div_rows"]}
        numpy_by_seed[s] = d["numpy"]
        nodiv_best_by_seed[s] = d["nodiv_global_best"]["spiking"]

    # operating points present in ALL seeds (same grid -> all shared)
    common = set(per_seed[SEEDS[0]])
    for s in SEEDS[1:]:
        common &= set(per_seed[s])

    rows = []
    for op in common:
        recs = [per_seed[s][op] for s in SEEDS]
        rows.append({"op": dict(zip(KEYS, op)), "per_seed": dict(zip(SEEDS, [round(r, 3) for r in recs])),
                     "min": min(recs), "mean": sum(recs) / len(recs)})
    rows.sort(key=lambda r: (r["min"], r["mean"]), reverse=True)

    print(f"numpy oracle per seed: {numpy_by_seed}")
    print(f"nodiv global best per seed: {nodiv_by_seed if False else nodiv_best_by_seed}")
    print(f"\n{len(common)} shared operating points. TOP 8 by worst-case (max-of-min across seeds):\n")
    for r in rows[:8]:
        print(f"  min={r['min']:.3f} mean={r['mean']:.3f}  per_seed={r['per_seed']}  op={r['op']}")

    best = rows[0]
    np_min = min(numpy_by_seed.values())
    print(f"\n[ROBUST BEST] min-across-seeds={best['min']:.3f} mean={best['mean']:.3f}  op={best['op']}")
    print(f"[numpy parity bar] min numpy across seeds={np_min:.3f}")
    margin = best["min"] - np_min
    verdict = "GO" if best["min"] >= 0.95 else "NEGATIVE"
    print(f"[VERDICT] robust worst-case {best['min']:.3f} vs numpy {np_min:.3f}  margin={margin:+.3f}  -> {verdict}")
    json.dump({"top": rows[:8], "robust_best": best, "numpy_by_seed": numpy_by_seed,
               "nodiv_best_by_seed": nodiv_best_by_seed, "verdict": verdict},
              open("research/findings/raw/_divnorm_robust_agg.json", "w"), indent=2)


if __name__ == "__main__":
    main()
