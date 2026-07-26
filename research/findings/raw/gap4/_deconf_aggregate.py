"""Aggregate the 6 per-seed deconfounded-credit JSONs -> mean-over-seeds table + GO verdict + regression check vs the
prior GO (bdsp arm should be byte-reproducible since credit_lesion=None is a faithful copy of the parent rule)."""
import json, glob, os
import numpy as np

DIR = os.path.dirname(__file__)
seeds = [42, 43, 44, 100, 101, 102]
per = []
for s in seeds:
    p = os.path.join(DIR, f"deconf_s{s}.json")
    if not os.path.exists(p):
        print(f"MISSING seed {s}: {p}"); continue
    per.append(json.load(open(p))["per"][0])

# prior GO per-seed bdsp (from bdsp_faithful_s*.log), fracs 1.0/0.1/0.05
prior_bdsp = {42: [0.810, 0.909, 0.823], 43: [None, 0.729, 0.744], 44: [None, 0.824, 0.815],
              100: [None, None, None], 101: [None, None, None], 102: [None, None, None]}

fracs = [r["frac"] for r in per[0]["rows"]]
keys = ["reservoir", "fa_linear", "bdsp", "bdsp_shufE", "bdsp_permB", "bdsp_shuffled_target", "ncc_init"]
print(f"n_seeds={len(per)}  fracs={fracs}\n")
print(f"{'frac':>5} | {'RES':>6} {'fa_lin':>6} {'bdsp':>6} {'shufE':>6} {'permB':>6} {'shufY':>6} {'ncc_i':>6} | GO")
for i, frac in enumerate(fracs):
    agg = {k: float(np.mean([p["rows"][i][k] for p in per])) for k in keys}
    n_go = sum(p["rows"][i]["GO"] for p in per)
    print(f"{frac:>5.2f} | {agg['reservoir']:>6.3f} {agg['fa_linear']:>6.3f} {agg['bdsp']:>6.3f} "
          f"{agg['bdsp_shufE']:>6.3f} {agg['bdsp_permB']:>6.3f} {agg['bdsp_shuffled_target']:>6.3f} "
          f"{agg['ncc_init']:>6.3f} | {n_go}/{len(per)}")

print("\nPer-seed GO detail:")
for p in per:
    s = p["seed"]
    for i, r in enumerate(p["rows"]):
        print(f"  seed {s:>3} frac={r['frac']:.2f}: bdsp={r['bdsp']:.3f} res={r['reservoir']:.3f} "
              f"shufE={r['bdsp_shufE']:.3f} shufY={r['bdsp_shuffled_target']:.3f} ncc_i={r['ncc_init']:.3f} GO={r['GO']}")

print("\nRegression check (my bdsp vs prior GO bdsp; should match within ~0.02 -- credit_lesion=None is faithful copy):")
for p in per:
    s = p["seed"]
    for i, r in enumerate(p["rows"]):
        pb = prior_bdsp.get(s, [None, None, None])[i]
        if pb is not None:
            d = r["bdsp"] - pb
            flag = "OK" if abs(d) < 0.03 else "**DRIFT**"
            print(f"  seed {s} frac={r['frac']:.2f}: mine={r['bdsp']:.3f} prior={pb:.3f} d={d:+.3f} {flag}")

# selectivity headline (init, worst hidden layer ncc) per frac, mean
print("\nInput-selectivity at INIT (ncc_acc, mean over seeds; chance=0.10):")
for i, frac in enumerate(fracs):
    sel = [min(h["ncc_acc"] for h in p["rows"][i]["sel_init"]) for p in per]
    print(f"  frac={frac:.2f}: ncc_init mean={np.mean(sel):.3f} min={np.min(sel):.3f}")
