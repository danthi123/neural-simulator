"""Per-seed pass rates on dev seeds 1-60 for candidate operating points (for the binomial GO threshold)."""
import replica3 as R, numpy as np, sys, json
from math import comb
cfgs = [("base", 4800, 4, {}), ("base", 4800, 5, {}), ("base", 4800, 6, {}),
        ("dosed", 2400, 6, dict(joint_p=0.12, p_point=0.4, possg_use=1.0, ord_use=0.8)),
        ("dosed", 4800, 4, dict(joint_p=0.12, p_point=0.4, possg_use=1.0, ord_use=0.8))]
out = {}
for name, T, lt, kw in cfgs:
    res = [R.run_seed(s, "covpre", lt, T=T, **kw) for s in range(1, 61)]
    per = {k: sum(b[k] for b in res) / 60 for k in R.KEYS}; q = sum(all(b[k] for k in R.KEYS) for b in res) / 60
    p14 = sum(comb(15, j) * q ** j * (1 - q) ** (15 - j) for j in range(14, 16))
    p13 = sum(comb(15, j) * q ** j * (1 - q) ** (15 - j) for j in range(13, 16))
    print("%s T=%d lt=%d: per-bar pass rate %s | ALL q=%.3f | P(>=14/15)=%.3f P(>=13/15)=%.3f | minfire med %.2f maxsil med %.2f" % (
        name, T, lt, {k: round(v, 3) for k, v in per.items()}, q, p14, p13,
        np.median([b["min_fire"] for b in res]), np.median([b["max_silent"] for b in res])), flush=True)
    out["%s_%d_%d" % (name, T, lt)] = dict(per=per, q_all=q, p14=p14, p13=p13)
json.dump(out, open("rate60.json", "w"), indent=1)
