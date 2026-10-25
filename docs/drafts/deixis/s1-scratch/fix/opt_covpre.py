import replica3 as R, numpy as np, sys
from collections import Counter
def count_min(T, **kw):
    cs = []
    for sd in range(1, 16):
        c = Counter()
        for x, y, m in R.world(sd, T, **kw):
            for i in np.nonzero(x)[0]: c[R.TOK[i]] += 1
        cs.append(c)
    return {t: int(min(c[t] for c in cs)) for t in ["POSS_S", "POSS_A", "PLUR", "POSS_G", "DEM", "ORD1", "ORD2", "ANA", "CN", "N_V3", "OOI1", "PREV"]}
cfgs = {"base": {}, "dosed": dict(joint_p=0.12, p_point=0.4, possg_use=1.0, ord_use=0.8)}
for name, kw in cfgs.items():
    for T in (2400, 4800):
        print(name, T, "min occurrences:", count_min(T, **kw), flush=True)
        for lt in (4, 6, 9):
            c, a, res = R.sweep("covpre", lt, T=T, **kw)
            print("   lt=%d %s ALL %d/15 minfire med %.2f maxsil med %.2f" % (lt, c, a, np.median([b["min_fire"] for b in res]), np.median([b["max_silent"] for b in res])), flush=True)
