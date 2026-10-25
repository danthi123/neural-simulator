import replica3 as R, numpy as np, sys
from collections import Counter
lt = float(sys.argv[1]) if len(sys.argv) > 1 else 9
cat = Counter(); fails = Counter()
for s in range(1, 16):
    rows = R.world(s, 2400); lam = lt / 400
    W = R.train(rows, "covpre", lam, 400)[2400]; n95, c95 = R.nulls(rows, "covpre", lam, 400, s); b = R.bars(W, n95, c95)
    for k in R.KEYS:
        if not b[k]: fails[k] += 1
    r = lambda t: W[R.ti[t]]
    for t in R.TOK:
        for j in range(R.NPOOL):
            if not (t in R.TARGET and R.POOLS[j] in R.TARGET[t]) and r(t)[j] > R.TH_SILENT: cat["pool:%s>%s" % (t, R.POOLS[j])] += 1
    for n in R.NAMES:
        bearer = "V3" if n == "CN" else n[2:]; e = r(n)[R.NPOOL:].copy(); e[R.ENT.index(bearer)] = -1
        if e.max() > R.TH_SILENT: cat["name>E:%s>%s" % (n, R.ENT[int(e.argmax())])] += 1
    for t in R.TARGET:
        e = r(t)[R.NPOOL:]
        if e.max() > R.TH_SILENT: cat["ptr>E:%s>%s" % (t, R.ENT[int(e.argmax())])] += 1
    if not b["S5"]: print("seed", s, "S5: OOI1 pools", np.round(r("OOI1")[:R.NPOOL], 2), "2xn95", round(2 * n95["OOI1"], 3), "PREV pools", np.round(r("PREV")[:R.NPOOL], 2), "tA", round(b["tg"][("SHA", "ADR")], 2))
    if not b["S4"]: print("seed", s, "S4: top decoy", max(((round(float(r(t)[d]), 3), t, R.POOLS[d]) for t in R.TOK for d in R.DEC)))
print("lt", lt, "bar failures:", dict(fails))
print("routes above theta_silent (seed counts):", dict(cat.most_common(20)))
