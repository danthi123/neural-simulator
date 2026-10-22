"""Quick check (scratch): replace S2b's class-max null with the same name's own label-shuffled null (dose-matched)."""
import numpy as np, cov_decay_replica2 as R
TAU = 400
def pername_null(rows, rule, lam, tau, seed, n=5):
    samp = {t: [] for t in R.TOK}
    for i in range(n):
        Wn = R.train(R.shuffle_labels(rows, 10000 * seed + i), rule, lam, tau)[len(rows)]
        for t in R.TOK: samp[t].extend(list(Wn[R.ti[t], :4]))
    return {t: float(np.percentile(v, 95)) for t, v in samp.items()}
for rule in ("cov", "hebb", "bcm"):
  for lt in (4, 6, 9):
    ok = 0; ok_all = 0
    for s in range(1, 16):
        rows = R.world(s, 2400); W = R.train(rows, rule, lt / TAU, TAU)[2400]
        nt = R.null_thresholds(rows, rule, lt / TAU, TAU, s); b = R.bars(W, nt); pn = pername_null(rows, rule, lt / TAU, TAU, s)
        s2b = True
        for n in R.NAMES:
            bearer = "V3" if n == "CN" else n[2:]
            row = W[R.ti[n]]; ew = row[4:]; bw = ew[R.ENT.index(bearer)]
            s2b &= bw >= 2 * max(row[:4].max(), pn[n]) and bw >= 2 * np.delete(ew, R.ENT.index(bearer)).max()
        ok += s2b; ok_all += s2b and all(b[k] for k in ("S1", "S2a", "S3", "S4", "S5"))
    print(rule, lt, "S2b per-name null %d/15  ALL %d/15" % (ok, ok_all))
