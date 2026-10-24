"""S6 as revised: (i) every scored pointer route non-decreasing (tol 0.05) between 2/3 and full exposure; (ii) routes of
tokens with >= 450 occurrences (>= 300 = 3/alpha by the 2/3 checkpoint) within 20% between the two. covpre, lt=6 (the graded centre), T=4800."""
import replica3 as R, numpy as np
T = 4800; lt = 6; ok_i = ok_ii = 0; n_ii = 0; worst = []
for s in range(1, 16):
    rows = R.world(s, T); occ = np.sum([x for x, y, m in rows], axis=0)
    ck = R.train(rows, "covpre", lt / 400, 400, ckpts=(2 * T // 3,)); a, b = ck[2 * T // 3], ck[T]
    si = sii = True
    for t, ks in R.TARGET.items():
        for k in ks:
            w0, w1 = a[R.ti[t], R.pi[k]], b[R.ti[t], R.pi[k]]
            si &= w1 >= w0 - 0.05
            if occ[R.ti[t]] >= 450:
                n_ii += 1; good = abs(w1 - w0) / max(w1, 1e-9) <= 0.2; sii &= good
                if not good: worst.append((s, t, k, int(occ[R.ti[t]]), round(float(w0), 3), round(float(w1), 3)))
    for n in R.NAMES:
        bearer = "V3" if n == "CN" else n[2:]; j = R.NPOOL + R.ENT.index(bearer)
        w0, w1 = a[R.ti[n], j], b[R.ti[n], j]; si &= w1 >= w0 - 0.05
        if occ[R.ti[n]] >= 450:
            n_ii += 1; good = abs(w1 - w0) / max(w1, 1e-9) <= 0.2; sii &= good
    ok_i += si; ok_ii += sii
print("S6(i) non-decreasing on %d/15 seeds; S6(ii) dosed routes within 20%% on %d/15 seeds (%d dosed route-seed pairs); failures %s" % (ok_i, ok_ii, n_ii, worst))
