"""Fix-round step-0a record (scratch): every arm through the SAME tuning procedure at the preregistered schedule
(base world, 4800 utterances), the nulls on the GO candidate, the per-seed pass rate on 60 seeds, the dose curves and
convergence. Writes final_v3.out (stdout) and final_v3.json."""
import replica3 as R, numpy as np, json
from math import comb
T = 4800; GRID = (1, 2, 3, 4, 6, 9, 12, 18); out = {"T": T, "grid": GRID, "arms": {}}
def binom_ge(q, k, n=15): return sum(comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(k, n + 1))
print("== 1. every arm, same grid, T=%d, dev seeds 1-15 (counts = seeds passing)" % T, flush=True)
centre = {}
for rule in ("cov", "covpre", "hebb", "hebbwta", "bcm"):
    best = None; out["arms"][rule] = {}
    for lt in GRID:
        c, allp, res = R.sweep(rule, lt, T=T)
        out["arms"][rule][lt] = dict(c, ALL=allp, minfire=float(np.median([b["min_fire"] for b in res])), maxsil=float(np.median([b["max_silent"] for b in res])))
        print("%-7s lt=%2d %s ALL(null bars) %2d ALL(null-free) %2d | min fire med %.2f, max silent(W) med %.2f, max leak(H) med %.2f" % (
            rule, lt, {k: c[k] for k in ["S0_fire", "S0_silent", "S1", "S2a", "S2b", "S2b_classmax", "S3", "S4", "S5"]}, allp, c["ALL_nf"],
            np.median([b["min_fire"] for b in res]), np.median([b["max_silent"] for b in res]), np.median([b["max_leakH"] for b in res])), flush=True)
        key = (c["ALL_nf"], allp)
        if best is None or key > best[1]: best = (lt, key)
    centre[rule] = best[0]
    print("  => %s grid centre lt=%d (ALL null-free %d, with nulls %d)" % (rule, best[0], best[1][0], best[1][1]), flush=True)
out["centre"] = centre
print("== 2. per-seed pass rate, dev seeds 1-60, each arm at its own grid centre", flush=True)
out["rate60"] = {}
for rule in ("covpre", "cov", "bcm", "hebb", "hebbwta"):
    res = [R.run_seed(s, rule, centre[rule], T=T) for s in range(1, 61)]
    q = sum(all(b[k] for k in R.KEYS_NF) for b in res) / 60; qn = sum(all(b[k] for k in R.KEYS) for b in res) / 60
    per = {k: round(sum(b[k] for b in res) / 60, 3) for k in R.KEYS_NF}
    out["rate60"][rule] = dict(q=q, q_with_nulls=qn, per=per, p_ge14=binom_ge(q, 14), p_ge13=binom_ge(q, 13))
    print("  %-7s lt=%d q(ALL null-free)=%.3f q(with nulls)=%.3f per-bar %s P(>=14/15)=%.4f P(>=13/15)=%.4f" % (
        rule, centre[rule], q, qn, per, binom_ge(q, 14), binom_ge(q, 13)), flush=True)
lt = centre["covpre"]
print("== 3. nulls on the GO candidate (covpre lt=%d): each must FAIL the GO bars" % lt, flush=True)
out["nulls"] = {}
for null in ("label", "scene", "k0"):
    res = [R.run_seed(s, "covpre", lt, T=T, null=null) for s in range(1, 16)]
    c = {k: int(sum(b[k] for b in res)) for k in R.KEYS_NF}; a = int(sum(all(b[k] for k in R.KEYS_NF) for b in res))
    out["nulls"][null] = dict(c, ALL=a); print("  %-6s %s ALL %d/15" % (null, c, a), flush=True)
res = [R.run_seed(s, "covpre", lt, T=T, voice_perm=True) for s in range(1, 16)]
c = {k: int(sum(b[k] for b in res)) for k in R.KEYS_NF}; a = int(sum(all(b[k] for k in R.KEYS_NF) for b in res))
out["nulls"]["voice_perm"] = dict(c, ALL=a); print("  voice  %s ALL %d/15" % (c, a), flush=True)
print("== 4. dose curves (covpre lt=%d), medians over 15 seeds" % lt, flush=True)
out["dose"] = {}
for f_self in (1.0, 0.75, 0.5, 0.4, 0.25):
    res = [R.run_seed(s, "covpre", lt, T=T, f_self=f_self) for s in range(1, 16)]
    v = float(np.median([b["leakA"] for b in res])); a = int(sum(all(b[k] for k in R.KEYS_NF) for b in res))
    out["dose"]["overheard_%.2f" % (1 - f_self)] = dict(leak_SHA_to_SELF=v, ALL=a)
    print("  overheard %.2f: SHA->E_SELF %.3f (theta_silent 0.20) ALL %d/15" % (1 - f_self, v, a), flush=True)
for nv in (1, 2, 3, 5):
    vals = []
    for s in range(1, 16):
        rows = R.world(s, T, n_voices=nv, v5_speaks=True); W = R.train(rows, "covpre", lt / 400, 400)[T]
        vals.append(float(W[R.ti["SHS"], R.NPOOL:].max()))
    out["dose"]["voices_%d" % nv] = float(np.median(vals)); print("  speaking voices %d: SHS->best E %.3f" % (nv, np.median(vals)), flush=True)
for p in (0.0, 0.3, 0.6, 0.9):
    res = [R.run_seed(s, "covpre", lt, T=T, turn_taking=p) for s in range(1, 16)]
    v = float(np.median([b["prev_adr"] for b in res])); out["dose"]["turn_taking_%.1f" % p] = v
    print("  turn-taking p=%.1f: PREV->ADR / SHA->ADR %.3f" % (p, v), flush=True)
print("== 5. convergence (covpre lt=%d): weights at 2/3 vs full exposure" % lt, flush=True)
agree = []; mono = []
for s in range(1, 16):
    rows = R.world(s, T); ck = R.train(rows, "covpre", lt / 400, 400, ckpts=(2 * T // 3,))
    a, b = ck[2 * T // 3], ck[T]
    for t, ks in R.TARGET.items():
        for k in ks:
            w0, w1 = a[R.ti[t], R.pi[k]], b[R.ti[t], R.pi[k]]
            agree.append(abs(w1 - w0) / max(w1, 1e-9) <= 0.2); mono.append(w1 >= w0 - 0.05)
out["convergence"] = dict(within20=float(np.mean(agree)), monotone=float(np.mean(mono)))
print("  pointer targets within 20%%: %.3f of (token, pool, seed); non-decreasing (tol 0.05): %.3f" % (np.mean(agree), np.mean(mono)), flush=True)
json.dump(out, open("final_v3.json", "w"), indent=1, default=float)
