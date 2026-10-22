"""Judge's rate replica v2 (scratch only; not spiking). Changes from v1: decoys are per-person coin bits
(independent of the event by construction), PREV marks the previous speaker whether or not they are now the
addressee (the forced-fit test), 5 label-shuffled nulls per seed give a null-relative threshold on the class MAX,
tau=400 utterances with 2400 utterances of exposure (6 tau), S6 compares 1600 vs 2400."""
import numpy as np, sys
VOICES = ["V1", "V2", "V3", "V4", "V5"]; PERS = VOICES + ["B1", "SELF"]; OBJ = ["O1", "O2", "O3"]
ENT = PERS + OBJ; POOLS = ["SPK", "ADR", "LEFT", "NEWEST"]
TOK = (["SHS", "SHA", "POSS_S", "POSS_A", "PLUR", "OOI1", "PREV", "CN", "IDIO", "FILL"]
       + ["N_" + p for p in PERS] + ["T_" + o for o in OBJ] + ["P%d" % i for i in range(1, 6)])
ti = {t: i for i, t in enumerate(TOK)}; POST = POOLS + ENT; pi = {p: i for i, p in enumerate(POST)}
PRES = 0.2
NAMES = ["N_" + p for p in PERS] + ["CN"]
NONREF = ["FILL"] + ["T_" + o for o in OBJ] + ["P%d" % i for i in range(1, 6)]

def world(seed, T, f_self=0.4, n_voices=5, turn_taking=0.0, v5_speaks=False, prev_use=0.5):
    rng = np.random.default_rng(seed); rows = []; prev = None; disc = 0; present = None
    voices = VOICES[:n_voices]
    for _ in range(T):
        if disc == 0:
            k = min(3, len(voices)); present = list(rng.choice(voices, k, replace=False)) + ["B1", "SELF"]; disc = 3; prev = None
        disc -= 1
        spk_pool = [p for p in present if p in voices and (v5_speaks or p != "V5")] or [p for p in present if p in voices]
        S = spk_pool[rng.integers(len(spk_pool))]
        others = [p for p in present if p != S and p != "SELF" and p in voices]
        if rng.random() < f_self or not others: A = "SELF"
        elif prev is not None and prev != S and prev in others and rng.random() < turn_taking: A = prev
        else: A = others[rng.integers(len(others))]
        agent = present[rng.integers(len(present))]
        joint = rng.random() < 0.05 and A != "SELF"
        pat_person = None; obj = None; owner = None
        if rng.random() < 0.2:
            cands = [p for p in present if p != agent]; pat_person = cands[rng.integers(len(cands))]
        else:
            obj = OBJ[rng.integers(3)]
            if rng.random() < 0.35: owner = present[rng.integers(len(present))]
        left = {p: rng.random() < 0.5 for p in present}; newest = {p: rng.random() < 0.5 for p in present}
        def name(p):
            if p == "V3": return "CN" if rng.random() < 0.5 else "N_V3"
            if p == "V2" and p == S and rng.random() < 0.5: return "IDIO"
            return "N_" + p
        def refer(p):
            if prev is not None and p == prev and p != S and rng.random() < prev_use: return "PREV"
            if p == S: return "SHS" if rng.random() < 0.85 else name(p)
            if p == A: return "SHA" if rng.random() < 0.85 else name(p)
            if rng.random() < 0.10: return "OOI1"
            return name(p)
        toks = []
        if joint: toks.append("PLUR"); parts = {S, A}
        else: toks.append(refer(agent)); parts = {agent}
        toks.append("P%d" % (1 + rng.integers(5)))
        if pat_person: toks.append(refer(pat_person)); parts.add(pat_person)
        else:
            if owner is not None:
                toks.append("POSS_S" if owner == S else "POSS_A" if owner == A else name(owner)); parts.add(owner)
            toks.append("T_" + obj); parts.add(obj)
        if rng.random() < 0.5: toks.append("FILL")
        x = np.zeros(len(TOK))
        for t in toks: x[ti[t]] = 1
        y = np.zeros(len(POST))
        y[pi["SPK"]] = S in parts; y[pi["ADR"]] = A in parts
        y[pi["LEFT"]] = any(left[p] for p in parts if p in PERS); y[pi["NEWEST"]] = any(newest[p] for p in parts if p in PERS)
        for p in present: y[pi[p]] = PRES
        for p in parts: y[pi[p]] = 1.0
        rows.append((x, y, dict(S=S, A=A, prev=prev, parts=parts)))
        prev = S
    return rows

def train(rows, rule, lam, tau, alpha=0.005, warm=200, ckpts=()):
    W = np.zeros((len(TOK), len(POST))); ybar = np.mean([r[1] for r in rows[:warm]], axis=0); th = ybar ** 2; out = {}
    for t, (x, y, _) in enumerate(rows, 1):
        if rule == "cov": dw = lam * np.outer(x, y - ybar)
        elif rule == "hebb": dw = lam * np.outer(x, y)
        elif rule == "bcm": dw = lam * np.outer(x, y * (y - th))
        W += dw; W *= (1 - 1.0 / tau); np.clip(W, 0.0, 1.0, out=W)
        ybar = (1 - alpha) * ybar + alpha * y; th = (1 - alpha) * th + alpha * y * y
        if t in ckpts: out[t] = W.copy()
    out[len(rows)] = W.copy(); return out

def shuffle_labels(rows, seed):
    rng = np.random.default_rng(seed); out = []
    for x, y, m in rows:
        y2 = y.copy(); y2[:4] = y[rng.permutation(4)]; out.append((x, y2, m))
    return out

def null_thresholds(rows, rule, lam, tau, seed, n=5):
    mx_nonref, mx_decoy, mx_name_pool = [], [], []
    for i in range(n):
        Wn = train(shuffle_labels(rows, 10000 * seed + i), rule, lam, tau)[len(rows)]
        mx_nonref.append(max(Wn[ti[t], :4].max() for t in NONREF))
        mx_decoy.append(max(Wn[ti[t], 2:4].max() for t in TOK))
        mx_name_pool.append(max(Wn[ti[t], :4].max() for t in NAMES))
    return dict(nonref=float(np.percentile(mx_nonref, 95)), decoy=float(np.percentile(mx_decoy, 95)),
                name_pool=float(np.percentile(mx_name_pool, 95)))

def bars(W, nt):
    r = lambda t: W[ti[t]]
    def pm(t, k):
        row = r(t)[:4]; return row[pi[k]], max(np.delete(row, pi[k]))
    tS, nS = pm("SHS", "SPK"); tA, nA = pm("SHA", "ADR"); tPS, nPS = pm("POSS_S", "SPK"); tPA, nPA = pm("POSS_A", "ADR")
    small = min(tS, tA)
    S1 = tS >= 2 * nS and tA >= 2 * nA and tPS >= 2 * nPS and tPA >= 2 * nPA
    S2a = tS >= 2 * r("SHS")[4:].max()
    leakA = r("SHA")[4 + ENT.index("SELF")] / tA if tA > 0 else np.nan
    S2b = True; name_detail = {}
    for n in NAMES:
        bearer = "V3" if n == "CN" else n[2:]
        row = r(n); ew = row[4:]; b = ew[ENT.index(bearer)]
        ok = b >= 2 * max(row[:4].max(), nt["name_pool"]) and b >= 2 * np.delete(ew, ENT.index(bearer)).max()
        name_detail[n] = (round(float(b), 3), round(float(row[:4].max()), 3), round(float(np.delete(ew, ENT.index(bearer)).max()), 3), ok)
        S2b &= ok
    S3 = all(r(t)[:4].max() <= max(nt["nonref"], 0.25 * small) for t in NONREF)
    S4 = all(r(t)[2:4].max() <= max(nt["decoy"], 0.2 * small) for t in TOK) and \
         all(POOLS[int(r(t)[:4].argmax())] not in ("LEFT", "NEWEST") for t in TOK if r(t)[:4].max() >= 0.5 * small)
    S5 = r("OOI1")[:4].max() <= max(nt["nonref"], 0.25 * small) and all(r("PREV")[k] <= 0.2 * tA for k in range(4))
    return dict(S1=S1, S2a=S2a, S2b=S2b, S3=S3, S4=S4, S5=S5, tS=tS, tA=tA, leakA=leakA, nt=nt,
                prev_adr=r("PREV")[pi["ADR"]] / tA if tA else np.nan, names=name_detail,
                idio=(r("IDIO")[pi["SPK"]], r("IDIO")[4 + ENT.index("V2")]), plur=(r("PLUR")[pi["SPK"]], r("PLUR")[pi["ADR"]]))

KEYS = ["S1", "S2a", "S2b", "S3", "S4", "S5"]
def sweep(rule, lamtau, tau, T=2400, seeds=range(1, 16), **kw):
    lam = lamtau / tau; res = []
    for s in seeds:
        rows = world(s, T, **kw); W = train(rows, rule, lam, tau)[T]; nt = null_thresholds(rows, rule, lam, tau, s)
        res.append(bars(W, nt))
    return {k: int(sum(b[k] for b in res)) for k in KEYS}, int(sum(all(b[k] for k in KEYS) for b in res)), res

if __name__ == "__main__":
    TAU = 400
    print("== arms, 2400 utterances, tau=400, 15 seeds; counts = seeds passing; nulls = 5 label shuffles per seed")
    for rule in ("cov", "hebb", "bcm"):
        for lt in (4, 6, 9):
            c, allp, res = sweep(rule, lt, TAU)
            print(" %-4s lam*tau=%d %s ALL %2d/15 med tS %.2f tA %.2f nullmax(nonref) %.3f leakA %.2f" % (
                rule, lt, c, allp, np.median([b["tS"] for b in res]), np.median([b["tA"] for b in res]),
                np.median([b["nt"]["nonref"] for b in res]), np.median([b["leakA"] for b in res])))
    c, allp, res = sweep("cov", 6, TAU)
    bad = [b for b in res if not b["S2b"]]
    print("== cov 6: per-name detail (bearer w, max pool w, max other-E w, ok) on the first failing seed, if any")
    if bad: print("  ", bad[0]["names"])
    else: print("   all 15 seeds pass S2b")
    print("== S6 asymptote: cov 6, margins at 1600 (4 tau) vs 2400 (6 tau)")
    agree = 0
    for s in range(1, 16):
        rows = world(s, 2400); ck = train(rows, "cov", 6 / TAU, TAU, ckpts=(1600,))
        a, b = ck[1600][ti["SHS"], pi["SPK"]], ck[2400][ti["SHS"], pi["SPK"]]; agree += abs(a - b) / max(b, 1e-9) <= 0.2
    print("   SHS->SPK agrees within 20%% on %d/15 seeds" % agree)
    print("== nulls (cov 6): label-shuffled stream scored against fresh nulls; K=0 is all zeros")
    cnt = 0
    for s in range(1, 16):
        rows = shuffle_labels(world(s, 2400), 777 + s); W = train(rows, "cov", 6 / TAU, TAU)[2400]; nt = null_thresholds(rows, "cov", 6 / TAU, TAU, s)
        b = bars(W, nt); cnt += b["S1"] and b["S2a"]
    print("   passes S1+S2a on %d/15" % cnt)
    print("== dose (cov 6): overheard fraction -> SHA->SELF leak / pool")
    for f_self in (1.0, 0.75, 0.5, 0.4, 0.25):
        c, allp, res = sweep("cov", 6, TAU, f_self=f_self); print("   overheard %.2f: leakA med %.2f ALL %d/15" % (1 - f_self, np.median([b["leakA"] for b in res]), allp))
    print("== dose (cov 6): speaking voices -> SHS entity leak")
    for nv in (1, 2, 3, 5):
        res = []
        for s in range(1, 16):
            W = train(world(s, 2400, n_voices=nv, v5_speaks=True), "cov", 6 / TAU, TAU)[2400]; res.append(W[ti["SHS"], 4:].max() / max(W[ti["SHS"], pi["SPK"]], 1e-9))
        print("   n_voices=%d: leak med %.2f" % (nv, np.median(res)))
    print("== forced-fit arm (cov 6): PREV marks the previous speaker; addressee = previous speaker with p")
    for p in (0.0, 0.3, 0.6, 0.9):
        c, allp, res = sweep("cov", 6, TAU, turn_taking=p)
        print("   p=%.1f: PREV->ADR / SHA->ADR med %.2f  S5 %d/15" % (p, np.median([b["prev_adr"] for b in res]), c["S5"]))
    c, allp, res = sweep("cov", 6, TAU)
    print("== IDIO (SPK, E_V2) med (%.3f, %.3f); PLUR (SPK, ADR) med (%.2f, %.2f)" % (
        np.median([b["idio"][0] for b in res]), np.median([b["idio"][1] for b in res]), np.median([b["plur"][0] for b in res]), np.median([b["plur"][1] for b in res])))
