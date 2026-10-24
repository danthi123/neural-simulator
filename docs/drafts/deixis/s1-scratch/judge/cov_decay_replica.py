"""Judge's rate replica (scratch only; not spiking) of the step-0 world + bars in round2/PACKET.md.
Rule under test: covariance + the engine's own gated uniform decay (the leak), per utterance:
    w += lam * x_j * (y_i - ybar_i);  w *= (1 - 1/tau);  clip [0, 1]
Fixed point w* = lam*tau*E[x (y - ybar)]  (graded in the covariance, not a sign).
Compared on the SAME streams: plain Hebbian + decay (w* ∝ E[x y]) and BCM-with-binary-y + decay.
Bars S1-S5 relative to a label-shuffled null (VOICE/ADDR/LEFT/NEWEST relabelled per utterance).
Prints per-(lam*tau) pass counts over 15 seeds, the dose curves, and a realistic turn-taking arm."""
import numpy as np, sys
VOICES = ["V1", "V2", "V3", "V4", "V5"]; PERS = VOICES + ["B1", "SELF"]; OBJ = ["O1", "O2", "O3"]
ENT = PERS + OBJ
POOLS = ["SPK", "ADR", "LEFT", "NEWEST"]
TOK = (["SHS", "SHA", "POSS_S", "POSS_A", "PLUR", "OOI1", "PREV", "CN", "IDIO", "FILL"]
       + ["N_" + p for p in PERS] + ["T_" + o for o in OBJ] + ["P%d" % i for i in range(1, 6)])
ti = {t: i for i, t in enumerate(TOK)}; POST = POOLS + ENT; pi = {p: i for i, p in enumerate(POST)}
PRES = 0.2   # presence rate of an uninvolved present entity

def world(seed, T, f_self=0.4, n_voices=5, turn_taking=0.0, v5_speaks=False):
    rng = np.random.default_rng(seed); rows = []; prev = None; disc = 0; present = None; entered = None
    voices = VOICES[:n_voices]
    for _ in range(T):
        if disc == 0:
            k = min(3, len(voices)); present = list(rng.choice(voices, k, replace=False)) + ["B1", "SELF"]
            present = [p for p in present if p in PERS]; entered = present[rng.integers(len(present))]
            disc = 3; prev = None
        disc -= 1
        spk_pool = [p for p in present if p in voices and (v5_speaks or p != "V5")]
        if not spk_pool: spk_pool = [p for p in present if p in voices]
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
        left = present[0]
        def name(p):
            if p == "V3": return "CN" if rng.random() < 0.5 else "N_V3"
            if p == "V2" and p == S and rng.random() < 0.5: return "IDIO"
            return "N_" + p
        def refer(p):
            if p == S: return "SHS" if rng.random() < 0.85 else name(p)
            if p == A: return "SHA" if rng.random() < 0.85 else name(p)
            if prev is not None and p == prev and rng.random() < 0.5: return "PREV"
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
        y[pi["SPK"]] = S in parts; y[pi["ADR"]] = A in parts; y[pi["LEFT"]] = left in parts; y[pi["NEWEST"]] = entered in parts
        for p in present: y[pi[p]] = PRES
        for p in parts: y[pi[p]] = 1.0
        rows.append((x, y, dict(S=S, A=A, prev=prev, parts=parts)))
        prev = S
    return rows

def train(rows, rule, lam, tau, alpha=0.01, warm=100):
    W = np.zeros((len(TOK), len(POST))); ybar = np.mean([r[1] for r in rows[:warm]], axis=0); th = ybar ** 2
    for x, y, _ in rows:
        if rule == "cov": dw = lam * np.outer(x, y - ybar)
        elif rule == "hebb": dw = lam * np.outer(x, y)
        elif rule == "bcm": dw = lam * np.outer(x, y * (y - th))
        W += dw; W *= (1 - 1.0 / tau); np.clip(W, 0.0, 1.0, out=W)
        ybar = (1 - alpha) * ybar + alpha * y; th = (1 - alpha) * th + alpha * y * y
    return W

def shuffle_labels(rows, seed):
    rng = np.random.default_rng(seed + 1000); out = []
    for x, y, m in rows:
        y2 = y.copy(); y2[:4] = y[rng.permutation(4)]; out.append((x, y2, m))
    return out

def bars(W, Wnull):
    null95 = float(np.percentile(Wnull[:, :4], 95))
    r = lambda t: W[ti[t]]
    def pool_margin(t, k):
        row = r(t)[:4]; tv = row[pi[k]]; nxt = max(np.delete(row, pi[k])); return tv, nxt
    tS, nS = pool_margin("SHS", "SPK"); tA, nA = pool_margin("SHA", "ADR")
    tPS, nPS = pool_margin("POSS_S", "SPK"); tPA, nPA = pool_margin("POSS_A", "ADR")
    S1 = tS >= 2 * nS and tA >= 2 * nA and tPS >= 2 * nPS and tPA >= 2 * nPA
    small = min(tS, tA)
    e = lambda t: r(t)[4:]
    S2a = tS >= 2 * e("SHS").max()
    leakA = e("SHA")[pi["SELF"] - 4] / tA if tA > 0 else np.nan     # the "you is the brain's name" leak, reported
    names = ["N_" + p for p in PERS] + ["CN"]
    S2b = True
    for n in names:
        bearer = n[2:] if n.startswith("N_") else "V3"
        row = r(n); ew = row[4:]; b = ew[ENT.index(bearer)]
        S2b &= b >= 2 * row[:4].max() and b >= 2 * np.delete(ew, ENT.index(bearer)).max()
    nonref = ["FILL"] + ["T_" + o for o in OBJ] + ["P%d" % i for i in range(1, 6)]
    S3 = all(r(t)[:4].max() <= max(null95, 0.25 * small) for t in nonref)
    S4 = True
    for t in TOK:
        row = r(t)[:4]
        if row.max() >= 0.5 * small and POOLS[int(row.argmax())] in ("LEFT", "NEWEST"): S4 = False
        if max(row[pi["LEFT"]], row[pi["NEWEST"]]) > max(null95, 0.2 * small): S4 = False
    S5 = r("OOI1")[:4].max() <= max(null95, 0.25 * small) and all(r("PREV")[k] <= 0.2 * tA for k in range(4))
    return dict(S1=S1, S2a=S2a, S2b=S2b, S3=S3, S4=S4, S5=S5, null95=null95, tS=tS, tA=tA, leakA=leakA,
                prev_adr=r("PREV")[pi["ADR"]] / tA if tA else np.nan, idio_spk=r("IDIO")[pi["SPK"]],
                idio_e=r("IDIO")[4 + ENT.index("V2")], plur=(r("PLUR")[pi["SPK"]], r("PLUR")[pi["ADR"]]))

def sweep(rule, lamtau, tau, T=600, seeds=range(1, 16), **kw):
    lam = lamtau / tau; res = []
    for s in seeds:
        rows = world(s, T, **kw); W = train(rows, rule, lam, tau); Wn = train(shuffle_labels(rows, s), rule, lam, tau)
        res.append(bars(W, Wn))
    keys = ["S1", "S2a", "S2b", "S3", "S4", "S5"]
    allpass = sum(all(b[k] for k in keys) for b in res)
    return {k: sum(b[k] for b in res) for k in keys}, allpass, res

if __name__ == "__main__":
    print("== main arms, 600 utterances, tau=150 (4 tau), 15 seeds; counts are seeds passing")
    for rule in ("cov", "hebb", "bcm"):
        for lt in (3, 6, 12):
            c, allp, res = sweep(rule, lt, 150)
            print(" %-4s lam*tau=%3d  %s  ALL %2d/15  med tS %.2f tA %.2f null95 %.3f leakA(SHA->SELF/pool) %.2f prev/adr %.2f" % (
                rule, lt, c, allp, np.median([b["tS"] for b in res]), np.median([b["tA"] for b in res]),
                np.median([b["null95"] for b in res]), np.median([b["leakA"] for b in res]), np.median([b["prev_adr"] for b in res])))
    print("== asymptote check (S6): cov lam*tau=6 at 600 vs 2400 utterances")
    for T in (600, 2400):
        c, allp, res = sweep("cov", 6, 150, T=T)
        print("  T=%d  %s ALL %d/15 med tS %.2f tA %.2f" % (T, c, allp, np.median([b["tS"] for b in res]), np.median([b["tA"] for b in res])))
    print("== nulls on cov lam*tau=6 (K=0 is trivially all-zero weights): voice-permuted stream scored as if intact")
    # voice permuted per utterance = the shuffle itself; score the shuffled run against a second shuffle
    cnt = 0
    for s in range(1, 16):
        rows = world(s, 600); Wn = train(shuffle_labels(rows, s), "cov", 6 / 150, 150); Wn2 = train(shuffle_labels(rows, s + 7), "cov", 6 / 150, 150)
        b = bars(Wn, Wn2); cnt += b["S1"] and b["S2a"]
    print("  shuffled stream passes S1+S2a on %d/15 seeds" % cnt)
    print("== dose: overheard fraction (1 - f_self) -> SHA->SELF leak / pool  (cov, lam*tau=6)")
    for f_self in (1.0, 0.75, 0.5, 0.4, 0.25):
        c, allp, res = sweep("cov", 6, 150, f_self=f_self)
        print("  f_self=%.2f (overheard %.2f): leakA med %.2f  S2a %d/15 ALL %d/15" % (f_self, 1 - f_self, np.median([b["leakA"] for b in res]), c["S2a"], allp))
    print("== dose: number of speaking voices -> SHS entity leak (max E / pool)")
    for nv in (1, 2, 3, 5):
        res = []
        for s in range(1, 16):
            rows = world(s, 600, n_voices=nv, v5_speaks=True); W = train(rows, "cov", 6 / 150, 150)
            res.append(W[ti["SHS"], 4:].max() / max(W[ti["SHS"], pi["SPK"]], 1e-9))
        print("  n_voices=%d: SHS leak med %.2f" % (nv, np.median(res)))
    print("== realistic turn-taking arm (addressee = previous speaker p=0.6): PREV marker vs ADR pool")
    c, allp, res = sweep("cov", 6, 150, turn_taking=0.6)
    print("  %s ALL %d/15  med PREV->ADR / SHA->ADR = %.2f (prediction: rises; a true ambiguity in that world)" % (c, allp, np.median([b["prev_adr"] for b in res])))
    print("== IDIO (V2's private self-name) and PLUR (speaker+addressee): cov lam*tau=6")
    c, allp, res = sweep("cov", 6, 150)
    print("  IDIO->SPK med %.2f  IDIO->E_V2 med %.2f  PLUR (SPK, ADR) med (%.2f, %.2f)" % (
        np.median([b["idio_spk"] for b in res]), np.median([b["idio_e"] for b in res]),
        np.median([b["plur"][0] for b in res]), np.median([b["plur"][1] for b in res])))
