"""Fix-round rate replica v3 (scratch only; not spiking). Built from scratch/judge/cov_decay_replica2.py (copied
here unchanged as cov_decay_replica2.py). Changes, each answering a final-verification finding:
  * GIVEN relation (pool GIV) and its tokens: ANA (refers to an entity named in the previous utterance of the
    discourse), DEM (refers to an entity the speaker points at; also a determiner before a pointed object), POSS_G
    (possessor is given), ORD1/ORD2 (two entities named in the previous utterance; first / second named). The
    toy renders the givenness afferent (world/teacher, declared); production renders none.
  * PLUR dose is a parameter (joint speaker+addressee agent probability), default 0.10.
  * Bars use the SAME token's label-shuffled null (dose-matched): null95[t] = 95th percentile of token t's pool
    weights over 5 shuffled runs x 5 pool columns (25 samples). The class-max null of v2 is kept as a reported
    variant for S2b only.
  * S0, the absolute route bar from the spiking transfer curve (scratch/fix/wscale_curve.out, NT=12 assemblies,
    W_MAX=20): a learned route must reach theta_fire = 8/20 = 0.40 of W_MAX (every target neuron fires in a 40-ms
    token on 10/10 seeds), every route that must not act must stay <= theta_silent = 4/20 = 0.20 (<= 1 target
    spike on 10/10 seeds). K = 0 (all weights 0) now FAILS S0, as it must.
  * Rules: cov (uniform decay leak, the judged design), covpre (the same covariance term with a PRESYNAPTICALLY
    gated, weight-proportional leak: dw = lam*x*(y-ybar) - kap*x*w, the outstar fixed point w* = (E[y|x]-ybar)/kap,
    dose-independent), hebb (rate-window / causal potentiation + decay), hebbwta (hebb with winner-take-all among the
    active pools, the CAUSAL+WTA arm), bcm (BCM + decay).
  * Nulls: label-shuffled, scene-shuffled (x and y from different utterances), voice-permuted (the SPK/ADR afferents
    follow a random present voice), K = 0.
"""
import numpy as np, sys, json
VOICES = ["V1", "V2", "V3", "V4", "V5"]; PERS = VOICES + ["B1", "SELF"]; OBJ = ["O1", "O2", "O3"]
ENT = PERS + OBJ; POOLS = ["SPK", "ADR", "GIV", "LEFT", "NEWEST"]; NPOOL = len(POOLS); DEC = [3, 4]
GIVTOK = ["ANA", "DEM", "POSS_G", "ORD1", "ORD2"]
TOK = (["SHS", "SHA", "POSS_S", "POSS_A", "PLUR", "OOI1", "PREV", "CN", "IDIO", "FILL"] + GIVTOK
       + ["N_" + p for p in PERS] + ["T_" + o for o in OBJ] + ["P%d" % i for i in range(1, 6)])
ti = {t: i for i, t in enumerate(TOK)}; POST = POOLS + ENT; pi = {p: i for i, p in enumerate(POST)}
PRES = 0.2
NAMES = ["N_" + p for p in PERS] + ["CN"]
NONREF = ["FILL"] + ["T_" + o for o in OBJ] + ["P%d" % i for i in range(1, 6)]
TARGET = {"SHS": ["SPK"], "SHA": ["ADR"], "POSS_S": ["SPK"], "POSS_A": ["ADR"], "PLUR": ["SPK", "ADR"],
          "ANA": ["GIV"], "DEM": ["GIV"], "POSS_G": ["GIV"], "ORD1": ["GIV"], "ORD2": ["GIV"]}
TH_FIRE, TH_SILENT = 0.40, 0.20

def world(seed, T, f_self=0.4, n_voices=5, turn_taking=0.0, v5_speaks=False, prev_use=0.5, joint_p=0.10,
          p_point=0.30, ana_use=0.7, ord_use=0.6, dem_use=0.8, possg_use=0.8, voice_perm=False):
    rng = np.random.default_rng(seed); rows = []; prev = None; disc = 0; present = None; prev_named = []
    voices = VOICES[:n_voices]
    for _ in range(T):
        if disc == 0:
            k = min(3, len(voices)); present = list(rng.choice(voices, k, replace=False)) + ["B1", "SELF"]
            disc = 3; prev = None; prev_named = []
        disc -= 1
        spk_pool = [p for p in present if p in voices and (v5_speaks or p != "V5")] or [p for p in present if p in voices]
        S = spk_pool[rng.integers(len(spk_pool))]
        others = [p for p in present if p != S and p != "SELF" and p in voices]
        if rng.random() < f_self or not others: A = "SELF"
        elif prev is not None and prev != S and prev in others and rng.random() < turn_taking: A = prev
        else: A = others[rng.integers(len(others))]
        pointed = None
        if rng.random() < p_point:
            cands = [e for e in present + OBJ if e not in (S, A)]; pointed = cands[rng.integers(len(cands))]
        given = set(e for e in prev_named if e not in (S, A)) | ({pointed} if pointed else set())
        agent = present[rng.integers(len(present))]
        joint = rng.random() < joint_p and A != "SELF"
        pat_person = None; obj = None; owner = None
        if rng.random() < 0.2:
            cands = [p for p in present if p != agent]; pat_person = cands[rng.integers(len(cands))]
        else:
            obj = OBJ[rng.integers(3)]
            if rng.random() < 0.35: owner = present[rng.integers(len(present))]
        left = {p: rng.random() < 0.5 for p in present}; newest = {p: rng.random() < 0.5 for p in present}
        named_now = []
        def name(p):
            named_now.append(p)
            if p == "V3": return "CN" if rng.random() < 0.5 else "N_V3"
            if p == "V2" and p == S and rng.random() < 0.5: return "IDIO"
            return "N_" + p
        def refer(p):
            if prev is not None and p == prev and p != S and rng.random() < prev_use: return "PREV"
            if p == S: return "SHS" if rng.random() < 0.85 else name(p)
            if p == A: return "SHA" if rng.random() < 0.85 else name(p)
            if p in given:
                if p == pointed and rng.random() < dem_use: return "DEM"
                if p in prev_named:
                    if len(prev_named) >= 2 and p in prev_named[:2] and rng.random() < ord_use:
                        return "ORD1" if p == prev_named[0] else "ORD2"
                    if rng.random() < ana_use: return "ANA"
            if rng.random() < 0.10: return "OOI1"
            return name(p)
        toks = []
        if joint: toks.append("PLUR"); parts = {S, A}
        else: toks.append(refer(agent)); parts = {agent}
        toks.append("P%d" % (1 + rng.integers(5)))
        if pat_person: toks.append(refer(pat_person)); parts.add(pat_person)
        else:
            if owner is not None:
                if owner == S: toks.append("POSS_S")
                elif owner == A: toks.append("POSS_A")
                elif owner in given and rng.random() < possg_use: toks.append("POSS_G")
                else: toks.append(name(owner))
                parts.add(owner)
            if obj in given and obj == pointed and rng.random() < dem_use: toks += ["DEM", "T_" + obj]; named_now.append(obj)
            elif obj in given and obj in prev_named and len(prev_named) >= 2 and obj in prev_named[:2] and rng.random() < ord_use:
                toks.append("ORD1" if obj == prev_named[0] else "ORD2")
            elif obj in given and obj in prev_named and rng.random() < ana_use: toks.append("ANA")
            else: toks.append("T_" + obj); named_now.append(obj)
            parts.add(obj)
        if rng.random() < 0.5: toks.append("FILL")
        x = np.zeros(len(TOK))
        for t in toks: x[ti[t]] = 1
        Sh, Ah = S, A
        if voice_perm:
            vv = [p for p in present if p in voices]; Sh = vv[rng.integers(len(vv))]
            aa = [p for p in present if p != Sh]; Ah = aa[rng.integers(len(aa))]
        y = np.zeros(len(POST))
        y[pi["SPK"]] = Sh in parts; y[pi["ADR"]] = Ah in parts; y[pi["GIV"]] = any(e in given for e in parts)
        y[pi["LEFT"]] = any(left[p] for p in parts if p in PERS); y[pi["NEWEST"]] = any(newest[p] for p in parts if p in PERS)
        for p in present: y[pi[p]] = PRES
        for p in parts: y[pi[p]] = 1.0
        rows.append((x, y, dict(S=S, A=A, prev=prev, parts=parts, given=given)))
        prev = S; prev_named = [e for e in dict.fromkeys(named_now)]
    return rows

def train(rows, rule, lam, tau, alpha=0.005, warm=200, ckpts=(), kap=1.0):
    W = np.zeros((len(TOK), len(POST))); ybar = np.mean([r[1] for r in rows[:warm]], axis=0); th = ybar ** 2; out = {}
    for t, (x, y, _) in enumerate(rows, 1):
        if rule == "cov": W += lam * np.outer(x, y - ybar); W *= (1 - 1.0 / tau)
        elif rule == "covpre": W += lam * (np.outer(x, y - ybar) - kap * x[:, None] * W)
        elif rule == "hebb": W += lam * np.outer(x, y); W *= (1 - 1.0 / tau)
        elif rule == "hebbwta":
            yy = y.copy(); act = [k for k in range(NPOOL) if y[k] > 0]
            if len(act) > 1:
                drive = [y[k] + float(x @ W[:, k]) + 1e-6 * k for k in act]; keep = act[int(np.argmax(drive))]
                for k in act:
                    if k != keep: yy[k] = 0.0
            W += lam * np.outer(x, yy); W *= (1 - 1.0 / tau)
        elif rule == "bcm": W += lam * np.outer(x, y * (y - th)); W *= (1 - 1.0 / tau)
        np.clip(W, 0.0, 1.0, out=W)
        ybar = (1 - alpha) * ybar + alpha * y; th = (1 - alpha) * th + alpha * y * y
        if t in ckpts: out[t] = W.copy()
    out[len(rows)] = W.copy(); return out

def shuffle_labels(rows, seed):
    rng = np.random.default_rng(seed); out = []
    for x, y, m in rows:
        y2 = y.copy(); y2[:NPOOL] = y[rng.permutation(NPOOL)]; out.append((x, y2, m))
    return out

def shuffle_scene(rows, seed):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(rows))
    return [(rows[i][0], rows[j][1], rows[i][2]) for i, j in enumerate(perm)]

def nulls(rows, rule, lam, tau, seed, n=5, **kw):
    """Per-token null95 over the token's pool weights in n label-shuffled runs (n x NPOOL samples), and the v2 class
    maxima (kept for the reported S2b variant)."""
    samp = {t: [] for t in TOK}; cls_name = []
    for i in range(n):
        Wn = train(shuffle_labels(rows, 10000 * seed + i), rule, lam, tau, **kw)[len(rows)]
        for t in TOK: samp[t].extend(list(Wn[ti[t], :NPOOL]))
        cls_name.append(max(Wn[ti[t], :NPOOL].max() for t in NAMES))
    return {t: float(np.percentile(v, 95)) for t, v in samp.items()}, float(np.percentile(cls_name, 95))

def bars(W, n95, cls95):
    r = lambda t: W[ti[t]]
    S1 = True; tg = {}
    for t, ks in TARGET.items():
        row = r(t)[:NPOOL]
        for k in ks:
            others = [row[j] for j in range(NPOOL) if POOLS[j] not in ks]
            ok = row[pi[k]] >= 2 * max(others); S1 &= ok; tg[(t, k)] = float(row[pi[k]])
    small = min(tg[("SHS", "SPK")], tg[("SHA", "ADR")])
    S2a = all(max(r(t)[pi[k]] for k in TARGET[t]) >= 2 * r(t)[NPOOL:].max() for t in ["SHS", "SHA", "ANA", "DEM", "ORD1", "ORD2"])
    S2b = True; S2b_cls = True; names = {}
    for n in NAMES:
        bearer = "V3" if n == "CN" else n[2:]
        row = r(n); ew = row[NPOOL:]; b = ew[ENT.index(bearer)]; oth = np.delete(ew, ENT.index(bearer)).max()
        ok = b >= 2 * max(row[:NPOOL].max(), n95[n]) and b >= 2 * oth
        okc = b >= 2 * max(row[:NPOOL].max(), cls95) and b >= 2 * oth
        names[n] = (round(float(b), 3), ok); S2b &= ok; S2b_cls &= okc
    S3 = all(r(t)[:NPOOL].max() <= max(2 * n95[t], 0.25 * small) for t in NONREF)
    S4 = all(r(t)[DEC].max() <= max(2 * n95[t], 0.2 * small) for t in TOK) and \
         all(int(r(t)[:NPOOL].argmax()) not in DEC for t in TOK if r(t)[:NPOOL].max() >= 0.5 * small)
    # S5: the out-of-inventory marker as S3; the previous-speaker marker's forced fit onto the ADDRESSEE pool stays
    # <= 0.2x the addressee pointer in the decorrelated world. PREV -> GIV is a genuine overlap (the previous speaker
    # was just heard, so is discourse-given) and is REPORTED, not barred.
    S5 = r("OOI1")[:NPOOL].max() <= max(2 * n95["OOI1"], 0.25 * small) and r("PREV")[pi["ADR"]] <= 0.2 * tg[("SHA", "ADR")]
    # S0: absolute routes from the engine transfer curve. MUST FIRE (>= TH_FIRE): every pointer-class target, the
    # self-name's E route, every person name's bearer route. CLASS W, MUST STAY SILENT (<= TH_SILENT): a route whose
    # firing could produce a single WRONG latch -- every pool route of a non-referential token, a name, OOI1, PREV
    # (except PREV->GIV and PREV->ADR, above; IDIO, the idiolect forced-fit marker, is barred by T9 and reported), every non-target pool route of a single-target pointer form, every
    # entity route of FILL/predicates, every non-bearer entity route of a name. CLASS H, REPORTED: routes of tokens
    # whose own pool already fires (pointer forms' entity routes; PLUR's non-target routes): firing adds a second
    # latch, which the hold rule reads as HOLD (fail-closed); their cost is counted by the organ-gate recall bars.
    fire = [tg[k] for k in tg] + [r("N_SELF")[NPOOL + ENT.index("SELF")]] + [names[n][0] for n in NAMES]
    silent = []; leakH = []
    for t in TOK:
        row = r(t)
        for j in range(NPOOL):
            if t in TARGET and POOLS[j] in TARGET[t]: continue
            if t == "PREV" and POOLS[j] in ("GIV", "ADR"): continue
            if t == "IDIO": continue          # forced-fit idiolect marker: barred by T9 (held or V2), reported
            (leakH if t == "PLUR" else silent).append(row[j])
    for t in ["FILL"] + ["P%d" % i for i in range(1, 6)]: silent.append(r(t)[NPOOL:].max())
    for n in NAMES:
        bearer = "V3" if n == "CN" else n[2:]; ew = r(n)[NPOOL:]; silent.append(np.delete(ew, ENT.index(bearer)).max())
    for t in TARGET: leakH.append(r(t)[NPOOL:].max())
    S0_fire = min(fire) >= TH_FIRE; S0_silent = max(silent) <= TH_SILENT
    S0 = S0_fire and S0_silent
    # NULL-FREE variants (the GO-rule form for the spiking census): within-token contrasts; the noise allowance that the
    # label-shuffled null gave S3-S5 is replaced by the PHYSICS bound theta_silent (a route at or below it cannot fire
    # its target), so no label-shuffled run is needed to set any threshold. PLUR is a hold class (co-commit).
    S2b_nf = all(names[n][0] >= 2 * r(n)[:NPOOL].max() and names[n][0] >= 2 * np.delete(r(n)[NPOOL:], ENT.index("V3" if n == "CN" else n[2:])).max() for n in NAMES)
    S3_nf = all(r(t)[:NPOOL].max() <= max(TH_SILENT, 0.25 * small) for t in NONREF)
    S4_nf = all(r(t)[DEC].max() <= max(TH_SILENT, 0.2 * small) for t in TOK if t != "PLUR") and \
            all(int(r(t)[:NPOOL].argmax()) not in DEC for t in TOK if r(t)[:NPOOL].max() >= 0.5 * small)
    S5_nf = r("OOI1")[:NPOOL].max() <= max(TH_SILENT, 0.25 * small) and r("PREV")[pi["ADR"]] <= 0.2 * tg[("SHA", "ADR")]
    return dict(S2b_nf=S2b_nf, S3_nf=S3_nf, S4_nf=S4_nf, S5_nf=S5_nf, S0=S0, S0_fire=S0_fire, S0_silent=S0_silent, S1=S1, S2a=S2a, S2b=S2b, S3=S3, S4=S4, S5=S5, S2b_classmax=S2b_cls, min_fire=float(min(fire)),
                max_silent=float(max(silent)), max_leakH=float(max(leakH)), S0_strict=bool(S0 and max(leakH) <= TH_SILENT), tg=tg, names=names, leakA=float(r("SHA")[NPOOL + ENT.index("SELF")]),
                prev_adr=float(r("PREV")[pi["ADR"]] / max(tg[("SHA", "ADR")], 1e-9)))

KEYS = ["S0", "S1", "S2a", "S2b", "S3", "S4", "S5"]
KEYS_NF = ["S0", "S1", "S2a", "S2b_nf", "S3_nf", "S4_nf", "S5_nf"]
def run_seed(s, rule, lt, tau=400, T=2400, null="none", **kw):
    lam = lt / tau
    rows = world(s, T, **{k: v for k, v in kw.items() if k != "kap"})
    if null == "label": rows = shuffle_labels(rows, 777 + s)
    elif null == "scene": rows = shuffle_scene(rows, 555 + s)
    tkw = {"kap": kw["kap"]} if "kap" in kw else {}
    W = train(rows, rule, lam, tau, **tkw)[T] if null != "k0" else np.zeros((len(TOK), len(POST)))
    n95, c95 = nulls(rows, rule, lam, tau, s, **tkw)
    return bars(W, n95, c95)

def sweep(rule, lt, seeds=range(1, 16), **kw):
    res = [run_seed(s, rule, lt, **kw) for s in seeds]
    c = {k: int(sum(b[k] for b in res)) for k in KEYS + ["S0_fire", "S0_silent", "S2b_classmax"] + KEYS_NF[3:]}
    c["ALL_nf"] = int(sum(all(b[k] for k in KEYS_NF) for b in res))
    return c, int(sum(all(b[k] for k in KEYS) for b in res)), res

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if mode == "grid":
        # the SAME tuning procedure for every arm: lam*tau over one grid; the arm's grid centre is its best ALL count
        GRID = (1, 2, 3, 4, 6, 9, 12, 18)
        for rule in ("cov", "covpre", "hebb", "hebbwta", "bcm"):
            best = None
            for lt in GRID:
                c, allp, res = sweep(rule, lt)
                print("%-7s lam*tau=%2d %s ALL %2d/15 | min fire med %.2f max silent med %.2f" % (
                    rule, lt, c, allp, np.median([b["min_fire"] for b in res]), np.median([b["max_silent"] for b in res])), flush=True)
                if best is None or allp > best[1]: best = (lt, allp)
            print("  => %s grid centre lam*tau=%d (ALL %d/15)" % (rule, best[0], best[1]), flush=True)
