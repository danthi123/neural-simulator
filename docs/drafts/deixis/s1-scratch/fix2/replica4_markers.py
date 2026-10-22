"""Round-3 scratch: (1) the toy world with the morphology markers the English fixture's rendering implies, and (2) the
step-0a record re-scored with PER-TARGET thresholds (the calibration finding). Imports replica3 (copied unchanged).

(1) Markers (nonce tokens; the scene and every original token are drawn exactly as replica3.world draws them -- the
    marker rendering uses no draw from the world RNG, and ASP uses its own RNG -- so the original routes are unchanged):
  EN-matching (EN v1 renders these; class W, inside the GO bars):
    ART  the uniform article: before every object token rendered by its noun with no other determiner, and before
         ORD1 / ORD2 ('the former', 'the latter');
    ASP  the progressive frame marker (EN v1 renders 0.2 of utterances in the progressive, 0.2 in the perfect, 0.6 in
         the simple past, drawn independently of the scene; the perfect has no free token in EN v1, its auxiliary is
         fused into the subject form): present with probability 0.2;
    IND  an indefinite article before an object token whose object is NOT given (stress pair of DEF; class W).
  Stress (co-varying; EN v1 does NOT render them; HOLD-COST class, reported, outside the GO bars):
    AGR  person agreement (the English present-tense base form): on the verb when the agent is the speaker, the
         addressee or the joint speaker+addressee;
    DEF  a givenness-informative article: before an object token whose object IS given (named in the previous
         utterance or pointed at) and rendered by its noun.
(2) Per-target thresholds: pool routes (TOK -> P) use theta_P, entity routes (TOK -> E) use theta_E. The prototype
    calibration (calib_proto*.out) gives P 8/4 and E 9/4 at the agency-motif FS weights; E 16/4 with a strong real-FS pool.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replica3 as R
from replica3 import VOICES, PERS, OBJ, ENT, POOLS, NPOOL, POST, pi, PRES

MARK = ["ART", "ASP", "IND", "AGR", "DEF"]
TOKM = R.TOK + MARK; tim = {t: i for i, t in enumerate(TOKM)}
CLASS_W_MARK = ["ART", "ASP", "IND"]; HOLD_COST = ["AGR", "DEF"]


def world_m(seed, T, f_self=0.4, n_voices=5, turn_taking=0.0, v5_speaks=False, prev_use=0.5, joint_p=0.10,
            p_point=0.30, ana_use=0.7, ord_use=0.6, dem_use=0.8, possg_use=0.8, voice_perm=False, p_prog=0.2):
    """replica3.world, copied, with the five markers added (see the module docstring)."""
    rng = np.random.default_rng(seed); rng_asp = np.random.default_rng(10 ** 6 + seed)
    rows = []; prev = None; disc = 0; present = None; prev_named = []
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
        if joint or agent in (S, A): toks.append("AGR")                 # stress: person agreement
        if pat_person: toks.append(refer(pat_person)); parts.add(pat_person)
        else:
            det = False
            if owner is not None:
                if owner == S: toks.append("POSS_S")
                elif owner == A: toks.append("POSS_A")
                elif owner in given and rng.random() < possg_use: toks.append("POSS_G")
                else: toks.append(name(owner))
                parts.add(owner); det = True
            if obj in given and obj == pointed and rng.random() < dem_use: toks += ["DEM", "T_" + obj]; named_now.append(obj)
            elif obj in given and obj in prev_named and len(prev_named) >= 2 and obj in prev_named[:2] and rng.random() < ord_use:
                toks.append("ORD1" if obj == prev_named[0] else "ORD2")
            elif obj in given and obj in prev_named and rng.random() < ana_use: toks.append("ANA")
            else:
                if not det: toks.append("ART")                           # EN-matching uniform article
                toks.append("DEF" if obj in given else "IND")            # stress: givenness-informative article
                toks.append("T_" + obj); named_now.append(obj)
            parts.add(obj)
        if "ORD1" in toks or "ORD2" in toks: toks.append("ART")         # 'the former' / 'the latter'
        if rng_asp.random() < p_prog: toks.append("ASP")                 # EN-matching aspect frame
        if rng.random() < 0.5: toks.append("FILL")
        x = np.zeros(len(TOKM))
        for t in toks: x[tim[t]] = 1
        Sh, Ah = S, A
        if voice_perm:
            vv = [p for p in present if p in voices]; Sh = vv[rng.integers(len(vv))]
            aa = [p for p in present if p != Sh]; Ah = aa[rng.integers(len(aa))]
        y = np.zeros(len(POST))
        y[pi["SPK"]] = Sh in parts; y[pi["ADR"]] = Ah in parts; y[pi["GIV"]] = any(e in given for e in parts)
        y[pi["LEFT"]] = any(left[p] for p in parts if p in PERS); y[pi["NEWEST"]] = any(newest[p] for p in parts if p in PERS)
        for p in present: y[pi[p]] = PRES
        for p in parts: y[pi[p]] = 1.0
        rows.append((x, y))
        prev = S; prev_named = [e for e in dict.fromkeys(named_now)]
    return rows


def train_m(rows, rule, lam, tau, alpha=0.005, warm=200, kap=1.0):
    """replica3.train's arithmetic, copied, over the extended token set."""
    W = np.zeros((len(TOKM), len(POST))); ybar = np.mean([r[1] for r in rows[:warm]], axis=0); th = ybar ** 2
    for x, y in rows:
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
    return W


ZERO = {t: 0.0 for t in R.TOK}


def score(W, thP=(0.40, 0.20), thE=(0.40, 0.20)):
    """GO bars on the original routes (replica3.bars with its S0 replaced by the per-target S0), plus the markers."""
    Wo = W[:len(R.TOK)]
    b = R.bars(Wo, ZERO, 0.0)
    r = lambda t: W[tim[t]]
    fire, silent = [], []
    for t, ks in R.TARGET.items():
        for k in ks: fire.append((r(t)[pi[k]], thP[0]))
    fire.append((r("N_SELF")[NPOOL + ENT.index("SELF")], thE[0]))
    for n in R.NAMES:
        bearer = "V3" if n == "CN" else n[2:]
        fire.append((r(n)[NPOOL + ENT.index(bearer)], thE[0]))
        silent += [(v, thE[1]) for j, v in enumerate(r(n)[NPOOL:]) if ENT[j] != bearer]
    for t in R.TOK:
        for j, pool in enumerate(POOLS):
            if t in R.TARGET and pool in R.TARGET[t]: continue
            if t == "PREV" and pool in ("GIV", "ADR"): continue
            if t in ("IDIO", "PLUR"): continue
            silent.append((r(t)[j], thP[1]))
    for t in ["FILL"] + ["P%d" % i for i in range(1, 6)]:
        silent += [(v, thE[1]) for v in r(t)[NPOOL:]]
    S0 = all(v >= th for v, th in fire) and all(v <= th for v, th in silent)
    small = min(r("SHS")[pi["SPK"]], r("SHA")[pi["ADR"]])
    mk = {}
    for t in MARK:
        mk[t] = dict(max_pool=float(r(t)[:NPOOL].max()), top_pool=POOLS[int(r(t)[:NPOOL].argmax())],
                     max_ent=float(r(t)[NPOOL:].max()))
    W_ok = all(mk[t]["max_pool"] <= max(thP[1], 0.25 * small) and mk[t]["max_pool"] <= thP[1] and mk[t]["max_ent"] <= thE[1]
               for t in CLASS_W_MARK)
    go = dict(S0=S0, S1=b["S1"], S2a=b["S2a"], S2b=b["S2b_nf"], S3=b["S3_nf"], S4=b["S4_nf"], S5=b["S5_nf"], MARK_W=W_ok)
    return go, mk


if __name__ == "__main__":
    CENTRE = {"covpre": 6, "cov": 6, "hebb": 3, "hebbwta": 3, "bcm": 1}
    HERE = os.path.dirname(os.path.abspath(__file__)); out = {"centre": CENTRE}
    TH = {"P8/4,E8/4 (round-2 single curve)": ((0.40, 0.20), (0.40, 0.20)),
          "P8/4,E9/4 (prototype, agency-motif FS)": ((0.40, 0.20), (0.45, 0.20)),
          "P8/4,E16/4 (prototype, strong real-FS pool)": ((0.40, 0.20), (0.80, 0.20))}
    for rule in ("covpre", "cov", "hebb", "hebbwta", "bcm"):
        res = {}
        for seeds, lab in ((range(1, 16), "1-15"), (range(1, 61), "1-60")):
            if lab == "1-60" and rule != "covpre": continue
            per = []
            for s in seeds:
                W = train_m(world_m(s, 4800), rule, CENTRE[rule] / 400, 400)
                per.append({k: score(W, *v) for k, v in TH.items()})
            for k in TH:
                gos = [p[k][0] for p in per]
                n_go = sum(all(g[x] for x in ("S0", "S1", "S2a", "S2b", "S3", "S4", "S5")) for g in gos)
                n_go_w = sum(all(g.values()) for g in gos)
                res["%s|%s" % (lab, k)] = dict(GO_bars=n_go, GO_bars_and_markerW=n_go_w, n=len(gos),
                                               per_bar={x: sum(g[x] for g in gos) for x in gos[0]})
            mk = {t: dict(max_pool_med=float(np.median([p[list(TH)[0]][1][t]["max_pool"] for p in per])),
                          max_pool_max=float(np.max([p[list(TH)[0]][1][t]["max_pool"] for p in per])),
                          top_pool=max(set(p[list(TH)[0]][1][t]["top_pool"] for p in per),
                                       key=lambda q: sum(pp[list(TH)[0]][1][t]["top_pool"] == q for pp in per)),
                          max_ent_med=float(np.median([p[list(TH)[0]][1][t]["max_ent"] for p in per])))
                  for t in MARK}
            res["%s|markers" % lab] = mk
        out[rule] = res
        for k, v in res.items():
            if k.endswith("markers"):
                print("%-7s %s" % (rule, k), {t: ("pool med %.2f (max %.2f, %s) ent med %.2f" % (m["max_pool_med"], m["max_pool_max"], m["top_pool"], m["max_ent_med"])) for t, m in v.items()}, flush=True)
            else:
                print("%-7s %-45s GO %2d/%d  GO+markerW %2d/%d  %s" % (rule, k, v["GO_bars"], v["n"], v["GO_bars_and_markerW"], v["n"], v["per_bar"]), flush=True)
    json.dump(out, open(os.path.join(HERE, "replica4_markers.json"), "w"), indent=1, default=float)
