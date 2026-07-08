"""OPEN-WORLD INFERENCE beyond stored structure -- spreading-activation semantic completion (the open-domain
research-gate's #1 cheapest mechanism). A query about a concept with NO stored fact does NOT hard-abstain if it is
ADJACENT (in the brain's own learned co-occurrence code space) to concepts that DO have facts: activation spreads to
the nearest neighbours and their shared property is offered as a HEDGED, graded-confidence best guess. The moat is
UPGRADED not weakened: a genuinely DISJOINT unknown (no close neighbour) still hard-abstains. Biology: Collins-Loftus
spreading activation; Rogers-McClelland semantic cognition; CA3 pattern completion (catalog D.13). Reuse-by-import
(the emergent co-occurrence codes); numpy. NO `sim/` edit.

Anti-cheats (from the research gate): (A) DERANGED neighbourhood (shuffle codes) -> completion collapses to chance;
(B) DISJOINT code (a genuinely novel concept) -> still hard-abstains (moat preserved); (C) CONFIDENCE tracks
neighbourhood tightness (in-category nearest-cosine >> disjoint nearest-cosine).
"""
from __future__ import annotations
import argparse
import numpy as np
from research.runners._emergent_vocab_breadth_scale_derisk import learn_stream_codes
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows

N_CAT = 4
MEMBERS_PER_CAT = 8
PROPS = ["run", "swim", "fly", "dig"]


def _cooccur_stream(rng):
    """A category-structured co-occurrence stream (NOT a taught is-a): each category has shared CONTEXT tokens; its
    members appear with those contexts -> same-category members get similar codes (EMERGE-30 shared-context). The
    completion later uses only the resulting code SIMILARITY, never a stored is-a link."""
    cats = {f"cat{c}": [f"m{c}_{i}" for i in range(MEMBERS_PER_CAT)] for c in range(N_CAT)}
    ctx = {f"cat{c}": [f"ctx{c}_{k}" for k in range(4)] for c in range(N_CAT)}
    sents = []
    for c in range(N_CAT):
        cc = f"cat{c}"
        for m in cats[cc]:
            for k in ctx[cc]:
                sents.append([m, k])                             # member co-occurs with its category contexts
                sents.append([k, m])
    stream, order = [], list(range(len(sents)))
    for _ in range(60):
        rng.shuffle(order); stream += [list(sents[i]) for i in order]
    return stream, cats


def _real_animal_stream(seed):
    """REAL codes: learn from the Simple-Wikipedia definitional stream (CYCLE 1049/1050), using FOUR WELL-SEPARATED
    categories from DIFFERENT domains (mammal/tree/vehicle/tool) -- each tight (members share one super token) AND
    mutually dissimilar (no shared super-context, unlike nested mammal/bird/fish under 'animal'). Real co-occurrence
    codes, no synthetic scaffold."""
    import json
    from research.runners._realcorpus_simplewiki_isa_derisk import _first_sentence
    from research.runners._realcorpus_simplewiki_taxonomy_qa_derisk import mine_multi
    defs = json.load(open("research/findings/raw/_simplewiki_defs.json"))
    by_super = {}
    for t, ex in defs.items():
        for (c, p) in mine_multi(t, ex):
            by_super.setdefault(p, []).append(c)
    want = ["mammal", "tree", "vehicle", "tool"]                  # 4 domains, mutually dissimilar
    supers = [s for s in want if len(set(by_super.get(s, []))) >= 4]
    cats = {f"cat{i}": sorted(set(by_super[s])) for i, s in enumerate(supers)}
    rng = np.random.default_rng(seed)
    sents = []
    for i, cc in enumerate(cats):
        sup = supers[i]
        for m in cats[cc]:
            sents += [[m, sup], [sup, m], ["the", m, "is", "a", sup]]
    stream, order = [], list(range(len(sents)))
    for _ in range(60):
        rng.shuffle(order); stream += [list(sents[i]) for i in order]
    return stream, cats, supers


def run_seed(seed, real=False):
    rng = np.random.default_rng(seed)
    if real:
        stream, cats, supers = _real_animal_stream(seed)
        ncat = len(cats)
        members = [m for c in cats for m in cats[c]]
        hubs = supers + ["the", "is", "a"]
        codes, _ = learn_stream_codes(seed, stream, members, hubs, window=3)
        cat_keys = list(cats)
        cat_of = {m: k for k, ms in cats.items() for m in ms}
        cat_of = {m: cat_keys.index(cat_of[m]) for m in members}
        prop_of_cat = {i: PROPS[i] for i in range(ncat)}
        cats = {f"cat{i}": cats[cat_keys[i]] for i in range(ncat)}
    else:
        stream, cats = _cooccur_stream(rng)
        ncat = N_CAT
        members = [m for c in range(N_CAT) for m in cats[f"cat{c}"]]
        ctxs = sorted({t for st in stream for t in st if t.startswith("ctx")})
        codes, _ = learn_stream_codes(seed, stream, members, ctxs, window=2)
        cat_of = {m: c for c in range(N_CAT) for m in cats[f"cat{c}"]}
        prop_of_cat = {c: PROPS[c] for c in range(N_CAT)}
    U = _unit_rows(codes); row = {m: i for i, m in enumerate(members)}
    n_cats = ncat
    _keep = 5 if not real else 0.6                               # real cats vary in size -> hold out ~40%
    # TEACH the category property to most members ("known" concepts with stored facts); hold out the rest
    known, held = {}, []
    for c in range(n_cats):
        ms = list(cats[f"cat{c}"]); rng.shuffle(ms)
        n_known = _keep if isinstance(_keep, int) else max(2, int(round(len(ms) * _keep)))
        n_known = min(n_known, len(ms) - 1)                     # always hold out >=1
        for m in ms[:n_known]:
            known[m] = prop_of_cat[c]                            # stored fact: m -> its property
        held += ms[n_known:]                                    # held-out (no stored fact)
    known_rows = {m: row[m] for m in known}

    def complete(vec, theta):
        """Spreading-activation completion: nearest KNOWN concept by code cosine. Returns (prop, confidence) if the
        nearest neighbour clears theta (adjacent unknown -> hedged guess), else (None, conf) -> hard-abstain."""
        best_m, best_cos = None, -1.0
        for m, r in known_rows.items():
            cs = float(vec @ U[r])
            if cs > best_cos:
                best_cos, best_m = cs, m
        if best_cos >= theta:
            return known[best_m], best_cos
        return None, best_cos

    # calibrate theta from the KNOWN set's own nearest-OTHER-known cosine (a self-consistency floor), frozen per seed
    withins = []
    for m, r in known_rows.items():
        cs = max(float(U[r] @ U[rr]) for mm, rr in known_rows.items() if mm != m)
        withins.append(cs)
    theta = float(np.percentile(withins, 20))                   # 20th pct of known-neighbour tightness

    # (main) held-out ADJACENT unknowns: spreading activation should guess the TRUE category property
    hit = tot = 0
    held_conf = []
    for m in held:
        prop, conf = complete(U[row[m]], theta)
        held_conf.append(conf)
        if prop is not None:
            tot += 1; hit += int(prop == prop_of_cat[cat_of[m]])
    main_acc = hit / max(1, tot)
    main_cover = tot / max(1, len(held))                        # fraction of adjacent unknowns that got a guess

    # (A) DERANGED neighbourhood: shuffle which code-row each known concept sits at -> the nearest neighbour becomes
    # random -> the guess collapses to chance. Averaged over 10 shuffles (a single shuffle is high-variance at N_CAT=4).
    km = list(known_rows)
    der_accs = []
    for _ in range(10):
        perm = list(known_rows.values()); rng.shuffle(perm)
        der_rows = {km[i]: perm[i] for i in range(len(km))}
        der_hit = der_tot = 0
        for m in held:
            best_m, best = None, -1.0
            for mm in der_rows:
                cs = float(U[row[m]] @ U[der_rows[mm]])
                if cs > best:
                    best, best_m = cs, mm
            if best >= theta:
                der_tot += 1; der_hit += int(known[best_m] == prop_of_cat[cat_of[m]])
        der_accs.append(der_hit / max(1, der_tot))
    der_acc = float(np.mean(der_accs))

    # (B) DISJOINT unknowns: random unit codes (a genuinely novel concept, no category) -> must hard-ABSTAIN
    dis_abstain = 0
    dis_conf = []
    for _ in range(30):
        v = rng.standard_normal(U.shape[1]); v /= np.linalg.norm(v)
        prop, conf = complete(v, theta)
        dis_conf.append(conf)
        dis_abstain += int(prop is None)
    dis_abstain /= 30.0

    # (C) CONFIDENCE tracks tightness: adjacent-unknown nearest-cosine >> disjoint nearest-cosine
    conf_gap = float(np.mean(held_conf) - np.mean(dis_conf))
    return {"acc": main_acc, "cover": main_cover, "deranged": der_acc, "disjoint_abstain": dis_abstain,
            "held_conf": float(np.mean(held_conf)), "disj_conf": float(np.mean(dis_conf)), "conf_gap": conf_gap,
            "n_cats": n_cats, "n_held": len(held)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--real", action="store_true", help="use REAL Simple-Wiki animal-domain codes instead of synthetic")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    src = "REAL Simple-Wiki codes (mammal/tree/vehicle/tool domains)" if a.real else f"synthetic ({N_CAT} cats x {MEMBERS_PER_CAT})"
    print(f"[spreading-activation completion] guess the adjacent-unknown's property via nearest learned-code neighbour "
          f"| {src}, props={PROPS}", flush=True)
    A, C, D, B, G, NC = [], [], [], [], [], []
    for s in seeds:
        r = run_seed(s, real=a.real)
        A.append(r["acc"]); C.append(r["cover"]); D.append(r["deranged"]); B.append(r["disjoint_abstain"]); G.append(r["conf_gap"]); NC.append(r["n_cats"])
        print(f"  [seed {s}] guess-acc={r['acc']:.3f} coverage={r['cover']:.3f} | deranged={r['deranged']:.3f} "
              f"disjoint-abstain={r['disjoint_abstain']:.3f} conf(held={r['held_conf']:.2f} vs disj={r['disj_conf']:.2f}, gap={r['conf_gap']:.2f}) [{r['n_cats']}cat/{r['n_held']}held]", flush=True)
    chance = 1.0 / float(np.mean(NC))
    go = (all(x > 0.85 for x in A) and all(c > 0.85 for c in C) and all(d < 0.45 for d in D)
          and all(b > 0.85 for b in B) and all(g > 0.15 for g in G))
    print(f"\n  AGGREGATE: guess-acc={np.mean(A):.3f} coverage={np.mean(C):.3f} deranged={np.mean(D):.3f} "
          f"disjoint-abstain={np.mean(B):.3f} conf-gap={np.mean(G):.3f} (chance={chance:.3f})", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- an ADJACENT unknown (no stored fact) gets a HEDGED best-guess "
          f"property via spreading activation to its nearest learned-code neighbour "
          f"{'(guess-acc + coverage high; DERANGED codes collapse it to chance; DISJOINT unknowns still hard-abstain = moat preserved; confidence tracks neighbourhood tightness) -> open-world inference beyond stored structure, moat UPGRADED not weakened' if go else '-- some gate unmet; honest boundary'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
