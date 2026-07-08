"""TRUE 2-up inheritance via a CHAINED read (the stacked-pooler logic, CYCLE-1042 fix): the flat leaf->grandparent
read was NEGATIVE (codes don't propagate 2 hops); but BOTH single-level steps work (CYCLE 1041). So chain them:
leaf -> its SUPER (L1: nearest super by the leaf's code among the taught supers' leaves) -> the super's GRANDPARENT
(L2: nearest grandparent by the super's code among same-grandparent supers) -> inherit the grandparent property.
A HELD-OUT super's leaves inherit their grandparent 2-up via this chain, vs a grandparent-derangement control.
Reuses the 3-level Wikidata tree (CYCLE 1042) + learn_stream_codes. numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from research.runners._emergent_vocab_breadth_scale_derisk import learn_stream_codes
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows


def _stream(tree, rng):
    sents = []
    for g, smap in tree.items():
        for s, leaves in smap.items():
            sents.append(["the", s, "is", "a", g])
            for lf in leaves:
                sents.append(["the", lf, "is", "a", s])
    stream, order = [], list(range(len(sents)))
    for _ in range(40):
        rng.shuffle(order)
        stream += [list(sents[i]) for i in order]
    return stream


def run_seed(seed, tree, deranged=False):
    rng = np.random.default_rng(seed)
    gps = [g for g in tree if len(tree[g]) >= 2]
    if len(gps) < 2:
        return None
    supers = sorted({s for g in gps for s in tree[g]})
    leaves = sorted({lf for g in gps for s in tree[g] for lf in tree[g][s]})
    super_of = {lf: s for g in gps for s in tree[g] for lf in tree[g][s]}
    # codes: learn leaf + super codes from the is-a stream (leaves coded via super context; supers via grandparent)
    stream = _stream(tree, rng)
    allw = leaves + supers
    codes, _ = learn_stream_codes(seed, stream, allw, gps + supers, window=4)
    U = _unit_rows(codes); row = {w: i for i, w in enumerate(allw)}
    # hold out one whole super per grandparent (its leaves are the 2-up queries); the rest are "taught"
    held_supers = {}
    for g in gps:
        sl = list(tree[g]); rng.shuffle(sl); held_supers[g] = sl[-1]
    # L1 prototypes: each super = mean code of ITS leaves (from taught leaves only, exclude the query leaf)
    def super_proto(s, exclude):
        ls = [lf for g in gps for ss in tree[g] if ss == s for lf in tree[g][ss] if lf != exclude and lf in row]
        return U[[row[lf] for lf in ls]].mean(0) if ls else np.zeros(U.shape[1])
    # L2 prototypes: each grandparent's proto = mean code of the TAUGHT supers ASSIGNED to it. DERANGEMENT assigns
    # each grandparent a DIFFERENT grandparent's supers -> a super's code no longer matches its own grandparent's
    # proto -> the chain routes to the WRONG grandparent -> collapse (the grouping is load-bearing).
    gp_supers = {g: [s for s in tree[g] if s != held_supers[g]] for g in gps}
    if deranged:
        perm = list(gps)
        while any(perm[i] == gps[i] for i in range(len(perm))):
            rng.shuffle(perm)
        gp_supers = {gps[i]: gp_supers[perm[i]] for i in range(len(gps))}   # grandparent gets a WRONG super-set
    def gp_proto(g):
        ss = [s for s in gp_supers[g] if s in row]
        return U[[row[s] for s in ss]].mean(0) if ss else np.zeros(U.shape[1])
    gp_protos = {g: gp_proto(g) for g in gps}
    correct = tot = 0
    for g in gps:
        hs = held_supers[g]
        for lf in tree[g][hs]:
            if lf not in row:
                continue
            tot += 1
            # L1: leaf -> nearest super (by leaf code vs super protos, over ALL supers incl held-out)
            s_scores = {s: U[row[lf]] @ super_proto(s, lf) for s in supers if s in row}
            s_hat = max(s_scores, key=s_scores.get)
            # L2: that super -> nearest grandparent (by the super's code vs grandparent protos)
            g_scores = {gg: U[row[s_hat]] @ gp_protos[gg] for gg in gps}
            g_hat = max(g_scores, key=g_scores.get)
            correct += int(g_hat == g)
    return correct / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="research/findings/raw/_wikidata_3level.json")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    tree = json.load(open(a.tree))
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[wikidata 2-up CHAINED] leaf->super->grandparent chained inheritance | grandparents={[g for g in tree if len(tree[g])>=2]}", flush=True)
    accs, ders = [], []
    for s in seeds:
        acc = run_seed(s, tree, deranged=False); der = run_seed(s, tree, deranged=True)
        if acc is None:
            print("  NOT-EVALUABLE"); return
        accs.append(acc); ders.append(der)
        print(f"  [seed {s}] 2up-chained={acc:.3f} deranged={der:.3f} chance={1.0/len([g for g in tree if len(tree[g])>=2]):.3f}", flush=True)
    ch = 1.0 / len([g for g in tree if len(tree[g]) >= 2])
    go = all(accs[i] - ders[i] > 0.15 and accs[i] - ch > 0.15 for i in range(len(accs)))
    print(f"\n  AGGREGATE: 2up-chained={np.mean(accs):.3f} deranged={np.mean(ders):.3f} chance={ch:.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- a held-out super's leaves inherit their GRANDPARENT 2-up via "
          f"the CHAINED read (leaf->super->grandparent) {'(beats chance + grandparent-derangement all seeds) -> TRUE multi-level (2-up) inheritance on real Wikidata is-a via the stacked/chained mechanism' if go else '-> the chained 2-up still does not cleanly hold; honest boundary'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
