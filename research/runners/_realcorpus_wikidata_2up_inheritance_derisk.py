"""TRUE 2-up (grandparent) multi-level inheritance over a 3-level Wikidata P279 taxonomy: fetch grandparent ->
supers -> leaves (a real 3-level is-a tree), learn leaf codes from the is-a DEFINITIONAL stream, and test whether
a HELD-OUT super's leaves inherit their GRANDPARENT's property 2 levels up (never taught on that super), vs a
grandparent-derangement control. This validates the MULTI-level (not just single-level) inheritance claim on real
data: does the emergent code-learning propagate is-a similarity 2 hops (leaf~super~grandparent)? numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from collections import defaultdict
from research.runners._fluidconv_phase15_wikidata_breadth_derisk import _sparql, _head_token
from research.runners._emergent_vocab_breadth_scale_derisk import learn_stream_codes
from research.runners._realcorpus_inheritance_multilevel_derisk import _teach_test

# high-level GRANDPARENTS (fetch their subclasses = supers; each super's subclasses = leaves) -> a 3-level tree.
GRANDPARENTS = {"animal": "Q729", "plant": "Q756", "vehicle": "Q42889", "food": "Q2095", "tool": "Q39546"}


def _subclasses(qid, limit):
    q = (f'SELECT ?cLabel WHERE {{ ?c wdt:P279 wd:{qid}. ?c wdt:P279 wd:{qid}. '
         f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT {limit}')
    out = []
    for row in _sparql(q):
        tok = _head_token(row.get("cLabel", {}).get("value", ""))
        if tok and len(tok) > 2:
            out.append((tok, None))
    # need the QIDs of the subclasses to go a level deeper -> refetch with the item
    q2 = (f'SELECT ?c ?cLabel WHERE {{ ?c wdt:P279 wd:{qid}. '
          f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT {limit}')
    res = []
    for row in _sparql(q2):
        qid_c = row.get("c", {}).get("value", "").rsplit("/", 1)[-1]
        tok = _head_token(row.get("cLabel", {}).get("value", ""))
        if tok and len(tok) > 2 and qid_c.startswith("Q"):
            res.append((tok, qid_c))
    return res


def fetch_3level(out_path):
    tree = {}                                               # grandparent -> {super -> [leaves]}
    for gname, gqid in GRANDPARENTS.items():
        supers = _subclasses(gqid, 8); time.sleep(0.3)
        smap = {}
        for (sname, sqid) in supers[:6]:
            if sqid is None or sname == gname:
                continue
            leaves = [t for (t, _q) in _subclasses(sqid, 12) if t != sname and t != gname]
            time.sleep(0.3)
            if len(leaves) >= 3:
                smap[sname] = sorted(set(leaves))
        if len(smap) >= 2:
            tree[gname] = smap
        print(f"  {gname}: {len(smap)} supers with >=3 leaves", flush=True)
    json.dump(tree, open(out_path, "w"), indent=2)
    return tree


def _stream(tree, rng):
    sents = []
    for g, smap in tree.items():
        for s, leaves in smap.items():
            sents.append(["the", s, "is", "a", g])
            for lf in leaves:
                sents.append(["the", lf, "is", "a", s])
    stream = []
    order = list(range(len(sents)))
    for _ in range(40):
        rng.shuffle(order)
        for i in order:
            stream.append(list(sents[i]))
    return stream


def run_seed(seed, tree):
    rng = np.random.default_rng(seed)
    gps = [g for g, sm in tree.items() if len(sm) >= 2]
    if len(gps) < 2:
        return None
    stream = _stream(tree, rng)
    leaves = sorted({lf for g in gps for s in tree[g] for lf in tree[g][s]})
    supers = sorted({s for g in gps for s in tree[g]})
    codes, _ = learn_stream_codes(seed, stream, leaves, supers, window=4)   # code the LEAVES; supers=context hubs
    row_of = {w: i for i, w in enumerate(leaves)}
    P = rng.standard_normal((len(gps), 64))
    # teach a grandparent property via the leaves of SOME of its supers; HOLD OUT a whole different super's leaves
    taught, queries = defaultdict(list), []
    for g in gps:
        sl = list(tree[g]); rng.shuffle(sl)
        if len(sl) < 2:
            continue
        for s in sl[:-1]:                                   # taught supers
            taught[g] += [row_of[lf] for lf in tree[g][s] if lf in row_of]
        for lf in tree[g][sl[-1]]:                          # HELD-OUT super's leaves -> inherit grandparent 2-up
            if lf in row_of:
                queries.append((row_of[lf], g))
    acc = _teach_test(codes, taught, queries, P, gps)
    tk = [g for g in gps if g in taught]
    ders = []
    for _ in range(5):
        perm = list(tk)
        while any(perm[i] == tk[i] for i in range(len(perm))):
            rng.shuffle(perm)
        wrong = {tk[i]: taught[perm[i]] for i in range(len(tk))}
        ders.append(_teach_test(codes, wrong, queries, P, gps))
    return {"seed": seed, "n_gp": len(gps), "n_q": len(queries), "acc": acc,
            "deranged": float(np.mean(ders)), "chance": 1.0 / len(gps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="research/findings/raw/_wikidata_3level.json")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--margin", type=float, default=0.15)
    a = ap.parse_args()
    import os
    if a.refetch or not os.path.exists(a.tree):
        print("[wikidata 3-level fetch] grandparent -> supers -> leaves ...", flush=True)
        try:
            tree = fetch_3level(a.tree)
        except Exception as ex:
            print(f"  FETCH FAILED: {type(ex).__name__}: {ex}; VERDICT: NOT-EVALUABLE", flush=True); return
    else:
        tree = json.load(open(a.tree))
    gps = [g for g in tree if len(tree[g]) >= 2]
    print(f"  3-level tree: grandparents={gps}", flush=True)
    recs = [r for r in (run_seed(s, tree) for s in [int(x) for x in a.seeds.split(",")]) if r]
    if not recs:
        print("  NOT-EVALUABLE (need >=2 rich grandparents)"); return
    for r in recs:
        print(f"  [seed {r['seed']}] 2up-inherit={r['acc']:.3f} deranged={r['deranged']:.3f} chance={r['chance']:.3f} "
              f"({r['n_gp']} grandparents, {r['n_q']} held-out-super queries)", flush=True)
    m = lambda k: float(np.mean([r[k] for r in recs]))
    go = all(r["acc"] - r["deranged"] > a.margin and r["acc"] - r["chance"] > a.margin for r in recs)
    print(f"\n  AGGREGATE: 2up-inherit={m('acc'):.3f} deranged={m('deranged'):.3f} chance={m('chance'):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- a HELD-OUT super's leaves inherit their GRANDPARENT's "
          f"property 2 levels up {'(beats chance + grandparent-derangement all seeds) -> TRUE multi-level (2-up) inheritance on real Wikidata is-a; the emergent codes propagate leaf~super~grandparent' if go else '-> 2-up does NOT cleanly propagate through the emergent codes (single-level works; 2-up needs a richer code scheme -- honest scope)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
