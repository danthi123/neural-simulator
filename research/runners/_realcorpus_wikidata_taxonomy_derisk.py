"""Wikidata P279 taxonomic acquisition (the taxonomy data-acquisition de-risk, per the CYCLE-1039 research):
fetch a real multi-level is-a graph from Wikidata subclass-of (P279) -- BREADTH by fetching the subclasses of
a few superordinates (mammal/bird/fish/tree/vehicle/tool), DEPTH by fetching those superordinates' own parents
-- and measure whether it forms a CLEAN multi-level taxonomy (hub superordinates with >=3 members + is-a chains
super->grandparent) that the ready EMERGE-44/45 stacked pooler can consume. This is the first-look: does REAL
curated is-a data (unlike the NEGATIVE distributional/copular extraction from TinyStories/WikiText) yield a clean
multi-level graph? If GO, the stacked-pooler inheritance test is the follow-on. numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from research.runners._fluidconv_phase15_wikidata_breadth_derisk import _sparql, _head_token

# superordinate QIDs (breadth: fetch their subclasses) -- common, high-confidence categories.
SUPERS = {"mammal": "Q7377", "bird": "Q5113", "fish": "Q152", "insect": "Q1390",
          "tree": "Q10884", "vehicle": "Q42889", "tool": "Q39546", "fruit": "Q3314483"}


def _subclasses(qid, limit=25):
    q = (f'SELECT ?cLabel WHERE {{ ?c wdt:P279 wd:{qid}. '
         f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT {limit}')
    out = []
    for row in _sparql(q):
        tok = _head_token(row.get("cLabel", {}).get("value", ""))
        if tok and len(tok) > 2:
            out.append(tok)
    return out


def _parents(qid, limit=4):
    q = (f'SELECT ?pLabel WHERE {{ wd:{qid} wdt:P279 ?p. '
         f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT {limit}')
    out = []
    for row in _sparql(q):
        tok = _head_token(row.get("pLabel", {}).get("value", ""))
        if tok and len(tok) > 2:
            out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="research/findings/raw/_wikidata_taxonomy.json")
    a = ap.parse_args()
    print(f"[wikidata P279 taxonomy] fetching subclasses of {list(SUPERS)} + their parents ...", flush=True)
    isa = defaultdict(set)                                   # child -> {parents}
    n_children_of = defaultdict(set)
    try:
        for sname, sqid in SUPERS.items():
            kids = _subclasses(sqid); time.sleep(0.3)
            for k in kids:
                if k != sname:
                    isa[k].add(sname); n_children_of[sname].add(k)
            pars = _parents(sqid); time.sleep(0.3)           # DEPTH: the super's own parent (2nd is-a level)
            for p in pars:
                if p != sname:
                    isa[sname].add(p); n_children_of[p].add(sname)
            print(f"  {sname}: {len(kids)} subclasses, parents={pars}", flush=True)
    except Exception as ex:
        print(f"  FETCH FAILED (network): {type(ex).__name__}: {ex}", flush=True)
        print("  VERDICT: NOT-EVALUABLE (Wikidata endpoint unreachable this run) -- de-risk when reachable.", flush=True)
        return
    hubs = {p for p, ks in n_children_of.items() if len(ks) >= 3}
    chains = sum(1 for c in isa for p in isa[c] if p in isa)   # child->parent->grandparent
    hub_pairs = sum(len(ks) for p, ks in n_children_of.items() if p in hubs)
    json.dump({"isa": {c: sorted(ps) for c, ps in isa.items()}}, open(a.out, "w"), indent=2)
    print(f"\n  is-a graph: {sum(len(v) for v in isa.values())} pairs, {len(isa)} children, "
          f"{len(n_children_of)} parents; HUBS(>=3 children)={len(hubs)} ({hub_pairs} hub-pairs); chains={chains}", flush=True)
    print(f"  hubs: {[(p, len(n_children_of[p])) for p in hubs]}", flush=True)
    go = len(hubs) >= 5 and hub_pairs >= 40 and chains >= 5
    print(f"  VERDICT: {'GO (clean multi-level is-a from REAL Wikidata P279)' if go else 'PARTIAL'} -- "
          f"{'the data-acquisition premise HOLDS: real curated is-a yields a clean multi-level taxonomy (hubs+chains) -> feed the ready stacked pooler (inheritance follow-on)' if go else 'fewer clean hubs/chains than target; inspect'}. Saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
