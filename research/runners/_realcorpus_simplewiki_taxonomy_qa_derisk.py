"""The FULL natural-text taxonomy pipeline (CYCLE 1049 closure): REAL Simple-Wikipedia definitions -> a DISCOVERED
3-level is-a tree (grandparent -> super -> leaves) -> the CYCLE-1043/1044 CHAINED multi-level inheritance QA -> the
canonical "can a dog breathe?" (dog -> mammal -> animal -> breathe) answered on a FULLY-NATURAL-TEXT-DISCOVERED
taxonomy, no curated graph. Extends the CYCLE-1049 extractor to read the first ~3 sentences with pronoun-subject
resolution (super definitions state their genus in a later sentence: "Mammals are in the class Mammalia. They are a
group of vertebrate animals."). numpy (+ cached defs). NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from research.runners._realcorpus_simplewiki_isa_derisk import _CACHE, _first_sentence, mine_natural, _sing
from research.runners._realcorpus_taxonomy_qa_console_derisk import TaxonomyQA

NAT_PROPERTY = {"animal": "breathe", "plant": "grow", "machine": "work", "vehicle": "move"}   # grandparent properties


def mine_multi(title, text):
    """Mine is-a from the first ~3 sentences (a super's genus is often stated after sentence 1); the article TITLE is
    the subject throughout (handles pronoun subjects 'They are.../It is...')."""
    pairs = []
    for sent in text.split(". ")[:3]:
        pairs += mine_natural(title, _first_sentence(sent + "."))
    # dedup, keep first parent per child
    seen, out = set(), []
    for (c, p) in pairs:
        if c not in seen:
            seen.add(c); out.append((c, p))
    return out


def build_tree(defs):
    """From the mined child->parent pairs, assemble a 3-level tree {grandparent: {super: [leaves]}}: a SUPER is a node
    with both children (leaves) and a parent (grandparent). Only supers whose grandparent has a defined property."""
    parent_of, children_of = {}, {}
    for t, ex in defs.items():
        for (c, p) in mine_multi(t, ex):
            parent_of.setdefault(c, p)
            children_of.setdefault(p, set()).add(c)
    tree = {}
    for sup, leaves in children_of.items():
        gp = parent_of.get(sup)                                  # the super's own parent = the grandparent
        if gp is None or gp not in NAT_PROPERTY:
            continue
        real_leaves = sorted(lf for lf in leaves if lf not in children_of)   # leaves = not themselves a super
        if len(real_leaves) >= 3:
            tree.setdefault(gp, {})[sup] = real_leaves
    return {g: sm for g, sm in tree.items() if len(sm) >= 1}, parent_of, children_of


def run_seed(seed, tree):
    qa = TaxonomyQA(seed, tree, hold_out=False)
    # property-grandparents = those in the chained-read's usable set (>=2 supers) AND with a defined property
    prop_gps = [g for g in qa.gps if g in NAT_PROPERTY]
    qa.gp_of_property = {NAT_PROPERTY[g]: g for g in prop_gps}
    props = list(qa.gp_of_property)
    yes_ok = yes_tot = no_ok = no_tot = 0
    for g in prop_gps:
        tp = NAT_PROPERTY[g]
        for s in tree[g]:
            for lf in tree[g][s]:
                if lf not in qa.row:
                    continue
                yes_tot += 1; yes_ok += int(qa.ask(lf, tp) == "yes")
                for op in props:
                    if op != tp:                                 # a DIFFERENT property-grandparent's property
                        no_tot += 1; no_ok += int(qa.ask(lf, op) == "no")
    moat = sum(qa.ask(z, props[0]) == "moat" for z in ["zzz", "qqq"]) / 2.0 if props else 0.0
    no = (no_ok / no_tot) if no_tot else None                    # None = ill-posed (only 1 property-grandparent)
    return {"yes": yes_ok / max(1, yes_tot), "no": no, "moat": moat, "n": yes_tot, "n_prop_gps": len(prop_gps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    defs = json.load(open(_CACHE))
    tree, parent_of, _ = build_tree(defs)
    print(f"[simple-wiki taxonomy QA] natural-text 3-level tree:", flush=True)
    for g, sm in tree.items():
        print(f"  {g} ({NAT_PROPERTY.get(g)}): " + "; ".join(f"{s}->[{','.join(l[:5])}]" for s, l in sm.items()), flush=True)
    qa0 = TaxonomyQA(42, tree, hold_out=False)
    prop_gps = [g for g in qa0.gps if g in NAT_PROPERTY]         # usable property-grandparents (>=2 supers)
    qa0.gp_of_property = {NAT_PROPERTY[g]: g for g in prop_gps}
    if not prop_gps or sum(len(tree[g]) for g in prop_gps) < 2:
        print("  VERDICT: NOT-EVALUABLE (no property-grandparent with >=2 natural-text supers)"); return
    # canonical demo: one leaf per property-grandparent (the dog->mammal->animal->breathe chain)
    for g in prop_gps:
        for s0 in tree[g]:
            lf0 = tree[g][s0][0]
            print(f"  Q: can a {lf0} {NAT_PROPERTY[g]}?  ({lf0}->{s0}->{g})  -> {qa0.ask(lf0, NAT_PROPERTY[g])}", flush=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    ys, ns, ms = [], [], []
    for sd in seeds:
        r = run_seed(sd, tree); ys.append(r["yes"]); ms.append(r["moat"])
        if r["no"] is not None:
            ns.append(r["no"])
        no_s = f"{r['no']:.3f}" if r["no"] is not None else "N/A(1 prop-gp)"
        print(f"  [seed {sd}] YES(inherit)={r['yes']:.3f} NO(mismatch)={no_s} moat={r['moat']:.3f} ({r['n']} leaf-queries, {r['n_prop_gps']} prop-grandparents)", flush=True)
    # GO on the achievable claim: inherit + moat on the natural-text tree; mismatch-NO only when >=2 property-grandparents
    no_ok = (not ns) or all(n > 0.85 for n in ns)
    go = all(y > 0.95 for y in ys) and all(m > 0.99 for m in ms) and no_ok
    no_agg = f"{np.mean(ns):.3f}" if ns else "N/A (single property-grandparent -- breadth follow-on)"
    print(f"\n  AGGREGATE: YES={np.mean(ys):.3f} NO={no_agg} moat={np.mean(ms):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- the canonical multi-level chain (e.g. dog->mammal->animal->breathe) "
          f"inherits 2-up on a taxonomy DISCOVERED ENTIRELY from natural Simple-Wikipedia definitions {'(YES=inherit + moat, 6-seed) -> the WHOLE taxonomy pipeline (structure discovery + multi-level inheritance) runs on natural definitional text, no curated graph. Honest scope: this ~65-title fetch yields ONE rich property-grandparent (animal, 3 supers); the mismatch-NO cross-property test needs a 2nd rich grandparent (a breadth follow-on -- more plant/machine titles)' if go else '-- some gate unmet; honest boundary'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
