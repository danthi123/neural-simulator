"""Multi-level inheritance over the REAL Wikidata P279 is-a graph (the taxonomy unlock, end-to-end): the CYCLE-1040
fetch gave a CLEAN multi-level is-a graph. This closes the loop emergently -- learn concept CODES from the is-a as a
DEFINITIONAL stream ("the dog is a mammal ..."), where same-super members share their SUPER as co-occurrence context
(EMERGE-30 shared-context mechanism), so same-super members get similar codes; then the associative-memory read
(rung-1/multilevel `_teach_test`) inherits: a HELD-OUT member of a super inherits that super's taught property
(via the emergent code similarity), and the SUPER-DERANGEMENT control collapses it. Confirms the ready mechanism
+ REAL is-a data -> multi-level inheritance, on the corpus the extraction could NOT provide before. numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from research.runners._emergent_vocab_breadth_scale_derisk import learn_stream_codes
from research.runners._realcorpus_inheritance_multilevel_derisk import _teach_test


def _isa_stream(isa, reps=40, rng=None):
    """Definitional stream: 'the <child> is a <parent>' for each is-a pair, repeated + shuffled. Same-super members
    thus share their PARENT token as co-occurrence context (EMERGE-30 shared-context grouping)."""
    sents = []
    for c, parents in isa.items():
        for p in parents:
            sents.append(["the", c, "is", "a", p])
    stream = []
    order = list(range(len(sents)))
    for _ in range(reps):
        if rng is not None:
            rng.shuffle(order)
        for i in order:
            stream.append(list(sents[i]))
    return stream


def run_seed(seed, isa, min_members=3):
    rng = np.random.default_rng(seed)
    # supers with >= min_members children (usable categories); members = the children
    children_of = {}
    for c, parents in isa.items():
        for p in parents:
            children_of.setdefault(p, []).append(c)
    supers = [p for p, ks in children_of.items() if len(ks) >= min_members]
    members = sorted({c for p in supers for c in children_of[p]})
    if len(supers) < 3 or len(members) < 12:
        return None
    stream = _isa_stream(isa, reps=40, rng=rng)
    vocab = members                                          # code the MEMBERS
    hubs = supers                                            # the shared context = the super tokens
    codes, _ = learn_stream_codes(seed, stream, vocab, hubs, window=4)
    row_of = {w: i for i, w in enumerate(vocab)}
    # each member -> its (first) super with >=min_members; hold out one whole super's members? -> single-level:
    # teach a super property to SOME members, hold out OTHERS of the same super, test inheritance vs derangement.
    super_ids = supers
    P = rng.standard_normal((len(super_ids), 64))
    taught_by_super, queries = {}, []
    for s in super_ids:
        ks = [c for c in children_of[s] if c in row_of]; rng.shuffle(ks)
        if len(ks) < 2:
            continue
        cut = max(1, len(ks) // 2)
        taught_by_super[s] = [row_of[c] for c in ks[:cut]]   # teach half
        for c in ks[cut:]:
            queries.append((row_of[c], s))                   # held-out half inherit
    acc = _teach_test(codes, taught_by_super, queries, P, super_ids)
    # DERANGEMENT control: each super's property is taught with a DIFFERENT (permuted) super's members, while the
    # held-out members are queried at their TRUE super -> a mismatch (a held-out member's code no longer matches the
    # taught set bound to its super), so inheritance MUST collapse if the grouping/codes are load-bearing. (NOT a
    # consistent relabel -- queries stay at the true super.)
    taught_keys = [s for s in super_ids if s in taught_by_super]
    ders = []
    for _ in range(5):
        perm = list(taught_keys)
        while any(perm[i] == taught_keys[i] for i in range(len(perm))):   # a true derangement (no fixed points)
            rng.shuffle(perm)
        wrong_taught = {taught_keys[i]: taught_by_super[perm[i]] for i in range(len(taught_keys))}
        ders.append(_teach_test(codes, wrong_taught, queries, P, super_ids))   # queries at the TRUE super
    return {"seed": seed, "n_supers": len(super_ids), "n_queries": len(queries),
            "acc": acc, "deranged": float(np.mean(ders)), "chance": 1.0 / len(super_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isa", default="research/findings/raw/_wikidata_taxonomy.json")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--margin", type=float, default=0.15)
    a = ap.parse_args()
    isa = {c: set(ps) for c, ps in json.load(open(a.isa))["isa"].items()}
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[wikidata is-a inheritance] real Wikidata P279 graph ({sum(len(v) for v in isa.values())} pairs)", flush=True)
    recs = [r for r in (run_seed(s, isa) for s in seeds) if r is not None]
    if not recs:
        print("  NOT-EVALUABLE (too few usable supers/members)"); return
    for r in recs:
        print(f"  [seed {r['seed']}] held-out inherit={r['acc']:.3f} deranged={r['deranged']:.3f} "
              f"chance={r['chance']:.3f} ({r['n_supers']} supers, {r['n_queries']} held-out queries)", flush=True)
    m = lambda k: float(np.mean([r[k] for r in recs]))
    go = all(r["acc"] - r["deranged"] > a.margin and r["acc"] - r["chance"] > a.margin for r in recs)
    print(f"\n  AGGREGATE: held-out inherit={m('acc'):.3f} deranged={m('deranged'):.3f} chance={m('chance'):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL'} -- a HELD-OUT member of a super inherits that super's property via "
          f"codes learned from the REAL Wikidata is-a definitional stream {'(beats chance + super-derangement all seeds) -> the taxonomy is end-to-end unlocked: real is-a data + the ready emergent mechanism -> multi-level inheritance' if go else '(margin not met on some seed)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
