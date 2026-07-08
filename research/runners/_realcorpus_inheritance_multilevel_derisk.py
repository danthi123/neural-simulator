"""KNOWLEDGE-half of breadth, MULTI-LEVEL: 2-level taxonomy inheritance over real-corpus-discovered
hierarchy. The single-level rungs inherit within a flat category; this inherits ACROSS taxonomy levels:
a never-taught SUB-category inherits its SUPER-category's property (EMERGE-27-style, over a hierarchy
DISCOVERED from a real corpus by hierarchical clustering).

Mechanism:
  * Discover vocab + codes (breadth) -> cosine k-means into FINE clusters -> cluster the fine centroids
    into COARSE super-clusters (a 2-level taxonomy: word -> fine-cluster -> super-cluster), NO labels.
  * Teach a SUPER-category property to the members of SOME fine-clusters in a super-cluster (the TAUGHT
    fine-clusters); HOLD OUT an entire DIFFERENT fine-cluster of the same super-cluster.
  * TEST: does a member of the HELD-OUT fine-cluster inherit its super-cluster's property (argmax over
    super properties)? -- purely via the super-cluster structure (its fine-cluster was NEVER taught).
This is harder than single-level (whole sub-categories held out), and is genuine taxonomic
generalization: a never-seen sub-category inherits from its super-category.

Anti-cheat: super-label DERANGEMENT (shuffle super assignments) collapses it. Reuse-by-import of the
breadth discovery + the emergent-cluster k-means + rung-1's associative-memory read. NO sim/ edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners._realcorpus_inheritance_emergent_clusters_derisk import _kmeans
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows
from research.runners.corpus_stream import load_token_stream_multi


def _teach_test(codes, taught_by_super, queries, P, super_ids):
    """Associative memory over taught members (bound to their SUPER property); a query inherits the super
    whose taught members its code is most similar to. Returns argmax-accuracy over (query, true_super)."""
    U = _unit_rows(codes)
    M = np.zeros((codes.shape[1], P.shape[1]))
    for s, members in taught_by_super.items():
        idx = super_ids.index(s)
        for r in members:
            M += np.outer(U[r], P[idx])
    correct = 0
    for q, true_s in queries:
        phat = U[q] @ M
        pred = super_ids[int(np.argmax(P @ phat))]
        correct += int(pred == true_s)
    return correct / max(1, len(queries))


def run_seed(seed, stories, K, k_fine, k_coarse, verbose=False):
    vocab, gfreq = discover_vocab(stories, K)
    target_set = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    U = _unit_rows(codes)

    fine = _kmeans(codes, k_fine, seed)
    # fine-cluster centroids -> coarse super-clusters
    fine_ids = sorted(set(fine.tolist()))
    cent = np.stack([U[fine == c].mean(0) for c in fine_ids])
    coarse_of_fineidx = _kmeans(cent, k_coarse, seed)
    fine_to_super = {fine_ids[i]: int(coarse_of_fineidx[i]) for i in range(len(fine_ids))}
    # group: super -> {fine -> [rows]}
    from collections import defaultdict
    super_fine_rows = defaultdict(lambda: defaultdict(list))
    for r in range(len(vocab)):
        f = int(fine[r]); super_fine_rows[fine_to_super[f]][f].append(r)
    # usable supers: >=2 fine-clusters each with >=3 members (so 1 held-out fine + >=1 taught fine)
    usable = {}
    for s, fr in super_fine_rows.items():
        big = {f: rows for f, rows in fr.items() if len(rows) >= 3}
        if len(big) >= 2:
            usable[s] = big
    if len(usable) < 2:
        return None
    super_ids = sorted(usable.keys())
    rng = np.random.RandomState(seed)
    P = rng.randn(len(super_ids), 64)

    taught_by_super, queries = {}, []
    for s in super_ids:
        fines = list(usable[s]); rng.shuffle(fines)
        held_fine = fines[0]                      # HOLD OUT an entire fine-cluster
        taught_fines = fines[1:]
        taught_rows = [r for f in taught_fines for r in usable[s][f]]
        taught_by_super[s] = taught_rows
        for r in usable[s][held_fine]:
            queries.append((r, s))                # its members must inherit s via the super structure

    ml_acc = _teach_test(codes, taught_by_super, queries, P, super_ids)
    # DERANGE the super labels
    der_accs = []
    for _ in range(5):
        perm = list(super_ids); rng.shuffle(perm)
        der_map = {super_ids[i]: perm[i] for i in range(len(super_ids))}
        d_taught = {der_map[s]: taught_by_super[s] for s in super_ids}
        d_queries = [(q, der_map[s]) for q, s in queries]
        der_accs.append(_teach_test(codes, d_taught, d_queries, P, super_ids))
    der = float(np.mean(der_accs))
    chance = 1.0 / len(super_ids)

    if verbose:
        print(f"    discovered 2-level taxonomy: {k_fine} fine -> {len(super_ids)} usable super-clusters")
        for s in super_ids[:4]:
            fines = list(usable[s])
            ex = [vocab[usable[s][fines[0]][0]], vocab[usable[s][fines[-1]][0]]]
            print(f"      super {s}: {len(fines)} fine-clusters, e.g. members {ex}")
    return {"seed": seed, "n_supers": len(super_ids), "chance": chance,
            "multilevel_inherit_acc": ml_acc, "deranged_acc": der, "n_queries": len(queries)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--k-fine", type=int, default=20)
    ap.add_argument("--k-coarse", type=int, default=5)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--margin", type=float, default=0.12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[multilevel inheritance] corpus={a.corpus_path} stories={len(stories)} K={a.K} "
          f"k_fine={a.k_fine} k_coarse={a.k_coarse}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a.k_fine, a.k_coarse, verbose=(s == seeds[0]))
        if r is None:
            print(f"  [seed {s}] too few usable super-clusters -- skip", flush=True); continue
        recs.append(r)
        print(f"  [seed {s}] MULTI-LEVEL inherit (held-out SUB-category -> SUPER property)={r['multilevel_inherit_acc']:.3f} | "
              f"deranged={r['deranged_acc']:.3f} | chance={r['chance']:.3f} (supers={r['n_supers']}, nq={r['n_queries']})",
              flush=True)
    if not recs:
        print("  VERDICT: NOT-EVALUABLE"); return
    def m(k): return float(np.mean([r[k] for r in recs]))
    ml, der, ch = m("multilevel_inherit_acc"), m("deranged_acc"), m("chance")
    bc = all(r["multilevel_inherit_acc"] - r["chance"] > a.margin for r in recs)
    bd = all(r["multilevel_inherit_acc"] - r["deranged_acc"] > a.margin for r in recs)
    go = bc and bd
    print(f"\n  AGGREGATE ({len(recs)} seeds): MULTI-LEVEL inherit={ml:.3f} | deranged={der:.3f} | chance={ch:.3f}",
          flush=True)
    print(f"  beats_chance={bc} | beats_deranged={bd}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'NEGATIVE'} -- a member of a NEVER-TAUGHT sub-category "
          f"{'INHERITS its SUPER-category property via the discovered 2-level taxonomy' if go else 'does NOT clearly inherit across levels'} "
          f"{'(above chance + super-derangement) -> multi-level taxonomic generalization over a real-corpus-discovered hierarchy' if go else ''}.",
          flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "K": a.K, "k_fine": a.k_fine, "k_coarse": a.k_coarse,
                   "aggregate": {"multilevel": ml, "deranged": der, "chance": ch}, "per_seed": recs},
                  open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
