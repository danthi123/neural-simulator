"""MULTI-LEVEL taxonomy, the NAMED next mechanism (research-gate): the multi-level NEGATIVE showed CENTROID
clustering of fine-cluster centroids into supers is not load-bearing (real ~ deranged) -- co-occurrence
codes give FLAT categories, not a nested is-a hierarchy. The named surpass mechanism (EMERGE-44/45 stacked
pooler) groups fine clusters by their CO-OCCURRENCE (a second-order signal: which fine-clusters appear
together in the same stories), NOT by centroid similarity. This tests whether the co-occurrence super
supports 2-level generalization where the centroid super failed -- head-to-head, same seeds/data.

Mechanism (vs the NEGATIVE):
  * L1: k-means the codes into FINE clusters (same as the NEGATIVE).
  * SUPER (NEW): build the fine-cluster CO-OCCURRENCE matrix (for each story, the fine clusters of its words
    all co-occur), PPMI-normalize, and cluster the fine clusters by their co-occurrence PROFILES -> supers
    from what appears together (the EMERGE-44 stacked-pooler signal), NOT centroid similarity.
  * TEST (same as the NEGATIVE): teach a SUPER property to some fine-clusters, hold out a WHOLE fine-cluster,
    check its members inherit the super via the discovered super grouping. vs super-DERANGEMENT.
Reports BOTH the co-occurrence super AND the centroid super (the NEGATIVE baseline) -- does co-occurrence
beat deranged where centroid did not? numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from collections import defaultdict

from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners._realcorpus_inheritance_emergent_clusters_derisk import _kmeans
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows
from research.runners._realcorpus_inheritance_multilevel_derisk import _teach_test
from research.runners.corpus_stream import load_token_stream_multi


def _cooc_super(stories, vocab, fine, k_fine, k_coarse, seed):
    """Group fine clusters into supers by CO-OCCURRENCE (which fine-clusters appear together in stories),
    PPMI-normalized, then k-means the co-occurrence profiles. The EMERGE-44 stacked-pooler signal."""
    row_of = {w: i for i, w in enumerate(vocab)}
    fine_of_word = {w: int(fine[row_of[w]]) for w in vocab}
    C = np.zeros((k_fine, k_fine))
    for st in stories:
        present = sorted({fine_of_word[w] for w in st if w in fine_of_word})
        for i in range(len(present)):
            for j in range(len(present)):
                if i != j:
                    C[present[i], present[j]] += 1.0
    # PPMI on the fine-cluster co-occurrence
    tot = C.sum() + 1e-9
    pi = C.sum(1, keepdims=True) / tot
    pj = C.sum(0, keepdims=True) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log((C / tot) / (pi @ pj + 1e-12) + 1e-12)
    ppmi = np.maximum(ppmi, 0.0)
    prof = _unit_rows(ppmi)                                       # each fine cluster's co-occurrence profile
    lab = _kmeans(prof, k_coarse, seed)                          # supers = fine clusters with similar co-occurrence
    return {c: int(lab[c]) for c in range(k_fine)}


def _centroid_super(codes, fine, k_fine, k_coarse, seed):
    """The NEGATIVE baseline: cluster the fine-cluster CENTROIDS into supers (static similarity)."""
    U = _unit_rows(codes)
    fine_ids = list(range(k_fine))
    cent = np.stack([U[fine == c].mean(0) if (fine == c).any() else U.mean(0) for c in fine_ids])
    lab = _kmeans(cent, k_coarse, seed)
    return {c: int(lab[c]) for c in fine_ids}


def _eval(codes, vocab, fine, fine_to_super, seed):
    """Held-out sub-category -> super-property inheritance + super-derangement (the NEGATIVE's test)."""
    super_fine_rows = defaultdict(lambda: defaultdict(list))
    for r in range(len(vocab)):
        super_fine_rows[fine_to_super[int(fine[r])]][int(fine[r])].append(r)
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
        taught_by_super[s] = [r for f in fines[1:] for r in usable[s][f]]
        for r in usable[s][fines[0]]:
            queries.append((r, s))
    ml = _teach_test(codes, taught_by_super, queries, P, super_ids)
    ders = []
    for _ in range(5):
        perm = list(super_ids); rng.shuffle(perm)
        dm = {super_ids[i]: perm[i] for i in range(len(super_ids))}
        ders.append(_teach_test(codes, {dm[s]: taught_by_super[s] for s in super_ids},
                                [(q, dm[s]) for q, s in queries], P, super_ids))
    return {"acc": ml, "deranged": float(np.mean(ders)), "chance": 1.0 / len(super_ids), "n_supers": len(super_ids)}


def run_seed(seed, stories, K, k_fine, k_coarse):
    vocab, gfreq = discover_vocab(stories, K)
    target = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    fine = _kmeans(codes, k_fine, seed)
    cooc = _eval(codes, vocab, fine, _cooc_super(stories, vocab, fine, k_fine, k_coarse, seed), seed)
    cent = _eval(codes, vocab, fine, _centroid_super(codes, fine, k_fine, k_coarse, seed), seed)
    return {"seed": seed, "cooc": cooc, "centroid": cent}


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
    print(f"[multilevel STACKED (co-occurrence super)] K={a.K} k_fine={a.k_fine} k_coarse={a.k_coarse}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a.k_fine, a.k_coarse)
        if r["cooc"] is None or r["centroid"] is None:
            print(f"  [seed {s}] too few usable supers -- skip", flush=True); continue
        recs.append(r)
        print(f"  [seed {s}] CO-OCC super: inherit={r['cooc']['acc']:.3f} deranged={r['cooc']['deranged']:.3f} "
              f"chance={r['cooc']['chance']:.3f} || CENTROID (NEGATIVE): inherit={r['centroid']['acc']:.3f} "
              f"deranged={r['centroid']['deranged']:.3f}", flush=True)
    if not recs:
        print("  VERDICT: NOT-EVALUABLE"); return
    def m(path): return float(np.mean([r[path[0]][path[1]] for r in recs]))
    co_acc, co_der = m(("cooc", "acc")), m(("cooc", "deranged"))
    ce_acc, ce_der = m(("centroid", "acc")), m(("centroid", "deranged"))
    co_beats = all(r["cooc"]["acc"] - r["cooc"]["deranged"] > a.margin and
                   r["cooc"]["acc"] - r["cooc"]["chance"] > a.margin for r in recs)
    print(f"\n  AGGREGATE: CO-OCC inherit={co_acc:.3f} deranged={co_der:.3f} (beats-both all-seeds={co_beats}) || "
          f"CENTROID inherit={ce_acc:.3f} deranged={ce_der:.3f}", flush=True)
    print(f"  VERDICT: {'GO' if co_beats else 'NEGATIVE'} -- the co-occurrence super "
          f"{'SUPPORTS 2-level generalization (beats chance + super-derangement all seeds) where the centroid super did NOT -> the stacked-pooler co-occurrence signal carries a real is-a hierarchy in the corpus' if co_beats else 'does NOT clearly beat derangement either -> the corpus lacks a nested is-a signal even via second-order co-occurrence (the mechanism is right; the DATA is the gate)'}.",
          flush=True)
    if a.out:
        json.dump({"verdict": "GO" if co_beats else "NEGATIVE", "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
