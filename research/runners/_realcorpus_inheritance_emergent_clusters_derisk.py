"""KNOWLEDGE-half of breadth, PROBE-FREE extension: inheritance over FULLY-EMERGENT categories.

Rungs 1-4 defined the categories with the hand-labeled TAXONOMY_8x8 probe -- the last hand-designed
scaffold in the inheritance pipeline (everything else -- vocab discovery, category structure, the
inheritance -- is emergent). This removes it: the categories are DISCOVERED by clustering the
real-corpus co-occurrence codes (k-means, NO a-priori labels), and inheritance rides the discovered
clusters. Master-directive-aligned (emergent, not hand-designed) AND corpus-agnostic (no probe to
mismatch -- works on TinyStories AND WikiText).

Test: cluster the discovered vocab's codes -> for each cluster with >=4 members, teach a distinct
property to HALF, test whether a HELD-OUT cluster member inherits ITS cluster's property (argmax over
cluster properties). Anti-cheat: label-DERANGEMENT (shuffle cluster assignments) collapses it. Report
example clusters + their coherence so the emergent semantics are legible.

Reuse-by-import: breadth discovery + rung-1 inheritance. numpy-only (simple k-means), offline. NO sim/ edit.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners._realcorpus_inheritance_rung1_derisk import _unit_rows, _inherit_accuracy, _build_splits
from research.runners.corpus_stream import load_token_stream_multi


def _kmeans(X, k, seed, iters=50):
    """Simple cosine k-means on unit-normed rows (deterministic given seed)."""
    rng = np.random.RandomState(seed)
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cent = U[rng.choice(len(U), k, replace=False)].copy()
    labels = np.zeros(len(U), int)
    for _ in range(iters):
        sim = U @ cent.T
        new = sim.argmax(1)
        if np.array_equal(new, labels) and _ > 0:
            labels = new; break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                v = U[m].mean(0); cent[c] = v / (np.linalg.norm(v) + 1e-12)
    return labels


def run_seed(seed, stories, K, n_clusters, verbose=False):
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

    labels = _kmeans(codes, n_clusters, seed)
    # keep clusters with >= 4 members
    from collections import Counter
    cnt = Counter(labels.tolist())
    keep = sorted([c for c, n in cnt.items() if n >= 4])
    if len(keep) < 2:
        return None
    # cluster coherence (within-cluster mean cosine) + example words
    clusters = {c: [i for i in range(len(vocab)) if labels[i] == c] for c in keep}

    def coh(rows):
        if len(rows) < 2:
            return 0.0
        S = U[rows] @ U[rows].T
        iu = np.triu_indices(len(rows), 1)
        return float(S[iu].mean())
    cluster_coh = {c: coh(clusters[c]) for c in keep}
    # rank clusters by coherence; use the top ones (the semantically-tight discovered groups)
    ranked = sorted(keep, key=lambda c: -cluster_coh[c])

    rng = np.random.RandomState(seed)
    usable = {c: clusters[c] for c in ranked}
    cat_ids = list(usable.keys())
    P = rng.randn(len(cat_ids), 64)
    taught_by_cat, heldout_q, taught_q = _build_splits(usable, cat_ids, rng)
    ho_acc, n_ho = _inherit_accuracy(codes, taught_by_cat, heldout_q, P, cat_ids)

    # derangement
    all_rows = [r for rs in usable.values() for r in rs]
    der_accs = []
    for _ in range(5):
        pool = list(all_rows); rng.shuffle(pool)
        der = {}; i = 0
        for c in cat_ids:
            n = len(usable[c]); der[c] = pool[i:i + n]; i += n
        d_t, d_h, _ = _build_splits(der, cat_ids, rng)
        da, _ = _inherit_accuracy(codes, d_t, d_h, P, cat_ids)
        der_accs.append(da)

    if verbose:
        print(f"    discovered {len(keep)} emergent clusters (>=4 members); top by coherence:")
        for c in ranked[:6]:
            ws = [vocab[i] for i in clusters[c]][:9]
            print(f"      cluster {c} (coh {cluster_coh[c]:+.3f}, n={len(clusters[c])}): {ws}")
    return {"seed": seed, "n_clusters_used": len(cat_ids), "chance": 1.0 / len(cat_ids),
            "heldout_inherit_acc": ho_acc, "deranged_acc": float(np.mean(der_accs)), "n_heldout": n_ho,
            "mean_cluster_coherence": float(np.mean(list(cluster_coh.values())))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    stories = load_token_stream_multi(a.corpus_path, max_stories=None)
    print(f"[emergent-clusters inheritance] corpus={a.corpus_path} stories={len(stories)} "
          f"tokens={sum(len(s) for s in stories)} K={a.K} n_clusters={a.n_clusters}", flush=True)

    recs = []
    for s in seeds:
        r = run_seed(s, stories, a.K, a.n_clusters, verbose=(s == seeds[0]))
        if r is None:
            continue
        recs.append(r)
        print(f"  [seed {s}] EMERGENT-cluster held-out inherit={r['heldout_inherit_acc']:.3f} | "
              f"deranged={r['deranged_acc']:.3f} | chance={r['chance']:.3f} | "
              f"mean-cluster-coh={r['mean_cluster_coherence']:+.3f} (clusters={r['n_clusters_used']})", flush=True)

    if not recs:
        print("  VERDICT: NOT-EVALUABLE"); return
    def m(k): return float(np.mean([r[k] for r in recs]))
    ho, der, ch = m("heldout_inherit_acc"), m("deranged_acc"), m("chance")
    beats_chance = all(r["heldout_inherit_acc"] - r["chance"] > a.margin for r in recs)
    beats_der = all(r["heldout_inherit_acc"] - r["deranged_acc"] > a.margin for r in recs)
    go = beats_chance and beats_der
    print(f"\n  AGGREGATE ({len(recs)} seeds): held-out inherit={ho:.3f} | deranged={der:.3f} | chance={ch:.3f}",
          flush=True)
    print(f"  beats_chance={beats_chance} | beats_deranged={beats_der}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'NEGATIVE'} -- a held-out member of a FULLY-EMERGENT (clustered, "
          f"NO hand-labeled probe) category {'INHERITS its cluster property' if go else 'does NOT clearly inherit'} "
          f"{'above chance + derangement -> inheritance rides categories DISCOVERED by clustering, the last hand-designed probe removed' if go else ''}.",
          flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "K": a.K, "n_clusters": a.n_clusters,
                   "aggregate": {"heldout": ho, "deranged": der, "chance": ch}, "per_seed": recs},
                  open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
