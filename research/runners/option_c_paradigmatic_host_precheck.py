"""Option C fairer de-risk -- STAGE A (CPU-numpy): the PARADIGMATIC host pre-check that GATES the GPU.

Design: docs/plans/2026-06-14-option-c-fairer-derisk-design.md. The prior Option-C host ceiling was a
target x target FIRST-ORDER (syntagmatic) measure -> it could not recover the paradigmatic taxonomy, so the
de-risk was inconclusive. This builds the correct SECOND-ORDER measure: a target x FULL-context PPMI matrix
(rows = each target word's distribution over a large context vocabulary, NOT restricted to the targets) ->
truncated SVD -> cosine of the row vectors = paradigmatic similarity. Sweeps the standard knobs and asks:
does ANY setting clear the host gate (Pearson(sim, S_true) >= 0.50)? If yes -> Option C is viable (a GPU
brain-based fair test is warranted, owner-directed); if no -> NEGATIVE_data_too_syntagmatic (close the
cheap-local Option-C question for free). NO GPU, NO sim/ edits, reuse-by-import for the taxonomy.

Usage: python -m research.runners.option_c_paradigmatic_host_precheck
"""
from __future__ import annotations
import re, sys
from collections import Counter
import numpy as np

from research.runners.option_c_real_cooccurrence_derisk import TAXONOMY_8x8, taxonomy_to_vocab_categories

CORPUS = "data/corpus/tinystories.txt"
STOPLIST = set("the a an and to of in is it he she they you i we was were that this his her him "
               "for on with as at by be are had have has not but so all one out up".split())


def build_target_context_counts(tokens, vocab, ctx_words, window):
    tgt_row = {w: i for i, w in enumerate(vocab)}
    ctx_col = {w: j for j, w in enumerate(ctx_words)}
    M = np.zeros((len(vocab), len(ctx_words)), dtype=np.float64)
    n = len(tokens)
    for i, t in enumerate(tokens):
        r = tgt_row.get(t)
        if r is None:
            continue
        lo, hi = max(0, i - window), min(n, i + window + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            c = ctx_col.get(tokens[j])
            if c is not None:
                M[r, c] += 1.0
    return M


def ppmi_svd_sim(M, svd_dim, alpha):
    # PPMI with optional context-distribution smoothing (alpha<1 sharpens paradigmatic recovery; Levy-Goldberg)
    row_sum = M.sum(1, keepdims=True)
    col_sum = M.sum(0, keepdims=True)
    if alpha != 1.0:
        col_sum = (col_sum ** alpha)
    total = col_sum.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * total) / (row_sum * col_sum + 1e-12) + 1e-12)
    ppmi = np.maximum(pmi, 0.0)
    k = min(svd_dim, min(ppmi.shape) - 1)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    emb = U[:, :k] * S[:k]
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb @ emb.T


def score(sim, labels):
    Nt = len(labels)
    S_true = (labels[:, None] == labels[None, :]).astype(np.float64)
    iu = np.triu_indices(Nt, k=1)
    pearson = float(np.corrcoef(sim[iu], S_true[iu])[0, 1])
    within = sim[iu][S_true[iu] == 1.0]; between = sim[iu][S_true[iu] == 0.0]
    margin = float(within.mean() - between.mean())
    s2 = sim.copy(); np.fill_diagonal(s2, -2.0); nn = s2.argmax(1)
    nn_same = float(np.mean([labels[i] == labels[nn[i]] for i in range(Nt)]))
    return pearson, margin, nn_same, nn


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    labels = np.asarray(cat_ids, dtype=int)
    txt = open(CORPUS, encoding="utf-8", errors="ignore").read().lower()
    tokens = re.findall(r"[a-z]+", txt)
    freq = Counter(tokens)
    print(f"[Option-C STAGE-A paradigmatic host pre-check] {len(tokens):,} tokens, {len(vocab)} targets, "
          f"8 categories\n  gate: host Pearson(sim,S_true) >= 0.50 (the prior FIRST-ORDER host was +0.13)")
    best = None
    # build the target x context matrix once per (window, ctx_size); sweep svd_dim x alpha on it (cheap).
    for window in (2, 3, 5, 10):
        for ctx_size in (5000, 10000):
            ctx_words = [w for w, _ in freq.most_common(ctx_size + len(STOPLIST)) if w not in STOPLIST][:ctx_size]
            M = build_target_context_counts(tokens, vocab, ctx_words, window)
            for svd_dim in (50, 100, 200, 300):
                for alpha in (1.0, 0.75):
                    sim = ppmi_svd_sim(M, svd_dim, alpha)
                    pearson, margin, nn_same, nn = score(sim, labels)
                    tag = f"w={window} ctx={ctx_size} svd={svd_dim} a={alpha}"
                    if best is None or pearson > best[0]:
                        best = (pearson, margin, nn_same, tag, nn)
                    print(f"    {tag:34s} Pearson={pearson:+.3f} margin={margin:+.3f} nn-same={nn_same:.3f}")
    pearson, margin, nn_same, tag, nn = best
    print(f"\n  BEST: {tag}  Pearson={pearson:+.3f} margin={margin:+.3f} nn-same={nn_same:.3f}")
    # per-category recovery at the best setting (the Option-2 subset selector)
    print("  per-category nearest-neighbour same-category rate (host-recovered if high):")
    for ci, cn in enumerate(cat_names):
        members = [k for k in range(len(vocab)) if labels[k] == ci]
        rec = float(np.mean([labels[nn[k]] == ci for k in members]))
        print(f"    {cn:14s}: {rec:.3f}")
    host_passes = pearson >= 0.50
    print(f"\n  >>> HOST GATE Pearson>=0.50: {'PASS' if host_passes else 'FAIL'} (best {pearson:+.3f})")
    print("  VERDICT:", "Option C VIABLE -> a GPU brain-based fair test is warranted (owner-directed)"
          if host_passes else
          "NEGATIVE_data_too_syntagmatic-CANDIDATE -> even the tuned 2nd-order host falls short; "
          "the cheap-local Option-C is near-closed (the recovered-category SUBSET may still be fair)")


if __name__ == "__main__":
    main()
