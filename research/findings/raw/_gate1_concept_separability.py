"""Cheap-first GATE 1 (CPU, decisive): is the 28-word "front-end wall" a representation limit or a lossy
READOUT artifact?

The wall was measured as pool-LABEL recognition (argmax over pool-mean firing) = 0.57 / among concept pools
0.54. But the internal asset map established the substrate's 16-concept ACTIVITY is already 100% nearest-
neighbor identifiable (within 0.896 > between 0.768) -- so pool-argmax (which collapses each 200-neuron pool
to a scalar mean, discarding within-pool structure) may be a LOSSY readout, not the true separability.

This gate measures, on the captured multi-sample 28-concept activity (train/test split so the decoder must
GENERALIZE across the substrate's OU-noise, NOT memorize):
  (a) pool-argmax accuracy            -- reproduces the documented wall (~0.54-0.57)
  (b) nearest-centroid on the FULL per-neuron code (cosine)  -- the readout-free separability
  (c) a LEARNED linear classifier (logistic regression / LDA) -- the best linear decoder

DECISION (pre-registered, compute-protecting):
  - If (b) or (c) >> (a) and clears ~0.80  -> the wall is a LOSSY-READOUT artifact. CHEAP FIX (a better
    decoder); NO 100hr, NO representation learning needed. Best possible outcome for the compute budget.
  - If (b),(c) ~ (a) ~ 0.54              -> genuine representation limit; the internal map's lesson applies
    (operating on the same activity is bounded) -> the 100hr / different-acquisition question is live.
  - Three-state; malformed input -> CANNOT-CONCLUDE, not a crash.

No allow_pickle (numeric arrays only; n_pools derives from the code dimension). Reuse-by-import (sklearn +
the cached npz). No protected-module change; no autograd; no GPU.
Run: python -m research.findings.raw._gate1_concept_separability
"""
from __future__ import annotations
import os
import numpy as np

NPZ = "research/findings/raw/_28concept_activity_seed42.npz"
N_PER_POOL = 200   # each concept pool = 200 neurons, concatenated in pool order (D = n_pools * 200)
TRAIN_FRAC = 0.625  # 10 of 16 samples train, 6 test


def _split(y, seed=0):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        k = int(round(len(idx) * TRAIN_FRAC))
        tr += idx[:k].tolist(); te += idx[k:].tolist()
    return np.array(tr), np.array(te)


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def main():
    if not os.path.exists(NPZ):
        print(f"CANNOT-CONCLUDE: {NPZ} not found (run _capture_28concept_activity first)", flush=True); return
    d = np.load(NPZ)                      # numeric arrays only -> no allow_pickle needed
    X, y = d["X"].astype(np.float64), d["y"].astype(np.int64)
    pool_of_word = d["pool_of_word"].astype(np.int64)
    D = X.shape[1]
    if D % N_PER_POOL != 0:
        print(f"CANNOT-CONCLUDE: D={D} not divisible by {N_PER_POOL} (pool layout mismatch)", flush=True)
        return
    n_pools = D // N_PER_POOL
    n_words = len(np.unique(y))
    if n_words != n_pools:
        print(f"CANNOT-CONCLUDE: {n_words} words != {n_pools} pools", flush=True); return
    print(f"=== GATE 1: 28-concept separability (N={X.shape[0]}, D={D}, {n_pools} pools/words) ===", flush=True)

    tr, te = _split(y, seed=0)

    # (a) pool-argmax accuracy: collapse each pool to its mean firing, argmax -> pool == word's pool
    pool_means = X.reshape(X.shape[0], n_pools, N_PER_POOL).mean(axis=2)   # [N, n_pools]
    pred_pool = pool_means.argmax(axis=1)
    a_all = float(np.mean(pred_pool == pool_of_word[y]))
    a_te = float(np.mean(pred_pool[te] == pool_of_word[y[te]]))
    print(f"  (a) pool-argmax accuracy:           all {a_all:.3f}   test {a_te:.3f}   (the documented wall; "
          f"front-end probe got 0.571 -- faithfulness check)", flush=True)

    # MEAN-CENTER each code (remove common-mode = the project's ON/OFF / denoiser methodology) for (b),(c)
    Xmc = X - X.mean(axis=1, keepdims=True)

    # (b) nearest-centroid on mean-centered per-neuron code (cosine), train centroids -> classify test by WORD
    Xn = _l2(Xmc)
    cents = _l2(np.stack([Xn[tr][y[tr] == c].mean(axis=0) for c in range(n_pools)]))
    b_te = float(np.mean((Xn[te] @ cents.T).argmax(axis=1) == y[te]))
    print(f"  (b) nearest-centroid (mean-centered):   test {b_te:.3f}   (readout-free word separability)",
          flush=True)

    # (c) learned linear decoders: PCA-reduce (p>>n) then logreg; LDA with Ledoit-Wolf shrinkage
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        npc = min(len(tr) - 1, 150)
        pca = PCA(n_components=npc).fit(Xmc[tr])
        Ztr, Zte = pca.transform(Xmc[tr]), pca.transform(Xmc[te])
        sc = StandardScaler().fit(Ztr)
        lr = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(Ztr), y[tr])
        c_lr = float(lr.score(sc.transform(Zte), y[te]))
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xmc[tr], y[tr])
        c_lda = float(lda.score(Xmc[te], y[te]))
        print(f"  (c) learned linear (PCA-{npc}): logreg {c_lr:.3f}   LDA {c_lda:.3f}   (best linear decoder)",
              flush=True)
    except Exception as e:
        c_lr = c_lda = float("nan")
        print(f"  (c) sklearn unavailable/failed ({e}); skipping learned linear", flush=True)

    best_decode = max(v for v in [b_te, c_lr, c_lda] if v == v)
    chance = 1.0 / n_pools
    print(f"\n  chance = {chance:.3f}; documented wall (pool-argmax test) = {a_te:.3f}; best decoder = "
          f"{best_decode:.3f}", flush=True)

    if best_decode >= 0.80 and best_decode >= a_te + 0.15:
        print("VERDICT: LOSSY-READOUT ARTIFACT -- the 28-concept codes ARE highly separable with a proper "
              "decoder; the pool-argmax wall is a lossy readout, not a representation limit. CHEAP FIX (a "
              "learned/NN decoder over the full code); NO 100hr representation learning needed. Confirm "
              "multi-seed, then ship the decoder.", flush=True)
    elif best_decode >= a_te + 0.15:
        print(f"VERDICT: PARTIAL readout gain (best {best_decode:.2f} vs wall {a_te:.2f}) but below 0.80 -- a "
              "better decoder helps but does not fully clear the wall; both readout AND representation "
              "contribute. Weigh a learned-expansion follow-up + representation learning.", flush=True)
    else:
        print(f"VERDICT: REPRESENTATION LIMIT -- even the best decoder ({best_decode:.2f}) ~ the pool-argmax "
              "wall; the 28-concept codes are genuinely inseparable. The internal map's lesson applies "
              "(transforming the SAME activity is bounded) -> richer reps need different ACQUISITION "
              "(training distribution / larger model) = the owner-strategic question.", flush=True)


if __name__ == "__main__":
    main()
