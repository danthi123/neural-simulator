"""The SECOND brain-grounded lever for "learn naturally over time": NONNEGATIVE similarity-matching (the
brain's firing rates are >= 0) does CLUSTERING where LINEAR SM does PCA -- and the task target is CATEGORICAL.

THE INSIGHT (why this is the right tool, not a knob): the real-corpus target S_true is the same-CATEGORY block
matrix (8 categories). LINEAR similarity-matching (the validated learn_simmatch: signed y, Oja-W +
anti-Hebbian-M) converges to PCA -- the top-k PRINCIPAL COMPONENTS. PCA is the WRONG objective for a
CATEGORICAL target: the offline PPMI+PCA(64) optimum is only +0.518 because PCA spreads category structure
across continuous components. NONNEGATIVE similarity-matching (Pehlevan-Chklovskii 2018; Hu-Pehlevan-Chklovskii
2014 -- rectify y >= 0 in the settle) instead does manifold/CLUSTERING decomposition, which matches a
categorical block target much better. AND nonnegative y is the brain-FAITHFUL rate code (firing rates can't be
negative), so this is strictly MORE biological than the signed linear SM, not less.

So this de-risks: does the BRAIN-FAITHFUL nonnegative rate rule (clustering) reach a HIGHER categorical ceiling
than the signed linear PCA rule -- i.e. was the +0.518 "offline optimum" itself the WRONG (PCA) ceiling for a
categorical task? Reuse-by-import of the VALIDATED learn_simmatch for the linear baseline; the ONLY change in
the NSM arm is the relu in the settle (the nonnegativity), holding W/M rules identical.

THE ARMS (same PPMI input, same timescale separation, multi-seed):
  HOST            PPMI+SVD                         -- the project's ceiling (+0.44).
  OFFLINE_PCA     offline PPMI+PCA(k)              -- the LINEAR (PCA) offline optimum (+0.52).
  OFFLINE_KMEANS  offline k-means(8) same-cluster  -- the CATEGORICAL/clustering ceiling (could be >> +0.52).
  LINEAR_SM       online signed SM (learn_simmatch)-- the validated PCA-rule baseline (the +0.30-0.35 regime).
  NSM             online nonnegative SM (relu y)   -- *** the brain-faithful clustering rule under test ***.

THE FORK:
  NSM_WINS  : NSM beats LINEAR_SM and approaches OFFLINE_KMEANS ==> the brain-faithful nonnegative rate rule
              does the categorical clustering the signed PCA rule can't; the "+0.518 ceiling" was the wrong
              (PCA) target -- the categorical ceiling is higher and the brain-rule reaches it. A genuinely
              new, hopeful answer to the owner.
  NSM_EVEN  : NSM ~= LINEAR_SM ==> nonnegativity alone doesn't unlock the categorical structure online (the
              local rule still can't separate the categories) -> the locality wall stands (the (B) fork).
Anti-cheat: OFFLINE_KMEANS is the same-input clustering reference (a high value shows the structure IS
recoverable, so a low NSM is the rule's limit not the data's); permuted ~0; eff-rank reported.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_nonnegative_sm_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, effective_rank,
)
from research.runners.learned_graded_cortex_fair_test import (  # noqa: E402
    build_real_corpus, ppmi_matrix, pca_lowrank_sim, learn_simmatch,
)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402

N_HUB = 500
K = 64
ALPHA = 0.75
SEEDS = (42, 43, 44)
# the timescale-separated regime (the SM theory's fast-lateral requirement; settled by the companion probe):
LR_FF, LR_M, SETTLE, EPOCHS = 0.005, 0.100, 40, 400


def learn_nonnegative_sm(X, S_true, k, epochs, lr_ff, lr_m, settle_steps, seed, track_every=0):
    """NONNEGATIVE similarity-matching: IDENTICAL to learn_simmatch EXCEPT y is rectified (relu) at each settle
    step -- the brain-faithful nonnegative rate code. The rectified dynamics do clustering, not PCA."""
    rng = np.random.RandomState(seed * 104729 + 3)
    Nc, H = X.shape
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W_ff = rng.randn(k, H) * 0.1
    M = np.zeros((k, k), dtype=np.float64)
    order = np.arange(Nc)
    traj = []

    def settle(ff):
        y = np.zeros(k)
        for _ in range(settle_steps):
            y = np.maximum(0.5 * y + 0.5 * (ff - M @ y), 0.0)   # the ONLY change vs linear SM: relu (y >= 0)
        return y

    def read_codes():
        out = np.zeros((Nc, k))
        for i in range(Nc):
            out[i] = settle(W_ff @ Xn[i])
        return out

    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = Xn[i]
            y = settle(W_ff @ x)
            W_ff += lr_ff * (np.outer(y, x) - (y ** 2)[:, None] * W_ff)
            dM = np.outer(y, y) - M
            np.fill_diagonal(dM, 0.0)
            M += lr_m * dM
        if track_every and (ep + 1) % track_every == 0:
            traj.append((ep + 1, float(_pearson_vs_Strue(_cos_sim(read_codes()), S_true))))
    return read_codes(), traj


def offline_kmeans_sim(X, n_clusters, seed, n_iter=100):
    """Offline k-means on the (row-normalized) PPMI rows -> same-cluster block matrix. The CATEGORICAL ceiling
    (if clustering recovers the categories this is near +1; it shows the categorical structure IS recoverable)."""
    rng = np.random.RandomState(seed * 7 + 11)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Nc = Xn.shape[0]
    cent = Xn[rng.choice(Nc, n_clusters, replace=False)].copy()
    assign = np.zeros(Nc, dtype=int)
    for _ in range(n_iter):
        d = ((Xn[:, None, :] - cent[None, :, :]) ** 2).sum(2)
        new = d.argmin(1)
        if np.array_equal(new, assign):
            break
        assign = new
        for c in range(n_clusters):
            m = assign == c
            if m.any():
                cent[c] = Xn[m].mean(0)
    return (assign[:, None] == assign[None, :]).astype(np.float64)


def run_seed(seed):
    C, labels, S_true = build_real_corpus(seed, N_HUB)
    Nc, H = C.shape
    svd_dim = min(50, min(C.shape) - 1)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=svd_dim, alpha=ALPHA), labels)
    Xppmi = ppmi_matrix(C, ALPHA)
    offline_pca = _pearson_vs_Strue(pca_lowrank_sim(Xppmi, K), S_true)
    n_cat = len(np.unique(labels))
    offline_km = _pearson_vs_Strue(offline_kmeans_sim(Xppmi, n_cat, seed), S_true)
    print(f"\n[NSM seed {seed}] {Nc}c x {H}h, {n_cat} categories | host {host_p:+.3f} | offline PCA {offline_pca:+.3f}"
          f" | offline k-means({n_cat}) {offline_km:+.3f} (the CATEGORICAL ceiling)", flush=True)

    lin_codes, _, lin_traj = learn_simmatch(Xppmi, S_true, K, EPOCHS, LR_FF, LR_M, SETTLE, seed,
                                            track_every=max(1, EPOCHS // 4))
    lin_p = _pearson_vs_Strue(_cos_sim(lin_codes), S_true); lin_r = effective_rank(lin_codes)
    nsm_codes, nsm_traj = learn_nonnegative_sm(Xppmi, S_true, K, EPOCHS, LR_FF, LR_M, SETTLE, seed,
                                               track_every=max(1, EPOCHS // 4))
    nsm_p = _pearson_vs_Strue(_cos_sim(nsm_codes), S_true); nsm_r = effective_rank(nsm_codes)
    print(f"  LINEAR_SM (PCA, signed) : Pearson={lin_p:+.3f}  eff-rank={lin_r:.1f}   traj "
          + "  ".join(f"{p:+.2f}" for _, p in lin_traj), flush=True)
    print(f"  NSM (clustering, relu>=0): Pearson={nsm_p:+.3f}  eff-rank={nsm_r:.1f}   traj "
          + "  ".join(f"{p:+.2f}" for _, p in nsm_traj), flush=True)
    # anti-cheat: permuted on the NSM arm.
    rng = np.random.RandomState(seed * 2718281 + 5)
    perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    nsm_perm = _pearson_vs_Strue(_cos_sim(nsm_codes), S_perm)
    print(f"  [anti-cheat] NSM permuted={nsm_perm:+.3f} (~0)", flush=True)
    return {"seed": seed, "host": host_p, "offline_pca": offline_pca, "offline_kmeans": offline_km,
            "linear_sm": lin_p, "linear_rank": lin_r, "nsm": nsm_p, "nsm_rank": nsm_r, "nsm_permuted": nsm_perm}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[nonnegative-SM de-risk] seeds={SEEDS} n_hub={N_HUB} k={K} regime lr_ff={LR_FF} lr_m={LR_M} "
          f"settle={SETTLE} epochs={EPOCHS} -- does the brain-faithful NONNEGATIVE rate rule (clustering) beat "
          f"the signed LINEAR (PCA) rule on the CATEGORICAL target?", flush=True)
    rows = [run_seed(s) for s in SEEDS]

    def m(key):
        return float(np.mean([r[key] for r in rows]))
    host, pca, km = m("host"), m("offline_pca"), m("offline_kmeans")
    lin, nsm = m("linear_sm"), m("nsm")
    lin_r, nsm_r = m("linear_rank"), m("nsm_rank")
    print(f"\n{'='*96}\n  MEAN ({len(SEEDS)} seeds): host {host:+.3f} | offline PCA {pca:+.3f} | "
          f"offline k-means {km:+.3f} (CATEGORICAL ceiling)", flush=True)
    print(f"  LINEAR_SM (PCA)   {lin:+.3f}  eff-rank {lin_r:.1f}", flush=True)
    print(f"  NSM (clustering)  {nsm:+.3f}  eff-rank {nsm_r:.1f}", flush=True)
    print(f"{'='*96}", flush=True)
    lift = nsm - lin
    if lift >= 0.04 and nsm >= 0.70 * km:
        print(f"  NSM_WINS: the brain-faithful nonnegative rate rule (clustering) BEATS the signed linear PCA "
              f"rule (+{lift:.3f}: {lin:+.3f} -> {nsm:+.3f}) and approaches the CATEGORICAL ceiling "
              f"(k-means {km:+.3f}, {nsm/km:.0%}). ==> the '+0.518 offline optimum' was the WRONG (PCA) target "
              f"for a categorical task; the categorical ceiling is {km:+.3f} and the brain-rule (NSM) reaches "
              f"toward it. The marginality was the WRONG-OBJECTIVE artifact -- a genuinely new, hopeful answer: "
              f"nonnegativity (the brain's rate code) IS what we were missing.", flush=True)
    elif lift >= 0.04:
        print(f"  NSM_HELPS: NSM beats LINEAR_SM (+{lift:.3f}) -- nonnegativity helps the categorical structure "
              f"-- but stays below the categorical ceiling (k-means {km:+.3f}). Real gain; push the regime/scale "
              f"or stack with the timescale separation.", flush=True)
    else:
        print(f"  NSM_EVEN: NSM ~= LINEAR_SM ({nsm:+.3f} vs {lin:+.3f}) -- nonnegativity alone does NOT unlock "
              f"the categorical structure online (the local rule still can't separate the categories, even "
              f"though offline k-means {km:+.3f} shows they're recoverable) ==> the online-local-vs-offline "
              f"separation gap stands (the point-neuron locality wall, the (B) fork).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host": host, "offline_pca": pca, "offline_kmeans": km, "linear_sm": lin, "nsm": nsm,
           "linear_rank": lin_r, "nsm_rank": nsm_r, "lift": lift, "per_seed": rows}
    raw_dir = os.path.join(_REPO, "research", "findings", "raw")
    path = os.path.join(raw_dir, "_phaseB_nonnegative_sm.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
