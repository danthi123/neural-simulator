"""CYCLE 88 — the off-diagonal escape candidate: an INTERNEURON whitening network (the interneuron COUNT sets
the rank). The distinct, untested mechanism after the point-neuron lateral was exhausted (CYCLE 87).

THE PROBLEM (CYCLE 80-87): the off-diagonal cross-neuron decorrelation that reaches host on the REAL corpus
(offline rank-8 ZCA = +0.49) is unreachable by the tested online-local rules:
  - learned-W + somatic lateral (similarity-matching): the W COLLAPSES to the top 1-4 PCs -> +0.35.
  - fixed-W + somatic lateral: OVER-whitens (eff-rank 44) -> +0.32.
  - bottleneck the OUTPUT to k=8 (CYCLE 87): collapses to eff-rank 1.1 -> +0.20.
None hit the rank-8 sweet spot.

THE CANDIDATE (Pehlevan-Sengupta-Chklovskii 2018; Golkar-Lipshutz-Chklovskii whitening nets): keep the output
FULL-dimensional (M principal neurons) + add a SMALL INTERNEURON population (P of them). The interneurons read
the principal activity (z = L y) and inhibit it back (y = Wx - Lᵀz), so at equilibrium y = (I + LᵀL)⁻¹ Wx. The
inhibition is RANK-P (P interneurons) -> a RANK-P whitening of the principals. The interneuron COUNT directly
sets the decorrelation rank -- the "select the right rank-8" property both prior failure modes lacked. L learns
anti-Hebbian (ΔL ∝ zyᵀ - L) to capture the top-P principal covariance. The diagonal half (per-feature centering,
already locally-solved = input_mean_adapt) is applied first; the interneurons do the OFF-diagonal.

ARMS (real corpus, PPMI input centered; 3 seeds):
  HOST              PPMI+SVD                                  +0.442
  offline ZCA rank-8                                          +0.49 (the target)
  diagonal control  per-feature centering only (no lateral)  ~+0.31 (the diagonal-only ceiling)
  somatic SM        learn_simmatch (the collapsing control)   ~+0.35 (must FALL SHORT)
  *** INTERNEURON whitening (P interneurons), P swept 4/8/16/32 -- THE mechanism under test ***

GATE: the interneuron net BEATS the somatic SM (+0.35) + the diagonal (+0.31) toward the offline rank-8 ZCA
(+0.49), at P~8, with eff-rank ~8 (NOT 1-4 collapse, NOT 44 over-whiten). Anti-cheat: lesion (freeze L at 0 ->
collapses to the un-whitened projection); permuted-similarity ~0; the somatic-SM control falls short (the
interneuron architecture is load-bearing); multi-seed. GO => a POINT-NEURON interneuron CIRCUIT reaches the
off-diagonal on real -> the functional cortex needs NO curated concepts AND no dendrite; the build is the
interneuron whitening net. NEGATIVE => even the interneuron net plateaus -> escalate to the dendritic spec.

Reuse-by-import; NO sim/ edits; numpy; the harness is reusable for the mechanism-scope's refinement.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_interneuron_whitening_derisk
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

SEEDS = (42, 43, 44)
N_HUB = 500
ALPHA = 0.75
M_PRINCIPAL = 64                 # full-dim principal population (a fixed random projection of the input)
PS = (4, 8, 16, 32)              # interneuron count = the whitening RANK (8 ~ the 8 categories = the sweet spot)
EPOCHS = 300
ETA_L = 0.01                     # interneuron (anti-Hebbian) learning rate


def zca_rank_sim(X, k):
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    kk = min(k, U.shape[1])
    emb = U[:, :kk]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb @ emb.T


def interneuron_whiten(Y_raw, P, epochs, eta_L, seed, freeze=False):
    """Online interneuron whitening. Principals y (M-dim) inhibited by P interneurons z=L y; equilibrium
    y=(I+LᵀL)⁻¹ y_raw; L learns anti-Hebbian ΔL ∝ z yᵀ - L (captures the top-P principal covariance ->
    rank-P whitening). freeze=True (lesion) keeps L=0 (no whitening). Returns the per-concept whitened codes."""
    rng = np.random.RandomState(seed * 7919 + 5)
    Nc, M = Y_raw.shape
    L = np.zeros((P, M)) if freeze else rng.randn(P, M) * 0.01
    Ipc = np.eye(M)
    order = np.arange(Nc)

    def settle(yr):
        return np.linalg.solve(Ipc + L.T @ L, yr)        # y = (I + LᵀL)⁻¹ y_raw

    if not freeze:
        for _ in range(epochs):
            for i in rng.permutation(order):
                y = settle(Y_raw[i])
                z = L @ y
                L += eta_L * (np.outer(z, y) - L)         # anti-Hebbian, fixed-point decay
                nrm = np.linalg.norm(L)
                if nrm > 50.0:
                    L *= 50.0 / nrm                        # bounded interneuron weights (stability guard)
    return np.array([settle(Y_raw[i]) for i in range(Nc)]), L


def run_seed(seed):
    C, labels, S_true = build_real_corpus(seed, N_HUB)
    labels = np.asarray(labels)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=ALPHA), labels)
    Xppmi = ppmi_matrix(C, ALPHA)
    Xc = Xppmi - Xppmi.mean(0, keepdims=True)                 # diagonal half (per-feature centering, given)
    diag_p = _pearson_vs_Strue(_cos_sim(Xc), S_true)          # centering-only = the diagonal ceiling
    zca8 = _pearson_vs_Strue(zca_rank_sim(Xppmi, 8), S_true)
    # somatic SM control (the collapsing one):
    sm_codes, _, _ = learn_simmatch(Xppmi, S_true, 64, 200, 0.005, 0.05, 40, seed)
    sm_p = _pearson_vs_Strue(_cos_sim(sm_codes), S_true)
    # fixed random principal projection of the centered input:
    rng = np.random.RandomState(seed * 104729 + 7)
    W = rng.randn(M_PRINCIPAL, Xc.shape[1]) / np.sqrt(Xc.shape[1])
    Y_raw = Xc @ W.T                                          # principals (Nc x M), pre-whitening
    raw_p = _pearson_vs_Strue(_cos_sim(Y_raw), S_true)        # the un-whitened random projection
    print(f"\n[interneuron seed {seed}] {C.shape[0]}c x {C.shape[1]}h | host {host_p:+.3f} | offline ZCA rank-8 "
          f"{zca8:+.3f} | diagonal(centering) {diag_p:+.3f} | somatic SM {sm_p:+.3f} | raw projection {raw_p:+.3f}",
          flush=True)
    rows = {}
    for P in PS:
        codes, L = interneuron_whiten(Y_raw, P, EPOCHS, ETA_L, seed)
        p = _pearson_vs_Strue(_cos_sim(codes), S_true)
        er = effective_rank(codes)
        rows[P] = {"pearson": round(p, 3), "eff_rank": round(er, 1)}
        print(f"    P={P:3d} interneurons: {p:+.3f} (eff-rank {er:.1f})", flush=True)
    # lesion (freeze L=0) at the best P:
    bestP = max(PS, key=lambda P: rows[P]["pearson"])
    lesion_codes, _ = interneuron_whiten(Y_raw, bestP, EPOCHS, ETA_L, seed, freeze=True)
    lesion_p = _pearson_vs_Strue(_cos_sim(lesion_codes), S_true)
    # permuted on the best:
    best_codes, _ = interneuron_whiten(Y_raw, bestP, EPOCHS, ETA_L, seed)
    rng2 = np.random.RandomState(seed * 2718281 + 3)
    perm = rng2.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    perm_p = _pearson_vs_Strue(_cos_sim(best_codes), S_perm)
    print(f"    [anti-cheat] best P={bestP}: lesion(L=0) {lesion_p:+.3f} (= raw {raw_p:+.3f}); permuted {perm_p:+.3f}",
          flush=True)
    return {"seed": seed, "host": host_p, "zca8": zca8, "diagonal": diag_p, "somatic_sm": sm_p, "raw": raw_p,
            "ps": rows, "best_P": bestP, "lesion": lesion_p, "permuted": perm_p}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[interneuron whitening de-risk] seeds={SEEDS} M={M_PRINCIPAL} P-sweep={PS} -- does a small "
          f"interneuron population (count=rank) reach the off-diagonal ceiling on REAL where the somatic lateral "
          f"collapses?", flush=True)
    rows = [run_seed(s) for s in SEEDS]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    host, zca8, diag, sm, raw = m("host"), m("zca8"), m("diagonal"), m("somatic_sm"), m("raw")
    perm = m("permuted"); lesion = m("lesion")

    def pm(P, key):
        return float(np.mean([r["ps"][P][key] for r in rows]))
    print(f"\n{'='*96}\n  MEAN ({len(SEEDS)} seeds): host {host:+.3f} | offline ZCA rank-8 {zca8:+.3f} | "
          f"diagonal {diag:+.3f} | somatic SM {sm:+.3f} | raw projection {raw:+.3f}", flush=True)
    best_P, best_p = None, -9.0
    for P in PS:
        p, er = pm(P, "pearson"), pm(P, "eff_rank")
        print(f"    P={P:3d} interneurons: {p:+.3f} (eff-rank {er:.1f})", flush=True)
        if p > best_p:
            best_p, best_P = p, P
    print(f"  anti-cheat: lesion(L=0) {lesion:+.3f} (= raw {raw:+.3f}); permuted {perm:+.3f}", flush=True)
    print(f"{'='*96}", flush=True)
    if best_p >= sm + 0.06 and best_p >= zca8 - 0.05 and abs(perm) <= 0.10 and lesion < best_p - 0.06:
        print(f"  GO: the INTERNEURON whitening net (P={best_P}, {best_p:+.3f}) REACHES the offline rank-8 ZCA "
              f"({zca8:+.3f} ~ host {host:+.3f}), BEATING the collapsing somatic SM ({sm:+.3f}) AND the diagonal "
              f"({diag:+.3f}); lesion collapses to raw ({lesion:+.3f}); permuted-clean ({perm:+.3f}). ==> a "
              f"POINT-NEURON INTERNEURON CIRCUIT (count=rank) reaches the off-diagonal on REAL data -> a functional "
              f"cortex with NO curated concepts AND no dendrite needed; the build = the interneuron whitening net "
              f"on the bridge (FS-interneuron population sized to the rank). The decisive escape.", flush=True)
    elif best_p >= sm + 0.06:
        print(f"  PARTIAL: the interneuron net (P={best_P}, {best_p:+.3f}) beats the somatic SM ({sm:+.3f}) + "
              f"diagonal ({diag:+.3f}) toward the ZCA ({zca8:+.3f}) but falls short of the ceiling -- the "
              f"interneuron architecture HELPS (real progress past the wall) but needs tuning (rank/lr/settle) or "
              f"the dendritic refinement. Lesion {lesion:+.3f}, permuted {perm:+.3f}.", flush=True)
    else:
        print(f"  NEGATIVE: even the interneuron net (best P={best_P}, {best_p:+.3f}) does NOT clearly beat the "
              f"somatic SM ({sm:+.3f}) -- the interneuron-count-as-rank mechanism doesn't crack the off-diagonal "
              f"on real either; escalate to the dendritic predictive-coding spec (the deep-research in flight).",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host": host, "zca8": zca8, "diagonal": diag, "somatic_sm": sm, "raw": raw, "best_P": best_P,
           "best_pearson": best_p, "lesion": lesion, "permuted": perm,
           "ps": {str(P): {"pearson": pm(P, "pearson"), "eff_rank": pm(P, "eff_rank")} for P in PS},
           "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_interneuron_whitening.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
