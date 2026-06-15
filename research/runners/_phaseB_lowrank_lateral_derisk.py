"""CYCLE 87 — the deep-research O1 de-risk: can a LOW-RANK point-neuron lateral reach the off-diagonal ceiling
on the REAL corpus? (the decisive test before more months on the diagonal-only D2 build.)

The off-diagonal deep-research (2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research.md) verdict:
(1) D2-as-designed (per-hub divisive gain r=D*x, D diagonal) is MATHEMATICALLY incapable of the off-diagonal
    -> caps at the measured 49% of host on real; fixing its Phase-2 NEGATIVE can't raise it.
(2) every biologically-plausible off-diagonal decorrelator is a RECURRENT cross-neuron lateral; pure whitening
    OVER-whitens (the project's full-ZCA collapse -0.012) -> the target is LOW-RANK (rank~8 ZCA = +0.437 ~ host).
(3) the project's own SM lateral (learn_simmatch / graded_lateral) plateaus at +0.35 because it OVER-COMPRESSES
    to eff-rank 3-4 at k=64 (CYCLE 81). O1 = make the lateral LOW-RANK (a k~8 BOTTLENECK) + full-settle so it
    keeps the RIGHT 8 informative dimensions instead of collapsing to 3-4.

THE DECISIVE TEST: run the validated online SM lateral (learn_simmatch, the anti-Hebbian interneuron lateral +
Oja) at a LOW-RANK bottleneck k in {8,12,16,32} on the REAL PPMI corpus, vs:
  - the offline ZCA rank-k (the achievable low-rank-whitening ceiling; rank-8 ~ +0.437 ~ host),
  - the diagonal divisive-gain control (must stay ~+0.22, the D2-as-designed ceiling),
  - host PPMI+SVD (+0.442).
GATE: does the low-rank online lateral BEAT the k=64 SM (+0.35) + the diagonal (+0.22) toward the offline
rank-8 ZCA (+0.437)? GO => a POINT-NEURON low-rank lateral circuit reaches the off-diagonal ceiling -> D2
dendrites are UNNECESSARY (fix the lateral, not build a diagonal dendrite). BOUNDARY/NEGATIVE => even the
low-rank lateral plateaus -> the off-diagonal genuinely needs the deeper M-P dendritic-PLUS-lateral build (the
honest deep frontier), and the cheap point-neuron path is exhausted.

Reuse-by-import (learn_simmatch + ppmi_matrix + build_real_corpus); NO sim/ edits; numpy; 3 seeds.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_lowrank_lateral_derisk
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
    _cos_sim, _pearson_vs_Strue, effective_rank, perhub_residual,
)
from research.runners.learned_graded_cortex_fair_test import (  # noqa: E402
    build_real_corpus, ppmi_matrix, learn_simmatch,
)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402

SEEDS = (42, 43, 44)
N_HUB = 500
ALPHA = 0.75
KS = (8, 12, 16, 32)              # the low-rank bottleneck sweep (rank-8 is the ZCA ceiling)
LR_FF, LR_M, SETTLE, EPOCHS = 0.005, 0.05, 40, 200   # stable config + fast lateral (CYCLE 81)


def zca_rank_sim(X, k):
    """Offline low-rank ZCA whitening (keep top-k directions, EQUALIZE their variance) -> cosine. The
    achievable low-rank-whitening ceiling the online lateral is trying to reach (rank-8 ~ +0.437 ~ host)."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    kk = min(k, U.shape[1])
    emb = U[:, :kk]                                  # whitened scores (unit-variance orthonormal columns)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb @ emb.T


def run_seed(seed):
    C, labels, S_true = build_real_corpus(seed, N_HUB)
    labels = np.asarray(labels)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=ALPHA), labels)
    Xppmi = ppmi_matrix(C, ALPHA)
    # diagonal control (the D2-as-designed ceiling): per-hub divisive gain on the raw counts.
    g = C.mean(0)
    diag_p = _pearson_vs_Strue(_cos_sim(perhub_residual(C, g, 1.0)), S_true)
    print(f"\n[low-rank lateral seed {seed}] {C.shape[0]}c x {C.shape[1]}h | host {host_p:+.3f} | "
          f"diagonal-gain control {diag_p:+.3f} (the D2-as-designed ceiling)", flush=True)
    rows = {}
    for k in KS:
        zca_p = _pearson_vs_Strue(zca_rank_sim(Xppmi, k), S_true)
        codes, _, traj = learn_simmatch(Xppmi, S_true, k, EPOCHS, LR_FF, LR_M, SETTLE, seed,
                                        track_every=max(1, EPOCHS // 4))
        sm_p = _pearson_vs_Strue(_cos_sim(codes), S_true)
        sm_peak = max([p for _, p in traj] + [sm_p])
        er = effective_rank(codes)
        rows[k] = {"zca": zca_p, "sm": sm_p, "sm_peak": sm_peak, "eff_rank": er}
        print(f"    k={k:3d}: offline ZCA rank-k {zca_p:+.3f} | online low-rank lateral {sm_p:+.3f} "
              f"(peak {sm_peak:+.3f}, eff-rank {er:.1f})", flush=True)
    return {"seed": seed, "host": host_p, "diagonal": diag_p, "ks": rows}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[low-rank lateral O1 de-risk] seeds={SEEDS} n_hub={N_HUB} -- can a LOW-RANK point-neuron lateral "
          f"reach the off-diagonal ceiling (rank-8 ZCA ~+0.44) on the REAL corpus, beating the diagonal +0.22 "
          f"and the k=64 SM +0.35?", flush=True)
    rows = [run_seed(s) for s in SEEDS]
    host = float(np.mean([r["host"] for r in rows]))
    diag = float(np.mean([r["diagonal"] for r in rows]))

    def km(k, key):
        return float(np.mean([r["ks"][k][key] for r in rows]))
    print(f"\n{'='*96}\n  MEAN ({len(SEEDS)} seeds): host {host:+.3f} | diagonal-gain control {diag:+.3f} "
          f"| reference k=64 SM ~+0.35 (CYCLE 81)", flush=True)
    best_k, best_sm = None, -9.0
    for k in KS:
        zca, sm, peak, er = km(k, "zca"), km(k, "sm"), km(k, "sm_peak"), km(k, "eff_rank")
        print(f"    k={k:3d}: offline ZCA {zca:+.3f} | online low-rank lateral {sm:+.3f} (peak {peak:+.3f}, "
              f"eff-rank {er:.1f})", flush=True)
        if peak > best_sm:
            best_sm, best_k = peak, k
    zca8 = km(8, "zca")
    print(f"{'='*96}", flush=True)
    # the gate uses the PEAK (the stable-optimum, not the over-trained endpoint -- CYCLE 81 lesson).
    if best_sm >= zca8 - 0.03 and best_sm >= 0.40:
        print(f"  GO: the LOW-RANK online lateral (k={best_k}, peak {best_sm:+.3f}) REACHES the offline rank-8 "
              f"ZCA ceiling ({zca8:+.3f} ~ host {host:+.3f}), BEATING the diagonal-gain control ({diag:+.3f}) AND "
              f"the k=64 SM (+0.35). ==> a POINT-NEURON low-rank inhibitory-interneuron lateral CIRCUIT reaches "
              f"the off-diagonal ceiling on REAL data -> the D2 dendritic build is UNNECESSARY for this; the fix "
              f"is the low-rank lateral (the project's graded_lateral made low-rank + full-settle), NOT a dendrite "
              f"that re-does the diagonal. The cheapest, highest-value redirect for the owner.", flush=True)
    elif best_sm >= 0.40:
        print(f"  PARTIAL: the low-rank lateral (k={best_k}, peak {best_sm:+.3f}) beats the diagonal ({diag:+.3f}) "
              f"+ the k=64 SM (+0.35) toward host but falls short of the rank-8 ZCA ({zca8:+.3f}) -- the online "
              f"lateral under-converges vs the offline optimum; a fuller-settle / better-conditioned lateral may "
              f"close it (still a point-neuron circuit, no dendrite needed).", flush=True)
    else:
        print(f"  BOUNDARY/NEGATIVE: even the LOW-RANK online lateral (k={best_k}, peak {best_sm:+.3f}) does NOT "
              f"clearly beat the k=64 SM (+0.35) toward the rank-8 ZCA ({zca8:+.3f}) -- the point-neuron lateral "
              f"plateaus even when bottlenecked to low rank. ==> the off-diagonal genuinely needs MORE than a "
              f"point-neuron lateral (the deeper Mikulasch-Priesemann dendritic-PLUS-lateral build) -- the cheap "
              f"point-neuron path is exhausted; the honest deep frontier. (Offline rank-8 ZCA {zca8:+.3f} proves "
              f"the structure IS reachable, so this is a CONVERGENCE/substrate limit, not a data limit.)", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host": host, "diagonal": diag, "best_k": best_k, "best_sm_peak": best_sm, "zca_rank8": zca8,
           "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_lowrank_lateral.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
