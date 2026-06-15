"""The OWNER'S QUESTION made decisive: "the brain needs to learn naturally over time -- what could we be
missing?" -> is the online similarity-matching (SM) +0.348 plateau vs the offline PPMI+PCA optimum +0.518 a
CONVERGENCE-REGIME artifact (the wrong learning-rate timescale separation), or the genuine point-neuron
locality wall?

THE UNTESTED VARIABLE (the crux): the validated `learn_simmatch` (the arc's own L1 GO machinery) defaults to
`lr_ff == lr_m` (0.01 == 0.01) -- EQUAL feedforward and lateral rates. EVERY fair_test run, including the
CYCLE-80 "+0.296 BOUNDARY", used equal rates. But Pehlevan-Chklovskii similarity-matching REQUIRES the lateral
M to adapt FASTER than the feedforward W (the anti-Hebbian decorrelation must keep up with / lead the Hebbian
learning, else W partially collapses onto the top few components -> eff-rank ~5 instead of the needed ~8).
Biologically: inhibitory/interneuron plasticity is fast; pyramidal feedforward plasticity is slow. The
"natural learning over time" regime = a STABLE low feedforward rate + a FASTER lateral + many epochs (replay /
consolidation) + a tracked plateau (NOT the over-trained endpoint).

This probe runs the VALIDATED learn_simmatch (reuse-by-import -- NOT a re-implementation) on the REAL corpus
(PPMI input, the host-matched encoding), sweeping the lr_m/lr_ff timescale ratio + settle depth, multi-seed,
tracking the convergence trajectory to confirm a stable plateau (not over-training). Reports Pearson(S,S_true)
+ eff-rank vs BOTH ceilings: host PPMI+SVD and the offline PPMI+PCA(k) optimum.

THE DECISIVE FORK:
  CLOSES   : a timescale-separated SM (lr_m >> lr_ff) lifts toward the offline +0.518 (eff-rank -> ~8) ==> the
             "marginality" was a CONVERGENCE-REGIME artifact; the brain-based rule reaches the offline optimum
             in its correct (fast-lateral, stable, replayed) regime -> a HOPEFUL answer to the owner, and the
             bridge target rises from the marginal +0.296 to the offline +0.518.
  PLATEAUS : even timescale-separated + stable + long, SM plateaus at ~+0.348 / eff-rank ~5 < offline +0.518 /
             rank ~8 ==> the residual is the genuine online-LOCAL-vs-offline-GLOBAL gap (the point-neuron
             locality wall, == the 2026-06-11 cortex-fork (B) dendritic path) -- an honest, well-localized
             negative that CONVERGES the natural-learning angle with the existing fork.
Anti-cheat: the offline PPMI+PCA optimum is the SAME-input reference (so a lift is convergence, not input
change); permuted ~0; eff-rank reported (rank-1 collapse vs the needed ~8 is the mechanism, not the number).

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_timescale_convergence_derisk
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

N_HUB = 500          # match the CYCLE-80 numbers (host +0.442, the BOUNDARY +0.296 was at n_hub=500)
K = 64               # the embedding dim (the offline PPMI+PCA(64) optimum is +0.518)
ALPHA = 0.75         # host PPMI context-smoothing (matched)
SEEDS = (42, 43, 44)

# (lr_ff, lr_m, settle, epochs, label): the timescale-separation sweep. The 1:1 baseline IS the fair_test
# default (the +0.296/+0.348 regime); the 5:1/10:1/20:1 give the lateral the faster timescale the SM theory
# demands. The stable low-ff long-epoch arms ARE the "natural learning over time" (replay/consolidation) regime.
CONFIGS = [
    (0.010, 0.010, 30, 200, "1:1 baseline (fair_test default)"),
    (0.010, 0.050, 30, 200, "5:1 lateral-faster"),
    (0.010, 0.100, 30, 200, "10:1 lateral-faster"),
    (0.005, 0.100, 40, 400, "20:1 lateral-faster + stable-ff + long (natural-over-time)"),
    (0.003, 0.060, 50, 600, "20:1 + deeper-settle + longer (consolidation regime)"),
]


def run_seed(seed):
    C, labels, S_true = build_real_corpus(seed, N_HUB)
    Nc, H = C.shape
    svd_dim = min(50, min(C.shape) - 1)
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=svd_dim, alpha=ALPHA)
    host_p, _, _, _ = score(host_sim, labels)
    Xppmi = ppmi_matrix(C, ALPHA)
    offline_p = _pearson_vs_Strue(pca_lowrank_sim(Xppmi, K), S_true)
    offline_rank = effective_rank(pca_lowrank_sim_codes(Xppmi, K))
    print(f"\n[timescale seed {seed}] {Nc}c x {H}h | host PPMI+SVD={host_p:+.3f} | "
          f"offline PPMI+PCA(k={K})={offline_p:+.3f} (eff-rank {offline_rank:.1f})", flush=True)
    rows = []
    for lr_ff, lr_m, settle, epochs, label in CONFIGS:
        codes, _, traj = learn_simmatch(Xppmi, S_true, K, epochs, lr_ff, lr_m, settle, seed,
                                        track_every=max(1, epochs // 5))
        pear = _pearson_vs_Strue(_cos_sim(codes), S_true)
        rank = effective_rank(codes)
        trajs = "  ".join(f"ep{e}:{p:+.3f}" for e, p in traj)
        # plateau detection: is the END within 0.02 of the PEAK (stable) or well below (over-trained)?
        peak = max(p for _, p in traj) if traj else pear
        stable = "STABLE" if pear >= peak - 0.02 else f"OVER-TRAINED(peak {peak:+.3f})"
        print(f"  [{label:52s}] Pearson={pear:+.3f}  eff-rank={rank:4.1f}  [{stable}]\n"
              f"      traj: {trajs}", flush=True)
        rows.append({"label": label, "lr_ff": lr_ff, "lr_m": lr_m, "ratio": lr_m / lr_ff, "settle": settle,
                     "epochs": epochs, "pearson": round(pear, 4), "eff_rank": round(rank, 2),
                     "peak": round(peak, 4), "stable": pear >= peak - 0.02})
    return {"seed": seed, "host": host_p, "offline": offline_p, "offline_rank": round(offline_rank, 2),
            "configs": rows}


def pca_lowrank_sim_codes(X, k):
    """The offline PCA embedding (codes, not the gram) -- to report its eff-rank as the rank ceiling target."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    kk = min(k, len(S))
    emb = U[:, :kk] * S[:kk]
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[timescale-convergence de-risk] seeds={SEEDS} n_hub={N_HUB} k={K} -- does lr_m>>lr_ff (the SM "
          f"theory's required timescale separation) close the online +0.348 -> offline +0.518 gap?", flush=True)
    rows = [run_seed(s) for s in SEEDS]

    def cfg_mean(i, key):
        return float(np.mean([r["configs"][i][key] for r in rows]))
    host = float(np.mean([r["host"] for r in rows]))
    offline = float(np.mean([r["offline"] for r in rows]))
    offline_rank = float(np.mean([r["offline_rank"] for r in rows]))
    print(f"\n{'='*96}\n  MEAN ({len(SEEDS)} seeds): host {host:+.3f} | offline PPMI+PCA {offline:+.3f} "
          f"(eff-rank {offline_rank:.1f})", flush=True)
    best_i, best_p = 0, -9.0
    for i, cfg in enumerate(CONFIGS):
        mp = cfg_mean(i, "pearson"); mr = cfg_mean(i, "eff_rank"); ratio = cfg[1] / cfg[0]
        print(f"  [{cfg[4]:52s}] ratio {ratio:4.0f}:1  Pearson={mp:+.3f}  eff-rank={mr:.1f}", flush=True)
        if mp > best_p:
            best_p, best_i = mp, i
    baseline_p = cfg_mean(0, "pearson")
    best_label = CONFIGS[best_i][4]
    print(f"{'='*96}", flush=True)
    # The fork: did timescale separation lift toward offline, or plateau?
    lift = best_p - baseline_p
    frac_off = best_p / offline if offline > 1e-9 else 0.0
    if best_p >= offline - 0.04:
        print(f"  CLOSES: timescale-separated SM ({best_label}) = {best_p:+.3f} REACHES the offline optimum "
              f"({offline:+.3f}, {frac_off:.0%}). ==> the marginality was a CONVERGENCE-REGIME artifact (the "
              f"fair_test's equal-rate default violated the SM theory's fast-lateral requirement). The brain-"
              f"based rule reaches the offline optimum in its correct (fast-lateral, stable, replayed) regime. "
              f"The bridge target rises from the marginal +0.296 to the offline +0.52. HOPEFUL answer to the "
              f"owner: this IS what we were missing.", flush=True)
    elif lift >= 0.04:
        print(f"  PARTIAL: timescale separation LIFTS the SM (+{lift:.3f}: baseline {baseline_p:+.3f} -> "
              f"{best_p:+.3f} at {best_label}) toward offline {offline:+.3f} ({frac_off:.0%}) but does NOT fully "
              f"reach it. Real convergence gain (load-bearing -- the fast lateral matters) but a residual "
              f"online-local-vs-offline gap remains -> push the regime further (faster lateral / deeper settle / "
              f"more replay) OR the residual is the locality wall.", flush=True)
    else:
        print(f"  PLATEAUS: even timescale-separated + stable + long, the SM plateaus ({best_p:+.3f} at "
              f"{best_label}, baseline {baseline_p:+.3f}) below offline {offline:+.3f} ({frac_off:.0%}). ==> the "
              f"residual is the genuine online-LOCAL-vs-offline-GLOBAL gap (the point-neuron locality wall == "
              f"the 2026-06-11 cortex-fork (B) dendritic path). The natural-learning angle CONVERGES with the "
              f"existing fork: replay/homeostasis stabilize + un-do over-training, but the last ~3 of 8 PCs need "
              f"a non-local (dendritic) computation a point-neuron local rule cannot do.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)

    out = {"host_mean": host, "offline_mean": offline, "offline_rank_mean": offline_rank,
           "baseline_pearson": baseline_p, "best_pearson": best_p, "best_label": best_label,
           "lift": lift, "frac_of_offline": frac_off, "per_seed": rows, "configs": [c[4] for c in CONFIGS]}
    import json
    raw_dir = os.path.join(_REPO, "research", "findings", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, "_phaseB_timescale_convergence.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
