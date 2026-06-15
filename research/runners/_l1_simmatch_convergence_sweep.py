"""L1 follow-up: is the online similarity-matching PLATEAU on the real corpus a TUNING problem or a WALL?

The fair-test (`learned_graded_cortex_fair_test.py --real-corpus`) found online similarity-matching plateaus
at Pearson ~+0.29 (offdiag-cos +0.97 = SATURATED) while the OFFLINE PPMI+PCA optimum is +0.523 -> the
structure IS recoverable from real PPMI, so the gap is the ONLINE rule saturating, not a data wall. The
diagnosed cause: a CONSTANT feedforward learning rate leaves a residual noise floor (Robbins-Monro: the
Pehlevan-Chklovskii online rule converges to the offline optimum only as lr->0), and k=64 (= #concepts) is
over-expressive for an 8-category structure. This sweeps the principled fixes:
  - LR DECAY (lr_ff_t = lr_ff0 / (1 + decay*epoch))  -- the Robbins-Monro schedule.
  - SMALLER lr_ff (constant).
  - SMALLER k (compress toward the 8 categories).
  - LONGER settle.
Against the FIXED offline PPMI+PCA(k) ceiling. If any config approaches the offline optimum (de-saturates,
offdiag-cos drops, Pearson rises toward +0.5) -> the plateau was TUNING -> the L1 verdict flips toward GO.
If ALL configs plateau/saturate -> the online brain-based rule genuinely cannot carve the weak real
structure -> the NEGATIVE is solid. Corpus is seed-independent at n_hub=2000 (host identical across seeds),
so build once; vary the LEARNER seed for robustness. CPU/numpy, NO sim/ edits.

Run: python -u -m research.runners._l1_simmatch_convergence_sweep --n-hub 2000 --seeds 42,43,44
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import (  # noqa: E402
    build_real_corpus, ppmi_matrix, pca_lowrank_sim,
)


def learn_simmatch_sched(X, k, epochs, lr_ff0, lr_m, settle_steps, seed, lr_decay=0.0):
    """Pehlevan-Chklovskii online similarity-matching with an optional Robbins-Monro lr schedule on the
    FEEDFORWARD rate (lr_ff_t = lr_ff0 / (1 + lr_decay*ep)). lr_decay=0 -> constant (the baseline)."""
    rng = np.random.RandomState(seed * 104729 + 3)
    Nc, H = X.shape
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W_ff = rng.randn(k, H) * 0.1
    M = np.zeros((k, k), dtype=np.float64)
    order = np.arange(Nc)
    for ep in range(epochs):
        lr_ff = lr_ff0 / (1.0 + lr_decay * ep)
        rng.shuffle(order)
        for i in order:
            x = Xn[i]
            ff = W_ff @ x
            y = np.zeros(k)
            for _ in range(settle_steps):
                y = 0.5 * y + 0.5 * (ff - M @ y)
            W_ff += lr_ff * (np.outer(y, x) - (y ** 2)[:, None] * W_ff)
            dM = np.outer(y, y) - M
            np.fill_diagonal(dM, 0.0)
            M += lr_m * dM
    codes = np.zeros((Nc, k))
    for i in range(Nc):
        ff = W_ff @ Xn[i]
        y = np.zeros(k)
        for _ in range(settle_steps):
            y = 0.5 * y + 0.5 * (ff - M @ y)
        codes[i] = y
    return codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-hub", type=int, default=2000)
    p.add_argument("--host-alpha", type=float, default=0.75)
    p.add_argument("--out", default="research/findings/raw/_l1_simmatch_convergence_sweep.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    # corpus is seed-independent at this n_hub -> build once
    C, labels, S_true = build_real_corpus(42, args.n_hub)
    Xppmi = ppmi_matrix(C, args.host_alpha)
    print(f"[L1 simmatch convergence sweep] {C.shape[0]} concepts x {C.shape[1]} hubs", flush=True)
    for kk in (8, 16, 32, 64):
        pca_p = _pearson_vs_Strue(pca_lowrank_sim(Xppmi, kk), S_true)
        print(f"  offline PPMI+PCA(k={kk:2d}) ceiling = {pca_p:+.3f}", flush=True)

    # (k, epochs, lr_ff0, lr_m, settle, lr_decay, label)
    configs = [
        (64, 200, 0.010, 0.010, 30, 0.00, "baseline (reproduce +0.29)"),
        (64, 300, 0.010, 0.010, 30, 0.05, "k64 + LR-DECAY"),
        (64, 300, 0.003, 0.010, 30, 0.00, "k64 + low lr_ff=0.003"),
        (32, 300, 0.010, 0.010, 30, 0.05, "k32 + LR-DECAY"),
        (16, 300, 0.010, 0.010, 40, 0.05, "k16 + LR-DECAY + settle40"),
        (16, 300, 0.003, 0.010, 40, 0.00, "k16 + low lr_ff=0.003 + settle40"),
        (8,  400, 0.010, 0.010, 50, 0.05, "k8  + LR-DECAY + settle50"),
        (8,  400, 0.003, 0.020, 50, 0.00, "k8  + low lr_ff + fast lateral"),
    ]
    results = []
    best = None
    for (k, ep, lrff, lrm, settle, dec, label) in configs:
        pears, offs, gens = [], [], []
        for s in seeds:
            codes = learn_simmatch_sched(Xppmi, k, ep, lrff, lrm, settle, s, lr_decay=dec)
            pears.append(_pearson_vs_Strue(_cos_sim(codes), S_true))
            offs.append(float(_cos_sim(codes)[np.triu_indices(codes.shape[0], 1)].mean()))
            g, ch = heldout_generalization(codes, labels)
            gens.append(g)
        pm, om, gm = float(np.mean(pears)), float(np.mean(offs)), float(np.mean(gens))
        rec = {"label": label, "k": k, "epochs": ep, "lr_ff0": lrff, "lr_m": lrm, "settle": settle,
               "lr_decay": dec, "pearson_mean": pm, "pearson_seeds": pears, "offdiag_mean": om, "gen_mean": gm}
        results.append(rec)
        if best is None or pm > best["pearson_mean"]:
            best = rec
        print(f"  [{label:34s}] Pearson={pm:+.3f} {['%+.3f'%x for x in pears]}  "
              f"offdiag-cos={om:+.3f} (saturated if ~1)  gen={gm:.3f}", flush=True)

    # the decisive comparison
    pca64 = _pearson_vs_Strue(pca_lowrank_sim(Xppmi, 64), S_true)
    pca_best_k = max((_pearson_vs_Strue(pca_lowrank_sim(Xppmi, kk), S_true), kk) for kk in (8, 16, 32, 64))
    frac = best["pearson_mean"] / pca_best_k[0] if pca_best_k[0] > 1e-9 else 0.0
    if best["pearson_mean"] >= 0.70 * pca_best_k[0] and best["pearson_mean"] >= 0.40:
        verdict = "PLATEAU_WAS_TUNING_GO"
        why = (f"best online config '{best['label']}' reaches {best['pearson_mean']:+.3f} = {frac:.0%} of the "
               f"offline PPMI+PCA(k={pca_best_k[1]}) optimum {pca_best_k[0]:+.3f} (de-saturated, offdiag "
               f"{best['offdiag_mean']:+.3f}) -> the plateau WAS tuning; online similarity-matching CAN reach the "
               f"ceiling on real data -> L1 flips toward GO; escalate to the spiking similarity-matching build.")
    elif best["pearson_mean"] >= 0.40:
        verdict = "PARTIAL_BOUNDARY"
        why = (f"best online '{best['label']}' {best['pearson_mean']:+.3f} = {frac:.0%} of offline "
               f"{pca_best_k[0]:+.3f} -> tuning lifts it materially but it still falls short of the offline "
               f"optimum -> BOUNDARY with a sharp target ({pca_best_k[0]-best['pearson_mean']:+.3f} residual).")
    else:
        verdict = "PLATEAU_IS_A_WALL_NEGATIVE"
        why = (f"NO online config clears Pearson +0.40 (best '{best['label']}' {best['pearson_mean']:+.3f}, "
               f"offdiag {best['offdiag_mean']:+.3f}) while the offline optimum is {pca_best_k[0]:+.3f} -> the "
               f"online brain-based rule genuinely cannot carve the weak real structure -> the L1 NEGATIVE is "
               f"solid; ship the flat 2,048-concept cortex.")
    print(f"\n{'='*92}\n  SWEEP VERDICT: {verdict}\n  {why}\n{'='*92}", flush=True)
    print(f"  best: {best['label']}  Pearson={best['pearson_mean']:+.3f}  offline-PCA-best(k={pca_best_k[1]})="
          f"{pca_best_k[0]:+.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
    out = {"verdict": verdict, "why": why, "best": best, "offline_pca_best": {"pearson": pca_best_k[0], "k": pca_best_k[1]},
           "offline_pca_k64": pca64, "results": results, "seeds": seeds, "n_hub": args.n_hub}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
