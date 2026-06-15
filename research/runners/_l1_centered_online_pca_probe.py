"""L1 root-cause probe: is the online plateau (+0.29) the RULE's wall, or the un-removed COMMON MODE?

The convergence sweep found online similarity-matching saturates at +0.29 (offdiag-cos +0.97) while the
OFFLINE PPMI+PCA optimum is +0.523. The diagnostic tell: lr-decay DE-saturated (offdiag 0.97->0.75) yet
Pearson FELL to +0.258 -- de-saturating moved AWAY from the +0.485 input-cosine structure. Root cause
hypothesis: the offline pca_lowrank_sim CENTERS (subtracts the column mean = removes the COMMON MODE); the
online rule on UN-CENTERED PPMI puts its dominant component on the common mode (the shared high-frequency
context every concept co-occurs with) -> the codes are dominated by it -> offdiag-cos ~1 saturation. This is
the SAME common-mode/whitening wall the project has hit 5+ times (Mikulasch-Priesemann: a point neuron can't
remove the common mode pre-spike).

THE DECISIVE TEST: run a PROVABLY-CONVERGENT online PCA rule (Oja's symmetric subspace rule, GHA-class --
local Hebbian + lateral, brain-plausible) on (a) un-centered PPMI [must reproduce ~+0.29] vs (b) CENTERED
PPMI [common mode removed]. Centering = subtract the column mean, which a brain realizes as a slow
subtractive-inhibition EMA (feedforward inhibition) -- brain-plausible. Compare to the offline PCA ceiling.
  - If CENTERED-online reaches ~+0.52 ==> the online RULE CAN extract the structure GIVEN common-mode
    removal; L1 flips toward GO/BOUNDARY (the missing piece is subtractive inhibition, which is brain-based).
  - If CENTERED-online ALSO saturates ==> the online local rule genuinely cannot carve the weak real
    structure even with the common mode removed ==> the L1 NEGATIVE is solid.
Also runs Oja on the RAW (non-PPMI) input as a control. CPU/numpy, seed-independent corpus -> build once.

Run: python -u -m research.runners._l1_centered_online_pca_probe --n-hub 2000 --seeds 42,43,44
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
    build_real_corpus, ppmi_matrix, pca_lowrank_sim, encode_raw,
)


def oja_subspace(X, k, epochs, lr, seed):
    """Oja's symmetric SUBSPACE rule (provably converges to the top-k principal subspace; local Hebbian
    feedforward + symmetric lateral decay): y = W x ; dW = lr (y xT - y yT W). Brain-plausible online PCA."""
    rng = np.random.RandomState(seed * 6151 + 5)
    Nc, H = X.shape
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, H) * 0.1
    order = np.arange(Nc)
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = Xn[i]
            y = W @ x
            W += lr * (np.outer(y, x) - np.outer(y, y) @ W)
    return (W @ Xn.T).T


def center_cols(X):
    """Remove the COMMON MODE: subtract the column (context-hub) mean. Brain-plausible as a slow subtractive-
    inhibition EMA of the shared population drive (feedforward inhibition)."""
    return X - X.mean(0, keepdims=True)


def _measure(name, codes, S_true, labels):
    pear = _pearson_vs_Strue(_cos_sim(codes), S_true)
    g, ch = heldout_generalization(codes, labels)
    off = float(_cos_sim(codes)[np.triu_indices(codes.shape[0], 1)].mean())
    print(f"  [{name:34s}] Pearson={pear:+.3f}  offdiag-cos={off:+.3f}  gen={g:.3f} (chance {ch:.3f})", flush=True)
    return {"pearson": pear, "offdiag": off, "gen": g, "chance": ch}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-hub", type=int, default=2000)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--host-alpha", type=float, default=0.75)
    p.add_argument("--out", default="research/findings/raw/_l1_centered_online_pca_probe.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    C, labels, S_true = build_real_corpus(42, args.n_hub)
    Xppmi = ppmi_matrix(C, args.host_alpha)
    Xppmi_c = center_cols(Xppmi)
    Xraw = encode_raw(C)
    Xraw_c = center_cols(Xraw)

    offline_pca = _pearson_vs_Strue(pca_lowrank_sim(Xppmi, args.k), S_true)
    ppmi_cos = _pearson_vs_Strue(_cos_sim(Xppmi), S_true)
    ppmi_cos_c = _pearson_vs_Strue(_cos_sim(Xppmi_c), S_true)
    print(f"[L1 centered-online PCA probe] {C.shape[0]} concepts x {C.shape[1]} hubs", flush=True)
    print(f"  reference: offline PPMI+PCA(k={args.k})={offline_pca:+.3f}  |  cos(PPMI rows) uncentered="
          f"{ppmi_cos:+.3f}  centered={ppmi_cos_c:+.3f}", flush=True)

    arms = {"oja_ppmi_uncentered": Xppmi, "oja_ppmi_CENTERED": Xppmi_c,
            "oja_raw_uncentered": Xraw, "oja_raw_CENTERED": Xraw_c}
    agg = {}
    for label, Xin in arms.items():
        ps, offs, gens = [], [], []
        for s in seeds:
            codes = oja_subspace(Xin, args.k, args.epochs, args.lr, s)
            ps.append(_pearson_vs_Strue(_cos_sim(codes), S_true))
            offs.append(float(_cos_sim(codes)[np.triu_indices(codes.shape[0], 1)].mean()))
            g, _ = heldout_generalization(codes, labels)
            gens.append(g)
        pm, om, gm = float(np.mean(ps)), float(np.mean(offs)), float(np.mean(gens))
        agg[label] = {"pearson_mean": pm, "pearson_seeds": ps, "offdiag_mean": om, "gen_mean": gm}
        print(f"  [{label:24s}] Pearson={pm:+.3f} {['%+.3f'%x for x in ps]}  offdiag-cos={om:+.3f}  gen={gm:.3f}",
              flush=True)

    cen = agg["oja_ppmi_CENTERED"]["pearson_mean"]
    unc = agg["oja_ppmi_uncentered"]["pearson_mean"]
    frac = cen / offline_pca if offline_pca > 1e-9 else 0.0
    if cen >= 0.70 * offline_pca and cen >= 0.40:
        verdict = "COMMON_MODE_WAS_THE_WALL_L1_FLIPS"
        why = (f"CENTERED online Oja-PCA reaches {cen:+.3f} = {frac:.0%} of the offline optimum {offline_pca:+.3f} "
               f"(vs un-centered {unc:+.3f}) -> the online plateau was the UN-REMOVED COMMON MODE, not the rule. "
               f"L1 flips toward GO/BOUNDARY: a brain-plausible online PCA rule CAN extract the real structure "
               f"GIVEN common-mode removal (subtractive-inhibition EMA, itself brain-based). The learned cortex "
               f"path is the online rule + a subtractive-inhibition front-end.")
    elif cen >= unc + 0.10:
        verdict = "CENTERING_HELPS_PARTIAL_BOUNDARY"
        why = (f"centering lifts online Oja-PCA materially ({unc:+.3f}->{cen:+.3f}) but it still falls short of "
               f"the offline optimum {offline_pca:+.3f} ({frac:.0%}) -> common-mode removal is PART of it; a "
               f"residual online-vs-offline gap remains -> BOUNDARY with a sharp target.")
    else:
        verdict = "RULE_IS_THE_WALL_L1_NEGATIVE_SOLID"
        why = (f"even CENTERED (common mode removed) online Oja-PCA saturates ({cen:+.3f} vs offline "
               f"{offline_pca:+.3f}); centering did not rescue it -> the online LOCAL rule genuinely cannot carve "
               f"the weak real structure -> the L1 NEGATIVE is SOLID; ship the flat 2,048-concept cortex.")
    print(f"\n{'='*92}\n  PROBE VERDICT: {verdict}\n  {why}\n{'='*92}", flush=True)
    print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    out = {"verdict": verdict, "why": why, "offline_pca": offline_pca, "ppmi_cos_uncentered": ppmi_cos,
           "ppmi_cos_centered": ppmi_cos_c, "arms": agg, "seeds": seeds, "n_hub": args.n_hub, "k": args.k}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
