"""L1 learning-spiking axis: does the NONNEGATIVE similarity-matching variant (the biologically-correct one
-- firing rates are non-negative) still extract the structure, OR does the signed->nonneg change break it?

The validated GO + caveat-4 used SIGNED output (y can be negative). But a spiking neuron emits non-negative
spike counts; the Oja/Pehlevan decorrelation relies on negative output correlations the signed rule has and
a rectified rate does not. Pehlevan's NONNEGATIVE similarity-matching (NSM, rectified output + lateral
inhibition) is the brain-correct version. This tests whether NSM preserves the category structure that the
signed rule recovers (+0.515). Staged to isolate the nonnegativity from spike noise:
  SIGNED  (reference)      -- the +0.515 result.
  NONNEG rate             -- rectified output, exact rates (isolates the nonnegativity).
  NONNEG + spike output   -- rectified output delivered as Poisson spike counts (the full spiking learner).
All on CENTERED PPMI input (common-mode removal = subtractive inhibition). Multi-seed + permuted control.
GATE: NONNEG (rate and/or spike) >= 0.70 * signed AND >= 0.30, permuted ~0 -> the learning-spiking axis is
de-risked (non-negativity does not break it). If NONNEG collapses -> the build needs a signed-via-two-
populations (ON/OFF) trick; flag it BEFORE the commit. CPU/numpy, build once. NO sim/ edits.

Run: python -u -m research.runners._l1_nonneg_simmatch_check --n-hub 2000 --seeds 42,43,44
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
from research.runners._l1_centered_online_pca_probe import center_cols  # noqa: E402


def simmatch(X, k, epochs, lr_ff0, lr_m, settle_steps, seed, lr_decay=0.0, nonneg=False,
             spike_output=False, out_gain=30.0):
    """Pehlevan-Chklovskii similarity-matching. nonneg -> rectified output (NSM, brain-correct). spike_output
    -> the rectified rate is delivered as a Poisson spike count in the Hebbian terms (the full spiking learner)."""
    rng = np.random.RandomState(seed * 104729 + 3)
    Nc, H = X.shape
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W_ff = rng.randn(k, H) * 0.1
    M = np.zeros((k, k), dtype=np.float64)
    order = np.arange(Nc)

    def settle(ff):
        y = np.zeros(k)
        for _ in range(settle_steps):
            y = 0.7 * y + 0.3 * (ff - M @ y)
            if nonneg:
                y = np.maximum(y, 0.0)
        return y

    for ep in range(epochs):
        lr_ff = lr_ff0 / (1.0 + lr_decay * ep)
        rng.shuffle(order)
        for i in order:
            x = Xn[i]; y = settle(W_ff @ x)
            yh = y
            if spike_output:
                yh = rng.poisson(np.maximum(y, 0.0) * out_gain).astype(np.float64) / out_gain  # noisy rate est
            W_ff += lr_ff * (np.outer(yh, x) - (yh ** 2)[:, None] * W_ff)
            dM = np.outer(yh, yh) - M
            np.fill_diagonal(dM, 0.0)
            M += lr_m * dM
    return np.array([settle(W_ff @ Xn[i]) for i in range(Nc)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-hub", type=int, default=2000)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--host-alpha", type=float, default=0.75)
    p.add_argument("--out", default="research/findings/raw/_l1_nonneg_simmatch_check.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    C, labels, S_true = build_real_corpus(42, args.n_hub)
    Xc = center_cols(ppmi_matrix(C, args.host_alpha))
    offline = _pearson_vs_Strue(pca_lowrank_sim(ppmi_matrix(C, args.host_alpha), args.k), S_true)
    print(f"[L1 nonneg-simmatch check] {C.shape[0]} concepts x {C.shape[1]} hubs; offline {offline:+.3f}", flush=True)

    arms = {
        "SIGNED (reference)": dict(nonneg=False, spike_output=False),
        "NONNEG rate": dict(nonneg=True, spike_output=False),
        "NONNEG + spike output": dict(nonneg=True, spike_output=True),
    }
    out_arms = {}
    for label, kw in arms.items():
        ps, gens, perms = [], [], []
        for s in seeds:
            codes = simmatch(Xc, args.k, 200, 0.010, 0.030, 40, s, lr_decay=0.0, **kw)
            ps.append(_pearson_vs_Strue(_cos_sim(codes), S_true))
            g, ch = heldout_generalization(codes, labels); gens.append(g)
            rng = np.random.RandomState(s * 2718281 + 1); perm = rng.permutation(labels)
            S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
            perms.append(_pearson_vs_Strue(_cos_sim(codes), S_perm))
        out_arms[label] = {"pearson_mean": float(np.mean(ps)), "pearson_seeds": ps,
                           "gen_mean": float(np.mean(gens)), "perm_mean": float(np.mean(perms))}
        print(f"  [{label:24s}] Pearson={out_arms[label]['pearson_mean']:+.3f} {['%+.3f'%x for x in ps]}  "
              f"gen={out_arms[label]['gen_mean']:.3f}  perm={out_arms[label]['perm_mean']:+.3f}", flush=True)

    signed = out_arms["SIGNED (reference)"]["pearson_mean"]
    nn_rate = out_arms["NONNEG rate"]["pearson_mean"]
    nn_spk = out_arms["NONNEG + spike output"]["pearson_mean"]
    best_nn = max(nn_rate, nn_spk)
    clean = abs(out_arms["NONNEG + spike output"]["perm_mean"]) <= 0.15
    if best_nn >= 0.70 * signed and best_nn >= 0.30 and clean:
        verdict = "NONNEG_SURVIVES_LEARNING_SPIKING_DERISKED"
        why = (f"the brain-correct NONNEGATIVE similarity-matching preserves the structure (nonneg-rate "
               f"{nn_rate:+.3f}, nonneg+spike {nn_spk:+.3f} vs signed {signed:+.3f}, {best_nn/signed:.0%}), "
               f"permuted clean -> the learning-spiking axis is de-risked; non-negativity (rectified firing) "
               f"does NOT break the learner. ALL THREE axes (rule, input-spiking, learning-nonneg/spiking) GO "
               f"at the rate/spike-smoke level -> the spiking build is comprehensively de-risked (scale + the "
               f"bridge assembly remain the build itself).")
    elif best_nn >= 0.30 and clean:
        verdict = "NONNEG_PARTIAL"
        why = (f"nonneg similarity-matching partially survives (best {best_nn:+.3f} vs signed {signed:+.3f}, "
               f"{best_nn/signed:.0%}) -> non-negativity costs some structure; the build likely needs a "
               f"signed-via-ON/OFF-population trick to recover the signed performance. Note it.")
    else:
        verdict = "NONNEG_BREAKS_IT"
        why = (f"the brain-correct nonneg variant collapses (best {best_nn:+.3f} vs signed {signed:+.3f}) -> "
               f"rectified firing breaks the decorrelation -> the spiking build MUST use a signed-via-two-"
               f"populations (ON/OFF) representation; material qualification of the GO -- flag BEFORE the commit.")
    print(f"\n{'='*92}\n  LEARNING-SPIKING VERDICT: {verdict}\n  {why}\n{'='*92}", flush=True)
    print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    out = {"verdict": verdict, "why": why, "arms": out_arms, "signed_ref": signed, "offline_pca": offline,
           "seeds": seeds, "n_hub": args.n_hub, "k": args.k}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
