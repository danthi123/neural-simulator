"""L1 FINAL validation: a brain-plausible online local PCA rule (Oja) on TRUE PPMI input vs the full
anti-cheat battery -- with the LOAD-BEARING honesty control (is the LEARNING the work, or the PPMI INPUT?).

Context: the convergence sweep made similarity-matching look like a wall (+0.29); the centered-online probe
flipped it -- Oja's subspace rule (same brain-plausible online-local-Hebbian-PCA class) reaches +0.44-0.48
= 85-92% of the offline PPMI+PCA optimum (+0.523), beating the project's own ppmi_svd_sim (+0.323). The
+0.29 was an UNDER-CONVERGED similarity-matching implementation, not an online-rule wall.

THE HONESTY CONTROL THAT GATES THE CLAIM: cos(PPMI rows) full-rank = +0.485 ~= Oja's +0.48. So the category
structure is ALREADY in the PPMI input cosines. Is the LEARNING load-bearing, or just the PPMI input? A
RANDOM (untrained) projection of PPMI approximately preserves cosine (Johnson-Lindenstrauss) -> if it ALSO
reaches ~+0.48, the learner's job is similarity-PRESERVATION (don't saturate), and the load-bearing operation
is the PPMI INPUT ENCODING (log + marginal-ratio + threshold = brain-realizable as dendritic-log +
Phase-1 divisive-gain + spike-threshold). That is an HONEST, important reframe -- report it either way.

ARMS (multi-seed real corpus):
  CEILINGS  offline PPMI+PCA(k); host ppmi_svd_sim(k=50); cos(PPMI rows) uncentered/centered.
  LEARNER   Oja-subspace on CENTERED PPMI            -- the brain-plausible online local PCA rule.
  CONTROLS  random-projection(PPMI) [is learning load-bearing?]; Oja on RAW [input lesion]; faithful
            truly-saturating Hebbian [the destroy-the-structure failure]; permuted-similarity [artifact].

GATES (GO): host_carries; A_saturating_fails(<=0.15); learner_reaches(Oja >= 0.70*offline AND >= 0.30);
  generalizes(gen > chance+margin); permuted_collapses; input_lesion(Oja_PPMI >> Oja_raw). The random-proj
  control does NOT gate GO -- it gates the HONEST FRAMING (learning-extracts vs input-encodes-+-learning-
  preserves). CPU/numpy, seed-independent corpus -> build once, vary the learner seed. NO sim/ edits.

Run: python -u -m research.runners._l1_oja_validated --n-hub 2000 --seeds 42,43,44
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
    build_real_corpus, ppmi_matrix, pca_lowrank_sim, encode_raw, learn_faithful_saturating,
)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners._l1_centered_online_pca_probe import oja_subspace, center_cols  # noqa: E402


def random_projection(X, k, seed):
    """Untrained random Gaussian projection (the 'is learning load-bearing' control). Unit-normalize input ->
    JL approximately preserves cosine -> reveals how much structure survives with NO learning."""
    rng = np.random.RandomState(seed * 7919 + 11)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, X.shape[1]) / np.sqrt(X.shape[1])
    return (W @ Xn.T).T


def _stat(codes, S_true, labels):
    pear = _pearson_vs_Strue(_cos_sim(codes), S_true)
    g, ch = heldout_generalization(codes, labels)
    off = float(_cos_sim(codes)[np.triu_indices(codes.shape[0], 1)].mean())
    return pear, off, g, ch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-hub", type=int, default=2000)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--host-alpha", type=float, default=0.75)
    p.add_argument("--structure-bar", type=float, default=0.30)
    p.add_argument("--a-fail-bar", type=float, default=0.15)
    p.add_argument("--host-bar", type=float, default=0.30)
    p.add_argument("--gen-margin", type=float, default=0.10)
    p.add_argument("--reach-frac", type=float, default=0.70)
    p.add_argument("--out", default="research/findings/raw/_l1_oja_validated.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    C, labels, S_true = build_real_corpus(42, args.n_hub)
    Xppmi = ppmi_matrix(C, args.host_alpha); Xppmi_c = center_cols(Xppmi); Xraw = encode_raw(C)
    offline = _pearson_vs_Strue(pca_lowrank_sim(Xppmi, args.k), S_true)
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=args.host_alpha)
    host_p, _, host_nn, _ = score(host_sim, labels)
    ppmi_cos = _pearson_vs_Strue(_cos_sim(Xppmi), S_true)
    print(f"[L1 Oja validated] {C.shape[0]} concepts x {C.shape[1]} hubs", flush=True)
    print(f"  CEILINGS: offline PPMI+PCA(k={args.k})={offline:+.3f} | host ppmi_svd_sim(k=50)={host_p:+.3f} "
          f"(nn-same {host_nn:.3f}) | cos(PPMI rows)={ppmi_cos:+.3f}", flush=True)

    def multiseed(fn):
        ps, offs, gens, chs = [], [], [], []
        for s in seeds:
            codes = fn(s)
            pe, of, g, ch = _stat(codes, S_true, labels)
            ps.append(pe); offs.append(of); gens.append(g); chs.append(ch)
        return {"pearson_mean": float(np.mean(ps)), "pearson_seeds": ps, "offdiag_mean": float(np.mean(offs)),
                "gen_mean": float(np.mean(gens)), "chance": float(np.mean(chs))}

    oja_ppmi = multiseed(lambda s: oja_subspace(Xppmi_c, args.k, args.epochs, args.lr, s))
    oja_raw = multiseed(lambda s: oja_subspace(Xraw, args.k, args.epochs, args.lr, s))
    randproj = multiseed(lambda s: random_projection(Xppmi, args.k, s))
    sat = multiseed(lambda s: learn_faithful_saturating(Xraw, args.k, args.epochs, 0.02, s)[0])
    # permuted-similarity anti-cheat on the learner codes
    perm_ps = []
    for s in seeds:
        codes = oja_subspace(Xppmi_c, args.k, args.epochs, args.lr, s)
        rng = np.random.RandomState(s * 2718281 + 1)
        perm = rng.permutation(labels)
        S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
        perm_ps.append(_pearson_vs_Strue(_cos_sim(codes), S_perm))
    perm_mean = float(np.mean(perm_ps))

    for nm, r in [("Oja(PPMI centered)", oja_ppmi), ("Oja(raw) [input lesion]", oja_raw),
                  ("random-proj(PPMI) [learn?]", randproj), ("faithful saturating [fail]", sat)]:
        print(f"  [{nm:30s}] Pearson={r['pearson_mean']:+.3f} {['%+.3f'%x for x in r['pearson_seeds']]}  "
              f"offdiag={r['offdiag_mean']:+.3f}  gen={r['gen_mean']:.3f}", flush=True)
    print(f"  [anti-cheat] Oja permuted-similarity Pearson={perm_mean:+.3f} (~0)", flush=True)

    chance = oja_ppmi["chance"]
    reach = args.reach_frac * max(offline, 1e-9)
    gates = {
        "host_carries": bool(offline >= args.host_bar),
        "A_saturating_fails": bool(abs(sat["pearson_mean"]) <= args.a_fail_bar),
        "learner_reaches_ceiling": bool(oja_ppmi["pearson_mean"] >= reach and oja_ppmi["pearson_mean"] >= args.structure_bar),
        "generalizes": bool(oja_ppmi["gen_mean"] > chance + args.gen_margin),
        "permuted_collapses": bool(abs(perm_mean) <= args.a_fail_bar),
        "input_lesion_collapses": bool(oja_ppmi["pearson_mean"] >= oja_raw["pearson_mean"] + 0.10),
    }
    all_go = all(gates.values())
    # the honesty axis: is the LEARNING load-bearing, or the PPMI INPUT?
    learning_adds = oja_ppmi["pearson_mean"] - randproj["pearson_mean"]
    learning_loadbearing = learning_adds >= 0.10
    frac = oja_ppmi["pearson_mean"] / offline if offline > 1e-9 else 0.0
    print(f"\n  [gates] {gates}  ALL={all_go}", flush=True)
    print(f"  [honesty] random-proj(PPMI)={randproj['pearson_mean']:+.3f} vs Oja(PPMI)={oja_ppmi['pearson_mean']:+.3f}"
          f"  -> learning adds {learning_adds:+.3f} ({'LOAD-BEARING' if learning_loadbearing else 'PRESERVATION-ONLY: the PPMI INPUT encodes the structure; the learner just must NOT saturate'})",
          flush=True)

    if all_go and learning_loadbearing:
        verdict = "GO_LEARNING_EXTRACTS"
        why = (f"Oja online local PCA on PPMI reaches {oja_ppmi['pearson_mean']:+.3f} ({frac:.0%} of offline "
               f"{offline:+.3f}, beating host {host_p:+.3f}), generalizes {oja_ppmi['gen_mean']:.3f}, controls "
               f"clean, AND the learning is load-bearing (adds {learning_adds:+.3f} over random projection).")
    elif all_go:
        verdict = "GO_INPUT_ENCODES_LEARNER_PRESERVES"
        why = (f"Oja online local PCA on PPMI reaches {oja_ppmi['pearson_mean']:+.3f} ({frac:.0%} of offline "
               f"{offline:+.3f}, beating host {host_p:+.3f}), generalizes {oja_ppmi['gen_mean']:.3f}, controls "
               f"clean. HONEST REFRAME: the load-bearing operation is the PPMI INPUT ENCODING (the structure is "
               f"already in cos(PPMI rows)={ppmi_cos:+.3f}; random-proj reaches {randproj['pearson_mean']:+.3f}); "
               f"the learner's job is similarity-PRESERVATION (a saturating learner DESTROYS it -> "
               f"{sat['pearson_mean']:+.3f}; Oja preserves it). PPMI = log(dendritic) + /marginal(Phase-1 "
               f"divisive gain) + threshold(spike) -> the brain-based pieces EXIST. ==> the learned graded "
               f"cortex from real experience is VIABLE: PPMI-shaped input + a non-saturating (Oja-class) online "
               f"code. The CYCLE-53 'data-limited' + the sweep's apparent NEGATIVE are BOTH overturned.")
    else:
        verdict = "NOT_CLEARED"
        why = f"gates not all green ({gates}); the GO does not hold under the full battery."
    print(f"\n{'='*96}\n  L1 VALIDATED VERDICT: {verdict}\n  {why}\n{'='*96}", flush=True)
    print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    out = {"verdict": verdict, "why": why, "gates": gates, "all_go": all_go,
           "learning_loadbearing": learning_loadbearing, "learning_adds_over_random": learning_adds,
           "offline_pca": offline, "host_ppmi_svd": host_p, "ppmi_cos": ppmi_cos,
           "oja_ppmi": oja_ppmi, "oja_raw": oja_raw, "random_proj": randproj, "faithful_saturating": sat,
           "permuted_mean": perm_mean, "reach_fraction": frac, "seeds": seeds, "n_hub": args.n_hub, "k": args.k}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
