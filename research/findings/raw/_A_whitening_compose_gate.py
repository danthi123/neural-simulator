"""Option-1 RIGOROUS test (owner: "proceed regardless, make all necessary preparations"). The two prior de-risks each
nearly shipped a false positive judged on COHERENCE; this gates on COMPOSITION (the agent benchmark) and fixes the
rank-deficiency that blew up the learning rule.

Design (lessons baked in):
- Whiten in a K≤N SUBSPACE (the realizable IT-pool dimension), so the rule cannot amplify an empty null-space → no M
  blow-up. K=300 ≤ N=320 concepts → full rank.
- GATE ON COMPOSITION, not coherence: a noise-collapse passes coherence but won't compose.
- CONTROLS bracket every result: raw grounded codes (≈66.7% floor) and analytic CONCEPT-whiten (the proven 100% target).
- Guards: M-ratio (learned vs analytic) + blow-up detector.

Conditions, each → phases → NestedCompositionAgent → composition %:
  (1) RAW            — no whitening (floor control)
  (2) CONCEPT-whiten — _decorrelate (N×N gram; the proven 100% target control)
  (3) DIM-analytic   — project to K, analytic C_K^{-1/2} (the substrate-realizable analytic; UNTESTED for composition)
  (4) LEARNED        — project to K, LEARN the K×K whitening (ΔM∝⟨yyᵀ⟩−I) (the open question)
NO sim/ edits. CIFAR grounding (Track A); reuse-by-import.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from research.runners.unified_agent_realobject_grounded import build_realobject_features, run_seed
from research.runners.unified_agent_visual_grounded import _decorrelate
from research.runners.unified_agent_benchmark import build_vocab, aggregate
from research.runners._visual_grounding_probe import _v1_matrix


def _proj(feat_dim, K, seed):
    return np.random.default_rng(seed).standard_normal((feat_dim, K)) / np.sqrt(feat_dim)


def _analytic_whiten_K(Zc, eps):
    C = Zc.T @ Zc / len(Zc)
    w, V = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
    w = np.clip(w, 1e-9, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T, (V * np.sqrt(w)) @ V.T   # C^-1/2, C^1/2


def dim_analytic(feats, K, seed, eps=1e-2):
    Z = feats @ _proj(feats.shape[1], K, seed)
    Zc = Z - Z.mean(0)
    Cinv_sqrt, _ = _analytic_whiten_K(Zc, eps)
    return Zc @ Cinv_sqrt


def learned_whiten(feats, K, seed, n_iters=4000, eta=0.01, eps=1e-2):
    """Project to K (≤N → full rank, no null-space blow-up) then LEARN the K×K whitening lateral M via ΔM∝⟨yyᵀ⟩−I.
    Returns the whitened codes, the M-ratio vs analytic, and a blow-up flag."""
    Z = feats @ _proj(feats.shape[1], K, seed)
    Zc = Z - Z.mean(0)
    Cinv_sqrt, Csqrt = _analytic_whiten_K(Zc, eps)
    M_analytic = Csqrt - np.eye(K)
    I = np.eye(K)
    M = np.zeros((K, K))
    blew = False
    for _ in range(n_iters):
        Y = np.linalg.solve(I + M, Zc.T).T
        if not np.all(np.isfinite(Y)) or np.abs(Y).max() > 1e6:
            blew = True
            break
        M = M + eta * (Y.T @ Y / len(Zc) - I)
        M = 0.5 * (M + M.T)
    Y = np.linalg.solve(I + M, Zc.T).T if not blew else Zc @ Cinv_sqrt
    mratio = float(np.linalg.norm(M - M_analytic) / (np.linalg.norm(M_analytic) + 1e-9))
    return Y, mratio, blew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--out", default="research/findings/raw/_A_whitening_compose_gate.json")
    args = ap.parse_args()
    nouns, verbs, adjs = build_vocab()
    W, _ = _v1_matrix()
    feats, dim, tokens, src = build_realobject_features(nouns, verbs, adjs, W, seed=args.seeds[0])
    print(f"=== whitening COMPOSITION gate | grounding={src} | {len(tokens)} concepts | K={args.K} ===", flush=True)

    def comp(label, codes):
        d = codes.shape[1]
        seed_res = [run_seed(s, codes, d, tokens, nouns, verbs, adjs, decorrelate=False) for s in args.seeds]
        _, gok, gtot = aggregate(seed_res)
        return gok, gtot, seed_res

    out = {"source": src, "K": args.K, "seeds": args.seeds}
    # (1) RAW floor control, (2) CONCEPT-whiten 100% target control
    for label, codes in (("RAW (floor control)", feats), ("CONCEPT-whiten (100% target control)", _decorrelate(feats))):
        gok, gtot, _ = comp(label, codes)
        out[label] = [gok, gtot]
        print(f"  {label:<40} {gok}/{gtot} = {gok/gtot*100:.1f}%", flush=True)
    # (3) DIM-analytic (realizable analytic)
    gok, gtot, _ = comp("DIM-analytic", dim_analytic(feats, args.K, args.seeds[0]))
    out["DIM-analytic"] = [gok, gtot]
    print(f"  {'DIM-analytic (realizable, K-subspace)':<40} {gok}/{gtot} = {gok/gtot*100:.1f}%", flush=True)
    # (4) LEARNED whitening
    Yl, mratio, blew = learned_whiten(feats, args.K, args.seeds[0])
    gok, gtot, _ = comp("LEARNED", Yl)
    out["LEARNED"] = {"compose": [gok, gtot], "m_ratio": mratio, "blew_up": blew}
    guard = "  ⚠ M BLEW UP" if (blew or mratio > 0.5) else "  (M matches analytic)"
    print(f"  {'LEARNED (local rule, K-subspace)':<40} {gok}/{gtot} = {gok/gtot*100:.1f}%   "
          f"| M-ratio={mratio:.2f} blew={blew}{guard}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
