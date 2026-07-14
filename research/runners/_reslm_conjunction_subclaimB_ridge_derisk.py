"""Settle Rung-1 Sub-claim B (the adversarial-verify's recommended follow-on): under a CLOSED-FORM RIDGE read-out (zero
epoch/lr dependence), does the Sigma-Pi PRODUCT (conjunction) advantage over a PARAM-MATCHED strong nonlinear basis
(random Fourier features) hold robustly at 6 seeds? The verify found it comfortable under the delta rule (+0.09 vs RFF)
but narrowing to +0.02-0.04 under ridge at 1-2 seeds. This runs it at 6 seeds on BOTH the adjacent and non-adjacent
conjunction tasks. Reuse-by-import of the Rung-1 runner; NO `sim/` edit.

GO/verdict: if product > RFF under ridge by >+0.05 on >=5/6 seeds -> Sub-claim B is robust (conjunction structure, not
capacity, even under regularization). Else -> honestly a small-but-consistent effect (product still > RFF directionally),
and Rung 2+ rest on the ROBUST Sub-claim A (linear cannot extract the conjunction) regardless.

Run: python -m research.runners._reslm_conjunction_subclaimB_ridge_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np

from research.runners._reslm_conjunction_readout_derisk import (
    _reservoir, _states_and_targets, _prod_feats, _std, V, N_POOL, N_PROD)


def _rff_feats(S, seed):
    """Random Fourier features cos(Wf @ s + b) -- a strong, universal nonlinear basis, param-matched to N_PROD."""
    rng = np.random.default_rng(seed * 23 + 8)
    Wf = rng.standard_normal((N_PROD, N_POOL)) / np.sqrt(N_POOL)
    b = rng.uniform(0, 2 * np.pi, N_PROD)
    return np.cos(S @ Wf.T + b)


def _ridge_acc(X, Y, ntr, lam=1.0):
    Ytr = np.eye(V)[Y[:ntr]]
    W = np.linalg.solve(X[:ntr].T @ X[:ntr] + lam * np.eye(X.shape[1]), X[:ntr].T @ Ytr)
    return float(np.mean((X[ntr:] @ W).argmax(1) == Y[ntr:]))


def _second_order(seed, Win, Wres, nonadjacent):
    """Rebuild the states/targets for the adjacent (rule[prev2,prev1]) OR non-adjacent (rule[prev3,prev1]) task."""
    from research.runners._reslm_conjunction_readout_derisk import N_SEQ, SEQ_LEN
    rng = np.random.default_rng(seed * 11 + 3)
    rule = rng.integers(0, V, (V, V))
    S, Y = [], []
    for _ in range(N_SEQ):
        toks = list(rng.integers(0, V, 3 if nonadjacent else 2))
        x = np.zeros(N_POOL)
        for t in toks:
            e = np.zeros(V); e[t] = 1.0; x = np.tanh(Wres @ x + Win @ e)
        for _ in range(SEQ_LEN):
            nxt = int(rule[toks[-3], toks[-1]]) if nonadjacent else int(rule[toks[-2], toks[-1]])
            S.append(x.copy()); Y.append(nxt)
            e = np.zeros(V); e[nxt] = 1.0; x = np.tanh(Wres @ x + Win @ e); toks.append(nxt)
    return np.array(S), np.array(Y)


def run(seed):
    Win, Wres = _reservoir(seed)
    out = {"seed": seed}
    for task, nonadj in (("adjacent", False), ("nonadjacent", True)):
        S, Y = _second_order(seed, Win, Wres, nonadj)
        ntr = int(0.7 * len(S))
        P = _prod_feats(S, seed); R = _rff_feats(S, seed)
        lin = _ridge_acc(_std(S), Y, ntr)
        prod = _ridge_acc(_std(np.hstack([S, P])), Y, ntr)
        rff = _ridge_acc(_std(np.hstack([S, R])), Y, ntr)
        out[task] = {"linear": lin, "product": prod, "rff": rff, "prod_minus_rff": prod - rff, "prod_minus_lin": prod - lin}
    print(f"[subB seed={seed}] ADJ prod-rff={out['adjacent']['prod_minus_rff']:+.3f} (prod {out['adjacent']['product']:.3f} "
          f"rff {out['adjacent']['rff']:.3f}) | NONADJ prod-rff={out['nonadjacent']['prod_minus_rff']:+.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    for task in ("adjacent", "nonadjacent"):
        m = [r[task]["prod_minus_rff"] for r in res]
        n_robust = sum(1 for x in m if x > 0.05)
        print(f"[subB] {task}: product>RFF+0.05 on {n_robust}/{len(res)} seeds (mean prod-rff {np.mean(m):+.3f})", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
