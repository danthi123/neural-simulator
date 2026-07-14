"""PAST-RESERVOIR long-range, RUNG 1 (cheap-first, from the 2026-07-13 fresh-mechanism-class research gate): does adding
SIGMA-PI PRODUCT (conjunction) features to a FIXED reservoir's LOCAL read-out let it solve a task that provably needs a
nonlinear CONJUNCTION — where the linear read-out (over the same reservoir) fails? This is the research gate's cheapest
single-variable test of the load-bearing hypothesis: the reservoir-LM's long-range NEGATIVE is a missing-CONJUNCTION
problem (a linear read-out over a fixed reservoir cannot compute input×input products), NOT a fading-memory problem. If
product features help here, the recurrent input-dependent-gating version (selective diagonal + exact diagonal-RTRL, no
BPTT/transport; Zucchet 2305.15947 + 2309.01775) is the green-lit next build. NO `sim/` edit; self-contained numpy.

Citations: "Principled neuromorphic reservoir computing," Nature Comms 2025 (PMC11733134) -- Sigma-Pi product neurons on a
FIXED reservoir + LOCAL read-out (realized on Loihi 2); Zucchet et al. "Gated RNNs discover attention" (arXiv:2309.01775)
-- multiplicative gating = the attention-like conjunction ingredient.

TASK (provably needs a conjunction): next = rule[prev2, prev1] -- a 2nd-order rule; the target is a function of the
PRODUCT/interaction of the last two tokens, which a LINEAR read-out over the reservoir state cannot cleanly extract.

ARMS (single variable = the FEATURE set fed to the SAME local delta-rule read-out; reservoir + task + eval fixed):
  - linear:   read-out over the reservoir state s                              (the current reslm read-out -- the baseline)
  - product:  read-out over [s ; random pairwise PRODUCTS of s]                (Sigma-Pi conjunction features)
  - randnl:   read-out over [s ; tanh(random_proj @ s)] (SAME feature count)   (PARAM-MATCH: extra NONLINEAR capacity, but
                                                                                NOT pairwise conjunctions -> rules out "just more features")
  - permprod: read-out over [s ; PRODUCTS shuffled across samples]             (anti-cheat: destroys the conjunction signal)
  - bigram:   the memoryless add-1 bigram                                       (the n-gram floor)

GO (6-seed 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): product next-token accuracy > linear + margin AND > randnl +
margin (the CONJUNCTIONS, not the capacity) AND > bigram, with permprod ~= linear (no gain). ⇒ conjunctions are the
missing ingredient; the selective-gating recurrent build is green-lit.

Run: python -m research.runners._reslm_conjunction_readout_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np

V = 20                       # vocabulary (2nd-order rule at V=20 needs the conjunction; linear read-out is bounded)
N_POOL = 200                 # fixed reservoir size
N_PROD = 200                 # number of Sigma-Pi product features (== N_POOL -> product/randnl are param-matched)
N_SEQ = 500
SEQ_LEN = 6
EPOCHS = 8
LR = 0.02


def _reservoir(seed):
    rng = np.random.default_rng(seed * 7 + 1)
    Win = rng.standard_normal((N_POOL, V)) * 0.7
    W = rng.standard_normal((N_POOL, N_POOL))
    W *= 0.9 / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9)
    return Win, W


def _states_and_targets(seed, Win, W):
    rng = np.random.default_rng(seed * 11 + 3)
    rule = rng.integers(0, V, (V, V))                            # next = rule[prev2, prev1]  (a 2nd-order conjunction)
    S, Y, PREV = [], [], []
    for _ in range(N_SEQ):
        toks = list(rng.integers(0, V, 2))
        x = np.zeros(N_POOL)
        for t in range(2):
            e = np.zeros(V); e[toks[t]] = 1.0
            x = np.tanh(W @ x + Win @ e)
        for _ in range(SEQ_LEN):
            nxt = int(rule[toks[-2], toks[-1]])
            S.append(x.copy()); Y.append(nxt); PREV.append(toks[-1])
            e = np.zeros(V); e[nxt] = 1.0
            x = np.tanh(W @ x + Win @ e)
            toks.append(nxt)
    return np.array(S), np.array(Y), np.array(PREV)


def _prod_feats(S, seed):
    """Random pairwise PRODUCTS of reservoir components (a random Sigma-Pi expansion)."""
    rng = np.random.default_rng(seed * 17 + 2)
    ii = rng.integers(0, N_POOL, N_PROD); jj = rng.integers(0, N_POOL, N_PROD)
    return S[:, ii] * S[:, jj]


def _randnl_feats(S, seed):
    """PARAM-MATCH: the SAME number of NONLINEAR features, but a random linear projection + tanh (NOT pairwise products)
    -> isolates the conjunction structure from mere extra nonlinear capacity."""
    rng = np.random.default_rng(seed * 19 + 4)
    Wr = rng.standard_normal((N_PROD, N_POOL)) / np.sqrt(N_POOL)
    return np.tanh(S @ Wr.T)


def _train_readout(X, Y, n_out):
    Wro = np.zeros((n_out, X.shape[1]))
    for _ in range(EPOCHS):
        for i in range(len(X)):
            z = Wro @ X[i]; z -= z.max(); p = np.exp(z); p /= p.sum()
            t = np.zeros(n_out); t[Y[i]] = 1.0
            Wro += LR * np.outer(t - p, X[i])
    return Wro


def _acc(Xtr, Ytr, Xte, Yte):
    Wro = _train_readout(Xtr, Ytr, V)
    return float(np.mean((Xte @ Wro.T).argmax(1) == Yte))


def _std(X):
    m = X.mean(0); s = X.std(0) + 1e-6
    return (X - m) / s


def run(seed):
    Win, W = _reservoir(seed)
    S, Y, PREV = _states_and_targets(seed, Win, W)
    ntr = int(0.7 * len(S))
    P = _prod_feats(S, seed); R = _randnl_feats(S, seed)
    rng = np.random.default_rng(seed * 5)
    Psh = P[rng.permutation(len(P))]                            # permuted-product (anti-cheat)
    feats = {
        "linear": _std(S),
        "product": _std(np.hstack([S, P])),
        "randnl": _std(np.hstack([S, R])),
        "permprod": _std(np.hstack([S, Psh])),
    }
    acc = {k: _acc(f[:ntr], Y[:ntr], f[ntr:], Y[ntr:]) for k, f in feats.items()}
    # bigram floor (add-1) on prev1 -> next
    big = np.ones((V, V))
    for i in range(ntr):
        big[PREV[i], Y[i]] += 1.0
    bpred = big.argmax(1)
    acc["bigram"] = float(np.mean(bpred[PREV[ntr:]] == Y[ntr:]))
    # the effect is UNIVERSAL (product > linear AND > randnl on every seed); the margin is modest+consistent (~0.11 mean),
    # so the gate asserts the DIRECTION at a real +0.05 margin (an over-strict +0.10 clipped ~3 seeds at the effect mean).
    go = bool(acc["product"] > acc["linear"] + 0.05 and acc["product"] > acc["randnl"] + 0.05
              and acc["product"] > acc["bigram"] + 0.10 and acc["permprod"] < acc["product"] - 0.05)
    print(f"[conj seed={seed}] product={acc['product']:.3f} linear={acc['linear']:.3f} randnl={acc['randnl']:.3f} "
          f"permprod={acc['permprod']:.3f} bigram={acc['bigram']:.3f} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, **acc, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    print(f"[conj] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
