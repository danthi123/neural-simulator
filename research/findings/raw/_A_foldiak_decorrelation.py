"""A deep-grounding arc, cheap-first de-risk: does the Földiák 1990 anti-Hebbian decorrelating network (the on-bridge-
realizable biological replacement for the numpy ZCA the 2026-06-04 grounding used) decorrelate a correlated grounded
codebook AS WELL AS ZCA? The 2026-06-04 finding showed a correlated V1+word codebook collapses composition to 0% raw
and 100% once ZCA-decorrelated; the cheat-A research's named residual is to replace that numpy ZCA with a local-rule
on-bridge decorrelation (Földiák: Hebbian feed-forward W + anti-Hebbian lateral V -> sparse decorrelated codes).
GATE: Földiák output cross-concept coherence ≈ ZCA's (both << raw). If GO, build it on the bridge. Honest: Földiák
local rules APPROXIMATE, not equal, ZCA (cheat-A research) -- measure the gap.
"""
import numpy as np


def make_correlated_codebook(n_concepts, n_feat, n_blocks, seed):
    """Concepts grouped in blocks; within-block strongly correlated (the V1-block / word-block modality structure of
    the 2026-06-04 finding -- e.g. all adjectives live in the word block -> correlated -> composition collapses)."""
    rng = np.random.default_rng(seed)
    block = rng.standard_normal((n_blocks, n_feat))
    X = np.zeros((n_concepts, n_feat))
    for i in range(n_concepts):
        X[i] = 0.75 * block[i % n_blocks] + 0.25 * rng.standard_normal(n_feat)
    return np.maximum(X, 0)   # non-negative (firing-rate-like grounded features)


def coherence(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    G = np.abs(Xn @ Xn.T)
    off = G[~np.eye(len(X), dtype=bool)]
    return float(off.mean()), float(off.max())


def zca(X):
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / len(X)
    U, S, _ = np.linalg.svd(C + 1e-2 * np.eye(X.shape[1]))
    return Xc @ (U @ np.diag(1.0 / np.sqrt(S)) @ U.T)


def foldiak(X, n_out, n_epochs=600, aq=0.05, aw=0.15, at=0.08, p=0.08, settle=20, seed=42):
    """Földiák 1990 decorrelating sparse coder (correct form): BINARY threshold outputs, feed-forward Q (Hebbian
    toward x), inhibitory lateral W (anti-Hebbian toward target p² -> decorrelates), adaptive thresholds t (keep each
    output active ~p of the time -> prevents collapse). y_i = 1 if (Q_i x - W_i y - t_i) > 0."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Q = rng.standard_normal((n_out, d)) * (1.0 / np.sqrt(d))
    W = np.zeros((n_out, n_out))
    t = np.zeros(n_out)
    Xs = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    def settle_out(x):
        s = Q @ x
        y = (s - t > 0).astype(float)
        for _ in range(settle):
            y = (s - W @ y - t > 0).astype(float)
        return y

    for _ in range(n_epochs):
        for x in Xs[rng.permutation(n)]:
            y = settle_out(x)
            Q += aq * y[:, None] * (x[None, :] - Q)            # Hebbian feed-forward (toward x for active outputs)
            W += aw * (np.outer(y, y) - p * p)                 # anti-Hebbian lateral, target p^2 -> decorrelate
            np.fill_diagonal(W, 0.0)
            W = np.maximum(W, 0.0)
            t += at * (y - p)                                  # threshold adapt -> each output active ~p (no collapse)
    return np.stack([settle_out(x) for x in Xs])


if __name__ == "__main__":
    for seed in (42, 43, 44):
        X = make_correlated_codebook(n_concepts=16, n_feat=256, n_blocks=4, seed=seed)
        Yz = zca(X)
        Yf = foldiak(X, n_out=256, seed=seed)
        rm, rx = coherence(X)
        zm, zx = coherence(Yz)
        fm, fx = coherence(Yf)
        print(f"seed={seed}: cross-concept coherence  RAW mean={rm:.3f}/max={rx:.3f}  "
              f"ZCA mean={zm:.3f}/max={zx:.3f}  FOLDIAK mean={fm:.3f}/max={fx:.3f}", flush=True)
