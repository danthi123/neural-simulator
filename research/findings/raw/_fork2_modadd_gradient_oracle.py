"""Disambiguation oracle for EMERGE-7: is the held-out (p,m)->g=(p+m)%n_suffix split learnable-by-GRADIENT AT ALL,
given the factors p,m directly (no context-carrying)? A 2-layer MLP trained with backprop on the 12 off-diagonal
cells, evaluated on the 4 held diagonal cells. If it GENERALIZES -> the held-out split is gradient-learnable, so the
RNN+local-rule failure is a genuine mechanism gap (BOUNDARY). If it FAILS too -> the held-out modular-addition split
is grokking-hard at this scale = a TASK-design issue (reframe the milestone), not a clean mechanism boundary.
"""
import numpy as np


def oracle(n=4, hidden=64, epochs=4000, lr=0.1, wd=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    cells = [(p, m) for p in range(n) for m in range(n)]
    held = [(p, p) for p in range(n)]
    train = [c for c in cells if c not in held]

    def feat(p, m):
        v = np.zeros(2 * n); v[p] = 1.0; v[n + m] = 1.0; return v      # onehot(p) concat onehot(m)

    W1 = rng.normal(0, 1 / np.sqrt(2 * n), (hidden, 2 * n)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), (n, hidden)); b2 = np.zeros(n)
    for ep in range(epochs):
        for (p, m) in [train[i] for i in rng.permutation(len(train))]:
            x = feat(p, m); g = (p + m) % n
            h = np.maximum(0, W1 @ x + b1)
            lo = W2 @ h + b2; pr = np.exp(lo - lo.max()); pr /= pr.sum()
            d = pr.copy(); d[g] -= 1.0
            gW2 = np.outer(d, h) + wd * W2; gb2 = d
            dh = (W2.T @ d) * (h > 0)
            gW1 = np.outer(dh, x) + wd * W1; gb1 = dh
            W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1

    def acc(cellset):
        ok = 0
        for (p, m) in cellset:
            h = np.maximum(0, W1 @ feat(p, m) + b1); lo = W2 @ h + b2
            ok += int(int(np.argmax(lo)) == (p + m) % n)
        return ok / len(cellset)
    return acc(train), acc(held)


if __name__ == "__main__":
    for seed in (0, 1, 2):
        tr, he = oracle(seed=seed)
        print(f"seed {seed}: MLP train-acc {tr:.3f}  held-acc {he:.3f}  (chance {1/4:.3f})")
