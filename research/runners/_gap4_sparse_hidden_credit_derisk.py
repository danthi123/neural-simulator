"""gap#4 spiking-proxy de-risk — does the working RATE credit rule survive a SPARSE (spiking-like) hidden code?

The MNIST de-risk (`_gap4_credit_vs_reservoir_mnist_derisk.py`, 6-seed) showed the biological local credit rule (FA/DFA)
beats a reservoir on a proper deep task in RATE. The deep-research's remaining gap#4 open piece (cause #3): the SAME BDSP
algorithm reaches accuracy on numpy graded signals but degenerates ON the sparse spiking bridge (firing 0.04-0.07) -- the
sparse spike code may not carry the graded class signal. This isolates the SPARSITY effect cheaply on the proper task
(MNIST) WITHOUT the full on-bridge build: make the hidden a SPARSE-BINARY code (top-k% active, like spikes) with a
straight-through estimator (STE) for the credit (the standard spiking-net surrogate: forward = binary, credit derivative
= the dense sigmoid derivative). Sweep sparsity dense->2%; does FA collapse toward the reservoir as the code gets sparser?

- FA robust to sparsity (still >> reservoir at 2-5%): the SPARSE CODE per se is NOT the blocker -> the on-bridge negative
  is something else (op-point / burst-credit-specific / population size), a narrower diagnosis.
- FA collapses toward the reservoir as sparsity rises: the sparse code IS the blocker -> the frontier is carrying the
  graded class signal in few spikes (population coding / more spikes), confirming cause #3.

RESERVOIR (frozen sparse-random hidden + trained readout) is the credit-independent baseline at EACH sparsity. numpy CPU.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
from sim.dendritic_mlp import DendriticMLP, _MOMENTUM, _sig  # noqa: E402

xp, _ = get_backend()


def _sparsify(a, frac):
    """Top-`frac` fraction active per sample -> binary sparse code (like a spike raster). frac>=1.0 -> dense sigmoid."""
    if frac >= 1.0:
        return a
    k = max(1, int(round(frac * a.shape[1])))
    thr = xp.sort(a, axis=1)[:, -k][:, None]        # per-sample k-th largest
    return (a >= thr).astype(a.dtype)


class SparseNet(DendriticMLP):
    """DendriticMLP with a SPARSE-BINARY hidden code + STE credit (forward binary; credit uses the dense sigmoid
    derivative). Also a reservoir mode (freeze hidden). Sparsity `frac` applied to every hidden layer."""

    def __init__(self, sizes, seed=0, frac=1.0):
        super().__init__(sizes, seed=seed)
        self.frac = frac

    def _forward_dense_sparse(self, X):
        """Return (dense_acts, sparse_acts, logits): dense for the STE derivative, sparse for the forward pass."""
        dense = [xp.asarray(X, float)]
        sparse = [xp.asarray(X, float)]
        for li in range(len(self.W) - 1):
            pre = sparse[-1] @ self.W[li]
            d = _sig(pre)
            dense.append(d)
            sparse.append(_sparsify(d, self.frac))
        return dense, sparse, sparse[-1] @ self.W[-1]

    def _fwd_err(self, X, y):
        dense, sparse, lg = self._forward_dense_sparse(X)
        y = xp.asarray(y)
        e = self._softmax_local(lg)
        e[xp.arange(len(y)), y] -= 1.0
        return dense, sparse, e

    @staticmethod
    def _softmax_local(z):
        z = z - z.max(1, keepdims=True)
        ez = xp.exp(z)
        return ez / ez.sum(1, keepdims=True)

    def accuracy(self, X, y):
        _, _, lg = self._forward_dense_sparse(X)
        y = xp.asarray(y)
        return float(to_host(xp.mean(xp.argmax(lg, 1) == y)))

    def train_step_sparse(self, X, y, mode, lr):
        dense, sparse, e = self._fwd_err(X, y)
        nW = len(self.W)
        upd = [None] * nW
        upd[-1] = -(sparse[-1].T @ e)                 # readout uses the sparse hidden (trained in all modes)
        for li in range(nW - 1):
            if mode == "reservoir":
                upd[li] = xp.zeros_like(self.W[li]); continue
            a_prev_sparse, d_l = sparse[li], dense[li + 1]
            ap = e @ self.B[li]                        # fixed-random DFA feedback
            # STE: forward uses the SPARSE code (a_prev_sparse); credit derivative uses the DENSE sigmoid derivative
            base = a_prev_sparse.T @ (ap * d_l * (1.0 - d_l))
            upd[li] = -base
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [xp.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]


def _load_mnist(n_train, n_test, seed):
    d = np.load("data/mnist.npz")
    rng = np.random.default_rng(seed)
    xtr = d["x_train"].reshape(-1, 784).astype(np.float64) / 255.0
    xte = d["x_test"].reshape(-1, 784).astype(np.float64) / 255.0
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    itr = rng.choice(len(xtr), n_train, replace=False)
    ite = rng.choice(len(xte), n_test, replace=False)
    return xtr[itr], d["y_train"][itr].astype(np.int64), xte[ite], d["y_test"][ite].astype(np.int64)


def _run(mode, Xtr, ytr, Xte, yte, sizes, seed, frac, epochs, batch, lr):
    net = SparseNet(sizes, seed=seed, frac=frac)
    rng = np.random.default_rng(seed * 131 + 7)
    for _ in range(epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            idx = order[s:s + batch]
            net.train_step_sparse(Xtr[idx], ytr[idx], mode, lr)
    return net.accuracy(Xte, yte)


def one_seed(seed, a):
    Xtr, ytr, Xte, yte = _load_mnist(a.n_train, a.n_test, seed)
    sizes = [784] + [a.hidden] * a.depth + [10]
    rows = []
    for frac in a.fracs:
        fa = _run("fa", Xtr, ytr, Xte, yte, sizes, seed, frac, a.epochs, a.batch, a.lr)
        res = _run("reservoir", Xtr, ytr, Xte, yte, sizes, seed, frac, a.epochs, a.batch, a.lr)
        rows.append(dict(frac=frac, fa=round(float(fa), 4), reservoir=round(float(res), 4),
                         fa_beats_res=bool(fa > res + 0.01)))
        print(f"  [seed {seed}] sparsity={frac:.2f} (active {frac*100:.0f}%): FA={fa:.3f} RESERVOIR={res:.3f} "
              f"| FA>RES:{fa > res + 0.01}")
    return dict(seed=seed, sizes=sizes, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.2, 0.1, 0.05, 0.02])
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--out", default="research/findings/raw/gap4/sparse_hidden_credit.json")
    a = ap.parse_args()
    _, backend = get_backend()
    print(f"[gap4-sparse] does FA credit survive a SPARSE-BINARY hidden (STE)? hidden={a.hidden} depth={a.depth} "
          f"fracs={a.fracs} seeds={a.seeds} backend={backend}")
    per = [one_seed(s, a) for s in a.seeds]
    # aggregate: at each frac, how many seeds FA>RES + mean gap
    print("[gap4-sparse] SUMMARY (mean over seeds):")
    for i, frac in enumerate(a.fracs):
        fas = [p["rows"][i]["fa"] for p in per]
        ress = [p["rows"][i]["reservoir"] for p in per]
        nbeat = sum(p["rows"][i]["fa_beats_res"] for p in per)
        print(f"  sparsity={frac:.2f}: FA={np.mean(fas):.3f} RESERVOIR={np.mean(ress):.3f} gap={np.mean(fas)-np.mean(ress):+.3f} FA>RES {nbeat}/{len(per)}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, per=per), f, indent=2)


if __name__ == "__main__":
    main()
