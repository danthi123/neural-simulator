"""gap#4 — a FAITHFUL numpy replica of the on-bridge BDSP rule on MNIST: WHICH feature degenerates it?

Reading the actual kernel (`sim/kernels.py fused_bdsp_update`, M1.2) corrected my earlier guess. The on-bridge rule is

    dw_ij = eta * Etilde_pre_j * E_post_i * (P_post_i - Pbar_post_i)     [Payeur-Naud 2021 M1.2]

which differs from the WORKING FA rate rule (`sim.dendritic_mlp`, `base = a_prev.T @ (e@B * sig')`) in TWO faithful ways:
  (1) COINCIDENCE GATE: the update is gated by the product of pre AND post EVENT (spike) rates `Etilde_pre * E_post`.
      At firing 0.05 that product is ~0.0025 -- a ~400x starvation vs the dense case. My sparse-hidden de-risk gated only
      the PRE side (a_prev binary) and FA still beat the reservoir 3-seed at 2% -- so the PRE gate alone is NOT the
      blocker. The UNTESTED half is the POST gate (true pre*post coincidence).
  (2) SIGMOID+BASELINE CREDIT: the credit is `(sigmoid(beta*apical) - Pbar)` (bounded [0,1], baseline-subtracted by a
      slow EMA init p0=0.30), not FA's unbounded linear `e@B`.

This isolates the two features on the SAME MNIST task + the SAME fixed-random DFA feedback B, 4 arms:
  - fa_linear   (the working rule): pre=binary spike, post=DENSE sigmoid deriv, credit=linear e@B.  [= the 3-seed winner]
  - fa_coinc    (add ONLY the coincidence gate): pre=binary, post=BINARY spike, credit=linear.
  - bdsp_nocoinc(add ONLY the sigmoid+baseline credit): pre=binary, post=DENSE deriv, credit=(sigmoid(beta*ap)-Pbar).
  - bdsp        (the FAITHFUL on-bridge rule): pre=binary, post=BINARY spike, credit=(sigmoid(beta*ap)-Pbar).
plus RESERVOIR (frozen hidden) = the credit-independent baseline at each firing rate.

If bdsp collapses toward the reservoir at low firing while fa_linear holds -> we know the on-bridge degeneracy is the
coincidence gate and/or the sigmoid-baseline credit (a precise, narrowed on-bridge fix: population coincidences / a
longer eligibility window / drop the p0 dead-zone), NOT a fundamental "sparse code can't carry credit" wall. numpy CPU.
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
    """Top-`frac` active per sample -> binary spike raster. frac>=1.0 -> dense sigmoid rate."""
    if frac >= 1.0:
        return a
    k = max(1, int(round(frac * a.shape[1])))
    thr = xp.sort(a, axis=1)[:, -k][:, None]
    return (a >= thr).astype(a.dtype)


class BdspNet(DendriticMLP):
    """DendriticMLP with a sparse-binary hidden + selectable credit arm (fa_linear / fa_coinc / bdsp_nocoinc / bdsp /
    reservoir). Faithful to the on-bridge BDSP: post-side gate = the binary spike (event rate), credit = sigmoid+EMA
    baseline. p0 = resting burst-probability baseline; beta = apical->P sigmoid slope."""

    def __init__(self, sizes, seed=0, frac=1.0, p0=0.30, beta=1.0):
        super().__init__(sizes, seed=seed)
        self.frac, self.p0, self.beta = frac, p0, beta
        self.Pbar = [None] * (len(self.W) - 1)   # slow EMA burst-prob baseline per hidden layer (init p0)

    def _forward(self, X):
        dense = [xp.asarray(X, float)]
        sparse = [xp.asarray(X, float)]
        for li in range(len(self.W) - 1):
            d = _sig(sparse[-1] @ self.W[li])
            dense.append(d)
            sparse.append(_sparsify(d, self.frac))
        return dense, sparse, sparse[-1] @ self.W[-1]

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True); ez = xp.exp(z); return ez / ez.sum(1, keepdims=True)

    def accuracy(self, X, y):
        _, _, lg = self._forward(X); y = xp.asarray(y)
        return float(to_host(xp.mean(xp.argmax(lg, 1) == y)))

    def train_step(self, X, y, mode, lr):
        dense, sparse, lg = self._forward(X)
        y = xp.asarray(y); e = self._softmax(lg); e[xp.arange(len(y)), y] -= 1.0
        nW = len(self.W); upd = [None] * nW
        upd[-1] = -(sparse[-1].T @ e)                              # readout trained in all arms (uses sparse hidden)
        for li in range(nW - 1):
            if mode == "reservoir":
                upd[li] = xp.zeros_like(self.W[li]); continue
            a_prev = sparse[li]                                    # PRE side = binary spike (event) in every arm
            ap = e @ self.B[li]                                    # fixed-random DFA apical credit
            d_dense = dense[li + 1]                                # dense sigmoid rate (for STE post-deriv)
            spk_post = sparse[li + 1]                              # POST binary spike (event rate)
            # POST-side gate: coincidence arms use the binary spike; non-coincidence arms use the dense sigmoid deriv.
            post_gate = spk_post if mode in ("fa_coinc", "bdsp") else d_dense * (1.0 - d_dense)
            # CREDIT: bdsp arms use sigmoid(beta*ap)-Pbar (EMA baseline); fa arms use the linear ap.
            if mode in ("bdsp", "bdsp_nocoinc"):
                P = _sig(self.beta * ap)
                if self.Pbar[li] is None:
                    self.Pbar[li] = xp.full(P.shape[1], self.p0)
                self.Pbar[li] = 0.99 * self.Pbar[li] + 0.01 * P.mean(0)   # slow EMA baseline
                credit = P - self.Pbar[li][None, :]
            else:
                credit = ap
            upd[li] = -(a_prev.T @ (credit * post_gate))
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [xp.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]


def _load_mnist(n_train, n_test, seed):
    d = np.load("data/mnist.npz"); rng = np.random.default_rng(seed)
    xtr = d["x_train"].reshape(-1, 784).astype(np.float64) / 255.0
    xte = d["x_test"].reshape(-1, 784).astype(np.float64) / 255.0
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    itr = rng.choice(len(xtr), n_train, replace=False); ite = rng.choice(len(xte), n_test, replace=False)
    return xtr[itr], d["y_train"][itr].astype(np.int64), xte[ite], d["y_test"][ite].astype(np.int64)


def _run(mode, Xtr, ytr, Xte, yte, sizes, seed, frac, a):
    net = BdspNet(sizes, seed=seed, frac=frac, p0=a.p0, beta=a.beta)
    rng = np.random.default_rng(seed * 131 + 7)
    for _ in range(a.epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), a.batch):
            idx = order[s:s + a.batch]
            net.train_step(Xtr[idx], ytr[idx], mode, a.lr)
    return net.accuracy(Xte, yte)


def one_seed(seed, a):
    Xtr, ytr, Xte, yte = _load_mnist(a.n_train, a.n_test, seed)
    sizes = [784] + [a.hidden] * a.depth + [10]
    rows = []
    for frac in a.fracs:
        acc = {m: _run(m, Xtr, ytr, Xte, yte, sizes, seed, frac, a)
               for m in ["reservoir", "fa_linear", "fa_coinc", "bdsp_nocoinc", "bdsp"]}
        rows.append(dict(frac=frac, **{k: round(float(v), 4) for k, v in acc.items()}))
        print(f"  [seed {seed}] frac={frac:.2f}: RES={acc['reservoir']:.3f} fa_lin={acc['fa_linear']:.3f} "
              f"fa_coinc={acc['fa_coinc']:.3f} bdsp_noco={acc['bdsp_nocoinc']:.3f} bdsp={acc['bdsp']:.3f}")
    return dict(seed=seed, sizes=sizes, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.1, 0.05])
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--out", default="research/findings/raw/gap4/bdsp_faithful.json")
    a = ap.parse_args()
    _, backend = get_backend()
    print(f"[gap4-bdsp] faithful on-bridge BDSP replica: which feature degenerates it? hidden={a.hidden} depth={a.depth} "
          f"fracs={a.fracs} p0={a.p0} beta={a.beta} seeds={a.seeds} backend={backend}")
    per = [one_seed(s, a) for s in a.seeds]
    print("[gap4-bdsp] SUMMARY (mean over seeds):")
    for i, frac in enumerate(a.fracs):
        agg = {m: np.mean([p["rows"][i][m] for p in per]) for m in ["reservoir", "fa_linear", "fa_coinc", "bdsp_nocoinc", "bdsp"]}
        print(f"  frac={frac:.2f}: RES={agg['reservoir']:.3f} fa_lin={agg['fa_linear']:.3f} "
              f"fa_coinc={agg['fa_coinc']:.3f} bdsp_noco={agg['bdsp_nocoinc']:.3f} bdsp={agg['bdsp']:.3f} "
              f"| bdsp>RES:{agg['bdsp'] > agg['reservoir'] + 0.01}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, per=per), f, indent=2)


if __name__ == "__main__":
    main()
