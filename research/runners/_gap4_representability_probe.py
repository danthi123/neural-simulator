"""gap#4 representability probe — the load-bearing diagnostic behind
`research/findings/2026-07-24-gap4-sparse-spiking-forward-representability-degeneracy-...`.

Question: does the on-bridge SPARSE-SPIKING forward pass preserve the input's GENERALIZABLE class structure? Measured by
HELD-OUT decodability (train a probe on the train items' representation, evaluate on held-out inheritance items) — the
INIT forward (no training/credit) is the purest representability read. INPUT is the ceiling (the generalizable structure
that exists). A hidden readout "preserves structure" iff it generalizes near INPUT.

Why HELD-OUT and not train: the raw UNPOOLED hidden code (256 features, 96 samples) is train-separable (mlp 1.000) by
high-dim OVERFITTING — it collapses to chance held-out (0.247). Only held-out distinguishes real structure from overfit.

Result (6 seeds, semantic-inheritance, k=5, n_ho=27): INPUT ho mlp 0.988; every sparse-spiking hidden readout ho <= 0.34
(pooled event 0.34, unpooled event 0.247, graded-soma-V 0.29). => the sparse-spiking forward does not carry
generalizable class structure to the hidden layer. NOT a pooling artifact (unpooled overfits), NOT dendritic
(2026-07-22), NOT credit-at-sparse (credit reaches the weights but has no separable features to shape).

Run: SIM_BACKEND=cupy python -m research.runners._gap4_representability_probe
"""
import os
import sys

os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time

import numpy as np

from research.runners._gap4_onbridge_spiking_selfpredict_derisk import Gap4OnBridgeNet
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance

DRIVE = dict(tonic_h_pA=560, tonic_o_pA=620, in_current_pA=520, in_bias_pA=260, ff_w_init=4.0)
SEEDS = (42, 43, 44, 100, 101, 102)
ROWS = ["INPUT", "H2 event pooled", "H2 event UNPOOLED", "H2 graded-V pooled"]


def _sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_lin(X, y, k, iters=600, lr=0.5, l2=3e-3):
    n, d = X.shape
    W = np.zeros((d, k))
    b = np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(iters):
        P = _sm(X @ W + b)
        g = (P - Y) / n
        W -= lr * (X.T @ g + l2 * W)
        b -= lr * g.sum(0)
    return lambda Z: np.argmax(Z @ W + b, 1)


def fit_mlp(X, y, k, h=64, iters=1500, lr=0.2, l2=1e-3, seed=0):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d, h)) / np.sqrt(d)
    b1 = np.zeros(h)
    W2 = rng.standard_normal((h, k)) / np.sqrt(h)
    b2 = np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(iters):
        Z1 = X @ W1 + b1
        A1 = np.maximum(Z1, 0.0)
        P = _sm(A1 @ W2 + b2)
        gO = (P - Y) / n
        gW2 = A1.T @ gO + l2 * W2
        gb2 = gO.sum(0)
        gZ1 = (gO @ W2.T) * (Z1 > 0)
        gW1 = X.T @ gZ1 + l2 * W1
        gb1 = gZ1.sum(0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2

    def f(Z):
        A = np.maximum(Z @ W1 + b1, 0.0)
        return np.argmax(A @ W2 + b2, 1)
    return f


def _acc(f, X, y):
    return float(np.mean(f(X) == y))


def _ev_pooled(net, X):
    return np.asarray([net._forward_spiking(X[i])[2] for i in range(len(X))])


def _ev_unpooled(net, X):
    from sim.backend import to_host
    o = []
    for i in range(len(X)):
        net._forward_spiking(X[i])
        o.append(np.asarray(to_host(net.br.cp_bdsp_E))[net.slices[2]].copy())
    return np.asarray(o)


def _gr_pooled(net, X):
    o = []
    for i in range(len(X)):
        net._forward_spiking(X[i])
        o.append(net.soma_rate_proxy(2))
    return np.asarray(o)


def main():
    agg = {r: {"tr_lin": [], "ho_lin": [], "tr_mlp": [], "ho_mlp": []} for r in ROWS}
    for seed in SEEDS:
        t0 = time.time()
        (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(
            seed, n_super=12, n_members=8, held_per_super=3, n_prop=2, n_obs=16, member_id_dim=3, noise=0.02)
        n_in = Xtr.shape[1]
        k = meta["k_classes"]
        inh = idx["inh_idx"]
        srng = np.random.default_rng(seed * 13 + 1)
        keep = srng.permutation(len(Xtr))[:96]
        Xb, yb = Xtr[keep], ytr[keep]
        Xh, yh = Xte[inh], yte[inh]
        net = Gap4OnBridgeNet(n_in, 32, k, seed=seed, feedback="transport_ceiling", n_hidden_layers=2, pool_k=8,
                              settle_steps=25, credit_steps=20, lr=0.25, beta=1.0, p0=0.30, graded_credit=True,
                              apical_gain_pA=2000.0, **DRIVE)
        net.cfg.bdsp_w_max = 30.0
        net.cfg.bdsp_w_min = -30.0
        getters = {"INPUT": (Xb, Xh),
                   "H2 event pooled": (_ev_pooled(net, Xb), _ev_pooled(net, Xh)),
                   "H2 event UNPOOLED": (_ev_unpooled(net, Xb), _ev_unpooled(net, Xh)),
                   "H2 graded-V pooled": (_gr_pooled(net, Xb), _gr_pooled(net, Xh))}
        for r in ROWS:
            Rtr, Rho = getters[r]
            fl = fit_lin(Rtr, yb, k)
            fm = fit_mlp(Rtr, yb, k, seed=seed)
            agg[r]["tr_lin"].append(_acc(fl, Rtr, yb))
            agg[r]["ho_lin"].append(_acc(fl, Rho, yh))
            agg[r]["tr_mlp"].append(_acc(fm, Rtr, yb))
            agg[r]["ho_mlp"].append(_acc(fm, Rho, yh))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s) chance={1.0/k:.2f} n_ho={len(inh)}", flush=True)

    print("\n===== HELD-OUT representability (mean over 6 seeds) — TRAIN / HELDOUT =====", flush=True)
    for r in ROWS:
        a = agg[r]
        print(f"  {r:20s} lin {np.mean(a['tr_lin']):.3f}/{np.mean(a['ho_lin']):.3f}   "
              f"mlp {np.mean(a['tr_mlp']):.3f}/{np.mean(a['ho_mlp']):.3f}", flush=True)


if __name__ == "__main__":
    main()
