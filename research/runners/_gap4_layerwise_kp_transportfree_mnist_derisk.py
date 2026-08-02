"""gap#4 de-risk -- does TRANSPORT-FREE KP-learned feedback close the FA->backprop credit gap at DEPTH on MNIST?

Context. Our 2026-07-22 finding
(`research/findings/2026-07-22-gap4-credit-BEATS-reservoir-on-MNIST-cleanxor-was-the-wrong-instrument.md`) established
that on MNIST the biological local-credit rule BEATS a frozen reservoir at depth 2/4/6 (FA 0.93 vs reservoir 0.10 at
depth-4). But that FA arm is DIRECT feedback alignment (DFA): `sim.dendritic_mlp.DendriticMLP.train_step` projects the
OUTPUT error DIRECTLY to every hidden layer (`e @ self.B[li]`, line ~148), which BYPASSES the deep multiplicative chain
-- so it already matched the oracle at every depth (no gap for learned feedback to close; the finding says exactly this).

The STILL-OPEN question WF-Act-PC (arXiv 2607.13380, 2026) answers is about the LAYERWISE recursion
`e_l = (e_{l+1} @ FEEDBACK_l) * sigma'(a_l)` (Lillicrap 2016 feedback-alignment, NOT DFA). There, fixed-RANDOM per-layer
feedback DOES degrade with depth (the random matrices compound; alignment decays through the chain), leaving a gap to
backprop -- and a LEARNED feedback that converges to W^T (Kolen-Pollack, transport-free) should close it. We have never
tested transport-free KP-learned-feedback + sigma' against backprop on MNIST at depth in the LAYERWISE regime.

This runner (additive; NO `sim/` edit; the 2026-07-22 runner + `sim/` are untouched) builds a UNIFIED LAYERWISE credit
path whose SINGLE free variable is the FEEDBACK SOURCE, sigma' ON for all three:
  - fa        : Fb_l = fixed-RANDOM per-layer feedback (transport-free; Lillicrap-2016 layerwise FA).
  - kp        : Fb_l = KP-LEARNED per-layer feedback (Kolen-Pollack; transport-free; ported EXACTLY from
                `research/runners/_gnw_d1_spiking_bdsp_derisk.py::_kp_update`, dY = kp_lr*(post^T @ pre) - kp_decay*Y).
  - backprop  : Fb_l = true W_l^T (ORACLE ceiling; the ONLY mode that reads a forward weight, by design).
  - reservoir : hidden FROZEN at random init, readout trained (the credit-independent baseline from the finding).
Everything else (forward net, seeded init, Xavier scale, softmax-CE output delta, mean-over-batch + heavy-ball momentum
optimizer, lr) is IDENTICAL across fa/kp/backprop => the ONLY thing that differs is which matrix carries the descending
error. KEY read: does `kp` close the `fa`->`backprop` gap at depth-4?

Anti-cheats (printed): permuted-label kp -> chance; the kp feedback update NEVER reads a forward W (source-asserted);
the layerwise backprop arm reproduces the committed `DendriticMLP` oracle exactly (validity). CPU/numpy, seed-42 smoke.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU-only (rate numpy MLP), per task

import argparse
import hashlib
import inspect
import json

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
from sim.dendritic_mlp import DendriticMLP, _MOMENTUM  # noqa: E402

xp, _ = get_backend()


class LayerwiseNet(DendriticMLP):
    """DendriticMLP + a UNIFIED LAYERWISE credit path parameterized ONLY by the feedback source, + a reservoir mode.

    Additive: the base `DendriticMLP` (forward, init, `train_step`, DFA `self.B`) is untouched. The new per-transition
    feedback matrices `self.Fb[j]` (shape (sizes[j+1], sizes[j]) == W[j]^T shape) are used ONLY by `train_step_layerwise`
    and are random/seeded/TRANSPORT-FREE (never derived from W). fa and kp START from the SAME random draw, so the ONLY
    fa->kp difference is whether KP updates them.
    """

    def __init__(self, sizes, seed=0, kp_lr=0.2, kp_decay=1e-4):
        super().__init__(sizes, seed=seed)
        self.kp_lr = float(kp_lr)
        self.kp_decay = float(kp_decay)
        rng = np.random.default_rng(seed * 977 + 13)     # separate stream; TRANSPORT-FREE (random, not from W)
        nW = len(self.W)
        self.Fb = [None] * nW                            # Fb[j] carries error from layer j+1 down to layer j
        for j in range(1, nW):                           # transitions W[1]..W[nW-1] are descended (W[0] never is)
            lim = np.sqrt(6.0 / (self.sizes[j] + self.sizes[j + 1]))   # Xavier scale ~ |W[j]^T| for a fair FA start
            self.Fb[j] = xp.asarray(rng.uniform(-lim, lim, (self.sizes[j + 1], self.sizes[j])))

    def _kp_fb_update(self, j, pre, graderr, lr):
        """Kolen-Pollack update of the layerwise feedback Fb[j] (mode='kp' ONLY). TRANSPORT-FREE by construction: reads
        ONLY the local pre-activity + the descending error + Fb[j] itself -- it NEVER references any forward weight W.

        Ported EXACTLY from `_gnw_d1_spiking_bdsp_derisk.py::_kp_update`: dY = kp_lr*(post^T @ pre) - kp_decay*Y, applied
        as  Y += lr*(kp_lr*outer - kp_decay*Y),  outer = (post^T @ pre)/m. W[j]'s applied DESCENT increment here is
        upd[j] = -(acts[j]^T @ graderr_{j+1}); its transpose (the target increment for Fb[j] == W[j]^T) is
        -(graderr_{j+1}^T @ acts[j]) = (post^T @ pre) with post = -graderr_{j+1} (the descent-signed error) and
        pre = acts[j]. So Fb[j]^T tracks W[j] and (W[j]^T - Fb[j]) decays under the shared kp_decay (Akrout 2019 / KP,
        no weight copy). `lr` is the forward step (as in `_gnw`, the mirror rate is lr*kp_lr)."""
        pre = xp.asarray(pre)
        post = -xp.asarray(graderr)                      # descent-signed error, matching W[j]'s -(a^T @ graderr) sign
        m = max(1, pre.shape[0])
        outer = (post.T @ pre) / m                       # (sizes[j+1], sizes[j]) == Fb[j].shape; LOCAL, no W read
        self.Fb[j] = self.Fb[j] + lr * (self.kp_lr * outer - self.kp_decay * self.Fb[j])

    def train_step_layerwise(self, X, y, mode, lr):
        """One SGD step. mode in {reservoir, fa, kp, backprop}. Output readout trained IDENTICALLY in every mode; the
        ONLY thing that differs across fa/kp/backprop is the feedback matrix used in the layerwise recursion."""
        acts, e = self._debug_fwd_err(X, y)              # e = softmax - onehot = +gradient at the output
        nW = len(self.W)
        m = max(1, X.shape[0])
        upd = [None] * nW
        upd[-1] = -(acts[-1].T @ e)                      # output readout descent (ALL modes, identical)
        if mode == "reservoir":
            for j in range(nW - 1):
                upd[j] = xp.zeros_like(self.W[j])        # hidden FROZEN at random init
        elif mode in ("fa", "kp", "backprop"):
            d = e                                        # gradient-error, descends top -> bottom
            for li in range(nW - 2, -1, -1):
                a = acts[li + 1]                         # hidden layer li+1 activity
                if mode == "backprop":
                    Fb = self.W[li + 1].T                # TRUE transpose -- ORACLE (only mode that reads W)
                else:                                    # fa OR kp: use the stored layerwise feedback matrix
                    Fb = self.Fb[li + 1]
                d_below = (d @ Fb) * a * (1.0 - a)       # e_l = (e_{l+1} @ FEEDBACK_l) * sigma'(a_l)   [sigma' ON]
                if mode == "kp":                         # learn the feedback (transport-free) BEFORE d is overwritten
                    self._kp_fb_update(li + 1, pre=acts[li + 1], graderr=d, lr=lr)
                upd[li] = -(acts[li].T @ d_below)        # W[li] descent update from the descended error
                d = d_below
        else:
            raise ValueError("unknown mode %r" % mode)
        # Optimizer: mean-over-batch + heavy-ball momentum -- IDENTICAL to DendriticMLP.train_step (mode-agnostic).
        if self._vel is None:
            self._vel = [xp.zeros_like(w) for w in self.W]
        for j in range(nW):
            self._vel[j] = _MOMENTUM * self._vel[j] + upd[j] / m
            self.W[j] = self.W[j] + lr * self._vel[j]


def _load_mnist(n_train, n_test, seed):
    """Identical to the 2026-07-22 runner's loader (same standardization + seeded subsample)."""
    d = np.load("data/mnist.npz")
    rng = np.random.default_rng(seed)
    xtr = d["x_train"].reshape(-1, 784).astype(np.float64) / 255.0
    xte = d["x_test"].reshape(-1, 784).astype(np.float64) / 255.0
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd
    itr = rng.choice(len(xtr), n_train, replace=False)
    ite = rng.choice(len(xte), n_test, replace=False)
    return xtr[itr], d["y_train"][itr].astype(np.int64), xte[ite], d["y_test"][ite].astype(np.int64)


def _train_eval(mode, Xtr, ytr, Xte, yte, sizes, seed, epochs, batch, lr, kp_lr, kp_decay, permute=False):
    net = LayerwiseNet(sizes, seed=seed, kp_lr=kp_lr, kp_decay=kp_decay)
    rng = np.random.default_rng(seed * 131 + 7)          # identical training-order stream to the 2026-07-22 runner
    ytr_use = ytr
    if permute:
        ytr_use = rng.permutation(ytr)                   # permuted-label anti-cheat: destroy X<->y => must -> chance
    n = len(Xtr)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            net.train_step_layerwise(Xtr[idx], ytr_use[idx], mode, lr)
    return net.accuracy(Xte, yte), net.accuracy(Xtr, ytr_use)


def _oracle_matches_backprop(seed, sizes):
    """Validity anti-cheat: the layerwise `backprop` arm must reproduce the committed DendriticMLP oracle grads exactly
    (proves my recursion == the shipped `_true_grads`, so `backprop` is a faithful ceiling). One batch, at init."""
    net = LayerwiseNet(sizes, seed=seed)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((64, sizes[0]))
    y = rng.integers(0, sizes[-1], 64)
    g_true = net._true_grads(X, y)                        # committed DendriticMLP oracle (hand-derived backprop)
    # reproduce the layerwise-backprop hidden updates and compare to -g_true (descent):
    acts, e = net._debug_fwd_err(X, y)
    nW = len(net.W)
    upd = [None] * nW
    upd[-1] = -(acts[-1].T @ e)
    d = e
    for li in range(nW - 2, -1, -1):
        a = acts[li + 1]
        d = (d @ net.W[li + 1].T) * a * (1.0 - a)
        upd[li] = -(acts[li].T @ d)
    max_abs = 0.0
    for li in range(nW):
        max_abs = max(max_abs, float(to_host(xp.max(xp.abs(upd[li] - (-g_true[li]))))))
    return max_abs


def one_seed(seed, a):
    Xtr, ytr, Xte, yte = _load_mnist(a.n_train, a.n_test, seed)
    sizes = [784] + [a.hidden] * a.depth + [10]
    res = {}
    for mode in ("reservoir", "fa", "kp", "backprop"):
        te, tr = _train_eval(mode, Xtr, ytr, Xte, yte, sizes, seed, a.epochs, a.batch, a.lr, a.kp_lr, a.kp_decay)
        res[mode] = dict(test=round(float(te), 4), train=round(float(tr), 4))
    # anti-cheat: permuted-label kp -> chance
    kp_perm_te, _ = _train_eval("kp", Xtr, ytr, Xte, yte, sizes, seed, a.epochs, a.batch, a.lr, a.kp_lr, a.kp_decay,
                                permute=True)
    res["kp_permuted"] = dict(test=round(float(kp_perm_te), 4))
    FA, KP, BP, RES = res["fa"]["test"], res["kp"]["test"], res["backprop"]["test"], res["reservoir"]["test"]
    fa_bp_gap = round(BP - FA, 4)                         # how far fixed-random FA sits below the backprop ceiling
    kp_closes = KP >= FA + 0.01 and (BP - KP) <= 0.5 * max(1e-9, (BP - FA))   # KP beats FA AND covers >=50% of the gap
    print(f"  [seed {seed}] MNIST {sizes}: RESERVOIR={RES:.3f} FA={FA:.3f} KP={KP:.3f} BACKPROP={BP:.3f} "
          f"| FA->BP gap={fa_bp_gap:+.3f}  KP-closes-gap={kp_closes}  kp_permuted={res['kp_permuted']['test']:.3f}")
    return dict(seed=seed, sizes=sizes, res=res, fa_bp_gap=fa_bp_gap, kp_closes_gap=bool(kp_closes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4], help="run each of these depths")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--kp-lr", type=float, default=0.2, help="Kolen-Pollack feedback rate (mirror = lr*kp_lr)")
    ap.add_argument("--kp-decay", type=float, default=1e-4, help="Kolen-Pollack symmetric weight decay on Fb")
    ap.add_argument("--out", default="research/findings/raw/gap4/layerwise_kp_transportfree_mnist.json")
    a = ap.parse_args()
    _, backend = get_backend()
    print(f"[gap4-layerwise-kp] transport-free KP-learned feedback vs fixed-random FA vs backprop (LAYERWISE recursion "
          f"e_l=(e_l+1 @ Fb)*sigma') hidden={a.hidden} depths={a.depths} n_train={a.n_train} epochs={a.epochs} "
          f"seeds={a.seeds} backend={backend}")

    # --- source-level anti-cheat: the KP feedback update must never read a forward W ---
    kp_src = inspect.getsource(LayerwiseNet._kp_fb_update)
    assert "self.W" not in kp_src, "KP feedback update references self.W -- NOT transport-free!"
    print("  [anti-cheat] KP feedback update source contains NO 'self.W' -> transport-free: OK")

    all_out = {}
    for depth in a.depths:
        a.depth = depth
        # --- validity anti-cheat: layerwise backprop == committed DendriticMLP oracle ---
        mism = _oracle_matches_backprop(a.seeds[0], [784] + [a.hidden] * depth + [10])
        print(f"  [validity] depth-{depth}: layerwise-backprop vs committed _true_grads oracle max|diff|={mism:.2e} "
              f"({'MATCH' if mism < 1e-9 else 'MISMATCH'})")
        print(f"  --- depth-{depth} ---")
        per = [one_seed(s, a) for s in a.seeds]
        mfa = float(np.mean([p["res"]["fa"]["test"] for p in per]))
        mkp = float(np.mean([p["res"]["kp"]["test"] for p in per]))
        mbp = float(np.mean([p["res"]["backprop"]["test"] for p in per]))
        mres = float(np.mean([p["res"]["reservoir"]["test"] for p in per]))
        ncloses = sum(p["kp_closes_gap"] for p in per)
        print(f"  [depth-{depth}] means: RES={mres:.3f} FA={mfa:.3f} KP={mkp:.3f} BACKPROP={mbp:.3f} | "
              f"FA->BP gap={mbp - mfa:+.3f}  KP-closes-gap {ncloses}/{len(per)}")
        all_out[f"depth{depth}"] = dict(per=per, means=dict(reservoir=mres, fa=mfa, kp=mkp, backprop=mbp),
                                        kp_closes=ncloses)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, depths=a.depths, hidden=a.hidden, epochs=a.epochs, n_train=a.n_train,
                       kp_lr=a.kp_lr, kp_decay=a.kp_decay, results=all_out), f, indent=2)
    print(f"[gap4-layerwise-kp] wrote {a.out}")


if __name__ == "__main__":
    main()
