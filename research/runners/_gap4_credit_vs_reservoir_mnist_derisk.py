"""gap#4 de-risk — does deep credit beat a RESERVOIR on a PROPER deep task (MNIST), and does LEARNED feedback beat
fixed-random feedback-alignment?

Context (2026-07-22 deep-research, `2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS...`): the standing
gap#4 "clean negative" (credit-training the hidden LOSES to a fixed-random reservoir readout, 0.55 vs 0.77) was measured
ONLY on cleanxor. cleanxor has a ZERO linear discriminant by construction, so `e = onehot - softmax` (which sums to 0)
gives a rank-1 update in a zero-information subspace -> credit CANNOT help there, and the learned-feedback (KP) fix has
nothing to align to. The deep-read flagged cleanxor as the WRONG INSTRUMENT and named two untested fixes: (1) learned
feedback weights (PAL/weight-mirror/KP), (2) a learned interneuron self-predicting microcircuit. This de-risk isolates
the FIRST claim on a PROPER deep compositional task with an INFORMATIVE gradient (MNIST: linear ~92%, deep ~98%; nonzero
class-mean differences -> the credit is NOT rank-1-zero).

ARMS (same seeded deep net 784->H->H->10, same optimizer, multi-seed):
- RESERVOIR : hidden layers FROZEN at random init; train ONLY the output readout (the credit-INDEPENDENT baseline).
- FA        : fixed-random feedback matrices B (never learned) = the project's current `local_correct` DFA method.
- KP        : LEARNED feedback (Kolen-Pollack symmetric update: B tracks W^T with no weight transport) = untested fix #1.
- ORACLE    : hand-derived backprop (aligned feedback ceiling; measurement only, NOT a shipped local rule).

READ: if FA > RESERVOIR on MNIST, the cleanxor negative was a TASK ARTIFACT -> credit DOES build accuracy on a proper
task (a real correction). If KP > FA, learned feedback closes the depth-degradation gap. All no-weight-transport except
oracle (fenced as measurement). CPU/numpy, coexists with GPU training.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json

import numpy as np

from sim.backend import get_backend, to_host  # noqa: E402
from sim.dendritic_mlp import DendriticMLP, _MOMENTUM  # noqa: E402

xp, _ = get_backend()


class Net(DendriticMLP):
    """DendriticMLP + a RESERVOIR mode (freeze hidden) + a KP learned-feedback mode (B tracks W^T, no weight transport)."""

    def train_step_ext(self, X, y, mode, lr, lr_fb=0.05, fb_decay=1e-4):
        if mode in ("oracle", "local_correct", "permuted", "local_wrongsign", "global_scalar"):
            return self.train_step(X, y, mode, lr)
        acts, e = self._debug_fwd_err(X, y)
        nW = len(self.W)
        upd = [None] * nW
        upd[-1] = -(acts[-1].T @ e)                     # output readout (trained in ALL modes)
        if mode == "reservoir":
            for li in range(nW - 1):
                upd[li] = xp.zeros_like(self.W[li])     # hidden FROZEN at random init
        elif mode == "kp":
            # LEARNED feedback: Kolen-Pollack symmetric update drives B[li] -> (effective forward path)^T with NO
            # weight transport (B is updated by its OWN local product, not copied from W). For the DFA layout B[li] is
            # (n_out, sizes[li+1]); the symmetric partner of the output-path update a_{li+1}^T e is e^T a_{li+1}.
            for li in range(nW - 1):
                a_prev, a_l = acts[li], acts[li + 1]
                ap = e @ self.B[li]
                upd[li] = -(a_prev.T @ (ap * a_l * (1.0 - a_l)))    # same FA hidden update, but with the LEARNED B
                dB = -(e.T @ a_l) / max(1, X.shape[0]) - fb_decay * self.B[li]   # KP symmetric + weight decay
                self.B[li] = self.B[li] + lr_fb * dB
        else:
            raise ValueError("unknown mode %r" % mode)
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
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd
    itr = rng.choice(len(xtr), n_train, replace=False)
    ite = rng.choice(len(xte), n_test, replace=False)
    return xtr[itr], d["y_train"][itr].astype(np.int64), xte[ite], d["y_test"][ite].astype(np.int64)


def _train_eval(mode, Xtr, ytr, Xte, yte, sizes, seed, epochs, batch, lr):
    net = Net(sizes, seed=seed)
    rng = np.random.default_rng(seed * 131 + 7)
    n = len(Xtr)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            net.train_step_ext(Xtr[idx], ytr[idx], mode, lr)
    return net.accuracy(Xte, yte), net.accuracy(Xtr, ytr)


def one_seed(seed, a):
    Xtr, ytr, Xte, yte = _load_mnist(a.n_train, a.n_test, seed)
    sizes = [784] + [a.hidden] * a.depth + [10]
    res = {}
    for mode in ("reservoir", "local_correct", "kp", "oracle"):
        te, tr = _train_eval(mode, Xtr, ytr, Xte, yte, sizes, seed, a.epochs, a.batch, a.lr)
        res[mode] = dict(test=round(float(te), 4), train=round(float(tr), 4))
    FA, RES, KP, OR = res["local_correct"], res["reservoir"], res["kp"], res["oracle"]
    fa_beats_res = FA["test"] > RES["test"] + 0.01
    kp_beats_fa = KP["test"] > FA["test"] + 0.01
    print(f"  [seed {seed}] MNIST {sizes}: RESERVOIR={RES['test']:.3f} FA={FA['test']:.3f} KP={KP['test']:.3f} "
          f"ORACLE={OR['test']:.3f} | FA>RES:{fa_beats_res} KP>FA:{kp_beats_fa}")
    return dict(seed=seed, sizes=sizes, res=res, fa_beats_res=fa_beats_res, kp_beats_fa=kp_beats_fa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2, help="hidden layers (>=2 = deep; FA degrades with depth)")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--out", default="research/findings/raw/gap4/credit_vs_reservoir_mnist.json")
    a = ap.parse_args()
    _, backend = get_backend()
    print(f"[gap4-mnist] deep credit vs RESERVOIR vs learned-feedback on MNIST (hidden={a.hidden} depth={a.depth} "
          f"n_train={a.n_train} epochs={a.epochs}) seeds={a.seeds} backend={backend}")
    per = [one_seed(s, a) for s in a.seeds]
    nfa = sum(p["fa_beats_res"] for p in per)
    nkp = sum(p["kp_beats_fa"] for p in per)
    mres = float(np.mean([p["res"]["reservoir"]["test"] for p in per]))
    mfa = float(np.mean([p["res"]["local_correct"]["test"] for p in per]))
    mkp = float(np.mean([p["res"]["kp"]["test"] for p in per]))
    mor = float(np.mean([p["res"]["oracle"]["test"] for p in per]))
    print(f"[gap4-mnist] VERDICT: FA>RESERVOIR {nfa}/{len(per)}, KP>FA {nkp}/{len(per)} | means: RES={mres:.3f} "
          f"FA={mfa:.3f} KP={mkp:.3f} ORACLE={mor:.3f}")
    print("  => " + ("credit DOES beat reservoir on a proper task (cleanxor negative was a task artifact); "
                     if nfa == len(per) else "credit does NOT reliably beat reservoir even on MNIST; ")
          + ("learned feedback (KP) closes the FA->oracle gap." if nkp >= (len(per) + 1) // 2 else "learned feedback (KP) does not clearly beat FA here."))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seeds=a.seeds, per=per), f, indent=2)


if __name__ == "__main__":
    main()
