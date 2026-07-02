"""EMERGE-2 DE-RISK (v2): does the confirmed burst mechanism learn deep structure SELF-SUPERVISED (no labels)?

EMERGE-1b confirmed burst-multiplexed dendritic credit assignment develops deep structure under SUPERVISION. The goal
(an emergent brain that learns from EXPERIENCE) needs the UNSUPERVISED / self-supervised regime the brain lives in: it
predicts the lawful structure of its own input, and the prediction error drives plasticity (Urbanczik-Senn unifies
sup/unsup -- only what drives the soma changes). This tests the burst rule there.

v2 fix (v1 was mis-designed -- its "entangled" latents were linearly readable straight off the input [raw probe 0.99],
so recovering them never required depth; the raw-probe control caught it): use a task where PREDICTION ITSELF requires
depth, and measure PREDICTION QUALITY (clean), not linear disentanglement (fraught).

THE SELF-SUPERVISED TASK: an observation x = [a, b] where the second part b is a LAWFUL depth-2 function of the first
part a -- b is a vector of K_B distinct threshold-of-XORs of a's bits (XOR needs one hidden layer; threshold-over-XORs
needs a second). Training = predict b from a (masked/self-supervised: b is part of the observed world, NOT an external
label). To predict b the net must DEVELOP the depth-2 structure. Held-out = unseen a-patterns. A shallow / linear
predictor provably can't represent threshold-of-XORs, so held-out b-prediction measures whether the deep net DEVELOPED
the structure from self-supervised experience.

ARMS (identical data/splits/seeds; the burst credit = the SELF-GENERATED prediction error, NO labels):
deep_burst_linearized (TEST) · point_shallow (single hidden, must fail on depth-2) · linear_baseline (ridge a->b, the
raw floor -- depth-2 is not linear) · oracle_bp (fenced backprop, task-sanity) · apical_lesion (Y=0 -> frozen hidden ->
can't compute the depth-2 function) · wrong_sign (anti-learn) · no_teaching_null (prediction error zeroed -> no
learning). GO = deep_burst held-out b-accuracy >= 0.75 AND > shallow + 0.07 AND > linear + 0.10 AND > apical_lesion +
0.07; apical-lesion collapses; wrong-sign/null at the shallow/chance floor; oracle >= 0.85 (task-sane); no weight
transport; multi-seed (42/43/44). Reuse-by-import; NO `sim/` edit; CPU (small net -> numpy is the right tool; GPU
launch-overhead would be slower at this width). Run: python -m research.runners._emerge2_selfsup_burst_emergence_derisk
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge2_selfsup_burst.json"
A_BITS = 10          # the visible part a
K_B = 6              # the lawful part b = K_B depth-2 functions of a
_MOMENTUM = 0.9


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def make_task(seed):
    """x=[a,b]; b = K_B LEARNABLE depth-2 boolean functions of a's CLEAN disjoint-consecutive-pair XORs (the level-1
    latents -- EMERGE-1's learnable structure: each input bit in exactly one pair, so backprop reaches ~0.95). Predict
    b from a (self-supervised: b is the lawful structure of the observation, no external label). Held-out = unseen
    a-patterns. Returns (a_tr, b_tr), (a_te, b_te), a in +/-1, b in {0,1}."""
    rng = np.random.default_rng(seed)
    n = 1 << A_BITS
    a = ((np.arange(n)[:, None] >> np.arange(A_BITS)[None, :]) & 1).astype(np.float64)   # (n, A_BITS) {0,1}
    px = np.logical_xor(a[:, 0::2].astype(bool), a[:, 1::2].astype(bool)).astype(float)  # clean disjoint-pair XORs (n, 5)
    npair = px.shape[1]; pb = px > 0.5
    b = np.zeros((n, K_B))
    b[:, 0] = (px.sum(1) >= (npair + 1) // 2)                                            # majority (EMERGE-1's, oracle~0.95)
    b[:, 1] = (px.sum(1) >= 2)                                                           # lower threshold
    b[:, 2] = (px.sum(1) >= npair - 1)                                                   # high threshold
    b[:, 3] = np.logical_or(np.logical_and(pb[:, 0], pb[:, 1]), np.logical_and(pb[:, 2], pb[:, 3]))  # (AND) OR (AND)
    b[:, 4] = (px[:, [0, 2, 4]].sum(1) >= 2)                                             # threshold of a subset
    b[:, 5] = np.logical_xor(pb[:, 0], np.logical_and(pb[:, 1], pb[:, 2]))               # XOR of a bit and an AND
    b = b.astype(np.float64)
    X = a * 2.0 - 1.0                                              # +/-1
    idx = rng.permutation(n); cut = int(0.65 * n)
    tr, te = idx[:cut], idx[cut:]
    return (X[tr], b[tr]), (X[te], b[te])


class BurstpropRegressor:
    """Self-supervised burst regressor (sigmoid hiddens, LINEAR output predicting b). Faithful Burstprop credit
    (multiplexed event/burst, layer-wise burst-coded error via fixed-random Y, recurrent linearization, BDSP; no weight
    transport). Teaching signal = the SELF-GENERATED prediction error (pred - b) -- no labels."""

    def __init__(self, sizes, seed=0, beta=1.0, p0=0.5, ema=0.9):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes); self.beta = float(beta); self.p0 = float(p0); self.ema = float(ema)
        self.W = []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
        yrng = np.random.default_rng(seed + 9973)
        self.Y = [yrng.normal(0, 1.0, (sizes[k + 2], sizes[k + 1])) for k in range(len(sizes) - 2)]
        self.pbar = [np.full(sizes[k + 1], p0) for k in range(len(sizes) - 2)]
        self._vel = None

    def _forward(self, X):
        acts = [np.asarray(X, float)]
        for li in range(len(self.W) - 1):
            acts.append(_sig(acts[-1] @ self.W[li]))
        return acts, _sig(acts[-1] @ self.W[-1])            # SIGMOID output (bounded 0-1; per-bit BCE head)

    def predict(self, X):
        return self._forward(X)[1]

    def bit_acc(self, X, T):
        return float(np.mean((self.predict(X) >= 0.5) == (T >= 0.5)))

    def _true_grads(self, X, T):                            # backprop for the sigmoid-output BCE net (oracle only)
        acts, out = self._forward(X); d = (out - T)         # BCE+sigmoid gradient at the output (NOT /m; matches EMERGE-1b)
        nW = len(self.W); grads = [None] * nW; grads[-1] = acts[-1].T @ d
        for li in range(nW - 2, -1, -1):
            aa = acts[li + 1]; d = (d @ self.W[li + 1].T) * aa * (1.0 - aa); grads[li] = acts[li].T @ d
        return grads

    def train_step(self, X, T, mode, lr):
        acts, out = self._forward(X)
        delta_out = (out - T)                               # BCE+sigmoid error (NOT /m -- the /m is in the optimizer)
        nW = len(self.W); nhid = nW - 1; upd = [None] * nW
        if mode == "oracle":
            upd = [-g for g in self._true_grads(X, T)]
        else:
            upd[-1] = -(acts[-1].T @ delta_out)
            linearize = (mode == "burst_linearized")
            b = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out
            for k in range(nhid - 1, -1, -1):
                post = acts[k + 1]
                Yk = np.zeros_like(self.Y[k]) if mode == "apical_lesion" else self.Y[k]
                v_api = b @ Yk
                if linearize:
                    v_api = v_api * (post * (1.0 - post))
                p = _sig(self.beta * v_api)
                self.pbar[k] = self.ema * self.pbar[k] + (1.0 - self.ema) * p.mean(0)
                dev = post * (p - self.pbar[k])
                g = acts[k].T @ dev
                upd[k] = -g if mode == "wrong_sign" else g
                b = dev
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [np.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m     # /m in the optimizer (EMERGE-1b convention)
            self.W[li] = self.W[li] + lr * self._vel[li]


def _train(net, X, T, mode, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], T[b], mode=mode, lr=lr)


def _linear_baseline(Xtr, Btr, Xte, Bte):
    """Ridge a->b, held-out bit-accuracy: the raw floor. depth-2 (threshold-of-XORs) is NOT linear -> should be ~chance."""
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1); Ae = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    lam = 1e-2 * np.eye(A.shape[1]); lam[-1, -1] = 0.0
    W = np.linalg.solve(A.T @ A + lam, A.T @ Btr)
    return float(np.mean(((Ae @ W) >= 0.5) == (Bte >= 0.5)))


def run(seed, epochs, lr, batch, hidden):
    (Xtr, Btr), (Xte, Bte) = make_task(seed)
    din, dout = Xtr.shape[1], Btr.shape[1]
    deep = [din, hidden, hidden, dout]; shal = [din, hidden, dout]
    res = {"linear_baseline": _linear_baseline(Xtr, Btr, Xte, Bte),
           "chance": float(np.mean(np.maximum(Bte.mean(0), 1 - Bte.mean(0))))}
    for name, sizes, mode in [("deep_burst_linearized", deep, "burst_linearized"),
                              ("point_shallow", shal, "burst_linearized"),
                              ("oracle_bp", deep, "oracle"), ("apical_lesion", deep, "apical_lesion"),
                              ("wrong_sign", deep, "wrong_sign"), ("no_teaching_null", deep, "no_teaching_null")]:
        net = BurstpropRegressor(sizes, seed=seed)
        wt_ok = all(not any(np.array_equal(Yk, w) or np.array_equal(Yk, w.T) for w in net.W) for Yk in net.Y)
        _train(net, Xtr, Btr, mode, epochs, lr, batch, seed)
        res[name] = {"heldout_bitacc": net.bit_acc(Xte, Bte), "train_bitacc": net.bit_acc(Xtr, Btr),
                     "no_weight_transport": bool(wt_ok)}
    return {"seed": seed, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s, a.epochs, a.lr, a.batch, a.hidden); per.append(r)
            print(f"  [seed {s}] deep_burst held {r['deep_burst_linearized']['heldout_bitacc']:.3f} "
                  f"(train {r['deep_burst_linearized']['train_bitacc']:.3f}) | shallow "
                  f"{r['point_shallow']['heldout_bitacc']:.3f} | linear {r['linear_baseline']:.3f} | oracle "
                  f"{r['oracle_bp']['heldout_bitacc']:.3f} | lesion {r['apical_lesion']['heldout_bitacc']:.3f} | "
                  f"wrong {r['wrong_sign']['heldout_bitacc']:.3f} | null {r['no_teaching_null']['heldout_bitacc']:.3f} "
                  f"| chance {r['chance']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mh(k):
            return float(np.mean([p[k]["heldout_bitacc"] for p in per]))
        deep, shal, orac = mh("deep_burst_linearized"), mh("point_shallow"), mh("oracle_bp")
        les, wrong, null = mh("apical_lesion"), mh("wrong_sign"), mh("no_teaching_null")
        lin = float(np.mean([p["linear_baseline"] for p in per])); ch = float(np.mean([p["chance"] for p in per]))
        wt = all(p["deep_burst_linearized"]["no_weight_transport"] for p in per)
        task_ok = orac >= 0.85                                       # backprop CAN predict b (depth-2) self-supervised
        develops = (deep >= 0.75) and (deep > shal + 0.07) and (deep > lin + 0.10) and (deep > les + 0.07)
        lesion_collapses = les <= max(shal, lin, ch) + 0.05
        wrong_anti = wrong <= max(shal, ch) + 0.05
        null_flat = null <= max(shal, ch) + 0.05
        go = bool(task_ok and develops and lesion_collapses and wrong_anti and null_flat and wt)
        if not task_ok:
            verdict = (f"INCONCLUSIVE -- oracle predicts b only {orac:.3f} self-supervised; task/config needs tuning "
                       f"(epochs/lr/hidden/K_B) before reading the burst arms. (linear floor {lin:.3f}, chance {ch:.3f}.)")
        elif go:
            verdict = (f"GO -- the confirmed burst mechanism learns deep structure SELF-SUPERVISED (no labels): trained "
                       f"only to predict the lawful part b (depth-2 threshold-of-XORs) of the observation from a, deep "
                       f"burstprop reaches held-out b-accuracy {deep:.3f} >> shallow {shal:.3f} + linear {lin:.3f} + "
                       f"apical-lesion {les:.3f} + chance {ch:.3f}; apical-lesion collapses, wrong-sign/no-teaching-null "
                       f"at floor, no weight transport; oracle {orac:.3f}. Multi-seed. ⇒ the emergent-cortex primitive "
                       f"holds UNSUPERVISED -- the substrate DEVELOPS deep structure from experience (predict-your-input) "
                       f"via biological burst credit assignment. Carry it to the spiking substrate. NO sim/ edit.")
        else:
            miss = []
            if not develops: miss.append(f"deep didn't clearly beat the floors (deep {deep:.3f} vs shallow {shal:.3f}/"
                                        f"linear {lin:.3f}/lesion {les:.3f})")
            if not lesion_collapses: miss.append("apical-lesion didn't collapse")
            if not (wrong_anti and null_flat): miss.append("wrong-sign/null not at floor")
            if not wt: miss.append("weight-transport check failed")
            verdict = ("BOUNDARY (the next mechanism to find, not a stop) -- " + "; ".join(miss) + f" (oracle {orac:.3f}"
                       f", linear {lin:.3f}). Per the master directive: iterate -- wider/ensemble (sharper burst "
                       f"estimate, as in EMERGE-1b), a predictive-coding target (`sim/predictive_coding.py`), or the "
                       f"Sacramento-Senn microcircuit's local error.")
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge2_selfsup_burst_v2", "GO": go, "verdict": verdict,
               "task": f"self-supervised: predict the lawful part b ({K_B} depth-2 threshold-of-XORs of {A_BITS} bits) "
                       f"from a; emergence = held-out b-prediction accuracy (deep vs shallow/linear); NO external labels",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "v2: measures PREDICTION quality (clean), not linear disentanglement (v1's flaw, caught by "
                              "the raw-probe control). Teaching signal = self-generated prediction error, no labels. "
                              "BOUNDARY = the next mechanism (scale/PC/microcircuit), not a stop. Oracle = fenced "
                              "backprop task-sanity, NOT a shipped biological mode. Small net -> numpy is the right tool."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge2] VERDICT: {verdict}", flush=True)
    print(f"[emerge2] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
