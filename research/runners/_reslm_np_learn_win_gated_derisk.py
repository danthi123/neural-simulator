"""NP-toward-recurrent-language de-risk (the confound-checked spec from the design+adversarial-verify workflow
wf_f69e1f86-fb5): does NODE PERTURBATION learn the reservoir INPUT weights W_in through GENUINE cross-timestep nonlinear
recurrent computation -- the R3 lever (learn W_in on a FIXED reservoir), on the language architecture (recurrent),
BPTT-free? The naive "NP learns W_in on a reservoir with zero-input fillers" is a STATIC 2-layer problem in a recurrent
costume (W_in enters at ONE step) -> uninterpretable. This uses an ORDER-GATED DELAYED next-class task that forces BOTH
axes (learned input rep AND recurrent nonlinear binding), with CEILING-FIRST controls that ARBITRATE (not assume) the credit.

TASK (order-gated delayed next-class prediction):
  pos 0      : GATE cue P   (role by position)
  pos 1..d   : input-bearing distractor fillers (disjoint distractor vocab)
  pos d+1    : CONTENT cue Q
  pos d+2    : blank read step -> read h here
  target = class(Q)          if class(P)%2 == 0
           (class(Q)+1)%G    if class(P)%2 == 1     (the EARLIER gate conditionally shifts the LATER content)
  - order-sensitive (P@0 vs Q@d+1; swap changes target) -> a BAG (order-invariant) is provably below on an order-probe.
  - nonlinear gate x content -> a LINEAR reservoir's linear readout can't represent the conditional shift.
  - the delay makes recurrent RETENTION load-bearing (gP survives d fillers THEN binds with Q).
  - learn-W_in headroom: read gP (parity of P's class) + class(Q) through SHARED class blocks despite the identity
    confound + HELD-OUT rare synonyms (the R3 regime `_reslm_generalize_rate_check` shows has headroom).

WHAT NP TRAINS: W_in ONLY (W_rec FROZEN, per R3). Readout re-fit by the SAME z-scored ridge each eval for EVERY arm ->
the single variable is the W_in credit rule. NP update = WEIGHT-perturbation of W_in (dim N*m, T-INDEPENDENT -> no
sequence-length variance blowup), antithetic-k, common-random-numbers (noise=0), running-mean baseline.

RUNG 0 (MANDATORY, cheap, oracle/frozen/linear/bag ONLY -- run BEFORE any NP; the CEILING-first discipline): the
instrument is interpretable iff ALL hold at the chosen dist (dev seeds):
  (a) oracle(tanh) - frozen(tanh) >= +0.10 held-out AND frozen <= 0.90       [learn-W_in headroom]
  (b) oracle(tanh) - oracle(linear) >= +0.10 at dist>0, GROWING with dist    [nonlinear recurrence load-bearing]
  (c) oracle(tanh,recurrent) - best-bag(oracle W_in) >= +0.10 AND order-probe bag ~ chance  [temporal binding, not bag]
  (d) recurrence-lesion (W_rec->0) collapses the oracle margin at dist>0 (z-scored readout)  [right-reason collapse]
  If any sub-gate fails -> STOP + re-tune the regime on dev seeds; do NOT run NP in a no-headroom regime (CEILING lesson).
Reuse-by-import: `build_codes` from `_reslm_generalize_rate_check`; the NP estimator shape from `_nodepert_deep_credit_derisk`.
NO `sim/` edit. numpy/CPU.

Run (rung-0 first): SIM_BACKEND=numpy python -m research.runners._reslm_np_learn_win_gated_derisk --rung0 --seeds 42 43 44
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._reslm_generalize_rate_check import build_codes


def _softmax_rows(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


# --------- task ---------------------------------------------------------------------------------------------
def _distractor_codes(seed, m, n_distract):
    rng = np.random.default_rng(seed * 613 + 41)
    D = (rng.random((n_distract, m)) < 0.15).astype(np.float64)      # sparse input-bearing distractors in the same m-space
    return D


def build_gated(seed, G, syn, dist, n_ex, n_distract, order_probe=False):
    """Return (train, eval) lists of (P, Q, fillers, target). Held-out: the CONTENT cue Q uses the rare synonym
    (syn-1) at eval only (novel identity dims -> class read from the SHARED block = the generalization test); the GATE
    cue P uses train synonyms. order_probe=True -> eval set is the SAME (P,Q) pairs presented in BOTH orders (P,Q) and
    (Q,P) with the order-correct target -> a bag scores ~chance (its two orders map to one representation)."""
    rng = np.random.default_rng(seed * 733 + 19)
    held = {c: c * syn + (syn - 1) for c in range(G)}
    def tok(c, held_ok):
        j = (syn - 1) if held_ok else int(rng.integers(0, syn - 1))
        return c * syn + j
    def target(P, Q):
        gP = (P // syn) % 2
        cq = Q // syn
        return (cq + 1) % G if gP == 1 else cq
    def one(heldQ):
        cP = int(rng.integers(0, G)); cQ = int(rng.integers(0, G))
        P = tok(cP, False); Q = tok(cQ, heldQ)
        fillers = [int(rng.integers(0, n_distract)) for _ in range(dist)]
        return P, Q, fillers, target(P, Q)
    train = [one(False) for _ in range(n_ex)]
    if order_probe:
        evl = []
        for _ in range(max(60, n_ex // 4)):
            P, Q, fillers, _ = one(True)
            evl.append((P, Q, fillers, target(P, Q)))            # order (P,Q)
            evl.append((Q, P, fillers, target(Q, P)))            # swapped order (Q,P) -> generally different target
        return train, evl
    evl = [one(True) for _ in range(max(60, n_ex // 4))]
    return train, evl


# --------- reservoir forward --------------------------------------------------------------------------------
def _seq_x(ex, dist, codes, dcodes, m):
    P, Q, fillers, _ = ex
    T = dist + 3
    xs = []
    for t in range(T):
        if t == 0:            xs.append(codes[P])
        elif t == dist + 1:   xs.append(codes[Q])
        elif t <= dist:       xs.append(dcodes[fillers[t - 1]])
        else:                 xs.append(np.zeros(m))              # blank read step
    return xs                                                    # list of T m-vectors


def _fwd_read(xs, W_in, W_rec, b, n, alpha, recur, lesion=False):
    """Run the reservoir over xs; return h_read (the last step). lesion=True zeroes the recurrence (W_rec->0)."""
    h = np.zeros(n)
    Wr = None if lesion else W_rec
    for t, x in enumerate(xs):
        pre = (0.0 if lesion else Wr @ h) + W_in @ x + b
        act = pre if recur == "linear" else np.tanh(pre)
        h = (1 - alpha) * h + alpha * act
    return h


def _reads(examples, dist, W_in, W_rec, b, n, alpha, recur, codes, dcodes, m, lesion=False):
    R = np.array([_fwd_read(_seq_x(ex, dist, codes, dcodes, m), W_in, W_rec, b, n, alpha, recur, lesion)
                  for ex in examples])
    Y = np.array([ex[3] for ex in examples])
    return R, Y


def _zscore_ridge_acc(Rtr, Ytr, Rev, Yev, G, lam=1.0, want_train=False):
    """z-score the read-states (per-dim, train stats), ridge-fit the G-way readout on train, held-out acc on eval."""
    mu = Rtr.mean(0, keepdims=True); sd = Rtr.std(0, keepdims=True) + 1e-6
    Xtr = np.concatenate([(Rtr - mu) / sd, np.ones((len(Rtr), 1))], 1)
    Xev = np.concatenate([(Rev - mu) / sd, np.ones((len(Rev), 1))], 1)
    Tt = np.zeros((len(Ytr), G)); Tt[np.arange(len(Ytr)), Ytr] = 1.0
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Tt)
    he = float((Xev @ W).argmax(1).__eq__(Yev).mean())
    if want_train:
        return he, float((Xtr @ W).argmax(1).__eq__(Ytr).mean())
    return he


# --------- W_in learners ------------------------------------------------------------------------------------
def _init(seed, n, m):
    rng = np.random.RandomState(seed * 3 + 17)
    W_rec = rng.normal(0, 1, (n, n)) / np.sqrt(n)
    W_rec *= 0.95 / max(np.max(np.abs(np.linalg.eigvals(W_rec))), 1e-6)
    W_in = rng.normal(0, 1, (n, m)) / np.sqrt(m)
    b = np.zeros(n)
    return W_rec, W_in, b


def train_win_bptt(train, dist, W_rec, W_in0, b, n, m, G, alpha, recur, codes, dcodes, epochs, lr, seed):
    """ORACLE: learn W_in (+ a throwaway softmax Wout) by full BPTT through the recurrence. The eval readout is re-fit
    by ridge afterwards (fair across arms)."""
    rng = np.random.RandomState(seed * 7 + 1)
    W_in = W_in0.copy(); Wout = np.zeros((G, n))
    Xs = [_seq_x(ex, dist, codes, dcodes, m) for ex in train]
    Ys = [ex[3] for ex in train]
    for _ep in range(epochs):
        order = rng.permutation(len(train))
        for si in order:
            xs = Xs[si]; y = Ys[si]; T = len(xs)
            hs = [np.zeros(n)]; acts = []
            for t in range(T):
                pre = W_rec @ hs[-1] + W_in @ xs[t] + b
                a = pre if recur == "linear" else np.tanh(pre)
                acts.append(a); hs.append((1 - alpha) * hs[-1] + alpha * a)
            read = hs[-1]
            z = np.clip(Wout @ read, -30, 30); z = z - z.max(); e = np.exp(z); p = e / e.sum()
            g = p.copy(); g[y] -= 1.0
            gW = np.outer(g, read); gn = np.linalg.norm(gW)
            Wout -= lr * (gW / gn * min(gn, 5.0))            # norm-clip the Wout gradient
            dh = Wout.T @ g
            dW_in = np.zeros_like(W_in)
            for t in range(T - 1, -1, -1):
                dpre = alpha * dh if recur == "linear" else alpha * (1 - acts[t] ** 2) * dh
                dW_in += np.outer(dpre, xs[t])
                dh = (1 - alpha) * dh + W_rec.T @ dpre
            dn = np.linalg.norm(dW_in)
            W_in -= lr * (dW_in / dn * min(dn, 5.0))         # norm-clip the W_in gradient (BPTT stability)
    return W_in


def train_win_np(train, dist, W_rec, W_in0, b, n, m, G, alpha, recur, codes, dcodes, epochs, lr, sigma, k, seed):
    """NODE PERTURBATION on W_in (weight-perturbation, T-independent dim N*m; antithetic-k; common-random-numbers via a
    deterministic forward noise=0; running-mean baseline). A throwaway softmax Wout supplies the scalar loss the global
    dL rides; the eval readout is re-fit by ridge afterwards (fair across arms)."""
    rng = np.random.RandomState(seed * 11 + 5)
    W_in = W_in0.copy(); Wout = np.zeros((G, n))
    Xs = [_seq_x(ex, dist, codes, dcodes, m) for ex in train]
    Ys = [ex[3] for ex in train]
    def _loss(Wi, xs, y):
        read = _fwd_read(xs, Wi, W_rec, b, n, alpha, recur)
        z = np.clip(Wout @ read, -30, 30); z = z - z.max(); e = np.exp(z); p = e / e.sum()
        return float(-np.log(max(p[y], 1e-12))), read, p
    for _ep in range(epochs):
        order = rng.permutation(len(train))
        for si in order:
            xs = Xs[si]; y = Ys[si]
            L0, read, p = _loss(W_in, xs, y)
            grad = np.zeros_like(W_in)
            for _r in range(k):
                xi = sigma * rng.standard_normal(W_in.shape)      # perturb the N*m WEIGHTS (T-independent)
                Lp = _loss(W_in + xi, xs, y)[0]
                Lm = _loss(W_in - xi, xs, y)[0]                   # antithetic
                grad += (0.5 * (Lp - Lm) / (sigma * sigma)) * xi
            W_in -= lr * grad / k
            # throwaway Wout: clean delta on the clean read (keeps the loss signal meaningful as W_in moves)
            g = p.copy(); g[y] -= 1.0
            Wout -= 0.05 * np.outer(g, read)
    return W_in


def _bag_oracle_acc(train, evl, G, codes, dcodes, m, dist, seed, epochs, lr):
    """BAG control: order-INVARIANT. The 'read' = a learned linear map of the SUM of the sequence's input codes (oracle
    W_in-equivalent: fit the readout directly on the summed codes = the best any order-invariant reader can do). If this
    reaches the recurrent oracle, the task is bag-fakeable; on the order-probe it must be ~chance."""
    def bag(ex):
        xs = _seq_x(ex, dist, codes, dcodes, m); return np.sum(xs, 0)
    Rtr = np.array([bag(e) for e in train]); Ytr = np.array([e[3] for e in train])
    Rev = np.array([bag(e) for e in evl]);   Yev = np.array([e[3] for e in evl])
    return _zscore_ridge_acc(Rtr, Ytr, Rev, Yev, G)


# --------- arms + rung-0 ------------------------------------------------------------------------------------
def _arm_acc(train, evl, dist, recur, learner, G, n, m, alpha, codes, dcodes, seed, epochs, lr, lesion=False,
             sigma=0.0, k=1, want_train=False):
    W_rec, W_in0, b = _init(seed, n, m)
    if learner == "frozen":
        W_in = W_in0
    elif learner == "oracle":
        W_in = train_win_bptt(train, dist, W_rec, W_in0, b, n, m, G, alpha, recur, codes, dcodes, epochs, lr, seed)
    elif learner == "np":
        W_in = train_win_np(train, dist, W_rec, W_in0, b, n, m, G, alpha, recur, codes, dcodes, epochs, lr, sigma, k, seed)
    else:
        raise ValueError(learner)
    Rtr, Ytr = _reads(train, dist, W_in, W_rec, b, n, alpha, recur, codes, dcodes, m, lesion)
    Rev, Yev = _reads(evl, dist, W_in, W_rec, b, n, alpha, recur, codes, dcodes, m, lesion)
    return _zscore_ridge_acc(Rtr, Ytr, Rev, Yev, G, want_train=want_train)


def rung0(seed, G, syn, sf, idn, id_pool, n, dists, epochs, lr, n_ex, n_distract):
    codes, V, m = build_codes(seed, G, syn, sf, idn, id_pool=id_pool)
    out = {"seed": seed, "G": G, "m": int(m), "chance": round(1.0 / G, 3), "by_dist": {}}
    for dist in dists:
        train, evl = build_gated(seed, G, syn, dist, n_ex, n_distract)
        _, oprobe = build_gated(seed, G, syn, dist, n_ex, n_distract, order_probe=True)
        dcodes = _distractor_codes(seed, m, n_distract)
        d = {}
        d["oracle_tanh"], d["oracle_tanh_train"] = _arm_acc(train, evl, dist, "tanh", "oracle", G, n, m, 0.3, codes, dcodes, seed, epochs, lr, want_train=True)
        d["frozen_tanh"] = _arm_acc(train, evl, dist, "tanh", "frozen", G, n, m, 0.3, codes, dcodes, seed, epochs, lr)
        d["oracle_linear"] = _arm_acc(train, evl, dist, "linear", "oracle", G, n, m, 0.3, codes, dcodes, seed, epochs, lr)
        d["oracle_tanh_lesion"] = _arm_acc(train, evl, dist, "tanh", "oracle", G, n, m, 0.3, codes, dcodes, seed, epochs, lr, lesion=True)
        d["bag_oracle"] = _bag_oracle_acc(train, evl, G, codes, dcodes, m, dist, seed, epochs, lr)
        d["bag_orderprobe"] = _bag_oracle_acc(train, oprobe, G, codes, dcodes, m, dist, seed, epochs, lr)
        out["by_dist"][dist] = {kk: round(vv, 3) for kk, vv in d.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--G", type=int, default=6); ap.add_argument("--syn", type=int, default=5)
    ap.add_argument("--sf", type=int, default=3); ap.add_argument("--idn", type=int, default=20)
    ap.add_argument("--id-pool", type=int, default=0); ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--dists", type=int, nargs="+", default=[0, 3, 8])
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-ex", type=int, default=400); ap.add_argument("--n-distract", type=int, default=12)
    ap.add_argument("--rung0", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [rung0(s, a.G, a.syn, a.sf, a.idn, a.id_pool, a.n, a.dists, a.epochs, a.lr, a.n_ex, a.n_distract) for s in a.seeds]
    # aggregate by dist
    print(f"=== RUNG-0 ceiling-first (order-gated delayed next-class; seeds {a.seeds}, chance {1.0/a.G:.3f}) ===", flush=True)
    for dist in a.dists:
        agg = {kk: float(np.mean([r["by_dist"][dist][kk] for r in res]))
               for kk in res[0]["by_dist"][dist]}
        head = agg["oracle_tanh"] - agg["frozen_tanh"]
        nonlin = agg["oracle_tanh"] - agg["oracle_linear"]
        vsbag = agg["oracle_tanh"] - agg["bag_oracle"]
        print(f"dist={dist}: oracle_tanh={agg['oracle_tanh']:.3f}(tr{agg['oracle_tanh_train']:.2f}) frozen={agg['frozen_tanh']:.3f} "
              f"oracle_lin={agg['oracle_linear']:.3f} lesion={agg['oracle_tanh_lesion']:.3f} "
              f"bag={agg['bag_oracle']:.3f} bag_op={agg['bag_orderprobe']:.3f} "
              f"| headroom={head:+.3f} nonlin_adv={nonlin:+.3f} vs_bag={vsbag:+.3f}", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
