"""The LANGUAGE escalation of the SSM-extract GO (`2026-07-13-SSM-fixed-...6seed-GO.md`): does a FIXED structured
multi-timescale (SSM/HiPPO-extract) reservoir improve DEEP-CONTEXT next-token CE on REAL TEXT, where the random reservoir
fades (the reslm's fading-memory ceiling that bounds the whole generation ladder)? A rate reservoir over the token stream
in blocks (state reset per block so within-block position = context depth); a LOCAL next-token read-out (the reslm's
delta-rule `train_readout`); CE bucketed by context depth (shallow ctx≤4 vs DEEP ctx≥17). ARMS: random-ESN vs
multi-timescale-diagonal vs bigram. GATE: the multi-timescale DEEP-context CE beats the random reservoir's (and both beat
bigram), i.e. the structured recurrence exposes deep-context language structure a random reservoir cannot. Reuse-by-import
`Vocab`/`train_readout`/`eval_ce`/`fit_bigram`/`bigram_ce` from `_emerge_reservoir_lm_derisk` + `load_sentences`; NO `sim/` edit.

Run: SIM_BACKEND=numpy python -m research.runners._ssm_reservoir_lm_derisk --seed 42 --n-sent 4000
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import math
import time

import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram, bigram_ce
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences


def _softmax_rows(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def _fit_ridge_readout(Xn, Y, V, lam=1.0):
    """Closed-form ridge read-out on FLAT pre-aligned (state -> next-token) rows: ONE solve (the proven-fast pattern
    from the committed memory-task de-risk). Z=[X,1]; Wd [n_feat,V] = solve(Z^T Z + lam I, Z^T onehot(Y)). The
    Z^T @ onehot(Y) is done by SCATTER (np.add.at) so the [n_samples, V] one-hot is NEVER materialized (scales to
    V=2000+ at large n). A LINEAR read-out fit by least-squares to the next-token one-hot -- NO backprop, NO recurrent credit."""
    n, d = Xn.shape
    Z = np.concatenate([Xn, np.ones((n, 1))], 1)
    ZtOH = np.zeros((V, d + 1)); np.add.at(ZtOH, Y, Z)            # column c = sum of Z[i] where Y[i]==c (no giant one-hot)
    Wd = np.linalg.solve(Z.T @ Z + lam * np.eye(d + 1), ZtOH.T)
    return Wd                                                     # [d+1, V]


def _fit_temperature(Z, Wd, Y, rng, sub=20000):
    """Fit a scalar temperature T minimizing mean CE of softmax((Z@Wd)/T) vs Y (1-D golden-section), on a random
    SUBSAMPLE (the [sub, V] logits fit in memory for V=2000+). Calibrates the ridge logits into a proper next-token
    DISTRIBUTION so the CE is comparable to the reslm ladder + the bigram baseline."""
    idx = rng.choice(len(Y), size=min(sub, len(Y)), replace=False)
    logits = Z[idx] @ Wd; Ys = Y[idx]
    def ce(T):
        P = _softmax_rows(logits / T); return float(-np.log(P[np.arange(len(Ys)), Ys] + 1e-12).mean())
    lo, hi = 0.02, 5.0; gr = (np.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo); d = lo + gr * (hi - lo); fc, fd = ce(c), ce(d)
    for _ in range(28):
        if fc < fd: hi, d, fd = d, c, fc; c = hi - gr * (hi - lo); fc = ce(c)
        else: lo, c, fc = c, d, fd; d = lo + gr * (hi - lo); fd = ce(d)
    return 0.5 * (lo + hi)

_N = 300


def _win(rng, n_in):
    return (rng.standard_normal((_N, n_in)) * (1.0 / np.sqrt(n_in))).astype(np.float64)


def _build_A(kind, rng):
    if kind in ("random", "hetero_esn"):
        W = rng.standard_normal((_N, _N)); ev = np.max(np.abs(np.linalg.eigvals(W)))
        W = (0.95 / ev) * W                                      # spectral radius 0.95 (echo-state mixing)
        if kind == "random":
            return W
        alpha = 1.0 / np.exp(np.linspace(np.log(1.5), np.log(400.0), _N))   # per-unit leak: fast (α≈0.67)→slow (α≈0.0025)
        return (W, alpha)                                        # leaky-ESN: mixing recurrence + heterogeneous time constants
    tau = np.exp(np.linspace(np.log(1.5), np.log(400.0), _N))     # multi-timescale diagonal (SSM extract); "mtbounded" adds tanh
    return np.exp(-1.0 / tau)


def _block_states(kind, A, W_in, emb, ids, block):
    """State reset per block; within-block position = context depth. Returns (states[T,n], targets[T], pos[T]).
    BATCHED over blocks: blocks are independent (reset per block), so ALL blocks step in LOCKSTEP as a [n_blocks, _N]
    matrix -- `block` vectorized matmuls TOTAL, independent of corpus size (the parallel-across-blocks speedup that
    makes the validated-scale run tractable; the per-token Python loop was the toy-scale bottleneck)."""
    ids = np.asarray(ids)
    B1 = block + 1
    n_full = len(ids) // B1
    if n_full == 0:
        return np.zeros((0, _N)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    chunks = ids[:n_full * B1].reshape(n_full, B1)                 # [n_blocks, block+1] non-overlapping
    Wmat, alpha = (A if kind == "hetero_esn" else (None, None))
    X = np.zeros((n_full, _N))
    S = np.empty((block, n_full, _N)); Y = np.empty((block, n_full), dtype=int)
    for k in range(block):
        drive = emb[chunks[:, k]] @ W_in.T                        # [n_blocks, _N] — one matmul per within-block position
        if kind == "random":
            X = np.tanh(X @ A.T + drive)                          # ESN: mixing tanh recurrence
        elif kind == "hetero_esn":
            X = (1.0 - alpha) * X + alpha * np.tanh(X @ Wmat.T + drive)   # leaky-ESN: mixing + heterogeneous leak
        elif kind == "mtbounded":
            X = np.tanh(A * X + drive)                            # multi-timescale diagonal, BOUNDED (tanh)
        else:
            X = A * X + drive                                     # multi-timescale diagonal, LINEAR
        S[k] = X; Y[k] = chunks[:, k + 1]
    P = np.broadcast_to(np.arange(block)[:, None], (block, n_full))
    return S.reshape(block * n_full, _N), Y.reshape(-1), np.ascontiguousarray(P).reshape(-1)


def _ce_by_depth(kind, rng, vocab, tr_ids, ev_ids, block, epochs, lr):
    n_in = 64
    A = _build_A(kind, rng); W_in = _win(rng, n_in)
    emb = (rng.standard_normal((vocab.size, n_in)) * (1.0 / np.sqrt(n_in)))    # fixed random token embedding
    Str, Ytr, _ = _block_states(kind, A, W_in, emb, tr_ids, block)
    Sev, Yev, Pev = _block_states(kind, A, W_in, emb, ev_ids, block)
    m, s = Str.mean(0), Str.std(0) + 1e-6
    Wd = _fit_ridge_readout((Str - m) / s, Ytr, vocab.size)
    Ztr = np.concatenate([(Str - m) / s, np.ones((len(Str), 1))], 1)
    Zev = np.concatenate([(Sev - m) / s, np.ones((len(Sev), 1))], 1)
    T = _fit_temperature(Ztr, Wd, Ytr, rng)                      # temperature on a TRAIN subsample -> calibrated CE
    logits = Zev @ Wd
    P = _softmax_rows(logits / T)
    ce = -np.log(P[np.arange(len(Yev)), Yev] + 1e-12)
    acc = (np.argmax(logits, 1) == Yev)
    def bucket(arr, lo, hi):
        msk = (Pev >= lo) & (Pev < hi)
        return float(arr[msk].mean()) if msk.any() else float("nan")
    ce_d = {"shallow(1-4)": bucket(ce, 0, 4), "mid(5-16)": bucket(ce, 4, 16), "deep(17+)": bucket(ce, 16, block)}
    acc_d = {"shallow(1-4)": bucket(acc, 0, 4), "mid(5-16)": bucket(acc, 4, 16), "deep(17+)": bucket(acc, 16, block)}
    mag = np.linalg.norm(Sev, axis=1)                            # raw state magnitude (drift diagnostic)
    return {**ce_d, "acc": acc_d, "mag_sh": bucket(mag, 0, 4), "mag_deep": bucket(mag, 16, block)}


def _bigram_by_depth(P_bi, ev_ids, block):
    """Bigram CE bucketed by the SAME within-block depth as the reservoirs (reset per block). If the bigram's DEEP-bucket
    CE ≈ the reservoirs' deep CE, then deep-context prediction is essentially memoryless at this scale ⇒ the task is a
    NULL DISCRIMINATOR for any long-range mechanism (the thin-deep-signal scale confound the reslm ladder already hit)."""
    ev = np.asarray(ev_ids); ce = []; pos = []
    for b0 in range(0, len(ev) - 1, block):
        hi = min(b0 + block + 1, len(ev))
        for k in range(b0, hi - 1):
            p = P_bi[ev[k], ev[k + 1]]
            ce.append(-math.log(max(p, 1e-12))); pos.append(k - b0)
    ce = np.asarray(ce); pos = np.asarray(pos)
    def bk(lo, hi):
        msk = (pos >= lo) & (pos < hi); return float(ce[msk].mean()) if msk.any() else float("nan")
    return {"shallow(1-4)": bk(0, 4), "mid(5-16)": bk(4, 16), "deep(17+)": bk(16, block)}


def run(seed, n_sent, block=64, epochs=300, lr=0.5, vocab_size=1000):
    rng = np.random.default_rng(seed)
    sents = load_sentences("data/corpus/tinystories.txt", n_sent)
    perm = rng.permutation(len(sents)); ev = [sents[i] for i in perm[-max(200, n_sent // 10):]]
    pool = [sents[i] for i in perm[:-max(200, n_sent // 10)]]
    vocab = Vocab.build(pool, V=vocab_size)
    def flat(ss):
        out = []
        for s in ss:
            out += vocab.ids(s)
        return out
    tr_ids = flat(pool); ev_ids = flat(ev)
    # bigram baseline (position-independent) — overall + by-depth (the null-discriminator control)
    P_bi = fit_bigram([vocab.ids(s) for s in pool], vocab.size)
    bi_ce, _, _ = bigram_ce(P_bi, [vocab.ids(s) for s in ev])
    bi_depth = _bigram_by_depth(P_bi, ev_ids, block)
    out = {}
    seed_off = {"random": 0, "multitimescale": 7, "mtbounded": 13, "hetero_esn": 19}
    arms = ("random", "multitimescale", "mtbounded", "hetero_esn")
    for kind in arms:
        out[kind] = _ce_by_depth(kind, np.random.default_rng(seed * 31 + seed_off[kind]),
                                 vocab, tr_ids, ev_ids, block, epochs, lr)
    r_deep = out["random"]["deep(17+)"]; h_deep = out["hetero_esn"]["deep(17+)"]
    print(f"[ssm-lm seed={seed}] bigram_ce={bi_ce:.3f} chance={math.log(vocab.size):.3f} | by-depth next-token CE (lower=better):")
    for kind in arms:
        a = out[kind]["acc"]
        print(f"    {kind:14s}: CE shallow={out[kind]['shallow(1-4)']:.3f} mid={out[kind]['mid(5-16)']:.3f} deep={out[kind]['deep(17+)']:.3f}"
              f"  | mag sh={out[kind]['mag_sh']:.2f} deep={out[kind]['mag_deep']:.2f} | acc deep={a['deep(17+)']:.3f}")
    print(f"    {'bigram':14s}: CE shallow={bi_depth['shallow(1-4)']:.3f} mid={bi_depth['mid(5-16)']:.3f} deep={bi_depth['deep(17+)']:.3f}  (by-depth null-discriminator control)")
    bi_deep = bi_depth["deep(17+)"]; bi_mid = bi_depth["mid(5-16)"]
    # DISCRIMINATOR-VALIDITY: does the plain ESN beat the by-depth bigram at MID (5-16)? (per 2026-07-11-SCALE-reservoir-wins-mid)
    r_mid = out["random"]["mid(5-16)"]
    disc_valid = r_mid < bi_mid - 0.02
    print(f"    DISCRIMINATOR: random-ESN mid={r_mid:.3f} vs bigram-mid={bi_mid:.3f} -> "
          f"{'VALID (reservoir beats bigram at mid = real signal present)' if disc_valid else 'INVALID (no mid signal -> scale up more)'}")
    # The mission-relevant lever: does the leaky-ESN (mixing + heterogeneous time constants) beat the PLAIN ESN at deep?
    go = (h_deep < r_deep - 0.02) and (h_deep < bi_deep - 0.02)
    best_res_deep = min(out[k]["deep(17+)"] for k in arms)
    null_disc = best_res_deep >= bi_deep - 0.02
    print(f"    DEEP-context: hetero_esn={h_deep:.3f} mt_linear={out['multitimescale']['deep(17+)']:.3f} best_reservoir={best_res_deep:.3f} vs plain_random={r_deep:.3f} vs bigram_deep={bi_deep:.3f} "
          f"-> {'GO' if go else ('NULL-DISCRIMINATOR (no reservoir beats bigram at deep)' if null_disc else 'no')}")
    return dict(seed=seed, bigram_ce=round(bi_ce, 3), out=out, disc_valid=bool(disc_valid),
                hetero_deep=round(h_deep, 3), rand_deep=round(r_deep, 3), bigram_deep=round(bi_deep, 3),
                mt_deep=round(out["multitimescale"]["deep(17+)"], 3), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-sent", type=int, default=4000); ap.add_argument("--out", default=None)
    ap.add_argument("--vocab-size", type=int, default=1000); ap.add_argument("--block", type=int, default=64)
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    t0 = time.time()
    res = [run(s, a.n_sent, block=a.block, vocab_size=a.vocab_size) for s in seeds]
    if len(res) > 1:
        print(f"[ssm-lm] {sum(1 for r in res if r['go'])}/{len(res)} seeds GO")
    if a.out:
        json.dump(dict(results=res, elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
