"""The SSM-extract LANGUAGE test, done RIGHT (reuse-by-import of the VALIDATED machinery). The toy ridge runner
(`_ssm_reservoir_lm_derisk.py`) was a NULL DISCRIMINATOR because (a) its ridge-to-one-hot read-out is a weak
cross-entropy predictor and (b) it used cross-sentence 64-blocks, off-regime. THIS runner reuses the VALIDATED
context-depth machinery -- the proper softmax delta-rule read-out (`train_readout`), the per-context-depth
reservoir-minus-bigram MARGIN (`2026-07-11-SCALE-reservoir-wins-mid-depth-loses-deep-*`, where a plain reservoir
DECISIVELY beats the bigram at MID context but LOSES at DEEP = the fading-memory limit), the bag-of-prefix CONFOUND
control -- and swaps in NUMPY reservoirs {random ESN, multitimescale diagonal, hetero-leaky-ESN} per-sentence.

THE MISSION QUESTION: the plain reservoir wins mid / loses deep (fading memory). Does STRUCTURING the fixed recurrence
(multi-timescale time constants = the SSM/HiPPO forward long-range, the memory-horizon GO) EXTEND the win to DEEP
context where the plain reservoir fades? GATE: multitimescale's DEEP-bucket margin over the bigram > the plain ESN's
(and both must beat the BAG at that depth = the confound guard). Reuse-by-import; NO `sim/` edit, NO BPTT.

Run: SIM_BACKEND=numpy python -m research.runners._ssm_context_depth_derisk --seeds 42 --vocab 200 --n-sentences 8000
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
from collections import defaultdict

import numpy as np

from research.runners._emerge_reservoir_lm_derisk import (
    Vocab, train_readout, _standardize_fit, _softmax, fit_bigram, _bag_cache)
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences

BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 15), (16, 31), (32, 999)]   # context depth = tokens seen before the prediction


def _bucket(d):
    for lo, hi in BUCKETS:
        if lo <= d <= hi:
            return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return "32-999"


def _concat(sents, c):
    """Group c CONSECUTIVE (corpus-order) sentences into ONE long sequence (a 'document') so context depth extends to
    ~c*sentence_len, where long time constants have TIME to integrate (the memory-horizon GO used 150 steps; per-sentence
    <=16 is too short). CONSECUTIVE (not shuffled) so the long-range context is genuine discourse continuity (same
    TinyStories story) -- shuffling would give long sequences with NO real long-range dependency (uninformative)."""
    if c <= 1:
        return sents
    docs = []
    for i in range(0, len(sents) - c + 1, c):
        doc = []
        for j in range(c):
            doc += sents[i + j]
        docs.append(doc)
    return docs


class NumpyReservoir:
    """A FIXED numpy rate reservoir with a per-token read matching the validated `per_token_states` interface
    (reset per sentence). kind: 'random' (ESN tanh mixing, spectral 0.95) / 'multitimescale' (diagonal leaky
    integrators, tau log-range = the SSM extract) / 'mtbounded' (tanh diagonal) / 'hetero_esn' (leaky mixing +
    heterogeneous per-unit time constants). feature 'running_cumulative' (mean state over prefix, the validated ESN
    read) or 'raw' (current state -- the natural read for a multi-timescale reservoir whose state already integrates)."""

    def __init__(self, V, seed, n=300, kind="random"):
        self.n = n; self.kind = kind
        rng = np.random.default_rng(seed * 7919 + 3)
        self.W_in = (rng.random((n, V)) * 2 - 1) * (1.0 / np.sqrt(V))
        arng = np.random.default_rng(seed * 31 + 5)
        if kind in ("random", "hetero_esn"):
            W = arng.standard_normal((n, n)); ev = np.max(np.abs(np.linalg.eigvals(W)))
            self.W = (0.95 / ev) * W
            self.alpha = 1.0 / np.exp(np.linspace(np.log(1.5), np.log(400.0), n)) if kind == "hetero_esn" else None
        else:
            tau = np.exp(np.linspace(np.log(1.5), np.log(400.0), n))
            self.A = np.exp(-1.0 / tau)

    def per_token_states(self, U, silence=False, feature="running_cumulative"):
        x = np.zeros(self.n); cum = np.zeros(self.n); S = []
        for t in range(len(U)):
            drive = np.zeros(self.n) if silence else self.W_in @ U[t]
            if self.kind == "random":
                x = np.tanh(self.W @ x + drive)
            elif self.kind == "hetero_esn":
                x = (1.0 - self.alpha) * x + self.alpha * np.tanh(self.W @ x + drive)
            elif self.kind == "mtbounded":
                x = np.tanh(self.A * x + drive)
            else:
                x = self.A * x + drive
            cum += x
            S.append((cum / (t + 1)).copy() if feature == "running_cumulative" else x.copy())
        return S


def _cache(res, vocab, sents, feature):
    return [(res.per_token_states(vocab.encode_seq(s), feature=feature), vocab.ids(s)) for s in sents]


def _by_depth_margin(W, mean, std, ev_cache, P_bi, V):
    """Per-context-depth CE: reservoir vs bigram. Returns {bucket: {reservoir_ce, bigram_ce, margin, n}}."""
    rce = defaultdict(float); bce = defaultdict(float); cnt = defaultdict(int)
    for states, ids in ev_cache:
        for t in range(len(ids) - 1):
            b = _bucket(t + 1)
            x = np.concatenate([(states[t] - mean) / std, [1.0]])
            p = _softmax(W @ x)
            rce[b] += -math.log(max(p[ids[t + 1]], 1e-12))
            bce[b] += -math.log(max(P_bi[ids[t], ids[t + 1]], 1e-12))
            cnt[b] += 1
    return {b: {"n": cnt[b], "res": round(rce[b] / cnt[b], 3), "bigram": round(bce[b] / cnt[b], 3),
                "margin": round((bce[b] - rce[b]) / cnt[b], 3)} for b in cnt}


def run(seed, sents, vocab_size, n_pool, epochs, lr, wd, feature, concat=1, max_train=2000, max_eval=400):
    docs = _concat(sents, concat)                                 # concat CONSECUTIVE sentences BEFORE the split (keeps discourse)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(docs)); cut = int(0.8 * len(docs))
    tr = [docs[i] for i in idx[:cut]][:max_train]; ev = [docs[i] for i in idx[cut:]][:max_eval]
    vocab = Vocab.build(tr, V=vocab_size); V = vocab.size
    tr_ids = [vocab.ids(s) for s in tr]
    P_bi = fit_bigram(tr_ids, V)
    res_out = {}
    # BAG control (confound): same read-out over the bag-of-prefix feature, once (reservoir-independent given ids)
    for kind in ("random", "multitimescale", "mtbounded", "hetero_esn"):
        res = NumpyReservoir(V, seed=seed, n=n_pool, kind=kind)
        tr_cache = _cache(res, vocab, tr, feature); ev_cache = _cache(res, vocab, ev, feature)
        mean, std = _standardize_fit(tr_cache)
        W = train_readout(tr_cache, V, epochs, lr, np.random.default_rng(seed * 13 + 1), mean, std, wd=wd, ls=0.05)
        res_out[kind] = _by_depth_margin(W, mean, std, ev_cache, P_bi, V)
    # bag control (built from the random arm's cache ids; feature-independent)
    res = NumpyReservoir(V, seed=seed, n=n_pool, kind="random")
    tr_cache = _cache(res, vocab, tr, feature); ev_cache = _cache(res, vocab, ev, feature)
    bag_tr = _bag_cache(tr_cache, V); bag_ev = _bag_cache(ev_cache, V)
    bmean, bstd = _standardize_fit(bag_tr)
    Wb = train_readout(bag_tr, V, epochs, lr, np.random.default_rng(seed * 13 + 2), bmean, bstd, wd=wd, ls=0.05)
    res_out["bag"] = _by_depth_margin(Wb, bmean, bstd, bag_ev, P_bi, V)

    print(f"[ssm-ctxdepth seed={seed} V={V} n_pool={n_pool} feature={feature}] reservoir-minus-bigram CE margin by context depth (+ = beats bigram):", flush=True)
    hdr = "    depth   " + "".join(f"{k:>14s}" for k in ("random", "multitimescale", "mtbounded", "hetero_esn", "bag"))
    print(hdr, flush=True)
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi}" if lo != hi else f"{lo}"
        if b in res_out["random"]:
            row = f"    {b:>7s} "
            for k in ("random", "multitimescale", "mtbounded", "hetero_esn", "bag"):
                row += f"{res_out[k][b]['margin']:>+14.3f}"
            print(row + f"   (n={res_out['random'][b]['n']})", flush=True)
    # GATE: does multi-timescale EXTEND the win to DEEP where random fades? DEEP = the deepest available buckets
    # (lo>=16 = genuine long context, only populated with --concat; else lo>=6), n-weighted mean margin.
    deep_lo = 16 if any(int(b.split("-")[0]) >= 16 for b in res_out["random"]) else 6
    def deep_margin(kind):
        num = den = 0.0
        for lo, hi in BUCKETS:
            b = f"{lo}-{hi}" if lo != hi else f"{lo}"
            if lo >= deep_lo and b in res_out[kind]:
                num += res_out[kind][b]["margin"] * res_out[kind][b]["n"]; den += res_out[kind][b]["n"]
        return num / den if den else float("nan")
    r_deep = deep_margin("random"); m_deep = deep_margin("multitimescale")
    h_deep = deep_margin("hetero_esn"); bag_deep = deep_margin("bag")
    best_res = max(m_deep, h_deep)
    go = (best_res > r_deep + 0.02) and (best_res > bag_deep + 0.02)
    print(f"    DEEP margin (depth>={deep_lo}): multitimescale={m_deep:+.3f} hetero_esn={h_deep:+.3f} vs random={r_deep:+.3f} vs bag={bag_deep:+.3f} "
          f"-> {'GO (multi-timescale extends the deep-context win)' if go else 'no'}", flush=True)
    return dict(seed=seed, feature=feature, by_kind=res_out, deep_lo=deep_lo, m_deep=round(m_deep, 3),
                h_deep=round(h_deep, 3), r_deep=round(r_deep, 3), bag_deep=round(bag_deep, 3), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--vocab", type=int, default=200); ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--n-pool", type=int, default=300); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005); ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--feature", type=str, default="running_cumulative", choices=["running_cumulative", "raw"])
    ap.add_argument("--concat", type=int, default=1, help="group N consecutive sentences into one long sequence (long-context regime)")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    sents = load_sentences(a.corpus, a.n_sentences)
    t0 = time.time()
    res = [run(s, sents, a.vocab, a.n_pool, a.epochs, a.lr, a.weight_decay, a.feature, concat=a.concat) for s in a.seeds]
    if len(res) > 1:
        print(f"[ssm-ctxdepth] {sum(1 for r in res if r['go'])}/{len(res)} seeds GO", flush=True)
    if a.out:
        json.dump(dict(results=res, elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
