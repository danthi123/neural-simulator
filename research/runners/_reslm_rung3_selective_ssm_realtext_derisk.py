"""PAST-RESERVOIR Rung 3: the per-neuron SELECTIVE diagonal SSM (eligibility-trained, no BPTT/transport) on REAL TEXT
next-token prediction — does the LEARNED input-dependent gate beat a FIXED reservoir at DEEP context depth on TinyStories?
This escalates Rung 2 (synthetic gated-conjunction) to real language, reusing the reslm real-corpus + by-context-depth CE
machinery, comparing LIKE-FOR-LIKE (same token embedding, same local delta-rule read-out, same corpus/eval — the ONLY
variable is fixed lambda vs learned selective lambda_t = sigmoid(w.E[tok]+c)). NO `sim/` edit; self-contained numpy.

CEILING/REFERENCE (built-in, per the run-the-ceiling-early discipline): the bigram (memoryless n-gram floor) + the FIXED
reservoir (Rung-1/Ueda baseline). The claim is NOT "beats a transformer" — it is "the LOCALLY-trained selective gate
captures MORE deep-context than the fixed reservoir it upgrades," on real text, transport-free.

MECHANISM (Rung 2, online RTRL at EVERY token since an LM predicts the next token at every position):
  h_{t,i} = lam_{t,i}*h_{t-1,i} + (1-lam_{t,i})*inj_i ,  lam_{t,i}=sigmoid(w_i.E[tok_t]+c_i) ,  inj=W_in E[tok_t]
  e^w_{i,t}=lam*e^w + lam(1-lam)E[tok]*(h_prev-inj) ;  Δtheta ∝ -delta*e ;  delta = W_ro^T (p - onehot)  (read-out spatial bp)
  forget-bias init c=2.5 (survives the vanishing-eligibility trap). Reset h + traces per sentence.

ARMS (single variable = the gate; same E, W_in, read-out training, corpus):
  - selective:  lam input-dependent, gate (w,c) TRAINED online by the eligibility trace
  - fixed_res:  FIXED per-neuron lambda (leaky ESN, the reservoir baseline), only the read-out trained
  - bigram:     add-1 memoryless floor

GO (6-seed 42/43/44/100/101/102): at DEEP context depth (>=4 tokens), the selective SSM's next-token CE is LOWER than the
fixed reservoir's (selective captures more deep context), AND selective beats the bigram at deep depth, on >=5/6.

Run (fan <=6 procs, one per seed): python -m research.runners._reslm_rung3_selective_ssm_realtext_derisk --seeds 42
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import math
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket

D_IN = 32
N_HID = 200
FORGET_BIAS = 2.5
EPOCHS = 5
LR_RO = 0.02
LR_GATE = 0.02


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def _params(seed, V):
    rng = np.random.default_rng(seed * 7 + 2)
    E = rng.standard_normal((V, D_IN)) * 0.8
    Win = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    w = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    c = np.full(N_HID, FORGET_BIAS)
    fixed_lam = _sig(rng.standard_normal(N_HID) * 0.5 + FORGET_BIAS)
    return E, Win, w, c, fixed_lam


def _train_eval(seed, tr_ids, ev_ids, V, arm, feedback="transport"):
    """Arms differ ONLY in the gate. selective: lam=sig(w.E[tok]+c), gate trained. fixed_res: fixed lam. detached:
    input-dependent lam but gate UNTRAINED (tests LEARNING matters). randgate: gate reads a RANDOM token's embedding per
    step (VALID wrong-input control -- destroys CURRENT-token conditioning; unlike a dim-permutation which is invertible),
    gate trained on the random signal (tests the gate must read the CURRENT token). `feedback` = the gate's spatial
    learning-signal path: 'transport' (Wro^T = the committed run, biologically the weight-transport ceiling) or 'random'
    (fixed random feedback Bc = broadcast alignment = TRANSPORT-FREE; the honesty-closure the coupling's adversarial-verify
    established -- the temporal eligibility is always no-BPTT, only the SPATIAL feedback differs)."""
    E, Win, w, c, fixed_lam = _params(seed, V)
    train_gate = (arm in ("selective", "randgate"))
    rgate = np.random.default_rng(seed * 101 + 7)                 # random-token stream for the randgate control
    Bc = np.random.default_rng(seed * 191 + 11).standard_normal((N_HID, V)) / np.sqrt(V)  # fixed random feedback (gate)
    Wro = np.zeros((V, N_HID))
    for _ep in range(EPOCHS):
        for ids in tr_ids:
            h = np.zeros(N_HID); ew = np.zeros((N_HID, D_IN)); ec = np.zeros(N_HID)
            for t in range(len(ids) - 1):
                u = E[ids[t]]; inj = Win @ u
                ug = E[int(rgate.integers(V))] if arm == "randgate" else u   # gate input
                lam = fixed_lam if arm == "fixed_res" else _sig(w @ ug + c)
                h_prev = h; h = lam * h_prev + (1.0 - lam) * inj
                if train_gate:
                    dl = lam * (1.0 - lam); base = (h_prev - inj)
                    ew = lam[:, None] * ew + (dl * base)[:, None] * ug[None, :]
                    ec = lam * ec + dl * base
                z = Wro @ h; p = _softmax(z)
                err = p.copy(); err[ids[t + 1]] -= 1.0
                delta = (Bc @ err) if feedback == "random" else (Wro.T @ err)   # transport-free vs the ceiling
                Wro -= LR_RO * np.outer(err, h)
                if train_gate:
                    w -= LR_GATE * (delta[:, None] * ew)
                    c -= LR_GATE * (delta * ec)
    ce = defaultdict(float); cnt = defaultdict(int)
    for ids in ev_ids:
        h = np.zeros(N_HID)
        for t in range(len(ids) - 1):
            u = E[ids[t]]; inj = Win @ u
            ug = E[int(rgate.integers(V))] if arm == "randgate" else u
            lam = fixed_lam if arm == "fixed_res" else _sig(w @ ug + c)
            h = lam * h + (1.0 - lam) * inj
            p = _softmax(Wro @ h)
            b = _bucket(t + 1)
            ce[b] += -math.log(max(p[ids[t + 1]], 1e-12)); cnt[b] += 1
    return {b: ce[b] / cnt[b] for b in cnt}, dict(cnt)


def run(seed, corpus, n_sent, vocab_sz, feedback="transport"):
    sents = load_sentences(corpus, n_sent)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sents)); cut = int(0.8 * len(sents))
    tr = [sents[i] for i in idx[:cut]][:1500]; ev = [sents[i] for i in idx[cut:]][:400]
    vocab = Vocab.build(tr, V=vocab_sz); V = vocab.size
    tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
    P_bi = fit_bigram(tr_ids, V)
    arms = {}
    cnt = None
    for arm in ("selective", "fixed_res", "detached", "randgate"):
        arms[arm], c2 = _train_eval(seed, tr_ids, ev_ids, V, arm, feedback=feedback)
        cnt = c2 if cnt is None else cnt
    # bigram by depth
    bce = defaultdict(float); bcnt = defaultdict(int)
    for ids in ev_ids:
        for t in range(len(ids) - 1):
            b = _bucket(t + 1)
            bce[b] += -math.log(max(P_bi[ids[t], ids[t + 1]], 1e-12)); bcnt[b] += 1
    arms["bigram"] = {b: bce[b] / bcnt[b] for b in bce}
    deep = ["4-5", "6-9", "10-99"]                                # deep-context aggregate (depth >= 4)
    def _agg(d):
        num = sum(d.get(b, 0) * cnt.get(b, 0) for b in deep); den = sum(cnt.get(b, 0) for b in deep)
        return num / den if den else float("nan")
    dp = {a: _agg(arms[a]) for a in arms}
    # GO: LOWER CE is better. selective must beat fixed_res AND detached AND permgate AND bigram at deep context.
    go = bool(dp["selective"] < dp["fixed_res"] - 0.02 and dp["selective"] < dp["detached"] - 0.02
              and dp["selective"] < dp["randgate"] - 0.02 and dp["selective"] < dp["bigram"] - 0.02)
    print(f"[rung3 seed={seed}] DEEP(d>=4) CE: selective={dp['selective']:.3f} fixed_res={dp['fixed_res']:.3f} "
          f"detached={dp['detached']:.3f} randgate={dp['randgate']:.3f} bigram={dp['bigram']:.3f} "
          f"| sel<fix {dp['fixed_res']-dp['selective']:+.3f} sel<det {dp['detached']-dp['selective']:+.3f} "
          f"sel<rand {dp['randgate']-dp['selective']:+.3f} sel<big {dp['bigram']-dp['selective']:+.3f} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "deep": dp, "by_depth": arms, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--n-sent", type=int, default=6000)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--gate-feedback", type=str, default="transport", choices=["transport", "random"])  # committed default
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s, a.corpus, a.n_sent, a.vocab, a.gate_feedback) for s in a.seeds]
    print(f"[rung3] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
