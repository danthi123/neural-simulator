"""PAST-RESERVOIR — the MISSION-CENTRAL COUPLING: does adding the validated learned SELECTIVE-SSM context channel to the
EMERGENT GENERATOR's read-out carry the DEEP context the (e-prop-trained) reservoir alone loses? This is the step from the
isolated selective-SSM ladder (Rung 1-4b: it beats a fixed reservoir on real text + on the spiking substrate + it SCALES)
to the ACTUAL emergent conversational cortex — the e-prop-trained rate reservoir LM (`_emerge_reservoir_lm_eprop_recurrent`,
the rate analogue of on-bridge BDSP: W_rec LEARNS by one-step-local eligibility, NO BPTT/transport).

THE SETUP (single variable = the selective channel; the reservoir is IDENTICAL across arms):
  1. Build + e-prop-TRAIN the reservoir ONCE per seed (mode='plastic' -> the emergent generator's learned cortex). FREEZE it.
  2. Precompute the frozen reservoir's h_t sequences ONCE (shared across all arms -> the O(n^2) cost is paid once).
  3. Per arm, add (or not) a per-neuron selective-SSM context channel c_t and train a read-out over the arm's feature:
       c_{t,i} = lam_{t,i} c_{t-1,i} + (1-lam_{t,i}) inj_i ,  lam_{t,i} = sigmoid(w_i . E[tok_t] + b_i) ,  inj = W_in_sel E[tok_t]
       one-step-local eligibility (Zucchet 2305.15947, survives input-dependent selectivity; NO BPTT/transport):
         e^w = lam e^w + lam(1-lam) E[tok] (c_prev - inj) ;  Dtheta ∝ -delta_c e ;  delta_c = (Wro over c-part)^T (p - onehot)
       forget-bias init b=2.5 (survives the vanishing-eligibility trap). Reset h + c + traces per sentence.

ARMS (the ONLY variable is the read-out feature / the gate):
  - eprop         : read-out over h_t ONLY  (the emergent generator AS-IS; the fading reservoir carries context)  [BASELINE]
  - eprop_sel     : read-out over [h_t, c_t], gate INPUT-DEPENDENT + trained  (the coupling under test)
  - eprop_sel_rand: [h_t, c_t] but the gate reads a RANDOM token/step (selectivity broken) -> ANTI-CHEAT (extra read-out
                    capacity, WRONG content -- must NOT beat eprop by the selective margin)
  - eprop_sel_fix : [h_t, c_t] but lam FIXED (input-INDEPENDENT constant leak = an extra slow LINEAR integrator, ~ALIF) ->
                    CONTROL: isolates that the gain is the INPUT-DEPENDENT selectivity (the conjunction), not merely an
                    extra slow memory channel.

METRIC: per-context-depth held-out CE (reuse BUCKETS/_bucket), deep aggregate d>=4. GO (>=5/6 seeds): eprop_sel beats eprop
at deep context by >=0.02, AND eprop_sel_rand does NOT beat eprop by that margin, AND eprop_sel's advantage > eprop_sel_fix's
(input-dependent selectivity adds MORE than a plain slow integrator). A GO = the generator's deep context is carried BETTER
by the learned selective gate than by the (even e-prop-trained) fading reservoir -> the rate-level path to fluent long-range
conversation (the selective channel is byte-equivalent on the spiking substrate, Rung 4b-iii-a). Reuse-by-import; NO sim/ edit.

Run (fan <=6 procs, one per seed): python -m research.runners._reslm_couple_selssm_into_eprop_generator_derisk --seeds 42
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse
import json
import math
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir, train
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket

# ---- reservoir (emergent generator) ----
N_HID = 140
SPECTRAL = 1.1
ALPHA = 0.3
EPROP_EPOCHS = 3
LR_OUT = 0.02
LR_REC = 0.02
# ---- selective channel ----
D_IN = 32
FORGET_BIAS = 2.5
ARM_EPOCHS = 4
LR_RO = 0.02
LR_GATE = 0.02


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def _sel_params(seed, V):
    rng = np.random.default_rng(seed * 7 + 5)
    E = rng.standard_normal((V, D_IN)) * 0.8
    Win = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    w = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    b = np.full(N_HID, FORGET_BIAS)
    fixed_lam = _sig(rng.standard_normal(N_HID) * 0.5 + FORGET_BIAS)
    return E, Win, w, b, fixed_lam


def _forward_h(res, ids):
    """Frozen-reservoir h_t sequence (the emergent generator cortex state). Shared across arms."""
    return res.forward_states(ids)   # non-ALIF -> list of h_t (N_HID,)


def _train_eval(seed, tr_hids, tr_ids, ev_hids, ev_ids, V, arm):
    """arm in {eprop, eprop_sel, eprop_sel_rand, eprop_sel_fix}. tr_hids/ev_hids = precomputed frozen-reservoir h_t seqs
    (aligned to tr_ids/ev_ids). Trains a read-out over the arm's feature; for the selective arms co-trains the gate by the
    one-step-local eligibility rule. Reservoir W_rec is FROZEN (already e-prop-trained = the emergent generator)."""
    use_sel = arm != "eprop"
    train_gate = arm in ("eprop_sel", "eprop_sel_rand")
    E, Win, w, b, fixed_lam = _sel_params(seed, V)
    feat_dim = N_HID + (N_HID if use_sel else 0)
    Wro = np.zeros((V, feat_dim))
    rgate = np.random.default_rng(seed * 131 + 3)                 # random-token stream for the rand control
    for _ep in range(ARM_EPOCHS):
        for hids, ids in zip(tr_hids, tr_ids):
            c = np.zeros(N_HID); ew = np.zeros((N_HID, D_IN)); ec = np.zeros(N_HID)
            for t in range(len(ids) - 1):
                h = hids[t]
                if use_sel:
                    u = E[ids[t]]; inj = Win @ u
                    ug = E[int(rgate.integers(V))] if arm == "eprop_sel_rand" else u
                    lam = fixed_lam if arm == "eprop_sel_fix" else _sig(w @ ug + b)
                    c_prev = c; c = lam * c_prev + (1.0 - lam) * inj
                    if train_gate:
                        dl = lam * (1.0 - lam); base = (c_prev - inj)
                        ew = lam[:, None] * ew + (dl * base)[:, None] * ug[None, :]
                        ec = lam * ec + dl * base
                    feat = np.concatenate([h, c])
                else:
                    feat = h
                p = _softmax(Wro @ feat)
                err = p.copy(); err[ids[t + 1]] -= 1.0
                Wro -= LR_RO * np.outer(err, feat)
                if train_gate:
                    delta_c = Wro[:, N_HID:].T @ err              # spatial bp through the read-out to the c-channel
                    w -= LR_GATE * (delta_c[:, None] * ew)
                    b -= LR_GATE * (delta_c * ec)
    # eval per-depth CE
    ce = defaultdict(float); cnt = defaultdict(int)
    reval = np.random.default_rng(seed * 131 + 3)                 # SAME random stream as training for the rand control
    for hids, ids in zip(ev_hids, ev_ids):
        c = np.zeros(N_HID)
        for t in range(len(ids) - 1):
            h = hids[t]
            if use_sel:
                u = E[ids[t]]; inj = Win @ u
                ug = E[int(reval.integers(V))] if arm == "eprop_sel_rand" else u
                lam = fixed_lam if arm == "eprop_sel_fix" else _sig(w @ ug + b)
                c = lam * c + (1.0 - lam) * inj
                feat = np.concatenate([h, c])
            else:
                feat = h
            p = _softmax(Wro @ feat)
            b_d = _bucket(t + 1)
            ce[b_d] += -math.log(max(p[ids[t + 1]], 1e-12)); cnt[b_d] += 1
    return {k: ce[k] / cnt[k] for k in cnt}, dict(cnt)


def run(seed, corpus, n_sent, vocab_sz, tr_cap=1200, ev_cap=300):
    sents = load_sentences(corpus, n_sent)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sents)); cut = int(0.8 * len(sents))
    tr = [sents[i] for i in idx[:cut]][:tr_cap]; ev = [sents[i] for i in idx[cut:]][:ev_cap]
    vocab = Vocab.build(tr, V=vocab_sz); V = vocab.size
    tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
    tr_ids = [s for s in tr_ids if len(s) >= 2]; ev_ids = [s for s in ev_ids if len(s) >= 2]
    P_bi = fit_bigram(tr_ids, V)

    # 1. build + e-prop-train the emergent generator (reservoir); then FREEZE.
    res = RateReservoir(V, N_HID, seed, alpha=ALPHA, spectral=SPECTRAL)
    train(res, tr_ids, V, EPROP_EPOCHS, LR_OUT, LR_REC, seed, mode="plastic")

    # 2. precompute frozen-reservoir h_t seqs ONCE (shared across arms).
    tr_hids = [_forward_h(res, ids) for ids in tr_ids]
    ev_hids = [_forward_h(res, ids) for ids in ev_ids]

    arms = {}
    cnt = None
    for arm in ("eprop", "eprop_sel", "eprop_sel_rand", "eprop_sel_fix"):
        arms[arm], c2 = _train_eval(seed, tr_hids, tr_ids, ev_hids, ev_ids, V, arm)
        cnt = c2 if cnt is None else cnt
    # bigram per depth (floor)
    bce = defaultdict(float); bcnt = defaultdict(int)
    for ids in ev_ids:
        for t in range(len(ids) - 1):
            bd = _bucket(t + 1)
            bce[bd] += -math.log(max(P_bi[ids[t], ids[t + 1]], 1e-12)); bcnt[bd] += 1
    arms["bigram"] = {k: bce[k] / bcnt[k] for k in bce}

    deep = ["4-5", "6-9", "10-99"]
    def _agg(d):
        num = sum(d.get(x, 0) * cnt.get(x, 0) for x in deep); den = sum(cnt.get(x, 0) for x in deep)
        return num / den if den else float("nan")
    dp = {a: _agg(arms[a]) for a in arms}
    sel_gain = dp["eprop"] - dp["eprop_sel"]         # >0 = selective LOWERS deep CE vs the generator baseline
    rand_gain = dp["eprop"] - dp["eprop_sel_rand"]
    fix_gain = dp["eprop"] - dp["eprop_sel_fix"]
    go = bool(sel_gain > 0.02 and rand_gain < sel_gain - 0.02 and fix_gain < sel_gain - 0.01)
    print(f"[couple seed={seed}] DEEP(d>=4) CE: eprop={dp['eprop']:.3f} sel={dp['eprop_sel']:.3f} "
          f"rand={dp['eprop_sel_rand']:.3f} fix={dp['eprop_sel_fix']:.3f} bigram={dp['bigram']:.3f} "
          f"| sel_gain={sel_gain:+.3f} rand_gain={rand_gain:+.3f} fix_gain={fix_gain:+.3f} -> {'GO' if go else 'no'}",
          flush=True)
    return {"seed": seed, "deep": dp, "by_depth": arms,
            "sel_gain": sel_gain, "rand_gain": rand_gain, "fix_gain": fix_gain, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--n-sent", type=int, default=4000)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--tr-cap", type=int, default=1200)          # max training sentences (scale lever; default = committed)
    ap.add_argument("--ev-cap", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s, a.corpus, a.n_sent, a.vocab, a.tr_cap, a.ev_cap) for s in a.seeds]
    print(f"[couple] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
