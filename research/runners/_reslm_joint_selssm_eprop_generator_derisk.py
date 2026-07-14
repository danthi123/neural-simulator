"""PAST-RESERVOIR — the DECISIVE mission test the frozen coupling reframed: does input-dependent SELECTIVITY help a
JOINTLY-TRAINED recurrent generator carry deep context? The adversarial-verify of the FROZEN coupling showed that adding
the selective channel to a frozen e-prop reservoir is mostly a shallow readout FIX (the frozen reservoir already carries
the context; the channel just fixes its readout) + a modest deep-selective residual. The honest mission question lives in
the JOINT setting: e-prop-train the reservoir W_rec AND a co-resident selective channel TOGETHER (both by one-step-local
eligibility + FIXED RANDOM FEEDBACK — transport-free, no BPTT), and ask whether the joint model carries MORE deep context
than the reservoir-alone e-prop generator. (Rung 3 showed the selective SSM AS the primary memory beats a fixed reservoir
strongly at deep context; here it is a co-trained channel on top of a co-trained reservoir.)

MECHANISM (all transport-free; the reservoir e-prop is the committed rate-analogue-of-BDSP; the gate is the Rung-2 selective
eligibility). Per token:
  reservoir:  h_t   = (1-a) h_{t-1} + a tanh(W_rec h_{t-1} + W_in x_t + b)
  selective:  c_t   = lam_t c_{t-1} + (1-lam_t) inj_t ,  lam_t = sigmoid(w.E[tok_t]+b_g) ,  inj = W_in_sel E[tok_t]
  read-out:   p_t   = softmax(W_out [h_t; c_t]) ,  err = p - onehot(next)
  W_out += lr_out (delta-rule)
  reservoir W_rec  += lr_rec * (Bh @ err) * e_rec     [e_rec = forward eligibility of h wrt W_rec; Bh fixed random]
  gate (w,b_g)     += lr_gate * (Bc @ err) * e_gate   [e_gate = forward selective eligibility; Bc fixed random]
NO BPTT, NO weight transport (both learning signals are fixed-random-feedback broadcasts of the read-out error).

ARMS (single variable = the selective channel present-and-co-trained):
  - joint_eprop      : reservoir e-prop + read-out over h_t ONLY (the emergent generator baseline)
  - joint_eprop_sel  : reservoir e-prop + a CO-TRAINED selective channel, read-out over [h_t, c_t]
  - joint_eprop_fix  : same but lam FIXED (input-independent accumulator) -> isolates the SELECTIVITY vs extra co-trained memory
GO (>=5/6): joint_eprop_sel beats joint_eprop at deep context (d>=4) by margin, AND beats joint_eprop_fix (selectivity, not
just an extra co-trained channel). A GO = input-dependent selectivity helps a TRAINED recurrent generator carry deep context
(the mission claim, in the right setting). An honest NEGATIVE (the trained reservoir absorbs the selective function) redirects.

Run (fan <=6 procs): python -m research.runners._reslm_joint_selssm_eprop_generator_derisk --seeds 42
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

from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket

N_HID = 120
SPECTRAL = 1.1
ALPHA = 0.3
EPOCHS = 5
LR_OUT = 0.02
LR_REC = 0.02
LR_GATE = 0.02
D_IN = 32
FORGET_BIAS = 2.5


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def _train_eval(seed, tr_ids, ev_ids, V, arm):
    """Joint e-prop training. arm in {joint_eprop, joint_eprop_sel, joint_eprop_fix}. Reservoir W_rec is CO-TRAINED by
    e-prop (random feedback) in ALL arms; the selective arms ALSO co-train a selective channel c_t (gate by eligibility +
    random feedback). Single variable = whether/how the selective channel is present. NO BPTT, NO transport."""
    use_sel = arm != "joint_eprop"
    rng = np.random.default_rng(seed * 13 + 7)
    res = RateReservoir(V, N_HID, seed, alpha=ALPHA, spectral=SPECTRAL)
    a = res.alpha
    # selective-channel params
    Es = rng.standard_normal((V, D_IN)) * 0.8
    Win_s = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    wg = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    bg = np.full(N_HID, FORGET_BIAS)
    fixed_lam = _sig(rng.standard_normal(N_HID) * 0.5 + FORGET_BIAS)
    feat_dim = N_HID + (N_HID if use_sel else 0)
    W_out = rng.standard_normal((V, feat_dim)) * 0.01
    Bh = rng.standard_normal((N_HID, V)) / np.sqrt(V)             # fixed random feedback -> reservoir (transport-free)
    Bc = rng.standard_normal((N_HID, V)) / np.sqrt(V)             # fixed random feedback -> gate
    order = np.arange(len(tr_ids))
    for _ep in range(EPOCHS):
        rng.shuffle(order)
        for si in order:
            ids = tr_ids[si]
            if len(ids) < 2:
                continue
            h = np.zeros(N_HID); e_rec = np.zeros((N_HID, N_HID))
            c = np.zeros(N_HID); ew = np.zeros((N_HID, D_IN)); ec = np.zeros(N_HID)
            for t in range(len(ids) - 1):
                h_prev = h
                pre = res.W_rec @ h_prev + res.W_in[:, ids[t]] + res.b
                act = np.tanh(pre)
                h = (1 - a) * h_prev + a * act
                if use_sel:
                    u = Es[ids[t]]; inj = Win_s @ u
                    lam = fixed_lam if arm == "joint_eprop_fix" else _sig(wg @ u + bg)
                    c_prev = c; c = lam * c_prev + (1.0 - lam) * inj
                    feat = np.concatenate([h, c])
                else:
                    feat = h
                p = _softmax(W_out @ feat)
                err = p.copy(); err[ids[t + 1]] -= 1.0
                W_out -= LR_OUT * np.outer(err, feat)
                # reservoir e-prop (random feedback, forward eligibility)
                psi = a * (1.0 - act * act)
                e_rec = (1 - a)[:, None] * e_rec + np.outer(psi, h_prev)
                L_h = Bh @ err
                res.W_rec -= LR_REC * (L_h[:, None] * e_rec)
                # gate eligibility (selective arm, trained; fixed arm holds lam constant so no gate update)
                if arm == "joint_eprop_sel":
                    dl = lam * (1.0 - lam); base = (c_prev - inj)
                    ew = lam[:, None] * ew + (dl * base)[:, None] * u[None, :]
                    ec = lam * ec + dl * base
                    L_c = Bc @ err
                    wg -= LR_GATE * (L_c[:, None] * ew)
                    bg -= LR_GATE * (L_c * ec)
    # eval per-depth CE
    ce = defaultdict(float); cnt = defaultdict(int)
    for ids in ev_ids:
        h = np.zeros(N_HID); c = np.zeros(N_HID)
        for t in range(len(ids) - 1):
            h = (1 - a) * h + a * np.tanh(res.W_rec @ h + res.W_in[:, ids[t]] + res.b)
            if use_sel:
                u = Es[ids[t]]; inj = Win_s @ u
                lam = fixed_lam if arm == "joint_eprop_fix" else _sig(wg @ u + bg)
                c = lam * c + (1.0 - lam) * inj
                feat = np.concatenate([h, c])
            else:
                feat = h
            p = _softmax(W_out @ feat)
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
    arms = {}; cnt = None
    for arm in ("joint_eprop", "joint_eprop_sel", "joint_eprop_fix"):
        arms[arm], c2 = _train_eval(seed, tr_ids, ev_ids, V, arm)
        cnt = c2 if cnt is None else cnt
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
    dp = {aa: _agg(arms[aa]) for aa in arms}
    sel_gain = dp["joint_eprop"] - dp["joint_eprop_sel"]      # >0 = co-trained selectivity LOWERS deep CE vs reservoir-alone
    fix_gain = dp["joint_eprop"] - dp["joint_eprop_fix"]
    go = bool(sel_gain > 0.02 and fix_gain < sel_gain - 0.01)
    print(f"[joint seed={seed}] DEEP(d>=4) CE: eprop={dp['joint_eprop']:.3f} sel={dp['joint_eprop_sel']:.3f} "
          f"fix={dp['joint_eprop_fix']:.3f} bigram={dp['bigram']:.3f} | sel_gain={sel_gain:+.3f} fix_gain={fix_gain:+.3f} "
          f"-> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "deep": dp, "by_depth": arms, "sel_gain": sel_gain, "fix_gain": fix_gain, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--n-sent", type=int, default=4000)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--tr-cap", type=int, default=1200)
    ap.add_argument("--ev-cap", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s, a.corpus, a.n_sent, a.vocab, a.tr_cap, a.ev_cap) for s in a.seeds]
    print(f"[joint] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
