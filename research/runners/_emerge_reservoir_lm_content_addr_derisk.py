"""LONG-RANGE de-risk (the mechanism the state-memory fork located): a NON-FADING, CONTENT-ADDRESSABLE store. The
capstone + e-prop + hetero-tau findings proved the long-range (d10+) wall is the reservoir's FADING STATE, and it is NOT
surpassable by a bigger/learned/slower reservoir (a leaky state dilutes distal items). The research gate (a9f7285e64d7865ce)
forked to A2: reach long-range with a content-addressable associative-memory READ (attention == modern-Hopfield read,
Ramsauer 2020; the biological form = hippocampal CA3 pattern completion / the project's FHRR cleanup). This de-risks the
DIRECTION cheaply: does appending a FIXED content-addressable retrieval to the reservoir read-out lift d10+ where the
fading state alone (and longer tau) could not?

MECHANISM (per token t, predicting token t+1 from context 0..t):
  keys    K_tau = reservoir state h_tau (the CONTEXT at which past token tau occurred), tau in [max(0,t-N) .. t-1]
  query   q     = h_t (the current context)
  attn    a_tau = softmax( beta * <q, K_tau> / sqrt(n) )      # content-addressable (modern-Hopfield / attention) read
  value   v_tau = onehot(ids[tau+1])                          # the token that FOLLOWED the retrieved context (kNN-LM read)
  retrieved r_t = sum_tau a_tau v_tau                          # a soft-retrieved next-token distribution from similar past
  feature_t = [ h_t  ||  r_t ]  (dim n + V)  -> softmax(W_out feature_t) predicts ids[t+1]
This is NON-FADING (any past tau is reachable by content) + CONTENT-ADDRESSED (retrieved by current-context similarity),
exactly the two properties a leaky state lacks. FIXED reservoir (homogeneous a=0.3, the n-gram-level baseline) so the
single variable is the content-addressable read.

ARMS (all reuse the SAME fixed reservoir + the same delta-rule read-out; only the appended feature differs):
  base      : read-out over h_t only  (= the fading-state baseline that loses at deep context)
  content   : + the content-addressable read (the mechanism under test)
  shuffle   : content, but the KEYS are shuffled within the sentence (query no longer matches the right context) ANTI-CHEAT
  uniform   : content, but beta=0 (uniform attention -> retrieves the average past token, no content-addressing) ANTI-CHEAT
  recent1   : + onehot(prev token) (the bigram feature) -> does content add BEYOND the recent token at DEEP context?
GO = content beats base at d10+ (where the fading state failed) AND beats recent1 at d10+ (adds beyond the recent token)
AND shuffle/uniform do NOT (content-addressing is load-bearing). Reuse-by-import; numpy; NO `sim/` edit, NO BPTT.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import (
    Vocab, _softmax, fit_bigram, train_readout, _standardize_fit)
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket
from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir, train as eprop_train

OUT = Path("research/findings/raw/_reslm_content_addr.json")


def content_read(states, ids, V, beta, n, N_window, arm, rng):
    """Per token t: the content-addressable retrieval r_t (dim V). arm controls the read structure."""
    L = len(ids)
    ids = np.asarray(ids)
    key_order = np.arange(L)
    if arm == "shuffle":
        key_order = rng.permutation(L)                       # scramble WHICH past context each key represents
    R = np.zeros((L, V))
    inv = 1.0 / math.sqrt(n)
    for t in range(L):
        lo = max(0, t - N_window)
        past = np.arange(lo, t)                              # tau in [lo, t-1]
        if past.size == 0:
            continue
        q = states[t]
        Kt = states[key_order[past]]                        # (p, n) keys (shuffled-context under 'shuffle')
        if arm == "uniform":
            a = np.full(past.size, 1.0 / past.size)         # beta=0 -> uniform (no content-addressing)
        else:
            sc = beta * (Kt @ q) * inv
            a = _softmax(sc)
        # value = onehot(token that FOLLOWED the retrieved context) = ids[tau+1] (clamp tau+1<=t so no future leak: tau<t so tau+1<=t)
        vtok = ids[np.minimum(past + 1, t)]                 # tau+1, capped at t (all <= t = past/current, NO future leak)
        for j, tk in zip(a, vtok):
            R[t, tk] += j
    return R


def features(res, ids, V, beta, N_window, arm, rng):
    states = res.forward_states(ids)
    S = np.array(states)                                    # (L, n)
    if arm == "base":
        return [S[t] for t in range(len(ids))]
    if arm == "recent1":
        out = []
        for t in range(len(ids)):
            oh = np.zeros(V); oh[ids[t]] = 1.0              # prev token (the bigram feature)
            out.append(np.concatenate([S[t], oh]))
        return out
    R = content_read(S, ids, V, beta, res.n, N_window, arm, rng)
    return [np.concatenate([S[t], R[t]]) for t in range(len(ids))]


def train_and_eval(res, tr, ev, V, beta, N_window, arm, epochs, lr, wd, seed, P_bi):
    """Reuse the VALIDATED _standardize_fit + train_readout (same as the ngram-hybrid runner) so the augmented-feature
       training is the same proven path (label-smoothed, robust standardization)."""
    rng = np.random.default_rng(seed * 13 + 5)
    trc = [(features(res, ids, V, beta, N_window, arm, rng), ids) for ids in tr]
    evc = [(features(res, ids, V, beta, N_window, arm, rng), ids) for ids in ev]
    mean, std = _standardize_fit(trc)
    W = train_readout(trc, V, epochs, lr, np.random.default_rng(seed * 7 + 1), mean, std, wd=wd, ls=0.05)
    rce = defaultdict(float); bce = defaultdict(float); cnt = defaultdict(int)
    for feats, ids in evc:
        for t in range(len(ids) - 1):
            b = _bucket(t + 1)
            x = np.concatenate([(feats[t] - mean) / std, [1.0]])
            p = _softmax(W @ x); tgt = ids[t + 1]
            rce[b] += -math.log(max(p[tgt], 1e-12)); bce[b] += -math.log(max(P_bi[ids[t], tgt], 1e-12)); cnt[b] += 1
    depth = {k: {"n": cnt[k], "ce": round(rce[k] / cnt[k], 3), "bigram_ce": round(bce[k] / cnt[k], 3)} for k in cnt}
    agg = sum(rce.values()) / sum(cnt.values())
    return depth, round(agg, 3)


def interp_eval(res, tr, ev, V, beta, N_window, epochs, lr, wd, seed, P_bi):
    """kNN-LM-style interpolation read: train the BASE read-out on the reservoir state, then combine its prediction with
       the content-addressable retrieval distribution r_t as p_final = (1-lam)*p_base + lam*r_t (lam swept). Biologically
       a cortex(base) + hippocampal-retrieval(r_t) complementary-systems mix. Reports the BEST-lam CE by depth vs base
       (does a PROPER integration of the learned-key retrieval beat base at deep, where content-append could not?)."""
    rng = np.random.default_rng(seed * 13 + 5)
    trc = [(features(res, ids, V, beta, N_window, "base", rng), ids) for ids in tr]
    mean, std = _standardize_fit(trc)
    W = train_readout(trc, V, epochs, lr, np.random.default_rng(seed * 7 + 1), mean, std, wd=wd, ls=0.05)
    lams = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]
    # per-lam, per-depth CE
    agg = {la: defaultdict(float) for la in lams}; cnt = defaultdict(int)
    for ids in ev:
        S = np.array(res.forward_states(ids))
        R = content_read(S, ids, V, beta, res.n, N_window, "content", rng)
        for t in range(len(ids) - 1):
            b = _bucket(t + 1); tgt = ids[t + 1]
            x = np.concatenate([(S[t] - mean) / std, [1.0]])
            pb = _softmax(W @ x)
            rt = R[t]; rt = rt / max(rt.sum(), 1e-9)                # normalize the retrieval to a distribution
            cnt[b] += 1
            for la in lams:
                pf = (1 - la) * pb + la * rt
                agg[la][b] += -math.log(max(pf[tgt], 1e-12))
    # HONEST: pick a SINGLE global lam that minimizes AGGREGATE eval CE (1 hyperparameter), report per-depth gain at it.
    tot = {la: sum(agg[la].values()) for la in lams}
    glam = min(tot, key=tot.get)                                # single global lambda
    depth = {}
    for b in cnt:
        ces = {la: agg[la][b] / cnt[b] for la in lams}
        bestpd = min(ces, key=ces.get)                          # per-depth best (OPTIMISTIC ceiling)
        depth[b] = {"n": cnt[b], "base_ce": round(ces[0.0], 3), "global_lam": glam,
                    "global_ce": round(ces[glam], 3), "gain": round(ces[0.0] - ces[glam], 3),
                    "gain_bestpd": round(ces[0.0] - ces[bestpd], 3), "best_lam_pd": bestpd}
    return depth


def gated_interp_eval(res, tr, ev, V, beta, N_window, epochs, lr, wd, seed, P_bi, lam_max=0.5):
    """CLS-GATED interpolation on the RETRIEVAL's own CONFIDENCE (not base uncertainty -- that conflates short-context
       ambiguity with deep-context fading). lam_t = lam_max * clip((conf_t - c_lo)/(c_hi - c_lo)), conf_t = max(r_t) (the
       content-addressable read's peak = it found a strongly-matching past context). Trust the hippocampal retrieval when
       IT is confident. c_lo/c_hi = 20th/80th percentiles of conf fit on TRAIN (no eval tuning). Gated CE by depth vs base."""
    rng = np.random.default_rng(seed * 13 + 5)
    trc = [(features(res, ids, V, beta, N_window, "base", rng), ids) for ids in tr]
    mean, std = _standardize_fit(trc)
    W = train_readout(trc, V, epochs, lr, np.random.default_rng(seed * 7 + 1), mean, std, wd=wd, ls=0.05)

    def confs(seqs):
        out = []
        for ids in seqs:
            S = np.array(res.forward_states(ids)); R = content_read(S, ids, V, beta, res.n, N_window, "content", rng)
            for t in range(len(ids) - 1):
                rt = R[t]; s = rt.sum(); out.append(float(rt.max() / s) if s > 1e-9 else 0.0)
        return np.array(out)
    c = confs([ids for ids in tr]); c_lo, c_hi = np.percentile(c, 20), np.percentile(c, 80)  # gate calibrated on TRAIN

    gag = defaultdict(float); bag = defaultdict(float); cnt = defaultdict(int)
    for ids in ev:
        S = np.array(res.forward_states(ids)); R = content_read(S, ids, V, beta, res.n, N_window, "content", rng)
        for t in range(len(ids) - 1):
            b = _bucket(t + 1); tgt = ids[t + 1]
            x = np.concatenate([(S[t] - mean) / std, [1.0]]); pb = _softmax(W @ x)
            rt = R[t]; s = rt.sum(); rt = rt / max(s, 1e-9)
            conf = float(rt.max())
            lam = lam_max * float(np.clip((conf - c_lo) / max(c_hi - c_lo, 1e-9), 0.0, 1.0))
            pf = (1 - lam) * pb + lam * rt
            gag[b] += -math.log(max(pf[tgt], 1e-12)); bag[b] += -math.log(max(pb[tgt], 1e-12)); cnt[b] += 1
    return {b: {"n": cnt[b], "base_ce": round(bag[b] / cnt[b], 3), "gated_ce": round(gag[b] / cnt[b], 3),
                "gain": round((bag[b] - gag[b]) / cnt[b], 3)} for b in cnt}


def _gate_feats(pb, rt, t, maxpos=40.0):
    """Features for the learned CLS gate (per token): base uncertainty, retrieval confidence/entropy, depth proxy,
       base-retrieval agreement, base top-prob."""
    Hb = -np.sum(pb * np.log(pb + 1e-12)); Hr = -np.sum(rt * np.log(rt + 1e-12))
    agree = 1.0 if int(np.argmax(pb)) == int(np.argmax(rt)) else 0.0
    return np.array([Hb, rt.max(), Hr, min(t, maxpos) / maxpos, pb.max(), agree, 1.0])


def learned_gate_eval(res, tr, ev, V, beta, N_window, epochs, lr, wd, seed, P_bi, gate_epochs=4, gate_lr=0.2):
    """LEARNED CLS gate: lam_t = sigmoid(w . feats_t), w trained by gradient on the interpolated CE over TRAIN. Opens the
       hippocampal retrieval only where it helps (deep-where-retrieval-is-right), combining the confounded hand-signals.
       Reports gated CE by depth vs base on EVAL (w trained on train only)."""
    rng = np.random.default_rng(seed * 13 + 5)
    trc = [(features(res, ids, V, beta, N_window, "base", rng), ids) for ids in tr]
    mean, std = _standardize_fit(trc)
    W = train_readout(trc, V, epochs, lr, np.random.default_rng(seed * 7 + 1), mean, std, wd=wd, ls=0.05)

    def make_cache(seqs):
        out = []
        for ids in seqs:
            S = np.array(res.forward_states(ids)); R = content_read(S, ids, V, beta, res.n, N_window, "content", rng)
            rows = []
            for t in range(len(ids) - 1):
                x = np.concatenate([(S[t] - mean) / std, [1.0]]); pb = _softmax(W @ x)
                rt = R[t]; s = rt.sum(); rt = rt / max(s, 1e-9)
                rows.append((pb, rt, _gate_feats(pb, rt, t), ids[t + 1], t + 1))
            out.append(rows)
        return out
    trg = make_cache(tr); evg = make_cache(ev)
    # standardize gate features on train
    allf = np.array([r[2] for rows in trg for r in rows]); fmu = allf.mean(0); fsd = allf.std(0) + 1e-6
    w = np.zeros(len(fmu))
    for ep in range(gate_epochs):
        order = rng.permutation(len(trg))
        for si in order:
            for pb, rt, f, tgt, d in trg[si]:
                fs = (f - fmu) / fsd; lam = 1.0 / (1.0 + math.exp(-float(w @ fs))); lam *= 0.6  # cap lam<=0.6
                pf = (1 - lam) * pb + lam * rt; pft = max(pf[tgt], 1e-12)
                dL_dlam = -(rt[tgt] - pb[tgt]) / pft                    # d(-log pf[tgt])/dlam
                dlam_dz = (lam / 0.6) * (1 - lam / 0.6) * 0.6           # d lam / d z  (z = w.fs)
                w -= gate_lr * dL_dlam * dlam_dz * fs
    gag = defaultdict(float); bag = defaultdict(float); cnt = defaultdict(int); lam_by = defaultdict(list)
    for rows in evg:
        for pb, rt, f, tgt, d in rows:
            b = _bucket(d); fs = (f - fmu) / fsd
            lam = 0.6 / (1.0 + math.exp(-float(w @ fs)))
            pf = (1 - lam) * pb + lam * rt
            gag[b] += -math.log(max(pf[tgt], 1e-12)); bag[b] += -math.log(max(pb[tgt], 1e-12)); cnt[b] += 1
            lam_by[b].append(lam)
    return {b: {"n": cnt[b], "base_ce": round(bag[b] / cnt[b], 3), "gated_ce": round(gag[b] / cnt[b], 3),
                "gain": round((bag[b] - gag[b]) / cnt[b], 3), "mean_lam": round(float(np.mean(lam_by[b])), 3)} for b in cnt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/wikitext.txt")
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--n-sentences", type=int, default=6000)
    ap.add_argument("--max-train-sents", type=int, default=1500)
    ap.add_argument("--max-eval-sents", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--n-window", type=int, default=64)         # how far back the content read can reach
    ap.add_argument("--learned-keys", action="store_true")      # e-prop-train the reservoir first -> LEARNED keys (rung-2 test)
    ap.add_argument("--lr-rec", type=float, default=0.006)      # e-prop recurrent lr (when --learned-keys)
    ap.add_argument("--eprop-epochs", type=int, default=8)
    ap.add_argument("--arms", type=str, nargs="+", default=["base", "content", "shuffle", "uniform", "recent1"])
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    sents = load_sentences(args.corpus, args.n_sentences)
    t0 = time.time(); per_seed = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(sents)); cut = int(0.8 * len(sents))
        tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
        ev = [sents[i] for i in idx[cut:]][:args.max_eval_sents]
        vocab = Vocab.build(tr, V=args.vocab); V = vocab.size
        tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
        P_bi = fit_bigram(tr_ids, V)
        res = RateReservoir(V, args.n_pool, seed, alpha=0.3, spectral=1.1)   # FIXED homogeneous (the fading baseline)
        if args.learned_keys:                                   # RUNG-2 test: e-prop-LEARN the recurrent weights first
            eprop_train(res, tr_ids, V, args.eprop_epochs, 0.02, args.lr_rec, seed, mode="plastic")
            print(f"[seed {seed}] reservoir e-prop-trained -> LEARNED keys for the content read", flush=True)
        rec = {"V": V, "by_arm": {}}
        for arm in args.arms:
            if arm == "interp":                                 # kNN-LM interpolation read (special eval)
                depth = interp_eval(res, tr_ids, ev_ids, V, args.beta, args.n_window,
                                    args.epochs, args.lr, args.weight_decay, seed, P_bi)
                rec["by_arm"]["interp"] = {"by_depth": depth}
                glam = next(iter(depth.values()))["global_lam"]
                row = " ".join(f"d{k}:{depth[k]['gain']:+.3f}"
                               for lo,hi in BUCKETS for k in [f'{lo}-{hi}' if lo!=hi else f'{lo}'] if k in depth)
                print(f"[seed {seed}] INTERP single-global-λ={glam} CE gain over base (pos=interp BEATS base): {row}", flush=True)
                continue
            if arm == "gated":                                  # CLS entropy-gated interpolation (special eval)
                depth = gated_interp_eval(res, tr_ids, ev_ids, V, args.beta, args.n_window,
                                          args.epochs, args.lr, args.weight_decay, seed, P_bi)
                rec["by_arm"]["gated"] = {"by_depth": depth}
                row = " ".join(f"d{k}:{depth[k]['gain']:+.3f}"
                               for lo,hi in BUCKETS for k in [f'{lo}-{hi}' if lo!=hi else f'{lo}'] if k in depth)
                print(f"[seed {seed}] GATED (entropy-gated) CE gain over base (pos=gated BEATS base, NET): {row}", flush=True)
                continue
            if arm == "learned_gate":                           # LEARNED CLS gate (special eval)
                depth = learned_gate_eval(res, tr_ids, ev_ids, V, args.beta, args.n_window,
                                          args.epochs, args.lr, args.weight_decay, seed, P_bi)
                rec["by_arm"]["learned_gate"] = {"by_depth": depth}
                row = " ".join(f"d{k}:{depth[k]['gain']:+.3f}(λ{depth[k]['mean_lam']})"
                               for lo,hi in BUCKETS for k in [f'{lo}-{hi}' if lo!=hi else f'{lo}'] if k in depth)
                print(f"[seed {seed}] LEARNED-GATE CE gain over base (pos=BEATS base, NET; λ=mean gate): {row}", flush=True)
                continue
            depth, agg = train_and_eval(res, tr_ids, ev_ids, V, args.beta, args.n_window, arm,
                                        args.epochs, args.lr, args.weight_decay, seed, P_bi)
            rec["by_arm"][arm] = {"aggregate_ce": agg, "by_depth": depth}
        per_seed[str(seed)] = rec
        if "base" not in rec["by_arm"]:                          # e.g. interp-only run -> skip the CE-minus-base print
            continue
        base = rec["by_arm"]["base"]["by_depth"]
        def dd(arm, k):
            b = rec["by_arm"].get(arm, {}).get("by_depth", {})
            return round(b[k]["ce"] - base[k]["ce"], 3) if (arm in rec["by_arm"] and k in b and k in base) else None
        print(f"[seed {seed}] V={V} CE-minus-base by depth (neg=better than fading-state baseline):", flush=True)
        for arm in args.arms:
            if arm == "base":
                continue
            row = " ".join(f"d{k}:{dd(arm,k):+.2f}" for lo,hi in BUCKETS for k in [f'{lo}-{hi}' if lo!=hi else f'{lo}'] if dd(arm,k) is not None)
            print(f"    {arm:9s} agg{rec['by_arm'][arm]['aggregate_ce']-rec['by_arm']['base']['aggregate_ce']:+.3f}  {row}", flush=True)

    out = {"runner": "_emerge_reservoir_lm_content_addr_derisk", "corpus": args.corpus, "seeds": args.seeds,
           "beta": args.beta, "n_window": args.n_window, "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
