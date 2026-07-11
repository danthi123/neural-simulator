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
from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir

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
        rec = {"V": V, "by_arm": {}}
        for arm in args.arms:
            depth, agg = train_and_eval(res, tr_ids, ev_ids, V, args.beta, args.n_window, arm,
                                        args.epochs, args.lr, args.weight_decay, seed, P_bi)
            rec["by_arm"][arm] = {"aggregate_ce": agg, "by_depth": depth}
        per_seed[str(seed)] = rec
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
