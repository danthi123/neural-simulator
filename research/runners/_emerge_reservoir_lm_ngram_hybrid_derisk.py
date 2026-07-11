"""SCALE mechanism test (the fix for the deep-context loss): the context-depth analysis showed the reservoir beats the
bigram at MID depth (2-5 tokens) but LOSES at deep context (6+) because its running-cumulative feature washes the recent
tokens out. The principled fix (standard echo-state read-out design: read the INPUT alongside the reservoir state): give
the read-out the RECENT-K token identities it needs for the sharp local prediction, ALONGSIDE the reservoir's higher-order
context. Feature = [reservoir running-cumulative state || onehot(prev token) || ... onehot(t-K+1)]. The read-out then has
BOTH the bigram's signal (recent tokens) AND the reservoir's distal context -> it should beat the bigram at ALL depths
(strictly more information than the bigram). K=0 = reservoir only (the baseline that loses deep); K=1 = + the bigram's
own feature; K=2 = + trigram context. Per-context-depth CE vs the bigram. Reuse-by-import; NO `sim/` edit, NO BPTT. CPU.
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
    Vocab, ReservoirStates, train_readout, _cache, _standardize_fit, _softmax, fit_bigram)
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket

OUT = Path("research/findings/raw/_reslm_ngram_hybrid.json")


def augment(cache, V, K, use_reservoir=True):
    """Feature per position t (predicts t+1): [reservoir_state (if use_reservoir)] ++ last-K token one-hots (ids[t..t-K+1]).
       use_reservoir=False -> a LEARNED K-gram (recent tokens only, no reservoir) -- the fair strong baseline to isolate
       the reservoir's contribution beyond the recent tokens."""
    out = []
    for states, ids in cache:
        aug = []
        for t in range(len(ids)):
            feat = [states[t]] if use_reservoir else []
            for k in range(K):
                oh = np.zeros(V)
                if t - k >= 0:
                    oh[ids[t - k]] = 1.0
                feat.append(oh)
            aug.append(np.concatenate(feat) if feat else np.zeros(1))
        out.append((aug, ids))
    return out


def per_depth_ce(W, mean, std, cache, P_bi):
    rce = defaultdict(float); bce = defaultdict(float); cnt = defaultdict(int)
    for states, ids in cache:
        for t in range(len(ids) - 1):
            b = _bucket(t + 1)
            x = np.concatenate([(states[t] - mean) / std, [1.0]])
            p = _softmax(W @ x); tgt = ids[t + 1]
            rce[b] += -math.log(max(p[tgt], 1e-12)); bce[b] += -math.log(max(P_bi[ids[t], tgt], 1e-12)); cnt[b] += 1
    tot_r = sum(rce.values()); tot_b = sum(bce.values()); n = sum(cnt.values())
    return ({b: {"n": cnt[b], "margin": round((bce[b] - rce[b]) / cnt[b], 3),
                 "reservoir_ce": round(rce[b] / cnt[b], 3), "bigram_ce": round(bce[b] / cnt[b], 3)} for b in cnt},
            round(tot_r / n, 3), round(tot_b / n, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--max-train-sents", type=int, default=2500)
    ap.add_argument("--max-eval-sents", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--k-recent", type=int, nargs="+", default=[0, 1, 2])
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
        tr_ids = [vocab.ids(s) for s in tr]
        res = ReservoirStates(V, seed=seed, n=args.n_pool)
        tr_cache0 = _cache(res, vocab, tr); ev_cache0 = _cache(res, vocab, ev)
        P_bi = fit_bigram(tr_ids, V)
        per_seed[str(seed)] = {"V": V, "n_train": len(tr), "by_k": {}}
        for K in args.k_recent:
            entry = {}
            for use_res in (True, False):
                if not use_res and K == 0:
                    continue                                      # K=0 without reservoir = empty feature (skip)
                trc = augment(tr_cache0, V, K, use_res); evc = augment(ev_cache0, V, K, use_res)
                mean, std = _standardize_fit(trc)
                W = train_readout(trc, V, args.epochs, args.lr, np.random.default_rng(seed * 13 + 1), mean, std,
                                  wd=args.weight_decay, ls=0.05)
                depth, agg_m, agg_b = per_depth_ce(W, mean, std, evc, P_bi)
                tag = "res+tok" if use_res else "tok_only"
                entry[tag] = {"aggregate_ce": agg_m, "aggregate_margin_vs_bigram": round(agg_b - agg_m, 3), "by_depth": depth}
            per_seed[str(seed)]["by_k"][str(K)] = entry
            # print: for K>=1, both res+tok and tok_only margins vs bigram + the RESERVOIR CONTRIBUTION (res+tok CE - tok_only CE)
            rt = entry.get("res+tok"); to = entry.get("tok_only")
            if rt and to:
                contrib = {b: round(to["by_depth"][b]["reservoir_ce"] - rt["by_depth"][b]["reservoir_ce"], 3)
                           for b in rt["by_depth"] if b in to["by_depth"]}      # reservoir contribution = tok_only_CE - res+tok_CE
                deep = [b for b in ("6-9", "10-99") if b in contrib]
                deepc = np.mean([contrib[b] for b in deep]) if deep else float("nan")
                print(f"[seed {seed}] K={K}: res+tok agg-margin-vs-bigram {rt['aggregate_margin_vs_bigram']:+.3f} | "
                      f"tok_only(learned {K}-gram) {to['aggregate_margin_vs_bigram']:+.3f} | RESERVOIR CONTRIB (nats, +=reservoir helps beyond tokens): "
                      + " ".join(f"d{b}:{contrib[b]:+.2f}" for lo,hi in BUCKETS for b in [f'{lo}-{hi}' if lo!=hi else f'{lo}'] if b in contrib)
                      + f" | deep(6+) {deepc:+.3f}", flush=True)
            elif rt:
                print(f"[seed {seed}] K={K}: reservoir-only agg-margin-vs-bigram {rt['aggregate_margin_vs_bigram']:+.3f}", flush=True)

    out = {"runner": "_emerge_reservoir_lm_ngram_hybrid_derisk", "corpus": args.corpus, "seeds": args.seeds,
           "n_pool": args.n_pool, "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
