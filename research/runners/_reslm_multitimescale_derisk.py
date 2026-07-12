"""DeepESN MULTI-TIMESCALE de-risk (#1 external-scan lever) on the real-text reservoir generator — reuse-by-import.

Ueda 2025 (arXiv:2503.01724): a fixed reservoir is n-gram-bounded at small scale (only reaches syntactic capability at
~16k units/100M words). Our probes confirm the n-gram bound at tractable scale. The DeepESN multi-timescale lever asks
the ONE relative question tractable at our scale: at a FIXED reservoir/read budget, does a MULTI-TIMESCALE forward state
(read the reservoir at slow + fast timescales together) capture more next-token structure than a SINGLE-timescale read?
If yes, multi-timescale reaches the reservoir's ceiling with fewer units (cheaper path to Ueda-scale). Cheap: read the
SAME OnBridgeLSM at feature='running_cumulative' (slow, whole-prefix mean) AND 'per_window' (fast, recency), concat ->
read-out. NO new reservoir, NO shared edit; two deterministic reads of the same washed trajectory.

Arms (same reservoir, same read-out pipeline): SINGLE (running_cumulative, n_pool feats) · MULTI (concat slow+fast,
2*n_pool feats) · SIZE-MATCHED FLAT control (a single-timescale FLAT reservoir with 2*n_pool units = same feature dim
as MULTI). GATE: MULTI beats SINGLE on held-out next-token CE AND beats the SIZE-MATCHED flat (so the gain is TIMESCALE
diversity, not just more features) AND beats the bigram. numpy/CPU; NO sim/ edit.

Run:  OMP_NUM_THREADS=4 python -u -m research.runners._reslm_multitimescale_derisk --seeds 42 --json raw/_multits_s42.json
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import (
    Vocab, ReservoirStates, train_readout, eval_ce, _standardize_fit, fit_bigram, bigram_ce)

_WORD = re.compile(r"[a-z']+")


def load_sentences(path, max_sents, min_len=3, max_len=16):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            for chunk in re.split(r"[.!?]", line.lower()):
                w = _WORD.findall(chunk)
                if min_len <= len(w) <= max_len:
                    out.append(w)
                    if len(out) >= max_sents:
                        return out
    return out


def cache_feature(res, vocab, sents, feature):
    return [(res.per_token_states(vocab.encode_seq(s), feature=feature), vocab.ids(s)) for s in sents]


def cache_multi(res, vocab, sents):
    """Concat the SLOW (running_cumulative) + FAST (per_window) reads of the SAME reservoir trajectory per token."""
    out = []
    for s in sents:
        U = vocab.encode_seq(s)
        slow = res.per_token_states(U, feature="running_cumulative")
        fast = res.per_token_states(U, feature="per_window")
        cat = [np.concatenate([slow[t], fast[t]]) for t in range(len(slow))]
        out.append((cat, vocab.ids(s)))
    return out


def _train_eval(cache_tr, cache_ev, V, a, seed):
    mean, std = _standardize_fit(cache_tr)
    W = train_readout(cache_tr, V, a.epochs, a.lr, np.random.default_rng(seed * 13 + 1), mean, std,
                      wd=a.weight_decay, ls=a.label_smoothing)
    ce, acc, _ = eval_ce(W, mean, std, cache_ev, V)
    return round(ce, 3), round(acc, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--n-sentences", type=int, default=4000)
    ap.add_argument("--max-train-sents", type=int, default=700)
    ap.add_argument("--max-eval-sents", type=int, default=220)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--json", type=str, default="research/findings/raw/_reslm_multits.json")
    args = ap.parse_args()

    sents = load_sentences(args.corpus, args.n_sentences)
    per_seed = {}
    t0 = time.time()
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(sents)); cut = int(0.8 * len(sents))
        tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
        ev = [sents[i] for i in idx[cut:]][:args.max_eval_sents]
        vocab = Vocab.build(tr, V=args.vocab); V = vocab.size
        ev_ids = [vocab.ids(s) for s in ev]; tr_ids = [vocab.ids(s) for s in tr]

        res = ReservoirStates(V, seed=seed, n=args.n_pool)                    # n_pool reservoir
        single_ce, _ = _train_eval(cache_feature(res, vocab, tr, "running_cumulative"),
                                   cache_feature(res, vocab, ev, "running_cumulative"), V, args, seed)
        multi_ce, _ = _train_eval(cache_multi(res, vocab, tr), cache_multi(res, vocab, ev), V, args, seed)
        res2 = ReservoirStates(V, seed=seed, n=2 * args.n_pool)               # size-matched flat control (2*n_pool)
        flat2_ce, _ = _train_eval(cache_feature(res2, vocab, tr, "running_cumulative"),
                                  cache_feature(res2, vocab, ev, "running_cumulative"), V, args, seed)
        P_bi = fit_bigram(tr_ids, V); bi_ce, _, _ = bigram_ce(P_bi, ev_ids)

        d = {"single_ce": single_ce, "multi_ce": multi_ce, "flat2x_ce": flat2_ce, "bigram_ce": round(bi_ce, 3),
             "multi_minus_single": round(multi_ce - single_ce, 3), "multi_minus_flat2x": round(multi_ce - flat2_ce, 3),
             "multi_beats_single": multi_ce < single_ce, "multi_beats_flat2x": multi_ce < flat2_ce,
             "multi_beats_bigram": multi_ce < bi_ce}
        per_seed[str(seed)] = d
        print(f"[seed {seed}] single {single_ce} | MULTI-TS {multi_ce} (vs single {multi_ce-single_ce:+.3f}) | "
              f"flat-2x {flat2_ce} (vs flat {multi_ce-flat2_ce:+.3f}) | bigram {bi_ce:.3f} | "
              f"multi<single {multi_ce<single_ce} multi<flat2x {multi_ce<flat2_ce}", flush=True)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)},
              open(args.json, "w"), indent=2)
    print(f"[multits] wrote {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
