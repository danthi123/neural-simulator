"""SCALE mechanism probe (WHERE does the reservoir beat the bigram on real text?). The aggregate next-token CE has the
reservoir hovering AT the bigram once the bigram is well-estimated (the data sweep) -- but aggregate CE is dominated by
the many EASY local predictions where a bigram is already near-optimal. The reservoir's whole point is HIGHER-ORDER
context (its fading memory over the prefix), which a bigram (previous token only) structurally cannot use. So we break
the held-out CE down by CONTEXT DEPTH d (= number of tokens seen before the prediction): the bigram uses only the
previous token regardless of d, so its per-depth CE is ~flat; the reservoir sees tokens 0..t, so if it is genuinely using
longer context its ADVANTAGE over the bigram should GROW with d. This isolates the reservoir's real contribution from the
bigram-dominated aggregate. Reuse-by-import (the Rung-1 machinery + the real-corpus loader); NO `sim/` edit, NO BPTT. CPU.
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

OUT = Path("research/findings/raw/_reslm_context_depth.json")
BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 99)]     # context-depth d = tokens seen before the prediction


def _bucket(d):
    for lo, hi in BUCKETS:
        if lo <= d <= hi:
            return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return f"{BUCKETS[-1][0]}-{BUCKETS[-1][1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--n-sentences", type=int, default=6000)
    ap.add_argument("--max-train-sents", type=int, default=2000)
    ap.add_argument("--max-eval-sents", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--n-pool", type=int, default=300)
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

        res = ReservoirStates(V, seed=seed, n=args.n_pool)
        tr_cache = _cache(res, vocab, tr); ev_cache = _cache(res, vocab, ev)
        mean, std = _standardize_fit(tr_cache)
        W = train_readout(tr_cache, V, args.epochs, args.lr, np.random.default_rng(seed * 13 + 1), mean, std,
                          wd=args.weight_decay, ls=0.05)
        P_bi = fit_bigram(tr_ids, V)

        # per-context-depth CE: reservoir (uses states[t] = context 0..t) vs bigram (uses only ids[t])
        rce = defaultdict(float); bce = defaultdict(float); cnt = defaultdict(int)
        for (states, ids) in ev_cache:
            for t in range(len(ids) - 1):
                d = t + 1                                          # context depth = tokens seen (0..t)
                b = _bucket(d)
                x = np.concatenate([(states[t] - mean) / std, [1.0]])
                p = _softmax(W @ x)
                tgt = ids[t + 1]
                rce[b] += -math.log(max(p[tgt], 1e-12))
                bce[b] += -math.log(max(P_bi[ids[t], tgt], 1e-12))
                cnt[b] += 1
        depth = {b: {"n": cnt[b], "reservoir_ce": round(rce[b] / cnt[b], 3), "bigram_ce": round(bce[b] / cnt[b], 3),
                     "margin": round((bce[b] - rce[b]) / cnt[b], 3)} for b in cnt}
        per_seed[str(seed)] = {"V": V, "n_train": len(tr), "by_depth": depth}
        print(f"[seed {seed}] V={V} n_tr={len(tr)} n_pool={args.n_pool} -- reservoir-minus-bigram CE margin by context depth:", flush=True)
        for lo, hi in BUCKETS:
            b = f"{lo}-{hi}" if lo != hi else f"{lo}"
            if b in depth:
                dd = depth[b]
                print(f"    depth {b:>5} (n={dd['n']:>5}): reservoir {dd['reservoir_ce']:.3f} vs bigram {dd['bigram_ce']:.3f} -> margin {dd['margin']:+.3f}", flush=True)

    out = {"runner": "_emerge_reservoir_lm_context_depth_derisk", "corpus": args.corpus, "seeds": args.seeds,
           "n_pool": args.n_pool, "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
