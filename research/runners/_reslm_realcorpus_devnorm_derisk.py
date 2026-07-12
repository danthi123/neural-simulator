"""DEV-NORM on the real-text emergent GENERATOR (the R3<->generation-ladder convergence de-risk).

The RUNG-1 reservoir generator uses a FIXED input projection (the ESN/LSM invariant). My R3 arc showed learning the
INPUT representation beats the fixed reservoir at rate, but the credit-based input learning is coarseness-bound on spikes;
the DEVELOPMENTAL local normalization (PPMI-family divisive gain, validated on count-corpus, works on spikes) is the
spiking-compatible input-scale lever. This applies it to the generator's input WITHOUT any shared-machinery edit: each
token's input code is pre-scaled by its frequency gain `g_v = scale/(sigma + freq_v)^k` (common tokens down-weighted,
rare/informative tokens emphasized -- PPMI/TF-IDF-like), so the reservoir state emphasizes informative tokens. Since the
drive is `W_in @ (onehot * g) = g * (W_in @ onehot)`, pre-scaling the code IS the divisive input gain -- no edit to
`per_token_states`.

GATE: does dev-norm's held-out next-token CE beat the fixed-input baseline (and both vs the bigram)? Anti-cheats:
PERMUTED-gain (shuffle which token gets which gain -> collapse to baseline) + the runner's own permuted-corpus.
Reuse-by-import (ReservoirStates / train_readout / eval_ce / bigram); numpy/CPU; NO sim/ edit, NO shared-runner edit.

Run (fan 6 seeds across cores):
  for s in 42 43 44 100 101 102; do OMP_NUM_THREADS=3 SIM_BACKEND=numpy python -u -m \
    research.runners._reslm_realcorpus_devnorm_derisk --seeds $s --json raw/_devnorm_s$s.json > raw/_devnorm_s$s.log 2>&1 & done; wait
"""
import argparse, json, re, time
from collections import Counter
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


def token_gain(vocab, tr, sigma, k, permute_seed=None):
    """g_v = scale/(sigma + freq_v)^k, mean-normalized over the vocab so the MEAN drive scale is preserved (keeps the
    reservoir in-regime). Common tokens (high freq) -> small g -> down-weighted; rare -> large g -> emphasized."""
    c = Counter(w for s in tr for w in s)
    total = max(1, sum(c.values()))
    freq = np.zeros(vocab.size)
    for w, n in c.items():
        freq[vocab.id(w)] += n
    freq = freq / total
    g = 1.0 / (sigma + freq) ** k
    if permute_seed is not None:                                    # anti-cheat: break the freq<->token correspondence
        np.random.default_rng(permute_seed).shuffle(g)
    g = g / g.mean()                                                # mean-normalize -> mean gain 1.0
    return g


def cache_scaled(res, vocab, sents, gain):
    """Same (states, ids) format as _cache, but each token's one-hot is pre-scaled by its gain before the reservoir."""
    out = []
    for s in sents:
        codes = vocab.encode_seq(s)
        ids = vocab.ids(s)
        if len(ids):
            scaled = np.asarray([codes[t] * gain[ids[t]] for t in range(len(ids))])
        else:
            scaled = codes
        out.append((res.per_token_states(scaled), ids))
    return out


def _train_eval(cache_tr, cache_ev, V, args, seed):
    mean, std = _standardize_fit(cache_tr)
    W = train_readout(cache_tr, V, args.epochs, args.lr, np.random.default_rng(seed * 13 + 1), mean, std,
                      wd=args.weight_decay, ls=args.label_smoothing)
    ce, acc, _ = eval_ce(W, mean, std, cache_ev, V)
    return round(ce, 3), round(acc, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--n-sentences", type=int, default=3000)
    ap.add_argument("--max-train-sents", type=int, default=900)
    ap.add_argument("--max-eval-sents", type=int, default=220)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--dev-sigma", type=float, default=0.01)
    ap.add_argument("--dev-k", type=float, default=1.0)
    ap.add_argument("--json", type=str, default="research/findings/raw/_reslm_devnorm.json")
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
        tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
        res = ReservoirStates(V, seed=seed, n=args.n_pool)

        # baseline (fixed input) vs dev-norm (frequency-gained input) vs permuted-gain anti-cheat -- SAME reservoir.
        g = token_gain(vocab, tr, args.dev_sigma, args.dev_k)
        gp = token_gain(vocab, tr, args.dev_sigma, args.dev_k, permute_seed=seed * 613 + 5)
        base_ce, base_acc = _train_eval(*(lambda c: (c(tr), c(ev)))(lambda ss: cache_scaled(res, vocab, ss, np.ones(V))), V, args, seed)
        dev_ce, dev_acc = _train_eval(cache_scaled(res, vocab, tr, g), cache_scaled(res, vocab, ev, g), V, args, seed)
        perm_ce, perm_acc = _train_eval(cache_scaled(res, vocab, tr, gp), cache_scaled(res, vocab, ev, gp), V, args, seed)
        P_bi = fit_bigram(tr_ids, V); bi_ce, bi_acc, _ = bigram_ce(P_bi, ev_ids)

        d = {"V": V, "n_tr": len(tr), "baseline_ce": base_ce, "devnorm_ce": dev_ce, "permgain_ce": perm_ce,
             "bigram_ce": round(bi_ce, 3), "dev_minus_base": round(dev_ce - base_ce, 3),
             "dev_beats_base": dev_ce < base_ce, "dev_beats_bigram": dev_ce < bi_ce,
             "perm_collapses": perm_ce >= base_ce - 0.01}
        per_seed[str(seed)] = d
        print(f"[seed {seed}] V={V} n_tr={len(tr)} | baseline CE {base_ce:.3f} | DEVNORM CE {dev_ce:.3f} "
              f"(dev-base {dev_ce-base_ce:+.3f}) | permgain CE {perm_ce:.3f} | bigram {bi_ce:.3f} | "
              f"dev<base {dev_ce<base_ce} dev<bigram {dev_ce<bi_ce}", flush=True)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)},
              open(args.json, "w"), indent=2)
    print(f"[devnorm] wrote {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
