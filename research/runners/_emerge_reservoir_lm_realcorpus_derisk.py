"""SCALE TEST (the decisive one): does the emergent generator -- a FIXED spiking reservoir (EMERGE-82 OnBridgeLSM) + a
shallow one-step-local-delta read-out (NO BPTT) -- beat the bigram on REAL TEXT, not just the controlled EMERGE-62
template stream? Rung 1 was validated on the bounded template grammar; the honest open question (the scale frontier) is
whether the same dynamics-earned next-token generation holds on a REAL corpus, where the bigram is a MUCH stronger
baseline (real text has rich local structure). Here we run the exact Rung-1 machinery over TinyStories / WikiText words.
Reuse-by-import: `Vocab`, `ReservoirStates`, `train_readout`, `eval_ce`, `_cache`, `fit_bigram`/`bigram_ce`,
`fit_trigram`/`trigram_ce`, `_standardize_fit` from the Rung-1 runner. NO `sim/` edit, NO BPTT. CPU numpy.

GO = the reservoir's held-out next-token cross-entropy BEATS the bigram AND the permuted-corpus control (scramble train
word order, fresh read-out) does NOT (i.e. the reservoir is capturing REAL higher-order structure, not an artifact). An
honest NEGATIVE (the tiny reservoir does not beat the bigram on real text) is a first-class result -- it maps where the
reservoir path needs scale (a bigger reservoir / more data) to generalize beyond a template grammar.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, re, time, math
from pathlib import Path
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import (
    Vocab, ReservoirStates, train_readout, eval_ce, _cache, _standardize_fit,
    fit_bigram, bigram_ce, fit_trigram, trigram_ce)

OUT = Path("research/findings/raw/_reslm_realcorpus.json")
_WORD = re.compile(r"[a-z']+")


def load_sentences(path, max_sents, min_len=3, max_len=16):
    """Read a real corpus, split into sentences on sentence-final punctuation, tokenize to lowercase words."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore").lower()
    sents = []
    for raw in re.split(r"[.!?]", txt):
        w = _WORD.findall(raw)
        if min_len <= len(w) <= max_len:
            sents.append(w)
            if len(sents) >= max_sents:
                break
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--n-sentences", type=int, default=4000)
    ap.add_argument("--max-train-sents", type=int, default=1400)
    ap.add_argument("--max-eval-sents", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.001, help="L2 decay on the read-out (regularization)")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    sents = load_sentences(args.corpus, args.n_sentences)
    t0 = time.time(); per_seed = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(sents))
        cut = int(0.8 * len(sents))
        tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
        ev = [sents[i] for i in idx[cut:]][:args.max_eval_sents]
        vocab = Vocab.build(tr, V=args.vocab)
        V = vocab.size
        tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]

        res = ReservoirStates(V, seed=seed, n=args.n_pool)
        tr_cache = _cache(res, vocab, tr)
        ev_cache = _cache(res, vocab, ev)
        mean, std = _standardize_fit(tr_cache)
        W = train_readout(tr_cache, V, args.epochs, args.lr, np.random.default_rng(seed * 13 + 1), mean, std,
                          wd=args.weight_decay, ls=args.label_smoothing)
        res_ce, res_acc, _n = eval_ce(W, mean, std, ev_cache, V)

        P_bi = fit_bigram(tr_ids, V); bi_ce, bi_acc, _ = bigram_ce(P_bi, ev_ids)
        ctx = fit_trigram(tr_ids, V); tri_ce, tri_acc, _ = trigram_ce(ctx, P_bi, ev_ids)
        chance = math.log(V)

        # anti-cheat: permuted-corpus (scramble each train sentence's word order, fresh read-out) -> no real structure
        prng = np.random.default_rng(seed * 7 + 3)
        perm = [list(prng.permutation(s)) for s in tr]
        perm_cache = _cache(res, vocab, perm)
        pmean, pstd = _standardize_fit(perm_cache)
        Wp = train_readout(perm_cache, V, args.epochs, args.lr, np.random.default_rng(seed * 23 + 1), pmean, pstd,
                           wd=args.weight_decay, ls=args.label_smoothing)
        perm_ce, _pa, _ = eval_ce(Wp, pmean, pstd, ev_cache, V)

        beats_bigram = res_ce < bi_ce
        perm_collapses = perm_ce >= bi_ce
        seed_go = bool(beats_bigram and perm_collapses)
        per_seed[str(seed)] = {
            "V": V, "n_train": len(tr), "n_eval": len(ev),
            "reservoir_ce": round(res_ce, 3), "reservoir_acc": round(res_acc, 3),
            "bigram_ce": round(bi_ce, 3), "bigram_acc": round(bi_acc, 3),
            "trigram_ce": round(tri_ce, 3), "unigram_chance_ce": round(chance, 3),
            "permuted_corpus_ce": round(perm_ce, 3),
            "beats_bigram": beats_bigram, "permuted_collapses": perm_collapses, "seed_go": seed_go}
        print(f"[seed {seed}] V={V} n_tr={len(tr)} | RESERVOIR CE {res_ce:.3f} (acc {res_acc:.3f}) vs bigram {bi_ce:.3f} "
              f"(acc {bi_acc:.3f}) [trigram {tri_ce:.3f} | chance {chance:.3f}] | permuted-corpus {perm_ce:.3f} | "
              f"seed_go {seed_go}", flush=True)

    n_go = sum(per_seed[s]["seed_go"] for s in per_seed)
    out = {"runner": "_emerge_reservoir_lm_realcorpus_derisk", "corpus": args.corpus, "seeds": args.seeds,
           "vocab": args.vocab, "per_seed": per_seed, "n_go": n_go, "n_seeds": len(args.seeds),
           "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nAGG GO {n_go}/{len(args.seeds)} on REAL corpus {args.corpus} ({out['elapsed_s']}s) -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
