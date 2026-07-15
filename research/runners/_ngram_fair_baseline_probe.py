"""FAIR n-gram baseline probe — the rigorous baseline any selective-SSM "fluency crossover" must beat.

The 2026-07-15 confound was caught TWICE using the tuned add-k bigram; the DEEPER control (2026-07-14 CONTROLS-REFUTED)
is the INTERPOLATED TRIGRAM — the reservoir-LM lost to it even when it beat the bigram. So a genuine "beats the n-gram"
crossover must beat BOTH a tuned add-k bigram AND an interpolated (deleted-interpolation) trigram, on the IDENTICAL
data/split the vectorized runner uses (`default_rng(seed).permutation`, ev=last n_eval, tr=first n_train of the rest).

Run: python -m research.runners._ngram_fair_baseline_probe --n-sentences 60000 --n-train 24000 --n-eval 500 --vocab 200 --seed 42
Compares to the selective-SSM sel_ce in raw/_fluency_np*_nt<nt>_s<seed>.json if present.
"""
import argparse, json, os, math, glob
import numpy as np
from research.runners._emerge_reservoir_lm_derisk import Vocab
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences


def _split(corpus, n_sentences, n_eval, n_train, V, seed):
    sents = load_sentences(corpus, n_sentences)
    perm = np.random.default_rng(seed).permutation(len(sents))
    ev = [sents[i] for i in perm[-n_eval:]]; pool = [sents[i] for i in perm[:-n_eval]]
    vocab = Vocab.build(pool, V=V); Veff = vocab.size
    tr = pool[:n_train]
    tr_ids = [np.asarray(vocab.ids(s), dtype=np.int64) for s in tr]
    ev_ids = [np.asarray(vocab.ids(s), dtype=np.int64) for s in ev]
    return tr_ids, ev_ids, Veff


def tuned_bigram_ce(tr_ids, ev_ids, V):
    c = np.zeros((V, V))
    for a in tr_ids:
        for x, y in zip(a[:-1], a[1:]): c[x, y] += 1.0
    best = None
    for k in (0.3, 0.1, 0.03, 0.01, 0.003):
        Pk = (c + k) / (c + k).sum(1, keepdims=True); e = 0.0; n = 0
        for a in ev_ids:
            for x, y in zip(a[:-1], a[1:]): e += -math.log(max(float(Pk[x, y]), 1e-12)); n += 1
        ce = e / max(n, 1); best = ce if best is None else min(best, ce)
    return best


def interp_trigram_ce(tr_ids, ev_ids, V):
    """Deleted-interpolation trigram: P(w3|w1,w2) = l3*tri + l2*bi + l1*uni, lambdas tuned on a held-out slice of tr."""
    from collections import defaultdict
    uni = np.zeros(V); bi = defaultdict(lambda: np.zeros(V)); tri = defaultdict(lambda: np.zeros(V))
    ntok = 0
    for a in tr_ids:
        for i in range(len(a)):
            uni[a[i]] += 1; ntok += 1
            if i >= 1: bi[a[i - 1]][a[i]] += 1
            if i >= 2: tri[(a[i - 2], a[i - 1])][a[i]] += 1
    uni_p = (uni + 0.01) / (uni + 0.01).sum()

    def bi_p(w1):
        r = bi.get(w1)
        if r is None: return uni_p
        s = r.sum()
        return (r / s) if s > 0 else uni_p

    def tri_p(w1, w2):
        r = tri.get((w1, w2))
        if r is None: return None
        s = r.sum()
        return (r / s) if s > 0 else None

    def mixed(w1, w2, l1, l2, l3):
        t = tri_p(w1, w2); b = bi_p(w2)
        p = l1 * uni_p + l2 * b
        if t is not None: p = p + l3 * t
        else: p = p / (l1 + l2)  # renorm when no trigram evidence
        return p / p.sum()

    # tune lambdas on a small held-out slice of tr (deleted interpolation, coarse grid)
    dev = tr_ids[-max(1, len(tr_ids) // 10):]
    best = None; best_l = (0.1, 0.3, 0.6)
    for l3 in (0.3, 0.5, 0.7, 0.85):
        for l2 in (0.1, 0.2, 0.3):
            l1 = max(0.02, 1.0 - l3 - l2)
            e = 0.0; n = 0
            for a in dev:
                for i in range(2, len(a)):
                    p = mixed(a[i - 2], a[i - 1], l1, l2, l3); e += -math.log(max(float(p[a[i]]), 1e-12)); n += 1
            ce = e / max(n, 1)
            if best is None or ce < best: best = ce; best_l = (l1, l2, l3)
    l1, l2, l3 = best_l

    def _bucket(depth):   # depth = 1-indexed position of the predicted token (matches the reslm runner's buckets)
        return "1" if depth == 1 else "2" if depth == 2 else "3" if depth == 3 else \
               "4-5" if depth <= 5 else "6-9" if depth <= 9 else "10-99"
    e = 0.0; n = 0
    bde = defaultdict(float); bdn = defaultdict(int)
    for a in ev_ids:
        if len(a) >= 2:
            c = -math.log(max(float(bi_p(a[0])[a[1]]), 1e-12)); e += c; n += 1
            bde["1"] += c; bdn["1"] += 1     # position 0->1 (predict token at depth 1) scored by bigram
        for i in range(2, len(a)):
            p = mixed(a[i - 2], a[i - 1], l1, l2, l3); c = -math.log(max(float(p[a[i]]), 1e-12)); e += c; n += 1
            k = _bucket(i)                    # predicting token at index i = depth i (1-indexed within the sentence)
            bde[k] += c; bdn[k] += 1
    by_depth = {k: round(bde[k] / bdn[k], 4) for k in bdn if bdn[k]}
    return e / max(n, 1), best_l, by_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-sentences", type=int, default=60000)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    tr_ids, ev_ids, V = _split(a.corpus, a.n_sentences, a.n_eval, a.n_train, a.vocab, a.seed)
    tb = tuned_bigram_ce(tr_ids, ev_ids, V)
    tri, lam, tri_bd = interp_trigram_ce(tr_ids, ev_ids, V)
    # find the matching selective result (+ its by-depth CE)
    sel = None; sel_bd = None
    for f in glob.glob(f"raw/_fluency_np*_nt{a.n_train}_s{a.seed}*.json"):
        try:
            d = json.load(open(f)); sel = d.get("sel_ce"); sel_bd = (d.get("by_depth") or {}).get("sel")
        except Exception: pass
    print(f"[fair-ngram] V={V} nt={a.n_train} s={a.seed} | tuned_bigram={tb:.3f} interp_trigram={tri:.3f} (lam={tuple(round(x,2) for x in lam)})"
          + (f" | sel_ce={sel:.3f} sel_over_tuned_bi={tb - sel:+.3f} sel_over_trigram={tri - sel:+.3f} -> "
             + ("CROSSES BOTH (genuine)" if (sel < tb - 0.02 and sel < tri - 0.02) else
                "beats bigram NOT trigram (not higher-order)" if sel < tb - 0.02 else "below both") if sel is not None else " | (no selective json yet)"), flush=True)
    # THE by-depth deep-position question: does the reservoir-LM beat the FAIR TRIGRAM at d>=10, where the order-2
    # trigram has no long-range info? (trigram by-depth reveals if it degrades at depth or is ~position-independent)
    print(f"[fair-ngram BY-DEPTH] trigram: {tri_bd}", flush=True)
    if sel_bd:
        deep = [k for k in ("6-9", "10-99") if k in tri_bd and k in sel_bd]
        for k in deep:
            marg = round(tri_bd[k] - sel_bd[k], 4)
            print(f"  d={k}: sel={sel_bd[k]:.3f} trigram={tri_bd[k]:.3f} sel_over_trigram={marg:+.4f} "
                  f"-> {'SEL BEATS trigram at depth (real long-range!)' if marg > 0.02 else 'trigram wins/ties (n-gram-bound even at depth)'}", flush=True)


if __name__ == "__main__":
    main()
