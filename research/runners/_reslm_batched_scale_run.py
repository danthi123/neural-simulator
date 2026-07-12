"""THE DECISIVE SCALE RUN (batched, affordable): does the emergent reservoir generator's CE margin over the bigram
GROW with more DATA, or stay bounded? The 2026-07-11 co-scale probe went NEGATIVE at (n_pool=800, 2400 sents) —
data-starved. This runs the SAME machinery but collects reservoir states via the 67×-faster batched collection
(`_reslm_batched_reservoir_derisk`), so we can push the data well past that point and get the answer.

The batched reservoir is a CONSISTENT (copy-identical) reservoir that differs slightly (~0.08) from the shipped serial
one — irrelevant to the SCALE QUESTION, since res_ce and the bigram are compared AT EACH data scale (a self-consistent
trend). Reuse-by-import: `Vocab`, `train_readout`, `eval_ce`, `fit_bigram`/`bigram_ce`, `load_sentences`; NO `sim/` edit.

Run: E:/.../python.exe -m research.runners._reslm_batched_scale_run --n-pool 300 --n-train 2800 --seed 42
"""
import argparse, time, json, math
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import (
    Vocab, train_readout, eval_ce, fit_bigram, bigram_ce, _standardize_fit,
)
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
import research.runners._reslm_batched_reservoir_derisk as BR


def _cache_batched(b, copy_res, W_in, snap, vocab, sents, M):
    """Same output as the shipped `_cache` (list of (per-token states, token-id list)), collected in M-sentence batches
    through the block-diagonal batched reservoir."""
    out = []
    for i in range(0, len(sents), M):
        chunk = sents[i:i + M]
        U_list = [vocab.encode_seq(s) for s in chunk]
        pad = M - len(U_list)
        if pad:                                              # pad the last partial chunk with 1-token dummies (dropped)
            U_list = U_list + [[np.zeros(W_in.shape[1])] for _ in range(pad)]
        S = BR.per_token_states_batch(b, copy_res, W_in, snap, U_list)
        for c, s in enumerate(chunk):
            out.append((np.asarray(S[c]), vocab.ids(s)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-sentences", type=int, default=20000)
    ap.add_argument("--n-train", type=int, default=1400)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--batch-m", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="raw/_reslm_batched_scale.json")
    a = ap.parse_args()

    t0 = time.time()
    sents = load_sentences(a.corpus, a.n_sentences)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(sents))
    cut = min(a.n_train, len(sents) - a.n_eval)
    tr = [sents[i] for i in idx[:cut]]
    ev = [sents[i] for i in idx[cut:cut + a.n_eval]]
    vocab = Vocab.build(tr, V=a.vocab)
    V = vocab.size
    in_dim = len(vocab.encode_seq(tr[0])[0])
    tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]

    b, copy_res, W_in, snap = BR.build_batched(a.seed, a.n_pool, in_dim, a.batch_m)
    tc = time.time()
    tr_cache = _cache_batched(b, copy_res, W_in, snap, vocab, tr, a.batch_m)
    ev_cache = _cache_batched(b, copy_res, W_in, snap, vocab, ev, a.batch_m)
    collect_s = time.time() - tc

    mean, std = _standardize_fit(tr_cache)
    W = train_readout(tr_cache, V, a.epochs, a.lr, np.random.default_rng(a.seed * 13 + 1), mean, std, wd=a.weight_decay)
    res_ce, res_acc, _n = eval_ce(W, mean, std, ev_cache, V)
    P_bi = fit_bigram(tr_ids, V); bi_ce, bi_acc, _ = bigram_ce(P_bi, ev_ids)
    margin = bi_ce - res_ce                                   # >0 => the generator beats the bigram (in nats)

    res = dict(n_pool=a.n_pool, n_train=len(tr), n_eval=len(ev), V=V, batch_m=a.batch_m,
               res_ce=round(res_ce, 4), bi_ce=round(bi_ce, 4), margin=round(margin, 4),
               res_acc=round(res_acc, 4), bi_acc=round(bi_acc, 4), chance=round(math.log(V), 4),
               collect_s=round(collect_s, 1), total_s=round(time.time() - t0, 1))
    json.dump(res, open(a.out, "w"))
    print(f"[batched-scale] n_pool={a.n_pool} n_train={len(tr)} V={V}: "
          f"margin(bi-res)={margin:+.4f} nats (res_ce={res_ce:.3f} bi_ce={bi_ce:.3f}) "
          f"collect={collect_s:.0f}s -> {'BEATS bigram' if margin > 0 else 'loses to bigram'}")


if __name__ == "__main__":
    main()
