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
    Vocab, train_readout, eval_ce, fit_bigram, bigram_ce, _standardize_fit, _bag_cache, ACTIVE_MIN,
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
    # FIXED eval set + FIXED vocab across the whole n_train sweep (the verify-Workflow scope-fix: a sliding eval window /
    # per-point vocab would measure the margin-vs-n_train curve on DRIFTING data). ev = a fixed held-out slice; the train
    # POOL is disjoint; the vocab is built ONCE from the whole pool so token-ids are constant; tr = a growing prefix.
    perm = np.random.default_rng(a.seed).permutation(len(sents))
    ev = [sents[i] for i in perm[-a.n_eval:]]
    pool = [sents[i] for i in perm[:-a.n_eval]]
    vocab = Vocab.build(pool, V=a.vocab)                       # FIXED vocab (whole pool -> invariant to n_train)
    V = vocab.size
    in_dim = len(vocab.encode_seq(pool[0])[0])
    tr = pool[:a.n_train]                                      # growing PREFIX of the fixed pool
    tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]

    b, copy_res, W_in, snap = BR.build_batched(a.seed, a.n_pool, in_dim, a.batch_m)
    tc = time.time()
    tr_cache = _cache_batched(b, copy_res, W_in, snap, vocab, tr, a.batch_m)
    ev_cache = _cache_batched(b, copy_res, W_in, snap, vocab, ev, a.batch_m)
    collect_s = time.time() - tc
    # reservoir ACTIVITY (rule out a degenerate/near-silent pool faking a read-out-vs-baseline effect): the states are
    # running-cumulative pool rates -> the last-token state's mean ~ mean spike-rate/neuron/step.
    mean_rate = float(np.mean([st[-1].mean() for st, _ in tr_cache if len(st)])) if tr_cache else 0.0

    def _fit_eval(tr_c, ev_c, salt):
        m, s = _standardize_fit(tr_c)
        W = train_readout(tr_c, V, a.epochs, a.lr, np.random.default_rng(a.seed * 13 + salt), m, s, wd=a.weight_decay)
        ce, acc, _ = eval_ce(W, m, s, ev_c, V)
        return ce, acc

    res_ce, res_acc = _fit_eval(tr_cache, ev_cache, 1)
    # THE KEY CONTROL (verify-Workflow PRIMARY): the memoryless BAG-OF-PREFIX read-out over the SAME positions. If the
    # reservoir's recurrent DYNAMICS are load-bearing, res_ce must beat BAG_ce -- and that margin must GROW with data.
    bag_ce, bag_acc = _fit_eval(_bag_cache(tr_cache, V), _bag_cache(ev_cache, V), 7)
    P_bi = fit_bigram(tr_ids, V); bi_ce, bi_acc, _ = bigram_ce(P_bi, ev_ids)

    margin_bag = bag_ce - res_ce                              # HEADLINE: dynamics load-bearing iff >0 AND grows with data
    margin_bi = bi_ce - res_ce                               # the weak comparator (a bag can fake a growing margin here)
    res = dict(n_pool=a.n_pool, n_train=len(tr), n_eval=len(ev), V=V, batch_m=a.batch_m,
               res_ce=round(res_ce, 4), bag_ce=round(bag_ce, 4), bi_ce=round(bi_ce, 4),
               margin_over_bag=round(margin_bag, 4), margin_over_bigram=round(margin_bi, 4),
               res_acc=round(res_acc, 4), bag_acc=round(bag_acc, 4), bi_acc=round(bi_acc, 4),
               mean_spike_rate=round(mean_rate, 6), active=bool(mean_rate > ACTIVE_MIN),
               chance=round(math.log(V), 4), collect_s=round(collect_s, 1), total_s=round(time.time() - t0, 1))
    json.dump(res, open(a.out, "w"))
    print(f"[batched-scale] np={a.n_pool} nt={len(tr)} V={V} active={res['active']}(rate={mean_rate:.4f}): "
          f"margin_over_BAG={margin_bag:+.4f} (res={res_ce:.3f} bag={bag_ce:.3f}) | over_bigram={margin_bi:+.4f} "
          f"-> {'dynamics load-bearing' if margin_bag > 0 else 'BAG matches/beats reservoir (dynamics NOT load-bearing)'}")


if __name__ == "__main__":
    main()
