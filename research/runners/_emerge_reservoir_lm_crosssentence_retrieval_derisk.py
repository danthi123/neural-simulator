"""CROSS-SENTENCE content-addressable retrieval de-risk (the GENUINE long-range test; corrects a caught confound).

WHY THIS RUNNER. `2026-07-11-learned-keys-plus-interpolation-BEATS-base-at-deep-context-*.md` (read its CORRECTION banner)
found a content-addressable retrieval interpolated with the base read-out BEAT base at deep context -- but it was a
CONFOUND: the retrieval was INTRA-SENTENCE over <=16-token sentences, so the "deep gain" was a within-sentence-token CACHE
(unigram-recency / Merity-2016 neural-cache prior), which is SHUFFLE-INVARIANT (a random-key retrieval gives the same bag).
The genuine content-addressing test must be CROSS-SENTENCE: retrieve over a persistent store of PRIOR SENTENCES' content
within the same contiguous passage. A RANDOM cross-corpus/cross-context bag should NOT help (too diffuse), but a
content-addressed retrieval of the RELEVANT prior context SHOULD -> the load-bearing signal is `content > shuffle` at
CROSS-SENTENCE positions (NOT `content > base`; a within-passage bag can beat base without any content-addressing, which is
exactly the confound just corrected).

MECHANISM (per token t in sentence s of a contiguous PASSAGE, predicting token t+1):
  keys    K_tau = reservoir state h_tau at a PAST token tau, from EARLIER SENTENCES (0..s-1) of the SAME passage.
  value   v_tau = onehot(ids[tau+1])  (the token that FOLLOWED the retrieved past context; kNN-LM read).
  query   q     = h_t (the current context).
  attn    a_tau = softmax(beta * <q,K_tau> / sqrt(n)).
  r_t     = sum_tau a_tau v_tau        (a soft cross-sentence next-token recall).
  p_final = (1-lam) * p_base + lam * r_t   (CLS cortex(base) + hippocampal-retrieval(r_t) interpolation, kNN-LM/Khandelwal).
The store holds ONLY tokens from COMPLETED PRIOR SENTENCES of the current passage -> no within-sentence cache, no future
leak. Reservoir state is washed BETWEEN passages (h=0 at each passage), carried WITHIN a passage; the STORE is what holds
the cross-sentence memory.

ARMS (single variable = the retrieval; all share the SAME base read-out + reservoir):
  base     : no retrieval (lam=0)                                     -- the fading-state baseline.
  content  : cross-sentence content-addressed retrieval (the mechanism under test).
  shuffle  : SAME store, KEYS PERMUTED vs their values (query matches a RANDOM prior context) -> a random cross-sentence
             bag. THE LOAD-BEARING ANTI-CHEAT: content must BEAT shuffle at cross-sentence positions, else it is a bag
             confound again (the exact defect this runner corrects).
  uniform  : beta=0 -> uniform attention over the store = the average prior-sentence next-token (a cross-sentence bag,
             no content-addressing). A second bag control.
HEADLINE METRIC = content-minus-shuffle CE at CROSS-SENTENCE positions (NOT content-minus-base). Also reported:
content-minus-base (the confoundable number) so both are visible. Each arm gets its OWN best single-global-lam (min pooled
cross CE) so the anti-cheat is not handicapped. CE broken down by within-passage token depth (deeper == more prior
sentences == longer-range).

HONESTY. If content ~ shuffle at cross-sentence positions (a bag again) OR content does not beat base, that is a VALID
honest negative -- reported, not faked. Numpy rate reservoir (the RateReservoir + e-prop from the eprop runner); WikiText
(real cross-sentence discourse); NO `sim/` edit, NO BPTT. Multi-seed + the load-bearing shuffle control are the
CONTROLLER's job; this runner + its cheap 1-seed smoke build + exercise the harness.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, math, re, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import (
    Vocab, _softmax, train_readout, _standardize_fit)
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket
from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir, train as eprop_train

OUT = Path("research/findings/raw/_reslm_crosssentence.json")
_WORD = re.compile(r"[a-z']+")
LAMS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]


# ---------------------------------------------------------------------------------------------------------------------
# CONTIGUOUS-DOCUMENT loader: PRESERVE order, group consecutive sentences into passages (NO within-passage shuffle, NO
# length-filter of sentences inside a passage -> the token stream stays contiguous with real cross-sentence discourse).
# ---------------------------------------------------------------------------------------------------------------------
def load_passages(path, n_passages, sents_per_passage, min_sent_len=2):
    """Read the corpus preserving order; split into sentences on [.!?] (kept IN ORDER); group consecutive sentences into
    passages of `sents_per_passage`. A sentence with < min_sent_len word-tokens (empty/punctuation-only fragment) is
    dropped, but LONG sentences are KEPT (no 3-16 length filter) so the contiguous stream is preserved."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore").lower()
    need = n_passages * sents_per_passage
    sents = []
    for raw in re.split(r"[.!?]", txt):
        w = _WORD.findall(raw)
        if len(w) >= min_sent_len:
            sents.append(w)
            if len(sents) >= need:
                break
    passages = []
    for i in range(0, len(sents) - sents_per_passage + 1, sents_per_passage):
        passages.append(sents[i:i + sents_per_passage])
        if len(passages) >= n_passages:
            break
    return passages


def passage_to_ids(passage, vocab):
    """Concatenate a passage's sentences into one contiguous id stream + a per-token sentence index."""
    ids = []
    sent_of = []
    for si, s in enumerate(passage):
        for w in s:
            ids.append(vocab.id(w))
            sent_of.append(si)
    return np.asarray(ids), np.asarray(sent_of)


def build_cache(res, passages, vocab):
    """Reservoir states over each passage (contiguous stream; h washed to 0 at each passage boundary by forward_states)."""
    out = []
    for p in passages:
        ids, _ = passage_to_ids(p, vocab)
        if len(ids) < 2:
            continue
        S = res.forward_states(ids.tolist())
        out.append((S, ids))
    return out


def retrieve(K, valids, q, V, beta, inv, arm, rng):
    """Cross-sentence content-addressable retrieval distribution r_t (dim V) from the store (K: keys, valids: next-token
    ids). arm='content' = attend by <q,key>; 'shuffle' = permute keys vs values (random prior context) = bag anti-cheat;
    'uniform' = beta=0 uniform attention = the average prior-sentence next-token (bag)."""
    M = K.shape[0]
    if arm == "uniform":
        a = np.full(M, 1.0 / M)
    else:
        Kk = K[rng.permutation(M)] if arm == "shuffle" else K
        a = _softmax(beta * (Kk @ q) * inv)
    r = np.zeros(V)
    np.add.at(r, valids, a)                                    # r_t = sum_tau a_tau onehot(v_tau); sums to 1 (a sums to 1)
    return r


def eval_passages(res, W, mean, std, ev_pass, vocab, V, beta, arms, seed):
    """For each eval passage: stream sentence-by-sentence; a token in sentence s is scored against the store of PRIOR
    sentences 0..s-1 (appended only after a sentence completes -> never the current/within sentence, no future leak).
    Accumulate CE per arm per lam per within-passage token-depth bucket, over CROSS-SENTENCE positions only (store>0)."""
    rng = np.random.default_rng(seed * 13 + 5)
    n = res.n
    inv = 1.0 / math.sqrt(n)
    ret_arms = [a for a in arms if a != "base"]
    agg = {arm: {la: defaultdict(float) for la in LAMS} for arm in ret_arms}
    agg_all = {arm: {la: 0.0 for la in LAMS} for arm in ret_arms}
    base_ce = defaultdict(float)
    cnt = defaultdict(int)
    base_all = 0.0
    cnt_all = 0
    n_first_sent = 0                                           # positions with no prior-sentence store (arms == base)
    for passage in ev_pass:
        ids, sent_of = passage_to_ids(passage, vocab)
        L = len(ids)
        if L < 2:
            continue
        S = np.asarray(res.forward_states(ids.tolist()))       # (L, n)
        n_sent = len(passage)
        pos_by_sent = [[] for _ in range(n_sent)]
        for t in range(L):
            pos_by_sent[int(sent_of[t])].append(t)
        store_K = []
        store_val = []
        for si in range(n_sent):
            for t in pos_by_sent[si]:
                if t >= L - 1:
                    continue
                tgt = int(ids[t + 1])
                x = np.concatenate([(S[t] - mean) / std, [1.0]])
                pb = _softmax(W @ x)
                if not store_K:                                # first sentence -> no prior context; arms==base -> skip
                    n_first_sent += 1
                    continue
                b = _bucket(t + 1)                             # within-passage token depth (deeper == more prior sents)
                ceb = -math.log(max(pb[tgt], 1e-12))
                base_ce[b] += ceb
                cnt[b] += 1
                base_all += ceb
                cnt_all += 1
                K = np.asarray(store_K)
                valids = np.asarray(store_val)
                q = S[t]
                for arm in ret_arms:
                    r = retrieve(K, valids, q, V, beta, inv, arm, rng)
                    for la in LAMS:
                        pf = (1 - la) * pb + la * r
                        ce = -math.log(max(pf[tgt], 1e-12))
                        agg[arm][la][b] += ce
                        agg_all[arm][la] += ce
            for t in pos_by_sent[si]:                          # after the sentence completes -> add to the store
                if t + 1 < L:
                    store_K.append(S[t])
                    store_val.append(int(ids[t + 1]))
    # each arm's OWN best single-global-lam (min pooled cross-sentence CE) so the anti-cheat is not handicapped.
    out = {"n_cross_positions": cnt_all, "n_first_sentence_positions": n_first_sent,
           "base_cross_ce": round(base_all / max(1, cnt_all), 4),
           "base_by_depth": {b: round(base_ce[b] / cnt[b], 4) for b in cnt}, "by_arm": {}}
    for arm in ret_arms:
        best_la = min(LAMS, key=lambda la: agg_all[arm][la])
        out["by_arm"][arm] = {
            "best_lam": best_la,
            "cross_ce": round(agg_all[arm][best_la] / max(1, cnt_all), 4),
            "by_depth": {b: round(agg[arm][best_la][b] / cnt[b], 4) for b in cnt},
            "cross_ce_by_lam": {str(la): round(agg_all[arm][la] / max(1, cnt_all), 4) for la in LAMS},
        }
    return out


def _derisk_one(seed, args):
    passages = load_passages(args.corpus, args.passages, args.sents_per_passage)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(passages))
    cut = int(0.8 * len(passages))
    tr_pass = [passages[i] for i in idx[:cut]]
    ev_pass = [passages[i] for i in idx[cut:]]
    tr_sents = [s for p in tr_pass for s in p]                 # Vocab from TRAIN passages only
    vocab = Vocab.build(tr_sents, V=args.vocab)
    V = vocab.size

    res = RateReservoir(V, args.n_pool, seed, alpha=0.3, spectral=1.1)
    if args.learned_keys:                                      # e-prop-LEARN the recurrent weights (the learned-key test)
        tr_ids = [passage_to_ids(p, vocab)[0].tolist() for p in tr_pass]
        eprop_train(res, tr_ids, V, args.eprop_epochs, args.lr, args.lr_rec, seed, mode="plastic")

    tr_cache = build_cache(res, tr_pass, vocab)
    mean, std = _standardize_fit(tr_cache)
    W = train_readout(tr_cache, V, args.epochs, args.lr, np.random.default_rng(seed * 13 + 1), mean, std,
                      wd=args.weight_decay, ls=0.05)

    ev = eval_passages(res, W, mean, std, ev_pass, vocab, V, args.beta, args.arms, seed)

    # headline signals (positive = content BETTER = lower CE)
    ba = ev["by_arm"]
    base_cross = ev["base_cross_ce"]
    content = ba.get("content", {})
    shuffle = ba.get("shuffle", {})
    content_minus_base = round(base_cross - content.get("cross_ce", base_cross), 4) if content else None
    content_minus_shuffle = (round(shuffle.get("cross_ce", 0.0) - content.get("cross_ce", 0.0), 4)
                             if (content and shuffle) else None)
    # per-depth content-minus-shuffle (the load-bearing signal broken out by depth)
    cms_by_depth = {}
    if content and shuffle:
        for b in content["by_depth"]:
            if b in shuffle["by_depth"]:
                cms_by_depth[b] = round(shuffle["by_depth"][b] - content["by_depth"][b], 4)
    return {
        "seed": seed, "V": V, "n_pool": args.n_pool, "n_train_pass": len(tr_pass), "n_eval_pass": len(ev_pass),
        "sents_per_passage": args.sents_per_passage, "learned_keys": bool(args.learned_keys),
        "eval": ev,
        "content_minus_base_cross": content_minus_base,          # confoundable number (a bag can win this)
        "content_minus_shuffle_cross": content_minus_shuffle,    # THE REAL SIGNAL (positive = content > bag)
        "content_minus_shuffle_by_depth": cms_by_depth,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/wikitext.txt")
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--passages", type=int, default=400, help="number of contiguous passages (train/eval split at passage level)")
    ap.add_argument("--sents-per-passage", type=int, default=10, help="consecutive sentences per passage (the cross-sentence horizon)")
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=8, help="base read-out delta-rule epochs")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--beta", type=float, default=4.0, help="retrieval temperature")
    ap.add_argument("--learned-keys", action="store_true", help="e-prop-train the reservoir recurrent weights (LEARNED keys)")
    ap.add_argument("--eprop-epochs", type=int, default=6)
    ap.add_argument("--lr-rec", type=float, default=0.006, help="e-prop recurrent lr (when --learned-keys)")
    ap.add_argument("--arms", type=str, nargs="+", default=["base", "content", "shuffle", "uniform"])
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    per_seed = {}
    for seed in args.seeds:
        d = _derisk_one(seed, args)
        per_seed[str(seed)] = d
        ba = d["eval"]["by_arm"]
        print(f"[seed {seed}] V={d['V']} n_pool={d['n_pool']} passages(tr/ev)={d['n_train_pass']}/{d['n_eval_pass']} "
              f"sents/pass={d['sents_per_passage']} learned_keys={d['learned_keys']} | "
              f"cross positions n={d['eval']['n_cross_positions']} (first-sentence skipped {d['eval']['n_first_sentence_positions']})",
              flush=True)
        print(f"    base cross-CE {d['eval']['base_cross_ce']:.4f}"
              + "".join(f" | {a} {ba[a]['cross_ce']:.4f}(λ{ba[a]['best_lam']})" for a in ba), flush=True)
        print(f"    content-minus-base    (confoundable, +=content better) {d['content_minus_base_cross']:+.4f}", flush=True)
        print(f"    content-minus-SHUFFLE (THE REAL SIGNAL, +=content>bag)  {d['content_minus_shuffle_cross']:+.4f}", flush=True)
        row = " ".join(f"d{k}:{d['content_minus_shuffle_by_depth'][k]:+.4f}"
                       for lo, hi in BUCKETS for k in [f'{lo}-{hi}' if lo != hi else f'{lo}']
                       if k in d['content_minus_shuffle_by_depth'])
        print(f"    content-minus-shuffle by within-passage depth: {row}", flush=True)

    out = {"runner": "_emerge_reservoir_lm_crosssentence_retrieval_derisk", "corpus": args.corpus,
           "seeds": args.seeds, "beta": args.beta, "arms": args.arms, "lams": LAMS,
           "headline": "content_minus_shuffle_cross > 0 (content beats a random cross-sentence bag) = the genuine "
                       "cross-sentence content-addressing signal; content_minus_base alone is confoundable by a bag",
           "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
