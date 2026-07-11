"""LEARNED-ATTENTION cross-sentence de-risk -- the CEILING test that isolates the reservoir-substrate from the open-text
long-range question. Read `2026-07-11-ALIF-adaptation-state-NEGATIVE-...-arc-synthesis-...md` (the SYNTHESIS) +
`2026-07-11-cross-sentence-content-addressable-retrieval-NEGATIVE-...md` first.

WHY THIS RUNNER (the un-tested isolation). The whole 2026-07-11 arc showed every RESERVOIR-substrate long-range mechanism
is NEGATIVE on open text: a fading/adaptive recurrent STATE dilutes distal items, and a content-addressable retrieval over
RESERVOIR-STATE keys retrieves a random cross-sentence BAG (content ~= shuffle ~= uniform; the uniform bag is best). The
synthesis pinned the binding limit on the KEYS: reservoir states don't encode distal linguistic structure finely enough
for content-addressing to beat a bag. That leaves ONE question the arc never isolated: is there ANY exploitable open-text
cross-sentence long-range signal that a *LEARNED* attention (learned query/key PROJECTIONS over token-identity embeddings,
not reservoir states) can capture -- or is open-text cross-sentence long-range genuinely THIN at this scale?

  * If even a FULL-GRADIENT learned attention over the same cross-sentence store does NOT beat the bag/shuffle at
    long-range -> the reservoir substrate is not the only problem: open-text cross-sentence long-range is thin here (a
    clean SCALE/DATA verdict that bounds the whole frontier).
  * If it DOES beat the bag -> there IS signal a learned attention finds that reservoir keys could not, and the frontier
    is precisely "make the attention's learning LOCAL/biological" (the CEILING-vs-LOCAL gap is the residual to close).

THE MECHANISM (a single LEARNABLE attention head over the SAME cross-sentence store as the reservoir runner; reuse
`load_passages` / `passage_to_ids` / the buckets). Per token t in sentence s of a contiguous PASSAGE, predicting t+1:
  store  = tokens tau of PRIOR sentences 0..s-1 of the same passage (never within-sentence; no future leak) -- identical
           store semantics to the cross-sentence reservoir runner, so this is a like-for-like KEY-QUALITY swap.
  key    raw material = a FIXED-random token EMBEDDING context of the stored token: c(tau) = concat(E[x_tau], E[x_{tau-1}])
           (a `win`-token identity context; E is fixed-random d~32 per vocab id). NO reservoir state anywhere.
  value  v_tau = onehot(x_{tau+1})  (the token that FOLLOWED -- the kNN-LM / induction-head "copy the continuation" read).
  query  q_t  = Wq @ c(t)   (Wq LEARNED);  key k_tau = Wk @ c(tau)   (Wk LEARNED).   [so a k-gram context MATCH -> retrieve
           the continuation is LEARNABLE: the canonical induction-head long-range mechanism, Olsson-2022.]
  attn   a_tau = softmax((Wk c(tau)) . (Wq c(t)) / sqrt(d_head)).
  r_t    = sum_tau a_tau v_tau   (a soft cross-sentence next-token recall, dim V).
  p_final= (1-lam) p_base + lam r_t   (CLS cortex(base) + retrieval interpolation; base = bigram, the cheap fair comparator).

TRAINING Wq, Wk (the two arms -- single variable = HOW the projections learn; store/embeddings/eval identical):
  * CEILING (full gradient) -- the load-bearing "is there signal" test. Train Wq, Wk by the EXACT gradient of the
    retrieval next-token CE (-log p_ret[target], p_ret = (1-eps)r_t + eps/V) over train cross-sentence positions, batched
    per passage. A rate-level CEILING; biological plausibility NOT required for THIS arm -- it answers "is there ANY
    signal a learned attention can get". (The exact softmax-CE gradient through the attention; NO deep net -- the base is
    a fixed bigram, so the only learned thing is the single attention head.)
  * LOCAL (feedback-alignment / three-factor, NO BPTT) -- the biological version. Same forward, but Wq/Wk are updated by
    a LOCAL rule on the CLEAN read error: per store slot the learning signal L_tau = (onehot(target) - r_t) . v_tau =
    (x_{tau+1}==target) - r_t[x_{tau+1}] (the slot's value dotted with the observable read error -- a broadcast
    three-factor signal, NO exact -target/p division, NO softmax Jacobian transport), baseline-subtracted by the retrieved
    mean (the softmax structure), driving Wq/Wk through the attention weights as eligibility. This is delta-rule /
    feedback-alignment attention learning.

ANTI-CHEATS (FOREGROUND -- bag/cache confounds were caught repeatedly this session; headline is content vs the BAG, NOT
content vs base). For EACH trained arm, at eval the SAME trained Wq/Wk are read three ways over the store, each with its
OWN best single-global-lam (min pooled cross CE, so the anti-cheat is NOT handicapped):
  content : attend by the learned (Wk c(tau)) . (Wq c(t))  -- the mechanism under test.
  shuffle : SAME scores but the store KEYS are PERMUTED vs their values -> a learned query now matches a RANDOM prior
            context = a random cross-sentence bag. THE LOAD-BEARING ANTI-CHEAT.
  uniform : beta=0 -> uniform attention = the average prior-sentence next-token (a second cross-sentence bag).
HEADLINE = content-minus-SHUFFLE and content-minus-UNIFORM CE at CROSS-SENTENCE positions (NOT content-minus-base, which
any bag inflates), broken out by within-passage token depth (deeper == more prior sentences == longer range). A GO =
learned-attention content >> shuffle AND >> uniform at long range.

HONESTY (the decision this de-risk makes):
  * If even the full-gradient CEILING does NOT beat the bag at long range -> open-text cross-sentence long-range is THIN at
    this scale (a clean scale/data honest negative that bounds the whole frontier -- REPORT it, do NOT fake a positive).
  * If the CEILING beats the bag but the LOCAL rule does not -> the frontier is precisely "local-rule attention learning"
    (report the ceiling-vs-local gap).
  * If both beat the bag -> a learned attention finds long-range signal reservoir keys could not, AND it is learnable
    locally.
Numpy rate-level; reuse-by-import (the cross-sentence loader/store/buckets); WikiText (real cross-sentence discourse); NO
`sim/` edit, NO BPTT (the CEILING is a single-head gradient, not through-time). Multi-seed + sweeps are the CONTROLLER's
job; this runner + its cheap 1-seed smoke build + exercise the harness.
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

from research.runners._emerge_reservoir_lm_derisk import Vocab, _softmax, fit_bigram
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket
from research.runners._emerge_reservoir_lm_crosssentence_retrieval_derisk import load_passages, passage_to_ids

OUT = Path("research/findings/raw/_reslm_learned_attention.json")
LAMS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]
ARMS = ["content", "shuffle", "uniform"]
MARGIN = 0.03   # content must beat the bag CE by at least this many nats to count as a real signal


# ---------------------------------------------------------------------------------------------------------------------
# Per-passage store + positions. Store = tokens of PRIOR (completed) sentences of the passage; key/query material = a
# `win`-token FIXED-random embedding context (identity only, NO reservoir). Value = onehot(next token).
# ---------------------------------------------------------------------------------------------------------------------
def _ctx_matrix(ids, E, win):
    """ctx[t] = concat(E[ids[t]], E[ids[t-1]], ..., E[ids[t-win+1]]) with zero-pad at the passage start. (L, win*d)."""
    L = len(ids)
    d = E.shape[1]
    C = np.zeros((L, win * d))
    for t in range(L):
        for k in range(win):
            j = t - k
            if j >= 0:
                C[t, k * d:(k + 1) * d] = E[ids[j]]
    return C


def build_passage(passage, vocab, E, win):
    """Return (key_ctx, val_ids, positions) for one contiguous passage.
      key_ctx : (n_store, win*d) context vectors for every STORABLE token tau (0..L-2, i.e. tau+1 exists), in stream order.
      val_ids : (n_store,)      value = ids[tau+1] for each storable tau.
      positions: list of (q_ctx (win*d,), target_id, M, depth, cur_id) for each CROSS-SENTENCE position t (sentence>=1
                 with a non-empty prior-sentence store). M = store slots available (all storable tau in sentences < s);
                 the store adds a sentence's tokens only AFTER it completes -> never within/future. cur_id = the current
                 token id (for the bigram base row P(next | current)). depth = t+1 (tokens seen before the prediction).
    """
    ids, sent_of = passage_to_ids(passage, vocab)
    L = len(ids)
    if L < 3:
        return np.zeros((0, win * E.shape[1])), np.zeros(0, int), []
    C = _ctx_matrix(ids, E, win)
    storable = list(range(L - 1))                              # tau in 0..L-2 (tau+1 exists)
    key_ctx = C[storable]                                      # (n_store, win*d), stream order
    val_ids = np.asarray([ids[t + 1] for t in storable], int)
    st_sent = np.asarray([int(sent_of[t]) for t in storable])  # sentence of each storable token (non-decreasing)
    n_sent = len(passage)
    # slots_before[s] = # storable tokens with sentence < s (= store size available to a token in sentence s)
    slots_before = np.zeros(n_sent + 1, int)
    for s in range(1, n_sent + 1):
        slots_before[s] = int(np.sum(st_sent < s))
    positions = []
    for t in range(L - 1):                                    # predict ids[t+1]
        s = int(sent_of[t])
        if s < 1:
            continue
        M = int(slots_before[s])
        if M <= 0:
            continue
        positions.append((C[t], int(ids[t + 1]), M, t + 1, int(ids[t])))
    return key_ctx, val_ids, positions


# ---------------------------------------------------------------------------------------------------------------------
# The learnable single-head attention: Wq, Wk (d_head x win*d). Train by the CEILING (exact CE gradient) or the LOCAL
# (feedback-alignment three-factor) rule. Batched per passage (project the passage's keys once per epoch).
# ---------------------------------------------------------------------------------------------------------------------
def _init_proj(rng, d_head, in_dim):
    return rng.standard_normal((d_head, in_dim)) / math.sqrt(in_dim)


def train_attention(pdata, V, d_head, rule, epochs, lr, eps, clip, seed):
    """Train Wq, Wk over the train passages' cross-sentence positions. `rule` in {ceiling, local}.
       CEILING: exact gradient of -log p_ret[target], p_ret = (1-eps) r_t + eps/V  (the softmax-CE gradient through the
                single attention head -- NO BPTT, NO deep net; the base is a fixed bigram so the head is the only learned
                thing). Batched per passage (accumulate mean gradient, clip global norm, apply).
       LOCAL:   feedback-alignment / three-factor delta rule on the CLEAN read error (onehot(target) - r_t), baseline-
                subtracted by the retrieved mean; NO exact -target/p division, NO softmax-Jacobian transport.
       Returns (Wq, Wk, train_ce_history)."""
    rng = np.random.default_rng(seed * 101 + (0 if rule == "ceiling" else 1))
    in_dim = pdata[0][0].shape[1] if pdata and pdata[0][0].shape[0] else next(
        (k.shape[1] for k, _, _ in pdata if k.shape[0]), d_head)
    Wq = _init_proj(rng, d_head, in_dim)
    Wk = _init_proj(rng, d_head, in_dim)
    inv = 1.0 / math.sqrt(d_head)
    order = list(range(len(pdata)))
    hist = []
    for ep in range(epochs):
        rng.shuffle(order)
        tot_ce = 0.0
        tot_n = 0
        for pi in order:
            key_ctx, val_ids, positions = pdata[pi]
            if not positions:
                continue
            Kp_all = key_ctx @ Wk.T                            # (n_store, d_head) -- project once per passage/epoch
            gWq = np.zeros_like(Wq)
            gWk = np.zeros_like(Wk)
            npos = 0
            for q_ctx, tgt, M, _depth, _cur in positions:
                q = Wq @ q_ctx                                  # (d_head,)
                Kp = Kp_all[:M]                                 # (M, d_head)
                a = _softmax((Kp @ q) * inv)                    # (M,)
                val = val_ids[:M]
                r = np.zeros(V)
                np.add.at(r, val, a)                            # r_t (dim V)
                p_ret_tgt = (1.0 - eps) * r[tgt] + eps / V
                tot_ce += -math.log(max(p_ret_tgt, 1e-12))
                tot_n += 1
                npos += 1
                match = (val == tgt).astype(np.float64)         # slots whose value IS the target
                if rule == "ceiling":
                    m = (-(1.0 - eps) / p_ret_tgt) * match       # dL/da_tau (nonzero only where value==target -> stable)
                    gscore = a * (m - float(a @ m))              # dL/dscore = a*(m - a.m)
                    dq = (Kp.T @ gscore) * inv                   # dL/dq
                    gWq += np.outer(dq, q_ctx)
                    gWk += np.outer(q, gscore @ key_ctx[:M]) * inv
                else:                                            # LOCAL feedback-alignment / three-factor (ascent)
                    Lt = match - r[val]                          # (onehot(tgt) - r) . v_tau = clean read error on the slot
                    adv = Lt - float(a @ Lt)                     # baseline-subtract by the retrieved mean (softmax struct)
                    gs = a * adv                                 # score learning signal (ASCEND -> reduces loss)
                    dq = (Kp.T @ gs) * inv
                    gWq += np.outer(dq, q_ctx)
                    gWk += np.outer(q, gs @ key_ctx[:M]) * inv
            if npos == 0:
                continue
            gWq /= npos
            gWk /= npos                                          # mean gradient over the passage (length-robust)
            gn = math.sqrt(float(np.sum(gWq * gWq) + np.sum(gWk * gWk)))
            scale = (clip / gn) if (clip > 0 and gn > clip) else 1.0
            if rule == "ceiling":
                Wq -= lr * scale * gWq                           # DESCEND the exact CE gradient
                Wk -= lr * scale * gWk
            else:
                Wq += lr * scale * gWq                           # ASCEND the three-factor advantage
                Wk += lr * scale * gWk
        hist.append(round(tot_ce / max(1, tot_n), 4))
    return Wq, Wk, hist


# ---------------------------------------------------------------------------------------------------------------------
# Eval: read the trained Wq/Wk three ways (content / shuffle / uniform) over the eval store, each interpolated with the
# bigram base at its OWN best single-global-lam. Headline = content-minus-shuffle and content-minus-uniform (the BAGS).
# ---------------------------------------------------------------------------------------------------------------------
def _attend(Kp, q, val, V, arm, inv, beta, rng):
    M = Kp.shape[0]
    if arm == "uniform":
        a = np.full(M, 1.0 / M)
    else:
        Kk = Kp[rng.permutation(M)] if arm == "shuffle" else Kp   # shuffle: keys permuted vs values (random prior context)
        a = _softmax((Kk @ q) * (beta * inv))
    r = np.zeros(V)
    np.add.at(r, val, a)
    return r


def eval_attention(pdata_ev, Wq, Wk, P_bi, V, d_head, beta, seed):
    """For each eval passage, at each cross-sentence position score content/shuffle/uniform with the trained Wq/Wk;
       accumulate CE per arm per lam per within-passage depth bucket. Each arm then gets its OWN best single-global-lam.
       Base = bigram P(next | current token) (cur_id carried in each position)."""
    rng = np.random.default_rng(seed * 777 + 3)
    inv = 1.0 / math.sqrt(d_head)
    agg = {arm: {la: defaultdict(float) for la in LAMS} for arm in ARMS}
    agg_all = {arm: {la: 0.0 for la in LAMS} for arm in ARMS}
    base_ce = defaultdict(float)
    base_all = 0.0
    cnt = defaultdict(int)
    cnt_all = 0
    for key_ctx, val_ids, positions in pdata_ev:
        if not positions:
            continue
        Kp_all = key_ctx @ Wk.T
        for q_ctx, tgt, M, depth, cur_id in positions:
            q = Wq @ q_ctx
            b = _bucket(depth)
            pb = P_bi[cur_id]                                   # bigram base row P(next | current token)
            ceb = -math.log(max(pb[tgt], 1e-12))
            base_ce[b] += ceb
            base_all += ceb
            cnt[b] += 1
            cnt_all += 1
            Kp = Kp_all[:M]
            val = val_ids[:M]
            for arm in ARMS:
                r = _attend(Kp, q, val, V, arm, inv, beta, rng)
                for la in LAMS:
                    pf = (1.0 - la) * pb + la * r
                    ce = -math.log(max(pf[tgt], 1e-12))
                    agg[arm][la][b] += ce
                    agg_all[arm][la] += ce
    out = {"n_cross_positions": cnt_all,
           "base_cross_ce": round(base_all / max(1, cnt_all), 4),
           "base_by_depth": {b: round(base_ce[b] / cnt[b], 4) for b in cnt}, "by_arm": {}}
    for arm in ARMS:
        best_la = min(LAMS, key=lambda la: agg_all[arm][la])
        out["by_arm"][arm] = {
            "best_lam": best_la,
            "cross_ce": round(agg_all[arm][best_la] / max(1, cnt_all), 4),
            "by_depth": {b: round(agg[arm][best_la][b] / cnt[b], 4) for b in cnt},
            "cross_ce_by_lam": {str(la): round(agg_all[arm][la] / max(1, cnt_all), 4) for la in LAMS},
        }
    return out


def _headline(ev):
    """content-minus-shuffle + content-minus-uniform at cross positions (the BAGS), + per-depth breakdowns."""
    ba = ev["by_arm"]
    c = ba["content"]["cross_ce"]
    sh = ba["shuffle"]["cross_ce"]
    un = ba["uniform"]["cross_ce"]
    cms = round(sh - c, 4)                                      # + = content beats the shuffle bag
    cmu = round(un - c, 4)                                      # + = content beats the uniform bag
    cmb = round(ev["base_cross_ce"] - c, 4)                     # confoundable (a bag inflates this)
    by_depth = {}
    for b in ba["content"]["by_depth"]:
        cc = ba["content"]["by_depth"][b]
        ss = ba["shuffle"]["by_depth"].get(b)
        uu = ba["uniform"]["by_depth"].get(b)
        by_depth[b] = {"content_minus_shuffle": round(ss - cc, 4) if ss is not None else None,
                       "content_minus_uniform": round(uu - cc, 4) if uu is not None else None}
    return {"content_minus_shuffle": cms, "content_minus_uniform": cmu, "content_minus_base": cmb, "by_depth": by_depth}


def _derisk_one(seed, args):
    passages = load_passages(args.corpus, args.passages, args.sents_per_passage)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(passages))
    cut = int(0.8 * len(passages))
    tr_pass = [passages[i] for i in idx[:cut]]
    ev_pass = [passages[i] for i in idx[cut:]]
    tr_sents = [s for p in tr_pass for s in p]
    vocab = Vocab.build(tr_sents, V=args.vocab)
    V = vocab.size
    E = np.random.default_rng(seed * 555 + 7).standard_normal((V, args.d))  # FIXED-random token embeddings (identity only)

    # bigram base on TRAIN token stream
    tr_ids = [passage_to_ids(p, vocab)[0].tolist() for p in tr_pass]
    P_bi = fit_bigram(tr_ids, V)

    tr_data = [build_passage(p, vocab, E, args.win) for p in tr_pass]
    ev_data = [build_passage(p, vocab, E, args.win) for p in ev_pass]
    n_cross_tr = sum(len(p[2]) for p in tr_data)

    arms_out = {}
    for rule in args.rules:
        Wq, Wk, hist = train_attention(tr_data, V, args.d_head, rule, args.epochs, args.lr, args.eps, args.clip, seed)
        ev = eval_attention(ev_data, Wq, Wk, P_bi, V, args.d_head, args.beta, seed)
        head = _headline(ev)
        arms_out[rule] = {"train_ce_history": hist, "eval": ev, "headline": head}
    return {
        "seed": seed, "V": V, "d": args.d, "d_head": args.d_head, "win": args.win,
        "n_train_pass": len(tr_pass), "n_eval_pass": len(ev_pass), "n_cross_train": n_cross_tr,
        "sents_per_passage": args.sents_per_passage, "rules": args.rules, "by_rule": arms_out,
    }


def _print_seed(d):
    print(f"[seed {d['seed']}] V={d['V']} d={d['d']} d_head={d['d_head']} win={d['win']} "
          f"passages(tr/ev)={d['n_train_pass']}/{d['n_eval_pass']} n_cross_train={d['n_cross_train']}", flush=True)
    for rule in d["rules"]:
        r = d["by_rule"][rule]
        ba = r["eval"]["by_arm"]
        h = r["headline"]
        print(f"  [{rule:>7}] train-CE {r['train_ce_history']}", flush=True)
        print(f"           cross-CE: base {r['eval']['base_cross_ce']:.4f} | content {ba['content']['cross_ce']:.4f}"
              f"(λ{ba['content']['best_lam']}) | shuffle {ba['shuffle']['cross_ce']:.4f}(λ{ba['shuffle']['best_lam']}) | "
              f"uniform {ba['uniform']['cross_ce']:.4f}(λ{ba['uniform']['best_lam']})", flush=True)
        print(f"           HEADLINE (+ = content beats the bag): content-SHUFFLE {h['content_minus_shuffle']:+.4f} | "
              f"content-UNIFORM {h['content_minus_uniform']:+.4f}   (content-base {h['content_minus_base']:+.4f}, "
              f"confoundable)", flush=True)
        row = " ".join(f"d{k}:{h['by_depth'][k]['content_minus_shuffle']:+.3f}"
                       for lo, hi in BUCKETS for k in [f'{lo}-{hi}' if lo != hi else f'{lo}']
                       if k in h["by_depth"] and h["by_depth"][k]["content_minus_shuffle"] is not None)
        print(f"           content-minus-shuffle by within-passage depth: {row}", flush=True)


def _verdict(per_seed, rules):
    """1-seed/aggregate verdict on the three decision branches (mean over seeds)."""
    def mean(rule, key):
        return float(np.mean([per_seed[s]["by_rule"][rule]["headline"][key] for s in per_seed]))
    out = {}
    ceiling_beats = None
    if "ceiling" in rules:
        cs = mean("ceiling", "content_minus_shuffle")
        cu = mean("ceiling", "content_minus_uniform")
        ceiling_beats = bool(cs > MARGIN and cu > MARGIN)
        out["ceiling_content_minus_shuffle"] = round(cs, 4)
        out["ceiling_content_minus_uniform"] = round(cu, 4)
        out["ceiling_beats_bag"] = ceiling_beats
    local_beats = None
    if "local" in rules:
        ls = mean("local", "content_minus_shuffle")
        lu = mean("local", "content_minus_uniform")
        local_beats = bool(ls > MARGIN and lu > MARGIN)
        out["local_content_minus_shuffle"] = round(ls, 4)
        out["local_content_minus_uniform"] = round(lu, 4)
        out["local_beats_bag"] = local_beats
    if ceiling_beats is False:
        out["verdict"] = ("THIN -- even the FULL-GRADIENT learned-attention CEILING does NOT beat the cross-sentence BAG "
                          "(content ~<= shuffle/uniform) at this scale: open-text cross-sentence long-range is thin here. "
                          "The reservoir substrate is not the only limit -- there is little content-selective distal "
                          "signal for a learned attention to capture at this scale (a clean SCALE/DATA bound on the whole "
                          "frontier). HONEST NEGATIVE -- do NOT fake a positive.")
    elif ceiling_beats and local_beats is False:
        out["verdict"] = ("CEILING-ONLY -- the full-gradient learned attention BEATS the bag (there IS content-selective "
                          "cross-sentence signal reservoir keys could not capture), but the LOCAL feedback-alignment rule "
                          "does NOT reach it. The frontier is precisely 'make the attention's learning LOCAL/biological' "
                          "-- the ceiling-vs-local gap is the residual to close.")
    elif ceiling_beats and local_beats:
        out["verdict"] = ("BOTH -- a learned attention BEATS the cross-sentence bag AND it is learnable by a LOCAL "
                          "feedback-alignment rule: there is open-text long-range signal reservoir keys could not capture, "
                          "and the biological local rule reaches it. (Confirm at scale / multi-seed before generalizing.)")
    else:
        out["verdict"] = ("MIXED/INCONCLUSIVE -- see per-arm headlines; the load-bearing number is the CEILING "
                          "content-minus-shuffle/uniform.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/wikitext.txt")
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--passages", type=int, default=300, help="contiguous passages (train/eval split at passage level)")
    ap.add_argument("--sents-per-passage", type=int, default=10, help="consecutive sentences per passage (cross-sent horizon)")
    ap.add_argument("--d", type=int, default=32, help="fixed-random token embedding dim")
    ap.add_argument("--d-head", type=int, default=32, help="attention head dim")
    ap.add_argument("--win", type=int, default=2, help="query/key context window (tokens): concat(E[x_t],E[x_{t-1}],...)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1, help="attention learning rate (on the per-passage MEAN gradient)")
    ap.add_argument("--eps", type=float, default=0.02, help="uniform floor on the retrieval distribution (train stability)")
    ap.add_argument("--clip", type=float, default=1.0, help="global-norm clip on the per-passage mean gradient (0=off)")
    ap.add_argument("--beta", type=float, default=1.0, help="eval attention temperature multiplier (content/shuffle)")
    ap.add_argument("--rules", type=str, nargs="+", default=["ceiling", "local"], choices=["ceiling", "local"])
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    per_seed = {}
    for seed in args.seeds:
        d = _derisk_one(seed, args)
        per_seed[str(seed)] = d
        _print_seed(d)

    verdict = _verdict(per_seed, args.rules)
    out = {"runner": "_emerge_reservoir_lm_learned_attention_derisk", "corpus": args.corpus, "seeds": args.seeds,
           "beta": args.beta, "lams": LAMS, "margin_nats": MARGIN, "rules": args.rules,
           "headline": ("content_minus_shuffle / content_minus_uniform at cross-sentence positions for the CEILING "
                        "(full-gradient) learned attention = does a learned attention find long-range signal the reservoir "
                        "keys could not; content_minus_base alone is confoundable by a bag"),
           "verdict": verdict, "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 118, flush=True)
    print(f"[learned-attention] VERDICT: {verdict['verdict']}", flush=True)
    print(f"[learned-attention] {json.dumps({k: v for k, v in verdict.items() if k != 'verdict'})}", flush=True)
    print(f"[learned-attention] -> {args.json} ({out['elapsed_s']}s)\n" + "=" * 118, flush=True)


if __name__ == "__main__":
    main()
