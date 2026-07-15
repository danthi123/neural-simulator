"""Controlled-LAG recurrent e-prop de-risk (2026-07-14, the emergence-engine recurrent-language-cortex frontier).

WHY: the prior attempt to train a recurrent language cortex via e-prop was ADVERSARIALLY REFUTED
(`2026-07-14-eprop-recurrent-synthesis-CONTROLS-REFUTED.md`): the "deep-context" win on WikiText-CE was a
credit-direction-INDEPENDENT memory-timescale artifact (sign_flip==plastic, true-gradient HURTS, distal-scramble
SURVIVES). The deep-research gate (Bellec-2020 e-prop + Marschall-Savin taxonomy + FPTT, read in depth) diagnosed the
cause: (a) the eligibility was already CORRECT (the faithful ALIF 2-component form, finite-difference-checked), and
(b) the decisive controls ran on the SHORT-horizon LIF arm on a GAMEABLE next-token-CE metric that has NO controlled
long-range dependency. The fix is a TASK change, not a mechanism fix.

THE TASK (Bellec store-recall, symbolized): a DELAYED CUED-RECALL (copy) stream with a CONTROLLED dependency length T.
Each trial = [STORE, x, f_1..f_T, RECALL, x] with x a CONTENT symbol and f_i disjoint FILLER symbols. Score accuracy
of predicting x AT the RECALL position (a T+1-step dependency). PROVABLY beats no n-gram: at RECALL the last n<T
tokens are fillers+RECALL, statistically INDEPENDENT of x -> any bigram/trigram is at chance (1/K) by construction.
Run on the ALIF arm (Bellec: plain LIF cannot bridge the delay even with BPTT; the slow-adaptation eligibility is what
carries long-range credit).

GO BAR: at a T* where the FIXED reservoir is at/near chance, PLASTIC recall >= 0.8 (or >= 3x chance), with a
dependency-length curve where plastic's solvable horizon STRICTLY exceeds fixed's. MANDATORY anti-cheats (each catches
the refuted artifact): (1) sign_flip DIVERGES (~chance, NOT ~plastic); (2) symmetric (true gradient) is the CEILING
(>= plastic, does NOT hurt); (3) cue_scramble COLLAPSES (train with x->random target -> eval ~chance); (4) the n-gram
is ~chance at RECALL; (5) zero_signal == fixed, shuffle_elig ~ fixed (sanity). Reuse-by-import of the validated
`RateReservoir` (ALIF) + the finite-difference-checked eligibility; numpy CPU; NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from research.runners._emerge_reservoir_lm_eprop_recurrent_derisk import RateReservoir, grad_check_alif
from research.runners._emerge_reservoir_lm_derisk import _softmax


def make_recall_task(K, F, T, n_trials, seed, cue_scramble=False):
    """[STORE, x, f_1..f_T, RECALL, x] per trial. recall_pos = the RECALL token index (its read-out predicts the final
    x). targets = the STORED x (eval ALWAYS measures the stored x). cue_scramble: the training target after RECALL is a
    RANDOM content symbol (breaks cue->target) -> a model can learn nothing -> eval-vs-stored-x collapses to chance."""
    rng = np.random.default_rng(seed)
    STORE, RECALL, FILL0 = K, K + 1, K + 2
    trials, recall_pos, targets = [], [], []
    for _ in range(n_trials):
        x = int(rng.integers(K))
        fillers = [FILL0 + int(rng.integers(F)) for _ in range(T)]
        tgt = int(rng.integers(K)) if cue_scramble else x
        seq = [STORE, x] + fillers + [RECALL, tgt]
        trials.append(np.asarray(seq, dtype=np.int64))
        recall_pos.append(len(seq) - 2)               # the RECALL token; its read-out predicts seq[-1]
        targets.append(x)                              # eval measures the STORED x
    return trials, recall_pos, targets


def make_xor_task(K, F, T, n_trials, seed, cue_scramble=False):
    """DELAYED MODULAR-SUM (recurrent-computation discriminator): [STORE, x1, f_1..f_T, x2, RECALL, y] with
    y=(x1+x2) mod K. x1 is DISTAL (T+2 back), x2 is RECENT (2 back). y is a NONLINEAR function of TWO temporally-
    separated cues -> a LINEAR read-out over [held-x1-code, recent-x2-code] provably CANNOT compute it (XOR-like), so
    the learned RECURRENCE must combine them; a memory-timescale nudge (holding x1 OR reading x2 alone) CANNOT fake it.
    cue_scramble: the training target is a RANDOM y (breaks cue->target) -> collapse."""
    rng = np.random.default_rng(seed)
    STORE, RECALL, FILL0 = K, K + 1, K + 2
    trials, recall_pos, targets = [], [], []
    for _ in range(n_trials):
        x1 = int(rng.integers(K)); x2 = int(rng.integers(K))
        y = (x1 + x2) % K
        fillers = [FILL0 + int(rng.integers(F)) for _ in range(T)]
        tgt = int(rng.integers(K)) if cue_scramble else y
        seq = [STORE, x1] + fillers + [x2, RECALL, tgt]
        trials.append(np.asarray(seq, dtype=np.int64))
        recall_pos.append(len(seq) - 2)               # the RECALL token; its read-out predicts y
        targets.append(y)
    return trials, recall_pos, targets


def make_accum_task(K, F, T, n_trials, seed, cue_scramble=False):
    """EVIDENCE ACCUMULATION (Bellec's OWN validated e-prop+ALIF task) = the POSITIVE-CONTROL ceiling. A stream of T
    LEFT/RIGHT cues (content 0/1), interspersed with fillers, then RECALL -> y = the MAJORITY side. The recurrence must
    INTEGRATE (count) the cues -> a memory-nudge that holds one cue can't fake it. If plastic/symmetric LEARN this
    (>> chance) but XOR does NOT, the XOR null is specifically the cross-cue-COMBINATION limitation (not a bad
    implementation). K is forced to 2 (LEFT/RIGHT). cue_scramble: random target -> collapse."""
    rng = np.random.default_rng(seed)
    RECALL, FILL0 = 2, 3
    trials, recall_pos, targets = [], [], []
    for _ in range(n_trials):
        cues = rng.integers(2, size=T)                 # T LEFT(0)/RIGHT(1) cues
        y = int(cues.sum() * 2 > T)                     # majority (ties -> 0)
        seq = []
        for c in cues:
            seq.append(int(c))
            if F > 0:
                seq.append(FILL0 + int(rng.integers(F)))   # a filler between cues (spacing)
        tgt = int(rng.integers(2)) if cue_scramble else y
        seq = seq + [RECALL, tgt]
        trials.append(np.asarray(seq, dtype=np.int64))
        recall_pos.append(len(seq) - 2)
        targets.append(y)
    return trials, recall_pos, targets


def train_recall(res, trials, V, epochs, lr_out, lr_rec, seed, arm, wd=1e-3):
    """ALIF e-prop trainer carrying the FAITHFUL Bellec-2020 2-component eligibility (identical to the validated
    `_train_alif`), parameterized by the credit ARM. arm in {fixed, plastic, symmetric, sign_flip, zero_signal,
    shuffle_elig}. W_out by the one-step next-token delta rule; W_rec by e-prop (NO BPTT). W_rec updated in place."""
    rng = np.random.default_rng(seed * 13 + 7)
    n = res.n; a = res.alpha; rho = res.rho; beta = res.beta
    W_out = rng.standard_normal((V, 2 * n)) * 0.01
    B = rng.standard_normal((2 * n, V)) / np.sqrt(V)   # fixed random feedback over BOTH compartments [h; a]
    frozen = (arm == "fixed")
    order = np.arange(len(trials))
    for ep in range(epochs):
        rng.shuffle(order)
        for si in order:
            ids = trials[si]
            if len(ids) < 2:
                continue
            h = np.zeros(n); ad = np.zeros(n)
            eps_h = np.zeros((n, n)); eps_a = np.zeros((n, n))
            for t in range(len(ids) - 1):
                h_prev = h
                ad = rho * ad + (1.0 - rho) * h_prev
                x = res.W_in[:, ids[t]]
                pre = res.W_rec @ h_prev + x + res.b - beta * ad
                act = np.tanh(pre)
                h = (1 - a) * h_prev + a * act
                feat = np.concatenate([h, ad])
                p = _softmax(W_out @ feat)
                delta = -p; delta[ids[t + 1]] += 1.0
                W_out += lr_out * (np.outer(delta, feat) - wd * W_out)
                if frozen:
                    continue
                psi = a * (1.0 - act * act)
                eps_a = rho[:, None] * eps_a + (1.0 - rho)[:, None] * eps_h
                eps_h = (1 - a)[:, None] * eps_h + psi[:, None] * (h_prev[None, :] - beta * eps_a)
                if arm == "zero_signal":
                    continue                            # L:=0 -> W_rec never moves (sanity == fixed)
                L = (W_out.T @ delta) if arm == "symmetric" else (B @ delta)  # weight-transport ceiling vs random FA
                if arm == "sign_flip":
                    L = -L                              # credit-direction flipped -> must DIVERGE from plastic
                L_h, L_a = L[:n], L[n:]
                Eh, Ea = eps_h, eps_a
                if arm == "shuffle_elig":
                    Eh = eps_h.reshape(-1)[rng.permutation(n * n)].reshape(n, n)
                    Ea = eps_a.reshape(-1)[rng.permutation(n * n)].reshape(n, n)
                res.W_rec += lr_rec * (L_h[:, None] * Eh + L_a[:, None] * Ea)
    return W_out


def eval_recall(res, W_out, trials, recall_pos, targets):
    """RECALL-position accuracy: for each trial, run the (trained) reservoir fresh, read the RECALL-token state, and
    check argmax(W_out @ feat) == the stored x."""
    correct = 0
    for si, ids in enumerate(trials):
        states = res.forward_states(ids)                # ALIF -> concat([h_t, a_t]) per t; fresh state per trial
        p = _softmax(W_out @ states[recall_pos[si]])
        correct += int(np.argmax(p) == targets[si])
    return correct / max(1, len(trials))


def ngram_recall_acc(K, F, T, tr_trials, tr_targets, ev_trials, ev_recall, ev_targets, order=2):
    """The strongest n-gram at the RECALL position: P(x | last `order` tokens before-and-including RECALL). By
    construction those tokens are fillers/RECALL (independent of x) -> chance. Empirical confirmation."""
    from collections import defaultdict, Counter
    ctx = defaultdict(Counter)
    for ids, x in zip(tr_trials, tr_targets):
        rp = len(ids) - 2
        key = tuple(int(v) for v in ids[max(0, rp - order + 1):rp + 1])
        ctx[key][x] += 1
    glob = Counter(tr_targets)
    correct = 0
    for ids, rp, x in zip(ev_trials, ev_recall, ev_targets):
        key = tuple(int(v) for v in ids[max(0, rp - order + 1):rp + 1])
        c = ctx.get(key) or glob
        correct += int(c.most_common(1)[0][0] == x)
    return correct / max(1, len(ev_trials))


def language_2stage_test(seed, n_sentences=6000, V=300, n_pool=200, n_hidden=128, epochs=8, lr=0.02):
    """MISSION-CONNECTED: does the identified fix (a 2-STAGE cortical read-out) beat a LINEAR read-out on a REAL
    language stream (the EMERGE SVO+function-word stream), or does the bigram-dominated scale wall dominate? Fixed ALIF
    reservoir + cached per-token states; train a linear vs a 2-layer read-out for next-token prediction; compare CE +
    accuracy to a bigram. Tests whether the language cortex has the same linear-readout limitation the XOR task had."""
    import research.runners._emerge62_discover_function_words_derisk as m62
    from research.runners._emerge_reservoir_lm_derisk import Vocab, _split_sentences, fit_bigram
    stream = m62.build_stream(seed, n_sentences=n_sentences)
    sents = _split_sentences(stream)
    vocab = Vocab.build(sents, V)
    ids_sents = [np.asarray([vocab.id(w) for w in s], dtype=np.int64) for s in sents]
    ids_sents = [s for s in ids_sents if len(s) >= 2]
    ntr = int(0.8 * len(ids_sents)); tr, ev = ids_sents[:ntr], ids_sents[ntr:]
    Veff = vocab.size
    res = RateReservoir(Veff, n_pool, seed=seed, alpha=0.3, spectral=1.1, alif=True, beta=1.0)

    def pairs(sset):
        X, Y = [], []
        for ids in sset:
            S = res.forward_states(ids)
            for t in range(len(ids) - 1):
                X.append(S[t]); Y.append(int(ids[t + 1]))
        return np.array(X), np.array(Y)
    Xtr, Ytr = pairs(tr); Xev, Yev = pairs(ev)
    d = Xtr.shape[1]; rng = np.random.default_rng(seed)

    def ce_acc(fn):
        ce = 0.0; acc = 0
        for x, y in zip(Xev, Yev):
            p = fn(x); ce += -np.log(p[y] + 1e-12); acc += int(np.argmax(p) == y)
        return ce / len(Xev), acc / len(Xev)
    # LINEAR read-out (delta rule)
    W = np.zeros((Veff, d))
    for ep in range(epochs):
        for i in rng.permutation(len(Xtr)):
            p = _softmax(W @ Xtr[i]); g = -p; g[Ytr[i]] += 1.0; W += lr * np.outer(g, Xtr[i])
    lin_ce, lin_acc = ce_acc(lambda x: _softmax(W @ x))
    # 2-STAGE read-out (backprop through one hidden layer)
    W1 = rng.standard_normal((n_hidden, d)) * 0.1; W2 = rng.standard_normal((Veff, n_hidden)) * 0.1
    for ep in range(epochs):
        for i in rng.permutation(len(Xtr)):
            hh = np.tanh(W1 @ Xtr[i]); p = _softmax(W2 @ hh); g = -p; g[Ytr[i]] += 1.0
            W2 += lr * np.outer(g, hh); W1 += lr * np.outer((W2.T @ g) * (1 - hh * hh), Xtr[i])
    two_ce, two_acc = ce_acc(lambda x: _softmax(W2 @ np.tanh(W1 @ x)))
    # ANTI-CHEAT (permuted-corpus): shuffle each TRAIN sentence's token order, recompute states, retrain a FRESH
    # 2-stage read-out, eval on the REAL held-out. If the 2-stage bigram-beating is genuine word-ORDER structure it
    # COLLAPSES (>= bigram); if it survives, the advantage is a unigram/artifact, not real structure.
    tr_perm = [s[rng.permutation(len(s))] for s in tr]
    Xtp, Ytp = pairs(tr_perm)
    W1p = rng.standard_normal((n_hidden, d)) * 0.1; W2p = rng.standard_normal((Veff, n_hidden)) * 0.1
    for ep in range(epochs):
        for i in rng.permutation(len(Xtp)):
            hh = np.tanh(W1p @ Xtp[i]); p = _softmax(W2p @ hh); g = -p; g[Ytp[i]] += 1.0
            W2p += lr * np.outer(g, hh); W1p += lr * np.outer((W2p.T @ g) * (1 - hh * hh), Xtp[i])
    perm_ce, _ = ce_acc(lambda x: _softmax(W2p @ np.tanh(W1p @ x)))
    # bigram baseline CE
    P_bi = fit_bigram(tr, Veff)
    bi_ce = 0.0
    for ids in ev:
        for t in range(len(ids) - 1):
            bi_ce += -np.log(P_bi[int(ids[t]), int(ids[t + 1])] + 1e-12)
    bi_ce /= max(1, sum(len(s) - 1 for s in ev))
    return {"V": Veff, "linear_ce": lin_ce, "twostage_ce": two_ce, "bigram_ce": bi_ce, "perm_ce": perm_ce,
            "linear_acc": lin_acc, "twostage_acc": two_acc, "n_eval": len(Xev)}


def train_recall_nl(res, trials, V, epochs, lr_out, lr_rec, seed, arm, n_hidden=64, wd=1e-3):
    """e-prop W_rec + a 2-layer NONLINEAR read-out (feat->hidden tanh->softmax), read-out by backprop. Removes the
    linear-readout confound so the HORIZON-EXTENSION question is clean: does training W_rec (plastic) EXTEND the
    fixed reservoir's distal-cue preservation horizon vs a fixed reservoir (both with the nonlinear read-out)?
    arm in {fixed, plastic, symmetric, sign_flip}. Returns (W1, W2)."""
    rng = np.random.default_rng(seed * 13 + 7)
    n = res.n; a = res.alpha; rho = res.rho; beta = res.beta
    W1 = rng.standard_normal((n_hidden, 2 * n)) * 0.1
    W2 = rng.standard_normal((V, n_hidden)) * 0.1
    B = rng.standard_normal((2 * n, V)) / np.sqrt(V)   # random feedback to feat (e-prop W_rec signal)
    frozen = (arm == "fixed")
    order = np.arange(len(trials))
    for ep in range(epochs):
        rng.shuffle(order)
        for si in order:
            ids = trials[si]
            if len(ids) < 2:
                continue
            h = np.zeros(n); ad = np.zeros(n)
            eps_h = np.zeros((n, n)); eps_a = np.zeros((n, n))
            for t in range(len(ids) - 1):
                h_prev = h
                ad = rho * ad + (1.0 - rho) * h_prev
                x = res.W_in[:, ids[t]]
                pre = res.W_rec @ h_prev + x + res.b - beta * ad
                act = np.tanh(pre)
                h = (1 - a) * h_prev + a * act
                feat = np.concatenate([h, ad])
                hh = np.tanh(W1 @ feat)
                z = W2 @ hh; z -= z.max(); e = np.exp(z); p = e / e.sum()
                delta = -p; delta[ids[t + 1]] += 1.0
                W2 += lr_out * (np.outer(delta, hh) - wd * W2)          # read-out backprop
                dhh = (W2.T @ delta) * (1.0 - hh * hh)
                W1 += lr_out * (np.outer(dhh, feat) - wd * W1)
                if frozen:
                    continue
                psi = a * (1.0 - act * act)
                eps_a = rho[:, None] * eps_a + (1.0 - rho)[:, None] * eps_h
                eps_h = (1 - a)[:, None] * eps_h + psi[:, None] * (h_prev[None, :] - beta * eps_a)
                L = (W1.T @ dhh) if arm == "symmetric" else (B @ delta)  # true-grad-to-feat vs random FA
                if arm == "sign_flip":
                    L = -L
                res.W_rec += lr_rec * (L[:n][:, None] * eps_h + L[n:][:, None] * eps_a)
    return W1, W2


def eval_recall_nl(res, W1, W2, trials, recall_pos, targets):
    correct = 0
    for si, ids in enumerate(trials):
        feat = res.forward_states(ids)[recall_pos[si]]
        correct += int(np.argmax(W2 @ np.tanh(W1 @ feat)) == targets[si])
    return correct / max(1, len(trials))


def horizon_ext_test(seed, task="xor", K=4, F=6, T=20, n_train=400, n_eval=200, n_pool=100, epochs=25,
                     lr_out=0.02, lr_rec=0.0005, n_hidden=64):
    """Does PLASTIC recurrent e-prop EXTEND the fixed reservoir's horizon (with the nonlinear read-out removing the
    linear confound)? Returns {fixed, plastic, symmetric, sign_flip} recall accuracy at lag T."""
    V = K + 2 + F
    gen = {"xor": make_xor_task, "accum": make_accum_task}.get(task, make_recall_task)
    tr, tr_rp, tr_tg = gen(K, F, T, n_train, seed * 100 + 1)
    ev, ev_rp, ev_tg = gen(K, F, T, n_eval, seed * 100 + 2)
    out = {}
    for arm in ("fixed", "plastic", "symmetric", "sign_flip"):
        res = RateReservoir(V, n_pool, seed=seed, alpha=0.3, spectral=1.1, alif=True, beta=1.0)
        W1, W2 = train_recall_nl(res, tr, V, epochs, lr_out, lr_rec, seed, arm, n_hidden=n_hidden)
        out[arm] = eval_recall_nl(res, W1, W2, ev, ev_rp, ev_tg)
    out["chance"] = 1.0 / K
    return out


def nonlinear_readout_test(seed, task="xor", K=4, F=6, T=5, n_train=400, n_eval=200, n_pool=100,
                           epochs=150, n_hidden=64, lr=0.05):
    """DECISIVE reframe test: does a NONLINEAR (2-layer MLP) read-out on the FIXED reservoir already solve the task? If
    yes for XOR, the XOR null was a LINEAR-READOUT limitation (the fixed reservoir's features already contain the two
    cues; a linear softmax just can't COMBINE them), NOT a recurrent-credit limitation -> the recurrent W_rec learning
    was never the bottleneck, and SnAp-1 is unnecessary. Returns (mlp_acc, chance)."""
    V = K + 2 + F
    gen = {"xor": make_xor_task, "accum": make_accum_task}.get(task, make_recall_task)
    tr, tr_rp, tr_tg = gen(K, F, T, n_train, seed * 100 + 1)
    ev, ev_rp, ev_tg = gen(K, F, T, n_eval, seed * 100 + 2)
    res = RateReservoir(V, n_pool, seed=seed, alpha=0.3, spectral=1.1, alif=True, beta=1.0)
    Xtr = np.array([res.forward_states(ids)[rp] for ids, rp in zip(tr, tr_rp)])   # FIXED reservoir RECALL features
    Xev = np.array([res.forward_states(ids)[rp] for ids, rp in zip(ev, ev_rp)])
    ytr = np.array(tr_tg); yev = np.array(ev_tg)
    rng = np.random.default_rng(seed); d = Xtr.shape[1]
    W1 = rng.standard_normal((n_hidden, d)) * 0.1; W2 = rng.standard_normal((V, n_hidden)) * 0.1
    for ep in range(epochs):
        for i in rng.permutation(len(Xtr)):
            x = Xtr[i]
            hh = np.tanh(W1 @ x)
            z = W2 @ hh; z -= z.max(); e = np.exp(z); p = e / e.sum()
            g = -p; g[ytr[i]] += 1.0
            W2 += lr * np.outer(g, hh)
            W1 += lr * np.outer((W2.T @ g) * (1 - hh * hh), x)
    acc = float(np.mean([np.argmax(W2 @ np.tanh(W1 @ x)) == y for x, y in zip(Xev, yev)]))
    return acc, 1.0 / K


def run_one(seed, K=6, F=6, T=10, n_train=400, n_eval=200, n_pool=220, epochs=25,
            lr_out=0.02, lr_rec=0.01, beta=1.0, awin_lo=30.0, awin_hi=300.0, task="copy",
            arms=("fixed", "plastic", "symmetric", "sign_flip", "zero_signal", "shuffle_elig")):
    V = K + 2 + F
    gen = {"xor": make_xor_task, "accum": make_accum_task}.get(task, make_recall_task)
    tr, tr_rp, tr_tg = gen(K, F, T, n_train, seed * 100 + 1)
    ev, ev_rp, ev_tg = gen(K, F, T, n_eval, seed * 100 + 2)   # FRESH fillers = the filler-scramble-survives eval
    out = {}
    for arm in arms:
        res = RateReservoir(V, n_pool, seed=seed, alpha=0.3, spectral=1.1, alif=True, beta=beta,
                            adapt_win_lo=awin_lo, adapt_win_hi=awin_hi)      # fresh reservoir per arm (same seed = same init)
        W_out = train_recall(res, tr, V, epochs, lr_out, lr_rec, seed, arm)
        out[arm] = eval_recall(res, W_out, ev, ev_rp, ev_tg)
    # cue_scramble: train PLASTIC on a cue-scrambled task (random target), eval on the normal held-out -> collapse
    trc, _, _ = gen(K, F, T, n_train, seed * 100 + 1, cue_scramble=True)
    res_c = RateReservoir(V, n_pool, seed=seed, alpha=0.3, spectral=1.1, alif=True, beta=beta,
                          adapt_win_lo=awin_lo, adapt_win_hi=awin_hi)
    W_c = train_recall(res_c, trc, V, epochs, lr_out, lr_rec, seed, "plastic")
    out["cue_scramble"] = eval_recall(res_c, W_c, ev, ev_rp, ev_tg)
    out["ngram2"] = ngram_recall_acc(K, F, T, tr, tr_tg, ev, ev_rp, ev_tg, order=2)
    out["chance"] = 1.0 / K
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--T", type=int, nargs="+", default=[2, 5, 10, 20])
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-pool", type=int, default=220)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr-rec", type=float, default=0.01, help="e-prop W_rec learning rate (destabilization test: lower it)")
    ap.add_argument("--task", default="copy", choices=["copy", "xor", "accum"], help="copy=single-cue hold (ALIF solves it); xor=delayed modular-sum of TWO cues; accum=evidence-accumulation majority (Bellec's validated e-prop+ALIF positive control)")
    ap.add_argument("--grad-check", action="store_true", help="assert the ALIF 2-component eligibility matches finite differences")
    ap.add_argument("--nonlinear-readout", action="store_true", help="DECISIVE: does a 2-layer read-out on the FIXED reservoir solve the task? (reframes XOR null as readout-vs-recurrent-credit)")
    ap.add_argument("--horizon-test", action="store_true", help="does PLASTIC recurrent e-prop EXTEND the fixed reservoir's horizon (with a nonlinear read-out removing the linear confound)?")
    ap.add_argument("--language-test", action="store_true", help="MISSION: does a 2-STAGE read-out beat a LINEAR one on the real EMERGE SVO language stream?")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    if a.grad_check:
        gc = grad_check_alif()
        print(f"[grad_check_alif] {gc}", flush=True)
    if a.language_test:
        rs = [language_2stage_test(s, n_pool=a.n_pool, epochs=a.epochs) for s in a.seeds]
        agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        beats_bigram = agg["twostage_ce"] < agg["bigram_ce"] - 0.05
        perm_collapses = agg["perm_ce"] >= agg["bigram_ce"] - 0.05    # permuted-corpus must NOT beat the bigram
        verdict = ("GENUINE: 2-STAGE read-out beats the bigram via real word-order structure (permuted-corpus COLLAPSES)"
                   if (beats_bigram and perm_collapses)
                   else ("ARTIFACT: 2-stage beats bigram BUT permuted-corpus ALSO does (unigram/overfit, not structure)"
                         if beats_bigram else "2-stage does NOT beat the bigram"))
        print(f"[language-test] V={agg['V']:.0f} n_eval={agg['n_eval']:.0f} | linear_ce={agg['linear_ce']:.3f} "
              f"twostage_ce={agg['twostage_ce']:.3f} bigram_ce={agg['bigram_ce']:.3f} perm_ce={agg['perm_ce']:.3f} | "
              f"twostage_acc={agg['twostage_acc']:.3f} -> {verdict}", flush=True)
        return
    if a.nonlinear_readout:
        for T in a.T:
            accs = [nonlinear_readout_test(s, task=a.task, K=a.K, T=T, n_pool=a.n_pool) for s in a.seeds]
            acc = float(np.mean([x[0] for x in accs])); chance = accs[0][1]
            verdict = ("READOUT-LIMITED (fixed reservoir already has the features; a nonlinear read-out solves it -> "
                       "recurrent credit was NOT the bottleneck)" if acc >= max(0.8, 3 * chance)
                       else "NOT readout-limited (even a nonlinear read-out on the fixed reservoir fails -> the "
                       "features/recurrent-computation are genuinely missing)")
            print(f"[nonlinear-readout] task={a.task} T={T} fixed-reservoir + 2-layer MLP acc={acc:.3f} "
                  f"(chance={chance:.3f}) -> {verdict}", flush=True)
        return
    if a.horizon_test:
        for T in a.T:
            rs = [horizon_ext_test(s, task=a.task, K=a.K, T=T, n_pool=a.n_pool, epochs=a.epochs, lr_rec=a.lr_rec) for s in a.seeds]
            agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            ext = agg["plastic"] - agg["fixed"]
            verdict = ("EXTENDS (plastic>fixed AND symmetric>=plastic AND sign_flip collapses)"
                       if (ext > 0.08 and agg["symmetric"] >= agg["plastic"] - 0.05
                           and agg["sign_flip"] <= agg["fixed"] + 0.05)
                       else "does NOT genuinely extend (or fails an anti-cheat)")
            print(f"[horizon-test] task={a.task} T={T} chance={agg['chance']:.3f} | fixed={agg['fixed']:.3f} "
                  f"plastic={agg['plastic']:.3f} symmetric={agg['symmetric']:.3f} sign_flip={agg['sign_flip']:.3f} "
                  f"-> {verdict}", flush=True)
        return
    results = {}
    for T in a.T:
        per_seed = [run_one(s, K=a.K, T=T, n_pool=a.n_pool, epochs=a.epochs, lr_rec=a.lr_rec, task=a.task) for s in a.seeds]
        agg = {}
        for k in per_seed[0]:
            agg[k] = float(np.mean([r[k] for r in per_seed]))
        results[T] = agg
        chance = agg["chance"]
        print(f"T={T:3d}  chance={chance:.3f} | fixed={agg['fixed']:.3f} plastic={agg['plastic']:.3f} "
              f"symmetric={agg['symmetric']:.3f} sign_flip={agg['sign_flip']:.3f} "
              f"zero={agg['zero_signal']:.3f} shuf={agg['shuffle_elig']:.3f} "
              f"cue_scr={agg['cue_scramble']:.3f} ngram={agg['ngram2']:.3f}", flush=True)
    # GO logic: a T* where fixed ~ chance AND plastic strong AND all anti-cheats hold
    chance = results[a.T[0]]["chance"]
    go_T = None
    for T in a.T:
        r = results[T]
        if (r["plastic"] >= max(0.8, 3 * chance) and r["fixed"] <= 2 * chance
                and r["sign_flip"] <= 2 * chance and r["symmetric"] >= r["plastic"] - 0.1
                and r["cue_scramble"] <= 2 * chance and r["ngram2"] <= 2 * chance):
            go_T = T; break
    # horizon: highest T where plastic still strong vs highest T where fixed still strong
    plastic_horizon = max([T for T in a.T if results[T]["plastic"] >= max(0.8, 3 * chance)], default=None)
    fixed_horizon = max([T for T in a.T if results[T]["fixed"] >= max(0.8, 3 * chance)], default=0)
    verdict = ("GO" if (go_T is not None and (fixed_horizon == 0 or (plastic_horizon or 0) > fixed_horizon))
               else "NO/PARTIAL")
    print(f"\n=== {verdict} : go_T={go_T}  plastic_horizon={plastic_horizon} fixed_horizon={fixed_horizon} "
          f"({time.time()-t0:.0f}s) ===", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"results": results, "go_T": go_T, "plastic_horizon": plastic_horizon,
                       "fixed_horizon": fixed_horizon, "verdict": verdict}, f, indent=2)


if __name__ == "__main__":
    main()
