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


def run_one(seed, K=6, F=6, T=10, n_train=400, n_eval=200, n_pool=220, epochs=25,
            lr_out=0.02, lr_rec=0.01, beta=1.0, awin_lo=30.0, awin_hi=300.0,
            arms=("fixed", "plastic", "symmetric", "sign_flip", "zero_signal", "shuffle_elig")):
    V = K + 2 + F
    tr, tr_rp, tr_tg = make_recall_task(K, F, T, n_train, seed * 100 + 1)
    ev, ev_rp, ev_tg = make_recall_task(K, F, T, n_eval, seed * 100 + 2)   # FRESH fillers = the filler-scramble-survives eval
    out = {}
    for arm in arms:
        res = RateReservoir(V, n_pool, seed=seed, alpha=0.3, spectral=1.1, alif=True, beta=beta,
                            adapt_win_lo=awin_lo, adapt_win_hi=awin_hi)      # fresh reservoir per arm (same seed = same init)
        W_out = train_recall(res, tr, V, epochs, lr_out, lr_rec, seed, arm)
        out[arm] = eval_recall(res, W_out, ev, ev_rp, ev_tg)
    # cue_scramble: train PLASTIC on a cue-scrambled task (random target), eval on the normal held-out -> collapse
    trc, _, _ = make_recall_task(K, F, T, n_train, seed * 100 + 1, cue_scramble=True)
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
    ap.add_argument("--grad-check", action="store_true", help="assert the ALIF 2-component eligibility matches finite differences")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    if a.grad_check:
        gc = grad_check_alif()
        print(f"[grad_check_alif] {gc}", flush=True)
    results = {}
    for T in a.T:
        per_seed = [run_one(s, K=a.K, T=T, n_pool=a.n_pool, epochs=a.epochs, lr_rec=a.lr_rec) for s in a.seeds]
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
