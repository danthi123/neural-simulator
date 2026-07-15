"""2026-07-15 — the highest-leverage next (ROADMAP-sync): LEARN the comprehension->intent DISPATCH on the GO feedforward
spiking deep-credit substrate (e-prop + population coding), replacing the hand membership-aware router (EMERGE-58).

THE TEST (rate-first, reuse-by-import from `_deep_eprop_binder_bundling_derisk`): does a DEEP transport-free e-prop-trained
spiking classifier learn a COMPOSITIONAL dispatch rule from a stream and GENERALIZE to held-out (subject x question-type)
COMPOSITIONS -- the systematicity axis? Unlike the binder (which failed systematicity on invertible superposition), dispatch
is a FEEDFORWARD label-map in the KNOWN-GO classification regime, so the 2026-07-14 binder boundary should NOT bind it.

The task mirrors the real router: the response-FRAME (intent) is a compositional function of (the subject's CATEGORY, the
QUESTION-TYPE). Subject codes carry CATEGORY structure (shared category block + unique identity) so the net must READ the
category from the code and combine it with the qtype -> a held-out (subject, qtype) combo generalizes IFF the net learned
"read category x qtype -> intent", not memorized (subject, qtype) -> intent.

ARMS (reuse `_train_snn`/`score_snn`): armA linear (n_in->INTENT), armB 2-hidden e-prop (the pick), armB_bptt 2-hidden BPTT
(the ceiling -> a NEGATIVE is interpretable), permuted (shuffle labels -> chance = the load-bearing anti-cheat vs memorization).
GO: armB held-out-compositional >> chance AND >> the linear armA AND ~ the BPTT ceiling; permuted -> chance; 6-seed.

Run: SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m research.runners._learned_dispatch_derisk
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._deep_eprop_binder_bundling_derisk import _train_snn, score_snn, standardize, _forward_logits, T

N_CAT = 4               # subject categories (e.g. animal / vehicle / person / place)
N_SUBJ_PER_CAT = 6      # subjects per category (24 subjects)
N_QTYPE = 5             # question types (property-query / describe / yes-no / inheritance / unknown)
N_INTENT = 5            # response frames (fact-lookup / inherit-reason / describe / abstain / emerge-frame)
D_CAT = 24              # shared category-block dims (carries category => enables compositional read)
D_ID = 24              # unique subject-identity dims
D_Q = 24               # question-type code dims
SPARS = 0.25


def _sparse_code(rng, D):
    v = np.zeros(D); k = max(1, int(SPARS * D)); v[rng.choice(D, k, replace=False)] = 1.0; return v


def build_task(seed, n_holdout_per_cat=3, hard=False):
    rng = np.random.default_rng(seed * 911 + 7)
    cat_code = np.stack([_sparse_code(rng, D_CAT) for _ in range(N_CAT)])          # shared per category
    subj_cat = np.repeat(np.arange(N_CAT), N_SUBJ_PER_CAT)                          # category of each subject
    subj_id = np.stack([_sparse_code(rng, D_ID) for _ in range(N_CAT * N_SUBJ_PER_CAT)])
    q_code = np.stack([_sparse_code(rng, D_Q) for _ in range(N_QTYPE)])
    if hard:
        # HARD (clean deep-credit systematicity): a STRUCTURED-NONLINEAR rule intent = (a[cat] XOR b[qt] bits) -> class,
        # inferable ONLY by composing the two SEPARATELY-attested factors; hold out whole (cat,qtype) COMBINATIONS
        # (never attested) so 1-NN has NO same-(cat,qtype) neighbor AND linear can't do the XOR -> only deep+compose wins.
        nb = 3                                                                      # bits -> 2^3=8 >= N_INTENT
        a = rng.integers(0, 2, size=(N_CAT, nb)); b = rng.integers(0, 2, size=(N_QTYPE, nb))
        intent_map = np.array([[int("".join(map(str, (a[c] ^ b[q])[:nb])), 2) % N_INTENT
                                for q in range(N_QTYPE)] for c in range(N_CAT)])
        # hold out ~30% of (cat,qtype) CELLS entirely (each cat + each qtype still attested in OTHER cells)
        cells = [(c, q) for c in range(N_CAT) for q in range(N_QTYPE)]
        rng.shuffle(cells)
        held_cells = set()
        catcount = {c: 0 for c in range(N_CAT)}; qcount = {q: 0 for q in range(N_QTYPE)}
        for (c, q) in cells:
            if len(held_cells) >= int(0.3 * len(cells)):
                break
            # keep each cat + each qtype attested in >=2 train cells
            train_c = sum(1 for (c2, q2) in cells if c2 == c and (c2, q2) not in held_cells) - 1
            train_q = sum(1 for (c2, q2) in cells if q2 == q and (c2, q2) not in held_cells) - 1
            if train_c >= 2 and train_q >= 2:
                held_cells.add((c, q))
        while len(np.unique(intent_map)) < N_INTENT:
            a = rng.integers(0, 2, size=(N_CAT, nb)); b = rng.integers(0, 2, size=(N_QTYPE, nb))
            intent_map = np.array([[int("".join(map(str, (a[c] ^ b[q])[:nb])), 2) % N_INTENT
                                    for q in range(N_QTYPE)] for c in range(N_CAT)])
        held_pred = lambda s, q: (subj_cat[s], q) in held_cells
    else:
        intent_map = rng.integers(0, N_INTENT, size=(N_CAT, N_QTYPE))
        while len(np.unique(intent_map)) < N_INTENT:
            intent_map = rng.integers(0, N_INTENT, size=(N_CAT, N_QTYPE))
        held_set = set()
        for c in range(N_CAT):
            subs = np.where(subj_cat == c)[0]
            for _ in range(n_holdout_per_cat):
                s = int(rng.choice(subs)); q = int(rng.integers(0, N_QTYPE))
                if sum(1 for s2 in subs if s2 != s and (s2, q) not in held_set) >= 2:
                    held_set.add((s, q))
        held_pred = lambda s, q: (s, q) in held_set
    n_subj = N_CAT * N_SUBJ_PER_CAT
    X, y, is_held = [], [], []
    for s in range(n_subj):
        for q in range(N_QTYPE):
            x = np.concatenate([cat_code[subj_cat[s]], subj_id[s], q_code[q]])
            X.append(x); y.append(int(intent_map[subj_cat[s], q])); is_held.append(bool(held_pred(s, q)))
    return np.array(X), np.array(y), np.array(is_held), N_INTENT


def run_one(seed, hidden=48, epochs=120, lr=0.05, in_gain=1.0, do_bptt=True, hard=False):
    X, y, is_held, F = build_task(seed, hard=hard)
    tr = ~is_held
    Xtr, ytr = X[tr], y[tr]
    Xev, yev, held = X, y, is_held                       # eval on ALL, tag held-out compositions
    Xtr_n, Xev_n = standardize(Xtr, Xev)
    n_in = Xtr.shape[1]
    out = {"seed": seed, "n_in": n_in, "F": F, "chance": round(1.0 / F, 4),
           "n_train": int(tr.sum()), "n_held": int(is_held.sum())}
    # armA: linear (shallow) -- memorization/linear baseline
    la = _train_snn(Xtr_n, ytr, [n_in, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    out["armA_lin_train"], out["armA_lin_held"], _ = score_snn(la, Xev_n, yev, held, in_gain)
    # armB: DEEP 2-hidden e-prop (THE PICK)
    lb = _train_snn(Xtr_n, ytr, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    out["armB_eprop_train"], out["armB_eprop_held"], _ = score_snn(lb, Xev_n, yev, held, in_gain)
    # ceiling: 2-hidden BPTT (best credit) -> a NEGATIVE is interpretable (task vs credit-rule)
    if do_bptt:
        lc = _train_snn(Xtr_n, ytr, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="bptt")
        out["armB_bptt_train"], out["armB_bptt_held"], _ = score_snn(lc, Xev_n, yev, held, in_gain)
    # ANTI-CHEAT permuted: shuffle labels -> held must collapse to chance (proves it's the real dispatch rule)
    rp = np.random.default_rng(seed + 5); yperm = ytr[rp.permutation(len(ytr))]
    lp = _train_snn(Xtr_n, yperm, [n_in, hidden, hidden, F], T, epochs, lr, in_gain, seed, credit_mode="eprop")
    _, out["armB_permuted_held"], _ = score_snn(lp, Xev_n, yev, held, in_gain)
    # ANTI-CHEAT memorization floor: 1-NN on the RAW code over TRAIN combos -> held-out-combo (novel subject,qtype) ~ chance
    from numpy.linalg import norm
    def nn_pred(xq):
        d = [norm(xq - Xtr[i]) for i in range(len(Xtr))]; return ytr[int(np.argmin(d))]
    hi = np.where(is_held)[0]
    out["memfloor_held"] = round(float(np.mean([nn_pred(X[i]) == y[i] for i in hi])), 4) if len(hi) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--no-bptt", action="store_true")
    ap.add_argument("--hard", action="store_true", help="clean deep-credit systematicity: nonlinear XOR rule + held-out (cat,qtype) COMBINATIONS (1-NN + linear both fail)")
    ap.add_argument("--out", default="research/findings/raw/_learned_dispatch.json")
    a = ap.parse_args()
    rows = [run_one(s, hidden=a.hidden, epochs=a.epochs, do_bptt=not a.no_bptt, hard=a.hard) for s in a.seeds]
    for r in rows:
        print(f"[dispatch s{r['seed']}] chance={r['chance']} memfloor={r['memfloor_held']} | "
              f"armA_lin_held={r['armA_lin_held']:.3f} armB_EPROP_held={r['armB_eprop_held']:.3f} "
              f"(train {r['armB_eprop_train']:.3f}) armB_bptt_held={r.get('armB_bptt_held', float('nan')):.3f} "
              f"permuted_held={r['armB_permuted_held']:.3f}", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
