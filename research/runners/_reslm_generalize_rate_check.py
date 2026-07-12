"""Ceiling-first for the CORRECT instrument: next-token PREDICTION with distributional generalization (rate).

The cue-classification task is the WRONG instrument for the R3 learn-W_in benefit: classification of DISTINCT inputs
never needs a learned W_in (Johnson-Lindenstrauss: a fixed random projection already separates distinct codes -> fixed
ceilings at 1.000; see the 2026-07-12 overlap-check NEGATIVE). R3's headroom (learn +1.257 vs fixed -1.657 on real LM)
is a PREDICTION + GENERALIZATION phenomenon: the same tokens recur across contexts, and learning the embedding places
distributionally-similar tokens together so the fixed reservoir GENERALIZES to rare tokens.

This instrument reproduces that minimally:
- G classes x `syn` synonyms => V tokens. Class-mates SHARE `sf` code dims (the class feature) + each token has `id`
  unique identity dims that are CLASS-IRRELEVANT (a confound). m = G*sf-shared-block? no: shared class dims are per-class,
  identity dims per-token; all over one code space. (Overlap via the shared class block.)
- Markov class transition: current token's CLASS -> next CLASS (near-deterministic G x G). Task = predict NEXT class
  (G-way) from the current token, read at the step after the token.
- HELD-OUT: 1 rare synonym/class is EXCLUDED from train; at eval its class must be predicted from its SHARED class dims
  (seen via class-mates) despite novel identity dims. Fixed W_in: weights identity-confound dims equally => noisier
  generalization; learned W_in (input-synapse e-prop, no weight transport): can up-weight the shared class dims and
  suppress the confounding identity dims => cleaner held-out generalization. Metric = HELD-OUT next-class accuracy.

Gate: learn - fixed >= 0.10 on held-out (fixed <= 0.90) => headroom => build the spiking next-token version here.
numpy/CPU; NO sim/ edit, NO runner edit.

Run:  OMP_NUM_THREADS=4 python -u -m research.runners._reslm_generalize_rate_check --json raw/_reslm_generalize.json
"""
import argparse, json, os
import numpy as np


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def _fit_ridge(R, Y, n_classes, lam=1.0):
    N, d = R.shape
    T = np.zeros((N, n_classes)); T[np.arange(N), Y] = 1.0
    return np.linalg.solve(R.T @ R + lam * np.eye(d), R.T @ T)


def _decode_acc(R, Y, W):
    return float((R @ W).argmax(1).astype(int).__eq__(np.asarray(Y)).mean())


def build_codes(seed, G, syn, sf, idn):
    """m = G*sf (per-class shared blocks) + G*syn*idn (per-token identity). Each token code = its class's shared block
    (all-ones over that class's sf dims) + its own identity dims (ones over idn dims). Class-mates share the class block."""
    rng = np.random.default_rng(seed * 131 + 7)
    V = G * syn
    m = G * sf + V * idn
    codes = np.zeros((V, m))
    for c in range(G):
        cls_dims = np.arange(c * sf, (c + 1) * sf)
        for j in range(syn):
            v = c * syn + j
            codes[v, cls_dims] = 1.0                                   # shared class feature
            id0 = G * sf + v * idn
            codes[v, id0:id0 + idn] = 1.0                              # unique identity (class-irrelevant confound)
    return codes, V, m


def build_stream(seed, G, syn, n_seq, held_frac=0.2):
    """Near-deterministic class Markov: class c -> next class (c+1)%G with prob 0.85 else uniform. Each example =
    (current_token, next_class). 1 held-out synonym/class excluded from train (appears only in eval)."""
    rng = np.random.default_rng(seed * 977 + 3)
    held = {c: c * syn + (syn - 1) for c in range(G)}                  # last synonym of each class = held-out
    train, evl = [], []
    for c in range(G):
        for _ in range(n_seq):
            j = int(rng.integers(0, syn - 1))                          # train tokens = synonyms 0..syn-2
            cur = c * syn + j
            nc = (c + 1) % G if rng.random() < 0.85 else int(rng.integers(0, G))
            train.append((cur, nc))
        # held-out eval: the rare synonym, its true next class distribution
        for _ in range(max(4, n_seq // 4)):
            nc = (c + 1) % G if rng.random() < 0.85 else int(rng.integers(0, G))
            evl.append((held[c], nc))
    rng.shuffle(train)
    return train, evl


def rate_ref_generalize(train, evl, codes, m, n, G, seed, epochs, lr_out, lr_in,
                        alpha=0.3, noise=0.0, learn=True):
    rng = np.random.RandomState(seed * 3 + 17)
    W_rec = rng.normal(0, 1, (n, n)) / np.sqrt(n)
    sr = max(np.abs(np.linalg.eigvals(W_rec)))
    W_rec = W_rec * (0.95 / max(sr, 1e-6))
    W_in = rng.normal(0, 1, (n, m)) / np.sqrt(m)
    Bfb = rng.normal(0, 1, (n, G)); b = np.zeros(n); Wout = np.zeros((G, n))

    def _fwd(tok, collect, nrng):
        # 2-step: present token, then a blank "predict" step; read at the predict step.
        h = np.zeros(n); e_in = np.zeros((n, m)) if collect else None; hq = None
        for t in range(2):
            x = codes[tok] if t == 0 else np.zeros(m)
            pre = W_rec @ h + W_in @ x + b
            act = np.tanh(pre); h = (1 - alpha) * h + alpha * act
            if noise > 0.0:
                h = h + noise * nrng.standard_normal(n)
            if collect:
                psi = alpha * (1 - act * act); e_in = (1 - alpha) * e_in + np.outer(psi, x)
            if t == 1:
                hq = h.copy()
        return hq, e_in

    trng = np.random.RandomState(seed * 9 + 5)
    if learn:
        for _ep in range(epochs):
            order = list(range(len(train))); rng.shuffle(order)
            for si in order:
                tok, nc = train[si]
                hq, e_in = _fwd(tok, True, trng)
                p = _softmax(Wout @ hq); delta = -p; delta[nc] += 1.0
                Wout += lr_out * np.outer(delta, hq)
                L = Bfb @ delta
                W_in = W_in + lr_in * (L[:, None] * e_in)

    def _reads(sents):
        R, Y = [], []
        for tok, nc in sents:
            hq, _ = _fwd(tok, False, trng); R.append(np.concatenate([hq, [1.0]])); Y.append(nc)
        return np.asarray(R), np.asarray(Y)

    Rtr, Ytr = _reads(train); Rev, Yev = _reads(evl)
    W = _fit_ridge(Rtr, Ytr, G, lam=1.0)
    return _decode_acc(Rtr, Ytr, W), _decode_acc(Rev, Yev, W)   # (train acc, HELD-OUT acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr-out", type=float, default=0.02)
    ap.add_argument("--lr-in", type=float, default=0.05)
    ap.add_argument("--n-seq", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--json", type=str, default="raw/_reslm_generalize.json")
    args = ap.parse_args()

    # G classes, syn synonyms, sf shared-class dims, idn identity-confound dims, n reservoir
    CONFIGS = []
    for G in [6]:
        for syn in [5]:
            for sf in [2, 3]:
                for idn in [10, 20, 30]:     # bigger identity confound (idn >> sf) => learn's suppression matters more
                    for n in [50, 60]:
                        CONFIGS.append((G, syn, sf, idn, n))

    rows = []
    for (G, syn, sf, idn, n) in CONFIGS:
        lrn_tr, lrn_ho, fix_tr, fix_ho = [], [], [], []
        for seed in args.seeds:
            codes, V, m = build_codes(seed, G, syn, sf, idn)
            train, evl = build_stream(seed, G, syn, args.n_seq)
            ltr, lho = rate_ref_generalize(train, evl, codes, m, n, G, seed, args.epochs,
                                           args.lr_out, args.lr_in, noise=args.noise, learn=True)
            ftr, fho = rate_ref_generalize(train, evl, codes, m, n, G, seed, args.epochs,
                                           args.lr_out, args.lr_in, noise=args.noise, learn=False)
            lrn_tr.append(ltr); lrn_ho.append(lho); fix_tr.append(ftr); fix_ho.append(fho)
        L_ho, F_ho = float(np.mean(lrn_ho)), float(np.mean(fix_ho))
        rows.append({"G": G, "syn": syn, "sf": sf, "idn": idn, "n": n, "m": G * sf + V * idn,
                     "chance": round(1.0 / G, 3),
                     "learn_train": round(float(np.mean(lrn_tr)), 3), "learn_heldout": round(L_ho, 4),
                     "fixed_train": round(float(np.mean(fix_tr)), 3), "fixed_heldout": round(F_ho, 4),
                     "margin_heldout": round(L_ho - F_ho, 4)})

    rows.sort(key=lambda r: r["margin_heldout"], reverse=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump({"seeds": args.seeds, "noise": args.noise, "n_seq": args.n_seq, "rows": rows},
              open(args.json, "w"), indent=2)

    print(f"=== next-token GENERALIZATION: learn vs fixed W_in on HELD-OUT synonyms (rate, seeds {args.seeds}) ===", flush=True)
    print(f"{'G':>2}{'syn':>4}{'sf':>3}{'idn':>4}{'n':>4} | {'L_ho':>6}{'F_ho':>6}{'margin':>8}{'L_tr':>6}{'F_tr':>6}{'chance':>7}  verdict", flush=True)
    hits = 0
    for r in rows:
        hd = r["margin_heldout"] >= 0.10 and r["fixed_heldout"] <= 0.90 and r["learn_heldout"] > r["fixed_heldout"]
        v = "*** HEADROOM" if hd else ""
        hits += hd
        print(f"{r['G']:>2}{r['syn']:>4}{r['sf']:>3}{r['idn']:>4}{r['n']:>4} | {r['learn_heldout']:>6.3f}"
              f"{r['fixed_heldout']:>6.3f}{r['margin_heldout']:>+8.3f}{r['learn_train']:>6.2f}{r['fixed_train']:>6.2f}"
              f"{r['chance']:>7.3f}  {v}", flush=True)
    print(f"\n{hits} config(s) with held-out GENERALIZATION headroom (margin>=0.10, fixed<=0.90).", flush=True)
    if hits:
        b = next(r for r in rows if r["margin_heldout"] >= 0.10 and r["fixed_heldout"] <= 0.90)
        print(f"BEST -> G={b['G']} syn={b['syn']} sf={b['sf']} idn={b['idn']} n={b['n']}: held-out learn {b['learn_heldout']:.3f} "
              f"vs fixed {b['fixed_heldout']:.3f} (margin {b['margin_heldout']:+.3f}). This is the correct spiking instrument.", flush=True)
    else:
        print("NO held-out generalization headroom -> fixed W_in already generalizes via the shared class dims (JL again). "
              "Next lever: stronger identity confound (idn>>sf), correlated-wrong-class identity dims, or noise.", flush=True)


if __name__ == "__main__":
    main()
