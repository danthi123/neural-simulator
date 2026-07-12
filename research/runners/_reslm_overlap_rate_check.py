"""Ceiling-first for the OVERLAPPING-SPARSE-CODE instrument (rate, deterministic, noise-free by default).

The K-cue one-hot task is a deterministic LOOKUP => a fixed random W_in never collides distinct cues, so learning
W_in shows no headroom (see 2026-07-12 finding). The faithful, noise-independent instrument = OVERLAPPING SPARSE
CODES under a BOTTLENECK: each cue = a sparse s-of-m code (codes OVERLAP because K*s > m), and the reservoir is
smaller than the code space (n < m), so the FIXED random W_in (n x m) is a lossy random compression that ALIASES
confusable codes -> fixed decode drops below ceiling; a LEARNED W_in (input-synapse e-prop, broadcast random
feedback, no weight transport) organizes the projection to preserve the discriminative directions -> learn decode
higher. This is the R3 thesis ("the input representation is the lever") in its cleanest transferable form, and the
collision is STRUCTURAL (no noise) so it transfers to the spiking reservoir.

Gate: find (m, s, K, n) where learn - fixed >= 0.10 AND fixed <= 0.90 (genuine collision) at noise=0. That regime
is where the spiking overlapping-code run is meaningful. numpy/CPU; NO sim/ edit, NO runner edit.

Run:  OMP_NUM_THREADS=4 python -u -m research.runners._reslm_overlap_rate_check --json raw/_reslm_overlap.json
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


def build_overlap_task(seed, n_cues, s, m, dist, n_per_cue, jitter=0):
    """Each cue k = a fixed random s-of-m sparse code (codes overlap when K*s > m). Sequence per example =
    [CUE_CODE] FILLER*dist [QUERY]. Input dims: 0..m-1 = code dims, m = FILLER, m+1 = QUERY. Label = k at QUERY."""
    rng = np.random.default_rng(seed * 5227 + 11)
    codes = np.zeros((n_cues, m))
    for k in range(n_cues):
        act = rng.choice(m, size=s, replace=False)
        codes[k, act] = 1.0
    FILL, QRY, D = m, m + 1, m + 2
    train, evl = [], []
    for k in range(n_cues):
        for j in range(n_per_cue):
            d = max(0, int(dist + (rng.integers(-jitter, jitter + 1) if jitter > 0 else 0)))
            (evl if j == 0 else train).append((k, d))
    rng.shuffle(train)
    return train, evl, codes, D, FILL, QRY


def rate_ref_overlap(train, evl, codes, D, FILL, QRY, n, n_classes, seed, dist,
                     epochs, lr_out, lr_in, alpha=0.3, noise=0.0, learn=True):
    rng = np.random.RandomState(seed * 3 + 17)
    W_rec = rng.normal(0, 1, (n, n)) / np.sqrt(n)
    sr = max(np.abs(np.linalg.eigvals(W_rec)))
    W_rec = W_rec * (0.95 / max(sr, 1e-6))
    W_in = rng.normal(0, 1, (n, D)) / np.sqrt(D)            # n x (m+2): the lossy random compression when n < m
    Bfb = rng.normal(0, 1, (n, n_classes))
    b = np.zeros(n); Wout = np.zeros((n_classes, n))

    def _seq(k, d):
        cue = np.zeros(D); cue[:codes.shape[1]] = codes[k]   # cue step = the multi-hot code, padded to D dims
        x = [cue]
        for _ in range(d):
            f = np.zeros(D); f[FILL] = 1.0; x.append(f)
        q = np.zeros(D); q[QRY] = 1.0; x.append(q)
        return x

    def _fwd(k, d, collect, nrng):
        h = np.zeros(n); e_in = np.zeros((n, D)) if collect else None; hq = None
        seq = _seq(k, d)
        for t, x in enumerate(seq):
            pre = W_rec @ h + W_in @ x + b
            act = np.tanh(pre)
            h = (1 - alpha) * h + alpha * act
            if noise > 0.0:
                h = h + noise * nrng.standard_normal(n)
            if collect:
                psi = alpha * (1 - act * act)
                e_in = (1 - alpha) * e_in + np.outer(psi, x)
            if t == len(seq) - 1:
                hq = h.copy()
        return hq, e_in

    trng = np.random.RandomState(seed * 9 + 5)
    if learn:
        for _ep in range(epochs):
            order = list(range(len(train))); rng.shuffle(order)
            for si in order:
                k, d = train[si]
                hq, e_in = _fwd(k, d, True, trng)
                p = _softmax(Wout @ hq); delta = -p; delta[k] += 1.0
                Wout += lr_out * np.outer(delta, hq)
                L = Bfb @ delta
                W_in = W_in + lr_in * (L[:, None] * e_in)

    def _reads(sents):
        R, Y = [], []
        for k, d in sents:
            hq, _ = _fwd(k, d, False, trng); R.append(np.concatenate([hq, [1.0]])); Y.append(k)
        return np.asarray(R), np.asarray(Y)

    Rtr, Ytr = _reads(train); Rev, Yev = _reads(evl)
    W = _fit_ridge(Rtr, Ytr, n_classes, lam=1.0)
    return _decode_acc(Rev, Yev, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--n-per-cue", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr-out", type=float, default=0.02)
    ap.add_argument("--lr-in", type=float, default=0.05)
    ap.add_argument("--dist", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--json", type=str, default="raw/_reslm_overlap.json")
    args = ap.parse_args()

    # bottleneck grid: n reservoir < m code-dims, K cues, s active. Overlap = K*s / m; bottleneck = m / n.
    CONFIGS = []
    for n in [40, 60]:
        for m in [120, 200]:
            for s in [8, 20]:
                for K in [30, 60]:
                    CONFIGS.append((n, m, s, K))

    rows = []
    for (n, m, s, K) in CONFIGS:
        learn_accs, fixed_accs = [], []
        for seed in args.seeds:
            tr, ev, codes, D, FILL, QRY = build_overlap_task(seed, K, s, m, args.dist, args.n_per_cue)
            la = rate_ref_overlap(tr, ev, codes, D, FILL, QRY, n, K, seed, args.dist,
                                  args.epochs, args.lr_out, args.lr_in, noise=args.noise, learn=True)
            fa = rate_ref_overlap(tr, ev, codes, D, FILL, QRY, n, K, seed, args.dist,
                                  args.epochs, args.lr_out, args.lr_in, noise=args.noise, learn=False)
            learn_accs.append(la); fixed_accs.append(fa)
        lm, fm = float(np.mean(learn_accs)), float(np.mean(fixed_accs))
        rows.append({"n": n, "m": m, "s": s, "K": K, "overlap_ratio": round(K * s / m, 2),
                     "bottleneck": round(m / n, 2), "chance": round(1.0 / K, 3),
                     "learn": round(lm, 4), "fixed": round(fm, 4), "margin": round(lm - fm, 4)})

    rows.sort(key=lambda r: r["margin"], reverse=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump({"seeds": args.seeds, "dist": args.dist, "noise": args.noise, "lr_in": args.lr_in, "rows": rows},
              open(args.json, "w"), indent=2)

    print(f"=== overlapping-code learn-W_in vs fixed-W_in (rate, noise={args.noise}, seeds {args.seeds}) ===", flush=True)
    print(f"{'n':>3}{'m':>4}{'s':>3}{'K':>4}{'ovl':>5}{'bneck':>6} | {'learn':>6}{'fixed':>6}{'margin':>8}{'chance':>7}  verdict", flush=True)
    hits = 0
    for r in rows:
        hd = r["margin"] >= 0.10 and r["fixed"] <= 0.90 and r["learn"] > r["fixed"]
        v = "*** HEADROOM" if hd else ("no-collision" if r["fixed"] > 0.90 else "")
        hits += hd
        print(f"{r['n']:>3}{r['m']:>4}{r['s']:>3}{r['K']:>4}{r['overlap_ratio']:>5}{r['bottleneck']:>6} | "
              f"{r['learn']:>6.3f}{r['fixed']:>6.3f}{r['margin']:>+8.3f}{r['chance']:>7.3f}  {v}", flush=True)
    print(f"\n{hits} config(s) with STRUCTURAL headroom (noise={args.noise}, margin>=0.10, fixed<=0.90).", flush=True)
    if hits:
        b = next(r for r in rows if r["margin"] >= 0.10 and r["fixed"] <= 0.90)
        print(f"BEST -> n={b['n']} m={b['m']} s={b['s']} K={b['K']} (overlap {b['overlap_ratio']}, bottleneck {b['bottleneck']}): "
              f"learn {b['learn']:.3f} vs fixed {b['fixed']:.3f}, margin {b['margin']:+.3f}. Build the spiking overlapping-code run here.", flush=True)
    else:
        print("NO structural headroom at noise=0 -> even overlapping codes stay ridge-separable under this bottleneck; "
              "next: stronger bottleneck (n<<m), more overlap (K*s>>m), or add small noise.", flush=True)


if __name__ == "__main__":
    main()
