"""Ceiling-first headroom map for the learn-W_in-on-a-fixed-reservoir rung (rate level, cheap, reuse-by-import).

WHY (the ceiling-first discipline): the spiking BDSP learn-W_in runner's shipped rate reference returns
`learn 1.000 vs fixed 1.000` at (n_cues=12, dist=3) — NO headroom, so it cannot tell whether learning the
input projection HELPS. Root cause: that rate reference is DETERMINISTIC + NOISELESS (build_task with no
jitter => all examples per cue are the identical token sequence; the leaky-tanh reservoir has no noise), so
12 cues always map to 12 distinct reads a ridge separates trivially at ANY dist. The SPIKING version has real
membrane/Poisson noise => genuine cue collision after fading. So the shipped rate ceiling is NOT a like-for-like
difficulty proxy for the spiking task. This sweep adds matched difficulty (within-class filler JITTER + reservoir
state NOISE + larger K + smaller pool) and finds the regime where learning W_in genuinely beats a fixed random
W_in (real headroom). THAT regime is where the spiking run is meaningful.

Reuses `build_task` from the spiking runner; local noisy rate reference (same leaky-tanh + input-synapse e-prop
rule as the runner's rate_reference, plus a noise term). NO runner edit, NO sim/ edit. numpy/CPU.

Run:  OMP_NUM_THREADS=4 python -u -m research.runners._reslm_rate_headroom_sweep --json raw/_reslm_headroom.json
"""
import argparse, json, os
import numpy as np
from research.runners._reslm_onbridge_learn_win_derisk import build_task


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _fit_ridge(R, Y, n_classes, lam=1.0):
    N, d = R.shape
    T = np.zeros((N, n_classes))
    T[np.arange(N), Y] = 1.0
    A = R.T @ R + lam * np.eye(d)
    return np.linalg.solve(A, R.T @ T)


def _decode_acc(R, Y, W):
    return float((R @ W).argmax(1).astype(int).__eq__(np.asarray(Y)).mean())


def rate_ref_noisy(train, evl, V, n, n_classes, seed, epochs, lr_out, lr_in,
                   alpha=0.3, noise=0.0, learn=True):
    """Leaky-tanh reservoir, FIXED W_rec (spectral radius 0.95), W_in learned by input-synapse e-prop
    (broadcast random feedback, no weight transport). `noise` adds Gaussian state noise each step =
    a like-for-like difficulty proxy for the spiking version's membrane noise. Ridge decode on query reads."""
    rng = np.random.RandomState(seed * 3 + 17)
    W_rec = rng.normal(0, 1, (n, n)) / np.sqrt(n)
    sr = max(np.abs(np.linalg.eigvals(W_rec)))
    W_rec = W_rec * (0.95 / max(sr, 1e-6))
    W_in = rng.normal(0, 1, (n, V)) / np.sqrt(V)
    Bfb = rng.normal(0, 1, (n, n_classes))
    b = np.zeros(n)
    Wout = np.zeros((n_classes, n))

    def _fwd(toks, collect_elig, nrng):
        h = np.zeros(n)
        e_in = np.zeros((n, V)) if collect_elig else None
        hq = None
        for t, tok in enumerate(toks):
            x = np.zeros(V); x[tok] = 1.0
            pre = W_rec @ h + W_in @ x + b
            act = np.tanh(pre)
            h = (1 - alpha) * h + alpha * act
            if noise > 0.0:
                h = h + noise * nrng.standard_normal(n)          # matched-difficulty state noise
            if collect_elig:
                psi = alpha * (1 - act * act)
                e_in = (1 - alpha) * e_in + np.outer(psi, x)
            if t == len(toks) - 1:
                hq = h.copy()
        return hq, e_in

    trng = np.random.RandomState(seed * 9 + 5)
    if learn:
        for _ep in range(epochs):
            order = list(range(len(train))); rng.shuffle(order)
            for si in order:
                toks, k = train[si]
                hq, e_in = _fwd(toks, True, trng)
                p = _softmax(Wout @ hq); delta = -p; delta[k] += 1.0
                Wout += lr_out * np.outer(delta, hq)
                L = Bfb @ delta
                W_in = W_in + lr_in * (L[:, None] * e_in)

    # metric: freeze, clean ridge on the query reads (noise on, matched to train).
    def _reads(sents):
        R, Y = [], []
        for toks, k in sents:
            hq, _ = _fwd(toks, False, trng)
            R.append(np.concatenate([hq, [1.0]])); Y.append(k)
        return np.asarray(R), np.asarray(Y)

    Rtr, Ytr = _reads(train); Rev, Yev = _reads(evl)
    W = _fit_ridge(Rtr, Ytr, n_classes, lam=1.0)
    return _decode_acc(Rev, Yev, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--n-per-cue", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr-out", type=float, default=0.02)
    ap.add_argument("--lr-in", type=float, default=0.05)
    ap.add_argument("--json", type=str, default="raw/_reslm_headroom.json")
    args = ap.parse_args()

    # difficulty grid: dist (cue->query fillers) x jitter (within-class variance) x noise x (n_cues, n_pool)
    DISTS = [3, 6, 10, 14, 20]
    JITTERS = [0, 3]
    NOISES = [0.0, 0.05, 0.15]
    NK = [(12, 120), (30, 80)]           # (n_cues, reservoir n): easy vs harder (more cues, smaller pool)

    rows = []
    for (n_cues, n_pool) in NK:
        for dist in DISTS:
            for jit in JITTERS:
                for noise in NOISES:
                    learn_accs, fixed_accs = [], []
                    for seed in args.seeds:
                        train, evl, V, FILL, QRY = build_task(seed, n_cues, args.n_per_cue, dist, dist_jitter=jit)
                        la = rate_ref_noisy(train, evl, V, n_pool, n_cues, seed, args.epochs,
                                            args.lr_out, args.lr_in, noise=noise, learn=True)
                        fa = rate_ref_noisy(train, evl, V, n_pool, n_cues, seed, args.epochs,
                                            args.lr_out, args.lr_in, noise=noise, learn=False)
                        learn_accs.append(la); fixed_accs.append(fa)
                    lm, fm = float(np.mean(learn_accs)), float(np.mean(fixed_accs))
                    chance = 1.0 / n_cues
                    rows.append({"n_cues": n_cues, "n_pool": n_pool, "dist": dist, "jitter": jit,
                                 "noise": noise, "chance": round(chance, 3),
                                 "learn": round(lm, 4), "fixed": round(fm, 4), "margin": round(lm - fm, 4)})

    rows.sort(key=lambda r: r["margin"], reverse=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"seeds": args.seeds, "n_per_cue": args.n_per_cue, "epochs": args.epochs,
                   "lr_in": args.lr_in, "lr_out": args.lr_out, "rows": rows}, f, indent=2)

    print(f"=== learn-W_in vs fixed-W_in headroom map (rate, seeds {args.seeds}) ===", flush=True)
    print(f"{'nK':>4} {'nP':>4} {'dist':>4} {'jit':>3} {'noise':>5} | {'learn':>6} {'fixed':>6} "
          f"{'margin':>7} {'chance':>6}   verdict", flush=True)
    hits = 0
    for r in rows:
        # HEADROOM = learn clearly beats fixed AND fixed is genuinely below ceiling (real collision, not both-perfect)
        headroom = r["margin"] >= 0.10 and r["fixed"] <= 0.90 and r["learn"] > r["fixed"]
        verdict = "*** HEADROOM" if headroom else ("no-collision" if r["fixed"] > 0.90 else "")
        hits += headroom
        print(f"{r['n_cues']:>4} {r['n_pool']:>4} {r['dist']:>4} {r['jitter']:>3} {r['noise']:>5} | "
              f"{r['learn']:>6.3f} {r['fixed']:>6.3f} {r['margin']:>+7.3f} {r['chance']:>6.3f}   {verdict}", flush=True)
    print(f"\n{hits} config(s) show genuine rate-level HEADROOM (margin>=0.10, fixed<=0.90).", flush=True)
    if hits:
        best = next(r for r in rows if r["margin"] >= 0.10 and r["fixed"] <= 0.90)
        print(f"BEST headroom regime -> n_cues={best['n_cues']} n_pool={best['n_pool']} dist={best['dist']} "
              f"jitter={best['jitter']} noise={best['noise']}  (learn {best['learn']:.3f} vs fixed {best['fixed']:.3f}, "
              f"margin {best['margin']:+.3f}). Run the SPIKING learn_win arm at this task difficulty.", flush=True)
    else:
        print("NO rate headroom in this grid -> the fixed reservoir separates the cues at every tested difficulty; "
              "learning W_in is not needed HERE. Next lever: OVERLAPPING SPARSE input codes (shared active dims per "
              "cue => fixed W_in projection collides even at dist=0; learn must de-correlate) -- the named rung.", flush=True)


if __name__ == "__main__":
    main()
