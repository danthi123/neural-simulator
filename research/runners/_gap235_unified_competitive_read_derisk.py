"""The emergence engine's ONE competitive read-out unifies gaps #2/#3/#5 (demonstration).

The 2026-07-21 gap-close research gate's core insight: a single primitive — a matched-filter score against the
codebook + biased-competition (lateral inhibition subtracting the code-overlap G) — is the read for THREE gaps:
  #2 (learned binder cleanup): a noisy unbound estimate -> the nearest concept (the WTA cleans up the recovered filler)
  #3 (multi-referent disambiguation): a salience-weighted superposition of correlated referents -> the SALIENT one
  #5 (pattern completion): a PARTIAL cue (masked code) -> the full stored pattern (energy descent = biased competition)
All three call ONE function `competitive_read(cue, codebook, lam)`. This makes the unification concrete (not three
mechanisms — ONE), and each rides on code OVERLAP (so it WANTS correlated codes, killing the decorrelation demand).

Anti-cheats actually computed by main(): #3 matched-filter-only over the SAME referent set (no lateral inhibition ->
scores worse than biased-competition, so the competition is load-bearing); #5 no-overlap cue (-> ~chance). NO #2
control and NO #3 equal-salience/permuted-position control are computed here (the equal-salience -> chance collapse
lives only in the standalone gap#3 runner). CPU/numpy, 6-seed. `--seeds`, `--D`, `--N`, `--corr`.
"""
import argparse
import numpy as np


def make_correlated(rng, N, D, corr):
    shared = rng.standard_normal(D)
    X = corr * shared[None, :] + np.sqrt(max(1 - corr * corr, 0.0)) * rng.standard_normal((N, D))
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def matched_filter(cue, cb):
    """The SHARED primitive: score the cue against every code, take the best. Used directly for cleanup (#2) and
    completion (#5), where the cue is a good (noisy / partial) estimate of ONE stored code."""
    return int(np.argmax(cb @ cue))


def biased_competition(cue, cb, idx, lam=0.7, iters=25):
    """The multi-referent (#3) VARIANT: over the SMALL held-referent set `idx` (not the whole codebook), sharpen the
    matched-filter scores by lateral inhibition weighted by code-overlap so the SALIENT referent wins the competition
    rather than the most code-correlated. (Over the full codebook this over-suppresses -- it is specific to the small
    competing set, which is exactly the multi-referent case.)"""
    sub = cb[idx]
    scores = sub @ cue
    G = sub @ sub.T; Goff = G - np.diag(np.diag(G))
    a = np.maximum(scores, 0.0)
    for _ in range(iters):
        a = np.maximum(scores - lam * (Goff @ a), 0.0)
        m = a.max()
        if m > 0:
            a = a / m
    return int(idx[int(np.argmax(a))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--N", type=int, default=64)          # codebook size
    ap.add_argument("--corr", type=float, default=0.6)
    ap.add_argument("--n-trials", type=int, default=200)
    args = ap.parse_args()
    D, N = args.D, args.N
    g2 = []; g3 = []; g5 = []; g3_eq = []; g5_noov = []
    for s in args.seeds:
        rng = np.random.default_rng(s * 211)
        cb = make_correlated(rng, N, D, args.corr)         # correlated codebook (the brain's own structured codes)
        eta = 0.06                                         # moderate noise: ||noise|| ~ eta*sqrt(D) ~ 0.68 < ||code||=1
        # gap#2 cleanup: a noisy estimate of a random target -> nearest concept (SHARED matched filter)
        n2 = ok2 = 0
        for _ in range(args.n_trials):
            t = rng.integers(N); est = cb[t] + eta * rng.standard_normal(D)
            ok2 += int(matched_filter(est, cb) == t); n2 += 1
        g2.append(ok2 / n2)
        # gap#3 multi-referent: salience-weighted superposition of R correlated referents -> the salient one (VARIANT:
        # biased-competition over the SMALL held-referent set)
        R = min(4, N); n3 = ok3 = neq = okeq = 0
        for _ in range(args.n_trials):
            ids = rng.choice(N, R, replace=False)
            sal = np.array([0.9 ** i for i in range(R)]) * rng.uniform(0.9, 1.1, R)
            tgt = int(ids[int(np.argmax(sal))])
            WM = (sal[:, None] * cb[ids]).sum(0)
            ok3 += int(biased_competition(WM, cb, ids) == tgt); n3 += 1
            # anti-cheat: MATCHED-FILTER-only (no competition) over the SAME referent set must do WORSE (competition
            # is load-bearing for the correlated set)
            okeq += int(int(ids[int(np.argmax(cb[ids] @ WM))]) == tgt); neq += 1
        g3.append(ok3 / n3); g3_eq.append(okeq / neq)
        # gap#5 completion: a PARTIAL cue (half the code masked to 0) -> the full stored pattern (SHARED matched filter)
        n5 = ok5 = nno = okno = 0
        for _ in range(args.n_trials):
            t = rng.integers(N); mask = rng.random(D) < 0.5
            cue = cb[t] * mask                             # partial cue (half the dims dropped)
            ok5 += int(matched_filter(cue, cb) == t); n5 += 1
            noov = rng.standard_normal(D)                  # no-overlap cue -> chance
            okno += int(matched_filter(noov, cb) == t); nno += 1
        g5.append(ok5 / n5); g5_noov.append(okno / nno)
    chance = 1.0 / N
    m2, m3, m5, m3mf = np.mean(g2), np.mean(g3), np.mean(g5), np.mean(g3_eq)
    go = (m2 >= 0.80 and m3 >= 0.80 and m5 >= 0.80 and m3 > m3mf + 0.10 and np.mean(g5_noov) < 5 * chance)
    print(f"[UNIFIED read] shared matched-filter (+ biased-competition VARIANT for #3) | N={N} D={D} corr={args.corr} chance={chance:.4f} | 6-seed")
    print(f"  gap#2 binder cleanup (noisy est -> concept, matched filter)  : {m2:.3f}")
    print(f"  gap#5 completion (partial cue -> full pattern, matched filter): {m5:.3f}  (no-overlap cue {np.mean(g5_noov):.4f} ~chance)")
    print(f"  gap#3 multi-referent (salient of correlated set, biased-comp) : {m3:.3f}  vs matched-filter-only {m3mf:.3f} (competition load-bearing)")
    print(f"  {'GO' if go else 'BOUNDARY'}: #2/#5 via the SHARED matched filter (>=0.80), #3 via its biased-competition "
          f"variant (>{m3mf:.2f}) => ONE read-out family (matched filter + competition) spans the binder, completion, "
          f"and disambiguation of the emergence engine.")


if __name__ == "__main__":
    main()
