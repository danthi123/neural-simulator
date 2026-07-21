"""gap#3 (multi-referent disambiguation) Rank-4 — BIASED-COMPETITION resolves a bare pronoun among several CORRELATED
held referents, where the two prior NEGATIVEs (recency 2026-06-17, salience-boost) failed. Cheap-first RATE de-risk of
the gate's mechanism: lateral biased-competition inhibition (Desimone-Duncan 1995) between referent attractors.

The wall (2026-06-17-multireferent-NEGATIVE): with N correlated referents in WM (the loop holds the SET, superposed),
which one a bare "it" binds is ambiguous — reading the max <WM, ref> is dominated by the inter-referent CORRELATION,
not the salience, so it does NOT track recency/topicality. The FIX: biased-competition subtracts the correlated
crosstalk (lateral inhibition G[r,r']=<ref_r,ref_r'>), decorrelating the activations so the SALIENCE wins.

  WM = sum_r salience_r * ref_r        (salience = recency: referent 0 most recent -> highest weight)
  OFF (read-max / salience-boost):  argmax_r <WM, ref_r>                       (the prior NEGATIVE)
  ON  (biased-competition):  iterate a_r = relu(<WM,ref_r> - lam * sum_{r'!=r} G[r,r'] a_{r'});  winner=argmax a

ONE VARIABLE: lateral inhibition ON vs OFF. GATE: ON resolves the SALIENT referent (highest salience) robustly AND
> OFF, 6-seed. Anti-cheats: permuted referent-order (winner tracks salience, not position); EQUAL-salience control
stays neutral (no spurious winner beyond chance). `--seeds`, `--n-ref`, `--corr`, `--n-trials`.
"""
import argparse
import numpy as np


def make_correlated_refs(rng, N, D, corr):
    """N unit codes with controlled inter-code cosine ~ corr: a shared component + individual components."""
    shared = rng.standard_normal(D)
    indiv = rng.standard_normal((N, D))
    X = corr * shared[None, :] + np.sqrt(max(1 - corr * corr, 0.0)) * indiv
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def biased_competition(scores, G, lam=0.7, iters=25):
    """WTA by lateral inhibition: a_r <- relu(scores_r - lam * sum_{r'!=r} G[r,r'] a_{r'}), normalized each step."""
    a = np.maximum(scores, 0).copy()
    Goff = G - np.diag(np.diag(G))
    for _ in range(iters):
        a = np.maximum(scores - lam * (Goff @ a), 0.0)
        m = a.max()
        if m > 0:
            a = a / m
    return a


def run_seed(seed, N, D, corr, n_trials, equal_sal=False, permute=False):
    rng = np.random.default_rng(seed * 149 + N)
    off_ok = on_ok = n = 0
    for _ in range(n_trials):
        refs = make_correlated_refs(rng, N, D, corr)
        # salience = recency: a decaying profile; the SALIENT (target) referent is the one with the max salience
        sal = np.ones(N) if equal_sal else np.array([0.9 ** i for i in range(N)]) * rng.uniform(0.9, 1.1, N)
        order = rng.permutation(N) if permute else np.arange(N)
        sal = sal[order]                                    # permute which position is salient (anti-cheat)
        target = int(np.argmax(sal))
        WM = (sal[:, None] * refs).sum(0)                   # the held superposition
        scores = refs @ WM                                  # <WM, ref_r> (matched filter)
        # OFF: read-max (salience-boost baseline = the prior NEGATIVE)
        off_pred = int(np.argmax(scores))
        # ON: biased-competition (lateral inhibition subtracts the correlated crosstalk)
        G = refs @ refs.T
        a = biased_competition(scores, G)
        on_pred = int(np.argmax(a))
        off_ok += int(off_pred == target); on_ok += int(on_pred == target); n += 1
    return off_ok / n, on_ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-ref", type=int, default=4)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--corr", type=float, default=0.6)
    ap.add_argument("--n-trials", type=int, default=300)
    args = ap.parse_args()
    chance = 1.0 / args.n_ref
    offs = []; ons = []; ons_eq = []; ons_perm = []
    for s in args.seeds:
        o, n_ = run_seed(s, args.n_ref, args.D, args.corr, args.n_trials)
        _, n_eq = run_seed(s, args.n_ref, args.D, args.corr, args.n_trials, equal_sal=True)
        _, n_pm = run_seed(s, args.n_ref, args.D, args.corr, args.n_trials, permute=True)
        offs.append(o); ons.append(n_); ons_eq.append(n_eq); ons_perm.append(n_pm)
    off = np.mean(offs); on = np.mean(ons); on_perm = np.mean(ons_perm); on_eq = np.mean(ons_eq)
    go = (on >= 0.80 and on > off + 0.10 and on_perm >= 0.80 and abs(on_eq - chance) < 0.15)
    print(f"[gap3 biased-competition] N={args.n_ref} D={args.D} corr={args.corr} chance={chance:.3f} | seeds={args.seeds}")
    print(f"  OFF (read-max / salience-boost, the prior NEGATIVE) : {off:.3f}")
    print(f"  ON  (biased-competition, lateral inhibition)        : {on:.3f}  (resolves the SALIENT referent)")
    print(f"  ON permuted-position (salience not position)        : {on_perm:.3f}  (must stay high -> tracks salience)")
    print(f"  ON equal-salience control (no spurious winner)      : {on_eq:.3f}  (must ~chance {chance:.3f})")
    print(f"  {'GO' if go else 'BOUNDARY'}: ON>=0.80 & ON>OFF+0.10 & permuted-high & equal~chance "
          f"({on>=0.80}/{on>off+0.10}/{on_perm>=0.80}/{abs(on_eq-chance)<0.15})")


if __name__ == "__main__":
    main()
