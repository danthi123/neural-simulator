"""THROWAWAY (raw/): is the near-orthogonality boundary CAPACITY-limited (fewer
concepts -> reachable -> a richer substrate with more dims/concept would help ->
the months-scale richer-substrate fork has merit) or FUNDAMENTAL (unreachable at
any concept count -> accept the oracle)? Decision-relevant for the strategic fork.

Cheap: for N concepts (subset the 16), random-project (the clean reliable method)
+ measure between-concept cosine (separation) + within-concept (reliability) of
the codes. If between drops below near-ortho (~0.30) for small N while within
stays high -> capacity-limited (more dims/concept = richer substrate helps). If
between stays > 0.30 even at N=4 -> fundamental to the activity structure.

stdlib+numpy + cached activity. No protected import.
"""
from __future__ import annotations
import os
import numpy as np

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
SEEDS = [42, 43, 44]
N_OUT = 200
RP_K = 16
N_LIST = [4, 8, 12, 16]
NEAR = 0.30


def _cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _rn(v):
    v = np.maximum(v.astype(np.float64), 0.0)
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    seeds = [s for s in SEEDS if os.path.exists(CACHE % s)]
    print("=== near-ortho frontier vs concept-count (random-proj k-WTA) ===", flush=True)
    print(f"seeds={seeds} N_OUT={N_OUT} k={RP_K}; near-ortho bar between < {NEAR}", flush=True)
    if not seeds:
        print("VERDICT: CANNOT-CONCLUDE (no caches)", flush=True)
        return

    print(f"\n{'N':>3} | {'raw_btw':>8} {'rp_btw':>8} {'rp_within':>9}", flush=True)
    rp_btw_byN = {}
    for N in N_LIST:
        rb, pb, pw = [], [], []
        for s in seeds:
            d = np.load(CACHE % s)
            words = [k[5:] for k in d.files if k.startswith("obs__")][:N]
            store = {w: _rn(d["obs__" + w][:32].mean(0)) for w in words}
            query = {w: _rn(d["obs__" + w][32:].mean(0)) for w in words}
            rb += [_cos(store[a], store[b]) for i, a in enumerate(words) for b in words[i+1:]]
            rng = np.random.default_rng(1000 + s)
            P = rng.standard_normal((N_OUT, store[words[0]].shape[0]))

            def enc(x):
                v = np.zeros(N_OUT); v[np.argpartition(-(P @ x), RP_K)[:RP_K]] = 1.0
                return v
            cs = {w: enc(store[w]) for w in words}
            cq = {w: enc(query[w]) for w in words}
            pb += [_cos(cs[a], cs[b]) for i, a in enumerate(words) for b in words[i+1:]]
            pw += [_cos(cs[w], cq[w]) for w in words]
        rp_btw_byN[N] = float(np.mean(pb))
        print(f"{N:>3} | {np.mean(rb):>8.3f} {np.mean(pb):>8.3f} {np.mean(pw):>9.3f}", flush=True)

    b4, b16 = rp_btw_byN[4], rp_btw_byN[16]
    print(f"\n[trend] random-proj between: N=4 {b4:.3f} -> N=16 {b16:.3f} (delta {b16-b4:+.3f})", flush=True)
    if b4 < NEAR <= b16:
        print("VERDICT: CAPACITY-LIMITED -- near-ortho is reachable at small N (%.3f<%.2f) but not at "
              "N=16 (%.3f). The boundary scales with concept count -> a RICHER SUBSTRATE (more "
              "dims/concept) would push the frontier toward near-ortho; the months-scale richer-substrate "
              "fork has genuine merit. Decision-relevant: richer substrate is worth owner consideration."
              % (b4, NEAR, b16), flush=True)
    elif b4 >= NEAR:
        print("VERDICT: FUNDAMENTAL -- near-ortho is NOT reachable even at N=4 (%.3f >= %.2f). The boundary "
              "is intrinsic to the activity structure, not capacity -> a richer substrate (more concepts/"
              "dims) would NOT help reach near-ortho; accept the oracle near-ortho code as the engineering "
              "component. Decision-relevant: the richer-substrate fork is LOW-merit; advance P4 with the oracle."
              % (b4, NEAR), flush=True)
    else:
        print("VERDICT: MIXED -- near-ortho reachable across the range or trend unclear; "
              f"N=4 {b4:.3f}, N=16 {b16:.3f} (see numbers).", flush=True)


if __name__ == "__main__":
    main()
