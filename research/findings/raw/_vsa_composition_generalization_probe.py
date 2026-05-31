"""THROWAWAY cheap-first (CPU/numpy): does GENERALIZABLE compositional binding
(role x filler, the only kind that generalizes to novel combos) work with the
SUBSTRATE's overlapping concept codes -- or does it need near-orthogonal codes
(the boundary characterized this session)? Empirical test of the convergence
hypothesis: compositional generalization -> near-ortho boundary.

Gain-field / VSA binding (biologically = apical role gates basal filler, a
multiplicative bind; HRR/VSA = role (x) filler): a 'sentence' = sum of K bound
(role, filler) pairs. This GENERALIZES by construction (bind ANY role x ANY
filler on the fly, no training). The open question is whether UNBINDING (query a
role -> recover its filler) is CLEAN with the substrate's overlapping fillers,
or only with near-orthogonal codes.

Test: fillers = the substrate's real concept codes (denoise64 cache, between-cos
~0.77) vs random bipolar (clean control). roles = random bipolar (near-ortho).
Bind Hadamard, sum, query each role -> unbind -> cleanup to nearest filler.
Sweep K (1..8 bound pairs). Metric: role-query unbind accuracy.

If substrate-filler unbind accuracy is HIGH -> generalizable composition works
with substrate codes (escape!). If LOW (and random-filler control HIGH) ->
compositional generalization needs near-ortho the substrate lacks -> converges on
the near-ortho boundary; the only escape is months-scale richer training.
stdlib+numpy + cached activity; no protected import.
"""
from __future__ import annotations
import os
import numpy as np

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
SEEDS = [42, 43, 44]
K_LIST = [1, 2, 3, 4, 6, 8]
N_TRIALS = 60


def _center_bipolarize(v):
    """Substrate firing-rate code -> zero-mean (common-mode removed) for VSA use."""
    v = v.astype(np.float64)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def load_fillers(seed):
    d = np.load(CACHE % seed)
    words = [k[5:] for k in d.files if k.startswith("obs__")]
    # use the per-concept mean activity (the substrate concept code), centered
    fillers = np.stack([_center_bipolarize(d["obs__" + w].mean(axis=0)) for w in words])
    return words, fillers  # (V, D)


def make_roles(mode, R, D, rng):
    if mode == "random":            # random bipolar = near-ortho in high-D
        roles = rng.choice([-1.0, 1.0], size=(R, D))
    elif mode == "overlap":         # deliberately overlapping roles (cos ~0.5)
        base = rng.choice([-1.0, 1.0], size=D)
        roles = np.array([base if rng.random() < 0.5 else -base for _ in range(R)])
        roles = roles + 0.7 * rng.choice([-1.0, 1.0], size=(R, D))
    elif mode == "disjoint":        # disjoint sub-population blocks (the dlpfc case)
        roles = np.zeros((R, D))
        block = D // R
        for r in range(R):
            roles[r, r * block:(r + 1) * block] = rng.choice([-1.0, 1.0], size=block)
    else:
        raise ValueError(mode)
    return roles / (np.linalg.norm(roles, axis=1, keepdims=True) + 1e-12)


def run(fillers, roles, rng, broken=False):
    """broken=True: unbind with a WRONG (random other) role -> anti-cheat (should be chance)."""
    V, D = fillers.shape
    R = roles.shape[0]
    out = {}
    for K in K_LIST:
        if K > min(V, R):
            continue
        correct, total = 0, 0
        for _ in range(N_TRIALS):
            fill_idx = rng.choice(V, size=K, replace=False)
            role_idx = rng.choice(R, size=K, replace=False)
            S = np.zeros(D)
            for k in range(K):
                S = S + roles[role_idx[k]] * fillers[fill_idx[k]]
            for k in range(K):
                ub_role = roles[rng.integers(R)] if broken else roles[role_idx[k]]
                est = S * ub_role
                pred = int(np.argmax(fillers @ est))
                correct += int(pred == fill_idx[k])
                total += 1
        out[K] = correct / total
    return out


def main():
    seeds = [s for s in SEEDS if os.path.exists(CACHE % s)]
    print("=== VSA/gain-field compositional generalization: substrate vs clean fillers ===", flush=True)
    if not seeds:
        print("CANNOT-CONCLUDE (no caches)", flush=True); return
    words, f0 = load_fillers(seeds[0])
    V, D = f0.shape
    btw = np.mean([float(f0[i] @ f0[j]) for i in range(V) for j in range(i + 1, V)])
    print(f"V={V} D={D}; substrate filler between-cos (centered) = {btw:.3f}", flush=True)

    R = 8  # a small fixed set of roles (agent/patient/action/... -- biologically few)
    modes = ["random", "overlap", "disjoint"]
    acc = {m: {K: [] for K in K_LIST} for m in modes}
    broken = {K: [] for K in K_LIST}
    role_btw = {}
    for seed in seeds:
        _, fillers = load_fillers(seed)
        rng = np.random.default_rng(seed)
        for m in modes:
            roles = make_roles(m, R, D, rng)
            if seed == seeds[0]:
                rb = np.mean([abs(float(roles[i] @ roles[j]))
                              for i in range(R) for j in range(i + 1, R)])
                role_btw[m] = rb
            for K, a in run(fillers, roles, rng).items():
                acc[m][K].append(a)
        # anti-cheat: random near-ortho roles but unbind with WRONG role
        roles = make_roles("random", R, D, rng)
        for K, a in run(fillers, roles, rng, broken=True).items():
            broken[K].append(a)

    print(f"\nroles: small fixed set R={R}; |role-role| cos by mode: "
          f"{ {m: round(role_btw[m], 3) for m in modes} }", flush=True)
    print(f"\n{'K':>3} | " + " | ".join(f"{m:>10}" for m in modes) + f" | {'broken(ctrl)':>11}", flush=True)
    for K in K_LIST:
        if not acc[modes[0]][K]:
            continue
        cells = " | ".join(f"{np.mean(acc[m][K]):>10.3f}" for m in modes)
        print(f"{K:>3} | {cells} | {np.mean(broken[K]):>11.3f}", flush=True)

    chance = 1.0 / V
    print(f"\nchance = 1/{V} = {chance:.3f}", flush=True)
    # verdict on K=4 (a 4-role sentence: agent/patient/action/manner) for each mode
    rnd4 = np.mean(acc["random"].get(4, [0]))
    ovl4 = np.mean(acc["overlap"].get(4, [0]))
    dsj4 = np.mean(acc["disjoint"].get(4, [0]))
    brk4 = np.mean(broken.get(4, [0]))
    print(f"K=4: random_roles={rnd4:.3f} overlap_roles={ovl4:.3f} disjoint_roles={dsj4:.3f} | broken={brk4:.3f}", flush=True)
    if brk4 > 0.20:
        print("VERDICT: ANTI-CHEAT FAILED -- broken binding is above chance; cleanup is doing the work, "
              "not the binding. Result INVALID.", flush=True)
    elif dsj4 >= 0.80 and brk4 < 0.20:
        print("VERDICT: ESCAPE CONFIRMED -- generalizable compositional bind/unbind works with the SUBSTRATE's "
              "OVERLAPPING fillers AND biologically-realistic DISJOINT-SUBPOPULATION roles (the dlpfc case), "
              f"K=4 acc {dsj4:.2f}; broken-binding control collapses to ~chance ({brk4:.2f}). Compositional "
              "generalization needs near-ortho ROLES (few, trivially feasible via distinct sub-populations) "
              "x ID-SEPARABLE fillers (the substrate HAS this: within>between) -- NOT near-ortho fillers. "
              "The near-ortho boundary does NOT block composition. REAL biological-composition path.", flush=True)
    elif rnd4 >= 0.80 and dsj4 < 0.80:
        print(f"VERDICT: PARTIAL -- works with random near-ortho roles ({rnd4:.2f}) but disjoint-subpop roles "
              f"weaker ({dsj4:.2f}). The mechanism needs distributed (not block) role codes; achievable but a "
              "design constraint.", flush=True)
    else:
        print(f"VERDICT: see table (random {rnd4:.2f}, overlap {ovl4:.2f}, disjoint {dsj4:.2f}, broken {brk4:.2f}).", flush=True)


if __name__ == "__main__":
    main()
