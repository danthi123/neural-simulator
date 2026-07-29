"""Is there a GRADED, thresholdable OCCUPANCY signal on the substrate? (2026-07-29)

The toy says retrieve-vs-allocate gives 6/6 permutations, but it needs a graded read of BINDING STRENGTH
to threshold on: slot(c) = argmax(W.c) if max > theta ELSE allocate. This arc has measured graded
quantities getting CRUSHED at every layer (BTSP soft bound, Hebbian bound, apical plateau uniform to ~1%,
competition not steerable by 7x cue current). So before building the allocator, measure whether ANY
separation exists between "slot bound to this fact" and "slot not bound".

DECISIVE either way: if the distributions overlap, no theta exists and the toy result CANNOT transfer as
designed — which redirects the build rather than wasting a GPU week on it.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_occupancy_separability --seed 42
"""
import os, sys, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay, CONSOLIDATED_FACTS)
from research.runners._consol_direct_weight_probe import BASE
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", type=int, default=30)
    args = ap.parse_args()

    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=0.15,
             comp_kir_g=3.0, comp_v_hold=-50.0, comp_apical_R=0.15, comp_gc_read=0.5,
             comp_btsp=True, comp_btsp_lr=0.0005, comp_btsp_wmax=2000.0, comp_btsp_elig_tau=30.0,
             comp_no_pool_slot=False, comp_pool_slot_weight=1.5, comp_attractor_slots=N,
             enable_hebbian=True)
    b = build_substrate(args.seed, SimpleNamespace(**a))
    b.core_config.hebbian_max_weight = 2.5
    b.core_config.enable_stdp = False
    rm = b.region_manager
    slot = {i: np.asarray(sorted(rm.indices(f"comp_attr_{i}")), dtype=np.int64) for i in range(N)}
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)

    def wmat():
        """per-fact CA1-core -> per-slot mean weight (the quantity a retrieve rule would threshold)."""
        c = b.cp_connections; nz = int(c.nnz)
        po = to_host(c.indices).astype(np.int64)[:nz]
        ip = to_host(c.indptr).astype(np.int64)
        pr = np.repeat(np.arange(len(ip) - 1), np.diff(ip))[:nz]
        wd = to_host(c.data).astype(np.float64)[:nz]
        M = np.zeros((N, N))
        for i in range(N):
            try:
                core = np.asarray(to_host(b.get_engram_tag_indices(tags[i])), dtype=np.int64).ravel()
            except Exception:
                continue
            if core.size == 0:
                continue
            mc = np.isin(pr, core)
            for j in range(N):
                m = mc & np.isin(po, slot[j])
                M[i, j] = float(wd[m].mean()) if m.sum() else 0.0
        return M

    before = wmat()
    coactivation_replay(b, CONSOLIDATED_FACTS, tags, int(args.cycles), args.seed,
                        coactivate=True, attractor_on=True)
    after = wmat()

    print(f"[seed {args.seed}] cycles={args.cycles} backend={BACKEND}")
    print("  per-fact CA1core -> slot weights AFTER replay:")
    for i in range(N):
        print("    fact %d: %s" % (i, [round(v, 4) for v in after[i]]))

    # ⚠️ THE OBVIOUS METRIC IS TRIVIALLY TRUE. own=max(row) vs other=mean(rest) satisfies own>other by
    # ARITHMETIC, for any matrix whatsoever — it is not evidence. (Same 'true by construction' trap as
    # scoring a scramble control against a permuted target: it passes without testing anything.)
    #
    # What retrieve-vs-allocate ACTUALLY needs is not that a max exists, but that the max is
    #   (a) STABLE — a fact returns to the SAME slot on every read (it need NOT be the host's intended
    #       slot; a self-organized store yields a permutation, not the identity map), and
    #   (b) SEPARATED ENOUGH to threshold — the runner-up must be far enough below to place theta.
    # Both are measured below. (a) is the load-bearing one: this arc measured own-is-max at 2/9, i.e. a
    # graded max EXISTS but usually points at a slot other than the host's intended one — which is fine
    # for a permutation, and fatal only if it is unstable.
    own = np.array([after[i].max() for i in range(N)])
    oth = np.array([np.delete(after[i], int(np.argmax(after[i]))).max() for i in range(N)])   # RUNNER-UP, not mean
    gap = own - oth
    print("\n  own(max) per fact : %s" % [round(v, 4) for v in own])
    print("  runner-up          : %s" % [round(v, 4) for v in oth])
    print("  GAP               : %s" % [round(v, 5) for v in gap])
    spread = float(after.max() - after.min())
    print("  full weight spread across the whole matrix: %.5f" % spread)
    # STABILITY: re-read the matrix twice more and check argmax(W[f]) does not move.
    a2, a3 = wmat(), wmat()
    stable = [int(np.argmax(after[i])) == int(np.argmax(a2[i])) == int(np.argmax(a3[i])) for i in range(N)]
    print("  argmax slot per fact, 3 successive reads: %s / %s / %s"
          % ([int(np.argmax(after[i])) for i in range(N)],
             [int(np.argmax(a2[i])) for i in range(N)],
             [int(np.argmax(a3[i])) for i in range(N)]))
    print("  STABLE argmax (what RETRIEVE needs): %d/%d facts" % (sum(stable), N))
    amap = {i: int(np.argmax(after[i])) for i in range(N)}
    perm = len(set(amap.values())) == N
    print("  fact->slot map %s  permutation_valid=%s" % (amap, perm))
    sep = bool(gap.min() > 0.05 * max(abs(own.mean()), 1e-9)) and all(stable) and perm
    print("\n  => %s" % ("SEPARABLE: a theta exists (min gap is >5%% of the weight scale) -- retrieve-vs-allocate "
                         "has something to threshold on." if sep else
                         "NOT SEPARABLE: the min gap is under 5% of the weight scale. NO theta distinguishes a "
                         "bound slot from a free one, so retrieve-vs-allocate CANNOT be implemented on this "
                         "quantity -- the toy result does not transfer as designed."))
    print("OCCUPANCY-SEPARABILITY DONE")


if __name__ == "__main__":
    main()
