"""ADVERSARIAL: do OCCUPANCY and CUE-MATCH have OPPOSITE saturation fates on pool->slot? (2026-07-29)

The neural-occupancy design reads BOTH of its signals off ONE matrix (concept_to_comp_attr):
    o[s] = mean W[ all 8 pools -> slot s ]   -> occupancy      (ACROSS-slot contrast)
    m[s] = mean W[ this fact's 2 pools -> s] -> cue match      (WITHIN-slot, across-POOL contrast)
and its Stage 0 only gates on the ACROSS-slot spread (`rownorm_spread`).

sim/bridge.py:7712-7715 (default Hebbian branch, hebbian_rate_window/branchless both OFF):
    active = where(pre_fired[t-1] & post_fired[t]);  dw = lr * (hebbian_max_weight - w)
=> the increment is RATE-INDEPENDENT: a binary-gated exponential relaxation to w_max, i.e. a
   SATURATING EVENT COUNTER  w(n) = wmax - (wmax-w0)*(1-lr)^n.
PREDICTION: for a slot that keeps winning, EVERY pool synapse into it -> w_max, so the
WITHIN-slot pool contrast (the retrieve branch) DECAYS TO ZERO with replay, while the
ACROSS-slot contrast (occupancy) GROWS. The design's Stage 0 is structurally blind to this.

Measured at slot_drive=0 (self-organized, per the design's own mandatory setting).
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_occupancy_vs_match_saturation --seed 42
"""
import os, sys, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay,
    CONSOLIDATED_FACTS, _NOUN_POOLS, _ADJ_POOLS, _POOL_OF, _mean_gate_weight)
from research.runners._consol_direct_weight_probe import BASE
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)
POOLS = list(_NOUN_POOLS) + list(_ADJ_POOLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=4)      # measure after each chunk
    ap.add_argument("--cycles-per-chunk", type=int, default=8)
    ap.add_argument("--hebb-max", type=float, default=2.5)
    args = ap.parse_args()

    a = dict(BASE)
    # EXACTLY the design's Stage-0 config block
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=0.15,
             comp_kir_g=3.0, comp_v_hold=-50.0, comp_apical_R=0.15, comp_gc_read=0.5,
             comp_btsp=True, comp_btsp_lr=0.0005, comp_btsp_wmax=2000.0, comp_btsp_elig_tau=30.0,
             comp_no_pool_slot=False, comp_pool_slot_weight=1.5, comp_attractor_slots=N,
             enable_hebbian=True)
    b = build_substrate(args.seed, SimpleNamespace(**a))
    b.core_config.hebbian_max_weight = float(args.hebb_max)
    b.core_config.enable_stdp = False
    rm = b.region_manager

    slot = {s: np.asarray(sorted(rm.indices(f"comp_attr_{s}")), dtype=np.int64) for s in range(N)}
    pool = {p: np.asarray(sorted(rm.indices(p)), dtype=np.int64) for p in POOLS}
    fact_pools = [[_POOL_OF[n], _POOL_OF[adj]] for (n, adj) in CONSOLIDATED_FACTS]

    # PRE-FLIGHT, printed not assumed
    print(f"[seed {args.seed}] backend={BACKEND}  hebb_max={b.core_config.hebbian_max_weight} "
          f"pool_slot_init={a['comp_pool_slot_weight']}  comp_no_pool_slot={a['comp_no_pool_slot']}")
    print(f"  live mean W(concept_to_comp_attr) = {_mean_gate_weight(b, 'concept_to_comp_attr'):.4f}")
    assert b.core_config.hebbian_max_weight > a["comp_pool_slot_weight"], "bound BELOW init -> rule inverts"

    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    # ⚠ THE CHECK MUST BE PLACED HERE, NOT AT BUILD. Encode itself potentiates pool->slot, so a bound that
    # passes the build-time check can be BELOW the live weight by the time replay starts -> the rule INVERTS.
    _w_post = _mean_gate_weight(b, "concept_to_comp_attr")
    print(f"  POST-ENCODE mean W(concept_to_comp_attr) = {_w_post:.4f}   bound = {b.core_config.hebbian_max_weight}")
    print("  REGIME: %s" % ("INVERTED (bound < live weight) -- every 'potentiation' is DEPRESSION; results VOID"
                            if _w_post > b.core_config.hebbian_max_weight else "potentiating (bound > live weight)"))

    def W():
        """W[pool p, slot s] = mean concept_to_comp_attr weight, from the LIVE CSR."""
        c = b.cp_connections
        nz = int(c.nnz)
        po = to_host(c.indices).astype(np.int64)[:nz]
        ip = to_host(c.indptr).astype(np.int64)
        pr = np.repeat(np.arange(len(ip) - 1), np.diff(ip))[:nz]
        wd = to_host(c.data).astype(np.float64)[:nz]
        M = np.zeros((len(POOLS), N))
        for pi, p in enumerate(POOLS):
            mp = np.isin(pr, pool[p])
            for s in range(N):
                m = mp & np.isin(po, slot[s])
                M[pi, s] = float(wd[m].mean()) if m.sum() else np.nan
        return M

    hdr = ("  cyc | occupancy o[s] (mean over 8 pools)      | ACROSS-slot | WITHIN-slot pool spread per slot | "
           "retrieve contrast m-o per fact")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    done = 0
    for ch in range(args.chunks + 1):
        M = W()
        o = np.nanmean(M, axis=0)                                   # occupancy per slot
        across = float((o.max() - o.min()) / max(o.max(), 1e-12))   # design's Stage-0 rownorm_spread
        within = [float((np.nanmax(M[:, s]) - np.nanmin(M[:, s])) / max(np.nanmax(M[:, s]), 1e-12))
                  for s in range(N)]                                # the check Stage 0 OMITS
        mminus = []
        for f in range(N):
            pi = [POOLS.index(p) for p in fact_pools[f]]
            m = np.nanmean(M[pi, :], axis=0)
            mminus.append(float((m - o).max()))                     # best retrieve contrast for this fact
        print("  %4d | %s | %11.4f | %s | %s"
              % (done, [round(v, 4) for v in o], across,
                 [round(v, 4) for v in within], [round(v, 5) for v in mminus]))
        rows.append((done, across, float(np.mean(within)), float(np.mean(mminus))))
        if ch == args.chunks:
            break
        coactivation_replay(b, CONSOLIDATED_FACTS, tags, int(args.cycles_per_chunk), args.seed,
                            coactivate=True, attractor_on=True, slot_drive_pA=0.0)   # MANDATORY per design §2
        done += args.cycles_per_chunk

    print("\n  TREND (the falsifiable prediction of the critique):")
    print("    cycles : ACROSS-slot(occupancy)  WITHIN-slot(retrieve)  mean(m-o)")
    for (c_, ac, wi, mm) in rows:
        print("    %6d :  %18.4f  %20.4f  %9.5f" % (c_, ac, wi, mm))
    a0, aN = rows[0][1], rows[-1][1]
    w0, wN = rows[0][2], rows[-1][2]
    print("\n  VERDICT")
    print("   occupancy (across-slot) : %.4f -> %.4f  %s (design Stage-0 bar >= 0.20 : %s)"
          % (a0, aN, "GROWS" if aN > a0 else "does NOT grow", "PASS" if aN >= 0.20 else "FAIL"))
    print("   retrieve  (within-slot) : %.4f -> %.4f  %s"
          % (w0, wN, "DECAYS (saturating counter confirmed)" if wN < w0 else "does NOT decay"))
    if aN < 0.20:
        print("   => STAGE-0 KILLER FIRES ON THE DESIGN'S OWN BAR: occupancy is not readable from this matrix.")
    if wN < w0 and wN < 0.15:
        print("   => RETRIEVE BRANCH IS UNDEFINED: the within-slot pool contrast the retrieve rule thresholds on "
              "has collapsed. Stage 0 as written CANNOT see this.")
    print("OCC-VS-MATCH-SATURATION DONE")


if __name__ == "__main__":
    main()
