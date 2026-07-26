"""Slot-route CORTICAL STORE probe (2026-07-25) — the corrected next step after the ca1→slot 6-seed GO.

WHY THIS, AND WHY IT LOOKS DIFFERENT FROM THE A1 TEST:
  * The `ca1→comp_attr` write is validated (6-seed GO at the calibrated operating point). But **CA1 is hippocampus**,
    so that pathway is REMOVED by the hippo lesion the capability test applies — it cannot itself deliver
    hippo-independent recall. It is the replay-time reinstatement half.
  * The store that must survive the lesion is the CORTEX-resident `concept_to_comp_attr` (noun/adj pool → slot,
    plastic, already wired). **That is what this probe measures.**
  * The A1 capability test cues a noun through `language_input`, which needs word→pool binding — and that is **UNBUILT**
    (never above chance on any reproducible configuration; the recorded 87.5% baseline is retired as unreproducible).
    So this probe **cues the concept pools DIRECTLY by teacher current**, bypassing the unbuilt path entirely. That is
    a deliberate scope choice: it tests CONSOLIDATION (the actual open question) without depending on a faculty that
    does not exist yet. It is therefore NOT the full A1 gate and must never be reported as such.

Operating point is the CALIBRATED one (`comp_apical_R=0.15`, `comp_gc_read=0.5`, default pyramidal phenotype) — the
arc's earlier settings were an artifact of a 333x pA→mV miscalibration.

Every ratio is reported with the MASS TRIAD (permuted-target control · raw per-target magnitudes · per-fact passes,
never a mean), per `.claude/skills/verify-go/SKILL.md` lens 7.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_cortical_store_probe --seed 42
"""
import os, sys, json, argparse, hashlib
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay, _mean_gate_weight,
    CONSOLIDATED_FACTS, _try_tgate, _try_pgate, hippo_lesioned, _NOUN_POOLS, _ADJ_POOLS)
from research.runners._consol_direct_weight_probe import BASE
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)


def run(seed, cycles=10, btsp_lr=0.0005, drive_pA=1400.0, read_steps=60):
    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=0.15, comp_kir_g=3.0,
             comp_v_hold=-50.0, comp_apical_R=0.15, comp_gc_read=0.5,          # CALIBRATED operating point
             comp_btsp=True, comp_btsp_lr=float(btsp_lr), comp_btsp_wmax=2000.0,
             # BASE sets comp_no_pool_slot=True, which DROPS the pool->slot pathways entirely (it was added because the
             # ALL-pools->ALL-slots broadcast is a write-selectivity killer for the ca1->slot measurement). But those
             # pathways ARE the cortical store this probe exists to measure, so it must re-enable them. Without this the
             # probe reports dw=0 and per-slot mass=0 — i.e. it measures a pathway that does not exist.
             comp_no_pool_slot=False)
    b = build_substrate(seed, SimpleNamespace(**a))
    thr_hash = hashlib.md5(to_host(b.cp_neuron_firing_thresholds).tobytes()).hexdigest()[:12]
    rm = b.region_manager
    names = {r.name for r in b.core_config.brain_regions}
    slot = {i: np.asarray(sorted(rm.indices(f"comp_attr_{i}")), dtype=np.int64) for i in range(N)
            if f"comp_attr_{i}" in names}
    pool_of = {}
    for i, (noun, adj) in enumerate(CONSOLIDATED_FACTS):
        ps = [f"noun_pool_{noun.upper()}", f"adjective_pool_{adj.upper()}"]
        pool_of[i] = [np.asarray(sorted(rm.indices(p)), dtype=np.int64) for p in ps if p in names]

    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    va = to_host(b.cp_v_apical) if getattr(b, "cp_v_apical", None) is not None else None
    v_ok = bool(va is not None and va.min() >= -90 and va.max() <= 50)

    w0 = _mean_gate_weight(b, "concept_to_comp_attr")
    coactivation_replay(b, CONSOLIDATED_FACTS, tags, int(cycles), seed, coactivate=True, attractor_on=True)
    w1 = _mean_gate_weight(b, "concept_to_comp_attr")

    # ---- (A) is the CORTICAL store selective?  pool_i -> slot_j weights
    csr = b.cp_connections
    data = to_host(csr.data).astype(np.float64)[:int(csr.nnz)]
    post_of = to_host(csr.indices).astype(np.int64)[:int(csr.nnz)]
    indptr = to_host(csr.indptr).astype(np.int64)
    pre_of = np.zeros(int(csr.nnz), dtype=np.int64)
    for r in range(len(indptr) - 1):
        pre_of[indptr[r]:indptr[r + 1]] = r
    post_slot = np.full(csr.shape[0], -1, dtype=np.int64)
    for s in slot:
        post_slot[slot[s]] = s
    syn_slot = post_slot[post_of]
    W = np.zeros((N, N))
    for i in range(N):
        pre = np.concatenate(pool_of[i]) if pool_of[i] else np.array([], dtype=np.int64)
        m_pre = np.isin(pre_of, pre)
        for j in slot:
            m = m_pre & (syn_slot == j)
            W[i, j] = float(data[m].mean()) if m.sum() else 0.0
    oo = [float(W[i, i] / np.mean([W[i, j] for j in range(N) if j != i]))
          if np.mean([W[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
    # permuted-target control: read fact i's pools against a ROTATED slot assignment -> must collapse to ~1.0
    perm = [(i + 1) % N for i in range(N)]
    oo_perm = [float(W[i, perm[i]] / np.mean([W[i, j] for j in range(N) if j != perm[i]]))
               if np.mean([W[i, j] for j in range(N) if j != perm[i]]) > 1e-12 else 0.0 for i in range(N)]

    # ---- (B) CAPABILITY: hippo LESIONED, drive fact i's pools directly, read slot activity
    recall, recall_rates = [], []
    with hippo_lesioned(b):
        _try_tgate(b, "nmda_attractor", 1.0)
        for i in range(N):
            b.cp_external_input_current[:] = 0.0
            drv = cp.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=cp.float32)
            for arr in pool_of[i]:
                drv[cp.asarray(arr)] = float(drive_pA)
            acc = np.zeros(int(b.cp_membrane_potential_v.shape[0]))
            for _ in range(int(read_steps)):
                b.cp_external_input_current[:] = drv
                b._run_one_simulation_step()
                acc += to_host(b.cp_firing_states).astype(np.float64)
            rates = [float(acc[slot[j]].mean()) if j in slot else 0.0 for j in range(N)]
            recall_rates.append([round(r, 3) for r in rates])
            recall.append(bool(int(np.argmax(rates)) == i and max(rates) > 0))
            b.cp_external_input_current[:] = 0.0
    return dict(seed=int(seed), thr_hash=thr_hash, v_apical_physiological=v_ok,
                v_apical_range=[round(float(va.min()), 2), round(float(va.max()), 2)] if va is not None else None,
                dw_cortical=round(w1 - w0, 5),
                store_own_over_other=[round(x, 3) for x in oo],
                store_own_is_max=[bool(np.argmax(W[i]) == i) for i in range(N)],
                permuted_target_control=[round(x, 3) for x in oo_perm],
                per_slot_mass=[round(float(W[:, j].mean()), 4) for j in range(N)],
                recall_correct=recall, recall_slot_rates=recall_rates,
                n_recall=int(sum(recall)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--btsp-lr", type=float, default=0.0005)
    ap.add_argument("--out", default="research/findings/raw/cortical_store")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run(args.seed, args.cycles, args.btsp_lr)
    Path(f"{args.out}/cortstore_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    print(f"[seed {args.seed}] backend={BACKEND} thr_hash={r['thr_hash']} dw_cortical={r['dw_cortical']}")
    print(f"  v_apical={r['v_apical_range']} physiological={r['v_apical_physiological']}"
          + ("" if r['v_apical_physiological'] else "   <-- INVALID SUBSTRATE, stop"))
    print(f"  (A) CORTICAL STORE concept→slot own/other={r['store_own_over_other']} own_is_max={r['store_own_is_max']}")
    print(f"      permuted-target control={r['permuted_target_control']}  <- MUST collapse to ~1.0")
    print(f"      per-slot mass={r['per_slot_mass']}  <- must be balanced (else winner-slot artifact)")
    print(f"  (B) HIPPO-LESIONED recall (pools driven directly): {r['n_recall']}/{N} correct  slot rates={r['recall_slot_rates']}")
    ok = r['n_recall'] >= 2 and sum(r['store_own_is_max']) >= 2
    print(f"  VERDICT: {'GO-ish — cortical store selective AND survives the lesion (verify controls + 6 seeds)' if ok else 'NO — see (A)/(B); report which half failed'}")
    print("  SCOPE: cues pools DIRECTLY (teacher current), NOT via word→pool binding (unbuilt) — this is NOT the full A1 gate.")
    print("CORTICAL-STORE-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
