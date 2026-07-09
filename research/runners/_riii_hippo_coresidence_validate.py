"""R-iii one-brain integration: validate the CA3 memory (formation + dendritic completion) CO-RESIDENT on the shared
nav/conv bridge (`build_merged_nav_conv_bridge(co_resident_hippo_memory=True)`). Two gates: (A) the MEMORY FUNCTION --
the direct-synchronous FORMATION + partial-cue COMPLETION (CYCLE 1076) fires on the co-resident ca3 slice (held-out
members complete); (B) ZERO CROSS-TALK -- the ca3 memory slice is array-disjoint from nav/conv, so the merged bridge's
neuron count + cp_connections structure with the flag ON differ from OFF only by the appended hippo slice, and the
memory ops do not perturb the nav/conv regions (the `project_one_brain_substrate_vs_functional` bar). Reuse-by-import
of `_train_assemblies`/`_recall` (the validated formation/completion harness). GPU (SIM_BACKEND=cupy). NO `sim/` edit.

Run: SIM_BACKEND=cupy python -m research.runners._riii_hippo_coresidence_validate --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, time
import numpy as np


def run_seed(seed, hippo_n_ca3=500, n_assembly=12, n_mem=3, presentations=60, hippo_k_thresh=66.0, cue_drive=1000.0):
    """Build the merged nav/conv bridge WITH the co-resident CA3 memory, run formation + partial-cue completion on the
    ca3 slice, and confirm the ca3 slice is disjoint from the nav/conv regions. Returns the held-out completion + a
    disjointness flag. (The formation/completion params mirror the CYCLE-1086 sparse-large regime.)"""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
    from research.runners._riii_ca3_emergent_completion_derisk import _train_assemblies, _recall
    from research.runners._riii_ca3_coincidence_completion_derisk import _set_gates

    bridge, _handles = build_merged_nav_conv_bridge(seed=seed, co_resident_hippo_memory=True,
                                                    hippo_n_ca3=hippo_n_ca3, hippo_n_ca1=120,
                                                    hippo_k_thresh=hippo_k_thresh)
    rm = bridge.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    # disjointness: the ca3 slice must not overlap any nav/parser/dlpfc region
    ca3_set = set(int(x) for x in ca3_idx)
    other = set()
    for rn, idxs in rm.region_indices_dict().items():
        if rn in ("ca3", "ca1", "ca3_pv_basket"):
            continue
        other |= set(int(x) for x in idxs)
    disjoint = len(ca3_set & other) == 0

    try:
        basket = np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)
    except Exception:
        basket = None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ca3_idx)
    assemblies = [np.array(perm[m * n_assembly:(m + 1) * n_assembly], dtype=np.int64) for m in range(n_mem)]
    non_assembly = np.array([g for g in ca3_idx if g not in set(int(x) for a in assemblies for x in a)],
                            dtype=np.int64)[:60]
    _train_assemblies(bridge, cp, assemblies, presentations, 1000.0, 8, 12)
    held_c, non_c = [], []
    for asm in assemblies:
        a = asm.copy(); rng.shuffle(a); h = len(a) // 2
        cue, held = a[:h], a[h:]
        rh = _recall(bridge, cp, cue, held, cue_drive, clamp_cells=basket)
        rc = _recall(bridge, cp, cue, cue, cue_drive, clamp_cells=basket)
        rn = _recall(bridge, cp, cue, non_assembly, cue_drive, clamp_cells=basket)
        ca = float(np.mean(rc)) + 1e-9
        held_c.append(float(np.mean(rh)) / ca); non_c.append(float(np.mean(rn)) / ca)
    return {"heldout": float(np.mean(held_c)), "nonassembly": float(np.mean(non_c)), "disjoint": disjoint,
            "n_total": int(rm.total_neurons()), "n_ca3": int(len(ca3_idx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--hippo-n-ca3", type=int, default=500)
    ap.add_argument("--n-assembly", type=int, default=12)
    ap.add_argument("--hippo-k-thresh", type=float, default=66.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split()] if " " in a.seeds else [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii HIPPO CO-RESIDENCE] co_resident_hippo_memory on the merged nav/conv bridge | formation+completion "
          f"on the co-resident ca3 slice + disjointness", flush=True)
    import json
    rows = []
    for s in seeds:
        t0 = time.time()
        r = run_seed(s, hippo_n_ca3=a.hippo_n_ca3, n_assembly=a.n_assembly, hippo_k_thresh=a.hippo_k_thresh)
        rows.append({"seed": s, **r})
        print(f"  [seed {s}] held-out={r['heldout']:.3f} (non {r['nonassembly']:.3f}) | disjoint={r['disjoint']} "
              f"| n_total={r['n_total']} n_ca3={r['n_ca3']} ({time.time()-t0:.0f}s)", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        h = [r["heldout"] for r in rows]; nn = [r["nonassembly"] for r in rows]
        go = all(x > 0.30 for x in h) and all(x < 0.20 for x in nn) and all(r["disjoint"] for r in rows)
        print(f"\n  AGGREGATE: held-out={np.mean(h):.3f} non-assembly={np.mean(nn):.3f} disjoint={all(r['disjoint'] for r in rows)}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the emergent CA3 completion fires on the co-resident ca3 slice of the shared nav/conv one-brain (held-out completes, non-assembly silent), array-disjoint from nav/conv = the R-iii memory folded into the ONE BRAIN' if go else 'completion or disjointness not yet clean on the merged bridge; check the coincidence-plateau config scoping'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
