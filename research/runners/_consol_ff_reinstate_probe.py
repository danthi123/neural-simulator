"""Consolidation Rank-2 stack element 2 (gentle FEEDFORWARD reinstatement) — decisive cheap test (2026-07-25).
Element 1 (sparse commit) alone was INSUFFICIENT: `stimulate_tag` floods CA1 directly at 1500 pA and CA1 recurrence
re-densifies a sparse tag to ~85. The biologically-faithful SWR replay reinstates CA1 via CA3->CA1 pattern completion
from the SPARSE CA3 engram, NOT by flooding CA1. TEST: drive ONLY the CA3 portion of the tag (gentle) with ca3_to_ca1
open, and measure the CA1 response density + per-fact Jaccard vs the direct-flood baseline. If CA3-driven reinstatement
gives a SPARSE + DISTINCT CA1 (Jaccard << the flood's ~0.58, active-frac << 0.7) -> element 2 is the load-bearing lever.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_ff_reinstate_probe --seed 42
"""
import os, sys, json, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, CONSOLIDATED_FACTS, _try_tgate, _try_pgate)
from research.runners.text_minimal_isolation import set_sleep_gates
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)
BASE = dict(ca1_concept_density=0.25, ca1_concept_weight=0.0, nmda_self_weight=12.0, nmda_self_density=0.15,
            nmda_recurrent_ratio=0.6, cross_pool_density=0.10, stdp_w_max=8.0, enable_global_nmda=False,
            enable_hebbian=True, skip_nmda_additions=True, comp_attractor_slots=N, comp_attractor_n_per=120,
            comp_self_weight=12.0, comp_no_pool_slot=True, comp_dendritic=True, comp_wta_weight=5.0,
            comp_k_thresh=2.0, comp_self_regen=0.0, comp_kir_g=1.0)


def _jac(a, c):
    A, C = set(a.tolist()), set(c.tolist())
    return len(A & C) / max(1, len(A | C))


def _ca1_fire(idx_ca1, acc_full):
    return acc_full[idx_ca1]


def run(seed, commit_top_k, ca3_drive):
    b = build_substrate(seed, SimpleNamespace(**BASE))
    rm = b.region_manager
    ca1_idx = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    ca3_idx = np.asarray(sorted(rm.indices("ca3")), dtype=np.int64)
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS, commit_top_k=commit_top_k)
    ca3set = set(ca3_idx.tolist())

    def _fire(method, tag):
        _try_tgate(b, "nmda_attractor", 0.0)
        set_sleep_gates(b)
        _try_tgate(b, "ca3_to_ca1", 1.0); _try_pgate(b, "ca3_to_ca1", 1.0)   # open CA3->CA1 for reinstatement
        b.cp_external_input_current[:] = 0.0
        for _ in range(30):
            b._run_one_simulation_step()
        if method == "flood":
            b.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        else:  # ff: drive ONLY the CA3 portion of the tag, gently -> ca3->ca1 reinstates CA1
            _ti = b.get_engram_tag_indices(tag)
            ti = np.asarray(_ti.get() if hasattr(_ti, "get") else list(_ti), dtype=np.int64)
            ca3_tag = np.asarray([x for x in ti.tolist() if x in ca3set], dtype=np.int64)
            drv = cp.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=cp.float32)
            if ca3_tag.size:
                drv[cp.asarray(ca3_tag)] = float(ca3_drive)
            b.cp_external_input_current[:] = drv
        acc = np.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=np.float64)
        for _ in range(40):
            b._run_one_simulation_step()
            acc += to_host(b.cp_firing_states).astype(np.float64)
            if method != "flood" and ca3_tag.size:   # sustain the gentle CA3 drive
                b.cp_external_input_current[:] = drv
        try:
            b.clear_tag_drive(tag)
        except Exception:
            pass
        b.cp_external_input_current[:] = 0.0
        return acc

    out = {"seed": seed, "commit_top_k": commit_top_k, "ca3_drive": ca3_drive, "methods": {}}
    for method in ("flood", "ff"):
        engr = {}
        for i, tag in enumerate(tags):
            acc = _fire(method, tag)
            f = _ca1_fire(ca1_idx, acc)
            engr[i] = ca1_idx[f > 0]
        sizes = [int(engr[i].size) for i in range(N)]
        js = [_jac(engr[i], engr[j]) for i in range(N) for j in range(i + 1, N)]
        # ca3 tag portion sizes (for context)
        out["methods"][method] = dict(ca1_active=sizes, ca1_active_frac=[round(s / ca1_idx.size, 3) for s in sizes],
                                       ca1_mean_jaccard=round(float(np.mean(js)), 3))
    out["n_ca1"] = int(ca1_idx.size); out["n_ca3"] = int(ca3_idx.size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--commit-top-k", type=int, default=20)
    ap.add_argument("--ca3-drive", type=float, default=400.0)
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run(args.seed, args.commit_top_k, args.ca3_drive)
    Path(f"{args.out}/ff_reinstate_seed{args.seed}_tk{args.commit_top_k}_d{args.ca3_drive:g}.json").write_text(json.dumps(r, indent=2))
    fl = r["methods"]["flood"]; ff = r["methods"]["ff"]
    print(f"[seed {args.seed} tk={args.commit_top_k} ca3_drive={args.ca3_drive}] CA1 reinstatement density + overlap:")
    print(f"  FLOOD (direct stimulate_tag): active_frac={fl['ca1_active_frac']} Jaccard={fl['ca1_mean_jaccard']}")
    print(f"  FF (CA3->CA1 reinstatement):  active_frac={ff['ca1_active_frac']} Jaccard={ff['ca1_mean_jaccard']}")
    sparser = np.mean(ff['ca1_active_frac']) < 0.5 * np.mean(fl['ca1_active_frac'])
    distinct = ff['ca1_mean_jaccard'] < 0.3 and ff['ca1_mean_jaccard'] < fl['ca1_mean_jaccard'] - 0.15
    print(f"  VERDICT: {'FF reinstatement gives a SPARSE + DISTINCT CA1 -> element 2 works (build the write on it)' if (sparser and distinct) else 'FF not clearly sparse+distinct (tune ca3_drive/top_k, or CA3->CA1 reinstatement itself is dense)'}")
    print("FF-REINSTATE-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
