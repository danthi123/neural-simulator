"""Consolidation Rank-2 DG-source de-risk (2026-07-25): the A1 boundary is a DENSE/OVERLAPPING CA1 code (6-seed
Jaccard 0.557). The workflow's Rank-2 names TWO separation levers; naive CA1 FFI-kWTA was INERT (fixed top_k +
teacher-drive). The OTHER lever = source the write from DG, the pattern-separated locus (dg + dg_pv_basket; P1: DG cos
0.218 from input 0.800). DECISIVE cheap test: measure the per-tag engram OVERLAP for each hippocampal region
(ec/dg/ca3/ca1) under the same tags. If DG is SPARSE + DISTINCT (Jaccard << CA1's) then routing ca1->slot's role to
dg->slot would localize the write; if DG is ALSO dense, the separation is lost by the trisynaptic wiring (deeper).

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_dg_overlap_probe --seed 42
"""
import os, sys, json, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, CONSOLIDATED_FACTS, _try_tgate)
from research.runners.text_minimal_isolation import set_sleep_gates
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)
REGIONS = ["ec", "dg", "ca3", "ca1"]
BASE = dict(ca1_concept_density=0.25, ca1_concept_weight=0.0, nmda_self_weight=12.0, nmda_self_density=0.15,
            nmda_recurrent_ratio=0.6, cross_pool_density=0.10, stdp_w_max=8.0, enable_global_nmda=False,
            enable_hebbian=True, skip_nmda_additions=True, comp_attractor_slots=N, comp_attractor_n_per=120,
            comp_self_weight=12.0, comp_no_pool_slot=True, comp_dendritic=True, comp_wta_weight=5.0,
            comp_k_thresh=2.0, comp_self_regen=0.0, comp_kir_g=1.0)


def _jac(a, c):
    A, C = set(a.tolist()), set(c.tolist())
    return len(A & C) / max(1, len(A | C))


def run(seed):
    b = build_substrate(seed, SimpleNamespace(**BASE))
    rm = b.region_manager
    ridx = {}
    for r in REGIONS:
        try:
            ridx[r] = np.asarray(sorted(rm.indices(r)), dtype=np.int64)
        except Exception:
            ridx[r] = np.asarray([], dtype=np.int64)
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    # fire under each tag (full accumulation), then slice per region
    fire = {}
    for i, tag in enumerate(tags):
        _try_tgate(b, "nmda_attractor", 0.0)
        set_sleep_gates(b)
        b.cp_external_input_current[:] = 0.0
        for _ in range(30):
            b._run_one_simulation_step()
        b.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        acc = np.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=np.float64)
        for _ in range(40):
            b._run_one_simulation_step()
            acc += to_host(b.cp_firing_states).astype(np.float64)
        try:
            b.clear_tag_drive(tag)
        except Exception:
            pass
        b.cp_external_input_current[:] = 0.0
        fire[i] = acc
    # per-region: active fraction + pairwise Jaccard (at any-spike + >25%-fire)
    out = {"seed": seed, "backend": BACKEND, "regions": {}}
    for r in REGIONS:
        idx = ridx[r]
        if idx.size == 0:
            out["regions"][r] = {"n": 0}; continue
        rowj = {}
        for frac in (0.0, 0.25):
            engr = {i: idx[fire[i][idx] > frac * 40] for i in range(N)}
            sizes = [int(engr[i].size) for i in range(N)]
            js = [_jac(engr[i], engr[j]) for i in range(N) for j in range(i + 1, N)]
            rowj[str(frac)] = dict(active=sizes, active_frac=[round(s / idx.size, 3) for s in sizes],
                                   mean_jaccard=round(float(np.mean(js)), 3))
        out["regions"][r] = dict(n=int(idx.size), by_frac=rowj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run(args.seed)
    Path(f"{args.out}/dg_overlap_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    print(f"[seed {args.seed}] per-region engram (fire under tag) — is DG separated where CA1 is dense?")
    for reg in REGIONS:
        d = r["regions"].get(reg, {})
        if not d.get("n"):
            print(f"  {reg}: (absent)"); continue
        f0 = d["by_frac"]["0.0"]
        print(f"  {reg:4s} (n={d['n']:3d}): active_frac@0={f0['active_frac']} Jaccard@0={f0['mean_jaccard']}  "
              f"(@25%: frac={d['by_frac']['0.25']['active_frac']} J={d['by_frac']['0.25']['mean_jaccard']})")
    dg = r["regions"].get("dg", {}); ca1 = r["regions"].get("ca1", {})
    if dg.get("n") and ca1.get("n"):
        dgj = dg["by_frac"]["0.0"]["mean_jaccard"]; ca1j = ca1["by_frac"]["0.0"]["mean_jaccard"]
        print(f"\n  VERDICT: DG Jaccard {dgj} vs CA1 {ca1j} -> "
              f"{'DG SEPARATED (route the write from DG = the Rank-2 fix)' if dgj < 0.3 and dgj < ca1j - 0.15 else 'DG NOT clearly more separated (separation lost in the trisynaptic wiring -> deeper)'}")
    print("DG-OVERLAP-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
