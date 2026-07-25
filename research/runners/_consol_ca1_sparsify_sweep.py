"""Consolidation Option-2 lever de-risk (2026-07-25): can a feedback-inhibition kWTA SPARSIFY the CA1 code enough to
lift the code-overlap ceiling above the 2.5 gate? The decoupled-plateau probe proved the write is code-bounded
(dense ceiling 1.54; sparse >25%-max ceiling 5.56). This sweeps the CA1 FFI-kWTA (ca1->ca1_ffi->ca1 feedback loop)
over (ffi_inh, ffi_drive, ffi_n) and reports, for the RESULTING CA1 code: n_active (want <~5% => sparse), the
dense-code overlap ceiling (want >2.5), and Jaccard (want low/disjoint). A config with n_active sparse AND ceiling>2.5
=> CA1 separation is realizable on the point-neuron substrate => the consolidation write passes.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_ca1_sparsify_sweep --seed 42
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
from research.runners._consol_direct_weight_probe import BASE, _fire_under_tag, _jac
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)


def measure_code(seed, ffi_inh, ffi_drive, ffi_n, tag_drive=1500.0):
    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=0.15, comp_kir_g=3.0)
    if ffi_inh > 0:
        a.update(ca1_ffi_kwta=True, ca1_ffi_inh=float(ffi_inh), ca1_ffi_drive=float(ffi_drive), ca1_ffi_n=int(ffi_n))
    b = build_substrate(seed, SimpleNamespace(**a))
    rm = b.region_manager
    ca1_idx = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    n_ca1 = ca1_idx.size
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    fire = {}
    for i, tag in enumerate(tags):
        fc, _ = _fire_under_tag(b, tag, ca1_idx, drive=tag_drive)
        fire[i] = fc
    F = np.stack([fire[i] for i in range(N)])
    G = F @ F.T
    ceil = float(np.mean([G[i, i] / np.mean([G[i, j] for j in range(N) if j != i])
                          if np.mean([G[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]))
    n_active = [int((F[i] > 0).sum()) for i in range(N)]
    frac_active = float(np.mean(n_active) / max(1, n_ca1))
    jac = float(np.mean([_jac(ca1_idx[F[i] > 0], ca1_idx[F[j] > 0]) for i in range(N) for j in range(i + 1, N)]))
    return dict(n_ca1=int(n_ca1), n_active=n_active, frac_active=round(frac_active, 3), ceiling=round(ceil, 3),
                jaccard=round(jac, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # grid: baseline (no ffi) + feedback-inhibition kWTA over (inh, drive, n)
    grid = [(0.0, 0.0, 0)]
    for n in (30, 60):
        for drive in (3.0, 6.0):
            for inh in (10.0, 25.0, 50.0):
                grid.append((inh, drive, n))
    results = []
    for (inh, drive, n) in grid:
        r = measure_code(args.seed, inh, drive, n)
        r.update(ffi_inh=inh, ffi_drive=drive, ffi_n=n)
        results.append(r)
        tag = "baseline" if inh == 0 else f"inh{inh:g}_drv{drive:g}_n{n}"
        go = "**SPARSE+SEP**" if (r["frac_active"] < 0.10 and r["ceiling"] > 2.5) else ""
        print(f"  [{tag:20s}] frac_active={r['frac_active']:.3f} n_active={r['n_active']} ceiling={r['ceiling']:.2f} jac={r['jaccard']:.3f} {go}", flush=True)
    Path(f"{args.out}/ca1_sparsify_sweep_seed{args.seed}.json").write_text(json.dumps(results, indent=2))
    best = max(results, key=lambda x: (x["ceiling"] if x["frac_active"] < 0.15 else 0.0))
    print(f"[seed {args.seed}] BEST (sparse & high-ceiling): {best}")
    print("CA1-SPARSIFY-SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
