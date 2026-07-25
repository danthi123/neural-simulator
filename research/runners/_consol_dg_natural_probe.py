"""Consolidation Family-D Step 1 (2026-07-25 workflow-recommended PRIMARY) — the ~1-GPU-min density de-risk that decides
the whole DG-direct premise. DG has ZERO recurrent excitation (only out-edge dg->ca3, verified) so a SPARSE DG code
CANNOT re-densify into a halo (unlike CA1). The measured DG density (Jaccard 0.58) was a stimulate_tag(1500 pA) FLOOD
artifact. TEST: reinstate fact i via the NATURAL perforant path -- a per-fact orthogonal pattern (~200 pA) into
language_input, letting language_input->ec->dg + dg_pv_basket FFI (shipped) sparsify -- and read DG active-frac + Jaccard.

  GO: DG active_frac < 0.1 AND Jaccard < 0.2 (vs the flood's ~0.72/0.58) -> the halo was a drive artifact; the sparse
      pattern-separated DG index is real -> build the dg->slot write (Step 2). NO-GO: DG dense even under natural drive.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_dg_natural_probe --seed 42
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
from research.runners._consol_dg_overlap_probe import BASE
from research.runners.text_minimal_isolation import set_sleep_gates
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)


def _jac(a, c):
    A, C = set(a.tolist()), set(c.tolist())
    return len(A & C) / max(1, len(A | C))


def run(seed, li_drive, li_sparsity=0.1, ffi_lesion=False):
    b = build_substrate(seed, SimpleNamespace(**BASE))
    rm = b.region_manager
    if ffi_lesion:   # anti-cheat: zero dg_pv_basket->dg to confirm the FFI is load-bearing (DG should re-densify)
        _try_pgate(b, "dg_pv_basket_to_dg", 0.0); _try_tgate(b, "dg_pv_basket_to_dg", 0.0)
    dg_idx = np.asarray(sorted(rm.indices("dg")), dtype=np.int64)
    ca1_idx = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    li_idx = np.asarray(sorted(rm.indices("language_input")), dtype=np.int64)
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)

    def fire(method, tag, i):
        _try_tgate(b, "nmda_attractor", 0.0)
        set_sleep_gates(b)
        b.cp_external_input_current[:] = 0.0
        drv = None
        if method == "natural":  # perforant-path reinstatement: language_input -> ec -> dg (dg_pv_basket FFI sparsifies)
            for g in ("lang_to_ec", "ec_to_dg", "ec_context_to_dg"):
                _try_tgate(b, g, 1.0); _try_pgate(b, g, 1.0)
            pat = orthogonal_drive_pattern(i, n_cues=N, n_neurons=li_idx.size, drive_max_pA=float(li_drive), sparsity=float(li_sparsity))
            drv = cp.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=cp.float32)
            drv[cp.asarray(li_idx)] = cp.asarray(pat, dtype=cp.float32)
            b.cp_external_input_current[:] = drv
        for _ in range(30):
            b._run_one_simulation_step()
            if drv is not None:
                b.cp_external_input_current[:] = drv
        if method == "flood":
            b.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        acc = np.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=np.float64)
        for _ in range(40):
            b._run_one_simulation_step()
            acc += to_host(b.cp_firing_states).astype(np.float64)
            if drv is not None:
                b.cp_external_input_current[:] = drv
        try:
            b.clear_tag_drive(tag)
        except Exception:
            pass
        b.cp_external_input_current[:] = 0.0
        return acc

    out = {"seed": seed, "li_drive": li_drive, "methods": {}}
    for method in ("flood", "natural"):
        dg_engr, ca1_engr = {}, {}
        for i, tag in enumerate(tags):
            acc = fire(method, tag, i)
            dg_engr[i] = dg_idx[acc[dg_idx] > 0]
            ca1_engr[i] = ca1_idx[acc[ca1_idx] > 0]
        dgj = [_jac(dg_engr[i], dg_engr[j]) for i in range(N) for j in range(i + 1, N)]
        ca1j = [_jac(ca1_engr[i], ca1_engr[j]) for i in range(N) for j in range(i + 1, N)]
        out["methods"][method] = dict(
            dg_active=[int(dg_engr[i].size) for i in range(N)],
            dg_active_frac=[round(dg_engr[i].size / dg_idx.size, 3) for i in range(N)],
            dg_jaccard=round(float(np.mean(dgj)), 3),
            ca1_active_frac=[round(ca1_engr[i].size / ca1_idx.size, 3) for i in range(N)],
            ca1_jaccard=round(float(np.mean(ca1j)), 3))
    out["n_dg"] = int(dg_idx.size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--li-drive", type=float, default=200.0)
    ap.add_argument("--li-sparsity", type=float, default=0.1)
    ap.add_argument("--ffi-lesion", action="store_true")
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run(args.seed, args.li_drive, args.li_sparsity, args.ffi_lesion)
    Path(f"{args.out}/dg_natural_seed{args.seed}_d{args.li_drive:g}_s{args.li_sparsity:g}{'_lesion' if args.ffi_lesion else ''}.json").write_text(json.dumps(r, indent=2))
    fl = r["methods"]["flood"]; na = r["methods"]["natural"]
    print(f"[seed {args.seed} li_drive={args.li_drive}] DG density: does NATURAL perforant drive sparsify DG?")
    print(f"  FLOOD (stimulate_tag 1500pA):     DG active_frac={fl['dg_active_frac']} Jaccard={fl['dg_jaccard']}")
    print(f"  NATURAL (language_input->ec->dg):  DG active_frac={na['dg_active_frac']} Jaccard={na['dg_jaccard']}")
    go = np.mean(na['dg_active_frac']) < 0.1 and na['dg_jaccard'] < 0.2
    print(f"  VERDICT: {'GO -- sparse pattern-separated DG index is real -> build dg->slot write (Step 2)' if go else 'NO-GO -- DG not sparse/separated under natural drive (active %.2f, J %.2f)' % (float(np.mean(na['dg_active_frac'])), na['dg_jaccard'])}")
    print("DG-NATURAL-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
