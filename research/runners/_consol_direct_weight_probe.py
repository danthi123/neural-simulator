"""Consolidation STEP-A decisive diagnostic (promoted + extended from the workflow verifier's direct_weight_probe.py,
2026-07-25). Reads the ca1_engram_i -> slot_j WEIGHTS DIRECTLY from cp_connections (rows=pre, cols=post) after the
co-activation write — the load-bearing measurement the saturating g_coincidence + the confounded g_e (a 1.4-1.9x
noise floor from near-uniform weights) cannot give. ROUTES the build:

  - distinctive rate-weighted own/other >= ~1.1 AND own-is-max on >=4/6 seeds  -> the presynaptic signal is SELECTABLE
    -> lever = the SELECTIVE WRITE (heterosynaptic competition, Rank 1).
  - distinctive ~1.0 (or distinctive sets vanish at thresh_frac=0.5 while Jaccard>0.4)  -> CA1 is a DENSE OVERLAPPING
    code -> lever = upstream CA1 pattern-separation (FFI kWTA / DG-sourced drive, Rank 2).

Also resolves the record's Jaccard contradiction (0.55-0.67 dense vs 0.00-0.11 disjoint) by reporting overlap at
thresh_frac in {0.0, 0.25, 0.5} + rate-weighted — the two camps used different engram definitions of the SAME substrate.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_direct_weight_probe --seed 42
"""
import os, sys, json, argparse, hashlib
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay, _mean_gate_weight, CONSOLIDATED_FACTS,
    _try_tgate)
from research.runners.text_minimal_isolation import set_sleep_gates
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)
BASE = dict(ca1_concept_density=0.25, ca1_concept_weight=0.0, nmda_self_weight=12.0, nmda_self_density=0.15,
            nmda_recurrent_ratio=0.6, cross_pool_density=0.10, stdp_w_max=8.0, enable_global_nmda=False,
            enable_hebbian=True, skip_nmda_additions=True, comp_attractor_slots=N, comp_attractor_n_per=120,
            comp_self_weight=12.0, comp_no_pool_slot=True)
OP = dict(self_regen=0.0, k_thresh=2.0, wta=5.0, kir_g=1.0, slot_drive=700.0)   # op000 = biggest write dw
THRESH_FRACS = [0.0, 0.25, 0.5]


def _fire_under_tag(b, tag, ca1_idx, steps=40, drive=1500.0):
    """Accumulated ca1 firing under tag (compute ONCE; threshold at multiple fracs after). `drive`: gentler = sparser."""
    _try_tgate(b, "nmda_attractor", 0.0)
    set_sleep_gates(b)
    b.cp_external_input_current[:] = 0.0
    for _ in range(30):
        b._run_one_simulation_step()
    b.stimulate_tag(tag, drive_pA=float(drive), additive=False)
    acc = np.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=np.float64)
    for _ in range(steps):
        b._run_one_simulation_step()
        acc += to_host(b.cp_firing_states).astype(np.float64)
    try:
        b.clear_tag_drive(tag)
    except Exception:
        pass
    b.cp_external_input_current[:] = 0.0
    return acc[ca1_idx], steps   # fire counts per ca1 neuron


def _jac(a, c):
    A, C = set(a.tolist()), set(c.tolist())
    return len(A & C) / max(1, len(A | C))


def run_seed(seed, ffi_inh=0.0, ffi_drive=3.0, tag_drive=1500.0, commit_top_k=None,
             btsp_hetero_dep=0.0, btsp_hetero_theta=0.0, btsp_elig_exp=1.0):
    a = dict(BASE); a.update(comp_dendritic=True, comp_wta_weight=OP["wta"], comp_k_thresh=OP["k_thresh"],
                             comp_self_regen=OP["self_regen"], comp_kir_g=OP["kir_g"])
    if ffi_inh > 0:   # Rank-2 CA1 FFI-kWTA sparsification de-risk
        a.update(ca1_ffi_kwta=True, ca1_ffi_inh=float(ffi_inh), ca1_ffi_drive=float(ffi_drive))
    if btsp_hetero_dep > 0 or btsp_elig_exp > 1.0:   # Rank-2 element 3: rate-gated heterosynaptic write (+ supralinear elig)
        a.update(comp_btsp=True, comp_btsp_hetero_dep=float(btsp_hetero_dep),
                 comp_btsp_hetero_theta=float(btsp_hetero_theta), comp_btsp_elig_exp=float(btsp_elig_exp))
    b = build_substrate(seed, SimpleNamespace(**a))
    # SEED VERIFICATION (the seed-never-controlled-substrate trap): hash the per-neuron firing thresholds
    thr_hash = hashlib.md5(to_host(b.cp_neuron_firing_thresholds).tobytes()).hexdigest()[:12]
    rm = b.region_manager
    ca1_idx = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    slot_idx = {s: np.asarray(sorted(rm.indices(f"comp_attr_{s}")), dtype=np.int64) for s in range(N)}
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS, commit_top_k=commit_top_k)
    w0 = _mean_gate_weight(b, "ca1_to_comp_attr")
    coactivation_replay(b, CONSOLIDATED_FACTS, tags, 40, seed, coactivate=True, attractor_on=True,
                        slot_drive_pA=OP["slot_drive"], tag_drive_pA=float(tag_drive))
    w1 = _mean_gate_weight(b, "ca1_to_comp_attr")
    # reconstruct (pre,post,weight)
    csr = b.cp_connections
    data = to_host(csr.data).astype(np.float64)[:int(csr.nnz)]
    post_of = to_host(csr.indices).astype(np.int64)[:int(csr.nnz)]
    indptr = to_host(csr.indptr).astype(np.int64)
    pre_of = np.zeros(int(csr.nnz), dtype=np.int64)
    for r in range(len(indptr) - 1):
        pre_of[indptr[r]:indptr[r + 1]] = r
    post_slot = np.full(csr.shape[0], -1, dtype=np.int64)
    for s in range(N):
        post_slot[slot_idx[s]] = s
    syn_slot = post_slot[post_of]
    # fire per fact (once)
    fire = {}
    for i, tag in enumerate(tags):
        fc, steps = _fire_under_tag(b, tag, ca1_idx, drive=tag_drive)
        fire[i] = fc

    def engram_at(i, frac):
        return ca1_idx[fire[i] > frac * 40]

    res = {"seed": seed, "thr_hash": thr_hash, "dw": round(w1 - w0, 5), "by_thresh": {}, "rate_weighted": {}}
    for frac in THRESH_FRACS:
        engr = {i: engram_at(i, frac) for i in range(N)}
        sizes = {i: int(engr[i].size) for i in range(N)}
        jac = [[round(_jac(engr[i], engr[j]), 3) for j in range(N)] for i in range(N)]
        mean_jac = float(np.mean([jac[i][j] for i in range(N) for j in range(i + 1, N)]))
        # distinctive (engram_i minus union others) MEAN weight own/other
        Ddir = np.zeros((N, N)); dist_sizes = {}; own_is_max = []
        for i in range(N):
            others = set()
            for j in range(N):
                if j != i:
                    others |= set(engr[j].tolist())
            dist = np.asarray([x for x in engr[i].tolist() if x not in others], dtype=np.int64)
            dist_sizes[i] = int(dist.size)
            for j in range(N):
                m = np.isin(pre_of, dist) & (syn_slot == j)
                Ddir[i, j] = float(data[m].mean()) if m.sum() > 0 else 0.0
            own_is_max.append(bool(dist.size > 0 and np.argmax(Ddir[i]) == i))
        d_oo = [float(Ddir[i, i] / np.mean([Ddir[i, j] for j in range(N) if j != i]))
                if np.mean([Ddir[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
        res["by_thresh"][str(frac)] = dict(engram_sizes=sizes, mean_jaccard=mean_jac, jaccard=jac,
                                            distinctive_sizes=dist_sizes, distinctive_own_over_other=d_oo,
                                            own_is_max=own_is_max)
    # rate-weighted: weight each ca1 pre by its fire count -> own/other of the rate-weighted ca1->slot_j weight sum
    rw = np.zeros((N, N))
    for i in range(N):
        w_pre = np.zeros(csr.shape[0]); w_pre[ca1_idx] = fire[i]
        for j in range(N):
            m = (syn_slot == j) & (w_pre[pre_of] > 0)
            rw[i, j] = float((data[m] * w_pre[pre_of][m]).sum())
    rw_oo = [float(rw[i, i] / np.mean([rw[i, j] for j in range(N) if j != i]))
             if np.mean([rw[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
    res["rate_weighted"] = dict(own_over_other=rw_oo, own_is_max=[bool(np.argmax(rw[i]) == i) for i in range(N)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ca1-ffi-inh", type=float, default=0.0, help="Rank-2: CA1 FFI-kWTA inhibition strength (0=off)")
    ap.add_argument("--ca1-ffi-drive", type=float, default=3.0)
    ap.add_argument("--tag-drive", type=float, default=1500.0, help="replay+probe tag drive (gentler = sparser distinct core)")
    ap.add_argument("--commit-top-k", type=int, default=None, help="Rank-2 el.1: sparse engram-tag commit size (None=~85)")
    ap.add_argument("--btsp-hetero-dep", type=float, default=0.0, help="Rank-2 el.3: heterosynaptic-depression coeff (0=off)")
    ap.add_argument("--btsp-hetero-theta", type=float, default=0.0, help="Rank-2 el.3: eligibility threshold for depression")
    ap.add_argument("--btsp-elig-exp", type=float, default=1.0, help="Rank-2 el.3b: supralinear eligibility exponent (>1 widens core-halo gap)")
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run_seed(args.seed, ffi_inh=args.ca1_ffi_inh, ffi_drive=args.ca1_ffi_drive, tag_drive=args.tag_drive,
                 commit_top_k=args.commit_top_k, btsp_hetero_dep=args.btsp_hetero_dep,
                 btsp_hetero_theta=args.btsp_hetero_theta, btsp_elig_exp=args.btsp_elig_exp)
    r["ca1_ffi_inh"] = args.ca1_ffi_inh; r["tag_drive"] = args.tag_drive; r["commit_top_k"] = args.commit_top_k
    r["btsp_hetero_dep"] = args.btsp_hetero_dep; r["btsp_elig_exp"] = args.btsp_elig_exp
    tag = (f"_ffi{args.ca1_ffi_inh:g}" if args.ca1_ffi_inh > 0 else "") + (f"_td{args.tag_drive:g}" if args.tag_drive != 1500 else "") + (f"_tk{args.commit_top_k}" if args.commit_top_k else "") + (f"_hd{args.btsp_hetero_dep:g}" if args.btsp_hetero_dep > 0 else "") + (f"_ee{args.btsp_elig_exp:g}" if args.btsp_elig_exp > 1 else "")
    Path(f"{args.out}/directwrite{tag}_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    # decision summary
    d0 = r["by_thresh"]["0.0"]; d5 = r["by_thresh"]["0.5"]
    print(f"[seed {args.seed}] backend={BACKEND} thr_hash={r['thr_hash']} dw={r['dw']}")
    print(f"  Jaccard: frac0={d0['mean_jaccard']} frac0.5={d5['mean_jaccard']} | engram_sizes@0={d0['engram_sizes']}")
    print(f"  DISTINCTIVE own/other @frac0={[round(x,3) for x in d0['distinctive_own_over_other']]} own_is_max={d0['own_is_max']}")
    print(f"  RATE-WEIGHTED own/other={[round(x,3) for x in r['rate_weighted']['own_over_other']]} own_is_max={r['rate_weighted']['own_is_max']}")
    print(f"  ROUTE: {'SELECTABLE->WRITE(Rank1)' if (np.mean(r['rate_weighted']['own_over_other'])>=1.1 and sum(r['rate_weighted']['own_is_max'])>=2) else 'DENSE-CODE->CA1-SEPARATION(Rank2)'}")
    print("DIRECTWRITE-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
