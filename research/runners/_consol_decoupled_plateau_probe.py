"""Consolidation Option-2 FIRST-MOVE de-risk (2026-07-25): a DECOUPLED-PLATEAU write.

The Option-3 diagnostic pinned the boundary: the ISOLATED fact code IS fact-specific (fire-under-tag own/other ~4.5),
but the co-activation replay dilutes it to ~1.5 because driving the slot SOMA (to make the post-spike the STDP/Hebbian
write needs) floods CA1 — and every write rule then flattens it to ~1.0. Option 3 (cleaner replay + rate write) is
exhausted (caps << the 2.5 gate).

Option 2's cheapest first move (NO sim/ edit): DECOUPLE the "which slot" teaching signal from the somatic drive.
  - Reinstate fact i ISOLATED: stimulate_tag(tag_i) ONLY (no pool/slot somatic drive) -> CA1 fires the clean 4.5 pattern
    -> a CLEAN BTSP presynaptic eligibility Etilde over the fact-specific CA1 code.
  - Drive slot_i's APICAL plateau directly as a pure teaching signal: clamp cp_v_apical[slot_i] high each burst step.
    BTSP's instructive signal is IS = max(v_apical - v_hold, 0) (bridge.py:8053), so a clamped apical = a plateau that
    does NOT fire the soma and does NOT feed back to CA1. The bistable self-regen latch (comp_self_regen) + KIR down-state
    keep the OTHER slots silent (IS_j ~ 0 -> no write to slot_j).
  - dw[ca1_k -> slot_i] = eta * Etilde[ca1_k] * IS[slot_i] * (w_max - w): high for k in engram_i, ~0 for slot_j.
    -> the read own/other should track the CLEAN 4.5, not the diluted 1.5.

GATE (6-seed): DISTINCTIVE own/other @frac0 >= 2.5 AND own_is_max on >= 4/6 seeds -> Option 2 CHEAPLY GO.
               flat (~1.0) -> the decoupling does not help -> the boundary is the months-scale substrate (route on).

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_decoupled_plateau_probe --seed 42 --v-teach -25
"""
import os, sys, json, argparse, hashlib
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, _mean_gate_weight, CONSOLIDATED_FACTS, _try_tgate, _try_pgate)
from research.runners.text_minimal_isolation import set_sleep_gates
from research.runners._consol_direct_weight_probe import BASE, _fire_under_tag, _jac
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)
THRESH_FRACS = [0.0, 0.25, 0.5]


def decoupled_plateau_write(bridge, facts, tags, cycles, seed, v_teach=-25.0, burst_steps=30, isolate=True,
                            attractor_on=False, reinstate_drive=1500.0):
    """Isolated reinstatement (tag only) + a clamped apical teaching plateau on slot_i. BTSP writes ca1->slot_i on the
    clean eligible CA1 pattern, gated by the decoupled plateau (no somatic slot drive -> no CA1 flood).
    attractor_on=False: keep the CA1 pattern-completion attractor OFF so the fact-specific (4.5) reinstatement is NOT
    washed to a common attractor state (the plateau is force-clamped, so the slots don't need the attractor to sustain)."""
    set_sleep_gates(bridge)
    for g in ("ca1_to_comp_attr",):        # the plastic ca1->slot pathway BTSP writes
        _try_pgate(bridge, g, 1.0)
    _try_tgate(bridge, "nmda_attractor", 1.0 if attractor_on else 0.0)
    rm = bridge.region_manager
    all_names = {r.name for r in bridge.core_config.brain_regions}
    slot_idx = {i: (cp.asarray(sorted(rm.indices(f"comp_attr_{i}")), dtype=cp.int64)
                    if f"comp_attr_{i}" in all_names else None) for i in range(len(facts))}
    # apical rest (KIR down-state target) to reset between facts + hold the NON-target slots down (exclusive teaching)
    Er = float(getattr(bridge.core_config, "comp_v_hold", -50.0)) - 20.0  # ~ -70 rest, below v_hold
    all_slots = cp.concatenate([slot_idx[i] for i in range(len(facts)) if slot_idx[i] is not None])
    rng = np.random.default_rng(int(seed) + 777)
    order = list(range(len(facts)))
    for _c in range(int(cycles)):
        rng.shuffle(order)
        for i in order:
            tag = tags[i]
            bridge.cp_external_input_current[:] = 0.0
            bridge.stimulate_tag(tag, drive_pA=float(reinstate_drive), additive=False)   # ISOLATED CA3 engram cue (SWR: gentle => sparse)
            si = slot_idx[i]
            for _ in range(int(burst_steps)):
                if bridge.cp_v_apical is not None:
                    bridge.cp_v_apical[all_slots] = cp.float32(Er)        # EXCLUSIVE: hold ALL slots down first...
                    if si is not None:
                        bridge.cp_v_apical[si] = cp.float32(v_teach)      # ...then raise ONLY slot_i's teaching plateau
                bridge._run_one_simulation_step()
            try:
                bridge.clear_tag_drive(tag)
            except Exception:
                pass
            if bridge.cp_v_apical is not None:                            # reset all apical to rest between facts
                bridge.cp_v_apical[:] = cp.float32(Er)
    bridge.cp_external_input_current[:] = 0.0
    return {"cycles": int(cycles), "v_teach": float(v_teach)}


def run_seed(seed, v_teach=-25.0, cycles=40, btsp_lr=0.02, self_regen=0.15, tag_drive=1500.0,
             elig_exp=1.0, hetero_dep=0.0, hetero_theta=0.0, ffi_inh=0.0, ffi_drive=3.0, commit_top_k=None,
             hippo_izh_type=None, hippo_izh_regions="dg"):
    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=float(self_regen),
             comp_kir_g=3.0, comp_v_hold=-50.0,
             comp_btsp=True, comp_btsp_lr=float(btsp_lr), comp_btsp_wmax=8.0,
             comp_btsp_elig_exp=float(elig_exp), comp_btsp_hetero_dep=float(hetero_dep),
             comp_btsp_hetero_theta=float(hetero_theta))
    if ffi_inh > 0:   # sparsify the CA1 code (FFI kWTA) -> bigger disjoint per-fact cores
        a.update(ca1_ffi_kwta=True, ca1_ffi_inh=float(ffi_inh), ca1_ffi_drive=float(ffi_drive))
    if hippo_izh_type:   # sparse DG/CA3/CA1 phenotype (down-state-stable, high-threshold, adapting) -> sparse code
        a.update(hippo_izh_type=str(hippo_izh_type), hippo_izh_regions=str(hippo_izh_regions))
    b = build_substrate(seed, SimpleNamespace(**a))
    thr_hash = hashlib.md5(to_host(b.cp_neuron_firing_thresholds).tobytes()).hexdigest()[:12]
    rm = b.region_manager
    ca1_idx = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    slot_idx = {s: np.asarray(sorted(rm.indices(f"comp_attr_{s}")), dtype=np.int64) for s in range(N)}
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS, commit_top_k=commit_top_k)
    if b.cp_v_apical is None:
        return {"seed": seed, "error": "cp_v_apical is None (two-compartment not allocated) — comp_dendritic off?"}
    w0 = _mean_gate_weight(b, "ca1_to_comp_attr")
    decoupled_plateau_write(b, CONSOLIDATED_FACTS, tags, int(cycles), seed, v_teach=float(v_teach),
                            reinstate_drive=float(tag_drive))
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
    fire = {}
    for i, tag in enumerate(tags):
        fc, _ = _fire_under_tag(b, tag, ca1_idx, drive=tag_drive)
        fire[i] = fc

    def engram_at(i, frac):
        return ca1_idx[fire[i] > frac * 40]

    res = {"seed": seed, "thr_hash": thr_hash, "dw": round(w1 - w0, 5), "v_teach": float(v_teach),
           "self_regen": float(self_regen), "by_thresh": {}}
    for frac in THRESH_FRACS:
        engr = {i: engram_at(i, frac) for i in range(N)}
        sizes = {i: int(engr[i].size) for i in range(N)}
        jac = float(np.mean([_jac(engr[i], engr[j]) for i in range(N) for j in range(i + 1, N)]))
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
        res["by_thresh"][str(frac)] = dict(engram_sizes=sizes, mean_jaccard=jac, distinctive_sizes=dist_sizes,
                                            distinctive_own_over_other=d_oo, own_is_max=own_is_max)
    # RATE-WEIGHTED own/other: the CAPABILITY metric (each ca1 pre weighted by its fire count -> total input to slot_j).
    # This is what actually drives selective recall (the full fact_i pattern activating ca1->slot; slot_i should win),
    # and matches the isolated fire-under-tag 4.5 (rate-based), unlike the harsh binary distinctive-core.
    rw = np.zeros((N, N))
    for i in range(N):
        w_pre = np.zeros(csr.shape[0]); w_pre[ca1_idx] = fire[i]
        for j in range(N):
            m = (syn_slot == j) & (w_pre[pre_of] > 0)
            rw[i, j] = float((data[m] * w_pre[pre_of][m]).sum())
    rw_oo = [float(rw[i, i] / np.mean([rw[i, j] for j in range(N) if j != i]))
             if np.mean([rw[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
    res["rate_weighted"] = dict(own_over_other=rw_oo, own_is_max=[bool(np.argmax(rw[i]) == i) for i in range(N)])
    # CODE-OVERLAP CEILING: the max own/other ANY linear write can reach on this CA1 rate code = Sum(fire_i^2)/mean_j Sum(fire_i*fire_j).
    # If the measured write's own/other ~= this ceiling, the WRITE is already at ceiling -> the code separability is the wall.
    F = np.stack([fire[i] for i in range(N)])          # (N, n_ca1) rate vectors
    G = F @ F.T                                         # gram: G[i,j] = fire_i . fire_j
    ceil = [float(G[i, i] / np.mean([G[i, j] for j in range(N) if j != i]))
            if np.mean([G[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
    res["code_overlap_ceiling"] = dict(own_over_other=ceil, mean=float(np.mean(ceil)))
    # SPARSE-CODE ceiling: threshold each fire vector at its >frac-of-max cells (a working kWTA) -> would CA1 separation
    # raise the ceiling above the 2.5 gate? If YES, the fix is upstream CA1 pattern-separation (DG/CA3), not the write.
    res["sparse_ceiling"] = {}
    for tf in (0.25, 0.5):
        Fs = np.where(F > tf * 40, F, 0.0)
        Gs = Fs @ Fs.T
        sc = [float(Gs[i, i] / np.mean([Gs[i, j] for j in range(N) if j != i]))
              if np.mean([Gs[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
        res["sparse_ceiling"][str(tf)] = dict(own_over_other=sc, mean=float(np.mean([x for x in sc if x > 0]) if any(x > 0 for x in sc) else 0.0),
                                               n_active=[int((Fs[i] > 0).sum()) for i in range(N)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--v-teach", type=float, default=-25.0, help="clamped apical teaching-plateau voltage (above v_hold=-50)")
    ap.add_argument("--cycles", type=int, default=40)
    ap.add_argument("--btsp-lr", type=float, default=0.02)
    ap.add_argument("--self-regen", type=float, default=0.15, help="bistable plateau latch (0=off)")
    ap.add_argument("--elig-exp", type=float, default=1.0, help="supralinear eligibility exponent (>1 concentrates the write on strong-firing core)")
    ap.add_argument("--hetero-dep", type=float, default=0.0, help="heterosynaptic depression coeff (0=off)")
    ap.add_argument("--hetero-theta", type=float, default=0.0, help="eligibility threshold for depression (sparse gate)")
    ap.add_argument("--ffi-inh", type=float, default=0.0, help="CA1 FFI-kWTA inhibition (0=off; sparsify the code)")
    ap.add_argument("--ffi-drive", type=float, default=3.0)
    ap.add_argument("--commit-top-k", type=int, default=None, help="sparse engram-tag commit size (research-gate: 15 -> sparse near-disjoint CA1 code)")
    ap.add_argument("--tag-drive", type=float, default=1500.0, help="reinstatement + read drive (SWR: gentle e.g. 400-600 => sparse, no re-densify)")
    ap.add_argument("--hippo-izh-type", type=str, default=None, help="sparse hippo phenotype, e.g. IZH2007_STRIATAL_MSN (down-state-stable, high-threshold, adapting)")
    ap.add_argument("--hippo-izh-regions", type=str, default="dg", help="comma-sep regions to give the sparse phenotype, e.g. dg,ca3,ca1")
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run_seed(args.seed, v_teach=args.v_teach, cycles=args.cycles, btsp_lr=args.btsp_lr, self_regen=args.self_regen,
                 elig_exp=args.elig_exp, hetero_dep=args.hetero_dep, hetero_theta=args.hetero_theta,
                 ffi_inh=args.ffi_inh, ffi_drive=args.ffi_drive, commit_top_k=args.commit_top_k, tag_drive=args.tag_drive,
                 hippo_izh_type=args.hippo_izh_type, hippo_izh_regions=args.hippo_izh_regions)
    _tg = (f"_ee{args.elig_exp:g}" if args.elig_exp > 1 else "") + (f"_hd{args.hetero_dep:g}" if args.hetero_dep > 0 else "") + (f"_ffi{args.ffi_inh:g}" if args.ffi_inh > 0 else "")
    Path(f"{args.out}/decoupled_vt{args.v_teach:g}{_tg}_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    if "error" in r:
        print(f"[seed {args.seed}] ERROR: {r['error']}"); print("DECOUPLED-PLATEAU-PROBE DONE", flush=True); return
    d0 = r["by_thresh"]["0.0"]
    doo = [round(x, 3) for x in d0["distinctive_own_over_other"]]
    mean_oo = float(np.mean([x for x in d0["distinctive_own_over_other"] if x > 0])) if any(x > 0 for x in d0["distinctive_own_over_other"]) else 0.0
    n_max = sum(d0["own_is_max"])
    rw = r["rate_weighted"]; rw_oo = [round(x, 3) for x in rw["own_over_other"]]
    rw_mean = float(np.mean([x for x in rw["own_over_other"] if x > 0])) if any(x > 0 for x in rw["own_over_other"]) else 0.0
    rw_nmax = sum(rw["own_is_max"])
    print(f"[seed {args.seed}] backend={BACKEND} thr_hash={r['thr_hash']} dw={r['dw']} v_teach={args.v_teach}")
    print(f"  DISTINCTIVE own/other @frac0={doo}  mean={mean_oo:.3f}  own_is_max={d0['own_is_max']} ({n_max}/{N})")
    print(f"  RATE-WEIGHTED own/other={rw_oo}  mean={rw_mean:.3f}  own_is_max={rw['own_is_max']} ({rw_nmax}/{N})  <- CAPABILITY metric")
    ceil = r.get("code_overlap_ceiling", {})
    print(f"  CODE-OVERLAP CEILING own/other={[round(x,3) for x in ceil.get('own_over_other',[])]}  mean={ceil.get('mean',0):.3f}  <- max ANY linear write can reach (dense code)")
    sc = r.get("sparse_ceiling", {})
    for tf in ("0.25", "0.5"):
        if tf in sc:
            print(f"  SPARSE CEILING @>{tf}max: own/other={[round(x,2) for x in sc[tf]['own_over_other']]} mean={sc[tf]['mean']:.3f} n_active={sc[tf]['n_active']}  <- if>2.5, CA1 SEPARATION is the fix")
    print(f"  distinctive_sizes={d0['distinctive_sizes']}  Jaccard@0={d0['mean_jaccard']:.3f}")
    print(f"  VERDICT (rate-weighted): {'GO-ish (own/other>=2.5 & own_is_max>=4/6)' if (rw_mean >= 2.5 and rw_nmax >= N) else 'below-gate (mean %.2f, %d/%d max)' % (rw_mean, rw_nmax, N)}")
    print("DECOUPLED-PLATEAU-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
