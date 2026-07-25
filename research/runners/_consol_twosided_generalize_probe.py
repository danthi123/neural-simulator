"""Consolidation two-sided-read GENERALIZATION de-risk (2026-07-25): does the unsaturated-graded-write +
core-gated-recall lead (fact-1 own/other 3.67 at seed 42) GENERALIZE across all 3 facts (and seeds)?

Reuses the shipped `_consol_decoupled_plateau_probe` machinery (build_substrate, isolated-reinstatement + exclusive
apical-clamp teaching write, BTSP) and ADDS the decisive diagnostics the lead needs:

  1. CROSS-FACT CORE-FIRING matrix -- for each fact-K's >25% CORE cells, their mean firing under EVERY fact's tag
     (fire-under-tag) AND during the ACTUAL write reinstatement windows. Tests hypothesis (a): fact-2's core leaks to
     other slots because it FIRES during fact-0/1's clamp windows (builds eligibility -> written to the wrong slot).
  2. write-loop instrumentation: per-fact-window CA1 firing capture; optional eligibility RESET between facts (tests
     hypothesis (b): temporal eligibility bleed across the tau=30ms low-pass in the shuffled order); optional FIXED order;
     optional SETTLE gap between facts (let eligibility decay + slots return to KIR down-state).
  3. PERMUTED-CORE control: read slot_i against a DIFFERENT fact's core -> must collapse to ~1.0 (the selectivity is EARNED
     by the write, not a metric artifact).
  4. per-fact CORE-GATED RECALL own/other + core_sizes + dw (the GO metric).

GO-in-principle (per the task): core-gated own/other >= 2.5 AND own-is-max on >= 2/3 facts, and the permuted-core control
collapses (~1.0), and cores are not 1-2-cell degenerate. Multi-seed = >=3 seeds.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_twosided_generalize_probe --seed 42 \
    --commit-top-k 15 --hippo-izh-type IZH2007_STRIATAL_MSN --hippo-izh-regions dg,ca3,ca1 \
    --elig-tau 30 --elig-hard-thresh 0.4 --cycles 3 --btsp-wmax 2000 --btsp-lr 0.000003
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
_CORE_THR = 0.25 * 40.0    # >25% of the 40-step fire-under-tag window (same as the shipped SPARSE CEILING)


def instrumented_write(bridge, facts, tags, cycles, seed, v_teach=-25.0, burst_steps=30, reinstate_drive=1500.0,
                       reset_elig=False, fixed_order=False, settle_steps=0, ca1_idx=None, blocked=False):
    """Isolated-reinstatement + exclusive-apical-clamp teaching write (== decoupled_plateau_write) with instrumentation:
    captures per-fact-window CA1 firing, and optionally resets BTSP eligibility / settles / fixes order between facts.
    blocked=True: write ALL cycles of fact i before fact i+1 (full per-fact isolation — the decisive test of whether the
    leak is the interleaved schedule vs a deeper per-fact property)."""
    set_sleep_gates(bridge)
    _try_pgate(bridge, "ca1_to_comp_attr", 1.0)
    _try_tgate(bridge, "nmda_attractor", 0.0)
    rm = bridge.region_manager
    all_names = {r.name for r in bridge.core_config.brain_regions}
    slot_idx = {i: (cp.asarray(sorted(rm.indices(f"comp_attr_{i}")), dtype=cp.int64)
                    if f"comp_attr_{i}" in all_names else None) for i in range(len(facts))}
    Er = float(getattr(bridge.core_config, "comp_v_hold", -50.0)) - 20.0
    all_slots = cp.concatenate([slot_idx[i] for i in range(len(facts)) if slot_idx[i] is not None])
    ca1_h = np.asarray(ca1_idx, dtype=np.int64)
    # window_fire[write_fact] accumulates CA1 firing during that fact's reinstatement bursts (summed over cycles)
    window_fire = {i: np.zeros(ca1_h.size, dtype=np.float64) for i in range(len(facts))}
    rng = np.random.default_rng(int(seed) + 777)

    def _one_fact_burst(i):
        tag = tags[i]
        if reset_elig and getattr(bridge, "cp_btsp_pre_elig", None) is not None:
            bridge.cp_btsp_pre_elig[:] = cp.float32(0.0)
        bridge.cp_external_input_current[:] = 0.0
        bridge.stimulate_tag(tag, drive_pA=float(reinstate_drive), additive=False)
        si = slot_idx[i]
        for _ in range(int(burst_steps)):
            if bridge.cp_v_apical is not None:
                bridge.cp_v_apical[all_slots] = cp.float32(Er)
                if si is not None:
                    bridge.cp_v_apical[si] = cp.float32(v_teach)
            bridge._run_one_simulation_step()
            window_fire[i] += to_host(bridge.cp_firing_states).astype(np.float64)[ca1_h]
        try:
            bridge.clear_tag_drive(tag)
        except Exception:
            pass
        if bridge.cp_v_apical is not None:
            bridge.cp_v_apical[:] = cp.float32(Er)
        if settle_steps > 0:      # let eligibility decay + slots return to down-state before the next fact
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(int(settle_steps)):
                if bridge.cp_v_apical is not None:
                    bridge.cp_v_apical[all_slots] = cp.float32(Er)
                bridge._run_one_simulation_step()

    order0 = list(range(len(facts)))
    if blocked:
        for i in order0:
            for _c in range(int(cycles)):
                _one_fact_burst(i)
    else:
        for _c in range(int(cycles)):
            order = list(order0) if fixed_order else (rng.shuffle(order0) or order0)
            for i in order:
                _one_fact_burst(i)
    bridge.cp_external_input_current[:] = 0.0
    return {"window_fire": window_fire}


def run_seed(seed, v_teach=-25.0, cycles=3, btsp_lr=0.000003, self_regen=0.15, tag_drive=1500.0,
             elig_exp=1.0, hetero_dep=0.0, hetero_theta=0.0, commit_top_k=15,
             hippo_izh_type="IZH2007_STRIATAL_MSN", hippo_izh_regions="dg,ca3,ca1",
             elig_hard_thresh=0.4, elig_tau=30.0, btsp_wmax=2000.0,
             reset_elig=False, fixed_order=False, settle_steps=0, core_thr_frac=0.25, blocked=False):
    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=float(self_regen),
             comp_kir_g=3.0, comp_v_hold=-50.0,
             comp_btsp=True, comp_btsp_lr=float(btsp_lr), comp_btsp_wmax=float(btsp_wmax),
             comp_btsp_elig_exp=float(elig_exp), comp_btsp_hetero_dep=float(hetero_dep),
             comp_btsp_hetero_theta=float(hetero_theta), comp_btsp_elig_hard_thresh=float(elig_hard_thresh))
    if elig_tau is not None:
        a.update(comp_btsp_elig_tau=float(elig_tau))
    if hippo_izh_type:
        a.update(hippo_izh_type=str(hippo_izh_type), hippo_izh_regions=str(hippo_izh_regions))
    b = build_substrate(seed, SimpleNamespace(**a))
    thr_hash = hashlib.md5(to_host(b.cp_neuron_firing_thresholds).tobytes()).hexdigest()[:12]
    rm = b.region_manager
    ca1_idx = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    slot_idx = {s: np.asarray(sorted(rm.indices(f"comp_attr_{s}")), dtype=np.int64) for s in range(N)}
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS, commit_top_k=commit_top_k)
    if b.cp_v_apical is None:
        return {"seed": seed, "error": "cp_v_apical is None (comp_dendritic off?)"}

    # cores are write-INDEPENDENT (fire-under-tag drives the committed tag; ca1->slot weights don't feed back to CA1).
    # Compute them BEFORE the write so the write can be instrumented against a fixed core set.
    fire = {}
    for i, tag in enumerate(tags):
        fc, _ = _fire_under_tag(b, tag, ca1_idx, drive=tag_drive)
        fire[i] = fc
    _thr = float(core_thr_frac) * 40.0
    core = {i: np.where(fire[i] > _thr)[0] for i in range(N)}    # indices INTO ca1_idx
    core_sizes = [int(core[i].size) for i in range(N)]

    # CROSS-FACT CORE-FIRING (fire-under-tag): mean firing of fact-K's core cells under fact-W's tag.
    xfire_tag = np.zeros((N, N))   # [core_of_K, under_tag_W]
    for k in range(N):
        for w in range(N):
            xfire_tag[k, w] = float(fire[w][core[k]].mean()) if core[k].size > 0 else 0.0

    w0 = _mean_gate_weight(b, "ca1_to_comp_attr")
    inst = instrumented_write(b, CONSOLIDATED_FACTS, tags, int(cycles), seed, v_teach=float(v_teach),
                              reinstate_drive=float(tag_drive), reset_elig=reset_elig, fixed_order=fixed_order,
                              settle_steps=int(settle_steps), ca1_idx=ca1_idx, blocked=blocked)
    w1 = _mean_gate_weight(b, "ca1_to_comp_attr")
    wf = inst["window_fire"]
    # CROSS-FACT CORE-FIRING (actual write windows): mean firing of fact-K's core cells during fact-W's write reinstatement
    xfire_write = np.zeros((N, N))
    for k in range(N):
        for w in range(N):
            xfire_write[k, w] = float(wf[w][core[k]].mean()) if core[k].size > 0 else 0.0

    # ---- M0 (2026-07-25 research gate, `-dendritic-spike-count-read-research-gate.md`): the GATED CEILING, computed on
    # the DURING-WRITE counts. The proposed dendritic surpass applies a per-source ABSOLUTE windowed-spike-count gate
    # BEFORE the synaptic sum. This asks the FREE question the gate would otherwise be built to answer: after such a
    # gate, is the code the write actually sees fact-specific enough to support the 2.5 selectivity gate?
    #   best gated ceiling >= 2.5  -> the gate has a real signal to amplify -> build M1' (the ~25-line additive sim edit).
    #   best gated ceiling <  2.5  -> the WRITE WINDOWS are the lever, not dendrites (a gate would amplify a flat
    #                                 signal — the arc's recurring error). Contrast vs the ISOLATED counts shows which.
    WF = np.stack([wf[i] for i in range(N)])       # (N, n_ca1) per-cell spike counts DURING the write windows
    FT = np.stack([fire[i] for i in range(N)])     # (N, n_ca1) per-cell counts under ISOLATED tag (the reference)

    def _ceiling(G):
        out = []
        for i in range(N):
            own = float((G[i] * G[i]).sum())
            oth = float(np.mean([float((G[i] * G[j]).sum()) for j in range(N) if j != i]))
            out.append(own / oth if oth > 1e-12 else 0.0)
        return out

    def _gated_scan(X, label):
        xmax = float(X.max()) if X.size else 0.0
        res = {"max_count": round(xmax, 1), "ungated_ceiling": [round(v, 3) for v in _ceiling(X)], "binary": {}, "hill": {}}
        for frac in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
            th = round(frac * xmax, 2)
            Gb = (X > th).astype(np.float64)                       # binary absolute-threshold gate
            res["binary"][f"{frac:g}"] = dict(theta=th, ceiling=[round(v, 3) for v in _ceiling(Gb)],
                                              n_active=[int(Gb[i].sum()) for i in range(N)])
            Gh = X**8 / (X**8 + th**8 + 1e-30)                     # Hill n=8 (CaMKII-like), ABSOLUTE theta
            res["hill"][f"{frac:g}"] = dict(theta=th, ceiling=[round(v, 3) for v in _ceiling(Gh)],
                                            n_active=[int((Gh[i] > 0.5).sum()) for i in range(N)])
        return res

    m0 = {"during_write": _gated_scan(WF, "during_write"), "isolated_tag": _gated_scan(FT, "isolated_tag")}

    # reconstruct (pre, post-slot, weight)
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

    def core_gated_matrix(core_map):
        """rwc[i,j] = sum over ca1 in core_map[i] (weighted by fire_i) of w[ca1->slot_j]."""
        M = np.zeros((N, N))
        for i in range(N):
            w_pre = np.zeros(csr.shape[0]); ci = ca1_idx[core_map[i]]; w_pre[ci] = fire[i][core_map[i]]
            for j in range(N):
                m = (syn_slot == j) & (w_pre[pre_of] > 0)
                M[i, j] = float((data[m] * w_pre[pre_of][m]).sum())
        return M

    rwc = core_gated_matrix(core)
    rwc_oo = [float(rwc[i, i] / np.mean([rwc[i, j] for j in range(N) if j != i]))
              if np.mean([rwc[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
    rwc_max = [bool(np.argmax(rwc[i]) == i) for i in range(N)]

    # PERMUTED-CORE control: gate fact-i's recall read by a DIFFERENT fact's core (cyclic i->(i+1)%N).
    # If the selectivity is EARNED, reading slot_i against another fact's core has NO own-slot preference (~1.0).
    perm = {i: core[(i + 1) % N] for i in range(N)}
    rwc_perm = np.zeros((N, N))
    for i in range(N):
        w_pre = np.zeros(csr.shape[0]); ci = ca1_idx[perm[i]]; w_pre[ci] = fire[(i + 1) % N][perm[i]]
        for j in range(N):
            m = (syn_slot == j) & (w_pre[pre_of] > 0)
            rwc_perm[i, j] = float((data[m] * w_pre[pre_of][m]).sum())
    perm_oo = [float(rwc_perm[i, i] / np.mean([rwc_perm[i, j] for j in range(N) if j != i]))
               if np.mean([rwc_perm[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]

    # RANDOM-CA1 control: read each slot's own/other with a RANDOM set of CA1 cells (size ~mean core, uniform weight).
    # If own/other for the winner fact is STILL high, the "selectivity" is a WINNER-SLOT weight artifact, not fact-specific.
    rng_ctrl = np.random.default_rng(int(seed) + 999)
    ncore = int(np.mean([c.size for c in core.values()])) or 10
    rand_cells = rng_ctrl.choice(ca1_idx.size, size=min(ncore, ca1_idx.size), replace=False)
    rand_oo = []
    for i in range(N):
        w_pre = np.zeros(csr.shape[0]); w_pre[ca1_idx[rand_cells]] = 1.0
        row = np.array([float((data[(syn_slot == j) & (w_pre[pre_of] > 0)]).sum()) for j in range(N)])
        rand_oo.append(float(row[i] / np.mean([row[j] for j in range(N) if j != i])) if np.mean([row[j] for j in range(N) if j != i]) > 1e-12 else 0.0)
    # PER-SLOT MEAN ca1->slot weight (the winner-slot smoking gun): if one slot's mean >> the others, the own/other is that.
    slot_mean_w = []
    for j in range(N):
        m = (syn_slot == j)
        slot_mean_w.append(round(float(data[m].mean()) if m.sum() > 0 else 0.0, 4))

    # SPARSE (code) ceiling for the cores (max own/other any write could reach on these cores)
    F = np.stack([fire[i] for i in range(N)])
    Fs = np.where(F > _thr, F, 0.0)
    Gs = Fs @ Fs.T
    sc = [float(Gs[i, i] / np.mean([Gs[i, j] for j in range(N) if j != i]))
          if np.mean([Gs[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]

    res = dict(seed=int(seed), thr_hash=thr_hash, dw=round(w1 - w0, 5), btsp_wmax=float(btsp_wmax),
               btsp_lr=float(btsp_lr), elig_tau=elig_tau, elig_hard_thresh=float(elig_hard_thresh),
               commit_top_k=commit_top_k, hippo_izh=hippo_izh_type, cycles=int(cycles),
               reset_elig=bool(reset_elig), fixed_order=bool(fixed_order), settle_steps=int(settle_steps), blocked=bool(blocked),
               core_thr_frac=float(core_thr_frac), core_sizes=core_sizes,
               core_gated_own_over_other=[round(x, 3) for x in rwc_oo],
               core_gated_own_is_max=rwc_max, n_pass=int(sum(1 for i in range(N) if rwc_oo[i] >= 2.5 and rwc_max[i])),
               permuted_core_own_over_other=[round(x, 3) for x in perm_oo],
               random_ca1_own_over_other=[round(x, 3) for x in rand_oo],
               slot_mean_weight=slot_mean_w,
               sparse_core_ceiling=[round(x, 3) for x in sc],
               m0_gated_ceiling=m0,
               xfire_under_tag=[[round(x, 2) for x in row] for row in xfire_tag.tolist()],
               xfire_during_write=[[round(x, 2) for x in row] for row in xfire_write.tolist()])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--v-teach", type=float, default=-25.0)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--btsp-lr", type=float, default=0.000003)
    ap.add_argument("--btsp-wmax", type=float, default=2000.0)
    ap.add_argument("--self-regen", type=float, default=0.15)
    ap.add_argument("--elig-exp", type=float, default=1.0)
    ap.add_argument("--hetero-dep", type=float, default=0.0)
    ap.add_argument("--hetero-theta", type=float, default=0.0)
    ap.add_argument("--commit-top-k", type=int, default=15)
    ap.add_argument("--tag-drive", type=float, default=1500.0)
    ap.add_argument("--hippo-izh-type", type=str, default="IZH2007_STRIATAL_MSN")
    ap.add_argument("--hippo-izh-regions", type=str, default="dg,ca3,ca1")
    ap.add_argument("--elig-hard-thresh", type=float, default=0.4)
    ap.add_argument("--elig-tau", type=float, default=30.0)
    ap.add_argument("--core-thr-frac", type=float, default=0.25)
    ap.add_argument("--reset-elig", action="store_true", help="reset BTSP eligibility between facts (test temporal bleed)")
    ap.add_argument("--fixed-order", action="store_true", help="fixed (not shuffled) fact order in the write loop")
    ap.add_argument("--settle-steps", type=int, default=0, help="settle steps between facts (eligibility decay + down-state)")
    ap.add_argument("--blocked", action="store_true", help="write ALL cycles of fact i before fact i+1 (full per-fact isolation)")
    ap.add_argument("--out", default="research/findings/raw/consol_opsweep_gpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run_seed(args.seed, v_teach=args.v_teach, cycles=args.cycles, btsp_lr=args.btsp_lr, self_regen=args.self_regen,
                 elig_exp=args.elig_exp, hetero_dep=args.hetero_dep, hetero_theta=args.hetero_theta,
                 commit_top_k=args.commit_top_k, tag_drive=args.tag_drive, hippo_izh_type=args.hippo_izh_type,
                 hippo_izh_regions=args.hippo_izh_regions, elig_hard_thresh=args.elig_hard_thresh, elig_tau=args.elig_tau,
                 btsp_wmax=args.btsp_wmax, reset_elig=args.reset_elig, fixed_order=args.fixed_order,
                 settle_steps=args.settle_steps, core_thr_frac=args.core_thr_frac, blocked=args.blocked)
    _tg = (args.tag or "") + (f"_reset" if args.reset_elig else "") + (f"_fixed" if args.fixed_order else "") + \
          (f"_blocked" if args.blocked else "") + \
          (f"_settle{args.settle_steps}" if args.settle_steps else "") + (f"_ctf{args.core_thr_frac:g}" if args.core_thr_frac != 0.25 else "")
    Path(f"{args.out}/twosided{_tg}_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    if "error" in r:
        print(f"[seed {args.seed}] ERROR: {r['error']}"); print("TWOSIDED-PROBE DONE", flush=True); return
    print(f"[seed {args.seed}] thr_hash={r['thr_hash']} dw={r['dw']} wmax={r['btsp_wmax']} lr={r['btsp_lr']} "
          f"reset={r['reset_elig']} fixed={r['fixed_order']} settle={r['settle_steps']}")
    print(f"  CORE-GATED own/other={r['core_gated_own_over_other']}  own_is_max={r['core_gated_own_is_max']}  "
          f"core_sizes={r['core_sizes']}  n_pass(>=2.5&max)={r['n_pass']}/{N}")
    print(f"  PERMUTED-CORE control own/other={r['permuted_core_own_over_other']}  <- must collapse to ~1.0 (else winner-slot artifact)")
    print(f"  RANDOM-CA1  control own/other={r['random_ca1_own_over_other']}  <- must collapse to ~1.0 (else winner-slot artifact)")
    print(f"  PER-SLOT mean ca1->slot weight={r['slot_mean_weight']}  <- one slot >> others = winner-slot artifact")
    print(f"  sparse_core_ceiling={r['sparse_core_ceiling']}  <- max own/other the code supports on these cores")
    _m0 = r.get("m0_gated_ceiling", {})
    if _m0:
        print("  === M0: GATED CEILING (does a per-source absolute spike-count gate recover fact-specificity?) ===")
        for src in ("during_write", "isolated_tag"):
            d = _m0[src]
            best, bestlab = 0.0, ""
            for kind in ("binary", "hill"):
                for frac, e in d[kind].items():
                    act = e["n_active"]
                    if min(act) < 2:          # degenerate-gate guard: a 0/1-cell gate is not a real code
                        continue
                    mv = float(np.mean(e["ceiling"]))
                    if mv > best:
                        best, bestlab = mv, f"{kind}@{frac}x(theta={e['theta']},n_active={act})"
            print(f"    {src:13s}: ungated={d['ungated_ceiling']} (mean {np.mean(d['ungated_ceiling']):.2f}) | BEST gated mean={best:.3f} via {bestlab}")
        # VERDICT on PER-FACT passes, never the MEAN. (2026-07-25: a mean-based verdict called this a GO when only
        # fact 0 — the FIRST-written fact — passed and facts 1/2 sat at 1.2-2.0 on 3-6-cell gates. That is the same
        # one-of-N artifact class as the winner-slot bias. A mechanism must work for MOST facts, not carry the mean.)
        _dw = _m0["during_write"]
        best_np, best_lab, best_cl = 0, "", []
        for kind in ("binary", "hill"):
            for frac, e in _dw[kind].items():
                if min(e["n_active"]) < 3:          # degenerate-gate guard: <3 cells makes own/other a small-number artifact
                    continue
                npass = sum(1 for c in e["ceiling"] if c >= 2.5)
                if npass > best_np:
                    best_np, best_lab, best_cl = npass, f"{kind}@{frac}x(n_active={e['n_active']})", e["ceiling"]
        _need = N - 1                                # require MOST facts (>= N-1), not one
        print(f"    M0 VERDICT: best per-fact passes = {best_np}/{N} via {best_lab or 'none'} ceilings={best_cl}")
        print(f"      {'GO — >=%d/%d facts clear 2.5 -> the gate has UNIFORM signal to amplify -> build M1' % (_need, N) if best_np >= _need else 'KILL — only %d/%d facts clear 2.5 (typically just the FIRST-written fact, on a fresh network) -> the write windows degrade specificity CUMULATIVELY across the schedule; a gate would amplify a flat signal for every later fact' % (best_np, N)}")
    print(f"  xfire[core_K under tag_W] (fire-under-tag):")
    for k in range(N):
        print(f"     core_{k} (size {r['core_sizes'][k]}): {r['xfire_under_tag'][k]}  <- diag should dominate if core is fact-specific")
    print(f"  xfire[core_K during write window_W] (the leak driver):")
    for k in range(N):
        print(f"     core_{k}: {r['xfire_during_write'][k]}  <- off-diag = core_K fires during OTHER facts' write -> leaks to their slot")
    print("TWOSIDED-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
