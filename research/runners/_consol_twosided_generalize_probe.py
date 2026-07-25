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
                       reset_elig=False, fixed_order=False, settle_steps=0, ca1_idx=None, blocked=False, write_order=None, reset_neurons=False, freeze_hippo=False):
    """Isolated-reinstatement + exclusive-apical-clamp teaching write (== decoupled_plateau_write) with instrumentation:
    captures per-fact-window CA1 firing, and optionally resets BTSP eligibility / settles / fixes order between facts.
    blocked=True: write ALL cycles of fact i before fact i+1 (full per-fact isolation — the decisive test of whether the
    leak is the interleaved schedule vs a deeper per-fact property)."""
    set_sleep_gates(bridge)
    if freeze_hippo:
        # 2026-07-25 (order-permutation follow-up): set_sleep_gates leaves ca3_to_ca1 PLASTIC (=1.0) during the write —
        # the very pathway that determines CA1's code for each tag. So the FIRST fact's write modifies ca3->ca1 and
        # every LATER fact's tag maps onto an altered CA1, which is exactly the positional/cumulative specificity loss
        # (and it survives eligibility-reset + settle + neuron-reset because it lives in the WEIGHTS). Freeze all the
        # named hippocampal/cortical gates, leaving ONLY ca1->comp_attr (the consolidation write) plastic.
        from research.runners.text_minimal_isolation import freeze_all_gates
        freeze_all_gates(bridge)
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
    # M1' diagnostic (2026-07-25): the per-cell PRESYNAPTIC ELIGIBILITY actually integrated over each window, and the
    # gate value that multiplies it. These separate "firing -> eligibility" from "eligibility -> weight": the write is
    # dw ~ eta * sum_t Etilde_k(t) * gate_k(t) * IS, so if window_elig tracks window_fire but the WEIGHT does not,
    # the break is downstream of the eligibility (thresholding / saturation / wiring), not in the count gate.
    window_elig = {i: np.zeros(ca1_h.size, dtype=np.float64) for i in range(len(facts))}
    window_gelig = {i: np.zeros(ca1_h.size, dtype=np.float64) for i in range(len(facts))}
    # M1' diagnostic: the INSTRUCTIVE SIGNAL actually seen by each slot during each fact's window,
    # IS_j = mean_t max(v_apical[slot_j](t) - plateau_v_hold, 0), measured INSIDE the step loop (i.e. AFTER the
    # bridge's own apical dynamics have run, not the clamp value we wrote). dw ~ Etilde_pre * IS_post, so if the
    # off-diagonal IS is not ~0 the "exclusive apical clamp" is not exclusive at the point the write reads it, and
    # every slot receives the SAME eligibility-weighted write -> no fact-specific structure for any gate to shape.
    slot_is = np.zeros((len(facts), len(facts)))
    slot_is_n = np.zeros(len(facts))
    slot_vap = np.zeros((len(facts), len(facts)))
    _pvh = float(getattr(bridge.core_config, "coincidence_plateau_v_hold", -35.0))
    rng = np.random.default_rng(int(seed) + 777)

    # M1' (2026-07-25): per-fact-window gate-engagement stats (CA1 cells whose BOX-CAR count clears theta at the
    # end of the burst = the sources the write actually integrates). Empty when the gate is off.
    gate_stats = {}

    def _one_fact_burst(i):
        tag = tags[i]
        if reset_elig and getattr(bridge, "cp_btsp_pre_elig", None) is not None:
            bridge.cp_btsp_pre_elig[:] = cp.float32(0.0)
        # M1' BOX-CAR RESET: zero the windowed spike count at burst ONSET. Without this the count is cumulative
        # across facts and is no longer a per-window count (the reset is part of the primitive, not an optimisation).
        bridge.reset_btsp_window()
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
            if bridge.cp_v_apical is not None:
                _va = to_host(bridge.cp_v_apical).astype(np.float64)
                for _j in range(len(facts)):
                    _sj = slot_idx[_j]
                    if _sj is not None:
                        slot_is[i, _j] += float(np.maximum(_va[to_host(_sj).astype(np.int64)] - _pvh, 0.0).mean())
                        slot_vap[i, _j] += float(_va[to_host(_sj).astype(np.int64)].mean())
                slot_is_n[i] += 1
            if getattr(bridge, "cp_btsp_pre_elig", None) is not None:
                _e = to_host(bridge.cp_btsp_pre_elig).astype(np.float64)[ca1_h]
                window_elig[i] += _e
                if getattr(bridge, "cp_btsp_win_count", None) is not None:
                    _c = to_host(bridge.cp_btsp_win_count).astype(np.float64)[ca1_h]
                    _th = float(getattr(bridge.core_config, "btsp_win_gate_theta", 0.0))
                    _hn = float(getattr(bridge.core_config, "btsp_win_gate_hill_n", 8.0))
                    window_gelig[i] += _e * (_c ** _hn / (_c ** _hn + _th ** _hn + 1e-30))
                else:
                    window_gelig[i] += _e
        # M1' gate engagement, measured at the END of the burst (before any settle/reset), on the CA1 sources.
        if getattr(bridge, "cp_btsp_win_count", None) is not None:
            _wc = to_host(bridge.cp_btsp_win_count).astype(np.float64)
            _th = float(getattr(bridge.core_config, "btsp_win_gate_theta", 0.0))
            _ca1c = _wc[ca1_h]
            st = gate_stats.setdefault(i, {"pass_ca1": [], "n_ca1": int(_ca1c.size), "max_count": [],
                                           "mean_count": [], "pass_all": [], "n_all": int(_wc.size)})
            st["pass_ca1"].append(int((_ca1c >= _th).sum()))
            st["max_count"].append(float(_ca1c.max()))
            st["mean_count"].append(round(float(_ca1c.mean()), 3))
            st["pass_all"].append(int((_wc >= _th).sum()))
        try:
            bridge.clear_tag_drive(tag)
        except Exception:
            pass
        if bridge.cp_v_apical is not None:
            bridge.cp_v_apical[:] = cp.float32(Er)
        if reset_neurons:
            # TRUE per-fact network re-initialisation (2026-07-25 order-permutation follow-up). The permutation proved
            # the specificity loss is POSITIONAL: whichever fact is written FIRST keeps it (~1.75) and every later fact
            # is flat (~0.85), with eligibility-reset + settle ALREADY on. So what carries over is NEURON STATE, not
            # eligibility: spike-triggered adaptation (the MSN phenotype's d_increment=150 loads cp_recovery_variable_u),
            # membrane state, and short-term synaptic depression. Restore all three to rest so every fact sees the same
            # fresh network the first one does.
            _Vr = float(getattr(bridge.core_config, 'izh_c_reset', -65.0))
            if getattr(bridge, 'cp_membrane_potential_v', None) is not None:
                bridge.cp_membrane_potential_v[:] = cp.float32(_Vr)
            if getattr(bridge, 'cp_recovery_variable_u', None) is not None:
                bridge.cp_recovery_variable_u[:] = cp.float32(0.0)
            if getattr(bridge, 'cp_stp_x', None) is not None:
                bridge.cp_stp_x[:] = cp.float32(1.0)
            if getattr(bridge, 'cp_stp_u', None) is not None:
                bridge.cp_stp_u[:] = cp.float32(float(getattr(bridge.core_config, 'stp_U', 0.5)))
        if settle_steps > 0:      # let eligibility decay + slots return to down-state before the next fact
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(int(settle_steps)):
                if bridge.cp_v_apical is not None:
                    bridge.cp_v_apical[all_slots] = cp.float32(Er)
                bridge._run_one_simulation_step()

    # WRITE ORDER (2026-07-25 M0 follow-up): blocked mode always wrote 0,1,2, so "fact 0" and "written first" were
    # perfectly confounded — and M0 found only the first-written fact keeps its specificity. write_order permutes the
    # schedule to separate them: if the passing fact FOLLOWS the order, the degradation is cumulative-across-schedule;
    # if fact 0 passes even when written last, the cause is specific to fact 0 instead.
    order0 = list(write_order) if write_order else list(range(len(facts)))
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
    return {"window_fire": window_fire, "gate_stats": gate_stats,
            "window_elig": window_elig, "window_gelig": window_gelig,
            "slot_is": slot_is / np.maximum(slot_is_n, 1)[:, None],
            "slot_vap": slot_vap / np.maximum(slot_is_n, 1)[:, None], "plateau_v_hold": _pvh}


def run_seed(seed, v_teach=-25.0, cycles=3, btsp_lr=0.000003, self_regen=0.15, tag_drive=1500.0,
             elig_exp=1.0, hetero_dep=0.0, hetero_theta=0.0, commit_top_k=15,
             hippo_izh_type="IZH2007_STRIATAL_MSN", hippo_izh_regions="dg,ca3,ca1",
             elig_hard_thresh=0.4, elig_tau=30.0, btsp_wmax=2000.0,
             reset_elig=False, fixed_order=False, settle_steps=0, core_thr_frac=0.25, blocked=False, write_order=None, reset_neurons=False, freeze_hippo=False,
             btsp_win_theta=0.0, btsp_win_hill_n=8.0, apical_R=None, gc_read=None, encode_btsp_lr=0.0):
    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=float(self_regen),
             comp_kir_g=3.0, comp_v_hold=-50.0,
             comp_btsp=True, comp_btsp_lr=float(btsp_lr), comp_btsp_wmax=float(btsp_wmax),
             comp_btsp_elig_exp=float(elig_exp), comp_btsp_hetero_dep=float(hetero_dep),
             comp_btsp_hetero_theta=float(hetero_theta), comp_btsp_elig_hard_thresh=float(elig_hard_thresh),
             # M1' dendritic sustained-count gate (0.0 => OFF => byte-identical to the pre-edit substrate)
             comp_btsp_win_gate_theta=float(btsp_win_theta), comp_btsp_win_gate_hill_n=float(btsp_win_hill_n))
    if apical_R is not None:
        # M1' follow-up lever (2026-07-25): the apical fixed point is ~ Er + comp_apical_R * I_coincidence, so at the
        # shipped R=50 the apical parks at ~1.9e5 mV and the +-45 mV teaching clamp is numerically irrelevant -> the
        # instructive signal is only 3.5:1 selective instead of exclusive. Lowering R is the direct test of whether an
        # EXCLUSIVE instructive signal is reachable at a physiological operating point.
        a.update(comp_apical_R=float(apical_R))
    if gc_read is not None:
        # 2026-07-25: apical->soma READ conductance. At the shipped gc_read=5.0 the apical DOMINATES every soma; combined
        # with the miscalibrated R=50 that is what produced the 93%-active "dense CA1 code" the whole arc mis-diagnosed as
        # a boundary. At a physiological R, gc_read=5.0 instead SILENCES CA1 (apical sits below rest and clamps the soma).
        # gc_read must be retuned WITH R: (R=0.15, gc_read=0.5) restores a sparse, near-disjoint, fact-specific CA1 code.
        a.update(comp_gc_read=float(gc_read))
    if elig_tau is not None:
        a.update(comp_btsp_elig_tau=float(elig_tau))
    if hippo_izh_type:
        a.update(hippo_izh_type=str(hippo_izh_type), hippo_izh_regions=str(hippo_izh_regions))
    b = build_substrate(seed, SimpleNamespace(**a))
    thr_hash = hashlib.md5(to_host(b.cp_neuron_firing_thresholds).tobytes()).hexdigest()[:12]
    rm = b.region_manager
    ca1_idx = np.asarray(sorted(rm.indices("ca1")), dtype=np.int64)
    slot_idx = {s: np.asarray(sorted(rm.indices(f"comp_attr_{s}")), dtype=np.int64) for s in range(N)}
    # ENCODE/WRITE learning-rate SEPARATION (2026-07-25): BTSP is active during encode_facts_with_reinstatement, so a
    # write-scale btsp_lr silently CORRUPTS the codes before they are ever measured (observed: core_sizes=[3,7,112] at
    # lr=1e-2). Encoding must lay down the codes with the write rule quiescent; only the consolidation write that follows
    # should learn. cfg is read per-step by the bridge, so setting it at runtime takes effect immediately.
    _wr_lr = float(b.core_config.btsp_learning_rate)
    b.core_config.btsp_learning_rate = float(encode_btsp_lr)
    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS, commit_top_k=commit_top_k)
    # NOTE: the write lr stays OFF through the fire-under-tag MEASUREMENT below as well — restoring it here made the
    # measurement itself plastic (BTSP learned while the codes were being read), which inflated core_sizes from [2,1,2]
    # to [22,120,120] purely as a function of the WRITE lr. A measurement must never be plastic.
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
    b.core_config.btsp_learning_rate = _wr_lr      # write rule ON only for the consolidation write itself
    inst = instrumented_write(b, CONSOLIDATED_FACTS, tags, int(cycles), seed, v_teach=float(v_teach),
                              reinstate_drive=float(tag_drive), reset_elig=reset_elig, fixed_order=fixed_order,
                              settle_steps=int(settle_steps), ca1_idx=ca1_idx, blocked=blocked, write_order=write_order, reset_neurons=reset_neurons, freeze_hippo=freeze_hippo)
    gate_stats = inst.get("gate_stats", {})
    we = inst.get("window_elig", {}); wge = inst.get("window_gelig", {})
    slot_is_m = inst.get("slot_is"); slot_vap_m = inst.get("slot_vap"); _pvh_used = inst.get("plateau_v_hold")
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
        # MAGNITUDE-FREE specificity (2026-07-25): the raw ceiling Sum(x_i^2)/Sum(x_i.x_j) rewards whichever window simply
        # FIRES MORE — and the first-written window fires on a fresh network. That is a mass artifact of the same class as
        # the winner-slot bias. Cosine specificity divides it out: spec_i = 1 / mean_j!=i cos(x_i, x_j). If the positional
        # effect survives here it is REAL code specificity; if it vanishes, the "positional specificity" was just firing mass.
        nrm = np.linalg.norm(X, axis=1); U = X / np.maximum(nrm, 1e-12)[:, None]
        cos_spec = []
        for i in range(N):
            m = float(np.mean([float(U[i] @ U[j]) for j in range(N) if j != i]))
            cos_spec.append(1.0 / m if m > 1e-12 else 0.0)
        res = {"max_count": round(xmax, 1), "ungated_ceiling": [round(v, 3) for v in _ceiling(X)],
               "total_spikes": [round(float(X[i].sum()), 1) for i in range(N)],
               "cosine_specificity": [round(v, 3) for v in cos_spec], "binary": {}, "hill": {}}
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

    # MASS-FREE TWIN of core_gated_own_over_other (verify-go lens 7b, 2026-07-25 hard rule). The row-i ratio compares
    # slots j, so the confound is PER-SLOT WEIGHT MASS: divide each column by that slot's mean ca1->slot weight and the
    # ratio can no longer be bought by one slot simply being heavier. If own/other survives slot-normalisation the
    # selectivity is in the PATTERN; if it collapses, it was mass. Reported ALONGSIDE the raw ratio, never instead of it.
    _smw = np.array(slot_mean_w, dtype=np.float64)
    rwc_n = rwc / np.maximum(_smw, 1e-12)[None, :]
    rwc_oo_norm = [float(rwc_n[i, i] / np.mean([rwc_n[i, j] for j in range(N) if j != i]))
                   if np.mean([rwc_n[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
    slot_mass_ratio = round(float(max(slot_mean_w) / max(min(slot_mean_w), 1e-12)), 3)

    # ---- M1' WRITE-FIDELITY / TWO-SIDED-MATCH DIAGNOSTICS (2026-07-25). If the realized own/other stays flat while
    # the M0 oracle predicted 2.9-4.0, exactly one of two things is true and these separate them:
    #   (A) the WRITE does not store the (gated) during-write code   -> w_fidelity_gated ~ 0
    #   (B) the write DOES store it, but the RECALL cue (the isolated tag's core) addresses a DIFFERENT cell set than
    #       the write window did -> w_fidelity_gated high, jaccard(write-core, isolated-core) low, and the
    #       SELF-CUED read (cue with the same during-write gated set the write saw) is high while the
    #       isolated-tag-cued read is flat.
    # NOTE: M0's ceiling is SYMMETRIC (the same during-write code on both sides). The deployed metric is NOT: it
    # cues with the isolated tag. self_cued_own_over_other is the honest in-run reproduction of M0's ceiling
    # condition — it is a DIAGNOSTIC, not the GO metric (cueing with the write window is circular as a capability).
    w_to_slot = np.zeros((N, ca1_idx.size))     # w_to_slot[j, k] = summed ca1_k -> slot_j weight
    _pos_of = {int(g): kk for kk, g in enumerate(ca1_idx)}
    for j in range(N):
        m = (syn_slot == j)
        pres = pre_of[m]; vals = data[m]
        for p, v in zip(pres, vals):
            kk = _pos_of.get(int(p))
            if kk is not None:
                w_to_slot[j, kk] += v

    def _corr(a, b):
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    _wth = float(btsp_win_theta)
    WFm = np.stack([wf[i] for i in range(N)])
    write_core = {j: np.where(WFm[j] >= _wth)[0] for j in range(N)} if _wth > 0 else \
                 {j: np.where(WFm[j] >= 0.5 * WFm.max())[0] for j in range(N)}
    w_fid_raw = [round(_corr(w_to_slot[j], WFm[j]), 3) for j in range(N)]
    w_fid_gated = [round(_corr(w_to_slot[j], (WFm[j] >= _wth).astype(float) if _wth > 0 else WFm[j]), 3) for j in range(N)]
    w_fid_isolated = [round(_corr(w_to_slot[j], fire[j]), 3) for j in range(N)]
    wc_iso_jaccard = [round(_jac(write_core[j], core[j]), 3) for j in range(N)]
    write_core_sizes = [int(write_core[j].size) for j in range(N)]
    # SELF-CUED read: weight the read by the write-window code itself (M0's symmetric condition).
    self_cued = np.zeros((N, N))
    for i in range(N):
        gi = (WFm[i] >= _wth).astype(np.float64) * WFm[i] if _wth > 0 else WFm[i]
        for j in range(N):
            self_cued[i, j] = float((gi * w_to_slot[j]).sum())
    self_cued_oo = [round(float(self_cued[i, i] / np.mean([self_cued[i, j] for j in range(N) if j != i])), 3)
                    if np.mean([self_cued[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]
    # cross-condition per-cell correlation: does a cell's isolated-tag firing predict its during-write firing?
    iso_vs_write_corr = [round(_corr(fire[j], WFm[j]), 3) for j in range(N)]
    # THE LOCALISER: firing -> eligibility -> weight, one link at a time.
    elig_vs_fire = [round(_corr(we[j], WFm[j]), 3) if j in we else 0.0 for j in range(N)]
    w_vs_elig = [round(_corr(w_to_slot[j], we[j]), 3) if j in we else 0.0 for j in range(N)]
    w_vs_gelig = [round(_corr(w_to_slot[j], wge[j]), 3) if j in wge else 0.0 for j in range(N)]
    w_cv = [round(float(w_to_slot[j].std() / max(abs(w_to_slot[j].mean()), 1e-12)), 3) for j in range(N)]
    w_cross_corr = [[round(_corr(w_to_slot[i], w_to_slot[j]), 3) for j in range(N)] for i in range(N)]
    # CLOSING ATTRIBUTION: the BTSP rule is dw[k->slot_j] ~ eta * sum_i sum_t Etilde_i(k,t)*gate * IS_j(t), so with the
    # MEASURED per-window gated eligibility E_i and the MEASURED instructive-signal matrix IS[i,j] the realized weight
    # is predicted by  w_pred[k,j] = sum_i E_i[k] * IS[i,j].  If w_pred reproduces the realized own/other, the write is
    # FULLY explained by (eligibility x instructive signal) and the only lever left is IS's off-diagonal leak.
    pred_oo, pred_corr = [], []
    if slot_is_m is not None and len(wge) == N:
        Emat = np.stack([wge[i] for i in range(N)])                 # (N, n_ca1) gated per-window eligibility
        Wpred = Emat.T @ np.asarray(slot_is_m)                      # (n_ca1, N)
        pred_corr = [round(_corr(w_to_slot[j], Wpred[:, j]), 3) for j in range(N)]
        for i in range(N):
            w_pre_i = np.zeros(ca1_idx.size); w_pre_i[core[i]] = fire[i][core[i]]
            row = np.array([float((w_pre_i * Wpred[:, j]).sum()) for j in range(N)])
            oth = float(np.mean([row[j] for j in range(N) if j != i]))
            pred_oo.append(round(float(row[i] / oth), 3) if oth > 1e-12 else 0.0)

    # SPARSE (code) ceiling for the cores (max own/other any write could reach on these cores)
    F = np.stack([fire[i] for i in range(N)])
    Fs = np.where(F > _thr, F, 0.0)
    Gs = Fs @ Fs.T
    sc = [float(Gs[i, i] / np.mean([Gs[i, j] for j in range(N) if j != i]))
          if np.mean([Gs[i, j] for j in range(N) if j != i]) > 1e-12 else 0.0 for i in range(N)]

    res = dict(seed=int(seed), thr_hash=thr_hash, dw=round(w1 - w0, 5), btsp_wmax=float(btsp_wmax),
               btsp_lr=float(btsp_lr), elig_tau=elig_tau, elig_hard_thresh=float(elig_hard_thresh),
               commit_top_k=commit_top_k, hippo_izh=hippo_izh_type, cycles=int(cycles),
               reset_elig=bool(reset_elig), fixed_order=bool(fixed_order), settle_steps=int(settle_steps), blocked=bool(blocked), write_order=(list(write_order) if write_order else None),
               core_thr_frac=float(core_thr_frac), core_sizes=core_sizes,
               core_gated_own_over_other=[round(x, 3) for x in rwc_oo],
               core_gated_own_is_max=rwc_max, n_pass=int(sum(1 for i in range(N) if rwc_oo[i] >= 2.5 and rwc_max[i])),
               core_gated_own_over_other_slotnorm=[round(x, 3) for x in rwc_oo_norm],
               slot_mass_ratio=slot_mass_ratio,
               btsp_win_theta=float(btsp_win_theta), btsp_win_hill_n=float(btsp_win_hill_n),
               win_gate_stats={str(k): v for k, v in gate_stats.items()},
               w_fidelity_raw_count=w_fid_raw, w_fidelity_gated_count=w_fid_gated,
               w_fidelity_isolated_fire=w_fid_isolated,
               write_core_sizes=write_core_sizes, write_core_vs_isolated_core_jaccard=wc_iso_jaccard,
               self_cued_own_over_other=self_cued_oo, iso_vs_write_percell_corr=iso_vs_write_corr,
               elig_vs_fire_corr=elig_vs_fire, w_vs_elig_corr=w_vs_elig, w_vs_gated_elig_corr=w_vs_gelig,
               w_to_slot_cv=w_cv, w_to_slot_cross_corr=w_cross_corr,
               predicted_own_over_other=pred_oo, predicted_vs_realized_w_corr=pred_corr,
               slot_v_apical_mean=[[round(x, 1) for x in row] for row in (slot_vap_m.tolist() if slot_vap_m is not None else [])],
               plateau_v_hold_used=_pvh_used,
               slot_instructive_signal=[[round(x, 3) for x in row] for row in (slot_is_m.tolist() if slot_is_m is not None else [])],
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
    ap.add_argument("--freeze-hippo", action="store_true", help="freeze ca3->ca1 etc during the write (only ca1->comp_attr plastic) — tests whether the WRITE corrupts the hippocampal code")
    ap.add_argument("--reset-neurons", action="store_true", help="TRUE per-fact network re-init between facts (v/u/STP to rest) — tests whether ADAPTATION state is what accumulates")
    ap.add_argument("--write-order", type=str, default=None, help="comma-sep fact write order, e.g. 2,1,0 — separates 'fact 0' from 'written first'")
    ap.add_argument("--blocked", action="store_true", help="write ALL cycles of fact i before fact i+1 (full per-fact isolation)")
    ap.add_argument("--btsp-win-theta", type=float, default=0.0,
                    help="M1': ABSOLUTE windowed-spike-count threshold for the dendritic sustained-count write gate "
                         "(0.0 = OFF = byte-identical). Counts run ~0-40 over a 30-step burst; try 5/8/10/12/15.")
    ap.add_argument("--encode-btsp-lr", type=float, default=0.0, help="btsp_lr DURING encode (default 0 = write rule quiescent while codes are laid down)")
    ap.add_argument("--comp-gc-read", type=float, default=None, help="apical->soma read conductance; must be retuned WITH --comp-apical-R (physiological pair: R=0.15 gc_read=0.5)")
    ap.add_argument("--comp-apical-R", type=float, default=None,
                    help="override comp_apical_R (default 50.0): the apical fixed point is ~Er + R*I_coincidence")
    ap.add_argument("--btsp-win-hill-n", type=float, default=8.0, help="M1': Hill cooperativity of the count gate (CaMKII ~8)")
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
                 settle_steps=args.settle_steps, core_thr_frac=args.core_thr_frac, blocked=args.blocked,
                 write_order=[int(x) for x in args.write_order.split(',')] if args.write_order else None,
                 reset_neurons=args.reset_neurons, freeze_hippo=args.freeze_hippo,
                 btsp_win_theta=args.btsp_win_theta, btsp_win_hill_n=args.btsp_win_hill_n,
                 apical_R=args.comp_apical_R, gc_read=args.comp_gc_read, encode_btsp_lr=args.encode_btsp_lr)
    _tg = (args.tag or "") + (f"_wt{args.btsp_win_theta:g}" if args.btsp_win_theta > 0 else "") + \
          (f"_reset" if args.reset_elig else "") + (f"_fixed" if args.fixed_order else "") + \
          (f"_blocked" if args.blocked else "") + \
          (f"_settle{args.settle_steps}" if args.settle_steps else "") + (f"_ctf{args.core_thr_frac:g}" if args.core_thr_frac != 0.25 else "")
    Path(f"{args.out}/twosided{_tg}_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    if "error" in r:
        print(f"[seed {args.seed}] ERROR: {r['error']}"); print("TWOSIDED-PROBE DONE", flush=True); return
    print(f"[seed {args.seed}] thr_hash={r['thr_hash']} dw={r['dw']} wmax={r['btsp_wmax']} lr={r['btsp_lr']} "
          f"reset={r['reset_elig']} fixed={r['fixed_order']} settle={r['settle_steps']}")
    print(f"  CORE-GATED own/other={r['core_gated_own_over_other']}  own_is_max={r['core_gated_own_is_max']}  "
          f"core_sizes={r['core_sizes']}  n_pass(>=2.5&max)={r['n_pass']}/{N}")
    print(f"  MASS-FREE twin (slot-weight-normalized) own/other={r['core_gated_own_over_other_slotnorm']}  "
          f"slot_mass_ratio(max/min)={r['slot_mass_ratio']} <- ratio must SURVIVE slot-normalisation (else it was mass)")
    if r.get("btsp_win_theta", 0.0) > 0:
        print(f"  M1' WIN-GATE theta={r['btsp_win_theta']} hill_n={r['btsp_win_hill_n']}  per-fact engagement "
              f"(CA1 sources clearing theta at burst end, of {next(iter(r['win_gate_stats'].values()))['n_ca1'] if r['win_gate_stats'] else '?'}):")
        for k in sorted(r["win_gate_stats"], key=lambda x: int(x)):
            g = r["win_gate_stats"][k]
            print(f"     fact {k}: pass_ca1={g['pass_ca1']} (frac={[round(p / max(g['n_ca1'],1), 3) for p in g['pass_ca1']]})  "
                  f"max_count={g['max_count']}  mean_count={g['mean_count']}  pass_bridge_wide={g['pass_all']}/{g['n_all']}")
    print(f"  WRITE-FIDELITY corr(w->slot_j, during-write count_j)={r['w_fidelity_raw_count']}  "
          f"corr(w, GATED count)={r['w_fidelity_gated_count']}  corr(w, isolated fire)={r['w_fidelity_isolated_fire']}")
    print(f"  WRITE-CORE sizes={r['write_core_sizes']}  jaccard(write-core, isolated-core)={r['write_core_vs_isolated_core_jaccard']}  "
          f"per-cell corr(isolated, during-write)={r['iso_vs_write_percell_corr']}")
    print(f"  LOCALISER  corr(window_elig, window_fire)={r['elig_vs_fire_corr']}  corr(w, window_elig)={r['w_vs_elig_corr']}  "
          f"corr(w, GATED window_elig)={r['w_vs_gated_elig_corr']}")
    print(f"  w->slot CV(per-cell spread)={r['w_to_slot_cv']}  cross-slot corr={r['w_to_slot_cross_corr']}  "
          f"<- CV~0 or cross-corr~1 => the write is UNIFORM/shared, not fact-specific")
    print(f"  PREDICTED own/other from (measured gated eligibility x measured IS matrix)={r['predicted_own_over_other']}  "
          f"corr(w_realized, w_predicted)={r['predicted_vs_realized_w_corr']}")
    print(f"  v_apical[window_i, slot_j] mean (mV, INSIDE the step)={r['slot_v_apical_mean']}  plateau_v_hold={r['plateau_v_hold_used']}")
    print(f"  IS[window_i, slot_j] (mean instructive signal INSIDE the step)={r['slot_instructive_signal']}  "
          f"<- off-diagonal must be ~0 or the apical clamp is NOT exclusive at write time")
    print(f"  SELF-CUED own/other (cue = the write-window code itself; M0's SYMMETRIC condition, DIAGNOSTIC not GO)"
          f"={r['self_cued_own_over_other']}")
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
            print(f"    {'':13s}  total_spikes={d.get('total_spikes')}  COSINE-spec (magnitude-free)={d.get('cosine_specificity')}")
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
