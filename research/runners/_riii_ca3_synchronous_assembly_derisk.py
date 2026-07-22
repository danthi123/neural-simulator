"""R-iii gap#5 — SYNCHRONY-ISOLATION de-risk: does DIRECT SYNCHRONOUS assembly encoding grow the within-ensemble
CA3 recurrent weights to the completion scale (the diagnostic-pinned residual)?

2026-07-18. The 2026-07-14 arc pinned the functional-completion blocker: the learned within-ensemble weights stay
~7.5 (co-activity-limited), ~200× below the ~1600 the hand-installed attractor needs, because the members fire
ASYNCHRONOUSLY. This session's cap-vs-synchrony test confirmed it (hebb_max 30→2000 byte-identical → NOT the cap).
The 2026-07-14 finding named but NEVER BUILT the fix: "my tests drove the UPSTREAM input, not the assembly cells
directly" (Kopsick-Ascoli 2024: drive the assembly PCs together in a gamma window → dense co-firing → strong LTP).

This de-risk ISOLATES that hypothesis: pre-assign a sparse CA3 assembly per pattern, drive THOSE cells DIRECTLY with
strong SYNCHRONOUS gamma-pulsed current during encoding (all fire together each ON window), rate-window LTP + the
committed EMERGE-40 competition ON, then recall a 50% partial cue directly on CA3 → does the held-out 50% FIRE
(functional pattern completion)? GO = h_comp≥0.30 & ≥2× non-stored, competition load-bearing (lam=0 vs 0.5),
async-control collapses (sync OFF → back to the ~7.5 weak weights → no completion). If GO → synchrony IS the fix →
the follow-on wires the EMERGENT mossy/DG assembly selection (experience-derived). If NO even with perfect synchrony
→ a deeper issue. GPU.
"""
import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates  # noqa: E402
from research.runners._riii_ca3_competitive_completion_payoff_derisk import _extract_ca3ca3_coincidence  # noqa: E402
from research.runners.validate_trisynaptic_loop import measure_region_response  # noqa: E402


def _extract_ca3ca3_all(bridge, ca3_idx, to_host):
    """ca3->ca3 synapses (ALL, NO coincidence mask) -> (flat_pos, pre_local, post_local). For the Wang nmda_slow mode,
    where cp_coincidence_synapse_mask is None (coincidence detection off)."""
    conn = bridge.cp_connections
    nnz = int(conn.nnz)
    indptr = np.asarray(to_host(conn.indptr)); indices = np.asarray(to_host(conn.indices))
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    post_of = indices[:nnz]
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    ca3_set = set(ca3_pos.keys())
    flat, pre_l, post_l = [], [], []
    for k in range(nnz):
        pre, post = int(pre_of[k]), int(post_of[k])
        if pre in ca3_set and post in ca3_set:
            flat.append(k); pre_l.append(ca3_pos[pre]); post_l.append(ca3_pos[post])
    return (np.asarray(flat, dtype=np.int64), np.asarray(pre_l, dtype=np.int64), np.asarray(post_l, dtype=np.int64))


def run(seed, n_ca3=1000, n_mem=2, assembly_frac=0.012, train_events=120, sync_on=2, sync_off=4,
        encode_drive=700.0, recall_drive=250.0, lam_dep_wi=0.5, hebb_max=2000.0, ca3_fb_inhib=20.0,
        reset_steps=15, drive_steps=48, recall_steps=60, ens_thresh=2, no_sync=False,
        coact_thresh=0.02, hebb_lr=None, k_thresh=18.0, plateau_strength=120.0, apical_R=50.0, apical_gc=None,
        permute_recall=False, bistable=False, nmda_recurrent=False, nmda_tau=100.0, nmda_ratio=1.0,
        homeostatic=False, homeo_target=None,
        rate_homeo=False, rate_homeo_target=0.02, rate_homeo_alpha=0.1, rate_homeo_adapt=15.0,
        rate_homeo_steps=400, rate_homeo_cap=800.0, enable_ou=True, ca3_density=0.5,
        selective_inhib=False, sel_inhib_spare=0.0, recall_k_thresh=None, structural_sep=False,
        plateau_self_regen=0.0, plateau_v_hold=-35.0, apical_kir_g=0.0, apical_gc_read=None, read_apical=False,
        read_ca1=False, schaffer_boost=1.0,
        encode_btsp=False, btsp_lr=0.02, encode_ca3w=None, encode_plateau_pA=250.0, encode_structural_sep=0,
        encode_hetero=0.0, encode_btsp_hetero=0.0, assemblies_ext=None, swr_ripple_pA=800.0, swr_ca1_ff_inhib=None,
        swr_learn_schaffer=False, swr_target_frac=0.15, swr_schaffer_hi=60.0, swr_schaffer_lo=0.2, swr_disjoint=False,
        swr_ca1_topk=None, interassembly_isolate=False,
        per_assembly_sel_inhib=False, per_assembly_inhib_w=40.0, per_assembly_ei_w=None,
        per_assembly_apical_inhib=False, per_assembly_apical_w=0.7, per_assembly_apical_gate=2.0,
        per_assembly_apical_spare_own=True,
        swr_disjoint_targets=False):
    # DIAGNOSED LEVERS (2026-07-18 workflow): the rate-window LTP is an EMA-trace rule -- a cell's co-activity trace
    # tops out ~0.03-0.2 (point Izh fires ~0.2 duty @700pA), so coact_thresh MUST be BELOW it (~0.02) or nothing
    # potentiates; the gamma OFF-gap DECAYS the EMA (0.9^off) so CONTINUOUS drive (sync_off<=1) is required, NOT
    # synchrony; higher hebb_lr + strong drive (~3000pA -> ~0.5 duty) climb the weight toward the completion scale.
    from sim.backend import get_backend, to_host
    from sim.kernels import fused_htm_winner_inactive_depression
    cp, _ = get_backend()
    # WANG-2002 mode (nmda_recurrent): the ca3->ca3 recurrent is SOMATIC slow-NMDA (the bistable attractor itself);
    # the dendritic-coincidence dAP readout (coincidence/two_comp) is OFF. Else: the dAP-coincidence readout (default).
    _init_ca3w = float(encode_ca3w) if (encode_btsp and encode_ca3w is not None) else 6.0   # BTSP-encode: init recurrent LOW so BTSP builds it
    bridge = _build(seed, n_ca3=n_ca3, ca3w=_init_ca3w, ca3_density=ca3_density,
                    coincidence=(not nmda_recurrent), two_comp=(not nmda_recurrent),
                    nmda_recurrent=nmda_recurrent, nmda_tau=nmda_tau, nmda_ratio=nmda_ratio, apical_R=apical_R,
                    apical_gc=apical_gc, k_thresh=k_thresh, plateau_strength=plateau_strength,
                    train=True, hebb_max=hebb_max, hebb_rate=True, ca3_fb_inhib=ca3_fb_inhib,
                    coact_thresh=coact_thresh, hebb_lr=hebb_lr, enable_ou=enable_ou,
                    plateau_self_regen=plateau_self_regen, plateau_v_hold=plateau_v_hold, apical_kir_g=apical_kir_g,
                    apical_gc_read=apical_gc_read, ca1_ff_inhib=swr_ca1_ff_inhib)
    if os.environ.get("SWR_NO_STP"):
        bridge.core_config.enable_short_term_plasticity = False   # DIAGNOSTIC: does the ca3->ca1 boost reach g_e without STP?
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3")); ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    n = bridge.core_config.num_neurons
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    rng = np.random.default_rng(seed * 17 + 3)

    # PRE-ASSIGN sparse assemblies (~1% of CA3), disjoint-ish (random draw).
    # NOTE (2026-07-18): an EXCITATORY-only assembly was tried to remove ca1 g_i for the SWR read, but it BROKE the
    # completion (cue 0.29->0.038) -- the completion's recurrent dynamics depend on the assembly's full composition. The
    # ca1 inhibition and the completion are COUPLED; the SWR ca1-drive is a hard fresh-pass integration, not a quick fix.
    n_assy = max(6, int(assembly_frac * n_ca3))
    if assemblies_ext is not None:
        # EMERGENT-DG integrated select-and-store: use the externally-SELECTED assemblies (e.g. mossy-seeded from a
        # synchronized DG volley) instead of the random draw. Each entry = global CA3 indices; the rest of run() (BTSP
        # store, structural_sep, selective_inhib, bistable completion, anti-cheats) is parameterized purely by this list.
        assemblies = [np.asarray(sorted(int(x) for x in a), dtype=np.int64) for a in assemblies_ext]
        n_mem = len(assemblies)
    elif swr_disjoint:
        # DISJOINT assemblies (SWR specificity test): draw all n_mem*n_assy cells from ONE without-replacement pool so
        # the assemblies share NO cells -> removes the overlap that seeds cross-assembly completion spreading.
        _pool = rng.choice(ca3_idx, n_assy * n_mem, replace=False)
        assemblies = [np.asarray(sorted(_pool[i * n_assy:(i + 1) * n_assy]), dtype=np.int64) for i in range(n_mem)]
    else:
        assemblies = [np.asarray(sorted(rng.choice(ca3_idx, n_assy, replace=False)), dtype=np.int64) for _ in range(n_mem)]

    _extract = _extract_ca3ca3_all if nmda_recurrent else _extract_ca3ca3_coincidence
    flat_h, pre_l_h, post_l_h = _extract(bridge, ca3_idx, to_host)
    conn = bridge.cp_connections
    do_comp = lam_dep_wi > 0.0 and len(flat_h) > 0
    if do_comp:
        flat_pos = cp.asarray(flat_h, dtype=cp.int64)
        pre_local = cp.asarray(pre_l_h, dtype=cp.int64)
        post_local = cp.asarray(post_l_h, dtype=cp.int64)

    def _apply_competition(member_mask_local):
        fpre = member_mask_local[pre_local]; fpost = member_mask_local[post_local]
        w = conn.data[flat_pos]
        w = fused_htm_winner_inactive_depression(w, fpre, fpost, float(lam_dep_wi), 0.0, float(hebb_max))
        w = fused_htm_winner_inactive_depression(w, fpost, fpre, float(lam_dep_wi), 0.0, float(hebb_max))
        conn.data[flat_pos] = w

    _set_gates(bridge, 1.0)
    period = int(sync_on) + int(sync_off)
    if encode_btsp:
        # gap#4<->gap#5 UNIFICATION ENCODE: BTSP plateau-gated ONE-SHOT storing instead of rate-window Hebbian. Disable
        # the Hebbian; enable the bistable BDSP apical (my keystone) + the BTSP block. During the co-fire, drive the
        # PLATEAU DIRECTLY on the pre-assigned assembly (the "encode-this" teaching signal, analogous to the Hebbian
        # path's DIRECT synchronous assembly drive) -> only the assembly cells have BOTH pre-eligibility (co-firing) AND
        # a plateau (IS_post) -> BTSP potentiates the WITHIN-assembly recurrent SPECIFICALLY (member->non-member post has
        # no plateau -> not stored). Specificity is BY CONSTRUCTION (where the plateau is), not from avalanche. Recall
        # (below) uses the two_comp coincidence plateau for completion, so enable_bdsp/enable_btsp are DISABLED after.
        cfg_b = bridge.core_config
        cfg_b.enable_hebbian_learning = False
        cfg_b.enable_bdsp = True; cfg_b.bdsp_apical_bistable = True; cfg_b.bdsp_learning_rate = 0.0
        cfg_b.coincidence_plateau_self_regen = float(plateau_self_regen)
        cfg_b.coincidence_plateau_v_hold = float(plateau_v_hold); cfg_b.apical_kir_g = float(apical_kir_g)
        cfg_b.enable_btsp = True; cfg_b.btsp_learning_rate = float(btsp_lr)
        cfg_b.btsp_elig_tau_ms = 1000.0; cfg_b.btsp_w_max = float(hebb_max)
        # gap#4<->gap#5 UNIFICATION FIX: the STRUCTURED one-shot storing rule -- add the heterosynaptic-COMPETITION arm
        # (Milstein-Magee bidirectional; sharpen the stored assembly the way Hebbian's lam_dep_wi does). 0 = uniform
        # (the characterized ~0.18 residual); >0 = competition-shaped (the head-to-head predicts a stronger completion).
        cfg_b.btsp_hetero_dep = float(encode_btsp_hetero)
        n_all_b = cfg_b.num_neurons
        bridge.cp_bdsp_apical_drive = cp.zeros(n_all_b, dtype=cp.float32)
        for m, assy in enumerate(assemblies):
            assy_arr = cp.asarray(assy, dtype=cp.int64)
            # STRUCTURED storing (encode_hetero>0, default 0 = uniform/byte-preserved): a per-cell plateau multiplier so
            # assembly cells latch/store at HETEROGENEOUS strengths -> a varied within-assembly distribution (like
            # Hebbian's heterogeneous co-firing), closing the uniform-BTSP-vs-structured-Hebbian magnitude residual.
            if float(encode_hetero) > 0.0:
                _hrng = np.random.default_rng(seed * 131 + m + 7)
                _hmul = (1.0 + float(encode_hetero) * (2.0 * _hrng.random(len(assy)) - 1.0)).clip(0.1, 3.0)
                plateau_vec = cp.asarray((float(encode_plateau_pA) * _hmul).astype(np.float32))
            else:
                plateau_vec = cp.full(len(assy), float(encode_plateau_pA), dtype=cp.float32)
            for ev in range(train_events):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_bdsp_apical_drive[:] = 0.0
                for _ in range(reset_steps):
                    bridge._run_one_simulation_step()
                for _st in range(drive_steps):
                    bridge.cp_external_input_current[:] = 0.0
                    bridge.cp_external_input_current[assy_arr] = float(encode_drive)      # co-fire the assembly (pre-elig)
                    bridge.cp_bdsp_apical_drive[:] = 0.0
                    bridge.cp_bdsp_apical_drive[assy_arr] = plateau_vec                   # plateau ON the assembly (IS_post; per-cell when hetero>0)
                    bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
        cfg_b.enable_bdsp = False; cfg_b.enable_btsp = False; bridge.cp_bdsp_apical_drive = None  # recall uses two_comp only
    else:
        for m, assy in enumerate(assemblies):
            assy_arr = cp.asarray(assy, dtype=cp.int64)
            # the KNOWN assembly is the competition member set (pre-assigned; local ca3 positions)
            member_mask = cp.zeros(len(ca3_idx), dtype=cp.float32)
            member_mask[cp.asarray([ca3_pos[int(g)] for g in assy], dtype=cp.int64)] = 1.0
            for ev in range(train_events):
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(reset_steps):
                    bridge._run_one_simulation_step()
                for _st in range(drive_steps):
                    bridge.cp_external_input_current[:] = 0.0
                    on = True if no_sync else ((_st % period) < int(sync_on))   # no_sync = drive every step (async control)
                    if on:
                        bridge.cp_external_input_current[assy_arr] = float(encode_drive)   # DIRECT synchronous assembly drive
                    bridge._run_one_simulation_step()
                if do_comp:
                    _apply_competition(member_mask)
            bridge.cp_external_input_current[:] = 0.0
    _set_gates(bridge, 0.0)

    # within-ensemble vs member->silent weight read (did the weights GROW to the completion scale?)
    def _wstats():
        d = np.asarray(to_host(conn.data))
        wi, ws = [], []
        assy_set = set(int(g) for a in assemblies for g in a)
        for k, (pl, ql) in enumerate(zip(pre_l_h, post_l_h)):
            pre_g = ca3_idx[pl]; post_g = ca3_idx[ql]
            if pre_g in assy_set and post_g in assy_set:
                wi.append(d[flat_h[k]])
            elif pre_g in assy_set and post_g not in assy_set:
                ws.append(d[flat_h[k]])
        return (float(np.mean(wi)) if wi else 0.0), (float(np.mean(ws)) if ws else 0.0)
    w_within, w_silent = _wstats()

    if homeostatic and homeo_target is not None:
        # KOPSICK homeostatic working point (research gate 2026-07-18): divisively normalize each assembly member's
        # TOTAL incoming within-assembly recurrent weight to a common set-point T, so every seed/neuron gets the SAME
        # effective recurrent gain -> the SAME bistable working point (fixes the seed-fragility: a too-high per-seed gain
        # self-sustains, too-low doesn't complete). Runner-side rescale of cp_connections; NO sim/ edit.
        member_pos = set(ca3_pos[int(g)] for a in assemblies for g in a)
        d = np.asarray(to_host(conn.data)).copy()
        wk = [k for k in range(len(flat_h)) if int(pre_l_h[k]) in member_pos and int(post_l_h[k]) in member_pos]
        by_post = {}
        for k in wk:
            by_post.setdefault(int(post_l_h[k]), []).append(k)
        for _post, ks in by_post.items():
            s = float(sum(d[int(flat_h[k])] for k in ks))
            if s > 1e-6:
                sc = float(homeo_target) / s
                for k in ks:
                    d[int(flat_h[k])] *= sc
        if wk:
            idxs = cp.asarray([int(flat_h[k]) for k in wk], dtype=cp.int64)
            vals = cp.asarray([float(d[int(flat_h[k])]) for k in wk], dtype=conn.data.dtype)
            conn.data[idxs] = vals
        w_within, w_silent = _wstats()

    if structural_sep:
        # STRUCTURAL PATTERN SEPARATION (the DG's job, Kandel Ch 54 / Marr): zero the recurrent synapses FROM
        # non-members INTO assembly members, so an assembly cell receives recurrent drive ONLY from its within-assembly
        # partners. Then a permuted cue (random non-member cells) cannot leak-activate the assembly even with STRONG
        # within-weights -> strong AND specific completion (resolving the strength-vs-specificity tension: the leaked-
        # activation path is removed structurally, so the within-weights can be strong). Realized runner-side with the
        # pre-assigned assembly mask (like `_apply_competition`); the emergent DG-selected version is the follow-on.
        # structural_sep=1: zero non-member->member (the permuted cue can't reach the assembly). structural_sep=2: FULL
        # bidirectional isolation -- ALSO zero member->non-member, so a STRONG completed assembly cannot SPREAD to
        # non-members (the network recurrent->coincidence->latch loop that self-sustains when the read is strong). Full
        # isolation = the complete DG pattern-separation outcome (the assembly is a closed set), the emergent version.
        assy_pos_set2 = set(ca3_pos[int(g)] for a in assemblies for g in a)   # local ca3 positions of members
        _full = int(structural_sep) >= 2
        zk = [k for k in range(len(flat_h))
              if (int(post_l_h[k]) in assy_pos_set2) != (int(pre_l_h[k]) in assy_pos_set2)] if _full else \
             [k for k in range(len(flat_h))
              if int(post_l_h[k]) in assy_pos_set2 and int(pre_l_h[k]) not in assy_pos_set2]  # non-member->member (+member->non-member if full)
        if zk:
            idxs = cp.asarray([int(flat_h[k]) for k in zk], dtype=cp.int64)
            conn.data[idxs] = cp.zeros(len(zk), dtype=conn.data.dtype)
        w_within, w_silent = _wstats()

    if recall_k_thresh is not None:
        # DECOUPLE encoding vs recall dAP threshold: a LOW k_thresh during encoding lets the plateau fire freely so the
        # rate-window LTP grows STRONG within-ensemble weights; a HIGH k_thresh at RECALL means only the strong LEARNED
        # coincident drive (correct cue) crosses the plateau, not scattered generic/avalanche drive (permuted cue) ->
        # specificity WITHOUT starving the encoding (the two were coupled through one cfg.coincidence_k_threshold).
        bridge.core_config.coincidence_k_threshold = float(recall_k_thresh)

    if interassembly_isolate and len(assemblies) > 1 and len(flat_h) > 0:
        # BETWEEN-ASSEMBLY recurrent isolation (the emergent equivalent of swr_disjoint / the DG's between-memory
        # pattern-separation, Kandel Ch 54): zero ca3->ca3 recurrent edges whose pre and post are in DIFFERENT stored
        # assemblies, so a partial cue of assembly A completes A but does NOT cross-complete assembly B through the shared
        # dense recurrent substrate. structural_sep isolates the assembly-UNION from non-members; this isolates the
        # assemblies from EACH OTHER (required for the SWR readout to discriminate two co-stored assemblies). Default
        # False => byte-identical. Realizes R1's emergent SEPARATION in recurrent space; NO sim/ edit.
        local_to_asm = {}
        for _m, _a in enumerate(assemblies):
            for _g in _a:
                local_to_asm[ca3_pos[int(_g)]] = _m
        zk = [k for k in range(len(flat_h))
              if int(pre_l_h[k]) in local_to_asm and int(post_l_h[k]) in local_to_asm
              and local_to_asm[int(pre_l_h[k])] != local_to_asm[int(post_l_h[k])]]
        if zk:
            idxs = cp.asarray([int(flat_h[k]) for k in zk], dtype=cp.int64)
            conn.data[idxs] = cp.zeros(len(zk), dtype=conn.data.dtype)

    non_stored0 = np.array([g for g in ca3_idx if g not in set(int(x) for a in assemblies for x in a)], dtype=np.int64)

    if selective_inhib:
        # ASSEMBLY-SELECTIVE INHIBITION -- "spare your own engram" (Kim-Kim 2025 PMC12244581, research gate
        # 2026-07-18). The GLOBAL shared basket (ca3_pv_basket) inhibits every cell equally -> a permuted cue's
        # non-member cells avalanche through the generic recurrents and spuriously complete the stored assembly. The
        # paper's fix: an assembly's co-active interneurons SPARE the assembly's excitatory cells (weak I->E onto them)
        # but INHIBIT non-members -> a random cue's cells are suppressed before they can avalanche, while the correct
        # cue's (spared) assembly cells build up + complete. Realized here as the LEARNED OUTCOME (like `_apply_competition`
        # uses the pre-assigned assembly mask): DEPRESS the ca3_pv_basket->ca3 (I->E) synapses ONTO assembly members
        # (spare them) toward `sel_inhib_spare`; leave those onto non-members intact (inhibit the avalanche). The
        # emergent DG-selected + E->I-plasticity-tuned version is the follow-on. NO sim/ edit.
        bask_idx = list(rm.indices("ca3_pv_basket"))
        bask_set = set(int(g) for g in bask_idx)
        assy_pos_set = set(int(g) for a in assemblies for g in a)   # global CA3 indices that are assembly members
        conn2 = bridge.cp_connections
        nnz = int(conn2.nnz)
        indptr = np.asarray(to_host(conn2.indptr)); indices = np.asarray(to_host(conn2.indices))
        pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
        d = np.asarray(to_host(conn2.data)).copy()
        spare_k = [k for k in range(nnz)
                   if int(pre_of[k]) in bask_set and int(indices[k]) in assy_pos_set]  # basket -> assembly-member (I->E)
        for k in spare_k:
            d[k] = float(sel_inhib_spare)                          # spare the assembly's own cells from inhibition
        if spare_k:
            idxs = cp.asarray(spare_k, dtype=cp.int64)
            conn2.data[idxs] = cp.asarray([float(d[k]) for k in spare_k], dtype=conn2.data.dtype)

    if per_assembly_sel_inhib and len(assemblies) > 1:
        # PER-ASSEMBLY selective inhibition (Kim-Kim 2025 "spare your own engram" + Kopsick 2024 learned E->I), the
        # BETWEEN-MEMORY separation: the shared ca3_pv_basket is PARTITIONED into one sub-pool per stored assembly, and
        #   (E->I) sub-pool m is driven ONLY by assembly-m's excitatory cells (co-active interneurons), and
        #   (I->E) sub-pool m SPARES assembly-m's members (weak, sel_inhib_spare) but INHIBITS the OTHER assemblies'
        #          members (strong, per_assembly_inhib_w) and leaves non-members at the default general inhibition.
        # => a partial cue of A ignites A's cells -> A's basket sub-pool fires -> it SUPPRESSES B's cells before they can
        # avalanche through the shared recurrents, while sparing A -> A completes, B stays silent -> the two co-stored
        # assemblies are INDEPENDENTLY ADDRESSABLE (the SWR readout can discriminate). Realized as the learned outcome on
        # the emergent assemblies (like selective_inhib/structural_sep), fully vectorized. Default False => byte-identical.
        _bask = np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)
        _nb = len(_bask); _na = len(assemblies)
        conn3 = bridge.cp_connections; nnz3 = int(conn3.nnz)
        _ip = np.asarray(to_host(conn3.indptr)); _ind = np.asarray(to_host(conn3.indices))
        _pre = (np.searchsorted(_ip, np.arange(nnz3), side="right") - 1).astype(np.int64)
        _post = _ind[:nnz3].astype(np.int64)
        _d = np.asarray(to_host(conn3.data)).copy()
        N = bridge.core_config.num_neurons
        ca3_asm_arr = np.full(N, -1, dtype=np.int64)          # ca3 global -> assembly m (members only)
        for _m, _a in enumerate(assemblies):
            ca3_asm_arr[np.asarray(_a, dtype=np.int64)] = _m
        bask_asm_arr = np.full(N, -1, dtype=np.int64)          # basket global -> sub-pool m (round-robin partition)
        bask_asm_arr[_bask] = np.arange(_nb, dtype=np.int64) % _na
        pre_asm = ca3_asm_arr[_pre]; post_asm = ca3_asm_arr[_post]
        pre_bmask = bask_asm_arr[_pre]; post_bmask = bask_asm_arr[_post]
        # E->I: member m -> basket sub-pool != m  => zero (sub-pool m is driven ONLY by assembly-m)
        _ei = (pre_asm >= 0) & (post_bmask >= 0) & (pre_asm != post_bmask)
        _d[_ei] = 0.0
        # WITHIN-assembly E->I POTENTIATION (Kim-Kim 2025 heterosynaptic E->I potentiation -- the CORE of the mechanism,
        # was missing): SET member m -> own basket sub-pool m to a strong weight so a cue of m drives sub-pool m HARD ->
        # it actually suppresses the OTHER assemblies (inhibit-other). The un-potentiated default left the sub-pools too
        # weak, which both let the small assembly avalanche AND globally over-suppressed own-completion. Default None ->
        # unchanged (byte-identical); only strengthens EXISTING member->own-sub-pool edges (biological potentiation).
        if per_assembly_ei_w is not None:
            _ei_within = (pre_asm >= 0) & (post_bmask >= 0) & (pre_asm == post_bmask)
            print(f"[pa_ei_w] within-assembly E->I edges matched={int(_ei_within.sum())} "
                  f"(cross-EI zeroed={int(_ei.sum())}); setting them to {float(per_assembly_ei_w)}", flush=True)
            _d[_ei_within] = float(per_assembly_ei_w)
        # I->E: basket sub-pool m -> member: spare own (== m), inhibit other (!= m); magnitude only (pre is inhibitory)
        _ie_spare = (pre_bmask >= 0) & (post_asm >= 0) & (pre_bmask == post_asm)
        _ie_inhib = (pre_bmask >= 0) & (post_asm >= 0) & (pre_bmask != post_asm)
        _d[_ie_spare] = float(sel_inhib_spare)
        _d[_ie_inhib] = float(per_assembly_inhib_w)
        conn3.data[:] = cp.asarray(_d, dtype=conn3.data.dtype)

    if bistable:
        # GENUINE CUE-GATED COMPLETION TEST: hard-SILENCE the network (clear v/u/firing/conductances to rest), then
        # drive a condition, and read the HELD (non-cued stored) members' firing. Real pattern completion requires:
        #   NO-CUE       -> held SILENT (the attractor is not self-sustaining/always-on)
        #   CORRECT cue  -> held FIRES (partial cue A ignites the full pattern A)
        #   PERMUTED cue -> held SILENT (specific: a random cue does NOT ignite A)
        # (The prior "completion" failed because measure_region_response never silenced the self-sustaining attractor.)
        from sim.backend import from_host
        # FREEZE plasticity for the recall/calibration phase: the learned attractor is FIXED here; leaving hebbian on
        # lets the recall's own co-firing keep potentiating the ca3->ca3 recurrents (strengthening the attractor DURING
        # the rest/calibration stepping -> the intrinsic-homeostatic suppression is fought by ongoing LTP). Testing a
        # fixed autoassociator requires no learning at recall.
        bridge.core_config.enable_hebbian_learning = False
        n_all = bridge.core_config.num_neurons
        ca3_arr_host = np.asarray(ca3_idx, dtype=int)
        # PER-NEURON INTRINSIC-EXCITABILITY OFFSET (default all-zero == byte-identical). When rate_homeo, the calibration
        # phase below fills the CA3 entries with a suppressive tonic bias so the low (rest) state is a genuine bistable
        # rest across seeds.
        bias_full = np.zeros(n_all, dtype=np.float64)
        bias_dev = from_host(bias_full)

        def _hard_silence(settle=30):
            if getattr(bridge, "cp_izh_c_reset", None) is not None:
                bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
            else:
                bridge.cp_membrane_potential_v[:] = -65.0
            bridge.cp_recovery_variable_u[:] = 0.0
            if getattr(bridge, "cp_firing_states", None) is not None:
                bridge.cp_firing_states[:] = False
            for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_e", "cp_conductance_g_i",
                       "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
                       # CRITICAL for the BISTABLE test: reset the dendritic-plateau conductance too, or a plateau
                       # latched during encoding PERSISTS through "silence" (the apical read-out caught this: nocue
                       # showed plateaus ON). A valid no-cue/permuted anti-cheat must start from a genuine silent down state.
                       "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise"):
                _arr = getattr(bridge, _a, None)
                if _arr is not None:
                    _arr[:] = 0.0
            if getattr(bridge, "cp_v_apical", None) is not None:   # reset the apical compartment to rest (the bistable up state must be cleared)
                bridge.cp_v_apical[:] = cp.float32(getattr(bridge.core_config, "apical_E_rest", -65.0))
            bridge.cp_external_input_current[:] = bias_dev   # the calibrated intrinsic bias (0 by default -> silence)
            for _ in range(settle):     # confirm it stays silent (a bistable attractor will; a self-sustaining one re-ignites)
                bridge._run_one_simulation_step()

        # ---- gap#5 DENDRITE-TARGETING apical inhibition (Muller-Remy 2014 O-LM/SOM), default-off byte-identical ----
        # The SOMATIC selective-inhibition family (Kim-Kim PV-basket, --per-assembly-inhib/--pa-ei-w) VERIFIED-does-NOT
        # gate the two-assembly cross-completion, because R4 completion is a DENDRITIC APICAL PLATEAU (cp_v_apical >
        # plateau_v_hold, self-regenerating) that somatic g_i cannot shunt (wrong compartment). Here a per-assembly
        # O-LM/SOM pool, driven by whichever assembly is CURRENTLY ACTIVE (emergent -- read from soma firing, NOT the
        # host cue identity), shunts the OTHER assemblies' APICAL compartment toward E_inh each recall step. Because the
        # plateau self-regen sigmoid reads cp_v_apical (bridge.py:6647,6654-6656), pulling v_apical below v_hold cascades
        # to killing g_coincidence -> the other assembly's plateau cannot latch/hold; the active assembly's OWN apical is
        # SPARED (spare-your-own-engram, on the CORRECT compartment the somatic basket could not reach). Activity-gated:
        # no-cue / permuted-cue have no active winner -> zero inhibition (anti-cheats untouched). NO sim/ edit -- a
        # per-recall-step runner term on cp_v_apical (the substrate updates the plateau; this is the O-LM shunt onto it).
        _pa_apical_on = bool(per_assembly_apical_inhib) and len(assemblies) > 1 and (bridge.cp_v_apical is not None)
        _pa_mem_dev = [cp.asarray(np.asarray(a, dtype=np.int64)) for a in assemblies] if _pa_apical_on else []
        _pa_E_inh = cp.float32(-75.0)                          # GABA_A apical reversal (dendrite-targeting SOM/O-LM)
        _pa_w = cp.float32(float(per_assembly_apical_w))        # per-step shunt fraction toward E_inh (0..1)
        _pa_gate = float(per_assembly_apical_gate)              # accumulated soma-firing count that marks a genuine winner
        _pa_spare = bool(per_assembly_apical_spare_own)         # False = GLOBAL-suppression anti-cheat control (own collapses)
        _pa_stats = {"steps_fired": 0, "cells_shunted": 0, "printed": False}

        def _apply_apical_inhib(acc):
            """acc = list[nA] accumulated per-assembly soma-firing counts, mutated in place. Determine the currently-
            active (winner) assembly from EMERGENT soma firing; shunt every OTHER assembly's cp_v_apical toward E_inh
            (dendrite-targeting O-LM inhibition), sparing the winner. Gated: no inhibition until a winner is genuinely
            active (>=_pa_gate accumulated spikes) so the initial transient + no-cue/permuted apply nothing."""
            if not _pa_apical_on:
                return
            for _m in range(len(_pa_mem_dev)):
                acc[_m] += float(bridge.cp_firing_states[_pa_mem_dev[_m]].astype(cp.float32).sum())
            _win = int(np.argmax(acc))
            if acc[_win] < _pa_gate:
                return
            for _m in range(len(_pa_mem_dev)):
                if _pa_spare and _m == _win:
                    continue
                _idx = _pa_mem_dev[_m]
                bridge.cp_v_apical[_idx] = bridge.cp_v_apical[_idx] + _pa_w * (_pa_E_inh - bridge.cp_v_apical[_idx])
                _pa_stats["cells_shunted"] += int(_idx.size)
            _pa_stats["steps_fired"] += 1

        def _measure(cue_idx):
            _hard_silence()
            cur = bias_full.copy()                          # start from the intrinsic bias (0 by default)
            if cue_idx is not None and len(cue_idx):
                cur[np.asarray(cue_idx, dtype=int)] += float(recall_drive)
            dev = from_host(cur.astype(np.float64)); spk = np.zeros(len(ca3_idx))
            _pa_acc = [0.0] * len(assemblies)                  # fresh winner accumulator per (silenced) measure call
            for _ in range(recall_steps):
                bridge.cp_external_input_current[:] = dev; bridge._run_one_simulation_step()
                _apply_apical_inhib(_pa_acc)                    # dendrite-targeting O-LM shunt (default-off byte-identical)
                if read_apical:
                    # DECOUPLED READ-OUT: read the held APICAL PLATEAU state (cp_v_apical > plateau_v_hold = the
                    # intrinsically-bistable UP state = the held memory) instead of soma firing, so a WEAK apical->soma
                    # coupling lets the plateau HOLD (completion) without the soma driving the recurrent loop (self-sustain).
                    va = np.asarray(to_host(bridge.cp_v_apical))[ca3_arr_host]
                    spk += (va > float(plateau_v_hold)).astype(float)
                else:
                    spk += np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(float)
            bridge.cp_external_input_current[:] = 0.0
            return spk / recall_steps

        if rate_homeo:
            # PER-NEURON RATE HOMEOSTATIC (Turrigiano intrinsic plasticity, the ranked fix for the Wang seed-fragility,
            # 2026-07-18): the Wang bistable WORKING POINT is seed-dependent (heterogeneity + connectivity + E/I) and the
            # weight-sum T-homeostatic could NOT equalize it (per-seed-T diagnostic: 43 self-sustains / 44 non-specific at
            # every T). Each CA3 cell's INTRINSIC excitability (a tonic bias current) auto-calibrates so its NO-CUE rest
            # firing drops to a low target -> the self-sustaining seeds' over-firers are SUPPRESSED into a genuine low
            # rest state, equalizing the working point across seeds. Suppressive-only (bias <= 0): damps the always-on
            # high state, never excites silent cells (which would break specificity).
            #   OUTER-LOOP form: each iteration MEASURES the no-cue rest firing FROM A FRESH HARD-SILENCE over the full
            #   recall window (the exact quantity the GO gate reads), then increases the bias on whatever fired. This
            #   targets the cold-start rest state directly (an online per-step EMA finds only the metastable edge-bias
            #   that holds during its own continuous trajectory but re-ignites on a fresh 150-step recall).
            for _it in range(int(rate_homeo_steps)):
                r0 = _measure(None)                                   # per-CA3-cell no-cue rest rate (fresh cold start)
                over = np.maximum(r0 - float(rate_homeo_target), 0.0)
                if float(np.max(over)) < 1e-3:
                    break                                            # rest is a genuine low state -> calibrated
                bias_full[ca3_arr_host] = np.clip(bias_full[ca3_arr_host] - float(rate_homeo_adapt) * over,
                                                  -float(rate_homeo_cap), 0.0)
                bias_dev = from_host(bias_full)

        nocue_l, cue_l, perm_l, silence_l = [], [], [], []
        for m, assy in enumerate(assemblies):
            a = assy.copy(); np.random.default_rng(seed + m).shuffle(a)
            half = max(2, len(a) // 2); cue, held = a[:half], a[half:]
            hp = [ca3_pos[int(g)] for g in held]
            # NO-CUE: also read the whole assembly's rest firing (self-sustain check)
            r0 = _measure(None); nocue_l.append(float(np.mean(r0[hp])))
            silence_l.append(float(np.mean(r0[[ca3_pos[int(g)] for g in a]])))
            r1 = _measure(cue); cue_l.append(float(np.mean(r1[hp])))
            perm = np.random.default_rng(seed * 7 + m + 999).choice(non_stored0, len(cue), replace=False)
            r2 = _measure(perm); perm_l.append(float(np.mean(r2[hp])))
        held_cue = float(np.mean(cue_l)); held_nocue = float(np.mean(nocue_l)); held_perm = float(np.mean(perm_l))
        rest = float(np.mean(silence_l))
        if _pa_apical_on and not _pa_stats["printed"]:
            # VERIFY-THE-EDIT-TOOK-EFFECT (silent-failure discipline): confirm the mechanism actually fired + touched a
            # non-trivial number of cells. cells_shunted==0 => silent no-op (the #1 failure mode) -> alarm.
            _pa_stats["printed"] = True
            print(f"[pa_apical] dendrite-targeting apical inhib FIRED: shunt-steps={_pa_stats['steps_fired']} "
                  f"cells_shunted={_pa_stats['cells_shunted']} (w={float(_pa_w):.2f} gate={_pa_gate} "
                  f"spare_own={_pa_spare}); assembly_sizes={[len(a) for a in assemblies]}", flush=True)
        # GENUINE bistable completion (relative to the Wang low-rate background, NOT a dead net): the correct cue must
        # IGNITE the high state (>=0.20) AND be >=3x BOTH the no-cue low state AND the permuted -- i.e. only the correct
        # partial cue reaches the high attractor state; no-cue/permuted stay in the low state. The low background is
        # capped (<=0.10) so it is a genuine LOW state, not a near-self-sustaining one. Above-baseline completion signal
        # (cue-rest) vs permuted residual (perm-rest) reported for transparency.
        sig = held_cue - rest; perm_sig = held_perm - rest
        go = (held_cue >= 0.20 and held_cue >= 3.0 * (held_nocue + 1e-6) and held_cue >= 3.0 * (held_perm + 1e-6)
              and held_nocue <= 0.10)
        out = {"seed": seed, "w_within": w_within, "held_cue": held_cue, "held_nocue": held_nocue,
               "held_perm": held_perm, "rest_firing": rest, "sig": float(sig), "perm_sig": float(perm_sig),
               "mean_bias": float(np.mean(bias_full[ca3_arr_host])), "go": bool(go)}
        if read_ca1:
            # SWR GENERATIVE-REPLAY Rung 1 on the VALIDATED completion: does a PARTIAL cue's dendritic completion drive
            # the SAME ca1 (Schaffer) pattern as the FULL assembly (correct), and a DIFFERENT one for another assembly
            # (specific)? schaffer_boost potentiates ca3->ca1 so the completed assembly drives ca1 above threshold.
            ca1_idx = np.asarray(list(rm.indices("ca1")), dtype=int)
            if swr_learn_schaffer:
                # LEARNED SCHAFFER (2026-07-19): the biology-correct SWR readout. The fixed-random DENSE Schaffer drives
                # every ca1 near-identically (no specificity). Instead, ASSOCIATIVELY POTENTIATE ca3(assembly_m)->
                # ca1(target_m): each assembly gets a distinct sparse ca1 TARGET pattern, and only the assembly->target
                # synapses are potentiated (Schaffer collateral LTP during encoding); all other ca3->ca1 held LOW. So
                # recall of an assembly drives ITS specific ca1 pattern -> ca1_match(same) high, ca1_cross(other) low.
                cc = bridge.cp_connections; nnzc = int(cc.nnz)
                ip = np.asarray(to_host(cc.indptr)); ind = np.asarray(to_host(cc.indices))
                prec = np.searchsorted(ip, np.arange(nnzc), side="right") - 1
                try:
                    ca3_inh = set(int(g) for g in rm.inhibitory_indices("ca3"))
                except Exception:
                    ca3_inh = set()
                c1_list = list(ca1_idx); n_ca1_l = len(c1_list)
                n_tgt = max(2, int(swr_target_frac * n_ca1_l))
                # per-assembly: a distinct sparse ca1 target (deterministic random draw)
                asm_of = {}                                       # ca3 global -> assembly m
                tgt_of = {}                                       # assembly m -> set(ca1 global)
                if swr_disjoint_targets and n_tgt * len(assemblies) <= n_ca1_l:
                    # DISJOINT ca1 targets: draw ALL assemblies' targets from ONE pool WITHOUT replacement so no two
                    # assemblies share a ca1 target cell -> removes the CA1-readout overlap (the seed-fragility source:
                    # independent random 18-of-120 draws overlap ~3 cells by chance, inflating ca1_cross). The
                    # discrimination then reflects the CA3 completion, not the target-draw luck. Each target still sparse.
                    _tp = np.random.default_rng(seed * 71 + 13).choice(n_ca1_l, n_tgt * len(assemblies), replace=False)
                    for m in range(len(assemblies)):
                        tgt_of[m] = set(int(c1_list[i]) for i in _tp[m * n_tgt:(m + 1) * n_tgt])
                    for m, assy in enumerate(assemblies):
                        for g in assy:
                            asm_of[int(g)] = m
                else:
                    for m, assy in enumerate(assemblies):
                        for g in assy:
                            asm_of[int(g)] = m
                        _tr = np.random.default_rng(seed * 71 + m + 13)
                        tgt_of[m] = set(int(c1_list[i]) for i in _tr.choice(n_ca1_l, n_tgt, replace=False))
                data_h = np.asarray(to_host(cc.data)).copy()
                _npot = 0
                for k in range(nnzc):
                    pre = int(prec[k]); post = int(ind[k])
                    if pre in ca3_inh or post not in set(c1_list):
                        continue
                    if pre in asm_of and post in tgt_of[asm_of[pre]]:
                        data_h[k] = float(swr_schaffer_hi); _npot += 1     # associative LTP: assembly -> its target
                    elif post in set(c1_list):
                        data_h[k] = float(swr_schaffer_lo)                 # non-associated Schaffer held LOW
                cc.data[:] = cp.asarray(data_h, dtype=cc.data.dtype)
                if os.environ.get("SWR_DEBUG"):
                    print(f"    [SWR debug] LEARNED Schaffer: {_npot} assembly->target synapses potentiated to "
                          f"{swr_schaffer_hi}, others->{swr_schaffer_lo}; n_target/assembly={n_tgt}", flush=True)
            elif schaffer_boost != 1.0:
                cc = bridge.cp_connections; nnzc = int(cc.nnz)
                ip = np.asarray(to_host(cc.indptr)); ind = np.asarray(to_host(cc.indices))
                prec = np.searchsorted(ip, np.arange(nnzc), side="right") - 1
                # EXCITATORY ca3->ca1 ONLY: the synapse sign is the PRE cell's trait, so boosting ALL ca3->ca1 amplifies
                # the INHIBITORY ca3 cells too (g_i > g_e -> ca1 stays silent, root-caused 2026-07-18). Exclude inhibitory ca3.
                try:
                    ca3_inh = set(int(g) for g in rm.inhibitory_indices("ca3"))
                except Exception:
                    ca3_inh = set()
                c3s = set(int(g) for g in ca3_idx) - ca3_inh; c1s = set(int(g) for g in ca1_idx)
                sk = [k for k in range(nnzc) if int(prec[k]) in c3s and int(ind[k]) in c1s]
                if os.environ.get("SWR_DEBUG"):
                    _wm = float(cp.asnumpy(cc.data[cp.asarray(sk, dtype=cp.int64)]).mean()) if sk else 0.0
                    print(f"    [SWR debug] Schaffer ca3->ca1 synapses found={len(sk)} mean_w(pre-boost)={_wm:.3f} "
                          f"n_ca1={len(c1s)}", flush=True)
                if sk:
                    ix = cp.asarray(sk, dtype=cp.int64); cc.data[ix] = cc.data[ix] * cp.float32(schaffer_boost)

            def _measure_ca1(cue_idx, ripple_pA=None, g_on=8, g_off=4):
                if ripple_pA is None:
                    ripple_pA = float(swr_ripple_pA)
                # SWR two-phase read (honest ripple mechanism): PHASE 1 -- hold the cue so the bistable dendrite completes
                # (settle), then READ which cells LATCHED (cp_v_apical > v_hold = the completed assembly, a correct cue
                # latches the full assembly, a wrong cue latches ~none). PHASE 2 -- a SHARP-WAVE RIPPLE drives ONLY the
                # LATCHED cells DIRECTLY in strong gamma volleys (no bistable suppression) -> they fire SYNCHRONOUSLY -> a
                # coincident Schaffer volley -> ca1 fires. The ca1 pattern reflects the COMPLETED assembly; specificity is
                # inherited from WHICH cells latched.
                _hard_silence(); base = bias_full.copy()
                if cue_idx is not None and len(cue_idx):
                    base[np.asarray(cue_idx, dtype=int)] += float(recall_drive)
                dev_base = from_host(base.astype(np.float64))
                fire_acc = np.zeros(len(ca3_arr_host))
                _pa_acc = [0.0] * len(assemblies)                    # fresh winner accumulator per SWR phase-1 read
                for _ in range(recall_steps):                        # PHASE 1: establish completion, accumulate CA3 firing
                    bridge.cp_external_input_current[:] = dev_base; bridge._run_one_simulation_step()
                    _apply_apical_inhib(_pa_acc)                      # dendrite-targeting O-LM shunt -> the OTHER assembly does not latch -> not in `latched` -> ca1_cross drops
                    fire_acc += np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(float)
                # the COMPLETED assembly = cells that FIRED (soma) during phase 1 (the network completion is soma-driven,
                # NOT a sustained apical latch -- apical>v_hold identifies only ~3%; diagnostic 2026-07-18).
                _far = fire_acc / recall_steps
                latched = ca3_arr_host[_far > 0.08]
                if os.environ.get("SWR_DEBUG"):
                    _ls = set(int(x) for x in latched)
                    _brk = [len(_ls & set(int(g) for g in _a)) for _a in assemblies]   # latched-in-each-assembly
                    _nonasm = len(_ls) - sum(_brk)
                    print(f"    [SWR debug] latched-breakdown per-assembly={_brk} non-assembly={_nonasm} (total {len(_ls)})", flush=True)
                rip = bias_full.copy()
                if len(latched):
                    rip[latched] += float(ripple_pA)               # PHASE 2: strong burst of the LATCHED completed cells
                dev_rip = from_host(rip.astype(np.float64)); dev_off = from_host(bias_full.astype(np.float64))
                c1 = np.zeros(len(ca1_idx)); period = g_on + g_off; n_rip = 60
                _dbg = os.environ.get("SWR_DEBUG"); _ca3fire = 0.0; _lat_arr = np.asarray(latched, dtype=int)
                _peak_ge = 0.0; _peak_ca3sync = 0.0
                # E%-max CA1 winner-set read (de Almeida-Idiart-Lisman 2009; Valero 2017 cell-specific-drive selectivity):
                # accumulate per-CA1-cell PEAK g_e (the Schaffer drive) so we can fire ONLY the top-k most-driven CA1 cells
                # (the winner-set), converting the all-fire g_e-180+ collapse into a discriminating sparse pattern.
                # Default None => byte-identical (no extra reads, returns c1 firing as before).
                _topk_ge = np.zeros(len(ca1_idx)) if swr_ca1_topk is not None else None
                # SWR readout fix (2026-07-19): STP depression on the Schaffer (ca3->ca1) crushes g_e under the ripple's
                # sustained firing -> disable STP during PHASE 2 ONLY so the completed assembly's volley reaches ca1
                # (phase 1 keeps STP so the completion is normal). Gated (env), restored after.
                _p2_nostp = bool(os.environ.get("SWR_PHASE2_NOSTP"))
                _stp_was = bridge.core_config.enable_short_term_plasticity
                if _p2_nostp:
                    bridge.core_config.enable_short_term_plasticity = False
                for t in range(n_rip):
                    bridge.cp_external_input_current[:] = dev_rip if (t % period) < g_on else dev_off
                    bridge._run_one_simulation_step()
                    c1 += np.asarray(to_host(bridge.cp_firing_states))[ca1_idx].astype(float)
                    if _topk_ge is not None:
                        _ge_v = getattr(bridge, "cp_conductance_g_e", None)
                        if _ge_v is not None:
                            _topk_ge = np.maximum(_topk_ge, np.asarray(to_host(_ge_v))[ca1_idx].astype(float))
                    if _dbg and len(_lat_arr):
                        _ca3_now = float(np.asarray(to_host(bridge.cp_firing_states))[_lat_arr].mean())
                        _ca3fire += _ca3_now; _peak_ca3sync = max(_peak_ca3sync, _ca3_now)  # max SIMULTANEOUS latched-fire fraction
                        _ge_now = getattr(bridge, "cp_conductance_g_e", None)
                        if _ge_now is not None:
                            _peak_ge = max(_peak_ge, float(np.asarray(to_host(_ge_now))[ca1_idx].max()))  # peak ca1 g_e DURING ripple
                bridge.cp_external_input_current[:] = 0.0
                if _p2_nostp:
                    bridge.core_config.enable_short_term_plasticity = _stp_was   # restore
                if _dbg:
                    _ge = getattr(bridge, "cp_conductance_g_e", None)
                    _ca1_ge = float(np.asarray(to_host(_ge))[ca1_idx].mean()) if _ge is not None else -1.0
                    _ca1_v = float(np.asarray(to_host(bridge.cp_membrane_potential_v))[ca1_idx].mean())
                    print(f"    [SWR debug] latched={len(latched)} | phase2: ca3-latched-fire-rate={_ca3fire/n_rip:.3f} "
                          f"PEAK-ca3-sync={_peak_ca3sync:.3f} PEAK-ca1_g_e={_peak_ge:.3f} end-ca1_g_e={_ca1_ge:.3f} "
                          f"ca1_v={_ca1_v:.1f} ca1_fire_sum={float((c1/n_rip).sum()):.3f}", flush=True)
                if _topk_ge is not None:
                    # E%-max winner-set: keep the top-k CA1 cells by peak Schaffer g_e (fire only the cells within E% of
                    # the max-driven cell), the pattern = their graded g_e; the rest zeroed. This is the discriminating
                    # read (a specific CA3 assembly -> its strongest-driven CA1 target cells win; the weak leak is cut).
                    _k = max(1, int(float(swr_ca1_topk) * len(ca1_idx)))
                    _thr = float(np.sort(_topk_ge)[-_k])
                    return _topk_ge * (_topk_ge >= _thr).astype(float)
                return c1 / n_rip

            def _cos(x, y):
                nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
                return float(x @ y / (nx * ny)) if nx > 1e-9 and ny > 1e-9 else 0.0
            fulls, parts = [], []
            for m, assy in enumerate(assemblies):
                a = assy.copy(); np.random.default_rng(seed + m).shuffle(a); cue = a[:max(2, len(a) // 2)]
                fulls.append(_measure_ca1(assy)); parts.append(_measure_ca1(cue))
            nA = len(assemblies)
            matches = [_cos(parts[m], fulls[m]) for m in range(nA)]
            crosses = [_cos(parts[m], fulls[(m + 1) % nA]) for m in range(nA)] if nA > 1 else [0.0]
            out.update(ca1_match=float(np.mean(matches)), ca1_cross=float(np.mean(crosses)),
                       ca1_fire=float(np.mean([np.mean(f) for f in fulls])))
        return out

    # RECALL: partial cue (50% of each assembly) DIRECT on CA3 -> does the held-out 50% fire?
    non_stored = np.array([g for g in ca3_idx if g not in set(int(x) for a in assemblies for x in a)], dtype=np.int64)
    held_list, ns_list = [], []
    held_abs_l, cue_abs_l, ns_abs_l = [], [], []
    for m, assy in enumerate(assemblies):
        a = assy.copy(); np.random.default_rng(seed + m).shuffle(a)
        half = max(2, len(a) // 2); cue, held = a[:half], a[half:]
        if permute_recall:
            # ANTI-CHEAT: cue a RANDOM NON-assembly set (same size) -> the stored assembly's held members must NOT
            # complete (rules out "any cue completes anything" / a drive artifact independent of the learned attractor).
            cue = np.asarray(np.random.default_rng(seed * 7 + m + 999).choice(non_stored, len(cue), replace=False), dtype=np.int64)
        resp = measure_region_response(bridge, "ca3", cue.tolist(), drive_pA=recall_drive,
                                       drive_region="ca3", n_steps=recall_steps)
        held_abs = float(np.mean(resp[[ca3_pos[int(g)] for g in held]]))
        cue_abs = float(np.mean(resp[[ca3_pos[int(g)] for g in cue]]))
        ns_abs = float(np.mean(resp[[ca3_pos[int(g)] for g in non_stored[:40]]]))
        held_abs_l.append(held_abs); cue_abs_l.append(cue_abs); ns_abs_l.append(ns_abs)
        cue_act = cue_abs or 1.0
        held_list.append(held_abs / (cue_act + 1e-9))
        ns_list.append(ns_abs / (cue_act + 1e-9))
    h_comp, n_comp = float(np.mean(held_list)), float(np.mean(ns_list))
    go = h_comp >= 0.30 and h_comp >= 2.0 * (n_comp + 1e-9)
    return {"seed": seed, "w_within": w_within, "w_silent": w_silent,
            "w_ratio": (w_within / (w_silent + 1e-9)), "h_comp": h_comp, "n_comp": n_comp, "go": bool(go),
            "held_abs": float(np.mean(held_abs_l)), "cue_abs": float(np.mean(cue_abs_l)), "ns_abs": float(np.mean(ns_abs_l))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-ca3", type=int, default=1000)
    ap.add_argument("--lam-dep-wi", type=float, default=0.5)
    ap.add_argument("--hebb-max", type=float, default=2000.0)
    ap.add_argument("--encode-drive", type=float, default=700.0)
    ap.add_argument("--no-sync", action="store_true", help="ASYNC control: drive every step (no gamma pulse)")
    a = ap.parse_args()
    t0 = time.time()
    print(f"[R-iii synchrony-isolation] n_ca3={a.n_ca3} lam={a.lam_dep_wi} hebb_max={a.hebb_max} "
          f"encode_drive={a.encode_drive} no_sync={a.no_sync}", flush=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        r = run(s, n_ca3=a.n_ca3, lam_dep_wi=a.lam_dep_wi, hebb_max=a.hebb_max,
                encode_drive=a.encode_drive, no_sync=a.no_sync)
        print(f"  [seed {s}] w_within={r['w_within']:.1f} w_silent={r['w_silent']:.1f} ratio={r['w_ratio']:.2f} | "
              f"FUNCTIONAL h_comp={r['h_comp']:.3f} non-stored={r['n_comp']:.3f} -> {'GO' if r['go'] else 'NO'} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
