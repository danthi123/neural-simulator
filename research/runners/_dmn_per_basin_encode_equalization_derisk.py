"""PER-BASIN ENCODE / IGNITABILITY EQUALIZATION -- make EVERY disjoint CA3 basin ignite SOLO before competition.

Closes the boundary `2026-08-14-self-initiated-all-basins-ignite-PARTIAL.md`: on the multibasin/utterance GO substrate
the LAST-ENCODED disjoint basin (index N-1) fails to ignite EVEN IN SOLO isolation on 6/6 seeds (member 0.12-0.15 vs
random 0.04) while basins 0..N-2 ignite solo -- a systematic ABSOLUTE-THRESHOLD / structural weakness, NOT competition
(so STP, a competition-reshaper, cannot rescue it; STP was banked). Multi-lever falsification showed intrinsic-boost +
3x within-recurrent weight boost barely move the tail: the failure is NOT weight MAGNITUDE, it is that the sequential
one-shot BTSP encode writes a WEAKER PATTERN (fewer synapses cross the plateau threshold) to whichever basin is encoded
LAST -- later encodes land on a substrate whose slow state has drifted from the pristine build the first basin saw.

THE DIAGNOSIS (this arc, GPU-measured, seed 42): the collapse is DEFINITIVELY POSITIONAL -- re-ordering the encode
moves the collapse to WHICHEVER basin lands LAST (order [3,2,1,0] -> basin 0 collapses; [1,3,0,2] -> basin 2). It is
NOT the cells (rules out connectivity of a subset), NOT weight magnitude (5.6x recall-time within-recurrent scaling to
match the strong basins' w does NOT ignite the tail), NOT co-basin interference (clamping every competitor silent
during the write leaves the tail collapsed), NOT accumulated dendritic/adaptation state (a full _hard_silence reset
between basins does not rescue it). THE CAUSE: the one-shot BTSP write is NOT instantaneous -- the plateau sets a slow
ELIGIBILITY trace (btsp_elig_tau_ms=1000) that converts to a synaptic weight over SUBSEQUENT steps. Every basin's
eligibility is converted by the NEXT basin's drive -- except the LAST, which has no subsequent encode -> its trace never
converts -> a sparse write (w~30 vs 150-210). The GO encode's PROTOCOL omitted the consolidation tail (the missing
companion process).

THE SURPASS that WORKS = "consolidated" encode: the sequential BTSP encode + a POST-ENCODE CONSOLIDATION settle (BTSP
still active, zero input) that lets the final basin's eligibility convert like all the others (measured: settle 600 ->
tail w 29->281, ALL basins strong). ONE global parameter, NOT a per-basin thumb -- the last basin benefits most, the
equalization EMERGES. Biology: synaptic/behavioural-timescale consolidation after the plateau (eligibility->weight).
BANKED NEGATIVE levers (comparison arms, --compare-modes): interleaved (destroys consecutive-drive compounding, all
writes collapse), isolated (removes the amplifying ambient), clamped (co-basin activity is not the cause), homeostatic
recall-time gain (magnitude is not the cause). Then FREEZE (conn.data byte-frozen during the measured wander) and run
the noise-driven DMN wander + the closed spontaneous-thought->utterance loop. NO STP (banked). NO sim/ edit;
reuse-by-import. Sequential baseline is LITERALLY `_prepare_balanced` (the GO substrate). FUNCTIONAL CORRELATE ONLY.

Smoke(GPU): SIM_BACKEND=cupy python -u -m research.runners._dmn_per_basin_encode_equalization_derisk --solo-compare --compare-modes sequential consolidated --seeds 42 --n-mem 4 --solo-steps 2200
Full (GPU): SIM_BACKEND=cupy python -u -m research.runners._dmn_per_basin_encode_equalization_derisk --seeds 42 43 44 100 101 102 --n-mem 4 --rest-steps 8000 --acid-steps 1200 --solo-steps 3000 --encode-mode consolidated
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

# reuse-by-import the VALIDATED building blocks (each 6-seed GO)
from research.runners._gap5_spontaneous_reactivation_derisk import GO_CFG  # noqa: E402
from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates  # noqa: E402
from research.runners._self_initiated_spontaneous_thought_derisk import (  # noqa: E402
    _scale_within_assembly, _steered_rest, _assembly_stats, _curiosity_wants,
)
import research.runners._self_initiation_multibasin_derisk as _MB  # noqa: E402
from research.runners._self_initiation_multibasin_derisk import (  # noqa: E402
    _selection, NOV_BY_NMEM, _extract_ca3ca3_vec,
)
from research.runners._self_initiated_utterance_derisk import (  # noqa: E402
    _lexicon, _build_mouth, _episodes, _utterance_stream, _derangement,
)
from tools.lab import attributable_to, void_if  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_dmn_per_basin_encode_equalization_derisk.json"


# ----------------------------------------------------------------------------------------------------------------------
# THE EQUALIZED PREP. Sequential (interleave=False) is the byte-clean multibasin/utterance GO substrate (delegates to
# _MB._prepare_balanced VERBATIM). Interleaved (interleave=True) reproduces that build + DISJOINT partition + post-encode
# steps (structural_sep / recall_k / selective_inhib) EXACTLY but round-robins the BTSP encode passes across basins.
# ----------------------------------------------------------------------------------------------------------------------
def _interleaved_encode(bridge, assemblies, cfg, cp):
    """Round-robin the one-shot BTSP encode: for each event, drive EVERY basin once (basin 0..N-1), so no basin is
    systematically encoded on the most-drifted substrate. Identical per-basin total (train_events reset+drive passes),
    identical BTSP parameters; ONLY the loop nesting differs from _prepare_balanced's sequential encode."""
    train_events = int(cfg["train_events"]); reset_steps = int(cfg["reset_steps"])
    drive_steps = int(cfg["drive_steps"]); encode_drive = float(cfg["encode_drive"])
    cfg_b = bridge.core_config
    cfg_b.enable_hebbian_learning = False
    cfg_b.enable_bdsp = True; cfg_b.bdsp_apical_bistable = True; cfg_b.bdsp_learning_rate = 0.0
    cfg_b.coincidence_plateau_self_regen = float(cfg["plateau_self_regen"])
    cfg_b.coincidence_plateau_v_hold = float(cfg["plateau_v_hold"]); cfg_b.apical_kir_g = float(cfg["apical_kir_g"])
    cfg_b.enable_btsp = True; cfg_b.btsp_learning_rate = float(cfg["btsp_lr"])
    cfg_b.btsp_elig_tau_ms = 1000.0; cfg_b.btsp_w_max = float(cfg["hebb_max"])
    cfg_b.btsp_hetero_dep = float(cfg["encode_btsp_hetero"])
    bridge.cp_bdsp_apical_drive = cp.zeros(int(cfg_b.num_neurons), dtype=cp.float32)
    assy_arrs = [cp.asarray(a, dtype=cp.int64) for a in assemblies]
    plateau_vecs = [cp.full(len(a), float(cfg["encode_plateau_pA"]), dtype=cp.float32) for a in assemblies]
    for ev in range(train_events):
        for m, assy_arr in enumerate(assy_arrs):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_bdsp_apical_drive[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            for _st in range(drive_steps):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[assy_arr] = encode_drive
                bridge.cp_bdsp_apical_drive[:] = 0.0
                bridge.cp_bdsp_apical_drive[assy_arr] = plateau_vecs[m]
                bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    cfg_b.enable_bdsp = False; cfg_b.enable_btsp = False; bridge.cp_bdsp_apical_drive = None


def _within_syn_indices(bridge, assemblies):
    """Per-basin WITHIN-assembly recurrent synapse index arrays (pre AND post both in the basin)."""
    from sim.backend import to_host
    conn = bridge.cp_connections
    n_all = int(bridge.core_config.num_neurons); nnz = int(conn.nnz)
    indptr = np.asarray(to_host(conn.indptr)); indices = np.asarray(to_host(conn.indices))
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    out = []
    for a in assemblies:
        memb = np.zeros(n_all, dtype=bool); memb[np.asarray(a, dtype=np.int64)] = True
        w = memb[pre_of] & memb[indices[:nnz]]
        out.append(np.nonzero(w)[0].astype(np.int64))
    return out


def _isolated_encode(bridge, assemblies, cfg, cp):
    """ISOLATED (init-ambient) sequential encode -- the equalization lever that WORKS. Keep the CONSECUTIVE per-basin
    drive that BTSP needs to compound a strong write (round-robin interleaving DESTROYS it -> all writes collapse), but
    during EACH basin's encode hold the OTHER basins' within-recurrence at the WEAK INIT level (ca3w) they had before
    any encoding -- NOT their strong ENCODED value. MEASURED cause of the last-basin collapse: the strong write is
    NETWORK-AMPLIFIED (the ambient weak recurrence boosts the plateau); by the last basin the already-encoded basins are
    STRONG attractors that ignite on the driven basin's spillover and drive the shared ca3_pv_basket to SUPPRESS it ->
    collapse (sequential w_within 163/166/213/33). Zeroing the ambient ALSO kills the amplification (all ~25 -- as bad).
    Holding co-basins at INIT gives EVERY basin the favourable weak ambient the FIRST basin saw -> equalized UP. Each
    basin's own strong write is snapshotted + restored so all persist (dentate-gyrus orthogonalisation: engrams encoded
    without mutual attractor interference; McNaughton & Morris 1987). Restored region is disjoint from the BTSP write."""
    train_events = int(cfg["train_events"]); reset_steps = int(cfg["reset_steps"])
    drive_steps = int(cfg["drive_steps"]); encode_drive = float(cfg["encode_drive"])
    cfg_b = bridge.core_config
    cfg_b.enable_hebbian_learning = False
    cfg_b.enable_bdsp = True; cfg_b.bdsp_apical_bistable = True; cfg_b.bdsp_learning_rate = 0.0
    cfg_b.coincidence_plateau_self_regen = float(cfg["plateau_self_regen"])
    cfg_b.coincidence_plateau_v_hold = float(cfg["plateau_v_hold"]); cfg_b.apical_kir_g = float(cfg["apical_kir_g"])
    cfg_b.enable_btsp = True; cfg_b.btsp_learning_rate = float(cfg["btsp_lr"])
    cfg_b.btsp_elig_tau_ms = 1000.0; cfg_b.btsp_w_max = float(cfg["hebb_max"])
    cfg_b.btsp_hetero_dep = float(cfg["encode_btsp_hetero"])
    bridge.cp_bdsp_apical_drive = cp.zeros(int(cfg_b.num_neurons), dtype=cp.float32)
    n_mem = len(assemblies)
    within_idx = _within_syn_indices(bridge, assemblies)
    conn = bridge.cp_connections
    init_within = conn.data.copy()                      # the WEAK INIT within-recurrence (ca3w), before any encoding
    assy_arrs = [cp.asarray(a, dtype=cp.int64) for a in assemblies]
    plateau_vecs = [cp.full(len(a), float(cfg["encode_plateau_pA"]), dtype=cp.float32) for a in assemblies]
    for m in range(n_mem):
        others = np.concatenate([within_idx[j] for j in range(n_mem) if j != m]) if n_mem > 1 else np.zeros(0, np.int64)
        other_dev = cp.asarray(others, dtype=cp.int64) if others.size else None
        saved = conn.data[other_dev].copy() if other_dev is not None else None
        if other_dev is not None:
            conn.data[other_dev] = init_within[other_dev]    # hold co-basins at WEAK INIT ambient (not strong, not 0)
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_bdsp_apical_drive[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            for _st in range(drive_steps):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[assy_arrs[m]] = encode_drive
                bridge.cp_bdsp_apical_drive[:] = 0.0
                bridge.cp_bdsp_apical_drive[assy_arrs[m]] = plateau_vecs[m]
                bridge._run_one_simulation_step()
        if other_dev is not None:
            conn.data[other_dev] = saved                                                # restore (disjoint from write)
        bridge.cp_external_input_current[:] = 0.0
    cfg_b.enable_bdsp = False; cfg_b.enable_btsp = False; bridge.cp_bdsp_apical_drive = None


def _clamped_encode(bridge, assemblies, cfg, cp, clamp_pA):
    """CLAMPED (attention-gated) sequential encode -- the equalization that WORKS. Keep the consecutive per-basin
    drive, but during EACH basin's write HYPERPOLARISE every OTHER basin's cells (a strong negative current) so the
    already-encoded strong attractors CANNOT ignite on the driven basin's recurrent spillover and drive the shared
    ca3_pv_basket to SUPPRESS the plateau being written. MEASURED cause of the last-basin collapse (finding
    2026-08-14 + this arc): with >=3 strong co-basins the spillover crosses a basket-saturation threshold that shuts
    down the last basin's plateau -> a SPARSE (connectivity-poor) BTSP write (w~30) that no recall-time gain rescues
    (5.6x scaling -> still no ignition). Silencing the competitors during the write lets every basin's plateau develop
    like the FIRST basin's -> a DENSE write. Biology: lateral inhibition / attentional gating restricts plasticity to
    the attended engram (Hasselmo ACh encode-vs-recall; only the driven assembly is plastic-active). No sim/ edit --
    the clamp is a host-supplied external CURRENT during encode (the world/teacher gating attention), the write itself
    is the substrate's own BTSP plateau mechanism."""
    train_events = int(cfg["train_events"]); reset_steps = int(cfg["reset_steps"])
    drive_steps = int(cfg["drive_steps"]); encode_drive = float(cfg["encode_drive"])
    cfg_b = bridge.core_config
    cfg_b.enable_hebbian_learning = False
    cfg_b.enable_bdsp = True; cfg_b.bdsp_apical_bistable = True; cfg_b.bdsp_learning_rate = 0.0
    cfg_b.coincidence_plateau_self_regen = float(cfg["plateau_self_regen"])
    cfg_b.coincidence_plateau_v_hold = float(cfg["plateau_v_hold"]); cfg_b.apical_kir_g = float(cfg["apical_kir_g"])
    cfg_b.enable_btsp = True; cfg_b.btsp_learning_rate = float(cfg["btsp_lr"])
    cfg_b.btsp_elig_tau_ms = 1000.0; cfg_b.btsp_w_max = float(cfg["hebb_max"])
    cfg_b.btsp_hetero_dep = float(cfg["encode_btsp_hetero"])
    bridge.cp_bdsp_apical_drive = cp.zeros(int(cfg_b.num_neurons), dtype=cp.float32)
    n_mem = len(assemblies)
    assy_arrs = [cp.asarray(a, dtype=cp.int64) for a in assemblies]
    plateau_vecs = [cp.full(len(a), float(cfg["encode_plateau_pA"]), dtype=cp.float32) for a in assemblies]
    others_arrs = [cp.asarray(np.concatenate([assemblies[j] for j in range(n_mem) if j != m]), dtype=cp.int64)
                   if n_mem > 1 else None for m in range(n_mem)]
    for m in range(n_mem):
        oth = others_arrs[m]
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            if oth is not None:
                bridge.cp_external_input_current[oth] = -float(clamp_pA)   # silence competitors during reset too
            bridge.cp_bdsp_apical_drive[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            for _st in range(drive_steps):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[assy_arrs[m]] = encode_drive
                if oth is not None:
                    bridge.cp_external_input_current[oth] = -float(clamp_pA)   # hyperpolarise competitors
                bridge.cp_bdsp_apical_drive[:] = 0.0
                bridge.cp_bdsp_apical_drive[assy_arrs[m]] = plateau_vecs[m]
                bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
    cfg_b.enable_bdsp = False; cfg_b.enable_btsp = False; bridge.cp_bdsp_apical_drive = None


def _consolidated_encode(bridge, assemblies, cfg, cp, settle_steps):
    """CONSOLIDATED sequential encode -- the equalization that WORKS (the mechanism is now identified). The one-shot
    BTSP write is NOT instantaneous: the plateau sets a slow ELIGIBILITY trace (btsp_elig_tau_ms=1000) that converts to
    a synaptic weight over SUBSEQUENT simulation steps. In the plain sequential encode every basin's eligibility is
    converted by the drive of the NEXT basin -- EXCEPT the LAST basin, which has no subsequent encode, so its trace
    never fully converts -> a sparse (connectivity-poor) write that no recall-time gain rescues (measured: 5.6x scaling
    does not ignite it; DEFINITIVELY POSITIONAL -- re-ordering moves the collapse to whatever basin lands last).
    THE FIX is the missing companion process: a post-encode CONSOLIDATION period (BTSP still active, zero input) that
    lets the final basin's eligibility convert like all the others (measured: settle 600 -> tail w 29->281, all basins
    strong). ONE global parameter (settle_steps), NOT a per-basin thumb -- the last basin benefits most, equalization
    EMERGES. Biology: synaptic/behavioural-timescale consolidation AFTER the plateau (the eligibility-trace-to-weight
    conversion is the mechanism; systems consolidation needs offline time after encoding)."""
    train_events = int(cfg["train_events"]); reset_steps = int(cfg["reset_steps"])
    drive_steps = int(cfg["drive_steps"]); encode_drive = float(cfg["encode_drive"])
    cfg_b = bridge.core_config
    cfg_b.enable_hebbian_learning = False
    cfg_b.enable_bdsp = True; cfg_b.bdsp_apical_bistable = True; cfg_b.bdsp_learning_rate = 0.0
    cfg_b.coincidence_plateau_self_regen = float(cfg["plateau_self_regen"])
    cfg_b.coincidence_plateau_v_hold = float(cfg["plateau_v_hold"]); cfg_b.apical_kir_g = float(cfg["apical_kir_g"])
    cfg_b.enable_btsp = True; cfg_b.btsp_learning_rate = float(cfg["btsp_lr"])
    cfg_b.btsp_elig_tau_ms = 1000.0; cfg_b.btsp_w_max = float(cfg["hebb_max"])
    cfg_b.btsp_hetero_dep = float(cfg["encode_btsp_hetero"])
    bridge.cp_bdsp_apical_drive = cp.zeros(int(cfg_b.num_neurons), dtype=cp.float32)
    for m, assy in enumerate(assemblies):                      # SEQUENTIAL order (matches the GO substrate)
        assy_arr = cp.asarray(assy, dtype=cp.int64)
        plateau_vec = cp.full(len(assy), float(cfg["encode_plateau_pA"]), dtype=cp.float32)
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_bdsp_apical_drive[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            for _st in range(drive_steps):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[assy_arr] = encode_drive
                bridge.cp_bdsp_apical_drive[:] = 0.0
                bridge.cp_bdsp_apical_drive[assy_arr] = plateau_vec
                bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
    for _ in range(int(settle_steps)):                          # POST-ENCODE CONSOLIDATION: convert the LAST basin's
        bridge.cp_external_input_current[:] = 0.0               # eligibility (btsp still on, zero input) -> dense write
        bridge.cp_bdsp_apical_drive[:] = 0.0
        bridge._run_one_simulation_step()
    cfg_b.enable_bdsp = False; cfg_b.enable_btsp = False; bridge.cp_bdsp_apical_drive = None


def _prepare_equalized(seed, cfg, do_encode=True, encode_mode="sequential", clamp_pA=4000.0, settle_steps=600):
    """encode_mode="sequential" -> the byte-clean multibasin/utterance GO substrate (delegates verbatim).
    "interleaved" -> round-robin BTSP encode (BANKED NEGATIVE: destroys consecutive-drive compounding, all writes
    collapse). "isolated" -> hold co-basins at weak init during each write (BANKED NEGATIVE: removes the amplification,
    all writes collapse). "clamped" -> hyperpolarise co-basins during each write so the plateau develops fully (the
    equalization that works)."""
    if encode_mode == "sequential":
        return _MB._prepare_balanced(seed, cfg, do_encode=do_encode)
    assert encode_mode in ("interleaved", "isolated", "clamped", "consolidated"), encode_mode
    settle_steps = int(cfg.get("settle_steps", settle_steps))       # CLI override via cfg

    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    n_ca3 = int(cfg["n_ca3"])
    _init_ca3w = float(cfg["encode_ca3w"]) if cfg.get("encode_btsp") else 6.0
    bridge = _build(seed, n_ca3=n_ca3, ca3w=_init_ca3w, ca3_density=cfg["ca3_density"],
                    coincidence=True, two_comp=True, nmda_recurrent=False, nmda_tau=100.0, nmda_ratio=1.0,
                    apical_R=cfg["apical_R"], apical_gc=cfg["apical_gc"], k_thresh=cfg["k_thresh"],
                    plateau_strength=cfg["plateau_strength"], train=True, hebb_max=cfg["hebb_max"], hebb_rate=True,
                    ca3_fb_inhib=cfg["ca3_fb_inhib"], coact_thresh=cfg["coact_thresh"], hebb_lr=None, enable_ou=False,
                    plateau_self_regen=cfg["plateau_self_regen"], plateau_v_hold=cfg["plateau_v_hold"],
                    apical_kir_g=cfg["apical_kir_g"], apical_gc_read=cfg["apical_gc_read"], ca1_ff_inhib=None)
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3"))
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    rng = np.random.default_rng(seed * 17 + 3)      # SAME partition RNG as _prepare_balanced -> identical disjoint sets
    n_assy = max(6, int(cfg["assembly_frac"] * n_ca3))
    n_mem = int(cfg["n_mem"])
    assert n_mem * n_assy <= len(ca3_idx), (
        f"disjoint requires n_mem*n_assy <= n_ca3: {n_mem}*{n_assy}={n_mem * n_assy} > {len(ca3_idx)}")
    perm = rng.permutation(np.asarray(ca3_idx, dtype=np.int64))
    assemblies = [np.asarray(sorted(perm[i * n_assy:(i + 1) * n_assy]), dtype=np.int64) for i in range(n_mem)]

    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_vec(bridge, ca3_idx, to_host)
    conn = bridge.cp_connections

    _set_gates(bridge, 1.0)
    if do_encode and cfg.get("encode_btsp"):
        if encode_mode == "consolidated":
            _consolidated_encode(bridge, assemblies, cfg, cp, settle_steps)   # <-- the equalization that works
        elif encode_mode == "clamped":
            _clamped_encode(bridge, assemblies, cfg, cp, clamp_pA)            # banked-negative comparison arm
        elif encode_mode == "isolated":
            _isolated_encode(bridge, assemblies, cfg, cp)                     # banked-negative comparison arm
        else:
            _interleaved_encode(bridge, assemblies, cfg, cp)                  # banked-negative comparison arm
    _set_gates(bridge, 0.0)

    n_ca3_loc = len(ca3_idx)
    member_local = np.zeros(n_ca3_loc, dtype=bool)
    member_local[np.asarray(sorted(ca3_pos[int(g)] for a in assemblies for g in a), dtype=np.int64)] = True
    pre_mem = member_local[pre_l_h]; post_mem = member_local[post_l_h]
    within = pre_mem & post_mem
    within_flat = flat_h[within].astype(np.int64)
    d = np.asarray(to_host(conn.data))
    w_within = float(np.mean(d[within_flat])) if within_flat.size else 0.0

    if int(cfg["structural_sep"]) >= 1:
        zsel = post_mem & (~pre_mem)
        if zsel.any():
            idxs = cp.asarray(flat_h[zsel], dtype=cp.int64)
            conn.data[idxs] = cp.zeros(int(zsel.sum()), dtype=conn.data.dtype)
    if cfg.get("recall_k_thresh") is not None:
        bridge.core_config.coincidence_k_threshold = float(cfg["recall_k_thresh"])
    if cfg["selective_inhib"]:
        n_all = int(bridge.core_config.num_neurons)
        bask_bool = np.zeros(n_all, dtype=bool)
        bask_bool[np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)] = True
        assy_bool = np.zeros(n_all, dtype=bool)
        assy_bool[np.asarray(sorted(int(g) for a in assemblies for g in a), dtype=np.int64)] = True
        conn2 = bridge.cp_connections; nnz = int(conn2.nnz)
        indptr = np.asarray(to_host(conn2.indptr)); indices = np.asarray(to_host(conn2.indices))
        pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
        spare = bask_bool[pre_of] & assy_bool[indices[:nnz]]
        if spare.any():
            idxs = cp.asarray(np.nonzero(spare)[0], dtype=cp.int64)
            conn2.data[idxs] = cp.full(int(spare.sum()), float(cfg["sel_inhib_spare"]), dtype=conn2.data.dtype)

    assembly_local = np.asarray(sorted(ca3_pos[int(g)] for a in assemblies for g in a), dtype=np.int64)
    assemblies_local = [np.asarray(sorted(ca3_pos[int(g)] for g in a), dtype=np.int64) for a in assemblies]
    ca3_arr_host = np.asarray(ca3_idx, dtype=np.int64)
    try:
        ca3_inh = set(int(g) for g in rm.inhibitory_indices("ca3"))
    except Exception:
        ca3_inh = set()
    ca3_exc_local = np.asarray([i for i, g in enumerate(ca3_idx) if int(g) not in ca3_inh], dtype=np.int64)
    union = sorted(int(g) for a in assemblies for g in a)
    max_overlap = int(max((len(set(a.tolist()) & set(b.tolist())) for i, a in enumerate(assemblies)
                           for b in assemblies[i + 1:]), default=0))
    return dict(bridge=bridge, ca3_idx=ca3_idx, ca3_arr_host=ca3_arr_host, assemblies=assemblies,
                assembly_local=assembly_local, assemblies_local=assemblies_local, ca3_exc_local=ca3_exc_local,
                within_flat=within_flat, w_within=w_within, n_assy=n_assy, max_pair_overlap=max_overlap,
                n_union=len(union))


# ----------------------------------------------------------------------------------------------------------------------
def _w_within_per(prep):
    """Per-basin mean within-assembly recurrent weight (the encode-strength proxy) from the CURRENT conn.data. Call
    BEFORE any gain/lesion modifies conn.data."""
    from sim.backend import to_host
    bridge = prep["bridge"]; conn = bridge.cp_connections
    n_all = int(bridge.core_config.num_neurons); nnz = int(conn.nnz)
    indptr = np.asarray(to_host(conn.indptr)); indices = np.asarray(to_host(conn.indices))
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    d = np.asarray(to_host(conn.data))
    out = []
    for a in prep["assemblies"]:
        memb = np.zeros(n_all, dtype=bool); memb[np.asarray(a, dtype=np.int64)] = True
        w = memb[pre_of] & memb[indices[:nnz]]
        out.append(float(d[w].mean()) if w.any() else 0.0)
    return out


def _solo_ignition_table(seed, cfg, rest_steps, min_frac, *, encode_mode="sequential", homeo_gains=None, log=True):
    """SOLO diagnosis (the DECISIVE metric), gold-standard fresh-bridge-per-basin (matches the boundary finding). For
    each basin m: build the equalized store, apply homeo gains to ALL basins (if any), LESION every other basin's
    within-recurrence to 0, run rest -> does basin m ignite UNCONTESTED on its own encoded weights? Reports per-basin
    w_within so the reader sees the encode-strength equalization directly. array_equal(before,after) is verified."""
    from sim.backend import to_host
    n_mem = int(cfg["n_mem"]); rows = []; w_within_all = None
    for m in range(n_mem):
        prep = _prepare_equalized(seed, cfg, do_encode=True, encode_mode=encode_mode)
        wperm = _w_within_per(prep)                                       # per-basin encode strength (clean store)
        if w_within_all is None:
            w_within_all = list(wperm)                                    # clean single-build per-basin table (m==0)
        if homeo_gains is not None:
            for j in range(n_mem):
                _scale_within_assembly(prep, j, float(homeo_gains[j]))
        for j in range(n_mem):
            if j != m:
                _scale_within_assembly(prep, j, 0.0)                     # zero competitors -> basin m solo
        w_before = np.asarray(to_host(prep["bridge"].cp_connections.data)).copy()
        F, _ = _steered_rest(prep, [0.0] * n_mem, rest_steps, seed, noise_on=True)
        w_after = np.asarray(to_host(prep["bridge"].cp_connections.data))
        frozen = bool(np.array_equal(w_before, w_after))
        st = _assembly_stats(F, prep["assemblies_local"], m, seed, min_frac)
        ignites = bool(st["dwell"] > 0 and st["member"] >= min_frac and st["member"] > 2.0 * (st["random"] + 1e-6))
        rows.append({"basin": m, "solo_ignites": ignites, "dwell": float(st["dwell"]), "member": float(st["member"]),
                     "random": float(st["random"]), "w_within": float(wperm[m]), "weights_frozen": frozen})
        if log:
            print(f"    [seed {seed}] SOLO basin {m}: ignites={ignites} dwell={st['dwell']:.0f} "
                  f"member {st['member']:.2f} vs rand {st['random']:.2f} w_within {wperm[m]:.1f} frozen={frozen}",
                  flush=True)
    n_solo = int(sum(1 for r in rows if r["solo_ignites"]))
    return {"per_basin": rows, "n_solo_ignite": n_solo, "all_solo_ignite": bool(n_solo == n_mem),
            "w_within_per": [r["w_within"] for r in rows], "w_within_all": w_within_all}


def _homeostatic_calibrate(seed, cfg, *, target, n_iter, eta, probe_steps, min_frac, encode_mode,
                           gmin=0.5, gmax=8.0):
    """EMERGENT per-basin ignitability equalization (synaptic-scaling homeostasis; Turrigiano 2008). Build ONCE,
    snapshot the clean encoded store. Each iteration, for each basin m: restore the clean store, apply the CURRENT
    per-basin gains to ALL basins, LESION the others (solo), run a short rest probe, read basin m's OWN solo member,
    and multiplicatively adjust g[m] toward a SINGLE SCALAR `target` (g *= 1 + eta*(target-member)/target, clipped).
    Each basin reads its OWN error against ONE global setpoint -> the per-basin gains EMERGE (not a host per-basin
    set). Returns the emergent gain vector, applied (frozen) to every downstream wander."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    prep = _prepare_equalized(seed, cfg, do_encode=True, encode_mode=encode_mode)
    conn = prep["bridge"].cp_connections
    snap = conn.data.copy()
    n_mem = int(cfg["n_mem"])
    gains = [1.0] * n_mem
    history = []
    for it in range(int(n_iter)):
        members = []
        for m in range(n_mem):
            conn.data[:] = snap                                          # restore the clean encoded store
            for j in range(n_mem):
                _scale_within_assembly(prep, j, gains[j] if j == m else 0.0)   # gain on m, lesion others (solo)
            F, _ = _steered_rest(prep, [0.0] * n_mem, probe_steps, seed, noise_on=True)
            st = _assembly_stats(F, prep["assemblies_local"], m, seed, min_frac)
            mem = float(st["member"]); members.append(mem)
            err = float(target) - mem                                    # basin m reads its OWN error vs ONE setpoint
            gains[m] = float(np.clip(gains[m] * (1.0 + eta * err / float(target)), gmin, gmax))
        history.append({"iter": it, "members": members, "gains": list(gains)})
        print(f"    [seed {seed}] HOMEO iter {it}: member {[round(x,2) for x in members]} "
              f"-> gains {[round(g,2) for g in gains]}", flush=True)
    conn.data[:] = snap                                                  # leave the store clean; caller re-applies gains
    return {"gains": gains, "target": float(target), "single_scalar_target": True, "history": history}


def _run_wander(seed, cfg, rest_steps, noise_on, *, gains, encode_mode, homeo_gains=None, do_encode=True):
    """A FRESH deterministic bridge (same seed -> byte-identical substrate + encode + Poisson stream). Applies the
    homeostatic per-basin gains (if any) then the per-condition curiosity gains, arms nothing (NO STP), runs the
    noise-driven wander with plasticity byte-frozen."""
    n_mem = int(cfg["n_mem"])
    prep = _prepare_equalized(seed, cfg, do_encode=do_encode, encode_mode=encode_mode)
    if homeo_gains is not None:
        for i in range(n_mem):
            _scale_within_assembly(prep, i, float(homeo_gains[i]))
    if gains is not None:
        for i in range(n_mem):
            _scale_within_assembly(prep, i, float(gains[i]))
    F, diag = _steered_rest(prep, [0.0] * n_mem, rest_steps, seed, noise_on=noise_on)
    return F, prep, diag


def _visit_order(F, assemblies_local, min_frac, cap=48):
    order = [int(c) for (c, s, e) in _episodes(F, assemblies_local, min_frac)]
    return order[:cap]


# ----------------------------------------------------------------------------------------------------------------------
def one_seed(seed, n_mem, rest_steps, acid_steps, solo_steps, gain_scale, min_frac, D, encode_mode, homeostatic,
             homeo_target, homeo_iter, homeo_eta, homeo_probe, settle_steps=600):
    t0 = time.time()
    out = {"seed": seed, "n_mem": n_mem, "encode_mode": encode_mode, "homeostatic": bool(homeostatic),
           "settle_steps": int(settle_steps)}
    cfg = dict(GO_CFG); cfg["n_ca3"] = 2000; cfg["n_mem"] = int(n_mem); cfg["settle_steps"] = int(settle_steps)
    agents, verbs, patients, vocab = _lexicon(n_mem)
    out["facts"] = [f"{agents[i]} {verbs[i]} {patients[i]}" for i in range(n_mem)]

    # (0) OPTIONAL homeostatic calibration (emergent per-basin gains; single scalar target)
    homeo_gains = None
    if homeostatic:
        cal = _homeostatic_calibrate(seed, cfg, target=homeo_target, n_iter=homeo_iter, eta=homeo_eta,
                                     probe_steps=homeo_probe, min_frac=min_frac, encode_mode=encode_mode)
        homeo_gains = cal["gains"]; out["homeo"] = cal

    # (1) THE DECISIVE METRIC: per-basin SOLO ignition on the EQUALIZED store (baseline sequential for contrast)
    out["solo_equalized"] = _solo_ignition_table(seed, cfg, solo_steps, min_frac, encode_mode=encode_mode,
                                                  homeo_gains=homeo_gains)
    out["solo_baseline_seq"] = _solo_ignition_table(seed, cfg, solo_steps, min_frac, encode_mode="sequential",
                                                     homeo_gains=None)
    print(f"  [seed {seed}] SOLO equalized {out['solo_equalized']['n_solo_ignite']}/{n_mem} "
          f"(baseline seq {out['solo_baseline_seq']['n_solo_ignite']}/{n_mem}) | per-basin member "
          f"eq {[round(r['member'],2) for r in out['solo_equalized']['per_basin']]} "
          f"seq {[round(r['member'],2) for r in out['solo_baseline_seq']['per_basin']]} | w_within(1-build) "
          f"eq {[round(w,0) for w in out['solo_equalized']['w_within_all']]} "
          f"seq {[round(w,0) for w in out['solo_baseline_seq']['w_within_all']]} ({time.time()-t0:.0f}s)", flush=True)

    # curiosity gains (identical construction to multibasin/utterance/all-basins)
    nov_rng = np.random.default_rng(seed * 7919 + 1)
    novelties = [float(v) for v in nov_rng.permutation(np.asarray(NOV_BY_NMEM[n_mem], dtype=float))]
    wants, _ = _curiosity_wants(seed, novelties)
    wmax = max(wants) if wants else 1.0
    gains_on = [1.0 + gain_scale * (w / wmax if wmax > 1e-9 else 0.0) for w in wants]
    order = [int(i) for i in np.argsort(-np.asarray(novelties))]
    gvals = sorted(gains_on, reverse=True)
    gains_reversed = [0.0] * n_mem
    for k, ci in enumerate(order):
        gains_reversed[ci] = gvals[n_mem - 1 - k]
    novel_set = np.asarray(order[:max(1, n_mem // 2)], dtype=int)
    out["novelties"] = novelties; out["gains_on"] = gains_on; out["novel_set"] = novel_set.tolist()

    # the MOUTH
    comp, utt_by_agent, decode_ok, moat = _build_mouth(seed, agents, verbs, patients, vocab, D)
    ident = list(range(n_mem))
    out["mouth_fidelity"] = bool(all(decode_ok)); out["moat_abstains"] = bool(moat)

    # (2) BALANCED, UNIFORM gain -> IGNITION COMPLETENESS in the competitive wander (the secondary headline)
    F_bal, prep_b, d_bal = _run_wander(seed, cfg, rest_steps, True, gains=[1.0] * n_mem, encode_mode=encode_mode,
                                       homeo_gains=homeo_gains)
    sel_bal = _selection(F_bal, prep_b["assemblies_local"], seed, min_frac)
    n_ig_bal = int(sel_bal["n_visited_coherent"]); all_ig_bal = (n_ig_bal == n_mem)
    out["balanced"] = {"n_visited_coherent": n_ig_bal, "all_ignite": all_ig_bal, "dwell": sel_bal["dwell"],
                       "coherent": sel_bal["coherent"], "per_member": sel_bal["per_member"],
                       "max_pair_overlap": int(prep_b["max_pair_overlap"]), "w_within": float(prep_b["w_within"]),
                       "weights_frozen": bool(d_bal["weights_frozen"]),
                       "visit_order": _visit_order(F_bal, prep_b["assemblies_local"], min_frac)}
    print(f"  [seed {seed}] BALANCED wander: n_ignite {n_ig_bal}/{n_mem} all={all_ig_bal} "
          f"dwell {[int(x) for x in sel_bal['dwell']]} overlap {prep_b['max_pair_overlap']} "
          f"frozen {d_bal['weights_frozen']} ({time.time()-t0:.0f}s)", flush=True)

    # (3) CURIOSITY-ON production wander + CLOSED LOOP -> speak about EVERY concept
    F_on, prep_on, d_on = _run_wander(seed, cfg, rest_steps, True, gains=gains_on, encode_mode=encode_mode,
                                      homeo_gains=homeo_gains)
    sel_on = _selection(F_on, prep_on["assemblies_local"], seed, min_frac)
    st_on = _utterance_stream(F_on, prep_on["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    out["on"] = {"n_utt": st_on["n_utt"], "about_rate": st_on["about_rate"],
                 "n_concepts_spoken": st_on["n_concepts_spoken"], "share": st_on["share"],
                 "member": sel_on["pooled_member"], "random": sel_on["pooled_random"],
                 "n_visited_coherent": sel_on["n_visited_coherent"],
                 "apical_rest_max": d_on["apical_rest_max"], "weights_frozen": bool(d_on["weights_frozen"]),
                 "visit_order": _visit_order(F_on, prep_on["assemblies_local"], min_frac)}
    print(f"  [seed {seed}] ON: utt {st_on['n_utt']} about {st_on['about_rate']:.2f} concepts_spoken "
          f"{st_on['n_concepts_spoken']}/{n_mem} member {sel_on['pooled_member']:.2f} vs rand "
          f"{sel_on['pooled_random']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # (4) controls: SCRAMBLE routing, REVERSED curiosity, NO-NOISE (internal trigger), STORE-LESION (substrate)
    scr = _derangement(n_mem, seed)
    st_scr = _utterance_stream(F_on, prep_on["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, scr)
    F_rv, prep_rv, _ = _run_wander(seed, cfg, rest_steps, True, gains=gains_reversed, encode_mode=encode_mode,
                                   homeo_gains=homeo_gains)
    st_rv = _utterance_stream(F_rv, prep_rv["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    F_nn, prep_nn, d_nn = _run_wander(seed, cfg, acid_steps, False, gains=gains_on, encode_mode=encode_mode,
                                      homeo_gains=homeo_gains)
    st_nn = _utterance_stream(F_nn, prep_nn["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    F_sl, prep_sl, _ = _run_wander(seed, cfg, rest_steps, True, gains=gains_on, encode_mode=encode_mode,
                                   homeo_gains=homeo_gains, do_encode=False)
    sel_sl = _selection(F_sl, prep_sl["assemblies_local"], seed, min_frac)
    st_sl = _utterance_stream(F_sl, prep_sl["assemblies_local"], agents, utt_by_agent, decode_ok, min_frac, ident)
    share_on = np.asarray(st_on["share"]); share_rv = np.asarray(st_rv["share"])
    novel_on = float(share_on[novel_set].sum()); novel_rv = float(share_rv[novel_set].sum())
    out["scramble_about"] = st_scr["about_rate"]
    out["reversed"] = {"novel_share": novel_rv, "n_utt": st_rv["n_utt"]}
    out["novel_share_on"] = novel_on
    out["no_noise"] = {"n_utt": st_nn["n_utt"], "apical_rest_max": d_nn["apical_rest_max"]}
    out["store_lesion"] = {"n_utt": st_sl["n_utt"], "about_n": st_sl["n_about"],
                           "member": sel_sl["pooled_member"], "random": sel_sl["pooled_random"]}
    out["bias"] = {"novel_share_on": novel_on, "novel_share_reversed": novel_rv,
                   "attributable": attributable_to("curiosity-gain @ novel-concept utterance share (on vs reversed)",
                                                    novel_on, novel_rv)}
    print(f"  [seed {seed}] SCRAMBLE about {st_scr['about_rate']:.2f} | NO-NOISE utt {st_nn['n_utt']} | "
          f"STORE-LESION utt {st_sl['n_utt']} member {sel_sl['pooled_member']:.2f} | novel-share on {novel_on:.2f} "
          f"rev {novel_rv:.2f}", flush=True)

    void_if(st_on["n_utt"] == 0, f"seed {seed}: ON wander produced 0 utterances (nothing to interpret)")

    # ---- per-seed GO gate ----
    store_lesion_lb = bool(st_sl["n_utt"] <= max(1, int(0.25 * st_on["n_utt"])) or st_sl["n_about"] == 0
                           or sel_sl["pooled_member"] < 0.5 * sel_on["pooled_member"])
    checks = dict(
        disjoint_ok=(out["balanced"]["max_pair_overlap"] == 0),
        mouth_fidelity=(out["mouth_fidelity"] and out["moat_abstains"]),
        ALL_BASINS_SOLO_IGNITE=bool(out["solo_equalized"]["all_solo_ignite"]),                  # THE PRIMARY HEADLINE
        equalization_load_bearing=bool(out["solo_equalized"]["n_solo_ignite"]
                                       > out["solo_baseline_seq"]["n_solo_ignite"]),            # eq beats sequential
        ALL_BASINS_IGNITE_WANDER=bool(all_ig_bal),                                              # secondary (competition)
        loop_speaks_every_concept=(st_on["n_concepts_spoken"] == n_mem),
        about_selected=bool(st_on["about_rate"] >= 0.90 and sel_on["pooled_member"] >= min_frac
                            and sel_on["pooled_member"] > 2.0 * (sel_on["pooled_random"] + 1e-6)),
        scramble_collapses=bool(st_scr["about_rate"] <= 0.15),
        curiosity_steered=bool(novel_on >= novel_rv + 0.10),
        internally_triggered=bool(st_nn["n_utt"] == 0 and out["on"]["weights_frozen"]
                                  and (out["on"]["apical_rest_max"] is None
                                       or out["on"]["apical_rest_max"] <= float(GO_CFG["plateau_v_hold"]) + 1e-3)),
        store_lesion_load_bearing=store_lesion_lb,
        weights_byte_frozen=bool(out["balanced"]["weights_frozen"] and out["on"]["weights_frozen"]
                                 and all(r["weights_frozen"] for r in out["solo_equalized"]["per_basin"])),
    )
    out["checks"] = checks
    out["seed_go"] = bool(all(checks.values()))
    print(f"  [seed {seed}] => {'GO' if out['seed_go'] else 'no'}  {checks}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def _main_solo_compare(a):
    """CHEAP diagnostic: per-basin SOLO ignition table for {sequential baseline, interleaved, (opt) homeostatic}. The
    decisive smoke -- does the equalization make the LAST-encoded basin ignite solo?"""
    print(f"[eq SOLO-COMPARE] n_mem={a.n_mem} solo={a.solo_steps} seeds={a.seeds} "
          f"backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    cfg = dict(GO_CFG); cfg["n_ca3"] = 2000; cfg["n_mem"] = int(a.n_mem); cfg["settle_steps"] = int(a.settle_steps)
    modes = [m for m in a.compare_modes]
    per = []
    for s in a.seeds:
        row = {"seed": s}
        for mode in modes:
            print(f"  [seed {s}] --- {mode.upper()} ---", flush=True)
            row[mode] = _solo_ignition_table(s, cfg, a.solo_steps, a.min_frac, encode_mode=mode)
        if a.homeostatic:
            base_mode = "consolidated" if "consolidated" in modes else "sequential"
            print(f"  [seed {s}] --- HOMEOSTATIC (on {base_mode}) ---", flush=True)
            cal = _homeostatic_calibrate(s, cfg, target=a.homeo_target, n_iter=a.homeo_iter, eta=a.homeo_eta,
                                         probe_steps=a.homeo_probe, min_frac=a.min_frac, encode_mode=base_mode)
            row["homeostatic"] = _solo_ignition_table(s, cfg, a.solo_steps, a.min_frac, encode_mode=base_mode,
                                                      homeo_gains=cal["gains"])
            row["homeo_gains"] = cal["gains"]
        per.append(row)
        msg = " ".join(f"{mode} {row[mode]['n_solo_ignite']}/{a.n_mem}" for mode in modes)
        if a.homeostatic:
            msg += f" homeo {row['homeostatic']['n_solo_ignite']}/{a.n_mem}"
        print(f"  [seed {s}] SOLO n_ignite: {msg}", flush=True)
    summary = {"probe": "dmn_encode_equalization_SOLO_COMPARE", "seeds": a.seeds, "n_mem": a.n_mem,
               "solo_steps": a.solo_steps, "min_frac": a.min_frac, "per_seed": per}
    Path(a.out).with_suffix(".solocompare.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=4, choices=[4, 5, 6, 8])
    ap.add_argument("--rest-steps", type=int, default=8000)
    ap.add_argument("--acid-steps", type=int, default=1200)
    ap.add_argument("--solo-steps", type=int, default=3000)
    ap.add_argument("--gain-scale", type=float, default=1.0)
    ap.add_argument("--min-frac", type=float, default=0.30)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--encode-mode", choices=["sequential", "interleaved", "isolated", "clamped", "consolidated"],
                    default="consolidated", help="consolidated = sequential + post-encode consolidation (WORKS)")
    ap.add_argument("--compare-modes", nargs="+", default=["sequential", "consolidated"],
                    help="solo-compare arms (subset of sequential/interleaved/isolated/clamped/consolidated)")
    ap.add_argument("--settle-steps", type=int, default=600, help="post-encode BTSP consolidation steps (consolidated)")
    ap.add_argument("--homeostatic", action="store_true", help="emergent per-basin ignitability calibration (lever B)")
    ap.add_argument("--homeo-target", type=float, default=0.32, help="SINGLE scalar solo-member setpoint")
    ap.add_argument("--homeo-iter", type=int, default=3)
    ap.add_argument("--homeo-eta", type=float, default=1.2)
    ap.add_argument("--homeo-probe", type=int, default=1000)
    ap.add_argument("--solo-compare", action="store_true", help="cheap diag: solo table across --compare-modes")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.solo_compare:
        return _main_solo_compare(a)

    print(f"[eq] n_mem={a.n_mem} rest={a.rest_steps} acid={a.acid_steps} solo={a.solo_steps} "
          f"encode_mode={a.encode_mode} homeostatic={a.homeostatic} seeds={a.seeds} "
          f"backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    t0 = time.time(); per = []; err = None
    partial_path = Path(a.out).with_suffix(".partial.json")
    try:
        for s in a.seeds:
            per.append(one_seed(s, a.n_mem, a.rest_steps, a.acid_steps, a.solo_steps, a.gain_scale, a.min_frac, a.D,
                                a.encode_mode, a.homeostatic, a.homeo_target, a.homeo_iter, a.homeo_eta, a.homeo_probe,
                                settle_steps=a.settle_steps))
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps({"partial": True, "seeds_done": [p["seed"] for p in per],
                                                "per_seed": per}, indent=2, default=str))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    preconditions = []; attribution = None
    if err is None and per:
        n = len(per)
        thresh = max(1, (n + 1) // 2) if a.smoke else max(1, (5 * n + 5) // 6)
        n_solo_all = sum(1 for p in per if p["solo_equalized"]["all_solo_ignite"])
        n_wander_all = sum(1 for p in per if p["balanced"]["all_ignite"])
        n_go = sum(1 for p in per if p["seed_go"])
        go = (n_solo_all >= thresh) and (n_wander_all >= thresh) and (n_go >= thresh)
        m_solo = float(np.mean([p["solo_equalized"]["n_solo_ignite"] for p in per]))
        m_solo_seq = float(np.mean([p["solo_baseline_seq"]["n_solo_ignite"] for p in per]))
        m_ig_bal = float(np.mean([p["balanced"]["n_visited_coherent"] for p in per]))
        m_concepts = float(np.mean([p["on"]["n_concepts_spoken"] for p in per]))
        m_about = float(np.mean([p["on"]["about_rate"] for p in per]))
        m_scr = float(np.mean([p["scramble_about"] for p in per]))
        m_member = float(np.mean([p["on"]["member"] for p in per]))
        m_random = float(np.mean([p["on"]["random"] for p in per]))
        m_novel_on = float(np.mean([p["novel_share_on"] for p in per]))
        m_novel_rv = float(np.mean([p["reversed"]["novel_share"] for p in per]))
        attribution = attributable_to("curiosity-gain @ novel-concept utterance share (6-seed, on vs reversed)",
                                       m_novel_on, m_novel_rv)

        vd = Verdict("DMN per-basin encode equalization: all basins ignite SOLO (6-seed)", chance=m_random)
        vd.require("PRIMARY: every disjoint basin ignites SOLO on >= threshold seeds",
                   n_solo_all, expect=lambda x, t=thresh: x >= t)
        vd.control("equalization load-bearing: SOLO n_ignite equalized vs sequential baseline",
                   m_solo, m_solo_seq, min_separation=0.5)
        vd.require("basins DISJOINT (max pairwise overlap == 0) every seed",
                   all(p["balanced"]["max_pair_overlap"] == 0 for p in per), expect=True)
        vd.require("SECONDARY: all basins ignite in the balanced uniform-gain WANDER on >= threshold seeds",
                   n_wander_all, expect=lambda x, t=thresh: x >= t)
        vd.require("closed loop speaks about EVERY concept (mean n_concepts_spoken == n_mem)",
                   m_concepts, expect=lambda x, nm=a.n_mem: x >= nm - 1e-9)
        vd.require("ABOUT-THE-SELECTED-CONCEPT rate (mean) >= 0.9", m_about, expect=lambda x: x >= 0.9)
        vd.control("about-selected: production vs SCRAMBLE-routing", m_about, m_scr, min_separation=0.5)
        vd.control("curiosity-steered: novel utterance share on vs reversed", m_novel_on, m_novel_rv,
                   min_separation=0.05)
        vd.control("coherent: surfaced member vs random floor", m_member, m_random, min_separation=0.15)
        vd.require("internally-triggered: NO-NOISE -> 0 utterances every seed",
                   all(p["no_noise"]["n_utt"] == 0 for p in per), expect=True)
        vd.require("substrate-attributable: STORE-LESION collapses the utterance stream every seed",
                   all(p["checks"]["store_lesion_load_bearing"] for p in per), expect=True)
        vd.require("plasticity byte-frozen (solo + wander) every seed",
                   all(p["checks"]["weights_byte_frozen"] for p in per), expect=True)
        vd.disabled("hebbian/BTSP plasticity during every wander/solo probe", "frozen store measurement")
        decided = vd.decide(go)
        preconditions = decided["preconditions"]

        lever = f"{a.encode_mode} encode" + ("+homeostatic" if a.homeostatic else "")
        verdict = (f"{'GO' if go else 'PARTIAL/NEGATIVE'} {n_go}/{n} -- per-basin ENCODE equalization ({lever}) makes "
                   f"{m_solo:.1f}/{a.n_mem} disjoint basins ignite SOLO (baseline sequential {m_solo_seq:.1f}/{a.n_mem}); "
                   f"all-solo-ignite on {n_solo_all}/{n} seeds. Competitive wander ignites {m_ig_bal:.1f}/{a.n_mem} "
                   f"(all-{a.n_mem} on {n_wander_all}/{n} seeds); closed loop speaks about {m_concepts:.1f}/{a.n_mem} "
                   f"concepts (about-selected {m_about:.2f} vs SCRAMBLE {m_scr:.2f}); coherence member {m_member:.2f} vs "
                   f"random {m_random:.2f}; novel-share on {m_novel_on:.2f} vs reversed {m_novel_rv:.2f}"
                   f"{'; %.0f%% attributable to the curiosity gain' % (100 * attribution) if attribution is not None else ''}."
                   f"{' => every disjoint basin ignites; the DMN wander covers the FULL store.' if go else ' Per THE LAW: bank the method, name the residual + next lever; not a stop.'}")
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        vd = Verdict("DMN per-basin encode equalization (6-seed)")
        vd.require("run completed without error", err is None, expect=True)
        preconditions = vd.decide(False)["preconditions"]

    summary = {"probe": "dmn_per_basin_encode_equalization", "GO": go, "n_go": n_go, "seeds": a.seeds,
               "n_mem": a.n_mem, "rest_steps": a.rest_steps, "acid_steps": a.acid_steps, "solo_steps": a.solo_steps,
               "encode_mode": a.encode_mode, "homeostatic": a.homeostatic, "homeo_target": a.homeo_target,
               "gain_scale": a.gain_scale, "min_frac": a.min_frac, "D": a.D,
               "curiosity_bias_attribution": attribution, "preconditions": preconditions,
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110 + f"\n[eq] VERDICT: {verdict}\n[eq] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
