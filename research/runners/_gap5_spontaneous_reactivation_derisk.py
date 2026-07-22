"""gap#5 RANK 1 — SPONTANEOUS (no-cue) single-assembly reactivation on the CLOSED bistable store.

2026-07-21 (research gate 2026-07-21-gap5-SWR-generative-replay-research-gate.md). ALL prior SWR work is CUE-driven
completion->CA1; the genuinely-open piece of the generative-replay/imaginative-replay loop -- a stored assembly that
reactivates WITHOUT a cue, triggered by non-specific noise -- has only ever been tested in 3 RETRACTED confounds:
  (1) the SELF-SUSTAINING artifact (an attractor that never turned off);
  (2) the Wang plasticity+noise confound (recall-time LTP built the attractor; OU-noise-on gave 0.5 EVERYWHERE = uniform
      noise, not basin-selective);
  (3) the `_hard_silence` dendritic-reset bug (a latched plateau persisted through "silence").
This de-risk builds the FIRST non-artifact spontaneous test on the CLOSED completion machinery (intrinsic dendritic
bistability + KIR down-state = a GENUINE silent rest), with anti-cheats that each RETIRE a named confound. It is a
new research runner, reuse-by-import of the CLOSED completion `run()`'s building blocks (_build/_set_gates/_extract/
competition/structural-sep/selective-inhib/_hard_silence); NO `sim/` edit.

THE TEST: BTSP/rate-Hebbian-encode ONE assembly into the CLOSED bistable store, FREEZE plasticity, RESET the dendritic
state (the committed `_hard_silence` fix), then run a REST phase with WEAK NON-SPECIFIC background (OU noise, low sigma;
NO cue, NO recall_drive). Does the assembly SPONTANEOUSLY + BASIN-SELECTIVELY reactivate as DISCRETE events, then rest
silent between them?

GO GATE: discrete spontaneous events OCCUR (event_rate>0) AND are assembly-SPECIFIC (member_frac >> random-set frac,
spec>margin) AND the net RESTS silent between events (LOW duty cycle -- discrete events, NOT a continuous ON state).
Anti-cheats (each retires a named retracted confound; VERIFIED not asserted):
  - NO-NOISE (OU off, no drive) -> SILENT, 0 events            [retires the SELF-SUSTAINING artifact -- THE ACID TEST]
  - NO-ENCODING (skip the store, same noise) -> no specific events  [retires the noise artifact]
  - SHUFFLED within-assembly weights (keep noise, scramble learned recurrent) -> no specific events  [retires OU-uniform]
  - PERMUTED-ASSEMBLY (score events vs a random same-size non-assembly set) -> member_frac -> chance  [non-specific]
  - FROZEN plasticity MANDATORY during rest (verified: cp_connections.data byte-unchanged)  [retires the Wang confound]
  - DENDRITIC-RESET verified (cp_v_apical at rest-start <= plateau_v_hold, no latched plateau)  [retires _hard_silence bug]

CPU-smoke: SIM_BACKEND=numpy python -m research.runners._gap5_spontaneous_reactivation_derisk --seeds 42 --n-ca3 1000
Full run (GPU): SIM_BACKEND=cupy python -m research.runners._gap5_spontaneous_reactivation_derisk --seeds 42 43 44 100 101 102 --n-ca3 2000
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates  # noqa: E402
from research.runners._riii_ca3_competitive_completion_payoff_derisk import _extract_ca3ca3_coincidence  # noqa: E402
from research.runners._riii_ca3_synchronous_assembly_derisk import run as _riii_run  # noqa: E402 (positive-control cross-check)

OUT = _REPO / "research" / "findings" / "raw" / "_gap5_spontaneous_reactivation_derisk.json"

# The CLOSED completion store == research/runners/_gap4_btsp_completion_unification_6seed.py BTSP_CFG (the BTSP
# plateau-gated ONE-SHOT encode -- the path that runner actually tests, GO 5/6). NOTE (verified 2026-07-21): the plain
# Hebbian GO_CFG (encode_drive=3000/no_sync/recall_k_thresh=110) does NOT complete even at n_ca3=2000 (w_within~15,
# below the plateau k_thresh) -- the completing store is the BTSP encode (btsp_w_max=300, recall_k_thresh=40). RANK 1
# = single-assembly, n_mem=1. recall_drive/recall_steps used only by the positive-control cross-check.
CFG = dict(n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000.0, no_sync=True,
           bistable=True, nmda_recurrent=False, enable_ou=False, selective_inhib=True, structural_sep=1,
           plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc=1.0, apical_gc_read=5.0,
           recall_drive=700.0, recall_steps=150,
           # BTSP encode (== _gap4 BTSP_CFG):
           encode_btsp=True, encode_ca3w=0.5, encode_plateau_pA=250.0, btsp_lr=0.02, hebb_max=300.0,
           train_events=30, recall_k_thresh=40.0,
           # run() defaults made explicit (so _prepare below reproduces the encode faithfully):
           drive_steps=48, reset_steps=15, k_thresh=18.0, plateau_strength=120.0, coact_thresh=0.02,
           ca3_fb_inhib=20.0, apical_R=50.0, plateau_v_hold=-35.0, sel_inhib_spare=0.0, encode_btsp_hetero=0.0, n_mem=1)
GO_CFG = CFG  # back-compat alias used below


# ----------------------------------------------------------------------------------------------------------------------
# _prepare: reproduce the CLOSED completion GO_CFG bridge (build + encode + structural-sep + recall-k + selective-inhib),
# faithfully mirroring _riii...run()'s Hebbian path (lines 75-310), so the rest phase runs on the SAME basin the
# completion test validates. Cross-checked against run() (positive control). do_encode=False = the NO-ENCODING anti-cheat.
# ----------------------------------------------------------------------------------------------------------------------
def _extract_ca3ca3_vec(bridge, ca3_idx, to_host):
    """VECTORIZED (numpy) equivalent of _extract_ca3ca3_coincidence -- same (flat, pre_local, post_local) for the
    coincidence-masked ca3->ca3 recurrent synapses, without the O(nnz) python loop (the n_ca3=2000 CPU bottleneck)."""
    conn = bridge.cp_connections
    nnz = int(conn.nnz)
    n_all = int(bridge.core_config.num_neurons)
    mask = np.asarray(to_host(bridge.cp_coincidence_synapse_mask[:nnz])).astype(bool)
    indptr = np.asarray(to_host(conn.indptr)); indices = np.asarray(to_host(conn.indices))
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    post_of = indices[:nnz]
    loc = np.full(n_all, -1, dtype=np.int64); loc[np.asarray(ca3_idx, dtype=np.int64)] = np.arange(len(ca3_idx))
    pre_l_all = loc[pre_of]; post_l_all = loc[post_of]
    sel = mask & (pre_l_all >= 0) & (post_l_all >= 0)
    flat = np.nonzero(sel)[0].astype(np.int64)
    return flat, pre_l_all[sel].astype(np.int64), post_l_all[sel].astype(np.int64)


def _prepare(seed, cfg, do_encode=True):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    n_ca3 = int(cfg["n_ca3"])
    # BTSP: init the recurrent LOW (encode_ca3w=0.5) so the one-shot BTSP builds it; build hebb_max == btsp_w_max.
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
    rng = np.random.default_rng(seed * 17 + 3)   # SAME draw as run() -> the same assembly the positive control validates
    n_assy = max(6, int(cfg["assembly_frac"] * n_ca3))
    n_mem = int(cfg["n_mem"])
    assemblies = [np.asarray(sorted(rng.choice(ca3_idx, n_assy, replace=False)), dtype=np.int64) for _ in range(n_mem)]

    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_vec(bridge, ca3_idx, to_host)
    conn = bridge.cp_connections

    # ENCODE (BTSP plateau-gated ONE-SHOT storing -- mirrors _riii...run() encode_btsp path, lines 134-178).
    # do_encode=False -> NO-ENCODING anti-cheat (weights stay at init 0.5 -> no completing basin).
    _set_gates(bridge, 1.0)
    if do_encode and cfg.get("encode_btsp"):
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
        for m, assy in enumerate(assemblies):
            assy_arr = cp.asarray(assy, dtype=cp.int64)
            plateau_vec = cp.full(len(assy), float(cfg["encode_plateau_pA"]), dtype=cp.float32)
            for ev in range(train_events):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_bdsp_apical_drive[:] = 0.0
                for _ in range(reset_steps):
                    bridge._run_one_simulation_step()
                for _st in range(drive_steps):
                    bridge.cp_external_input_current[:] = 0.0
                    bridge.cp_external_input_current[assy_arr] = encode_drive       # co-fire the assembly (pre-elig)
                    bridge.cp_bdsp_apical_drive[:] = 0.0
                    bridge.cp_bdsp_apical_drive[assy_arr] = plateau_vec             # plateau ON the assembly (IS_post)
                    bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
        cfg_b.enable_bdsp = False; cfg_b.enable_btsp = False; bridge.cp_bdsp_apical_drive = None   # recall uses two_comp only
    _set_gates(bridge, 0.0)

    # membership bool arrays (vectorized; no python loops over nnz)
    n_ca3_loc = len(ca3_idx)
    member_local = np.zeros(n_ca3_loc, dtype=bool)
    member_local[np.asarray(sorted(ca3_pos[int(g)] for a in assemblies for g in a), dtype=np.int64)] = True
    pre_mem = member_local[pre_l_h]; post_mem = member_local[post_l_h]
    within = pre_mem & post_mem
    within_flat = flat_h[within].astype(np.int64)

    # within-ensemble weight read (did the store grow to the completion scale?)
    d = np.asarray(to_host(conn.data))
    w_within = float(np.mean(d[within_flat])) if within_flat.size else 0.0

    # STRUCTURAL SEPARATION (structural_sep=1: zero non-member->member recurrent edges)
    if int(cfg["structural_sep"]) >= 1:
        zsel = post_mem & (~pre_mem)
        if zsel.any():
            idxs = cp.asarray(flat_h[zsel], dtype=cp.int64)
            conn.data[idxs] = cp.zeros(int(zsel.sum()), dtype=conn.data.dtype)

    # DECOUPLE recall dendritic threshold (recall_k_thresh: high at recall/rest so only strong learned coincident
    # drive crosses the plateau -> specificity)
    if cfg.get("recall_k_thresh") is not None:
        bridge.core_config.coincidence_k_threshold = float(cfg["recall_k_thresh"])

    # ASSEMBLY-SELECTIVE INHIBITION (selective_inhib: spare the assembly's own cells from the shared basket)
    if cfg["selective_inhib"]:
        n_all = int(bridge.core_config.num_neurons)
        bask_bool = np.zeros(n_all, dtype=bool); bask_bool[np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)] = True
        assy_bool = np.zeros(n_all, dtype=bool); assy_bool[np.asarray(sorted(int(g) for a in assemblies for g in a), dtype=np.int64)] = True
        conn2 = bridge.cp_connections; nnz = int(conn2.nnz)
        indptr = np.asarray(to_host(conn2.indptr)); indices = np.asarray(to_host(conn2.indices))
        pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
        spare = bask_bool[pre_of] & assy_bool[indices[:nnz]]        # basket -> assembly-member (I->E)
        if spare.any():
            idxs = cp.asarray(np.nonzero(spare)[0], dtype=cp.int64)
            conn2.data[idxs] = cp.full(int(spare.sum()), float(cfg["sel_inhib_spare"]), dtype=conn2.data.dtype)

    assembly_local = np.asarray(sorted(ca3_pos[int(g)] for a in assemblies for g in a), dtype=np.int64)   # union
    assemblies_local = [np.asarray(sorted(ca3_pos[int(g)] for g in a), dtype=np.int64) for a in assemblies]  # per-assembly
    ca3_arr_host = np.asarray(ca3_idx, dtype=np.int64)
    # CA3 EXC local positions (for the CA3-targeted Poisson noise -- exclude the region's inhibitory cells so the
    # background perturbs the excitatory attractor, not the FS pool that would clamp it)
    try:
        ca3_inh = set(int(g) for g in rm.inhibitory_indices("ca3"))
    except Exception:
        ca3_inh = set()
    ca3_exc_local = np.asarray([i for i, g in enumerate(ca3_idx) if int(g) not in ca3_inh], dtype=np.int64)
    return dict(bridge=bridge, ca3_idx=ca3_idx, ca3_arr_host=ca3_arr_host, assemblies=assemblies,
                assembly_local=assembly_local, assemblies_local=assemblies_local, ca3_exc_local=ca3_exc_local,
                within_flat=within_flat, w_within=w_within, n_assy=n_assy)


# ----------------------------------------------------------------------------------------------------------------------
# _hard_silence: the committed dendritic-reset (mirrors _riii...run() lines 378-399). OU forced OFF during silence so a
# genuine bistable rest is a genuine SILENT down state (a self-sustaining net re-ignites during the settle).
# ----------------------------------------------------------------------------------------------------------------------
def _hard_silence(bridge, settle=30):
    from sim.backend import get_backend
    cp, _ = get_backend()
    _ou_was = bridge.core_config.enable_ou_process
    bridge.core_config.enable_ou_process = False
    if getattr(bridge, "cp_izh_c_reset", None) is not None:
        bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
    else:
        bridge.cp_membrane_potential_v[:] = -65.0
    bridge.cp_recovery_variable_u[:] = 0.0
    if getattr(bridge, "cp_firing_states", None) is not None:
        bridge.cp_firing_states[:] = False
    for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_e", "cp_conductance_g_i",
               "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
               "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise"):
        _arr = getattr(bridge, _a, None)
        if _arr is not None:
            _arr[:] = 0.0
    if getattr(bridge, "cp_v_apical", None) is not None:
        bridge.cp_v_apical[:] = cp.float32(getattr(bridge.core_config, "apical_E_rest", -65.0))
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(settle):
        bridge._run_one_simulation_step()
    bridge.core_config.enable_ou_process = _ou_was


def _configure_ou(bridge, sigma, seed):
    """Enable weak NON-SPECIFIC OU background (sigma pA) on ALL neurons for the rest phase, or disable it (sigma None).
    Re-inits cp_ou_current to mean 0. Seeds the backend RNG for reproducibility."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    cfg = bridge.core_config
    if sigma is None:
        cfg.enable_ou_process = False
        bridge.cp_ou_current = None
        return
    cp.random.seed(int(seed) * 100003 + 7)     # deterministic OU noise stream
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 0.0
    cfg.ou_std_current_pA = float(sigma)
    bridge._initialize_ou_process_state(cfg, int(cfg.num_neurons))


# ----------------------------------------------------------------------------------------------------------------------
# event detection: DISCRETE spontaneous reactivation events, scored for assembly-specificity + duty cycle.
# ----------------------------------------------------------------------------------------------------------------------
def _detect_events(F, assembly_local, seed, other_local=None, W=5, ev_floor=0.5, ev_k=4.0, n_rand=8, min_frac=0.30):
    """F: bool [T, n_ca3] rest-phase CA3 firing. assembly_local: local CA3 positions of the assembly UNDER TEST.
    Events = windows where the SMOOTHED total CA3 co-firing crosses a threshold (unbiased: detected on ALL CA3, then
    classified). Per event: member_frac (mean per-step assembly-active fraction), random-set frac (chance / permuted-
    assembly control), specificity = member_frac - random_frac; cross_frac = a co-stored OTHER assembly's participation
    (basin-selectivity vs a real competing memory). duty_cycle = fraction of rest steps in an event."""
    T, nca3 = F.shape
    A = np.asarray(assembly_local, dtype=np.int64); asize = int(len(A))
    O = np.asarray(other_local, dtype=np.int64) if other_local is not None and len(other_local) else None
    pop = F.sum(1).astype(float)                          # per-step # CA3 firing
    a_fire = F[:, A].sum(1).astype(float)                 # per-step # assembly firing
    o_fire = F[:, O].sum(1).astype(float) if O is not None else None
    exclude = set(A.tolist()) | (set(O.tolist()) if O is not None else set())
    nonmember = np.asarray([i for i in range(nca3) if i not in exclude], dtype=np.int64)
    rng = np.random.default_rng(seed * 991 + 5)
    rand_sets = [rng.choice(nonmember, asize, replace=False) for _ in range(n_rand)] if len(nonmember) >= asize else []

    S = np.convolve(pop, np.ones(W), mode="same")
    med = float(np.median(S)); mad = float(np.median(np.abs(S - med))) * 1.4826
    thr = max(med + ev_k * mad, ev_floor * asize)
    in_event = S > thr

    events = []
    t = 0
    while t < T:
        if in_event[t]:
            s = t
            while t < T and (in_event[t] or (t + W < T and in_event[min(T - 1, t + 1):min(T, t + 1 + W)].any())):
                t += 1
            events.append((s, min(t, T)))
        t += 1

    duty = float(in_event.mean())
    pop_rate = float(pop.mean() / max(1, nca3))
    assembly_rest_frac = float(a_fire.mean() / max(1, asize))

    mfs, rfs, cfs, specs, peaks = [], [], [], [], []
    for (s, e) in events:
        e = max(e, s + 1)
        mf = float(a_fire[s:e].mean() / asize)
        rf = float(np.mean([F[s:e][:, rs].sum(1).mean() / asize for rs in rand_sets])) if rand_sets else 0.0
        cf = float(o_fire[s:e].mean() / max(1, len(O))) if O is not None else 0.0
        mfs.append(mf); rfs.append(rf); cfs.append(cf); specs.append(mf - rf)
        peaks.append(float((a_fire[s:e] / asize).max()))
    n_events = len(events)
    n_specific = int(sum(1 for i in range(n_events) if mfs[i] >= min_frac and mfs[i] > 2.0 * (rfs[i] + 1e-6)))
    return dict(n_events=n_events, n_specific=n_specific,
                event_rate_per1k=float(1000.0 * n_events / max(1, T)),
                duty_cycle=duty, pop_rate=pop_rate, assembly_rest_frac=assembly_rest_frac,
                member_frac=float(np.mean(mfs)) if mfs else 0.0,
                random_frac=float(np.mean(rfs)) if rfs else 0.0,
                cross_frac=float(np.mean(cfs)) if cfs else 0.0,
                specificity=float(np.mean(specs)) if specs else 0.0,
                event_peak_frac=float(np.mean(peaks)) if peaks else 0.0,
                thr=thr, med_S=med, mad_S=mad, asize=asize)


def _rest_and_detect(prep, noise, rest_steps, seed, assembly_idx=0, W=5, ev_floor=0.5, ev_k=4.0, min_frac=0.30):
    """Freeze plasticity, hard-silence (verify dendritic reset), enable weak NON-SPECIFIC background, run REST, detect
    events. Verifies cp_connections.data is byte-unchanged across the rest phase (FROZEN plasticity mandatory).
    noise = ("none",) | ("ou", sigma) | ("poisson", rate, pA): OU = sigma-pA fluctuation on ALL neurons; poisson = a
    per-step suprathreshold pulse to a random `rate` fraction of the CA3 EXC cells (bypasses the FS-basket clamp)."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    assert bridge.core_config.enable_hebbian_learning is False

    _hard_silence(bridge)
    apical_max = None; n_latched = 0
    if getattr(bridge, "cp_v_apical", None) is not None:
        va = np.asarray(to_host(bridge.cp_v_apical))[prep["ca3_arr_host"][prep["assembly_local"]]]
        apical_max = float(np.max(va)); n_latched = int((va > float(GO_CFG["plateau_v_hold"])).sum())

    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()

    kind = noise[0]
    _configure_ou(bridge, (noise[1] if kind == "ou" else None), seed)   # OU (or disabled for none/poisson)
    # Poisson-noise setup (CA3-EXC-targeted; a deterministic host RNG stream). Each drawn cell is driven for `dur`
    # consecutive steps (a single-step pulse cannot fire an Izhikevich cell from rest -- the completion cue was 700pA x
    # 150 steps), so a coincidental within-window volley to several assembly cells can build a recurrent ignition.
    poisson = kind == "poisson"
    if poisson:
        p_rate, p_pa = float(noise[1]), float(noise[2])
        p_dur = int(noise[3]) if len(noise) > 3 else 5
        exc_glob = prep["ca3_arr_host"][prep["ca3_exc_local"]]
        exc_dev = cp.asarray(exc_glob, dtype=cp.int64)
        prng = np.random.default_rng(int(seed) * 100003 + 11)
        countdown = np.zeros(len(exc_glob), dtype=np.int32)

    ca3_arr_host = prep["ca3_arr_host"]
    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = 0.0     # OU (if enabled) added internally; no cue, no recall_drive
        if poisson:
            new = prng.random(len(exc_glob)) < p_rate           # newly-triggered CA3-exc cells this step (non-specific)
            countdown[new] = p_dur
            active = countdown > 0
            if active.any():
                bridge.cp_external_input_current[exc_dev[cp.asarray(np.nonzero(active)[0], dtype=cp.int64)]] = p_pa
            countdown[active] -= 1
        bridge._run_one_simulation_step()
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)
    bridge.core_config.enable_ou_process = False

    w_after = np.asarray(to_host(bridge.cp_connections.data))
    weights_frozen = bool(np.array_equal(w_before, w_after))

    al = prep.get("assemblies_local", [prep["assembly_local"]])
    a_test = al[assembly_idx] if assembly_idx < len(al) else prep["assembly_local"]
    a_other = al[(assembly_idx + 1) % len(al)] if len(al) > 1 else None
    ev = _detect_events(F, a_test, seed, other_local=a_other, W=W, ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    ev.update(noise=str(noise), apical_rest_max=apical_max, apical_n_latched=n_latched,
              weights_frozen=weights_frozen, rest_steps=rest_steps)
    return ev, F


def _shuffle_within_weights(prep, seed):
    """SHUFFLED within-assembly weights anti-cheat: permute the learned within-assembly recurrent edge weights AMONG
    themselves (same multiset, destroyed pairing) -> the basin structure is gone, the weight budget preserved."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    conn = prep["bridge"].cp_connections
    wf = prep["within_flat"]
    if len(wf) < 2:
        return 0
    d = np.asarray(to_host(conn.data))
    vals = d[wf].copy()
    np.random.default_rng(seed * 13 + 1).shuffle(vals)
    conn.data[cp.asarray(wf, dtype=cp.int64)] = cp.asarray(vals, dtype=conn.data.dtype)
    return int(len(wf))


# ----------------------------------------------------------------------------------------------------------------------
def _noise_label(ns):
    if ns[0] == "ou":
        return f"OU sig={ns[1]:g}"
    if ns[0] == "poisson":
        dur = ns[3] if len(ns) > 3 else 5
        return f"Poisson r={ns[1]:g} pA={ns[2]:g} dur={dur:g}"
    return "none"


def one_seed(seed, cfg, noise_specs, rest_steps, W, ev_floor, ev_k, min_frac, skip_poscontrol=False):
    t0 = time.time()
    out = {"seed": seed}

    # -- POSITIVE CONTROL (independent cross-check that the CLOSED config completes from a CUE at this scale, and that
    #    _prepare reproduces run()'s basin): reuse _riii...run() directly with the BTSP CFG (n_mem=1). --
    run_kwargs = dict(n_ca3=cfg["n_ca3"], ca3_density=cfg["ca3_density"], assembly_frac=cfg["assembly_frac"],
                      encode_drive=cfg["encode_drive"], no_sync=cfg["no_sync"], recall_k_thresh=cfg["recall_k_thresh"],
                      recall_drive=cfg["recall_drive"], recall_steps=cfg["recall_steps"], bistable=True,
                      nmda_recurrent=False, enable_ou=False, selective_inhib=cfg["selective_inhib"],
                      structural_sep=cfg["structural_sep"], plateau_self_regen=cfg["plateau_self_regen"],
                      apical_kir_g=cfg["apical_kir_g"], apical_gc=cfg["apical_gc"], apical_gc_read=cfg["apical_gc_read"],
                      # BTSP encode (== _gap4 BTSP_CFG) so the positive control validates the SAME completing store:
                      encode_btsp=cfg["encode_btsp"], encode_ca3w=cfg["encode_ca3w"],
                      encode_plateau_pA=cfg["encode_plateau_pA"], btsp_lr=cfg["btsp_lr"], hebb_max=cfg["hebb_max"],
                      train_events=cfg["train_events"], n_mem=cfg["n_mem"])
    if skip_poscontrol:
        out["poscontrol"] = None
        print(f"  [seed {seed}] POS-CTRL skipped ({time.time()-t0:.0f}s)", flush=True)
    else:
        pc = _riii_run(seed, **run_kwargs)
        out["poscontrol"] = {"held_cue": pc["held_cue"], "held_nocue": pc["held_nocue"], "held_perm": pc["held_perm"],
                             "rest_firing": pc["rest_firing"], "w_within": pc["w_within"], "go": pc["go"]}
        print(f"  [seed {seed}] POS-CTRL (cue completion): cue {pc['held_cue']:.3f} nocue {pc['held_nocue']:.3f} "
              f"perm {pc['held_perm']:.3f} rest {pc['rest_firing']:.3f} w_within {pc['w_within']:.1f} "
              f"-> {'store OK' if pc['go'] else 'store WEAK'} ({time.time()-t0:.0f}s)", flush=True)

    # -- GO condition: encoded bridge, sweep the noise level. Reuse ONE encoded bridge (weights frozen -> safe). --
    prep = _prepare(seed, cfg, do_encode=True)
    out["w_within_prepare"] = prep["w_within"]
    go_runs = {}
    best_ns, best = None, None
    for ns in noise_specs:
        ev, F = _rest_and_detect(prep, ns, rest_steps, seed, W=W, ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
        go_runs[_noise_label(ns)] = ev
        print(f"  [seed {seed}] GO {_noise_label(ns):>22}: events={ev['n_events']:>3} specific={ev['n_specific']:>3} "
              f"rate/1k={ev['event_rate_per1k']:.2f} duty={ev['duty_cycle']:.3f} memb={ev['member_frac']:.3f} "
              f"rand={ev['random_frac']:.3f} cross={ev['cross_frac']:.3f} spec={ev['specificity']:+.3f} "
              f"peak={ev['event_peak_frac']:.3f} pop={ev['pop_rate']:.4f} frozen={ev['weights_frozen']} "
              f"apical_max={ev['apical_rest_max']} latched={ev['apical_n_latched']} ({time.time()-t0:.0f}s)", flush=True)
        # "best" = most specific events, tie-broken by higher specificity
        score = (ev["n_specific"], ev["specificity"])
        if best is None or score > best:
            best, best_ns = score, ns
    out["go_runs"] = go_runs
    out["best_noise"] = _noise_label(best_ns)
    go = go_runs[_noise_label(best_ns)]

    # -- ANTI-CHEATS at best noise --
    # NO-NOISE on the SAME encoded bridge -> must be SILENT (acid test for the self-sustaining artifact)
    nn, _ = _rest_and_detect(prep, ("none",), rest_steps, seed, W=W, ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    out["nonoise"] = nn
    print(f"  [seed {seed}] NO-NOISE (acid): events={nn['n_events']} specific={nn['n_specific']} "
          f"duty={nn['duty_cycle']:.4f} memb={nn['member_frac']:.3f} pop={nn['pop_rate']:.5f} "
          f"apical_max={nn['apical_rest_max']} latched={nn['apical_n_latched']} ({time.time()-t0:.0f}s)", flush=True)

    # NO-ENCODING (fresh bridge, store skipped, same noise) -> no assembly-specific events
    prep_ne = _prepare(seed, cfg, do_encode=False)
    ne, _ = _rest_and_detect(prep_ne, best_ns, rest_steps, seed, W=W, ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    out["noencode"] = ne
    print(f"  [seed {seed}] NO-ENCODE {_noise_label(best_ns)}: events={ne['n_events']} specific={ne['n_specific']} "
          f"memb={ne['member_frac']:.3f} rand={ne['random_frac']:.3f} spec={ne['specificity']:+.3f} "
          f"pop={ne['pop_rate']:.4f} w_within(prepare)={prep_ne['w_within']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # SHUFFLED within-assembly weights (fresh encoded bridge, scramble, same noise) -> no assembly-specific events
    prep_sh = _prepare(seed, cfg, do_encode=True)
    n_shuf = _shuffle_within_weights(prep_sh, seed)
    sh, _ = _rest_and_detect(prep_sh, best_ns, rest_steps, seed, W=W, ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    out["shuffled"] = sh; out["shuffled"]["n_within_shuffled"] = n_shuf
    print(f"  [seed {seed}] SHUFFLED-W {_noise_label(best_ns)}: shuffled {n_shuf} edges; events={sh['n_events']} "
          f"specific={sh['n_specific']} memb={sh['member_frac']:.3f} rand={sh['random_frac']:.3f} "
          f"spec={sh['specificity']:+.3f} pop={sh['pop_rate']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # NO-STRUCTURE (learned weights INTACT, but structural_sep=0 + selective_inhib=False) -> the DECISIVE isolation of the
    # LEARNED-WEIGHT contribution. The SHUFFLED-W caveat: structural_sep + selective_inhib SURVIVE a weight-shuffle, so the
    # shuffle only PARTIALLY collapses (structure carries the residual). This control removes the STRUCTURE while KEEPING
    # the learned within-assembly weights: if events stay SELECTIVE (memb >> random), the LEARNED attractor carries the
    # selectivity (clean GO on the learned-weight question); if memb -> random, the structural wiring was carrying it.
    cfg_nostruct = {**cfg, "structural_sep": 0, "selective_inhib": False}
    prep_nostruct = _prepare(seed, cfg_nostruct, do_encode=True)
    nstr, _ = _rest_and_detect(prep_nostruct, best_ns, rest_steps, seed, W=W, ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    out["nostructure"] = nstr
    print(f"  [seed {seed}] NO-STRUCT (learned wts, no sep/sel-inhib) {_noise_label(best_ns)}: events={nstr['n_events']} "
          f"specific={nstr['n_specific']} memb={nstr['member_frac']:.3f} rand={nstr['random_frac']:.3f} "
          f"spec={nstr['specificity']:+.3f} pop={nstr['pop_rate']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # -- PER-SEED VERDICT --
    # GO: discrete assembly-specific spontaneous events + net rests silent + all anti-cheats retire their confound.
    acid_noise_off = (nn["n_specific"] == 0 and nn["assembly_rest_frac"] < 0.05)   # NO-NOISE -> silent
    frozen_ok = bool(go["weights_frozen"] and nn["weights_frozen"])
    dendrite_reset_ok = (go["apical_rest_max"] is None or go["apical_rest_max"] <= float(GO_CFG["plateau_v_hold"]) + 1e-3)
    specific_events = (go["n_specific"] >= 1 and go["member_frac"] >= min_frac
                       and go["member_frac"] > 2.0 * (go["random_frac"] + 1e-6))
    discrete = (go["duty_cycle"] <= 0.40)                                          # NOT a continuous ON state
    noencode_retired = (ne["n_specific"] == 0 or ne["member_frac"] < 0.5 * go["member_frac"])
    shuffle_retired = (sh["n_specific"] == 0 or sh["member_frac"] < 0.5 * go["member_frac"])
    permuted_retired = (go["member_frac"] > 2.0 * (go["random_frac"] + 1e-6))
    # DIAGNOSTIC (does the LEARNED attractor carry the selectivity, or the structural wiring?): NO-STRUCT keeps the learned
    # weights but removes sep/sel-inhib -> if still selective, the learned attractor carries it. A CLEAN RANK-1 GO wants
    # this True; if False, the reactivation is real+spontaneous but its selectivity is structural, not learned (weaker).
    learned_weight_carries = bool(nstr["n_specific"] >= 1 and nstr["member_frac"] > 2.0 * (nstr["random_frac"] + 1e-6))
    seed_go = bool(specific_events and discrete and acid_noise_off and frozen_ok and dendrite_reset_ok
                   and noencode_retired and shuffle_retired and permuted_retired)
    out["checks"] = dict(specific_events=specific_events, discrete=discrete, acid_noise_off=acid_noise_off,
                         frozen_ok=frozen_ok, dendrite_reset_ok=dendrite_reset_ok, noencode_retired=noencode_retired,
                         shuffle_retired=shuffle_retired, permuted_retired=permuted_retired,
                         learned_weight_carries=learned_weight_carries)
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']}  best_noise={_noise_label(best_ns)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000, help="CPU smoke ok at 2000 (BTSP encode is cheap); the store only completes at 2000")
    ap.add_argument("--n-mem", type=int, default=2, help="stored assemblies (2 = the validated _gap4 basin; reactivation tested on assembly 0)")
    ap.add_argument("--noise", choices=["ou", "poisson"], default="poisson",
                    help="ou = sigma-pA fluctuation on ALL neurons; poisson = CA3-EXC-targeted pulse volley (bypasses the FS-basket clamp)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[100.0, 200.0, 400.0], help="OU sigma sweep (pA), --noise ou")
    ap.add_argument("--poisson-rate", type=float, default=0.01, help="fraction of CA3-EXC cells NEWLY triggered per step, --noise poisson")
    ap.add_argument("--poisson-pa", type=float, nargs="+", default=[500.0, 1000.0, 2000.0], help="per-pulse pA sweep, --noise poisson")
    ap.add_argument("--poisson-dur", type=int, default=5, help="pulse duration (steps) each triggered CA3-EXC cell is driven, --noise poisson")
    ap.add_argument("--rest-steps", type=int, default=1500)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.5, help="event window-sum floor as a fraction of assembly size")
    ap.add_argument("--ev-k", type=float, default=4.0, help="event threshold = med + ev_k*MAD (robust)")
    ap.add_argument("--min-frac", type=float, default=0.30, help="assembly-active fraction for a 'specific' event")
    ap.add_argument("--skip-poscontrol", action="store_true", help="skip the slow run() cue-completion cross-check")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cfg = dict(GO_CFG); cfg["n_ca3"] = int(a.n_ca3); cfg["n_mem"] = int(a.n_mem)
    if a.noise == "ou":
        noise_specs = [("ou", s) for s in a.sigmas]
    else:
        noise_specs = [("poisson", a.poisson_rate, p, a.poisson_dur) for p in a.poisson_pa]
    print(f"[gap5-spont] n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']} assy~{max(6,int(cfg['assembly_frac']*cfg['n_ca3']))} "
          f"noise={a.noise} levels={[_noise_label(n) for n in noise_specs]} rest_steps={a.rest_steps} "
          f"seeds={a.seeds} backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, cfg, noise_specs, a.rest_steps, a.window, a.ev_floor, a.ev_k, a.min_frac,
                                skip_poscontrol=a.skip_poscontrol))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        go = n_go >= max(1, (len(per) + 1) // 2)      # >=50% for the smoke; the full-run gate is >=5/6 on GPU
        mg = [p["go_runs"][p["best_noise"]] for p in per]
        mm = float(np.mean([g["member_frac"] for g in mg])); mr = float(np.mean([g["random_frac"] for g in mg]))
        mc = float(np.mean([g["cross_frac"] for g in mg]))
        md = float(np.mean([g["duty_cycle"] for g in mg])); mnn = float(np.mean([p["nonoise"]["assembly_rest_frac"] for p in per]))
        verdict = (f"{'GO' if go else 'PARTIAL/NEGATIVE'} {n_go}/{len(per)} -- a stored bistable CA3 assembly "
                   f"{'SPONTANEOUSLY + BASIN-SELECTIVELY reactivates' if go else 'did NOT cleanly spontaneously reactivate'} "
                   f"under weak non-specific background: member_frac {mm:.3f} vs random {mr:.3f} / cross-assembly {mc:.3f}, "
                   f"duty {md:.3f}; NO-NOISE assembly rest {mnn:.4f} (acid: must be ~0). "
                   f"{'=> the FIRST non-artifact spontaneous-reactivation piece of the SWR loop is de-risked; run the 6-seed GPU confirm.' if go else 'Per THE LAW: tune the noise (sigma/poisson pA+rate) / apical_gc_read / apical_kir_g / recall_k_thresh; not a stop.'}")
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = {"probe": "gap5_spontaneous_reactivation", "GO": go, "n_go": n_go, "seeds": a.seeds,
               "n_ca3": cfg["n_ca3"], "n_mem": cfg["n_mem"], "noise": a.noise,
               "noise_levels": [_noise_label(n) for n in noise_specs], "rest_steps": a.rest_steps,
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110 + f"\n[gap5-spont] VERDICT: {verdict}\n[gap5-spont] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
