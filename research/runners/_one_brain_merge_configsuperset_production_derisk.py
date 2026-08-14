"""CONFIG-SUPERSET PRODUCTION merge de-risk: surprise (GABA_B) + Wong-Wang comprehension (NMDA) on ONE bridge.

The named 'larger next step' of 2026-08-13-one-brain-merge-2organ-BOUNDARY.md: two DIFFERENT-builder production
organs on ONE spiking pool via a config SUPERSET (enable_gabab AND enable_nmda coexist; per-region NMDA mask ->
only sel_agent/sel_patient carry NMDA). Confronts the ONE genuine conflict rung-2 mapped: comprehension's GRADED
well/ill AUC needs dt=0.5, the surprise organ natively runs dt=1.0. We SWEEP the (dt, homeostasis) 2x2 and ask:
is there a shared operating point where BOTH organs' production ANSWERS are preserved?

  * comprehension: well/ill AUC>=0.80 on the merged pool  AND  `comprehended` bool byte-id merged-vs-standalone
    (co-residence adds NO coupling: role read with the surprise->role cross intact == with it lesioned)  AND
    `comprehended` bool preserved vs the NATIVE production comprehension read (dt=0.5).
  * surprise: `surprised` bool byte-id merged-vs-(cross-lesioned)  AND  `surprised` bool preserved vs the NATIVE
    production surprise read (dt=1.0), across the surprise panel.

GO -> the pair can share one pool answer-preservingly (report the reconcilable cell). NO-GO -> the honest negative
is the per-region-dt engine boundary (the fused integrator steps ALL neurons at one dt), the mapped sim/ step.

REUSE-BY-IMPORT; NO sim/ edit. Two ADDITIVE default-preserving edits only:
  (1) `_spiking_comprehension_monitor_derisk._build_comp` gains dt_ms/homeostasis/per_region_thresh kwargs.
  (2) `_install_learned_cue_role` (below) copies the comprehension organ's FROZEN learned cue->role weights onto
      the merged role pathway BY NAME (mirrors `_install_block_diagonal`), because the GRADED AUC is load-bearing
      on the learned weights (lesion -> AUC 0.500); `build_merged_diffbuilder` alone installs only init weights.

SIM_BACKEND=numpy for the bit-exact CPU verify.
Run:
    SIM_BACKEND=numpy OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      python -m research.runners._one_brain_merge_configsuperset_production_derisk \
        --seeds 42,43,44,100,101,102 --cells 0.5:True,0.5:False,1.0:True,1.0:False \
        --out research/findings/raw/_one_brain_merge_configsuperset_6seed.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners._one_brain_merge_Norgan_derisk import (
    build_merged_diffbuilder, _role_rates, _surprise_current, _global_config_conflict_map,
)
from research.runners._one_brain_merge_2organ_derisk import (
    _idx_map, _arr_hash, _INIT_PER_NEURON_ARRAYS,
)
from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit, train_expectation, measure_conditions, _idx, _host, _hard_reset,
)
from research.runners._spiking_comprehension_monitor_derisk import (
    _build_comp, _evs_for, _agent_evidence_from_spikes, SEMANTIC_CUES, build_battery, roc_auc,
)
from research.runners._phaseB_multicue_competition_spiking_derisk import CUES, ROLES
from tools.verdict import Verdict

READ_STEPS = 60
CROSS_WEIGHT = 40.0
# native-production build params for the surprise organ (verbatim from SurpriseProductionOrgan._build_one)
SURP_N_TRAINED, SURP_N_NOVEL, SURP_BLK, SURP_CUE_BLK, SURP_CUE_W, SURP_NREPS = 8, 4, 24, 24, 0.8, 22

# The two co-resident state-restore sets the read-isolation guard snapshots/restores (generalizes
# MergedSubstrate._PER_NEURON_STATE to arbitrary index sets).
_PER_NEURON_STATE = (
    "cp_membrane_potential_v", "cp_recovery_variable_u",
    "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
    "cp_firing_states", "cp_prev_firing_states", "cp_refractory_timers", "cp_refractory",
    "cp_neuron_firing_thresholds", "cp_neuron_activity_ema", "cp_external_input_current",
)

EXP_REGIONS = ("cue_S", "patient_expected_S", "patient_asserted_S", "surprise_S")
ROLE_REGIONS = (["sel_agent", "sel_patient", "sel_FS_agent", "sel_FS_patient"]
                + [f"cue_{c}_{s}" for c in CUES for s in ("pos", "neg")])


def _forbidden(*_a, **_k):
    raise AssertionError("brain-based violated: host _semantic_contrast dot-product called during a read")


def _region_slice(bridge, names):
    """Boolean mask + concatenated int index over the given region names on the merged bridge."""
    idxs = [np.asarray(_idx(bridge, n), dtype=np.int64) for n in names]
    flat = np.concatenate(idxs)
    n = int(bridge.cp_membrane_potential_v.shape[0])
    mask = np.zeros(n, dtype=bool)
    mask[flat] = True
    return mask, flat


class _read_isolation:
    """Generalized `MergedSubstrate.read_isolation`: snapshot the FULL per-neuron state, run the read, then
    RESTORE only the CO-RESIDENT slice (`restore_mask`) -- so a read of the active organ leaves NO footprint on
    the co-resident's persistent neural state (thresholds / activity-EMA drift + spontaneous-firing carryover)."""

    def __init__(self, bridge, restore_mask, xp):
        self.b = bridge
        self.keep = xp.asarray(~restore_mask)   # True where we KEEP the (active organ's) evolution
        self.xp = xp

    def __enter__(self):
        self._snaps = []
        for name in _PER_NEURON_STATE:
            arr = getattr(self.b, name, None)
            self._snaps.append(None if arr is None else arr.copy())
        return self

    def __exit__(self, *exc):
        for name, snap in zip(_PER_NEURON_STATE, self._snaps):
            if snap is None:
                continue
            cur = getattr(self.b, name)
            setattr(self.b, name, self.xp.where(self.keep, cur, snap))
        return False


# ── (edit 2) install the comprehension organ's FROZEN learned cue->role weights onto the merged role pathway ──
def _install_learned_cue_role(merged, comp_src):
    """Copy `comp_src`'s frozen learned cue->role projection weights onto the merged bridge's role pathway BY
    NAME (mirrors `_install_block_diagonal`). `comp_src` installed its cue validities via `set_cue_weight` (a
    single scalar per cue), so we read each cue's scalar from its OWN synapses (`cue_weights()`) and write the
    same scalar onto the merged bridge's same-named `cue_{c}_{sgn} -> sel_{role}` edges. Then FREEZE the merged
    role plasticity gates so those weights never change (anti-cheat: frozen learned synapses, no host gradient).
    Returns the installed {cue: weight}."""
    w_by_cue = comp_src.cue_weights()
    installed = {}
    for c in CUES:
        wv = float(w_by_cue[c])
        for sgn, role in (("pos", "agent"), ("neg", "patient")):
            pre = np.asarray(_idx(merged, f"cue_{c}_{sgn}"), dtype=np.int64)
            post = np.asarray(_idx(merged, f"sel_{role}"), dtype=np.int64)
            p = np.repeat(pre, post.size).astype(np.int64)
            q = np.tile(post, pre.size).astype(np.int64)
            w = np.full(p.size, np.float32(wv), np.float32)
            merged.set_pathway_weights(f"install_{c}_{sgn}", pre_indices=p, post_indices=q,
                                       weights=w, add_missing=False)
        installed[c] = wv
    # FREEZE: role cue->role edges must never learn on the merged bridge (they are the organ's frozen weights).
    for c in CUES:
        merged.set_plasticity_gate(f"cue_{c}", 0.0)
    return installed


def _repoint_comp(comp_template, bridge):
    """A comprehension read VIEW: a shallow copy of `comp_template` whose sel/cue indices + bridge are re-resolved
    against `bridge` (the merged pool), so `SpikingRoleCompetition._noun_role_rates` reads the merged role slice.
    Installs the `_semantic_contrast` tripwire (asserts the host dot-product is never called)."""
    v = copy.copy(comp_template)
    v.bridge = bridge
    rm = bridge.region_manager
    v._sel_idx = {r: np.asarray(rm.indices(f"sel_{r}"), dtype=np.int64) for r in ROLES}
    v._cue_idx = {(c, sgn): np.asarray(rm.indices(f"cue_{c}_{sgn}"), dtype=np.int64)
                  for c in CUES for sgn in ("pos", "neg")}
    v._n = int(bridge.cp_membrane_potential_v.shape[0])
    v._semantic_contrast = _forbidden          # tripwire: reads are cp_firing_states only, never the host formula
    return v


def _margin(comp_view, n0, v, n1, read_steps=READ_STEPS):
    """|agentEv_0 - agentEv_1| off the sel pools for transitive (n0, v, n1), SEMANTIC (content) cues only.
    Each `_agent_evidence_from_spikes` reads `cp_firing_states` off sel_agent/sel_patient (brain-based)."""
    evs = _evs_for(n0, v, n1)
    a0 = float(_agent_evidence_from_spikes(comp_view, evs[0], SEMANTIC_CUES, read_steps))
    a1 = float(_agent_evidence_from_spikes(comp_view, evs[1], SEMANTIC_CUES, read_steps))
    return abs(a0 - a1)


def _calibrate_threshold(well, ill):
    """Production-organ threshold: place it in the well/ill GAP, biased to the ill side (midway between the well
    FLOOR and the ill CEILING when they separate; else the class-mean midpoint). Mirrors ComprehensionProductionOrgan."""
    if not well or not ill:
        return 0.0
    min_well, max_ill = float(np.min(well)), float(np.max(ill))
    mean_well, mean_ill = float(np.mean(well)), float(np.mean(ill))
    return 0.5 * (min_well + max_ill) if min_well > max_ill else 0.5 * (mean_well + mean_ill)


def _comp_margins(comp_view, battery, bridge=None, iso_mask=None, xp=None):
    """Battery margins on `comp_view`'s bridge. If `iso_mask` is given, each read is wrapped in the read-isolation
    guard (restore the co-resident slice) BEFORE the next read so co-resident drift does not accumulate. Also
    hard-resets the bridge to its resting snapshot before the loop so the first read starts silent."""
    if bridge is not None and getattr(bridge, "_rest_v", None) is not None:
        _hard_reset(bridge)                     # start from the byte-id resting snapshot (co-resident silent)
    margins = {}
    for (_lab, _tag, n0, v, n1) in battery:
        if iso_mask is not None:
            with _read_isolation(bridge, iso_mask, xp):
                m = _margin(comp_view, n0, v, n1)
        else:
            m = _margin(comp_view, n0, v, n1)
        margins[(n0, v, n1)] = m
    return margins


def _comprehended(margins, battery):
    """Per-item comprehended bool + AUC, using the production threshold calibrated on THIS build's battery."""
    well = [margins[(n0, v, n1)] for (lab, _t, n0, v, n1) in battery if lab == 1]
    ill = [margins[(n0, v, n1)] for (lab, _t, n0, v, n1) in battery if lab == 0]
    thr = _calibrate_threshold(well, ill)
    scores = [margins[(n0, v, n1)] for (_l, _t, n0, v, n1) in battery]
    labels = [lab for (lab, *_r) in battery]
    auc = roc_auc(scores, labels)
    comp_bool = {(n0, v, n1): bool(margins[(n0, v, n1)] >= thr) for (_l, _t, n0, v, n1) in battery}
    return comp_bool, float(thr), float(auc), well, ill


# ── surprise reads on the merged bridge (Norgan-style: measure_conditions confirm/contradict/novel per fact) ──
def _surprise_bools(bridge, cfg, idx_map, meta, xp, iso_mask=None):
    """Per-fact-per-condition `surprised` bool over the surprise panel via `measure_conditions`, thresholded
    exactly as SurpriseProductionOrgan (midpoint of confirm and the weaker violation). If `iso_mask` given, run
    the whole measurement inside the read-isolation guard (restore the co-resident role slice)."""
    guard = _read_isolation(bridge, iso_mask, xp) if iso_mask is not None else None
    if guard is not None:
        with guard:
            res = measure_conditions(bridge, cfg, idx_map, meta, xp)
    else:
        res = measure_conditions(bridge, cfg, idx_map, meta, xp)
    conf, contra, nov = res["confirm_hz"], res["contradict_hz"], res["novel_hz"]
    thr = 0.5 * (conf + min(contra, nov))
    bools = ([m >= thr for m in res["confirm_per"]]      # expect all False (not surprised)
             + [m >= thr for m in res["contradict_per"]]  # expect all True
             + [m >= thr for m in res["novel_per"]])      # expect all True
    return [bool(b) for b in bools], dict(confirm_hz=float(conf), contradict_hz=float(contra),
                                          novel_hz=float(nov), threshold=float(thr),
                                          separation=float(contra / max(conf, 1e-6)))


# ── cached native baselines (dt/homeo-independent -> built ONCE per seed) ──
def _native_comp(seed, battery):
    """SHIPPED production comprehension answer: `_build_comp(seed)` at the pure production defaults
    (dt=0.5, homeostasis OFF, per-region-thresh OFF). Its per-item `comprehended` bool + AUC."""
    comp = _build_comp(seed)                    # production defaults
    comp._semantic_contrast = _forbidden
    margins = {}
    for (_lab, _t, n0, v, n1) in battery:
        margins[(n0, v, n1)] = _margin(comp, n0, v, n1)
    return _comprehended(margins, battery)


def _native_surprise(seed, xp):
    """SHIPPED production surprise answer: the standalone expectation circuit at native dt=1.0 (the params
    SurpriseProductionOrgan builds), trained + measured. Per-fact-per-condition `surprised` bool."""
    bridge, cfg, meta = build_expectation_circuit(seed, n_trained=SURP_N_TRAINED, n_novel=SURP_N_NOVEL,
                                                  blk=SURP_BLK, cue_blk=SURP_CUE_BLK,
                                                  cue_to_expected_weight=SURP_CUE_W)
    bridge._blk = meta["blk"]
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in ("cue", "patient_expected", "patient_asserted", "surprise")}
    train_expectation(bridge, cfg, idx_map, meta, xp, n_reps=SURP_NREPS)
    cfg.enable_hebbian_learning = False
    return _surprise_bools(bridge, cfg, idx_map, meta, xp)


def _build_and_prep(seed, dt_ms, homeo, comp_src, cross_weight):
    """A config-superset bridge (surprise_S + Wong-Wang role) at (dt_ms, homeo) with the comprehension organ's
    frozen learned cue->role weights installed. `cross_weight`=0 -> the surprise->role cross is present but SILENT
    (the DECOUPLED baseline: byte-id everything except the one cross edge, no in-place mutation)."""
    merged, cfg_m, meta = build_merged_diffbuilder(seed, dt_ms=dt_ms, homeostasis=homeo,
                                                   per_region_thresh=True, cross_weight=cross_weight)
    _install_learned_cue_role(merged, comp_src)     # graded AUC is load-bearing on the learned weights (frozen)
    merged._blk = meta["blk"]
    return merged, cfg_m, meta


def run_cell(seed, dt_ms, homeo, *, native_cache, verbose=True):
    from sim.backend import get_backend
    from tools.lab import attributable_to
    xp, _ = get_backend()
    battery = build_battery(seed, n_per_cond=6)

    # ---- weight source + read template: the comprehension organ at THIS operating point ----
    comp_src = _build_comp(seed, dt_ms=dt_ms, homeostasis=homeo, per_region_thresh=True)

    # ---- the merged config-superset bridge + a fresh DECOUPLED twin (cross=0). No in-place CSR mutation:
    #      the _install_block_diagonal_full toggle does NOT round-trip cleanly, so each comparison is a
    #      FRESH build differing only in the cross weight (the one coupling channel surprise_S->sel_agent). ----
    merged, cfg_m, meta = _build_and_prep(seed, dt_ms, homeo, comp_src, CROSS_WEIGHT)
    dec, cfg_d, _ = _build_and_prep(seed, dt_ms, homeo, comp_src, 0.0)     # decoupled twin
    idx_S = _idx_map(merged, "_S", xp)
    idx_S_d = _idx_map(dec, "_S", xp)

    # ── (1) ONE POOL: every surprise + role region index lives in the one cp_ array ──
    n_all = int(merged.cp_membrane_potential_v.shape[0])
    all_names = list(EXP_REGIONS) + list(ROLE_REGIONS)
    sizes = {nm: int(_idx(merged, nm).size) for nm in all_names}
    n_surp = sum(sizes[n] for n in EXP_REGIONS)
    n_role = sum(sizes[n] for n in ROLE_REGIONS)
    one_pool = bool(n_all >= n_surp + n_role) and all(int(_idx(merged, nm).max()) < n_all for nm in all_names)

    # ── (2) DETERMINISM: build twice -> hash membrane + thresholds ──
    b2, _, _ = build_merged_diffbuilder(seed, dt_ms=dt_ms, homeostasis=homeo,
                                        per_region_thresh=True, cross_weight=CROSS_WEIGHT)
    determ = bool(_arr_hash(merged.cp_membrane_potential_v) == _arr_hash(b2.cp_membrane_potential_v)
                  and _arr_hash(merged.cp_neuron_firing_thresholds) == _arr_hash(b2.cp_neuron_firing_thresholds))

    # ── (3) GABA_B + NMDA COEXIST (the config superset), per-region NMDA mask restricts to sel_agent/sel_patient ──
    n_nmda = int(np.asarray(_host(merged.cp_nmda_neuron_mask)).sum()) if getattr(merged, "cp_nmda_neuron_mask", None) is not None else -1
    n_sel = sizes["sel_agent"] + sizes["sel_patient"]
    gabab_nmda_coexist = bool(getattr(merged, "cp_conductance_g_gabab", None) is not None
                              and getattr(merged, "cp_conductance_g_nmda", None) is not None
                              and cfg_m.enable_gabab and cfg_m.enable_nmda and n_nmda == n_sel)

    # ── brain-based anti-cheats (asserted, not just claimed) ──
    assert float(cfg_m.current_reward_signal) == 0.0 and float(cfg_m.reward_baseline) == 0.0, "reward leak"
    assert not cfg_m.enable_stdp and not cfg_m.enable_reward_modulation, "stdp/reward-mod must be off"

    role_mask, _ = _region_slice(merged, ROLE_REGIONS)
    surp_mask, _ = _region_slice(merged, EXP_REGIONS)

    # ── surprise INIT byte-id: the merged surprise SLICE == a true standalone expectation organ (dt-independent
    #    per-neuron init; the per-region threshold fix). Measured AT BUILD (before any read evolves v/u), so it
    #    compares the initial per-neuron arrays, not post-simulation membrane states. ──
    solo, _, _ = build_expectation_circuit(seed, n_trained=meta["n_trained"], n_novel=meta["n_novel"],
                                           blk=meta["blk"], cue_blk=meta["cue_blk"], region_suffix="_S",
                                           per_region_thresh=True)
    surp_init_err = 0.0
    for r in EXP_REGIONS:
        mi = _idx(merged, r); si = _idx(solo, r)
        for nm in _INIT_PER_NEURON_ARRAYS:
            am = np.asarray(_host(getattr(merged, nm)))[mi]
            aso = np.asarray(_host(getattr(solo, nm)))[si]
            surp_init_err = max(surp_init_err, float(np.abs(am - aso).max()))
    surp_init_byte_id = bool(surp_init_err <= 1e-6)

    # ── (4) COMPREHENSION on the merged pool (surprise UNtrained -> silent co-resident): AUC + comprehended bool.
    #        Read-isolation ON (production read path). The DECOUPLED twin (cross=0) is the standalone-with-flags
    #        baseline: co-residence is clean iff the comprehended bools are identical merged-vs-decoupled. ──
    comp_view = _repoint_comp(comp_src, merged)
    comp_view_d = _repoint_comp(comp_src, dec)
    m_merged = _comp_margins(comp_view, battery, bridge=merged, iso_mask=surp_mask, xp=xp)
    m_dec = _comp_margins(comp_view_d, battery, bridge=dec, iso_mask=None, xp=xp)   # decoupled: no co-resident to isolate
    comp_bool_m, thr_m, auc_m, well_m, ill_m = _comprehended(m_merged, battery)
    comp_bool_d, _thr_d, _auc_d, _wd, _id = _comprehended(m_dec, battery)
    comp_byte_id_err = max(abs(m_merged[k] - m_dec[k]) for k in m_merged)
    comp_byte_id = bool(all(comp_bool_m[k] == comp_bool_d[k] for k in comp_bool_m))   # comprehended bool identical

    # ── (4c) comprehension answer-preserved vs the SHIPPED NATIVE read (dt=0.5, homeo OFF, production defaults) ──
    comp_bool_native, thr_nat, auc_nat, _wn, _in = native_cache["comp"]
    comp_match = sum(comp_bool_m[k] == comp_bool_native[k] for k in comp_bool_m)
    comp_answer_preserved = bool(comp_match == len(comp_bool_m))

    # ── READ-ISOLATION verify (comp): a comprehension read leaves the surprise slice bit-for-bit unchanged ──
    _hard_reset(merged)
    ref = {nm: np.asarray(_host(getattr(merged, nm))).copy() for nm in _PER_NEURON_STATE
           if getattr(merged, nm, None) is not None}
    with _read_isolation(merged, surp_mask, xp):
        _margin(comp_view, *battery[0][2:])
    read_iso_comp = all(np.array_equal(np.asarray(_host(getattr(merged, nm)))[surp_mask], ref[nm][surp_mask])
                        for nm in ref)

    # ── (5) SURPRISE on the merged pool AND on the decoupled twin: train, read confirm/contradict/novel ──
    for br, cf, ix in ((merged, cfg_m, idx_S), (dec, cfg_d, idx_S_d)):
        _hard_reset(br)
        cf.enable_hebbian_learning = True
        train_expectation(br, cf, ix, meta, xp, n_reps=SURP_NREPS)
        cf.enable_hebbian_learning = False
    surp_bool_m, surp_stats_m = _surprise_bools(merged, cfg_m, idx_S, meta, xp, iso_mask=role_mask)
    surp_bool_d, surp_stats_d = _surprise_bools(dec, cfg_d, idx_S_d, meta, xp)
    surprise_functional = bool(surp_stats_m["separation"] >= 2.0 and surp_stats_m["contradict_hz"] >= 5.0)

    # ── (5b) surprise byte-id: role cannot perturb surprise (no sel->surprise path). Merged(cross=40) vs
    #        decoupled(cross=0) surprised bools must match (upstream organ; confirms no back-coupling). ──
    surp_byte_id_err = max(abs(surp_stats_m[k] - surp_stats_d[k]) for k in ("confirm_hz", "contradict_hz", "novel_hz"))
    surp_byte_id = bool(surp_bool_m == surp_bool_d)

    # ── (5c) surprise answer-preserved vs the SHIPPED NATIVE read (dt=1.0, homeo ON, production defaults) ──
    surp_bool_native, surp_stats_native = native_cache["surp"]
    surp_match = sum(a == b for a, b in zip(surp_bool_m, surp_bool_native))
    surp_answer_preserved = bool(surp_match == len(surp_bool_native))

    # ── READ-ISOLATION verify (surp): a surprise read, under the guard, leaves the role slice bit-for-bit unchanged ──
    _hard_reset(merged)
    ref2 = {nm: np.asarray(_host(getattr(merged, nm))).copy() for nm in _PER_NEURON_STATE
            if getattr(merged, nm, None) is not None}
    with _read_isolation(merged, role_mask, xp):
        measure_conditions(merged, cfg_m, idx_S, meta, xp)
    read_iso_surp = all(np.array_equal(np.asarray(_host(getattr(merged, nm)))[role_mask], ref2[nm][role_mask])
                        for nm in ref2)

    # ── (6) CROSS synapse LOAD-BEARING: surprise_S (contradict vs confirm) biases sel_agent via the cross.
    #        Intact = merged(cross=40); lesioned = the DECOUPLED twin(cross=0). Fresh builds, no CSR toggle. ──
    def _agent_bias(br, mt):
        def _one(cond):
            cur = _surprise_current(br, mt, xp, condition=cond)
            cur = cur.copy(); cur[_idx(br, "cue_position_neg")] = np.float32(3500.0)
            return _role_rates(br, None, None, xp, extra_current=xp.asarray(cur))["agent"]
        return _one("contradict") - _one("confirm")

    cross_intact = _agent_bias(merged, meta)
    cross_lesion = _agent_bias(dec, meta)
    cross_frac = attributable_to("sel_agent bias @ surprise_S->sel_agent cross", cross_intact, cross_lesion)
    cross_lb = bool(abs(cross_intact) >= 1.0 and abs(cross_intact) >= 5.0 * max(abs(cross_lesion), 1e-6)
                    and (cross_frac is None or cross_frac >= 0.8))

    # ── VERDICT (merge-specific axes must be exact; the organ-variance axes carry the 6-seed slack) ──
    V = Verdict(f"config-superset merge seed={seed} dt={dt_ms} homeo={homeo}")
    V.require("one shared pool", one_pool, expect=True)
    V.require("determinism (build-twice byte-id)", determ, expect=True)
    V.require("GABA_B + NMDA coexist (config superset)", gabab_nmda_coexist, expect=True)
    V.require("comprehension byte-id merged-vs-decoupled", comp_byte_id, expect=True)
    V.require("comprehension AUC >= 0.80", bool(auc_m >= 0.80), expect=True)
    V.require("comprehension answer-preserved vs native(dt=0.5)", comp_answer_preserved, expect=True)
    V.require("surprise byte-id merged-vs-decoupled", surp_byte_id, expect=True)
    V.require("surprise answer-preserved vs native(dt=1.0)", surp_answer_preserved, expect=True)
    V.require("cross synapse surprise_S->sel_agent load-bearing", cross_lb, expect=True)
    V.require("read-isolation: comp leaves surprise slice untouched", bool(read_iso_comp), expect=True)
    V.require("read-isolation: surp guard restores role slice", bool(read_iso_surp), expect=True)
    go = bool(one_pool and determ and gabab_nmda_coexist and comp_byte_id and auc_m >= 0.80
              and comp_answer_preserved and surp_byte_id and surp_answer_preserved and cross_lb
              and read_iso_comp and read_iso_surp)
    V.decide(go=go)

    row = {
        "seed": seed, "dt_ms": dt_ms, "homeo": homeo, "go": go,
        "one_pool": one_pool, "n_all": n_all, "n_surprise": n_surp, "n_role": n_role,
        "determinism": determ, "gabab_nmda_coexist": gabab_nmda_coexist, "n_nmda_neurons": n_nmda, "n_sel": n_sel,
        "comp_auc_merged": float(auc_m), "comp_threshold_merged": thr_m,
        "comp_byte_id": comp_byte_id, "comp_byte_id_err": float(comp_byte_id_err),
        "comp_answer_preserved": comp_answer_preserved, "comp_match": int(comp_match), "comp_n": len(comp_bool_m),
        "comp_mean_well": float(np.mean(well_m)) if well_m else None,
        "comp_mean_ill": float(np.mean(ill_m)) if ill_m else None,
        "comp_auc_native": float(auc_nat),
        "surprise_functional": surprise_functional, "surprise_stats_merged": surp_stats_m,
        "surp_byte_id": surp_byte_id, "surp_byte_id_err": float(surp_byte_id_err),
        "surp_init_byte_id": surp_init_byte_id, "surp_init_err": float(surp_init_err),
        "surp_answer_preserved": surp_answer_preserved, "surp_match": int(surp_match), "surp_n": len(surp_bool_native),
        "surprise_stats_native": surp_stats_native,
        "cross_intact_hz": float(cross_intact), "cross_lesion_hz": float(cross_lesion),
        "cross_attribution_frac": (float(cross_frac) if cross_frac is not None else None),
        "cross_load_bearing": cross_lb,
        "read_iso_comp": bool(read_iso_comp), "read_iso_surp": bool(read_iso_surp),
    }
    if verbose:
        print(f"  [seed {seed} dt={dt_ms} homeo={homeo}] pool={one_pool} det={determ} nmda={n_nmda}/{n_sel} | "
              f"COMP auc={auc_m:.3f}(nat {auc_nat:.3f}) byteid={comp_byte_id}({comp_byte_id_err:.1e}) "
              f"ans={comp_answer_preserved}({comp_match}/{len(comp_bool_m)}) | "
              f"SURP sep={surp_stats_m['separation']:.1f}x func={surprise_functional} "
              f"byteid={surp_byte_id} ans={surp_answer_preserved}({surp_match}/{len(surp_bool_native)}) | "
              f"cross {cross_intact:+.1f}/{cross_lesion:+.1f}({cross_lb}) | "
              f"iso c={read_iso_comp} s={read_iso_surp} | GO={go}", flush=True)
    return row


def _gate(rows, cells, seeds):
    """GO iff there EXISTS a cell where ALL axes hold on >=5/6 seeds (merge-specific axes 6/6)."""
    n = len(seeds)
    by_cell = {}
    for (dt, h) in cells:
        cr = [r for r in rows if r["dt_ms"] == dt and r["homeo"] == h]
        key = f"dt{dt}_homeo{h}"
        by_cell[key] = {
            "n_go": sum(r["go"] for r in cr),
            "n_seeds": len(cr),
            "one_pool": sum(r["one_pool"] for r in cr),
            "determinism": sum(r["determinism"] for r in cr),
            "gabab_nmda_coexist": sum(r["gabab_nmda_coexist"] for r in cr),
            "comp_byte_id": sum(r["comp_byte_id"] for r in cr),
            "comp_auc_ge_080": sum(r["comp_auc_merged"] >= 0.80 for r in cr),
            "comp_answer_preserved": sum(r["comp_answer_preserved"] for r in cr),
            "surp_byte_id": sum(r["surp_byte_id"] for r in cr),
            "surp_answer_preserved": sum(r["surp_answer_preserved"] for r in cr),
            "cross_load_bearing": sum(r["cross_load_bearing"] for r in cr),
            "read_iso_comp": sum(r["read_iso_comp"] for r in cr),
            "read_iso_surp": sum(r["read_iso_surp"] for r in cr),
        }
    go_cells = [k for k, v in by_cell.items() if v["n_go"] >= max(5, n - 1)]
    return {"verdict": "GO" if go_cells else "BOUNDARY", "go_cells": go_cells,
            "go_by_cell": {k: v["n_go"] for k, v in by_cell.items()}, "per_cell": by_cell}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--cells", default="0.5:True,0.5:False,1.0:True,1.0:False")
    ap.add_argument("--smoke", action="store_true", help="1 seed x the 2x2 (short) sanity")
    ap.add_argument("--out", default="research/findings/raw/_one_brain_merge_configsuperset_6seed.json")
    a = ap.parse_args()

    seeds = [42] if a.smoke else [int(s) for s in a.seeds.split(",") if s.strip()]
    cells = [(float(c.split(":")[0]), c.split(":")[1] == "True") for c in a.cells.split(",")]

    from sim.backend import get_backend
    xp, _ = get_backend()

    print("=== CONFIG-SUPERSET PRODUCTION MERGE: surprise (GABA_B) + Wong-Wang comprehension (NMDA) on ONE bridge ===")
    print(f"    seeds={seeds}  cells(dt:homeo)={a.cells}")
    print("    GLOBAL CONFIG CONFLICT MAP:")
    for f, ev, rv, cls, note in _global_config_conflict_map():
        print(f"      {f:26s} surprise={ev!s:6s} role={rv!s:6s}  [{cls}]  {note}")

    rows = []
    for s in seeds:
        native_cache = {"comp": _native_comp(s, build_battery(s, n_per_cond=6)), "surp": _native_surprise(s, xp)}
        for (dt, h) in cells:
            rows.append(run_cell(s, dt, h, native_cache=native_cache))

    g = _gate(rows, cells, seeds)
    print("\n=== VERDICT ===")
    print(f"  go_by_cell: {g['go_by_cell']}")
    print(f"  reconcilable cell(s): {g['go_cells']}   ->   {g['verdict']}")
    for k, v in g["per_cell"].items():
        print(f"    {k}: GO {v['n_go']}/{v['n_seeds']} | pool {v['one_pool']} det {v['determinism']} "
              f"nmda {v['gabab_nmda_coexist']} | comp[byteid {v['comp_byte_id']} auc>=.8 {v['comp_auc_ge_080']} "
              f"ans {v['comp_answer_preserved']}] surp[byteid {v['surp_byte_id']} ans {v['surp_answer_preserved']}] "
              f"cross {v['cross_load_bearing']} iso[c {v['read_iso_comp']} s {v['read_iso_surp']}]")

    payload = {"mode": "one_brain_merge_configsuperset_production", "seeds": seeds, "cells": a.cells,
               "smoke": bool(a.smoke), "cross_weight": CROSS_WEIGHT, "read_steps": READ_STEPS,
               "config_conflict_map": [{"field": f, "surprise": ev, "role": rv, "class": cls, "note": nt}
                                       for f, ev, rv, cls, nt in _global_config_conflict_map()],
               "rows": rows, "gate": g}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
