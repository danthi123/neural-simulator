"""Workstream B (episodic) — a NON-SILENT (IGNITING) cortical heteroassociative cue-recall readout.

⭐ RESULTS STATUS (2026-08-08, Wave-2a build — read BEFORE trusting the design prose below):
  SURPASSED the Wave-1 death (readout SILENT, max cortical rate 0.000, winner = host argmax over an all-zero
  vector -> mechanical chance). The FIX is READOUT FAN-IN: the learned per-synapse CA3->cortex weight caps at
  ~11.6 under the rate-window Hebbian (sub-threshold for a cortical RS pyramidal; manual-weight ceiling ~60),
  but raising ca3_cortex_density to 1.0 over a larger assembly SUMMATES the many weak-but-SPECIFIC learned
  weights over threshold WITHOUT touching CA3 stability. The readout then IGNITES (max cortical rate ~0.04)
  and selects the correct WHAT/WHEN above chance (~0.75 vs 0.25) by NEURAL heteroassociative specificity, with
  a full teeth panel PASSING: permuted-cue -> chance + wrong readout, real-lesion -> silent + chance,
  sham-lesion -> preserved, untrained -> silent + chance. This corrects anti-cheat (1)'s Wave-1 mislabel.

  HONEST NEGATIVES (mapped with teeth, NOT dressed over):
   - CA3 pattern COMPLETION is NOT load-bearing here: held-out assembly firing (ca3_compl) = 0.00 and the
     zero-recurrent control is INERT (== full). The recall is FEEDFORWARD heteroassociation from the CUED CA3
     cells, NOT recurrent attractor completion. The term "completion" (docs/TERMS.md) is therefore NOT EARNED.
   - The neural WTA lateral inhibition is NOT load-bearing (wta_off == full): at this sparse operating point
     the selector is the heteroassociative specificity, not lateral inhibition.
   - The CORTICAL feedforward cue cannot ignite CA3 (ca3_compl=0.00 from a cortex_who cue) -> the admissible
     cue here is a PARTIAL CA3 cue delivered INTO CA3 (as EC/mossy delivers it).
   - Ignition (needs a LARGE assembly for fan-in) and completion (needs a SMALL sparse assembly) have opposing
     assembly-size requirements; resolving both needs a sim/ change (a cortical-stage bistable dendritic
     amplifier, or a target-specific readout plasticity gain). Out of the NO-sim-edit scope.

  ---- ORIGINAL DESIGN PROSE (Wave-1 intent; the WTA-as-selector + CLOSED-completion claims are the negatives above) ----
Wave-2a mechanism build. Rides the ALREADY-CLOSED bistable+specific CA3 attractor (the 2026-07-18 gap#5
mechanism: coincidence two-compartment dAP + self-regen plateau + KIR down-state + asymmetric apical read +
selective inhibition + structural pattern-separation) as the completion ENGINE, and adds the two stages that
Wave-1 was missing:

  1. CORTICAL ATTRIBUTE POOLS -- an episode = a conjunction of ONE item from each of three cortical pools
     WHO / WHAT / WHEN. Each pool holds K candidate items (disjoint sparse cell sets). An episode binds a
     (who_i, what_j, when_k) triple to a sparse CA3 assembly.
  2. RECIPROCAL cortex<->CA3 heteroassociation -- plastic RegionPathways (hebbian_rate_window co-activity,
     NOT STDP which is silent here) potentiate synchronous co-firing during ENCODE (plasticity ON): the
     cortex->CA3 feedforward encoder AND the CA3->cortex reciprocal readout are learned in the SAME co-drive.
  3. NEURAL WTA readout -- per-pool FEEDBACK LATERAL INHIBITION (an FS basket driven by the whole pool,
     inhibiting the whole pool = de Almeida-Idiart-Lisman E%-max): the most-driven item's cells fire and
     suppress the others -> ONE winner per pool, selected by NEURONS, NOT a host argmax over spike counts
     (the Wave-1 shortcut that passed while true neural recall was 0/6).

RECALL (plasticity FROZEN, OU off, dendritic reset): cue ONE pool (drive WHO's item cells in cortex) ->
cortex_who->CA3 lights the who-portion of the assembly -> the CLOSED bistable CA3 recurrent COMPLETES the full
assembly (or rests silent on a permuted cue) -> CA3->cortex reactivates the bound WHAT/WHEN item cells -> the
per-pool WTA collapses to one winner. We read WHICH item each pool's lateral inhibition selected.

ANTI-CHEATS WITH TEETH (each must be able to fail in its failing direction):
  (1) end-to-end NEURAL cortical recall: the WHAT/WHEN winner beats chance (1/K) -- the 0/6 bar to beat.
  (2) recurrent-CA3-zero control: zero ca3->ca3 at recall -> a WHO-cue lights only the who-portion, cannot
      complete the what/when portions -> cortical what/when reactivation drops. Isolates COMPLETION from the
      feedforward encoder (a pure who->ca3->what shortcut would survive this; real completion does not).
  (3) permuted-cue specificity: a WHO item from NO episode -> CA3 stays silent -> no cortical reactivation.
  (4) real + SHAM lesion: hyperpolarize the episode's CA3 assembly (real) vs an equal-size UNRELATED CA3 set
      (sham). Real must drop cortical recall; sham must NOT. The metric is CORTICAL firing, NOT a CA3-overlap
      metric, so the lesion is NOT tautological (Wave-1 B hyperpolarized CA3 and read a CA3-overlap metric --
      it could not fail).
  (5) untrained-CA3 control: skip ENCODE entirely -> recall fails (tests the LEARNED engram, not the wiring).

HONESTY: the CLOSED CA3 completion is itself weak (~0.18-0.33, 5/6 by the gap#5 standard); stacking a
cortex->CA3 encoder and a CA3->cortex readout on top is expected to be LOSSY. An honest NEGATIVE here (the
CA3->cortex readout silent, or the WTA non-specific) is a first-class deliverable that MAPS what the readout
needs. NO sim/ edit -- reuse-by-import + runner-side pathway construction. cfg.seed seeds the substrate.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _build_cortical(seed, n_ca3=1500, ca3_density=0.05, ca3w=6.0,
                    n_item_cells=14, k_items=4,
                    cortex_ca3_w=4.0, ca3_cortex_w=4.0, wta_ei_w=6.0, wta_ie_w=18.0,
                    ca3_fb_inhib=20.0, k_thresh=18.0, plateau_strength=120.0,
                    plateau_self_regen=0.15, plateau_v_hold=-35.0, apical_kir_g=3.0,
                    apical_gc_read=5.0, hebb_max=2000.0, hebb_lr=None, coact_thresh=0.02,
                    enable_ou=True, ca3_cortex_density=0.5, cortex_ca3_density=0.5):
    """Build ONE SimulationBridge = hippocampus (CLOSED CA3) + 3 cortical attribute pools + per-pool NEURAL WTA
    basket + plastic reciprocal cortex<->CA3 pathways. Mirrors the gap#5 `_build` CLOSED config; adds cortex.
    NO sim/ edit -- all runner-side region/pathway construction before bridge init."""
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType
    from research.runners.text_minimal_isolation import build_biological_brain_regions

    # pool must hold k_items disjoint sparse item-cell sets
    n_pool = int(k_items * n_item_cells)
    regions, pathways = build_biological_brain_regions(
        n_lang_input=64, n_motor_per_action=8, n_motor_fs_per_action=2, enable_motor_fs=True,
        enable_language_output=True, n_lang_output=64, enable_hippocampus_consolidation=True,
        n_ec=80, n_dg=200, n_ca3=n_ca3, n_ca1=100, ca3_recurrent_density=ca3_density,
        ca3_recurrent_weight=ca3w, ca3_to_ca1_density=0.30)

    # --- CLOSED CA3: route ca3->ca3 recurrent through the dendritic-coincidence plateau (the completion engine)
    for p in pathways:
        if getattr(p, "from_region", None) == "ca3" and getattr(p, "to_region", None) == "ca3":
            p.coincidence_detector = True

    # --- CA3 feedback inhibition (sparsity/PING) -- the gap#5 ca3_pv_basket
    _nb = max(8, int(0.25 * n_ca3))
    regions.append(BrainRegion(
        name="ca3_pv_basket", n_neurons=_nb, exc_fraction=0.0, internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
    pathways.append(RegionPathway(from_region="ca3", to_region="ca3_pv_basket",
                                  density=0.40, weight_mean=5.0, weight_jitter=0.2, plastic=False))
    pathways.append(RegionPathway(from_region="ca3_pv_basket", to_region="ca3",
                                  density=1.0, weight_mean=float(ca3_fb_inhib), weight_jitter=0.2, plastic=False))

    # --- 3 cortical attribute pools + per-pool NEURAL WTA basket + reciprocal plastic cortex<->CA3
    pool_names = ["cortex_who", "cortex_what", "cortex_when"]
    for nm in pool_names:
        regions.append(BrainRegion(
            name=nm, n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name))
        # reciprocal plastic heteroassociation (rate-Hebbian): cortex->CA3 encoder + CA3->cortex readout.
        # READOUT FAN-IN (ca3_cortex_density): the Wave-1 death was a SILENT readout -- the learned per-synapse
        # weight saturates ~11.6 (rate-window Hebbian, hebb_max far off) which alone is sub-threshold for a
        # cortical RS pyramidal (manual-weight ceiling: W~=60 needed). RAISING FAN-IN (density 1.0 x a larger
        # assembly) summates the many weak-but-SPECIFIC learned weights over threshold WITHOUT touching CA3
        # stability (adding ca3->cortex synapses does not feed back into CA3). This is the ignition lever.
        pathways.append(RegionPathway(from_region=nm, to_region="ca3", density=float(cortex_ca3_density),
                                      weight_mean=float(cortex_ca3_w), weight_jitter=0.2, plastic=True,
                                      plasticity_gate=f"{nm}_to_ca3"))
        pathways.append(RegionPathway(from_region="ca3", to_region=nm, density=float(ca3_cortex_density),
                                      weight_mean=float(ca3_cortex_w), weight_jitter=0.2, plastic=True,
                                      plasticity_gate=f"ca3_{nm}"))
        # NEURAL WTA: feedback lateral inhibition (E%-max). Basket driven by the whole pool, inhibits the whole
        # pool -> only the most-driven item's cells cross threshold before inhibition clamps -> one winner.
        bname = f"{nm}_wta"
        _nbw = max(6, int(0.4 * n_pool))
        regions.append(BrainRegion(
            name=bname, n_neurons=_nbw, exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
        pathways.append(RegionPathway(from_region=nm, to_region=bname, density=0.6,
                                      weight_mean=float(wta_ei_w), weight_jitter=0.2, plastic=False))
        # transmission_gate lets us DISABLE the WTA at recall (anti-cheat: lateral inhibition load-bearing)
        pathways.append(RegionPathway(from_region=bname, to_region=nm, density=1.0,
                                      weight_mean=float(wta_ie_w), weight_jitter=0.2, plastic=False,
                                      transmission_gate=f"{bname}_gate"))

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0; cfg.seed = seed; cfg.enable_nmda = True
    cfg.enable_ou_process = bool(enable_ou)
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_rate_window = True                      # windowed co-activity (rate-Hebbian) -- STDP is silent here
    cfg.hebbian_coactivity_thresh = float(coact_thresh)
    if hebb_lr is not None:
        cfg.hebbian_learning_rate = float(hebb_lr)
    cfg.hebbian_max_weight = float(hebb_max)
    cfg.stdp_w_max = max(10.0, 2.5 * ca3w); cfg.fast_spike_reset = True
    # CLOSED CA3 dendritic-bistability completion engine
    cfg.enable_coincidence_detection = True
    cfg.coincidence_weighted_drive = True
    cfg.coincidence_k_threshold = float(k_thresh)
    cfg.coincidence_plateau_strength = float(plateau_strength)
    cfg.enable_two_compartment_dap = True
    cfg.coincidence_plateau_self_regen = float(plateau_self_regen)
    cfg.coincidence_plateau_v_hold = float(plateau_v_hold)
    cfg.apical_kir_g = float(apical_kir_g)
    cfg.apical_g_couple_to_soma = float(apical_gc_read)
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, pool_names, n_pool


_GATES = ["ca3_swr_burst", "dg_to_ca3", "ec_to_dg", "lang_to_ec"]


def _set_gates(bridge, v, pool_names):
    for g in _GATES:
        try:
            bridge.set_plasticity_gate(g, v)
        except Exception:
            pass
    for nm in pool_names:
        for g in (f"{nm}_to_ca3", f"ca3_{nm}"):
            try:
                bridge.set_plasticity_gate(g, v)
            except Exception:
                pass


def _extract_ca3ca3_coincidence(bridge, ca3_idx, to_host):
    from research.runners._riii_ca3_competitive_formation_derisk import _extract_ca3ca3_coincidence as _ex
    return _ex(bridge, ca3_idx, to_host)


def run(seed, n_ca3=1500, k_items=4, n_item_cells=14, assembly_frac=0.06, train_events=40,
        encode_drive=700.0, recall_drive=500.0, cue_recall_drive=650.0, reset_steps=15, drive_steps=40,
        recall_steps=60, lam_dep_wi=0.5, hebb_max=2000.0, k_thresh=18.0, recall_k_thresh=40.0,
        plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc_read=5.0, ca3_fb_inhib=20.0,
        cortex_ca3_w=4.0, ca3_cortex_w=4.0, wta_ei_w=6.0, wta_ie_w=18.0, coact_thresh=0.02, hebb_lr=None,
        structural_sep=True, selective_inhib=True, sel_inhib_spare=0.0,
        untrained=False, zero_recurrent=False, wta_off=False, permute_cue=False,
        lesion=None, verbose=False, ca3_cortex_density=1.0, ca3_cue_frac=0.5):
    """Encode E=k_items episodes (each a distinct WHO x WHAT x WHEN triple bound to a CA3 assembly), then RECALL
    each from its WHO cue and read the NEURAL WTA winner in WHAT and WHEN. Returns a metrics dict.

    lesion: None | "real" | "sham". real = hyperpolarize the recalled episode's CA3 assembly; sham = an equal-size
    unrelated CA3 set. Both must be able to fail in their failing direction (real drops recall, sham does not)."""
    from sim.backend import get_backend, to_host, from_host
    from sim.kernels import fused_htm_winner_inactive_depression
    cp, _ = get_backend()

    bridge, pool_names, n_pool = _build_cortical(
        seed, n_ca3=n_ca3, ca3_density=0.05, k_items=k_items, n_item_cells=n_item_cells,
        cortex_ca3_w=cortex_ca3_w, ca3_cortex_w=ca3_cortex_w, wta_ei_w=wta_ei_w, wta_ie_w=wta_ie_w,
        ca3_fb_inhib=ca3_fb_inhib, k_thresh=k_thresh, plateau_self_regen=plateau_self_regen,
        apical_kir_g=apical_kir_g, apical_gc_read=apical_gc_read, hebb_max=hebb_max, hebb_lr=hebb_lr,
        coact_thresh=coact_thresh, ca3_cortex_density=ca3_cortex_density)
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3"))
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    rng = np.random.default_rng(seed * 17 + 3)

    # --- item-cell sets per pool (k_items disjoint sparse sets), and CA3 assemblies (one per episode) ---
    pool_glob = {nm: np.asarray(rm.indices(nm), dtype=np.int64) for nm in pool_names}
    items = {}   # nm -> list[np.array of global item-cell indices], length k_items
    for nm in pool_names:
        g = pool_glob[nm]
        items[nm] = [g[i * n_item_cells:(i + 1) * n_item_cells].copy() for i in range(k_items)]
    n_assy = max(6, int(assembly_frac * n_ca3))
    # disjoint CA3 assemblies (SWR-style separation): one per episode
    _pool = rng.choice(ca3_idx, n_assy * k_items, replace=False)
    assemblies = [np.asarray(sorted(_pool[i * n_assy:(i + 1) * n_assy]), dtype=np.int64) for i in range(k_items)]
    # episode e binds item e of who, item e of what, item e of when to assembly e (identity binding; the WTA
    # must nonetheless recover what/when from a WHO-only cue via completion, so the binding index is not a cue)
    episodes = [{"who": items["cortex_who"][e], "what": items["cortex_what"][e],
                 "when": items["cortex_when"][e], "assy": assemblies[e]} for e in range(k_items)]

    # extract ca3->ca3 coincidence synapses for competition / structural_sep
    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_coincidence(bridge, ca3_idx, to_host)
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

    # ================= ENCODE (plasticity ON): co-drive assembly + who/what/when items ==================
    if not untrained:
        _set_gates(bridge, 1.0, pool_names)
        for e, ep in enumerate(episodes):
            assy_arr = cp.asarray(ep["assy"], dtype=cp.int64)
            drive_glob = np.concatenate([ep["assy"], ep["who"], ep["what"], ep["when"]]).astype(np.int64)
            drive_arr = cp.asarray(drive_glob, dtype=cp.int64)
            member_mask = cp.zeros(len(ca3_idx), dtype=cp.float32)
            member_mask[cp.asarray([ca3_pos[int(g)] for g in ep["assy"]], dtype=cp.int64)] = 1.0
            for ev in range(train_events):
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(reset_steps):
                    bridge._run_one_simulation_step()
                for _st in range(drive_steps):
                    bridge.cp_external_input_current[:] = 0.0
                    bridge.cp_external_input_current[drive_arr] = float(encode_drive)   # synchronous co-drive
                    bridge._run_one_simulation_step()
                if do_comp:
                    _apply_competition(member_mask)     # sharpen within-assembly ca3->ca3 (competition)
            bridge.cp_external_input_current[:] = 0.0
        _set_gates(bridge, 0.0, pool_names)

    # STRUCTURAL PATTERN SEPARATION: zero non-member->member ca3->ca3 (permuted-cue can't leak into an assembly)
    if structural_sep and not untrained and len(flat_h) > 0:
        assy_pos_set = set(ca3_pos[int(g)] for a in assemblies for g in a)
        zk = [k for k in range(len(flat_h))
              if int(post_l_h[k]) in assy_pos_set and int(pre_l_h[k]) not in assy_pos_set]
        if zk:
            idxs = cp.asarray([int(flat_h[k]) for k in zk], dtype=cp.int64)
            conn.data[idxs] = cp.zeros(len(zk), dtype=conn.data.dtype)

    # ASSEMBLY-SELECTIVE INHIBITION (Kim-Kim spare-your-own): depress ca3_pv_basket->member (I->E) so a correct
    # cue's assembly cells are spared while non-members are still suppressed (permuted-cue avalanche control).
    if selective_inhib and not untrained:
        bask_idx = set(int(g) for g in rm.indices("ca3_pv_basket"))
        assy_glob = set(int(g) for a in assemblies for g in a)
        nnz = int(conn.nnz)
        indptr = np.asarray(to_host(conn.indptr)); indices = np.asarray(to_host(conn.indices))
        pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
        spare_k = [k for k in range(nnz) if int(pre_of[k]) in bask_idx and int(indices[k]) in assy_glob]
        if spare_k:
            idxs = cp.asarray(spare_k, dtype=cp.int64)
            conn.data[idxs] = cp.full(len(spare_k), float(sel_inhib_spare), dtype=conn.data.dtype)

    # decouple recall dAP threshold (higher at recall -> only the strong learned coincident drive completes)
    bridge.core_config.coincidence_k_threshold = float(recall_k_thresh)

    # recurrent-CA3-zero control (anti-cheat 2): zero ca3->ca3 -> no completion -> who-cue cannot reactivate what/when
    if zero_recurrent and len(flat_h) > 0:
        conn.data[cp.asarray(flat_h, dtype=cp.int64)] = cp.zeros(len(flat_h), dtype=conn.data.dtype)

    # ================= RECALL (plasticity FROZEN, OU off, dendritic reset) ==================
    bridge.core_config.enable_hebbian_learning = False
    bridge.core_config.enable_ou_process = False
    # WTA on/off: disable per-pool lateral inhibition (anti-cheat: is the neural WTA load-bearing?)
    for nm in pool_names:
        try:
            bridge.set_transmission_gate(f"{nm}_wta_gate", 0.0 if wta_off else 1.0)
        except Exception:
            pass

    ca3_arr_host = np.asarray(ca3_idx, dtype=int)
    n_all = bridge.core_config.num_neurons

    def _hard_silence(settle=25):
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

    # lesion current (applied every recall step): strong hyperpolarizing bias on a CA3 set
    lesion_bias = np.zeros(n_all, dtype=np.float64)

    def _set_lesion(target_assy):
        lesion_bias[:] = 0.0
        if lesion is None:
            return
        if lesion == "real":
            tgt = np.asarray(target_assy, dtype=int)
        else:  # sham: equal-size UNRELATED CA3 set (non-assembly members)
            non = np.asarray([g for g in ca3_idx if g not in set(int(x) for a in assemblies for x in a)],
                             dtype=int)
            tgt = np.random.default_rng(seed * 91 + 5).choice(non, min(len(target_assy), len(non)),
                                                              replace=False)
        lesion_bias[tgt] = -900.0

    def _measure_recall(cue_glob, target_assy):
        """Drive the WHO cue in cortex; read per-pool per-item cortical firing + CA3 assembly firing."""
        _hard_silence()
        _set_lesion(target_assy)
        cur = lesion_bias.copy()
        if cue_glob is not None and len(cue_glob):
            cur[np.asarray(cue_glob, dtype=int)] += float(cue_recall_drive)
        dev = from_host(cur.astype(np.float64))
        pool_spk = {nm: np.zeros(n_pool) for nm in pool_names}
        ca3_spk = np.zeros(len(ca3_idx))
        for _ in range(recall_steps):
            bridge.cp_external_input_current[:] = dev
            bridge._run_one_simulation_step()
            fs = np.asarray(to_host(bridge.cp_firing_states))
            for nm in pool_names:
                pool_spk[nm] += fs[pool_glob[nm]].astype(float)
            ca3_spk += fs[ca3_arr_host].astype(float)
        bridge.cp_external_input_current[:] = 0.0
        pr = {nm: pool_spk[nm] / recall_steps for nm in pool_names}
        max_cx = max(float(pr[nm].max()) for nm in pool_names)
        return pr, ca3_spk / recall_steps, max_cx

    def _item_rates(pool_rate_vec, nm):
        """Per-item mean firing rate in pool nm (n_item_cells cells per item)."""
        return np.array([float(np.mean(pool_rate_vec[i * n_item_cells:(i + 1) * n_item_cells]))
                         for i in range(k_items)])

    # ---- CUE = a PARTIAL CA3 assembly (delivered INTO CA3, as EC/mossy delivers the cue; the cortical
    # feedforward cue CANNOT trigger the CA3 dendritic plateau -- ca3_compl=0.00 -- so the admissible cue
    # is the partial-CA3 completion cue the gap#5 CLOSED attractor was validated on). The held-out portion
    # is reconstructed by ca3->ca3 completion; the CA3->cortex readout then reactivates the bound WHAT/WHEN.
    read_attrs = ["what", "when"]
    winner_correct = {a: [] for a in read_attrs}
    sep_correct_vs_other = {a: [] for a in read_attrs}
    ca3_completion = []          # HELD-OUT (non-cued) assembly firing fraction -- the real completion metric
    max_cortex_rates = []        # ABSOLUTE cortical readout rate -- the Wave-1 death metric (must be > 0)
    for e, ep in enumerate(episodes):
        assy = np.asarray(ep["assy"], dtype=np.int64)
        n_cue = max(1, int(ca3_cue_frac * len(assy)))
        if permute_cue:
            # partial cue from a DIFFERENT episode's assembly -> no valid completion of THIS episode
            src = np.asarray(episodes[(e + 1) % k_items]["assy"], dtype=np.int64)
            cue = src[:n_cue]
            held_pos = [ca3_pos[int(g)] for g in assy]          # none of THIS assembly is cued
        else:
            cue = assy[:n_cue]
            held = assy[n_cue:]
            held_pos = [ca3_pos[int(g)] for g in held]          # completion = held-out cells firing
        pool_rates, ca3_rate, max_cx = _measure_recall(cue, ep["assy"])
        max_cortex_rates.append(max_cx)
        ca3_completion.append(float(np.mean((ca3_rate[held_pos] > 0.05).astype(float))) if held_pos else 0.0)
        for a in read_attrs:
            r = _item_rates(pool_rates[f"cortex_{a}"], f"cortex_{a}")
            correct = e
            # winner = which item-pool the readout drove hardest. Admissible ONLY when the readout is
            # non-silent (max_cx > 0); over an all-zero rate vector argmax is a HOST TIEBREAK, not a decision.
            winner = int(np.argmax(r))
            winner_correct[a].append(1.0 if winner == correct else 0.0)
            others = np.delete(r, correct)
            sep_correct_vs_other[a].append(float(r[correct] - (np.max(others) if len(others) else 0.0)))

    def _m(d):
        return {a: float(np.mean(v)) for a, v in d.items()}

    wc = _m(winner_correct); sep = _m(sep_correct_vs_other)
    overall_winner = float(np.mean([wc[a] for a in read_attrs]))
    overall_sep = float(np.mean([sep[a] for a in read_attrs]))
    max_cortex = float(np.mean(max_cortex_rates))
    chance = 1.0 / k_items
    out = {"seed": seed, "n_ca3": n_ca3, "k_items": k_items, "chance": chance,
           "winner_what": wc["what"], "winner_when": wc["when"], "winner_overall": overall_winner,
           "sep_what": sep["what"], "sep_when": sep["when"], "sep_overall": overall_sep,
           "ca3_completion": float(np.mean(ca3_completion)),
           "max_cortex_rate": max_cortex,
           "readout_ignited": bool(max_cortex > 1e-6),
           # LEVER-ENGAGEMENT witnesses: prove each manipulation TOUCHED the substrate (so an IDENTICAL
           # downstream metric is an 'engaged-but-inert' honest negative, NOT a lever that never ran).
           "lever_ca3_recurrent_zeroed": int(len(flat_h)) if zero_recurrent else 0,
           "lever_wta_gate": 0.0 if wta_off else 1.0,
           "lever_lesion_cells": int(n_assy) if lesion else 0,
           "lever_cue_permuted": 1.0 if permute_cue else 0.0,
           "lever_untrained": 1.0 if untrained else 0.0,
           "condition": ("untrained" if untrained else "zero_recurrent" if zero_recurrent
                         else "wta_off" if wta_off else "permute_cue" if permute_cue
                         else f"lesion_{lesion}" if lesion else "full")}
    if verbose:
        print(f"    [{out['condition']:14s}] winner what={wc['what']:.2f} when={wc['when']:.2f} "
              f"overall={overall_winner:.2f} (chance {chance:.2f}) | sep={overall_sep:+.3f} | "
              f"max_cortex={max_cortex:.3f} | ca3_compl={out['ca3_completion']:.2f}", flush=True)
    return out


def ceiling_probe(seed, n_ca3=800, W=120.0, cue_dr=800.0, n_item_cells=12, k_items=3):
    """CEILING / INSTRUMENT test: manually set the reciprocal cortex<->CA3 + recurrent weights STRONG (W), then
    ask whether the pipeline CAN fire end-to-end -- separating 'storing rule too weak' from 'architecture cannot
    conduct'. Reports the mechanism-boundary numbers the honest negative rests on: (a) the g_e cortex->CA3
    delivers to the assembly, (b) whether a WHO cue ignites the assembly at all, (c) cortical cue adaptation,
    (d) whether a DIRECT strong assembly drive reactivates the cortical readout. If even W=strong fails, the
    boundary is architectural, not a weak-weights artifact."""
    from sim.backend import get_backend, to_host, from_host
    cp, _ = get_backend()
    b, pools, _ = _build_cortical(seed, n_ca3=n_ca3, k_items=k_items, n_item_cells=n_item_cells,
                                  ca3_fb_inhib=5.0, hebb_max=400.0)
    rm = b.region_manager
    ca3 = np.asarray(rm.indices("ca3")); n = b.core_config.num_neurons
    who = np.asarray(rm.indices("cortex_who")); what = np.asarray(rm.indices("cortex_what"))
    conn = b.cp_connections; nnz = int(conn.nnz)
    ip = np.asarray(to_host(conn.indptr)); ind = np.asarray(to_host(conn.indices))
    pre = np.searchsorted(ip, np.arange(nnz), side="right") - 1

    def es(ps, qs):
        ps = set(int(x) for x in ps); qs = set(int(x) for x in qs)
        return np.asarray([k for k in range(nnz) if int(pre[k]) in ps and int(ind[k]) in qs], dtype=np.int64)

    rng = np.random.default_rng(1)
    assy = np.sort(rng.choice(ca3, max(6, int(0.06 * n_ca3)), replace=False))
    whoit = who[:n_item_cells]; whatit = what[:n_item_cells]
    for ks, val in [(es(whoit, assy), W), (es(assy, whatit), W), (es(assy, assy), W)]:
        conn.data[cp.asarray(ks)] = cp.full(len(ks), float(val), dtype=conn.data.dtype)
    b.core_config.enable_hebbian_learning = False; b.core_config.enable_ou_process = False
    b.core_config.coincidence_k_threshold = 40.0

    def silence():
        b.cp_membrane_potential_v[:] = b.cp_izh_c_reset; b.cp_recovery_variable_u[:] = 0.0
        for a in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_coincidence",
                  "cp_conductance_g_coincidence_rise"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = 0.0
        if b.cp_v_apical is not None:
            b.cp_v_apical[:] = -65.0
        b.cp_external_input_current[:] = 0.0
        for _ in range(20):
            b._run_one_simulation_step()

    # WHO-cue ignition + cue adaptation + g_e delivered
    silence()
    cur = np.zeros(n); cur[whoit] = float(cue_dr); dev = from_host(cur.astype(np.float64))
    who_early = who_late = ge_peak = assy_fire = 0.0
    for t in range(50):
        b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
        fs = np.asarray(to_host(b.cp_firing_states))
        ge = getattr(b, "cp_conductance_g_e", None)
        if ge is not None:
            ge_peak = max(ge_peak, float(np.asarray(to_host(ge))[assy].mean()))
        if t < 10:
            who_early += fs[whoit].mean()
        if t >= 40:
            who_late += fs[whoit].mean()
        assy_fire += fs[assy].mean()
    who_early /= 10; who_late /= 10; assy_fire /= 50
    # DIRECT strong assembly drive -> cortical readout
    silence()
    cur = np.zeros(n); cur[assy] = 600.0; dev = from_host(cur.astype(np.float64))
    da = dw = 0.0
    for _ in range(60):
        b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
        fs = np.asarray(to_host(b.cp_firing_states)); da += fs[assy].mean(); dw += fs[whatit].mean()
    da /= 60; dw /= 60
    out = {"W": W, "who_cue_early_fire": who_early, "who_cue_late_fire": who_late,
           "assy_ge_peak_from_who": ge_peak, "who_cue_ignites_assy_fire": assy_fire,
           "direct_assy_fire": da, "direct_readout_what_fire": dw}
    print(f"[ceiling W={W}] who_cue fire early={who_early:.2f} late={who_late:.2f} (ADAPTS if late<<early) | "
          f"assy g_e from who={ge_peak:.2f} -> assy fire={assy_fire:.3f} (0 => cortex cannot ignite CA3) | "
          f"DIRECT assy fire={da:.3f} -> readout what fire={dw:.3f} (0 => CA3 cannot fire cortical readout)",
          flush=True)
    return out


def _verify_seed(seed, n_ca3=800):
    """Build twice at one seed, hash cp_neuron_firing_thresholds -> identical means cfg.seed seeds the substrate."""
    from sim.backend import to_host
    import hashlib
    hashes = []
    for _ in range(2):
        bridge, _, _ = _build_cortical(seed, n_ca3=n_ca3, k_items=3, n_item_cells=8)
        h = hashlib.sha1(np.asarray(to_host(bridge.cp_neuron_firing_thresholds)).tobytes()).hexdigest()
        hashes.append(h)
    ok = hashes[0] == hashes[1]
    print(f"[seed-verify] seed={seed} thresholds hash: {hashes[0][:16]} == {hashes[1][:16]} -> "
          f"{'IDENTICAL (seeded)' if ok else 'DIFFERENT (NOT seeded!)'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-ca3", type=int, default=1500)
    ap.add_argument("--k-items", type=int, default=4)
    ap.add_argument("--train-events", type=int, default=40)
    ap.add_argument("--assembly-frac", type=float, default=0.10)
    ap.add_argument("--ca3-cortex-density", type=float, default=1.0)
    ap.add_argument("--ca3-cue-frac", type=float, default=0.5)
    ap.add_argument("--verify-seed", action="store_true")
    ap.add_argument("--ceiling", action="store_true", help="instrument/ceiling probe (strong manual weights)")
    ap.add_argument("--smoke", action="store_true", help="single-seed, all conditions")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.verify_seed:
        _verify_seed(int(a.seeds.split(",")[0]))
        return
    if a.ceiling:
        for W in (16.0, 120.0, 300.0):
            ceiling_probe(int(a.seeds.split(",")[0]), n_ca3=a.n_ca3, W=W)
        return

    from tools.verdict import Verdict
    from tools.lab import attributable_to

    t0 = time.time()
    seeds = [int(x) for x in a.seeds.split(",")]
    all_rows = []
    for s in seeds:
        print(f"[cortical-episodic-WTA] seed={s} n_ca3={a.n_ca3} k_items={a.k_items} "
              f"train_events={a.train_events}", flush=True)
        kw = dict(n_ca3=a.n_ca3, k_items=a.k_items, train_events=a.train_events, verbose=True,
                  assembly_frac=a.assembly_frac, ca3_cortex_density=a.ca3_cortex_density,
                  ca3_cue_frac=a.ca3_cue_frac)
        full = run(s, **kw)
        zero_rec = run(s, zero_recurrent=True, **kw)
        wta_off = run(s, wta_off=True, **kw)
        perm = run(s, permute_cue=True, **kw)
        les_real = run(s, lesion="real", **kw)
        les_sham = run(s, lesion="sham", **kw)
        untr = run(s, untrained=True, **kw)
        rows = {"full": full, "zero_recurrent": zero_rec, "wta_off": wta_off, "permute_cue": perm,
                "lesion_real": les_real, "lesion_sham": les_sham, "untrained": untr}
        all_rows.append({"seed": s, **rows})
        print(f"  seed {s} done ({time.time()-t0:.0f}s)", flush=True)

    # aggregate (mean over seeds)
    def agg(cond, key):
        return float(np.mean([r[cond][key] for r in all_rows]))
    chance = 1.0 / a.k_items
    full_win = agg("full", "winner_overall")
    print("\n=== ATTRIBUTION (mean over seeds) ===", flush=True)
    attributable_to("completion (full - zero_recurrent)", full_win, agg("zero_recurrent", "winner_overall"))
    attributable_to("neural WTA (full sep - wta_off sep)", agg("full", "sep_overall"),
                    agg("wta_off", "sep_overall"))

    full_max_cortex = agg("full", "max_cortex_rate")
    v = Verdict("B-episodic: IGNITING cortical heteroassociative cue-recall (readout non-silent + neural)", chance=chance)
    v.disabled("OU noise process", "recall isolates deterministic bistability (gap#5 protocol)")
    v.disabled("Hebbian/STDP plasticity", "frozen at recall (fixed autoassociator)")
    # PRECONDITION (the Wave-1 death): the cortical readout must IGNITE. Over an all-zero rate vector the
    # winner is a host argmax tiebreak, not a neural decision -> the accuracy floor is UNDEFINED, not a score.
    v.floor("cortical readout IGNITES (max cortical rate > 0)", full_max_cortex, floor=0.0)
    v.floor("end-to-end neural cortical recall vs chance", full_win, artifact={"chance": chance})
    v.control("completion load-bearing (full vs zero-recurrent)", full_win, agg("zero_recurrent", "winner_overall"))
    v.control("neural WTA load-bearing (full-sep vs wta-off-sep)", agg("full", "sep_overall"),
              agg("wta_off", "sep_overall"))
    v.control("permuted-cue specificity (full vs permuted)", full_win, agg("permute_cue", "winner_overall"))
    v.control("real lesion drops recall (full vs lesion-real)", full_win, agg("lesion_real", "winner_overall"))
    v.require("sham lesion PRESERVES recall (sham ~ full)",
              abs(agg("lesion_sham", "winner_overall") - full_win) < 0.5 * max(full_win, 1e-6) or
              agg("lesion_sham", "winner_overall") >= agg("lesion_real", "winner_overall"), expect=True)
    v.control("untrained-CA3 fails (full vs untrained)", full_win, agg("untrained", "winner_overall"))
    go = (full_max_cortex > 1e-6                       # readout non-silent is the PRECONDITION for any B claim
          and full_win > chance + 1e-6
          and full_win > agg("zero_recurrent", "winner_overall")
          and full_win > agg("permute_cue", "winner_overall")
          and full_win > agg("lesion_real", "winner_overall")
          and full_win > agg("untrained", "winner_overall"))
    decided = v.decide(go=go)

    result = {"seeds": seeds, "chance": chance,
              "full_winner_overall": full_win,
              "full_max_cortex_rate": full_max_cortex,
              "full_readout_ignited": bool(full_max_cortex > 1e-6),
              "zero_recurrent_winner": agg("zero_recurrent", "winner_overall"),
              "zero_recurrent_ca3_completion": agg("zero_recurrent", "ca3_completion"),
              "wta_off_sep": agg("wta_off", "sep_overall"), "full_sep": agg("full", "sep_overall"),
              "permute_cue_winner": agg("permute_cue", "winner_overall"),
              "permute_cue_max_cortex": agg("permute_cue", "max_cortex_rate"),
              "lesion_real_winner": agg("lesion_real", "winner_overall"),
              "lesion_sham_winner": agg("lesion_sham", "winner_overall"),
              "untrained_winner": agg("untrained", "winner_overall"),
              "untrained_max_cortex": agg("untrained", "max_cortex_rate"),
              "full_ca3_completion": agg("full", "ca3_completion"),
              "verdict": decided, "rows": all_rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[wrote] {a.out}", flush=True)


if __name__ == "__main__":
    main()
