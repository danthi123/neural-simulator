"""ONE-BRAIN MERGE de-risk — the RECALL COMPOSER (+ its phase->spike transducer cleanup region) joins
PRODUCTION POOL #1 (SURPRISE + WORLD-MODEL) on ONE shared spiking substrate.

THE RUNG (this lane's mission)
------------------------------
Production pool #1 (`onebrain_merge_production.py`, `MergedSubstrate`, DEFAULT-ON 2026-08-13) puts the D2
SURPRISE expectation-violation organ and the E2 affective WORLD-MODEL organ on ONE shared `SimulationBridge`.
Separately, the RECALL COMPOSER (the RF-phasor VSA composer / the `/api/brain-chat` recall organ) was de-risked
onto ONE bridge WITH the surprise organ (`2026-08-13-onebrain-composer-merge-GO.md`), and its RF-phasor recall
was made to DRIVE a cross-organ synapse through a PHASE->SPIKE TRANSDUCER cleanup region
(`2026-08-13-onebrain-composer-transducer-GO.md`). This runner DE-RISKS the natural next step: the composer
(+ its transducer cleanup region) joining production pool #1 — i.e. the COMPOSER + SURPRISE + WORLD-MODEL all
on ONE shared pool, one `cp_membrane_potential_v`, all three reads byte-identical merged-vs-co-resident, the
moat preserved, and the recall->surprise cross-organ interaction still load-bearing WITH the world-model in the
pool. A DE-RISK, NOT a production flip.

THE FOUR CODES ON ONE POOL
--------------------------
* COMPOSER (recall): the production `RFPhasorComposer` on a masked SLICE (`SharedBridgeComposer`, the CAPSTONE
  index-shift port). RF-phasor resonate-and-fire ops bypass `_run_one_simulation_step`; masked writes touch ONLY
  the composer slice; the no-confab MOAT abstains on unstored cues.
* CLEANUP (phase->spike transducer): V word-blocks of Izhikevich WTA neurons on the shared pool. Driven by the
  recall's input-normalized matched-filter scores; the winner block SPIKES -> a genuine Izhikevich spike rate ->
  the same-code `cleanup->surprise` synapse (the recall drives the cross-organ edge; the transducer GO).
* SURPRISE (expectation-violation): the D2 organ (cue -> patient_expected(FS,GABA_A) -> surprise <-
  patient_asserted(exc)), Izhikevich + Hebbian + homeostasis + the merge flags.
* WORLD-MODEL (affective forward model): the E2 organ (state --learned-transition--> pred_{pos,neg}(FS,GABA_A);
  obs_{pos,neg}(exc) -> surprise_{pos,neg} <- pred_{pos,neg}(inh)), the same Izhikevich/Hebbian/homeostasis
  config. Its firing IS the affective prediction-error; the queryable predicted-valence sign is a spike-rate
  difference. DISJOINT region names from the surprise organ (state / pred_* / obs_* / surprise_*).

WHY BYTE-IDENTITY HOLDS (the mechanism, inherited from pool #1 + the composer merge)
-----------------------------------------------------------------------------------
The four organs read through DIFFERENT machinery on the SAME `cp_membrane_potential_v`:
  - The composer's RF ops never call `_run_one_simulation_step` and (masked) write only the composer slice, so
    the recall + moat are invariant to the three Izhikevich organs -> byte-identical to a standalone composer.
  - The SURPRISE and WORLD-MODEL organs have DISJOINT region names, NO cross synapse (in the byte-identity
    config), and both merge flags ON: `per_region_threshold_heterogeneity` makes each slice's per-neuron init
    NAME-keyed (invariant to co-residents / build order), and `per_region_homeostasis_isolation` freezes an idle
    co-resident's neurons so they do not drift while the active organ is read. Each organ trains + reads ONLY its
    own regions, so on the shared pool every read reproduces the co-resident-with-flags (standalone) read
    bit-for-bit. Verified per-seed (merged-vs-co-resident deltas; expect 0.0). The world-model byte-identity is
    checked on a SEPARATE fresh merged bridge trained ONLY on the world-model (Hebbian during surprise training
    could otherwise drift the world-model's plastic state->pred edges — a training-order confound, not a merge
    failure; the clean claim is per-organ isolation, exactly pool #1's protocol).

THE CROSS-ORGAN SYNAPSE (the recall drives surprise, WITH the world-model in the pool)
-------------------------------------------------------------------------------------
On a bridge WITH the `cleanup->surprise` edge, the composer's RF-phasor recall of fact i drives the cleanup WTA
(the transducer): the winner block SPIKES and the topographic `cleanup->surprise` synapse raises surprise block
i. LOAD-BEARING: lesion `cleanup->surprise` -> the interaction collapses (attribution frac ~1.0). This
reproduces the transducer GO with the world-model organ ALSO co-resident in the pool.

VERDICT
-------
GO if: one shared pool (composer + cleanup + surprise + world-model in one `cp_membrane_potential_v`) +
determinism (cfg.seed) + composer recall & moat byte-identical + surprise read byte-identical + world-model read
byte-identical (max delta 0.0 on all three) + both spiking faculties still SEPARATE (alive) + a load-bearing
recall->surprise cross-organ synapse. -> the composer can join production pool #1. Else: the mapped obstacle +
the named feature.

NO `sim/` edit; reuse-by-import; CPU-friendly (numpy). Run:
    SIM_BACKEND=numpy python -m research.runners._onebrain_composer_pool1_merge_derisk \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_composer_pool1_merge_6seed.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners._onebrain_composer_merge_derisk import (
    SharedBridgeComposer, FACTS, VOCAB, UNSTORED_CUE, _SURPRISE_KW, _SURPRISE_REGIONS,
    _surp_idx_map, _arr_hash, _maxerr_lists, _install_full_pathway_weight,
)
from research.runners._onebrain_composer_transducer_derisk import (
    PATIENT_WIDX, _transducer_drive, _install_mapped_blocks, _confirm_surprise_transducer,
)
from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit, train_expectation, measure_conditions,
    _idx, _install_block_diagonal, _host,
)
from research.runners import _affective_world_model_derisk as WM
from research.runners.rf_phasor_composer import RFPhasorComposer

# The world-model build parameter (must match WorldModelProductionOrgan / pool #1's default).
_WORLDMODEL_KW = dict(n_states=6)
_WORLDMODEL_REGIONS = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg",
                       "surprise_pos", "surprise_neg")

_PER_NEURON_STATE = (
    "cp_membrane_potential_v", "cp_recovery_variable_u",
    "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
    "cp_firing_states", "cp_prev_firing_states", "cp_refractory_timers", "cp_refractory",
    "cp_neuron_firing_thresholds", "cp_neuron_activity_ema", "cp_external_input_current",
)


@contextlib.contextmanager
def restore_except(bridge, active_idx, xp):
    """Snapshot the FULL per-neuron state, run the block, then RESTORE every neuron NOT in `active_idx` (the
    co-resident organs' slices). So a read of the active organ leaves the CO-RESIDENTS' persistent neural state
    exactly as it was -- there is no cross synapse in the byte-identity config, so this only guards against an
    incidental homeostatic footprint carried into a LATER read on the same bridge (the composer read after the
    surprise read). Generalizes `_onebrain_composer_merge_derisk.restore_composer_slice` to N co-residents."""
    keep = xp.zeros(int(bridge.cp_membrane_potential_v.shape[0]), dtype=bool)
    keep[xp.asarray(active_idx)] = True   # True over the ACTIVE organ -> it self-adapts; False elsewhere -> restored
    snaps = []
    for name in _PER_NEURON_STATE:
        arr = getattr(bridge, name, None)
        snaps.append(None if arr is None else arr.copy())
    try:
        yield
    finally:
        for name, snap in zip(_PER_NEURON_STATE, snaps):
            if snap is None:
                continue
            cur = getattr(bridge, name)
            setattr(bridge, name, xp.where(keep, cur, snap))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Build the merged bridge: any subset of {surprise, worldmodel, composer, cleanup} on ONE pool.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def build_merged_pool1(seed, D_cmp, cblk, *, organs=("surprise", "worldmodel", "composer", "cleanup"),
                       with_c2s=False, cross_weight=8.0):
    """ONE `SimulationBridge` holding a selectable UNION of the four organs' regions. The global config
    replicates `_onebrain_composer_merge_derisk.build_merged` / `..._transducer_derisk.build_transducer` exactly
    (Izhikevich, GENERIC_UNSTRUCTURED, Hebbian, homeostasis, GABA_B inert, the two merge flags) -- and it is
    identical to pool #1's config where the world-model's byte-identity depends on it (GABA_B inert makes the
    world-model's unset gabab tau/prop don't-cares). `organs` selects which slices are present: the full tuple
    for the real 4-way pool, or a single-organ tuple for the byte-identity CO-RESIDENT baseline (an organ alone
    on the SAME construction path + flags -> merged-vs-solo isolates the merge itself). `with_c2s` adds the
    word->fact-mapped `cleanup->surprise` cross synapse (the recall-driven cross-organ edge). NO `sim/` edit."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel

    # SPEC EXTRACTION (reuse-by-import): throwaway standalone builds -> the real region/pathway specs + meta.
    _brS, cfgS, metaS = build_expectation_circuit(seed, per_region_thresh=True, **_SURPRISE_KW)
    _brW, cfgW, metaW = WM.build_world_model_circuit(seed, **_WORLDMODEL_KW)
    surp_blk = metaS["blk"]; wm_blk = metaW["blk"]
    cmp_n = max(7, 2 * len(FACTS)) * D_cmp
    V = len(VOCAB)

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.per_region_threshold_heterogeneity = True    # merge flag #1 (INIT byte-identity)
    cfg.per_region_homeostasis_isolation = True      # merge flag #2 (idle-drift byte-identity)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = 0.06
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_max_weight = 45.0
    cfg.hebbian_weight_decay = 0.0
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = 0.85
    cfg.hebbian_coactivity_thresh = 0.20
    cfg.hebbian_mean_subtract = 1.0
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.22
    cfg.gabab_conductance_max = 0.0                  # GABA_B inert in ALL organs -> tau/prop are don't-cares
    cfg.enable_homeostasis = True

    regions = []; pathways = []
    if "surprise" in organs:
        regions += list(cfgS.brain_regions)
        pathways += list(cfgS.region_pathways)
    if "worldmodel" in organs:
        regions += list(cfgW.brain_regions)
        pathways += list(cfgW.region_pathways)
    if "composer" in organs:
        regions += [BrainRegion(name="composer", n_neurons=cmp_n, exc_fraction=1.0, internal_density=0.0,
                                exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    if "cleanup" in organs:
        regions += [BrainRegion(name="cleanup", n_neurons=V * cblk, exc_fraction=1.0, internal_density=0.0,
                                exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    add_c2s = bool(with_c2s and "cleanup" in organs and "surprise" in organs)
    if add_c2s:
        pathways += [RegionPathway(from_region="cleanup", to_region="surprise",
                                   density=1.0, weight_mean=float(cross_weight), weight_jitter=0.0, plastic=False)]

    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Surprise organ's topographic block-diagonal wiring (world-model needs none: its pathways stay as declared).
    if "surprise" in organs:
        _install_block_diagonal(bridge, "patient_asserted", "surprise", surp_blk, metaS["W_exc"])
        _install_block_diagonal(bridge, "patient_expected", "surprise", surp_blk, metaS["W_inh"])
        _install_block_diagonal(bridge, "cue", "patient_expected", surp_blk,
                                float(_SURPRISE_KW["cue_to_expected_weight"]))
    if add_c2s:
        pairs = [(PATIENT_WIDX[i], i) for i in range(len(FACTS))]   # cleanup word-block(patient_i) -> surprise fact-block i
        _install_mapped_blocks(bridge, "cleanup", "surprise", cblk, surp_blk, pairs, float(cross_weight))

    bridge._surp_blk = surp_blk; bridge._wm_blk = wm_blk; bridge._cblk = cblk
    bridge._blk = surp_blk if "surprise" in organs else wm_blk
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge, cfg, {"surprise": metaS, "worldmodel": metaW}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  One seed.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, *, D_cmp=64, cblk=24, n_reps=22, cross_weight=8.0, win_pa=600.0, verbose=True):
    from sim.backend import get_backend
    from tools.lab import attributable_to, void_if
    xp, _ = get_backend()

    # ── DETERMINISM: two FRESH full-pool builds at one seed -> identical substrate. ──
    d1, _, _ = build_merged_pool1(seed, D_cmp, cblk)
    d2, _, _ = build_merged_pool1(seed, D_cmp, cblk)
    det_ok = (_arr_hash(d1.cp_membrane_potential_v) == _arr_hash(d2.cp_membrane_potential_v)
              and _arr_hash(d1.cp_connections.tocsr().data) == _arr_hash(d2.cp_connections.tocsr().data)
              and _arr_hash(d1.cp_neuron_firing_thresholds) == _arr_hash(d2.cp_neuron_firing_thresholds))

    # ── THE 4-WAY MERGED BRIDGE (byte-identity config: NO cross edge). One pool check. ──
    merged, cfg_m, meta = build_merged_pool1(seed, D_cmp, cblk)
    cmp_idx = _idx(merged, "composer")
    cleanup_idx = _idx(merged, "cleanup")
    surp_idx = _surp_idx_map(merged, xp)
    wm_idx = {n: xp.asarray(_idx(merged, n)) for n in _WORLDMODEL_REGIONS}
    n_all = int(merged.core_config.num_neurons)
    n_surp = sum(len(_host(surp_idx[r])) for r in surp_idx)
    n_wm = sum(len(_host(wm_idx[r])) for r in _WORLDMODEL_REGIONS)
    n_cmp = len(cmp_idx); n_cl = len(cleanup_idx)
    v = merged.cp_membrane_potential_v
    one_pool = bool(int(v.shape[0]) == n_all and n_all == n_surp + n_wm + n_cmp + n_cl
                    and int(cmp_idx.max()) < n_all and int(cleanup_idx.max()) < n_all
                    and (int(cmp_idx.max()) - int(cmp_idx.min()) + 1 == n_cmp)
                    and (int(cleanup_idx.max()) - int(cleanup_idx.min()) + 1 == n_cl)
                    and all(int(_host(surp_idx[r]).max()) < n_all for r in surp_idx)
                    and all(int(_host(wm_idx[r]).max()) < n_all for r in _WORLDMODEL_REGIONS))

    # ── (1) SURPRISE READ byte-identity: train + read on the 4-way pool vs the surprise organ ALONE (same
    #    construction path + merge flags = the co-resident baseline). The composer/cleanup/world-model slices are
    #    idle (undriven, frozen by homeo-iso, no cross edge) -> restore them so the composer read below is pristine. ──
    merged._blk = merged._surp_blk
    train_expectation(merged, cfg_m, surp_idx, meta["surprise"], xp, n_reps=n_reps)
    cfg_m.enable_hebbian_learning = False
    surp_active = np.concatenate([np.asarray(_host(surp_idx[r])) for r in _SURPRISE_REGIONS])
    with restore_except(merged, surp_active, xp):
        resM = measure_conditions(merged, cfg_m, surp_idx, meta["surprise"], xp)

    co_s, cfg_cs, meta_cs = build_merged_pool1(seed, D_cmp, cblk, organs=("surprise",))
    co_s._blk = co_s._surp_blk
    sidx = _surp_idx_map(co_s, xp)
    train_expectation(co_s, cfg_cs, sidx, meta_cs["surprise"], xp, n_reps=n_reps)
    cfg_cs.enable_hebbian_learning = False
    resS = measure_conditions(co_s, cfg_cs, sidx, meta_cs["surprise"], xp)

    surprise_maxerr = _maxerr_lists(resM, resS, ["confirm_per", "contradict_per", "novel_per"])
    surprise_byte_id = bool(surprise_maxerr <= 1e-9)
    surp_sep = resM["contradict_hz"] / max(resM["confirm_hz"], 1e-6)
    surp_alive = bool(surp_sep >= 5.0)

    # ── (2) COMPOSER RECALL + MOAT byte-identity: the shared-slice composer on the 4-way pool vs a standalone
    #    RFPhasorComposer. INTERLEAVE ISOLATION: a composer store+query must leave the surprise slice byte-identical. ──
    iso = RFPhasorComposer(seed=seed, D=D_cmp, vocab=VOCAB)
    for a, vb, p in FACTS:
        iso.store(a, vb, p)
    iso_ans = [iso.query_patient(a, vb) for a, vb, p in FACTS]
    iso_abstain = iso.query_patient(*UNSTORED_CUE)

    sh = SharedBridgeComposer(seed=seed, D=D_cmp, vocab=VOCAB)
    sh.bind_to_shared(merged, cmp_idx)
    _snap = {nm: np.asarray(_host(getattr(merged, nm)))[surp_active].copy()
             for nm in ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_neuron_firing_thresholds")
             if getattr(merged, nm, None) is not None}
    for a, vb, p in FACTS:
        sh.store(a, vb, p)
    sh_ans = [sh.query_patient(a, vb) for a, vb, p in FACTS]
    sh_abstain = sh.query_patient(*UNSTORED_CUE)
    interleave_maxerr = 0.0
    for nm, before in _snap.items():
        after = np.asarray(_host(getattr(merged, nm)))[surp_active]
        interleave_maxerr = max(interleave_maxerr, float(np.abs(after - before).max()))
    composer_op_isolated = bool(interleave_maxerr <= 1e-9)
    recall_byte_id = bool(sh_ans == iso_ans)
    moat_preserved = bool(sh_abstain is None and iso_abstain is None and sh_abstain == iso_abstain)
    recall_correct = bool(sh_ans == [p for _a, _v, p in FACTS])

    # ── (3) WORLD-MODEL READ byte-identity: a SEPARATE fresh 4-way pool trained ONLY on the world-model (avoids
    #    the surprise-training Hebbian drift of the world-model's plastic state->pred edges) vs the world-model
    #    ALONE on the same construction path + flags (the co-resident baseline). ──
    mergedB, cfg_mB, metaB = build_merged_pool1(seed, D_cmp, cblk)
    mergedB._blk = mergedB._wm_blk
    wmB_idx = {n: xp.asarray(_idx(mergedB, n)) for n in _WORLDMODEL_REGIONS}
    v_true = WM._valence_map(seed, metaB["worldmodel"]["n_states"])
    WM.train_transition(mergedB, cfg_mB, wmB_idx, metaB["worldmodel"], xp, v_true, n_reps=n_reps)
    cfg_mB.enable_hebbian_learning = False
    resWM_m = WM.measure(mergedB, cfg_mB, wmB_idx, metaB["worldmodel"], xp, v_true)

    co_w, cfg_cw, meta_cw = build_merged_pool1(seed, D_cmp, cblk, organs=("worldmodel",))
    co_w._blk = co_w._wm_blk
    wS_idx = {n: xp.asarray(_idx(co_w, n)) for n in _WORLDMODEL_REGIONS}
    WM.train_transition(co_w, cfg_cw, wS_idx, meta_cw["worldmodel"], xp, v_true, n_reps=n_reps)
    cfg_cw.enable_hebbian_learning = False
    resWM_s = WM.measure(co_w, cfg_cw, wS_idx, meta_cw["worldmodel"], xp, v_true)

    wm_maxerr = _maxerr_lists(resWM_m, resWM_s, ["expected_per", "violated_per"])
    # ANTI-CHEAT: an EMPTY per-condition list makes _maxerr_lists return 0.0 spuriously -> byte-identity is
    # UNDEFINED, not a pass. VOID both byte-id arms if any read produced no measurements.
    byte_id_void = void_if(
        len(resM["confirm_per"]) == 0 or len(resM["contradict_per"]) == 0 or len(resM["novel_per"]) == 0
        or len(resWM_m["expected_per"]) == 0 or len(resWM_m["violated_per"]) == 0,
        "a byte-identity read produced an EMPTY per-condition list (UNDEFINED, not a 0.0 delta)")
    wm_byte_id = bool(not byte_id_void and wm_maxerr <= 1e-9 and resWM_m["pred_acc"] == resWM_s["pred_acc"])
    wm_ratio = resWM_m["violated_hz"] / max(resWM_m["expected_hz"], 1e-6)
    wm_alive = bool(wm_ratio >= 3.0 and resWM_m["violated_hz"] >= 5.0 and resWM_m["pred_acc"] >= 5.0 / 6.0)

    # ── (4) THE RECALL-DRIVEN CROSS-ORGAN SYNAPSE, WITH the world-model in the pool. On a bridge WITH the
    #    cleanup->surprise edge, the composer's recall of fact i drives the cleanup WTA (transducer) -> the winner
    #    block SPIKES -> surprise block i rises. LOAD-BEARING: lesion cleanup->surprise -> collapse. ──
    xb, cfg_x, meta_x = build_merged_pool1(seed, D_cmp, cblk, with_c2s=True, cross_weight=cross_weight)
    xb._blk = xb._surp_blk
    xsurp = _surp_idx_map(xb, xp)
    xcmp = _idx(xb, "composer"); xcl = _idx(xb, "cleanup")
    train_expectation(xb, cfg_x, xsurp, meta_x["surprise"], xp, n_reps=n_reps)
    cfg_x.enable_hebbian_learning = False
    shx = SharedBridgeComposer(seed=seed, D=D_cmp, vocab=VOCAB)
    shx.bind_to_shared(xb, xcmp)
    for a, vb, p in FACTS:
        shx.store(a, vb, p)
    drives = []; winners_ok = True
    for i, (a, vb, p) in enumerate(FACTS):
        mi = shx._scan_first_match(agent=a, action=vb)
        comp = shx.kb[mi][1] if mi is not None else None
        dv, win = _transducer_drive(shx, comp) if comp is not None else (None, None)
        drives.append(dv)
        winners_ok = winners_ok and (win == PATIENT_WIDX[i])
    inter_recall = []; cl_hz_recall = []
    for i in range(len(FACTS)):
        base_hz, _ = _confirm_surprise_transducer(xb, xsurp, xp, xcl, cblk, drive_vec=None, fact=i, win_pa=win_pa)
        rec_hz, cl_hz = _confirm_surprise_transducer(xb, xsurp, xp, xcl, cblk, drive_vec=drives[i], fact=i, win_pa=win_pa)
        inter_recall.append(rec_hz - base_hz); cl_hz_recall.append(cl_hz)
    _install_full_pathway_weight(xb, "cleanup", "surprise", 0.0)   # LESION
    inter_recall_lesion = []
    for i in range(len(FACTS)):
        base_hz, _ = _confirm_surprise_transducer(xb, xsurp, xp, xcl, cblk, drive_vec=None, fact=i, win_pa=win_pa)
        rec_hz, _ = _confirm_surprise_transducer(xb, xsurp, xp, xcl, cblk, drive_vec=drives[i], fact=i, win_pa=win_pa)
        inter_recall_lesion.append(rec_hz - base_hz)
    interaction_recall = float(np.mean(inter_recall))
    interaction_recall_lesion = float(np.mean(inter_recall_lesion))
    cl_fired = bool(np.mean(cl_hz_recall) >= 1.0)
    recall_frac = attributable_to("recall->surprise via the phase->spike transducer (world-model co-resident)",
                                  interaction_recall, interaction_recall_lesion)
    recall_drives_edge = bool(interaction_recall >= 5.0
                              and interaction_recall >= 5.0 * max(abs(interaction_recall_lesion), 1e-6)
                              and cl_fired and winners_ok
                              and (recall_frac is None or recall_frac >= 0.8))

    three_way_byte_id = bool(not byte_id_void and surprise_byte_id and recall_byte_id and moat_preserved
                             and recall_correct and wm_byte_id and composer_op_isolated)
    merge_go = bool(one_pool and det_ok and three_way_byte_id and surp_alive and wm_alive)
    pool1_go = bool(merge_go and recall_drives_edge)

    res = {
        "seed": seed, "D_cmp": D_cmp, "cblk": cblk, "cross_weight": cross_weight, "win_pa": win_pa,
        "one_shared_pool": one_pool, "n_all": n_all,
        "n_surp": n_surp, "n_wm": n_wm, "n_cmp": n_cmp, "n_cleanup": n_cl,
        "determinism_ok": det_ok,
        # (1) surprise
        "surprise_maxerr_hz": float(surprise_maxerr), "surprise_byte_identical": surprise_byte_id,
        "surprise_separation_ratio": float(surp_sep), "surprise_faculty_alive": surp_alive,
        "surprise_merged": {k: resM[k] for k in ("confirm_hz", "contradict_hz", "novel_hz")},
        # (2) composer
        "composer_recall_shared": sh_ans, "composer_recall_isolated": iso_ans,
        "composer_recall_byte_identical": recall_byte_id, "composer_recall_correct": recall_correct,
        "moat_shared_abstain": sh_abstain, "moat_preserved": moat_preserved,
        "interleave_maxerr": float(interleave_maxerr), "composer_op_isolated": composer_op_isolated,
        # (3) world-model
        "wm_maxerr_hz": float(wm_maxerr), "wm_byte_identical": wm_byte_id,
        "wm_ratio": float(wm_ratio), "wm_faculty_alive": wm_alive,
        "wm_merged": {k: resWM_m[k] for k in ("expected_hz", "violated_hz", "pred_acc")},
        # (4) cross-organ synapse (recall -> surprise via the transducer, world-model co-resident)
        "transducer_winners_ok": bool(winners_ok), "cleanup_fired_hz": float(np.mean(cl_hz_recall)),
        "cleanup_fired": cl_fired,
        "interaction_recall_hz": interaction_recall, "interaction_recall_lesion_hz": interaction_recall_lesion,
        "recall_attribution_frac": (float(recall_frac) if recall_frac is not None else None),
        "recall_drives_edge": recall_drives_edge,
        # verdicts
        "three_way_byte_identical": three_way_byte_id,
        "merge_go": merge_go, "pool1_go": pool1_go,
    }
    if verbose:
        print(f"  [seed {seed}] pool={one_pool}(N={n_all}={n_surp}s+{n_wm}wm+{n_cmp}c+{n_cl}cl) det={det_ok} | "
              f"BYTE-ID surp={surprise_maxerr:.1e}({surprise_byte_id}) "
              f"recall {sh_ans}=={iso_ans}->{recall_byte_id} moat={moat_preserved} "
              f"wm={wm_maxerr:.1e}({wm_byte_id}) op-iso={composer_op_isolated} | "
              f"ALIVE surp_sep={surp_sep:.1f}x({surp_alive}) wm_ratio={wm_ratio:.1f}x({wm_alive}) | "
              f"CROSS recall->surprise intact={interaction_recall:+.2f} lesion={interaction_recall_lesion:+.2f}Hz "
              f"cl_fired={np.mean(cl_hz_recall):.1f}Hz frac={recall_frac} DRIVES={recall_drives_edge} | "
              f"POOL1-GO={pool1_go}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--D-cmp", type=int, default=64)
    ap.add_argument("--cblk", type=int, default=24)
    ap.add_argument("--n-reps", type=int, default=22)
    ap.add_argument("--cross-weight", type=float, default=8.0)
    ap.add_argument("--win-pa", type=float, default=600.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== ONE-BRAIN MERGE: the COMPOSER (+ transducer cleanup) joins POOL #1 (SURPRISE + WORLD-MODEL) on ONE pool ===")
    results = [run_seed(s, D_cmp=args.D_cmp, cblk=args.cblk, n_reps=args.n_reps,
                        cross_weight=args.cross_weight, win_pa=args.win_pa) for s in seeds]

    n = len(results)
    def cnt(k):
        return sum(1 for r in results if r[k])
    n_pool = cnt("one_shared_pool"); n_det = cnt("determinism_ok")
    n_surp = cnt("surprise_byte_identical"); n_salive = cnt("surprise_faculty_alive")
    n_recall = cnt("composer_recall_byte_identical"); n_correct = cnt("composer_recall_correct")
    n_moat = cnt("moat_preserved"); n_isol = cnt("composer_op_isolated")
    n_wm = cnt("wm_byte_identical"); n_walive = cnt("wm_faculty_alive")
    n_3way = cnt("three_way_byte_identical")
    n_merge = cnt("merge_go"); n_cross = cnt("recall_drives_edge"); n_go = cnt("pool1_go")
    max_surp_err = max(r["surprise_maxerr_hz"] for r in results)
    max_wm_err = max(r["wm_maxerr_hz"] for r in results)
    _gate = lambda k: "GO" if ((n >= 6 and k >= 5) or (n < 6 and k == n)) else "BOUNDARY"

    print("\n=== VERDICT ===")
    print(f"  one shared neuron pool (surprise+world-model+composer+cleanup): {n_pool}/{n}")
    print(f"  determinism (cfg.seed incl. thresholds):                        {n_det}/{n}")
    print(f"  SURPRISE read byte-identical (merged vs co-resident):           {n_surp}/{n}  (max err {max_surp_err:.2e} Hz)")
    print(f"    surprise faculty alive (contradict>>confirm):                 {n_salive}/{n}")
    print(f"  COMPOSER recall byte-identical + correct:                       {n_recall}/{n} + {n_correct}/{n}")
    print(f"  no-confab MOAT preserved (unstored -> abstain):                 {n_moat}/{n}")
    print(f"  composer op byte-ISOLATED from surprise slice:                  {n_isol}/{n}")
    print(f"  WORLD-MODEL read byte-identical (merged vs co-resident):        {n_wm}/{n}  (max err {max_wm_err:.2e} Hz)")
    print(f"    world-model faculty alive (violated>>expected):              {n_walive}/{n}")
    print(f"  --> THREE-WAY BYTE-IDENTITY (composer+surprise+world-model):    {n_3way}/{n}  -> {_gate(n_3way)}")
    print(f"  --> MERGE GO (byte-id + both faculties alive):                  {n_merge}/{n}  -> {_gate(n_merge)}")
    print(f"  RECALL DRIVES the cross-organ synapse (world-model co-resident):{n_cross}/{n}  -> {_gate(n_cross)}")
    print(f"  ==> POOL #1 JOIN GO:                                            {n_go}/{n}  -> {_gate(n_go)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"mode": "onebrain_composer_pool1_merge", "n_seeds": n,
                       "n_one_shared_pool": n_pool, "n_determinism_ok": n_det,
                       "n_surprise_byte_identical": n_surp, "n_surprise_faculty_alive": n_salive,
                       "n_composer_recall_byte_identical": n_recall, "n_composer_recall_correct": n_correct,
                       "n_moat_preserved": n_moat, "n_composer_op_isolated": n_isol,
                       "n_wm_byte_identical": n_wm, "n_wm_faculty_alive": n_walive,
                       "n_three_way_byte_identical": n_3way,
                       "n_merge_go": n_merge, "n_recall_drives_edge": n_cross, "n_pool1_go": n_go,
                       "max_surprise_maxerr_hz": max_surp_err, "max_wm_maxerr_hz": max_wm_err,
                       "merge_verdict": _gate(n_merge), "cross_verdict": _gate(n_cross),
                       "pool1_verdict": _gate(n_go),
                       "cross_weight": args.cross_weight, "win_pa": args.win_pa, "results": results}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
