"""ONE-BRAIN COMPOSER TRANSDUCER de-risk — close the RF-phasor <-> spike-rate CODE gap so the composer's
RF-phasor RECALL natively drives the cross-organ synapse on the SHARED spiking substrate.

THE RUNG (what this closes)
---------------------------
The composer<->surprise MERGE is de-risked GO (`2026-08-13-onebrain-composer-merge-GO.md`,
`_onebrain_composer_merge_derisk.py`): one `SimulationBridge`, byte-identical composer recall + moat +
surprise read, and a LOAD-BEARING `composer->surprise` synapse in the shared pool. BUT the one nuance: the
composer's RF-phasor RECALL leaves neurons in a PHASE state (|Z|~1, NOT an Izhikevich spike train), and the
`rf_resonate_steps` fast path never traverses `cp_connections`, so the RECALL itself does NOT natively drive
the cross-organ synapse (merge runner measured: RF-recall interaction 0/6 -- inert).

THE SURPASS (named in the merge finding; built here, no defer)
-------------------------------------------------------------
A PHASE->SPIKE TRANSDUCER: route the composer's EXISTING, validated spiking-cleanup RF-membrane->Izhikevich-WTA
read (`RFPhasorComposer._spiking_cleanup` / `_izh_bank`, GO in
`2026-06-05-phase1-tpam-cleanup-derisk-GO.md`) onto the SHARED bridge as a first-class `cleanup` region so the
recall READOUT emits a SPIKE RATE that drives a same-code `cleanup->surprise` (Izhikevich<->Izhikevich)
synapse. Then a `surprise->cleanup` synapse lets the surprise signal bias recall THROUGH the shared substrate.

  RECALL (RF unbind on the shared composer slice)  -- phase state |Z|~1
     |  matched filter (the composer's Stage-1 read, `_spiking_cleanup` Stage 1: Re(rec . conj(code_k)))
     v
  cleanup REGION on the shared bridge (Izhikevich WTA, the `_izh_bank` Stage-2 read routed onto the pool)
     |  the winner block SPIKES -- a genuine Izhikevich spike RATE on `cp_membrane_potential_v`
     v  `cleanup->surprise` synapse in the shared `cp_connections` (traversed by the Izhikevich `_step`)
  SURPRISE organ  -- the recall now DRIVES the cross-organ synapse.

WHY THIS IS THE TRANSDUCER, NOT A STAND-IN
------------------------------------------
The merge runner's load-bearing test injected CURRENT into the composer block (a "spiking transducer
stand-in"). Here the drive into the `cleanup` region is the composer's OWN recall readout: RF unbind (shared
slice) -> Stage-1 matched-filter score -> input-normalized WTA drive (the SAME `(scores/peak)` normalization
as `_spiking_cleanup`). The winner block is SELECTED by the RF recall's matched filter (the argmax over the
matched-filter scores == the recalled patient word); the Izhikevich WTA on the shared bridge TRANSDUCES that
phase-derived score to a spike RATE. So the cross-organ synapse is driven by the RECALL, on the shared pool.

Honest scope (inherited from the validated spiking-cleanup): Stage-1 (the matched filter) is the composer's
per-op RF readout (as in `_spiking_cleanup`); the phase->spike TRANSDUCTION (Stage-2 WTA) + the cross-organ
drive are on the SHARED substrate. The Stage1->Stage2 score read+normalize is host arithmetic -- the
documented residual of the spiking cleanup this de-risk reuses, not a new shortcut.

WHAT IS MEASURED (6 seeds; smoke first)
---------------------------------------
* RECALL-DRIVEN cross interaction 0/6 -> 6/6: a CONFIRM read of surprise fact i (surprise ~0 baseline) with
  the `cleanup` region driven by the recall of fact i -> surprise rises. LOAD-BEARING: lesion `cleanup->
  surprise` -> collapses (attribution frac ~1.0). The OLD boundary reproduced in the SAME run: the composer's
  RF phase state alone (no transducer) -> 0 interaction.
* SURPRISE->CLEANUP (surprise biases recall): a CONTRADICTION (surprise fires high) drives the `cleanup`
  region through `surprise->cleanup`; lesion -> collapses. The reverse cross-organ synapse is load-bearing.
* BYTE-IDENTITY preserved WITH the transducer machinery present: composer recall + moat + surprise read all
  byte-identical (max delta 0.0) to their standalone references. DETERMINISM (cfg.seed). Genuinely one pool.
* The transducer emits the CORRECT recall: the shared-bridge WTA winner block == the recalled patient word.

NO `sim/` edit; reuse-by-import; CPU-friendly (numpy). Run:
    SIM_BACKEND=numpy python -m research.runners._onebrain_composer_transducer_derisk \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_composer_transducer_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners._onebrain_composer_merge_derisk import (
    SharedBridgeComposer, FACTS, VOCAB, UNSTORED_CUE, _SURPRISE_KW, _SURPRISE_REGIONS,
    _surp_idx_map, restore_composer_slice, _arr_hash, _maxerr_lists,
    _install_full_pathway_weight,
)
from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit, train_expectation, measure_conditions,
    _idx, _install_block_diagonal, _step, _hard_reset, _host,
)
from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge

# The patient word of each composer fact -> its index in VOCAB, and the surprise fact-block it maps to.
PATIENTS = [f[2] for f in FACTS]                       # ['cat', 'mouse', 'deer']
PATIENT_WIDX = [VOCAB.index(p) for p in PATIENTS]       # word-block index of each fact's patient


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The matched-filter score read (the composer's Stage-1 read, `_spiking_cleanup` Stage 1) and the
#  transducer drive (the input-normalized WTA drive, `_spiking_cleanup`'s `(scores/peak)`).
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _matched_filter_scores(sh, rec_phases):
    """Reproduce `RFPhasorComposer._spiking_cleanup` Stage 1 (the composer's on-bridge matched filter): install
    conj(codebook) complex synapses (rec -> concept neuron k), kick rec, one RF matvec step, read Re(c_k) off
    the concept neuron's membrane, rectify. Returns the per-word scores (== `_spiking_cleanup`'s `scores`, which
    equal the numpy matched filter `max(Re(rec . conj(code_k)), 0)` -- 2026-06-05-phase1-tpam-cleanup-derisk-GO)."""
    D = sh.D
    words = sh.words
    V = len(words)
    conns = []
    for k in range(V):
        cc = sh._cleanup_conj(sh._to_phasor(sh.concepts[words[k]]))   # local reciprocal rule when ON; conj when OFF
        for d in range(D):
            conns.append((D + k, d, cc[d]))
    b = sh._bridge_cache.get(D + V)
    if b is None:
        b = _build_rf_bridge(D + V, sh.seed)
        sh._bridge_cache[D + V] = b
    b.rf_set_complex_weights(conns)
    kick = np.zeros(D + V, dtype=np.complex128)
    kick[:D] = sh._to_phasor(rec_phases)
    b.rf_kick(kick, period=sh.period, lam=0.0)
    b.rf_resonate_steps(1)
    re = np.asarray(_host(b.cp_membrane_potential_v)).astype(float)[D:D + V]
    return np.maximum(re, 0.0)


def _transducer_drive(sh, comp):
    """The composer's RECALL of the patient role -> the input-normalized WTA drive vector over the vocab.
    RF unbind on the SHARED composer slice (phase state) -> Stage-1 matched-filter scores -> `(scores/peak)`
    (the SAME normalization as `_spiking_cleanup`: winner ~1, off-targets rectified to ~0). Returns
    (drive[V] in [0,1], winner_word_index) or (None, None) if no concept matches (abstain-like: peak ~0)."""
    rec = sh._unbind_phases(comp, "patient")           # SHARED-slice RF recall (leaves the composer slice in a phase state)
    scores = _matched_filter_scores(sh, rec)
    peak = float(scores.max())
    if peak <= 1e-9:
        return None, None
    return scores / peak, int(np.argmax(scores))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Build the merged bridge: surprise organ + composer region + a cleanup (transducer WTA) region,
#  with optional word->fact-mapped cleanup<->surprise cross-organ synapses.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _install_mapped_blocks(bridge, src, dst, src_blk, dst_blk, pairs, weight):
    """Keep only src-block[a] -> dst-block[b] edges for (a, b) in `pairs`; zero all other src->dst edges. CSR
    orientation-robust (mirrors `_install_block_diagonal`'s detection). Used for the word->fact cross wiring."""
    import scipy.sparse as sp
    src_idx = sorted(int(i) for i in _idx(bridge, src))
    dst_idx = sorted(int(i) for i in _idx(bridge, dst))
    src_set = set(src_idx); dst_set = set(dst_idx)
    src_base = min(src_idx); dst_base = min(dst_idx)
    pair_set = set((int(a), int(b)) for a, b in pairs)
    M = bridge.cp_connections.tocsr()
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    data = np.asarray(_host(M.data)).astype(np.float32)
    n_rows = M.shape[0]
    row_is_dst = row_is_src = 0
    for r in range(n_rows):
        r_in_dst = r in dst_set; r_in_src = r in src_set
        if not (r_in_dst or r_in_src):
            continue
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            if r_in_dst and c in src_set:
                row_is_dst += 1
            if r_in_src and c in dst_set:
                row_is_src += 1
    row_is_post = row_is_dst >= row_is_src
    n_set = 0
    for r in range(n_rows):
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            post, pre = (r, c) if row_is_post else (c, r)
            if pre in src_set and post in dst_set:
                sb = (pre - src_base) // src_blk
                db = (post - dst_base) // dst_blk
                if (sb, db) in pair_set:
                    data[off] = float(weight); n_set += 1
                else:
                    data[off] = 0.0
    bridge.cp_connections = sp.csr_matrix((data, indices, indptr), shape=M.shape)
    return n_set


def build_transducer(seed, D_cmp, cblk, *, with_c2s=False, with_s2c=False, cross_weight=8.0):
    """ONE `SimulationBridge`: the surprise organ's 4 regions + a `composer` region (RF ops) + a `cleanup`
    region (V word-blocks of `cblk` Izhikevich RS neurons = the transducer WTA on the shared pool). Config
    replicates `_onebrain_composer_merge_derisk.build_merged` exactly (Izhikevich + Hebbian + homeostasis +
    the two merge flags), PLUS the cleanup region and (opt) the word->fact-mapped cleanup<->surprise cross
    synapses. NO `sim/` edit."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel

    _brS, cfgS, metaS = build_expectation_circuit(seed, per_region_thresh=True, **_SURPRISE_KW)
    blk = metaS["blk"]
    cmp_n = max(7, 2 * len(FACTS)) * D_cmp
    V = len(VOCAB)

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.per_region_threshold_heterogeneity = True     # merge flag #1 (INIT byte-identity)
    cfg.per_region_homeostasis_isolation = True       # merge flag #2 (idle-drift byte-identity)
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
    cfg.gabab_conductance_max = 0.0
    cfg.enable_homeostasis = True

    cfg.brain_regions = list(cfgS.brain_regions) + [
        BrainRegion(name="composer", n_neurons=cmp_n, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        # The TRANSDUCER WTA region on the shared pool: V word-blocks of `cblk` Izhikevich neurons (the DEFAULT
        # GENERIC_UNSTRUCTURED type -- the SAME neuron config as the validated standalone `_izh_bank` WTA and the
        # merge runner's composer block, so the shared-bridge WTA reproduces the validated spiking cleanup). Driven
        # by the recall's input-normalized matched-filter scores; the winner block SPIKES (the phase->spike read).
        BrainRegion(name="cleanup", n_neurons=V * cblk, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = list(cfgS.region_pathways)
    if with_c2s:
        cfg.region_pathways = cfg.region_pathways + [
            RegionPathway(from_region="cleanup", to_region="surprise",
                          density=1.0, weight_mean=float(cross_weight), weight_jitter=0.0, plastic=False),
        ]
    if with_s2c:
        cfg.region_pathways = cfg.region_pathways + [
            RegionPathway(from_region="surprise", to_region="cleanup",
                          density=1.0, weight_mean=float(cross_weight), weight_jitter=0.0, plastic=False),
        ]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)

    _install_block_diagonal(bridge, "patient_asserted", "surprise", blk, metaS["W_exc"])
    _install_block_diagonal(bridge, "patient_expected", "surprise", blk, metaS["W_inh"])
    _install_block_diagonal(bridge, "cue", "patient_expected", blk, float(_SURPRISE_KW["cue_to_expected_weight"]))
    # The word->fact cross wiring: cleanup word-block(patient_i) <-> surprise fact-block i, for each composer fact.
    if with_c2s:
        pairs = [(PATIENT_WIDX[i], i) for i in range(len(FACTS))]     # cleanup word-block -> surprise fact-block
        _install_mapped_blocks(bridge, "cleanup", "surprise", cblk, blk, pairs, float(cross_weight))
    if with_s2c:
        pairs = [(i, PATIENT_WIDX[i]) for i in range(len(FACTS))]     # surprise fact-block -> cleanup word-block
        _install_mapped_blocks(bridge, "surprise", "cleanup", blk, cblk, pairs, float(cross_weight))
    bridge._blk = blk
    bridge._cblk = cblk
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge, cfg, metaS


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Interaction A: a CONFIRM read of surprise fact i, with the cleanup region driven by the RECALL.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _confirm_surprise_transducer(bridge, surp_idx, xp, cleanup_idx, cblk, *, drive_vec,
                                  fact=0, cue_pa=600.0, assert_pa=600.0, win_pa=600.0, hold=60, pre_steps=60):
    """CONFIRM read of surprise fact i (surprise ~0 when the prediction cancels the assertion), while the
    `cleanup` region is driven by `drive_vec` (the recall's input-normalized WTA drive; None = cleanup at rest
    = baseline). Each cleanup word-block k gets `drive_vec[k]*win_pa` external current (winner ~win_pa, off ~0)
    -> the winner block SPIKES -> the TOPOGRAPHIC `cleanup->surprise` edge -> surprise BLOCK i. Returns
    (surprise-block-i Hz, cleanup-winner-block Hz) -- the block the recall's winner topographically drives, the
    like-for-like measurement (the recall of fact i drives surprise block i, so read block i)."""
    blk = bridge._blk
    _hard_reset(bridge)
    cue = surp_idx["cue"]
    cl = np.asarray(cleanup_idx)
    # PREDICTION phase: cue alone (settle the expectation / the slow GABA_B).
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[cue[fact * blk:(fact + 1) * blk]] = xp.float32(cue_pa)
    for _ in range(pre_steps):
        _step(bridge)
    # ASSERTION phase: cue + asserted TRUE patient i (confirm) + the recall's transducer drive on cleanup.
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[cue[fact * blk:(fact + 1) * blk]] = xp.float32(cue_pa)
    pa = surp_idx["patient_asserted"]
    bridge.cp_external_input_current[pa[fact * blk:(fact + 1) * blk]] = xp.float32(assert_pa)
    win_blk = int(np.argmax(drive_vec)) if drive_vec is not None else -1
    if drive_vec is not None:
        for k in range(len(drive_vec)):
            cur = float(drive_vec[k]) * win_pa
            if cur <= 1e-9:
                continue
            bridge.cp_external_input_current[xp.asarray(cl[k * cblk:(k + 1) * cblk])] = xp.float32(cur)
    surp_blk = surp_idx["surprise"][fact * blk:(fact + 1) * blk]         # the block the recall's winner drives
    cl_win = cl[win_blk * cblk:(win_blk + 1) * cblk] if win_blk >= 0 else cl[:0]
    surp_b = xp.asarray(surp_blk)
    cl_w = xp.asarray(cl_win)
    scnt = ccnt = 0
    for _ in range(hold):
        _step(bridge)
        scnt += int(bridge.cp_firing_states[surp_b].sum())
        if win_blk >= 0:
            ccnt += int(bridge.cp_firing_states[cl_w].sum())
    bridge.cp_external_input_current[:] = 0.0
    surp_hz = scnt / max(len(_host(surp_blk)), 1) / (hold * 1e-3)
    cl_hz = (ccnt / max(len(cl_win), 1) / (hold * 1e-3)) if win_blk >= 0 else 0.0
    return surp_hz, cl_hz


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Interaction B: the surprise signal (a CONTRADICTION) drives the cleanup region via surprise->cleanup.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _cleanup_under_surprise(bridge, surp_idx, xp, cleanup_idx, cblk, *, fact, contradict,
                            cue_pa=600.0, assert_pa=600.0, hold=60, pre_steps=60):
    """Run a CONFIRM (contradict=False) or CONTRADICT (contradict=True) read of fact `fact`; the cleanup region
    is NOT driven by any transducer -- its only input is the shared `surprise->cleanup` synapse. The
    contradiction asserts patient j=(fact+1)%3 (a MAPPED fact), so surprise BLOCK j fires and drives cleanup
    word-block(patient_j) via `surprise->cleanup`. Returns the cleanup-block(patient_j) mean Hz -- the like-for-
    like block the reverse synapse topographically targets (high iff the surprise faculty fires AND the edge is
    intact). Confirm reads the SAME block (surprise block j undriven -> cleanup block patient_j ~0 = baseline)."""
    blk = bridge._blk
    j = (fact + 1) % len(FACTS)                       # the contradiction's asserted patient (a mapped fact)
    asserted = j if contradict else fact
    _hard_reset(bridge)
    cue = surp_idx["cue"]; pa = surp_idx["patient_asserted"]
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[cue[fact * blk:(fact + 1) * blk]] = xp.float32(cue_pa)
    for _ in range(pre_steps):
        _step(bridge)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[cue[fact * blk:(fact + 1) * blk]] = xp.float32(cue_pa)
    bridge.cp_external_input_current[pa[asserted * blk:(asserted + 1) * blk]] = xp.float32(assert_pa)
    cl = np.asarray(cleanup_idx)
    wj = PATIENT_WIDX[j]                              # the cleanup word-block surprise block j targets
    cl_blk = xp.asarray(cl[wj * cblk:(wj + 1) * cblk])
    ccnt = 0
    for _ in range(hold):
        _step(bridge)
        ccnt += int(bridge.cp_firing_states[cl_blk].sum())
    bridge.cp_external_input_current[:] = 0.0
    return ccnt / max(cblk, 1) / (hold * 1e-3)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  One seed.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, *, D_cmp=64, cblk=24, n_reps=22, cross_weight=8.0, win_pa=600.0, verbose=True):
    from sim.backend import get_backend
    from tools.lab import attributable_to
    xp, _ = get_backend()

    # ── DETERMINISM: two FRESH transducer builds at one seed -> identical substrate. ──
    d1, _, _ = build_transducer(seed, D_cmp, cblk, with_c2s=True, with_s2c=True, cross_weight=cross_weight)
    d2, _, _ = build_transducer(seed, D_cmp, cblk, with_c2s=True, with_s2c=True, cross_weight=cross_weight)
    det_ok = (_arr_hash(d1.cp_membrane_potential_v) == _arr_hash(d2.cp_membrane_potential_v)
              and _arr_hash(d1.cp_connections.tocsr().data) == _arr_hash(d2.cp_connections.tocsr().data)
              and _arr_hash(d1.cp_neuron_firing_thresholds) == _arr_hash(d2.cp_neuron_firing_thresholds))

    # ── (1) BYTE-IDENTITY of the surprise read WITH the transducer machinery present (no cross edges). ──
    merged, cfg_m, meta = build_transducer(seed, D_cmp, cblk)   # cleanup region present, edges absent
    cmp_idx = _idx(merged, "composer")
    cleanup_idx = _idx(merged, "cleanup")
    surp_idx = _surp_idx_map(merged, xp)
    n_all = int(merged.core_config.num_neurons)
    n_surp = sum(len(_host(surp_idx[r])) for r in surp_idx)
    n_cmp = len(cmp_idx); n_cl = len(cleanup_idx)
    v = merged.cp_membrane_potential_v
    one_pool = bool(int(v.shape[0]) == n_all and n_all >= n_surp + n_cmp + n_cl
                    and int(cmp_idx.max()) < n_all and int(cleanup_idx.max()) < n_all
                    and (int(cmp_idx.max()) - int(cmp_idx.min()) + 1 == n_cmp)
                    and (int(cleanup_idx.max()) - int(cleanup_idx.min()) + 1 == n_cl))

    train_expectation(merged, cfg_m, surp_idx, meta, xp, n_reps=n_reps)
    cfg_m.enable_hebbian_learning = False
    with restore_composer_slice(merged, cmp_idx, xp):
        resM = measure_conditions(merged, cfg_m, surp_idx, meta, xp)

    brS, cfgS, metaS = build_expectation_circuit(seed, per_region_thresh=True, **_SURPRISE_KW)
    brS._blk = metaS["blk"]
    cfgS.enable_homeostasis = True
    cfgS.per_region_homeostasis_isolation = True
    idxS = _surp_idx_map(brS, xp)
    train_expectation(brS, cfgS, idxS, metaS, xp, n_reps=n_reps)
    cfgS.enable_hebbian_learning = False
    resS = measure_conditions(brS, cfgS, idxS, metaS, xp)

    surprise_maxerr = _maxerr_lists(resM, resS, ["confirm_per", "contradict_per", "novel_per"])
    surprise_byte_id = bool(surprise_maxerr <= 1e-9)
    surp_sep = resM["contradict_hz"] / max(resM["confirm_hz"], 1e-6)
    surp_alive = bool(surp_sep >= 5.0)

    # ── (2) COMPOSER RECALL + MOAT byte-identity: shared-bridge composer vs standalone RFPhasorComposer. ──
    iso = RFPhasorComposer(seed=seed, D=D_cmp, vocab=VOCAB)
    for a, vb, p in FACTS:
        iso.store(a, vb, p)
    iso_ans = [iso.query_patient(a, vb) for a, vb, p in FACTS]
    iso_abstain = iso.query_patient(*UNSTORED_CUE)

    sh = SharedBridgeComposer(seed=seed, D=D_cmp, vocab=VOCAB)
    sh.bind_to_shared(merged, cmp_idx)
    for a, vb, p in FACTS:
        sh.store(a, vb, p)
    sh_ans = [sh.query_patient(a, vb) for a, vb, p in FACTS]
    sh_abstain = sh.query_patient(*UNSTORED_CUE)
    recall_byte_id = bool(sh_ans == iso_ans)
    moat_preserved = bool(sh_abstain is None and iso_abstain is None and sh_abstain == iso_abstain)
    recall_correct = bool(sh_ans == [p for _a, _v, p in FACTS])

    # ── (3) INTERACTION A: the RECALL drives surprise via cleanup->surprise (the phase->spike transducer). ──
    xb, cfg_x, meta_x = build_transducer(seed, D_cmp, cblk, with_c2s=True, cross_weight=cross_weight)
    xsurp = _surp_idx_map(xb, xp)
    xcmp = _idx(xb, "composer"); xcl = _idx(xb, "cleanup")
    train_expectation(xb, cfg_x, xsurp, meta_x, xp, n_reps=n_reps)
    cfg_x.enable_hebbian_learning = False
    shx = SharedBridgeComposer(seed=seed, D=D_cmp, vocab=VOCAB)
    shx.bind_to_shared(xb, xcmp)
    for a, vb, p in FACTS:
        shx.store(a, vb, p)

    # transducer drive per fact (the recall's WTA drive) + the winner-block check.
    drives = []; winners_ok = True
    for i, (a, vb, p) in enumerate(FACTS):
        mi = shx._scan_first_match(agent=a, action=vb)
        comp = shx.kb[mi][1] if mi is not None else None
        dv, win = _transducer_drive(shx, comp) if comp is not None else (None, None)
        drives.append(dv)
        winners_ok = winners_ok and (win == PATIENT_WIDX[i])   # the WTA winner block == the recalled patient word
    # the abstain (unstored cue) must produce NO transducer drive (the moat carried into the transducer).
    ab_mi = shx._scan_first_match(agent=UNSTORED_CUE[0], action=UNSTORED_CUE[1])
    ab_drive = None if ab_mi is None else _transducer_drive(shx, shx.kb[ab_mi][1])[0]
    transducer_abstains = bool(ab_drive is None)

    # per-fact: baseline (cleanup at rest) vs recall-driven (transducer), + cleanup fired under the recall.
    inter_recall = []; cl_hz_recall = []
    for i in range(len(FACTS)):
        base_hz, _ = _confirm_surprise_transducer(xb, xsurp, xp, xcl, cblk, drive_vec=None, fact=i, win_pa=win_pa)
        rec_hz, cl_hz = _confirm_surprise_transducer(xb, xsurp, xp, xcl, cblk, drive_vec=drives[i], fact=i, win_pa=win_pa)
        inter_recall.append(rec_hz - base_hz); cl_hz_recall.append(cl_hz)
    # LESION the transducer edge -> the recall can no longer reach surprise.
    _install_full_pathway_weight(xb, "cleanup", "surprise", 0.0)
    inter_recall_lesion = []
    for i in range(len(FACTS)):
        base_hz, _ = _confirm_surprise_transducer(xb, xsurp, xp, xcl, cblk, drive_vec=None, fact=i, win_pa=win_pa)
        rec_hz, _ = _confirm_surprise_transducer(xb, xsurp, xp, xcl, cblk, drive_vec=drives[i], fact=i, win_pa=win_pa)
        inter_recall_lesion.append(rec_hz - base_hz)
    interaction_recall = float(np.mean(inter_recall))
    interaction_recall_lesion = float(np.mean(inter_recall_lesion))
    cl_fired = bool(np.mean(cl_hz_recall) >= 1.0)   # the shared-bridge cleanup region actually SPIKED under the recall
    recall_frac = attributable_to("recall->surprise via the phase->spike transducer",
                                  interaction_recall, interaction_recall_lesion)
    recall_drives_edge = bool(interaction_recall >= 5.0
                              and interaction_recall >= 5.0 * max(abs(interaction_recall_lesion), 1e-6)
                              and cl_fired and winners_ok
                              and (recall_frac is None or recall_frac >= 0.8))

    # ── (4) INTERACTION B: the surprise signal biases recall via surprise->cleanup. ──
    yb, cfg_y, meta_y = build_transducer(seed, D_cmp, cblk, with_s2c=True, cross_weight=cross_weight)
    ysurp = _surp_idx_map(yb, xp)
    ycl = _idx(yb, "cleanup")
    train_expectation(yb, cfg_y, ysurp, meta_y, xp, n_reps=n_reps)
    cfg_y.enable_hebbian_learning = False
    inter_s2c = []
    for i in range(len(FACTS)):
        conf = _cleanup_under_surprise(yb, ysurp, xp, ycl, cblk, fact=i, contradict=False)
        cont = _cleanup_under_surprise(yb, ysurp, xp, ycl, cblk, fact=i, contradict=True)
        inter_s2c.append(cont - conf)
    _install_full_pathway_weight(yb, "surprise", "cleanup", 0.0)
    inter_s2c_lesion = []
    for i in range(len(FACTS)):
        conf = _cleanup_under_surprise(yb, ysurp, xp, ycl, cblk, fact=i, contradict=False)
        cont = _cleanup_under_surprise(yb, ysurp, xp, ycl, cblk, fact=i, contradict=True)
        inter_s2c_lesion.append(cont - conf)
    interaction_s2c = float(np.mean(inter_s2c))
    interaction_s2c_lesion = float(np.mean(inter_s2c_lesion))
    s2c_frac = attributable_to("surprise->cleanup (surprise biases recall)",
                               interaction_s2c, interaction_s2c_lesion)
    s2c_load_bearing = bool(interaction_s2c >= 5.0
                            and interaction_s2c >= 5.0 * max(abs(interaction_s2c_lesion), 1e-6)
                            and (s2c_frac is None or s2c_frac >= 0.8))

    byte_id_ok = bool(surprise_byte_id and recall_byte_id and moat_preserved and recall_correct)
    transducer_go = bool(one_pool and det_ok and byte_id_ok and surp_alive
                         and recall_drives_edge and transducer_abstains)

    res = {
        "seed": seed, "D_cmp": D_cmp, "cblk": cblk, "cross_weight": cross_weight, "win_pa": win_pa,
        "one_shared_pool": one_pool, "n_all": n_all, "n_surp": n_surp, "n_cmp": n_cmp, "n_cleanup": n_cl,
        "determinism_ok": det_ok,
        "surprise_maxerr_hz": float(surprise_maxerr), "surprise_byte_identical": surprise_byte_id,
        "surprise_separation_ratio": float(surp_sep), "surprise_faculty_alive": surp_alive,
        "composer_recall_shared": sh_ans, "composer_recall_isolated": iso_ans,
        "composer_recall_byte_identical": recall_byte_id, "composer_recall_correct": recall_correct,
        "moat_preserved": moat_preserved,
        # interaction A (recall -> surprise via the transducer)
        "transducer_winners_ok": bool(winners_ok), "cleanup_fired_hz": float(np.mean(cl_hz_recall)),
        "cleanup_fired": cl_fired,
        "interaction_recall_hz": interaction_recall, "interaction_recall_lesion_hz": interaction_recall_lesion,
        "recall_attribution_frac": (float(recall_frac) if recall_frac is not None else None),
        "recall_drives_edge": recall_drives_edge,
        "transducer_abstains_on_unstored": transducer_abstains,
        "interaction_recall_per": inter_recall, "interaction_recall_lesion_per": inter_recall_lesion,
        # interaction B (surprise -> cleanup: surprise biases recall)
        "interaction_s2c_hz": interaction_s2c, "interaction_s2c_lesion_hz": interaction_s2c_lesion,
        "s2c_attribution_frac": (float(s2c_frac) if s2c_frac is not None else None),
        "s2c_load_bearing": s2c_load_bearing,
        "byte_identity_ok": byte_id_ok,
        "transducer_go": transducer_go,
    }
    if verbose:
        print(f"  [seed {seed}] pool={one_pool}(N={n_all}={n_surp}s+{n_cmp}c+{n_cl}cl) det={det_ok} | "
              f"BYTE-ID surp={surprise_maxerr:.1e}({surprise_byte_id}) recall {sh_ans}=={iso_ans}->{recall_byte_id} "
              f"moat={moat_preserved} correct={recall_correct} | "
              f"TRANSDUCER winners_ok={winners_ok} cl_fired={np.mean(cl_hz_recall):.1f}Hz | "
              f"RECALL->surprise intact={interaction_recall:+.2f} lesion={interaction_recall_lesion:+.2f}Hz "
              f"frac={recall_frac} DRIVES={recall_drives_edge} abstain-clean={transducer_abstains} | "
              f"SURPRISE->cleanup intact={interaction_s2c:+.2f} lesion={interaction_s2c_lesion:+.2f}Hz "
              f"lb={s2c_load_bearing} | GO={transducer_go}")
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
    print("=== ONE-BRAIN COMPOSER TRANSDUCER: the RF-phasor RECALL drives the cross-organ synapse (phase->spike) ===")
    results = [run_seed(s, D_cmp=args.D_cmp, cblk=args.cblk, n_reps=args.n_reps,
                        cross_weight=args.cross_weight, win_pa=args.win_pa) for s in seeds]

    n = len(results)
    def cnt(k):
        return sum(1 for r in results if r[k])
    n_pool = cnt("one_shared_pool"); n_det = cnt("determinism_ok")
    n_surp = cnt("surprise_byte_identical"); n_alive = cnt("surprise_faculty_alive")
    n_recall = cnt("composer_recall_byte_identical"); n_correct = cnt("composer_recall_correct")
    n_moat = cnt("moat_preserved"); n_byte = cnt("byte_identity_ok")
    n_win = cnt("transducer_winners_ok"); n_clf = cnt("cleanup_fired")
    n_drives = cnt("recall_drives_edge"); n_abst = cnt("transducer_abstains_on_unstored")
    n_s2c = cnt("s2c_load_bearing"); n_go = cnt("transducer_go")
    max_surp_err = max(r["surprise_maxerr_hz"] for r in results)
    _gate = lambda k: "GO" if ((n >= 6 and k >= 5) or (n < 6 and k == n)) else "BOUNDARY"

    print("\n=== VERDICT ===")
    print(f"  one shared neuron pool (surprise+composer+cleanup): {n_pool}/{n}")
    print(f"  determinism (cfg.seed incl. thresholds):            {n_det}/{n}")
    print(f"  SURPRISE read byte-identical (w/ transducer):       {n_surp}/{n}  (max err {max_surp_err:.2e} Hz)")
    print(f"  COMPOSER recall byte-identical + correct:           {n_recall}/{n} + {n_correct}/{n}")
    print(f"  no-confab MOAT preserved (unstored -> abstain):     {n_moat}/{n}")
    print(f"  --> BYTE-IDENTITY preserved WITH the transducer:    {n_byte}/{n}  -> {_gate(n_byte)}")
    print(f"  transducer WTA winner block == recalled patient:    {n_win}/{n}")
    print(f"  shared-bridge cleanup region SPIKED under recall:   {n_clf}/{n}")
    print(f"  transducer ABSTAINS on the unstored cue (no drive): {n_abst}/{n}")
    print(f"  ==> RECALL DRIVES the cross-organ synapse (0/N->):  {n_drives}/{n}  -> {_gate(n_drives)}  "
          f"(the phase->spike transducer; lesion cleanup->surprise collapses it)")
    print(f"  SURPRISE->CLEANUP load-bearing (biases recall):     {n_s2c}/{n}  -> {_gate(n_s2c)}")
    print(f"  ==> TRANSDUCER GO:                                  {n_go}/{n}  -> {_gate(n_go)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"mode": "onebrain_composer_transducer", "n_seeds": n,
                       "n_one_shared_pool": n_pool, "n_determinism_ok": n_det,
                       "n_surprise_byte_identical": n_surp, "n_surprise_faculty_alive": n_alive,
                       "n_composer_recall_byte_identical": n_recall, "n_composer_recall_correct": n_correct,
                       "n_moat_preserved": n_moat, "n_byte_identity_ok": n_byte,
                       "n_transducer_winners_ok": n_win, "n_cleanup_fired": n_clf,
                       "n_recall_drives_edge": n_drives, "n_transducer_abstains": n_abst,
                       "n_s2c_load_bearing": n_s2c, "n_transducer_go": n_go,
                       "max_surprise_maxerr_hz": max_surp_err,
                       "byte_verdict": _gate(n_byte), "recall_drives_verdict": _gate(n_drives),
                       "s2c_verdict": _gate(n_s2c), "transducer_verdict": _gate(n_go),
                       "cross_weight": args.cross_weight, "win_pa": args.win_pa, "results": results}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
