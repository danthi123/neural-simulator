"""ONE-BRAIN MERGE de-risk — put TWO co-resident organ bridges onto ONE shared spiking substrate.

THE HONEST RECKONING (roadmap §0.2 residual #1 / this arc's mission)
--------------------------------------------------------------------
The production brain is CO-RESIDENCY: each conversational organ (recall / surprise / comprehension /
affect / episodic / ...) is a SEPARATE spiking `SimulationBridge`. The GNW bus now combines their READS
via ignition, but the organs themselves do NOT share ONE neuron pool with cross-region synapses. Genuine
cross-synaptic interaction is proven for exactly ONE pathway (acquisition) and, at the WKV-INTERNAL level,
for the WKV cortex's two internal bridges (2026-07-20-wkv-cortex-physically-merged-onto-one-bridge-GO —
byte-exact, but WITHIN one faculty). This runner takes the next rung: MERGE TWO DISTINCT ORGANS onto ONE
`SimulationBridge` (one shared `cp_` neuron array + a genuine CROSS-ORGAN synapse) and prove:

  (a) BYTE-IDENTITY (or a characterized bounded delta): each organ's spiking read is unchanged
      merged-vs-co-resident when the cross-organ synapse is inert (pure pool co-residence must not
      perturb either organ);
  (b) the CROSS-ORGAN synapse is LOAD-BEARING: with it intact, organ A's state changes a MEASURABLE
      interaction in organ B; lesion it -> the interaction vanishes;
  (c) DETERMINISM: `cfg.seed` set; build-twice-same-seed byte-identical (no unseeded-substrate confound).

THE TWO ORGANS (the mission's named "recall + surprise" pair; both reuse the adversarially-verified
D2 expectation circuit primitives from `_spiking_expectation_rpe_derisk`, 6/6 GO, lesion-decisive)
------------------------------------------------------------------------------------------------------
* Organ A = SURPRISE (expectation-violation): cueA (agent,action) --Hebbian topographic--> patient_expected_A
  (FS/PV-like, the recalled prediction, GABA_A subtractive) ; patient_asserted_A --exc--> surprise_A. The
  surprise_A windowed firing IS the read (CONFIRM cancels ~0 Hz; CONTRADICT/NOVEL fires).
* Organ B = RECALL (heteroassociative memory): cueB --Hebbian topographic--> patient_expected_B (the recalled
  patient). patient_expected_B firing IS the recall read (which/how-strongly a cue recalls its stored patient).

THE CROSS-ORGAN SYNAPSE (biologically motivated: novelty/surprise gates memory recall/encoding — the
LC-NE / hippocampal novelty signal): surprise_A --exc--> cueB. When organ A is SURPRISED (a violated
expectation), the surprise pool's firing adds drive to organ B's recall cue -> organ B recalls MORE
strongly. Lesion the surprise_A->cueB edges -> organ B's recall no longer responds to A's surprise.

WHAT MAKES THIS A GENUINE MERGE (not co-location)
-------------------------------------------------
Both organs' regions are allocated in ONE `SimulationBridge` -> ONE `cp_membrane_potential_v` array holds
BOTH organs' neurons (asserted in-code), stepped by ONE `_run_one_simulation_step`, one `cfg.seed`. The
cross-organ synapse is a real edge in the ONE `cp_connections` matrix. Contrast the co-resident baseline:
two separate bridges, two neuron arrays, no shared matrix.

BYTE-IDENTITY is ACHIEVABLE here because the substrate is deterministic + homogeneous: Izhikevich, NO
parameter heterogeneity, NO homeostasis (so `cp_neuron_firing_thresholds is None` — no per-neuron RNG
draw), NO OU / conductance noise, density=1.0 + jitter=0.0 connectivity (deterministic), Hebbian FROZEN
during reads. So adding organ B's neurons cannot perturb organ A's per-neuron parameters, and with organ
B silent (or the cross inert) organ B cannot feed back into organ A. This is exactly the regime where a
merge SHOULD be byte-clean; if it were not, that divergence would itself be the mapped obstacle.

NO `sim/` edit; reuse-by-import. CPU-friendly (~numpy). Run:
    SIM_BACKEND=numpy python -m research.runners._one_brain_merge_2organ_derisk \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_one_brain_merge_2organ_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit,
    train_expectation,
    measure_conditions,
    _drive_read,
    _hard_reset,
    _idx,
    _install_block_diagonal,
    _step,
    _host,
)


# ── canonical per-organ region-name maps (so we can reuse train_expectation/measure_conditions verbatim) ──
def _idx_map(bridge, suffix, xp):
    return {
        "cue": xp.asarray(_idx(bridge, "cue" + suffix)),
        "patient_expected": xp.asarray(_idx(bridge, "patient_expected" + suffix)),
        "patient_asserted": xp.asarray(_idx(bridge, "patient_asserted" + suffix)),
        "surprise": xp.asarray(_idx(bridge, "surprise" + suffix)),
    }


def build_merged_two_organ(seed, *, n_trained=8, n_novel=4, blk=24, cue_blk=24,
                           cue_to_expected_weight=0.8, asserted_to_surprise_weight=5.0,
                           expected_to_surprise_weight=14.0, gabab_prop=0.22,
                           gabab_tau_decay=150.0, hebbian_learning_rate=0.06,
                           hebbian_max_weight=45.0, cross_weight=12.0,
                           per_region_thresh=True):
    """ONE SimulationBridge holding BOTH organs (suffix _A = surprise, _B = recall) + the cross-organ
    synapse surprise_A -> cueB. Config is byte-identical to build_expectation_circuit (so each organ's
    slice matches its standalone build); only the region/pathway SETS are the union of the two organs."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    n_concepts = n_trained + n_novel
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    # PER-REGION threshold heterogeneity (the merge-closing fix): each region's
    # firing-threshold slice is drawn from a name-keyed substream, so organ B (the
    # SECOND organ in the merged pool) gets the SAME seeded thresholds it would get
    # standalone -- byte-identity is invariant to co-residents. Default ON here;
    # --legacy-global-thresh reproduces the 0/6 BOUNDARY (single global stream).
    cfg.per_region_threshold_heterogeneity = bool(per_region_thresh)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = float(hebbian_learning_rate)
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_max_weight = float(hebbian_max_weight)
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
    cfg.gabab_tau_decay = float(gabab_tau_decay)
    cfg.gabab_propagation_strength = float(gabab_prop)
    cfg.gabab_conductance_max = 0.0

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    FS = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name

    def organ_regions(suffix):
        return [
            BrainRegion(name="cue" + suffix, n_neurons=n_trained * cue_blk, exc_fraction=1.0,
                        internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
            BrainRegion(name="patient_expected" + suffix, n_neurons=n_concepts * blk, exc_fraction=0.0,
                        internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=FS,
                        syn_reversal_potential_i_override=-70.0),
            BrainRegion(name="patient_asserted" + suffix, n_neurons=n_concepts * blk, exc_fraction=1.0,
                        internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
            BrainRegion(name="surprise" + suffix, n_neurons=n_concepts * blk, exc_fraction=1.0,
                        internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
        ]

    def organ_pathways(suffix):
        return [
            RegionPathway(from_region="cue" + suffix, to_region="patient_expected" + suffix,
                          density=1.0, weight_mean=float(cue_to_expected_weight),
                          weight_jitter=0.0, plastic=True),
            RegionPathway(from_region="patient_asserted" + suffix, to_region="surprise" + suffix,
                          density=1.0, weight_mean=float(asserted_to_surprise_weight),
                          weight_jitter=0.0, plastic=False),
            RegionPathway(from_region="patient_expected" + suffix, to_region="surprise" + suffix,
                          density=1.0, weight_mean=float(expected_to_surprise_weight),
                          weight_jitter=0.0, plastic=False),
        ]

    # Organ A first (identical order to a standalone build), then Organ B, then the CROSS-ORGAN synapse.
    cfg.brain_regions = organ_regions("_A") + organ_regions("_B")
    cfg.region_pathways = organ_pathways("_A") + organ_pathways("_B") + [
        # THE CROSS-ORGAN SYNAPSE: surprise_A -> cueB (novelty/surprise gates recall). Fixed (non-plastic).
        RegionPathway(from_region="surprise_A", to_region="cue_B",
                      density=1.0, weight_mean=float(cross_weight),
                      weight_jitter=0.0, plastic=False),
    ]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge._blk = blk

    meta = dict(n_trained=n_trained, n_novel=n_novel, n_concepts=n_concepts, blk=blk,
                cue_blk=cue_blk, W_exc=float(asserted_to_surprise_weight),
                W_inh=float(expected_to_surprise_weight), cross_weight=float(cross_weight))

    # Install the TOPOGRAPHIC block-diagonal wiring for BOTH organs (concept c -> block c).
    for suf in ("_A", "_B"):
        _install_block_diagonal(bridge, "patient_asserted" + suf, "surprise" + suf, blk,
                                float(asserted_to_surprise_weight))
        _install_block_diagonal(bridge, "patient_expected" + suf, "surprise" + suf, blk,
                                float(expected_to_surprise_weight))
        _install_block_diagonal(bridge, "cue" + suf, "patient_expected" + suf, blk,
                                float(cue_to_expected_weight))
    # The cross-organ surprise_A->cueB is left FULL (any A-surprise concept drives B's recall cue): it is a
    # broadcast novelty gate, not a topographic map. cross_weight=0 (lesion) zeroes it entirely.

    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge, cfg, meta


def lesion_cross(bridge):
    """Zero the surprise_A -> cueB cross-organ edges (the load-bearing lesion). Returns (n_kept, n_zeroed)."""
    return _install_block_diagonal_full(bridge, "surprise_A", "cue_B", 0.0)


def _install_block_diagonal_full(bridge, src, dst, weight):
    """Set EVERY src->dst edge to `weight` (used to zero the full cross-organ pathway). Mirrors
    _install_block_diagonal's CSR handling but with no block-diagonal restriction."""
    import numpy as np
    import scipy.sparse as sp
    src_idx = set(int(i) for i in _idx(bridge, src))
    dst_idx = set(int(i) for i in _idx(bridge, dst))
    M = bridge.cp_connections.tocsr()
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    data = np.asarray(_host(M.data)).astype(np.float32)
    n_rows = M.shape[0]
    row_is_dst = 0; row_is_src = 0
    for r in range(n_rows):
        r_in_dst = r in dst_idx; r_in_src = r in src_idx
        if not (r_in_dst or r_in_src):
            continue
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            if r_in_dst and c in src_idx:
                row_is_dst += 1
            if r_in_src and c in dst_idx:
                row_is_src += 1
    row_is_post = row_is_dst >= row_is_src
    n_kept = n_set = 0
    for r in range(n_rows):
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            post, pre = (r, c) if row_is_post else (c, r)
            if pre in src_idx and post in dst_idx:
                data[off] = float(weight); n_set += 1
            else:
                n_kept += 1
    newM = sp.csr_matrix((data, indices, indptr), shape=M.shape)
    bridge.cp_connections = newM
    return n_kept, n_set


def read_recall(bridge, idx_map_B, meta, xp, *, cue_pa=600.0, hold=60, a_assert=None,
                cue_a_idx_map=None, a_cue=0, a_cue_pa=600.0):
    """Organ B recall read: drive cueB for each trained fact, read patient_expected_B firing (the recall).
    If `a_assert` is not None, ALSO drive organ A (cue a_cue + patient_asserted_A a_assert) so surprise_A
    fires DURING the recall -> the cross-organ synapse (if intact) modulates the recall. Returns mean recall
    Hz over the trained facts."""
    n_trained = meta["n_trained"]
    rates = []
    for i in range(n_trained):
        _hard_reset(bridge)
        drives = {"cue": (i, cue_pa)}
        if a_assert is not None and cue_a_idx_map is not None:
            # combined drive: organ B recall cue + organ A (cue + asserted patient) via a merged drive dict.
            _set_two_organ_drives(bridge, idx_map_B, cue_a_idx_map,
                                  {"cue": (i, cue_pa)},
                                  {"cue": (a_cue, a_cue_pa), "patient_asserted": (a_assert, a_cue_pa)}, xp)
            counts = 0
            for _ in range(hold):
                _step(bridge)
                counts += int(bridge.cp_firing_states[idx_map_B["patient_expected"]].sum())
            rates.append(counts / max(len(_host(idx_map_B["patient_expected"])), 1) / (hold * 1e-3))
        else:
            r = _drive_read(bridge, idx_map_B, drives, hold, xp, ["patient_expected"])
            rates.append(r["patient_expected"])
    return _st.mean(rates)


def _set_two_organ_drives(bridge, idx_map_B, idx_map_A, drives_B, drives_A, xp):
    """Set external drive for BOTH organs at once (organ B's recall cue + organ A's cue/asserted)."""
    bridge.cp_external_input_current[:] = 0.0
    blk = bridge._blk
    for idx_map, drives in ((idx_map_B, drives_B), (idx_map_A, drives_A)):
        for region, (concept, pA) in drives.items():
            idx = idx_map[region]
            sub = idx[concept * blk:(concept + 1) * blk] if concept is not None else idx
            bridge.cp_external_input_current[sub] = xp.float32(pA)


def _arr_hash(a):
    import hashlib
    import numpy as np
    h = np.asarray(_host(a)).astype(np.float64).tobytes()
    return hashlib.sha256(h).hexdigest()[:16]


def _region_span(bridge, suffix):
    """Absolute [lo, hi] neuron-index span covering all 4 of an organ's regions in the merged bridge."""
    import numpy as np
    idx = [np.asarray(_idx(bridge, n + suffix)) for n in
           ("cue", "patient_expected", "patient_asserted", "surprise")]
    lo = min(int(a.min()) for a in idx); hi = max(int(a.max()) for a in idx)
    return lo, hi


def homogenize_thresholds(bridge, value):
    """DIAGNOSTIC control that ISOLATES the byte-identity breaker. The substrate draws
    `cp_neuron_firing_thresholds` per-neuron from the GLOBAL RNG stream over ALL n neurons
    (sim/bridge.py:2307, `cp.random.uniform`), so an organ's slice depends on the total pool
    size — organ B (second in the merged stream) gets a DIFFERENT seeded heterogeneity than it
    would standalone. Setting every threshold to one constant REMOVES that per-neuron RNG
    variation entirely; applied IDENTICALLY to the merged and co-resident bridges it makes an
    organ's read invariant to pool position (a clean test that the threshold heterogeneity is
    the SOLE divergence source). NOTE: a constant threshold also DISABLES the surprise faculty's
    confirm/contradict separation (the heterogeneity is functionally required) — so this is a
    cause-isolating control, NOT the production fix. The production fix is a per-region RNG
    stream for the threshold draw (each organ seeded independently of co-residents)."""
    if bridge.cp_neuron_firing_thresholds is not None:
        bridge.cp_neuron_firing_thresholds[:] = float(value)


def _maxerr(m, s, keys):
    e = 0.0
    for k in keys:
        for a, b in zip(m[k], s[k]):
            e = max(e, abs(a - b))
    return e


def _train_coresident(seed, n_reps, xp, build_kw, homog=None, per_region_thresh=True, homeo=True,
                      homeo_iso=False):
    """Two SEPARATE standalone organ bridges (the co-resident baseline). homog!=None -> set every
    threshold to that constant before training (the cause-isolation control). homeo=False disables
    the homeostatic threshold-adaptation companion process (the SECOND, activity-history-coupled
    divergence source isolated 2026-08-13; static per-region thresholds are still heterogeneous).
    homeo_iso=True enables per_region_homeostasis_isolation (the merge-closing fix for the SECOND
    cause: freeze an idle region's homeostatic threshold drift so it is invariant to co-residence).

    Each standalone organ is built with the SAME region NAMES it carries in the merged bridge
    (organ A -> `_A`, organ B -> `_B`), so with per-region threshold seeding ON a region's
    name-keyed threshold slice matches its merged counterpart exactly -- the apples-to-apples
    'this organ, alone vs merged' comparison. (With the legacy global stream the names are inert
    to the draw, so this rename does not change the BOUNDARY baseline.)"""
    brA, cfgA, metaA = build_expectation_circuit(seed, n_trained=8, n_novel=4, blk=24, cue_blk=24,
                                                 region_suffix="_A", per_region_thresh=per_region_thresh,
                                                 **build_kw)
    brA._blk = 24
    brB, cfgB, metaB = build_expectation_circuit(seed, n_trained=8, n_novel=4, blk=24, cue_blk=24,
                                                 region_suffix="_B", per_region_thresh=per_region_thresh,
                                                 **build_kw)
    brB._blk = 24
    for c in (cfgA, cfgB):
        c.enable_homeostasis = bool(homeo)
        c.per_region_homeostasis_isolation = bool(homeo_iso)
    if homog is not None:
        homogenize_thresholds(brA, homog); homogenize_thresholds(brB, homog)
    idxA_solo = _idx_map(brA, "_A", xp); idxB_solo = _idx_map(brB, "_B", xp)
    train_expectation(brA, cfgA, idxA_solo, metaA, xp, n_reps=n_reps); cfgA.enable_hebbian_learning = False
    train_expectation(brB, cfgB, idxB_solo, metaB, xp, n_reps=n_reps); cfgB.enable_hebbian_learning = False
    return (brA, cfgA, metaA, idxA_solo), (brB, cfgB, metaB, idxB_solo)


def _train_merged(seed, cross_weight, n_reps, xp, build_kw, homog=None, per_region_thresh=True, homeo=True,
                  homeo_iso=False):
    """One merged 2-organ bridge; homog!=None -> constant threshold (cause-isolation control);
    homeo=False -> homeostatic threshold adaptation OFF (isolates the companion-process residual);
    homeo_iso=True -> per_region_homeostasis_isolation ON (freeze idle-region homeostatic drift)."""
    merged, cfg_m, meta = build_merged_two_organ(seed, cross_weight=cross_weight,
                                                 per_region_thresh=per_region_thresh, **build_kw)
    idxA = _idx_map(merged, "_A", xp); idxB = _idx_map(merged, "_B", xp)
    cfg_m.enable_homeostasis = bool(homeo)
    cfg_m.per_region_homeostasis_isolation = bool(homeo_iso)
    if homog is not None:
        homogenize_thresholds(merged, homog)
    train_expectation(merged, cfg_m, idxA, meta, xp, n_reps=n_reps)
    train_expectation(merged, cfg_m, idxB, meta, xp, n_reps=n_reps)
    cfg_m.enable_hebbian_learning = False
    if homog is not None:
        homogenize_thresholds(merged, homog)   # re-assert post-train (thresholds are static; belt-and-braces)
    return merged, cfg_m, meta, idxA, idxB


def _byte_identity(merged, cfg_m, meta, idxA, idxB, coA, coB, xp):
    """max|err| of each organ's read, merged vs its standalone co-resident bridge. Organ A =
    surprise (confirm/contradict/novel per-fact); organ B = recall (mean recall Hz). Reads with the
    OTHER organ undriven -> the cross synapse is inert (verified cross_w=0==cross_w=12)."""
    brA, cfgA, metaA, idxA_solo = coA
    brB, cfgB, metaB, idxB_solo = coB
    resA_m = measure_conditions(merged, cfg_m, idxA, meta, xp)
    resA_s = measure_conditions(brA, cfgA, idxA_solo, metaA, xp)
    surprise_maxerr = _maxerr(resA_m, resA_s, ["confirm_per", "contradict_per", "novel_per"])
    recall_m = read_recall(merged, idxB, meta, xp)
    recall_s = read_recall(brB, idxB_solo, metaB, xp)
    return surprise_maxerr, abs(recall_m - recall_s), resA_m, resA_s, recall_m, recall_s


_INIT_PER_NEURON_ARRAYS = (
    "cp_neuron_firing_thresholds", "cp_izh_C", "cp_izh_k", "cp_izh_vr", "cp_izh_vt",
    "cp_izh_vpeak", "cp_izh_a", "cp_izh_b", "cp_izh_c_reset", "cp_izh_d_increment",
    "cp_membrane_potential_v", "cp_recovery_variable_u",
)


def _init_byte_identity(seed, xp, build_kw, per_region_thresh):
    """INITIALIZATION byte-identity -- the axis the per-region threshold fix DIRECTLY closes, and
    the mission's literal framing ('a merged organ's init is invariant to its co-residents').
    Build the merged 2-organ bridge and each organ STANDALONE (same region names), compare EVERY
    per-neuron array on each organ's slice BEFORE any training. With the fix ON this is EXACT 0.0
    for both organs (the second organ no longer lands at a shifted global-RNG stream position);
    with --legacy-global-thresh cp_neuron_firing_thresholds diverges (the mapped BOUNDARY). This
    is build-only (cheap) and isolates the fix from the downstream homeostasis/training confounds."""
    import numpy as np
    merged, _, _ = build_merged_two_organ(seed, per_region_thresh=per_region_thresh, **build_kw)
    brA, _, _ = build_expectation_circuit(seed, n_trained=8, n_novel=4, blk=24, cue_blk=24,
                                          region_suffix="_A", per_region_thresh=per_region_thresh, **build_kw)
    brB, _, _ = build_expectation_circuit(seed, n_trained=8, n_novel=4, blk=24, cue_blk=24,
                                          region_suffix="_B", per_region_thresh=per_region_thresh, **build_kw)
    err = 0.0
    for suf, solo in (("_A", brA), ("_B", brB)):
        for r in ("cue", "patient_expected", "patient_asserted", "surprise"):
            mi = _idx(merged, r + suf); si = _idx(solo, r + suf)
            for nm in _INIT_PER_NEURON_ARRAYS:
                am = np.asarray(_host(getattr(merged, nm)))[mi]
                aso = np.asarray(_host(getattr(solo, nm)))[si]
                err = max(err, float(np.abs(am - aso).max()))
    return err


def run_seed(seed, *, n_reps=22, cross_weight=12.0, homog_control=-42.0, verbose=True,
             per_region_thresh=True, **build_kw):
    from sim.backend import get_backend
    xp, _ = get_backend()

    # ── DETERMINISM: build twice at the same seed, hash the substrate; identical => cfg.seed controls it.
    #    Includes cp_neuron_firing_thresholds so the NEW per-region draw is verified seed-deterministic. ──
    b1, _, _ = build_merged_two_organ(seed, cross_weight=cross_weight, per_region_thresh=per_region_thresh, **build_kw)
    b2, _, _ = build_merged_two_organ(seed, cross_weight=cross_weight, per_region_thresh=per_region_thresh, **build_kw)
    det_ok = (_arr_hash(b1.cp_membrane_potential_v) == _arr_hash(b2.cp_membrane_potential_v)
              and _arr_hash(b1.cp_connections.tocsr().data) == _arr_hash(b2.cp_connections.tocsr().data)
              and _arr_hash(b1.cp_neuron_firing_thresholds) == _arr_hash(b2.cp_neuron_firing_thresholds))

    # ── INIT byte-identity (the axis the per-region fix CLOSES): every per-neuron array of each
    #    organ is identical merged-vs-standalone BEFORE training. Fix ON -> EXACT; legacy -> thresholds diverge. ──
    init_err = _init_byte_identity(seed, xp, build_kw, per_region_thresh)
    init_byte_id = bool(init_err <= 1e-6)

    # ── PRODUCTION-HETEROGENEOUS config (the real organs). ──
    coA, coB = _train_coresident(seed, n_reps, xp, build_kw, per_region_thresh=per_region_thresh)
    merged, cfg_m, meta, idxA, idxB = _train_merged(seed, cross_weight, n_reps, xp, build_kw,
                                                    per_region_thresh=per_region_thresh)
    n_all = int(merged.cp_membrane_potential_v.shape[0])
    n_A = sum(len(_host(idxA[r])) for r in idxA)
    n_B = sum(len(_host(idxB[r])) for r in idxB)
    one_pool = bool(n_all >= n_A + n_B) and all(
        int(_host(idxA[r]).max()) < n_all and int(_host(idxB[r]).max()) < n_all for r in idxA)
    surp_err, recall_err, resA_m, resA_s, recall_m, recall_s = _byte_identity(
        merged, cfg_m, meta, idxA, idxB, coA, coB, xp)
    # the surprise faculty must actually SEPARATE (else "byte-identical" would be of a dead organ)
    surp_sep = resA_m["contradict_hz"] / max(resA_m["confirm_hz"], 1e-6)

    # ── (b) CROSS-ORGAN LOAD-BEARING: organ B recall when A is CONTRADICT (surprised) vs CONFIRM. ──
    a_cue, a_confirm, a_contra = 0, 0, 1 % meta["n_trained"]
    rc_confirm = read_recall(merged, idxB, meta, xp, a_assert=a_confirm, cue_a_idx_map=idxA, a_cue=a_cue)
    rc_contra = read_recall(merged, idxB, meta, xp, a_assert=a_contra, cue_a_idx_map=idxA, a_cue=a_cue)
    interaction_intact = rc_contra - rc_confirm
    nk, nz = lesion_cross(merged)
    rc_confirm_l = read_recall(merged, idxB, meta, xp, a_assert=a_confirm, cue_a_idx_map=idxA, a_cue=a_cue)
    rc_contra_l = read_recall(merged, idxB, meta, xp, a_assert=a_contra, cue_a_idx_map=idxA, a_cue=a_cue)
    interaction_lesion = rc_contra_l - rc_confirm_l

    # ── CAUSE-ISOLATION control: homogeneous threshold (removes the per-organ RNG heterogeneity) ──
    #    -> byte-identity should go EXACT, proving the threshold-RNG-stream is the sole divergence. ──
    coA_h, coB_h = _train_coresident(seed, n_reps, xp, build_kw, homog=homog_control)
    merged_h, cfgH, metaH, idxA_h, idxB_h = _train_merged(seed, cross_weight, n_reps, xp, build_kw, homog=homog_control)
    surp_err_h, recall_err_h, *_ = _byte_identity(merged_h, cfgH, metaH, idxA_h, idxB_h, coA_h, coB_h, xp)

    # ── SECOND-CAUSE ATTRIBUTION (2026-08-13): with the per-region INIT fix ON, the remaining
    #    production residual is the HOMEOSTATIC threshold-adaptation companion process (an
    #    activity-history-coupled intrinsic plasticity), NOT the init RNG. Disabling homeostasis
    #    (static, still per-region-HETEROGENEOUS thresholds) drives the full trained+read pipeline
    #    to EXACT byte-identity -- the decisive control that the init fix closes the init-RNG cause
    #    and homeostasis owns the rest. (Homeostasis is load-bearing for the surprise faculty, so
    #    this is a cause-isolating control, not the production operating point.) ──
    coA_n, coB_n = _train_coresident(seed, n_reps, xp, build_kw, per_region_thresh=per_region_thresh, homeo=False)
    merged_n, cfgN, metaN, idxA_n, idxB_n = _train_merged(seed, cross_weight, n_reps, xp, build_kw,
                                                          per_region_thresh=per_region_thresh, homeo=False)
    surp_err_n, recall_err_n, *_ = _byte_identity(merged_n, cfgN, metaN, idxA_n, idxB_n, coA_n, coB_n, xp)

    # ── THE FIX (2026-08-13): per_region_homeostasis_isolation ON, homeostasis STILL ON (the
    #    production operating point -- faculty stays alive). Root cause of the homeostatic residual
    #    (this arc): homeostatic threshold adaptation is a CONTINUOUS companion process that pulls
    #    even a SILENT neuron's threshold toward its target rate EVERY step, so on ONE shared,
    #    continuously-stepped substrate an IDLE co-resident region idle-drifts (~0.08 mV over a
    #    co-resident's training phase) -- an evolution the SEPARATE standalone bridge never undergoes.
    #    It is NOT pooled activity (the update is strictly per-neuron; an idle organ's activity EMA
    #    stays exactly 0.0) and NOT floating-point order -- it is a deterministic shared-CLOCK drift.
    #    The fix GATES the homeostatic threshold+EMA update to neurons that PARTICIPATED this step
    #    (fired OR received nonzero external drive), freezing idle co-resident drift while leaving the
    #    during-train / during-read adaptation (driven / firing neurons) untouched -> the operating
    #    point the faculty depends on is preserved. Closes the SURPRISE organ's read to byte-EXACT
    #    (its read has no B->A dependence). The RECALL organ retains a ~1-spike residual = the
    #    LOAD-BEARING cross synapse (surprise_A->cueB) firing cueB during organ A's surprise read
    #    (measure_conditions precedes read_recall), which nudges cueB's homeostatic threshold -- i.e.
    #    the cross synapse DOING ITS JOB leaking into the continuous companion process, not pool
    #    co-residence (see the finding's read-order control: reading recall BEFORE organ A's read is
    #    byte-exact). ──
    coA_i, coB_i = _train_coresident(seed, n_reps, xp, build_kw, per_region_thresh=per_region_thresh, homeo_iso=True)
    merged_i, cfgI, metaI, idxA_i, idxB_i = _train_merged(seed, cross_weight, n_reps, xp, build_kw,
                                                          per_region_thresh=per_region_thresh, homeo_iso=True)
    # READ-ORDER DECOMPOSITION of the recall organ's residual. read_recall drives ONLY cueB (organ A
    # undriven), and there is no B->A synapse, so reading recall FIRST leaves organ A's surprise read
    # untouched. recall_BEFORE = the pool-co-residence read (the cross has not yet fired) -> byte-EXACT
    # proves the MERGE ITSELF (init + trained + homeostatically-adapted) is byte-clean. recall_AFTER
    # (inside _byte_identity, which runs organ A's surprise read first) carries the cross synapse's own
    # LOAD-BEARING homeostatic footprint (surprise_A->cueB fired cueB during A's contradict/novel read),
    # NOT a co-residence defect.
    recall_before_m = read_recall(merged_i, idxB_i, metaI, xp)
    recall_before_s = read_recall(coB_i[0], coB_i[3], coB_i[2], xp)
    recall_before_err_i = abs(recall_before_m - recall_before_s)
    surp_err_i, recall_err_i, resA_mi, _, recall_mi, recall_si = _byte_identity(
        merged_i, cfgI, metaI, idxA_i, idxB_i, coA_i, coB_i, xp)
    recall_after_err_i = recall_err_i
    surp_sep_i = resA_mi["contradict_hz"] / max(resA_mi["confirm_hz"], 1e-6)

    # ATTRIBUTION (tools.lab): whose is the organ-B recall INTERACTION? treatment = intact interaction,
    # control = lesioned (cross zeroed). lesion ~0 -> the interaction is OWNED by the cross-organ synapse,
    # not a fixed input artifact (measuring both arms is not the same as asking whose the difference was).
    from tools.lab import attributable_to
    cross_frac = attributable_to("organ-B recall interaction @ surprise_A->cueB cross synapse",
                                 interaction_intact, interaction_lesion)

    load_bearing = bool(abs(interaction_intact) >= 1.0
                        and abs(interaction_intact) >= 5.0 * max(abs(interaction_lesion), 1e-6)
                        and (cross_frac is None or cross_frac >= 0.8))
    hetero_byte_id = bool(surp_err <= 1e-6 and recall_err <= 1e-6)
    homog_byte_id = bool(surp_err_h <= 1e-6 and recall_err_h <= 1e-6)
    homeo_off_byte_id = bool(surp_err_n <= 1e-6 and recall_err_n <= 1e-6)
    # THE FIX: with per_region_homeostasis_isolation ON (homeostasis STILL ON -> faculty alive),
    # POOL CO-RESIDENCE is byte-identical for BOTH organs -- surprise EXACT and recall EXACT when
    # read before the load-bearing cross fires. The faculty must still separate (else exact-of-a-
    # dead-organ). The recall_AFTER residual is the cross synapse's own footprint, tracked separately.
    homeo_iso_surp_exact = bool(surp_err_i <= 1e-6)
    homeo_iso_pool_byte_id = bool(surp_err_i <= 1e-6 and recall_before_err_i <= 1e-6)
    homeo_iso_alive = bool(surp_sep_i >= 5.0)
    res = {
        "seed": seed,
        "determinism_ok": bool(det_ok),
        "one_shared_pool": bool(one_pool),
        "n_all_neurons": n_all, "n_A": n_A, "n_B": n_B, "cross_edges_zeroed": int(nz),
        # INIT byte-identity (the axis the per-region threshold fix directly closes)
        "init_maxerr": float(init_err), "init_byte_identical": init_byte_id,
        # (a) PRODUCTION-hetero byte-identity (homeostasis ON; residual = the companion process)
        "surprise_maxerr_hz": float(surp_err), "recall_maxerr_hz": float(recall_err),
        "surprise_separation_ratio": float(surp_sep),
        "surprise_merged": {k: resA_m[k] for k in ("confirm_hz", "contradict_hz", "novel_hz")},
        "surprise_solo": {k: resA_s[k] for k in ("confirm_hz", "contradict_hz", "novel_hz")},
        "recall_merged_hz": float(recall_m), "recall_solo_hz": float(recall_s),
        # cause-isolation: homogeneous threshold -> EXACT byte-identity
        "homog_surprise_maxerr_hz": float(surp_err_h), "homog_recall_maxerr_hz": float(recall_err_h),
        "homog_byte_identical": homog_byte_id,
        # second-cause attribution: homeostasis OFF (static per-region-heterogeneous thresholds) -> EXACT
        "homeo_off_surprise_maxerr_hz": float(surp_err_n), "homeo_off_recall_maxerr_hz": float(recall_err_n),
        "homeo_off_byte_identical": homeo_off_byte_id,
        # THE FIX: per_region_homeostasis_isolation ON, homeostasis STILL ON (faculty alive).
        # POOL CO-RESIDENCE byte-exact (surprise + recall-before-cross); recall-after-cross residual
        # = the load-bearing cross synapse's homeostatic footprint (read-order control).
        "homeo_iso_surprise_maxerr_hz": float(surp_err_i),
        "homeo_iso_recall_before_maxerr_hz": float(recall_before_err_i),
        "homeo_iso_recall_after_maxerr_hz": float(recall_after_err_i),
        "homeo_iso_separation_ratio": float(surp_sep_i),
        "homeo_iso_surprise_byte_identical": homeo_iso_surp_exact,
        "homeo_iso_pool_byte_identical": homeo_iso_pool_byte_id,
        "homeo_iso_faculty_alive": homeo_iso_alive,
        # (b) load-bearing cross-organ synapse
        "recall_A_confirm_hz": float(rc_confirm), "recall_A_contra_hz": float(rc_contra),
        "interaction_intact_hz": float(interaction_intact),
        "interaction_lesion_hz": float(interaction_lesion),
        "cross_attribution_frac": (float(cross_frac) if cross_frac is not None else None),
        "hetero_byte_identical": hetero_byte_id,
        "cross_load_bearing": load_bearing,
    }
    # structural GO = one pool + determinism + a load-bearing cross synapse + INIT byte-identity
    # (each organ's per-neuron init invariant to co-residents -- the per-region fix) + the merge is
    # byte-clean once the homeostatic companion process is neutralised (homeo-off control). Exact
    # byte-identity of the fully HOMEOSTATICALLY-ADAPTED production read is bounded by that companion
    # process (+ a shared-numerical-context FP floor), not the init RNG -- see the finding.
    res["structural_go"] = bool(one_pool and det_ok and load_bearing and init_byte_id
                                and homog_byte_id and homeo_off_byte_id)
    res["homeo_iso_go"] = bool(homeo_iso_pool_byte_id and homeo_iso_alive)
    if verbose:
        print(f"  [seed {seed}] pool={one_pool}(N={n_all}={n_A}A+{n_B}B) det={det_ok} | "
              f"INIT byte-id err={init_err:.2e}({init_byte_id}) | "
              f"HOMEO-OFF byte-id surp={surp_err_n:.2e} recall={recall_err_n:.2e}({homeo_off_byte_id}) | "
              f"PROD(homeo-on,no-iso) surp={surp_err:.2e} recall={recall_err:.2e} (sep={surp_sep:.1f}x) | "
              f"HOMEO-ISO surp={surp_err_i:.2e} recall_before={recall_before_err_i:.2e} "
              f"recall_after={recall_after_err_i:.2e} (sep={surp_sep_i:.1f}x) pool-byte-id={homeo_iso_pool_byte_id} | "
              f"cross intact={interaction_intact:+.2f} lesion={interaction_lesion:+.2f}Hz | "
              f"struct-GO={res['structural_go']}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--n-reps", type=int, default=22)
    ap.add_argument("--cross-weight", type=float, default=12.0)
    ap.add_argument("--legacy-global-thresh", action="store_true",
                    help="Disable the per-region threshold-seeding fix (reproduces the 0/6 BOUNDARY: "
                         "the single global RNG stream shifts organ B to a divergent stream position).")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    per_region_thresh = not args.legacy_global_thresh
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== ONE-BRAIN MERGE (2 organs: surprise + recall) on ONE shared spiking substrate ===")
    print(f"    per-region threshold heterogeneity: {'ON (merge-closing fix)' if per_region_thresh else 'OFF (legacy global stream -> BOUNDARY)'}")
    results = [run_seed(s, n_reps=args.n_reps, cross_weight=args.cross_weight,
                        per_region_thresh=per_region_thresh) for s in seeds]

    n = len(results)
    n_pool = sum(1 for r in results if r["one_shared_pool"])
    n_det = sum(1 for r in results if r["determinism_ok"])
    n_lb = sum(1 for r in results if r["cross_load_bearing"])
    n_init = sum(1 for r in results if r["init_byte_identical"])
    n_homog = sum(1 for r in results if r["homog_byte_identical"])
    n_homeo_off = sum(1 for r in results if r["homeo_off_byte_identical"])
    n_hetero = sum(1 for r in results if r["hetero_byte_identical"])
    n_struct = sum(1 for r in results if r["structural_go"])
    n_iso_surp = sum(1 for r in results if r["homeo_iso_surprise_byte_identical"])
    n_iso_pool = sum(1 for r in results if r["homeo_iso_pool_byte_identical"])
    n_iso_alive = sum(1 for r in results if r["homeo_iso_faculty_alive"])
    n_iso_go = sum(1 for r in results if r["homeo_iso_go"])
    max_recall_err = max(r["recall_maxerr_hz"] for r in results)
    max_surp_err = max(r["surprise_maxerr_hz"] for r in results)
    max_iso_surp_err = max(r["homeo_iso_surprise_maxerr_hz"] for r in results)
    max_iso_recall_before_err = max(r["homeo_iso_recall_before_maxerr_hz"] for r in results)
    max_iso_recall_after_err = max(r["homeo_iso_recall_after_maxerr_hz"] for r in results)
    max_init_err = max(r["init_maxerr"] for r in results)
    _gate = lambda k: "GO" if ((n >= 6 and k >= 5) or (n < 6 and k == n)) else "BOUNDARY"
    # INIT byte-identity is the axis the per-region fix closes (the mission's 'init invariant to
    # co-residents'). STRUCTURAL GO now also requires it + the homeo-off byte-clean control. Exact
    # byte-identity of the fully HOMEOSTATICALLY-ADAPTED production read is bounded by the homeostatic
    # companion process (+ a shared-numerical-context FP floor) -> BOUNDARY on that one axis only.
    struct = _gate(n_struct)
    init_exact = _gate(n_init)
    homeo_off_exact = _gate(n_homeo_off)
    hetero_exact = _gate(n_hetero)
    iso_surp_exact = _gate(n_iso_surp)
    iso_pool_exact = _gate(n_iso_pool)
    iso_go = _gate(n_iso_go)
    print("\n=== VERDICT ===")
    print(f"  one shared neuron pool:                 {n_pool}/{n}")
    print(f"  determinism (cfg.seed incl. thresholds):{n_det}/{n}")
    print(f"  cross-organ synapse load-bearing:       {n_lb}/{n}")
    print(f"  INIT byte-identity (per-region fix):    {n_init}/{n}  ->  {init_exact}  (per-neuron init invariant to co-residents; max err {max_init_err:.2e}) <- the axis the fix CLOSES")
    print(f"  byte-identical, HOMEOSTASIS-OFF:        {n_homeo_off}/{n}  ->  {homeo_off_exact}  (static-threshold regime byte-clean; with INIT exact this isolates the production residual to the homeostatic DYNAMICS)")
    print(f"  byte-identical under HOMOG threshold:   {n_homog}/{n}  (cause-isolation: constant threshold)")
    print(f"  byte-identical, PRODUCTION (homeo ON, NO isolation):  {n_hetero}/{n}  ->  {hetero_exact}  (residual = homeostatic idle-drift; max err surp={max_surp_err:.2e} recall={max_recall_err:.2e} Hz)")
    print(f"  --- THE FIX: per_region_homeostasis_isolation ON (homeostasis STILL ON; faculty alive {n_iso_alive}/{n}) ---")
    print(f"  homeo-ISO surprise byte-identical:      {n_iso_surp}/{n}  ->  {iso_surp_exact}  (surprise organ EXACT; max err {max_iso_surp_err:.2e} Hz)")
    print(f"  homeo-ISO POOL CO-RESIDENCE byte-id:    {n_iso_pool}/{n}  ->  {iso_pool_exact}  (surprise + recall-BEFORE-cross both EXACT; recall-before max err {max_iso_recall_before_err:.2e} Hz) <- the MERGE is byte-clean")
    print(f"  homeo-ISO recall-AFTER-cross residual:  max {max_iso_recall_after_err:.2e} Hz  (the LOAD-BEARING cross synapse's OWN homeostatic footprint -- read-order control -- NOT a co-residence defect)")
    print(f"  --> HOMEO-ISO merge verdict:            {n_iso_go}/{n}  ->  {iso_go}  (pool co-residence byte-exact AND faculty alive)")
    print(f"  STRUCTURAL MERGE:                        {n_struct}/{n}  ->  {struct}")
    print(f"  --> per-region INIT fix CLOSES the init-RNG divergence ({init_exact}); the homeostatic idle-drift SECOND cause is CLOSED by per_region_homeostasis_isolation ({iso_pool_exact}); the recall-after-cross residual is the cross synapse's load-bearing footprint (characterized).")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"mode": "one_brain_merge_2organ",
                       "per_region_threshold_heterogeneity": per_region_thresh, "results": results, "n_seeds": n,
                       "n_one_shared_pool": n_pool, "n_determinism_ok": n_det,
                       "n_cross_load_bearing": n_lb, "n_init_byte_identical": n_init,
                       "n_homog_byte_identical": n_homog, "n_homeo_off_byte_identical": n_homeo_off,
                       "n_hetero_byte_identical": n_hetero, "n_structural_go": n_struct,
                       "n_homeo_iso_surprise_byte_identical": n_iso_surp,
                       "n_homeo_iso_pool_byte_identical": n_iso_pool,
                       "n_homeo_iso_faculty_alive": n_iso_alive, "n_homeo_iso_go": n_iso_go,
                       "max_init_maxerr": max_init_err,
                       "max_recall_maxerr_hz": max_recall_err, "max_surprise_maxerr_hz": max_surp_err,
                       "max_homeo_iso_surprise_maxerr_hz": max_iso_surp_err,
                       "max_homeo_iso_recall_before_maxerr_hz": max_iso_recall_before_err,
                       "max_homeo_iso_recall_after_maxerr_hz": max_iso_recall_after_err,
                       "structural_verdict": struct, "init_byteid_verdict": init_exact,
                       "homeo_off_byteid_verdict": homeo_off_exact, "exact_byteid_verdict": hetero_exact,
                       "homeo_iso_surprise_byteid_verdict": iso_surp_exact,
                       "homeo_iso_pool_byteid_verdict": iso_pool_exact, "homeo_iso_go_verdict": iso_go,
                       "cross_weight": args.cross_weight}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
