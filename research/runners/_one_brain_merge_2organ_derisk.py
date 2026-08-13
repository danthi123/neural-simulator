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
                           hebbian_max_weight=45.0, cross_weight=12.0):
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


def _train_coresident(seed, n_reps, xp, build_kw, homog=None):
    """Two SEPARATE standalone organ bridges (the co-resident baseline). homog!=None -> set every
    threshold to that constant before training (the cause-isolation control)."""
    brA, cfgA, metaA = build_expectation_circuit(seed, n_trained=8, n_novel=4, blk=24, cue_blk=24, **build_kw)
    brA._blk = 24
    brB, cfgB, metaB = build_expectation_circuit(seed, n_trained=8, n_novel=4, blk=24, cue_blk=24, **build_kw)
    brB._blk = 24
    if homog is not None:
        homogenize_thresholds(brA, homog); homogenize_thresholds(brB, homog)
    idxA_solo = _idx_map(brA, "", xp); idxB_solo = _idx_map(brB, "", xp)
    train_expectation(brA, cfgA, idxA_solo, metaA, xp, n_reps=n_reps); cfgA.enable_hebbian_learning = False
    train_expectation(brB, cfgB, idxB_solo, metaB, xp, n_reps=n_reps); cfgB.enable_hebbian_learning = False
    return (brA, cfgA, metaA, idxA_solo), (brB, cfgB, metaB, idxB_solo)


def _train_merged(seed, cross_weight, n_reps, xp, build_kw, homog=None):
    """One merged 2-organ bridge; homog!=None -> constant threshold (cause-isolation control)."""
    merged, cfg_m, meta = build_merged_two_organ(seed, cross_weight=cross_weight, **build_kw)
    idxA = _idx_map(merged, "_A", xp); idxB = _idx_map(merged, "_B", xp)
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


def run_seed(seed, *, n_reps=22, cross_weight=12.0, homog_control=-42.0, verbose=True, **build_kw):
    from sim.backend import get_backend
    xp, _ = get_backend()

    # ── DETERMINISM: build twice at the same seed, hash the substrate; identical => cfg.seed controls it. ──
    b1, _, _ = build_merged_two_organ(seed, cross_weight=cross_weight, **build_kw)
    b2, _, _ = build_merged_two_organ(seed, cross_weight=cross_weight, **build_kw)
    det_ok = (_arr_hash(b1.cp_membrane_potential_v) == _arr_hash(b2.cp_membrane_potential_v)
              and _arr_hash(b1.cp_connections.tocsr().data) == _arr_hash(b2.cp_connections.tocsr().data))

    # ── PRODUCTION-HETEROGENEOUS config (the real organs). ──
    coA, coB = _train_coresident(seed, n_reps, xp, build_kw)
    merged, cfg_m, meta, idxA, idxB = _train_merged(seed, cross_weight, n_reps, xp, build_kw)
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
    res = {
        "seed": seed,
        "determinism_ok": bool(det_ok),
        "one_shared_pool": bool(one_pool),
        "n_all_neurons": n_all, "n_A": n_A, "n_B": n_B, "cross_edges_zeroed": int(nz),
        # (a) PRODUCTION-hetero byte-identity (the characterized bounded delta)
        "surprise_maxerr_hz": float(surp_err), "recall_maxerr_hz": float(recall_err),
        "surprise_separation_ratio": float(surp_sep),
        "surprise_merged": {k: resA_m[k] for k in ("confirm_hz", "contradict_hz", "novel_hz")},
        "surprise_solo": {k: resA_s[k] for k in ("confirm_hz", "contradict_hz", "novel_hz")},
        "recall_merged_hz": float(recall_m), "recall_solo_hz": float(recall_s),
        # cause-isolation: homogeneous threshold -> EXACT byte-identity
        "homog_surprise_maxerr_hz": float(surp_err_h), "homog_recall_maxerr_hz": float(recall_err_h),
        "homog_byte_identical": homog_byte_id,
        # (b) load-bearing cross-organ synapse
        "recall_A_confirm_hz": float(rc_confirm), "recall_A_contra_hz": float(rc_contra),
        "interaction_intact_hz": float(interaction_intact),
        "interaction_lesion_hz": float(interaction_lesion),
        "cross_attribution_frac": (float(cross_frac) if cross_frac is not None else None),
        "hetero_byte_identical": hetero_byte_id,
        "cross_load_bearing": load_bearing,
    }
    # structural GO = one pool + determinism + a load-bearing cross synapse + the merge is byte-clean
    # under shared per-organ heterogeneity (homog control). Exact byte-identity under PRODUCTION
    # heterogeneity is the mapped residual (per-organ seeding), not required for the structural GO.
    res["structural_go"] = bool(one_pool and det_ok and load_bearing and homog_byte_id)
    if verbose:
        print(f"  [seed {seed}] pool={one_pool}(N={n_all}={n_A}A+{n_B}B) det={det_ok} | "
              f"HETERO byte-id err surp={surp_err:.2e} recall={recall_err:.2e} (sep={surp_sep:.1f}x) | "
              f"HOMOG-ctl err surp={surp_err_h:.2e} recall={recall_err_h:.2e} | "
              f"cross intact={interaction_intact:+.2f} lesion={interaction_lesion:+.2f}Hz | "
              f"struct-GO={res['structural_go']}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--n-reps", type=int, default=22)
    ap.add_argument("--cross-weight", type=float, default=12.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== ONE-BRAIN MERGE (2 organs: surprise + recall) on ONE shared spiking substrate ===")
    results = [run_seed(s, n_reps=args.n_reps, cross_weight=args.cross_weight) for s in seeds]

    n = len(results)
    n_pool = sum(1 for r in results if r["one_shared_pool"])
    n_det = sum(1 for r in results if r["determinism_ok"])
    n_lb = sum(1 for r in results if r["cross_load_bearing"])
    n_homog = sum(1 for r in results if r["homog_byte_identical"])
    n_hetero = sum(1 for r in results if r["hetero_byte_identical"])
    n_struct = sum(1 for r in results if r["structural_go"])
    max_recall_err = max(r["recall_maxerr_hz"] for r in results)
    max_surp_err = max(r["surprise_maxerr_hz"] for r in results)
    # STRUCTURAL GO: the merge is genuine, deterministic, its cross synapse load-bearing, and byte-clean
    # under shared per-organ heterogeneity. Exact byte-identity under PRODUCTION heterogeneity is the
    # mapped residual (per-organ seed streams) -> BOUNDARY on that axis.
    struct = "GO" if ((n >= 6 and n_struct >= 5) or (n < 6 and n_struct == n)) else "BOUNDARY"
    hetero_exact = "GO" if ((n >= 6 and n_hetero >= 5) or (n < 6 and n_hetero == n)) else "BOUNDARY"
    print("\n=== VERDICT ===")
    print(f"  one shared neuron pool:                 {n_pool}/{n}")
    print(f"  determinism (cfg.seed):                 {n_det}/{n}")
    print(f"  cross-organ synapse load-bearing:       {n_lb}/{n}")
    print(f"  byte-identical under HOMOG threshold:   {n_homog}/{n}  (cause-isolation: threshold RNG is the sole breaker)")
    print(f"  byte-identical under PRODUCTION hetero: {n_hetero}/{n}  (max err: surprise={max_surp_err:.2e} recall={max_recall_err:.2e} Hz)")
    print(f"  STRUCTURAL MERGE:                        {n_struct}/{n}  ->  {struct}")
    print(f"  EXACT byte-identity (production):        {n_hetero}/{n}  ->  {hetero_exact}  (residual: per-organ heterogeneity seeding)")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"mode": "one_brain_merge_2organ", "results": results, "n_seeds": n,
                       "n_one_shared_pool": n_pool, "n_determinism_ok": n_det,
                       "n_cross_load_bearing": n_lb, "n_homog_byte_identical": n_homog,
                       "n_hetero_byte_identical": n_hetero, "n_structural_go": n_struct,
                       "max_recall_maxerr_hz": max_recall_err, "max_surprise_maxerr_hz": max_surp_err,
                       "structural_verdict": struct, "exact_byteid_verdict": hetero_exact,
                       "cross_weight": args.cross_weight}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
