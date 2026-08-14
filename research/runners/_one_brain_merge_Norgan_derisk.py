"""ONE-BRAIN MERGE de-risk — SCALE the shared-substrate merge to (a) N ORGANS and (b) DIFFERENT BUILDERS.

CONTEXT (what just closed, and the named next rung)
---------------------------------------------------
The 2-organ merge CLOSED at INIT byte-identity (`2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md`;
runner `_one_brain_merge_2organ_derisk.py`) via the guarded engine flag
`cfg.per_region_threshold_heterogeneity=True` — a merged organ's per-neuron threshold init is now INVARIANT
to its co-residents (each region's thresholds are drawn from a name-keyed substream). But that result merged
TWO INSTANCES of ONE builder (`build_expectation_circuit`: surprise + recall). The named next step is:
  (a) 2 -> N organs on ONE shared spiking pool, and
  (b) merging organs built by DIFFERENT builders via a config SUPERSET (GABA_B + NMDA-accumulator coexistence).

This runner takes both. NO `sim/` edit — the engine flag already exists; everything here is additive,
reuse-by-import.

(a) N ORGANS  (`--mode norgan`, default N=3)
--------------------------------------------
N expectation-circuit organs (suffix _A/_B/_C/...) on ONE `SimulationBridge` (one `cp_` neuron array, one
step, one `cfg.seed`, `per_region_threshold_heterogeneity=True`). Each organ is the adversarially-verified D2
expectation circuit (surprise faculty + heteroassociative recall). GENUINE cross-organ synapses wire an
upper-triangular DAG: for every organ pair i<j, `surprise_i -> cue_j` (the LC-NE / hippocampal novelty motif:
organ i's SURPRISE gates organ j's recall). Verified per seed:
  * ONE shared pool: `len(cp_membrane_potential_v) >= sum(region sizes)`, every region index in the one array.
  * INIT byte-exact vs standalone (the flag ON): every per-neuron array of every organ's slice is identical
    merged-vs-standalone BEFORE training (the axis the per-region fix closes). `--legacy-global-thresh`
    reproduces the BOUNDARY.
  * >=1 cross-organ synapse LOAD-BEARING PER PAIR: drive organ i SURPRISED (contradict) vs not, read organ j's
    recall; lesion `surprise_i -> cue_j` -> the interaction collapses (attributable_to the cross synapse).
  * organs FUNCTIONAL: each organ's surprise faculty separates contradict/confirm on the merged bridge.
  * DETERMINISM: `cfg.seed`; build-twice-same-seed byte-identical (incl. thresholds).

(b) DIFFERENT BUILDERS  (`--mode diffbuilder`)
----------------------------------------------
Merge the SURPRISE expectation organ (`build_expectation_circuit`, GABA_B prediction) with the Wong-Wang
`SpikingRoleCompetition` comprehension/role monitor (`_phaseB_multicue_competition_spiking_derisk`, NMDA
mutual-inhibition WTA) onto ONE pool. This REQUIRES the config SUPERSET: `enable_gabab=True` AND
`enable_nmda=True` coexisting in one bridge, plus a single global `dt_ms` / `enable_homeostasis` the two
builders set differently. We build the superset, test BOTH organs functional (surprise separation; role WTA
selects the driven role), a load-bearing cross synapse (`surprise_S -> sel_agent`: surprise biases role toward
AGENT), init byte-exact for the co-resident expectation organ, and DETERMINISM — and we MAP precisely which
global fields COEXIST (nmda+gabab: benign union) vs which are GENUINE single-valued conflicts (dt_ms,
enable_homeostasis, the hebbian scalars). If a conflict makes an organ non-functional at the reconciled
operating point, that quantified boundary is the named next `sim/` step (per-organ dt / homeostasis scoping).

Run:
    SIM_BACKEND=numpy python -m research.runners._one_brain_merge_Norgan_derisk --mode norgan \
        --n-organs 3 --seeds 42,43,44,100,101,102 --out research/findings/raw/_one_brain_merge_Norgan_6seed.json
    SIM_BACKEND=numpy python -m research.runners._one_brain_merge_Norgan_derisk --mode diffbuilder \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_one_brain_merge_diffbuilder_6seed.json
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
from research.runners._one_brain_merge_2organ_derisk import (
    _idx_map,
    read_recall,
    _install_block_diagonal_full,
    _arr_hash,
    _INIT_PER_NEURON_ARRAYS,
)


def _sfx(i):
    """Organ suffix _A, _B, _C, ... (matches build_expectation_circuit region-name convention)."""
    return "_" + chr(ord("A") + i)


# ─────────────────────────────────────────────────────────────────────────────
# (a) N ORGANS — N expectation circuits on one bridge + an upper-triangular cross DAG.
# ─────────────────────────────────────────────────────────────────────────────
def build_merged_norgan(seed, *, n_organs=3, n_trained=8, n_novel=4, blk=24, cue_blk=24,
                        cue_to_expected_weight=0.8, asserted_to_surprise_weight=5.0,
                        expected_to_surprise_weight=14.0, gabab_prop=0.22, gabab_tau_decay=150.0,
                        hebbian_learning_rate=0.06, hebbian_max_weight=45.0, cross_weight=12.0,
                        per_region_thresh=True):
    """ONE SimulationBridge holding N expectation-circuit organs (suffix _A.._{A+N-1}) + cross-organ
    synapses surprise_i -> cue_j for every i<j. Config is byte-identical to build_expectation_circuit so
    each organ's slice matches its standalone build; only the region/pathway SETS are the N-organ union."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    n_concepts = n_trained + n_novel
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
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

    def organ_regions(sfx):
        return [
            BrainRegion(name="cue" + sfx, n_neurons=n_trained * cue_blk, exc_fraction=1.0,
                        internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
            BrainRegion(name="patient_expected" + sfx, n_neurons=n_concepts * blk, exc_fraction=0.0,
                        internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=FS,
                        syn_reversal_potential_i_override=-70.0),
            BrainRegion(name="patient_asserted" + sfx, n_neurons=n_concepts * blk, exc_fraction=1.0,
                        internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
            BrainRegion(name="surprise" + sfx, n_neurons=n_concepts * blk, exc_fraction=1.0,
                        internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
        ]

    def organ_pathways(sfx):
        return [
            RegionPathway(from_region="cue" + sfx, to_region="patient_expected" + sfx,
                          density=1.0, weight_mean=float(cue_to_expected_weight),
                          weight_jitter=0.0, plastic=True),
            RegionPathway(from_region="patient_asserted" + sfx, to_region="surprise" + sfx,
                          density=1.0, weight_mean=float(asserted_to_surprise_weight),
                          weight_jitter=0.0, plastic=False),
            RegionPathway(from_region="patient_expected" + sfx, to_region="surprise" + sfx,
                          density=1.0, weight_mean=float(expected_to_surprise_weight),
                          weight_jitter=0.0, plastic=False),
        ]

    suffixes = [_sfx(i) for i in range(n_organs)]
    regions = []
    pathways = []
    for sfx in suffixes:
        regions += organ_regions(sfx)
        pathways += organ_pathways(sfx)
    # THE CROSS-ORGAN DAG: surprise_i -> cue_j for every i<j (one directed edge per unordered pair).
    cross_pairs = []
    for i in range(n_organs):
        for j in range(i + 1, n_organs):
            pathways.append(RegionPathway(from_region="surprise" + suffixes[i],
                                          to_region="cue" + suffixes[j],
                                          density=1.0, weight_mean=float(cross_weight),
                                          weight_jitter=0.0, plastic=False))
            cross_pairs.append((i, j))
    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge._blk = blk

    meta = dict(n_trained=n_trained, n_novel=n_novel, n_concepts=n_concepts, blk=blk,
                cue_blk=cue_blk, W_exc=float(asserted_to_surprise_weight),
                W_inh=float(expected_to_surprise_weight), cross_weight=float(cross_weight),
                n_organs=n_organs, suffixes=suffixes, cross_pairs=cross_pairs)

    # TOPOGRAPHIC block-diagonal intra-organ wiring for every organ (concept c -> block c).
    for sfx in suffixes:
        _install_block_diagonal(bridge, "patient_asserted" + sfx, "surprise" + sfx, blk,
                                float(asserted_to_surprise_weight))
        _install_block_diagonal(bridge, "patient_expected" + sfx, "surprise" + sfx, blk,
                                float(expected_to_surprise_weight))
        _install_block_diagonal(bridge, "cue" + sfx, "patient_expected" + sfx, blk,
                                float(cue_to_expected_weight))
    # Cross synapses left FULL at cross_weight (broadcast novelty gate; lesion sets to 0).
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge, cfg, meta


def _init_byte_identity_norgan(seed, build_kw, per_region_thresh):
    """Every per-neuron array of every organ, merged-vs-standalone, BEFORE training. Fix ON -> 0.0."""
    import numpy as np
    n_organs = build_kw.get("n_organs", 3)
    merged, _, meta = build_merged_norgan(seed, per_region_thresh=per_region_thresh, **build_kw)
    err = 0.0
    per_organ = []
    for sfx in meta["suffixes"]:
        solo, _, _ = build_expectation_circuit(seed, n_trained=build_kw.get("n_trained", 8),
                                               n_novel=build_kw.get("n_novel", 4), blk=build_kw.get("blk", 24),
                                               cue_blk=build_kw.get("cue_blk", 24), region_suffix=sfx,
                                               per_region_thresh=per_region_thresh)
        oe = 0.0
        for r in ("cue", "patient_expected", "patient_asserted", "surprise"):
            mi = _idx(merged, r + sfx); si = _idx(solo, r + sfx)
            for nm in _INIT_PER_NEURON_ARRAYS:
                am = np.asarray(_host(getattr(merged, nm)))[mi]
                aso = np.asarray(_host(getattr(solo, nm)))[si]
                oe = max(oe, float(np.abs(am - aso).max()))
        per_organ.append(oe); err = max(err, oe)
    return err, per_organ


def run_seed_norgan(seed, *, n_organs=3, n_reps=22, cross_weight=12.0, verbose=True,
                    per_region_thresh=True, **build_kw):
    from sim.backend import get_backend
    xp, _ = get_backend()
    build_kw = dict(build_kw); build_kw["n_organs"] = n_organs

    # DETERMINISM: build twice, hash substrate incl. thresholds.
    b1, _, _ = build_merged_norgan(seed, cross_weight=cross_weight, per_region_thresh=per_region_thresh, **build_kw)
    b2, _, _ = build_merged_norgan(seed, cross_weight=cross_weight, per_region_thresh=per_region_thresh, **build_kw)
    det_ok = (_arr_hash(b1.cp_membrane_potential_v) == _arr_hash(b2.cp_membrane_potential_v)
              and _arr_hash(b1.cp_connections.tocsr().data) == _arr_hash(b2.cp_connections.tocsr().data)
              and _arr_hash(b1.cp_neuron_firing_thresholds) == _arr_hash(b2.cp_neuron_firing_thresholds))

    # INIT byte-identity (per-region fix closes it).
    init_err, init_per_organ = _init_byte_identity_norgan(seed, build_kw, per_region_thresh)
    init_byte_id = bool(init_err <= 1e-6)

    # Build + train the merged N-organ bridge.
    merged, cfg_m, meta = build_merged_norgan(seed, cross_weight=cross_weight,
                                              per_region_thresh=per_region_thresh, **build_kw)
    suffixes = meta["suffixes"]
    idx = {sfx: _idx_map(merged, sfx, xp) for sfx in suffixes}

    # ONE shared pool.
    n_all = int(merged.cp_membrane_potential_v.shape[0])
    n_each = {sfx: sum(len(_host(idx[sfx][r])) for r in idx[sfx]) for sfx in suffixes}
    one_pool = bool(n_all >= sum(n_each.values())) and all(
        int(_host(idx[sfx][r]).max()) < n_all for sfx in suffixes for r in idx[sfx])

    for sfx in suffixes:
        train_expectation(merged, cfg_m, idx[sfx], meta, xp, n_reps=n_reps)
    cfg_m.enable_hebbian_learning = False

    # Each organ FUNCTIONAL: surprise contradict/confirm separation.
    organ_sep = {}
    organ_funcs = []
    for sfx in suffixes:
        res = measure_conditions(merged, cfg_m, idx[sfx], meta, xp)
        sep = res["contradict_hz"] / max(res["confirm_hz"], 1e-6)
        organ_sep[sfx] = float(sep)
        organ_funcs.append(bool(sep >= 2.0 and res["contradict_hz"] >= 5.0))
    all_functional = all(organ_funcs)

    # CROSS-ORGAN LOAD-BEARING per pair (i<j): organ j recall when organ i CONTRADICT vs CONFIRM;
    # lesion surprise_i -> cue_j -> interaction collapses (attributable_to the cross synapse).
    from tools.lab import attributable_to
    a_confirm, a_contra, a_cue = 0, 1 % meta["n_trained"], 0
    pair_results = []
    all_load_bearing = True
    for (i, j) in meta["cross_pairs"]:
        si, sj = suffixes[i], suffixes[j]
        rc_conf = read_recall(merged, idx[sj], meta, xp, a_assert=a_confirm, cue_a_idx_map=idx[si], a_cue=a_cue)
        rc_cont = read_recall(merged, idx[sj], meta, xp, a_assert=a_contra, cue_a_idx_map=idx[si], a_cue=a_cue)
        intact = rc_cont - rc_conf
        nk, nz = _install_block_diagonal_full(merged, "surprise" + si, "cue" + sj, 0.0)
        rc_conf_l = read_recall(merged, idx[sj], meta, xp, a_assert=a_confirm, cue_a_idx_map=idx[si], a_cue=a_cue)
        rc_cont_l = read_recall(merged, idx[sj], meta, xp, a_assert=a_contra, cue_a_idx_map=idx[si], a_cue=a_cue)
        lesion = rc_cont_l - rc_conf_l
        frac = attributable_to(f"organ {sj} recall interaction @ surprise{si}->cue{sj}", intact, lesion)
        lb = bool(abs(intact) >= 1.0 and abs(intact) >= 5.0 * max(abs(lesion), 1e-6)
                  and (frac is None or frac >= 0.8))
        all_load_bearing = all_load_bearing and lb
        pair_results.append({"pair": [i, j], "src": si, "dst": sj, "cross_edges_zeroed": int(nz),
                             "interaction_intact_hz": float(intact), "interaction_lesion_hz": float(lesion),
                             "attribution_frac": (float(frac) if frac is not None else None),
                             "load_bearing": lb})
        # restore the cross edge so subsequent pairs test on the intact DAG.
        _install_block_diagonal_full(merged, "surprise" + si, "cue" + sj, float(cross_weight))

    structural_go = bool(one_pool and det_ok and init_byte_id and all_load_bearing and all_functional)
    res = {
        "seed": seed, "n_organs": n_organs,
        "determinism_ok": det_ok, "one_shared_pool": one_pool,
        "n_all_neurons": n_all, "n_per_organ": n_each,
        "init_maxerr": float(init_err), "init_per_organ_maxerr": init_per_organ, "init_byte_identical": init_byte_id,
        "organ_separation_ratio": organ_sep, "all_organs_functional": all_functional,
        "pairs": pair_results, "all_pairs_load_bearing": all_load_bearing,
        "structural_go": structural_go,
    }
    if verbose:
        seps = " ".join(f"{s}:{organ_sep[s]:.1f}x" for s in suffixes)
        lbs = " ".join(f"{p['src']}->{p['dst']}:{p['interaction_intact_hz']:+.1f}/{p['interaction_lesion_hz']:+.1f}"
                       for p in pair_results)
        print(f"  [seed {seed}] pool={one_pool}(N={n_all}) det={det_ok} INIT={init_err:.1e}({init_byte_id}) | "
              f"organs[{seps}] func={all_functional} | cross[{lbs}] LB={all_load_bearing} | "
              f"struct-GO={structural_go}", flush=True)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# (b) DIFFERENT BUILDERS — expectation (GABA_B) + Wong-Wang role WTA (NMDA) via a config SUPERSET.
# ─────────────────────────────────────────────────────────────────────────────
def _global_config_conflict_map():
    """Which GLOBAL cfg fields the two builders set, and whether the superset is a BENIGN union
    (both mechanisms coexist) or a GENUINE single-valued conflict (one bridge must pick one value).
    Values read from each builder's source (documented, not guessed)."""
    return [
        # field, expectation value, role value, classification, note
        ("enable_gabab", True, False, "BENIGN-UNION",
         "gabab ON is inert for the role organ (no GABA_B synapses; conductance_max=0) -> coexists"),
        ("enable_nmda", False, True, "BENIGN-UNION",
         "nmda ON is per-region gated; the expectation organ has no enable_nmda regions -> coexists"),
        ("dt_ms", 1.0, 0.5, "GENUINE-CONFLICT",
         "one global timestep; expectation tuned at 1.0, Wong-Wang accumulator at 0.5 -> must pick one"),
        ("enable_homeostasis", True, False, "GENUINE-CONFLICT",
         "one global flag; expectation surprise operating-point uses homeostasis, role WTA tuned OFF"),
        ("hebbian_learning_rate", 0.06, 0.02, "TRAIN-CONFLICT",
         "global during joint training; per-organ plasticity_gate mitigates but the RATE is shared"),
        ("hebbian_max_weight", 45.0, 60.0, "TRAIN-CONFLICT",
         "superset uses max(45,60)=60; both organs' working ranges are below it -> benign at 60"),
    ]


def build_merged_diffbuilder(seed, *, dt_ms=1.0, homeostasis=True, per_region_thresh=True,
                             cross_weight=40.0, n_trained=8, n_novel=4, blk=24, cue_blk=24,
                             per_region_homeo=False):
    """ONE bridge = SURPRISE expectation organ (suffix _S; GABA_B prediction) + Wong-Wang
    SpikingRoleCompetition (NMDA mutual-inhibition WTA), config = the SUPERSET. Reuse-by-import:
    both builders' actual BrainRegion / RegionPathway specs are pulled from their own construction
    and combined into one merged config with reconciled globals. Cross synapse surprise_S -> sel_agent.

    `per_region_homeo` (2026-08-14, DEFAULT-OFF -> byte-identical to the legacy global path): when True,
    the ONE binding conflict the config-superset BOUNDARY mapped (surprise needs homeostasis ON, role
    needs it OFF) is resolved WITHOUT a new engine primitive -- the global `cfg.enable_homeostasis` is
    forced OFF and the surprise (`_S`) `BrainRegion`s each opt IN via the EXISTING per-region
    `BrainRegion.enable_homeostasis=True` (sim/regions.py:171, built 2026-06-08): the fused engine then
    uses the adapted thresholds for the surprise slice and the fixed `cp_izh_vpeak` for the role slice.
    The `homeostasis` arg is ignored when this is True."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import RegionPathway
    from sim.enums import NeuronModel
    from research.runners._phaseB_multicue_competition_spiking_derisk import SpikingRoleCompetition

    W_exc, W_inh, cue_to_expected = 5.0, 14.0, 0.8

    # Expectation organ SPEC (suffix _S) — pull its region/pathway lists from the real builder.
    brS, cfgS, metaS = build_expectation_circuit(seed, n_trained=n_trained, n_novel=n_novel, blk=blk,
                                                 cue_blk=cue_blk, region_suffix="_S",
                                                 per_region_thresh=per_region_thresh)
    exp_regions = list(cfgS.brain_regions)
    exp_paths = list(cfgS.region_pathways)

    # Role organ SPEC — pull from the real Wong-Wang builder's own bridge config.
    rc = SpikingRoleCompetition(seed=seed)
    role_regions = list(rc.bridge.core_config.brain_regions)
    role_paths = list(rc.bridge.core_config.region_pathways)

    # PER-REGION HOMEOSTASIS (the config-superset BOUNDARY unblock, using the EXISTING engine primitive).
    # surprise (`_S`) regions opt IN to intrinsic-threshold adaptation; role regions keep the default
    # (`enable_homeostasis=False` -> fixed vpeak). Global flag forced OFF so the role WTA's graded margin
    # is not corrupted by adaptation, while the surprise slice keeps its native homeostasis operating point.
    if per_region_homeo:
        homeostasis = False
        for r in exp_regions:
            r.enable_homeostasis = True

    # SUPERSET config.
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.per_region_threshold_heterogeneity = bool(per_region_thresh)
    cfg.dt_ms = float(dt_ms)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = 0.06        # expectation's rate (role edges gated / installed, not trained here)
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_max_weight = 60.0           # superset = max(45, 60)
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
    # THE CONFIG SUPERSET — BOTH mechanisms' currents coexist in one bridge.
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.22
    cfg.gabab_conductance_max = 0.0
    cfg.enable_nmda = True                  # role's Wong-Wang accumulator
    cfg.enable_homeostasis = bool(homeostasis)
    cfg.fast_spike_reset = True

    cfg.brain_regions = exp_regions + role_regions
    cross = RegionPathway(from_region="surprise_S", to_region="sel_agent", density=1.0,
                          weight_mean=float(cross_weight), weight_jitter=0.0, plastic=False)
    cfg.region_pathways = exp_paths + role_paths + [cross]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge._blk = blk

    # Re-install the expectation organ's block-diagonal intra-wiring on the MERGED bridge.
    _install_block_diagonal(bridge, "patient_asserted_S", "surprise_S", blk, W_exc)
    _install_block_diagonal(bridge, "patient_expected_S", "surprise_S", blk, W_inh)
    _install_block_diagonal(bridge, "cue_S", "patient_expected_S", blk, cue_to_expected)

    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    meta = dict(metaS); meta["blk"] = blk
    return bridge, cfg, meta


def _role_rates(bridge, cue, sgn, xp, *, drive_pA=3500.0, settle=40, extra_current=None):
    """Drive one role cue sub-pop (cue_{cue}_{sgn}) [+ optional extra current], settle the WTA,
    return {role: mean sel firing rate}. Replicates SpikingRoleCompetition._noun_role_rates on the
    MERGED bridge (reads the merged region_manager indices)."""
    import numpy as np
    _hard_reset(bridge)
    cur = np.zeros(int(bridge.cp_membrane_potential_v.shape[0]), dtype=np.float32)
    if cue is not None:
        cur[_idx(bridge, f"cue_{cue}_{sgn}")] = np.float32(drive_pA)
    if extra_current is not None:
        cur = cur + extra_current
    bridge.cp_external_input_current[:] = xp.asarray(cur)
    sel = {r: _idx(bridge, f"sel_{r}") for r in ("agent", "patient")}
    counts = {r: 0 for r in sel}
    for _ in range(settle):
        _step(bridge)
        fs = bridge.cp_firing_states
        for r in sel:
            counts[r] += int(fs[sel[r]].sum())
    bridge.cp_external_input_current[:] = 0.0
    return {r: counts[r] / max(len(_host(sel[r])), 1) / (settle * 1e-3) for r in sel}


def _surprise_current(bridge, meta, xp, *, condition, cue_pa=600.0, assert_pa=600.0):
    """Build the external-current vector that drives the expectation organ into CONFIRM or CONTRADICT
    (so surprise_S fires or cancels). Used to feed the cross synapse during the role WTA read."""
    import numpy as np
    n = int(bridge.cp_membrane_potential_v.shape[0]); blk = meta["blk"]
    cur = np.zeros(n, dtype=np.float32)
    i = 0                                   # fact 0
    cue_idx = _idx(bridge, "cue_S"); asrt = _idx(bridge, "patient_asserted_S")
    cur[cue_idx[i * blk:(i + 1) * blk]] = np.float32(cue_pa)
    j = i if condition == "confirm" else (1 % meta["n_trained"])   # confirm: assert i; contradict: assert j
    cur[asrt[j * blk:(j + 1) * blk]] = np.float32(assert_pa)
    return cur


def run_seed_diffbuilder(seed, *, dt_ms=1.0, homeostasis=True, per_region_thresh=True,
                         cross_weight=40.0, n_reps=22, verbose=True):
    from sim.backend import get_backend
    xp, _ = get_backend()

    # DETERMINISM: build twice, hash substrate incl. thresholds.
    b1, _, _ = build_merged_diffbuilder(seed, dt_ms=dt_ms, homeostasis=homeostasis,
                                        per_region_thresh=per_region_thresh, cross_weight=cross_weight)
    b2, _, _ = build_merged_diffbuilder(seed, dt_ms=dt_ms, homeostasis=homeostasis,
                                        per_region_thresh=per_region_thresh, cross_weight=cross_weight)
    det_ok = (_arr_hash(b1.cp_membrane_potential_v) == _arr_hash(b2.cp_membrane_potential_v)
              and _arr_hash(b1.cp_connections.tocsr().data) == _arr_hash(b2.cp_connections.tocsr().data)
              and _arr_hash(b1.cp_neuron_firing_thresholds) == _arr_hash(b2.cp_neuron_firing_thresholds))

    merged, cfg_m, meta = build_merged_diffbuilder(seed, dt_ms=dt_ms, homeostasis=homeostasis,
                                                   per_region_thresh=per_region_thresh, cross_weight=cross_weight)

    # ONE shared pool: expectation organ (4 regions) + role organ (sel/sel_FS/cue pops) in one array.
    import numpy as np
    exp_regions = ("cue_S", "patient_expected_S", "patient_asserted_S", "surprise_S")
    role_regions = (["sel_agent", "sel_patient", "sel_FS_agent", "sel_FS_patient"]
                    + [f"cue_{c}_{s}" for c in ("position", "animacy", "verbfit", "lexbias")
                       for s in ("pos", "neg")])
    n_all = int(merged.cp_membrane_potential_v.shape[0])
    all_names = list(exp_regions) + role_regions
    sizes = {nm: len(_idx(merged, nm)) for nm in all_names}
    one_pool = bool(n_all >= sum(sizes.values())) and all(int(_idx(merged, nm).max()) < n_all for nm in all_names)

    # INIT byte-identity for the CO-RESIDENT expectation organ (its per-neuron init must be invariant to
    # the DIFFERENT-builder role organ). Compare its slice merged-vs-standalone (per-region threshold fix).
    solo, _, _ = build_expectation_circuit(seed, n_trained=meta["n_trained"], n_novel=meta["n_novel"],
                                           blk=meta["blk"], cue_blk=meta["cue_blk"], region_suffix="_S",
                                           per_region_thresh=per_region_thresh)
    init_err = 0.0
    for r in exp_regions:
        mi = _idx(merged, r); si = _idx(solo, r)
        for nm in _INIT_PER_NEURON_ARRAYS:
            am = np.asarray(_host(getattr(merged, nm)))[mi]
            aso = np.asarray(_host(getattr(solo, nm)))[si]
            init_err = max(init_err, float(np.abs(am - aso).max()))
    init_byte_id = bool(init_err <= 1e-6)

    # ── Expectation organ FUNCTIONAL (surprise separation) on the merged superset bridge. ──
    idx_S = _idx_map(merged, "_S", xp)
    train_expectation(merged, cfg_m, idx_S, meta, xp, n_reps=n_reps)
    cfg_m.enable_hebbian_learning = False
    resS = measure_conditions(merged, cfg_m, idx_S, meta, xp)
    surp_sep = resS["contradict_hz"] / max(resS["confirm_hz"], 1e-6)
    surprise_functional = bool(surp_sep >= 2.0 and resS["contradict_hz"] >= 5.0)

    # ── Role organ FUNCTIONAL (Wong-Wang WTA selects the DRIVEN role) on the merged bridge. ──
    #    Drive agent-cue only -> agent wins; patient-cue only -> patient wins (uses the init cue->role
    #    weight, no role training needed; tests the mutual-inhibition WTA at the merged dt/homeostasis). ──
    rr_agent = _role_rates(merged, "position", "pos", xp)     # -> sel_agent
    rr_patient = _role_rates(merged, "position", "neg", xp)   # -> sel_patient
    agent_wins = rr_agent["agent"] - rr_agent["patient"]
    patient_wins = rr_patient["patient"] - rr_patient["agent"]
    role_wta_functional = bool(agent_wins > 1.0 and patient_wins > 1.0)

    # ── CROSS synapse LOAD-BEARING: surprise_S (contradict vs confirm) biases sel_agent via surprise_S->sel_agent.
    #    Baseline drive = patient-cue (patient would win); surprise ON should lift sel_agent. Lesion -> no lift. ──
    from tools.lab import attributable_to
    base_neg = _idx(merged, "cue_position_neg")
    def _agent_with_surprise(cond):
        cur = _surprise_current(merged, meta, xp, condition=cond)
        cur = cur.copy(); cur[base_neg] = np.float32(3500.0)   # patient-biasing baseline
        rr = _role_rates(merged, None, None, xp, extra_current=xp.asarray(cur))
        return rr["agent"]
    agent_contra = _agent_with_surprise("contradict")
    agent_confirm = _agent_with_surprise("confirm")
    cross_intact = agent_contra - agent_confirm
    nk, nz = _install_block_diagonal_full(merged, "surprise_S", "sel_agent", 0.0)
    agent_contra_l = _agent_with_surprise("contradict")
    agent_confirm_l = _agent_with_surprise("confirm")
    cross_lesion = agent_contra_l - agent_confirm_l
    cross_frac = attributable_to("sel_agent bias @ surprise_S->sel_agent cross synapse", cross_intact, cross_lesion)
    cross_load_bearing = bool(abs(cross_intact) >= 1.0 and abs(cross_intact) >= 5.0 * max(abs(cross_lesion), 1e-6)
                              and (cross_frac is None or cross_frac >= 0.8))

    structural_go = bool(one_pool and det_ok and init_byte_id and surprise_functional
                         and role_wta_functional and cross_load_bearing)
    res = {
        "seed": seed, "dt_ms": dt_ms, "homeostasis": homeostasis,
        "determinism_ok": det_ok, "one_shared_pool": one_pool, "n_all_neurons": n_all,
        "region_sizes": sizes,
        "init_maxerr": float(init_err), "init_byte_identical": init_byte_id,
        "surprise_separation_ratio": float(surp_sep),
        "surprise_confirm_hz": float(resS["confirm_hz"]), "surprise_contradict_hz": float(resS["contradict_hz"]),
        "surprise_functional": surprise_functional,
        "role_agent_margin_hz": float(agent_wins), "role_patient_margin_hz": float(patient_wins),
        "role_wta_functional": role_wta_functional,
        "cross_intact_hz": float(cross_intact), "cross_lesion_hz": float(cross_lesion),
        "cross_attribution_frac": (float(cross_frac) if cross_frac is not None else None),
        "cross_load_bearing": cross_load_bearing,
        "structural_go": structural_go,
    }
    if verbose:
        print(f"  [seed {seed} dt={dt_ms} homeo={homeostasis}] pool={one_pool}(N={n_all}) det={det_ok} "
              f"INIT={init_err:.1e}({init_byte_id}) | surprise sep={surp_sep:.1f}x({surprise_functional}) | "
              f"role WTA a+{agent_wins:.1f}/p+{patient_wins:.1f}Hz({role_wta_functional}) | "
              f"cross intact={cross_intact:+.2f} lesion={cross_lesion:+.2f}({cross_load_bearing}) | "
              f"struct-GO={structural_go}", flush=True)
    return res


def _gate(n_go, n):
    return "GO" if ((n >= 6 and n_go >= 5) or (n < 6 and n_go == n)) else "BOUNDARY"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["norgan", "diffbuilder"], default="norgan")
    ap.add_argument("--n-organs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--n-reps", type=int, default=22)
    ap.add_argument("--cross-weight", type=float, default=None)
    ap.add_argument("--dt-ms", type=float, default=1.0, help="(diffbuilder) merged global timestep")
    ap.add_argument("--no-homeostasis", action="store_true", help="(diffbuilder) build with homeostasis OFF")
    ap.add_argument("--legacy-global-thresh", action="store_true",
                    help="Disable the per-region threshold fix (reproduces the INIT-invariance BOUNDARY).")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    per_region_thresh = not args.legacy_global_thresh
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    if args.mode == "norgan":
        cw = 12.0 if args.cross_weight is None else args.cross_weight
        print(f"=== ONE-BRAIN MERGE — {args.n_organs} ORGANS on ONE shared spiking substrate ===")
        print(f"    per-region threshold heterogeneity: {'ON (merge-closing fix)' if per_region_thresh else 'OFF (BOUNDARY)'}")
        results = [run_seed_norgan(s, n_organs=args.n_organs, n_reps=args.n_reps, cross_weight=cw,
                                   per_region_thresh=per_region_thresh) for s in seeds]
        n = len(results)
        n_pool = sum(r["one_shared_pool"] for r in results)
        n_det = sum(r["determinism_ok"] for r in results)
        n_init = sum(r["init_byte_identical"] for r in results)
        n_func = sum(r["all_organs_functional"] for r in results)
        n_lb = sum(r["all_pairs_load_bearing"] for r in results)
        n_struct = sum(r["structural_go"] for r in results)
        max_init = max(r["init_maxerr"] for r in results)
        print("\n=== VERDICT (N-organ) ===")
        print(f"  one shared neuron pool:              {n_pool}/{n}")
        print(f"  determinism (cfg.seed incl thresh):  {n_det}/{n}")
        print(f"  INIT byte-identity (per-region fix): {n_init}/{n}  ->  {_gate(n_init, n)}  (max err {max_init:.2e})")
        print(f"  all organs functional (surprise sep):{n_func}/{n}")
        print(f"  all cross pairs load-bearing:        {n_lb}/{n}")
        print(f"  STRUCTURAL {args.n_organs}-ORGAN MERGE:          {n_struct}/{n}  ->  {_gate(n_struct, n)}")
        payload = {"mode": "one_brain_merge_norgan", "n_organs": args.n_organs,
                   "per_region_threshold_heterogeneity": per_region_thresh, "results": results, "n_seeds": n,
                   "n_one_shared_pool": n_pool, "n_determinism_ok": n_det, "n_init_byte_identical": n_init,
                   "n_all_organs_functional": n_func, "n_all_pairs_load_bearing": n_lb,
                   "n_structural_go": n_struct, "max_init_maxerr": max_init,
                   "structural_verdict": _gate(n_struct, n), "init_byteid_verdict": _gate(n_init, n),
                   "cross_weight": cw}
    else:
        cw = 40.0 if args.cross_weight is None else args.cross_weight
        homeo = not args.no_homeostasis
        print("=== ONE-BRAIN MERGE — DIFFERENT BUILDERS (expectation GABA_B + Wong-Wang role NMDA) via config SUPERSET ===")
        print(f"    dt_ms={args.dt_ms}  homeostasis={homeo}  per-region-thresh={per_region_thresh}")
        print("    GLOBAL CONFIG CONFLICT MAP:")
        for f, ev, rv, cls, note in _global_config_conflict_map():
            print(f"      {f:26s} exp={ev!s:6s} role={rv!s:6s}  [{cls}]  {note}")
        results = [run_seed_diffbuilder(s, dt_ms=args.dt_ms, homeostasis=homeo,
                                        per_region_thresh=per_region_thresh, cross_weight=cw,
                                        n_reps=args.n_reps) for s in seeds]
        n = len(results)
        n_pool = sum(r["one_shared_pool"] for r in results)
        n_det = sum(r["determinism_ok"] for r in results)
        n_init = sum(r["init_byte_identical"] for r in results)
        n_surp = sum(r["surprise_functional"] for r in results)
        n_role = sum(r["role_wta_functional"] for r in results)
        n_lb = sum(r["cross_load_bearing"] for r in results)
        n_struct = sum(r["structural_go"] for r in results)
        max_init = max(r["init_maxerr"] for r in results)
        print("\n=== VERDICT (different-builders config superset) ===")
        print(f"  one shared neuron pool:              {n_pool}/{n}")
        print(f"  determinism (cfg.seed incl thresh):  {n_det}/{n}")
        print(f"  INIT byte-id (co-resident expect.):  {n_init}/{n}  ->  {_gate(n_init, n)}  (max err {max_init:.2e})")
        print(f"  expectation organ functional:        {n_surp}/{n}")
        print(f"  role WTA organ functional:           {n_role}/{n}")
        print(f"  cross synapse load-bearing:          {n_lb}/{n}")
        print(f"  STRUCTURAL DIFFERENT-BUILDER MERGE:  {n_struct}/{n}  ->  {_gate(n_struct, n)}")
        payload = {"mode": "one_brain_merge_diffbuilder", "dt_ms": args.dt_ms, "homeostasis": homeo,
                   "per_region_threshold_heterogeneity": per_region_thresh,
                   "config_conflict_map": [{"field": f, "expectation": ev, "role": rv, "class": cls, "note": nt}
                                           for f, ev, rv, cls, nt in _global_config_conflict_map()],
                   "results": results, "n_seeds": n, "n_one_shared_pool": n_pool, "n_determinism_ok": n_det,
                   "n_init_byte_identical": n_init, "n_surprise_functional": n_surp,
                   "n_role_wta_functional": n_role, "n_cross_load_bearing": n_lb, "n_structural_go": n_struct,
                   "max_init_maxerr": max_init, "structural_verdict": _gate(n_struct, n),
                   "init_byteid_verdict": _gate(n_init, n), "cross_weight": cw}

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
