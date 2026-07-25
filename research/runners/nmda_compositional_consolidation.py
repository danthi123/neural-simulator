"""A1 surpass: NMDA-supported concept pools enable REPLAY-driven consolidation
of a compositional (noun, adjective) binding into cortex, recallable with the
HIPPOCAMPUS LESIONED.

Background (the wall):
  * 2026-05-21 (TERMINAL): compositional (noun,adj) bindings are stranded in
    the hippocampal engram -- there is NO ca1 -> concept-pool consolidation
    wire, so replay (which strengthens ca1 -> motor / ca1 -> lang_out only)
    cannot transfer them to cortex.
  * 2026-05-22 (NEGATIVE): adding the ca1 -> concept-pool wire is NECESSARY BUT
    INSUFFICIENT. The concept pools are built with deliberately WEAK internal
    dynamics (a v14/v16 multi-concept-training stability choice), so the
    ca1 drive cannot ignite them into a readable consolidated attractor.

The fresh enabler (2026-07-24, P0.3 affect-state region GO): a slow-NMDA
reverberatory attractor holds a persistent assembly (the dlPFC WM-latch
operating point). The exact substrate primitive is the per-pathway
exc_receptor="nmda_slow" recurrent receptor (Wang 2001/2002): AMPA-suppressed
slow NR2B recurrence, so a recurrent can hold a graded attractor WITHOUT the
fast-AMPA synchronous ping-pong that breaks multi-concept training. That
dissolves the 05-22 tension.

The enhanced surpass (compose existing pieces, NO sim/ edit):
  1. ca1 -> concept-pool wire  (05-22, appended pathways; zero-init, learns
     selective reinstatement during ENCODING so the tag reinstates the fact's
     own noun+adj pools during replay).
  2. cross_pool_concept noun->adj cortical pathway (builder opt-in; the cortical
     store; ZEROED after encoding so REPLAY is the load-bearing binder).
  3. an nmda_slow self-loop attractor on the concept pools, held by a
     transmission gate "nmda_attractor" (CLOSED during training => weak pools =>
     clean Phase-1; OPEN during consolidation+recall => the reverberatory
     attractor ignites from the modest consolidated drive).
  4. concept-selective replay (run_concept_replay_phase): the tag reinstates the
     fact's noun+adj pools (via the learned ca1->concept wire), the attractor
     holds them, and STDP on the open cross_pool_concept pathway binds noun->adj
     cortically -- hippo-independent thereafter.

GO test (per seed): after N replay cycles, cue a NOUN with the HIPPOCAMPUS
LESIONED; the bound adjective pool is selectively (top-among-adj) and
above-floor active. Anti-cheats:
  (a) NO-REPLAY   -> no cortical recall (consolidation requires replay).
  (b) NMDA-LESION -> reproduces the 05-22 weak-dynamics failure (proves NMDA is
      the load-bearing new ingredient).
  (c) HIPPO-LESION-BEFORE-CONSOLIDATION -> fails (binding not yet consolidated).
  (d) NO-CONFAB   -> a withheld (never-consolidated) noun is not recalled.

Reuse-by-import: build_biological_brain_regions, _encode_facts,
run_concept_replay_phase, set_sleep_gates, freeze_all_gates,
apply_concept_topographic_bias, train_word_to_pool, measure_pool_firing.
NO sim/ edit (all attributes -- enable_nmda_recurrent, exc_receptor="nmda_slow",
transmission_gate, cross_pool_concept pathways -- are pre-existing additive
config). cfg.seed set (NOT actual_seed_used). Controller-driven; GPU (cupy).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.runners.concept_pool_demo as cpd
from research.runners.unified_per_regime_monitor_runner import (
    _phase1_recipe,
    _phase1_train_kwargs,
    _encode_facts,
    _all_words_word_to_idx,
    _N_WORDS_ORTHOGONAL,
)
from research.runners.consolidation_trainer import run_concept_replay_phase
from research.runners.text_minimal_isolation import (
    set_sleep_gates, freeze_all_gates,
)

# ---- experiment constants -------------------------------------------------
FACTS_ALL = [("apple", "big"), ("river", "small"),
             ("dog", "hot"), ("cat", "cold")]
# fact 3 ("cat","cold") is WITHHELD from encode+consolidate -> the no-confab
# probe: "cat" is a trained word-pool, but its fact is never consolidated.
CONSOLIDATED_FACTS = FACTS_ALL[:3]
WITHHELD_FACT = FACTS_ALL[3]

_POOL_OF = {
    "apple": "noun_pool_APPLE", "river": "noun_pool_RIVER",
    "dog": "noun_pool_DOG", "cat": "noun_pool_CAT",
    "big": "adjective_pool_BIG", "small": "adjective_pool_SMALL",
    "hot": "adjective_pool_HOT", "cold": "adjective_pool_COLD",
}
_ADJ_POOLS = ["adjective_pool_%s" % a for a in cpd.ADJECTIVE_NAMES]
_NOUN_POOLS = ["noun_pool_%s" % n for n in cpd.NOUN_NAMES]
_HIPPO_REGIONS = ["ec", "dg", "dg_pv_basket", "ca3", "ca1"]

# concept-pool plasticity gates that must be CLOSED so Phase-1 word training is
# strictly language_input -> pool (train_word_to_pool re-opens only what it
# needs, per word, and closes after).
_CONCEPT_GATES = [
    "language_input_to_motor", "language_input_to_noun_pool",
    "language_input_to_verb_pool", "language_input_to_adjective_pool",
    "motor_to_language_output", "noun_pool_to_language_output",
    "verb_pool_to_language_output", "adjective_pool_to_language_output",
    "cross_pool_concept", "ca1_to_concept_pool",
]


def _try_pgate(bridge, name, value):
    try:
        bridge.set_plasticity_gate(name, float(value))
        return True
    except KeyError:
        return False


def _try_tgate(bridge, name, value):
    try:
        bridge.set_transmission_gate(name, float(value))
        return True
    except KeyError:
        return False


# ---------------------------------------------------------------------------
# Substrate: build_biological_brain_regions + augmentations.
# ---------------------------------------------------------------------------
def build_substrate(seed, args):
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import RegionPathway
    from research.runners.text_minimal_isolation import build_biological_brain_regions

    dims = _phase1_recipe(False)
    n_lang_input = int(dims["n_lang_input"])
    n_per_pool = int(dims["n_per_pool"])
    n_fs_per_pool = int(dims["n_fs_per_pool"])
    n_dlpfc_verb = int(dims["n_dlpfc_verb"])

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=0.10,
        motor_exc_weight_mean=2.0,
        motor_inh_weight_mean=4.0,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_noun_pools=True,
        noun_pool_names=cpd.NOUN_NAMES,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=True,
        verb_pool_names=cpd.VERB_NAMES,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=True,
        adjective_pool_names=cpd.ADJECTIVE_NAMES,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        # WEAK concept dynamics (v14/v16 -- Phase-1 stability). The NMDA
        # attractor is supplied by the gated nmda_slow self-loop, not by the
        # pool's own (weak, AMPA) internal recurrence.
        concept_pool_internal_density=0.05,
        concept_pool_exc_weight_mean=0.3,
        concept_pool_inh_weight_mean=0.8,
        # the cortical noun<->adj store (zero-init, plastic, gate cross_pool_concept)
        enable_cross_pool_concept_pathways=True,
        cross_pool_concept_density=float(args.cross_pool_density),
        enable_hippocampus_consolidation=True,
        enable_dlpfc_verb=True,
        n_dlpfc_verb=n_dlpfc_verb,
        dlpfc_verb_internal_density=0.15,
    )

    pathways = list(pathways)
    regions = list(regions)
    # SPARSE-PHENOTYPE override (2026-07-25 escalation ladder): the consolidation write is code-overlap-bounded; the
    # lever is a SPARSE fact-specific CA1 code, which no cheap inhibition achieved (feedback-FFI / sparse-commit / gentle-
    # drive all left the code ~73-91% active). Real DG granule cells are ~2% active — down-state-STABLE, HIGH-threshold,
    # strongly-ADAPTING (a natural k-WTA). The default IZH2007_HIPPO_PYRAMIDAL (vt=-40, d=50, b=+5) is far too excitable
    # for a pattern-separator. Give the chosen hippocampal regions a sparse MSN-like phenotype (vt=-25, vr=-80, b=-20,
    # d=150) so only strongly-driven cells fire -> a sparse code the write CAN localize. Additive/default-off (byte-identical
    # when hippo_izh_type unset). Biologically motivated (DG/CA3 sparsity is a real high-threshold/adaptation phenotype).
    _hippo_izh = getattr(args, "hippo_izh_type", None)
    if _hippo_izh:
        _tgt = set(str(getattr(args, "hippo_izh_regions", "dg")).split(","))
        for r in regions:
            if getattr(r, "name", None) in _tgt:
                r.izh_neuron_type = _hippo_izh
    concept_pools = _NOUN_POOLS + ["verb_pool_%s" % v for v in cpd.VERB_NAMES] + _ADJ_POOLS
    skip_nmda = bool(getattr(args, "skip_nmda_additions", False))

    # (0) CA1 FFI-kWTA sparsification (Rank-2 de-risk, 2026-07-25 re-attribution): the A1 boundary is a DENSE/OVERLAPPING
    #     CA1 code (6-seed: ~90% of CA1 fires WEAKLY per tag, Jaccard 0.56 -> no ca1->slot write can localize). A
    #     feedforward+feedback inhibitory basket on CA1 (mirrors the shipped comp_attr_inh WTA pool) sparsifies CA1's
    #     tag-response toward a sparse DISTINCT ensemble so the write CAN localize. Additive/default-off
    #     (ca1_ffi_kwta=False -> byte-identical to the current A1 build).
    if bool(getattr(args, "ca1_ffi_kwta", False)):
        from sim.regions import BrainRegion as _BR
        regions = list(regions) + [_BR(name="ca1_ffi", n_neurons=int(getattr(args, "ca1_ffi_n", 30)),
            exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.1,
            plastic_internal=False)]
        pathways.append(RegionPathway(from_region="ca1", to_region="ca1_ffi", density=0.4,
            weight_mean=float(getattr(args, "ca1_ffi_drive", 3.0)), weight_jitter=0.2, plastic=False))
        pathways.append(RegionPathway(from_region="ca1_ffi", to_region="ca1", density=0.5,
            weight_mean=float(getattr(args, "ca1_ffi_inh", 5.0)), weight_jitter=0.2, plastic=False))

    # (1) ca1 -> concept-pool consolidation wire (zero-init; learns selective
    #     reinstatement during encoding). Gate: ca1_to_concept_pool.
    n_ca1_wire = 0
    for pool in concept_pools:
        pathways.append(RegionPathway(
            from_region="ca1", to_region=pool,
            density=float(args.ca1_concept_density),
            weight_mean=float(args.ca1_concept_weight), weight_jitter=0.3,
            plastic=True, plasticity_gate="ca1_to_concept_pool",
        ))
        n_ca1_wire += 1

    # (3) nmda_slow self-loop attractor on noun + adjective pools, held by the
    #     transmission gate "nmda_attractor" (closed during training).
    nmda_pools = _NOUN_POOLS + _ADJ_POOLS
    n_self = 0
    if not skip_nmda:
        for pool in nmda_pools:
            pathways.append(RegionPathway(
                from_region=pool, to_region=pool,
                density=float(args.nmda_self_density),
                weight_mean=float(args.nmda_self_weight), weight_jitter=0.05,
                plastic=False, exc_receptor="nmda_slow",
                transmission_gate="nmda_attractor",
            ))
            n_self += 1

    # (4) DEDICATED COMPOSITIONAL-ATTRACTOR region (Option 1, research-gate 2026-07-25):
    #     a SEPARATE strong Wang-2002 region (NOT the weak Phase-1 pools, which stay
    #     weak so Phase-1 is untouched). The consolidated composite lives + self-sustains
    #     HERE. Read out here. One sub-population per fact-slot (WTA between them via a
    #     shared inhibitory pool = one-of-N selective ignition, not a single global winner).
    #     Additive/default-off (comp_attractor_slots=0 -> byte-identical to the A1 build).
    from sim.regions import BrainRegion
    n_slots = int(getattr(args, "comp_attractor_slots", 0))
    comp_dend = bool(getattr(args, "comp_dendritic", False))   # dendritic surpass (DESIGN 2026-07-25): route ca1->slot through the two-compartment WEIGHTED-coincidence bistable plateau (on-bridge reuse, no sim/ edit)
    n_comp = 0
    if n_slots > 0:
        n_per = int(getattr(args, "comp_attractor_n_per", 120))
        # one excitatory sub-assembly per slot + a shared FS inhibitory pool for WTA
        for s in range(n_slots):
            regions = list(regions) + [BrainRegion(
                name=f"comp_attr_{s}", n_neurons=n_per, exc_fraction=1.0,
                internal_density=0.20, exc_weight_mean=2.0, inh_weight_mean=0.0,
                weight_jitter=0.3, plastic_internal=False)]
            # strong nmda_slow self-loop = the Wang-2002 hold (gate nmda_attractor)
            pathways.append(RegionPathway(
                from_region=f"comp_attr_{s}", to_region=f"comp_attr_{s}",
                density=0.20, weight_mean=float(getattr(args, "comp_self_weight", 12.0)),
                weight_jitter=0.05, plastic=False, exc_receptor="nmda_slow",
                transmission_gate="nmda_attractor"))
            # ca1 -> this slot (plastic, potentiates during co-activation replay). With comp_dendritic: route it through
            # the per-slot two-compartment coincidence plateau (WEIGHTED = graded by the potentiated ca1->slot weights).
            pathways.append(RegionPathway(
                from_region="ca1", to_region=f"comp_attr_{s}",
                density=float(args.ca1_concept_density), weight_mean=float(args.ca1_concept_weight),
                weight_jitter=0.3, plastic=True, plasticity_gate="ca1_to_comp_attr",
                coincidence_detector=comp_dend))
            n_comp += 1
        # shared inhibitory WTA pool: each slot drives it, it inhibits all slots
        regions = list(regions) + [BrainRegion(
            name="comp_attr_inh", n_neurons=int(n_per * 0.5), exc_fraction=0.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.1, plastic_internal=False)]
        for s in range(n_slots):
            pathways.append(RegionPathway(from_region=f"comp_attr_{s}", to_region="comp_attr_inh",
                                          density=0.30, weight_mean=3.0, weight_jitter=0.2, plastic=False))
            pathways.append(RegionPathway(from_region="comp_attr_inh", to_region=f"comp_attr_{s}",
                                          density=0.30, weight_mean=float(getattr(args, "comp_wta_weight", 5.0)),
                                          weight_jitter=0.2, plastic=False))
        # concept pools -> the slots (the cortical composite feeds the attractor; plastic). NOTE (c_drive probe 2026-07-25):
        # this is an ALL-pools->ALL-slots BROADCAST -> ca1_i->concept->ALL-slots drives every slot non-selectively (the
        # write-selectivity killer). comp_no_pool_slot=True drops it so routing is purely the potentiated (distinct-engram) ca1->slot.
        if not bool(getattr(args, "comp_no_pool_slot", False)):
            for pool in (_NOUN_POOLS + _ADJ_POOLS):
                for s in range(n_slots):
                    pathways.append(RegionPathway(
                        from_region=pool, to_region=f"comp_attr_{s}",
                        density=0.15, weight_mean=1.5, weight_jitter=0.3,
                        plastic=True, plasticity_gate="concept_to_comp_attr"))

    print(f"  augment: +{n_ca1_wire} ca1->concept wires, "
          f"+{n_self} nmda_slow self-loops (w={args.nmda_self_weight}, "
          f"d={args.nmda_self_density}) skip_nmda={skip_nmda} +{n_comp} comp_attractor slots")

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    # Family B (2026-07-25): ACTIVITY-SCALED divisive normalization on a target hippocampal region -- self-limits its
    # input by (sigma + gain*pop-mean) so a densely-driven region sparsifies WITHOUT the fixed-FFI knife-edge. Additive/
    # default-off (divnorm_regions="" -> byte-identical). Comma-sep region names, e.g. "dg" or "ca1".
    _dn = str(getattr(args, "divnorm_regions", "") or "")
    if _dn:
        cfg.enable_input_divisive_norm = True
        _dnset = set(x.strip() for x in _dn.split(",") if x.strip())
        for _r in cfg.brain_regions:
            if _r.name in _dnset:
                _r.input_divisive_norm = True
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)                      # <-- SEEDS THE SUBSTRATE (not actual_seed_used)
    cfg.enable_nmda = bool(args.enable_global_nmda)
    cfg.nmda_tau_decay = 100.0
    # the load-bearing new ingredient: slow NR2B recurrent receptor
    cfg.enable_nmda_recurrent = (not skip_nmda)
    cfg.nmda_recurrent_tau_decay_ms = 100.0
    cfg.nmda_recurrent_ratio = float(args.nmda_recurrent_ratio)
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = bool(args.enable_hebbian)
    # Option-3 (2026-07-25): a pure RATE-based write instead of spike-timing STDP -- the replay firing IS fact-specific
    # (own/other ~1.5) but STDP-timing flattens it; a rate-Hebbian write should preserve it. Additive/default-off.
    if bool(getattr(args, "no_stdp", False)):
        cfg.enable_stdp = False
    if getattr(args, "hebbian_lr", None) is not None:
        cfg.hebbian_learning_rate = float(args.hebbian_lr)
    if bool(getattr(args, "hebbian_rate_window", False)):
        cfg.hebbian_rate_window = True
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = float(args.stdp_w_max)
    cfg.fast_spike_reset = True

    if comp_dend:   # dendritic surpass: two-compartment bistable WEIGHTED-coincidence plateau on the slots (all default-off; byte-identical when comp_dendritic unset). gap5 GO_CFG + r-iii operating point.
        cfg.enable_coincidence_detection = True
        cfg.coincidence_weighted_drive = True             # grade the plateau by the potentiated ca1->slot weights
        cfg.coincidence_k_threshold = float(getattr(args, "comp_k_thresh", 3.0))   # calibrated to the per-step weighted ca1->slot c_drive
        cfg.enable_two_compartment_dap = True             # separate apical voltage cp_v_apical
        cfg.coincidence_plateau_self_regen = float(getattr(args, "comp_self_regen", 0.15))   # v-gated SUSTAIN latch
        cfg.coincidence_plateau_v_hold = float(getattr(args, "comp_v_hold", -50.0))
        cfg.apical_kir_g = float(getattr(args, "comp_kir_g", 3.0))                 # KIR down-state (silent rest)
        cfg.apical_g_couple = float(getattr(args, "comp_gc", 1.0))                 # apical<-soma back-coupling
        cfg.apical_g_couple_to_soma = float(getattr(args, "comp_gc_read", 5.0))    # apical->soma read (asymmetric)
        cfg.apical_R = float(getattr(args, "comp_apical_R", 50.0))                 # thin-high-R apical
        if bool(getattr(args, "comp_btsp", False)):   # Option-3: one-shot plateau-gated selective write (gap4 BTSP) for a LARGE held-vs-non c_drive separation the STDP write couldn't give
            cfg.enable_btsp = True
            cfg.btsp_learning_rate = float(getattr(args, "comp_btsp_lr", 0.01))
            cfg.btsp_w_max = float(getattr(args, "comp_btsp_wmax", 8.0))
            # Rank-2 stack element 3 (rate-gated HETEROSYNAPTIC write, 2026-07-25): potentiate the strong distinct-core
            # ca1 inputs (high Etilde) AND thresholded-depress the weak dense HALO (low Etilde) -> the write focuses on
            # the distinct core the flood otherwise dilutes. lam_dep=0 -> byte-identical to pure-potentiation BTSP.
            cfg.btsp_hetero_dep = float(getattr(args, "comp_btsp_hetero_dep", 0.0))
            cfg.btsp_hetero_theta = float(getattr(args, "comp_btsp_hetero_theta", 0.0))
            cfg.btsp_elig_exponent = float(getattr(args, "comp_btsp_elig_exp", 1.0))   # supralinear eligibility (widen core-halo gap)
            # HARD-THRESHOLD write-side k-WTA on the eligibility (2026-07-25 write-side de-risk): only the sustained-
            # firing CA1 core (pre-elig >= thresh*peak) contributes to the ca1->slot potentiation. 0.0 => byte-identical.
            cfg.btsp_elig_hard_thresh = float(getattr(args, "comp_btsp_elig_hard_thresh", 0.0))
            # PER-FACT-WINDOWED eligibility (2026-07-25 write-side NO-GO follow-up): the default tau=1000ms integrates
            # ACROSS the multi-fact write (cross-fact compressed -> a magnitude threshold can't isolate a per-fact core).
            # A SHORT tau (~the per-fact burst length) makes the eligibility track the CURRENT fact's firing only, so the
            # hard threshold CAN isolate the per-fact core. Default keeps the shipped 1000ms.
            if getattr(args, "comp_btsp_elig_tau", None) is not None:
                cfg.btsp_elig_tau_ms = float(args.comp_btsp_elig_tau)
            # M1' (2026-07-25): the DENDRITIC SUSTAINED-SPIKE-COUNT gate — a per-source BOX-CAR windowed spike COUNT
            # (reset per write window via bridge.reset_btsp_window()) through an ABSOLUTE Hill gate, applied to the
            # eligibility BEFORE the synaptic sum. Unlike elig_hard_thresh/elig_exponent above it is NOT normalized by
            # etilde.max() over the whole bridge (a spine has no network max — CaMKII thresholds an absolute Ca2+
            # set-point). theta <= 0.0 (default) => the counter is never allocated => byte-identical.
            cfg.btsp_win_gate_theta = float(getattr(args, "comp_btsp_win_gate_theta", 0.0))
            cfg.btsp_win_gate_hill_n = float(getattr(args, "comp_btsp_win_gate_hill_n", 8.0))

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


# ---------------------------------------------------------------------------
# Phase-1 training (word -> pool). nmda_attractor CLOSED throughout.
# ---------------------------------------------------------------------------
def train_phase1(bridge, seed, n_events, verbose=False):
    tk = _phase1_train_kwargs(False)
    all_words_ordered = (
        list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB)
        + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(all_words_ordered)}
    n_words_total = len(all_words_ordered)

    # strictly-isolated Phase-1: freeze everything; close attractor.
    freeze_all_gates(bridge)
    for g in _CONCEPT_GATES:
        _try_pgate(bridge, g, 0.0)
    _try_tgate(bridge, "nmda_attractor", 0.0)   # weak pools during training

    cpd.apply_concept_topographic_bias(
        bridge, n_lang_input=int(tk["n_lang_input"]),
        topographic_factor=float(tk["topographic_factor"]),
        off_target_factor=float(tk["off_target_factor"]),
        sparsity=float(tk["sparsity"]), orthogonal_codes=True,
        n_words_for_orthogonal=int(n_words_total), word_to_idx=word_to_idx,
        skip_motor=False, verbose=False)

    targets = []
    for w, a in cpd.DIRECTION_VOCAB.items():
        targets.append((w, "motor_%s" % a))
    for w, nm in cpd.NOUN_VOCAB.items():
        targets.append((w, "noun_pool_%s" % nm))
    for w, nm in cpd.VERB_VOCAB.items():
        targets.append((w, "verb_pool_%s" % nm))
    for w, nm in cpd.ADJECTIVE_VOCAB.items():
        targets.append((w, "adjective_pool_%s" % nm))

    rng = np.random.default_rng(int(seed))
    buf = [(w, t) for (w, t) in targets for _ in range(int(n_events))]
    rng.shuffle(buf)
    t0 = time.time()
    for i, (w, t) in enumerate(buf):
        cpd.train_word_to_pool(
            bridge, w, t, n_events=1, reset_steps=50,
            n_lang_input=int(tk["n_lang_input"]),
            n_lang_output=int(tk["n_lang_input"]),
            sparsity=float(tk["sparsity"]), orthogonal_codes=True,
            n_words_for_orthogonal=int(n_words_total),
            word_to_idx=word_to_idx, verbose=False)
        if verbose and (i + 1) % 400 == 0:
            print(f"    phase1 {i+1}/{len(buf)} ({(time.time()-t0)/60:.1f} min)")
    if verbose:
        print(f"  Phase-1 done ({len(buf)} events, {(time.time()-t0)/60:.1f} min)")


def direct_binding_sanity(bridge):
    """Direct word->pool retrieval (attractor OFF): sanity that Phase-1 trained."""
    _try_tgate(bridge, "nmda_attractor", 0.0)
    freeze_all_gates(bridge)
    dims = _phase1_recipe(False)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words = max(_N_WORDS_ORTHOGONAL, len(all_words))
    pools = (["motor_%s" % a for a in ("N", "E", "S", "W")]
             + _NOUN_POOLS + ["verb_pool_%s" % v for v in cpd.VERB_NAMES] + _ADJ_POOLS)
    def _tgt(w):
        if w in cpd.DIRECTION_VOCAB: return "motor_%s" % cpd.DIRECTION_VOCAB[w]
        if w in cpd.NOUN_VOCAB: return "noun_pool_%s" % cpd.NOUN_VOCAB[w]
        if w in cpd.VERB_VOCAB: return "verb_pool_%s" % cpd.VERB_VOCAB[w]
        return "adjective_pool_%s" % cpd.ADJECTIVE_VOCAB[w]
    n_ok = 0
    for w in all_words:
        per = cpd.measure_pool_firing(
            bridge, w, pools, stim_steps=100, reset_steps=50, drive_pA=200.0,
            sparsity=0.05, n_lang_input=int(dims["n_lang_input"]),
            orthogonal_codes=True, n_words_for_orthogonal=int(n_words),
            word_to_idx=word_to_idx)
        if max(per.items(), key=lambda kv: kv[1])[0] == _tgt(w):
            n_ok += 1
    return n_ok, len(all_words)


# ---------------------------------------------------------------------------
# Encoding (hippocampal engram + selective ca1->concept reinstatement weights).
# ---------------------------------------------------------------------------
def encode_facts_with_reinstatement(bridge, facts, commit_top_k=None):
    """Encode each fact into the hippocampal engram AND grow selective
    ca1 -> concept reinstatement weights. cross_pool_concept is opened by
    encode_concept_pair internally; its weights are ZEROED afterward so REPLAY
    is the load-bearing cortical binder. `commit_top_k` (default None -> ~85)
    overrides the engram-tag size: a SMALLER value commits only the strongly-firing
    distinct core -> a sparse separable CA1 tag (Rank-2 stack element 1)."""
    dims_r = _phase1_recipe(False)
    all_words, _ = _all_words_word_to_idx()
    dims = {
        "n_lang_input": int(dims_r["n_lang_input"]),
        "n_per_pool": int(dims_r["n_per_pool"]),
        "n_fs_per_pool": int(dims_r["n_fs_per_pool"]),
        "sparsity": 0.05, "dt_ms": 0.5,
        "n_words_for_orthogonal": max(_N_WORDS_ORTHOGONAL, len(all_words)),
    }
    # gate state for encoding: hippocampal ENCODING path open + ca1->concept
    # open+plastic (learns selective reinstatement). attractor OFF (encoding is
    # teacher-driven; the pools fire from lang+teacher, not from the attractor).
    freeze_all_gates(bridge)
    for g in ("lang_to_ec", "ec_to_dg", "dg_to_ca3", "ec_to_ca1", "ca3_to_ca1"):
        _try_pgate(bridge, g, 1.0)
    _try_pgate(bridge, "ca1_to_concept_pool", 1.0)
    _try_tgate(bridge, "nmda_attractor", 0.0)

    tags = _encode_facts(bridge, facts, dims, encoding_steps=200, commit_top_k=commit_top_k)

    # zero the cross_pool_concept weights encode_concept_pair grew -> clean
    # "cortex empty" pre-replay baseline (replay is the load-bearing binder).
    _zero_gate_weights(bridge, "cross_pool_concept")
    return tags, dims


def _zero_gate_weights(bridge, gate_name):
    from sim.backend import get_backend
    cp, _ = get_backend()
    idx = bridge._plasticity_gate_indices_gpu.get(gate_name, None)
    if idx is None or idx.size == 0:
        return 0
    nnz = int(bridge.cp_connections.nnz)
    idx = idx[idx < nnz]
    bridge.cp_connections.data[idx] = cp.float32(0.0)
    return int(idx.size)


def _mean_gate_weight(bridge, gate_name):
    from sim.backend import get_backend
    cp, _ = get_backend()
    idx = bridge._plasticity_gate_indices_gpu.get(gate_name, None)
    if idx is None or idx.size == 0:
        return 0.0
    nnz = int(bridge.cp_connections.nnz)
    idx = idx[idx < nnz]
    return float(bridge.cp_connections.data[idx].mean())


# ---------------------------------------------------------------------------
# Consolidation (concept-selective replay). ca1->concept + cross_pool open;
# attractor open so the reinstated pools ignite/hold and STDP binds noun->adj.
# ---------------------------------------------------------------------------
def consolidate(bridge, tags, cycles, seed, attractor_on=True):
    set_sleep_gates(bridge)                       # ca3_to_ca1, ca3_swr, ca1_to_motor/lang_out
    _try_pgate(bridge, "ca1_to_concept_pool", 1.0)   # reinstatement route (learned)
    _try_pgate(bridge, "cross_pool_concept", 1.0)    # cortical noun->adj binder
    # attractor ON -> reinstated pools ignite/hold so STDP binds noun->adj.
    # NMDA-LESION control keeps it CLOSED (weak pools, reproduces 2026-05-22).
    _try_tgate(bridge, "nmda_attractor", 1.0 if attractor_on else 0.0)
    rng = np.random.default_rng(int(seed) + 777)
    stats = run_concept_replay_phase(bridge, tags, n_replays_per_tag=int(cycles),
                                     drive_pA=1500.0, rng=rng)
    return stats


# ---------------------------------------------------------------------------
# CO-ACTIVATION replay (research-gate 2026-07-25, the potentiation fix). Per fact,
# drive the CA3 tag AND reinstate the fact's noun+adj concept pools, so the plastic
# ca1->slot / pool->slot / ca1->concept wires see pre(ca1)+post(pool/slot) coincidence
# and POTENTIATE (the A1 failure: CA3-only drive -> pools never fire -> wire frozen 0.01).
# ---------------------------------------------------------------------------
def coactivation_replay(bridge, facts, tags, cycles, seed, coactivate=True, attractor_on=True,
                        tag_drive_pA=1500.0, pool_drive_pA=1400.0, slot_drive_pA=1400.0, burst_steps=30):
    """Reinstate the FULL pattern per fact during replay: CA3 tag + the fact's concept pools + the fact's dedicated
    attractor slot (fact i -> comp_attr_i). The hippocampal replay reinstating the cortical target is biology-faithful;
    STDP then binds ca1->slot / pool->slot / ca1->concept from the pre(ca1)+post(pool,slot) coincidence."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    set_sleep_gates(bridge)
    for g in ("ca1_to_concept_pool", "ca1_to_comp_attr", "concept_to_comp_attr", "cross_pool_concept"):
        _try_pgate(bridge, g, 1.0)
    _try_tgate(bridge, "nmda_attractor", 1.0 if attractor_on else 0.0)
    rm = bridge.region_manager
    rng = np.random.default_rng(int(seed) + 777)
    all_region_names = {r.name for r in bridge.core_config.brain_regions}

    def _idx(nm):
        return cp.asarray(list(rm.indices(nm)), dtype=cp.int64) if nm in all_region_names else None
    # pool regions are UPPERCASE (noun_pool_APPLE); facts are lowercase (apple) -> .upper() the word.
    pool_idx, slot_idx = {}, {}
    for i, (noun, adj) in enumerate(facts):
        pool_idx[i] = [a for a in (_idx(f"noun_pool_{noun.upper()}"), _idx(f"adjective_pool_{adj.upper()}")) if a is not None]
        slot_idx[i] = _idx(f"comp_attr_{i}")      # fact i -> its dedicated attractor slot
    order = list(range(len(facts)))
    for _c in range(int(cycles)):
        rng.shuffle(order)                        # interleaved (CLS shuffled replay)
        for i in order:
            tag = tags[i]
            bridge.cp_external_input_current[:] = 0.0
            bridge.stimulate_tag(tag, drive_pA=float(tag_drive_pA), additive=False)   # CA3 engram cue
            if coactivate:                        # reinstate the fact's cortical pools + target slot -> post-spikes
                for a in pool_idx[i]:
                    bridge.cp_external_input_current[a] += float(pool_drive_pA)
                if slot_idx[i] is not None:
                    bridge.cp_external_input_current[slot_idx[i]] += float(slot_drive_pA)
            for _ in range(int(burst_steps)):
                bridge._run_one_simulation_step()
            try:
                bridge.clear_tag_drive(tag)
            except Exception:
                pass
    bridge.cp_external_input_current[:] = 0.0
    return {"cycles": int(cycles), "coactivate": bool(coactivate)}


# ---------------------------------------------------------------------------
# Hippo lesion (clamp hippo indices to -200 pA each step).
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def hippo_lesioned(bridge, silence_pA=-200.0):
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    idx = []
    for r in _HIPPO_REGIONS:
        try:
            idx.extend(list(rm.indices(r)))
        except Exception:
            pass
    arr = cp.asarray(idx, dtype=cp.int64)
    orig = bridge._run_one_simulation_step

    def silenced():
        bridge.cp_external_input_current[arr] = float(silence_pA)
        return orig()
    bridge._run_one_simulation_step = silenced
    try:
        yield
    finally:
        bridge._run_one_simulation_step = orig
        bridge.cp_external_input_current[arr] = 0.0


# ---------------------------------------------------------------------------
# Recall: cue a NOUN, read the adjective pools. Attractor reset per cue.
# ---------------------------------------------------------------------------
def recall_adj_rates(bridge, noun_word, dims, attractor_on=True,
                     cue_steps=120, quiet_steps=80):
    """Reset the attractor, cue the noun via language_input, read per-adj-pool
    firing. Returns dict adj_pool -> per-neuron mean spike count."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern
    rm = bridge.region_manager

    all_words, word_to_idx = _all_words_word_to_idx()
    n_words = int(dims["n_words_for_orthogonal"])
    n_lang = int(dims["n_lang_input"])
    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    adj_arr = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64) for p in _ADJ_POOLS}

    # RESET the attractor: close it, run quiet steps so any latched pool decays.
    _try_tgate(bridge, "nmda_attractor", 0.0)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(quiet_steps):
        bridge._run_one_simulation_step()
    # set the attractor to the condition value, then cue.
    _try_tgate(bridge, "nmda_attractor", 1.0 if attractor_on else 0.0)

    drive = cp.asarray(orthogonal_drive_pattern(
        cue_idx=word_to_idx[noun_word], n_cues=n_words, n_neurons=n_lang,
        drive_max_pA=200.0, sparsity=float(dims["sparsity"])), dtype=cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[lang_arr] = drive
    accum = {p: 0.0 for p in _ADJ_POOLS}
    for _ in range(cue_steps):
        bridge._run_one_simulation_step()
        fired = bridge.cp_firing_states
        for p in _ADJ_POOLS:
            accum[p] += float(cp.sum(fired[adj_arr[p]].astype(cp.float32)))
    bridge.cp_external_input_current[:] = 0.0
    return {p: accum[p] / (cue_steps * max(1, len(adj_arr[p]))) for p in _ADJ_POOLS}


def measure_recall(bridge, facts, dims, attractor_on, floor):
    """For each fact: cue the noun, is the bound adj pool TOP-among-adj AND
    above `floor`?  Also probe the withheld (no-confab) noun."""
    per_fact = []
    for (noun, adj) in facts:
        rates = recall_adj_rates(bridge, noun, dims, attractor_on=attractor_on)
        bound = _POOL_OF[adj]
        ranked = sorted(rates.items(), key=lambda kv: -kv[1])
        top_pool, top_rate = ranked[0]
        rank = [p for p, _ in ranked].index(bound) + 1
        per_fact.append({
            "noun": noun, "adj": adj, "bound_pool": bound,
            "bound_rate": rates[bound], "top_pool": top_pool, "top_rate": top_rate,
            "rank": rank, "selective": (rank == 1),
            "lifted": (rates[bound] > floor),
            "recalled": (rank == 1 and rates[bound] > floor),
            "rates": rates,
        })
    # no-confab: the withheld noun should NOT selectively ignite any adj.
    wn, wa = WITHHELD_FACT
    wrates = recall_adj_rates(bridge, wn, dims, attractor_on=attractor_on)
    w_top_pool, w_top_rate = max(wrates.items(), key=lambda kv: kv[1])
    noconfab = {
        "noun": wn, "top_pool": w_top_pool, "top_rate": w_top_rate,
        "max_rate": w_top_rate, "confabulated": (w_top_rate > floor),
        "rates": wrates,
    }
    return per_fact, noconfab


# ---------------------------------------------------------------------------
# Diagnostics: ca1->concept selectivity + cross_pool growth.
# ---------------------------------------------------------------------------
def diag_ca1_concept_selectivity(bridge, tags, facts):
    """Stimulate each fact's tag (hippo intact), read which concept pools fire
    -- confirms the learned ca1->concept reinstatement is fact-selective."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    pools = _NOUN_POOLS + _ADJ_POOLS
    parr = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64) for p in pools}
    freeze_all_gates(bridge)
    _try_pgate(bridge, "ca1_to_concept_pool", 0.0)
    set_sleep_gates(bridge)               # ca1->concept transmission needs ca3->ca1 etc open
    _try_pgate(bridge, "ca1_to_concept_pool", 0.0)  # freeze plasticity, keep current
    _try_tgate(bridge, "nmda_attractor", 0.0)
    out = []
    for (noun, adj), tag in zip(facts, tags):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()
        bridge.stimulate_tag(tag, drive_pA=1500.0, additive=False)
        acc = {p: 0.0 for p in pools}
        for _ in range(100):
            bridge._run_one_simulation_step()
            fired = bridge.cp_firing_states
            for p in pools:
                acc[p] += float(cp.sum(fired[parr[p]].astype(cp.float32)))
        bridge.clear_tag_drive(tag)
        rates = {p: acc[p] / (100 * max(1, len(parr[p]))) for p in pools}
        out.append({
            "fact": [noun, adj],
            "noun_bound": rates[_POOL_OF[noun]], "adj_bound": rates[_POOL_OF[adj]],
            "noun_top": max((rates[p] for p in _NOUN_POOLS)),
            "adj_top_pool": max(_ADJ_POOLS, key=lambda p: rates[p]),
        })
    bridge.cp_external_input_current[:] = 0.0
    return out


# ---------------------------------------------------------------------------
# One condition (from a post-Phase-1 weight snapshot).
# ---------------------------------------------------------------------------
def _snapshot(bridge):
    return bridge.cp_connections.data.copy()


def _restore(bridge, snap):
    _try_tgate(bridge, "nmda_attractor", 0.0)   # quiet the attractor before settling
    bridge.cp_connections.data[:] = snap
    # clear engram tags so each condition re-encodes cleanly
    for t in list(bridge.list_engram_tags()):
        try:
            bridge.delete_engram_tag(t["name"])
        except Exception:
            pass
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()


def run_condition(bridge, snap, condition, args, seed, diagnostic=False):
    """condition in {full, no_replay, nmda_lesion, hippo_before}."""
    _restore(bridge, snap)
    facts = CONSOLIDATED_FACTS
    diag = {}
    attractor_on = (condition != "nmda_lesion")   # (b): pools never get NMDA

    if condition == "hippo_before":
        # lesion the hippocampus for encode+consolidate: the binding is never
        # laid down (encoding can't reach hippo; replay can't reinstate).
        with hippo_lesioned(bridge):
            tags, dims = encode_facts_with_reinstatement(bridge, facts)
            consolidate(bridge, tags, args.replay_cycles, seed, attractor_on=attractor_on)
    else:
        tags, dims = encode_facts_with_reinstatement(bridge, facts)
        if diagnostic:
            diag["ca1_concept_selectivity"] = diag_ca1_concept_selectivity(bridge, tags, facts)
        cycles = 0 if condition == "no_replay" else args.replay_cycles
        if cycles > 0:
            consolidate(bridge, tags, cycles, seed, attractor_on=attractor_on)

    diag["cross_pool_mean_w"] = _mean_gate_weight(bridge, "cross_pool_concept")
    diag["ca1_concept_mean_w"] = _mean_gate_weight(bridge, "ca1_to_concept_pool")

    # RECALL with the hippocampus lesioned (systems-consolidation test).
    attractor_on = (condition != "nmda_lesion")
    freeze_all_gates(bridge)
    for g in _CONCEPT_GATES:
        _try_pgate(bridge, g, 0.0)
    with hippo_lesioned(bridge):
        per_fact, noconfab = measure_recall(
            bridge, facts, dims, attractor_on=attractor_on, floor=args.floor)

    n_recall = sum(f["recalled"] for f in per_fact)
    n_sel = sum(f["selective"] for f in per_fact)
    n_lift = sum(f["lifted"] for f in per_fact)
    return {
        "condition": condition, "n_facts": len(facts),
        "n_recalled": n_recall, "n_selective": n_sel, "n_lifted": n_lift,
        "per_fact": per_fact, "no_confab": noconfab, "diag": diag,
    }


# ---------------------------------------------------------------------------
def run_seed(seed, args):
    print(f"\n===== SEED {seed} =====", flush=True)
    t0 = time.time()
    bridge = build_substrate(seed, args)
    train_phase1(bridge, seed, args.train_events, verbose=True)
    db_ok, db_n = direct_binding_sanity(bridge)
    print(f"  direct-binding sanity: {db_ok}/{db_n} = {100.0*db_ok/db_n:.1f}%", flush=True)
    snap = _snapshot(bridge)

    conditions = ["full", "no_replay", "nmda_lesion", "hippo_before"]
    results = {}
    for c in conditions:
        r = run_condition(bridge, snap, c, args, seed,
                          diagnostic=(args.diagnostic and c == "full"))
        results[c] = r
        nf = r["n_facts"]
        print(f"  [{c:12s}] recalled {r['n_recalled']}/{nf}  "
              f"selective {r['n_selective']}/{nf}  lifted {r['n_lifted']}/{nf}  "
              f"| no-confab top={r['no_confab']['max_rate']:.4f} "
              f"confab={r['no_confab']['confabulated']} "
              f"| xpool_w={r['diag'].get('cross_pool_mean_w',0):.3f}", flush=True)
        if args.diagnostic and c == "full" and "ca1_concept_selectivity" in r["diag"]:
            for d in r["diag"]["ca1_concept_selectivity"]:
                print(f"        reinstate {d['fact']}: noun_bound={d['noun_bound']:.3f} "
                      f"adj_bound={d['adj_bound']:.3f} adj_top={d['adj_top_pool']}", flush=True)

    # per-seed verdict
    full = results["full"]
    grounded_ok = (full["n_recalled"] >= args.min_recall)
    noconfab_ok = (not full["no_confab"]["confabulated"])
    a_noreplay = (results["no_replay"]["n_recalled"] <= args.antichance)
    b_nmda = (results["nmda_lesion"]["n_recalled"] <= args.antichance)
    c_hippo = (results["hippo_before"]["n_recalled"] <= args.antichance)
    seed_go = grounded_ok and noconfab_ok and a_noreplay and b_nmda and c_hippo
    verdict = {
        "seed": seed, "direct_binding": [db_ok, db_n],
        "full_recalled": full["n_recalled"], "full_n": full["n_facts"],
        "grounded_ok": grounded_ok, "noconfab_ok": noconfab_ok,
        "a_noreplay_ok": a_noreplay, "b_nmda_lesion_ok": b_nmda,
        "c_hippo_before_ok": c_hippo, "seed_go": seed_go,
        "elapsed_min": (time.time() - t0) / 60.0,
        "results": results,
    }
    print(f"  --> SEED {seed} {'GO' if seed_go else 'NO'}  "
          f"(grounded {full['n_recalled']}/{full['n_facts']} | "
          f"a={a_noreplay} b={b_nmda} c={c_hippo} noconfab={noconfab_ok}) "
          f"[{verdict['elapsed_min']:.1f} min]", flush=True)
    del bridge
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--train-events", type=int, default=200)
    ap.add_argument("--replay-cycles", type=int, default=40)
    ap.add_argument("--nmda-self-weight", dest="nmda_self_weight", type=float, default=12.0)
    ap.add_argument("--nmda-self-density", dest="nmda_self_density", type=float, default=0.15)
    ap.add_argument("--nmda-recurrent-ratio", dest="nmda_recurrent_ratio", type=float, default=1.0)
    ap.add_argument("--ca1-concept-weight", dest="ca1_concept_weight", type=float, default=0.0)
    ap.add_argument("--ca1-concept-density", dest="ca1_concept_density", type=float, default=0.25)
    ap.add_argument("--cross-pool-density", dest="cross_pool_density", type=float, default=0.10)
    ap.add_argument("--stdp-w-max", dest="stdp_w_max", type=float, default=8.0)
    ap.add_argument("--floor", type=float, default=0.02)
    ap.add_argument("--min-recall", type=int, default=2, help="grounded facts recalled for GO (of 3)")
    ap.add_argument("--antichance", type=int, default=1, help="max recalled allowed in an anti-cheat control")
    ap.add_argument("--enable-global-nmda", dest="enable_global_nmda", action="store_true")
    ap.add_argument("--enable-hebbian", dest="enable_hebbian", action="store_true")
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--out", type=str, default="research/findings/raw/nmda_compositional_consolidation.json")
    args = ap.parse_args()

    print("=== A1: NMDA-supported compositional consolidation ===")
    print(f"seeds={args.seeds} train_events={args.train_events} replay_cycles={args.replay_cycles}")
    print(f"nmda_self_weight={args.nmda_self_weight} density={args.nmda_self_density} "
          f"recurrent_ratio={args.nmda_recurrent_ratio} | ca1_concept d={args.ca1_concept_density} "
          f"| floor={args.floor} hebbian={args.enable_hebbian} global_nmda={args.enable_global_nmda}")

    verdicts = []
    for s in args.seeds:
        verdicts.append(run_seed(s, args))

    n_go = sum(v["seed_go"] for v in verdicts)
    print(f"\n===== AGGREGATE: {n_go}/{len(verdicts)} seeds GO =====")
    for v in verdicts:
        print(f"  seed {v['seed']}: {'GO' if v['seed_go'] else 'NO'} "
              f"grounded={v['full_recalled']}/{v['full_n']} "
              f"a={v['a_noreplay_ok']} b={v['b_nmda_lesion_ok']} c={v['c_hippo_before_ok']} "
              f"noconfab={v['noconfab_ok']}")
    out = {
        "args": vars(args), "n_go": n_go, "n_seeds": len(verdicts),
        "verdicts": verdicts,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
