"""
Minimal language->motor isolation experiment.

Tests the decisive question: can the architecture learn word-action
mapping AT ALL when stripped of cascade interference?

Prior data (2026-05-03 autonomous overnight) shows 0/39 aligned across
all v2-architecture conditions. Pattern analysis shows misalignment is
seed-dependent (each random init creates its own private misalignment),
with a mild cascade-driven motor_E bias of ~3pp.

If THIS minimal architecture (NO cascade, NO PFC, NO retina, NO
visuomotor — just language_input -> motor_X with paired-stim training)
achieves aligned >= 4/6, the cascade IS the dominant interference.

If THIS also fails, the fundamental issue is deeper (plasticity dose,
soft-bound STDP, sparse-code overlap, or eval methodology).

Architecture:
  - language_input: 256 neurons (same as v2 baseline for fair compare)
  - motor_N, motor_E, motor_S, motor_W: 25 each (slightly larger than
    v2's 10 to reduce SNR noise; doesn't affect alignment if test
    works)
  - language_input -> motor_X pathways (4 plastic, all4 actions)
  - NO cluster_a, NO cluster_e, NO cortex_X cascade
  - NO retina, NO visual cortex, NO PFC
  - NO visuomotor pathways

Training:
  - paired-stim only (same _run_swr_replay_phase mechanism as H4)
  - synthetic balanced buffer: N events per direction, +1 reward
  - directly tests STDP's ability to differentiate words on a clean
    pathway

Eval:
  - same evaluate_word_to_action that everything else uses
  - 25 trials per word, interleaved
  - aligned ratio is the headline metric

Usage:
    python -m research.runners.text_minimal_isolation \\
        --seed 42 --n-events-per-direction 1000 \\
        --out-stats research/findings/raw/g11_bg/text_eval_minimal_iso_seed42.json
"""

import argparse
import json
import time
import numpy as np

# Backend-aware D->H transfer helper. 2026-05-11: enables this runner
# under SIM_BACKEND=numpy.
try:
    from sim.backend import to_host as _to_host
except ImportError:
    _to_host = lambda arr: arr.get() if hasattr(arr, "get") else arr


def build_minimal_brain_regions(
    n_lang_input: int = 256,
    n_motor_per_action: int = 25,
    text_input_to_motor_density: float = 0.30,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    enable_motor_fs: bool = False,
    n_motor_fs_per_action: int = 3,
    motor_to_fs_weight: float = 2.0,
    fs_to_motor_weight: float = 2.0,
):
    """Build a minimal language->motor architecture for isolation tests.

    Returns (regions, pathways) tuple compatible with the brain-region
    framework.

    Args:
        enable_motor_fs: add motor_FS_X interneuron pools providing
            cross-pool lateral inhibition (PV-FS in real motor cortex).
            Each motor_X drives its own motor_FS_X (excitatory), which
            inhibits the OTHER 3 motor pools (no self-inhibition). This
            is biology-grounded: real PV-FS interneurons provide
            ~10-15% of cortical population, mediating winner-takes-most
            competition without absolute veto. See Vogels et al 2011,
            Hofer et al 2011.
        n_motor_fs_per_action: FS interneurons per pool (default 3 ~12%
            of 25-neuron motor pool; biology range 10-15%).
        motor_to_fs_weight: excitatory drive from motor pyramidal to FS.
        fs_to_motor_weight: inhibitory weight from FS to other motor
            pools. Equal to motor_to_fs by default (graded competition,
            not absolute WTA).
    """
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    ACTION_NAMES = ["N", "E", "S", "W"]

    regions = []
    pathways = []

    # Language input region (sparse code substrate)
    regions.append(BrainRegion(
        name="language_input",
        n_neurons=n_lang_input,
        exc_fraction=0.8,
        internal_density=0.05,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2, plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))

    # Motor pools — separate region per action
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"motor_{action}",
            n_neurons=n_motor_per_action,
            exc_fraction=1.0,  # purely excitatory motor pool
            internal_density=0.0,  # no internal recurrence
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # language_input -> motor_X pathways (the ONE pathway being tested)
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="language_input", to_region=f"motor_{action}",
            density=text_input_to_motor_density,
            weight_mean=text_input_to_motor_weight,
            weight_jitter=text_input_to_motor_jitter,
            plastic=True,
            plasticity_gate="language_input_to_motor",
        ))

    # Motor lateral inhibition via PV-FS interneurons (biology-grounded)
    if enable_motor_fs:
        for action in ACTION_NAMES:
            regions.append(BrainRegion(
                name=f"motor_FS_{action}",
                n_neurons=n_motor_fs_per_action,
                exc_fraction=0.0,  # purely inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
            ))
        for action in ACTION_NAMES:
            # motor_X excites its own FS pool (not language_input -> FS:
            # FS recruitment must come from motor activity itself, not
            # language input directly, to prevent "language drive
            # directly suppresses wrong motor pools" shortcut)
            pathways.append(RegionPathway(
                from_region=f"motor_{action}", to_region=f"motor_FS_{action}",
                density=0.5,
                weight_mean=motor_to_fs_weight,
                weight_jitter=0.3,
                plastic=False,  # static recruitment (genetic-spec, not learned)
            ))
            # motor_FS_X inhibits the OTHER motor pools (no self-inhibition)
            for target_action in ACTION_NAMES:
                if target_action == action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"motor_FS_{action}",
                    to_region=f"motor_{target_action}",
                    density=0.5,
                    weight_mean=fs_to_motor_weight,
                    weight_jitter=0.3,
                    plastic=False,  # static inhibitory specification
                ))

    return regions, pathways


def build_biological_brain_regions(
    n_lang_input: int = 2048,
    n_motor_per_action: int = 500,
    text_input_to_motor_density: float = 0.30,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    enable_motor_fs: bool = False,
    n_motor_fs_per_action: int = 60,
    motor_to_fs_weight: float = 2.0,
    fs_to_motor_weight: float = 2.0,
    motor_internal_density: float = 0.10,
    motor_exc_fraction: float = 0.8,
    motor_exc_weight_mean: float = 2.0,
    motor_inh_weight_mean: float = 4.0,
    lang_internal_density: float = 0.05,
    lang_exc_fraction: float = 0.8,
    enable_language_output: bool = False,
    n_lang_output: int = 2048,
    motor_to_language_output_density: float = 0.30,
    motor_to_language_output_weight: float = 0.5,
    motor_to_language_output_jitter: float = 0.3,
    # Tier 2.3 (opt-in, design at
    # docs/plans/2026-05-06-Tier2.3-two-word-phrases-design.md):
    # add a PFC verb pool that holds verb context (~500ms NMDA
    # bistability) for compositional 2-word phrases like "go north".
    # Default OFF for full backward compatibility.
    enable_dlpfc_verb: bool = False,
    n_dlpfc_verb: int = 200,
    dlpfc_verb_internal_density: float = 0.15,
    dlpfc_verb_exc_weight_mean: float = 3.0,
    dlpfc_verb_inh_weight_mean: float = 4.0,
    lang_to_dlpfc_verb_density: float = 0.30,
    lang_to_dlpfc_verb_weight: float = 2.0,
    # Phase 1.3 (opt-in, design at
    # docs/plans/2026-05-06-Phase-1.3-consolidation-design.md):
    # add hippocampus regions + ca1 -> motor / language_output
    # consolidation pathways. Enables hippocampus -> cortex
    # transfer during sleep replay. Default OFF.
    enable_hippocampus_consolidation: bool = False,
    n_ec: int = 80,
    n_dg: int = 200,
    n_dg_pv_basket: int = 60,
    n_ca3: int = 100,
    n_ca1: int = 120,
    ca1_to_motor_density: float = 0.20,
    ca1_to_motor_weight: float = 2.0,
    ca1_to_lang_out_density: float = 0.20,
    ca1_to_lang_out_weight: float = 2.0,
    # CA3 autoassociator strength (Marr 1971; catalog D.05/D.13).
    # Default tuned for consolidation use case (Phase 1.3); pattern-
    # completion validation (P1 of realigned plan) may need stronger
    # recurrent connectivity.
    ca3_recurrent_density: float = 0.30,
    ca3_recurrent_weight: float = 1.5,
    # P4.1 episodic encoder: ec_context region for positional binding
    # (catalog D.01/D.02 + D.11 time cells). When enabled, adds a
    # separate region driven by positional embeddings; the pathway
    # ec_context -> dg gives DG (alongside ec -> dg) a combined
    # (word, position) input → distinct CA3 ensembles per
    # (word, position) tuple. Default OFF for backward compat with
    # Phase 1.3 retention tests.
    enable_episodic_context: bool = False,
    n_ec_context: int = 200,
    ec_context_to_dg_density: float = 0.40,
    ec_context_to_dg_weight: float = 4.0,
    # P5 ventral semantic stream (catalog G.11 + G.13): adds
    # semantic_cortex (~1000 neurons, sparse distributed concept
    # representations) + wernicke (~200 neurons, lang↔semantic bridge).
    # Pathways: lang_input → wernicke → semantic_cortex (comprehension);
    # semantic_cortex → wernicke → language_output (naming/production);
    # ca1 → semantic_cortex (consolidation via SWR replay).
    # Default OFF for backward compat.
    enable_ventral_semantic: bool = False,
    n_semantic_cortex: int = 1000,
    n_wernicke: int = 200,
    semantic_cortex_recurrent_density: float = 0.10,
    semantic_cortex_recurrent_weight: float = 1.0,
    lang_to_wernicke_density: float = 0.30,
    lang_to_wernicke_weight: float = 3.0,
    wernicke_to_semantic_density: float = 0.30,
    wernicke_to_semantic_weight: float = 4.0,
    semantic_to_wernicke_density: float = 0.20,  # weaker (production)
    semantic_to_wernicke_weight: float = 2.0,
    ca1_to_semantic_density: float = 0.20,
    ca1_to_semantic_weight: float = 3.0,
    # Path B+ (P5 architecture rework): semantic_FS lateral
    # inhibition. Real cortex has PV-FS interneurons that enforce
    # winner-take-most selection among co-active sub-populations
    # (Vogels 2011, Hofer 2011). Without this, attractor dynamics
    # are monolithic — every input drives the same big basin.
    # See P5 iter D FAIL (2026-05-11).
    enable_semantic_fs: bool = False,
    n_semantic_fs: int = 100,
    semantic_to_fs_density: float = 0.30,
    semantic_to_fs_weight: float = 3.0,
    fs_to_semantic_density: float = 0.50,
    fs_to_semantic_weight: float = 4.0,
    # Path G (iter G): wernicke_FS lateral inhibition for sparse
    # concept ensemble encoding. Per P5 iter E weight inspection
    # (selectivity=0.004), wernicke fires ALL neurons regardless
    # of concept — there's no selective ensemble. FS inhibition
    # forces sparse firing patterns that differ per input.
    enable_wernicke_fs: bool = False,
    n_wernicke_fs: int = 60,
    wernicke_to_fs_density: float = 0.30,
    wernicke_to_fs_weight: float = 3.0,
    wernicke_fs_to_wernicke_density: float = 0.50,
    wernicke_fs_to_wernicke_weight: float = 4.0,
    # Path A / Path G+ FULL (iter T): multi-pool wernicke with
    # per-concept FS cross-inhibition. Mirror of Tier 1 motor pool
    # architecture (which produced 6/6 multi-seed PASS). Each
    # concept gets dedicated wernicke_pool_<i> region (100 neurons)
    # + wernicke_fs_pool_<i> (12 PV-FS). Cross-pool inhibition:
    # each pool's FS inhibits OTHER pools (winner-take-most).
    # The single "wernicke" region is replaced by N pools.
    enable_multi_pool_wernicke: bool = False,
    n_wernicke_pools: int = 2,
    n_per_wernicke_pool: int = 100,
    n_per_wernicke_pool_fs: int = 12,
    wernicke_pool_to_fs_weight: float = 3.0,
    # Iter KK->LL: per-pool dynamics parameterized to test canon vs
    # weak iter-AA defaults. Iter KK seed 42 showed canon (0.10/2.0/4.0)
    # makes pools self-sustain and amplifies structural bias. Default
    # back to iter AA weak; add flags to override.
    wernicke_pool_internal_density: float = 0.05,
    wernicke_pool_exc_weight_mean: float = 0.3,
    wernicke_pool_inh_weight_mean: float = 0.8,
    lang_output_pool_internal_density: float = 0.05,
    lang_output_pool_exc_weight_mean: float = 0.3,
    lang_output_pool_inh_weight_mean: float = 0.8,
    wernicke_fs_cross_weight: float = 4.0,
    # Per-concept lang_output pools (mirror Tier 1 motor pool
    # at output side; addresses iter Z finding that shared
    # lang_output prevents bidirectional discrimination at toy
    # scale). When enabled AND multi_pool_wernicke is on:
    # - lang_output_pool_<i> regions created
    # - wernicke_pool_<i> -> lang_output_pool_<i> (dedicated)
    # - ca1 -> each lang_output_pool_<i>
    enable_per_concept_lang_out_pools: bool = False,
    n_per_lang_out_pool: int = 200,
    pool_to_lang_out_pool_weight: float = 3.0,
    ca1_to_lang_out_pool_weight: float = 2.0,
    # Iter CC: lang_output_FS pools (cross-inhibition at output
    # layer). Completes the full Tier 1 motor pool mirror —
    # each lang_output_pool_i has dedicated FS that inhibits
    # OTHER lang_output_pools (winner-take-most at output).
    # Fixes seed-101-style structural pool bias where one pool
    # is "always more active" regardless of input.
    enable_lang_out_fs_pools: bool = False,
    n_per_lang_out_fs_pool: int = 24,  # 12% of 200 = 24 PV-FS
    lang_out_to_fs_weight: float = 3.0,
    lang_out_fs_cross_weight: float = 4.0,
    # P6 Broca's area + compositional syntax (catalog G.12, Kandel
    # 6e Ch 55 pp 1382-1384). Adds broca region (~500 neurons,
    # recurrent for sentence working memory) + motor_speech region
    # (4 slots; will scale up later). Pathways:
    #   wernicke → broca (semantic content feeds syntax)
    #   broca → broca (sentence-level working memory)
    #   semantic_cortex → broca (meaning constraints)
    #   broca → motor_speech (articulation drive)
    #   broca → ec_context (Broca's drives positional context
    #                        during composition; only when
    #                        enable_episodic_context=True)
    # Default OFF for backward compat.
    enable_broca: bool = False,
    n_broca: int = 500,
    n_motor_speech: int = 64,  # 4 slots × 16 neurons each
    broca_recurrent_density: float = 0.15,
    broca_recurrent_weight: float = 2.0,
    wernicke_to_broca_density: float = 0.30,
    wernicke_to_broca_weight: float = 3.0,
    semantic_to_broca_density: float = 0.20,
    semantic_to_broca_weight: float = 2.0,
    broca_to_motor_speech_density: float = 0.40,
    broca_to_motor_speech_weight: float = 4.0,
    broca_to_ec_context_density: float = 0.20,
    broca_to_ec_context_weight: float = 2.0,
    # ─────────── Sensory grounding via Cluster K v2 (2026-05-12) ──────
    # P5 architectural pivot per iter KK/LL/MM/NN findings: at
    # biological scale, per-concept pool architecture has per-seed
    # structural pool bias that no parameter tuning fixes. Solution:
    # add a SECOND strong training signal (visual stream) independent
    # of random connectivity. Mirrors Tier 1's 6/6 success — motor
    # teacher current overrides random structure for direction words;
    # visual teacher does the same for abstract concepts.
    #
    # Catalog G.11 (Hickok & Poeppel ventral) + K.01 (V1/V2/IT
    # visual ventral) + Pulvermüller embodied semantics.
    # Design: docs/plans/2026-05-12-P5-sensory-grounding-design.md
    enable_visual_cortex: bool = False,
    visual_n_orientations: int = 8,
    visual_n_frequencies: int = 2,
    visual_n_positions_per_dim: int = 8,
    visual_image_size: int = 32,  # retina = 2 × 32² = 2048 ON/OFF
    visual_n_v2: int = 256,
    visual_n_it: int = 64,
    # Multimodal hub: where wernicke (auditory) + IT (visual) converge
    # to form the embodied semantic representation per concept. Real
    # biology: anterior temporal lobe (ATL) hub-and-spoke model
    # (Lambon Ralph 2017). Plastic recurrent + bidirectional pathways.
    enable_multimodal_hub: bool = False,
    n_multimodal_hub: int = 500,
    multimodal_hub_internal_density: float = 0.05,
    multimodal_hub_exc_weight_mean: float = 0.3,
    multimodal_hub_inh_weight_mean: float = 0.8,
    it_to_hub_density: float = 0.30,
    it_to_hub_weight: float = 2.0,
    wernicke_pool_to_hub_density: float = 0.30,
    wernicke_pool_to_hub_weight: float = 2.0,
    hub_to_lang_output_pool_density: float = 0.30,
    hub_to_lang_output_pool_weight: float = 2.0,
):
    """Biological-scale architecture with cortical canon ENABLED.

    vs build_minimal_brain_regions: motor pools have recurrent
    excitation + E/I balance + larger N. Specifically:
      - n_motor_per_action: 500 (vs 25). Schieber 2001 / Rathelot 2009
        estimate motor cortex sub-pools at 100-500 neurons per action.
      - motor exc_fraction: 0.8 (vs 1.0 pure-exc). Real cortex is 80E/20I.
      - motor internal_density: 0.10 (vs 0.0). Lefort 2009 estimates
        cortical recurrent connectivity at 10-20%.
      - motor exc_weight: 2.0, inh_weight: 4.0 (vs 0.0). Recurrent E
        amplifies signal; local I prevents runaway.
      - n_lang_input: 2048 (vs 256). Wernicke-area scale.

    Combined with cfg.enable_nmda=True (Wang 2002 NMDA bistability),
    these motor pools should produce attractor dynamics — transient
    differential drive locks into sustained differential firing.

    Memory budget at default sizes:
      - 2048 + 4*500 = 4048 neurons
      - lang->motor synapses: 2048 * 500 * 0.30 * 4 = 1.23M
      - motor recurrence: 500 * 500 * 0.10 * 4 = 100K (E/E + E/I + I/E)
      - lang internal recurrence: 2048 * 2048 * 0.05 = 209K
      - Total: ~1.55M synapses. Estimated ~1-2 GB GPU at peak.
        Single-process fit comfortably in RTX 3090 24 GB.

    Returns (regions, pathways) tuple.
    """
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    ACTION_NAMES = ["N", "E", "S", "W"]
    # Prereq checks (top-level so they fire regardless of which
    # downstream sub-flags are set).
    if enable_broca and not enable_ventral_semantic:
        raise ValueError(
            "enable_broca=True requires enable_ventral_semantic=True "
            "(Broca's reads from wernicke + semantic_cortex; "
            "catalog G.12 prereq)"
        )
    if enable_ventral_semantic and not enable_hippocampus_consolidation:
        raise ValueError(
            "enable_ventral_semantic=True requires "
            "enable_hippocampus_consolidation=True "
            "(ca1->semantic_cortex consolidation pathway needs ca1)"
        )
    regions = []
    pathways = []

    # Language input region — biological-scale Wernicke-like
    regions.append(BrainRegion(
        name="language_input",
        n_neurons=n_lang_input,
        exc_fraction=lang_exc_fraction,
        internal_density=lang_internal_density,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2, plastic_internal=False,  # frozen for clean test
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))

    # Motor pools — cortical canon: recurrent excitation + E/I balance
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"motor_{action}",
            n_neurons=n_motor_per_action,
            exc_fraction=motor_exc_fraction,
            internal_density=motor_internal_density,
            exc_weight_mean=motor_exc_weight_mean,
            inh_weight_mean=motor_inh_weight_mean,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # language_input -> motor_X (the pathway being tested)
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="language_input", to_region=f"motor_{action}",
            density=text_input_to_motor_density,
            weight_mean=text_input_to_motor_weight,
            weight_jitter=text_input_to_motor_jitter,
            plastic=True,
            plasticity_gate="language_input_to_motor",
        ))

    # Optional language_output region + reciprocal motor → language_output
    # pathway. Enables Tier 1 embodied Hebbian binding: when motor pool fires,
    # language_output develops the corresponding word pattern via STDP.
    # Biological basis: Felleman & Van Essen 1991 reciprocal cortical
    # connectivity; Broca's area is interleaved with premotor cortex
    # (Pulvermüller 2003). When motor pattern executes, premotor activity
    # propagates to linguistic representation.
    if enable_language_output:
        regions.append(BrainRegion(
            name="language_output",
            n_neurons=n_lang_output,
            exc_fraction=lang_exc_fraction,
            internal_density=lang_internal_density,
            exc_weight_mean=2.0, inh_weight_mean=4.0,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"motor_{action}", to_region="language_output",
                density=motor_to_language_output_density,
                weight_mean=motor_to_language_output_weight,
                weight_jitter=motor_to_language_output_jitter,
                plastic=True,
                plasticity_gate="motor_to_language_output",
            ))

    # Tier 2.3 PFC verb pool. Sub-region of dlPFC for verb-context
    # representation. NMDA-bistable for ~500ms persistent activity
    # (Wang 2002; Goldman-Rakic 1995 working memory). Holds verb
    # ("go") while next word ("north") arrives, allowing
    # compositional binding via co-firing.
    #
    # Architecture per design (docs/plans/2026-05-06-Tier2.3-two-
    # word-phrases-design.md): 200 neurons, exc_fraction 0.8,
    # internal_density 0.15 (high recurrence for working memory),
    # plastic_internal=False (frozen recurrence; only the input
    # pathway is plastic).
    if enable_dlpfc_verb:
        regions.append(BrainRegion(
            name="dlpfc_verb",
            n_neurons=n_dlpfc_verb,
            exc_fraction=0.8,
            internal_density=dlpfc_verb_internal_density,
            exc_weight_mean=dlpfc_verb_exc_weight_mean,
            inh_weight_mean=dlpfc_verb_inh_weight_mean,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        # language_input -> dlpfc_verb (plastic, gated). Verb words
        # in language_input drive dlpfc_verb; STDP at this pathway
        # binds specific verb codes to PFC pool activation.
        pathways.append(RegionPathway(
            from_region="language_input", to_region="dlpfc_verb",
            density=lang_to_dlpfc_verb_density,
            weight_mean=lang_to_dlpfc_verb_weight,
            weight_jitter=0.5,
            plastic=True,
            plasticity_gate="language_input_to_dlpfc_verb",
        ))

    # Phase 1.3 hippocampus consolidation regions + pathways.
    # Implements the trisynaptic loop EC -> DG -> CA3 -> CA1 plus
    # ec -> ca1 direct bypass (per Cluster D v1 architecture, see
    # research/runners/g11_bg_runner.py:741+ for the validated
    # specs). The KEY ADDITION over Cluster D is ca1 -> motor and
    # ca1 -> language_output consolidation pathways with their own
    # plasticity_gate, so sleep replay can drive cortex updates
    # while awake training uses the direct lang -> motor route.
    #
    # Biology source: Buzsaki & Moser 2013 (CA1 readout via Schaffer
    # collaterals); McClelland et al 1995 (complementary learning
    # systems theory: hippocampus stores fast, cortex consolidates
    # slow, sleep replay drives transfer).
    if enable_hippocampus_consolidation:
        regions.append(BrainRegion(
            name="ec",
            n_neurons=n_ec, exc_fraction=0.8,
            internal_density=0.05,
            exc_weight_mean=0.3, inh_weight_mean=0.8,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="dg",
            n_neurons=n_dg, exc_fraction=0.95,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="dg_pv_basket",
            n_neurons=n_dg_pv_basket, exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
        ))
        # CA3 with explicit recurrent pathway (SWR-gated). The
        # recurrent self-loop is added as a tagged pathway below
        # rather than internal_density, so plasticity can be gated
        # ON during ripple bursts and OFF otherwise (Cluster D v2
        # SWR-gated CA3 plasticity, Buzsaki 2015 ~150Hz ripples).
        regions.append(BrainRegion(
            name="ca3",
            n_neurons=n_ca3, exc_fraction=0.85,
            internal_density=0.0,  # rewired as explicit pathway
            exc_weight_mean=1.5, inh_weight_mean=2.0,
            weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="ca1",
            n_neurons=n_ca1, exc_fraction=0.85,
            internal_density=0.05,
            exc_weight_mean=0.3, inh_weight_mean=0.8,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ))
        # Trisynaptic loop pathways
        # language_input -> ec (cortex -> hippo)
        pathways.append(RegionPathway(
            from_region="language_input", to_region="ec",
            density=0.30, weight_mean=4.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="lang_to_ec",
        ))
        # ec -> dg (perforant path)
        pathways.append(RegionPathway(
            from_region="ec", to_region="dg",
            density=0.40, weight_mean=6.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ec_to_dg",
        ))
        # P4.1 episodic context: ec_context -> dg parallel pathway.
        # Catalog D.01/D.02 + D.11. When enable_episodic_context=True,
        # add ec_context region + this pathway so DG receives a
        # combined (word, position) drive → distinct CA3 ensembles
        # per (word, position) tuple.
        if enable_episodic_context:
            regions.append(BrainRegion(
                name="ec_context",
                n_neurons=n_ec_context, exc_fraction=1.0,
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
            pathways.append(RegionPathway(
                from_region="ec_context", to_region="dg",
                density=ec_context_to_dg_density,
                weight_mean=ec_context_to_dg_weight,
                weight_jitter=0.2,
                plastic=True, plasticity_gate="ec_context_to_dg",
            ))
        # P5 ventral semantic stream (catalog G.11 + G.13).
        # semantic_cortex: sparse distributed concept store
        #   (anterior-temporal-lobe analog).
        # wernicke: bidirectional bridge between phonological
        #   (lang_input/output) and semantic (semantic_cortex).
        if enable_ventral_semantic:
            # Path A / Path G+ FULL: multi-pool wernicke or single
            # wernicke region. If enable_multi_pool_wernicke=True,
            # create wernicke_pool_0 ... wernicke_pool_{N-1} with
            # cross-pool FS inhibition. Otherwise create single
            # "wernicke" region as before.
            if enable_multi_pool_wernicke:
                wernicke_names = [f"wernicke_pool_{i}"
                                   for i in range(n_wernicke_pools)]
                wernicke_fs_names = [f"wernicke_fs_pool_{i}"
                                       for i in range(n_wernicke_pools)]
                for name in wernicke_names:
                    regions.append(BrainRegion(
                        name=name,
                        n_neurons=n_per_wernicke_pool, exc_fraction=0.8,
                        # Iter KK->LL: parameterized. Default = iter AA
                        # weak (0.05/0.3/0.8). Override via CLI for
                        # canon (0.10/2.0/4.0) — but iter KK seed 42
                        # showed canon amplifies structural bias.
                        internal_density=wernicke_pool_internal_density,
                        exc_weight_mean=wernicke_pool_exc_weight_mean,
                        inh_weight_mean=wernicke_pool_inh_weight_mean,
                        weight_jitter=0.2, plastic_internal=False,
                        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                    ))
                for name in wernicke_fs_names:
                    regions.append(BrainRegion(
                        name=name,
                        n_neurons=n_per_wernicke_pool_fs,
                        exc_fraction=0.0,
                        internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0,
                        weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type=(
                            NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name
                        ),
                    ))
                # Per-concept lang_output pools (iter AA): mirror
                # Tier 1 motor pool architecture at the language
                # output. Each wernicke_pool routes to its dedicated
                # lang_output_pool, preventing the shared lang_output
                # bottleneck that limited iter Z bidirectional
                # discrimination.
                if enable_per_concept_lang_out_pools:
                    for i in range(n_wernicke_pools):
                        regions.append(BrainRegion(
                            name=f"lang_output_pool_{i}",
                            n_neurons=n_per_lang_out_pool,
                            exc_fraction=0.8,
                            # Iter KK->LL: parameterized. Default =
                            # iter AA weak (0.05/0.3/0.8).
                            internal_density=lang_output_pool_internal_density,
                            exc_weight_mean=lang_output_pool_exc_weight_mean,
                            inh_weight_mean=lang_output_pool_inh_weight_mean,
                            weight_jitter=0.2,
                            plastic_internal=False,
                            izh_neuron_type=(
                                NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
                            ),
                        ))
                    if enable_lang_out_fs_pools:
                        # Per-pool FS at lang_output (iter CC)
                        for i in range(n_wernicke_pools):
                            regions.append(BrainRegion(
                                name=f"lang_output_fs_pool_{i}",
                                n_neurons=n_per_lang_out_fs_pool,
                                exc_fraction=0.0,
                                internal_density=0.0,
                                exc_weight_mean=0.0,
                                inh_weight_mean=0.0,
                                weight_jitter=0.0,
                                plastic_internal=False,
                                izh_neuron_type=(
                                    NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name
                                ),
                            ))
            else:
                regions.append(BrainRegion(
                    name="wernicke",
                    n_neurons=n_wernicke, exc_fraction=0.8,
                    internal_density=0.05,
                    exc_weight_mean=0.3, inh_weight_mean=0.8,
                    weight_jitter=0.2, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                ))
            regions.append(BrainRegion(
                name="semantic_cortex",
                n_neurons=n_semantic_cortex, exc_fraction=0.85,
                internal_density=semantic_cortex_recurrent_density,
                exc_weight_mean=semantic_cortex_recurrent_weight,
                inh_weight_mean=1.5,
                weight_jitter=0.2, plastic_internal=True,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
            # Comprehension path: lang_input -> wernicke[_pool_i] -> semantic_cortex
            if enable_multi_pool_wernicke:
                # Each lang_input -> wernicke_pool_i with topographic
                # bias applied separately via apply_wernicke_pools_topographic_bias.
                # Cross-pool FS: each pool drives its own FS;
                # each FS inhibits OTHER pools (winner-take-most).
                for i, pool_name in enumerate(wernicke_names):
                    pathways.append(RegionPathway(
                        from_region="language_input", to_region=pool_name,
                        density=lang_to_wernicke_density,
                        weight_mean=lang_to_wernicke_weight,
                        weight_jitter=0.2,
                        plastic=True, plasticity_gate=f"lang_to_{pool_name}",
                    ))
                    pathways.append(RegionPathway(
                        from_region=pool_name, to_region="semantic_cortex",
                        density=wernicke_to_semantic_density,
                        weight_mean=wernicke_to_semantic_weight,
                        weight_jitter=0.2,
                        plastic=True, plasticity_gate=f"{pool_name}_to_semantic",
                    ))
                    pathways.append(RegionPathway(
                        from_region="semantic_cortex", to_region=pool_name,
                        density=semantic_to_wernicke_density,
                        weight_mean=semantic_to_wernicke_weight,
                        weight_jitter=0.2,
                        plastic=True, plasticity_gate=f"semantic_to_{pool_name}",
                    ))
                    fs_name = wernicke_fs_names[i]
                    pathways.append(RegionPathway(
                        from_region=pool_name, to_region=fs_name,
                        density=0.30,
                        weight_mean=wernicke_pool_to_fs_weight,
                        weight_jitter=0.2,
                        plastic=True, plasticity_gate=f"{pool_name}_to_fs",
                    ))
                    # FS_i inhibits all OTHER pools (cross-inhibition)
                    for j, other_pool in enumerate(wernicke_names):
                        if j == i:
                            continue
                        pathways.append(RegionPathway(
                            from_region=fs_name, to_region=other_pool,
                            density=0.50,
                            weight_mean=wernicke_fs_cross_weight,
                            weight_jitter=0.2,
                            plastic=False,
                            plasticity_gate=f"{fs_name}_to_{other_pool}",
                        ))
                    if enable_language_output:
                        if enable_per_concept_lang_out_pools:
                            # Path A++ (iter AA): dedicated per-
                            # concept lang_output_pool. Mirror of
                            # Tier 1 motor pool at output.
                            pool_lang_name = f"lang_output_pool_{i}"
                            pathways.append(RegionPathway(
                                from_region=pool_name,
                                to_region=pool_lang_name,
                                density=0.30,
                                weight_mean=pool_to_lang_out_pool_weight,
                                weight_jitter=0.2,
                                plastic=True,
                                plasticity_gate=(
                                    f"{pool_name}_to_lang_pool_{i}"),
                            ))
                        else:
                            pathways.append(RegionPathway(
                                from_region=pool_name,
                                to_region="language_output",
                                density=0.30, weight_mean=3.0,
                                weight_jitter=0.2,
                                plastic=True,
                                plasticity_gate=(
                                    f"{pool_name}_to_lang_out"),
                            ))
            else:
                pathways.append(RegionPathway(
                    from_region="language_input", to_region="wernicke",
                    density=lang_to_wernicke_density,
                    weight_mean=lang_to_wernicke_weight,
                    weight_jitter=0.2,
                    plastic=True, plasticity_gate="lang_to_wernicke",
                ))
                pathways.append(RegionPathway(
                    from_region="wernicke", to_region="semantic_cortex",
                    density=wernicke_to_semantic_density,
                    weight_mean=wernicke_to_semantic_weight,
                    weight_jitter=0.2,
                    plastic=True, plasticity_gate="wernicke_to_semantic",
                ))
                pathways.append(RegionPathway(
                    from_region="semantic_cortex", to_region="wernicke",
                    density=semantic_to_wernicke_density,
                    weight_mean=semantic_to_wernicke_weight,
                    weight_jitter=0.2,
                    plastic=True, plasticity_gate="semantic_to_wernicke",
                ))
                if enable_language_output:
                    pathways.append(RegionPathway(
                        from_region="wernicke", to_region="language_output",
                        density=0.30, weight_mean=3.0,
                        weight_jitter=0.2,
                        plastic=True, plasticity_gate="wernicke_to_lang_out",
                    ))
            # Hippo -> semantic_cortex consolidation pathway
            # (THE KEY BRIDGE — catalog D.01 consolidation; engrams
            # become durable cortical meanings via SWR replay).
            pathways.append(RegionPathway(
                from_region="ca1", to_region="semantic_cortex",
                density=ca1_to_semantic_density,
                weight_mean=ca1_to_semantic_weight,
                weight_jitter=0.3,
                plastic=True, plasticity_gate="ca1_to_semantic",
            ))
            # Path G: wernicke_FS lateral inhibition for sparse
            # concept ensemble encoding. Per P5 iter E weight
            # inspection (selectivity=0.004), wernicke fires ALL
            # neurons for both apple AND river — there's no
            # selective ensemble encoding. With FS inhibition,
            # only the top ~5-10% of wernicke neurons sustain
            # firing per input, producing distinct sparse codes.
            if enable_wernicke_fs:
                regions.append(BrainRegion(
                    name="wernicke_fs",
                    n_neurons=n_wernicke_fs, exc_fraction=0.0,
                    internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=(
                        NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name
                    ),
                ))
                pathways.append(RegionPathway(
                    from_region="wernicke",
                    to_region="wernicke_fs",
                    density=wernicke_to_fs_density,
                    weight_mean=wernicke_to_fs_weight,
                    weight_jitter=0.2,
                    plastic=True, plasticity_gate="wernicke_to_fs",
                ))
                pathways.append(RegionPathway(
                    from_region="wernicke_fs",
                    to_region="wernicke",
                    density=wernicke_fs_to_wernicke_density,
                    weight_mean=wernicke_fs_to_wernicke_weight,
                    weight_jitter=0.2,
                    plastic=False, plasticity_gate="wernicke_fs_to_wernicke",
                ))
            # Path B+: semantic_FS lateral inhibition for selective
            # attractor formation. Real cortex PV-FS interneurons
            # provide winner-take-most among co-active sub-populations.
            # Vogels 2011 / Hofer 2011: PV-FS ~12% of cortex; tonic
            # spiking; broad projections within column. Here:
            # semantic_cortex -> semantic_fs (excite all FS) and
            # semantic_fs -> semantic_cortex (inhibit broadly). The
            # winning sub-population is favored because it sustains
            # via recurrence; losing sub-populations get suppressed.
            if enable_semantic_fs:
                regions.append(BrainRegion(
                    name="semantic_fs",
                    n_neurons=n_semantic_fs, exc_fraction=0.0,
                    internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=(
                        NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name
                    ),
                ))
                pathways.append(RegionPathway(
                    from_region="semantic_cortex",
                    to_region="semantic_fs",
                    density=semantic_to_fs_density,
                    weight_mean=semantic_to_fs_weight,
                    weight_jitter=0.2,
                    plastic=True, plasticity_gate="semantic_to_fs",
                ))
                pathways.append(RegionPathway(
                    from_region="semantic_fs",
                    to_region="semantic_cortex",
                    density=fs_to_semantic_density,
                    weight_mean=fs_to_semantic_weight,
                    weight_jitter=0.2,
                    plastic=False, plasticity_gate="fs_to_semantic",
                ))
        # P6 Broca's area + motor_speech (catalog G.12). Adds
        # syntactic composition layer. Requires P5 ventral
        # semantic stream (broca reads from wernicke +
        # semantic_cortex).
        if enable_broca and enable_ventral_semantic:
            regions.append(BrainRegion(
                name="broca",
                n_neurons=n_broca, exc_fraction=0.8,
                internal_density=broca_recurrent_density,
                exc_weight_mean=broca_recurrent_weight,
                inh_weight_mean=1.5,
                weight_jitter=0.2, plastic_internal=True,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
            regions.append(BrainRegion(
                name="motor_speech",
                n_neurons=n_motor_speech, exc_fraction=0.85,
                internal_density=0.05,
                exc_weight_mean=0.3, inh_weight_mean=0.8,
                weight_jitter=0.2, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            ))
            # Pathways
            pathways.append(RegionPathway(
                from_region="wernicke", to_region="broca",
                density=wernicke_to_broca_density,
                weight_mean=wernicke_to_broca_weight,
                weight_jitter=0.2,
                plastic=True, plasticity_gate="wernicke_to_broca",
            ))
            pathways.append(RegionPathway(
                from_region="semantic_cortex", to_region="broca",
                density=semantic_to_broca_density,
                weight_mean=semantic_to_broca_weight,
                weight_jitter=0.2,
                plastic=True, plasticity_gate="semantic_to_broca",
            ))
            pathways.append(RegionPathway(
                from_region="broca", to_region="motor_speech",
                density=broca_to_motor_speech_density,
                weight_mean=broca_to_motor_speech_weight,
                weight_jitter=0.2,
                plastic=True, plasticity_gate="broca_to_motor_speech",
            ))
            # Optional: broca drives ec_context (Broca's generates
            # positional context during production). Only wired if
            # ec_context exists.
            if enable_episodic_context:
                pathways.append(RegionPathway(
                    from_region="broca", to_region="ec_context",
                    density=broca_to_ec_context_density,
                    weight_mean=broca_to_ec_context_weight,
                    weight_jitter=0.2,
                    plastic=True, plasticity_gate="broca_to_ec_context",
                ))
        # ec -> dg_pv_basket and dg_pv_basket -> dg (FFi for sparsity)
        pathways.append(RegionPathway(
            from_region="ec", to_region="dg_pv_basket",
            density=0.40, weight_mean=5.0, weight_jitter=0.2,
            plastic=False,
        ))
        pathways.append(RegionPathway(
            from_region="dg_pv_basket", to_region="dg",
            density=1.0, weight_mean=6.0, weight_jitter=0.2,
            plastic=False,
        ))
        # dg -> ca3 (mossy fibers)
        pathways.append(RegionPathway(
            from_region="dg", to_region="ca3",
            density=0.10, weight_mean=8.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="dg_to_ca3",
        ))
        # ec -> ca1 (direct cortical bypass)
        pathways.append(RegionPathway(
            from_region="ec", to_region="ca1",
            density=0.30, weight_mean=3.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ec_to_ca1",
        ))
        # ca3 -> ca3 recurrent (SWR-gated; awake = OFF, sleep = ON)
        # Parametrized 2026-05-11 for P1 pattern-completion validation
        # (catalog D.05/D.13; Marr 1971 autoassociator).
        pathways.append(RegionPathway(
            from_region="ca3", to_region="ca3",
            density=ca3_recurrent_density,
            weight_mean=ca3_recurrent_weight,
            weight_jitter=0.2,
            plastic=True, plasticity_gate="ca3_swr_burst",
        ))
        # ca3 -> ca1 (Schaffer collaterals)
        pathways.append(RegionPathway(
            from_region="ca3", to_region="ca1",
            density=0.30, weight_mean=4.0, weight_jitter=0.2,
            plastic=True, plasticity_gate="ca3_to_ca1",
        ))
        # *** Phase 1.3 KEY ADDITIONS: consolidation pathways ***
        # ca1 -> motor (per-action). Sleep replay drives motor
        # plasticity through this pathway, transferring hippo
        # patterns to cortex. Plasticity gated separately so
        # awake training keeps cortex pure (only direct lang ->
        # motor route is plastic awake; only ca1 -> motor is
        # plastic during sleep).
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region="ca1", to_region=f"motor_{action}",
                density=ca1_to_motor_density,
                weight_mean=ca1_to_motor_weight,
                weight_jitter=0.3, plastic=True,
                plasticity_gate="ca1_to_motor",
            ))
        # ca1 -> language_output: same logic for the bidirectional
        # binding pathway (motor -> lang_output in Tier 1; here
        # we add hippo -> lang_output for consolidation).
        if enable_language_output:
            pathways.append(RegionPathway(
                from_region="ca1", to_region="language_output",
                density=ca1_to_lang_out_density,
                weight_mean=ca1_to_lang_out_weight,
                weight_jitter=0.3, plastic=True,
                plasticity_gate="ca1_to_lang_out",
            ))
            # iter AA: also project ca1 to per-concept lang_output_pools
            # so the naming pathway (CA3 tag → CA1 → lang_pool) can
            # discriminate without going through shared lang_output.
            if (enable_multi_pool_wernicke
                    and enable_per_concept_lang_out_pools
                    and enable_ventral_semantic):
                for i in range(n_wernicke_pools):
                    pathways.append(RegionPathway(
                        from_region="ca1",
                        to_region=f"lang_output_pool_{i}",
                        density=ca1_to_lang_out_density,
                        weight_mean=ca1_to_lang_out_pool_weight,
                        weight_jitter=0.3, plastic=True,
                        plasticity_gate=f"ca1_to_lang_pool_{i}",
                    ))
                # Iter CC: lang_output_fs cross-inhibition
                if enable_lang_out_fs_pools:
                    for i in range(n_wernicke_pools):
                        # Each lang_pool drives its own FS
                        pathways.append(RegionPathway(
                            from_region=f"lang_output_pool_{i}",
                            to_region=f"lang_output_fs_pool_{i}",
                            density=0.30,
                            weight_mean=lang_out_to_fs_weight,
                            weight_jitter=0.2,
                            plastic=True,
                            plasticity_gate=(
                                f"lang_pool_{i}_to_fs"),
                        ))
                        # Cross-inhibition: FS_i inhibits OTHER pools
                        for j in range(n_wernicke_pools):
                            if j == i:
                                continue
                            pathways.append(RegionPathway(
                                from_region=f"lang_output_fs_pool_{i}",
                                to_region=f"lang_output_pool_{j}",
                                density=0.50,
                                weight_mean=lang_out_fs_cross_weight,
                                weight_jitter=0.2,
                                plastic=False,
                                plasticity_gate=(
                                    f"lang_fs_{i}_to_pool_{j}"),
                            ))

    # Motor lateral inhibition via PV-FSI (Vogels 2011 / Hofer 2011) -
    # biological 12% of motor pool size.
    if enable_motor_fs:
        for action in ACTION_NAMES:
            regions.append(BrainRegion(
                name=f"motor_FS_{action}",
                n_neurons=n_motor_fs_per_action,
                exc_fraction=0.0,
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
            ))
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"motor_{action}", to_region=f"motor_FS_{action}",
                density=0.5,
                weight_mean=motor_to_fs_weight,
                weight_jitter=0.3, plastic=False,
            ))
            for target_action in ACTION_NAMES:
                if target_action == action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"motor_FS_{action}",
                    to_region=f"motor_{target_action}",
                    density=0.5,
                    weight_mean=fs_to_motor_weight,
                    weight_jitter=0.3, plastic=False,
                ))

    # ─────────── Cluster K v2 visual ventral stream (2026-05-12) ───────
    # Sensory grounding for P5 abstract concepts. Mirror of g11_bg_runner
    # K v2 architecture: retina → V1 simple (Gabor) → V1 complex (phase
    # pooling) → V2 → IT. IT acts as concept-level visual hub feeding
    # multimodal_hub for cross-modal semantic binding.
    #
    # Catalog K.01 (V1/V2/IT ventral stream) + G.11 (Hickok & Poeppel
    # multi-stream language). Design:
    # docs/plans/2026-05-12-P5-sensory-grounding-design.md
    if enable_visual_cortex:
        n_retina = 2 * visual_image_size * visual_image_size  # ON + OFF
        n_v1_simple = (visual_n_orientations * visual_n_frequencies
                       * visual_n_positions_per_dim
                       * visual_n_positions_per_dim)
        n_v1_complex = (visual_n_orientations
                        * visual_n_positions_per_dim
                        * visual_n_positions_per_dim)
        regions.append(BrainRegion(
            name="retina",
            n_neurons=n_retina,
            exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="cortex_v1_simple",
            n_neurons=n_v1_simple,
            exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="cortex_v1_complex",
            n_neurons=n_v1_complex,
            exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="cortex_v2",
            n_neurons=visual_n_v2,
            exc_fraction=0.8, internal_density=0.05,
            exc_weight_mean=2.0, inh_weight_mean=4.0,
            weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        regions.append(BrainRegion(
            name="cortex_it",
            n_neurons=visual_n_it,
            exc_fraction=0.8, internal_density=0.10,
            exc_weight_mean=2.0, inh_weight_mean=4.0,
            weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        # retina → V1 simple — Gabor init applied post-build via
        # apply_v1_gabor_weights. Plastic so STDP can refine.
        pathways.append(RegionPathway(
            from_region="retina", to_region="cortex_v1_simple",
            density=0.05, weight_mean=0.5, weight_jitter=0.5,
            plastic=True, plasticity_gate="visual_cortex_v1",
        ))
        # V1 simple → V1 complex (phase pooling). Fixed.
        pathways.append(RegionPathway(
            from_region="cortex_v1_simple",
            to_region="cortex_v1_complex",
            density=visual_n_frequencies / float(n_v1_simple),
            weight_mean=2.0, weight_jitter=0.0, plastic=False,
        ))
        # V1 complex → V2 (higher-order features). Plastic.
        pathways.append(RegionPathway(
            from_region="cortex_v1_complex", to_region="cortex_v2",
            density=0.10, weight_mean=1.0, weight_jitter=0.5,
            plastic=True, plasticity_gate="visual_cortex_v2",
        ))
        # V2 → IT (object/category). Plastic.
        pathways.append(RegionPathway(
            from_region="cortex_v2", to_region="cortex_it",
            density=0.20, weight_mean=1.5, weight_jitter=0.5,
            plastic=True, plasticity_gate="visual_cortex_it",
        ))

    # Multimodal hub: ATL-like convergence zone where auditory
    # (wernicke) + visual (IT) semantic content binds together.
    # The "embodied" signal that Tier 1 motor binding got from motor
    # teacher current is now provided by visual co-firing.
    if enable_multimodal_hub:
        regions.append(BrainRegion(
            name="multimodal_hub",
            n_neurons=n_multimodal_hub,
            exc_fraction=0.8,
            internal_density=multimodal_hub_internal_density,
            exc_weight_mean=multimodal_hub_exc_weight_mean,
            inh_weight_mean=multimodal_hub_inh_weight_mean,
            weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))
        # IT → multimodal_hub (visual semantic content)
        if enable_visual_cortex:
            pathways.append(RegionPathway(
                from_region="cortex_it", to_region="multimodal_hub",
                density=it_to_hub_density,
                weight_mean=it_to_hub_weight,
                weight_jitter=0.2, plastic=True,
                plasticity_gate="it_to_hub",
            ))
        # wernicke_pool_i → multimodal_hub (auditory semantic content)
        if enable_ventral_semantic and enable_multi_pool_wernicke:
            for i in range(n_wernicke_pools):
                pathways.append(RegionPathway(
                    from_region=f"wernicke_pool_{i}",
                    to_region="multimodal_hub",
                    density=wernicke_pool_to_hub_density,
                    weight_mean=wernicke_pool_to_hub_weight,
                    weight_jitter=0.2, plastic=True,
                    plasticity_gate=f"wernicke_pool_{i}_to_hub",
                ))
        # multimodal_hub → lang_output_pool_i (so visual recognition
        # can drive the naming output via the hub)
        if (enable_ventral_semantic
                and enable_multi_pool_wernicke
                and enable_per_concept_lang_out_pools):
            for i in range(n_wernicke_pools):
                pathways.append(RegionPathway(
                    from_region="multimodal_hub",
                    to_region=f"lang_output_pool_{i}",
                    density=hub_to_lang_output_pool_density,
                    weight_mean=hub_to_lang_output_pool_weight,
                    weight_jitter=0.2, plastic=True,
                    plasticity_gate=f"hub_to_lang_pool_{i}",
                ))

    return regions, pathways


def set_awake_gates(bridge, enable_lang_to_motor: bool = True):
    """Phase 1.3 -- switch bridge to AWAKE mode plasticity gates.

    Awake mode plasticity (memory encoding):
    - language_input -> motor: ON (direct route plastic)
    - motor -> language_output: ON (Tier 1 reciprocal)
    - language_input -> dlpfc_verb: ON (Tier 2.3 if present)
    - lang_to_ec: ON (cortex -> hippo encoding)
    - ec_to_dg, dg_to_ca3, ec_to_ca1, ca3_to_ca1: ON (forward path)
    - ca3_swr_burst: OFF (no recurrent learning awake)
    - ca1_to_motor: OFF (no consolidation awake)
    - ca1_to_lang_out: OFF (no consolidation awake)

    Args:
        bridge: SimulationBridge with brain-region framework on.
        enable_lang_to_motor: if False, freezes the direct
            language_input -> motor pathway too. Use this when you
            want to FORCE all motor learning to come via the
            hippocampal route (rare; mostly for ablation studies).
    """
    awake_gates = {
        "language_input_to_motor": 1.0 if enable_lang_to_motor else 0.0,
        "motor_to_language_output": 1.0,
        "language_input_to_dlpfc_verb": 1.0,  # Tier 2.3
        "lang_to_ec": 1.0,
        "ec_to_dg": 1.0,
        "dg_to_ca3": 1.0,
        "ec_to_ca1": 1.0,
        "ca3_to_ca1": 1.0,
        "ca3_swr_burst": 0.0,    # OFF awake
        "ca1_to_motor": 0.0,     # OFF awake
        "ca1_to_lang_out": 0.0,  # OFF awake
    }
    for gate, value in awake_gates.items():
        try:
            bridge.set_plasticity_gate(gate, value)
        except Exception:
            pass  # gate may not exist if pathway not built


def set_sleep_gates(bridge):
    """Phase 1.3 -- switch bridge to SLEEP mode plasticity gates.

    Sleep mode plasticity (consolidation, transfer to cortex):
    - language_input -> motor: OFF (don't disturb cortex via direct)
    - motor -> language_output: OFF
    - language_input -> dlpfc_verb: OFF
    - lang_to_ec, ec_to_dg, ec_to_ca1, ca3_to_ca1: OFF (encoding off)
    - ca3_swr_burst: ON (recurrent autoassociator sharpens)
    - ca1_to_motor: ON (consolidation pathway plastic)
    - ca1_to_lang_out: ON (consolidation for bidirectional)

    During sleep:
    - CA3 fires SWR-style replay bursts (~150Hz, 100ms windows)
    - Replayed patterns drive CA1
    - CA1 -> motor / lang_out STDP transfers patterns to cortex
    - Cortex internal recurrence amplifies transferred patterns
    - Direct lang -> motor frozen so cortex isn't simultaneously
      retrained by the awake-time route
    """
    sleep_gates = {
        "language_input_to_motor": 0.0,
        "motor_to_language_output": 0.0,
        "language_input_to_dlpfc_verb": 0.0,
        "lang_to_ec": 0.0,
        "ec_to_dg": 0.0,
        "dg_to_ca3": 0.0,
        "ec_to_ca1": 0.0,
        "ca3_to_ca1": 1.0,        # rebuilt forward path stays open
        "ca3_swr_burst": 1.0,     # ON during sleep
        "ca1_to_motor": 1.0,      # ON during sleep
        "ca1_to_lang_out": 1.0,   # ON during sleep
    }
    for gate, value in sleep_gates.items():
        try:
            bridge.set_plasticity_gate(gate, value)
        except Exception:
            pass


def freeze_all_gates(bridge):
    """Set all known plasticity gates to 0. Used before evaluation
    so weights don't drift during the eval window."""
    for gate in (
        "language_input_to_motor",
        "motor_to_language_output",
        "language_input_to_dlpfc_verb",
        "lang_to_ec",
        "ec_to_dg",
        "dg_to_ca3",
        "ec_to_ca1",
        "ca3_to_ca1",
        "ca3_swr_burst",
        "ca1_to_motor",
        "ca1_to_lang_out",
    ):
        try:
            bridge.set_plasticity_gate(gate, 0.0)
        except Exception:
            pass


def make_concept_image(
    concept: str,
    retina_size: int = 32,
    drive_pA: float = 200.0,
) -> "np.ndarray":
    """Generate a deterministic retina drive pattern for a concept.

    For P5 sensory grounding (catalog G.11 + K.01 + Pulvermüller
    embodied semantics). Each concept gets a geometric visual prototype
    that's identifiable by V1 Gabor cells + V2 shape features.

    Returns: (2 * retina_size^2,) ndarray of pA values for the retina
    region. First half is ON-channel, second half is OFF-channel.

    Concept prototypes (designed for V1-detectable features):
    - "apple": round bright shape, center of image
                (Gabor cells with low-spatial-frequency tuning will fire)
    - "river": elongated horizontal wave pattern
                (Gabor cells with horizontal orientation will fire)
    - "alice": vertical bar (different orientation)
    - "table": rectangular flat shape
    - Default: faint sparse pattern

    The shapes are intentionally distinct in DOMINANT ORIENTATION and
    SHAPE so V1 simple cell tuning naturally separates them.
    """
    import numpy as np
    img_on = np.zeros((retina_size, retina_size), dtype=np.float32)
    img_off = np.zeros((retina_size, retina_size), dtype=np.float32)

    cx, cy = retina_size // 2, retina_size // 2

    if concept == "apple":
        # Round bright blob in center (low-freq, no preferred orient)
        r_main = retina_size * 0.25
        for y in range(retina_size):
            for x in range(retina_size):
                d = np.sqrt((x - cx)**2 + (y - cy)**2)
                if d < r_main:
                    img_on[y, x] = drive_pA
                elif d < r_main + 2.0:
                    # Edge transition: weak OFF
                    img_off[y, x] = drive_pA * 0.4
    elif concept == "river":
        # Horizontal wavy elongated band (HORIZONTAL orientation)
        center_band_h = retina_size * 0.20
        for y in range(retina_size):
            # Wavy mid-line
            mid = cy + int(2.5 * np.sin(y * 0.6))
            for x in range(retina_size):
                if abs(x - mid) < center_band_h * 0.5:
                    img_on[y, x] = drive_pA
                elif abs(x - mid) < center_band_h * 0.7:
                    img_off[y, x] = drive_pA * 0.4
        # Transpose so horizontal orientation dominates
        img_on = img_on.T.copy()
        img_off = img_off.T.copy()
    elif concept == "alice":
        # Vertical bar (VERTICAL orientation)
        bar_w = max(2, int(retina_size * 0.10))
        for y in range(retina_size):
            for x in range(cx - bar_w, cx + bar_w):
                if 0 <= x < retina_size:
                    img_on[y, x] = drive_pA
    elif concept == "table":
        # Rectangular flat shape (mixed orientations, distinct from
        # apple's roundness via aspect ratio)
        h, w = int(retina_size * 0.15), int(retina_size * 0.35)
        for y in range(cy - h, cy + h):
            for x in range(cx - w, cx + w):
                if 0 <= y < retina_size and 0 <= x < retina_size:
                    img_on[y, x] = drive_pA
    else:
        # Faint default: 5% random sparse pattern
        rng = np.random.default_rng(hash(concept) & 0xFFFFFFFF)
        mask = rng.random((retina_size, retina_size)) < 0.05
        img_on[mask] = drive_pA * 0.5

    # Pack to retina ON/OFF format (channel-first: ON first, then OFF)
    flat = np.concatenate([img_on.flatten(), img_off.flatten()])
    return flat


def apply_novel_key_topographic_bias(
    bridge,
    key: str,
    target_action: str,
    factor: float = 2.0,
    n_lang_input: int = 2048,
    sparsity: float = 0.1,
    verbose: bool = False,
):
    """Apply topographic bias to lang_input -> motor_<target> for a
    novel key BEFORE V_SCHEMA training. Pre-aligns weights so STDP
    has a stronger starting point.

    This is the analog of Tier 1's apply_topographic_bias (direction
    words -> motor pools) but for arbitrary new vocabulary. The novel
    key's active lang_input neurons get their edges to target motor
    pool multiplied by `factor`, edges to OTHER pools multiplied by
    1/factor (downweighted).

    Args:
        key: novel word string (e.g. "apple")
        target_action: "N"/"E"/"S"/"W"
        factor: boost factor for on-target (default 2.0)
        n_lang_input: language_input region size
        sparsity: drive pattern sparsity (matches vocab_to_drive_pattern default)
        verbose: print summary

    Returns:
        dict with edges_boosted, edges_downweighted counts.
    """
    from sim.text_embeddings import vocab_to_drive_pattern
    import numpy as _np

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    motor_target = list(rm.indices(f"motor_{target_action}"))
    other_actions = [a for a in ("N", "E", "S", "W") if a != target_action]

    drive = vocab_to_drive_pattern(
        key, n_neurons=n_lang_input, sparsity=sparsity,
    )
    local_active = _np.where(drive > 0)[0]
    global_active = [lang_input_indices[i] for i in local_active]

    indptr = _to_host(bridge.cp_connections.indptr)
    indices = _to_host(bridge.cp_connections.indices)
    data = _to_host(bridge.cp_connections.data)

    pair_to_idx = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    boosted = 0
    downweighted = 0
    off_factor = 1.0 / factor

    for src in global_active:
        # Boost on-target
        for dst in motor_target:
            key_pair = (src, dst)
            if key_pair in pair_to_idx:
                idx = pair_to_idx[key_pair]
                data[idx] = float(data[idx]) * factor
                boosted += 1
        # Downweight off-target
        for other_action in other_actions:
            motor_other = list(rm.indices(f"motor_{other_action}"))
            for dst in motor_other:
                key_pair = (src, dst)
                if key_pair in pair_to_idx:
                    idx = pair_to_idx[key_pair]
                    data[idx] = float(data[idx]) * off_factor
                    downweighted += 1

    from sim.backend import get_backend
    cp, _ = get_backend()
    bridge.cp_connections.data[...] = cp.asarray(data, dtype=cp.float32)

    if verbose:
        print(f"[novel-key-topographic] '{key}' -> motor_{target_action}: "
              f"boosted={boosted} edges (x{factor:.2f}), "
              f"downweighted={downweighted} (x{off_factor:.2f})")

    return {
        "key": key, "target_action": target_action, "factor": factor,
        "edges_boosted": boosted, "edges_downweighted": downweighted,
    }


def build_tier_2_3_action_gate(
    sensitivity: float = 0.01,
    decay_tau_ms: float = 300.0,
    drive_pA: float = 50.0,
    rate_threshold: float = 0.05,
):
    """Build the Tier 2.3 'action_gate' neuromodulator config.

    This neuromodulator implements the verb-context-required-for-action
    gating mechanism per design. dlpfc_verb activity drives a modulator
    that boosts excitability of all 4 motor pools. Without verb context,
    motor pools are quieter; with it, they fire normally.

    Mechanism:
    - rule_type='from_region_firing' on dlpfc_verb -> emit signal
      proportional to (dlpfc_verb mean firing - threshold)
    - target_type='excitability_drive' on motor_{N,E,S,W} group scope
    - decay_tau_ms=300 matches NMDA-aligned working memory timescale
      (verb context 'go' should boost motor for ~300ms while next
      direction word arrives)

    Biology: PFC -> motor cortex modulation is well-attested; PFC
    excitability biases premotor / M1 firing thresholds via direct
    glutamatergic projections. Goldman-Rakic 1995, Wang 2002 NMDA
    bistability in PFC; Miller & Cohen 2001 PFC top-down control.

    Returns:
        NeuromodulatorConfig with the 4-target action_gate spec.
        Caller must include this in cfg.neuromodulators when
        cfg.enable_neuromodulator_subsystem=True.

    Compatible with build_biological_brain_regions(enable_dlpfc_verb=True).
    """
    from sim.neuromodulators import (
        NeuromodulatorConfig, ModulatorTarget, ProductionRule,
    )
    targets = []
    for action in ["N", "E", "S", "W"]:
        targets.append(ModulatorTarget(
            target_type="excitability_drive",
            scope=f"group:motor_{action}",
            sensitivity=drive_pA,
        ))
    rules = [
        ProductionRule(
            rule_type="from_region_firing",
            sensitivity=sensitivity,
            threshold=rate_threshold,
            window_ms=200.0,  # EMA tau for PFC firing rate
            source_regions=["dlpfc_verb"],
        ),
    ]
    return NeuromodulatorConfig(
        name="action_gate",
        baseline=0.0,
        decay_tau_ms=decay_tau_ms,
        concentration_min=0.0,
        concentration_max=1.0,
        targets=targets,
        production_rules=rules,
    )


def apply_topographic_bias(
    bridge,
    topographic_factor: float = 1.5,
    off_target_factor: float = 0.7,
    n_lang_input: int = 256,
    sparsity: float = 0.1,
    orthogonal_cues: bool = False,
    apply_reciprocal: bool = False,  # Tier 1: also bias motor → language_output
    n_lang_output: int = 256,
    verbose: bool = True,
):
    """Apply biology-grounded topographic bias to language_input -> motor_X
    weights. Models the somatotopic Wernicke->motor projection that real
    cortex develops via early Hebbian co-firing (Pulvermüller 2001-2003,
    Hauk et al 2004).

    For each word w with active neuron set A_w (the same set that
    vocab_to_drive_pattern returns):
        weights[A_w -> motor_target(w)] *= topographic_factor
        weights[A_w -> motor_other]     *= off_target_factor

    With default 1.5 / 0.7, the ratio between target and off-target is
    ~2.1x — squarely within Pulvermüller's reported biology range
    (2-3x).

    With baseline weight=3.0, topographic_factor=1.5 gives target init
    of 4.5 — well below stdp_w_max=5.0, leaving STDP room to grow OR
    shrink. Off-target init of 2.1 has even more headroom.

    Args:
        bridge: initialized SimulationBridge (after _initialize_simulation_data)
        topographic_factor: multiplier for weights from active neurons
            of word w to motor_target(w). Default 1.5 (mid-biology).
        off_target_factor: multiplier for weights from active neurons
            of word w to other motor pools. Default 0.7.
        n_lang_input: language_input region size. Used to compute the
            same drive pattern as the eval/training pipeline.
        sparsity: same sparsity used for token drive elsewhere.

    Returns:
        dict with applied edge counts per pathway, for verification.
    """
    import numpy as np
    from sim.text_embeddings import vocab_to_drive_pattern, orthogonal_drive_pattern

    if bridge.region_manager is None:
        raise RuntimeError("apply_topographic_bias: region_manager is None")

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    n_lang = len(lang_input_indices)
    if n_lang != n_lang_input:
        raise ValueError(
            f"apply_topographic_bias: bridge has {n_lang} language_input "
            f"neurons but caller specified {n_lang_input}"
        )

    word_to_action = {"north": "N", "east": "E", "south": "S", "west": "W"}
    actions = ["N", "E", "S", "W"]
    # Stable index for orthogonal-cue mode. MUST match _VOCAB_ORDER in
    # bio_three_factor.py so the topographic prior boosts the same
    # neurons that vocab_to_drive_pattern_orthogonal will activate at
    # eval/training time.
    word_to_idx = {w: i for i, w in enumerate(word_to_action.keys())}

    # Extract current CSR weights once (avoids per-pathway pull)
    indptr = _to_host(bridge.cp_connections.indptr)
    indices = _to_host(bridge.cp_connections.indices)
    data = _to_host(bridge.cp_connections.data)

    # Pre-compute (pre, post) -> data index for fast lookup
    pair_to_idx = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    summary = {}
    for word, target_action in word_to_action.items():
        # Active language_input neurons for this word. Encoder MUST
        # match what bio_three_factor uses at training/eval time.
        if orthogonal_cues:
            drive = orthogonal_drive_pattern(
                cue_idx=word_to_idx[word],
                n_cues=len(word_to_action),
                n_neurons=n_lang_input,
                sparsity=sparsity,
            )
        else:
            drive = vocab_to_drive_pattern(word, n_neurons=n_lang_input,
                                            sparsity=sparsity)
        local_active = np.where(drive > 0)[0]
        global_active = [lang_input_indices[i] for i in local_active]

        for action in actions:
            motor_indices = list(rm.indices(f"motor_{action}"))
            factor = (topographic_factor if action == target_action
                      else off_target_factor)

            n_changed = 0
            for src in global_active:
                for dst in motor_indices:
                    key = (src, dst)
                    if key in pair_to_idx:
                        idx = pair_to_idx[key]
                        data[idx] = float(data[idx]) * factor
                        n_changed += 1
            summary[f"{word}->motor_{action}"] = {
                "factor": factor,
                "edges_modified": n_changed,
            }

    # Reciprocal direction: motor_X → language_output (Tier 1).
    # Same Pulvermüller-style somatotopic prior, just inverse direction:
    # motor_N edges to "north"-encoded neurons in language_output get
    # boosted; off-target reduced. Without this, motor pools project
    # uniformly and language_output can't differentiate inputs above
    # the architectural N-bias floor.
    if apply_reciprocal:
        try:
            lang_output_indices = list(rm.indices("language_output"))
        except Exception as e:
            raise RuntimeError(
                f"apply_topographic_bias(apply_reciprocal=True): "
                f"language_output region not found. Builder must be "
                f"called with enable_language_output=True. ({e})"
            )
        if len(lang_output_indices) != n_lang_output:
            raise ValueError(
                f"apply_topographic_bias: bridge has {len(lang_output_indices)} "
                f"language_output neurons but caller specified {n_lang_output}"
            )

        for word, target_action in word_to_action.items():
            # Active language_output neurons for this word — same
            # encoding as eval/training time.
            if orthogonal_cues:
                drive = orthogonal_drive_pattern(
                    cue_idx=word_to_idx[word],
                    n_cues=len(word_to_action),
                    n_neurons=n_lang_output,
                    sparsity=sparsity,
                )
            else:
                drive = vocab_to_drive_pattern(word, n_neurons=n_lang_output,
                                                sparsity=sparsity)
            local_active = np.where(drive > 0)[0]
            global_active = [lang_output_indices[i] for i in local_active]

            for action in actions:
                motor_indices = list(rm.indices(f"motor_{action}"))
                # If this action's motor is the target for this word,
                # boost its edges to the word's lang_output neurons.
                # Otherwise, dampen.
                factor = (topographic_factor if action == target_action
                          else off_target_factor)
                n_changed = 0
                for src in motor_indices:  # motor neuron is pre
                    for dst in global_active:  # lang_output is post
                        key = (src, dst)
                        if key in pair_to_idx:
                            idx = pair_to_idx[key]
                            data[idx] = float(data[idx]) * factor
                            n_changed += 1
                summary[f"motor_{action}->{word}"] = {
                    "factor": factor,
                    "edges_modified": n_changed,
                }

    # Push back to active backend (cupy or numpy)
    from sim.backend import get_backend
    cp, _ = get_backend()
    bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)

    if verbose:
        print(f"[topographic-bias] Applied factor={topographic_factor:.2f}/"
              f"{off_target_factor:.2f} to language_input -> motor_X"
              f"{' + motor_X -> language_output (reciprocal)' if apply_reciprocal else ''}")
        for k, v in summary.items():
            print(f"  {k}: x{v['factor']:.2f} on {v['edges_modified']} edges")

    return summary


def apply_wernicke_topographic_bias(
    bridge,
    concepts: list,  # e.g., ["apple", "river"]
    topographic_factor: float = 1.5,
    off_target_factor: float = 0.7,
    n_lang_input: int = 1024,
    sparsity: float = 0.1,
    verbose: bool = True,
):
    """Apply biology-grounded topographic bias to language_input ->
    wernicke weights. Each concept's active lang_input neurons get
    boosted weights to a DEDICATED contiguous subset of wernicke
    neurons (their "concept ensemble"). Off-target wernicke
    subsets get reduced weights.

    Path G+ MINIMAL: same wernicke region as default, but topographic
    structural prior creates per-concept ensembles. Mirror of the
    apply_topographic_bias for motor pools (which produced Tier 1
    multi-seed PASS).

    Args:
        concepts: list of concept names (e.g., ["apple", "river"]).
            Each concept gets a contiguous wernicke slice.
        topographic_factor: weight multiplier for concept's lang
            neurons -> concept's wernicke subset. Default 1.5.
        off_target_factor: weight multiplier for concept's lang
            neurons -> OTHER concept's wernicke subset. Default 0.7.
        n_lang_input: lang_input region size.
        sparsity: token drive sparsity.

    Returns: dict with edge modification summary per concept.

    Biology: Wernicke's area has topographic phoneme organization
    (Pulvermüller 2001-2003) — different phonemes activate different
    sub-regions. This function imposes that structure.
    """
    from sim.text_embeddings import vocab_to_drive_pattern
    import numpy as np

    if bridge.region_manager is None:
        raise RuntimeError("apply_wernicke_topographic_bias: "
                           "region_manager is None")

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    wernicke_indices = list(rm.indices("wernicke"))
    n_wernicke = len(wernicke_indices)

    if len(lang_input_indices) != n_lang_input:
        raise ValueError(
            f"apply_wernicke_topographic_bias: bridge has "
            f"{len(lang_input_indices)} language_input neurons but "
            f"caller specified {n_lang_input}"
        )

    n_concepts = len(concepts)
    if n_concepts < 2:
        raise ValueError("Need at least 2 concepts to bias")
    if n_wernicke < n_concepts:
        raise ValueError(
            f"Wernicke has {n_wernicke} neurons but {n_concepts} "
            "concepts requested — increase n_wernicke")

    # Split wernicke into contiguous slices per concept
    slice_size = n_wernicke // n_concepts
    concept_slices = {}
    for i, concept in enumerate(concepts):
        start = i * slice_size
        end = start + slice_size if i < n_concepts - 1 else n_wernicke
        concept_slices[concept] = wernicke_indices[start:end]

    # Extract CSR weights
    indptr = _to_host(bridge.cp_connections.indptr)
    indices = _to_host(bridge.cp_connections.indices)
    data = _to_host(bridge.cp_connections.data)

    # Pre-compute (pre, post) -> data index for fast lookup
    pair_to_idx = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    summary = {}
    for concept in concepts:
        drive = vocab_to_drive_pattern(
            concept, n_neurons=n_lang_input, sparsity=sparsity,
        )
        local_active = np.where(drive > 0)[0]
        global_active = [lang_input_indices[i] for i in local_active]

        for target_concept in concepts:
            wernicke_subset = concept_slices[target_concept]
            factor = (topographic_factor if target_concept == concept
                      else off_target_factor)
            n_changed = 0
            for src in global_active:
                for dst in wernicke_subset:
                    key = (src, dst)
                    if key in pair_to_idx:
                        idx = pair_to_idx[key]
                        data[idx] = float(data[idx]) * factor
                        n_changed += 1
            summary[f"{concept}->wernicke_{target_concept}"] = {
                "factor": factor,
                "edges_modified": n_changed,
            }

    # Push modified data back to bridge
    from sim.backend import get_backend
    cp, _ = get_backend()
    bridge.cp_connections.data[...] = cp.asarray(data,
                                                   dtype=cp.float32)

    if verbose:
        print(f"[wernicke-topographic] {len(concepts)} concepts, "
              f"slice size {slice_size}, factor={topographic_factor:.2f}/"
              f"{off_target_factor:.2f}")
        for k, v in summary.items():
            print(f"  {k}: x{v['factor']:.2f} on {v['edges_modified']} edges")

    return summary


def apply_wernicke_pool_topographic_bias(
    bridge,
    concepts: list,  # e.g., ["apple", "river"]
    topographic_factor: float = 2.0,
    off_target_factor: float = 0.5,
    n_lang_input: int = 1024,
    sparsity: float = 0.1,
    use_orthogonal_codes: bool = False,
    verbose: bool = True,
):
    """Apply topographic bias to lang_input -> wernicke_pool_i
    pathways. For each concept, boost its active lang_input
    neurons' weights to its ASSIGNED pool and reduce weights to
    OTHER pools. Path A iter U: routing per-concept.

    Args:
        concepts: list of concept names. concepts[i] maps to
            wernicke_pool_i.
        topographic_factor: weight multiplier for on-target
            (concept[i] -> wernicke_pool_i) connections.
            Default 2.0 (stronger than the single-region 1.5).
        off_target_factor: weight multiplier for off-target.
            Default 0.5 (more aggressive than 0.7).
        n_lang_input: lang_input region size.
        sparsity: token drive sparsity.
        use_orthogonal_codes: if True, use orthogonal_drive_pattern
            (zero overlap between concepts) instead of
            vocab_to_drive_pattern (8.8% overlap for 2 concepts).
            Iter NN test: removes input-code ambiguity to test if
            the per-seed pool bias at biological scale is caused
            by overlapping codes confusing the topographic bias.
    """
    from sim.text_embeddings import vocab_to_drive_pattern, orthogonal_drive_pattern
    import numpy as np

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    if len(lang_input_indices) != n_lang_input:
        raise ValueError(
            f"apply_wernicke_pool_topographic_bias: bridge has "
            f"{len(lang_input_indices)} language_input neurons but "
            f"caller specified {n_lang_input}"
        )

    pool_indices = {}
    for i in range(len(concepts)):
        pool_name = f"wernicke_pool_{i}"
        pool_indices[i] = list(rm.indices(pool_name))

    indptr = _to_host(bridge.cp_connections.indptr)
    indices = _to_host(bridge.cp_connections.indices)
    data = _to_host(bridge.cp_connections.data)

    pair_to_idx = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    summary = {}
    for concept_i, concept in enumerate(concepts):
        if use_orthogonal_codes:
            drive = orthogonal_drive_pattern(
                cue_idx=concept_i, n_cues=len(concepts),
                n_neurons=n_lang_input, sparsity=sparsity,
            )
        else:
            drive = vocab_to_drive_pattern(
                concept, n_neurons=n_lang_input, sparsity=sparsity,
            )
        local_active = np.where(drive > 0)[0]
        global_active = [lang_input_indices[i] for i in local_active]

        for pool_i in range(len(concepts)):
            pool_dst = pool_indices[pool_i]
            factor = (topographic_factor if pool_i == concept_i
                      else off_target_factor)
            n_changed = 0
            for src in global_active:
                for dst in pool_dst:
                    key = (src, dst)
                    if key in pair_to_idx:
                        idx = pair_to_idx[key]
                        data[idx] = float(data[idx]) * factor
                        n_changed += 1
            summary[f"{concept}->wernicke_pool_{pool_i}"] = {
                "factor": factor,
                "edges_modified": n_changed,
            }

    from sim.backend import get_backend
    cp, _ = get_backend()
    bridge.cp_connections.data[...] = cp.asarray(data,
                                                   dtype=cp.float32)

    if verbose:
        print(f"[wernicke-pool-topographic] {len(concepts)} concepts, "
              f"factor={topographic_factor:.2f}/{off_target_factor:.2f}")
        for k, v in summary.items():
            print(f"  {k}: x{v['factor']:.2f} on {v['edges_modified']} edges")

    return summary


def run_minimal_isolation(
    seed: int = 42,
    n_events_per_direction: int = 1000,
    stim_steps_per_step: int = 100,
    reset_steps: int = 50,
    lang_input_drive_pA: float = 200.0,
    lang_output_coactive_pA: float = 0.0,  # no language_output region
    motor_replay_drive_pA: float = 50.0,
    n_motor_per_action: int = 25,
    n_lang_input: int = 256,
    token_sparsity: float = 0.1,
    dt_ms: float = 1.0,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    stdp_w_max: float = 5.0,
    enable_hebbian: bool = False,
    # Biology-grounded additions (2026-05-03)
    enable_motor_fs: bool = False,
    n_motor_fs_per_action: int = 3,
    topographic_bias_factor: float = 1.0,  # 1.0 = off (uniform random)
    off_target_bias_factor: float = 1.0,   # 1.0 = off (uniform random)
    freeze_stdp: bool = False,             # anti-cheat control: skip STDP
    # Performance: fast-path spike reset (no GPU-CPU sync). 1.29x on
    # minimal arch under 4-way contention. Numerical equivalence verified
    # at tests/test_fast_spike_reset.py.
    fast_spike_reset: bool = True,
    # Biological-scale architecture (2026-05-04). When True, uses
    # build_biological_brain_regions: cortical canon (recurrence + E/I +
    # NMDA) + larger N. See function docstring for parameter details.
    biological: bool = False,
    enable_nmda: bool = False,
    ou_tau_ms: float = 15.0,
    ou_std_current_pA: float = 100.0,
    verbose: bool = True,
):
    """Run the minimal isolation experiment. Returns (bridge, stats)."""
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 60)
        print(f"MINIMAL LANGUAGE->MOTOR ISOLATION (seed={seed})")
        print(f"  n_lang_input={n_lang_input}, motor_per_action={n_motor_per_action}")
        total_neurons = (n_lang_input + 4 * n_motor_per_action +
                         (4 * n_motor_fs_per_action if enable_motor_fs else 0))
        print(f"  Total: {total_neurons} neurons")
        print(f"  {n_events_per_direction} paired-stim events per direction")
        print(f"  dt={dt_ms}ms, stim={stim_steps_per_step}, reset={reset_steps}")
        print(f"  enable_hebbian={enable_hebbian}, stdp_w_max={stdp_w_max}")
        print(f"  enable_motor_fs={enable_motor_fs} (n_fs_per_action="
              f"{n_motor_fs_per_action})")
        print(f"  topographic_bias: target={topographic_bias_factor:.2f}, "
              f"off={off_target_bias_factor:.2f} "
              f"(1.0/1.0 = no topography)")
        print(f"  freeze_stdp={freeze_stdp} (anti-cheat control)")
        if biological:
            print(f"  BIOLOGICAL ARCH: cortical canon (recurrence + E/I + NMDA)")
            print(f"  enable_nmda={enable_nmda}, ou_tau_ms={ou_tau_ms}, "
                  f"ou_std_current_pA={ou_std_current_pA}")
        print("=" * 60, flush=True)

    if biological:
        regions, pathways = build_biological_brain_regions(
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            text_input_to_motor_weight=text_input_to_motor_weight,
            text_input_to_motor_jitter=text_input_to_motor_jitter,
            enable_motor_fs=enable_motor_fs,
            n_motor_fs_per_action=n_motor_fs_per_action,
        )
    else:
        regions, pathways = build_minimal_brain_regions(
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            text_input_to_motor_weight=text_input_to_motor_weight,
            text_input_to_motor_jitter=text_input_to_motor_jitter,
            enable_motor_fs=enable_motor_fs,
            n_motor_fs_per_action=n_motor_fs_per_action,
        )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = dt_ms
    cfg.seed = seed
    cfg.enable_nmda = enable_nmda
    cfg.ou_tau_ms = ou_tau_ms
    cfg.ou_std_current_pA = ou_std_current_pA
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = enable_hebbian
    cfg.stdp_w_max = stdp_w_max
    cfg.fast_spike_reset = fast_spike_reset

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Apply topographic bias if requested (must come AFTER init, before
    # training, so STDP can refine from the biased starting point).
    if topographic_bias_factor != 1.0 or off_target_bias_factor != 1.0:
        apply_topographic_bias(
            bridge,
            topographic_factor=topographic_bias_factor,
            off_target_factor=off_target_bias_factor,
            n_lang_input=n_lang_input,
            sparsity=token_sparsity,
            verbose=verbose,
        )

    # Anti-cheat control: freeze STDP via plasticity gate. Tests whether
    # topographic bias alone (without learning) solves the task.
    if freeze_stdp:
        try:
            bridge.set_plasticity_gate("language_input_to_motor", 0.0)
            if verbose:
                print("[freeze_stdp] STDP frozen on language_input_to_motor "
                      "(anti-cheat control)", flush=True)
        except Exception as e:
            print(f"[freeze_stdp] WARNING: could not freeze gate: {e}",
                  flush=True)

    # Build synthetic balanced experience buffer
    DIRECTIONS = ["north", "east", "south", "west"]
    DIRECTION_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}
    synthetic_buffer = []
    for direction in DIRECTIONS:
        action = DIRECTION_TO_ACTION[direction]
        for _ in range(n_events_per_direction):
            synthetic_buffer.append({
                "token": direction,
                "action": action,
                "reward": 1.0,
                "correct_move": True,
            })
    rng.shuffle(synthetic_buffer)

    if verbose:
        print(f"\n[minimal-iso] Synthetic buffer: {len(synthetic_buffer)} events "
              f"({n_events_per_direction}/dir, shuffled)", flush=True)

    # Training: paired-stim using same mechanism as H4 SWR replay.
    # Inline since we don't have language_output region (curriculum's
    # _run_swr_replay_phase requires it).
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    motor_idx = {
        a: cp.asarray(list(rm.indices(f"motor_{a}")), dtype=cp.int64)
        for a in ["N", "E", "S", "W"]
    }
    n_lang = int(lang_input_idx.size)

    t_start = time.time()
    n_replays = 0
    for event_idx, event in enumerate(synthetic_buffer):
        token = event["token"]
        action = event["action"]
        reward = event["reward"]

        # Inter-trial reset
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Drive language_input only (no language_output in minimal arch)
        in_drive = vocab_to_drive_pattern(
            token, n_neurons=n_lang,
            drive_max_pA=lang_input_drive_pA, sparsity=token_sparsity,
        )
        bridge.cp_external_input_current[lang_input_idx] = cp.asarray(
            in_drive, dtype=cp.float32,
        )
        # Drive motor pool (the "nudge" toward correct action)
        bridge.cp_external_input_current[motor_idx[action]] += motor_replay_drive_pA

        # Stim window
        for _ in range(stim_steps_per_step):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Apply reward
        bridge.core_config.current_reward_signal = float(reward)
        for _ in range(20):  # reward window
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        n_replays += 1

        if verbose and (event_idx + 1) % 250 == 0:
            elapsed = time.time() - t_start
            print(f"  [minimal-iso] {event_idx+1}/{len(synthetic_buffer)} events "
                  f"({elapsed:.0f}s)", flush=True)
            # Tier-1 universal progress event for webapp
            from sim.progress import emit_progress
            emit_progress(
                "replay", event_idx + 1, len(synthetic_buffer),
                phase="paired-stim", unit="events",
                label="minimal-isolation",
                elapsed_seconds=elapsed,
            )

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n[minimal-iso] Training complete: {n_replays} events "
              f"({elapsed:.0f}s)", flush=True)

    training_stats = [{
        "phase": 1,
        "regime": "minimal_language_motor_isolation",
        "n_total_events": n_replays,
        "n_per_direction": n_events_per_direction,
        "elapsed_seconds": elapsed,
    }]

    return bridge, training_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events-per-direction", type=int, default=1000,
                    help="Paired-stim events per direction (default 1000)")
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--lang-input-drive-pA", type=float, default=200.0)
    ap.add_argument("--motor-replay-drive-pA", type=float, default=50.0)
    ap.add_argument("--n-motor-per-action", type=int, default=25)
    ap.add_argument("--n-lang-input", type=int, default=256)
    ap.add_argument("--stim-steps-per-step", type=int, default=100)
    ap.add_argument("--reset-steps", type=int, default=50)
    ap.add_argument("--token-sparsity", type=float, default=0.1)
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--text-input-to-motor-weight", type=float, default=3.0)
    ap.add_argument("--text-input-to-motor-jitter", type=float, default=0.5)
    ap.add_argument("--stdp-w-max", type=float, default=5.0)
    ap.add_argument("--enable-hebbian", action="store_true", default=False)
    # Biology-grounded additions (2026-05-03)
    ap.add_argument("--enable-motor-fs", action="store_true", default=False,
                    help="add motor PV-FS interneurons providing cross-pool "
                    "lateral inhibition (3 FS neurons per pool by default)")
    ap.add_argument("--n-motor-fs-per-action", type=int, default=3,
                    help="FS interneurons per motor pool (default 3 ~12%% "
                    "of 25-neuron pool, biology range 10-15%%)")
    ap.add_argument("--topographic-bias-factor", type=float, default=1.0,
                    help="multiplier for weights from word's active neurons "
                    "to its target motor pool. 1.0 = no topography (random). "
                    "1.5 = mid-biology range (Pulvermuller 2001-2003 ratio "
                    "~2-3x). Pair with --off-target-bias-factor < 1.0.")
    ap.add_argument("--off-target-bias-factor", type=float, default=1.0,
                    help="multiplier for weights from word's active neurons "
                    "to NON-target motor pools. 1.0 = no topography. 0.7 "
                    "with topographic-bias-factor=1.5 gives ratio ~2.1x.")
    ap.add_argument("--freeze-stdp", action="store_true", default=False,
                    help="anti-cheat control: freeze STDP on the language_"
                    "input_to_motor pathway. Combined with topographic bias, "
                    "tests whether the prior alone solves the task.")
    ap.add_argument("--no-fast-spike-reset", dest="fast_spike_reset",
                    action="store_false", default=True,
                    help="disable the fast spike-reset optimization "
                    "(cp.where masked-update, no GPU-CPU sync). Default "
                    "is enabled for ~1.3x speedup on minimal arch.")
    # Biological-scale architecture (2026-05-04). When --biological is set,
    # the runner uses build_biological_brain_regions and bumps default sizes:
    # n_lang_input=2048, n_motor_per_action=500, n_motor_fs_per_action=60.
    # Override individual sizes by passing the relevant flags after --biological.
    ap.add_argument("--biological", action="store_true", default=False,
                    help="use biological-scale architecture (cortical canon: "
                    "recurrent excitation + E/I balance + larger N). "
                    "Auto-bumps lang/motor/FS sizes; enables NMDA. See "
                    "build_biological_brain_regions docstring.")
    ap.add_argument("--enable-nmda", action="store_true", default=False,
                    help="enable NMDA synapses globally (Wang 2002 "
                    "bistability). Auto-set when --biological. Defaults off.")
    ap.add_argument("--ou-tau-ms", type=float, default=15.0,
                    help="OU noise correlation time. Default 15ms (synaptic-"
                    "timescale). Set 50-100ms for slower biological cortical "
                    "noise (alpha/beta-scale).")
    ap.add_argument("--ou-std-current-pA", type=float, default=100.0,
                    help="OU noise amplitude. Default 100pA (CoreSimConfig "
                    "default).")
    args = ap.parse_args()

    # --biological auto-bumps sizes if user didn't override them
    if args.biological:
        if args.n_lang_input == 256:
            args.n_lang_input = 2048
        if args.n_motor_per_action == 25:
            args.n_motor_per_action = 500
        if args.n_motor_fs_per_action == 3:
            args.n_motor_fs_per_action = 60
        # NMDA is integral to biological motor pool dynamics (Wang 2002).
        # Auto-enable unless user explicitly opts out (no opt-out flag yet,
        # so just force on when --biological).
        args.enable_nmda = True

    bridge, train_stats = run_minimal_isolation(
        seed=args.seed,
        n_events_per_direction=args.n_events_per_direction,
        stim_steps_per_step=args.stim_steps_per_step,
        reset_steps=args.reset_steps,
        lang_input_drive_pA=args.lang_input_drive_pA,
        motor_replay_drive_pA=args.motor_replay_drive_pA,
        n_motor_per_action=args.n_motor_per_action,
        n_lang_input=args.n_lang_input,
        token_sparsity=args.token_sparsity,
        dt_ms=args.dt_ms,
        text_input_to_motor_weight=args.text_input_to_motor_weight,
        text_input_to_motor_jitter=args.text_input_to_motor_jitter,
        stdp_w_max=args.stdp_w_max,
        enable_hebbian=args.enable_hebbian,
        enable_motor_fs=args.enable_motor_fs,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        topographic_bias_factor=args.topographic_bias_factor,
        off_target_bias_factor=args.off_target_bias_factor,
        freeze_stdp=args.freeze_stdp,
        fast_spike_reset=args.fast_spike_reset,
        biological=args.biological,
        enable_nmda=args.enable_nmda,
        ou_tau_ms=args.ou_tau_ms,
        ou_std_current_pA=args.ou_std_current_pA,
        verbose=True,
    )

    # Eval W->A only (no I->W since no visual cortex)
    from research.runners.text_eval import evaluate_word_to_action
    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_word_action} per word, "
          f"token_sparsity={args.token_sparsity})")
    print("=" * 60, flush=True)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_word_action,
        stim_steps_per_trial=args.stim_steps_per_step,
        n_reset_steps=args.reset_steps,
        token_sparsity=args.token_sparsity,
    )
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}")
    print(f"  Confusion: {wa_result['confusion_matrix']}", flush=True)

    if args.out_stats:
        out = {
            "regime": "minimal_language_motor_isolation",
            "seed": args.seed,
            "n_events_per_direction": args.n_events_per_direction,
            "n_total_events": 4 * args.n_events_per_direction,
            "training_stats": train_stats,
            "word_to_action_eval": wa_result,
            "config": {
                "n_lang_input": args.n_lang_input,
                "n_motor_per_action": args.n_motor_per_action,
                "lang_input_drive_pA": args.lang_input_drive_pA,
                "motor_replay_drive_pA": args.motor_replay_drive_pA,
                "stim_steps_per_step": args.stim_steps_per_step,
                "reset_steps": args.reset_steps,
                "token_sparsity": args.token_sparsity,
                "dt_ms": args.dt_ms,
                "text_input_to_motor_weight": args.text_input_to_motor_weight,
                "stdp_w_max": args.stdp_w_max,
                "enable_hebbian": args.enable_hebbian,
                "enable_motor_fs": args.enable_motor_fs,
                "n_motor_fs_per_action": args.n_motor_fs_per_action,
                "topographic_bias_factor": args.topographic_bias_factor,
                "off_target_bias_factor": args.off_target_bias_factor,
                "freeze_stdp": args.freeze_stdp,
            },
        }
        from pathlib import Path
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
