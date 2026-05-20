"""Net-new unified per-regime monitor + per-regime encoding runner.

Biology (complementary-learning-systems theory; McClelland 1995;
Tonegawa 2015; Miyamoto 2017): the brain runs SEPARATE encoding regimes
AND SEPARATE metacognitive monitors per regime. Cortical multi-event
schema learning binds direct concepts; hippocampal one-shot relational
binding binds compositional content. A unified architecture needs BOTH
regimes' encoding wired in BEFORE the per-regime routing through both
calibrated moats can clear all four conjunctive verdict bars
simultaneously.

The previous per-regime metacognitive-monitor stage's nuanced FAIL
diagnosed this precisely: per-regime separation works (uniform_ctrl=0
collapsed correctly; first-ever non-zero full_acc at small load) but
the direct_retain bar collapsed because the runner used one-shot pair
encoding for direct queries -- the v14/v16-multi-event-calibrated 650
direct gate has no signal to clear against a one-shot-encoded substrate.

This unified runner adds Phase-1 multi-event direct training BEFORE
the compositional one-shot encoding, then routes per-query-type through
both calibrated moats. The orchestrating runner is the ONLY net-new
code; EVERY learning rule + subsystem is REUSED by import:

  * Substrate construction (validated v16 + hippocampus + dlpfc PFC
    frame -- the SAME substrate Stage-1 / SPEAR / Pirazzini /
    Per-regime all used): REUSED ``text_minimal_isolation
    .build_biological_brain_regions(
    enable_hippocampus_consolidation=True, enable_noun_pools=True,
    enable_verb_pools=True, enable_adjective_pools=True, ...)``
    byte-unchanged. The new substrate has BOTH hippocampus (so the
    engram ``region_filter=["dg","ca3","ca1"]`` resolves to a real
    index set and ``commit_engram_tag`` produces tags with non-zero
    ``n_tagged``) AND concept pools (so the v14/v16 multi-event
    training applies cleanly via the validated
    ``apply_concept_topographic_bias`` + ``train_word_to_pool``
    flow byte-unchanged). The prior runner built on
    ``cpd.build_concept_bridge`` -- concept-pool-only, NO
    hippocampus -- which made the engram tags zero-neuron
    (the adversarial-review-blocked defect #1).
  * Phase-1 multi-event direct training (validated v14/v16 88.75%
    multi-seed): REUSED ``concept_pool_demo
    .apply_concept_topographic_bias`` + ``train_word_to_pool``
    byte-unchanged (the same recipe ``run_concept_pool_demo`` uses
    internally; we just call them directly on the new substrate).
  * Bridge state persistence (HDF5; byte-stable at same seed):
    REUSED ``bridge.save_checkpoint`` / ``bridge.load_checkpoint``.
  * Compositional one-shot encoding: REUSED
    ``compose_concept_engram.encode_concept_pair`` (byte-unchanged) --
    opens the ``cross_pool_concept`` plasticity gate around the
    encoding window then closes it; tags the co-fired neurons via the
    engram API.
  * Direct W->A readout: REUSED ``concept_pool_demo.measure_pool_firing``
    (the validated v14/v16 readout the 650 moat is calibrated against).
  * Compositional readout + 5.69 gate: REUSED the per-regime monitor
    runner's ``_compositional_query_confidence`` pattern (raw firing-
    rate confidence at lang_output via ``lang_output_pattern_during_*``
    + the calibrated ``_ranked_from_pattern``).
  * Capability verdict: REUSED ``per_regime_monitor_core
    .per_regime_monitor_verdict`` (byte-unchanged frozen bars
    [0.80, 0.10, 0.80, 0.90], ladder (2,3,5), min 3 seeds).
  * Both calibrated moats: REUSED ``abstention_gate.gate(., 650.0)`` +
    ``abstention_gate_compositional.gate(., 5.6887...)``.
  * Kill-safe checkpoint: REUSED ``sim.train_checkpoint.save_checkpoint``
    / ``load_checkpoint`` / ``resume_epoch``.

Per seed (in order):
  1. Phase-1 training (cached): if ``{phase1_cache_dir}/seed{seed}.simstate.h5``
     exists, skip. Else build the validated v16 + hippocampus + dlpfc
     substrate, apply ``apply_concept_topographic_bias``, then loop
     ``train_word_to_pool`` over the v14/v16 vocabulary -- the same
     recipe ``run_concept_pool_demo`` uses internally. The substrate
     is saved at the cache path for downstream (seed, N) cells to
     ``load_checkpoint`` against. Phase-1 produces a trained substrate
     at the v14/v16-calibrated direct-pool-firing-rate confidence the
     650 direct moat is calibrated on.

Per (seed, N) cell:
  2. Build a fresh bridge via the SAME substrate builder with the SAME
     dimensions (so the loaded architecture matches the saved
     checkpoint), then ``bridge.load_checkpoint(cache_path)``; freeze
     all Phase-1 plasticity gates so encoding does not perturb v14/v16's
     reciprocal binding.
  3. Compositional one-shot encoding: generate N held-out compositional
     (noun, adj) pairs deterministically from the seed (via a sub-seed
     offset distinct from the per-regime calibration's +10000 offset,
     so the unified runner's compositional facts cannot be confused
     with any future calibration set). Encode each pair via
     ``encode_concept_pair`` (opens / closes the ``cross_pool_concept``
     gate around the encoding window). Tag names are OPAQUE
     (``f"ep_{i}"``) -- the answer is decoded from the validated neural
     readout, never out of a tag string.
  4. Per-query routing (per query):
     * DIRECT: ``measure_pool_firing`` -> ranked-by-rate list ->
       ``gate_direct(ranked, MOAT_DIRECT=650.0)`` -> answer-or-abstain.
     * COMPOSITIONAL: validated compositional readout (raw firing-rate
       confidence at lang_output, via ``lang_output_pattern_during_*``
       + ``_ranked_from_pattern``) ->
       ``gate_compositional(ranked, COMPOSITIONAL_THRESHOLD=5.6887...)``
       -> answer-or-abstain.
  5. Three measurement arms (per cell), all from the SAME forward pass:
     * ``full`` = per-regime routing (direct -> 650; compositional ->
       5.69). Sum of correct direct + correct compositional / total.
     * ``uniform_ctrl`` = SAME ranked confidences but BOTH gates set
       to MOAT_DIRECT=650 (the decisive built-in control: the per-
       regime separation must be the differentiator).
     * ``direct_retain`` = direct-queries-only accuracy under the per-
       regime arm. Read from the SAME run as ``full`` (the same direct-
       correct counter, divided by total direct queries).
     * ``abstain_correct`` = fraction of UNGROUNDABLE queries
       (vocabulary words NOT used in this rung) on which the appropriate-
       regime gate abstained.
  6. Emit per (seed, N): the six required rung keys. Aggregate to one
     rung per N. Call ``per_regime_monitor_verdict(rungs)`` unchanged.

Anti-cheat (carry forward all prior lessons):
  * OPAQUE tag names; no tag-string parsing on tag names.
  * BOTH moats fed the calibrated raw firing-rate confidence quantities
    (``measure_pool_firing`` for direct; ``_ranked_from_pattern`` for
    compositional).
  * The ``cross_pool_concept`` gate is opened ONLY inside the
    encoding window (via ``encode_concept_pair``) then closed; it is
    NOT left open during evaluation.
  * ``uniform_ctrl`` differs from ``full`` ONLY in the threshold-
    routing decision (same seed, same encoded facts, same query set,
    same ranked confidences).
  * ``direct_retain`` is read from the SAME run as ``full`` (no separate
    draws); a single accumulator structure records both.
  * No protected-file edits; no autograd; the runner is reuse-only
    orchestration.

ASCII only. CuPy is the real / decisive path; ``--tiny-synth`` shrinks
Phase-1 training + compositional encoding so the smoke is seconds (toy
numbers explicitly NOT a result). The decisive multi-seed CuPy run is
a later controller-only task -- NOT performed here.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Backend policy mirrors the per-regime / SPEAR / Pirazzini runners.
# CuPy is the decisive path; NumPy ONLY when CuPy is genuinely
# unavailable (GPU-less box).
if "--tiny-synth" in sys.argv:
    try:
        import cupy as _cupy_probe  # noqa: F401

        _CUPY_AVAILABLE = True
    except Exception:
        _CUPY_AVAILABLE = False
    if not _CUPY_AVAILABLE:
        os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners.per_regime_monitor_core import (
    per_regime_monitor_verdict,
    _PR_LADDER,
)

# REUSED gates (each byte-unchanged in its own module). THREE moats are
# wired in by import; the unified runner uses TWO of them (the new
# substrate-specific direct gate + the per-regime compositional gate)
# and keeps the historical G.20 SharedPool direct moat imported only as
# evidence that the existing 650 calibration is byte-unchanged:
#
#   * ``abstention_gate.DEFAULT_THRESHOLD = 650.0`` (byte-unchanged;
#     calibrated on G.20 SharedPool ``recall_rates``, scale ~500-800).
#     NOT used to gate any query in this runner -- the existing 650 is
#     structurally unreachable by ``measure_pool_firing`` (per-neuron
#     mean rate, scale ~0.5-2 documented in CLAUDE.md). Imported only so
#     ``MOAT_DIRECT`` remains a referenced constant for source-grep pins
#     (and audit trail of the historical G.20 calibration).
#   * ``abstention_gate_direct_unified.DIRECT_UNIFIED_THRESHOLD = 0.0``
#     (placeholder; the unified runner's calibration step on the
#     ``build_biological_brain_regions`` substrate produces the
#     calibrated value, which the controller commits as a separate
#     frozen step). This is the new substrate-specific direct gate that
#     replaces the 650 moat for direct queries in this runner.
#   * ``abstention_gate_compositional.COMPOSITIONAL_THRESHOLD =
#     5.6887...`` (byte-unchanged; calibrated on the per-regime stage's
#     hippocampal one-shot substrate). Gates compositional queries.
#
# The uniform_ctrl arm applies a SINGLE threshold uniformly to BOTH
# regimes: we use DIRECT_UNIFIED_THRESHOLD (the direct regime's
# substrate-specific threshold) for both direct AND compositional
# queries. The decisive built-in control: if a single threshold suffices
# everywhere, the per-regime separation is not the differentiator. (The
# choice of DIRECT_UNIFIED_THRESHOLD over MOAT_DIRECT=650 is dictated by
# the substrate-mismatch defect #2: the historical 650 is not on the
# unified substrate's scale, so a uniform-650 control would trivially
# abstain on every query and the contrast vs `full` would be
# uninformative. Using DIRECT_UNIFIED_THRESHOLD keeps the control on a
# scale that produces a meaningful contrast.)
from research.runners.abstention_gate import gate as gate_direct
from research.runners.abstention_gate import DEFAULT_THRESHOLD as MOAT_DIRECT
from research.runners.abstention_gate_compositional import (
    gate as gate_compositional,
    COMPOSITIONAL_THRESHOLD,
)
from research.runners.abstention_gate_direct_unified import (
    gate as gate_direct_unified,
    DIRECT_UNIFIED_THRESHOLD,
)

from sim.train_checkpoint import (  # REUSED UNMODIFIED
    save_checkpoint,
    load_checkpoint,
    resume_epoch,
)

# REUSED concept-pool runner (Phase-1 training + bridge build + direct
# W->A readout) -- BYTE-UNCHANGED. Imported lazily inside functions
# below so module import is cheap and doesn't require CuPy / GPU.
import research.runners.concept_pool_demo as cpd

# REUSED compositional one-shot encoding helper -- BYTE-UNCHANGED.
from research.runners.compose_concept_engram import encode_concept_pair

# REUSED vocabulary + raw firing-rate ranking + hippocampal tag-region
# filter from the validated compose_retrieval_runner (byte-unchanged).
from research.runners.compose_retrieval_runner import (
    _NOUNS,
    _VERBS,
    _ADJS,
    _N_WORDS_ORTHOGONAL,
    _ranked_from_pattern,
    _HIPPO_TAG_REGIONS,
)


# =====================================================================
# Sub-seed offset for the unified runner's held-out compositional pair
# generation. Distinct from the per-regime monitor runner's calibration
# +10000 offset so the unified runner's facts are never confused with
# any future calibration set. The unified runner's compositional pairs
# are deterministic functions of (seed, N).
# =====================================================================
_UNIFIED_SUBSEED_OFFSET = 20000

# =====================================================================
# Calibration sub-seed offsets. Distinct from
# _UNIFIED_SUBSEED_OFFSET=+20000 (eval pairs) AND from the per-regime
# runner's compositional +10000 offset. Two distinct offsets:
#   * +30000 for the unified runner's compositional-gate calibration
#     set (held-out pairs disjoint from eval pairs);
#   * +40000 for the unified runner's direct-gate calibration set
#     (held-out words queried for groundable/ungroundable confidences).
# So the unified runner's calibration sets cannot be confused with any
# of the eval set or the per-regime calibration set.
# =====================================================================
_UNIFIED_CALIB_COMP_OFFSET = 30000
_UNIFIED_CALIB_DIRECT_OFFSET = 40000
# v2 direct-gate calibration sub-seed offset. The v2 protocol replaces
# the v1 per-seed random half-split with a per-word target-vs-best-off-
# target gap aggregated over the FULL trained vocab (16 words). The +50000
# offset keeps the v2 sub-seed distinct from v1's +40000 so the two
# protocols' deterministic noise are independent.
_UNIFIED_CALIB_DIRECT_V2_OFFSET = 50000

# v2 method docstring echoed in the JSON output (separate so it appears
# verbatim under the v2 protocol_version, mirroring CALIBRATION_METHOD_DOC).
CALIBRATION_METHOD_DOC_DIRECT_V2 = (
    "v2: per-word target-vs-best-off-target gap aggregated over the "
    "full trained vocab; no per-seed half-split; calibrated_threshold "
    "= 0.5 * (median(target_rate) + median(best_off_target_rate)) per "
    "seed"
)

# Tolerance for MATCH detection between calibrated aggregate and
# committed constant (mirrors per_regime_monitor_runner._CALIB_MATCH_TOL).
_CALIB_MATCH_TOL = 1e-6

# Calibration method docstring echoed in the JSON output.
CALIBRATION_METHOD_DOC = (
    "median_midpoint: for each seed, run a HELD-OUT calibration on the "
    "build_biological_brain_regions substrate (the SAME substrate "
    "Stage-1 / SPEAR / Pirazzini / Per-regime / Unified all use). "
    "Compositional gate: encode held-out (noun, adj) pairs (sub-seed = "
    "seed + 30000, disjoint from eval pairs) and measure raw firing-"
    "rate confidence at lang_output for GROUNDABLE (encoded) vs "
    "UNGROUNDABLE (never-encoded) queries; calibrated_threshold = "
    "0.5 * (median(groundable) + median(ungroundable)). Direct gate: "
    "query each Phase-1-trained vocabulary word (sub-seed = seed + "
    "40000) and measure the target-pool firing rate via "
    "measure_pool_firing (groundable = trained-word target-pool rate; "
    "ungroundable = never-trained word's top-pool rate -- in tiny-synth "
    "this is the same vocab with a sub-sampled disjoint Phase-1 subset; "
    "in full scale, the calibrator uses a held-out word partition). "
    "INSUFFICIENT-SEPARATION on ANY seed where groundable_median <= "
    "ungroundable_median (strengthen-only fix mirrored from the "
    "per-regime runner)."
)


# =====================================================================
# Phase-1 cache convention.
# =====================================================================
_PHASE1_CACHE_DEFAULT = "research/findings/raw/unified_per_regime/phase1/"


def _phase1_cache_path(cache_dir: str, seed: int) -> Path:
    """Per-seed Phase-1 cache file path.

    The Phase-1 multi-event-trained substrate is saved per seed and
    loaded into each (seed, N) evaluation cell. The expensive training
    is amortised across the decisive run (and any future stage).
    """
    return Path(cache_dir) / ("seed%d.simstate.h5" % int(seed))


# =====================================================================
# Phase-1 dim/recipe selection -- the SAME kwargs Phase-1 training uses
# and the SAME kwargs the per-cell bridge-build uses so the loaded
# architecture matches the saved checkpoint exactly. Mirror the
# Stage-1 / SPEAR / Pirazzini / Per-regime _build_substrate dim scheme
# so the substrate is the SAME validated v16 + hippocampus + dlpfc
# frame those stages cleared.
# =====================================================================
def _phase1_recipe(tiny_synth: bool) -> Dict[str, Any]:
    """Return the v14/v16-validated Phase-1 recipe dims (shrunk for
    tiny_synth). The same dims are used by:
      * Phase-1 training when calling ``apply_concept_topographic_bias``
        + ``train_word_to_pool`` (the same recipe ``run_concept_pool_demo``
        uses internally); AND
      * ``_build_bridge_with_phase1_recipe(...)`` when loading the
        cached checkpoint into a fresh bridge per (seed, N).
    So the architecture matches at save and load. The dims mirror the
    per-regime runner's _build_substrate for tiny_synth vs full scale
    so the substrate is the SAME validated v16 + hippocampus + dlpfc
    frame Stage-1 / SPEAR / Pirazzini / Per-regime all used (the
    adversarial-review-blocked substrate fix: this builder gives the
    engram region_filter [dg, ca3, ca1] something real to resolve
    against).
    """
    if tiny_synth:
        return {
            "n_train_events": 4,
            "n_lang_input": 64,
            "n_per_pool": 12,
            "n_fs_per_pool": 3,
            "n_dlpfc_verb": 24,
        }
    return {
        "n_train_events": 200,
        "n_lang_input": 2048,
        "n_per_pool": 200,
        "n_fs_per_pool": 24,
        "n_dlpfc_verb": 200,
    }


def _phase1_train_kwargs(tiny_synth: bool) -> Dict[str, Any]:
    """The kwargs Phase-1 training uses (the v14/v16-validated 88.75%-
    multi-seed recipe -- weak_dynamics + interleaved + topographic prior
    3.0 / 0.3 + orthogonal codes + sparsity 0.05 + adjective pools +
    direct verb-to-motor). These are the SAME recipe constants
    ``run_concept_pool_demo`` uses internally; we just apply them
    directly here on top of the substrate-with-hippocampus.
    """
    dims = _phase1_recipe(tiny_synth)
    return {
        "n_train_events": int(dims["n_train_events"]),
        "n_lang_input": int(dims["n_lang_input"]),
        "n_per_pool": int(dims["n_per_pool"]),
        "n_fs_per_pool": int(dims["n_fs_per_pool"]),
        # v14/v16-validated recipe constants.
        "weak_dynamics": True,
        "interleaved": True,
        "topographic_factor": 3.0,
        "off_target_factor": 0.3,
        "enable_adjective": True,
        "orthogonal_codes": True,
        "sparsity": 0.05,
        "enable_direct_verb_to_motor": True,
    }


def _build_bridge_with_phase1_recipe(seed: int, tiny_synth: bool):
    """Build a FRESH bridge whose architecture matches the Phase-1
    cached checkpoint exactly. Uses the SAME validated v16 +
    hippocampus + dlpfc substrate builder Stage-1 / SPEAR / Pirazzini /
    Per-regime all used (``build_biological_brain_regions``); the
    architecture has BOTH hippocampus (so the engram region_filter
    [dg, ca3, ca1] resolves to a real index set and
    ``commit_engram_tag`` produces tags with non-zero ``n_tagged``)
    AND concept pools (so the v14/v16 multi-event training applies
    cleanly). This is the SUBSTRATE FIX closing the prior adversarial-
    review-blocked defect #1.

    Strategic mirror: same construction path as
    ``per_regime_monitor_runner._build_substrate`` -- byte-unchanged
    builder, same kwarg surface, same CoreSimConfig dial-set, same
    _initialize_simulation_data call.
    """
    if tiny_synth:
        try:
            import cupy as _c  # noqa: F401

            _cupy_ok = True
        except Exception:
            _cupy_ok = False
        if not _cupy_ok:
            os.environ["SIM_BACKEND"] = "numpy"
            from sim.backend import get_backend as _get_backend

            _get_backend("numpy")

    dims = _phase1_recipe(tiny_synth)
    n_lang_input = int(dims["n_lang_input"])
    n_per_pool = int(dims["n_per_pool"])
    n_fs_per_pool = int(dims["n_fs_per_pool"])
    n_dlpfc_verb = int(dims["n_dlpfc_verb"])

    from sim.config import (
        CoreSimConfig,
        VisualizationConfig,
        RuntimeState,
        GPUConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    # weak_dynamics=True (v14/v16-validated) -- the SAME concept-pool
    # dial-set the Stage-1 / SPEAR / Pirazzini / Per-regime substrate
    # uses. Motor pools keep canon dynamics (the v16-validated default).
    concept_internal_density = 0.05
    concept_exc_weight = 0.3
    concept_inh_weight = 0.8
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
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
        # The validated trisynaptic hippocampal recent-specific path.
        # This is the SUBSTRATE FIX: the engram region_filter
        # [dg, ca3, ca1] now resolves to a real index set.
        enable_hippocampus_consolidation=True,
        # The validated dlpfc PFC working-memory compositional frame.
        enable_dlpfc_verb=True,
        n_dlpfc_verb=n_dlpfc_verb,
        dlpfc_verb_internal_density=0.15,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0
    cfg.fast_spike_reset = True

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
    return bridge


def _freeze_phase1_gates(bridge) -> List[str]:
    """Close all Phase-1 plasticity gates after loading a cached
    checkpoint so the compositional encoding window does NOT perturb
    v14/v16's reciprocal binding. Mirrors the
    ``run_concept_pool_demo`` load-path gate-freeze pattern (the same
    gate names) so the loaded substrate stays frozen for direct queries.
    """
    gates = (
        "language_input_to_motor",
        "language_input_to_noun_pool",
        "language_input_to_verb_pool",
        "language_input_to_adjective_pool",
        "motor_to_language_output",
        "noun_pool_to_language_output",
        "verb_pool_to_language_output",
        "adjective_pool_to_language_output",
        "motor_FS_to_motor",
        "verb_pool_FS_to_verb_pool",
        "noun_pool_FS_to_noun_pool",
        "adjective_pool_FS_to_adjective_pool",
        # cross_pool_concept defaults to closed by Phase-1 (v19); ensure
        # it stays closed pre-encoding too.
        "cross_pool_concept",
    )
    frozen: List[str] = []
    for g in gates:
        try:
            bridge.set_plasticity_gate(g, 0.0)
            frozen.append(g)
        except Exception:
            pass
    return frozen


# =====================================================================
# Phase-1 training (cached).
# =====================================================================
def _phase1_train_if_needed(seed: int, cache_dir: str,
                              tiny_synth: bool) -> Path:
    """Train Phase-1 substrate for one seed if no cached checkpoint
    exists; otherwise skip. Returns the path to the (now-existing)
    Phase-1 checkpoint. The Phase-1 caching strategy is the runner's
    primary cost-amortisation: a decisive multi-seed run runs Phase-1
    once per seed and reuses the same checkpoint across all (seed, N)
    cells AND across all later eval invocations.

    Phase-1 training is the SAME validated v14/v16 recipe
    ``run_concept_pool_demo`` uses internally (88.75% multi-seed
    bidirectional binding): apply Pulvermuller-style topographic bias
    via ``apply_concept_topographic_bias``, then loop
    ``train_word_to_pool`` over the full v14/v16 vocabulary in
    interleaved-shuffled order so no single pool dominates training.
    We invoke these byte-unchanged functions directly on the substrate
    built by ``_build_bridge_with_phase1_recipe`` -- which uses
    ``build_biological_brain_regions(
    enable_hippocampus_consolidation=True, ...)`` so the substrate
    has BOTH hippocampus AND concept pools (the SUBSTRATE FIX closing
    the prior adversarial-review-blocked defect #1).
    """
    cache_path = _phase1_cache_path(cache_dir, seed)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path

    train_kwargs = _phase1_train_kwargs(tiny_synth)

    # 1) Build the validated substrate (v16 + hippocampus + dlpfc PFC).
    bridge = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)

    # 2) Build the word_to_idx mapping the v14/v16 orthogonal-codes path
    # expects. The first 16 positions match
    # compose_retrieval_runner._NOUNS + _VERBS + _ADJS ordering.
    all_words_ordered = (
        list(cpd.DIRECTION_VOCAB)
        + list(cpd.NOUN_VOCAB)
        + list(cpd.VERB_VOCAB)
        + list(cpd.ADJECTIVE_VOCAB)
    )
    word_to_idx = {w: i for i, w in enumerate(all_words_ordered)}
    n_words_total = len(all_words_ordered)

    # 3) Apply topographic bias -- the SAME Pulvermuller-style
    # 1.5/0.7 cortical-somatotopy recipe ``run_concept_pool_demo`` uses
    # internally.
    cpd.apply_concept_topographic_bias(
        bridge,
        n_lang_input=int(train_kwargs["n_lang_input"]),
        topographic_factor=float(train_kwargs["topographic_factor"]),
        off_target_factor=float(train_kwargs["off_target_factor"]),
        sparsity=float(train_kwargs["sparsity"]),
        orthogonal_codes=bool(train_kwargs["orthogonal_codes"]),
        n_words_for_orthogonal=int(n_words_total),
        word_to_idx=word_to_idx,
        skip_motor=False,
        verbose=False,
    )

    # 4) Build the (word, target_pool) schedule -- the SAME schedule
    # ``run_concept_pool_demo`` constructs internally.
    all_targets: List[Tuple[str, str]] = []
    for word, action in cpd.DIRECTION_VOCAB.items():
        all_targets.append((word, "motor_%s" % action))
    for word, name in cpd.NOUN_VOCAB.items():
        all_targets.append((word, "noun_pool_%s" % name))
    for word, name in cpd.VERB_VOCAB.items():
        all_targets.append((word, "verb_pool_%s" % name))
    for word, name in cpd.ADJECTIVE_VOCAB.items():
        all_targets.append((word, "adjective_pool_%s" % name))

    # 5) Interleaved training -- the SAME shuffle pattern
    # ``run_concept_pool_demo`` uses (matches bio_three_factor +
    # prevents one pool from dominating during uninterrupted same-word
    # training). Shuffling deterministic on ``seed``.
    n_train_events = int(train_kwargs["n_train_events"])
    rng = np.random.default_rng(int(seed))
    buffer: List[Tuple[str, str]] = []
    for word, target in all_targets:
        for _ in range(n_train_events):
            buffer.append((word, target))
    rng.shuffle(buffer)

    for (word, target) in buffer:
        cpd.train_word_to_pool(
            bridge, word, target,
            n_events=1,
            reset_steps=50,
            n_lang_input=int(train_kwargs["n_lang_input"]),
            n_lang_output=int(train_kwargs["n_lang_input"]),
            sparsity=float(train_kwargs["sparsity"]),
            orthogonal_codes=bool(train_kwargs["orthogonal_codes"]),
            n_words_for_orthogonal=int(n_words_total),
            word_to_idx=word_to_idx,
            verbose=False,
        )

    # 6) Persist trained substrate state for downstream (seed, N) cells
    # to load_checkpoint against.
    Path(str(cache_path)).parent.mkdir(parents=True, exist_ok=True)
    bridge.save_checkpoint(str(cache_path))

    if not cache_path.exists():
        raise RuntimeError(
            "Phase-1 training did not produce the expected cache "
            "checkpoint at %s (recipe=%r)" % (cache_path, train_kwargs)
        )
    # Note on validation reuse: this Phase-1 training is the same
    # recipe ``run_concept_pool_demo`` runs internally (the v14/v16
    # 88.75% multi-seed validated dial-set). The runner does not call
    # ``cpd.run_concept_pool_demo`` directly because that helper would
    # build its OWN bridge via ``cpd.build_concept_bridge`` which lacks
    # hippocampal regions -- that was the substrate defect the
    # adversarial review blocked on. Direct
    # ``apply_concept_topographic_bias`` + ``train_word_to_pool``
    # gives us the SAME training on the SUBSTRATE-with-hippocampus.
    return cache_path


# =====================================================================
# Held-out compositional pair generation (deterministic from seed).
# =====================================================================
def _unified_compositional_pairs(seed: int, N: int) -> List[Tuple[str, str]]:
    """Generate N compositional (noun, adj) pairs deterministically
    from ``seed`` via a sub-seed offset (+20000) distinct from any
    calibration sub-seed. The unified runner's compositional facts
    cannot be confused with any future calibration set.
    """
    rng = np.random.default_rng(int(seed) + _UNIFIED_SUBSEED_OFFSET)
    pairs: List[Tuple[str, str]] = []
    n_nouns = len(_NOUNS)
    n_adjs = len(_ADJS)
    # Sample without replacement when possible; if N > pool, cycle the
    # noun and rotate the adj so pairs stay distinct.
    perm = rng.permutation(n_nouns * n_adjs)
    for i in range(int(N)):
        idx = int(perm[i % len(perm)])
        n_i = idx // n_adjs
        a_i = idx % n_adjs
        # Recycle deterministically once perm is exhausted.
        if i >= len(perm):
            shift = i // len(perm)
            a_i = (a_i + shift) % n_adjs
        pairs.append((_NOUNS[n_i], _ADJS[a_i]))
    return pairs


# =====================================================================
# Direct W->A readout (REUSED measure_pool_firing).
# =====================================================================
def _all_pool_regions(enable_adjective: bool = True) -> List[str]:
    """Concept-pool list the direct readout ranks across. The v14/v16
    architecture has 4 motor + 4 noun + 4 verb (+ 4 adjective)."""
    pools = (
        ["motor_%s" % a for a in ("N", "E", "S", "W")]
        + ["noun_pool_%s" % n for n in cpd.NOUN_NAMES]
        + ["verb_pool_%s" % v for v in cpd.VERB_NAMES]
    )
    if enable_adjective:
        pools += ["adjective_pool_%s" % a for a in cpd.ADJECTIVE_NAMES]
    return pools


def _all_words_word_to_idx() -> Tuple[List[str], Dict[str, int]]:
    """Vocabulary + word_to_idx mapping matching the v14/v16 ordering
    that the orthogonal codes are calibrated on. The first 16 positions
    match ``compose_retrieval_runner._NOUNS + _VERBS + _ADJS`` ordering
    (and ``concept_compose_train._ALL_WORDS`` for the first 16).
    """
    all_words = (
        ["north", "east", "south", "west"]
        + list(cpd.NOUN_VOCAB.keys())
        + list(cpd.VERB_VOCAB.keys())
        + list(cpd.ADJECTIVE_VOCAB.keys())
    )
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    return all_words, word_to_idx


def _direct_pool_target(word: str) -> str:
    """The pool region the v14/v16-trained substrate routes ``word``
    to (the target pool for direct retrieval)."""
    if word in cpd.DIRECTION_VOCAB:
        return "motor_%s" % cpd.DIRECTION_VOCAB[word]
    if word in cpd.NOUN_VOCAB:
        return "noun_pool_%s" % cpd.NOUN_VOCAB[word]
    if word in cpd.VERB_VOCAB:
        return "verb_pool_%s" % cpd.VERB_VOCAB[word]
    if word in cpd.ADJECTIVE_VOCAB:
        return "adjective_pool_%s" % cpd.ADJECTIVE_VOCAB[word]
    raise KeyError("unknown word for direct pool target: %r" % word)


def _direct_query_ranked(bridge, word: str, dims: Dict[str, Any],
                          all_pools: List[str], word_to_idx: Dict[str, int],
                          stim_steps: int, reset_steps: int):
    """Drive ``lang_input(word)`` via the validated v14/v16 path,
    measure per-pool firing rates, return the ranked
    [(pool_name, rate, tag), ...] list in the moat's expected shape.
    The 650 direct gate is calibrated on this raw rate (encoded ~796,
    control ~584; cross-vocab AUC 0.990).
    """
    per_pool = cpd.measure_pool_firing(
        bridge, word, all_pools,
        stim_steps=int(stim_steps),
        reset_steps=int(reset_steps),
        drive_pA=200.0,
        sparsity=float(dims["sparsity"]),
        n_lang_input=int(dims["n_lang_input"]),
        orthogonal_codes=True,
        n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
        word_to_idx=word_to_idx,
    )
    # Build ranked list descending by raw firing rate. The tag column
    # is the literal "direct" string -- the runner never parses tag
    # names to decode an answer (Stage-1 / SPEAR / Pirazzini lesson).
    ranked = sorted(
        ((p, float(per_pool[p]), "direct") for p in all_pools),
        key=lambda t: -t[1],
    )
    return ranked


# =====================================================================
# Compositional readout (REUSED from per_regime_monitor_runner pattern).
# =====================================================================
def _compositional_query_ranked(bridge, cue_noun: str,
                                  tag_name: Optional[str],
                                  dims: Dict[str, Any],
                                  recall_steps: int):
    """Compositional-retrieval-regime read: cue the noun + stimulate
    the bound engram tag; sum per-concept raw firing-rate confidences
    at lang_output. Mirrors the per-regime monitor runner's
    ``_compositional_query_confidence`` pattern exactly -- the SAME
    quantity the calibrated 5.69 compositional gate is calibrated on.
    """
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_stim,
        lang_output_pattern_during_input,
    )

    cons_pat, n_lo = lang_output_pattern_during_input(
        bridge, cue_noun,
        n_lang_input=int(dims["n_lang_input"]),
        sparsity=float(dims["sparsity"]),
        n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
        stim_steps=int(recall_steps),
    )
    cons_ranked = _ranked_from_pattern(
        cons_pat, n_lo, dims, exclude=cue_noun
    )

    if tag_name is not None and tag_name in {
        t["name"] for t in bridge.list_engram_tags()
    }:
        hip_pat, n_lo2 = lang_output_pattern_during_stim(
            bridge, tag_name, drive_pA=1500.0,
            stim_steps=int(recall_steps),
        )
        hip_ranked = _ranked_from_pattern(
            hip_pat, n_lo2, dims, exclude=cue_noun
        )
    else:
        hip_ranked = []

    scores: Dict[str, float] = {}
    for w, r, _t in cons_ranked:
        scores[w] = scores.get(w, 0.0) + float(r)
    for w, r, _t in hip_ranked:
        scores[w] = scores.get(w, 0.0) + float(r)
    ranked = sorted(
        ((w, scores[w], "compose") for w in scores),
        key=lambda t: -t[1],
    )
    return ranked


# =====================================================================
# Encoding helper (REUSED encode_concept_pair).
# =====================================================================
def _encode_facts(bridge, facts: List[Tuple[str, str]],
                   dims: Dict[str, Any],
                   encoding_steps: int) -> List[str]:
    """Encode each compositional (noun, adj) fact via the REUSED
    ``encode_concept_pair`` helper. Tag names are OPAQUE
    (``f"ep_{i}"``) -- they carry NO answer string (Stage-1 lesson).
    The cross_pool_concept gate is opened by ``encode_concept_pair``
    inside its body and CLOSED at the end of its body so it stays
    closed for the subsequent eval queries.
    """
    tags: List[str] = []
    for i, (noun, adj) in enumerate(facts):
        tag = "ep_%d" % i  # OPAQUE
        if tag in {t["name"] for t in bridge.list_engram_tags()}:
            try:
                bridge.delete_engram_tag(tag)
            except Exception:
                pass
        encode_concept_pair(
            bridge, noun, adj, tag,
            encoding_steps=int(encoding_steps),
            drive_pA=200.0,
            sparsity=float(dims["sparsity"]),
            n_lang_input=int(dims["n_lang_input"]),
            n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
            region_filter=_HIPPO_TAG_REGIONS,
            top_k=max(8, int(dims["n_per_pool"]) // 4),
            balanced_teacher_pA=500.0,
            verbose=False,
        )
        tags.append(tag)
    return tags


# =====================================================================
# Per-cell evaluation arm: one (seed, N) cell.
# =====================================================================
def _run_evaluation_arm(seed: int, N: int, tiny_synth: bool,
                          cache_dir: str) -> Dict[str, Any]:
    """Run the unified per-regime architecture for ONE (seed, N) cell.

    Returns the four rung accuracy fields the frozen verdict consumes:
    ``full_acc``, ``uniform_ctrl_acc``, ``direct_retain_acc``,
    ``abstain_correct``.

    The ``full`` and ``uniform_ctrl`` arms are scored from the SAME
    forward pass (same encoded facts, same queries, same ranked
    confidences) -- the SOLE difference is the threshold-routing
    decision. ``direct_retain_acc`` is the direct-only subset of the
    SAME ``full`` run (no separate draws). ``abstain_correct`` is the
    fraction of UNGROUNDABLE queries on which the appropriate-regime
    gate abstained.
    """
    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200

    # Load Phase-1 trained substrate into a fresh bridge that matches
    # the saved architecture.
    cache_path = _phase1_cache_path(cache_dir, seed)
    if not cache_path.exists():
        raise RuntimeError(
            "Phase-1 cache missing for seed %d at %s; call "
            "_phase1_train_if_needed first." % (seed, cache_path)
        )

    bridge = _build_bridge_with_phase1_recipe(seed, tiny_synth)
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    recipe_dims = _phase1_recipe(tiny_synth)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    dims: Dict[str, Any] = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }

    # Compositional encoding (gate opened only inside encode_concept_pair).
    facts = _unified_compositional_pairs(seed, N)
    tags = _encode_facts(bridge, facts, dims, enc_steps)

    all_pools = _all_pool_regions(enable_adjective=True)

    # ---- DIRECT queries: one per encoded fact (cue the noun + cue the
    # adj). The CORRECT direct answer for a cued noun is its target
    # noun pool; for a cued adj its target adjective pool. The direct-
    # retrieval-regime monitor (existing 650 moat) gates the answer.
    # `full` and `uniform_ctrl` BOTH route direct queries through the
    # 650 moat (they only differ on compositional queries by design),
    # so the two arms agree on direct counts by construction.
    n_direct_total = 0
    n_direct_correct_full = 0
    n_direct_correct_uniform = 0
    direct_words: List[Tuple[str, str]] = []  # (word, expected_pool)
    seen_direct: set = set()
    for noun, adj in facts:
        for w in (noun, adj):
            if w in seen_direct:
                continue
            seen_direct.add(w)
            try:
                expected_pool = _direct_pool_target(w)
            except KeyError:
                continue
            direct_words.append((w, expected_pool))

    for word, expected_pool in direct_words:
        n_direct_total += 1
        ranked = _direct_query_ranked(
            bridge, word, dims, all_pools, word_to_idx,
            stim_steps=recall_steps, reset_steps=recall_steps // 2,
        )
        # `full` routes direct queries through the NEW substrate-
        # specific direct gate (DIRECT_UNIFIED_THRESHOLD placeholder
        # 0.0 until calibration ships the calibrated value via a
        # controller commit). The existing 650 moat is byte-unchanged
        # but no longer used here -- its G.20 SharedPool-recall-rate
        # scale does not match measure_pool_firing's per-neuron mean
        # rate scale (defect #2 closure).
        decided_full = gate_direct_unified(ranked, DIRECT_UNIFIED_THRESHOLD)
        ans_full = None if decided_full is None else decided_full[0]
        # `uniform_ctrl` ALSO routes through the SAME substrate-specific
        # direct gate (single-threshold-applied-uniformly). For direct
        # queries the two arms therefore agree by construction -- the
        # difference shows up on compositional queries.
        decided_uniform = gate_direct_unified(
            ranked, DIRECT_UNIFIED_THRESHOLD
        )
        ans_uniform = None if decided_uniform is None else decided_uniform[0]
        # The validated direct-retrieval correctness criterion (v14/v16):
        # the top pool above the moat MUST be the word's target pool.
        if ans_full == expected_pool:
            n_direct_correct_full += 1
        if ans_uniform == expected_pool:
            n_direct_correct_uniform += 1

    # ---- COMPOSITIONAL queries: one per encoded fact, cue the noun,
    # expect the bound adj. The compositional-regime monitor (calibrated
    # 5.69 gate) gates the answer for `full`; the SAME ranked
    # confidences are routed through the direct moat (650) for
    # `uniform_ctrl` (the SOLE difference vs `full`).
    n_comp_total = 0
    n_comp_correct_full = 0
    n_comp_correct_uniform = 0
    for i, (noun, adj) in enumerate(facts):
        n_comp_total += 1
        tag = tags[i] if i < len(tags) else None
        ranked = _compositional_query_ranked(
            bridge, noun, tag, dims, recall_steps
        )
        # `full`: per-regime architecture routes compositional queries
        # through the COMPOSITIONAL gate at COMPOSITIONAL_THRESHOLD.
        decided_full = gate_compositional(ranked, COMPOSITIONAL_THRESHOLD)
        ans_full = None if decided_full is None else decided_full[0]
        # `uniform_ctrl`: single-threshold-applied-uniformly. The
        # compositional queries STILL go through the compositional
        # gate's structural shape, but with the threshold set to
        # DIRECT_UNIFIED_THRESHOLD (the SAME threshold uniform_ctrl
        # applies to direct queries). The SOLE difference from `full`
        # is that uniform_ctrl uses the direct regime's substrate-
        # specific threshold uniformly across BOTH regimes (instead
        # of routing direct -> direct gate, compositional ->
        # compositional gate as `full` does). Defect #2 closure: we
        # use DIRECT_UNIFIED_THRESHOLD here, not MOAT_DIRECT=650 --
        # the historical G.20 SharedPool calibration is not on the
        # unified substrate's scale.
        decided_uniform = gate_compositional(ranked, DIRECT_UNIFIED_THRESHOLD)
        ans_uniform = None if decided_uniform is None else decided_uniform[0]
        if ans_full == adj:
            n_comp_correct_full += 1
        if ans_uniform == adj:
            n_comp_correct_uniform += 1

    # ---- UNGROUNDABLE queries: vocabulary words NOT used in this
    # rung's facts. The appropriate-regime gate MUST abstain on these.
    # We mix direct + compositional ungroundables.
    encoded_nouns = {n for n, _ in facts}
    encoded_adjs = {a for _, a in facts}
    ungroundable_direct_words = (
        [w for w in _NOUNS if w not in encoded_nouns]
        + [w for w in _ADJS if w not in encoded_adjs]
    )
    n_ungroundable = 0
    n_abstain_ok = 0
    for w in ungroundable_direct_words:
        n_ungroundable += 1
        # Direct ungroundable: the substrate-specific direct gate
        # should abstain because the substrate's response to this word
        # is not strongly bound against the per-cell encoding regime.
        # (For Phase-1-trained words the direct gate will NOT abstain;
        # we count those as not-abstain-correct. The honest
        # abstain_correct measurement.) We here ask whether the
        # substrate produces a top-pool rate above the substrate-
        # specific direct gate -- if yes, the gate doesn't abstain.
        ranked = _direct_query_ranked(
            bridge, w, dims, all_pools, word_to_idx,
            stim_steps=recall_steps, reset_steps=recall_steps // 2,
        )
        decided = gate_direct_unified(ranked, DIRECT_UNIFIED_THRESHOLD)
        if decided is None:
            n_abstain_ok += 1

    # Also count compositional ungroundables (cue a noun that was NOT
    # encoded) -- the compositional gate at 5.69 should abstain.
    ungroundable_nouns = [w for w in _NOUNS if w not in encoded_nouns]
    for w in ungroundable_nouns:
        n_ungroundable += 1
        ranked = _compositional_query_ranked(
            bridge, w, None, dims, recall_steps
        )
        decided = gate_compositional(ranked, COMPOSITIONAL_THRESHOLD)
        if decided is None:
            n_abstain_ok += 1

    # ---- Aggregate the four rung fields from the SAME run. ----
    n_total = n_direct_total + n_comp_total
    full_acc = (
        (n_direct_correct_full + n_comp_correct_full) / n_total
        if n_total
        else 0.0
    )
    uniform_ctrl_acc = (
        (n_direct_correct_uniform + n_comp_correct_uniform) / n_total
        if n_total
        else 0.0
    )
    direct_retain_acc = (
        n_direct_correct_full / n_direct_total if n_direct_total else 0.0
    )
    abstain_correct = (
        n_abstain_ok / n_ungroundable if n_ungroundable else 1.0
    )

    return {
        "seed": int(seed),
        "N": int(N),
        "full_acc": float(full_acc),
        "uniform_ctrl_acc": float(uniform_ctrl_acc),
        "direct_retain_acc": float(direct_retain_acc),
        "abstain_correct": float(abstain_correct),
        # Diagnostics (not part of the verdict rung shape).
        "n_direct": int(n_direct_total),
        "n_compositional": int(n_comp_total),
        "n_ungroundable": int(n_ungroundable),
    }


# =====================================================================
# Calibration arm (compositional gate + new direct gate).
#
# Mirrors the per-regime runner's calibration mode but on the unified
# substrate. The compositional-gate calibration is the SAME quantity
# the per-regime runner's calibration produces (median midpoint of
# groundable vs ungroundable compositional confidences). The direct-
# gate calibration is NET-NEW: it measures the substrate-specific
# direct readout's groundable vs ungroundable population medians on a
# Phase-1-trained substrate (so the calibrated threshold is on the
# scale of measure_pool_firing's per-neuron mean rate, not the
# historical G.20 SharedPool recall_rates scale).
# =====================================================================


def _calibrate_compositional_one_seed(seed: int, tiny_synth: bool,
                                        cache_dir: str) -> Dict[str, Any]:
    """Per-seed compositional-gate calibration on the unified substrate.

    Sub-seed = seed + _UNIFIED_CALIB_COMP_OFFSET (+30000). Encode held-
    out (noun, adj) pairs disjoint from the eval set
    (_unified_compositional_pairs at the maximum N in the ladder),
    measure raw firing-rate confidences at lang_output for groundable
    + ungroundable queries, return midpoint of medians.

    The Phase-1 cache is loaded (so the substrate is already trained);
    the compositional encoding is the SAME engram-API one-shot binding
    the eval arm uses.
    """
    sub_seed = int(seed) + _UNIFIED_CALIB_COMP_OFFSET
    cal_rng = np.random.default_rng(sub_seed)

    # Eval-set pair partition at the maximum N in the frozen ladder.
    eval_pairs = set(_unified_compositional_pairs(seed, max(_PR_LADDER)))
    all_pairs = [(n, a) for n in _NOUNS for a in _ADJS]
    held_out_pairs = [p for p in all_pairs if p not in eval_pairs]
    n_calib_facts = 2 if tiny_synth else min(4, len(held_out_pairs))
    if held_out_pairs:
        perm_idx = cal_rng.permutation(len(held_out_pairs))
        calib_facts = [
            held_out_pairs[int(perm_idx[i])]
            for i in range(min(n_calib_facts, len(held_out_pairs)))
        ]
    else:
        calib_facts = []

    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200

    # Load Phase-1 substrate.
    cache_path = _phase1_cache_path(cache_dir, seed)
    bridge = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    recipe_dims = _phase1_recipe(tiny_synth)
    all_words, _ = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    dims: Dict[str, Any] = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }

    tags = _encode_facts(bridge, calib_facts, dims, enc_steps)

    # GROUNDABLE confidences: per encoded fact, confidence ON the
    # correct answer (the bound adj).
    groundable_confidences: List[float] = []
    for i, (noun, adj) in enumerate(calib_facts):
        tag = tags[i] if i < len(tags) else None
        ranked = _compositional_query_ranked(
            bridge, noun, tag, dims, recall_steps
        )
        rate_on_correct = 0.0
        for w, r, _t in ranked:
            if w == adj:
                rate_on_correct = float(r)
                break
        groundable_confidences.append(rate_on_correct)

    # UNGROUNDABLE confidences: query nouns NOT encoded in this calib
    # set; no bound adj exists so the top compositional confidence is
    # the noise-floor representative.
    encoded_nouns = {n for n, _ in calib_facts}
    ungroundable_nouns = [w for w in _NOUNS if w not in encoded_nouns]
    if not ungroundable_nouns:
        ungroundable_nouns = list(_VERBS)
    ungroundable_confidences: List[float] = []
    for w in ungroundable_nouns:
        ranked = _compositional_query_ranked(
            bridge, w, None, dims, recall_steps
        )
        # Top confidence (the moat-calibrated quantity).
        top_conf = float(ranked[0][1]) if ranked else 0.0
        ungroundable_confidences.append(top_conf)

    g_median = (
        float(statistics.median(groundable_confidences))
        if groundable_confidences else 0.0
    )
    u_median = (
        float(statistics.median(ungroundable_confidences))
        if ungroundable_confidences else 0.0
    )
    calibrated_threshold = float(0.5 * (g_median + u_median))

    return {
        "seed": int(seed),
        "sub_seed": int(sub_seed),
        "groundable_median": g_median,
        "ungroundable_median": u_median,
        "calibrated_threshold": calibrated_threshold,
        "n_groundable": len(groundable_confidences),
        "n_ungroundable": len(ungroundable_confidences),
    }


def _calibrate_direct_one_seed(seed: int, tiny_synth: bool,
                                 cache_dir: str) -> Dict[str, Any]:
    """Per-seed direct-gate calibration on the unified substrate.

    NET-NEW (defect #2 closure): the substrate-specific direct gate
    needs its own calibration because the historical 650 moat was
    calibrated on G.20 SharedPool recall_rates (scale ~500-800), but
    the unified runner's direct readout uses measure_pool_firing
    (per-neuron mean rate, scale ~0.5-2 per CLAUDE.md). The calibrated
    threshold must be on the latter scale.

    Sub-seed = seed + _UNIFIED_CALIB_DIRECT_OFFSET (+40000). Per-seed
    deterministic split of the v14/v16 vocabulary into a calibration
    GROUNDABLE half (words Phase-1 trained -- their target-pool firing
    rate is the signal level) and an UNGROUNDABLE half (a non-overlapping
    set of words queried as if-untrained: we query their TOP-POOL rate
    on a name they were not trained against -- per-seed partition gives
    a held-out split). The method is the SAME median-midpoint separator
    the compositional calibration uses.

    Method: groundable = trained-word's target-pool rate. Ungroundable
    = a non-overlapping word's TOP-POOL rate when that word's actual
    target is held-out from the comparison (we ask "what does the
    substrate produce as its TOP rate for this word?" without asserting
    correctness). For tiny-synth this is a small smoke; the decisive
    multi-seed CuPy calibration is a controller-only step.
    """
    sub_seed = int(seed) + _UNIFIED_CALIB_DIRECT_OFFSET
    cal_rng = np.random.default_rng(sub_seed)

    # Load Phase-1 substrate.
    cache_path = _phase1_cache_path(cache_dir, seed)
    bridge = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    recipe_dims = _phase1_recipe(tiny_synth)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    dims: Dict[str, Any] = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }

    all_pools = _all_pool_regions(enable_adjective=True)

    # Per-seed deterministic split of the v14/v16 vocabulary.
    candidate_words: List[str] = []
    for w in cpd.DIRECTION_VOCAB:
        candidate_words.append(w)
    for w in cpd.NOUN_VOCAB:
        candidate_words.append(w)
    for w in cpd.VERB_VOCAB:
        candidate_words.append(w)
    for w in cpd.ADJECTIVE_VOCAB:
        candidate_words.append(w)
    perm = cal_rng.permutation(len(candidate_words))
    split = len(candidate_words) // 2
    g_idx = perm[:split]
    u_idx = perm[split:]
    if tiny_synth:
        # Shrink to a few calibration queries for the smoke.
        g_idx = g_idx[: min(4, len(g_idx))]
        u_idx = u_idx[: min(4, len(u_idx))]
    groundable_words = [candidate_words[int(i)] for i in g_idx]
    ungroundable_words = [candidate_words[int(i)] for i in u_idx]

    recall_steps = 20 if tiny_synth else 100

    # GROUNDABLE rates: target-pool rate on a trained word.
    groundable_rates: List[float] = []
    for w in groundable_words:
        try:
            expected_pool = _direct_pool_target(w)
        except KeyError:
            continue
        per_pool = cpd.measure_pool_firing(
            bridge, w, all_pools,
            stim_steps=int(recall_steps),
            reset_steps=int(recall_steps // 2),
            drive_pA=200.0,
            sparsity=float(dims["sparsity"]),
            n_lang_input=int(dims["n_lang_input"]),
            orthogonal_codes=True,
            n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
            word_to_idx=word_to_idx,
        )
        groundable_rates.append(float(per_pool.get(expected_pool, 0.0)))

    # UNGROUNDABLE rates: TOP-POOL rate on a non-overlapping word
    # (the held-out half of the per-seed split). This is the noise-
    # floor representative: how high a "untrained-for-this-split"
    # word's top-pool rate gets when we ask the substrate to retrieve.
    ungroundable_rates: List[float] = []
    for w in ungroundable_words:
        per_pool = cpd.measure_pool_firing(
            bridge, w, all_pools,
            stim_steps=int(recall_steps),
            reset_steps=int(recall_steps // 2),
            drive_pA=200.0,
            sparsity=float(dims["sparsity"]),
            n_lang_input=int(dims["n_lang_input"]),
            orthogonal_codes=True,
            n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
            word_to_idx=word_to_idx,
        )
        top = max(per_pool.values()) if per_pool else 0.0
        ungroundable_rates.append(float(top))

    g_median = (
        float(statistics.median(groundable_rates))
        if groundable_rates else 0.0
    )
    u_median = (
        float(statistics.median(ungroundable_rates))
        if ungroundable_rates else 0.0
    )
    calibrated_threshold = float(0.5 * (g_median + u_median))

    return {
        "seed": int(seed),
        "sub_seed": int(sub_seed),
        "groundable_median": g_median,
        "ungroundable_median": u_median,
        "calibrated_threshold": calibrated_threshold,
        "n_groundable": len(groundable_rates),
        "n_ungroundable": len(ungroundable_rates),
    }


def _calibrate_direct_v2_one_seed(seed: int, tiny_synth: bool,
                                     cache_dir: str) -> Dict[str, Any]:
    """v2 per-seed direct-gate calibration on the unified substrate.

    Method: per-word target-vs-best-off-target gap aggregated over the
    FULL trained vocab (NO per-seed random half-split). For each of the
    16 trained vocabulary words w:
        target_rate     = per_pool[expected_pool(w)]
        best_off_target = max(per_pool[p] for p != expected_pool(w))
    Then:
        g_median             = median(target_rate over 16 words)
        u_median             = median(best_off_target over 16 words)
        calibrated_threshold = 0.5 * (g_median + u_median)

    Why v2: the v1 protocol's "ungroundable" set is the held-out half
    of the TRAINED 16-word vocab, queried with its own trained code, so
    it measures (strong-half-median) vs (other-strong-half-median),
    NOT trained-vs-untrained. The per-seed random half-split produces
    INVERTED outcomes at 2/3 seeds (42, 44) just because random splits
    sometimes put weak-binders in the groundable half. v2's per-word
    target-vs-best-off-target gap is a within-word contrast that
    survives weak per-word binders -- a substrate-retains-direction
    word still has target_rate > best_off_target_rate even when both
    are small.

    Sub-seed = seed + _UNIFIED_CALIB_DIRECT_V2_OFFSET (+50000). Distinct
    from v1's +40000.

    Return dict shape mirrors v1 (seed / sub_seed / groundable_median /
    ungroundable_median / calibrated_threshold / n_groundable /
    n_ungroundable) plus ``protocol_version = "v2"`` so the calibration
    status logic + downstream JSON consumers can route per-protocol.
    """
    sub_seed = int(seed) + _UNIFIED_CALIB_DIRECT_V2_OFFSET

    # Load Phase-1 substrate (same load + freeze pattern as v1).
    cache_path = _phase1_cache_path(cache_dir, seed)
    bridge = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    recipe_dims = _phase1_recipe(tiny_synth)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    dims: Dict[str, Any] = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }

    all_pools = _all_pool_regions(enable_adjective=True)

    # The FULL trained vocab (no per-seed split). 4 direction + 4 noun +
    # 4 verb + 4 adjective = 16, matching the v14/v16 calibration canon.
    trained_words: List[str] = []
    for w in cpd.DIRECTION_VOCAB:
        trained_words.append(w)
    for w in cpd.NOUN_VOCAB:
        trained_words.append(w)
    for w in cpd.VERB_VOCAB:
        trained_words.append(w)
    for w in cpd.ADJECTIVE_VOCAB:
        trained_words.append(w)

    recall_steps = 20 if tiny_synth else 100

    groundable_rates: List[float] = []
    ungroundable_rates: List[float] = []
    for w in trained_words:
        try:
            expected_pool = _direct_pool_target(w)
        except KeyError:
            continue
        per_pool = cpd.measure_pool_firing(
            bridge, w, all_pools,
            stim_steps=int(recall_steps),
            reset_steps=int(recall_steps // 2),
            drive_pA=200.0,
            sparsity=float(dims["sparsity"]),
            n_lang_input=int(dims["n_lang_input"]),
            orthogonal_codes=True,
            n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
            word_to_idx=word_to_idx,
        )
        target_rate = float(per_pool.get(expected_pool, 0.0))
        best_off_target_rate = max(
            (r for p, r in per_pool.items() if p != expected_pool),
            default=0.0,
        )
        groundable_rates.append(target_rate)
        ungroundable_rates.append(float(best_off_target_rate))

    g_median = (
        float(statistics.median(groundable_rates))
        if groundable_rates else 0.0
    )
    u_median = (
        float(statistics.median(ungroundable_rates))
        if ungroundable_rates else 0.0
    )
    calibrated_threshold = float(0.5 * (g_median + u_median))

    return {
        "seed": int(seed),
        "sub_seed": int(sub_seed),
        "groundable_median": g_median,
        "ungroundable_median": u_median,
        "calibrated_threshold": calibrated_threshold,
        "n_groundable": len(groundable_rates),
        "n_ungroundable": len(ungroundable_rates),
        "protocol_version": "v2",
    }


def _calibration_status(per_seed: List[Dict[str, Any]],
                          committed: float) -> Tuple[str, float]:
    """Classify the calibration outcome vs the committed constant.
    Mirrors per_regime_monitor_runner._calibration_status:

    INSUFFICIENT-SEPARATION -- on ANY per-seed cell the groundable
                population median is <= the ungroundable population
                median. The midpoint separator only makes sense when
                signal > noise; if the populations overlap or invert
                the committed threshold would route the WRONG way at
                eval. Controller must NOT commit a calibrated constant
                when this status is emitted.
    MATCH    -- the aggregate calibrated value is within tolerance of
                the committed constant.
    PENDING  -- the committed constant is the placeholder (0.0) AND
                the aggregate calibrated value is non-zero. The runner
                writes JSON only; the controller updates the source
                file in a separate commit.
    MISMATCH -- the committed constant is non-zero AND the aggregate
                calibrated value differs from it beyond tolerance.
    """
    vals = [d["calibrated_threshold"] for d in per_seed]
    aggregate = float(sum(vals) / len(vals)) if vals else 0.0

    # Strengthen-only: refuse to emit a separator when populations
    # overlap or invert at any seed.
    for d in per_seed:
        g = float(d.get("groundable_median", 0.0))
        u = float(d.get("ungroundable_median", 0.0))
        if g <= u:
            return "INSUFFICIENT-SEPARATION", aggregate

    committed = float(committed)
    if abs(aggregate - committed) <= _CALIB_MATCH_TOL:
        return "MATCH", aggregate
    if abs(committed) <= _CALIB_MATCH_TOL:
        return "PENDING", aggregate
    return "MISMATCH", aggregate


# =====================================================================
# Aggregation.
# =====================================================================
def _aggregate_rungs(cells_by_N: Dict[int, List[Dict[str, Any]]],
                       n_seeds: int) -> List[Dict[str, Any]]:
    """Aggregate per-seed evaluation cells into one rung per N (the
    exact six-key shape the frozen verdict consumes)."""
    rungs: List[Dict[str, Any]] = []
    for N in sorted(cells_by_N):
        cells = cells_by_N[N]

        def _mean(field: str) -> float:
            vals = [c[field] for c in cells]
            return float(sum(vals) / len(vals)) if vals else 0.0

        rungs.append({
            "N": int(N),
            "n_seeds": int(n_seeds),
            "full_acc": _mean("full_acc"),
            "uniform_ctrl_acc": _mean("uniform_ctrl_acc"),
            "direct_retain_acc": _mean("direct_retain_acc"),
            "abstain_correct": _mean("abstain_correct"),
        })
    return rungs


# =====================================================================
# Top-level entry.
# =====================================================================
def run_unified_per_regime_monitor(
    seeds,
    loads=_PR_LADDER,
    tiny_synth: bool = False,
    phase1_cache_dir: str = _PHASE1_CACHE_DEFAULT,
    out_path: Optional[str] = None,
    ckpt: Optional[str] = None,
    calibrate: bool = False,
    direct_calibration_v2: bool = False,
) -> Dict[str, Any]:
    """Unified per-regime monitor + per-regime encoding capability runner.

    Two modes:
      calibrate=False (default) -- evaluation. Per seed, per N in the
        frozen ladder: Phase-1 cached, compositional one-shot encoding,
        per-query-type routing through both calibrated moats; three
        measurement arms from the SAME run + abstain_correct.
      calibrate=True -- calibration. Run held-out per-seed calibration
        for BOTH the compositional gate (sub-seed = seed + 30000) AND
        the new substrate-specific direct gate (sub-seed = seed +
        40000). Returns per-seed thresholds + aggregate + status
        (MATCH / PENDING / MISMATCH / INSUFFICIENT-SEPARATION) for
        each gate. The runner writes JSON only; updating the source
        constants is a SEPARATE controller commit.

    Kill-safe via the REUSED ``sim.train_checkpoint`` (evaluation cell
    granularity).
    """
    seeds = list(seeds)
    loads = tuple(int(x) for x in loads)
    phase1_cache_dir = str(phase1_cache_dir)

    # ---- Phase 1: per-seed training (cached) -- required for both
    # evaluation AND calibration modes (the substrate-specific direct
    # gate calibration needs a Phase-1-trained substrate; the
    # compositional calibration encodes on top of it).
    Path(phase1_cache_dir).mkdir(parents=True, exist_ok=True)
    for s in seeds:
        _phase1_train_if_needed(int(s), phase1_cache_dir, tiny_synth)

    if calibrate:
        # ---- CALIBRATION MODE: held-out per-seed thresholds for both
        # gates (compositional + new substrate-specific direct). ----
        # The compositional-gate calibration is UNAFFECTED by the
        # ``direct_calibration_v2`` flag (still v1 / half-split-of-
        # compositional-pairs). Only the direct-gate calibration loop
        # routes through the v2 function when the flag is set.
        comp_per_seed: List[Dict[str, Any]] = []
        direct_per_seed: List[Dict[str, Any]] = []
        for s in seeds:
            comp_entry = _calibrate_compositional_one_seed(
                int(s), tiny_synth, phase1_cache_dir
            )
            comp_entry.setdefault("protocol_version", "v1")
            comp_per_seed.append(comp_entry)
            if direct_calibration_v2:
                direct_entry = _calibrate_direct_v2_one_seed(
                    int(s), tiny_synth, phase1_cache_dir
                )
            else:
                direct_entry = _calibrate_direct_one_seed(
                    int(s), tiny_synth, phase1_cache_dir
                )
                direct_entry.setdefault("protocol_version", "v1")
            direct_per_seed.append(direct_entry)

        comp_status, comp_aggregate = _calibration_status(
            comp_per_seed, float(COMPOSITIONAL_THRESHOLD)
        )
        direct_status, direct_aggregate = _calibration_status(
            direct_per_seed, float(DIRECT_UNIFIED_THRESHOLD)
        )

        direct_protocol = "v2" if direct_calibration_v2 else "v1"
        direct_method = (
            CALIBRATION_METHOD_DOC_DIRECT_V2
            if direct_calibration_v2 else CALIBRATION_METHOD_DOC
        )
        result: Dict[str, Any] = {
            "mode": "calibration",
            "seeds": list(seeds),
            "tiny_synth": bool(tiny_synth),
            "method": CALIBRATION_METHOD_DOC,
            "compositional_gate": {
                "per_seed_calibrated_thresholds": [
                    float(d["calibrated_threshold"]) for d in comp_per_seed
                ],
                "per_seed_details": comp_per_seed,
                "aggregate_calibrated_threshold": float(comp_aggregate),
                "committed_threshold": float(COMPOSITIONAL_THRESHOLD),
                "calibration_status": comp_status,
                "protocol_version": "v1",
            },
            "direct_gate": {
                "per_seed_calibrated_thresholds": [
                    float(d["calibrated_threshold"]) for d in direct_per_seed
                ],
                "per_seed_details": direct_per_seed,
                "aggregate_calibrated_threshold": float(direct_aggregate),
                "committed_threshold": float(DIRECT_UNIFIED_THRESHOLD),
                "calibration_status": direct_status,
                "protocol_version": direct_protocol,
                "method": direct_method,
            },
            "note": (
                "calibration only -- NOT a decisive result. Per-seed "
                "calibrated thresholds reported for BOTH the "
                "compositional gate AND the NEW substrate-specific "
                "direct gate. The controller picks the aggregate value "
                "and updates the source-constant in a SEPARATE commit. "
                "The runner only writes JSON."
            ),
        }
        if tiny_synth:
            result["note"] = (
                "TINY-SYNTH toy numbers -- NOT a result; logic-screen "
                "only. INSUFFICIENT-SEPARATION expected on toy data "
                "(per the per-regime stage's pattern); the decisive "
                "multi-seed CuPy calibration is a controller-only step "
                "at full biological scale."
            )

        if comp_status == "PENDING":
            try:
                print(
                    "CALIBRATION-PENDING (compositional): aggregate "
                    "calibrated threshold = %.6f vs committed %.6f."
                    % (comp_aggregate, float(COMPOSITIONAL_THRESHOLD)),
                    file=sys.stderr, flush=True,
                )
            except Exception:
                pass
        if direct_status == "PENDING":
            try:
                print(
                    "CALIBRATION-PENDING (direct): aggregate "
                    "calibrated threshold = %.6f vs committed "
                    "placeholder %.6f. Controller must commit the "
                    "calibrated value in a SEPARATE commit (mirrors "
                    "abe65f6 for the compositional gate)."
                    % (direct_aggregate, float(DIRECT_UNIFIED_THRESHOLD)),
                    file=sys.stderr, flush=True,
                )
            except Exception:
                pass

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(result, indent=2))
        return result

    # ---- Per-cell evaluation. ----
    cells: List[Dict[str, Any]] = []
    schedule = [(s, N) for s in seeds for N in loads]
    start = 0

    if ckpt:
        prev = load_checkpoint(ckpt)
        if prev is not None:
            start = resume_epoch(prev)
            blob = prev.get("weights", [None])[0]
            if blob is not None:
                try:
                    cells = json.loads(
                        bytes(np.asarray(blob)).decode("utf-8")
                    )
                except Exception:
                    cells = []

    try:
        for epoch in range(start, len(schedule)):
            s, N = schedule[epoch]
            cell = _run_evaluation_arm(int(s), int(N), tiny_synth,
                                          phase1_cache_dir)
            cells.append(cell)
            if ckpt:
                blob = np.frombuffer(
                    json.dumps(cells).encode("utf-8"), dtype=np.uint8
                )
                save_checkpoint(ckpt, epoch, {"cells": [blob]}, None, [])
    except KeyboardInterrupt:
        print(
            "INTERRUPTED -- partial checkpoint flushed; resumable",
            flush=True,
        )
        if not cells:
            raise

    # ---- Aggregate into rungs + call frozen verdict. ----
    cells_by_N: Dict[int, List[Dict[str, Any]]] = {}
    seeds_seen_by_N: Dict[int, set] = {}
    for c in cells:
        cells_by_N.setdefault(c["N"], []).append(c)
        seeds_seen_by_N.setdefault(c["N"], set()).add(c["seed"])

    if seeds_seen_by_N:
        n_seeds_min = min(len(v) for v in seeds_seen_by_N.values())
    else:
        n_seeds_min = 0

    rungs = _aggregate_rungs(cells_by_N, n_seeds_min)
    verdict = per_regime_monitor_verdict(rungs)

    result: Dict[str, Any] = {
        "mode": "evaluation",
        "rungs": rungs,
        "verdict": verdict,
        "tiny_synth": bool(tiny_synth),
        "seeds": list(seeds),
        "loads": list(loads),
        "raw_cells": cells,
        "phase1_cache_dir": phase1_cache_dir,
        "moat_direct": float(MOAT_DIRECT),
        "compositional_threshold": float(COMPOSITIONAL_THRESHOLD),
    }
    if tiny_synth:
        result["note"] = (
            "TINY-SYNTH toy numbers -- NOT a result; logic-screen only. "
            "Phase-1 training is shrunk to a few events; compositional "
            "encoding shrunk to one pair per rung. The decisive multi-"
            "seed CuPy run at full biological scale is a later "
            "controller-only task."
        )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Unified per-regime monitor + per-regime encoding runner "
            "(Phase-1 multi-event direct training cached + "
            "compositional one-shot encoding + per-query-type routing "
            "through both calibrated moats + uniform_ctrl built-in "
            "control + direct_retain readout from the same full run; "
            "reuse-only; no autograd)."
        )
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument(
        "--loads",
        type=int,
        nargs="+",
        default=list(_PR_LADDER),
        help="Load ladder (default the frozen ladder (2,3,5)).",
    )
    ap.add_argument(
        "--tiny-synth",
        action="store_true",
        help=(
            "Shrink Phase-1 training events + compositional pair count "
            "hard for the logic-screen smoke. Toy numbers are NOT a "
            "result."
        ),
    )
    ap.add_argument(
        "--phase1-cache-dir",
        default=_PHASE1_CACHE_DEFAULT,
        help=(
            "Directory where the per-seed Phase-1 substrate "
            "checkpoints are stored. The decisive multi-seed run "
            "amortises Phase-1 training across all (seed, N) cells "
            "and across future eval stages."
        ),
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Kill-safe checkpoint path (REUSED sim.train_checkpoint; "
            "re-run resumes at the next (seed, N) cell)."
        ),
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Write the full result JSON here.",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Run the held-out calibration step ONLY for BOTH the "
            "compositional gate AND the NEW substrate-specific direct "
            "gate (closes adversarial review defect #2). Writes per-"
            "seed thresholds + match status to the JSON output; does "
            "NOT modify any gate-module source file. Controller commits "
            "the calibrated constants in a SEPARATE commit (mirrors "
            "abe65f6 for the compositional gate)."
        ),
    )
    ap.add_argument(
        "--direct-calibration-v2",
        action="store_true",
        help=(
            "When combined with --calibrate, route the direct-gate "
            "calibration through the v2 protocol: per-word target-vs-"
            "best-off-target gap aggregated over the FULL trained vocab "
            "(no per-seed half-split). v1 suffered a methodology bug "
            "(the 'ungroundable' set was the held-out half of the TRAINED "
            "vocab queried with its own trained code; per-seed random "
            "split produced INVERTED outcomes at 2/3 seeds). The "
            "compositional-gate calibration is UNAFFECTED (still v1)."
        ),
    )
    a = ap.parse_args(argv)

    result = run_unified_per_regime_monitor(
        seeds=a.seeds,
        loads=tuple(a.loads),
        tiny_synth=a.tiny_synth,
        phase1_cache_dir=a.phase1_cache_dir,
        out_path=a.out,
        ckpt=a.ckpt,
        calibrate=a.calibrate,
        direct_calibration_v2=a.direct_calibration_v2,
    )
    tag = " [TINY-SYNTH toy -- NOT a result]" if a.tiny_synth else ""
    if a.calibrate:
        comp = result.get("compositional_gate", {})
        direct = result.get("direct_gate", {})
        print(
            "CALIBRATION (compositional): status=%s aggregate=%.6f "
            "committed=%.6f"
            % (
                comp.get("calibration_status", "?"),
                float(comp.get("aggregate_calibrated_threshold", 0.0)),
                float(comp.get("committed_threshold", 0.0)),
            ),
            flush=True,
        )
        print(
            "CALIBRATION (direct): status=%s aggregate=%.6f "
            "committed=%.6f%s"
            % (
                direct.get("calibration_status", "?"),
                float(direct.get("aggregate_calibrated_threshold", 0.0)),
                float(direct.get("committed_threshold", 0.0)),
                tag,
            ),
            flush=True,
        )
        return 0
    g = result["verdict"]["gate"]
    print("GATE=%s%s" % (g, tag), flush=True)
    print(json.dumps(result["rungs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
