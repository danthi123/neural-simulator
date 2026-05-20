"""Net-new per-regime metacognitive-monitor runner (Architecture A).

Biology (Miyamoto 2017 doubly-dissociable parallel metamemory streams):
the brain runs SEPARATE metacognitive monitors per regime. The triple-
convergent ceiling (Stage-1 static + SPEAR rhythm-multiplexed
synaptic_gain + Pirazzini disinhibition + Hasselmo ACh -- ALL hit the
SAME calibrated-650 no-confabulation threshold for compositional
queries) empirically localised the direct-retrieval-calibrated
trustworthy-abstention THRESHOLD itself as the rate-limiting factor.

This module is the ONLY genuinely net-new code: a per-regime ROUTING
controller + a pre-registered CALIBRATION block. EVERY learning rule
+ subsystem is reused BYTE-UNCHANGED by import:

  * existing direct-retrieval moat: REUSED abstention_gate.gate (7/7,
    DEFAULT_THRESHOLD = 650.0). Stays byte-unchanged as the direct-
    retrieval-regime monitor.
  * new compositional-regime moat: REUSED abstention_gate_compositional
    .gate (7/7, COMPOSITIONAL_THRESHOLD placeholder = 0.0). Sits
    ALONGSIDE the existing moat.
  * substrate + hippocampus + dlpfc PFC frame: REUSED the validated v16
    + hippocampus + dlpfc construction (mirror the Stage-1 / SPEAR /
    Pirazzini-cleared _build_substrate path).
  * encoding: REUSED the engram API (start_engram_recording /
    commit_engram_tag / stimulate_tag / clear_tag_drive). OPAQUE tag
    names (Stage-1 lesson: tags carry no answer).
  * readout: REUSED compose_concept_engram.lang_output_pattern_during_
    stim + lang_output_pattern_during_input + the calibrated
    `_ranked_from_pattern` raw firing-rate confidence formula (the
    same quantity the 650 moat is calibrated against).
  * kill-safe/resume: REUSED sim.train_checkpoint (per-cell save).
  * frozen verdict: REUSED per_regime_monitor_core.per_regime_monitor
    _verdict.

The runner has TWO MODES (a single threaded `calibrate: bool` flag):

  CALIBRATION MODE (calibrate=True) -- runs ONLY the calibration step:
    For each seed, generate a HELD-OUT calibration set distinguishable
    from the evaluation set via a separate sub-seed offset (seed +
    10000). Encode the calibration facts via the engram API. For each
    calibration query measure the raw firing-rate confidence at
    `lang_output` for the CORRECT (groundable) answer; for each
    UNGROUNDABLE control query measure the same quantity. Compute the
    calibrated threshold as the midpoint of the medians of the two
    populations (the simplest defensible separator; see CALIBRATION_
    METHOD_DOC). Records per-seed calibrated thresholds + committed
    constant + status (MATCH / PENDING / MISMATCH).
    Calibration writes the JSON output ONLY. The runner does NOT modify
    the source file `abstention_gate_compositional.py`; updating the
    committed constant is a SEPARATE controller commit (the runner
    prints to stderr when the constant is the placeholder and the
    calibrated value is non-zero).

  EVALUATION MODE (calibrate=False, default) -- runs the decisive
    capability test the Task 6 controller will invoke at full
    biological scale. Two query populations per (seed, N):
      * DIRECT queries: query each encoded fact by its primary concept;
        routed through gate_direct (existing 650 moat, byte-unchanged).
      * COMPOSITIONAL queries: query each encoded fact-pair
        compositionally (cue the noun; ask for the bound adj); routed
        through gate_compositional (new gate, threshold =
        COMPOSITIONAL_THRESHOLD).
    Three measurement arms per (seed, N):
      * `full`            = per-regime architecture: direct ->
                            gate_direct(., 650); compositional ->
                            gate_compositional(., COMPOSITIONAL_
                            THRESHOLD). full_acc = fraction of ALL
                            (direct + compositional) queries answered
                            correctly.
      * `uniform_ctrl`    = SAME run except BOTH gates set to
                            MOAT_DIRECT = 650 (single-threshold-applied-
                            uniformly). The decisive built-in control:
                            must collapse to the triple-convergent
                            ceiling. uniform_ctrl differs from `full`
                            ONLY in the threshold-routing decision;
                            same seed, same facts, same query set.
      * `direct_retain`   = direct-queries-only accuracy under the
                            per-regime architecture (must NOT degrade
                            vs the validated baseline). Read from the
                            SAME run as `full` (no separate draws).
      * abstain_correct   = fraction of UNGROUNDABLE queries on which
                            the appropriate-regime gate abstained.
    Emit per (seed, N): the six required rung keys (N, n_seeds,
    full_acc, uniform_ctrl_acc, direct_retain_acc, abstain_correct).
    Aggregate across seeds to one rung per N; call the frozen verdict.

Anti-cheat (carry forward Stage-1 / SPEAR / Pirazzini lessons):
  * OPAQUE tag names (f"ep_{i}"); no tag-string parsing on tag names
    (the answer is decoded from the validated neural readout, never
    out of an opaque tag string).
  * BOTH gates fed the calibrated raw firing-rate confidence
    (`pat[active].sum() / n_active` via _ranked_from_pattern); not
    cosine * norm.
  * uniform_ctrl differs from `full` ONLY in the threshold-routing
    decision; same seed, same encoded facts, same query set.
  * direct_retain is read from the SAME run as `full` (no separate
    draws).
  * Calibration set HELD OUT (sub-seed offset = +10000); evaluation
    set uses the seed itself.
  * Calibration mode does NOT write to abstention_gate_compositional
    .py; outputs to JSON only.

ASCII only. NO autograd anywhere. CuPy is the real / decisive path;
--tiny-synth shrinks pools/episodes/queries so the smoke is seconds
(toy numbers explicitly NOT a result). The decisive multi-seed CuPy
run is a later controller-only task -- NOT performed here.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Backend policy mirrors the SPEAR / Pirazzini runners. CuPy is the
# decisive path; NumPy ONLY when CuPy is genuinely unavailable (GPU-
# less box). SimulationBridge binds its array module at sim.bridge
# IMPORT time, so on a CuPy-capable box the tiny smoke still runs on
# the bridge's real backend.
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

# REUSED gates (each byte-unchanged in its own module).
from research.runners.abstention_gate import gate as gate_direct
from research.runners.abstention_gate import DEFAULT_THRESHOLD as MOAT_DIRECT
from research.runners.abstention_gate_compositional import (
    gate as gate_compositional,
    COMPOSITIONAL_THRESHOLD,
)

from sim.train_checkpoint import (  # REUSED UNMODIFIED
    save_checkpoint,
    load_checkpoint,
    resume_epoch,
)

# REUSED Stage-1 / SPEAR / Pirazzini-cleared substrate vocabulary +
# raw firing-rate ranking + hippocampal tag-region filter. Identity-
# imports only (byte-unchanged) -- duplicate no subsystem logic.
from research.runners.compose_retrieval_runner import (
    _NOUNS,
    _VERBS,
    _ADJS,
    _N_WORDS_ORTHOGONAL,
    _recent_facts,
    _ranked_from_pattern,
    _HIPPO_TAG_REGIONS,
)


# =====================================================================
# Calibration separator method (pre-registered; ASCII docstring is the
# `method` field echoed in the JSON output).
# =====================================================================
CALIBRATION_METHOD_DOC = (
    "median_midpoint: for each seed, encode a held-out calibration set "
    "(sub-seed = seed + 10000) of compositional facts, then measure the "
    "raw firing-rate confidence (the calibrated `pat[active].sum() / "
    "n_active` quantity) at language_output for (a) GROUNDABLE queries "
    "with known correct answers and (b) UNGROUNDABLE control queries "
    "(asking about facts that were never encoded). The calibrated "
    "threshold for that seed is the midpoint of the two population "
    "medians: 0.5 * (median(groundable) + median(ungroundable)). The "
    "runner returns the per-seed list; the controller picks the "
    "aggregate calibrated value and updates the COMPOSITIONAL_THRESHOLD "
    "source constant in a SEPARATE commit (the runner only writes JSON)."
)
_CALIB_SUBSEED_OFFSET = 10000  # held-out calibration sub-seed
_CALIB_MATCH_TOL = 1e-6        # tolerance for MATCH detection


# =====================================================================
# Substrate construction. Mirrors the Pirazzini / SPEAR / Stage-1
# cleared _build_substrate EXACTLY: same v16 + hippocampus + dlpfc
# fields; no cfg.num_traits override; the neuromodulator subsystem is
# left disabled here because the per-regime routing does NOT need ACh /
# disinhibition / lang_drive modulators. The calibration + evaluation
# pathways run through the validated substrate without any modulation.
# =====================================================================
def _build_substrate(seed: int, tiny_synth: bool):
    """Construct the validated v16 + hippocampus + dlpfc PFC frame
    bridge, REUSING the validated builders byte-unchanged. Returns
    (bridge, dims)."""
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

    import research.runners.concept_pool_demo as cpd
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

    if tiny_synth:
        n_lang_input = 64
        n_per_pool = 12
        n_fs_per_pool = 3
        n_dlpfc_verb = 24
    else:
        n_lang_input = 2048
        n_per_pool = 200
        n_fs_per_pool = 24
        n_dlpfc_verb = 200

    # weak_dynamics=True (validated v16) -- identical to Stage-1 / SPEAR
    # / Pirazzini.
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
    cfg.seed = seed
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

    dims = {
        "n_lang_input": n_lang_input,
        "n_per_pool": n_per_pool,
        "n_fs_per_pool": n_fs_per_pool,
        "sparsity": 0.05,
        "dt_ms": cfg.dt_ms,
    }
    return bridge, dims


# =====================================================================
# Encoding helpers (REUSE the engram API).
# =====================================================================
def _encode_facts(bridge, facts: List[Tuple[str, str]], dims: Dict[str, Any],
                  encoding_steps: int) -> List[str]:
    """Encode each (noun, adj) fact via the REUSED engram API. Tag
    names are OPAQUE (f"ep_{i}") -- carry NO answer string (Stage-1
    lesson)."""
    from research.runners.compose_concept_engram import encode_concept_pair

    tags: List[str] = []
    for i, (noun, adj) in enumerate(facts):
        tag = f"ep_{i}"  # OPAQUE
        if tag in {t["name"] for t in bridge.list_engram_tags()}:
            try:
                bridge.delete_engram_tag(tag)
            except Exception:
                pass
        encode_concept_pair(
            bridge, noun, adj, tag,
            encoding_steps=encoding_steps,
            drive_pA=200.0,
            sparsity=dims["sparsity"],
            n_lang_input=dims["n_lang_input"],
            n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
            region_filter=_HIPPO_TAG_REGIONS,
            top_k=max(8, dims["n_per_pool"] // 4),
            balanced_teacher_pA=500.0,
            verbose=False,
        )
        tags.append(tag)
    return tags


def _direct_query_confidence(bridge, cue_word: str, dims: Dict[str, Any],
                              recall_steps: int):
    """Direct-retrieval-regime read: drive lang_input(cue_word) and
    return the ranked raw firing-rate confidence list at lang_output.
    The single-concept retrieval mode -- the validated v16 path the
    direct moat is calibrated against."""
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_input,
    )

    pat, n_lo = lang_output_pattern_during_input(
        bridge, cue_word,
        n_lang_input=dims["n_lang_input"],
        sparsity=dims["sparsity"],
        n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
        stim_steps=recall_steps,
    )
    return _ranked_from_pattern(pat, n_lo, dims, exclude=cue_word)


def _compositional_query_confidence(bridge, cue_noun: str,
                                     tag_name: Optional[str],
                                     dims: Dict[str, Any],
                                     recall_steps: int):
    """Compositional-retrieval-regime read: cue the noun + stimulate
    the bound engram tag; sum per-concept raw firing-rate confidences
    at lang_output (Stage-1 cleared compose path). Same retrieval-
    augmented composition the convergent ceiling stack used; only the
    GATE the controller routes the answer through is what this stage
    varies."""
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_stim,
        lang_output_pattern_during_input,
    )

    # Consolidated-regime read (drive cue noun).
    cons_pat, n_lo = lang_output_pattern_during_input(
        bridge, cue_noun,
        n_lang_input=dims["n_lang_input"],
        sparsity=dims["sparsity"],
        n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
        stim_steps=recall_steps,
    )
    cons_ranked = _ranked_from_pattern(
        cons_pat, n_lo, dims, exclude=cue_noun
    )

    # Hippocampal-regime read (stim the bound tag).
    if tag_name is not None and tag_name in {
        t["name"] for t in bridge.list_engram_tags()
    }:
        hip_pat, n_lo2 = lang_output_pattern_during_stim(
            bridge, tag_name, drive_pA=1500.0, stim_steps=recall_steps,
        )
        hip_ranked = _ranked_from_pattern(
            hip_pat, n_lo2, dims, exclude=cue_noun
        )
    else:
        hip_ranked = []

    # Compose: sum per-concept raw firing-rate confidences.
    scores: Dict[str, float] = {}
    for w, r, _ in cons_ranked:
        scores[w] = scores.get(w, 0.0) + r
    for w, r, _ in hip_ranked:
        scores[w] = scores.get(w, 0.0) + r
    ranked = sorted(
        ((w, scores[w], "compose") for w in scores),
        key=lambda t: -t[1],
    )
    return ranked


def _decoded_answer(ranked, gate_fn, threshold: float) -> Optional[str]:
    """Route the ranked confidences through the chosen gate at the
    chosen threshold. Returns the decoded answer string or None
    (abstained)."""
    decided = gate_fn(ranked, threshold)
    return None if decided is None else decided[0]


def _top_confidence(ranked) -> float:
    """Top raw firing-rate confidence value (the moat-calibrated
    quantity). Returns 0.0 on empty input."""
    if not ranked:
        return 0.0
    return float(ranked[0][1])


# =====================================================================
# Evaluation arm: one full per-regime run for one (seed, N).
# =====================================================================
def _run_evaluation_arm(seed: int, N: int, tiny_synth: bool) -> Dict[str, Any]:
    """Run the per-regime architecture for ONE (seed, N) cell. Returns
    the four rung accuracy fields the frozen verdict consumes:
      full_acc, uniform_ctrl_acc, direct_retain_acc, abstain_correct.

    `full` and `uniform_ctrl` are scored from the SAME forward pass
    (same encoded facts, same queries, same ranked confidences) --
    only the threshold-routing decision differs. `direct_retain_acc`
    is the direct-only subset of `full`. `abstain_correct` is the
    fraction of UNGROUNDABLE queries on which the appropriate-regime
    gate abstained.
    """
    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200
    facts = _recent_facts(N)

    bridge, dims = _build_substrate(seed, tiny_synth)
    tags = _encode_facts(bridge, facts, dims, enc_steps)

    # ---- DIRECT queries: one per encoded fact, cue the noun -------
    # CORRECT direct answer is the noun itself read back. The direct-
    # retrieval-regime monitor (existing 650 moat) gates the answer.
    n_direct_total = 0
    n_direct_correct_full = 0
    n_direct_correct_uniform = 0
    # ABSTAIN tracking is on UNGROUNDABLE queries (below) -- direct
    # queries here are groundable by construction; we still record
    # whether the moat over- or under-confidently abstained for
    # diagnostics but the abstain_correct denominator is the
    # ungroundable set.
    for noun, _adj in facts:
        n_direct_total += 1
        ranked = _direct_query_confidence(
            bridge, noun, dims, recall_steps
        )
        # `full` routes direct queries through gate_direct (650 moat).
        ans_full = _decoded_answer(ranked, gate_direct, MOAT_DIRECT)
        # `uniform_ctrl` ALSO routes through the 650 moat (single-
        # threshold-applied-uniformly). For direct queries the two
        # arms therefore agree by construction -- the difference shows
        # up on compositional queries.
        ans_uniform = _decoded_answer(ranked, gate_direct, MOAT_DIRECT)
        # The validated direct-retrieval correctness criterion: the
        # decoded top word is the cued noun (the same v16 single-pool
        # retrieval used in the 88.75% multi-seed bidirectional binding).
        if ans_full == noun:
            n_direct_correct_full += 1
        if ans_uniform == noun:
            n_direct_correct_uniform += 1

    # ---- COMPOSITIONAL queries: one per encoded fact, cue the noun,
    # expect the bound adj. The compositional-regime monitor (new
    # gate, threshold = COMPOSITIONAL_THRESHOLD) gates the answer for
    # `full`; the SAME ranked confidences are also routed through the
    # direct moat (650) for `uniform_ctrl`. ----
    n_comp_total = 0
    n_comp_correct_full = 0
    n_comp_correct_uniform = 0
    for i, (noun, adj) in enumerate(facts):
        n_comp_total += 1
        tag = tags[i] if i < len(tags) else None
        ranked = _compositional_query_confidence(
            bridge, noun, tag, dims, recall_steps
        )
        # `full`: per-regime architecture routes compositional queries
        # through the COMPOSITIONAL gate at COMPOSITIONAL_THRESHOLD.
        ans_full = _decoded_answer(
            ranked, gate_compositional, COMPOSITIONAL_THRESHOLD
        )
        # `uniform_ctrl`: single-threshold-applied-uniformly --
        # compositional queries STILL routed through the direct moat
        # at MOAT_DIRECT (650). The SOLE difference from `full`.
        ans_uniform = _decoded_answer(
            ranked, gate_compositional, MOAT_DIRECT
        )
        if ans_full == adj:
            n_comp_correct_full += 1
        if ans_uniform == adj:
            n_comp_correct_uniform += 1

    # ---- UNGROUNDABLE queries: query each non-noun concept word that
    # was NOT encoded as a fact noun. The appropriate-regime gate
    # MUST abstain on these (no engram tag, no bound adj). Used for
    # abstain_correct. ----
    encoded_nouns = {n for n, _ in facts}
    ungroundable_words = [w for w in _NOUNS if w not in encoded_nouns]
    n_ungroundable = len(ungroundable_words)
    n_abstain_ok = 0
    for w in ungroundable_words:
        # Compositional ungroundable: no tag for this word. The
        # compositional gate at COMPOSITIONAL_THRESHOLD should abstain.
        ranked = _compositional_query_confidence(
            bridge, w, None, dims, recall_steps
        )
        ans = _decoded_answer(
            ranked, gate_compositional, COMPOSITIONAL_THRESHOLD
        )
        if ans is None:
            n_abstain_ok += 1

    # ---- Aggregate the four rung fields. ----
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
# Calibration arm: one held-out calibration on one seed.
# =====================================================================
def _calibrate_one_seed(seed: int, tiny_synth: bool) -> Dict[str, Any]:
    """Run the per-seed calibration on a HELD-OUT set distinguishable
    from the evaluation set via sub-seed = seed + 10000. The set
    encodes a small batch of compositional facts; per-query the runner
    reads the raw firing-rate confidence at lang_output for both
    GROUNDABLE and UNGROUNDABLE queries. Returns the per-seed
    calibrated threshold + the two population summaries.

    Strengthen-only fix (review:held-out-vocab-partition, Task 4):
    the calibration (noun, adj) PAIRS are explicitly drawn from the
    Cartesian-product MINUS the set of eval pairs ``_recent_facts(max(
    _PR_LADDER))`` so a calibrated threshold cannot be fitted on a
    pairing the eval will query. The vocabulary itself (only 4 nouns
    + 4 adjs) is unavoidably shared between eval and calibration, but
    the pair-level partition prevents the dominant failure mode
    (calibrating directly on an eval-set association). If the held-
    out pair pool is empty the runner returns a degenerate empty
    result and the calibration status downstream becomes
    INSUFFICIENT-SEPARATION.
    """
    sub_seed = int(seed) + _CALIB_SUBSEED_OFFSET
    cal_rng = np.random.default_rng(sub_seed)

    # Held-out calibration set: PAIRS guaranteed disjoint from the
    # eval set's _recent_facts pairs at the maximum N in the ladder
    # (so the calibrated threshold is never fitted on a pairing the
    # eval will encode + query). Sample without replacement; if the
    # held-out pool is too small fall back to whatever is available
    # and let the INSUFFICIENT-SEPARATION downstream check flag it.
    eval_pairs = set(_recent_facts(max(_PR_LADDER)))
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

    bridge, dims = _build_substrate(sub_seed, tiny_synth)
    tags = _encode_facts(bridge, calib_facts, dims, enc_steps)

    # GROUNDABLE: query each encoded calibration fact compositionally;
    # confidence on the CORRECT answer (the bound adj).
    groundable_confidences: List[float] = []
    for i, (noun, adj) in enumerate(calib_facts):
        tag = tags[i] if i < len(tags) else None
        ranked = _compositional_query_confidence(
            bridge, noun, tag, dims, recall_steps
        )
        # The confidence is the top raw firing-rate value when the
        # correct answer is on top, else the rate ON the correct adj's
        # row. We want the confidence ON the correct answer
        # (groundable means the system has signal toward the right
        # answer; we are measuring how high that signal goes).
        rate_on_correct = 0.0
        for w, r, _t in ranked:
            if w == adj:
                rate_on_correct = float(r)
                break
        groundable_confidences.append(rate_on_correct)

    # UNGROUNDABLE: query a noun that was NOT encoded. No bound adj
    # exists; the top compositional confidence is the population's
    # noise-floor representative for the ungroundable case.
    encoded_nouns = {n for n, _ in calib_facts}
    ungroundable_nouns = [w for w in _NOUNS if w not in encoded_nouns]
    if not ungroundable_nouns:
        # Vocab depleted (tiny-synth with full noun set); fall back to
        # querying the unbound concept words (verbs) which were never
        # encoded.
        ungroundable_nouns = list(_VERBS)
    ungroundable_confidences: List[float] = []
    for w in ungroundable_nouns:
        ranked = _compositional_query_confidence(
            bridge, w, None, dims, recall_steps
        )
        ungroundable_confidences.append(_top_confidence(ranked))

    # Separator: median midpoint (the pre-registered method).
    if groundable_confidences:
        g_median = float(statistics.median(groundable_confidences))
    else:
        g_median = 0.0
    if ungroundable_confidences:
        u_median = float(statistics.median(ungroundable_confidences))
    else:
        u_median = 0.0
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


# =====================================================================
# Aggregation.
# =====================================================================
def _aggregate_evaluation_rungs(cells_by_N: Dict[int, List[Dict[str, Any]]],
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


def _calibration_status(per_seed: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Classify the calibration outcome vs the committed constant.

    INSUFFICIENT-SEPARATION -- on ANY per-seed cell the groundable
                population median is <= the ungroundable population
                median. The midpoint separator only makes sense when
                signal > noise; if the populations overlap or invert
                the committed threshold would route the WRONG way at
                eval. STRENGTHEN-only review fix (Task 4); controller
                must NOT commit a calibrated constant when this status
                is emitted.
    MATCH    -- the aggregate calibrated value is within tolerance of
                the committed COMPOSITIONAL_THRESHOLD.
    PENDING  -- the committed constant is the placeholder (0.0) AND
                the aggregate calibrated value is non-zero. The runner
                writes JSON only; the controller updates the source
                file in a separate commit.
    MISMATCH -- the committed constant is non-zero AND the aggregate
                calibrated value differs from it beyond tolerance.
    """
    vals = [d["calibrated_threshold"] for d in per_seed]
    aggregate = float(sum(vals) / len(vals)) if vals else 0.0

    # STRENGTHEN-only: refuse to emit a separator when the populations
    # overlap or invert at any seed. The midpoint is only defensible
    # when groundable_median > ungroundable_median (signal genuinely
    # above noise floor); otherwise the committed threshold would
    # silently route the wrong way at eval. Tolerance is zero -- any
    # equal-or-inverted case flags.
    for d in per_seed:
        g = float(d.get("groundable_median", 0.0))
        u = float(d.get("ungroundable_median", 0.0))
        if g <= u:
            return "INSUFFICIENT-SEPARATION", aggregate

    committed = float(COMPOSITIONAL_THRESHOLD)
    if abs(aggregate - committed) <= _CALIB_MATCH_TOL:
        return "MATCH", aggregate
    if abs(committed) <= _CALIB_MATCH_TOL:
        # Placeholder; calibration value is non-zero => PENDING
        # (controller updates constant in a separate commit).
        return "PENDING", aggregate
    return "MISMATCH", aggregate


# =====================================================================
# Top-level entry.
# =====================================================================
def run_per_regime_monitor(seeds,
                            loads=_PR_LADDER,
                            tiny_synth: bool = False,
                            calibrate: bool = False,
                            out_path: Optional[str] = None,
                            ckpt: Optional[str] = None) -> Dict[str, Any]:
    """Per-regime metacognitive-monitor capability runner.

    Two modes:
      calibrate=False (default) -- evaluation. Per seed, per N in the
        frozen ladder: build the validated substrate, encode facts via
        the engram API, run direct + compositional queries through the
        per-regime architecture (`full`), the uniform-threshold control
        (`uniform_ctrl` = both gates at MOAT_DIRECT=650), and record
        direct_retain_acc + abstain_correct. Aggregate to rungs and
        score with the frozen `per_regime_monitor_verdict`.

      calibrate=True -- calibration. Per seed (sub-seed = seed +
        10000), encode a held-out calibration set and measure the raw
        firing-rate confidences for groundable vs ungroundable queries.
        Compute the per-seed calibrated threshold as the midpoint of
        the two population medians. Compare to the committed
        COMPOSITIONAL_THRESHOLD; record MATCH / PENDING / MISMATCH.
        The runner writes JSON only; updating the source-file constant
        is a SEPARATE controller commit. Calibration mode does NOT
        produce a decisive verdict.

    Kill-safe via the REUSED sim.train_checkpoint (evaluation mode
    only; calibration is a short pre-step the controller runs once).
    """
    seeds = list(seeds)
    loads = tuple(int(x) for x in loads)

    if calibrate:
        # ---- CALIBRATION MODE: held-out per-seed thresholds. ----
        per_seed: List[Dict[str, Any]] = []
        for s in seeds:
            d = _calibrate_one_seed(s, tiny_synth)
            per_seed.append(d)

        status, aggregate = _calibration_status(per_seed)
        result: Dict[str, Any] = {
            "mode": "calibration",
            "seeds": list(seeds),
            "per_seed_calibrated_thresholds": [
                float(d["calibrated_threshold"]) for d in per_seed
            ],
            "per_seed_details": per_seed,
            "aggregate_calibrated_threshold": float(aggregate),
            "committed_threshold": float(COMPOSITIONAL_THRESHOLD),
            "calibration_status": status,
            "method": CALIBRATION_METHOD_DOC,
            "tiny_synth": bool(tiny_synth),
            "note": (
                "calibration only -- NOT a decisive result. Per-seed "
                "calibrated thresholds reported; the controller picks "
                "the aggregate value and updates the COMPOSITIONAL_"
                "THRESHOLD source constant in a SEPARATE commit."
            ),
        }
        if tiny_synth:
            result["note"] = (
                "TINY-SYNTH toy numbers -- NOT a result; logic-screen "
                "only. Calibration on a held-out tiny-synth set is a "
                "smoke; the decisive calibration is a controller-only "
                "step at full biological scale."
            )

        if status == "PENDING":
            # Inform stderr that the controller must update the source
            # constant as a separate commit. The runner does NOT mutate
            # the source file.
            try:
                print(
                    "CALIBRATION-PENDING: aggregate calibrated threshold "
                    "= %.6f vs committed placeholder %.6f. The runner "
                    "writes JSON only; the controller updates the source "
                    "constant in a SEPARATE commit."
                    % (aggregate, float(COMPOSITIONAL_THRESHOLD)),
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(result, indent=2))
        return result

    # ---- EVALUATION MODE: per-regime architecture + uniform control. ----
    cells: List[Dict[str, Any]] = []
    start = 0
    schedule = [(s, N) for s in seeds for N in loads]
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
            cell = _run_evaluation_arm(s, N, tiny_synth)
            cells.append(cell)
            if ckpt:
                blob = np.frombuffer(
                    json.dumps(cells).encode("utf-8"), dtype=np.uint8
                )
                save_checkpoint(ckpt, epoch, {"cells": [blob]}, None, [])
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable",
              flush=True)
        if not cells:
            raise

    cells_by_N: Dict[int, List[Dict[str, Any]]] = {}
    seeds_seen_by_N: Dict[int, set] = {}
    for c in cells:
        cells_by_N.setdefault(c["N"], []).append(c)
        seeds_seen_by_N.setdefault(c["N"], set()).add(c["seed"])

    if seeds_seen_by_N:
        n_seeds = min(len(v) for v in seeds_seen_by_N.values())
    else:
        n_seeds = 0

    rungs = _aggregate_evaluation_rungs(cells_by_N, n_seeds)
    verdict = per_regime_monitor_verdict(rungs)

    result = {
        "mode": "evaluation",
        "rungs": rungs,
        "verdict": verdict,
        "tiny_synth": bool(tiny_synth),
        "seeds": list(seeds),
        "loads": list(loads),
        "raw_cells": cells,
        "committed_threshold": float(COMPOSITIONAL_THRESHOLD),
        "moat_direct": float(MOAT_DIRECT),
    }
    if tiny_synth:
        result["note"] = (
            "TINY-SYNTH toy numbers -- NOT a result; logic-screen only."
        )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-regime metacognitive-monitor runner "
                    "(Architecture A; calibration + per-query-type "
                    "routing; uniform_ctrl built-in control; "
                    "direct_retain readout; reuse-only; no autograd)."
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--loads", type=int, nargs="+",
                    default=list(_PR_LADDER),
                    help="Load ladder (default the frozen ladder).")
    ap.add_argument("--tiny-synth", action="store_true",
                    help="Shrink pools/episodes for the logic-screen "
                         "smoke. Toy numbers are NOT a result.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Run the held-out calibration step ONLY. "
                         "Writes per-seed calibrated thresholds + "
                         "match status to the JSON output; does NOT "
                         "modify abstention_gate_compositional.py.")
    ap.add_argument("--ckpt", default=None,
                    help="Kill-safe checkpoint path (REUSED "
                         "sim.train_checkpoint; re-run resumes; "
                         "evaluation mode only).")
    ap.add_argument("--out", default=None,
                    help="Write the full result JSON here.")
    a = ap.parse_args(argv)

    result = run_per_regime_monitor(
        seeds=a.seeds,
        loads=tuple(a.loads),
        tiny_synth=a.tiny_synth,
        calibrate=a.calibrate,
        out_path=a.out,
        ckpt=a.ckpt,
    )
    tag = " [TINY-SYNTH toy -- NOT a result]" if a.tiny_synth else ""
    if a.calibrate:
        status = result.get("calibration_status", "?")
        agg = result.get("aggregate_calibrated_threshold", 0.0)
        print(
            "CALIBRATION-STATUS=%s aggregate=%.6f committed=%.6f%s"
            % (
                status,
                float(agg),
                float(result.get("committed_threshold", 0.0)),
                tag,
            ),
            flush=True,
        )
        print(json.dumps(result["per_seed_calibrated_thresholds"], indent=2),
              flush=True)
    else:
        g = result["verdict"]["gate"]
        print("GATE=%s%s" % (g, tag), flush=True)
        print(json.dumps(result["rungs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
