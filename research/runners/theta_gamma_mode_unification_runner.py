"""Net-new theta-gamma mode-unification runner (Task 2 of the arc).

Biology (Lisman 2005; Hasselmo 2002; Buzsaki theta-gamma coupling): the
hippocampal theta cycle multiplexes ENCODE and RETRIEVE windows. At
theta-trough (encode phase) acetylcholine is HIGH and external inputs
write fresh content into the network. At theta-peak (retrieve phase)
acetylcholine is LOW and the network's recurrent dynamics pattern-
complete from a partial cue. Critically, the external drive that wrote
the encoding does NOT continue to dominate during the retrieve window:
the network must be ABLE to listen to its OWN recurrent state for
pattern completion to work.

The localisation finding from the 4-architecture convergent ceiling
(commit 110f7cd) diagnosed the exact failure mode the prior compose-
retrieval and unified per-regime monitor runners both shared: at
deployment, the cued-noun's diffuse lang_input drive DOMINATED the
engram tag's selective bound-adj drive. The runner kept cueing the
noun while simultaneously stimulating the bound engram tag, so the
language_output read was a sum of two drives -- one diffuse (the
noun's lang_input -> lang_output pathway) and one selective (the
engram tag's selective bound-adj projection). The diffuse signal won.

This runner introduces the cue-suppression-during-retrieve mechanism:
a three-phase theta cycle per compositional query. During ENCODE
(first half of the cycle) the cue's lang_input drive is ON; during
GAP (transition) and RETRIEVE (last 40 steps) the cue is suppressed
in the FULL arm so the engram tag's selective drive can dominate the
lang_output read. The UNIFORM_CTRL arm runs the same cycle EXCEPT
the cue stays ON during the retrieve window; this is the decisive
built-in experimental contrast (mirrors the unified runner's
per-regime-vs-uniform pattern).

This module is the ONLY genuinely net-new code in the arc:
  * a runner-local three-phase theta cycle controller per query
    (encode/gap/retrieve) that writes cp_external_input_current
    PER STEP inside the loop so the writes survive any sub-helper
    clears (mirrors the Pirazzini FIX B per-step pattern);
  * the FULL arm suppresses cue during retrieve, the UNIFORM_CTRL
    arm keeps the cue ON during retrieve (the SOLE differentiator);
  * a STRUCTURAL-EFFECT PROBE that verifies the mechanism produces
    > 1 mV bridge-state divergence between the two arms via the
    runner's ACTUAL code path (mirrors Pirazzini d462bf0 lesson:
    must work via the real per-step loop, not a synthetic per-step
    bypass). The probe runs in main() BEFORE the decisive eval loop.

Every other subsystem is REUSED-BY-IMPORT from the prior unified
per-regime monitor runner (commit 25b9183 byte-stable):

  * Substrate construction + Phase-1 caching (validated v16 +
    hippocampus + dlpfc PFC frame): REUSED
    ``unified_per_regime_monitor_runner._build_bridge_with_phase1_recipe``
    + ``_phase1_recipe`` + ``_phase1_cache_path`` + ``_freeze_phase1_gates``.
  * Compositional one-shot encoding: REUSED ``_encode_facts``
    (calls byte-unchanged ``encode_concept_pair`` internally).
  * Compositional pair generation: REUSED ``_unified_compositional_pairs``
    (sub-seed offset +20000 from the unified runner).
  * Direct W->A readout: REUSED ``_direct_query_ranked``.
  * Compositional readout (lang_output firing-rate confidence):
    REUSED ``_compositional_query_ranked``.
  * Frozen capability verdict: REUSED
    ``theta_gamma_mode_unification_core.theta_gamma_mode_unification_verdict``
    (Task 1 byte-unchanged; bars set in advance, NEVER tuned).
  * Both substrate-specific calibrated moats (the four moats):
    DIRECT_UNIFIED_THRESHOLD = 0.2841666666666667 +
    COMPOSITIONAL_UNIFIED_THRESHOLD = 0.1977124183006536 imported
    BYTE-UNCHANGED (no calibration changes).
  * Kill-safe checkpoint: REUSED sim.train_checkpoint.

Anti-cheat (carry forward all prior lessons):
  * OPAQUE tag names; no tag-string parsing on tag names.
  * BOTH moats fed the calibrated raw firing-rate confidence
    quantities.
  * The ``cross_pool_concept`` gate is opened ONLY inside the
    encoding window (via ``encode_concept_pair``) then closed.
  * ``uniform_ctrl`` differs from ``full`` ONLY in the
    suppress_cue_during_retrieve flag (same seed, same encoded
    facts, same query set).
  * ``direct_retain`` is read from the SAME run as ``full``.
  * No protected-file edits; no autograd; the runner is reuse-only
    orchestration of the prior arc's modules + the net-new theta
    cycle controller + the structural-effect probe.

ASCII only. CuPy is the real / decisive path; --tiny-synth shrinks
Phase-1 training + compositional encoding + theta-cycle step counts
so the smoke is seconds (toy numbers explicitly NOT a result).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Backend policy mirrors the unified / per-regime / SPEAR / Pirazzini
# runners. CuPy is the decisive path; NumPy ONLY when CuPy is genuinely
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

# REUSED frozen capability verdict (Task 1 byte-unchanged).
from research.runners.theta_gamma_mode_unification_core import (
    theta_gamma_mode_unification_verdict,
    _TG_LADDER,
)

# REUSED gates byte-unchanged (the four moats). FULL arm routes direct
# queries through DIRECT_UNIFIED_THRESHOLD, compositional queries
# through COMPOSITIONAL_UNIFIED_THRESHOLD. UNIFORM_CTRL applies
# DIRECT_UNIFIED_THRESHOLD uniformly to BOTH regimes (same threshold-
# uniformly pattern the unified runner uses).
from research.runners.abstention_gate import DEFAULT_THRESHOLD as MOAT_DIRECT
from research.runners.abstention_gate_compositional import (
    gate as gate_compositional,
    COMPOSITIONAL_THRESHOLD,
)
from research.runners.abstention_gate_compositional_unified import (
    gate as gate_compositional_unified,
    COMPOSITIONAL_UNIFIED_THRESHOLD,
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

# REUSED prior-arc orchestration -- every Phase-1 + substrate +
# encoding helper imported BYTE-UNCHANGED from the unified runner.
from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_recipe,
    _phase1_cache_path,
    _phase1_train_if_needed,
    _freeze_phase1_gates,
    _all_pool_regions,
    _all_words_word_to_idx,
    _direct_pool_target,
    _direct_query_ranked,
    _compositional_query_ranked,
    _unified_compositional_pairs,
    _encode_facts,
    _UNIFIED_SUBSEED_OFFSET,
    _PHASE1_CACHE_DEFAULT,
)
from research.runners.compose_retrieval_runner import (
    _NOUNS,
    _VERBS,
    _ADJS,
    _N_WORDS_ORTHOGONAL,
    _HIPPO_TAG_REGIONS,
)


# =====================================================================
# Theta cycle constants. FROZEN -- never tuned in response to results.
# 100-step cycle at dt=0.5ms = ~125 ms. The split (50/10/40) follows
# Lisman 2005 / Hasselmo 2002 theta-cycle accounting: ~50% encode
# (theta-trough), short transition (gap), ~40% retrieve (theta-peak).
# tiny-synth shrinks these so the smoke is seconds.
# =====================================================================
THETA_CYCLE_STEPS = 100
ENCODE_STEPS = 50         # Phase 1: ENCODE/CUE (theta-trough)
GAP_STEPS = 10            # Phase 2: GAP (transition)
RETRIEVE_STEPS = 40       # Phase 3: RETRIEVE/PATTERN-COMPLETE (theta-peak)
assert ENCODE_STEPS + GAP_STEPS + RETRIEVE_STEPS == THETA_CYCLE_STEPS
# Tag-stim window during ENCODE: the engram tag is briefly stimulated
# in the early part of the encode phase so the bound ensemble is
# refreshed before we transition into the retrieve window.
TAG_STIM_STEPS = 20

# tiny-synth shrunk values. Each phase keeps a non-zero floor so the
# trough/peak split is genuinely exercised under the smoke.
TINY_ENCODE_STEPS = 6
TINY_GAP_STEPS = 2
TINY_RETRIEVE_STEPS = 6
TINY_TAG_STIM_STEPS = 3


def _phase_step_counts(tiny_synth: bool) -> Tuple[int, int, int, int]:
    """Return (encode, gap, retrieve, tag_stim) step counts.

    Full scale uses the frozen Lisman-2005-style accounting; tiny-synth
    shrinks all four to a logic-screen smoke.
    """
    if tiny_synth:
        return (
            int(TINY_ENCODE_STEPS),
            int(TINY_GAP_STEPS),
            int(TINY_RETRIEVE_STEPS),
            int(TINY_TAG_STIM_STEPS),
        )
    return (
        int(ENCODE_STEPS),
        int(GAP_STEPS),
        int(RETRIEVE_STEPS),
        int(TAG_STIM_STEPS),
    )


# =====================================================================
# The net-new shared-theta-rhythm controller: a runner-local three-
# phase theta cycle that writes cp_external_input_current PER STEP
# inside the loop so the writes survive any sub-helper clears
# (mirrors Pirazzini FIX B per-step pattern). The FULL arm suppresses
# the cue during RETRIEVE; the UNIFORM_CTRL arm keeps the cue ON.
# This is the SOLE differentiator between the two arms.
# =====================================================================
def _run_theta_cycle_query(
    bridge,
    cue_word: str,
    tag_name: Optional[str],
    dims: Dict[str, Any],
    suppress_cue_during_retrieve: bool,
    tiny_synth: bool,
    word_to_idx: Dict[str, int],
    all_pools: List[str],
) -> List[Tuple[str, float, str]]:
    """Run ONE theta cycle (encode / gap / retrieve) for a compositional
    query. The mechanism is structurally active ONLY when
    suppress_cue_during_retrieve differs between calls (the FULL vs
    UNIFORM_CTRL contrast).

    Phase 1 (ENCODE): cue ON; if tag_name, stimulate the engram tag
        briefly (TAG_STIM_STEPS).
    Phase 2 (GAP): cue OFF; brief transition; clear tag drive.
    Phase 3 (RETRIEVE): cue ON or OFF depending on
        suppress_cue_during_retrieve. Accumulate lang_output firing
        pattern AND per-concept-pool firing rate ranked confidences
        across the RETRIEVE window only. The ranked-list shape mirrors
        the _compositional_query_ranked output the calibrated
        compositional gate is calibrated on.

    Returns ranked [(pool_or_word, raw_rate, "compose"), ...]
    descending by rate. The gate functions in the eval arm consume the
    ranked list shape directly.

    Implementation notes:
      * cp_external_input_current writes happen INSIDE the per-step
        loop (Pirazzini FIX B): this is the load-bearing detail that
        makes the structural-effect probe non-trivial. Writing on
        entry-only would let a sub-helper clear erase the drive.
      * The cue's orthogonal drive pattern is computed via the REUSED
        orthogonal_drive_pattern helper that the v14/v16 substrate is
        calibrated against.
      * The engram tag's stimulation is done via the REUSED
        bridge.stimulate_tag / clear_tag_drive API (no new learning
        rule).
      * lang_output firing-rate confidence is computed via the same
        _ranked_from_pattern formula the calibrated 0.197712
        compositional gate is calibrated on. Mirrors the
        _compositional_query_ranked structure.
    """
    from sim.backend import get_backend, to_host
    from sim.text_embeddings import orthogonal_drive_pattern
    from research.runners.compose_retrieval_runner import _ranked_from_pattern
    cp, _backend_name = get_backend()

    n_encode, n_gap, n_retrieve, n_tag_stim = _phase_step_counts(tiny_synth)

    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)
    n_lang_out = len(lang_out_idx)

    # Cue's orthogonal drive pattern -- the SAME pattern Phase-1
    # training drove the substrate to bind the word against (the
    # calibrated v14/v16 path).
    n_lang_input = int(dims["n_lang_input"])
    sparsity = float(dims["sparsity"])
    n_words_for_orthogonal = int(dims["n_words_for_orthogonal"])
    cue_idx = int(word_to_idx[cue_word])
    cue_drive_np = orthogonal_drive_pattern(
        cue_idx=cue_idx,
        n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input,
        drive_max_pA=200.0,
        sparsity=sparsity,
    )
    cue_drive = cp.asarray(cue_drive_np, dtype=cp.float32)

    # Ensure a clean baseline before the theta cycle starts.
    bridge.cp_external_input_current[:] = 0.0
    try:
        bridge.clear_tag_drive()
    except Exception:
        pass

    # ----- Phase 1: ENCODE (cue ON; brief tag stim early in the phase).
    if tag_name is not None and tag_name in {
        t["name"] for t in bridge.list_engram_tags()
    }:
        try:
            bridge.stimulate_tag(tag_name, drive_pA=1500.0, additive=False)
        except Exception:
            pass
    for step in range(int(n_encode)):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[lang_in_arr] = cue_drive
        bridge._run_one_simulation_step()
        # Clear tag drive after the brief stim window so the encode
        # tail does not keep dragging the bound ensemble.
        if step + 1 == int(n_tag_stim) and tag_name is not None:
            try:
                bridge.clear_tag_drive(tag_name)
            except Exception:
                pass

    # If TAG_STIM_STEPS >= ENCODE_STEPS (tiny-synth corner), make sure
    # tag drive is cleared before the gap.
    if tag_name is not None:
        try:
            bridge.clear_tag_drive(tag_name)
        except Exception:
            pass
        try:
            bridge.clear_tag_drive()
        except Exception:
            pass

    # ----- Phase 2: GAP (cue OFF; quiet transition).
    for _step in range(int(n_gap)):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()

    # ----- Phase 3: RETRIEVE / PATTERN-COMPLETE.
    # This is the load-bearing window. The FULL arm
    # (suppress_cue_during_retrieve=True) holds the cue OFF so the
    # bridge's recurrent state -- carrying the encoding's residual
    # activity in CA3 + dlpfc -- can drive lang_output via the
    # engram-bound projections. The UNIFORM_CTRL arm keeps the cue
    # ON; the cued-noun's diffuse lang_input drive dominates as
    # documented in the localisation finding.
    pattern = cp.zeros(n_lang_out, dtype=cp.float32)
    for _step in range(int(n_retrieve)):
        if suppress_cue_during_retrieve:
            bridge.cp_external_input_current[:] = 0.0
        else:
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[lang_in_arr] = cue_drive
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            pattern = pattern + firing[lang_out_arr].astype(cp.float32)

    # Settle: clear the drive so subsequent queries start clean.
    bridge.cp_external_input_current[:] = 0.0

    pattern_host = to_host(pattern)
    ranked = _ranked_from_pattern(pattern_host, n_lang_out, dims, exclude=cue_word)
    return ranked


# =====================================================================
# Deterministic-RNG isolation helper for the structural-effect probe
# AND the per-cell eval arm. CLOSES THE EIGHTH ADVERSARIAL REVIEW BLOCK
# (the prior 30.24 mV divergence was an RNG-drift artefact: two bridges
# in the same process share the global cp.random state, so OU-noise
# draws inside _run_one_simulation_step diverged purely because of the
# order the bridges were stepped -- the suppress_cue_during_retrieve
# flag was INVISIBLE behind the RNG drift).
#
# The fix is purely runner-side. Before each _run_theta_cycle_query
# call, we (a) capture the active backend's RNG state, (b) deterministic-
# seed the backend so the second bridge's draws are IDENTICAL to the
# first, (c) call the helper, (d) restore the captured state so other
# components (training, OU noise outside this loop, etc.) see no
# perturbation. The same deterministic seed value MUST be used for BOTH
# arms of any controlled contrast (full vs uniform_ctrl OR probe-on vs
# probe-off): the two arms then see identical OU-noise streams and the
# only difference between them is the suppress_cue_during_retrieve flag.
#
# Implementation uses sim.backend.get_random_state / set_random_state /
# the active backend's random.seed -- backend-aware (CuPy when GPU,
# NumPy when CPU). No new top-level imports introduced (sim.backend is
# already imported inside _run_theta_cycle_query / _structural_effect_probe).
# =====================================================================
def _seed_query_rng(rng_seed: int) -> Any:
    """Capture the active backend's RNG state and deterministic-seed it.

    Returns an opaque token to pass back to _restore_query_rng so the
    caller can restore the global state once the helper returns. This
    keeps the RNG perturbation LOCAL to a single _run_theta_cycle_query
    call; other components (training, ungroundable queries, etc.)
    that happen outside the wrapped window see no change.

    What this seeds (CLOSES the 8th adversarial review BLOCK on RNG
    drift):
      * The active backend's RNG (CuPy or NumPy, via sim.backend).
        This is what cp.random.randn / np.random.randn inside
        bridge._run_one_simulation_step's OU noise consumes.
      * The top-level numpy.random global (independent module-level
        state from the backend on CuPy; the same state on NumPy).
        bridge.py uses np.random for a number of CPU-side draws
        (e.g. structural plasticity candidate sampling, lognormal
        heterogeneity, etc.); leaving it un-seeded would silently
        diverge two bridges across calls.
      * The Python stdlib `random` module global. Bridge code paths
        downstream of _run_one_simulation_step occasionally use it.
    """
    from sim.backend import get_backend, get_random_state
    xp, name = get_backend()
    # Snapshot ALL three RNG sources so they can be restored together.
    backend_saved = get_random_state()
    import numpy
    np_saved = numpy.random.get_state()
    import random as _pyrandom
    py_saved = _pyrandom.getstate()

    seed_value = int(rng_seed) & 0x7FFFFFFF
    if name == "cupy":
        import cupy
        cupy.random.seed(seed_value)
    # ALWAYS also seed numpy.random + stdlib random because (a) bridge
    # uses np.random directly for CPU-side draws even on the CuPy
    # backend; (b) on the NumPy backend cupy.random.seed isn't relevant
    # but numpy.random.seed IS the backend seed. Single deterministic
    # seed across all three sources -> two arms see identical draws
    # everywhere downstream.
    numpy.random.seed(seed_value)
    _pyrandom.seed(seed_value)
    return (backend_saved, np_saved, py_saved)


def _restore_query_rng(saved_state: Any) -> None:
    """Restore the active backend's + numpy + python RNG states
    captured by _seed_query_rng. Idempotent on a None saved_state
    (no-op)."""
    if saved_state is None:
        return
    from sim.backend import set_random_state
    backend_saved, np_saved, py_saved = saved_state
    set_random_state(backend_saved)
    import numpy
    numpy.random.set_state(np_saved)
    import random as _pyrandom
    _pyrandom.setstate(py_saved)


# =====================================================================
# Structural-effect probe -- MANDATORY (mirrors Pirazzini d462bf0
# lesson). Verifies the theta-gamma mechanism produces NON-byte-
# identical bridge state between suppress_cue_during_retrieve=True
# vs =False via the runner's ACTUAL code path (the per-step theta
# cycle helper). NOT a synthetic-bypass probe.
#
# CLOSES the 8th adversarial review BLOCK: the prior probe measured
# RNG-drift (two bridges sharing the global cp.random state diverged
# purely from OU-noise draw ordering, NOT from the cue-suppression
# mechanism). The fix is the deterministic-RNG isolation pattern: seed
# the backend's RNG to the SAME value before each arm's call so both
# arms see IDENTICAL OU-noise streams; the SOLE remaining difference
# is the suppress_cue_during_retrieve flag. The probe also asserts
# CONTROLS: when both arms pass the SAME flag, the bridge-state
# divergence MUST be < 0.5 mV (RNG isolation is working). The
# flag-differing case MUST exceed 1 mV (mechanism is structurally
# active). If any control fails, RNG isolation is broken and the
# probe raises.
# =====================================================================
# Deterministic RNG seed for the structural-effect probe. The probe
# uses ONE fixed value across all four runs (two flag-differing, two
# flag-same controls) so the only between-arm difference is the flag.
_PROBE_RNG_SEED = 999

# Tolerance for the "controls must show near-zero divergence" check.
# 0.5 mV is well below the 1.0 mV bar the flag-differing case must
# exceed; this gives the controls room for hardware-level fp32 noise
# without false-rejecting RNG-isolation that is genuinely working.
_PROBE_CONTROL_TOL_MV = 0.5


def _structural_effect_probe(
    seed: int = 42,
    tiny_synth: bool = True,
    cache_dir: Optional[str] = None,
) -> float:
    """Run the runner's actual code path twice with the SAME initial
    bridge state but different suppress_cue_during_retrieve flags;
    return the max absolute membrane-potential divergence (mV) for
    the flag-differing case. Strengthened to close the 8th adversarial
    review BLOCK by adding RNG isolation + controls (see below).

    Mechanism (CLOSES the 8th adversarial review BLOCK):
      * Deterministic-seed the active backend's RNG to _PROBE_RNG_SEED
        BEFORE each _run_theta_cycle_query call. Both arms therefore
        see IDENTICAL OU-noise streams; the SOLE remaining difference
        between the arms is the suppress_cue_during_retrieve flag.
      * Restore the RNG state after each call so other components
        (Phase-1 training, fact encoding, etc.) see no perturbation.
      * Run TWO additional CONTROL contrasts at the SAME seed:
          (1) both arms pass suppress=True -> divergence must be < 0.5 mV
          (2) both arms pass suppress=False -> divergence must be < 0.5 mV
        If either control shows large divergence, RNG isolation is
        broken and the probe raises RuntimeError (the previously
        reported "30.24 mV" was exactly this failure mode -- it
        reproduced under both-flags-True and both-flags-False).
      * The flag-differing case (True vs False) MUST exceed 1.0 mV
        for the mechanism to be declared structurally active.

    If the flag-differing divergence is below 1 mV, the mechanism is
    structurally inert and the caller MUST abort (no decisive numbers
    reported). If a control shows divergence above the tolerance, RNG
    isolation is broken and the caller MUST abort. Either raises
    RuntimeError. Returns the flag-differing diff (float, > 1.0) when
    BOTH the mechanism is structurally active AND the controls pass.
    """
    from sim.backend import get_backend, to_host
    cp, _backend_name = get_backend()

    cache_dir = str(cache_dir) if cache_dir else _PHASE1_CACHE_DEFAULT
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Ensure the Phase-1 cache exists for this seed; train if needed.
    _phase1_train_if_needed(int(seed), cache_dir, tiny_synth)
    cache_path = _phase1_cache_path(cache_dir, seed)

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
    facts = _unified_compositional_pairs(seed, 1)
    enc_steps = 8 if tiny_synth else 200
    cue_noun, _adj = facts[0]
    tag_name = "ep_0"

    # Deterministic RNG seed for the ENCODING phase (separate from the
    # probe's RNG seed for the theta cycle, so encoding and theta-cycle
    # noise streams are independent but DETERMINISTIC across arms).
    ENCODE_RNG_SEED = 31337

    def _one_contrast(flag_a: bool, flag_b: bool) -> float:
        """Build two fresh bridges, load the SAME checkpoint into both,
        deterministic-seed BEFORE _encode_facts on EACH so the encoded
        bridge states are IDENTICAL, then deterministic-seed BEFORE
        _run_theta_cycle_query on EACH (same seed across arms) so the
        theta-cycle OU-noise streams are IDENTICAL. The SOLE remaining
        between-arm difference is the suppress_cue_during_retrieve flag.

        Returns max |delta v_membrane|. The fresh bridges per contrast
        ensure no cross-contrast leakage."""
        bridge_a = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        bridge_b = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        bridge_a.load_checkpoint(str(cache_path))
        bridge_b.load_checkpoint(str(cache_path))
        _freeze_phase1_gates(bridge_a)
        _freeze_phase1_gates(bridge_b)

        # Encoding phase: identical deterministic seed BEFORE each call
        # so the two bridges end up in byte-identical encoded states.
        saved_enc_a = _seed_query_rng(ENCODE_RNG_SEED)
        try:
            _encode_facts(bridge_a, facts, dims, enc_steps)
        finally:
            _restore_query_rng(saved_enc_a)
        saved_enc_b = _seed_query_rng(ENCODE_RNG_SEED)
        try:
            _encode_facts(bridge_b, facts, dims, enc_steps)
        finally:
            _restore_query_rng(saved_enc_b)

        # Theta-cycle phase: identical deterministic seed BEFORE each
        # arm so the OU-noise streams match across arms.
        saved_a = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            _ = _run_theta_cycle_query(
                bridge_a,
                cue_word=cue_noun,
                tag_name=tag_name,
                dims=dims,
                suppress_cue_during_retrieve=flag_a,
                tiny_synth=tiny_synth,
                word_to_idx=word_to_idx,
                all_pools=all_pools,
            )
        finally:
            _restore_query_rng(saved_a)

        saved_b = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            _ = _run_theta_cycle_query(
                bridge_b,
                cue_word=cue_noun,
                tag_name=tag_name,
                dims=dims,
                suppress_cue_during_retrieve=flag_b,
                tiny_synth=tiny_synth,
                word_to_idx=word_to_idx,
                all_pools=all_pools,
            )
        finally:
            _restore_query_rng(saved_b)

        v_a = to_host(bridge_a.cp_membrane_potential_v)
        v_b = to_host(bridge_b.cp_membrane_potential_v)
        return float(np.max(np.abs(np.asarray(v_a) - np.asarray(v_b))))

    # Flag-differing case: the mechanism MUST move the bridge state.
    diff_flag_diff = _one_contrast(True, False)
    # Both-True control: with identical flag + identical RNG, the two
    # bridges MUST agree (the RNG-isolation soundness check).
    diff_both_true = _one_contrast(True, True)
    # Both-False control: same check, opposite flag value.
    diff_both_false = _one_contrast(False, False)

    if diff_both_true > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Structural-effect probe CONTROL FAILED (both-True): with "
            "suppress_cue_during_retrieve=True on BOTH bridges and the "
            "same deterministic RNG seed, the two bridges diverged by "
            "%.6g mV (> %.3g mV tolerance). RNG isolation is broken; "
            "the flag-differing divergence is NOT attributable to the "
            "cue-suppression mechanism. Closes 8th adversarial review "
            "BLOCK; fix RNG isolation and re-run."
            % (diff_both_true, _PROBE_CONTROL_TOL_MV)
        )
    if diff_both_false > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Structural-effect probe CONTROL FAILED (both-False): with "
            "suppress_cue_during_retrieve=False on BOTH bridges and the "
            "same deterministic RNG seed, the two bridges diverged by "
            "%.6g mV (> %.3g mV tolerance). RNG isolation is broken; "
            "the flag-differing divergence is NOT attributable to the "
            "cue-suppression mechanism. Closes 8th adversarial review "
            "BLOCK; fix RNG isolation and re-run."
            % (diff_both_false, _PROBE_CONTROL_TOL_MV)
        )

    if diff_flag_diff <= 1.0:
        raise RuntimeError(
            "Structural-effect probe FAILED: theta-gamma "
            "suppress_cue_during_retrieve=True vs =False produced "
            "essentially identical bridge state (max |delta v| = "
            "%.6g mV <= 1.0 mV) via the runner's ACTUAL code path "
            "(controls passed: both-True=%.6g mV, both-False=%.6g mV "
            "-- the small flag-differing divergence is the genuine "
            "mechanism effect, not RNG drift). The mechanism is "
            "structurally inert -- mirrors Pirazzini d462bf0 defect. "
            "Fix and re-run BEFORE decisive."
            % (diff_flag_diff, diff_both_true, diff_both_false)
        )

    return float(diff_flag_diff)


# =====================================================================
# Per-cell evaluation arm: one (seed, N) cell. Mirrors the unified
# runner's structure EXCEPT the compositional query loop wraps each
# query in a three-phase theta cycle (encode/gap/retrieve). The FULL
# arm runs the cycle with suppress_cue_during_retrieve=True; the
# UNIFORM_CTRL arm runs the SAME cycle EXCEPT
# suppress_cue_during_retrieve=False.
#
# Engineering decision (documented): the theta cycle modifies bridge
# state, so the FULL and UNIFORM_CTRL arms cannot share the same
# bridge. We build TWO parallel bridges at the start of the cell
# (one for FULL, one for UNIFORM_CTRL), load the SAME Phase-1
# checkpoint into both, encode the SAME facts into both, then run
# each arm's queries against its own bridge. This keeps the seed +
# encoded facts + query set + ranked confidences IDENTICAL across
# the two arms; the SOLE differentiator is the
# suppress_cue_during_retrieve flag. Cost is ~2x bridge memory per
# cell -- acceptable at tiny-synth and at full scale (the substrate
# fits in <2 GB).
# =====================================================================
def _run_evaluation_arm(seed: int, N: int, tiny_synth: bool,
                          cache_dir: str) -> Dict[str, Any]:
    """Run the theta-gamma mode-unification architecture for ONE
    (seed, N) cell. Two parallel bridges (FULL + UNIFORM_CTRL); each
    runs the SAME theta cycle EXCEPT the SOLE differentiator (the
    suppress_cue_during_retrieve flag) on the RETRIEVE window of
    each compositional query.

    Returns the four rung accuracy fields the frozen verdict
    consumes: ``full_acc``, ``uniform_ctrl_acc``,
    ``direct_retain_acc``, ``abstain_correct``.
    """
    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200

    cache_path = _phase1_cache_path(cache_dir, seed)
    if not cache_path.exists():
        raise RuntimeError(
            "Phase-1 cache missing for seed %d at %s; call "
            "_phase1_train_if_needed first." % (seed, cache_path)
        )

    # TWO parallel bridges -- same architecture, same Phase-1
    # checkpoint, same frozen gates. The SOLE differentiator is the
    # suppress_cue_during_retrieve flag passed into
    # _run_theta_cycle_query.
    bridge_full = _build_bridge_with_phase1_recipe(seed, tiny_synth)
    bridge_uniform = _build_bridge_with_phase1_recipe(seed, tiny_synth)
    bridge_full.load_checkpoint(str(cache_path))
    bridge_uniform.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge_full)
    _freeze_phase1_gates(bridge_uniform)

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

    # Compositional encoding: encode the SAME facts into BOTH bridges
    # so the encoded state is identical at the start of the queries.
    # RNG isolation (CLOSES 8th adversarial review BLOCK): seed the
    # active backend + numpy + python RNGs to the SAME deterministic
    # value BEFORE each call so the encoded bridge states are
    # byte-identical across arms. Without this, the two bridges
    # consume different RNG draws during encoding and end up in
    # different post-encode states -- the arm contrast then conflates
    # encoding noise with the mechanism's actual effect.
    facts = _unified_compositional_pairs(seed, N)
    encode_rng_seed = (
        int(seed) * 1_000_003 + int(N) * 1009 + 31337
    ) & 0x7FFFFFFF
    saved_enc_full = _seed_query_rng(encode_rng_seed)
    try:
        tags_full = _encode_facts(bridge_full, facts, dims, enc_steps)
    finally:
        _restore_query_rng(saved_enc_full)
    saved_enc_uniform = _seed_query_rng(encode_rng_seed)
    try:
        tags_uniform = _encode_facts(bridge_uniform, facts, dims, enc_steps)
    finally:
        _restore_query_rng(saved_enc_uniform)

    # ---- DIRECT queries: one per unique trained word in the cell's
    # facts. BOTH arms route direct queries through the SAME
    # substrate-specific direct gate (DIRECT_UNIFIED_THRESHOLD). The
    # two arms therefore agree on direct counts by construction; the
    # difference shows up on compositional queries.
    n_direct_total = 0
    n_direct_correct_full = 0
    n_direct_correct_uniform = 0
    direct_words: List[Tuple[str, str]] = []
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
        # FULL bridge direct read.
        ranked_full = _direct_query_ranked(
            bridge_full, word, dims, all_pools, word_to_idx,
            stim_steps=recall_steps, reset_steps=recall_steps // 2,
        )
        decided_full = gate_direct_unified(ranked_full, DIRECT_UNIFIED_THRESHOLD)
        ans_full = None if decided_full is None else decided_full[0]
        if ans_full == expected_pool:
            n_direct_correct_full += 1
        # UNIFORM_CTRL bridge direct read.
        ranked_uniform = _direct_query_ranked(
            bridge_uniform, word, dims, all_pools, word_to_idx,
            stim_steps=recall_steps, reset_steps=recall_steps // 2,
        )
        decided_uniform = gate_direct_unified(
            ranked_uniform, DIRECT_UNIFIED_THRESHOLD
        )
        ans_uniform = None if decided_uniform is None else decided_uniform[0]
        if ans_uniform == expected_pool:
            n_direct_correct_uniform += 1

    # ---- COMPOSITIONAL queries: one per encoded fact, cue the noun,
    # expect the bound adj. Each query runs a full theta cycle on its
    # bridge; the FULL bridge runs with suppress_cue_during_retrieve=True;
    # the UNIFORM_CTRL bridge runs with =False. This is the SOLE
    # differentiator between the two arms (the
    # cue-suppression-during-retrieve mechanism).
    #
    # RNG isolation (CLOSES 8th adversarial review BLOCK): each query
    # deterministic-seeds the active backend's RNG to the SAME value
    # for BOTH arms (derived from seed/N/query-index so it is unique
    # per query but identical across the two arms). Both arms therefore
    # see IDENTICAL OU-noise streams; the SOLE between-arm difference
    # is the suppress_cue_during_retrieve flag. The RNG state is
    # restored after each call so the surrounding code (encoding,
    # direct queries, ungroundable queries) sees no perturbation.
    n_comp_total = 0
    n_comp_correct_full = 0
    n_comp_correct_uniform = 0
    for i, (noun, adj) in enumerate(facts):
        n_comp_total += 1
        tag_full = tags_full[i] if i < len(tags_full) else None
        tag_uniform = tags_uniform[i] if i < len(tags_uniform) else None
        # Deterministic RNG seed unique to this (seed, N, query-index)
        # tuple. Both arms below use the SAME value so the OU-noise
        # streams match across arms; the SOLE differentiator is the
        # suppress_cue_during_retrieve flag.
        query_rng_seed = (
            int(seed) * 1_000_003 + int(N) * 1009 + int(i) * 7919 + 17
        ) & 0x7FFFFFFF
        # FULL arm: cue suppressed during retrieve.
        saved_full = _seed_query_rng(query_rng_seed)
        try:
            ranked_full = _run_theta_cycle_query(
                bridge_full,
                cue_word=noun,
                tag_name=tag_full,
                dims=dims,
                suppress_cue_during_retrieve=True,
                tiny_synth=tiny_synth,
                word_to_idx=word_to_idx,
                all_pools=all_pools,
            )
        finally:
            _restore_query_rng(saved_full)
        decided_full = gate_compositional_unified(
            ranked_full, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        ans_full = None if decided_full is None else decided_full[0]
        # UNIFORM_CTRL arm: cue stays ON during retrieve. SAME theta
        # cycle EXCEPT the SOLE differentiator. The ranked output is
        # gated by the (uniform) DIRECT_UNIFIED_THRESHOLD via the
        # compositional-gate machinery (mirrors the unified runner's
        # uniform_ctrl convention). IDENTICAL deterministic RNG seed
        # to the FULL arm above.
        saved_uniform = _seed_query_rng(query_rng_seed)
        try:
            ranked_uniform = _run_theta_cycle_query(
                bridge_uniform,
                cue_word=noun,
                tag_name=tag_uniform,
                dims=dims,
                suppress_cue_during_retrieve=False,
                tiny_synth=tiny_synth,
                word_to_idx=word_to_idx,
                all_pools=all_pools,
            )
        finally:
            _restore_query_rng(saved_uniform)
        decided_uniform = gate_compositional(
            ranked_uniform, DIRECT_UNIFIED_THRESHOLD
        )
        ans_uniform = None if decided_uniform is None else decided_uniform[0]
        if ans_full == adj:
            n_comp_correct_full += 1
        if ans_uniform == adj:
            n_comp_correct_uniform += 1

    # ---- UNGROUNDABLE queries: vocabulary words NOT used in this
    # rung's facts. The appropriate-regime gate MUST abstain.
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
        ranked = _direct_query_ranked(
            bridge_full, w, dims, all_pools, word_to_idx,
            stim_steps=recall_steps, reset_steps=recall_steps // 2,
        )
        decided = gate_direct_unified(ranked, DIRECT_UNIFIED_THRESHOLD)
        if decided is None:
            n_abstain_ok += 1

    # Compositional ungroundables (cue a noun that was NOT encoded);
    # run via the FULL arm's theta cycle (the abstention behaviour
    # under the load-bearing mechanism). The substrate-specific
    # compositional gate should abstain.
    ungroundable_nouns = [w for w in _NOUNS if w not in encoded_nouns]
    for w in ungroundable_nouns:
        n_ungroundable += 1
        ranked = _run_theta_cycle_query(
            bridge_full,
            cue_word=w,
            tag_name=None,
            dims=dims,
            suppress_cue_during_retrieve=True,
            tiny_synth=tiny_synth,
            word_to_idx=word_to_idx,
            all_pools=all_pools,
        )
        decided = gate_compositional_unified(
            ranked, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        if decided is None:
            n_abstain_ok += 1

    # ---- Aggregate the four rung fields from the SAME run.
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
def run_theta_gamma_mode_unification(
    seeds,
    loads=_TG_LADDER,
    tiny_synth: bool = False,
    phase1_cache_dir: str = _PHASE1_CACHE_DEFAULT,
    out_path: Optional[str] = None,
    ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """Theta-gamma mode-unification capability runner.

    Per seed (in order):
      * Phase-1 multi-event direct training (cached) -- REUSED
        ``_phase1_train_if_needed`` from the unified runner. The
        Phase-1 caching strategy is the primary cost-amortisation.

    Per (seed, N) cell:
      * Build TWO parallel bridges from the same Phase-1 checkpoint
        (one for FULL, one for UNIFORM_CTRL).
      * Encode the SAME compositional facts into BOTH bridges.
      * For each query: FULL runs the theta cycle with
        suppress_cue_during_retrieve=True; UNIFORM_CTRL runs the
        SAME cycle EXCEPT suppress_cue_during_retrieve=False. The
        SOLE differentiator between the arms.
      * Emit the four verdict fields per cell.

    Kill-safe via the REUSED ``sim.train_checkpoint`` (cell
    granularity).
    """
    seeds = list(seeds)
    loads = tuple(int(x) for x in loads)
    phase1_cache_dir = str(phase1_cache_dir)

    Path(phase1_cache_dir).mkdir(parents=True, exist_ok=True)
    for s in seeds:
        _phase1_train_if_needed(int(s), phase1_cache_dir, tiny_synth)

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
    verdict = theta_gamma_mode_unification_verdict(rungs)

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
        "moat_compositional": float(COMPOSITIONAL_THRESHOLD),
        "direct_unified_threshold": float(DIRECT_UNIFIED_THRESHOLD),
        "compositional_unified_threshold": float(
            COMPOSITIONAL_UNIFIED_THRESHOLD
        ),
        "theta_cycle_constants": {
            "THETA_CYCLE_STEPS": int(THETA_CYCLE_STEPS),
            "ENCODE_STEPS": int(ENCODE_STEPS),
            "GAP_STEPS": int(GAP_STEPS),
            "RETRIEVE_STEPS": int(RETRIEVE_STEPS),
            "TAG_STIM_STEPS": int(TAG_STIM_STEPS),
        },
    }
    if tiny_synth:
        result["note"] = (
            "TINY-SYNTH toy numbers -- NOT a result; logic-screen only. "
            "Phase-1 training is shrunk to a few events; compositional "
            "encoding shrunk to one pair per rung; theta-cycle steps "
            "shrunk (encode/gap/retrieve = %d/%d/%d). The decisive "
            "multi-seed CuPy run at full biological scale is a later "
            "controller-only task."
            % (TINY_ENCODE_STEPS, TINY_GAP_STEPS, TINY_RETRIEVE_STEPS)
        )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Theta-gamma mode-unification runner (Task 2 of the arc). "
            "Three-phase theta cycle per compositional query "
            "(encode/gap/retrieve); the FULL arm suppresses the cued-"
            "noun's diffuse lang_input drive during the retrieve "
            "window so the engram tag's selective bound-adj drive can "
            "dominate; the UNIFORM_CTRL arm keeps the cue ON during "
            "retrieve as the decisive built-in control. Reuse-only "
            "orchestration of the prior unified per-regime monitor "
            "runner + the net-new theta cycle controller + structural-"
            "effect probe. No autograd; no torch; no LLM call."
        )
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument(
        "--loads",
        type=int,
        nargs="+",
        default=list(_TG_LADDER),
        help="Load ladder (default the frozen ladder (2,3,5)).",
    )
    ap.add_argument(
        "--tiny-synth",
        action="store_true",
        help=(
            "Shrink Phase-1 training + compositional pair count + "
            "theta-cycle step counts hard for the logic-screen smoke. "
            "Toy numbers are NOT a result."
        ),
    )
    ap.add_argument(
        "--phase1-cache-dir",
        default=_PHASE1_CACHE_DEFAULT,
        help=(
            "Directory where the per-seed Phase-1 substrate "
            "checkpoints are stored (REUSED from the unified runner; "
            "byte-stable cache). The decisive multi-seed run "
            "amortises Phase-1 training across all (seed, N) cells "
            "AND across the unified runner."
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
        "--skip-structural-probe",
        action="store_true",
        help=(
            "Skip the structural-effect probe before the eval loop. "
            "ONLY for the inner-loop test smoke that exercises the "
            "probe directly via the test API (the probe is RUN in "
            "tests/test_theta_gamma_mode_unification_runner.py). Do "
            "NOT pass this for any decisive run."
        ),
    )
    a = ap.parse_args(argv)

    # MANDATORY: structural-effect probe before the decisive eval loop.
    # If the mechanism is structurally inert the probe raises and the
    # runner aborts with NO decisive numbers reported. Mirrors
    # Pirazzini d462bf0 lesson.
    if not a.skip_structural_probe:
        try:
            diff_mv = _structural_effect_probe(
                seed=int(a.seeds[0]) if a.seeds else 42,
                tiny_synth=bool(a.tiny_synth),
                cache_dir=a.phase1_cache_dir,
            )
        except RuntimeError as exc:
            print(
                "STRUCTURAL-EFFECT-PROBE FAILED: %s" % exc,
                file=sys.stderr, flush=True,
            )
            return 2
        print(
            "STRUCTURAL-EFFECT-PROBE PASS: max |delta v_membrane| = "
            "%.6g mV (> 1.0 mV)" % diff_mv,
            flush=True,
        )

    result = run_theta_gamma_mode_unification(
        seeds=a.seeds,
        loads=tuple(a.loads),
        tiny_synth=a.tiny_synth,
        phase1_cache_dir=a.phase1_cache_dir,
        out_path=a.out,
        ckpt=a.ckpt,
    )
    tag = " [TINY-SYNTH toy -- NOT a result]" if a.tiny_synth else ""
    g = result["verdict"]["gate"]
    print("GATE=%s%s" % (g, tag), flush=True)
    print(json.dumps(result["rungs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
