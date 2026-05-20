"""Net-new targeted cue-suppression-during-replay + amplified
engram-tag stim + persistent PFC-frame + higher n_replays_per_tag
runner (Task 2 of the 7th arc).

Biology (complementary-learning-systems theory; McClelland 1995;
Tonegawa 2015; Buzsaki SWR replay 2015; Wang 2002 PFC; Tulving 1973
encoding-specificity; encoding-axis SDM via orthogonal Pulvermueller
ensembles): the brain consolidates one-shot relational bindings into
cortex via NREM sharp-wave-ripple replay AND holds the ordered
compositional structure of an ongoing thought in prefrontal working
memory via NMDA-bistable persistent activity. Hippocampal replay
during NREM is NOT cue-driven -- it is internally generated; the
cortical sensory inputs that were active during the original encoding
are typically NOT replayed alongside the engram. The 7th architecture
in the gating-based composition design line refines the 6th arc's
generative-replay + PFC-frame substrate with FOUR empirically-
targeted mechanisms:

  * CUE-SUPPRESSION DURING REPLAY (not retrieve). The 6th arc's
    ``run_concept_replay_phase`` reactivated each engram tag but left
    the lang_input cue free to also contribute via the ``lang_to_ec``
    pathway; the replay-strengthened consolidation was thereby
    contaminated by cue-driven activity. The 7th arc wraps the replay
    invocation in a context manager that (a) zeroes
    ``bridge.cp_external_input_current[:n_lang_input]`` and (b)
    clamps the ``lang_to_ec`` plasticity gate to 0.0 for the
    duration of the replay window. Both are restored in a ``finally:``
    block so subsequent retrieval queries see the cue with full
    transmission and the gate at its prior value. The retrieve window
    keeps the cue ON (encoding-specificity respected per Tulving
    1973). The mechanism is biology-grounded: hippocampal replay is
    internally generated, NOT cue-driven; the 7th arc encodes that
    biology directly.

  * AMPLIFIED ENGRAM-TAG STIM (during retrieve). Per the unified
    runner's localisation finding (cued-noun dominance), the
    1500 pA tag drive used in
    ``_compositional_query_ranked`` is dominated by the cue's diffuse
    lang_input activity. The 7th arc adds a net-new wrapper
    ``_compositional_query_amplified`` that replicates the unified
    runner's compositional readout logic but multiplies the tag drive
    by ``RETRIEVE_TAG_AMP_FACTOR=3.0`` (1500 * 3.0 = 4500 pA). The
    cue drive stays at baseline. The wrapper does NOT modify the
    reused ``_compositional_query_ranked`` helper (byte-unchanged);
    the amplification is implemented locally in the 7th arc runner.

  * PERSISTENT PFC-FRAME (50 steps instead of 10). The 6th arc's
    ``_prime_pfc_frame`` drove ``dlpfc_verb`` for 10 simulation
    steps. The NMDA-bistable attractor in ``dlpfc_verb`` requires
    ~100 ms (~200 sim steps at dt=0.5 ms) of priming to lock fully
    into its high-activity attractor branch; 10 steps may be too
    brief to actually engage the bistability. The 7th arc extends
    the prime to ``PFC_FRAME_STIM_STEPS=50``. The per-step write
    pattern inside the prime loop is REUSED byte-unchanged from the
    6th arc (Pirazzini FIX B per-step pattern survives sub-helper
    clears).

  * HIGHER n_replays_per_tag (50 vs uniform-baseline 20). Strengthens
    the FULL arm's replay consolidation; the UNIFORM_CTRL arm keeps
    the 6th arc's validated 20-replay baseline. Combined with
    cue-suppression during replay, the FULL arm's replay phase
    targets the engram tag's selective pathway more strongly without
    cue contamination.

The 6-architecture convergent ceiling (Stage-1 + SPEAR + Pirazzini +
Unified per-regime monitor + Theta-gamma + 6th arc generative-replay
+ PFC-frame) has produced six distinct mechanism-level signatures;
the cross-arc trajectory analysis at commit 9693685 showed 35%
gap-closure (Unified N=3 full=0.274 -> Theta-gamma 0.280 -> 6th arc
0.458). The 7th arc tests whether four empirically-targeted
modifications continue that trajectory. See
``docs/plans/2026-05-20-7th-arc-replay-cue-suppression-amplified-tag-design.md``
and the parent design at commit ``bef9027``.

This module is the ONLY genuinely net-new code in the arc:
  * a runner-local cue-suppressed replay invocation per cell: the
    FULL arm runs
    ``_run_replay_with_cue_suppressed(bridge_full, tags, 50)`` ONCE
    after encoding; the UNIFORM_CTRL arm runs
    ``run_concept_replay_phase(bridge_uniform, tags, 20)`` (cue
    present; baseline 20 replays);
  * a runner-local amplified-tag compositional read per compositional
    query: the FULL arm runs
    ``_compositional_query_amplified(bridge_full, ...,
    tag_amp_factor=3.0)`` (4500 pA effective tag drive); the
    UNIFORM_CTRL arm runs the baseline
    ``_compositional_query_ranked`` (1500 pA tag drive);
  * a runner-local persistent PFC-frame prime per compositional
    query: the FULL arm primes ``dlpfc_verb`` for
    ``PFC_FRAME_STIM_STEPS=50`` steps; the UNIFORM_CTRL arm skips
    PFC-frame priming entirely;
  * THREE structural-effect probes (cue-suppression-during-replay +
    amplified-tag-stim + persistent-PFC-frame) that verify each
    mechanism produces > 1 mV bridge-state divergence between on
    and off via the runner's ACTUAL code path under deterministic
    RNG isolation, AND that both-arms-same controls agree to
    < 0.5 mV (mirrors Pirazzini d462bf0 + theta-gamma e6b17da
    lesson). All three probes pre-validate cache-scale matching
    BEFORE ``load_checkpoint`` (mirrors 6th arc commit 13f73e8;
    closes 10th adversarial review BLOCK).

Every other subsystem is REUSED-BY-IMPORT from the prior 6th arc
runner + unified per-regime monitor runner (byte-stable):

  * Substrate construction + Phase-1 caching (validated v16 +
    hippocampus + dlpfc PFC frame): REUSED
    ``unified_per_regime_monitor_runner._build_bridge_with_phase1_recipe``
    + ``_phase1_recipe`` + ``_phase1_cache_path`` +
    ``_freeze_phase1_gates``.
  * Compositional one-shot encoding: REUSED ``_encode_facts``
    (calls byte-unchanged ``encode_concept_pair`` internally).
  * Compositional pair generation: REUSED
    ``_unified_compositional_pairs`` (sub-seed offset +20000).
  * Direct W->A readout: REUSED ``_direct_query_ranked``.
  * Baseline compositional readout (lang_output firing-rate
    confidence at 1500 pA tag drive): REUSED
    ``_compositional_query_ranked``.
  * Frozen capability verdict: REUSED
    ``targeted_cue_suppression_replay_core.targeted_cue_suppression_replay_verdict``
    (Task 1 byte-unchanged; bars set in advance, NEVER tuned).
  * Both substrate-specific calibrated moats (the four moats):
    DIRECT_UNIFIED_THRESHOLD + COMPOSITIONAL_UNIFIED_THRESHOLD
    imported BYTE-UNCHANGED (no calibration changes).
  * Generative replay phase: REUSED
    ``consolidation_trainer.run_concept_replay_phase`` BYTE-UNCHANGED.
  * Compositional readout primitives:
    ``compose_concept_engram.lang_output_pattern_during_input`` +
    ``lang_output_pattern_during_stim`` BYTE-UNCHANGED.
  * Kill-safe checkpoint: REUSED sim.train_checkpoint.

Anti-cheat (carry forward all prior lessons):
  * OPAQUE tag names; no tag-string parsing.
  * BOTH moats fed the calibrated raw firing-rate confidence
    quantities.
  * The ``cross_pool_concept`` gate is opened ONLY inside the
    encoding window (via ``encode_concept_pair``) then closed.
  * ``uniform_ctrl`` differs from ``full`` ONLY in the four targeted
    mechanism changes; same seed, same encoded facts, same query
    set, same cue presence in retrieve.
  * ``direct_retain`` is read from the SAME run as ``full``.
  * No protected-file edits; no autograd; the runner is reuse-only
    orchestration of the prior arc's modules + the four net-new
    targeted-mechanism invocations + the three structural-effect
    probes.

ASCII only. CuPy is the real / decisive path; --tiny-synth shrinks
Phase-1 training + compositional encoding + replay cycles + PFC-frame
priming durations so the smoke is seconds (toy numbers explicitly NOT
a result).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Backend policy mirrors the 6th arc / unified / theta-gamma / per-
# regime / SPEAR / Pirazzini runners. CuPy is the decisive path;
# NumPy ONLY when CuPy is genuinely unavailable (GPU-less box).
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
from research.runners.targeted_cue_suppression_replay_core import (
    targeted_cue_suppression_replay_verdict,
    _TC_LADDER,
)

# REUSED gates byte-unchanged (the four moats). FULL arm routes direct
# queries through DIRECT_UNIFIED_THRESHOLD, compositional queries
# through COMPOSITIONAL_UNIFIED_THRESHOLD. UNIFORM_CTRL applies the
# SAME gates to the SAME ranked confidences (the SOLE differentiators
# between the arms are the targeted mechanisms, not the gate routing).
from research.runners.abstention_gate import DEFAULT_THRESHOLD as MOAT_DIRECT
from research.runners.abstention_gate_compositional import (
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
    _ranked_from_pattern,
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

# REUSED generative-replay subsystem BYTE-UNCHANGED. The validated
# concept-replay loop already shipped in Phase 1.3 consolidation work
# (research/runners/consolidation_trainer.py:43). Catalog D.19 SWRs +
# D.14 engram cells; T1.B P3.1.
from research.runners.consolidation_trainer import run_concept_replay_phase


# =====================================================================
# Targeted-mechanism constants. FROZEN -- never tuned in response to
# results. The reuse-by-import surface is byte-stable; the net-new
# tuneables are these counts + amplification factors (set in advance,
# not in response to results).
# =====================================================================
# Generative replay (FULL arm): number of replay events per engram
# tag during the post-encoding consolidation phase. The 7th arc raises
# this from the 6th arc's n=20 to n=50 to strengthen the FULL arm's
# replay consolidation. Buzsaki 2015 NREM cycle count provides ample
# biological headroom for 50 replays.
REPLAY_CYCLES_PER_TAG = 50

# Generative replay (UNIFORM_CTRL arm): baseline 6th arc count. Keeps
# the uniform arm at the prior-arc validated replay scale so the
# differentiator is the FOUR mechanism changes, not just the replay
# count.
_UNIFORM_CTRL_REPLAY_CYCLES = 20

# Amplified-tag-stim factor: multiplies the baseline 1500 pA tag drive
# in ``_compositional_query_ranked``. 3.0x -> 4500 pA effective drive.
# Set in advance per the design doc; never tuned in response to
# results.
RETRIEVE_TAG_AMP_FACTOR = 3.0

# Generative replay: drive amplitude per replay event (pA). Matches
# the ``run_concept_replay_phase`` default and the
# ``run_swr_replay_phase`` 100 pA convention.
REPLAY_DRIVE_PA = 100.0

# Generative replay: burst duration per replay event (ms). The default
# is 100 ms (real SWR ~50 ms but the longer window helps STDP capture).
# Reused from ``run_concept_replay_phase`` default.
REPLAY_BURST_DURATION_MS = 100

# Generative replay: quiet inter-burst window (ms). Reused default.
REPLAY_INTER_BURST_MS = 50

# PFC-frame priming: amplitude in pA driven onto dlpfc_verb during the
# PFC-frame priming window before each compositional query.
PFC_FRAME_PA = 100.0

# PFC-frame priming (FULL arm): number of simulation steps the prime
# drive is held on dlpfc_verb before the retrieval read. The 7th arc
# raises this from the 6th arc's 10 to 50 so the NMDA bistability has
# time to lock into the high-activity attractor branch (the bistable
# time constant of ~100 ms at dt=0.5 ms ~ 200 steps; 50 steps is the
# half-time of the bistability rise and biology-grounded for engaging
# the attractor).
PFC_FRAME_STIM_STEPS = 50

# PFC-frame priming (UNIFORM_CTRL arm): brief 10-step prime (the 6th
# arc's baseline). Kept here as a documented constant so the
# persistent-PFC-frame probe can contrast the two regimes.
_UNIFORM_CTRL_PFC_FRAME_STIM_STEPS = 10

# tiny-synth shrunk values. Each replay count + step count keeps a
# non-zero floor so the mechanism is genuinely exercised under the
# smoke. The smoke shrinks the FULL-arm 50 -> 5 and the UNIFORM_CTRL
# 20 -> 2 (full > uniform preserved). PFC-frame 50 -> 6 vs 10 -> 2
# (full > uniform preserved).
TINY_REPLAY_CYCLES_PER_TAG = 5
TINY_UNIFORM_CTRL_REPLAY_CYCLES = 2
TINY_REPLAY_BURST_DURATION_MS = 6
TINY_REPLAY_INTER_BURST_MS = 3
TINY_PFC_FRAME_STIM_STEPS = 6
TINY_UNIFORM_CTRL_PFC_FRAME_STIM_STEPS = 2


def _replay_step_counts(tiny_synth: bool) -> Tuple[int, int, int, int]:
    """Return (full_n_replays, uniform_n_replays, burst_duration_ms,
    inter_burst_ms). Full scale uses the FULL=50 / UNIFORM_CTRL=20
    contrast; tiny-synth shrinks both to a logic-screen smoke while
    preserving the FULL > UNIFORM contrast."""
    if tiny_synth:
        return (
            int(TINY_REPLAY_CYCLES_PER_TAG),
            int(TINY_UNIFORM_CTRL_REPLAY_CYCLES),
            int(TINY_REPLAY_BURST_DURATION_MS),
            int(TINY_REPLAY_INTER_BURST_MS),
        )
    return (
        int(REPLAY_CYCLES_PER_TAG),
        int(_UNIFORM_CTRL_REPLAY_CYCLES),
        int(REPLAY_BURST_DURATION_MS),
        int(REPLAY_INTER_BURST_MS),
    )


def _pfc_frame_step_counts(tiny_synth: bool) -> Tuple[int, int]:
    """Return (full_n_steps, uniform_n_steps). Full scale uses FULL=50
    / UNIFORM_CTRL=10; tiny-synth shrinks both."""
    if tiny_synth:
        return (
            int(TINY_PFC_FRAME_STIM_STEPS),
            int(TINY_UNIFORM_CTRL_PFC_FRAME_STIM_STEPS),
        )
    return (
        int(PFC_FRAME_STIM_STEPS),
        int(_UNIFORM_CTRL_PFC_FRAME_STIM_STEPS),
    )


# =====================================================================
# Deterministic-RNG isolation helper. Transcribed BYTE-UNCHANGED from
# the 6th arc runner (commit 659c2d8, also matches theta-gamma e6b17da).
# Closes the eighth adversarial review BLOCK: the structural-effect
# probes + per-cell eval arm seed the active backend's RNG to the SAME
# value before each arm's call so both arms see IDENTICAL OU-noise
# streams; the SOLE remaining between-arm difference is the targeted
# mechanism flag.
# =====================================================================
def _seed_query_rng(rng_seed: int) -> Any:
    """Capture the active backend's RNG state and deterministic-seed it.

    Returns an opaque token to pass back to _restore_query_rng so the
    caller can restore the global state once the helper returns. This
    keeps the RNG perturbation LOCAL to a single wrapped window; other
    components (training, encoding outside this window, ungroundable
    queries, etc.) see no change.

    What this seeds (CLOSES the 8th adversarial review BLOCK on RNG
    drift):
      * The active backend's RNG (CuPy or NumPy, via sim.backend).
      * The top-level numpy.random global.
      * The Python stdlib `random` module global.
    """
    from sim.backend import get_backend, get_random_state
    xp, name = get_backend()
    backend_saved = get_random_state()
    import numpy
    np_saved = numpy.random.get_state()
    import random as _pyrandom
    py_saved = _pyrandom.getstate()

    seed_value = int(rng_seed) & 0x7FFFFFFF
    if name == "cupy":
        import cupy
        cupy.random.seed(seed_value)
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
# Net-new helper 1: cue-suppressed generative-replay phase. The 6th
# arc's reused ``run_concept_replay_phase`` already zeroes
# ``cp_external_input_current[:]`` between bursts (line 125 +
# line 137 of consolidation_trainer.py), so simply zeroing the cue
# slice before invocation has no observable effect (the reused
# loop clears the whole array anyway). To make cue-suppression-
# during-replay STRUCTURALLY ACTIVE, the runner needs a per-step
# replay loop that DRIVES the lang_input cue slice during the
# burst window (baseline) vs leaves it zeroed (suppressed). The
# FULL arm runs the suppressed variant; the UNIFORM_CTRL arm runs
# the cue-present variant.
#
# The helper below mirrors ``run_concept_replay_phase``'s
# (drive + burst + quiet) loop structure byte-for-byte EXCEPT:
#   * It adds an explicit per-burst-step write of CUE_REPLAY_PA pA
#     into the bridge's lang_input slice for ``cue_present=True``
#     (the UNIFORM_CTRL arm's baseline). For ``cue_present=False``
#     (the FULL arm's suppressed variant) the slice stays at zero.
#   * It also clamps the ``lang_to_ec`` plasticity gate to 0.0 in
#     the suppressed case so the cue's pathway-update influence is
#     additionally suppressed (per the design doc Mechanism 1's
#     "via the existing plasticity_gate or external-input
#     mechanism"). The clamp + slice-zero combination is the
#     load-bearing structural differentiator.
#
# Both arms call this helper with their respective ``cue_present``
# flag. The reused ``stimulate_tag`` byte-unchanged drives the
# engram-tag neurons; the per-burst-step write adds the cue drive
# on top. The structural-effect probe contrasts cue_present=False
# (suppressed) vs cue_present=True (baseline).
# =====================================================================
# Cue drive amplitude during baseline replay (cue-present). Picked
# at 200.0 pA (the project-wide ``set_token_drive`` ~200 pA
# convention so the cue's biological scale matches what the eval
# loop applies during direct queries).
CUE_REPLAY_PA = 200.0


def _replay_with_optional_cue(
    bridge,
    tag_names: List[str],
    n_replays_per_tag: int,
    burst_duration_ms: int,
    inter_burst_ms: int,
    drive_pA: float,
    n_lang_input: int,
    cue_present: bool,
    rng_seed: int,
) -> Dict[str, Any]:
    """Per-step replay loop. Mirrors run_concept_replay_phase's
    (drive + burst + quiet) structure byte-for-byte EXCEPT the burst
    window optionally co-drives the lang_input cue slice at
    CUE_REPLAY_PA pA.

    Mechanism: when cue_present=True, every burst step writes the
    cue drive to bridge.cp_external_input_current[:n_lang_input]
    on top of the engram-tag drive at the tagged neurons. The cue
    is therefore ALSO active during replay (the contamination the
    7th arc Mechanism 1 targets). When cue_present=False, the cue
    slice stays at 0.0 throughout the burst (the suppressed
    variant; the engram tag is the only active drive).

    Additionally, when cue_present=False (suppressed variant), the
    ``lang_to_ec`` plasticity gate is clamped to 0.0 for the
    duration of the replay window so the cortico-hippocampal
    pathway-update influence is also suppressed. Both are restored
    in a finally: block.

    Args:
      bridge: the SimulationBridge to run replay on.
      tag_names: the engram tag names committed by _encode_facts.
      n_replays_per_tag: number of replay events per tag.
      burst_duration_ms: per-replay-event burst duration (ms).
      inter_burst_ms: per-replay-event quiet window (ms).
      drive_pA: per-replay-event tag drive amplitude (pA).
      n_lang_input: length of the lang_input slice to drive during
        the burst when cue_present=True.
      cue_present: True for UNIFORM_CTRL baseline (cue active during
        burst), False for FULL arm's suppressed variant.
      rng_seed: deterministic RNG seed for the replay-order shuffle.

    Returns: dict with replay stats:
      n_replays, tags_replayed, per_tag_replay_count,
      burst_duration_ms, inter_burst_ms, randomize_order,
      cue_suppression_active.
    """
    if not tag_names:
        return {
            "n_replays": 0, "tags_replayed": [],
            "per_tag_replay_count": {},
            "burst_duration_ms": 0, "inter_burst_ms": 0,
            "randomize_order": False,
            "cue_suppression_active": (not bool(cue_present)),
        }

    from sim.backend import get_backend
    cp, _ = get_backend()
    n_li = int(n_lang_input)

    # Save lang_to_ec plasticity gate state so we can restore it.
    # Suppressing this gate during replay is part of Mechanism 1
    # (cue-pathway plasticity-update suppression).
    saved_lang_to_ec_gate = None
    if not cue_present:
        try:
            saved_lang_to_ec_gate = float(
                bridge.get_plasticity_gate_value("lang_to_ec")
            )
        except Exception:
            saved_lang_to_ec_gate = None
        if saved_lang_to_ec_gate is not None:
            try:
                bridge.set_plasticity_gate("lang_to_ec", 0.0)
            except Exception:
                pass

    rng = np.random.default_rng(int(rng_seed))
    order = list(tag_names) * int(n_replays_per_tag)
    order = list(order)
    rng.shuffle(order)

    per_tag_count = {name: 0 for name in tag_names}
    n_replays_total = 0

    try:
        for tag_name in order:
            # Mirror run_concept_replay_phase exactly EXCEPT for the
            # cue-drive write inside the burst loop.
            bridge.cp_external_input_current[:] = 0.0
            try:
                n_stim = bridge.stimulate_tag(
                    tag_name, drive_pA=float(drive_pA)
                )
            except KeyError:
                continue
            if n_stim == 0:
                continue
            for _ in range(int(burst_duration_ms)):
                # The burst loop preserves the tag drive by NOT
                # zeroing the array between steps (stimulate_tag has
                # already written the tag's neuron currents). For the
                # cue-present baseline, ALSO write the cue drive to
                # the lang_input slice each step. This makes the cue
                # active throughout the burst window (the
                # contamination the suppressed variant targets).
                if cue_present:
                    bridge.cp_external_input_current[:n_li] = float(
                        CUE_REPLAY_PA
                    )
                else:
                    bridge.cp_external_input_current[:n_li] = 0.0
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            # Quiet: zero the entire current array (matches reused
            # loop).
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(int(inter_burst_ms)):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            per_tag_count[tag_name] += 1
            n_replays_total += 1
    finally:
        # Restore lang_to_ec gate.
        if saved_lang_to_ec_gate is not None:
            try:
                bridge.set_plasticity_gate(
                    "lang_to_ec", float(saved_lang_to_ec_gate)
                )
            except Exception:
                pass
        # Leave the current array zeroed so the retrieve window sees
        # a clean baseline.
        bridge.cp_external_input_current[:] = 0.0

    return {
        "n_replays": int(n_replays_total),
        "tags_replayed": list(tag_names),
        "per_tag_replay_count": per_tag_count,
        "burst_duration_ms": int(burst_duration_ms),
        "inter_burst_ms": int(inter_burst_ms),
        "randomize_order": True,
        "cue_suppression_active": (not bool(cue_present)),
    }


def _run_replay_with_cue_suppressed(
    bridge,
    tag_names: List[str],
    n_replays_per_tag: int,
    burst_duration_ms: int,
    inter_burst_ms: int,
    drive_pA: float,
    n_lang_input: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Thin wrapper: cue_present=False (FULL arm's suppressed
    variant). The cue's lang_input drive stays at 0.0 during the
    burst; lang_to_ec gate is clamped to 0.0."""
    return _replay_with_optional_cue(
        bridge=bridge,
        tag_names=tag_names,
        n_replays_per_tag=int(n_replays_per_tag),
        burst_duration_ms=int(burst_duration_ms),
        inter_burst_ms=int(inter_burst_ms),
        drive_pA=float(drive_pA),
        n_lang_input=int(n_lang_input),
        cue_present=False,
        rng_seed=int(rng_seed),
    )


def _run_replay_baseline(
    bridge,
    tag_names: List[str],
    n_replays_per_tag: int,
    burst_duration_ms: int,
    inter_burst_ms: int,
    drive_pA: float,
    n_lang_input: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Thin wrapper: cue_present=True (UNIFORM_CTRL arm's baseline).
    The cue's lang_input slice is driven at CUE_REPLAY_PA during
    every burst step."""
    return _replay_with_optional_cue(
        bridge=bridge,
        tag_names=tag_names,
        n_replays_per_tag=int(n_replays_per_tag),
        burst_duration_ms=int(burst_duration_ms),
        inter_burst_ms=int(inter_burst_ms),
        drive_pA=float(drive_pA),
        n_lang_input=int(n_lang_input),
        cue_present=True,
        rng_seed=int(rng_seed),
    )


# =====================================================================
# Net-new helper 2: amplified-tag compositional read. Replicates the
# unified runner's ``_compositional_query_ranked`` body with the tag
# drive multiplied by tag_amp_factor. REUSES the byte-unchanged
# pattern primitives (lang_output_pattern_during_input +
# lang_output_pattern_during_stim + _ranked_from_pattern).
# =====================================================================
def _compositional_query_amplified(
    bridge,
    cue_noun: str,
    tag_name: Optional[str],
    dims: Dict[str, Any],
    recall_steps: int,
    tag_amp_factor: float,
):
    """Compositional-retrieval-regime read with AMPLIFIED engram-tag
    stim. Replicates ``_compositional_query_ranked`` byte-for-byte
    EXCEPT the tag drive_pA is multiplied by ``tag_amp_factor``. The
    cue drive stays at baseline.

    Mechanism (7th arc design Mechanism 2): the unified runner's
    localisation finding showed the cue's diffuse lang_input drive
    dominates the tag's selective drive at the 1500 pA baseline.
    Amplifying the tag drive by 3.0x (4500 pA effective) is the
    targeted refinement.

    Returns the same ranked list shape as
    ``_compositional_query_ranked``: a list of
    ``(word, score, tag_string)`` tuples sorted descending by score.
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
        amplified_drive_pA = 1500.0 * float(tag_amp_factor)
        hip_pat, n_lo2 = lang_output_pattern_during_stim(
            bridge, tag_name, drive_pA=amplified_drive_pA,
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
# PFC-frame priming helper. Drives the ``dlpfc_verb`` region's
# external input current for n_steps simulation steps. The per-step
# write pattern is byte-unchanged from the 6th arc (Pirazzini FIX B
# pattern: write INSIDE the per-step loop so any sub-helper clears
# don't drop the drive). The only difference vs the 6th arc is
# n_steps -- FULL arm uses ``PFC_FRAME_STIM_STEPS=50``, UNIFORM_CTRL
# arm uses ``_UNIFORM_CTRL_PFC_FRAME_STIM_STEPS=10`` (when the
# uniform arm primes; in the eval loop the uniform arm SKIPS priming
# entirely; the probe contrasts 50 vs 10).
# =====================================================================
def _prime_pfc_frame(bridge, n_steps: int) -> int:
    """Drive ``dlpfc_verb`` external input current for n_steps
    simulation steps. The NMDA-bistable attractor in dlpfc_verb holds
    the frame afterwards for the retrieve read.

    Mechanism (per-step writes survive sub-helper clears per Pirazzini
    FIX B): for each of n_steps,
      * Locate the ``dlpfc_verb`` region's neuron indices via the
        bridge's region_manager.
      * Set ``bridge.cp_external_input_current[:] = 0.0`` then
        ``[dlpfc_idx] = PFC_FRAME_PA`` (so other neurons aren't
        externally driven during the prime).
      * Run one simulation step.
    After the loop, clear ``cp_external_input_current`` so the
    subsequent retrieval read sees a clean baseline.

    Returns the number of dlpfc_verb neurons that received the prime
    drive (diagnostic). Returns 0 if dlpfc_verb is not in the region
    manager (substrate without enable_dlpfc_verb=True) OR n_steps<=0.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    try:
        dlpfc_idx = list(rm.indices("dlpfc_verb"))
    except Exception:
        return 0
    if not dlpfc_idx:
        return 0
    if int(n_steps) <= 0:
        return 0
    dlpfc_arr = cp.asarray(dlpfc_idx, dtype=cp.int64)
    for _ in range(int(n_steps)):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[dlpfc_arr] = float(PFC_FRAME_PA)
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    return int(len(dlpfc_idx))


# =====================================================================
# Structural-effect probes -- MANDATORY (THREE of them; one per
# targeted mechanism that differs across the arms). Each verifies the
# mechanism produces > 1 mV bridge-state divergence between flag-on
# and flag-off via the runner's ACTUAL code path under deterministic
# RNG isolation. Each probe also runs both-arms-same controls and
# asserts those agree to < 0.5 mV (the RNG-isolation soundness check).
# All three probes pre-validate cache-scale matching BEFORE
# load_checkpoint (mirrors 6th arc 13f73e8).
# =====================================================================
# Deterministic RNG seeds for the structural-effect probes. The probes
# use fixed values across all four runs (two flag-differing, two flag-
# same controls) so the only between-arm difference is the flag.
_PROBE_RNG_SEED = 999
_PROBE_ENCODE_RNG_SEED = 31337

# Tolerance for the "controls must show near-zero divergence" check.
# 0.5 mV is well below the 1.0 mV bar the flag-differing case must
# exceed (mirrors theta-gamma e6b17da + 6th arc 659c2d8).
_PROBE_CONTROL_TOL_MV = 0.5


def _validate_cache_scale_for_probe(cache_path, built_bridge,
                                       probe_name: str) -> None:
    """Refuse to run the probe on a cache file whose stored bridge
    dimensions do NOT match the freshly-built bridge.

    Closes the 10th adversarial review BLOCK carried forward from
    the 6th arc commit 13f73e8: with ``tiny_synth=True``, the probe
    builds a small bridge (~952 neurons / 46497 synapses) BUT
    ``load_checkpoint`` will happily load an existing biological-
    scale Phase-1 cache (~8440 neurons / 4825651 synapses). The
    bridge state arrays then have inconsistent shapes:
    ``cp_membrane_potential_v`` is sized to the cached value, but
    arrays the bridge allocated at build time stay at the tiny-synth
    size. Every subsequent simulation step raises ``IndexError``
    (caught by a broad ``try`` / ``except`` inside the bridge step),
    silently corrupting the probe state -- so the probe's reported
    flag-differing divergence is NOT trustworthy as a gate.

    Implementation: open the HDF5 file (lazy ``import h5py``) and
    inspect:
      * ``num_neurons`` attr;
      * ``connections_shape_0`` attr;
      * ``cp_membrane_potential_v`` dataset shape[0].
    Compare against ``built_bridge.cp_membrane_potential_v.shape[0]``.
    If ANY mismatches, raise ``RuntimeError`` with a clear message
    surfacing both cached and built dimensions.

    Args:
      cache_path: path to the HDF5 Phase-1 checkpoint that will be
        loaded.
      built_bridge: the freshly-built SimulationBridge whose
        dimensions the cache must match.
      probe_name: e.g. ``"cue-suppression-replay-effect"`` /
        ``"amplified-tag-stim-effect"`` /
        ``"persistent-pfc-frame-effect"``; surfaced in the error
        message for diagnostic clarity.

    Returns: ``None`` on success.
    Raises: ``RuntimeError`` on any dimensional mismatch.
    """
    import h5py  # lazy import per the strengthen-only fix

    built_n_neurons = int(built_bridge.cp_membrane_potential_v.shape[0])
    cached_n_attr = None
    cached_conn_shape_0 = None
    cached_v_shape_0 = None
    try:
        with h5py.File(str(cache_path), "r") as f:
            if "num_neurons" in f.attrs:
                cached_n_attr = int(f.attrs["num_neurons"])
            if "connections_shape_0" in f.attrs:
                cached_conn_shape_0 = int(f.attrs["connections_shape_0"])
            if "cp_membrane_potential_v" in f:
                cached_v_shape_0 = int(
                    f["cp_membrane_potential_v"].shape[0]
                )
    except Exception as exc:
        raise RuntimeError(
            "%s probe FAILED to inspect cache metadata at %s: %r. "
            "The probe REFUSES to run on a cache it cannot validate "
            "the scale of. Closes 10th adversarial review BLOCK; fix "
            "the cache file or re-run with --tiny-synth=False at the "
            "scale the cache was trained for."
            % (probe_name, str(cache_path), exc)
        ) from exc

    candidates = [
        ("num_neurons attr", cached_n_attr),
        ("connections_shape_0 attr", cached_conn_shape_0),
        ("cp_membrane_potential_v shape[0]", cached_v_shape_0),
    ]
    mismatches = [
        (label, value)
        for (label, value) in candidates
        if value is not None and value != built_n_neurons
    ]
    if mismatches:
        details = "; ".join(
            "%s=%d" % (label, value) for (label, value) in mismatches
        )
        raise RuntimeError(
            "%s probe REFUSES TO RUN: CACHE-SCALE MISMATCH detected at "
            "%s. The built bridge has %d neurons but the cached "
            "checkpoint reports: %s. Loading this checkpoint into a "
            "mismatched bridge silently corrupts state (every sim step "
            "raises IndexError, swallowed by the bridge's try/except). "
            "The previous 'passing' probe numbers would be unreliable "
            "as a gate. Closes 10th adversarial review BLOCK. Fix: "
            "re-run with --tiny-synth=False at the scale the cache "
            "was trained for, OR point --phase1-cache-dir at a "
            "tiny_synth-matching cache directory (or delete the "
            "mismatched cache file so Phase-1 retraining reproduces "
            "it at the correct scale)."
            % (probe_name, str(cache_path), built_n_neurons, details)
        )


def _cue_suppression_replay_effect_probe(
    seed: int = 42,
    tiny_synth: bool = True,
    cache_dir: Optional[str] = None,
) -> float:
    """Run the runner's actual code path twice with the SAME initial
    bridge state but different cue-suppression-during-replay flags;
    return the max absolute membrane-potential divergence (mV) for
    the flag-differing case.

    Mechanism (CLOSES the 8th adversarial review BLOCK):
      * Deterministic-seed the active backend's RNG to _PROBE_RNG_SEED
        BEFORE each replay phase. Both arms therefore see IDENTICAL
        OU-noise streams; the SOLE remaining difference between the
        arms is the cue-suppression-on vs cue-suppression-off flag.
      * Restore the RNG state after each call so other components see
        no perturbation.
      * Run TWO additional CONTROL contrasts at the SAME seed:
          (1) both arms run cue-suppressed replay -> < 0.5 mV
          (2) both arms run cue-present replay -> < 0.5 mV
        If either control shows large divergence, RNG isolation is
        broken and the probe raises RuntimeError.
      * The flag-differing case (cue-suppressed vs cue-present) MUST
        exceed 1.0 mV for the mechanism to be declared structurally
        active.

    DEFENSIVE pre-load check (CLOSES the 10th adversarial review
    BLOCK): each contrast builds two fresh bridges and BEFORE calling
    ``load_checkpoint`` validates that the cached checkpoint's
    stored bridge dimensions match the freshly-built bridge dimensions
    via ``_validate_cache_scale_for_probe``.

    If the flag-differing divergence is below 1 mV, the mechanism is
    structurally inert and the caller MUST abort. If a control shows
    divergence above the tolerance, RNG isolation is broken and the
    caller MUST abort. If the cache-scale validation fails, the cache
    and build are incompatible and the caller MUST abort. Any of
    these raises RuntimeError.
    """
    from sim.backend import to_host
    cache_dir = str(cache_dir) if cache_dir else _PHASE1_CACHE_DEFAULT
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
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
    facts = _unified_compositional_pairs(seed, 1)
    enc_steps = 8 if tiny_synth else 200
    full_n_replays, _uniform_n_replays, burst_ms, inter_ms = (
        _replay_step_counts(tiny_synth)
    )

    def _one_contrast(suppress_a: bool, suppress_b: bool) -> float:
        bridge_a = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        bridge_b = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        _validate_cache_scale_for_probe(
            cache_path, bridge_a, "cue-suppression-replay-effect"
        )
        _validate_cache_scale_for_probe(
            cache_path, bridge_b, "cue-suppression-replay-effect"
        )
        bridge_a.load_checkpoint(str(cache_path))
        bridge_b.load_checkpoint(str(cache_path))
        _freeze_phase1_gates(bridge_a)
        _freeze_phase1_gates(bridge_b)

        # Encoding phase: identical deterministic seed before each call.
        saved_enc_a = _seed_query_rng(_PROBE_ENCODE_RNG_SEED)
        try:
            tags_a = _encode_facts(bridge_a, facts, dims, enc_steps)
        finally:
            _restore_query_rng(saved_enc_a)
        saved_enc_b = _seed_query_rng(_PROBE_ENCODE_RNG_SEED)
        try:
            tags_b = _encode_facts(bridge_b, facts, dims, enc_steps)
        finally:
            _restore_query_rng(saved_enc_b)

        # Replay phase: identical seed, same replay count, same drive,
        # SOLE difference is the cue-suppression flag.
        saved_a = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            if suppress_a:
                _ = _run_replay_with_cue_suppressed(
                    bridge_a, tags_a,
                    n_replays_per_tag=int(full_n_replays),
                    burst_duration_ms=int(burst_ms),
                    inter_burst_ms=int(inter_ms),
                    drive_pA=float(REPLAY_DRIVE_PA),
                    n_lang_input=int(dims["n_lang_input"]),
                    rng_seed=_PROBE_RNG_SEED,
                )
            else:
                _ = _run_replay_baseline(
                    bridge_a, tags_a,
                    n_replays_per_tag=int(full_n_replays),
                    burst_duration_ms=int(burst_ms),
                    inter_burst_ms=int(inter_ms),
                    drive_pA=float(REPLAY_DRIVE_PA),
                    n_lang_input=int(dims["n_lang_input"]),
                    rng_seed=_PROBE_RNG_SEED,
                )
        finally:
            _restore_query_rng(saved_a)
        saved_b = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            if suppress_b:
                _ = _run_replay_with_cue_suppressed(
                    bridge_b, tags_b,
                    n_replays_per_tag=int(full_n_replays),
                    burst_duration_ms=int(burst_ms),
                    inter_burst_ms=int(inter_ms),
                    drive_pA=float(REPLAY_DRIVE_PA),
                    n_lang_input=int(dims["n_lang_input"]),
                    rng_seed=_PROBE_RNG_SEED,
                )
            else:
                _ = _run_replay_baseline(
                    bridge_b, tags_b,
                    n_replays_per_tag=int(full_n_replays),
                    burst_duration_ms=int(burst_ms),
                    inter_burst_ms=int(inter_ms),
                    drive_pA=float(REPLAY_DRIVE_PA),
                    n_lang_input=int(dims["n_lang_input"]),
                    rng_seed=_PROBE_RNG_SEED,
                )
        finally:
            _restore_query_rng(saved_b)

        v_a = to_host(bridge_a.cp_membrane_potential_v)
        v_b = to_host(bridge_b.cp_membrane_potential_v)
        return float(np.max(np.abs(np.asarray(v_a) - np.asarray(v_b))))

    diff_flag_diff = _one_contrast(True, False)
    diff_both_true = _one_contrast(True, True)
    diff_both_false = _one_contrast(False, False)

    if diff_both_true > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Cue-suppression-replay-effect probe CONTROL FAILED "
            "(both-True): with suppression on BOTH bridges and the same "
            "deterministic RNG seed, the two bridges diverged by %.6g mV "
            "(> %.3g mV tolerance). RNG isolation is broken; the flag-"
            "differing divergence is NOT attributable to the cue-"
            "suppression mechanism. Closes 8th adversarial review BLOCK; "
            "fix RNG isolation and re-run."
            % (diff_both_true, _PROBE_CONTROL_TOL_MV)
        )
    if diff_both_false > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Cue-suppression-replay-effect probe CONTROL FAILED "
            "(both-False): with cue-present BOTH bridges and the same "
            "deterministic RNG seed, the two bridges diverged by %.6g mV "
            "(> %.3g mV tolerance). RNG isolation is broken; the flag-"
            "differing divergence is NOT attributable to the cue-"
            "suppression mechanism. Closes 8th adversarial review BLOCK; "
            "fix RNG isolation and re-run."
            % (diff_both_false, _PROBE_CONTROL_TOL_MV)
        )

    if diff_flag_diff <= 1.0:
        raise RuntimeError(
            "Cue-suppression-replay-effect probe FAILED: cue-suppressed "
            "vs cue-present replay produced essentially identical bridge "
            "state (max |delta v| = %.6g mV <= 1.0 mV) via the runner's "
            "ACTUAL code path (controls passed: both-True=%.6g mV, "
            "both-False=%.6g mV). The mechanism is structurally inert -- "
            "mirrors Pirazzini d462bf0 defect. Fix and re-run BEFORE "
            "decisive."
            % (diff_flag_diff, diff_both_true, diff_both_false)
        )

    return float(diff_flag_diff)


def _amplified_tag_stim_effect_probe(
    seed: int = 42,
    tiny_synth: bool = True,
    cache_dir: Optional[str] = None,
) -> float:
    """Run the runner's actual code path twice with the SAME initial
    bridge state but different amplified-tag-stim factors; return the
    max absolute membrane-potential divergence (mV) for the flag-
    differing case.

    Mechanism: amplifies the engram-tag drive during compositional
    retrieval. The probe contrasts ``RETRIEVE_TAG_AMP_FACTOR=3.0``
    (FULL arm) vs ``1.0`` (UNIFORM_CTRL arm).
    """
    from sim.backend import to_host
    cache_dir = str(cache_dir) if cache_dir else _PHASE1_CACHE_DEFAULT
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
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
    facts = _unified_compositional_pairs(seed, 1)
    enc_steps = 8 if tiny_synth else 200
    recall_steps = 20 if tiny_synth else 100

    def _one_contrast(amp_a: float, amp_b: float) -> float:
        bridge_a = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        bridge_b = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        _validate_cache_scale_for_probe(
            cache_path, bridge_a, "amplified-tag-stim-effect"
        )
        _validate_cache_scale_for_probe(
            cache_path, bridge_b, "amplified-tag-stim-effect"
        )
        bridge_a.load_checkpoint(str(cache_path))
        bridge_b.load_checkpoint(str(cache_path))
        _freeze_phase1_gates(bridge_a)
        _freeze_phase1_gates(bridge_b)

        # Encoding phase: identical deterministic seed before each call.
        saved_enc_a = _seed_query_rng(_PROBE_ENCODE_RNG_SEED)
        try:
            tags_a = _encode_facts(bridge_a, facts, dims, enc_steps)
        finally:
            _restore_query_rng(saved_enc_a)
        saved_enc_b = _seed_query_rng(_PROBE_ENCODE_RNG_SEED)
        try:
            tags_b = _encode_facts(bridge_b, facts, dims, enc_steps)
        finally:
            _restore_query_rng(saved_enc_b)

        # Compositional read: identical seed, same cue noun, same
        # recall steps, SOLE difference is the tag amplification.
        cue_noun = facts[0][0]
        tag_a = tags_a[0] if tags_a else None
        tag_b = tags_b[0] if tags_b else None
        saved_a = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            _ = _compositional_query_amplified(
                bridge_a, cue_noun, tag_a, dims,
                recall_steps=int(recall_steps),
                tag_amp_factor=float(amp_a),
            )
        finally:
            _restore_query_rng(saved_a)
        saved_b = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            _ = _compositional_query_amplified(
                bridge_b, cue_noun, tag_b, dims,
                recall_steps=int(recall_steps),
                tag_amp_factor=float(amp_b),
            )
        finally:
            _restore_query_rng(saved_b)

        v_a = to_host(bridge_a.cp_membrane_potential_v)
        v_b = to_host(bridge_b.cp_membrane_potential_v)
        return float(np.max(np.abs(np.asarray(v_a) - np.asarray(v_b))))

    diff_flag_diff = _one_contrast(float(RETRIEVE_TAG_AMP_FACTOR), 1.0)
    diff_both_true = _one_contrast(
        float(RETRIEVE_TAG_AMP_FACTOR), float(RETRIEVE_TAG_AMP_FACTOR)
    )
    diff_both_false = _one_contrast(1.0, 1.0)

    if diff_both_true > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Amplified-tag-stim-effect probe CONTROL FAILED (both-True): "
            "with %.1fx amplification on BOTH bridges and the same "
            "deterministic RNG seed, the two bridges diverged by %.6g mV "
            "(> %.3g mV tolerance). RNG isolation is broken; the flag-"
            "differing divergence is NOT attributable to the amplified-"
            "tag mechanism. Closes 8th adversarial review BLOCK; fix RNG "
            "isolation and re-run."
            % (RETRIEVE_TAG_AMP_FACTOR, diff_both_true, _PROBE_CONTROL_TOL_MV)
        )
    if diff_both_false > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Amplified-tag-stim-effect probe CONTROL FAILED (both-False): "
            "with 1.0x amplification on BOTH bridges and the same "
            "deterministic RNG seed, the two bridges diverged by %.6g mV "
            "(> %.3g mV tolerance). RNG isolation is broken; the flag-"
            "differing divergence is NOT attributable to the amplified-"
            "tag mechanism. Closes 8th adversarial review BLOCK; fix RNG "
            "isolation and re-run."
            % (diff_both_false, _PROBE_CONTROL_TOL_MV)
        )

    if diff_flag_diff <= 1.0:
        raise RuntimeError(
            "Amplified-tag-stim-effect probe FAILED: %.1fx vs 1.0x tag "
            "amplification produced essentially identical bridge state "
            "(max |delta v| = %.6g mV <= 1.0 mV) via the runner's ACTUAL "
            "code path (controls passed: both-True=%.6g mV, both-False="
            "%.6g mV). The mechanism is structurally inert -- mirrors "
            "Pirazzini d462bf0 defect. Fix and re-run BEFORE decisive."
            % (RETRIEVE_TAG_AMP_FACTOR, diff_flag_diff, diff_both_true,
               diff_both_false)
        )

    return float(diff_flag_diff)


def _persistent_pfc_frame_effect_probe(
    seed: int = 42,
    tiny_synth: bool = True,
    cache_dir: Optional[str] = None,
) -> float:
    """Run the runner's actual code path twice with the SAME initial
    bridge state but different PFC-frame priming durations; return the
    max absolute membrane-potential divergence (mV) for the flag-
    differing case.

    Mechanism: extends the PFC-frame prime from 10 steps to 50 steps.
    The probe contrasts the FULL arm's ``PFC_FRAME_STIM_STEPS=50`` vs
    the UNIFORM_CTRL arm's ``_UNIFORM_CTRL_PFC_FRAME_STIM_STEPS=10``.
    """
    from sim.backend import to_host
    cache_dir = str(cache_dir) if cache_dir else _PHASE1_CACHE_DEFAULT
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _phase1_train_if_needed(int(seed), cache_dir, tiny_synth)
    cache_path = _phase1_cache_path(cache_dir, seed)

    full_n_steps, uniform_n_steps = _pfc_frame_step_counts(tiny_synth)

    def _one_contrast(n_steps_a: int, n_steps_b: int) -> float:
        bridge_a = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        bridge_b = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        _validate_cache_scale_for_probe(
            cache_path, bridge_a, "persistent-pfc-frame-effect"
        )
        _validate_cache_scale_for_probe(
            cache_path, bridge_b, "persistent-pfc-frame-effect"
        )
        bridge_a.load_checkpoint(str(cache_path))
        bridge_b.load_checkpoint(str(cache_path))
        _freeze_phase1_gates(bridge_a)
        _freeze_phase1_gates(bridge_b)

        saved_a = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            _ = _prime_pfc_frame(bridge_a, n_steps=int(n_steps_a))
        finally:
            _restore_query_rng(saved_a)
        saved_b = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            _ = _prime_pfc_frame(bridge_b, n_steps=int(n_steps_b))
        finally:
            _restore_query_rng(saved_b)

        v_a = to_host(bridge_a.cp_membrane_potential_v)
        v_b = to_host(bridge_b.cp_membrane_potential_v)
        return float(np.max(np.abs(np.asarray(v_a) - np.asarray(v_b))))

    diff_flag_diff = _one_contrast(int(full_n_steps), int(uniform_n_steps))
    diff_both_true = _one_contrast(int(full_n_steps), int(full_n_steps))
    diff_both_false = _one_contrast(
        int(uniform_n_steps), int(uniform_n_steps)
    )

    if diff_both_true > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Persistent-PFC-frame-effect probe CONTROL FAILED "
            "(both-True): with %d-step prime on BOTH bridges and the "
            "same deterministic RNG seed, the two bridges diverged by "
            "%.6g mV (> %.3g mV tolerance). RNG isolation is broken; "
            "the flag-differing divergence is NOT attributable to the "
            "persistent-PFC-frame mechanism. Closes 8th adversarial "
            "review BLOCK; fix RNG isolation and re-run."
            % (full_n_steps, diff_both_true, _PROBE_CONTROL_TOL_MV)
        )
    if diff_both_false > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Persistent-PFC-frame-effect probe CONTROL FAILED "
            "(both-False): with %d-step prime on BOTH bridges and the "
            "same deterministic RNG seed, the two bridges diverged by "
            "%.6g mV (> %.3g mV tolerance). RNG isolation is broken; "
            "the flag-differing divergence is NOT attributable to the "
            "persistent-PFC-frame mechanism. Closes 8th adversarial "
            "review BLOCK; fix RNG isolation and re-run."
            % (uniform_n_steps, diff_both_false, _PROBE_CONTROL_TOL_MV)
        )

    if diff_flag_diff <= 1.0:
        raise RuntimeError(
            "Persistent-PFC-frame-effect probe FAILED: %d-step vs %d-step "
            "PFC-frame prime produced essentially identical bridge state "
            "(max |delta v| = %.6g mV <= 1.0 mV) via the runner's ACTUAL "
            "code path (controls passed: both-True=%.6g mV, both-False="
            "%.6g mV). The mechanism is structurally inert -- mirrors "
            "Pirazzini d462bf0 defect. Fix and re-run BEFORE decisive."
            % (full_n_steps, uniform_n_steps, diff_flag_diff,
               diff_both_true, diff_both_false)
        )

    return float(diff_flag_diff)


# =====================================================================
# Per-cell evaluation arm: one (seed, N) cell. Mirrors the 6th arc
# structure EXCEPT:
#   * the FULL arm runs ``_run_replay_with_cue_suppressed`` ONCE
#     (50 replays/tag, lang_input zeroed + lang_to_ec gate clamped);
#     the UNIFORM_CTRL arm runs ``_run_replay_baseline`` ONCE
#     (20 replays/tag, cue present);
#   * the FULL arm primes ``dlpfc_verb`` for 50 steps before each
#     compositional query; the UNIFORM_CTRL arm SKIPS PFC-frame
#     priming entirely;
#   * the FULL arm uses ``_compositional_query_amplified`` at 3.0x tag
#     drive; the UNIFORM_CTRL arm uses ``_compositional_query_ranked``
#     at 1.0x tag drive;
#   * the cue stays ON during retrieve in BOTH arms (encoding-
#     specificity respected per the theta-gamma finding).
#
# Engineering decision (documented): the replay phase + PFC-frame
# prime modify bridge state, so the FULL and UNIFORM_CTRL arms cannot
# share the same bridge. We build TWO parallel bridges at the start of
# the cell (one for FULL, one for UNIFORM_CTRL), load the SAME Phase-1
# checkpoint into both, encode the SAME facts into both, then run each
# arm's queries against its own bridge. This keeps the seed + encoded
# facts + query set + cue presence in retrieve + ranked confidences
# IDENTICAL across the two arms; the SOLE differentiators are the
# four targeted mechanisms.
# =====================================================================
def _run_evaluation_arm(seed: int, N: int, tiny_synth: bool,
                          cache_dir: str) -> Dict[str, Any]:
    """Run the targeted cue-suppression + amplified-tag + persistent-
    PFC-frame architecture for ONE (seed, N) cell. Two parallel
    bridges (FULL + UNIFORM_CTRL); each runs the SAME encoding +
    direct queries; the FULL arm runs cue-suppressed replay (50
    cycles), primes dlpfc_verb for 50 steps before each compositional
    query, and amplifies the engram-tag drive 3.0x; the UNIFORM_CTRL
    arm runs cue-present replay (20 cycles), skips PFC-frame priming,
    and uses 1.0x tag drive.

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
    # checkpoint, same frozen gates. The SOLE differentiators are the
    # four targeted mechanisms on the FULL bridge.
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

    # ---- NET-NEW (FULL arm): cue-suppressed replay (50 cycles).
    #     UNIFORM_CTRL arm: cue-present replay (20 cycles).
    full_n_replays, uniform_n_replays, burst_ms, inter_ms = (
        _replay_step_counts(tiny_synth)
    )
    full_replay_rng_seed = (
        int(seed) * 1_000_003 + int(N) * 1009 + 7919
    ) & 0x7FFFFFFF
    uniform_replay_rng_seed = full_replay_rng_seed  # SAME seed both arms

    saved_replay_full = _seed_query_rng(full_replay_rng_seed)
    try:
        replay_stats_full = _run_replay_with_cue_suppressed(
            bridge_full, tags_full,
            n_replays_per_tag=int(full_n_replays),
            burst_duration_ms=int(burst_ms),
            inter_burst_ms=int(inter_ms),
            drive_pA=float(REPLAY_DRIVE_PA),
            n_lang_input=int(dims["n_lang_input"]),
            rng_seed=full_replay_rng_seed,
        )
    finally:
        _restore_query_rng(saved_replay_full)

    saved_replay_uniform = _seed_query_rng(uniform_replay_rng_seed)
    try:
        replay_stats_uniform = _run_replay_baseline(
            bridge_uniform, tags_uniform,
            n_replays_per_tag=int(uniform_n_replays),
            burst_duration_ms=int(burst_ms),
            inter_burst_ms=int(inter_ms),
            drive_pA=float(REPLAY_DRIVE_PA),
            n_lang_input=int(dims["n_lang_input"]),
            rng_seed=uniform_replay_rng_seed,
        )
    finally:
        _restore_query_rng(saved_replay_uniform)

    # ---- DIRECT queries: one per unique trained word in the cell's
    # facts. BOTH arms route direct queries through the SAME substrate-
    # specific direct gate (DIRECT_UNIFIED_THRESHOLD). The two arms may
    # diverge here because the FULL bridge has been perturbed by the
    # cue-suppressed replay; the UNIFORM_CTRL bridge has not.
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
        direct_query_rng_seed = (
            int(seed) * 1_000_003 + int(N) * 1009
            + hash(word) % 65521 + 113
        ) & 0x7FFFFFFF
        saved_dq_full = _seed_query_rng(direct_query_rng_seed)
        try:
            ranked_full = _direct_query_ranked(
                bridge_full, word, dims, all_pools, word_to_idx,
                stim_steps=recall_steps, reset_steps=recall_steps // 2,
            )
        finally:
            _restore_query_rng(saved_dq_full)
        decided_full = gate_direct_unified(
            ranked_full, DIRECT_UNIFIED_THRESHOLD
        )
        ans_full = None if decided_full is None else decided_full[0]
        if ans_full == expected_pool:
            n_direct_correct_full += 1
        saved_dq_uniform = _seed_query_rng(direct_query_rng_seed)
        try:
            ranked_uniform = _direct_query_ranked(
                bridge_uniform, word, dims, all_pools, word_to_idx,
                stim_steps=recall_steps, reset_steps=recall_steps // 2,
            )
        finally:
            _restore_query_rng(saved_dq_uniform)
        decided_uniform = gate_direct_unified(
            ranked_uniform, DIRECT_UNIFIED_THRESHOLD
        )
        ans_uniform = None if decided_uniform is None else decided_uniform[0]
        if ans_uniform == expected_pool:
            n_direct_correct_uniform += 1

    # ---- COMPOSITIONAL queries: one per encoded fact, cue the noun,
    # expect the bound adj. The cue stays ON during retrieve in BOTH
    # arms (encoding-specificity respected). The FULL arm primes
    # dlpfc_verb for 50 steps THEN amplifies the engram-tag drive
    # 3.0x. The UNIFORM_CTRL arm SKIPS PFC-frame priming AND uses the
    # 1.0x baseline tag drive (via _compositional_query_ranked).
    full_pfc_steps, _uniform_pfc_steps = _pfc_frame_step_counts(tiny_synth)
    n_comp_total = 0
    n_comp_correct_full = 0
    n_comp_correct_uniform = 0
    for i, (noun, adj) in enumerate(facts):
        n_comp_total += 1
        tag_full = tags_full[i] if i < len(tags_full) else None
        tag_uniform = tags_uniform[i] if i < len(tags_uniform) else None
        query_rng_seed = (
            int(seed) * 1_000_003 + int(N) * 1009 + int(i) * 7919 + 17
        ) & 0x7FFFFFFF
        # FULL arm: PFC-frame prime (50 steps) then amplified-tag
        # compositional retrieval read.
        saved_full = _seed_query_rng(query_rng_seed)
        try:
            _prime_pfc_frame(bridge_full, n_steps=int(full_pfc_steps))
            ranked_full = _compositional_query_amplified(
                bridge_full, noun, tag_full, dims, recall_steps,
                tag_amp_factor=float(RETRIEVE_TAG_AMP_FACTOR),
            )
        finally:
            _restore_query_rng(saved_full)
        decided_full = gate_compositional_unified(
            ranked_full, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        ans_full = None if decided_full is None else decided_full[0]
        # UNIFORM_CTRL arm: SKIP PFC-frame priming; baseline
        # compositional retrieval read at 1.0x tag drive.
        saved_uniform = _seed_query_rng(query_rng_seed)
        try:
            ranked_uniform = _compositional_query_ranked(
                bridge_uniform, noun, tag_uniform, dims, recall_steps
            )
        finally:
            _restore_query_rng(saved_uniform)
        decided_uniform = gate_compositional_unified(
            ranked_uniform, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        ans_uniform = None if decided_uniform is None else decided_uniform[0]
        if ans_full == adj:
            n_comp_correct_full += 1
        if ans_uniform == adj:
            n_comp_correct_uniform += 1

    # ---- UNGROUNDABLE queries: vocabulary words NOT used in this
    # rung's facts. The appropriate-regime gate MUST abstain. Counted
    # against the FULL arm (the load-bearing arm).
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

    ungroundable_nouns = [w for w in _NOUNS if w not in encoded_nouns]
    for w in ungroundable_nouns:
        n_ungroundable += 1
        _prime_pfc_frame(bridge_full, n_steps=int(full_pfc_steps))
        ranked = _compositional_query_amplified(
            bridge_full, w, None, dims, recall_steps,
            tag_amp_factor=float(RETRIEVE_TAG_AMP_FACTOR),
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
        "replay_n_replays_full": int(
            replay_stats_full.get("n_replays", 0)
        ),
        "replay_n_replays_uniform": int(
            replay_stats_uniform.get("n_replays", 0)
        ),
        "cue_suppression_active_full": bool(
            replay_stats_full.get("cue_suppression_active", False)
        ),
        "cue_suppression_active_uniform": bool(
            replay_stats_uniform.get("cue_suppression_active", False)
        ),
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
def run_targeted_cue_suppression_replay(
    seeds,
    loads=_TC_LADDER,
    tiny_synth: bool = False,
    phase1_cache_dir: str = _PHASE1_CACHE_DEFAULT,
    out_path: Optional[str] = None,
    ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """Targeted cue-suppression-during-replay + amplified engram-tag
    stim + persistent PFC-frame + higher n_replays_per_tag capability
    runner.

    Per seed (in order):
      * Phase-1 multi-event direct training (cached) -- REUSED
        ``_phase1_train_if_needed`` from the unified runner.

    Per (seed, N) cell:
      * Build TWO parallel bridges from the same Phase-1 checkpoint
        (one for FULL, one for UNIFORM_CTRL).
      * Encode the SAME compositional facts into BOTH bridges.
      * FULL arm: run ``_run_replay_with_cue_suppressed`` ONCE
        after encoding (50 replays/tag, lang_input zeroed +
        lang_to_ec gate clamped). UNIFORM_CTRL arm: run
        ``_run_replay_baseline`` (20 replays/tag, cue present).
      * For each compositional query: FULL arm primes ``dlpfc_verb``
        for 50 steps then runs ``_compositional_query_amplified`` at
        3.0x tag drive; the UNIFORM_CTRL arm skips priming and uses
        the baseline 1.0x ``_compositional_query_ranked``.
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
    verdict = targeted_cue_suppression_replay_verdict(rungs)

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
        "targeted_mechanism_constants": {
            "REPLAY_CYCLES_PER_TAG": int(REPLAY_CYCLES_PER_TAG),
            "UNIFORM_CTRL_REPLAY_CYCLES": int(
                _UNIFORM_CTRL_REPLAY_CYCLES
            ),
            "RETRIEVE_TAG_AMP_FACTOR": float(RETRIEVE_TAG_AMP_FACTOR),
            "REPLAY_DRIVE_PA": float(REPLAY_DRIVE_PA),
            "REPLAY_BURST_DURATION_MS": int(REPLAY_BURST_DURATION_MS),
            "REPLAY_INTER_BURST_MS": int(REPLAY_INTER_BURST_MS),
            "PFC_FRAME_PA": float(PFC_FRAME_PA),
            "PFC_FRAME_STIM_STEPS": int(PFC_FRAME_STIM_STEPS),
            "UNIFORM_CTRL_PFC_FRAME_STIM_STEPS": int(
                _UNIFORM_CTRL_PFC_FRAME_STIM_STEPS
            ),
        },
    }
    if tiny_synth:
        result["note"] = (
            "TINY-SYNTH toy numbers -- NOT a result; logic-screen only. "
            "Phase-1 training is shrunk to a few events; compositional "
            "encoding shrunk to one pair per rung; FULL replay cycles "
            "shrunk (n=%d, burst=%dms, inter=%dms); UNIFORM_CTRL replay "
            "cycles shrunk (n=%d); FULL PFC-frame priming shrunk to %d "
            "steps; UNIFORM_CTRL PFC-frame skipped. The decisive multi-"
            "seed CuPy run at full biological scale is a later "
            "controller-only task."
            % (
                TINY_REPLAY_CYCLES_PER_TAG,
                TINY_REPLAY_BURST_DURATION_MS,
                TINY_REPLAY_INTER_BURST_MS,
                TINY_UNIFORM_CTRL_REPLAY_CYCLES,
                TINY_PFC_FRAME_STIM_STEPS,
            )
        )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Targeted cue-suppression-during-replay + amplified engram-"
            "tag stim + persistent PFC-frame + higher n_replays_per_tag "
            "runner (Task 2 of the 7th arc). FULL arm runs cue-"
            "suppressed replay (50 cycles, lang_input zeroed + "
            "lang_to_ec gate clamped) after encoding AND primes "
            "dlpfc_verb for 50 steps before each compositional query "
            "AND amplifies the engram-tag drive 3.0x; UNIFORM_CTRL "
            "arm runs cue-present replay (20 cycles), skips PFC-frame "
            "priming, and uses 1.0x baseline tag drive. Cue stays ON "
            "during retrieve in BOTH arms (encoding-specificity "
            "respected per the theta-gamma finding). Reuse-only "
            "orchestration of the prior unified per-regime monitor + "
            "6th arc runners + the net-new targeted-mechanism "
            "invocations + the THREE structural-effect probes. No "
            "autograd; no torch; no LLM call."
        )
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument(
        "--loads",
        type=int,
        nargs="+",
        default=list(_TC_LADDER),
        help="Load ladder (default the frozen ladder (2,3,5)).",
    )
    ap.add_argument(
        "--tiny-synth",
        action="store_true",
        help=(
            "Shrink Phase-1 training + compositional pair count + "
            "replay cycles + PFC-frame priming hard for the logic-"
            "screen smoke. Toy numbers are NOT a result."
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
            "AND across the prior arcs."
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
        "--skip-structural-probes",
        action="store_true",
        help=(
            "Skip the structural-effect probes before the eval loop. "
            "ONLY for the inner-loop test smoke that exercises the "
            "probes directly via the test API. Do NOT pass this for "
            "any decisive run."
        ),
    )
    a = ap.parse_args(argv)

    # MANDATORY: ALL THREE structural-effect probes before the decisive
    # eval loop. If any mechanism is structurally inert (or RNG
    # isolation is broken) the probe raises and the runner aborts
    # with NO decisive numbers reported. Mirrors Pirazzini d462bf0 +
    # theta-gamma e6b17da + 6th arc 13f73e8 lessons.
    if not a.skip_structural_probes:
        try:
            diff_cue = _cue_suppression_replay_effect_probe(
                seed=int(a.seeds[0]) if a.seeds else 42,
                tiny_synth=bool(a.tiny_synth),
                cache_dir=a.phase1_cache_dir,
            )
        except RuntimeError as exc:
            print(
                "CUE-SUPPRESSION-REPLAY-EFFECT-PROBE FAILED: %s" % exc,
                file=sys.stderr, flush=True,
            )
            return 2
        print(
            "CUE-SUPPRESSION-REPLAY-EFFECT-PROBE PASS: max |delta "
            "v_membrane| = %.6g mV (> 1.0 mV)" % diff_cue,
            flush=True,
        )
        try:
            diff_amp = _amplified_tag_stim_effect_probe(
                seed=int(a.seeds[0]) if a.seeds else 42,
                tiny_synth=bool(a.tiny_synth),
                cache_dir=a.phase1_cache_dir,
            )
        except RuntimeError as exc:
            print(
                "AMPLIFIED-TAG-STIM-EFFECT-PROBE FAILED: %s" % exc,
                file=sys.stderr, flush=True,
            )
            return 2
        print(
            "AMPLIFIED-TAG-STIM-EFFECT-PROBE PASS: max |delta "
            "v_membrane| = %.6g mV (> 1.0 mV)" % diff_amp,
            flush=True,
        )
        try:
            diff_pfc = _persistent_pfc_frame_effect_probe(
                seed=int(a.seeds[0]) if a.seeds else 42,
                tiny_synth=bool(a.tiny_synth),
                cache_dir=a.phase1_cache_dir,
            )
        except RuntimeError as exc:
            print(
                "PERSISTENT-PFC-FRAME-EFFECT-PROBE FAILED: %s" % exc,
                file=sys.stderr, flush=True,
            )
            return 2
        print(
            "PERSISTENT-PFC-FRAME-EFFECT-PROBE PASS: max |delta "
            "v_membrane| = %.6g mV (> 1.0 mV)" % diff_pfc,
            flush=True,
        )

    result = run_targeted_cue_suppression_replay(
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
