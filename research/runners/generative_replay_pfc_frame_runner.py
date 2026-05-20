"""Net-new generative-replay + PFC-held compositional frame runner
(Task 2 of the 6th arc).

Biology (complementary-learning-systems theory; McClelland 1995; Tonegawa
2015; Buzsaki SWR replay 2015; Fuster 1989 / Goldman-Rakic 1995 / Wang
2002 PFC working memory): the brain consolidates one-shot relational
bindings into cortex via NREM sharp-wave-ripple replay AND holds the
ordered compositional structure of an ongoing thought in prefrontal
working memory via NMDA-bistable persistent activity. The 6th
architecture in the gating-based composition design line combines BOTH
of these biology-grounded subsystems on top of the validated unified
substrate (already has v16 concept pools + hippocampus + dlpfc_verb):

  * GENERATIVE REPLAY (REUSED ``run_concept_replay_phase`` from
    ``consolidation_trainer.py`` -- the already-validated SWR replay
    subsystem). For each compositional one-shot encoding, the runner
    runs the engram-tagged ensemble through repeated SWR-style
    stimulation cycles; STDP at the ca3->ca1->cortex pathways
    consolidates the specific (noun, adj) binding into the substrate.
    This is the biology-grounded mechanism that lets a one-shot encoded
    pair survive the substrate's noise floor at retrieval time without
    relying on tag-stim alone (the per-regime + unified runners' tag-
    stim-only retrieval was overwhelmed by the cued-noun's diffuse
    lang_input drive per the localisation finding).
  * PFC-HELD COMPOSITIONAL FRAME (REUSED ``dlpfc_verb`` region from
    the validated Cluster G v2.5 substrate -- NMDA-bistable persistent
    activity). Before each compositional query, the runner briefly
    drives the dlpfc_verb region so the NMDA-bistable attractor in
    that region carries the compositional structure into the retrieve
    window. This is the biology-grounded mechanism that holds the
    "which slot am I asking about" frame in working memory while the
    substrate's hippocampus + concept pools handle retrieval.

The 5-architecture convergent ceiling (Stage-1 + SPEAR + Pirazzini +
Unified per-regime monitor + Theta-gamma all decisively FAILED at
biological scale with distinct mechanism-level signatures); the theta-
gamma finding established that cue-suppression-during-retrieve violates
the encoding-specificity principle (Tulving 1973). The 6th arc REMOVES
cue-suppression-during-retrieve entirely (cue stays ON during retrieve
in BOTH arms) and adds replay + PFC-frame as AUGMENTING mechanisms that
RESPECT encoding-specificity. See
docs/plans/2026-05-20-generative-replay-PFC-frame-design.md.

This module is the ONLY genuinely net-new code in the arc:
  * a runner-local generative-replay invocation per cell: the FULL
    arm runs ``run_concept_replay_phase(bridge_full, tag_names=...,
    n_replays_per_tag=...)`` ONCE after encoding; the UNIFORM_CTRL arm
    skips this (the SOLE replay differentiator);
  * a runner-local PFC-frame priming per compositional query: the
    FULL arm briefly drives ``bridge_full.cp_external_input_current``
    at the ``dlpfc_verb`` region's indices for a small number of
    steps before the retrieval read; the UNIFORM_CTRL arm skips this
    (the SOLE PFC-frame differentiator);
  * TWO structural-effect probes (replay + PFC-frame) that verify
    each mechanism produces > 1 mV bridge-state divergence between
    on and off via the runner's ACTUAL code path under deterministic
    RNG isolation, AND that both-arms-same controls agree to <
    0.5 mV (mirrors Pirazzini d462bf0 + theta-gamma e6b17da lesson).
    Both probes run in main() BEFORE the decisive eval loop. If
    either probe fails the runner aborts (no decisive numbers).

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
    ``generative_replay_pfc_frame_core.generative_replay_pfc_frame_verdict``
    (Task 1 byte-unchanged; bars set in advance, NEVER tuned).
  * Both substrate-specific calibrated moats (the four moats):
    DIRECT_UNIFIED_THRESHOLD = 0.2841666666666667 +
    COMPOSITIONAL_UNIFIED_THRESHOLD = 0.1977124183006536 imported
    BYTE-UNCHANGED (no calibration changes).
  * Generative replay phase: REUSED
    ``consolidation_trainer.run_concept_replay_phase`` BYTE-UNCHANGED.
  * Kill-safe checkpoint: REUSED sim.train_checkpoint.

Anti-cheat (carry forward all prior lessons):
  * OPAQUE tag names; no tag-string parsing on tag names.
  * BOTH moats fed the calibrated raw firing-rate confidence
    quantities.
  * The ``cross_pool_concept`` gate is opened ONLY inside the
    encoding window (via ``encode_concept_pair``) then closed.
  * ``uniform_ctrl`` differs from ``full`` ONLY in the augmenting
    mechanisms (replay phase + PFC-frame priming); same seed, same
    encoded facts, same query set, same cue presence in retrieve.
  * ``direct_retain`` is read from the SAME run as ``full``.
  * No protected-file edits; no autograd; the runner is reuse-only
    orchestration of the prior arc's modules + the net-new replay
    + PFC-frame invocations + the two structural-effect probes.

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

# Backend policy mirrors the unified / theta-gamma / per-regime / SPEAR /
# Pirazzini runners. CuPy is the decisive path; NumPy ONLY when CuPy is
# genuinely unavailable (GPU-less box).
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
from research.runners.generative_replay_pfc_frame_core import (
    generative_replay_pfc_frame_verdict,
    _GR_LADDER,
)

# REUSED gates byte-unchanged (the four moats). FULL arm routes direct
# queries through DIRECT_UNIFIED_THRESHOLD, compositional queries
# through COMPOSITIONAL_UNIFIED_THRESHOLD. UNIFORM_CTRL applies the
# SAME gates to the SAME ranked confidences (the SOLE differentiator
# between the arms is the augmenting mechanisms, not the gate
# routing).
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
# Generative-replay + PFC-frame constants. FROZEN -- never tuned in
# response to results. The reuse-by-import surface is byte-stable; the
# net-new tuneables are these step counts + drive amplitudes (set in
# advance, not in response to results).
# =====================================================================
# Generative replay: how many replay events per engram tag during the
# post-encoding consolidation phase. n=20 is the
# ``run_concept_replay_phase`` default and the Phase-1.3-validated
# value (Buzsaki 2015 NREM cycle count scaled down).
N_REPLAYS_PER_TAG = 20

# Generative replay: drive amplitude per replay event (pA). Matches the
# ``run_concept_replay_phase`` default and the
# ``run_swr_replay_phase`` 100 pA convention.
REPLAY_DRIVE_PA = 100.0

# Generative replay: burst duration per replay event (ms). The default
# is 100 ms (real SWR ~50 ms but the longer window helps STDP capture).
# Reused from ``run_concept_replay_phase`` default.
REPLAY_BURST_DURATION_MS = 100

# Generative replay: quiet inter-burst window (ms). Reused default.
REPLAY_INTER_BURST_MS = 50

# PFC-frame priming: amplitude in pA driven onto dlpfc_verb during the
# brief PFC-frame priming window before each compositional query.
# Picked at 100.0 pA (matches the existing
# ``research/runners/consolidation_trainer.py`` replay drive scale and
# the project-wide ``set_token_drive`` ~200 pA convention; 100 pA is a
# moderate prime that lets NMDA bistability hold the frame without
# saturating the dlpfc_verb region).
PFC_FRAME_PA = 100.0

# PFC-frame priming: number of simulation steps the prime drive is held
# on dlpfc_verb before the retrieval read. Picked at 10 (the
# NMDA-bistability time constant of ~100 ms at dt=0.5 ms ~ 200 steps
# is the persistence window; 10 steps is the prime kick; the frame is
# then held by the NMDA-bistable attractor for the retrieval read).
PFC_FRAME_STIM_STEPS = 10

# tiny-synth shrunk values. Each replay count + step count keeps a
# non-zero floor so the mechanism is genuinely exercised under the smoke.
TINY_N_REPLAYS_PER_TAG = 2
TINY_REPLAY_BURST_DURATION_MS = 6
TINY_REPLAY_INTER_BURST_MS = 3
TINY_PFC_FRAME_STIM_STEPS = 3


def _replay_step_counts(tiny_synth: bool) -> Tuple[int, int, int]:
    """Return (n_replays_per_tag, burst_duration_ms, inter_burst_ms).

    Full scale uses the ``run_concept_replay_phase`` validated defaults;
    tiny-synth shrinks all three to a logic-screen smoke.
    """
    if tiny_synth:
        return (
            int(TINY_N_REPLAYS_PER_TAG),
            int(TINY_REPLAY_BURST_DURATION_MS),
            int(TINY_REPLAY_INTER_BURST_MS),
        )
    return (
        int(N_REPLAYS_PER_TAG),
        int(REPLAY_BURST_DURATION_MS),
        int(REPLAY_INTER_BURST_MS),
    )


def _pfc_frame_step_count(tiny_synth: bool) -> int:
    """Return the PFC-frame priming step count. Full scale uses
    PFC_FRAME_STIM_STEPS; tiny-synth shrinks for the smoke.
    """
    if tiny_synth:
        return int(TINY_PFC_FRAME_STIM_STEPS)
    return int(PFC_FRAME_STIM_STEPS)


# =====================================================================
# Deterministic-RNG isolation helper. Transcribed BYTE-UNCHANGED from
# the theta-gamma runner (commit e6b17da). Closes the eighth adversarial
# review BLOCK: the structural-effect probes + per-cell eval arm seed
# the active backend's RNG to the SAME value before each arm's call so
# both arms see IDENTICAL OU-noise streams; the SOLE remaining between-
# arm difference is the augmenting mechanism flag.
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
        This is what cp.random.randn / np.random.randn inside
        bridge._run_one_simulation_step's OU noise consumes.
      * The top-level numpy.random global (independent module-level
        state from the backend on CuPy; the same state on NumPy).
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
# Net-new helpers: generative-replay phase invocation + PFC-frame
# priming window. Each wraps a REUSED bridge subsystem; the runner
# orchestrates only.
# =====================================================================
def _run_generative_replay(bridge, tag_names: List[str],
                             tiny_synth: bool,
                             rng_seed: int) -> Dict[str, Any]:
    """Run ONE generative-replay phase on the given bridge after the
    compositional encoding. REUSES ``run_concept_replay_phase`` byte-
    unchanged; the runner only chooses the replay counts (tiny-synth
    shrinks them) and the RNG seed (deterministic so replay phases are
    reproducible across runs).

    The replay phase drives each engram-tagged ensemble for
    ``n_replays_per_tag`` events; STDP at the ca3->ca1->cortex pathways
    auto-consolidates the specific (noun, adj) binding into cortex.
    See ``consolidation_trainer.run_concept_replay_phase`` for the
    validated implementation.

    Args:
      bridge: the SimulationBridge to run replay on (FULL arm bridge).
      tag_names: the engram tag names committed by ``_encode_facts``.
      tiny_synth: shrink-for-smoke flag.
      rng_seed: deterministic RNG seed for the replay-order shuffle.

    Returns:
      The stats dict from ``run_concept_replay_phase``.
    """
    if not tag_names:
        return {"n_replays": 0, "tags_replayed": [],
                "per_tag_replay_count": {},
                "burst_duration_ms": 0, "inter_burst_ms": 0,
                "randomize_order": False}
    n_replays, burst_ms, inter_ms = _replay_step_counts(tiny_synth)
    rng = np.random.default_rng(int(rng_seed))
    stats = run_concept_replay_phase(
        bridge,
        tag_names=list(tag_names),
        n_replays_per_tag=int(n_replays),
        burst_duration_ms=int(burst_ms),
        inter_burst_ms=int(inter_ms),
        drive_pA=float(REPLAY_DRIVE_PA),
        randomize_order=True,
        rng=rng,
    )
    return stats


def _prime_pfc_frame(bridge, tiny_synth: bool) -> int:
    """Drive ``dlpfc_verb`` external input current for a brief priming
    window. The NMDA-bistable attractor in dlpfc_verb holds the frame
    afterwards for the retrieve read.

    Mechanism:
      * Locate the ``dlpfc_verb`` region's neuron indices via the
        bridge's region_manager.
      * For PFC_FRAME_STIM_STEPS simulation steps (TINY_PFC_FRAME_STIM_STEPS
        under tiny-synth), set
        ``bridge.cp_external_input_current[dlpfc_idx] = PFC_FRAME_PA``
        and run one simulation step. Writes happen INSIDE the per-
        step loop so the drive survives any sub-helper clears (Pirazzini
        FIX B per-step pattern). Other neurons' external input is set
        to 0 during this window so the prime is isolated.
      * Clear ``cp_external_input_current`` after the window so the
        subsequent retrieval read sees a clean baseline.

    Returns the number of dlpfc_verb neurons that received the prime
    drive (diagnostic). Returns 0 if dlpfc_verb is not in the region
    manager (substrate without enable_dlpfc_verb=True).
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
    dlpfc_arr = cp.asarray(dlpfc_idx, dtype=cp.int64)
    n_steps = _pfc_frame_step_count(tiny_synth)
    for _ in range(int(n_steps)):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[dlpfc_arr] = float(PFC_FRAME_PA)
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    return int(len(dlpfc_idx))


# =====================================================================
# Structural-effect probes -- MANDATORY (TWO of them; mirror theta-
# gamma e6b17da + Pirazzini d462bf0). Each probe verifies one of the
# two augmenting mechanisms produces > 1 mV bridge-state divergence
# between flag-on and flag-off via the runner's ACTUAL code path under
# deterministic RNG isolation. Each probe also runs both-arms-same
# controls and asserts those agree to < 0.5 mV (the RNG-isolation
# soundness check).
# =====================================================================
# Deterministic RNG seeds for the structural-effect probes. The probes
# use fixed values across all four runs (two flag-differing, two flag-
# same controls) so the only between-arm difference is the flag.
_PROBE_RNG_SEED = 999
_PROBE_ENCODE_RNG_SEED = 31337

# Tolerance for the "controls must show near-zero divergence" check.
# 0.5 mV is well below the 1.0 mV bar the flag-differing case must
# exceed (mirrors theta-gamma e6b17da).
_PROBE_CONTROL_TOL_MV = 0.5


def _validate_cache_scale_for_probe(cache_path, built_bridge,
                                       probe_name: str) -> None:
    """Refuse to run the probe on a cache file whose stored bridge
    dimensions do NOT match the freshly-built bridge.

    Closes the 10th adversarial review BLOCK on the 6th-architecture
    arc: with ``tiny_synth=True``, the probe builds a small bridge
    (e.g. 952 neurons / 46497 synapses) BUT ``load_checkpoint`` will
    happily load an existing biological-scale Phase-1 cache (e.g. 8440
    neurons / 4825651 synapses) under ``_PHASE1_CACHE_DEFAULT``. The
    bridge state arrays then have inconsistent shapes:
    ``cp_membrane_potential_v`` is sized to the cached value, but the
    arrays the bridge allocated at build time (e.g.
    ``cp_plasticity_rate_gain``) stay at the tiny-synth size. Every
    subsequent simulation step raises ``IndexError`` (caught by a
    broad ``try`` / ``except`` inside the bridge step), silently
    corrupting the probe state -- so the probe's reported
    flag-differing divergence is NOT trustworthy as a gate.

    Implementation: open the HDF5 file (lazy ``import h5py`` so the
    runner has no new top-level dependency) and inspect:
      * ``num_neurons`` attr (the canonical stored neuron count);
      * ``connections_shape_0`` attr (the connection-matrix neuron
        count -- redundant cross-check);
      * ``cp_membrane_potential_v`` dataset shape[0] (the actual
        per-neuron array dim).
    Compare against ``built_bridge.cp_membrane_potential_v.shape[0]``.
    If ANY mismatches, raise ``RuntimeError`` with a clear message
    that surfaces both the cached and built dimensions so the
    operator can diagnose.

    Args:
      cache_path: path to the HDF5 Phase-1 checkpoint that will be
        loaded.
      built_bridge: the freshly-built SimulationBridge whose dimensions
        the cache must match.
      probe_name: e.g. ``"replay-effect"`` / ``"pfc-frame-effect"``;
        surfaced in the error message for diagnostic clarity.

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


def _replay_effect_probe(
    seed: int = 42,
    tiny_synth: bool = True,
    cache_dir: Optional[str] = None,
) -> float:
    """Run the runner's actual code path twice with the SAME initial
    bridge state but different generative-replay flags; return the max
    absolute membrane-potential divergence (mV) for the flag-differing
    case.

    Mechanism (CLOSES the 8th adversarial review BLOCK):
      * Deterministic-seed the active backend's RNG to _PROBE_RNG_SEED
        BEFORE each replay phase. Both arms therefore see IDENTICAL
        OU-noise streams; the SOLE remaining difference between the
        arms is the replay-on vs replay-off flag.
      * Restore the RNG state after each call so other components see
        no perturbation.
      * Run TWO additional CONTROL contrasts at the SAME seed:
          (1) both arms run replay -> divergence must be < 0.5 mV
          (2) both arms skip replay -> divergence must be < 0.5 mV
        If either control shows large divergence, RNG isolation is
        broken and the probe raises RuntimeError.
      * The flag-differing case (replay-on vs replay-off) MUST exceed
        1.0 mV for the mechanism to be declared structurally active.

    DEFENSIVE pre-load check (CLOSES the 10th adversarial review
    BLOCK): each contrast builds two fresh bridges and BEFORE calling
    ``load_checkpoint`` validates that the cached checkpoint's stored
    bridge dimensions match the freshly-built bridge dimensions via
    ``_validate_cache_scale_for_probe``. The 10th adversarial review
    caught a real defect: with ``tiny_synth=True`` and the
    biological-scale Phase-1 cache under ``_PHASE1_CACHE_DEFAULT``,
    the bridge was being silently corrupted (cached 8440-neuron state
    loaded into a 952-neuron build raises IndexError on every step,
    caught by the bridge's try/except). Pre-load scale validation
    surfaces the mismatch as a clean ``RuntimeError`` BEFORE the
    bridge is corrupted, so the probe's flag-differing divergence
    cannot be reported off a corrupted state.

    If the flag-differing divergence is below 1 mV, the mechanism is
    structurally inert and the caller MUST abort (no decisive numbers
    reported). If a control shows divergence above the tolerance, RNG
    isolation is broken and the caller MUST abort. If the cache-scale
    validation fails, the cache and build are incompatible and the
    caller MUST abort. Any of these raises RuntimeError. Returns the
    flag-differing diff (float, > 1.0) when BOTH the mechanism is
    structurally active AND the controls pass AND the cache scale
    matches.
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

    def _one_contrast(flag_a: bool, flag_b: bool) -> float:
        """Build two fresh bridges, load the SAME checkpoint into both,
        deterministic-seed BEFORE _encode_facts on EACH so the encoded
        bridge states are IDENTICAL, then deterministic-seed BEFORE
        _run_generative_replay on EACH (same seed across arms) so the
        replay-phase OU-noise streams are IDENTICAL. The SOLE remaining
        between-arm difference is the replay-on flag.

        Pre-load CACHE-SCALE validation: BEFORE load_checkpoint runs,
        _validate_cache_scale_for_probe inspects the HDF5 file's
        stored bridge dimensions and refuses to proceed if they
        mismatch the freshly-built bridge. Closes 10th adversarial
        review BLOCK; without this, tiny_synth=True with the
        biological-scale Phase-1 cache silently corrupts bridge
        state (IndexError every step, swallowed by bridge's
        try/except) and the reported flag-differing divergence is
        unreliable as a gate.

        Returns max |delta v_membrane|. Fresh bridges per contrast
        ensure no cross-contrast leakage."""
        bridge_a = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        bridge_b = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        _validate_cache_scale_for_probe(
            cache_path, bridge_a, "replay-effect"
        )
        _validate_cache_scale_for_probe(
            cache_path, bridge_b, "replay-effect"
        )
        bridge_a.load_checkpoint(str(cache_path))
        bridge_b.load_checkpoint(str(cache_path))
        _freeze_phase1_gates(bridge_a)
        _freeze_phase1_gates(bridge_b)

        # Encoding phase: identical deterministic seed BEFORE each call
        # so the two bridges end up in byte-identical encoded states.
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

        # Replay phase: identical deterministic seed BEFORE each arm so
        # the OU-noise streams match across arms; the SOLE
        # differentiator is the flag (run replay vs skip replay).
        saved_a = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            if flag_a:
                _ = _run_generative_replay(
                    bridge_a, tags_a, tiny_synth, _PROBE_RNG_SEED
                )
        finally:
            _restore_query_rng(saved_a)
        saved_b = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            if flag_b:
                _ = _run_generative_replay(
                    bridge_b, tags_b, tiny_synth, _PROBE_RNG_SEED
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
            "Replay-effect probe CONTROL FAILED (both-True): with replay "
            "on BOTH bridges and the same deterministic RNG seed, the two "
            "bridges diverged by %.6g mV (> %.3g mV tolerance). RNG "
            "isolation is broken; the flag-differing divergence is NOT "
            "attributable to the generative-replay mechanism. Closes 8th "
            "adversarial review BLOCK; fix RNG isolation and re-run."
            % (diff_both_true, _PROBE_CONTROL_TOL_MV)
        )
    if diff_both_false > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "Replay-effect probe CONTROL FAILED (both-False): with replay "
            "skipped on BOTH bridges and the same deterministic RNG seed, "
            "the two bridges diverged by %.6g mV (> %.3g mV tolerance). "
            "RNG isolation is broken; the flag-differing divergence is "
            "NOT attributable to the generative-replay mechanism. Closes "
            "8th adversarial review BLOCK; fix RNG isolation and re-run."
            % (diff_both_false, _PROBE_CONTROL_TOL_MV)
        )

    if diff_flag_diff <= 1.0:
        raise RuntimeError(
            "Replay-effect probe FAILED: replay-on vs replay-off produced "
            "essentially identical bridge state (max |delta v| = %.6g mV "
            "<= 1.0 mV) via the runner's ACTUAL code path (controls "
            "passed: both-True=%.6g mV, both-False=%.6g mV). The "
            "mechanism is structurally inert -- mirrors Pirazzini "
            "d462bf0 defect. Fix and re-run BEFORE decisive."
            % (diff_flag_diff, diff_both_true, diff_both_false)
        )

    return float(diff_flag_diff)


def _pfc_frame_effect_probe(
    seed: int = 42,
    tiny_synth: bool = True,
    cache_dir: Optional[str] = None,
) -> float:
    """Run the runner's actual code path twice with the SAME initial
    bridge state but different PFC-frame priming flags; return the max
    absolute membrane-potential divergence (mV) for the flag-differing
    case.

    Mechanism (CLOSES the 8th adversarial review BLOCK):
      * Deterministic-seed the active backend's RNG to _PROBE_RNG_SEED
        BEFORE each PFC-frame priming window. Both arms therefore see
        IDENTICAL OU-noise streams; the SOLE remaining difference
        between the arms is the prime-on vs prime-off flag.
      * Restore the RNG state after each call so other components see
        no perturbation.
      * Run TWO additional CONTROL contrasts at the SAME seed:
          (1) both arms prime PFC-frame -> divergence must be < 0.5 mV
          (2) both arms skip prime -> divergence must be < 0.5 mV
        If either control shows large divergence, RNG isolation is
        broken and the probe raises RuntimeError.
      * The flag-differing case (prime-on vs prime-off) MUST exceed
        1.0 mV for the mechanism to be declared structurally active.

    DEFENSIVE pre-load check (CLOSES the 10th adversarial review
    BLOCK): each contrast builds two fresh bridges and BEFORE calling
    ``load_checkpoint`` validates that the cached checkpoint's stored
    bridge dimensions match the freshly-built bridge dimensions via
    ``_validate_cache_scale_for_probe``. The 10th adversarial review
    caught a real defect: with ``tiny_synth=True`` and the
    biological-scale Phase-1 cache under ``_PHASE1_CACHE_DEFAULT``,
    the bridge was being silently corrupted (cached 8440-neuron state
    loaded into a 952-neuron build raises IndexError on every step,
    caught by the bridge's try/except). Pre-load scale validation
    surfaces the mismatch as a clean ``RuntimeError`` BEFORE the
    bridge is corrupted, so the probe's flag-differing divergence
    cannot be reported off a corrupted state.

    Returns the flag-differing diff (float, > 1.0) when BOTH the
    mechanism is structurally active AND the controls pass AND the
    cache scale matches the built bridge.
    """
    from sim.backend import to_host
    cache_dir = str(cache_dir) if cache_dir else _PHASE1_CACHE_DEFAULT
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _phase1_train_if_needed(int(seed), cache_dir, tiny_synth)
    cache_path = _phase1_cache_path(cache_dir, seed)

    def _one_contrast(flag_a: bool, flag_b: bool) -> float:
        """Build two fresh bridges, load the SAME checkpoint into both,
        deterministic-seed BEFORE each _prime_pfc_frame call so the
        OU-noise streams match across arms; the SOLE between-arm
        difference is the prime-on flag.

        Pre-load CACHE-SCALE validation: BEFORE load_checkpoint runs,
        _validate_cache_scale_for_probe inspects the HDF5 file's
        stored bridge dimensions and refuses to proceed if they
        mismatch the freshly-built bridge. Closes 10th adversarial
        review BLOCK; without this, tiny_synth=True with the
        biological-scale Phase-1 cache silently corrupts bridge
        state (IndexError every step, swallowed by bridge's
        try/except) and the reported flag-differing divergence is
        unreliable as a gate.
        """
        bridge_a = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        bridge_b = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
        _validate_cache_scale_for_probe(
            cache_path, bridge_a, "pfc-frame-effect"
        )
        _validate_cache_scale_for_probe(
            cache_path, bridge_b, "pfc-frame-effect"
        )
        bridge_a.load_checkpoint(str(cache_path))
        bridge_b.load_checkpoint(str(cache_path))
        _freeze_phase1_gates(bridge_a)
        _freeze_phase1_gates(bridge_b)

        saved_a = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            if flag_a:
                _ = _prime_pfc_frame(bridge_a, tiny_synth)
        finally:
            _restore_query_rng(saved_a)
        saved_b = _seed_query_rng(_PROBE_RNG_SEED)
        try:
            if flag_b:
                _ = _prime_pfc_frame(bridge_b, tiny_synth)
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
            "PFC-frame-effect probe CONTROL FAILED (both-True): with "
            "prime on BOTH bridges and the same deterministic RNG seed, "
            "the two bridges diverged by %.6g mV (> %.3g mV tolerance). "
            "RNG isolation is broken; the flag-differing divergence is "
            "NOT attributable to the PFC-frame priming mechanism. Closes "
            "8th adversarial review BLOCK; fix RNG isolation and re-run."
            % (diff_both_true, _PROBE_CONTROL_TOL_MV)
        )
    if diff_both_false > _PROBE_CONTROL_TOL_MV:
        raise RuntimeError(
            "PFC-frame-effect probe CONTROL FAILED (both-False): with "
            "prime skipped on BOTH bridges and the same deterministic "
            "RNG seed, the two bridges diverged by %.6g mV (> %.3g mV "
            "tolerance). RNG isolation is broken; the flag-differing "
            "divergence is NOT attributable to the PFC-frame priming "
            "mechanism. Closes 8th adversarial review BLOCK; fix RNG "
            "isolation and re-run."
            % (diff_both_false, _PROBE_CONTROL_TOL_MV)
        )

    if diff_flag_diff <= 1.0:
        raise RuntimeError(
            "PFC-frame-effect probe FAILED: prime-on vs prime-off "
            "produced essentially identical bridge state (max |delta v| "
            "= %.6g mV <= 1.0 mV) via the runner's ACTUAL code path "
            "(controls passed: both-True=%.6g mV, both-False=%.6g mV). "
            "The mechanism is structurally inert -- mirrors Pirazzini "
            "d462bf0 defect. Fix and re-run BEFORE decisive."
            % (diff_flag_diff, diff_both_true, diff_both_false)
        )

    return float(diff_flag_diff)


# =====================================================================
# Per-cell evaluation arm: one (seed, N) cell. Mirrors the unified
# runner's structure EXCEPT:
#   * the FULL arm runs ``_run_generative_replay`` ONCE after encoding;
#     the UNIFORM_CTRL arm skips that step;
#   * the FULL arm primes ``dlpfc_verb`` via ``_prime_pfc_frame`` before
#     each compositional query; the UNIFORM_CTRL arm skips that step;
#   * the cue (lang_input) stays ON during retrieve in BOTH arms
#     (encoding-specificity respected per the theta-gamma finding).
#
# Engineering decision (documented): the replay phase + PFC-frame
# prime modify bridge state, so the FULL and UNIFORM_CTRL arms cannot
# share the same bridge. We build TWO parallel bridges at the start of
# the cell (one for FULL, one for UNIFORM_CTRL), load the SAME Phase-1
# checkpoint into both, encode the SAME facts into both, then run each
# arm's queries against its own bridge. This keeps the seed + encoded
# facts + query set + cue presence + ranked confidences IDENTICAL
# across the two arms; the SOLE differentiator is the AUGMENTING
# mechanisms (replay phase + PFC-frame priming).
# =====================================================================
def _run_evaluation_arm(seed: int, N: int, tiny_synth: bool,
                          cache_dir: str) -> Dict[str, Any]:
    """Run the generative-replay + PFC-frame architecture for ONE
    (seed, N) cell. Two parallel bridges (FULL + UNIFORM_CTRL); each
    runs the SAME encoding + same direct queries; the FULL arm runs
    ``_run_generative_replay`` once after encoding AND primes
    ``dlpfc_verb`` before each compositional query; the UNIFORM_CTRL
    arm skips both augmenting mechanisms.

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
    # augmenting mechanisms (replay phase + PFC-frame priming) on the
    # FULL bridge.
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
    # RNG isolation: seed the active backend + numpy + python RNGs to
    # the SAME deterministic value BEFORE each call so the encoded
    # bridge states are byte-identical across arms.
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

    # ---- NET-NEW (FULL arm only): generative replay on bridge_full.
    # The UNIFORM_CTRL arm skips this. RNG isolation: identical
    # deterministic seed before the replay phase so the replay-phase
    # OU-noise stream is reproducible.
    replay_rng_seed = (
        int(seed) * 1_000_003 + int(N) * 1009 + 7919
    ) & 0x7FFFFFFF
    saved_replay = _seed_query_rng(replay_rng_seed)
    try:
        replay_stats = _run_generative_replay(
            bridge_full, tags_full, tiny_synth, replay_rng_seed
        )
    finally:
        _restore_query_rng(saved_replay)

    # ---- DIRECT queries: one per unique trained word in the cell's
    # facts. BOTH arms route direct queries through the SAME substrate-
    # specific direct gate (DIRECT_UNIFIED_THRESHOLD). The two arms may
    # diverge here because the FULL bridge has been perturbed by the
    # replay phase; the UNIFORM_CTRL bridge has not.
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
        # RNG isolation: identical deterministic seed for the direct
        # query across BOTH arms so the OU-noise stream is matched.
        direct_query_rng_seed = (
            int(seed) * 1_000_003 + int(N) * 1009 + hash(word) % 65521 + 113
        ) & 0x7FFFFFFF
        # FULL bridge direct read.
        saved_dq_full = _seed_query_rng(direct_query_rng_seed)
        try:
            ranked_full = _direct_query_ranked(
                bridge_full, word, dims, all_pools, word_to_idx,
                stim_steps=recall_steps, reset_steps=recall_steps // 2,
            )
        finally:
            _restore_query_rng(saved_dq_full)
        decided_full = gate_direct_unified(ranked_full, DIRECT_UNIFIED_THRESHOLD)
        ans_full = None if decided_full is None else decided_full[0]
        if ans_full == expected_pool:
            n_direct_correct_full += 1
        # UNIFORM_CTRL bridge direct read.
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
    # expect the bound adj. The cue (lang_input) stays ON during
    # retrieve in BOTH arms (encoding-specificity respected). The FULL
    # arm primes dlpfc_verb (PFC-frame) before the retrieval read; the
    # UNIFORM_CTRL arm skips that step. This is the SOLE
    # compositional-query differentiator between the two arms.
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
        # PFC-frame priming.
        query_rng_seed = (
            int(seed) * 1_000_003 + int(N) * 1009 + int(i) * 7919 + 17
        ) & 0x7FFFFFFF
        # FULL arm: PFC-frame prime, then compositional retrieval read.
        # The cue stays ON during retrieve (encoding-specificity).
        saved_full = _seed_query_rng(query_rng_seed)
        try:
            # NET-NEW (FULL arm only): brief PFC-frame priming before
            # the retrieval read. The NMDA-bistable attractor in
            # dlpfc_verb holds the frame for the retrieval window.
            _prime_pfc_frame(bridge_full, tiny_synth)
            ranked_full = _compositional_query_ranked(
                bridge_full, noun, tag_full, dims, recall_steps
            )
        finally:
            _restore_query_rng(saved_full)
        decided_full = gate_compositional_unified(
            ranked_full, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        ans_full = None if decided_full is None else decided_full[0]
        # UNIFORM_CTRL arm: skip PFC-frame priming; SAME compositional
        # retrieval read with SAME cue presence. IDENTICAL deterministic
        # RNG seed to the FULL arm above.
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
    # against the FULL arm (the load-bearing arm); this is the
    # trustworthiness property the verdict's abstain_correct bar gates.
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
    # run via the FULL arm's PFC-frame-primed compositional read (the
    # abstention behaviour under the load-bearing mechanism). The
    # substrate-specific compositional gate should abstain.
    ungroundable_nouns = [w for w in _NOUNS if w not in encoded_nouns]
    for w in ungroundable_nouns:
        n_ungroundable += 1
        # PFC-frame prime + compositional read on the FULL bridge.
        # No engram tag (ungroundable noun was never encoded).
        _prime_pfc_frame(bridge_full, tiny_synth)
        ranked = _compositional_query_ranked(
            bridge_full, w, None, dims, recall_steps
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
        "replay_n_replays": int(replay_stats.get("n_replays", 0)),
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
def run_generative_replay_pfc_frame(
    seeds,
    loads=_GR_LADDER,
    tiny_synth: bool = False,
    phase1_cache_dir: str = _PHASE1_CACHE_DEFAULT,
    out_path: Optional[str] = None,
    ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """Generative replay + PFC-held compositional frame capability runner.

    Per seed (in order):
      * Phase-1 multi-event direct training (cached) -- REUSED
        ``_phase1_train_if_needed`` from the unified runner. The
        Phase-1 caching strategy is the primary cost-amortisation.

    Per (seed, N) cell:
      * Build TWO parallel bridges from the same Phase-1 checkpoint
        (one for FULL, one for UNIFORM_CTRL).
      * Encode the SAME compositional facts into BOTH bridges.
      * FULL arm: run ``run_concept_replay_phase`` ONCE after encoding
        so STDP at ca3->ca1->cortex consolidates each (noun, adj)
        binding into cortex. UNIFORM_CTRL arm: skip.
      * For each compositional query: FULL arm primes ``dlpfc_verb``
        for PFC_FRAME_STIM_STEPS steps so the NMDA-bistable attractor
        holds the compositional frame for the retrieval read; the
        UNIFORM_CTRL arm skips that step. The cue stays ON during
        retrieve in BOTH arms (encoding-specificity respected).
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
    verdict = generative_replay_pfc_frame_verdict(rungs)

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
        "generative_replay_constants": {
            "N_REPLAYS_PER_TAG": int(N_REPLAYS_PER_TAG),
            "REPLAY_DRIVE_PA": float(REPLAY_DRIVE_PA),
            "REPLAY_BURST_DURATION_MS": int(REPLAY_BURST_DURATION_MS),
            "REPLAY_INTER_BURST_MS": int(REPLAY_INTER_BURST_MS),
        },
        "pfc_frame_constants": {
            "PFC_FRAME_PA": float(PFC_FRAME_PA),
            "PFC_FRAME_STIM_STEPS": int(PFC_FRAME_STIM_STEPS),
        },
    }
    if tiny_synth:
        result["note"] = (
            "TINY-SYNTH toy numbers -- NOT a result; logic-screen only. "
            "Phase-1 training is shrunk to a few events; compositional "
            "encoding shrunk to one pair per rung; replay cycles shrunk "
            "(n=%d, burst=%dms, inter=%dms); PFC-frame priming shrunk "
            "(%d steps). The decisive multi-seed CuPy run at full "
            "biological scale is a later controller-only task."
            % (
                TINY_N_REPLAYS_PER_TAG,
                TINY_REPLAY_BURST_DURATION_MS,
                TINY_REPLAY_INTER_BURST_MS,
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
            "Generative replay + PFC-held compositional frame runner "
            "(Task 2 of the 6th arc). FULL arm runs generative replay "
            "after encoding AND primes dlpfc_verb before each "
            "compositional query; UNIFORM_CTRL arm skips both. Cue "
            "stays ON during retrieve in BOTH arms (encoding-"
            "specificity respected per the theta-gamma finding). "
            "Reuse-only orchestration of the prior unified per-regime "
            "monitor runner + the net-new replay + PFC-frame "
            "invocations + the TWO structural-effect probes. No "
            "autograd; no torch; no LLM call."
        )
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument(
        "--loads",
        type=int,
        nargs="+",
        default=list(_GR_LADDER),
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

    # MANDATORY: BOTH structural-effect probes before the decisive eval
    # loop. If either mechanism is structurally inert (or RNG isolation
    # is broken) the probe raises and the runner aborts with NO
    # decisive numbers reported. Mirrors Pirazzini d462bf0 + theta-
    # gamma e6b17da lesson.
    if not a.skip_structural_probes:
        try:
            diff_replay = _replay_effect_probe(
                seed=int(a.seeds[0]) if a.seeds else 42,
                tiny_synth=bool(a.tiny_synth),
                cache_dir=a.phase1_cache_dir,
            )
        except RuntimeError as exc:
            print(
                "REPLAY-EFFECT-PROBE FAILED: %s" % exc,
                file=sys.stderr, flush=True,
            )
            return 2
        print(
            "REPLAY-EFFECT-PROBE PASS: max |delta v_membrane| = "
            "%.6g mV (> 1.0 mV)" % diff_replay,
            flush=True,
        )
        try:
            diff_pfc = _pfc_frame_effect_probe(
                seed=int(a.seeds[0]) if a.seeds else 42,
                tiny_synth=bool(a.tiny_synth),
                cache_dir=a.phase1_cache_dir,
            )
        except RuntimeError as exc:
            print(
                "PFC-FRAME-EFFECT-PROBE FAILED: %s" % exc,
                file=sys.stderr, flush=True,
            )
            return 2
        print(
            "PFC-FRAME-EFFECT-PROBE PASS: max |delta v_membrane| = "
            "%.6g mV (> 1.0 mV)" % diff_pfc,
            flush=True,
        )

    result = run_generative_replay_pfc_frame(
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
