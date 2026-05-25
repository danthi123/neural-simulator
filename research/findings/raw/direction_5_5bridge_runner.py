"""Direction 5 Task 5: 5-bridge HYBRID training runner (CONTROLLER-ONLY;
GPU-bound).

Trains 5 HYBRID bio_brain_regions + shared-sparse-pool bridges (one per
vocab category: noun / verb / adjective / spatial / functional, V=16 each)
across 3 seeds [42, 43, 44] = 15 bridge trainings total. Per event,
drives BOTH the dedicated pool target (via train_word_to_pool, mirroring
v14/v16 Phase-1 attractor formation) AND the shared sparse pattern (via
a local helper that mirrors train_concept_sparse, lighting the K=100
sparse pattern as teacher current concurrently with the lang_input drive
so the lang_input -> shared_concept_pool weights grow correctly). The
HYBRID design (per design doc Approach A) requires BOTH substrates to
form per concept on every event.

Captures per-bridge per-word activity vectors across the bridge's
SHARED_CONCEPT_POOL region (uniform d_act = 2000 per bridge); writes
per-seed-per-bridge .npz cache. After all bridges train, optionally
invokes direction_5_cross_bridge_probe (Task 4) inline to emit the
frozen Direction 5 verdict.

ETA (per CLAUDE.md v14/v16 production timing + sparse-pathway overhead):
- Full scale (n_lang_input=2048, n_per_pool=200, n_events=200):
  ~30 min/bridge x 5 bridges x 3 seeds = ~7-8 hr GPU
- Smoke scale (n_lang_input=1024, n_per_pool=100, n_events=50):
  ~5-7 min/bridge x 5 bridges x 3 seeds = ~75-105 min GPU

The smoke is a MECHANICAL PASS check (verifies the pipeline runs end-
to-end on all 5 bridges without API mismatch or OOM); numbers from
smoke are NOT propagated as a result. The decisive multi-seed full-
scale run is the controller's next step after this commits.

Pre-registered verdict (frozen): research/findings/raw/direction_5_verdict.py
Pre-registered cross-bridge probe primitive (reuse-by-import,
byte-unchanged): research/findings/raw/direction_5_cross_bridge_probe.py
(itself reusing pillar n=95 + Direction 4 byte-unchanged via its own
imports).

KILL-SAFE caches:
- Trained bridge per (bridge, seed):
  {CACHE_DIR}/bridge_{tag}_{bridge_name}_seed{N}.simstate.h5
- Per-(bridge, seed) activity cache:
  {CACHE_DIR}/activity_{tag}_{bridge_name}_seed{N}.npz
- Re-runs short-circuit both stages independently per bridge per seed.
- All npz files use numeric-only arrays (no object dtype).

Reuses every primitive byte-unchanged via import; no protected/frozen/
moat module modified; no autograd.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Direction 5 modules (this arc's net-new code).
from research.findings.raw.direction_5_vocab_spec import (
    DIRECTION_5_BRIDGE_A_WORDS,
    DIRECTION_5_BRIDGE_B_WORDS,
    DIRECTION_5_BRIDGE_C_WORDS,
    DIRECTION_5_BRIDGE_D_WORDS,
    DIRECTION_5_BRIDGE_E_WORDS,
    DIRECTION_5_NOUN_NAMES,
    DIRECTION_5_VERB_NAMES,
    DIRECTION_5_ADJECTIVE_NAMES,
    DIRECTION_5_SPATIAL_NAMES,
    DIRECTION_5_FUNCTIONAL_NAMES,
    DIRECTION_5_BRIDGE_CATALOG,
    DIRECTION_5_TOTAL,
)
from research.findings.raw.direction_5_bridge_builder import (
    build_direction_5_bridge_A_nouns,
    build_direction_5_bridge_B_verbs,
    build_direction_5_bridge_C_adj,
    build_direction_5_bridge_D_spatial,
    build_direction_5_bridge_E_functional,
)
from research.findings.raw.direction_5_verdict import (
    _DIRECTION_5_OB_MIN, _DIRECTION_5_OI_MIN,
    _DIRECTION_5_LOADS, _DIRECTION_5_MIN_SEEDS,
)

# Reuse-by-import only (validated training + capture primitives;
# pillar n=95 + Direction 4 patterns).
from research.findings.raw.vocabulary_scaling_run import (
    BAR, N_DIM, N_TRIALS,
)
from research.runners.concept_pool_demo import train_word_to_pool
from sim.backend import get_backend, is_gpu_backend


# -----------------------------------------------------------------------
# Per-bridge scale parameters (full and smoke; frozen for the runner).
# Mirrors Direction 4 V=16 runner constants byte-pattern; the only
# mechanical difference: HYBRID architecture trains BOTH dedicated pool
# AND shared sparse pool per event.
# -----------------------------------------------------------------------
FULL_N_LANG_INPUT = 2048
FULL_N_PER_POOL = 200
FULL_N_FS_PER_POOL = 24
FULL_N_TRAIN_EVENTS = 200
# At V=16 + n_lang_input=2048, stride = 2048/16 = 128 neurons per cue
# band. orthogonal_drive_pattern requires n_active = sparsity*n_lang
# <= stride. sparsity 0.05 -> n_active=102 < 128 (clean margin).
# (This matches v14/v16 production recipe sparsity AND the D5 builder
# G.20 sparse-prior sparsity.)
FULL_SPARSITY = 0.05
FULL_M_OBS = 16  # observations per concept (matches K_VOCAB_TARGET)

SMOKE_N_LANG_INPUT = 1024
SMOKE_N_PER_POOL = 100
SMOKE_N_FS_PER_POOL = 12
SMOKE_N_TRAIN_EVENTS = 50
# At V=16 + n_lang_input=1024, stride = 64. sparsity 0.05 -> n_active 51
# < 64 (clean margin).
SMOKE_SPARSITY = 0.05
SMOKE_M_OBS = 8

# Sparse-pool teacher current (matches concept_pool_sparse_distributed
# train_concept_sparse default).
SPARSE_TEACHER_PA: float = 500.0
SPARSE_LANG_OUT_TEACHER_PA: float = 200.0

SEEDS = [42, 43, 44]
LOADS = list(_DIRECTION_5_LOADS)
BAR_OB = _DIRECTION_5_OB_MIN
BAR_OI = _DIRECTION_5_OI_MIN

# Per-bridge per-seed cache directory.
CACHE_DIR = os.path.join(_HERE, "direction_5_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Per-bridge word + target-pool map (each bridge has its own V=16 vocab
# pointing to its own pool-name prefix).
_PER_BRIDGE_WORD_LISTS: Dict[str, List[str]] = {
    "A_nouns": DIRECTION_5_BRIDGE_A_WORDS,
    "B_verbs": DIRECTION_5_BRIDGE_B_WORDS,
    "C_adj": DIRECTION_5_BRIDGE_C_WORDS,
    "D_spatial": DIRECTION_5_BRIDGE_D_WORDS,
    "E_functional": DIRECTION_5_BRIDGE_E_WORDS,
}

# Per-bridge builder dispatch (local to this runner; the D5 builder
# module exposes the 5 functions individually but no catalog).
_BRIDGE_BUILDERS = {
    "A_nouns": build_direction_5_bridge_A_nouns,
    "B_verbs": build_direction_5_bridge_B_verbs,
    "C_adj": build_direction_5_bridge_C_adj,
    "D_spatial": build_direction_5_bridge_D_spatial,
    "E_functional": build_direction_5_bridge_E_functional,
}


def _per_bridge_target_pool_map(bridge_name: str) -> Dict[str, str]:
    """Per-bridge word -> target_pool_region map. Bridge A/B/C use their
    dedicated pool kind; bridges D/E reuse the noun_pool_ prefix (same
    rationale as the bridge builder: substrate concept-pool architecture
    is category-agnostic at the pool level).

    The pool-region names match build_biological_brain_regions's region
    naming convention (noun_pool_<UPPER> / verb_pool_<UPPER> /
    adjective_pool_<UPPER>).
    """
    spec = DIRECTION_5_BRIDGE_CATALOG[bridge_name]
    words = spec["words"]  # type: ignore[index]
    pool_names = spec["pool_names"]  # type: ignore[index]
    slot = spec["builder_slot"]  # type: ignore[index]
    # Each builder_slot corresponds to a pool-name prefix.
    if slot == "noun_pool_names":
        prefix = "noun_pool_"
    elif slot == "verb_pool_names":
        prefix = "verb_pool_"
    elif slot == "adjective_pool_names":
        prefix = "adjective_pool_"
    else:
        raise ValueError(
            "Direction 5 bridge_catalog has unknown builder_slot '"
            + str(slot) + "' for bridge " + bridge_name
        )
    return {
        w: prefix + str(pool_names[i])  # type: ignore[index]
        for i, w in enumerate(words)
    }


def _bridge_save_path(bridge_name: str, seed: int, smoke: bool) -> str:
    tag = "smoke" if smoke else "full"
    return os.path.join(
        CACHE_DIR,
        "bridge_" + tag + "_" + bridge_name + "_seed" + str(seed)
        + ".simstate.h5",
    )


def _activity_cache_path(bridge_name: str, seed: int, smoke: bool) -> str:
    tag = "smoke" if smoke else "full"
    return os.path.join(
        CACHE_DIR,
        "activity_" + tag + "_" + bridge_name + "_seed" + str(seed)
        + ".npz",
    )


def _sparse_patterns_cache_path(
    bridge_name: str, seed: int, smoke: bool,
) -> str:
    """Cache for per-bridge per-seed sparse patterns (preserved alongside
    the trained bridge so that activity-capture + post-hoc inspection
    have access to the deterministic K=100 indices without re-running
    the builder).
    """
    tag = "smoke" if smoke else "full"
    return os.path.join(
        CACHE_DIR,
        "sparse_patterns_" + tag + "_" + bridge_name + "_seed"
        + str(seed) + ".npz",
    )


# -----------------------------------------------------------------------
# Local helper: drive sparse pattern as teacher current concurrently
# with lang_input drive for ONE event. Mirrors train_concept_sparse but
# parameterised so that the same per-event call drives the shared sparse
# substrate to fire at the K=100 pattern; lang_input -> shared_concept_pool
# STDP then grows the binding weights for the active lang_input neurons.
# -----------------------------------------------------------------------
def _drive_sparse_event(
    bridge, word_idx: int, sparse_pattern: List[int],
    n_lang_input: int, n_lang_output: int, sparsity: float,
    n_words_for_orthogonal: int,
    teacher_pA: float = SPARSE_TEACHER_PA,
    lang_output_teacher_pA: float = SPARSE_LANG_OUT_TEACHER_PA,
):
    """Drive ONE sparse-pattern teacher event for word=word_idx.

    Mirrors concept_pool_sparse_distributed.train_concept_sparse but
    called from the per-event loop alongside train_word_to_pool. The
    reset (20 steps) + drive (1 step) + cooldown (10 steps) sequence is
    intentionally the same as train_concept_sparse so that the
    shared_concept_pool sees identical training stimulus shape per
    concept (and the lang_input -> shared_concept_pool plastic pathway
    grows weights per the apply_sparse_topographic_prior + per-event
    STDP).
    """
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")),
                            dtype=cp.int64)
    lang_out_arr = cp.asarray(list(rm.indices("language_output")),
                                dtype=cp.int64)
    shared_indices = list(rm.indices("shared_concept_pool"))
    pattern_global = [shared_indices[i] for i in sparse_pattern]
    pattern_arr = cp.asarray(pattern_global, dtype=cp.int64)

    drive_in = orthogonal_drive_pattern(
        cue_idx=word_idx, n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    drive_in_arr = cp.asarray(drive_in, dtype=cp.float32)
    drive_out = orthogonal_drive_pattern(
        cue_idx=word_idx, n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_output, drive_max_pA=lang_output_teacher_pA,
        sparsity=sparsity,
    )
    drive_out_arr = cp.asarray(drive_out, dtype=cp.float32)
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    ext.fill(0)
    ext[lang_arr] = drive_in_arr
    ext[lang_out_arr] = drive_out_arr
    ext[pattern_arr] = teacher_pA
    bridge.cp_external_input_current[:] = ext
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(10):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1


# -----------------------------------------------------------------------
# Per-bridge per-seed build + train. KILL-SAFE via save_checkpoint.
# -----------------------------------------------------------------------
def _build_and_train(
    bridge_name: str, seed: int, smoke: bool, verbose: bool,
) -> Tuple[Any, List[str], Dict[str, int], float, int, List[List[int]]]:
    """Build + train one (bridge, seed) cell. Returns the trained
    bridge + the bridge's V=16 word list + word_to_idx + sparsity +
    n_lang_input + sparse_patterns (the K=100 patterns for the
    bridge's 16 concepts; needed for activity capture / post-hoc
    inspection)."""
    bridge_p = _bridge_save_path(bridge_name, seed, smoke)
    sparse_p = _sparse_patterns_cache_path(bridge_name, seed, smoke)
    if smoke:
        n_lang_input = SMOKE_N_LANG_INPUT
        n_per_pool = SMOKE_N_PER_POOL
        n_fs_per_pool = SMOKE_N_FS_PER_POOL
        n_train_events = SMOKE_N_TRAIN_EVENTS
        sparsity = SMOKE_SPARSITY
    else:
        n_lang_input = FULL_N_LANG_INPUT
        n_per_pool = FULL_N_PER_POOL
        n_fs_per_pool = FULL_N_FS_PER_POOL
        n_train_events = FULL_N_TRAIN_EVENTS
        sparsity = FULL_SPARSITY

    words = list(_PER_BRIDGE_WORD_LISTS[bridge_name])
    n_words = len(words)
    if n_words != 16:
        raise ValueError(
            "Direction 5 bridge " + bridge_name + " expected V=16 words; "
            "got " + str(n_words)
        )
    word_to_idx = {w: i for i, w in enumerate(words)}

    builder = _BRIDGE_BUILDERS[bridge_name]

    t0 = time.time()
    bridge, sparse_patterns, build_meta = builder(
        seed=seed,
        n_lang_input=n_lang_input,
        n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool,
        weak_dynamics=True,
        verbose=verbose,
    )

    if os.path.exists(bridge_p):
        if verbose:
            print(
                "  [" + bridge_name + "/seed " + str(seed)
                + "] loading cached trained bridge (" + bridge_p + ")",
                flush=True,
            )
        bridge.load_checkpoint(bridge_p)
        # Freeze plasticity gates for the activity-capture phase (no
        # further STDP during capture / probe). Cover BOTH the
        # dedicated-pool gates AND the new shared-pool gate.
        for g in (
            "language_input_to_motor",
            "language_input_to_noun_pool",
            "language_input_to_verb_pool",
            "language_input_to_adjective_pool",
            "language_input_to_shared",
            "motor_to_language_output",
            "noun_pool_to_language_output",
            "verb_pool_to_language_output",
            "adjective_pool_to_language_output",
        ):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        return (bridge, words, word_to_idx, sparsity, n_lang_input,
                sparse_patterns)

    if verbose:
        print(
            "  [" + bridge_name + "/seed " + str(seed)
            + "] training HYBRID V=16 substrate (" + str(n_words)
            + " words x " + str(n_train_events) + " events; "
            + "dedicated pools AND shared sparse pool per event)",
            flush=True,
        )

    # Open the shared-pool gate during training (the dedicated-pool gates
    # are opened by train_word_to_pool per-word; the shared-pool gate
    # MUST be open throughout so the lang_input -> shared_concept_pool
    # STDP can grow weights on every sparse-teacher event).
    try:
        bridge.set_plasticity_gate("language_input_to_shared", 1.0)
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    target_pool = _per_bridge_target_pool_map(bridge_name)

    schedule = []
    for w in words:
        for _ in range(n_train_events):
            schedule.append(w)
    rng.shuffle(schedule)
    if verbose:
        print(
            "  [" + bridge_name + "/seed " + str(seed)
            + "] interleaved schedule: " + str(len(schedule)) + " events",
            flush=True,
        )

    for ei, w in enumerate(schedule):
        # 1. DEDICATED pool training (Tier 1 / v14/v16 attractor formation
        # for the dedicated category pool). Mirrors Direction 4 byte-
        # pattern.
        train_word_to_pool(
            bridge, word=w, target_pool_region=target_pool[w],
            n_events=1,
            n_lang_input=n_lang_input,
            n_lang_output=n_lang_input,
            sparsity=sparsity,
            orthogonal_codes=True,
            n_words_for_orthogonal=n_words,
            word_to_idx=word_to_idx,
            verbose=False,
        )
        # 2. SHARED sparse-pool training (per design doc Approach A:
        # "drive the K=100 sparse pattern in the shared pool" AT THE
        # SAME training time as the dedicated pool). The lang_input ->
        # shared_concept_pool STDP grows weights for the active
        # lang_input neurons paired with the teacher-driven sparse
        # pattern.
        _drive_sparse_event(
            bridge, word_idx=word_to_idx[w],
            sparse_pattern=sparse_patterns[word_to_idx[w]],
            n_lang_input=n_lang_input,
            n_lang_output=n_lang_input,
            sparsity=sparsity,
            n_words_for_orthogonal=n_words,
        )
        if verbose and (ei + 1) % max(1, len(schedule) // 10) == 0:
            elapsed = (time.time() - t0) / 60
            print(
                "    [" + bridge_name + "/seed " + str(seed) + "] "
                + str(ei + 1) + "/" + str(len(schedule))
                + " events (" + ("%.1f" % elapsed) + " min)",
                flush=True,
            )

    # Persist trained bridge + sparse patterns for kill-safe re-runs.
    bridge.save_checkpoint(bridge_p)
    np.savez_compressed(
        sparse_p,
        **{
            "pattern_" + str(i): np.asarray(p, dtype=np.int32)
            for i, p in enumerate(sparse_patterns)
        },
    )
    if verbose:
        elapsed = (time.time() - t0) / 60
        print(
            "  [" + bridge_name + "/seed " + str(seed)
            + "] trained + saved (bridge + sparse patterns) in "
            + ("%.1f" % elapsed) + " min",
            flush=True,
        )
    return (bridge, words, word_to_idx, sparsity, n_lang_input,
            sparse_patterns)


# -----------------------------------------------------------------------
# Per-bridge activity capture (SHARED_CONCEPT_POOL region only;
# d_act = 2000 uniform across all 5 bridges). The cross-bridge probe
# at Task 4 reads OUT of this readout.
# -----------------------------------------------------------------------
def _capture_shared_pool_activity(
    bridge, bridge_name: str, words: List[str],
    word_to_idx: Dict[str, int], sparsity: float, n_lang_input: int,
    m_obs: int, verbose: bool,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Per-neuron activity vectors across the SHARED_CONCEPT_POOL region
    (2000 neurons uniform per bridge per the D5 HYBRID architecture).
    Same shape / window / drive_max_pA as the OPTION 3 V=16 probe +
    Direction 4 V=16 runner, except the readout region is the shared
    sparse pool (not the dedicated pool union)."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern

    rm = bridge.region_manager
    n_words = len(words)

    shared_idx_list = list(rm.indices("shared_concept_pool"))
    shared_idx_arr_host = np.asarray(shared_idx_list, dtype=np.int64)
    shared_idx_arr = cp.asarray(shared_idx_arr_host)
    n_pool_union = shared_idx_arr.shape[0]
    if verbose:
        print(
            "  [" + bridge_name + " capture] shared_concept_pool: "
            + str(n_pool_union) + " neurons (uniform d_act)",
            flush=True,
        )

    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr_host = np.asarray(lang_input_idx, dtype=np.int64)
    lang_input_arr = cp.asarray(lang_input_arr_host)

    stim_steps = 50
    reset_steps = 25

    acts: Dict[str, np.ndarray] = {}
    for w in words:
        observations = np.zeros((m_obs, n_pool_union), dtype=np.float32)
        for obs in range(m_obs):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            drive_in = orthogonal_drive_pattern(
                cue_idx=word_to_idx[w], n_cues=n_words,
                n_neurons=n_lang_input,
                drive_max_pA=200.0, sparsity=sparsity,
            )
            drive_in_gpu = cp.asarray(drive_in, dtype=cp.float32)
            bridge.cp_external_input_current[lang_input_arr] = drive_in_gpu
            counts = cp.zeros(n_pool_union, dtype=cp.float32)
            for _ in range(stim_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                fired = bridge.cp_firing_states[shared_idx_arr]
                counts = counts + fired.astype(cp.float32)
            observations[obs] = cp.asnumpy(counts)
        acts[w] = observations
        if verbose:
            density = float(np.mean(observations > 0.0))
            mean_rate = float(np.mean(observations))
            print(
                "    [" + bridge_name + " capture] '" + w
                + "': mean_rate " + ("%.4f" % mean_rate)
                + " density " + ("%.4f" % density),
                flush=True,
            )
    return acts, n_pool_union


def _load_activity_cache(
    cache_p: str, words: List[str],
) -> Dict[str, np.ndarray]:
    """Plain npz loader; numeric arrays only (safe load: no object dtype)."""
    data = np.load(cache_p, allow_pickle=False)
    return {w: data[str(w)] for w in words}


# -----------------------------------------------------------------------
# Train one (bridge, seed) cell + capture activity + write cache.
# -----------------------------------------------------------------------
def train_one_bridge(
    bridge_name: str, seed: int, smoke: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Build + train one (bridge, seed) cell; capture activity; write
    per-bridge-per-seed npz cache; return acts + n_pool_union.

    KILL-SAFE: re-runs short-circuit both train + capture stages
    independently per (bridge, seed) via on-disk cache existence checks.
    """
    if verbose:
        print(
            "\n--- " + bridge_name + " / seed " + str(seed) + " ---",
            flush=True,
        )
    (bridge, words, word_to_idx, sparsity, n_lang_input,
     sparse_patterns) = _build_and_train(
        bridge_name, seed, smoke, verbose,
    )

    cache_p = _activity_cache_path(bridge_name, seed, smoke)
    m_obs = SMOKE_M_OBS if smoke else FULL_M_OBS
    if os.path.exists(cache_p):
        if verbose:
            print(
                "  [" + bridge_name + "/seed " + str(seed)
                + "] loading cached activity (" + cache_p + ")",
                flush=True,
            )
        acts = _load_activity_cache(cache_p, words)
        n_pool_union = acts[words[0]].shape[1]
    else:
        acts, n_pool_union = _capture_shared_pool_activity(
            bridge, bridge_name, words, word_to_idx, sparsity,
            n_lang_input, m_obs, verbose,
        )
        np.savez_compressed(
            cache_p, **{str(w): acts[w] for w in words},
        )
        if verbose:
            print(
                "  [" + bridge_name + "/seed " + str(seed)
                + "] cached activity (" + cache_p + ")",
                flush=True,
            )
    return acts, n_pool_union


def main():
    ap = argparse.ArgumentParser(
        description="Direction 5 5-bridge HYBRID multi-seed training + "
                    "cross-bridge probe runner (CONTROLLER-ONLY; "
                    "GPU-bound)",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="reduced-scale smoke (n_lang=1024, n_per_pool=100, "
             "events=50; numbers NOT propagated as a result)",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="seeds to train; default [42, 43, 44]",
    )
    ap.add_argument(
        "--bridges", type=str, nargs="+", default=None,
        help="bridges to train (subset of [A_nouns, B_verbs, C_adj, "
             "D_spatial, E_functional]); default = all 5",
    )
    ap.add_argument(
        "--skip-probe", action="store_true",
        help="train only; do NOT invoke direction_5_cross_bridge_probe "
             "inline. Caller invokes probe separately later.",
    )
    ap.add_argument(
        "--out", default=None,
        help="output JSON path (default: side-by-side with this module)",
    )
    args = ap.parse_args()
    smoke = bool(args.smoke)

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(
        "=== Direction 5 5-bridge HYBRID multi-seed training runner ===",
        flush=True,
    )
    print(
        "  backend=" + backend_name + " (GPU=" + str(gpu) + ")",
        flush=True,
    )
    print(
        "  V_per_bridge=16, n_bridges=5, V_total=" + str(DIRECTION_5_TOTAL),
        flush=True,
    )
    print(
        "  architecture=HYBRID dedicated bio pools + 2000-neuron shared "
        "sparse pool (G.20 pillar n=95 K=100 per concept)",
        flush=True,
    )
    if smoke:
        print(
            "  *** SMOKE MODE: reduced scale; numbers NOT propagated "
            "as a result ***",
            flush=True,
        )
    print(
        "  bar=" + str(BAR_OB) + "; loads=" + str(LOADS)
        + "; min_seeds=" + str(_DIRECTION_5_MIN_SEEDS),
        flush=True,
    )

    seeds = list(args.seeds) if args.seeds is not None else list(SEEDS)
    if args.bridges is not None:
        bridges = list(args.bridges)
        for b in bridges:
            if b not in _BRIDGE_BUILDERS:
                raise ValueError(
                    "Unknown Direction 5 bridge name: " + str(b)
                    + " (expected one of "
                    + str(list(_BRIDGE_BUILDERS.keys())) + ")"
                )
    else:
        bridges = list(_BRIDGE_BUILDERS.keys())

    print(
        "  bridges=" + str(bridges) + "; seeds=" + str(seeds),
        flush=True,
    )

    # -------------------------------------------------------------------
    # Train all (bridge, seed) cells (one at a time; KILL-SAFE per cell).
    # -------------------------------------------------------------------
    t0_all = time.time()
    train_log: List[Dict[str, object]] = []
    for bridge_name in bridges:
        for seed in seeds:
            t_cell = time.time()
            try:
                acts, n_pool_union = train_one_bridge(
                    bridge_name=bridge_name, seed=seed, smoke=smoke,
                    verbose=True,
                )
                elapsed_cell = (time.time() - t_cell) / 60
                train_log.append({
                    "bridge": bridge_name, "seed": seed,
                    "n_pool_union": int(n_pool_union),
                    "wall_clock_minutes": elapsed_cell,
                    "status": "OK",
                })
                print(
                    "  [" + bridge_name + "/seed " + str(seed) + " done in "
                    + ("%.1f" % elapsed_cell) + " min]",
                    flush=True,
                )
            except Exception as exc:
                elapsed_cell = (time.time() - t_cell) / 60
                train_log.append({
                    "bridge": bridge_name, "seed": seed,
                    "wall_clock_minutes": elapsed_cell,
                    "status": "FAILED",
                    "error": str(exc),
                })
                print(
                    "  [" + bridge_name + "/seed " + str(seed)
                    + " FAILED in " + ("%.1f" % elapsed_cell) + " min]: "
                    + str(exc),
                    flush=True,
                )
                raise

    total_train_time = (time.time() - t0_all) / 60
    print(
        "\nTotal training wall-clock: " + ("%.1f" % total_train_time)
        + " min (" + str(len(bridges)) + " bridges x "
        + str(len(seeds)) + " seeds)",
        flush=True,
    )

    # -------------------------------------------------------------------
    # Optionally invoke Task 4 cross-bridge probe inline.
    # -------------------------------------------------------------------
    probe_result: Optional[Dict[str, object]] = None
    if not args.skip_probe:
        # Only run probe if we trained all 5 bridges + all 3 seeds (the
        # cross-bridge probe contractually requires the full 80-concept
        # union; a partial training cannot inform the cross-bridge
        # verdict).
        all_bridges_trained = (
            set(bridges) == set(_BRIDGE_BUILDERS.keys())
            and len(seeds) >= _DIRECTION_5_MIN_SEEDS
        )
        if all_bridges_trained:
            from research.findings.raw.direction_5_cross_bridge_probe import (
                run_cross_bridge_probe,
            )
            tag = "smoke" if smoke else "full"
            print(
                "\n=== Invoking Direction 5 cross-bridge probe (Task 4) ===",
                flush=True,
            )
            probe_result = run_cross_bridge_probe(
                seeds=seeds, tag=tag,
                cache_dir=CACHE_DIR, verbose=True,
            )
        else:
            print(
                "\n[skip probe] cross-bridge probe requires all 5 bridges "
                "trained and >= " + str(_DIRECTION_5_MIN_SEEDS)
                + " seeds; have " + str(len(bridges)) + " bridges, "
                + str(len(seeds)) + " seeds",
                flush=True,
            )

    # -------------------------------------------------------------------
    # Emit run-level summary JSON (training log + probe verdict if run).
    # -------------------------------------------------------------------
    out = {
        "backend": backend_name, "gpu": gpu,
        "smoke": smoke,
        "seeds": seeds, "bridges": bridges,
        "loads": LOADS,
        "bar_ob": BAR_OB, "bar_oi": BAR_OI,
        "min_seeds": _DIRECTION_5_MIN_SEEDS,
        "V_total": DIRECTION_5_TOTAL,
        "n_bridges": len(bridges),
        "architecture": (
            "hybrid_bio_brain_regions_plus_shared_sparse_pool"
        ),
        "shared_pool_n_neurons": 2000,
        "shared_pool_pattern_size": 100,
        "shared_pool_prior_strength": (
            "topographic_factor=10.0, off_target_factor=0.1"
        ),
        "training_log": train_log,
        "training_wall_clock_minutes": total_train_time,
        "probe_result": probe_result,
    }
    tag = "smoke" if smoke else "full"
    out_path = args.out or os.path.join(
        _HERE, "direction_5_5bridge_" + tag + ".json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote " + out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
