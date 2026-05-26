"""Direction 6 Task 5: 5-bridge training runner (CONTROLLER-ONLY;
GPU-bound). V=32 per bridge x 5 = 160 cross-bridge concepts.

Trains 5 bio_brain_regions bridges (one per vocab category: noun /
verb / adjective / spatial / functional, V=32 each) across 3 seeds
[42, 43, 44] = 15 bridge trainings total. Captures per-bridge per-word
activity vectors across the bridge's V=32 category-pool neuron union;
writes per-seed-per-bridge .npz cache. After all bridges train,
optionally invokes direction_6_cross_bridge_probe (Task 4) inline to
emit the frozen Direction 6 verdict.

ETA (per CLAUDE.md v14/v16 production timing + V=32 doubling):
- Full scale (n_lang_input=2048, n_per_pool=200, n_events=200):
  ~60-120 min/bridge x 5 bridges x 3 seeds = ~15-30 hr GPU
- Smoke scale (n_lang_input=1024, n_per_pool=100, n_events=50):
  ~5-10 min/bridge x 5 bridges x 3 seeds = ~1.5-3 hr GPU

The smoke is a MECHANICAL PASS check (verifies the pipeline runs end-
to-end on all 5 bridges without API mismatch or OOM); numbers from
smoke are NOT propagated as a result. The decisive multi-seed full-
scale run is the controller's next step after this commits.

Pre-registered verdict (frozen): research/findings/raw/direction_6_verdict.py
Pre-registered cross-bridge probe primitive (reuse-by-import,
byte-unchanged): research/findings/raw/direction_6_cross_bridge_probe.py
(itself reusing pillar n=95 byte-unchanged via its own imports).

KILL-SAFE caches:
- Trained bridge per (bridge, seed):
  {CACHE_DIR}/bridge_{tag}_{bridge_name}_seed{N}.simstate.h5
- Per-(bridge, seed) activity cache:
  {CACHE_DIR}/activity_{tag}_{bridge_name}_seed{N}.npz
- Re-runs short-circuit both stages independently per bridge per seed.
- All npz files use numeric-only arrays (safety: no object arrays loaded).

Reuses every primitive byte-unchanged via import; no protected/frozen/
moat module modified; no autograd. Per-bridge topographic prior helper
is local to this runner (mirrors Direction 3's _apply_v32_topographic
_bias byte-pattern; only the per-bridge V=32 vocab differs).

STRIDE NOTE (deliberate, NOT tuned):
- At V=32 + n_lang_input=2048, stride = 2048/32 = 64 neurons per cue band.
- orthogonal_drive_pattern requires n_active = sparsity*n_lang <= stride.
- sparsity 0.03 -> n_active = 61 < 64 (clean margin; same as Direction 3
  V=32 production constant FULL_SPARSITY).
- At V=32 + n_lang_input=1024 (smoke), stride = 32. sparsity 0.02 ->
  n_active = 20 < 32.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Direction 6 modules (this arc's net-new code).
from research.findings.raw.direction_6_vocab_spec import (
    DIRECTION_6_BRIDGE_A_WORDS,
    DIRECTION_6_BRIDGE_B_WORDS,
    DIRECTION_6_BRIDGE_C_WORDS,
    DIRECTION_6_BRIDGE_D_WORDS,
    DIRECTION_6_BRIDGE_E_WORDS,
    DIRECTION_6_NOUN_NAMES,
    DIRECTION_6_VERB_NAMES,
    DIRECTION_6_ADJECTIVE_NAMES,
    DIRECTION_6_SPATIAL_NAMES,
    DIRECTION_6_FUNCTIONAL_NAMES,
    DIRECTION_6_BRIDGE_CATALOG,
    DIRECTION_6_TOTAL,
)
from research.findings.raw.direction_6_bridge_builder import (
    DIRECTION_6_BRIDGE_BUILDERS,
)
from research.findings.raw.direction_6_verdict import (
    _DIRECTION_6_OB_MIN, _DIRECTION_6_OI_MIN,
    _DIRECTION_6_LOADS, _DIRECTION_6_MIN_SEEDS,
)

# Reuse-by-import only (validated training + capture primitives;
# pillar n=96 OPTION 3 + Direction 3 V=32 + Direction 4 5-bridge patterns).
from research.findings.raw.vocabulary_scaling_run import (
    BAR, N_DIM, N_TRIALS,
)
from research.runners.concept_pool_demo import train_word_to_pool
from sim.backend import get_backend, is_gpu_backend


# -----------------------------------------------------------------------
# Per-bridge scale parameters (full and smoke; frozen for the runner).
# Mirrors Direction 4 5-bridge runner byte-pattern; the only mechanical
# difference: per-bridge V=32 instead of per-bridge V=16 -> sparsity
# adjusted to satisfy orthogonal_drive_pattern stride constraint.
# -----------------------------------------------------------------------
FULL_N_LANG_INPUT = 2048
FULL_N_PER_POOL = 200
FULL_N_FS_PER_POOL = 24
FULL_N_TRAIN_EVENTS = 200
# At V=32 + n_lang_input=2048, stride = 2048/32 = 64 neurons per cue
# band. orthogonal_drive_pattern requires n_active = sparsity*n_lang
# <= stride. sparsity 0.03 -> n_active=61 < 64 (clean margin).
# Matches Direction 3 V=32 FULL_SPARSITY constant byte-for-byte.
FULL_SPARSITY = 0.03
FULL_M_OBS = 16  # observations per concept (matches K_VOCAB_TARGET)
FULL_TOPOGRAPHIC_FACTOR = 3.0
FULL_OFF_TARGET_FACTOR = 0.3

SMOKE_N_LANG_INPUT = 1024
SMOKE_N_PER_POOL = 100
SMOKE_N_FS_PER_POOL = 12
SMOKE_N_TRAIN_EVENTS = 50
# At V=32 + n_lang_input=1024, stride = 32. sparsity 0.02 -> n_active=20
# < 32 (clean margin). Matches Direction 3 V=32 SMOKE_SPARSITY.
SMOKE_SPARSITY = 0.02
SMOKE_M_OBS = 8

SEEDS = [42, 43, 44]
LOADS = list(_DIRECTION_6_LOADS)
BAR_OB = _DIRECTION_6_OB_MIN
BAR_OI = _DIRECTION_6_OI_MIN

# Per-bridge per-seed cache directory.
CACHE_DIR = os.path.join(_HERE, "direction_6_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Per-bridge word + target-pool map (each bridge has its own V=32 vocab
# pointing to its own pool-name prefix).
_PER_BRIDGE_WORD_LISTS: Dict[str, List[str]] = {
    "A_nouns": DIRECTION_6_BRIDGE_A_WORDS,
    "B_verbs": DIRECTION_6_BRIDGE_B_WORDS,
    "C_adj": DIRECTION_6_BRIDGE_C_WORDS,
    "D_spatial": DIRECTION_6_BRIDGE_D_WORDS,
    "E_functional": DIRECTION_6_BRIDGE_E_WORDS,
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
    spec = DIRECTION_6_BRIDGE_CATALOG[bridge_name]
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
            "Direction 6 bridge_catalog has unknown builder_slot '"
            + str(slot) + "' for bridge " + bridge_name
        )
    return {
        w: prefix + str(pool_names[i])  # type: ignore[index]
        for i, w in enumerate(words)
    }


def _per_bridge_pool_region_names(bridge_name: str) -> List[str]:
    """Per-bridge list of V=32 concept-pool region names (in word-order
    matching _PER_BRIDGE_WORD_LISTS[bridge_name]). Used by activity
    capture to extract only the bridge's own category pools (motor
    pools are structurally present but not the cross-bridge concepts)."""
    target_map = _per_bridge_target_pool_map(bridge_name)
    return [target_map[w] for w in _PER_BRIDGE_WORD_LISTS[bridge_name]]


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


# -----------------------------------------------------------------------
# Per-bridge topographic-bias helper (V=32 per-bridge; mirrors
# Direction 4 _apply_per_bridge_topographic_bias byte-pattern, only the
# per-bridge vocab size differs).
# -----------------------------------------------------------------------
def _apply_per_bridge_topographic_bias(
    bridge, bridge_name: str, n_lang_input: int, sparsity: float,
    word_to_idx: Dict[str, int], topographic_factor: float,
    off_target_factor: float, verbose: bool = False,
):
    """Per-bridge topographic prior: boosts lang_input(w) -> target_pool(w)
    weights by topographic_factor; dampens lang_input(w) -> off_pool by
    off_target_factor. Same two-pass priority logic as v14/v16 +
    Direction 3 V=32 + Direction 4 5-bridge: each edge is "target" for
    exactly one word; an edge that is "off-target" for at least one word
    gets dampened once.

    Bridge has only V=32 category pools (its category) in the cross-
    bridge probe scope; motor pools also exist structurally but are
    untouched by the per-bridge bias (motor isn't part of the V=32
    cross-bridge concept set).
    """
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()

    def _to_host(arr):
        try:
            return cp.asnumpy(arr)
        except Exception:
            return np.asarray(arr)

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    if len(lang_input_indices) != n_lang_input:
        raise ValueError(
            "Direction 6 per-bridge topographic bias: bridge "
            + bridge_name + " has " + str(len(lang_input_indices))
            + " lang_input neurons but caller specified "
            + str(n_lang_input)
        )

    indptr = _to_host(bridge.cp_connections.indptr)
    indices = _to_host(bridge.cp_connections.indices)
    data = _to_host(bridge.cp_connections.data)

    # (pre, post) -> data offset lookup
    pair_to_idx: Dict[Tuple[int, int], int] = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    # All V=32 output pools for THIS bridge (motor pools deliberately
    # excluded; they aren't part of the cross-bridge concept set).
    all_output_pools = _per_bridge_pool_region_names(bridge_name)
    target_map = _per_bridge_target_pool_map(bridge_name)
    words = _PER_BRIDGE_WORD_LISTS[bridge_name]
    n_words = len(words)

    # Step 1: per-word active lang_input neuron sets via orthogonal codes
    word_to_active: Dict[str, List[int]] = {}
    for word in words:
        if word in word_to_active:
            continue
        drive = orthogonal_drive_pattern(
            cue_idx=word_to_idx[word], n_cues=n_words,
            n_neurons=n_lang_input, sparsity=sparsity,
        )
        local_active = np.where(drive > 0)[0]
        word_to_active[word] = [lang_input_indices[i] for i in local_active]

    # Pass 1 forward: target boost (lang_input(word) -> target_pool)
    target_edges = set()
    for word in words:
        target_region = target_map[word]
        peer_neurons = list(rm.indices(target_region))
        global_active = word_to_active[word]
        for src in global_active:
            for dst in peer_neurons:
                key = (src, dst)
                if key in pair_to_idx and key not in target_edges:
                    idx = pair_to_idx[key]
                    data[idx] = float(data[idx]) * topographic_factor
                    target_edges.add(key)

    # Pass 2 forward: off-target dampen (within-bridge V=32 pools only)
    dampened_edges = set()
    for word in words:
        target_region = target_map[word]
        global_active = word_to_active[word]
        for peer in all_output_pools:
            if peer == target_region:
                continue
            peer_neurons = list(rm.indices(peer))
            for src in global_active:
                for dst in peer_neurons:
                    key = (src, dst)
                    if (key in pair_to_idx
                            and key not in target_edges
                            and key not in dampened_edges):
                        idx = pair_to_idx[key]
                        data[idx] = float(data[idx]) * off_target_factor
                        dampened_edges.add(key)

    # Reciprocal: pool -> language_output (v9 pattern from Direction 3
    # V=32 + Direction 4 5-bridge).
    try:
        lang_output_indices_full = list(rm.indices("language_output"))
        n_lang_output_actual = len(lang_output_indices_full)
    except Exception:
        lang_output_indices_full = None
        n_lang_output_actual = 0

    target_edges_recip_count = 0
    dampened_edges_recip_count = 0
    if lang_output_indices_full is not None:
        # Per-word active lang_output (same orthogonal scheme as forward)
        word_to_lang_out_active: Dict[str, List[int]] = {}
        for word in word_to_active:
            drive = orthogonal_drive_pattern(
                cue_idx=word_to_idx[word], n_cues=n_words,
                n_neurons=n_lang_output_actual, sparsity=sparsity,
            )
            local_active = np.where(drive > 0)[0]
            word_to_lang_out_active[word] = [
                lang_output_indices_full[i] for i in local_active
            ]

        target_edges_recip = set()
        for word in words:
            target_region = target_map[word]
            pool_neurons = list(rm.indices(target_region))
            global_active_out = word_to_lang_out_active[word]
            for src in pool_neurons:
                for dst in global_active_out:
                    key = (src, dst)
                    if (key in pair_to_idx
                            and key not in target_edges_recip):
                        idx = pair_to_idx[key]
                        data[idx] = float(data[idx]) * topographic_factor
                        target_edges_recip.add(key)

        dampened_edges_recip = set()
        for word in words:
            target_region = target_map[word]
            for peer in all_output_pools:
                if peer == target_region:
                    continue
                peer_pool_neurons = list(rm.indices(peer))
                global_active_out = word_to_lang_out_active[word]
                for src in peer_pool_neurons:
                    for dst in global_active_out:
                        key = (src, dst)
                        if (key in pair_to_idx
                                and key not in target_edges_recip
                                and key not in dampened_edges_recip):
                            idx = pair_to_idx[key]
                            data[idx] = float(data[idx]) * off_target_factor
                            dampened_edges_recip.add(key)
        target_edges_recip_count = len(target_edges_recip)
        dampened_edges_recip_count = len(dampened_edges_recip)

    bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)

    if verbose:
        print(
            "[D6-" + bridge_name
            + " topographic-bias] forward target_edges="
            + str(len(target_edges))
            + " off-target dampened=" + str(len(dampened_edges))
            + " | reciprocal target_edges="
            + str(target_edges_recip_count)
            + " off=" + str(dampened_edges_recip_count),
            flush=True,
        )


# -----------------------------------------------------------------------
# Per-bridge per-seed build + train. KILL-SAFE via save_checkpoint.
# -----------------------------------------------------------------------
def _build_and_train(
    bridge_name: str, seed: int, smoke: bool, verbose: bool,
):
    """Build + train one (bridge, seed) cell. Returns the trained bridge
    + the bridge's V=32 word list + word_to_idx + sparsity + n_lang_input."""
    bridge_p = _bridge_save_path(bridge_name, seed, smoke)
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
    if n_words != 32:
        raise ValueError(
            "Direction 6 bridge " + bridge_name + " expected V=32 words; "
            "got " + str(n_words)
        )
    word_to_idx = {w: i for i, w in enumerate(words)}

    builder = DIRECTION_6_BRIDGE_BUILDERS[bridge_name]

    t0 = time.time()
    bridge = builder(
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
        # further STDP during capture / probe).
        for g in (
            "language_input_to_motor",
            "language_input_to_noun_pool",
            "language_input_to_verb_pool",
            "language_input_to_adjective_pool",
            "motor_to_language_output",
            "noun_pool_to_language_output",
            "verb_pool_to_language_output",
            "adjective_pool_to_language_output",
        ):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        return bridge, words, word_to_idx, sparsity, n_lang_input

    if verbose:
        print(
            "  [" + bridge_name + "/seed " + str(seed)
            + "] training V=32 substrate (" + str(n_words)
            + " words x " + str(n_train_events) + " events)",
            flush=True,
        )
    _apply_per_bridge_topographic_bias(
        bridge,
        bridge_name=bridge_name,
        n_lang_input=n_lang_input,
        sparsity=sparsity,
        word_to_idx=word_to_idx,
        topographic_factor=FULL_TOPOGRAPHIC_FACTOR,
        off_target_factor=FULL_OFF_TARGET_FACTOR,
        verbose=verbose,
    )

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
        if verbose and (ei + 1) % max(1, len(schedule) // 10) == 0:
            elapsed = (time.time() - t0) / 60
            print(
                "    [" + bridge_name + "/seed " + str(seed) + "] "
                + str(ei + 1) + "/" + str(len(schedule))
                + " events (" + ("%.1f" % elapsed) + " min)",
                flush=True,
            )
    bridge.save_checkpoint(bridge_p)
    if verbose:
        elapsed = (time.time() - t0) / 60
        print(
            "  [" + bridge_name + "/seed " + str(seed)
            + "] trained + saved in " + ("%.1f" % elapsed) + " min",
            flush=True,
        )
    return bridge, words, word_to_idx, sparsity, n_lang_input


# -----------------------------------------------------------------------
# Per-bridge activity capture (V=32 per-bridge category pools only).
# -----------------------------------------------------------------------
def _capture_concept_pool_activity(
    bridge, bridge_name: str, words: List[str],
    word_to_idx: Dict[str, int], sparsity: float, n_lang_input: int,
    m_obs: int, verbose: bool,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Per-neuron activity vectors across the V=32 per-bridge category
    pool union. Same shape / window / drive_max_pA as the OPTION 3 V=16
    probe + Direction 3 V=32 runner + Direction 4 5-bridge runner."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern

    rm = bridge.region_manager
    n_words = len(words)

    pool_names = _per_bridge_pool_region_names(bridge_name)
    pool_idx_lists: List[int] = []
    for p in pool_names:
        pool_idx_lists.extend(list(rm.indices(p)))
    pool_idx_arr_host = np.asarray(pool_idx_lists, dtype=np.int64)
    pool_idx_arr = cp.asarray(pool_idx_arr_host)
    n_pool_union = pool_idx_arr.shape[0]
    if verbose:
        print(
            "  [" + bridge_name + " capture] "
            + str(len(pool_names)) + " pools, "
            + str(n_pool_union) + " pool-union neurons",
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
                fired = bridge.cp_firing_states[pool_idx_arr]
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
    """Plain npz loader; numeric-only arrays (safety: object arrays
    rejected via the load mode)."""
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
    bridge, words, word_to_idx, sparsity, n_lang_input = _build_and_train(
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
        acts, n_pool_union = _capture_concept_pool_activity(
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
        description="Direction 6 5-bridge multi-seed training + cross-"
                    "bridge probe runner (CONTROLLER-ONLY; GPU-bound; "
                    "V=32 per bridge x 5 = 160 cross-bridge concepts)",
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
        help="train only; do NOT invoke direction_6_cross_bridge_probe "
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
        "=== Direction 6 5-bridge multi-seed training runner ===",
        flush=True,
    )
    print(
        "  backend=" + backend_name + " (GPU=" + str(gpu) + ")",
        flush=True,
    )
    print(
        "  V_per_bridge=32, n_bridges=5, V_total=" + str(DIRECTION_6_TOTAL),
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
        + "; min_seeds=" + str(_DIRECTION_6_MIN_SEEDS),
        flush=True,
    )

    seeds = list(args.seeds) if args.seeds is not None else list(SEEDS)
    if args.bridges is not None:
        bridges = list(args.bridges)
        for b in bridges:
            if b not in DIRECTION_6_BRIDGE_BUILDERS:
                raise ValueError(
                    "Unknown Direction 6 bridge name: " + str(b)
                    + " (expected one of "
                    + str(list(DIRECTION_6_BRIDGE_BUILDERS.keys())) + ")"
                )
    else:
        bridges = list(DIRECTION_6_BRIDGE_BUILDERS.keys())

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
        + " min (5 bridges x " + str(len(seeds)) + " seeds)",
        flush=True,
    )

    # -------------------------------------------------------------------
    # Optionally invoke Task 4 cross-bridge probe inline.
    # -------------------------------------------------------------------
    probe_result: Optional[Dict[str, object]] = None
    if not args.skip_probe:
        # Only run probe if we trained all 5 bridges + all 3 seeds (the
        # cross-bridge probe contractually requires the full 160-concept
        # union; a partial training cannot inform the cross-bridge
        # verdict).
        all_bridges_trained = (
            set(bridges) == set(DIRECTION_6_BRIDGE_BUILDERS.keys())
            and len(seeds) >= _DIRECTION_6_MIN_SEEDS
        )
        if all_bridges_trained:
            from research.findings.raw.direction_6_cross_bridge_probe import (
                run_cross_bridge_probe,
            )
            tag = "smoke" if smoke else "full"
            print(
                "\n=== Invoking Direction 6 cross-bridge probe (Task 4) ===",
                flush=True,
            )
            probe_result = run_cross_bridge_probe(
                seeds=seeds, tag=tag,
                cache_dir=CACHE_DIR, verbose=True,
            )
        else:
            print(
                "\n[skip probe] cross-bridge probe requires all 5 bridges "
                "trained and >= " + str(_DIRECTION_6_MIN_SEEDS)
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
        "min_seeds": _DIRECTION_6_MIN_SEEDS,
        "V_total": DIRECTION_6_TOTAL,
        "n_bridges": len(bridges),
        "training_log": train_log,
        "training_wall_clock_minutes": total_train_time,
        "probe_result": probe_result,
    }
    tag = "smoke" if smoke else "full"
    out_path = args.out or os.path.join(
        _HERE, "direction_6_5bridge_" + tag + ".json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote " + out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
