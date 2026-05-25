"""Direction 3 V=32 multi-seed runner: parallel-matching mode-unification
on the bio_brain_regions V=32 substrate.

The pre-registered Direction 3 test: does the validated parallel-matching
biologized mode-unification (pillars n=93/n=94, OPTION 3 PASS at V=16)
still PASS when the vocabulary doubles to V=32 (4 motor + 12 noun +
12 verb + 4 adjective = 32 distinct concept pools) on the same
bio_brain_regions substrate?

The pipeline reuses every primitive from the validated OPTION 3 V=16
probe byte-unchanged:
- v14/v16 substrate training (via direction_3_bridge_builder) +
  per-word topographic prior (V=32-aware extension, this runner)
- Per-word activity capture across the concept-pool union
- Mean-centred phasor-symbol derivation (DERIV_SEED=90909)
- Resonate-and-fire FHRR bind + unbind
- Parallel-population-matching decoder (batched_phase_similarity over
  the 32 substrate-derived grounded symbols)
- Pre-registered loads L=[2, 3, 5] + bar 0.80 multi-seed

WALL-CLOCK ESTIMATE:
- Full scale (n_lang_input=2048, n_per_pool=200, n_events=200):
  ~2-3 hr per seed * 3 seeds = 6-9 hr GPU
- Smoke scale (n_lang_input=1024, n_per_pool=100, n_events=100):
  ~5-15 min per seed * 3 seeds = 15-45 min GPU

The smoke is a MECHANICAL PASS check (verifies the pipeline runs end-
to-end on V=32 without API mismatch or OOM); numbers from smoke are
NOT propagated as a result. The decisive multi-seed full-scale run is
the controller's next step after this commits.

Pre-registered verdict (frozen): research/findings/raw/direction_3_verdict.py

KILL-SAFE caches:
- Trained bridge per seed: {CACHE_DIR}/bridge_{tag}_seed{N}.simstate.h5
- Per-seed activity cache: {CACHE_DIR}/activity_{tag}_seed{N}.npz
- Re-runs short-circuit both stages.

Reuses every primitive byte-unchanged; no protected/frozen/moat module
modified; no autograd.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Direction 3 modules (this arc's net-new code)
from research.findings.raw.direction_3_vocab_spec import (
    DIRECTION_3_V32_WORDS, DIRECTION_3_V32_TARGET_POOL,
    DIRECTION_3_V32_TOTAL,
    DIRECTION_3_NOUN_NAMES, DIRECTION_3_VERB_NAMES,
    DIRECTION_3_ADJECTIVE_NAMES, DIRECTION_3_MOTOR_NAMES,
)
from research.findings.raw.direction_3_bridge_builder import (
    build_direction_3_v32_bridge,
)
from research.findings.raw.direction_3_verdict import (
    compute_verdict, _DIRECTION_3_V32_OB_MIN, _DIRECTION_3_V32_OI_MIN,
    _DIRECTION_3_V32_LOADS, _DIRECTION_3_V32_MIN_SEEDS,
    DIRECTION_3_V32_VOID_MALFORMED,
)

# Reuse-by-import only (validated mode-unification primitives).
from research.findings.raw.vocabulary_scaling_run import (
    BAR, N_DIM, N_TRIALS,
)
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    DERIV_SEED,
)
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.findings.raw.cross_bridge_mode_unification_probe import (
    batched_phase_similarity, verify_batched_equivalent_to_scalar,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from research.runners.concept_pool_demo import train_word_to_pool
from sim.backend import get_backend, is_gpu_backend, to_host


# -----------------------------------------------------------------------
# V=32 scale parameters (frozen for the runner; smoke overrides apply).
# -----------------------------------------------------------------------
FULL_N_LANG_INPUT = 2048
FULL_N_PER_POOL = 200
FULL_N_FS_PER_POOL = 24
FULL_N_TRAIN_EVENTS = 200
# At V=32 + n_lang_input=2048, stride = 2048/32 = 64 neurons per cue band.
# orthogonal_drive_pattern requires n_active = sparsity*n_neurons <= stride.
# sparsity 0.03 -> n_active = 61 < 64 (clean margin; deliberate, NOT tuned).
FULL_SPARSITY = 0.03
FULL_M_OBS = 16  # observations per concept (matches V=16 K_VOCAB_TARGET)
FULL_TOPOGRAPHIC_FACTOR = 3.0
FULL_OFF_TARGET_FACTOR = 0.3

SMOKE_N_LANG_INPUT = 1024
SMOKE_N_PER_POOL = 100
SMOKE_N_FS_PER_POOL = 12
SMOKE_N_TRAIN_EVENTS = 100
# At V=32 + n_lang_input=1024, stride = 32. sparsity 0.02 -> n_active 20 < 32.
SMOKE_SPARSITY = 0.02
SMOKE_M_OBS = 8

SEEDS = [42, 43, 44]
LOADS = list(_DIRECTION_3_V32_LOADS)
BAR_OB = _DIRECTION_3_V32_OB_MIN
BAR_OI = _DIRECTION_3_V32_OI_MIN

CACHE_DIR = os.path.join(_HERE, "direction_3_v32_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _bridge_save_path(seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, "bridge_" + tag + "_seed" + str(seed)
                          + ".simstate.h5")


def _activity_cache_path(seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, "activity_" + tag + "_seed" + str(seed)
                          + ".npz")


def _apply_v32_topographic_bias(bridge, n_lang_input, sparsity,
                                  word_to_idx, n_words,
                                  topographic_factor, off_target_factor,
                                  verbose=False):
    """V=32-aware topographic prior, generalising the V=16 helper from
    research.runners.concept_pool_demo.apply_concept_topographic_bias.

    The V=16 helper hardcodes the 4+4+4+4 vocab; this V=32 generalisation
    uses the frozen DIRECTION_3_V32_* spec. Same two-pass priority logic
    (target boost; off-target dampen once); same Pulvermueller 2003
    rationale (cortical somatotopy).

    Mirrors the V=16 helper at the algorithmic level (same
    Pass 1 / Pass 2 / reciprocal pattern), only the vocab discovery is
    V=32-aware.
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
            "V=32 topographic bias: bridge has "
            + str(len(lang_input_indices)) + " lang_input neurons but "
            "caller specified " + str(n_lang_input)
        )

    indptr = _to_host(bridge.cp_connections.indptr)
    indices = _to_host(bridge.cp_connections.indices)
    data = _to_host(bridge.cp_connections.data)

    # (pre, post) -> data offset lookup
    pair_to_idx = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    # All V=32 output pools
    all_output_pools = [
        "motor_" + a for a in DIRECTION_3_MOTOR_NAMES
    ]
    all_output_pools += ["noun_pool_" + n for n in DIRECTION_3_NOUN_NAMES]
    all_output_pools += ["verb_pool_" + v for v in DIRECTION_3_VERB_NAMES]
    all_output_pools += ["adjective_pool_" + a
                          for a in DIRECTION_3_ADJECTIVE_NAMES]

    # Build (word, target_pool, peers) tuples
    bias_tasks = []
    for word, target_region in DIRECTION_3_V32_TARGET_POOL.items():
        bias_tasks.append((word, target_region, all_output_pools))

    # Step 1: per-word active lang_input neuron sets via orthogonal codes
    word_to_active = {}
    for word, _, _ in bias_tasks:
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
    for word, target_region, _ in bias_tasks:
        peer_neurons = list(rm.indices(target_region))
        global_active = word_to_active[word]
        for src in global_active:
            for dst in peer_neurons:
                key = (src, dst)
                if key in pair_to_idx and key not in target_edges:
                    idx = pair_to_idx[key]
                    data[idx] = float(data[idx]) * topographic_factor
                    target_edges.add(key)

    # Pass 2 forward: off-target dampen
    dampened_edges = set()
    for word, target_region, peer_regions in bias_tasks:
        global_active = word_to_active[word]
        for peer in peer_regions:
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

    # Reciprocal: pool -> language_output (v9 pattern)
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
        word_to_lang_out_active = {}
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
        for word, target_region, _ in bias_tasks:
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
        for word, target_region, peer_regions in bias_tasks:
            for peer in peer_regions:
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
        print("[V=32 topographic-bias] forward target_edges="
              + str(len(target_edges))
              + " off-target dampened=" + str(len(dampened_edges))
              + " | reciprocal target_edges="
              + str(target_edges_recip_count)
              + " off=" + str(dampened_edges_recip_count),
              flush=True)


def _build_and_train(seed, smoke, verbose):
    """V=32 build + train; kill-safe via save_checkpoint."""
    bridge_p = _bridge_save_path(seed, smoke)
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

    words = list(DIRECTION_3_V32_WORDS)
    n_words = len(words)
    if n_words != DIRECTION_3_V32_TOTAL:
        raise ValueError(
            "V=32 spec mismatch: DIRECTION_3_V32_WORDS has "
            + str(n_words) + " but DIRECTION_3_V32_TOTAL is "
            + str(DIRECTION_3_V32_TOTAL)
        )
    word_to_idx = {w: i for i, w in enumerate(words)}

    t0 = time.time()
    bridge = build_direction_3_v32_bridge(
        seed=seed,
        n_lang_input=n_lang_input,
        n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool,
        weak_dynamics=True,
        verbose=verbose,
    )

    if os.path.exists(bridge_p):
        if verbose:
            print("  [seed " + str(seed) + "] loading cached trained "
                  "bridge (" + bridge_p + ")", flush=True)
        bridge.load_checkpoint(bridge_p)
        # Freeze plasticity gates for the OPTION 3 capture phase (no
        # further STDP during capture / probe).
        for g in ("language_input_to_motor",
                  "language_input_to_noun_pool",
                  "language_input_to_verb_pool",
                  "language_input_to_adjective_pool",
                  "motor_to_language_output",
                  "noun_pool_to_language_output",
                  "verb_pool_to_language_output",
                  "adjective_pool_to_language_output"):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        return bridge, words, word_to_idx, sparsity, n_lang_input

    if verbose:
        print("  [seed " + str(seed) + "] training V=32 substrate ("
              + str(n_words) + " words x " + str(n_train_events)
              + " events)", flush=True)
    _apply_v32_topographic_bias(
        bridge,
        n_lang_input=n_lang_input,
        sparsity=sparsity,
        word_to_idx=word_to_idx,
        n_words=n_words,
        topographic_factor=FULL_TOPOGRAPHIC_FACTOR,
        off_target_factor=FULL_OFF_TARGET_FACTOR,
        verbose=verbose,
    )

    rng = np.random.default_rng(seed)
    target_pool = DIRECTION_3_V32_TARGET_POOL

    schedule = []
    for w in words:
        for _ in range(n_train_events):
            schedule.append(w)
    rng.shuffle(schedule)
    if verbose:
        print("  [seed " + str(seed) + "] interleaved schedule: "
              + str(len(schedule)) + " events", flush=True)

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
            print("    [seed " + str(seed) + "] " + str(ei + 1) + "/"
                  + str(len(schedule)) + " events ("
                  + ("%.1f" % elapsed) + " min)", flush=True)
    bridge.save_checkpoint(bridge_p)
    if verbose:
        elapsed = (time.time() - t0) / 60
        print("  [seed " + str(seed) + "] trained + saved in "
              + ("%.1f" % elapsed) + " min", flush=True)
    return bridge, words, word_to_idx, sparsity, n_lang_input


def _capture_concept_pool_activity(bridge, words, word_to_idx,
                                     sparsity, n_lang_input,
                                     m_obs, verbose):
    """Per-neuron activity vectors across the V=32 concept-pool union.
    Same shape / window / drive_max_pA as the OPTION 3 V=16 probe."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern

    rm = bridge.region_manager
    n_words = len(words)

    pool_names = ["motor_" + a for a in DIRECTION_3_MOTOR_NAMES]
    pool_names += ["noun_pool_" + n for n in DIRECTION_3_NOUN_NAMES]
    pool_names += ["verb_pool_" + v for v in DIRECTION_3_VERB_NAMES]
    pool_names += ["adjective_pool_" + a
                    for a in DIRECTION_3_ADJECTIVE_NAMES]

    pool_idx_lists = []
    for p in pool_names:
        pool_idx_lists.extend(list(rm.indices(p)))
    pool_idx_arr_host = np.asarray(pool_idx_lists, dtype=np.int64)
    pool_idx_arr = cp.asarray(pool_idx_arr_host)
    n_pool_union = pool_idx_arr.shape[0]
    if verbose:
        print("  [capture] " + str(len(pool_names)) + " pools, "
              + str(n_pool_union) + " pool-union neurons", flush=True)

    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr_host = np.asarray(lang_input_idx, dtype=np.int64)
    lang_input_arr = cp.asarray(lang_input_arr_host)

    stim_steps = 50
    reset_steps = 25

    acts = {}
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
                drive_max_pA=200.0, sparsity=sparsity)
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
            print("    [capture] '" + w + "': mean_rate "
                  + ("%.4f" % mean_rate) + " density "
                  + ("%.4f" % density), flush=True)
    return acts, n_pool_union


def _ground_symbols(consolidated, words, d_act):
    """Same pattern as (b)/(e)/n=95/OPTION 3 V=16."""
    common = np.mean([consolidated[w] for w in words], axis=0)
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    return {w: phases_to_spikes(deriver(consolidated[w] - common))
            for w in words}


def _load_activity_cache(cache_p, words):
    """Plain npz loader; numeric arrays only; no allow-arbitrary-object."""
    data = np.load(cache_p)
    return {w: data[str(w)] for w in words}


def run_one_seed(seed, smoke, xp, verbose):
    print("\n--- seed " + str(seed) + " ---", flush=True)
    bridge, words, word_to_idx, sparsity, n_lang_input = _build_and_train(
        seed, smoke, verbose)

    cache_p = _activity_cache_path(seed, smoke)
    m_obs = SMOKE_M_OBS if smoke else FULL_M_OBS
    if os.path.exists(cache_p):
        if verbose:
            print("  [seed " + str(seed) + "] loading cached activity ("
                  + cache_p + ")", flush=True)
        acts = _load_activity_cache(cache_p, words)
        n_pool_union = acts[words[0]].shape[1]
    else:
        acts, n_pool_union = _capture_concept_pool_activity(
            bridge, words, word_to_idx, sparsity, n_lang_input,
            m_obs, verbose)
        np.savez_compressed(cache_p,
                              **{str(w): acts[w] for w in words})
        if verbose:
            print("  [seed " + str(seed) + "] cached activity ("
                  + cache_p + ")", flush=True)
    d_act = n_pool_union

    # K_VOCAB averages -- use min(m_obs, ...) so smoke with M_OBS=8 doesn't
    # try to average over 16 observations.
    k_vocab = m_obs
    consolidated = {w: acts[w][:k_vocab].mean(axis=0) for w in words}
    grounded = _ground_symbols(consolidated, words, d_act)
    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, list(words), xp, rng_seed=seed)
    if verbose:
        print("  [seed " + str(seed) + "] V=" + str(len(words))
              + " d_act=" + str(d_act)
              + "; batched-vs-scalar max-diff="
              + ("%.2e" % max_diff), flush=True)

    positions = gamma_slot_positions(seed, 7, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)
    n_trials = (max(1, N_TRIALS // 4) if smoke else N_TRIALS)

    per_load = {}
    V = len(words)
    for load in LOADS:
        ob_ok = oi_ok = 0
        for _ in range(n_trials):
            items_idx = tuple(int(i) for i in
                              qrng.choice(V, size=load, replace=False))
            items = [words[i] for i in items_idx]
            C = net.encode([(grounded[items[k]], positions[k])
                            for k in range(load)])
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            recovered = []
            scores_oi_gpu = xp.zeros(V)
            for k in range(load):
                sims_k = batched_phase_similarity(
                    unbinds[k], vocab_phase_matrix, xp)
                recovered.append(int(xp.argmax(sims_k)))
                scores_oi_gpu = scores_oi_gpu + sims_k
            if tuple(recovered) == items_idx:
                ob_ok += 1
            scores_oi_host = to_host(scores_oi_gpu)
            topK = sorted(
                int(i) for i in np.argsort(scores_oi_host)[-load:])
            if tuple(topK) == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / n_trials,
            "order_invariant_accuracy": oi_ok / n_trials,
            "n_trials": n_trials,
        }
        print("    L=" + str(load)
              + ": OB="
              + ("%.3f" % per_load[load]["order_bearing_accuracy"])
              + " OI="
              + ("%.3f" % per_load[load]["order_invariant_accuracy"]),
              flush=True)
    return per_load, V, d_act, max_diff


def main():
    ap = argparse.ArgumentParser(
        description="Direction 3 V=32 multi-seed parallel-matching "
                    "mode-unification probe on bio_brain_regions")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-scale smoke (n_lang=1024, n_per_pool=100, "
                         "events=100; numbers NOT propagated)")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: side-by-side with .log)")
    args = ap.parse_args()
    smoke = bool(args.smoke)

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== Direction 3 V=32 probe: parallel-matching mode-unification "
          "on bio_brain_regions ===", flush=True)
    print("  backend=" + backend_name + " (GPU=" + str(gpu) + ")",
          flush=True)
    print("  V=" + str(DIRECTION_3_V32_TOTAL)
          + " (4 motor + 12 noun + 12 verb + 4 adjective)", flush=True)
    if smoke:
        print("  *** SMOKE MODE: reduced scale; numbers NOT propagated "
              "as a result ***", flush=True)
    print("  bar=" + str(BAR_OB)
          + "; loads=" + str(LOADS)
          + "; decoder=parallel_population_matching_batched", flush=True)

    seeds = list(SEEDS)
    seed_results = []
    t0 = time.time()
    for seed in seeds:
        t_seed = time.time()
        per_load, V, d_act, max_diff = run_one_seed(
            seed, smoke, xp, verbose=True)
        seed_results.append({
            "seed": seed, "V": V, "d_act": d_act,
            "batched_vs_scalar_max_diff": max_diff,
            "per_load": {str(l): v for l, v in per_load.items()},
            # Verdict-shaped per-seed entry (keys "L=2", "L=3", "L=5"
            # with {"OB", "OI"} -- direct input to direction_3_verdict)
            "verdict_entry": {
                ("L=" + str(l)): {
                    "OB": per_load[l]["order_bearing_accuracy"],
                    "OI": per_load[l]["order_invariant_accuracy"],
                }
                for l in LOADS
            },
        })
        elapsed_seed = (time.time() - t_seed) / 60
        print("  [seed " + str(seed) + " done in "
              + ("%.1f" % elapsed_seed) + " min]", flush=True)
    total_time = time.time() - t0
    print("\nTotal wall-clock: " + ("%.1f" % (total_time / 60))
          + " min (backend=" + backend_name + ")", flush=True)

    # Aggregate
    print("\n=== MULTI-SEED AGGREGATE ===", flush=True)
    print("            L=2 OB   OI    L=3 OB   OI    L=5 OB   OI",
          flush=True)
    agg = {}
    cells = []
    for load in LOADS:
        obs = [r["per_load"][str(load)]["order_bearing_accuracy"]
               for r in seed_results]
        ois = [r["per_load"][str(load)]["order_invariant_accuracy"]
               for r in seed_results]
        ob_m = float(np.mean(obs))
        oi_m = float(np.mean(ois))
        agg[load] = {"order_bearing_mean": ob_m,
                     "order_bearing_per_seed": obs,
                     "order_invariant_mean": oi_m,
                     "order_invariant_per_seed": ois}
        cells.append(("%.3f" % ob_m) + " " + ("%.3f" % oi_m))
    print("  multi-seed:  " + "   ".join(cells), flush=True)

    # Frozen verdict (the bar / verdict module is pre-registered)
    verdict_input = [r["verdict_entry"] for r in seed_results]
    verdict = compute_verdict(verdict_input)
    print("\n=== VERDICT (frozen, pre-registered) ===", flush=True)
    print("  " + verdict, flush=True)
    if smoke:
        print("  *** SMOKE: this verdict reflects reduced-scale geometry; "
              "the full-scale decisive run is the controller's next "
              "step ***", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu,
        "smoke": smoke,
        "seeds": seeds, "loads": LOADS,
        "bar_ob": BAR_OB, "bar_oi": BAR_OI,
        "min_seeds": _DIRECTION_3_V32_MIN_SEEDS,
        "V": DIRECTION_3_V32_TOTAL,
        "decoder_order_bearing": "parallel_population_matching_batched",
        "decoder_order_invariant": "marginal_sum_phase_similarity_batched",
        "substrate": "bio_brain_regions_v14v16_recipe_V32",
        "per_seed": seed_results,
        "aggregate": {str(l): v for l, v in agg.items()},
        "verdict": verdict,
        "wall_clock_minutes": total_time / 60,
    }
    tag = "smoke" if smoke else "full"
    out_path = args.out or os.path.join(
        _HERE, "direction_3_v32_" + tag + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote " + out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
