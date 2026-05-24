"""Decisive runner for the (c) generative-replay loop.

Pre-registered (per `docs/plans/2026-05-24-generative-replay-
implementation.md` Task 2 + Task 5):

  - Substrate: the validated dlpfc-extension at pillar n=98
    (build_biological_brain_regions + hippocampus + Phase 1.3 SWR
    consolidation + dlpfc_wm NMDA-bistable PFC + lang_input ->
    dlpfc_wm pathway). Bridges cached at
    `research/findings/raw/mode_unification_with_hippo_dlpfc_cache/
    bridge_full_seed{42,43,44}.simstate.h5` from the dlpfc-extension
    probe (~hours per seed of training avoided).
  - Sequences: K stored sequences generated deterministically per
    seed via Task 1's `generate_k_stored_sequences`. Each sequence
    is `slot_count`-many distinct words from the v16 16-word vocab.
  - Encoding: each sequence engram-tagged via the validated
    Tonegawa-style mechanism (catalog D.14): drive lang_input for
    each slot word during a recording window, commit the tag with
    top-K selection. The engram tag captures the co-fired
    (sequence-spanning) ensemble.
  - Consolidation: Phase 1.3 sleep mechanism (the validated 3/3
    strict anti-cheat multi-seed mechanism) -- alternate awake and
    sleep gate sets, drive replay for committed sequence tags via
    `run_concept_replay_phase`. The CA1 -> cortex pathways
    consolidate the sequence into cortex.
  - Per trial: initialise the PFC frame with a partial cue (first
    `slot_count - 1` words of a randomly-chosen stored sequence);
    run the generative-replay loop for 1 iteration (the loop's
    decoded continuation = the proposal for the LAST slot); score
    completion accuracy (post-hoc only) by comparing decoded word
    to the true continuation.
  - Multi-seed: 42, 43, 44 (matches all prior validated pillars).
  - K-ladder: {4, 8, 16} sequences per seed (pre-registered).
  - Frozen 0.80 bar (IMMOVABLE).
  - Kill-safe: per-seed cache of trial results.
  - Smoke mode (--smoke): K=4, n_trials=20; verifies the loop
    assembles mechanically; numbers NOT propagated.

NO ORACLE LEAK: the loop runtime never receives the true continuation
items. The decoder argument list is exactly
(activity_vector, grounded_vocab_phase_matrix, position, net, xp).
The true continuation is read ONLY by the post-hoc scoring path,
AFTER the loop has produced its decoded word.

Reuse-by-import only; protected/frozen/moat modules unchanged; no
autograd; no-confab moat 7/7 stays green throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse-by-import: the loop controller (Task 2 sibling) +
# task-1 sequence helper + every validated primitive.
from research.runners.generative_replay_loop import (
    encode_pfc_frame, run_generative_loop,
)
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences, V16_DEFAULT_VOCAB,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from research.findings.raw.cross_bridge_mode_unification_probe import (
    batched_phase_similarity, verify_batched_equivalent_to_scalar,
    build_vocab_phase_matrix,
)
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    K_VOCAB_TARGET, DERIV_SEED,
)
from research.findings.raw.mode_unification_on_bio_brain_regions_probe import (
    _ground_symbols, _load_activity_cache,
    DEFAULT_N_LANG_INPUT, DEFAULT_N_PER_POOL, DEFAULT_N_FS_PER_POOL,
    DEFAULT_SPARSITY, M_OBS_FULL,
)
from research.findings.raw.mode_unification_with_hippo_dlpfc_probe import (
    _build_bridge_with_hippo_and_dlpfc, CACHE_DIR as DLPFC_CACHE_DIR,
)
from research.findings.raw.vocabulary_scaling_run import (
    BAR, N_DIM,
)
from research.runners.concept_pool_demo import (
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB, ADJECTIVE_VOCAB,
)
from research.runners.text_minimal_isolation import (
    set_sleep_gates, set_awake_gates, freeze_all_gates,
)
from research.runners.consolidation_trainer import (
    run_concept_replay_phase,
)
from sim.backend import get_backend, to_host, is_gpu_backend


# =====================================================================
# Pre-registered constants (FROZEN; never tuned).
# =====================================================================
SEEDS = [42, 43, 44]
K_LADDER = [4, 8, 16]              # Pre-registered K-values
SLOT_COUNT = 3                       # Cue = first (slot_count-1) slots
N_TRIALS_FULL = 200
N_GAMMA_SLOTS = 7
SWR_STEPS = 100
CAPTURE_STEPS = 50
ENCODING_STEPS_PER_SLOT = 30        # 30 sim-steps per cue-word during
                                     # engram encoding
N_REPLAYS_PER_TAG = 8                # Phase 1.3 consolidation: per-tag
                                     # SWR replay events
ENGRAM_TOP_K = 100                   # Tonegawa-style top-K selection
ENGRAM_REGION_FILTER = ["ca3"]       # Tag hippocampal CA3 (Marr-class
                                     # autoassociator)
TOPOGRAPHIC_SPARSITY = 0.05

# Smoke mode (cheap mechanical-assembly verification; numbers NOT
# propagated).
SMOKE_K_LADDER = [4]
SMOKE_N_TRIALS = 20
SMOKE_SEED = 42

# Output paths.
OUT_DIR = _HERE
CACHE_TRIAL_DIR = os.path.join(
    _HERE, "generative_replay_decisive_cache")
os.makedirs(CACHE_TRIAL_DIR, exist_ok=True)


def _vocab_words(enable_adjective: bool = True) -> List[str]:
    """Build the v16 16-word vocabulary in the SAME declaration order
    the dlpfc-extension probe trained on (so the cached substrate's
    grounded vocabulary aligns with this runner's vocab)."""
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB))
    if enable_adjective:
        words += list(ADJECTIVE_VOCAB)
    return words


def _trial_cache_path(seed: int, smoke: bool) -> str:
    tag = "smoke" if smoke else "full"
    return os.path.join(
        CACHE_TRIAL_DIR, f"trials_{tag}_seed{seed}.json")


def _build_pool_idx_arr(bridge, enable_adjective: bool):
    """Build the backend int64 array of pool-union neuron indices --
    same pattern as `_capture_concept_pool_activity` byte-unchanged."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    rm = bridge.region_manager
    pool_names = [f"motor_{d}" for d in ("N", "E", "S", "W")]
    pool_names += [f"noun_pool_{w.upper()}" for w in NOUN_VOCAB]
    pool_names += [f"verb_pool_{w.upper()}" for w in VERB_VOCAB]
    if enable_adjective:
        pool_names += [f"adjective_pool_{w.upper()}" for w in
                        ADJECTIVE_VOCAB]
    pool_idx_lists = []
    for p in pool_names:
        pool_idx_lists.extend(list(rm.indices(p)))
    pool_idx_arr_host = np.asarray(pool_idx_lists, dtype=np.int64)
    return xp.asarray(pool_idx_arr_host), len(pool_idx_lists)


def _engram_tag_name(seq_idx: int) -> str:
    return f"generative_seq_{seq_idx:03d}"


def _drive_lang_input(bridge, word: str, words: List[str],
                      n_lang_input: int):
    """Apply orthogonal lang_input drive pattern for a single word
    (matches the OPTION 3 / dlpfc-extension probe's capture pattern
    byte-equivalent: orthogonal_drive_pattern -> set
    cp_external_input_current at lang_input indices)."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern
    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr = xp.asarray(lang_input_idx, dtype=xp.int64)
    drive = orthogonal_drive_pattern(
        cue_idx=words.index(word), n_cues=len(words),
        n_neurons=n_lang_input,
        drive_max_pA=200.0, sparsity=TOPOGRAPHIC_SPARSITY)
    drive_gpu = xp.asarray(drive, dtype=xp.float32)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[lang_input_arr] = drive_gpu


def _encode_engram_for_sequence(bridge, seq: Tuple[str, ...],
                                 seq_idx: int, words: List[str],
                                 n_lang_input: int) -> dict:
    """Engram-tag one sequence. Drive each slot word in turn during
    the recording window so the co-fired ensemble across all slots
    becomes the tag. top_k selection (Tonegawa-style sparse engram);
    region_filter=['ca3'] keeps the tag hippocampal (Marr
    autoassociator).
    """
    tag_name = _engram_tag_name(seq_idx)
    bridge.start_engram_recording(tag_name)
    for slot_word in seq:
        _drive_lang_input(bridge, slot_word, words, n_lang_input)
        for _ in range(ENCODING_STEPS_PER_SLOT):
            bridge._run_one_simulation_step()
            rs = getattr(bridge, "runtime_state", None)
            if rs is not None and hasattr(rs, "current_time_step"):
                try:
                    rs.current_time_step += 1
                except Exception:
                    pass
    bridge.cp_external_input_current[:] = 0.0
    stats = bridge.commit_engram_tag(
        tag_name,
        top_k=ENGRAM_TOP_K,
        region_filter=ENGRAM_REGION_FILTER,
    )
    return stats


def _grounded_from_activity_cache(seed: int, words: List[str]
                                    ) -> Tuple[dict, int]:
    """Load the cached cortex activity from the dlpfc-extension
    probe; build the substrate-grounded vocabulary via the SAME
    _ground_symbols pipeline (mean-centred + fixed-seed deriver).
    Returns (grounded, d_act)."""
    cache_p = os.path.join(
        DLPFC_CACHE_DIR, f"activity_full_seed{seed}.npz")
    if not os.path.exists(cache_p):
        raise FileNotFoundError(
            f"dlpfc-extension activity cache missing: {cache_p}")
    acts = _load_activity_cache(cache_p, words)
    d_act = acts[words[0]].shape[1]
    # K_VOCAB_TARGET=16 mean obs per concept (the validated recipe).
    consolidated = {w: acts[w][:K_VOCAB_TARGET].mean(axis=0)
                    for w in words}
    grounded = _ground_symbols(consolidated, words, d_act)
    return grounded, d_act


def _load_substrate(seed: int, verbose: bool):
    """Build the dlpfc-extension substrate (matches pillar n=98) and
    load the cached trained weights. Freeze all plasticity gates
    AFTER loading so the loop's SWR window doesn't drift weights
    (the (c) loop is INFERENCE; weight drift would be a soundness
    issue)."""
    if verbose:
        print(f"  [seed {seed}] building dlpfc-extension substrate "
              f"+ loading cached trained weights...", flush=True)
    enable_adjective = True
    bridge = _build_bridge_with_hippo_and_dlpfc(
        seed=seed, enable_adjective=enable_adjective, verbose=verbose)
    bridge_p = os.path.join(
        DLPFC_CACHE_DIR, f"bridge_full_seed{seed}.simstate.h5")
    if not os.path.exists(bridge_p):
        raise FileNotFoundError(
            f"dlpfc-extension bridge cache missing: {bridge_p}\n"
            f"  Required for (c) generative-replay decisive run.")
    bridge.load_checkpoint(bridge_p)
    # Freeze all known plasticity gates so neither engram
    # encoding nor SWR-replay drives weight changes; the (c) loop
    # is INFERENCE over the consolidated substrate.
    freeze_all_gates(bridge)
    # ca3_swr_burst gate must be freezable on this substrate (the
    # validated Phase 1.3 mechanism); trigger_swr_replay opens and
    # closes it during each iteration's replay window.
    try:
        bridge.set_plasticity_gate("ca3_swr_burst", 0.0)
    except KeyError as e:
        raise RuntimeError(
            f"Substrate missing ca3_swr_burst gate -- the validated "
            f"Phase 1.3 mechanism is required for the (c) loop. "
            f"Diagnosis: {e}")
    return bridge


def _build_grounded_vocab_phase_matrix(grounded: dict,
                                        words: List[str], xp):
    """Build the (V, N_DIM) phase matrix on the active backend.
    Verifies batched == scalar within 1e-10 (fail-closed; matches
    the validated probes)."""
    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, words, xp, rng_seed=0)
    if max_diff > 1e-10:
        raise RuntimeError(
            f"Batched phase_similarity diverges from scalar by "
            f"{max_diff:.3e} > 1e-10 -- refusing to run.")
    return vocab_phase_matrix, max_diff


def _score_completion(decoded_word: str,
                       true_continuation: str) -> bool:
    """Post-hoc scoring ONLY. The loop runtime never reads
    true_continuation; this function is called by the runner AFTER
    the loop has produced its decoded_word."""
    return decoded_word == true_continuation


def _run_one_seed(seed: int, k_ladder: List[int], n_trials: int,
                  enable_adjective: bool, smoke: bool, verbose: bool
                  ) -> dict:
    """Run all K-ladder evaluations for one seed. Kill-safe via
    per-seed cache."""
    cache_p = _trial_cache_path(seed, smoke)
    if os.path.exists(cache_p):
        if verbose:
            print(f"  [seed {seed}] loading cached trial results "
                  f"({cache_p})", flush=True)
        with open(cache_p, "r", encoding="utf-8") as f:
            return json.load(f)

    t_seed = time.time()
    xp, _ = get_backend()
    words = _vocab_words(enable_adjective=enable_adjective)
    bridge = _load_substrate(seed, verbose=verbose)
    grounded, d_act = _grounded_from_activity_cache(seed, words)
    vocab_phase_matrix, batched_vs_scalar = \
        _build_grounded_vocab_phase_matrix(grounded, words, xp)
    pool_idx_arr, n_pool_union = _build_pool_idx_arr(
        bridge, enable_adjective)
    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    if SLOT_COUNT > N_GAMMA_SLOTS:
        raise ValueError(
            f"SLOT_COUNT={SLOT_COUNT} exceeds N_GAMMA_SLOTS="
            f"{N_GAMMA_SLOTS}")

    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    per_k_results = {}
    for k in k_ladder:
        if verbose:
            print(f"  [seed {seed}, K={k}] generating + engramming + "
                  f"consolidating + trials...", flush=True)
        t_k = time.time()
        # Step A: deterministic K stored sequences.
        stored_sequences = generate_k_stored_sequences(
            seed=seed, k=k, n_words=len(words),
            slot_count=SLOT_COUNT, vocab=words)
        # Step B: engram-tag every stored sequence (catalog D.14).
        engram_stats = []
        for seq_idx, seq in enumerate(stored_sequences):
            stats = _encode_engram_for_sequence(
                bridge, seq, seq_idx, words, DEFAULT_N_LANG_INPUT)
            engram_stats.append(stats)
        # Step C: Phase 1.3 consolidation (sleep replay against the
        # K tagged sequences). Sleep gates ON; replay each tag.
        set_sleep_gates(bridge)
        replay_stats = run_concept_replay_phase(
            bridge,
            tag_names=[_engram_tag_name(i)
                       for i in range(len(stored_sequences))],
            n_replays_per_tag=N_REPLAYS_PER_TAG,
            burst_duration_ms=100,
            inter_burst_ms=50,
            drive_pA=100.0,
            randomize_order=True,
            rng=np.random.default_rng(seed + 2),
        )
        # Restore awake gates + RE-FREEZE plasticity for the loop
        # (inference over consolidated substrate).
        set_awake_gates(bridge)
        freeze_all_gates(bridge)
        try:
            bridge.set_plasticity_gate("ca3_swr_burst", 0.0)
        except KeyError:
            pass

        # Step D: per trial -- partial cue + loop + post-hoc score.
        n_correct = 0
        trial_log = []
        for trial in range(n_trials):
            # Pick a random stored sequence as the trial target.
            chosen = int(qrng.integers(low=0, high=len(stored_sequences)))
            full_seq = stored_sequences[chosen]
            cue = full_seq[:SLOT_COUNT - 1]
            true_continuation = full_seq[SLOT_COUNT - 1]
            # Initial PFC frame from the cue (first slot_count-1
            # words at the first slot_count-1 gamma positions).
            cue_positions = positions[:SLOT_COUNT - 1]
            initial_C = encode_pfc_frame(
                cue, cue_positions, net, grounded)
            # Run the generative-replay loop for 1 iteration --
            # decoded continuation = proposal for the FINAL slot.
            trajectory = run_generative_loop(
                initial_C=initial_C,
                n_iterations=1,
                bridge=bridge,
                grounded=grounded,
                vocab_words=words,
                positions=positions,
                net=net,
                xp=xp,
                pool_idx_arr=pool_idx_arr,
                grounded_vocab_phase_matrix=vocab_phase_matrix,
                start_position_idx=SLOT_COUNT - 1,
                d_act=d_act,
                swr_steps=SWR_STEPS,
                capture_steps=CAPTURE_STEPS,
                verbose=False,
            )
            decoded_word = trajectory[0]["decoded_word"]
            # POST-HOC scoring only; the loop runtime never saw
            # true_continuation.
            ok = _score_completion(decoded_word, true_continuation)
            if ok:
                n_correct += 1
            trial_log.append({
                "trial": trial,
                "chosen_seq_idx": chosen,
                "cue": list(cue),
                "true_continuation": true_continuation,
                "decoded_word": decoded_word,
                "correct": bool(ok),
            })
        completion_acc = n_correct / max(1, n_trials)
        per_k_results[str(k)] = {
            "K": k,
            "n_trials": n_trials,
            "n_correct": n_correct,
            "completion_accuracy": completion_acc,
            "n_engrams": len(engram_stats),
            "replay_stats": {
                "n_replays": replay_stats.get("n_replays", 0),
            },
            "engram_stats_first3": engram_stats[:3],
            "trial_log_first5": trial_log[:5],
            "wall_clock_s": time.time() - t_k,
        }
        if verbose:
            print(f"    [seed {seed} K={k}] completion_accuracy="
                  f"{completion_acc:.3f} ({n_correct}/{n_trials}); "
                  f"K-block wall {time.time()-t_k:.1f}s", flush=True)
    seed_result = {
        "seed": seed,
        "n_pool_union": int(n_pool_union),
        "d_act": int(d_act),
        "batched_vs_scalar_max_diff": float(batched_vs_scalar),
        "wall_clock_s": time.time() - t_seed,
        "per_k": per_k_results,
    }
    with open(cache_p, "w", encoding="utf-8") as f:
        json.dump(seed_result, f, indent=2)
    if verbose:
        print(f"  [seed {seed} done in {(time.time()-t_seed)/60:.1f} "
              f"min, cached]", flush=True)
    return seed_result


def main():
    ap = argparse.ArgumentParser(
        description="Decisive runner for the (c) generative-replay "
                    "loop on the validated dlpfc-extension substrate")
    ap.add_argument("--smoke", action="store_true",
                    help="K=4 + 20 trials on seed 42 only; verifies "
                         "the loop assembles mechanically; numbers "
                         "NOT propagated as a result")
    args = ap.parse_args()
    smoke = bool(args.smoke)

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== (c) generative-replay decisive runner ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  substrate=dlpfc-extension n=98; "
          f"cache={DLPFC_CACHE_DIR}", flush=True)
    if smoke:
        print("  *** SMOKE MODE: K=4 + 20 trials @ seed 42 only; "
              "numbers NOT propagated ***", flush=True)
    print(f"  bar={BAR} (frozen; never tuned); SEEDS="
          f"{SEEDS if not smoke else [SMOKE_SEED]}; "
          f"K_LADDER={K_LADDER if not smoke else SMOKE_K_LADDER}; "
          f"slot_count={SLOT_COUNT}; "
          f"n_trials/K={N_TRIALS_FULL if not smoke else SMOKE_N_TRIALS}",
          flush=True)
    print(f"  loop: swr_steps={SWR_STEPS}, capture_steps={CAPTURE_STEPS}, "
          f"encoding_steps_per_slot={ENCODING_STEPS_PER_SLOT}, "
          f"n_replays_per_tag={N_REPLAYS_PER_TAG}", flush=True)

    seeds_to_run = [SMOKE_SEED] if smoke else list(SEEDS)
    k_ladder = SMOKE_K_LADDER if smoke else K_LADDER
    n_trials = SMOKE_N_TRIALS if smoke else N_TRIALS_FULL

    all_seed_results = []
    t0 = time.time()
    for seed in seeds_to_run:
        seed_result = _run_one_seed(
            seed=seed, k_ladder=k_ladder, n_trials=n_trials,
            enable_adjective=True, smoke=smoke, verbose=True)
        all_seed_results.append(seed_result)
    total_minutes = (time.time() - t0) / 60.0
    print(f"\nTotal wall-clock: {total_minutes:.2f} min "
          f"(backend={backend_name})", flush=True)

    # Aggregate per K across seeds.
    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    aggregate = {}
    all_pass = True
    for k in k_ladder:
        per_seed_accs = [r["per_k"][str(k)]["completion_accuracy"]
                          for r in all_seed_results]
        mean_acc = float(np.mean(per_seed_accs))
        aggregate[str(k)] = {
            "completion_accuracy_mean": mean_acc,
            "completion_accuracy_per_seed": per_seed_accs,
        }
        passes = mean_acc >= BAR
        if not passes:
            all_pass = False
        print(f"  K={k}: completion_accuracy mean={mean_acc:.3f} "
              f"per-seed={['%.3f' % a for a in per_seed_accs]} "
              f"({'>=' if passes else '<'} {BAR})", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if smoke:
        verdict = "SMOKE"
        print("  SMOKE -- numbers NOT propagated. Mechanical-assembly "
              "indicator only: the loop iterates without error "
              "across the K-ladder; the decisive run requires --no-"
              "smoke + the full pre-registered config.", flush=True)
    elif all_pass:
        verdict = "GENERATIVE_REPLAY_PASS"
        print(f"  Multi-seed mean completion_accuracy >= {BAR} at every "
              f"K in {k_ladder}. The biology-grounded generative-"
              f"replay loop produces partial-sequence completion at "
              f"the validated substrate's multi-seed margin. NOT yet "
              f"a capability claim -- pending dedicated adversarial "
              f"review (Task 4 in the plan).", flush=True)
    else:
        verdict = "GENERATIVE_REPLAY_NEGATIVE"
        print(f"  Multi-seed mean completion_accuracy misses {BAR} at "
              f"at least one K. Honest finding: which integration "
              f"property of the (c) loop doesn't scale at this K-"
              f"ladder. Per-K breakdown above; per-seed trial logs in "
              f"per-seed cache files.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "smoke": smoke,
        "seeds": seeds_to_run, "k_ladder": k_ladder,
        "slot_count": SLOT_COUNT, "n_trials_per_k": n_trials,
        "bar": BAR, "swr_steps": SWR_STEPS,
        "capture_steps": CAPTURE_STEPS,
        "encoding_steps_per_slot": ENCODING_STEPS_PER_SLOT,
        "n_replays_per_tag": N_REPLAYS_PER_TAG,
        "engram_top_k": ENGRAM_TOP_K,
        "engram_region_filter": ENGRAM_REGION_FILTER,
        "decoder": "parallel_population_matching_batched",
        "substrate": "build_biological_brain_regions_v16_recipe_"
                      "WITH_HIPPO_AND_DLPFC_n=98",
        "per_seed": all_seed_results,
        "aggregate": aggregate,
        "verdict": verdict,
        "wall_clock_minutes": total_minutes,
    }
    tag = "smoke" if smoke else "full"
    out_path = os.path.join(
        OUT_DIR, f"generative_replay_decisive_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
