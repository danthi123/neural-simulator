"""Direction A CAPACITY SWEEP: scale slot_count from 3 -> {4, 5, 6, 7}
to characterize where the ec_context positional binding mechanism
ceilings. Reuses the trained bridges from the main Direction A run
(no extra training); ~30-60 min wall total for the sweep.

REVIEWER NOTE (2026-05-24, STRENGTHEN clarification per BLOCK
verdict #3): The bridge is trained at n_max_positions=3 (the only
positions encountered during training are slots {0, 1, 2}). The
sweep tests {3, 4, 5, 6, 7} by passing n_max_positions=slot_count
to positional_drive_pattern -- which uses a STRIDE-3 layout in v1,
STRIDE-N layout at slot_count=N. So at slot_count=7 the drives
occupy positional bands the substrate's trained ec_context->pool
weights NEVER saw. This sweep therefore measures the ENGRAM-
CAPTURE+STIM mechanism at varying stride (the engram is re-captured
per-slot_count), NOT extrapolation of trained positional weights.
Any PASS here is about engram-tagging capacity at varying stride,
not about the substrate having learned positions 3-6.
"""

This runs AFTER the main Direction A completes. If main PASSes at
slot_count=3, this characterizes the load-ceiling: at what slot count
does the mechanism break? Mirrors the FHRR capacity-curve pattern
(commit 731ed09 LOAD-SCALING CHARACTERISED).

For each slot_count in {3, 4, 5, 6, 7}:
  - reuses the same trained bridge cache per seed
  - re-encodes K=8 sequences with new slot_count
  - re-tests slot-(slot_count-1) retrieval with top-3 readout
  - multi-seed 42/43/44
  - reports per-slot_count accuracy

Pre-registered: same 0.80 multi-seed bar (frozen, not tuned).

If accuracy holds at slot_count=7 (the gamma-cycle cap per
Lisman-Idiart N.16), ec_context spatial binding scales to the
theoretical limit. If it ceilings earlier, that's the precise
biology-translatable bound.

NUMPY-friendly imports; GPU-only when bridge actually runs.
~30-60 min wall total.
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

from research.runners.concept_pool_demo import (
    build_concept_bridge, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _bridge_save_path, _build_region_filter,
    _encode_sequence_with_ec_context,
    _capture_lang_output_with_pos_cue, _word_score,
    N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL, N_EC_CONTEXT,
    K_PAIRS, TOP_K_READOUT, EC_DRIVE_PA, STIM_DRIVE_PA,
    STIM_STEPS, READOUT_STEPS, N_REPEATS_PER_TAG, ENGRAM_TOP_K,
    TEACHER_PA, ENCODING_STEPS_PER_SLOT, SPARSITY,
)
from sim.text_embeddings import (
    orthogonal_drive_pattern, positional_drive_pattern,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_A_capacity_sweep.json")
SLOT_COUNTS = [3, 4, 5, 6, 7]  # 7 = catalog N.16 gamma cap
SEEDS = [42, 43, 44]
BAR = 0.80


def generate_sequences_for_load(seed, k, slot_count, vocab):
    """Generate K sequences of slot_count words from vocab.
    Mirrors generate_k_stored_sequences semantics but parameterized
    slot_count (the original was hardcoded). RNG seeded by
    (seed * 1000 + slot_count) for reproducibility per (seed,load)
    combo."""
    rng = np.random.default_rng(seed * 10000 + slot_count)
    sequences = []
    n_vocab = len(vocab)
    for _ in range(k):
        idx = rng.choice(n_vocab, size=slot_count, replace=False)
        sequences.append([vocab[i] for i in idx])
    return sequences


def encode_seq_at_slot_count(bridge, seq, words, seq_idx, slot_count,
                              region_filter):
    """Like _encode_sequence_with_ec_context but parameterized
    slot_count (passes through to positional_drive_pattern).
    Reuses the same encoding mechanism byte-equivalent except
    n_max_positions = slot_count (so position phasor is unique
    per slot at THIS slot_count)."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    from research.runners.concept_compose_train import _WORD_TO_POOL
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    n_lang_input = len(lang_in_idx)
    ec_idx = list(rm.indices("ec_context"))
    ec_arr = cp.asarray(ec_idx, dtype=cp.int64)
    n_ec = len(ec_idx)
    word_to_idx = {w: i for i, w in enumerate(words)}
    n_total = bridge.cp_external_input_current.shape[0]
    tag_name = f"sc{slot_count}_seq_{seq_idx:03d}"

    pool_arrs = []
    for slot_word in seq:
        pool_region = _WORD_TO_POOL.get(slot_word, None)
        if pool_region is None:
            pool_arrs.append(None); continue
        try:
            pool_idx = list(rm.indices(pool_region))
            pool_arrs.append(cp.asarray(pool_idx, dtype=cp.int64))
        except Exception:
            pool_arrs.append(None)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.start_engram_recording(tag_name)
    ext = cp.zeros(n_total, dtype=cp.float32)
    for slot_idx, word in enumerate(seq):
        drive_word = orthogonal_drive_pattern(
            cue_idx=word_to_idx[word], n_cues=len(words),
            n_neurons=n_lang_input, drive_max_pA=200.0,
            sparsity=SPARSITY)
        drive_pos = positional_drive_pattern(
            slot_idx, n_neurons=n_ec, n_max_positions=slot_count)
        pos_pattern = (drive_pos > 0).astype(np.float32) * EC_DRIVE_PA
        for _ in range(ENCODING_STEPS_PER_SLOT):
            ext.fill(0)
            ext[lang_in_arr] = cp.asarray(drive_word, dtype=cp.float32)
            ext[ec_arr] = cp.asarray(pos_pattern, dtype=cp.float32)
            if pool_arrs[slot_idx] is not None:
                ext[pool_arrs[slot_idx]] = TEACHER_PA
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    stats = bridge.commit_engram_tag(
        tag_name, top_k=ENGRAM_TOP_K, region_filter=region_filter)
    bridge.cp_external_input_current[:] = 0.0
    return tag_name, stats


def capture_with_pos_cue_at_slot_count(bridge, tag_name, cue_slot_idx,
                                          slot_count, n_ec, ec_arr,
                                          n_lang_output):
    """Like _capture_lang_output_with_pos_cue but parameterized
    slot_count (mirror)."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    cue_pos = positional_drive_pattern(
        cue_slot_idx, n_neurons=n_ec, n_max_positions=slot_count)
    cue_pos_pa = (cue_pos > 0).astype(np.float32) * EC_DRIVE_PA
    bridge.cp_external_input_current[ec_arr] = cp.asarray(
        cue_pos_pa, dtype=cp.float32)
    bridge.stimulate_tag(tag_name, drive_pA=STIM_DRIVE_PA, additive=False)

    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    for _ in range(STIM_STEPS + READOUT_STEPS):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[lang_out_arr]
        lang_counts = lang_counts + fired.astype(cp.float32)
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(lang_counts)


def run_seed_load(seed, slot_count, verbose=True):
    bridge_p = _bridge_save_path(seed)
    if not os.path.exists(bridge_p):
        print(f"  [seed {seed}] bridge cache missing", flush=True)
        return None

    cp, _ = get_backend()
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)
    bridge = build_concept_bridge(
        seed=seed, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=True,
        n_ec_context=N_EC_CONTEXT, verbose=False,
    )
    bridge.load_checkpoint(bridge_p)
    for g in ("language_input_to_motor",
              "language_input_to_noun_pool",
              "language_input_to_verb_pool",
              "language_input_to_adjective_pool",
              "motor_to_language_output",
              "noun_pool_to_language_output",
              "verb_pool_to_language_output",
              "adjective_pool_to_language_output",
              "ec_context_to_noun_pool",
              "ec_context_to_verb_pool",
              "ec_context_to_adjective_pool",
              "ec_context_to_motor"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    rm = bridge.region_manager
    ec_idx = list(rm.indices("ec_context"))
    ec_arr = cp.asarray(ec_idx, dtype=cp.int64)
    n_ec = len(ec_idx)
    n_lang_output = N_LANG_INPUT
    region_filter = _build_region_filter(rm)

    sequences = generate_sequences_for_load(
        seed, K_PAIRS, slot_count, words)

    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = encode_seq_at_slot_count(
            bridge, seq, words, seq_idx, slot_count, region_filter)
        tag_names.append(tag)

    # Test slot-(slot_count-1) retrieval; same as main runner pattern.
    n_correct = 0
    cue_slot_idx = slot_count - 1
    per_seq = []
    for seq_idx, (seq, tag_name) in enumerate(zip(sequences, tag_names)):
        true_slot = seq[cue_slot_idx]
        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_with_pos_cue_at_slot_count(
                bridge, tag_name, cue_slot_idx, slot_count,
                n_ec, ec_arr, n_lang_output)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_counts, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)[:TOP_K_READOUT]
        topK_words = [w for w, s in topK]
        correct = true_slot in topK_words
        if correct: n_correct += 1
        per_seq.append({
            "seq_idx": seq_idx, "true_slot": true_slot,
            "topK_words": topK_words, "correct": correct,
        })

    acc = n_correct / K_PAIRS
    if verbose:
        print(f"  [seed {seed}, slot_count {slot_count}] "
              f"slot-{cue_slot_idx} acc = {n_correct}/{K_PAIRS}"
              f" = {acc:.3f}", flush=True)
    return {
        "seed": seed, "slot_count": slot_count,
        "n_correct": n_correct, "slot_last_accuracy": acc,
        "per_seq": per_seq,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-counts", nargs="+", type=int,
                     default=SLOT_COUNTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    print(f"=== Direction A CAPACITY SWEEP ===", flush=True)
    print(f"  slot_counts: {args.slot_counts}", flush=True)
    print(f"  seeds: {args.seeds}", flush=True)
    print(f"  K_PAIRS: {K_PAIRS}, TOP_K_READOUT: {TOP_K_READOUT}",
          flush=True)
    print(f"  Pre-registered bar: {BAR} (frozen, not tuned)",
          flush=True)

    chance = TOP_K_READOUT / 16.0
    print(f"  Chance baseline: {chance:.3f}", flush=True)

    t0 = time.time()
    per_slot_results = {}
    for slot_count in args.slot_counts:
        print(f"\n--- slot_count = {slot_count} ---", flush=True)
        per_seed = []
        for seed in args.seeds:
            r = run_seed_load(seed, slot_count, verbose=True)
            if r is not None:
                per_seed.append(r)
        if not per_seed:
            print(f"  [skipped: no cached bridges]", flush=True)
            continue
        accs = [r["slot_last_accuracy"] for r in per_seed]
        mean = float(np.mean(accs))
        per_slot_results[slot_count] = {
            "per_seed": per_seed, "mean": mean,
        }
        print(f"  slot_count {slot_count} mean = {mean:.3f} "
              f"per-seed=[{', '.join(f'{a:.3f}' for a in accs)}]",
              flush=True)

    total_min = (time.time() - t0) / 60
    print(f"\n=== SUMMARY ===", flush=True)
    for sc in args.slot_counts:
        if sc in per_slot_results:
            mean = per_slot_results[sc]["mean"]
            mark = "PASS" if mean >= BAR else (
                "BOUNDARY" if mean > 2 * chance else "AT-CHANCE")
            print(f"  slot_count {sc}: mean {mean:.3f} ({mark})",
                  flush=True)

    print(f"\nWall: {total_min:.1f} min", flush=True)

    # Find load ceiling
    passing_loads = [sc for sc in per_slot_results
                     if per_slot_results[sc]["mean"] >= BAR]
    if passing_loads:
        max_pass = max(passing_loads)
        verdict = f"CAPACITY_CEILING_AT_SLOT_COUNT_{max_pass}"
        print(f"\n  PASSes at slot_count <= {max_pass}; collapses "
              f"above this point. Biology-translatable capacity "
              f"limit on ec_context spatial positional binding.",
              flush=True)
    else:
        verdict = "BELOW_BAR_AT_ALL_SLOT_COUNTS"
        print(f"\n  Below bar at every tested slot_count; "
              f"ec_context mechanism may have been borderline at "
              f"slot_count=3.", flush=True)

    out = {
        "slot_counts": args.slot_counts, "seeds": args.seeds,
        "K_PAIRS": K_PAIRS, "TOP_K_READOUT": TOP_K_READOUT,
        "chance_baseline": chance, "bar": BAR,
        "per_slot_results": per_slot_results,
        "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
