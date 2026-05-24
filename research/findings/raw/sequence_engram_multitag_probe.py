"""Sequence-engram multitag probe: test whether the validated MULTITAG
mechanism (project's 90% multi-seed conversational primitive) works
for SEQUENCE COMPLETION on the bio_brain_regions substrate.

CONTEXT (the pivot informed by tonight's findings):
- (c) generative-replay decisive: NEGATIVE (pillar n=99; aggregate
  104/1800=5.78% vs chance 6.25%)
- (c) loop diagnostic: REPLAY_DOESNT_REACTIVATE (the SWR->cortex
  pathway doesn't carry sequence-specific signal)
- bio_brain_regions load-ceiling map: substrate PASSes at every load
  L=2..7 (substrate has HUGE capacity headroom; the (c) NEGATIVE is
  a (c)-integration limitation, NOT a substrate limitation)
- MULTITAG mechanism (project's validated conversational primitive,
  90% multi-seed): stim engram tag -> readout via lang_output cosine
  matches the bound concepts in top-K (bypasses SWR->cortex)

This probe asks: does the validated MULTITAG mechanism produce
SEQUENCE COMPLETION when sequence engrams are tagged on the dlpfc-
extension substrate (pillar n=98)?

DESIGN:
- Encode K sequences as engrams (one engram per sequence; drive
  lang_input for each slot word sequentially; commit_engram_tag for
  the whole sequence with top_k=100 region_filter=ca3)
- For each test: stim the engram (validated D.14 reactivation),
  read lang_output cosine to each vocab word
- Sequence completion = the slot-3 word appears in top-3 cosine of
  lang_output (regardless of cue; pure stim-driven recall)
- Multi-seed (42/43/44); pre-registered 0.80 bar

If PASS: VALIDATED sequence-completion capability via the multitag-
analogous mechanism on the substrate, sidestepping the (c) loop's
SWR-readout failure mode.

If NEGATIVE: even the validated multitag mechanism doesn't transfer
to sequence engrams; the SUBSTRATE may not store slot-position-
distinct info in a way that lang_output can read.

Reuses every primitive byte-unchanged; no protected/frozen/moat
module modified; no autograd; no-confab moat must stay 7/7 green.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.vocabulary_scaling_run import (
    BAR, SEEDS,
)
from research.findings.raw.generative_replay_decisive import (
    _vocab_words, _load_substrate, _encode_engram_for_sequence,
    _engram_tag_name, set_sleep_gates, set_awake_gates,
    freeze_all_gates, SLOT_COUNT, N_REPLAYS_PER_TAG,
)
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.runners.consolidation_trainer import (
    run_concept_replay_phase,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(
    _HERE, "sequence_engram_multitag_probe.json")

# Pre-registered config (fixed; never tuned).
K_TEST = 8  # test 8 sequences (matches the (c) decisive K-ladder mid-point)
N_REPEATS_PER_TAG = 5  # stim each engram 5 times; average lang_output
STIM_STEPS = 100
STIM_DRIVE_PA = 1500.0  # matches the validated multitag stim strength
READOUT_STEPS = 50


def _capture_lang_output_during_stim(bridge, tag_name: str, words,
                                       stim_steps=STIM_STEPS,
                                       readout_steps=READOUT_STEPS,
                                       drive_pA=STIM_DRIVE_PA):
    """Stim the engram tag + capture lang_output spike counts per
    word. Returns dict {word: spike_count} averaged across the
    readout window."""
    cp, _ = get_backend()

    rm = bridge.region_manager
    lang_output_idx = list(rm.indices("language_output"))
    lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)

    # Reset.
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):  # short reset
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Stim the tagged ensemble.
    bridge.stimulate_tag(tag_name, drive_pA=drive_pA, additive=False)

    # Run the stim+readout window; accumulate lang_output firing per neuron.
    n_lang_output = lang_output_arr.shape[0]
    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    for _ in range(stim_steps + readout_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[lang_output_arr]
        lang_counts = lang_counts + fired.astype(cp.float32)

    bridge.clear_tag_drive(tag_name)
    return cp.asnumpy(lang_counts)


def _word_score_from_lang_output_pattern(lang_counts, word,
                                          n_lang_output, words):
    """Match lang_output activity to each word's drive pattern.
    Each word has an orthogonal drive_pattern; cosine similarity
    between lang_counts and the word's pattern indicates how strongly
    the activity reflects that word's identity."""
    from sim.text_embeddings import orthogonal_drive_pattern
    word_to_idx = {w: i for i, w in enumerate(words)}
    # Build the word's pattern at the lang_output resolution.
    pattern = orthogonal_drive_pattern(
        cue_idx=word_to_idx[word], n_cues=len(words),
        n_neurons=n_lang_output, drive_max_pA=1.0, sparsity=0.05)
    a = lang_counts.astype(np.float64)
    b = pattern.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_one_seed(seed: int, verbose: bool = True) -> dict:
    print(f"\n--- seed {seed} ---", flush=True)
    enable_adjective = True
    words = _vocab_words(enable_adjective=enable_adjective)
    n_lang_output = 2048  # matches the v16/dlpfc-extension default

    bridge = _load_substrate(seed, verbose=verbose)
    if verbose:
        print(f"  [seed {seed}] V={len(words)}; n_lang_output="
              f"{n_lang_output}; K_TEST={K_TEST}", flush=True)

    # Generate K stored sequences.
    sequences = generate_k_stored_sequences(
        seed=seed, k=K_TEST, n_words=len(words),
        slot_count=SLOT_COUNT, vocab=words)
    tag_names = [_engram_tag_name(i) for i in range(K_TEST)]
    if verbose:
        print(f"  [seed {seed}] sequences:", flush=True)
        for i, seq in enumerate(sequences):
            print(f"    {tag_names[i]}: {list(seq)}", flush=True)

    # Encode each sequence as one engram.
    DEFAULT_N_LANG_INPUT = 2048
    for seq_idx, seq in enumerate(sequences):
        _encode_engram_for_sequence(
            bridge, seq, seq_idx, words, DEFAULT_N_LANG_INPUT)
    if verbose:
        print(f"  [seed {seed}] {K_TEST} sequence engrams committed",
              flush=True)

    # Run Phase 1.3 consolidation.
    if verbose:
        print(f"  [seed {seed}] running Phase 1.3 consolidation "
              f"({N_REPLAYS_PER_TAG} replays/tag)...", flush=True)
    set_sleep_gates(bridge)
    run_concept_replay_phase(
        bridge, tag_names=tag_names,
        n_replays_per_tag=N_REPLAYS_PER_TAG,
        burst_duration_ms=100, inter_burst_ms=50,
        drive_pA=100.0, randomize_order=True,
        rng=np.random.default_rng(seed + 2))
    set_awake_gates(bridge)
    freeze_all_gates(bridge)

    # For each engram: stim + read lang_output cosine to each vocab
    # word; check if all 3 slot words appear in top-3.
    per_seq_results = []
    n_correct_full = 0  # all 3 slot words in top-3
    n_correct_partial = 0  # at least 2 of 3 in top-3
    n_slot3_in_topK = 0  # the slot-3 word in top-3 (sequence completion)

    for seq_idx, (seq, tag_name) in enumerate(
            zip(sequences, tag_names)):
        # Average lang_output spike counts across N_REPEATS_PER_TAG
        # repeated stim events (reduces noise).
        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = _capture_lang_output_during_stim(
                bridge, tag_name, words, stim_steps=STIM_STEPS,
                readout_steps=READOUT_STEPS, drive_pA=STIM_DRIVE_PA)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG

        # Score every vocab word against avg_counts.
        word_scores = {w: _word_score_from_lang_output_pattern(
            avg_counts, w, n_lang_output, words) for w in words}
        # Sort by score, take top-3.
        top3 = sorted(word_scores.items(),
                       key=lambda x: x[1], reverse=True)[:3]
        top3_words = [w for w, s in top3]

        # Did all 3 slot words appear in top-3?
        slot_words = list(seq)
        in_top3 = set(slot_words) & set(top3_words)
        all_correct = (len(in_top3) == 3)
        partial_correct = (len(in_top3) >= 2)
        slot3_correct = (slot_words[2] in top3_words)

        if all_correct:
            n_correct_full += 1
        if partial_correct:
            n_correct_partial += 1
        if slot3_correct:
            n_slot3_in_topK += 1

        per_seq_results.append({
            "seq_idx": seq_idx, "tag_name": tag_name,
            "sequence": list(seq),
            "top3_words": top3_words,
            "top3_scores": [round(s, 4) for w, s in top3],
            "in_top3": list(in_top3),
            "all_3_correct": all_correct,
            "partial_correct": partial_correct,
            "slot3_in_top3": slot3_correct,
        })
        print(f"    seq {seq_idx} {list(seq)}: top3={top3_words}; "
              f"matched={list(in_top3)}; all={all_correct}; "
              f"slot3={slot3_correct}", flush=True)

    full_acc = n_correct_full / K_TEST
    partial_acc = n_correct_partial / K_TEST
    slot3_acc = n_slot3_in_topK / K_TEST
    print(f"\n  [seed {seed}] all-3 correct: {n_correct_full}/{K_TEST}"
          f" = {full_acc:.3f}", flush=True)
    print(f"  [seed {seed}] partial (>=2): {n_correct_partial}/{K_TEST}"
          f" = {partial_acc:.3f}", flush=True)
    print(f"  [seed {seed}] slot3 in top3: {n_slot3_in_topK}/{K_TEST}"
          f" = {slot3_acc:.3f}", flush=True)
    return {
        "seed": seed, "K_TEST": K_TEST,
        "n_correct_full": n_correct_full,
        "n_correct_partial": n_correct_partial,
        "n_slot3_in_top3": n_slot3_in_topK,
        "full_accuracy": full_acc,
        "partial_accuracy": partial_acc,
        "slot3_accuracy": slot3_acc,
        "per_seq": per_seq_results,
    }


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== sequence-engram multitag probe ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print("  Pivot from (c) NEGATIVE: test validated MULTITAG "
          "mechanism on sequence engrams; bypasses (c)'s broken "
          "SWR->cortex path.", flush=True)
    print(f"  Pre-registered bar={BAR}; SEEDS={list(SEEDS)}; "
          f"K_TEST={K_TEST}; SLOT_COUNT={SLOT_COUNT}; "
          f"stim_drive_pA={STIM_DRIVE_PA}; "
          f"N_REPEATS_PER_TAG={N_REPEATS_PER_TAG}",
          flush=True)

    seed_results = []
    t0 = time.time()
    for seed in SEEDS:
        r = run_one_seed(seed, verbose=True)
        seed_results.append(r)
    total_min = (time.time() - t0) / 60
    print(f"\nTotal wall-clock: {total_min:.2f} min "
          f"(backend={backend_name})", flush=True)

    # Multi-seed aggregate.
    full_accs = [r["full_accuracy"] for r in seed_results]
    partial_accs = [r["partial_accuracy"] for r in seed_results]
    slot3_accs = [r["slot3_accuracy"] for r in seed_results]
    full_mean = float(np.mean(full_accs))
    partial_mean = float(np.mean(partial_accs))
    slot3_mean = float(np.mean(slot3_accs))

    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    print(f"  All-3-correct (full sequence in top-3) multi-seed "
          f"mean: {full_mean:.3f} per-seed=[{', '.join(f'{a:.3f}' for a in full_accs)}]",
          flush=True)
    print(f"  Partial (>=2 of 3 in top-3) multi-seed mean: "
          f"{partial_mean:.3f} per-seed=[{', '.join(f'{a:.3f}' for a in partial_accs)}]",
          flush=True)
    print(f"  Slot-3-completion (slot-3 word in top-3) multi-seed "
          f"mean: {slot3_mean:.3f} per-seed=[{', '.join(f'{a:.3f}' for a in slot3_accs)}]",
          flush=True)

    print(f"\n=== VERDICT (pre-registered) ===", flush=True)
    if full_mean >= BAR:
        verdict = "SEQUENCE_MULTITAG_PASS_FULL"
        print(f"  Full sequence completion PASSes ({full_mean:.3f} >="
              f" {BAR}). Pending fresh adversarial review.", flush=True)
    elif slot3_mean >= BAR:
        verdict = "SEQUENCE_MULTITAG_PASS_SLOT3_ONLY"
        print(f"  Slot-3 completion PASSes ({slot3_mean:.3f} >="
              f" {BAR}) but full-sequence does not. Partial validation.",
              flush=True)
    elif partial_mean >= BAR:
        verdict = "SEQUENCE_MULTITAG_PARTIAL"
        print(f"  Partial (2/3) PASSes ({partial_mean:.3f} >= {BAR}) "
              f"but neither full nor slot-3 alone. Boundary result.",
              flush=True)
    else:
        verdict = "SEQUENCE_MULTITAG_NEGATIVE"
        print(f"  Even the validated multitag mechanism doesn't "
              f"transfer to sequence engrams (full {full_mean:.3f}, "
              f"slot3 {slot3_mean:.3f}, partial {partial_mean:.3f} "
              f"all below {BAR}). The SUBSTRATE may not store slot-"
              f"position-distinct info readable from lang_output.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu,
        "seeds": list(SEEDS), "K_TEST": K_TEST,
        "slot_count": SLOT_COUNT, "bar": BAR,
        "stim_drive_pA": STIM_DRIVE_PA,
        "n_repeats_per_tag": N_REPEATS_PER_TAG,
        "substrate": ("build_biological_brain_regions_v16_recipe_"
                       "WITH_HIPPO_AND_DLPFC_n=98"),
        "method": ("sequence engrams via D.14; stim_tag (validated "
                    "reactivation) -> lang_output cosine to vocab; "
                    "multitag-analogous readout"),
        "per_seed": seed_results,
        "aggregate": {
            "full_sequence_accuracy_mean": full_mean,
            "full_sequence_per_seed": full_accs,
            "partial_accuracy_mean": partial_mean,
            "partial_per_seed": partial_accs,
            "slot3_accuracy_mean": slot3_mean,
            "slot3_per_seed": slot3_accs,
        },
        "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
