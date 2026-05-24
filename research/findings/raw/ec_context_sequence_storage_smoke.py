"""ec_context-based sequence storage SMOKE probe (Direction A).

Per the overnight synthesis: the bio_brain_regions substrate doesn't
store SEQUENTIAL slot-position structure because the engram-tagging
captures the UNION of sequentially-fired activity. Biology-faithful
refinement: drive lang_input(slot_word) + ec_context(slot_position)
SIMULTANEOUSLY per slot, so the engram captures (word, position)
co-firing -- sequence STRUCTURE preserved.

The project HAS ec_context as a validated substrate component (catalog
D.01 + D.02 + D.11; positional_drive_pattern function in
sim/text_embeddings; ec_context region built when enable_positional_-
context=True in build_concept_bridge). The (c) build did NOT integrate
it. This smoke tests whether integration is feasible at small scale.

SMOKE SCALE (fast iteration):
- Bridge: build_concept_bridge(enable_positional_context=True);
  small scale (n_lang=512, n_per_pool=100, n_fs=12)
- Vocab: 12 words (motor + noun + verb; no adjective)
- K=4 sequences of 3 slots each
- Few training events
- Engram-tag each sequence with ec_context positional binding
- Stim each engram + read lang_output cosine to vocab words
- Check slot-3 word in top-3

INTERPRETATION:
- Slot-3 in top-3 multi-seed >> chance (3/12 = 0.25): ec_context
  positional binding genuinely preserves sequence structure;
  Direction A is viable; scale up to full
- Slot-3 in top-3 ~at chance: ec_context-augmented encoding doesn't
  help with this readout either; deeper rework needed

Reuses every primitive byte-unchanged where possible; no protected/
frozen/moat module modified; no autograd; no-confab moat must stay
7/7 green.
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

from research.runners.concept_pool_demo import (
    build_concept_bridge, apply_concept_topographic_bias,
    train_word_to_pool, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
)
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from sim.text_embeddings import (
    orthogonal_drive_pattern,
)
from sim.backend import get_backend, is_gpu_backend

OUT_JSON = os.path.join(
    _HERE, "ec_context_sequence_storage_smoke.json")

# SMOKE config
N_LANG_INPUT_SMOKE = 512
N_PER_POOL_SMOKE = 100
N_FS_PER_POOL_SMOKE = 12
N_TRAIN_EVENTS_SMOKE = 30
K_PAIRS = 4
SLOT_COUNT = 3
TOP_K_READOUT = 3
N_REPEATS_PER_TAG = 3
STIM_STEPS = 100
STIM_DRIVE_PA = 1500.0
READOUT_STEPS = 50
N_EC_CONTEXT = 100  # ec_context region size
EC_DRIVE_PA = 200.0  # ec_context positional drive strength


def positional_drive_pattern(slot_idx, n_slots, n_ec_neurons,
                              drive_pA=EC_DRIVE_PA, sparsity=0.1):
    """Per-slot positional drive pattern at ec_context resolution.
    Each slot gets a deterministic non-overlapping band of ec_context
    neurons (mirrors orthogonal_drive_pattern but for slot positions
    instead of word identities)."""
    band_size = max(1, int(n_ec_neurons * sparsity))
    start = (slot_idx * band_size) % n_ec_neurons
    end = start + band_size
    pattern = np.zeros(n_ec_neurons, dtype=np.float32)
    pattern[start:end] = drive_pA
    return pattern


def _build_smoke_bridge(seed: int, verbose: bool = True):
    """Build a small bridge WITH ec_context positional context
    enabled."""
    bridge = build_concept_bridge(
        seed=seed,
        n_lang_input=N_LANG_INPUT_SMOKE,
        n_per_pool=N_PER_POOL_SMOKE,
        n_fs_per_pool=N_FS_PER_POOL_SMOKE,
        enable_adjective=False,  # 12 words: motor + noun + verb
        weak_dynamics=True,
        enable_positional_context=True,  # KEY: enables ec_context
        verbose=verbose,
    )
    return bridge


def _train_substrate_minimal(bridge, words, seed, verbose=True):
    """Minimal training: topographic bias + small interleaved
    training schedule."""
    word_to_idx = {w: i for i, w in enumerate(words)}
    target_pool = {}
    for w in DIRECTION_VOCAB:
        if w == "north": target_pool[w] = "motor_N"
        elif w == "east": target_pool[w] = "motor_E"
        elif w == "south": target_pool[w] = "motor_S"
        elif w == "west": target_pool[w] = "motor_W"
    for w in NOUN_VOCAB:
        target_pool[w] = f"noun_pool_{w.upper()}"
    for w in VERB_VOCAB:
        target_pool[w] = f"verb_pool_{w.upper()}"

    apply_concept_topographic_bias(
        bridge, n_lang_input=N_LANG_INPUT_SMOKE,
        topographic_factor=3.0, off_target_factor=0.3,
        sparsity=0.05, orthogonal_codes=True,
        n_words_for_orthogonal=len(words), word_to_idx=word_to_idx,
        verbose=verbose)

    rng = np.random.default_rng(seed)
    schedule = []
    for w in words:
        for _ in range(N_TRAIN_EVENTS_SMOKE):
            schedule.append(w)
    rng.shuffle(schedule)
    for ei, w in enumerate(schedule):
        train_word_to_pool(
            bridge, word=w, target_pool_region=target_pool[w],
            n_events=1, n_lang_input=N_LANG_INPUT_SMOKE,
            n_lang_output=N_LANG_INPUT_SMOKE,
            sparsity=0.05, orthogonal_codes=True,
            n_words_for_orthogonal=len(words),
            word_to_idx=word_to_idx, verbose=False)


def _encode_sequence_with_ec_context(bridge, seq, words,
                                       seq_idx, ec_drive_pA=EC_DRIVE_PA):
    """Encode the sequence as an engram with simultaneous lang_input
    + ec_context drives per slot. Engram captures (word, position)
    co-firing across all slots."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    n_lang_input = len(lang_input_idx)
    ec_context_idx = list(rm.indices("ec_context"))
    ec_context_arr = cp.asarray(ec_context_idx, dtype=cp.int64)
    n_ec = len(ec_context_idx)
    word_to_idx = {w: i for i, w in enumerate(words)}
    tag_name = f"ec_seq_{seq_idx:03d}"

    # Reset.
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Begin engram recording.
    bridge.start_engram_recording(tag_name)
    encoding_steps_per_slot = 30
    for slot_idx, word in enumerate(seq):
        # Drive lang_input(word) + ec_context(slot_position) simultaneously.
        drive_word = orthogonal_drive_pattern(
            cue_idx=word_to_idx[word], n_cues=len(words),
            n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=0.05)
        drive_pos = positional_drive_pattern(
            slot_idx, SLOT_COUNT, n_ec,
            drive_pA=ec_drive_pA, sparsity=0.1)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[lang_input_arr] = cp.asarray(
            drive_word, dtype=cp.float32)
        bridge.cp_external_input_current[ec_context_arr] = cp.asarray(
            drive_pos, dtype=cp.float32)
        for _ in range(encoding_steps_per_slot):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
    # Commit engram tag.
    stats = bridge.commit_engram_tag(
        tag_name, top_k=100, region_filter=["ca3"])
    bridge.cp_external_input_current[:] = 0.0
    return tag_name, stats


def _capture_lang_output_with_pos_cue(bridge, tag_name, n_lang_output,
                                        cue_slot_idx, cue_n_ec,
                                        ec_context_arr,
                                        stim_steps=STIM_STEPS,
                                        readout_steps=READOUT_STEPS,
                                        drive_pA=STIM_DRIVE_PA,
                                        ec_drive_pA=EC_DRIVE_PA):
    """Stim engram + drive ec_context(slot=cue_slot_idx) +
    capture lang_output. The ec_context cue selects WHICH slot's
    word should be reactivated."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_output_idx = list(rm.indices("language_output"))
    lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Drive ec_context at the cue slot position.
    cue_pos_pattern = positional_drive_pattern(
        cue_slot_idx, SLOT_COUNT, cue_n_ec,
        drive_pA=ec_drive_pA, sparsity=0.1)
    bridge.cp_external_input_current[ec_context_arr] = cp.asarray(
        cue_pos_pattern, dtype=cp.float32)
    # Stim the engram tag.
    bridge.stimulate_tag(tag_name, drive_pA=drive_pA, additive=False)

    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    for _ in range(stim_steps + readout_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[lang_output_arr]
        lang_counts = lang_counts + fired.astype(cp.float32)
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(lang_counts)


def _word_score(lang_counts, word, n_lang_output, words):
    word_to_idx = {w: i for i, w in enumerate(words)}
    pattern = orthogonal_drive_pattern(
        cue_idx=word_to_idx[word], n_cues=len(words),
        n_neurons=n_lang_output, drive_max_pA=1.0, sparsity=0.05)
    a = lang_counts.astype(np.float64)
    b = pattern.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== ec_context-based sequence storage SMOKE (Direction A) ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  SMOKE: small bridge + 12 words + K={K_PAIRS} sequences "
          f"+ slot_count={SLOT_COUNT}", flush=True)

    seed = 42
    print(f"\n--- seed {seed} ---", flush=True)
    cp, _ = get_backend()

    words = list(DIRECTION_VOCAB) + list(NOUN_VOCAB) + list(VERB_VOCAB)
    print(f"  words (V={len(words)}): {words}", flush=True)

    print(f"  building substrate with ec_context enabled...",
          flush=True)
    bridge = _build_smoke_bridge(seed, verbose=True)
    rm = bridge.region_manager
    ec_context_idx = list(rm.indices("ec_context"))
    n_ec = len(ec_context_idx)
    ec_context_arr = cp.asarray(ec_context_idx, dtype=cp.int64)
    print(f"  ec_context region size: {n_ec} neurons", flush=True)
    print(f"  training substrate (~minimal recipe; {N_TRAIN_EVENTS_SMOKE}"
          f" events/word x {len(words)} words = "
          f"{N_TRAIN_EVENTS_SMOKE * len(words)} events)...", flush=True)
    t_train = time.time()
    _train_substrate_minimal(bridge, words, seed, verbose=False)
    print(f"  training done in {(time.time()-t_train)/60:.1f} min",
          flush=True)

    # Generate K sequences.
    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=len(words),
        slot_count=SLOT_COUNT, vocab=words)
    print(f"  sequences:", flush=True)
    for i, s in enumerate(sequences):
        print(f"    seq {i}: {list(s)}", flush=True)

    # Encode each sequence with ec_context positional binding.
    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = _encode_sequence_with_ec_context(
            bridge, seq, words, seq_idx)
        tag_names.append(tag)
        print(f"  encoded {tag} for {list(seq)} (n_tagged="
              f"{stats.get('n_tagged', 0)})", flush=True)

    # For each sequence, test slot-3 retrieval with ec_context cue:
    # stim engram + drive ec_context(slot=2) + check lang_output
    # cosine for slot-3 word in top-K.
    n_lang_output = N_LANG_INPUT_SMOKE  # same as lang_input
    per_seq_results = []
    n_slot3_correct = 0
    for seq_idx, (seq, tag_name) in enumerate(
            zip(sequences, tag_names)):
        cue_slot_idx = SLOT_COUNT - 1  # slot index 2 (the LAST slot)
        true_slot3 = seq[cue_slot_idx]

        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = _capture_lang_output_with_pos_cue(
                bridge, tag_name, n_lang_output,
                cue_slot_idx=cue_slot_idx, cue_n_ec=n_ec,
                ec_context_arr=ec_context_arr)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG

        scores = {w: _word_score(avg_counts, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(),
                       key=lambda x: x[1], reverse=True)[:TOP_K_READOUT]
        topK_words = [w for w, s in topK]
        correct = (true_slot3 in topK_words)
        if correct:
            n_slot3_correct += 1
        per_seq_results.append({
            "seq_idx": seq_idx, "sequence": list(seq),
            "true_slot3": true_slot3,
            "topK_words": topK_words,
            "topK_scores": [round(s, 4) for w, s in topK],
            "correct": correct,
        })
        print(f"    seq {seq_idx} {list(seq)}: cue slot {cue_slot_idx}"
              f"; true_slot3={true_slot3}; top{TOP_K_READOUT}="
              f"{topK_words}; correct={correct}", flush=True)

    slot3_acc = n_slot3_correct / K_PAIRS
    chance = TOP_K_READOUT / len(words)
    print(f"\n  smoke result (seed 42): slot3-completion = "
          f"{n_slot3_correct}/{K_PAIRS} = {slot3_acc:.3f} "
          f"(chance = {chance:.3f})", flush=True)

    if slot3_acc > 2 * chance:
        verdict = "EC_CONTEXT_PRELIMINARY_SIGNAL"
        print(f"  Smoke shows ABOVE-CHANCE signal "
              f"({slot3_acc:.3f} > 2*{chance:.3f}). Direction A "
              f"viable; scale up to multi-seed full.", flush=True)
    elif slot3_acc > chance:
        verdict = "EC_CONTEXT_MARGINAL_SIGNAL"
        print(f"  Marginal signal ({slot3_acc:.3f} > chance "
              f"{chance:.3f} but < 2x). Direction A may need "
              f"refinement before full scale.", flush=True)
    else:
        verdict = "EC_CONTEXT_NEGATIVE_SMOKE"
        print(f"  Smoke at chance or below ({slot3_acc:.3f} vs "
              f"{chance:.3f}). Direction A as designed doesn't "
              f"help with this readout; deeper rework needed.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": seed,
        "V": len(words), "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT,
        "TOP_K_READOUT": TOP_K_READOUT,
        "n_ec_context": n_ec,
        "n_train_events_per_word": N_TRAIN_EVENTS_SMOKE,
        "stim_drive_pA": STIM_DRIVE_PA,
        "ec_drive_pA": EC_DRIVE_PA,
        "chance_baseline": chance,
        "slot3_accuracy_smoke": slot3_acc,
        "per_seq": per_seq_results,
        "verdict_smoke": verdict,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
