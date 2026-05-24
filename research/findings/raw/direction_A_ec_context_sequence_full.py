"""Direction A FULL SCALE: ec_context-based sequence storage on a
properly-trained substrate with enable_positional_context=True.

Per overnight: the (c) loop NEGATIVE diagnostic (REPLAY_DOESNT_-
REACTIVATE) precisely identified that slot-position structure is
missing. ec_context (project catalog D.01+D.02+D.11) is the validated
substrate component for slot-position binding. Earlier smoke was
inconclusive (0-neuron engram tags at small scale). Full v16 training
should produce meaningful tags.

DESIGN:
- Build substrate via concept_pool_demo.build_concept_bridge with
  enable_positional_context=True (adds ec_context region; ~200
  neurons; pathways ec_context -> concept pools + motors)
- Train substrate via the full v16 recipe (200 events/word x 16
  words = 3200 events; topographic prior; orthogonal codes; weak
  dynamics)
- For each test: encode K sequences via SIMULTANEOUS lang_input(word)
  + ec_context(position) drives per slot; engram-tag each sequence
- Test: stim each engram + read lang_output cosine; check slot-3 word
  in top-3 (sequence completion)
- Multi-seed (42, 43, 44); kill-safe per-seed cache
- Pre-registered 0.80 bar on slot-3-completion

Reuses every validated primitive byte-unchanged; no protected/frozen/
moat module modified; no autograd; no-confab moat must stay 7/7
green.

Expected ~50 min train per seed (v16 recipe) + ~10 min sequence
encoding + ~5 min readout = ~65 min per seed at full scale x 3 seeds
= ~3 hr GPU total. Kill-safe per-seed cache means resumable.
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
    build_concept_bridge, apply_concept_topographic_bias,
    train_word_to_pool, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.runners.concept_compose_train import _WORD_TO_POOL
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from sim.text_embeddings import (
    orthogonal_drive_pattern, positional_drive_pattern,
)
from sim.backend import get_backend, is_gpu_backend

CACHE_DIR = os.path.join(
    _HERE, "direction_A_ec_context_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Pre-registered config matching v16 production recipe.
N_LANG_INPUT = 2048
N_PER_POOL = 200
N_FS_PER_POOL = 24
N_EC_CONTEXT = 200  # ec_context region size
N_TRAIN_EVENTS = 200
TOPOGRAPHIC_FACTOR = 3.0
OFF_TARGET_FACTOR = 0.3
SPARSITY = 0.05

K_PAIRS = 8  # 8 sequences (matches multitag K)
SLOT_COUNT = 3
TOP_K_READOUT = 3  # slot-3 in top-3
N_REPEATS_PER_TAG = 3
STIM_STEPS = 100
READOUT_STEPS = 50
STIM_DRIVE_PA = 1500.0
EC_DRIVE_PA = 200.0
ENCODING_STEPS_PER_SLOT = 60  # ~2x baseline since lang_input + ec_context
ENGRAM_TOP_K = 100
# Validated multitag uses balanced_teacher_pA=500 to ensure target pool
# fires during encoding (the v14/v16 substrate's weak concept-pool
# dynamics don't ignite from lang_input alone reliably). Without teacher,
# engram tags are 0-neuron and the readout collapses to chance. Mirrors
# the validated 2026-05-14 multitag recipe.
TEACHER_PA = 500.0

OUT_JSON = os.path.join(
    _HERE, "direction_A_ec_context_sequence_full.json")
SEEDS = [42, 43, 44]


def _bridge_save_path(seed):
    return os.path.join(CACHE_DIR, f"bridge_full_seed{seed}.simstate.h5")


def _trials_save_path(seed):
    return os.path.join(CACHE_DIR, f"trials_full_seed{seed}.json")


def _build_and_train(seed, verbose=True):
    """Build v16-style substrate WITH enable_positional_context=True;
    train via v16 recipe; save."""
    bridge_p = _bridge_save_path(seed)
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(words)}
    n_words = len(words)

    bridge = build_concept_bridge(
        seed=seed,
        n_lang_input=N_LANG_INPUT,
        n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL,
        enable_adjective=True,
        weak_dynamics=True,
        enable_positional_context=True,
        n_ec_context=N_EC_CONTEXT,
        verbose=verbose,
    )

    if os.path.exists(bridge_p):
        print(f"  [seed {seed}] loading cached trained bridge "
              f"({bridge_p})", flush=True)
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
        return bridge, words, word_to_idx

    print(f"  [seed {seed}] training v16+ec_context substrate "
          f"({n_words} words x {N_TRAIN_EVENTS} events)...",
          flush=True)
    t_train = time.time()
    apply_concept_topographic_bias(
        bridge, n_lang_input=N_LANG_INPUT,
        topographic_factor=TOPOGRAPHIC_FACTOR,
        off_target_factor=OFF_TARGET_FACTOR,
        sparsity=SPARSITY, orthogonal_codes=True,
        n_words_for_orthogonal=n_words,
        word_to_idx=word_to_idx, verbose=verbose)

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
    for w in ADJECTIVE_VOCAB:
        target_pool[w] = f"adjective_pool_{w.upper()}"

    rng = np.random.default_rng(seed)
    schedule = []
    for w in words:
        for _ in range(N_TRAIN_EVENTS):
            schedule.append(w)
    rng.shuffle(schedule)
    for ei, w in enumerate(schedule):
        train_word_to_pool(
            bridge, word=w, target_pool_region=target_pool[w],
            n_events=1, n_lang_input=N_LANG_INPUT,
            n_lang_output=N_LANG_INPUT,
            sparsity=SPARSITY, orthogonal_codes=True,
            n_words_for_orthogonal=n_words,
            word_to_idx=word_to_idx, verbose=False)
        if verbose and (ei + 1) % max(1, len(schedule) // 10) == 0:
            print(f"    [seed {seed}] {ei+1}/{len(schedule)} events"
                  f" ({(time.time()-t_train)/60:.1f} min)", flush=True)
    bridge.save_checkpoint(bridge_p)
    print(f"  [seed {seed}] trained + saved in "
          f"{(time.time()-t_train)/60:.1f} min", flush=True)
    return bridge, words, word_to_idx


def _build_region_filter(rm):
    """Build the concept-pool region filter the validated multitag
    uses (noun/verb/adjective pools + motor regions). Mirrors
    multitag_eval.py lines 62-72."""
    region_filter = []
    for kind, names in [
        ("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"]),
    ]:
        for n in names:
            try:
                rm.indices(f"{kind}_{n}")
                region_filter.append(f"{kind}_{n}")
            except Exception:
                pass
    for m in ["motor_N", "motor_E", "motor_S", "motor_W"]:
        try:
            rm.indices(m)
            region_filter.append(m)
        except Exception:
            pass
    return region_filter


def _encode_sequence_with_ec_context(bridge, seq, words, seq_idx,
                                       region_filter):
    """Drive lang_input(slot_word) + ec_context(slot_position) +
    teacher current on slot_word's target pool simultaneously per
    slot; engram (on the concept pools + motors, not CA3) captures
    (word, position) co-firing across all slots.

    Mirrors validated multitag encoding (compose_concept_engram.
    encode_concept_pair) byte-equivalent for the word side; adds
    ec_context positional drive per slot.
    """
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    n_lang_input = len(lang_in_idx)
    ec_idx = list(rm.indices("ec_context"))
    ec_arr = cp.asarray(ec_idx, dtype=cp.int64)
    n_ec = len(ec_idx)
    word_to_idx = {w: i for i, w in enumerate(words)}
    n_total = bridge.cp_external_input_current.shape[0]
    tag_name = f"ec_seq_{seq_idx:03d}"

    # Pre-build the per-slot target pool index arrays.
    pool_arrs = []
    for slot_word in seq:
        pool_region = _WORD_TO_POOL.get(slot_word, None)
        if pool_region is None:
            pool_arrs.append(None)
            continue
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
            slot_idx, n_neurons=n_ec, n_max_positions=SLOT_COUNT)
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


def _capture_lang_output_with_pos_cue(bridge, tag_name, cue_slot_idx,
                                        n_ec, ec_arr, n_lang_output):
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    cue_pos = positional_drive_pattern(
        cue_slot_idx, n_neurons=n_ec, n_max_positions=SLOT_COUNT)
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


def _word_score(lang_counts, word, n_lang_output, words):
    word_to_idx = {w: i for i, w in enumerate(words)}
    pattern = orthogonal_drive_pattern(
        cue_idx=word_to_idx[word], n_cues=len(words),
        n_neurons=n_lang_output, drive_max_pA=1.0, sparsity=SPARSITY)
    a = lang_counts.astype(np.float64); b = pattern.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_one_seed(seed, verbose=True):
    print(f"\n--- seed {seed} ---", flush=True)
    trials_p = _trials_save_path(seed)
    if os.path.exists(trials_p):
        print(f"  [seed {seed}] loading cached trials", flush=True)
        with open(trials_p, "r", encoding="utf-8") as f:
            return json.load(f)

    cp, _ = get_backend()
    bridge, words, word_to_idx = _build_and_train(seed, verbose=verbose)
    rm = bridge.region_manager
    ec_idx = list(rm.indices("ec_context"))
    ec_arr = cp.asarray(ec_idx, dtype=cp.int64)
    n_ec = len(ec_idx)
    n_lang_output = N_LANG_INPUT
    region_filter = _build_region_filter(rm)
    print(f"  [seed {seed}] region_filter ({len(region_filter)}"
          f" regions): {region_filter}", flush=True)

    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=len(words),
        slot_count=SLOT_COUNT, vocab=words)
    print(f"  [seed {seed}] sequences:", flush=True)
    for i, s in enumerate(sequences):
        print(f"    seq {i}: {list(s)}", flush=True)

    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = _encode_sequence_with_ec_context(
            bridge, seq, words, seq_idx, region_filter)
        tag_names.append(tag)
        print(f"  encoded {tag} ({list(seq)}); n_tagged="
              f"{stats.get('n_tagged', 0)}", flush=True)

    per_seq = []
    n_correct = 0
    for seq_idx, (seq, tag_name) in enumerate(zip(sequences, tag_names)):
        cue_slot_idx = SLOT_COUNT - 1
        true_slot3 = seq[cue_slot_idx]
        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = _capture_lang_output_with_pos_cue(
                bridge, tag_name, cue_slot_idx, n_ec, ec_arr,
                n_lang_output)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_counts, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)[:TOP_K_READOUT]
        topK_words = [w for w, s in topK]
        correct = (true_slot3 in topK_words)
        if correct: n_correct += 1
        per_seq.append({
            "seq_idx": seq_idx, "sequence": list(seq),
            "true_slot3": true_slot3,
            "topK_words": topK_words,
            "topK_scores": [round(s, 4) for w, s in topK],
            "correct": correct,
        })
        print(f"    seq {seq_idx} {list(seq)}: true_slot3="
              f"{true_slot3}; top-{TOP_K_READOUT}={topK_words};"
              f" correct={correct}", flush=True)

    acc = n_correct / K_PAIRS
    result = {
        "seed": seed, "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT,
        "n_correct": n_correct, "slot3_accuracy": acc,
        "per_seq": per_seq,
    }
    with open(trials_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [seed {seed}] slot-3 accuracy = {n_correct}/{K_PAIRS}"
          f" = {acc:.3f}", flush=True)
    return result


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== Direction A FULL: ec_context-based sequence storage ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  recipe: n_lang={N_LANG_INPUT}, n_per_pool={N_PER_POOL},"
          f" n_ec_context={N_EC_CONTEXT}, n_train_events={N_TRAIN_EVENTS}",
          flush=True)
    print(f"  K_PAIRS={K_PAIRS}, SLOT_COUNT={SLOT_COUNT}, "
          f"TOP_K_READOUT={TOP_K_READOUT}", flush=True)

    chance = TOP_K_READOUT / 16.0
    print(f"  Chance baseline (top-{TOP_K_READOUT} of 16): "
          f"{chance:.3f}", flush=True)
    print(f"  Pre-registered bar: 0.80 multi-seed slot-3-accuracy",
          flush=True)

    seed_results = []
    t0 = time.time()
    for seed in SEEDS:
        r = run_one_seed(seed, verbose=True)
        seed_results.append(r)
    total_min = (time.time() - t0) / 60

    accs = [r["slot3_accuracy"] for r in seed_results]
    mean = float(np.mean(accs))
    print(f"\n=== MULTI-SEED RESULT ===", flush=True)
    print(f"  slot-3-accuracy mean = {mean:.3f} per-seed="
          f"[{', '.join(f'{a:.3f}' for a in accs)}]", flush=True)
    print(f"  Wall-clock: {total_min:.1f} min", flush=True)

    if mean >= 0.80:
        verdict = "DIRECTION_A_PASS_EC_CONTEXT_SEQUENCE_STORAGE"
        print(f"  PASS multi-seed >= 0.80; ec_context-based sequence"
              f" storage works on the validated substrate; pending"
              f" fresh adversarial review for pillar n=103.",
              flush=True)
    elif mean > 2 * chance:
        verdict = "DIRECTION_A_BOUNDARY_ABOVE_CHANCE_BELOW_BAR"
        print(f"  Multi-seed {mean:.3f} > 2*chance {2*chance:.3f} "
              f"but < 0.80; partial recovery; ec_context helps but"
              f" doesn't fully solve sequence storage at this recipe.",
              flush=True)
    else:
        verdict = "DIRECTION_A_NEGATIVE_AT_CHANCE"
        print(f"  Multi-seed {mean:.3f} at chance ({chance:.3f}); "
              f"ec_context-based sequence storage doesn't work at"
              f" this recipe; deeper substrate-level work needed.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT,
        "n_train_events": N_TRAIN_EVENTS,
        "n_ec_context": N_EC_CONTEXT,
        "chance_baseline": chance,
        "slot3_accuracy_mean": mean,
        "slot3_accuracy_per_seed": accs,
        "per_seed": seed_results, "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
