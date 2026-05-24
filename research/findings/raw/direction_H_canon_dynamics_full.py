"""Direction H FULL: canon concept-pool dynamics + multi-seed v16
recipe training + Phase 1 W->A control + sequence storage test.

Per Direction H smoke (commit ff974d3): canon dynamics
(weak_dynamics=False) PRESERVE Phase 1 W->A at 0.50 single-seed
smoke. Full GPU run justified.

Recipe:
1. Build substrate with canon dynamics (weak_dynamics=False)
2. Train via v16 recipe (200 events/word, 16 words) per seed
3. CONTROL: Phase 1 W->A test (>= 0.70 multi-seed required to
   proceed)
4. SEQUENCE STORAGE: Direction A v1's engram-tag mechanism on
   canon-dynamics substrate (same recipe; 8 sequences x SLOT=3;
   strict top-1)
5. Multi-seed 3 seeds [42,43,44]
6. Pre-registered FROZEN bar: Phase 1 >= 0.70 AND sequence storage
   >= 0.80 multi-seed STRICT TOP-1

Outcomes (from design doc):
- (a) Both pass: pillar n=105 candidate (canon enables both)
- (b) Phase 1 pass, sequence fail: BOUNDARY (bound is mechanism)
- (c) Phase 1 fail: NEGATIVE (canon breaks trainability)

~5 hr GPU; kill-safe per-seed cache.

Reuses Direction A's engram-tag sequence storage mechanism
byte-unchanged.
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
    ADJECTIVE_VOCAB,
)
from research.runners.concept_compose_train import _WORD_TO_POOL
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _build_region_filter,
)
from research.findings.raw.direction_H_canon_dynamics_smoke import (
    w_to_a_test, N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL,
    TOPOGRAPHIC_FACTOR, OFF_TARGET_FACTOR, SPARSITY,
)
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, is_gpu_backend


CACHE_DIR = os.path.join(_HERE, "direction_H_canon_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_JSON = os.path.join(_HERE, "direction_H_canon_dynamics_full.json")
SEEDS = [42, 43, 44]
N_TRAIN_EVENTS = 200  # full
K_PAIRS = 8
SLOT_COUNT = 3
TEACHER_PA = 500.0
ENCODING_STEPS_PER_SLOT = 60
ENGRAM_TOP_K = 100
STIM_DRIVE_PA = 1500.0
STIM_STEPS = 100
READOUT_STEPS = 50
N_REPEATS_PER_TAG = 3
PHASE1_BAR = 0.70
SEQUENCE_BAR = 0.80


def _bridge_save_path(seed):
    return os.path.join(CACHE_DIR, f"bridge_seed{seed}.simstate.h5")


def _trials_save_path(seed):
    return os.path.join(CACHE_DIR, f"trials_seed{seed}.json")


def _build_and_train(seed, verbose=True):
    """Build canon-dynamics substrate + train via v16 recipe."""
    bridge_p = _bridge_save_path(seed)
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(words)}
    n_words = len(words)

    # CANON DYNAMICS via weak_dynamics=False
    bridge = build_concept_bridge(
        seed=seed, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=False,  # <-- THE CANON DYNAMICS FLAG
        enable_positional_context=False, verbose=verbose,
    )
    if os.path.exists(bridge_p):
        print(f"  [seed {seed}] loading cached canon bridge",
              flush=True)
        bridge.load_checkpoint(bridge_p)
        return bridge, words, word_to_idx

    print(f"  [seed {seed}] training canon substrate ({n_words}"
          f" words x {N_TRAIN_EVENTS} events)", flush=True)
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
    for w in NOUN_VOCAB: target_pool[w] = f"noun_pool_{w.upper()}"
    for w in VERB_VOCAB: target_pool[w] = f"verb_pool_{w.upper()}"
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
                  f" ({(time.time()-t_train)/60:.1f} min)",
                  flush=True)
    bridge.save_checkpoint(bridge_p)
    print(f"  [seed {seed}] trained in "
          f"{(time.time()-t_train)/60:.1f} min", flush=True)
    return bridge, words, word_to_idx


def encode_sequence_engram(bridge, seq, words, seq_idx, region_filter):
    """Direction A v1 engram-tag encoding on canon-dynamics substrate.
    Same as Direction A's _encode_sequence_with_ec_context but
    WITHOUT ec_context (we're testing if canon dynamics alone enable
    sequence storage via top-K engram capture)."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]
    word_to_idx = {w: i for i, w in enumerate(words)}
    tag_name = f"canon_seq_{seq_idx:03d}"

    pool_arrs = []
    for slot_word in seq:
        try:
            pool_idx = list(rm.indices(_WORD_TO_POOL[slot_word]))
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
            n_neurons=N_LANG_INPUT, drive_max_pA=200.0,
            sparsity=SPARSITY)
        for _ in range(ENCODING_STEPS_PER_SLOT):
            ext.fill(0)
            ext[lang_in_arr] = cp.asarray(
                drive_word, dtype=cp.float32)
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
    return tag_name, stats


def retrieve_sequence(bridge, tag_name, n_lang_output):
    """Stim engram + read lang_output cosines."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.stimulate_tag(tag_name, drive_pA=STIM_DRIVE_PA,
                          additive=False)

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
        n_neurons=n_lang_output, drive_max_pA=1.0,
        sparsity=SPARSITY)
    a = lang_counts.astype(np.float64); b = pattern.astype(
        np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_one_seed(seed, verbose=True):
    print(f"\n--- seed {seed} (canon dynamics) ---", flush=True)
    trials_p = _trials_save_path(seed)
    if os.path.exists(trials_p):
        print(f"  [seed {seed}] loading cached trials", flush=True)
        with open(trials_p, "r", encoding="utf-8") as f:
            return json.load(f)

    cp, _ = get_backend()
    bridge, words, word_to_idx = _build_and_train(seed, verbose=verbose)
    rm = bridge.region_manager
    n_lang_output = N_LANG_INPUT
    region_filter = _build_region_filter(rm)
    target_pool = {}
    for w in DIRECTION_VOCAB:
        if w == "north": target_pool[w] = "motor_N"
        elif w == "east": target_pool[w] = "motor_E"
        elif w == "south": target_pool[w] = "motor_S"
        elif w == "west": target_pool[w] = "motor_W"
    for w in NOUN_VOCAB: target_pool[w] = f"noun_pool_{w.upper()}"
    for w in VERB_VOCAB: target_pool[w] = f"verb_pool_{w.upper()}"
    for w in ADJECTIVE_VOCAB:
        target_pool[w] = f"adjective_pool_{w.upper()}"

    # Freeze for inference test
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

    # PHASE 1 CONTROL
    print(f"  [seed {seed}] Phase 1 W->A control",
          flush=True)
    phase1_acc = w_to_a_test(bridge, words, word_to_idx,
                                target_pool, verbose=True)
    print(f"  [seed {seed}] Phase 1 W->A acc = {phase1_acc:.3f}",
          flush=True)

    # SEQUENCE STORAGE TEST
    print(f"  [seed {seed}] sequence storage test ({K_PAIRS} seqs)",
          flush=True)
    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=len(words),
        slot_count=SLOT_COUNT, vocab=words)
    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = encode_sequence_engram(
            bridge, seq, words, seq_idx, region_filter)
        tag_names.append(tag)

    n_top1 = 0
    per_seq = []
    for seq_idx, (seq, tag_name) in enumerate(zip(sequences,
                                                       tag_names)):
        true = seq[SLOT_COUNT - 1]
        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = retrieve_sequence(
                bridge, tag_name, n_lang_output)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_counts, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1 = topK[0][0]
        correct = (top1 == true)
        if correct: n_top1 += 1
        per_seq.append({
            "seq_idx": seq_idx, "sequence": list(seq),
            "true_slot": true, "top1": top1,
            "top1_correct": correct,
            "topK_words": [w for w, _ in topK[:5]],
        })
        if verbose:
            print(f"    seq {seq_idx} {list(seq)} true={true} "
                  f"top1={top1} correct={correct}", flush=True)

    seq_acc = n_top1 / K_PAIRS
    print(f"  [seed {seed}] sequence storage top-1 = {n_top1}"
          f"/{K_PAIRS} = {seq_acc:.3f}", flush=True)
    result = {
        "seed": seed, "phase1_w_to_a_acc": phase1_acc,
        "sequence_storage_top1": seq_acc, "K_PAIRS": K_PAIRS,
        "SLOT_COUNT": SLOT_COUNT,
        "per_seq": per_seq,
    }
    with open(trials_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction H FULL: canon concept-pool dynamics ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Pre-registered FROZEN bars: Phase 1 >= {PHASE1_BAR},"
          f" Sequence Storage >= {SEQUENCE_BAR}", flush=True)

    t0 = time.time()
    seed_results = []
    for seed in SEEDS:
        r = run_one_seed(seed, verbose=True)
        seed_results.append(r)
    total_min = (time.time() - t0) / 60

    phase1_accs = [r["phase1_w_to_a_acc"] for r in seed_results]
    seq_accs = [r["sequence_storage_top1"] for r in seed_results]
    phase1_mean = float(np.mean(phase1_accs))
    seq_mean = float(np.mean(seq_accs))

    print(f"\n=== MULTI-SEED RESULTS ===", flush=True)
    print(f"  Phase 1 W->A mean = {phase1_mean:.3f} per-seed"
          f"={phase1_accs}", flush=True)
    print(f"  Sequence storage mean = {seq_mean:.3f} per-seed"
          f"={seq_accs}", flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if phase1_mean >= PHASE1_BAR and seq_mean >= SEQUENCE_BAR:
        verdict = "DIRECTION_H_BOTH_PASS_PILLAR_N105_CANDIDATE"
        print(f"  BOTH PASS: Phase 1 {phase1_mean:.3f} >= "
              f"{PHASE1_BAR} AND sequence storage {seq_mean:.3f}"
              f" >= {SEQUENCE_BAR}. Canon concept-pool dynamics "
              f"PRESERVE multi-concept trainability AND enable "
              f"substrate-level sequence storage. PILLAR N=105 "
              f"CANDIDATE.", flush=True)
    elif phase1_mean >= PHASE1_BAR:
        verdict = "DIRECTION_H_PHASE1_PASS_SEQUENCE_BOUNDED"
        print(f"  Phase 1 PASS but sequence storage below bar "
              f"({seq_mean:.3f} < {SEQUENCE_BAR}). Canon dynamics"
              f" preserve trainability but bound is mechanism-level"
              f" (engram-tag) not dynamics-level. Confirms pillar"
              f" n=104 diagnosis.", flush=True)
    else:
        verdict = "DIRECTION_H_PHASE1_BROKEN_TRAINABILITY_LOSS"
        print(f"  Phase 1 BROKEN ({phase1_mean:.3f} < {PHASE1_BAR})"
              f"; canon dynamics break trainability per v14"
              f" finding; pivot to Direction I or L.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "weak_dynamics": False, "canon_dynamics": True,
        "n_train_events": N_TRAIN_EVENTS,
        "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT,
        "phase1_bar": PHASE1_BAR, "sequence_bar": SEQUENCE_BAR,
        "phase1_w_to_a_mean": phase1_mean,
        "phase1_per_seed": phase1_accs,
        "sequence_storage_mean": seq_mean,
        "sequence_per_seed": seq_accs,
        "per_seed": seed_results, "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
