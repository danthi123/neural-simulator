"""Direction A v2: open ec_context_to_pool plasticity gates DURING
encoding (the intended mechanism; v1 had them frozen which forced
the engram tag to be the only load-bearing piece).

Per adversarial reviewer VERDICT BLOCK on v1: top-3 was degenerate
(set membership not positional); strict top-1 multi-seed expected
to be ~31% above chance ~6% but well below 0.80 bar. The mechanism's
intent was: during encoding, the simultaneous lang_input(word) +
ec_context(position) drive triggers STDP that binds ec_context(pos)
to the word's pool. At retrieval, the ec_context cue then
selectively activates the slot-i word's pool via the learned
pathway, providing genuine positional selectivity.

v1 froze ec_context_to_* gates ALWAYS -- so STDP did NOT grow the
binding during encoding; the engram tag was the only co-firing
record, and the retrieval cue had no learned-pathway selectivity.

v2 fix: temporarily OPEN ec_context_to_noun_pool, ec_context_to_
verb_pool, ec_context_to_adjective_pool, ec_context_to_motor
during the encoding window; freeze them otherwise (so the bound
weights are inference-stable, mirror the validated v16 compose
pattern).

Reuses cached trained bridges (~no extra training time); ~30 min
GPU for 3 seeds.

Pre-registered FROZEN bar: 0.80 multi-seed STRICT TOP-1 (the
reviewer-imposed strict metric). NO bar tuning.
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
    build_concept_bridge, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.runners.concept_compose_train import _WORD_TO_POOL
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _bridge_save_path, _build_region_filter, _word_score,
    N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL, N_EC_CONTEXT,
    K_PAIRS, SLOT_COUNT, EC_DRIVE_PA, STIM_DRIVE_PA,
    STIM_STEPS, READOUT_STEPS, N_REPEATS_PER_TAG, ENGRAM_TOP_K,
    TEACHER_PA, ENCODING_STEPS_PER_SLOT, SPARSITY,
)
from sim.text_embeddings import (
    orthogonal_drive_pattern, positional_drive_pattern,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_A_v2_with_plasticity.json")
SEEDS = [42, 43, 44]
TOP_K_READOUT_STRICT = 1  # STRICT TOP-1 per reviewer
TOP_K_READOUT_TOP3 = 3    # also report top-3 for comparison
BAR = 0.80
EC_PLASTIC_GATES = ("ec_context_to_noun_pool",
                      "ec_context_to_verb_pool",
                      "ec_context_to_adjective_pool",
                      "ec_context_to_motor")


def encode_with_plasticity(bridge, seq, words, seq_idx,
                              region_filter):
    """v2: opens ec_context plasticity gates DURING encoding so
    STDP grows the binding; closes after."""
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
    tag_name = f"v2_ec_seq_{seq_idx:03d}"

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

    # v2 FIX: open ec_context_to_pool plasticity during encoding
    for g in EC_PLASTIC_GATES:
        try:
            bridge.set_plasticity_gate(g, 1.0)
        except Exception:
            pass

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

    # v2 FIX: close gates after encoding (inference-stable)
    for g in EC_PLASTIC_GATES:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    return tag_name, stats


def capture_with_pos_cue(bridge, tag_name, cue_slot_idx, n_ec,
                            ec_arr, n_lang_output):
    """Same as v1's _capture_lang_output_with_pos_cue."""
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


def run_seed(seed, verbose=True):
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

    # Freeze all NON-ec_context gates (matches v1's freezing)
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
    # ec_context_to_* gates start CLOSED; encode_with_plasticity
    # opens/closes them per-sequence.
    for g in EC_PLASTIC_GATES:
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

    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=n_words,
        slot_count=SLOT_COUNT, vocab=words)

    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = encode_with_plasticity(
            bridge, seq, words, seq_idx, region_filter)
        tag_names.append(tag)
        if verbose:
            print(f"  [seed {seed}] encoded {tag}; n_tagged="
                  f"{stats.get('n_tagged', 0)}", flush=True)

    # Test: per sequence, cue slot 2 (last slot)
    n_top1 = 0
    n_top3 = 0
    per_seq = []
    for seq_idx, (seq, tag_name) in enumerate(zip(sequences, tag_names)):
        cue_slot_idx = SLOT_COUNT - 1
        true = seq[cue_slot_idx]
        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_with_pos_cue(
                bridge, tag_name, cue_slot_idx, n_ec, ec_arr,
                n_lang_output)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_counts, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1_word = topK[0][0]
        top3_words = [w for w, s in topK[:3]]
        top1_correct = (top1_word == true)
        top3_correct = (true in top3_words)
        if top1_correct: n_top1 += 1
        if top3_correct: n_top3 += 1
        per_seq.append({
            "seq_idx": seq_idx, "sequence": list(seq),
            "true_slot": true,
            "top1_word": top1_word, "top1_correct": top1_correct,
            "top3_words": top3_words, "top3_correct": top3_correct,
        })

    top1_acc = n_top1 / K_PAIRS
    top3_acc = n_top3 / K_PAIRS
    print(f"  [seed {seed}] top-1 = {n_top1}/{K_PAIRS} = "
          f"{top1_acc:.3f}; top-3 = {n_top3}/{K_PAIRS} = "
          f"{top3_acc:.3f}", flush=True)
    return {
        "seed": seed,
        "top1_acc": top1_acc, "n_top1": n_top1,
        "top3_acc": top3_acc, "n_top3": n_top3,
        "n_total": K_PAIRS, "per_seq": per_seq,
    }


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction A v2 (plasticity-during-encoding) ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  ec_context_to_pool plasticity OPENED during encoding"
          f"; closed after.", flush=True)
    print(f"  STRICT TOP-1 metric (load-bearing); top-3 also "
          f"reported.", flush=True)
    print(f"  Pre-registered FROZEN bar: {BAR} multi-seed strict "
          f"top-1.", flush=True)

    seed_results = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        r = run_seed(seed)
        if r is not None:
            seed_results.append(r)

    total_min = (time.time() - t0) / 60
    if not seed_results:
        print("[FATAL] no cached bridges; v2 cannot run", flush=True)
        return 1
    top1_accs = [r["top1_acc"] for r in seed_results]
    top3_accs = [r["top3_acc"] for r in seed_results]
    top1_mean = float(np.mean(top1_accs))
    top3_mean = float(np.mean(top3_accs))

    print(f"\n=== MULTI-SEED ===", flush=True)
    print(f"  strict top-1 mean: {top1_mean:.3f} per-seed="
          f"{top1_accs}", flush=True)
    print(f"  top-3 mean:        {top3_mean:.3f} per-seed="
          f"{top3_accs}", flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    chance_top1 = 1.0 / 16.0
    print(f"\n=== VERDICT ===", flush=True)
    if top1_mean >= BAR:
        verdict = "V2_STRICT_TOP1_PASS"
        print(f"  PASS at strict top-1 multi-seed >= {BAR} -- the"
              f" plasticity-during-encoding fix makes the "
              f"ec_context positional binding load-bearing.",
              flush=True)
    elif top1_mean > 2 * chance_top1:
        verdict = "V2_STRICT_TOP1_ABOVE_CHANCE_BELOW_BAR"
        print(f"  partial signal at strict top-1: {top1_mean:.3f}"
              f" > 2*chance {2*chance_top1:.3f} but < {BAR};"
              f" plasticity helps but doesn't fully solve.",
              flush=True)
    else:
        verdict = "V2_STRICT_TOP1_AT_CHANCE"
        print(f"  at-chance at strict top-1; the plasticity fix "
              f"didn't change the mechanism's positional binding.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "bar": BAR, "chance_top1": chance_top1,
        "strict_top1_mean": top1_mean,
        "strict_top1_per_seed": top1_accs,
        "top3_mean": top3_mean, "top3_per_seed": top3_accs,
        "per_seed": seed_results, "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
