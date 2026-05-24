"""Direction H smoke: canon concept-pool dynamics smoke + Phase 1
control before committing full GPU run.

Per Direction H design (commit 8144840): test whether canon dynamics
(weak_dynamics=False -> defaults to motor_internal_density=0.10,
exc_weight=2.0, inh_weight=4.0) enable substrate-level sequence
retrieval that v16 WEAK dynamics could not.

Pre-registered Phase 1 control: train substrate; verify W->A binding
still works at >= 0.50 single-seed smoke (full multi-seed pass is
>= 0.70 multi-seed strict). If Phase 1 fails completely (~chance),
canon dynamics broke the trainability per v14 finding -> Direction H
NEGATIVE.

~5-10 min wall single-seed smoke (training + W->A test).
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
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_H_canon_dynamics_smoke.json")
SEED = 42
N_LANG_INPUT = 2048
N_PER_POOL = 200
N_FS_PER_POOL = 24
N_TRAIN_EVENTS = 50  # smoke: less training; full would use 200
TOPOGRAPHIC_FACTOR = 3.0
OFF_TARGET_FACTOR = 0.3
SPARSITY = 0.05
W_TO_A_TEST_REPEATS = 8


def w_to_a_test(bridge, words, word_to_idx, target_pool, verbose=True):
    """Test W->A binding: drive lang_input(word); check if target pool
    fires more than off-target pools. Returns strict top-1 accuracy."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    # Pre-resolve all pool indices
    pool_kinds = [("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
                   ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
                   ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"])]
    pool_indices = {}
    for kind, names in pool_kinds:
        for n in names:
            try:
                pool_indices[f"{kind}_{n}"] = (
                    cp.asarray(list(rm.indices(f"{kind}_{n}")),
                                  dtype=cp.int64))
            except Exception:
                pass
    for m in ["motor_N", "motor_E", "motor_S", "motor_W"]:
        try:
            pool_indices[m] = cp.asarray(
                list(rm.indices(m)), dtype=cp.int64)
        except Exception:
            pass

    n_correct = 0
    n_total_tests = 0
    test_words = list(words)
    rng = np.random.default_rng(42)
    rng.shuffle(test_words)
    test_words = test_words[:W_TO_A_TEST_REPEATS]

    for word in test_words:
        target = target_pool[word]
        drive = orthogonal_drive_pattern(
            cue_idx=word_to_idx[word], n_cues=len(words),
            n_neurons=N_LANG_INPUT, drive_max_pA=200.0,
            sparsity=SPARSITY)
        # Settle + drive
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        # Per-pool spike counts during drive
        pool_counts = {name: 0 for name in pool_indices}
        ext = cp.zeros(n_total, dtype=cp.float32)
        for _ in range(100):
            ext.fill(0)
            ext[lang_in_arr] = cp.asarray(drive, dtype=cp.float32)
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            for name, idx_arr in pool_indices.items():
                fired = bridge.cp_firing_states[idx_arr]
                pool_counts[name] += int(fired.sum())
        bridge.cp_external_input_current[:] = 0.0
        # Rank pools
        sorted_pools = sorted(pool_counts.items(),
                                 key=lambda x: x[1], reverse=True)
        top1_pool = sorted_pools[0][0]
        correct = (top1_pool == target)
        if correct: n_correct += 1
        n_total_tests += 1
        if verbose:
            print(f"    word={word} target={target} top1_pool="
                  f"{top1_pool} correct={correct} "
                  f"(counts: target={pool_counts.get(target, 0)}, "
                  f"top1={sorted_pools[0][1]})", flush=True)
    return n_correct / n_total_tests if n_total_tests > 0 else 0.0


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction H CANON DYNAMICS SMOKE ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu}); seed={SEED}",
          flush=True)
    print(f"  Tests if CANON concept-pool dynamics (weak_dynamics=False)"
          f" preserve Phase 1 W->A binding at smoke scale.",
          flush=True)

    t0 = time.time()
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)
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
    for w in ADJECTIVE_VOCAB:
        target_pool[w] = f"adjective_pool_{w.upper()}"

    # Build with CANON dynamics (weak_dynamics=False)
    bridge = build_concept_bridge(
        seed=SEED, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=False,  # <-- THE CANON DYNAMICS FLAG
        enable_positional_context=False, verbose=False,
    )
    print(f"  built CANON-dynamics bridge in {(time.time()-t0):.1f}s",
          flush=True)

    # Apply topographic bias + train via v16 recipe (reduced events for smoke)
    apply_concept_topographic_bias(
        bridge, n_lang_input=N_LANG_INPUT,
        topographic_factor=TOPOGRAPHIC_FACTOR,
        off_target_factor=OFF_TARGET_FACTOR,
        sparsity=SPARSITY, orthogonal_codes=True,
        n_words_for_orthogonal=n_words,
        word_to_idx=word_to_idx, verbose=False)

    print(f"\n  Smoke training ({N_TRAIN_EVENTS} events/word; "
          f"reduced from full 200)", flush=True)
    rng = np.random.default_rng(SEED)
    schedule = []
    for w in words:
        for _ in range(N_TRAIN_EVENTS):
            schedule.append(w)
    rng.shuffle(schedule)
    t_train = time.time()
    for ei, w in enumerate(schedule):
        train_word_to_pool(
            bridge, word=w, target_pool_region=target_pool[w],
            n_events=1, n_lang_input=N_LANG_INPUT,
            n_lang_output=N_LANG_INPUT,
            sparsity=SPARSITY, orthogonal_codes=True,
            n_words_for_orthogonal=n_words,
            word_to_idx=word_to_idx, verbose=False)
    print(f"  trained in {(time.time()-t_train)/60:.1f} min",
          flush=True)

    # Freeze plasticity for clean inference test
    for g in ("language_input_to_motor",
              "language_input_to_noun_pool",
              "language_input_to_verb_pool",
              "language_input_to_adjective_pool"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    # W->A test
    print(f"\n  Phase 1 W->A test ({W_TO_A_TEST_REPEATS} words)",
          flush=True)
    w_to_a_acc = w_to_a_test(bridge, words, word_to_idx, target_pool,
                                verbose=True)
    print(f"\n  Phase 1 W->A acc: {w_to_a_acc:.3f}", flush=True)

    print(f"\n  Wall: {(time.time()-t0)/60:.1f} min", flush=True)

    # Smoke verdict
    print(f"\n=== VERDICT ===", flush=True)
    if w_to_a_acc >= 0.50:
        verdict = "DIRECTION_H_SMOKE_PHASE1_PRESERVED"
        print(f"  Phase 1 PRESERVED at smoke scale ({w_to_a_acc:.3f}"
              f" >= 0.50); canon dynamics don't break trainability;"
              f" full GPU run JUSTIFIED.", flush=True)
    elif w_to_a_acc > 1.0 / 16.0:
        verdict = "DIRECTION_H_SMOKE_PHASE1_PARTIAL"
        print(f"  Phase 1 partially preserved ({w_to_a_acc:.3f}); "
              f"may be reduced training events; full GPU could"
              f" help.", flush=True)
    else:
        verdict = "DIRECTION_H_SMOKE_PHASE1_BROKEN"
        print(f"  Phase 1 BROKEN by canon dynamics ({w_to_a_acc:.3f}"
              f" at chance); reconfirms v14 canon-amplifies-bias"
              f"-collapse finding; Direction H closed; pivot to"
              f" Direction I or L.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "weak_dynamics": False, "canon_dynamics": True,
        "n_train_events": N_TRAIN_EVENTS,
        "w_to_a_test_repeats": W_TO_A_TEST_REPEATS,
        "phase1_w_to_a_acc": w_to_a_acc,
        "verdict": verdict,
        "wall_clock_minutes": (time.time()-t0)/60,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
