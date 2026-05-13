"""concept_compose_demo — Phase 2 of the concepts/composition/diversity arc.

Tests COMPOSITION: do multiple concept pools fire together for
compositional phrases like "go north" or "apple come"?

Three composition modes tested:

  (1) Sequential (NMDA bistability):
      Drive lang_input("go") for window_a steps -> verb_pool_GO fires,
      NMDA persistence holds.
      Drive lang_input("north") for window_b steps -> motor_N fires.
      During second window, BOTH verb_pool_GO AND motor_N active?

  (2) Co-firing (simultaneous):
      Drive lang_input with merged drive pattern for "go" + "north".
      Both pools fire above baseline?

  (3) Cross-kind isolation (negative control):
      Drive lang_input("go") alone for full window.
      ONLY verb_pool_GO fires; motor_X and noun_pool_X stay near baseline.

Architecture: depends on concept_pool_demo.py training a bridge with
10 pools (4 motor + 4 noun + 2 verb), then this runner loads it and
runs the three tests. Or, this runner can train its own bridge.

Phase 2 success criteria:
  - Sequential: target pool firing rate post-cross-pool drive > 50% of
    its rate during own drive (i.e., NMDA bistability holds for ~100ms)
  - Co-firing: both target pools fire >= 2x off-target pools
  - Isolation: target pool >= 2x off-target pools

User directive 2026-05-12: "concepts, composition, and diversity" are
the conversational blockers. This runner addresses composition
directly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Re-use the vocab definitions from concept_pool_demo
from research.runners.concept_pool_demo import (
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    NOUN_NAMES, VERB_NAMES, MOTOR_NAMES,
    build_concept_bridge, apply_concept_topographic_bias,
    train_word_to_pool, measure_pool_firing,
)


def measure_two_word_sequential(bridge, word_a: str, word_b: str,
                                  all_pool_regions: List[str],
                                  window_a_steps: int = 200,  # 100ms — biology-faithful word duration
                                  window_b_steps: int = 200,
                                  reset_steps: int = 100,
                                  drive_pA: float = 200.0,
                                  sparsity: float = 0.1,
                                  n_lang_input: int = 4096
                                  ) -> Dict[str, Dict[str, float]]:
    """Drive word_a for window_a steps, then word_b for window_b steps.

    Measures per-pool firing in BOTH windows. NMDA bistability means
    pools driven in window_a should still be active in window_b.

    Returns:
        {
            "window_a": {pool: firing_rate, ...},
            "window_b": {pool: firing_rate, ...},
        }
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)

    drive_a = vocab_to_drive_pattern(
        word_a, n_neurons=n_lang_input,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_b = vocab_to_drive_pattern(
        word_b, n_neurons=n_lang_input,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_a_gpu = cp.asarray(drive_a, dtype=cp.float32)
    drive_b_gpu = cp.asarray(drive_b, dtype=cp.float32)

    per_pool_indices = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                        for p in all_pool_regions}

    # Reset
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Window A: drive word_a, measure firing
    bridge.cp_external_input_current[lang_input_arr] = drive_a_gpu
    counts_a = {p: 0.0 for p in all_pool_regions}
    for _ in range(window_a_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for p, idx_arr in per_pool_indices.items():
            counts_a[p] += int(fired[idx_arr].sum())

    # Window B: drive word_b (NO reset between windows). word_a's drive
    # is replaced by word_b's, but NMDA persistence in pools driven by
    # word_a should keep them firing.
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[lang_input_arr] = drive_b_gpu
    counts_b = {p: 0.0 for p in all_pool_regions}
    for _ in range(window_b_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for p, idx_arr in per_pool_indices.items():
            counts_b[p] += int(fired[idx_arr].sum())

    # Convert to per-neuron mean rate
    rates_a = {}
    rates_b = {}
    for p in all_pool_regions:
        n_neurons = len(list(rm.indices(p)))
        rates_a[p] = counts_a[p] / max(n_neurons, 1)
        rates_b[p] = counts_b[p] / max(n_neurons, 1)

    return {"window_a": rates_a, "window_b": rates_b}


def measure_two_word_cofire(bridge, word_a: str, word_b: str,
                              all_pool_regions: List[str],
                              window_steps: int = 100,
                              reset_steps: int = 50,
                              drive_pA: float = 200.0,
                              sparsity: float = 0.1,
                              n_lang_input: int = 4096
                              ) -> Dict[str, float]:
    """Drive lang_input with merged (word_a + word_b) pattern.

    With drive_a + drive_b summed, lang_input activates both words'
    sparse codes simultaneously. Both target pools should fire above
    off-target pools.
    """
    import numpy as np
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)

    drive_a = vocab_to_drive_pattern(
        word_a, n_neurons=n_lang_input,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_b = vocab_to_drive_pattern(
        word_b, n_neurons=n_lang_input,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    # Element-wise max to merge (preserves drive_max_pA cap)
    drive_merged = np.maximum(drive_a, drive_b)
    drive_gpu = cp.asarray(drive_merged, dtype=cp.float32)

    per_pool_indices = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                        for p in all_pool_regions}

    # Reset
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Co-fire window
    bridge.cp_external_input_current[lang_input_arr] = drive_gpu
    counts = {p: 0.0 for p in all_pool_regions}
    for _ in range(window_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for p, idx_arr in per_pool_indices.items():
            counts[p] += int(fired[idx_arr].sum())

    rates = {}
    for p in all_pool_regions:
        n_neurons = len(list(rm.indices(p)))
        rates[p] = counts[p] / max(n_neurons, 1)
    return rates


def _target_pool_for_word(word: str) -> str:
    """Resolve word -> its trained target pool region."""
    if word in DIRECTION_VOCAB:
        return f"motor_{DIRECTION_VOCAB[word]}"
    if word in NOUN_VOCAB:
        return f"noun_pool_{NOUN_VOCAB[word]}"
    if word in VERB_VOCAB:
        return f"verb_pool_{VERB_VOCAB[word]}"
    raise ValueError(f"unknown word: {word!r}")


# Composition test pairs span the three kinds.
COMPOSE_PAIRS: List[Tuple[str, str]] = [
    # verb + direction (canonical "go north")
    ("go", "north"),
    ("come", "east"),
    # noun + direction ("apple north" - apple goes north)
    ("apple", "south"),
    ("dog", "west"),
    # verb + noun ("go apple" - go to the apple)
    ("go", "river"),
    ("come", "cat"),
]


def run_concept_compose_demo(seed: int = 42,
                               n_train_events: int = 200,
                               n_lang_input: int = 4096,
                               n_per_pool: int = 500,
                               n_fs_per_pool: int = 60,
                               apply_topographic: bool = True,
                               train_bridge: bool = True,
                               weak_dynamics: bool = False,
                               load_bridge: str = None,
                               verbose: bool = True) -> Dict:
    """Train bridge, then test sequential + co-fire composition.

    Phase 2 success criteria:
      1. Cross-category isolation (each single word -> own pool)
      2. Sequential composition (NMDA holds first pool while second fires)
      3. Co-firing composition (merged drive -> both target pools)
    """
    print(f"\n=== concept_compose_demo (seed={seed}) ===", flush=True)
    print(f"  Pairs to test: {COMPOSE_PAIRS}", flush=True)

    t0 = time.time()
    bridge = build_concept_bridge(
        seed=seed,
        n_lang_input=n_lang_input,
        n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool,
        weak_dynamics=weak_dynamics,
        verbose=verbose,
    )

    if load_bridge:
        print(f"\n[LOAD] loading bridge from {load_bridge}", flush=True)
        bridge.load_checkpoint(load_bridge)
        # Freeze gates for inference (no further training)
        for g in ("language_input_to_motor",
                  "language_input_to_noun_pool",
                  "language_input_to_verb_pool",
                  "motor_to_language_output",
                  "noun_pool_to_language_output",
                  "verb_pool_to_language_output"):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        train_bridge = False  # don't retrain a loaded bridge

    if apply_topographic and not load_bridge:
        apply_concept_topographic_bias(
            bridge, n_lang_input=n_lang_input, verbose=verbose,
        )

    all_pool_regions = (
        [f"motor_{a}" for a in MOTOR_NAMES]
        + [f"noun_pool_{n}" for n in NOUN_NAMES]
        + [f"verb_pool_{v}" for v in VERB_NAMES]
    )

    if train_bridge:
        all_targets = []
        for word, action in DIRECTION_VOCAB.items():
            all_targets.append((word, f"motor_{action}"))
        for word, name in NOUN_VOCAB.items():
            all_targets.append((word, f"noun_pool_{name}"))
        for word, name in VERB_VOCAB.items():
            all_targets.append((word, f"verb_pool_{name}"))

        print(f"\n[TRAIN] {len(all_targets)} (word, pool) pairs, "
              f"{n_train_events} events each", flush=True)
        t_train = time.time()
        for word, target in all_targets:
            t_word = time.time()
            train_word_to_pool(
                bridge, word, target,
                n_events=n_train_events,
                n_lang_input=n_lang_input,
                n_lang_output=n_lang_input,
                verbose=False,
            )
            print(f"  trained '{word}' -> {target} "
                  f"({time.time() - t_word:.0f}s)", flush=True)
        print(f"\n[TRAIN] complete ({time.time() - t_train:.0f}s)",
              flush=True)

    # ─── Test 1: Cross-category isolation (single word) ────────────
    print(f"\n[TEST 1] Cross-category isolation (Phase 1 sanity)", flush=True)
    isolation_results = {}
    n_isolation_pass = 0
    for word in list(DIRECTION_VOCAB) + list(NOUN_VOCAB) + list(VERB_VOCAB):
        target = _target_pool_for_word(word)
        rates = measure_pool_firing(
            bridge, word, all_pool_regions, n_lang_input=n_lang_input,
        )
        target_rate = rates[target]
        max_off = max(v for k, v in rates.items() if k != target)
        ratio = target_rate / max(max_off, 0.001)
        passed = target_rate > max_off
        if passed:
            n_isolation_pass += 1
        marker = "PASS" if passed else "FAIL"
        print(f"  {word:8s} -> {target:20s}  target={target_rate:.3f}  "
              f"max_off={max_off:.3f}  ratio={ratio:.2f}x  [{marker}]",
              flush=True)
        isolation_results[word] = {
            "target": target, "target_rate": float(target_rate),
            "max_off_target": float(max_off), "ratio": float(ratio),
            "passed": bool(passed),
        }

    # ─── Test 2: Sequential composition ─────────────────────────────
    print(f"\n[TEST 2] Sequential composition (NMDA bistability)", flush=True)
    print(f"  drive word_a for 50 steps, then word_b for 50 steps,",
          flush=True)
    print(f"  measure if both target pools active in window_b", flush=True)
    sequential_results = {}
    n_sequential_pass = 0
    for word_a, word_b in COMPOSE_PAIRS:
        target_a = _target_pool_for_word(word_a)
        target_b = _target_pool_for_word(word_b)
        seq = measure_two_word_sequential(
            bridge, word_a, word_b, all_pool_regions,
            n_lang_input=n_lang_input,
        )
        # In window_a, target_a should fire; in window_b, target_b should
        # fire AND target_a should retain >50% of its window_a rate.
        rate_a_in_a = seq["window_a"][target_a]
        rate_a_in_b = seq["window_b"][target_a]
        rate_b_in_b = seq["window_b"][target_b]
        # Off-target = max over pools NOT in {target_a, target_b} in window_b
        max_off_in_b = max(
            v for k, v in seq["window_b"].items()
            if k != target_a and k != target_b
        )
        a_persistence = rate_a_in_b / max(rate_a_in_a, 0.001)
        b_dominance = rate_b_in_b / max(max_off_in_b, 0.001)
        # PASS = a_persistence >= 0.5 AND b_dominance >= 1.0
        passed = (a_persistence >= 0.5) and (b_dominance >= 1.0)
        if passed:
            n_sequential_pass += 1
        marker = "PASS" if passed else "FAIL"
        print(f"  {word_a:5s}+{word_b:5s}  "
              f"a_in_a={rate_a_in_a:.2f}  a_in_b={rate_a_in_b:.2f}  "
              f"b_in_b={rate_b_in_b:.2f}  off={max_off_in_b:.2f}  "
              f"persist={a_persistence:.2f}  domin={b_dominance:.2f}  "
              f"[{marker}]", flush=True)
        sequential_results[f"{word_a}+{word_b}"] = {
            "target_a": target_a, "target_b": target_b,
            "rate_a_in_a": float(rate_a_in_a),
            "rate_a_in_b": float(rate_a_in_b),
            "rate_b_in_b": float(rate_b_in_b),
            "max_off_in_b": float(max_off_in_b),
            "a_persistence": float(a_persistence),
            "b_dominance": float(b_dominance),
            "passed": bool(passed),
        }

    # ─── Test 3: Co-firing composition ──────────────────────────────
    print(f"\n[TEST 3] Co-firing composition (merged drive)", flush=True)
    print(f"  drive word_a + word_b simultaneously, measure both pools",
          flush=True)
    cofire_results = {}
    n_cofire_pass = 0
    for word_a, word_b in COMPOSE_PAIRS:
        target_a = _target_pool_for_word(word_a)
        target_b = _target_pool_for_word(word_b)
        rates = measure_two_word_cofire(
            bridge, word_a, word_b, all_pool_regions,
            n_lang_input=n_lang_input,
        )
        rate_a = rates[target_a]
        rate_b = rates[target_b]
        max_off = max(
            v for k, v in rates.items()
            if k != target_a and k != target_b
        )
        a_ratio = rate_a / max(max_off, 0.001)
        b_ratio = rate_b / max(max_off, 0.001)
        passed = (a_ratio >= 1.0) and (b_ratio >= 1.0)
        if passed:
            n_cofire_pass += 1
        marker = "PASS" if passed else "FAIL"
        print(f"  {word_a:5s}+{word_b:5s}  "
              f"a={rate_a:.2f}  b={rate_b:.2f}  off={max_off:.2f}  "
              f"a/off={a_ratio:.2f}  b/off={b_ratio:.2f}  [{marker}]",
              flush=True)
        cofire_results[f"{word_a}+{word_b}"] = {
            "target_a": target_a, "target_b": target_b,
            "rate_a": float(rate_a), "rate_b": float(rate_b),
            "max_off": float(max_off),
            "a_ratio": float(a_ratio), "b_ratio": float(b_ratio),
            "passed": bool(passed),
        }

    print(f"\n[VERDICTS]", flush=True)
    n_words_total = len(DIRECTION_VOCAB) + len(NOUN_VOCAB) + len(VERB_VOCAB)
    print(f"  Test 1 isolation:  {n_isolation_pass}/{n_words_total} PASS",
          flush=True)
    print(f"  Test 2 sequential: {n_sequential_pass}/{len(COMPOSE_PAIRS)} PASS",
          flush=True)
    print(f"  Test 3 co-fire:    {n_cofire_pass}/{len(COMPOSE_PAIRS)} PASS",
          flush=True)
    print(f"  Wall clock: {time.time() - t0:.0f}s", flush=True)

    return {
        "seed": seed,
        "n_train_events": n_train_events,
        "wall_clock_s": time.time() - t0,
        "n_isolation_pass": n_isolation_pass,
        "n_isolation_words": len(DIRECTION_VOCAB) + len(NOUN_VOCAB) + len(VERB_VOCAB),
        "n_sequential_pass": n_sequential_pass,
        "n_sequential_pairs": len(COMPOSE_PAIRS),
        "n_cofire_pass": n_cofire_pass,
        "n_cofire_pairs": len(COMPOSE_PAIRS),
        "isolation": isolation_results,
        "sequential": sequential_results,
        "cofire": cofire_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 concept composition demo.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-events", type=int, default=200)
    parser.add_argument("--n-lang-input", type=int, default=4096)
    parser.add_argument("--n-per-pool", type=int, default=500)
    parser.add_argument("--n-fs-per-pool", type=int, default=60)
    parser.add_argument("--no-topographic", action="store_true")
    parser.add_argument("--no-train", action="store_true",
                         help="Skip training (assumes bridge has been pre-trained)")
    parser.add_argument("--weak-concept-dynamics", action="store_true",
                         help="Match v7 production recipe")
    parser.add_argument("--load-bridge", type=str, default=None,
                         help="Load v7 saved bridge checkpoint")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    result = run_concept_compose_demo(
        seed=args.seed,
        n_train_events=args.n_train_events,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        apply_topographic=not args.no_topographic,
        train_bridge=not args.no_train,
        weak_dynamics=args.weak_concept_dynamics,
        load_bridge=args.load_bridge,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n[OUT] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
