"""concept_speak_demo — Phase 3 of the concepts/composition/diversity arc.

Tests A->W readout for concept pools: drive a pool, decode the word
the network "speaks" via language_output.

Mirrors chat_speak_synonym_demo.py but for 10 distinct pools instead
of 4 motor pools.

For each of the 10 concept pools (4 motor + 4 noun + 2 verb):
  1. Drive the pool with strong teacher current
  2. Step bridge for stim window
  3. Record language_output firing pattern
  4. Compare via cosine similarity to each word's drive pattern
  5. Top-1 = sim's "spoken" word for that pool

PASS criteria: For each pool, the top-1 spoken word should be the
TRAINED target word (motor_N -> "north", noun_APPLE -> "apple", etc.).

Together with Phase 1 (cross-category isolation) and Phase 2
(composition), this gives the full reading + speaking + composing
loop for a 10-concept vocabulary.

Note: depends on the same training as concept_pool_demo. After Phase 1
completes (and the bridge is saved via lineage or checkpoint), Phase 3
can reuse the trained bridge for fast iteration on A->W readout
mechanics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from research.runners.concept_pool_demo import (
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    NOUN_NAMES, VERB_NAMES, MOTOR_NAMES,
    build_concept_bridge, apply_concept_topographic_bias,
    train_word_to_pool,
)


def _all_words():
    """Return all trained words in canonical order."""
    return list(DIRECTION_VOCAB) + list(NOUN_VOCAB) + list(VERB_VOCAB)


def _target_pool_for_word(word: str) -> str:
    if word in DIRECTION_VOCAB:
        return f"motor_{DIRECTION_VOCAB[word]}"
    if word in NOUN_VOCAB:
        return f"noun_pool_{NOUN_VOCAB[word]}"
    if word in VERB_VOCAB:
        return f"verb_pool_{VERB_VOCAB[word]}"
    raise ValueError(f"unknown word: {word!r}")


def _target_word_for_pool(pool: str) -> str:
    """Inverse: pool -> trained word."""
    if pool.startswith("motor_"):
        action = pool[len("motor_"):]
        for w, a in DIRECTION_VOCAB.items():
            if a == action:
                return w
    elif pool.startswith("noun_pool_"):
        name = pool[len("noun_pool_"):]
        for w, n in NOUN_VOCAB.items():
            if n == name:
                return w
    elif pool.startswith("verb_pool_"):
        name = pool[len("verb_pool_"):]
        for w, n in VERB_VOCAB.items():
            if n == name:
                return w
    raise ValueError(f"unknown pool: {pool!r}")


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def drive_pool_and_read_lang_output(bridge, pool_region: str,
                                      stim_steps: int = 100,
                                      reset_steps: int = 50,
                                      teacher_pA: float = 1500.0,
                                      n_lang_output: int = 4096):
    """Drive a single pool, accumulate language_output firing across stim window.

    Returns:
        spike_pattern: 1D array of per-neuron spike counts in lang_output
    """
    import numpy as np
    from sim.backend import get_backend
    cp, _ = get_backend()

    rm = bridge.region_manager
    pool_idx = list(rm.indices(pool_region))
    pool_arr = cp.asarray(pool_idx, dtype=cp.int64)
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    # Reset
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Drive pool only (no lang_input drive — we want pool -> lang_output via STDP-trained pathway)
    bridge.cp_external_input_current[pool_arr] = float(teacher_pA)

    # Accumulate per-neuron spike counts in language_output
    spike_pattern = cp.zeros(n_lang_output, dtype=cp.float32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[lang_out_arr]
        spike_pattern += fired.astype(cp.float32)

    try:
        return cp.asnumpy(spike_pattern)
    except Exception:
        return np.asarray(spike_pattern)


def evaluate_A_to_W(bridge, n_lang_input: int = 4096,
                     stim_steps: int = 100,
                     verbose: bool = True):
    """For each pool, drive it and rank the trained words by cosine."""
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    n_lang_out = len(list(rm.indices("language_output")))

    # Pre-compute word reference patterns
    all_words = _all_words()
    word_patterns = {
        w: vocab_to_drive_pattern(w, n_neurons=n_lang_out, sparsity=0.1)
        for w in all_words
    }

    # Iterate over all 10 pools
    all_pools = (
        [f"motor_{a}" for a in MOTOR_NAMES]
        + [f"noun_pool_{n}" for n in NOUN_NAMES]
        + [f"verb_pool_{v}" for v in VERB_NAMES]
    )

    results = {}
    n_pass = 0

    for pool in all_pools:
        target_word = _target_word_for_pool(pool)
        spike_pattern = drive_pool_and_read_lang_output(
            bridge, pool,
            stim_steps=stim_steps,
            n_lang_output=n_lang_out,
        )
        # Cosine similarity to each candidate word
        scores = {
            w: _cosine(spike_pattern, word_patterns[w])
            for w in all_words
        }
        # Top-1 by cosine
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top1_word = ranked[0][0]
        top1_score = ranked[0][1]
        target_score = scores[target_word]
        passed = (top1_word == target_word)
        if passed:
            n_pass += 1

        marker = "PASS" if passed else "FAIL"
        if verbose:
            top3 = ", ".join(f"{w}={s:.2f}" for w, s in ranked[:3])
            print(f"  drive {pool:20s} -> top-1='{top1_word}' "
                  f"(cos={top1_score:.2f}, target='{target_word}' "
                  f"cos={target_score:.2f})  top3=[{top3}]  [{marker}]",
                  flush=True)

        results[pool] = {
            "target_word": target_word,
            "top1_word": top1_word,
            "top1_score": float(top1_score),
            "target_score": float(target_score),
            "all_scores": {w: float(s) for w, s in scores.items()},
            "passed": bool(passed),
        }

    return {"n_pass": n_pass, "n_total": len(all_pools), "per_pool": results}


def run_concept_speak_demo(seed: int = 42,
                             n_train_events: int = 200,
                             n_lang_input: int = 4096,
                             n_per_pool: int = 500,
                             n_fs_per_pool: int = 60,
                             apply_topographic: bool = True,
                             weak_dynamics: bool = False,
                             nmda_tau_decay_ms: float = 100.0,
                             load_bridge: str = None,
                             verbose: bool = True):
    """Train + evaluate A->W readout for all 12 pools (motor + noun + verb).

    If load_bridge is given, skip training and load checkpoint (e.g.,
    from concept_pool_demo --save-bridge). Fast iteration on A->W
    mechanics without retraining.
    """
    print(f"\n=== concept_speak_demo (seed={seed}) ===", flush=True)
    print(f"  Tests A->W: drive all pools, decode 'spoken' word", flush=True)

    t0 = time.time()
    bridge = build_concept_bridge(
        seed=seed,
        n_lang_input=n_lang_input,
        n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool,
        weak_dynamics=weak_dynamics,
        nmda_tau_decay_ms=nmda_tau_decay_ms,
        verbose=verbose,
    )

    if load_bridge:
        print(f"\n[LOAD] loading checkpoint from {load_bridge}", flush=True)
        bridge.load_checkpoint(load_bridge)
        # Freeze gates for inference
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
        train_sec = 0.0
    else:
        if apply_topographic:
            apply_concept_topographic_bias(
                bridge, n_lang_input=n_lang_input, verbose=verbose,
            )

        # Train all (word, pool) pairs
        all_targets = []
        for word, action in DIRECTION_VOCAB.items():
            all_targets.append((word, f"motor_{action}"))
        for word, name in NOUN_VOCAB.items():
            all_targets.append((word, f"noun_pool_{name}"))
        for word, name in VERB_VOCAB.items():
            all_targets.append((word, f"verb_pool_{name}"))

        print(f"\n[TRAIN] {len(all_targets)} (word, pool) pairs",
              flush=True)
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
        train_sec = time.time() - t_train
        print(f"[TRAIN] complete ({train_sec:.0f}s)", flush=True)

    # Phase 3: A->W readout
    print(f"\n[EVAL] A->W readout for all 10 pools", flush=True)
    aw_result = evaluate_A_to_W(bridge, n_lang_input=n_lang_input,
                                  verbose=verbose)

    print(f"\n[VERDICT] {aw_result['n_pass']}/{aw_result['n_total']} pools "
          f"speak the trained word", flush=True)
    print(f"  Wall clock: {time.time() - t0:.0f}s", flush=True)

    return {
        "seed": seed,
        "n_train_events": n_train_events,
        "wall_clock_s": time.time() - t0,
        "a_to_w_pass": aw_result["n_pass"],
        "a_to_w_total": aw_result["n_total"],
        "per_pool": aw_result["per_pool"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 concept pool A->W readout.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-events", type=int, default=200)
    parser.add_argument("--n-lang-input", type=int, default=4096)
    parser.add_argument("--n-per-pool", type=int, default=500)
    parser.add_argument("--n-fs-per-pool", type=int, default=60)
    parser.add_argument("--no-topographic", action="store_true")
    parser.add_argument("--weak-concept-dynamics", action="store_true",
                         help="Match v7 production recipe (concept pools "
                         "use weak dynamics 0.05/0.3/0.8)")
    parser.add_argument("--nmda-tau-decay-ms", type=float, default=100.0,
                         help="NMDA tau (ms); for loaded bridges must "
                         "match the bridge's training tau")
    parser.add_argument("--load-bridge", type=str, default=None,
                         help="Load checkpoint instead of training "
                         "(use with v7 saved bridge from concept_pool_demo)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    result = run_concept_speak_demo(
        seed=args.seed,
        n_train_events=args.n_train_events,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        apply_topographic=not args.no_topographic,
        weak_dynamics=args.weak_concept_dynamics,
        nmda_tau_decay_ms=args.nmda_tau_decay_ms,
        load_bridge=args.load_bridge,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n[OUT] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
