"""P5 concept-recognition demo V2 — cosine-based recognition.

V1 used wernicke pool spike counts as the readout. Multi-seed
results showed this is unreliable (mean ~25% trial accuracy across
seeds 43, 44, 100, 101, 102) despite iter W passing 6/6 multi-seed
at the COSINE-on-semantic_cortex test.

V2 uses the same readout that iter W validated: tag the
semantic_cortex ensemble for each concept after training, then
compare test-time semantic_cortex firing pattern via index
cosine.

This is the production-ready P5 concept recognition demo.

Usage:
    python -m research.runners.p5_concept_demo_v2 --seed 42 \\
        --test-words "apple,river,apple,river"
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np


def measure_region_spikes(bridge, region_name: str, n_steps: int = 100):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager
    indices = list(rm.indices(region_name))
    arr = cp.asarray(indices, dtype=cp.int64)
    counts = cp.zeros(len(indices), dtype=cp.float32)
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[arr]
        counts += fired.astype(cp.float32)
    return to_host(counts)


def index_cosine(a_idx, b_idx):
    """Cosine similarity between two index sets."""
    if len(a_idx) == 0 or len(b_idx) == 0:
        return 0.0
    s_a = set(int(x) for x in a_idx)
    s_b = set(int(x) for x in b_idx)
    overlap = len(s_a & s_b)
    return float(overlap / (np.sqrt(len(s_a)) * np.sqrt(len(s_b))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train-events", type=int, default=400)
    ap.add_argument("--n-replay-cycles", type=int, default=40)
    ap.add_argument("--test-words", type=str,
                    default="apple,river,apple,river")
    args = ap.parse_args()

    # Reuse the build_iterW_bridge from V1
    from research.runners.p5_concept_demo import build_iterW_bridge
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    bridge, concept_arrays = build_iterW_bridge(
        seed=args.seed,
        n_train_events=args.n_train_events,
        n_replay_cycles=args.n_replay_cycles,
    )

    rm = bridge.region_manager
    sem_indices = list(rm.indices("semantic_cortex"))

    print("\nTagging semantic_cortex ensembles for trained concepts...")
    sem_tags = {}
    for concept in ["apple", "river"]:
        tag_name = f"{concept}_sem"
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.start_engram_recording(tag_name)
        bridge.cp_external_input_current[concept_arrays[concept]] = 200.0
        for _ in range(100):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[:] = 0.0
        stats = bridge.commit_engram_tag(
            tag_name, top_k=50, region_filter=["semantic_cortex"],
        )
        sem_tags[concept] = to_host(
            bridge.get_engram_tag_indices(tag_name)
        )
        print(f"  {concept}: tagged {stats['n_tagged']} sem neurons")

    test_words = [w.strip() for w in args.test_words.split(",") if w.strip()]
    print("\n" + "=" * 60)
    print("P5 concept-recognition demo V2 (cosine-based)")
    print("=" * 60)
    n_correct = 0
    for word in test_words:
        print(f"\nUser drives lang_input('{word}')...")
        if word not in concept_arrays:
            print(f"  '{word}' not in vocabulary")
            continue
        # Drive, measure semantic_cortex firing
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[concept_arrays[word]] = 200.0
        spike_counts = measure_region_spikes(
            bridge, "semantic_cortex", n_steps=100,
        )
        bridge.cp_external_input_current[:] = 0.0
        # Local indices to global
        firing_local = np.where(spike_counts > 0)[0]
        firing_global = np.array(
            [sem_indices[i] for i in firing_local
             if i < len(sem_indices)],
            dtype=np.int64,
        )

        # Cosine vs each tag
        cosines = {}
        for concept_name, tag_idx in sem_tags.items():
            cos = index_cosine(firing_global, tag_idx)
            cosines[concept_name] = cos
        print(f"  semantic_cortex firing: {len(firing_global)} neurons")
        for c, v in cosines.items():
            print(f"  cosine vs '{c}' tag: {v:.3f}")

        recognized = max(cosines, key=cosines.get)
        recognized_cos = cosines[recognized]
        other_cos = max(v for k, v in cosines.items() if k != recognized)
        confidence = recognized_cos / max(recognized_cos + other_cos, 0.001)
        print(f"  -> recognized: '{recognized}' "
              f"(confidence {confidence:.1%}, margin "
              f"{recognized_cos - other_cos:+.3f})")
        correct = (recognized == word)
        print(f"  expected: '{word}', got: '{recognized}' — "
              f"{'CORRECT' if correct else 'WRONG'}")
        if correct:
            n_correct += 1
    print(f"\n{'=' * 60}")
    print(f"Accuracy: {n_correct}/{len(test_words)} "
          f"({100*n_correct/max(len(test_words),1):.0f}%)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
