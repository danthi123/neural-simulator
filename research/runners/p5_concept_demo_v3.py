"""P5 concept-recognition demo V3 — multi-trial averaging.

V2 with cosine-on-semantic_cortex gave ~48% single-trial
accuracy (chance) despite iter W's 6/6 statistical PASS.
V3 averages N trials per word to overcome single-trial noise.

Hypothesis: with margin +0.05 and single-trial noise ±0.03,
averaging 5 trials reduces effective noise to ±0.013 — clearly
below margin → high accuracy.

Usage:
    python -m research.runners.p5_concept_demo_v3 --seed 42 \\
        --trials-per-word 5 --test-words "apple,river,apple,river"
"""
from __future__ import annotations

import argparse

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
    if len(a_idx) == 0 or len(b_idx) == 0:
        return 0.0
    s_a = set(int(x) for x in a_idx)
    s_b = set(int(x) for x in b_idx)
    overlap = len(s_a & s_b)
    return float(overlap / (np.sqrt(len(s_a)) * np.sqrt(len(s_b))))


def single_trial_cosines(bridge, drive_arr, sem_indices, sem_tags):
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[drive_arr] = 200.0
    spike_counts = measure_region_spikes(
        bridge, "semantic_cortex", n_steps=100,
    )
    bridge.cp_external_input_current[:] = 0.0
    firing_local = np.where(spike_counts > 0)[0]
    firing_global = np.array(
        [sem_indices[i] for i in firing_local
         if i < len(sem_indices)],
        dtype=np.int64,
    )
    return {
        concept: index_cosine(firing_global, tag_idx)
        for concept, tag_idx in sem_tags.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train-events", type=int, default=400)
    ap.add_argument("--n-replay-cycles", type=int, default=40)
    ap.add_argument("--test-words", type=str,
                    default="apple,river,apple,river")
    ap.add_argument("--trials-per-word", type=int, default=5,
                    help="N trials averaged per recognition")
    ap.add_argument("--vote-mode", choices=["majority", "mean_cosine"],
                    default="mean_cosine",
                    help="majority: each trial votes; mean_cosine: "
                         "average raw cosines (default)")
    args = ap.parse_args()

    from research.runners.p5_concept_demo import build_iterW_bridge
    from sim.backend import to_host

    bridge, concept_arrays = build_iterW_bridge(
        seed=args.seed,
        n_train_events=args.n_train_events,
        n_replay_cycles=args.n_replay_cycles,
    )

    rm = bridge.region_manager
    sem_indices = list(rm.indices("semantic_cortex"))

    print("\nTagging semantic_cortex ensembles...")
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
        bridge.commit_engram_tag(
            tag_name, top_k=50, region_filter=["semantic_cortex"],
        )
        sem_tags[concept] = to_host(
            bridge.get_engram_tag_indices(tag_name)
        )

    test_words = [w.strip() for w in args.test_words.split(",") if w.strip()]
    print(f"\n{'=' * 60}")
    print(f"P5 V3 demo: {args.trials_per_word}-trial averaging "
          f"({args.vote_mode})")
    print(f"{'=' * 60}")
    n_correct = 0
    for word in test_words:
        print(f"\nUser drives lang_input('{word}'), {args.trials_per_word} trials:")
        if word not in concept_arrays:
            print(f"  '{word}' not in vocabulary")
            continue
        trial_cosines = []
        for t in range(args.trials_per_word):
            cos = single_trial_cosines(
                bridge, concept_arrays[word], sem_indices, sem_tags,
            )
            trial_cosines.append(cos)
            print(f"  trial {t+1}: apple={cos['apple']:.3f}, "
                  f"river={cos['river']:.3f}")

        if args.vote_mode == "majority":
            votes = {c: 0 for c in sem_tags}
            for tc in trial_cosines:
                winner = max(tc, key=tc.get)
                votes[winner] += 1
            print(f"  votes: {votes}")
            recognized = max(votes, key=votes.get)
            confidence = votes[recognized] / args.trials_per_word
        else:  # mean_cosine
            mean_cos = {c: 0.0 for c in sem_tags}
            for tc in trial_cosines:
                for c, v in tc.items():
                    mean_cos[c] += v
            for c in mean_cos:
                mean_cos[c] /= args.trials_per_word
            print(f"  mean cosines: " + ", ".join(
                f"{c}={v:.3f}" for c, v in mean_cos.items()))
            recognized = max(mean_cos, key=mean_cos.get)
            recognized_v = mean_cos[recognized]
            other_v = max(v for k, v in mean_cos.items() if k != recognized)
            confidence = recognized_v / max(recognized_v + other_v, 0.001)

        correct = (recognized == word)
        print(f"  -> recognized: '{recognized}' "
              f"(confidence {confidence:.1%})")
        print(f"  expected: '{word}', got: '{recognized}' — "
              f"{'CORRECT' if correct else 'WRONG'}")
        if correct:
            n_correct += 1
    print(f"\n{'=' * 60}")
    print(f"Accuracy: {n_correct}/{len(test_words)} "
          f"({100*n_correct/max(len(test_words),1):.0f}%)")
    print(f"({args.trials_per_word}-trial averaging, "
          f"{args.vote_mode} mode)")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
