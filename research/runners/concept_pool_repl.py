"""concept_pool_repl — interactive REPL for trained concept pool bridge.

Standalone alternative to chat_repl integration. Loads a saved bridge
(via --load-bridge or implicit training) and lets the user type:

  - Any word -> reports per-pool firing rates, identifies top-1
  - "speak <pool>" -> drives the pool, reports the "spoken" word from
                       language_output cosine ranking
  - "compose <w1> <w2>" -> sequential composition test
  - "quit" / "exit" -> exit

Demonstrates the three Phase 1/2/3 capabilities interactively:
  1. Hear: type word -> watch the matching pool light up
  2. Speak: name a pool -> watch the network produce a word
  3. Compose: two-word phrase -> watch both pools light up

Critical for autonomy: the user can verify the architecture works
end-to-end without watching 20-minute training runs.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

from research.runners.concept_pool_demo import (
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    NOUN_NAMES, VERB_NAMES, MOTOR_NAMES,
    build_concept_bridge, apply_concept_topographic_bias,
    train_word_to_pool, measure_pool_firing,
)


_ALL_WORDS = list(DIRECTION_VOCAB) + list(NOUN_VOCAB) + list(VERB_VOCAB)
_ALL_POOLS = (
    [f"motor_{a}" for a in MOTOR_NAMES]
    + [f"noun_pool_{n}" for n in NOUN_NAMES]
    + [f"verb_pool_{v}" for v in VERB_NAMES]
)


def _target_word_for_pool(pool: str) -> str:
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
    return "?"


def cmd_hear(bridge, word: str, n_lang_input: int):
    """Type a word; report per-pool firing rates."""
    rates = measure_pool_firing(bridge, word, _ALL_POOLS, n_lang_input=n_lang_input)
    sorted_rates = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n[HEAR '{word}'] per-pool firing rates (top-5):")
    for pool, rate in sorted_rates[:5]:
        target_word = _target_word_for_pool(pool)
        marker = "*" if target_word == word else " "
        print(f"  {marker} {pool:22s}: {rate:.3f}  (trained for '{target_word}')")
    top1_pool, top1_rate = sorted_rates[0]
    print(f"\n  Network 'hears': {_target_word_for_pool(top1_pool)} via {top1_pool}",
          flush=True)


def cmd_speak(bridge, pool: str, n_lang_input: int, stim_steps: int = 100):
    """Drive a pool; report the word the network 'speaks'."""
    from research.runners.concept_speak_demo import (
        drive_pool_and_read_lang_output, _cosine,
    )
    from sim.text_embeddings import vocab_to_drive_pattern

    if pool not in _ALL_POOLS:
        print(f"  unknown pool: {pool!r}. Valid: {_ALL_POOLS}")
        return

    rm = bridge.region_manager
    n_lang_out = len(list(rm.indices("language_output")))
    spike_pattern = drive_pool_and_read_lang_output(
        bridge, pool, stim_steps=stim_steps, n_lang_output=n_lang_out,
    )
    # Rank by cosine
    scores = {}
    for w in _ALL_WORDS:
        pattern = vocab_to_drive_pattern(w, n_neurons=n_lang_out, sparsity=0.1)
        scores[w] = _cosine(spike_pattern, pattern)
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    target_word = _target_word_for_pool(pool)
    print(f"\n[SPEAK {pool}] cosine-ranked words (top-5):")
    for word, score in sorted_scores[:5]:
        marker = "*" if word == target_word else " "
        print(f"  {marker} {word:10s}: {score:.3f}")
    top1, top_score = sorted_scores[0]
    print(f"\n  Network 'says': '{top1}' (cosine {top_score:.3f}, "
          f"trained for '{target_word}')", flush=True)


def cmd_compose(bridge, word_a: str, word_b: str, n_lang_input: int):
    """Two-word sequential composition test."""
    from research.runners.concept_compose_demo import measure_two_word_sequential

    seq = measure_two_word_sequential(
        bridge, word_a, word_b, _ALL_POOLS, n_lang_input=n_lang_input,
    )
    print(f"\n[COMPOSE '{word_a}' then '{word_b}']")
    print(f"  Window A (driving '{word_a}'):")
    for p, r in sorted(seq["window_a"].items(), key=lambda kv: kv[1], reverse=True)[:3]:
        print(f"    {p:22s}: {r:.3f}  ({_target_word_for_pool(p)})")
    print(f"  Window B (driving '{word_b}', NMDA persistence expected):")
    for p, r in sorted(seq["window_b"].items(), key=lambda kv: kv[1], reverse=True)[:3]:
        print(f"    {p:22s}: {r:.3f}  ({_target_word_for_pool(p)})", flush=True)


def cmd_help():
    print("\nCommands:")
    print("  <word>              Hear: which pool lights up for this word?")
    print("  speak <pool>        Speak: drive pool, report 'spoken' word")
    print("  compose <w1> <w2>   Compose: sequential two-word firing")
    print("  list                List all trained words + pools")
    print("  help / ?            Show this help")
    print("  quit / exit         Exit")
    print()
    print(f"Trained words: {', '.join(_ALL_WORDS)}")
    print(f"All pools: {', '.join(_ALL_POOLS)}", flush=True)


def cmd_list():
    print(f"\n[WORDS] trained vocabulary:")
    print(f"  Directions ({len(DIRECTION_VOCAB)}): "
          f"{', '.join(DIRECTION_VOCAB.keys())}")
    print(f"  Nouns ({len(NOUN_VOCAB)}): "
          f"{', '.join(NOUN_VOCAB.keys())}")
    print(f"  Verbs ({len(VERB_VOCAB)}): "
          f"{', '.join(VERB_VOCAB.keys())}")
    print(f"\n[POOLS] {len(_ALL_POOLS)} distinct output categories:")
    for p in _ALL_POOLS:
        print(f"  {p:22s} (trained for '{_target_word_for_pool(p)}')",
              flush=True)


def run_repl(bridge, n_lang_input: int = 4096):
    """Interactive command loop."""
    print("\n" + "=" * 60)
    print("CONCEPT POOL REPL - 12 distinct output categories")
    print("Type 'help' for commands. Type a word to hear it.")
    print("=" * 60, flush=True)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit"):
            print("Goodbye.")
            return
        if cmd in ("help", "?"):
            cmd_help()
            continue
        if cmd == "list":
            cmd_list()
            continue
        if cmd == "speak":
            if len(parts) != 2:
                print("Usage: speak <pool>")
                continue
            cmd_speak(bridge, parts[1], n_lang_input)
            continue
        if cmd == "compose":
            if len(parts) != 3:
                print("Usage: compose <word1> <word2>")
                continue
            if parts[1] not in _ALL_WORDS or parts[2] not in _ALL_WORDS:
                print(f"Words must be in trained vocab: {_ALL_WORDS}")
                continue
            cmd_compose(bridge, parts[1], parts[2], n_lang_input)
            continue
        # Default: treat as a word to hear
        if cmd in _ALL_WORDS:
            cmd_hear(bridge, cmd, n_lang_input)
        else:
            print(f"  unknown word/command: {cmd!r}. Type 'help' for usage.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive REPL for concept pool bridge.")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to .simstate.h5 from --save-bridge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-lang-input", type=int, default=4096)
    parser.add_argument("--n-per-pool", type=int, default=500)
    parser.add_argument("--n-fs-per-pool", type=int, default=60)
    args = parser.parse_args()

    print(f"[REPL] building bridge skeleton (seed={args.seed})", flush=True)
    bridge = build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        verbose=False,
    )
    print(f"[REPL] loading weights from {args.checkpoint}", flush=True)
    bridge.load_checkpoint(args.checkpoint)
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

    run_repl(bridge, n_lang_input=args.n_lang_input)


if __name__ == "__main__":
    main()
