"""Chat-with-online-learning demo (Track 3, 2026-05-09).

Demonstrates online vocabulary learning via embodied-Hebbian co-firing.
The chat REPL primitive ``learn_word_pairing`` (chat_repl.py) lets the
already-trained bridge bind a NEW word to an existing motor pool at
runtime. This runner exercises that primitive end-to-end and emits
machine-readable stats so the launcher / aggregator can graph it.

Pipeline:
  1. Train Tier 1 4-word bridge (same as chat_demo)
  2. Phase A: test 4 primary words (north/east/south/west) — baseline accuracy
  3. Phase B: call learn_word_pairing for 2 NEW words ("ahead"->N, "back"->S)
  4. Phase C: test the 2 new words — did the binding take?
  5. Phase D: re-test the 4 primary words — catastrophic-forgetting check

Output JSON shape (extends chat_demo's schema for chat_demo_aggregate):
  {
    "seed": 42,
    "accuracy": <Phase D primary accuracy>,
    "correct": <Phase D correct turns>,
    "total":   <Phase D total turns>,
    "per_word_accuracy": {north: 1.0, ...},
    "demo_kind": "chat_learn_demo",
    "learn_results": [
      {"word": "ahead", "target": "N", "predicted": "N", "bound_ok": True,
       "confidence": 4.2, "n_events": 50}, ...
    ],
    "primary_baseline_accuracy": <Phase A>,
    "primary_post_learn_accuracy": <Phase D>,
    "primary_retention_ratio": <D / A>,
    "learn_binding_rate": <fraction of new words correctly bound>,
  }

Usage:
    python -m research.runners.chat_learn_demo \\
        --seed 42 --train-events 200 --learn-events 50 \\
        --out-stats research/findings/raw/g11_bg/chat_learn_seed42.json

The exit code mirrors a "GO" verdict per autonomous-runs principle #6:
0 if learn_binding_rate >= 50% AND primary_retention_ratio >= 80%, else 1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# Reuse chat_demo's primary vocab + utilities
from research.runners.chat_demo import (
    DIRECTIONS, DIRECTION_TO_ACTION, ACTION_TO_DIRECTION,
    train_chat_bridge, chat_turn,
)
# Import the new --learn primitive
from research.runners.chat_repl import learn_word_pairing


# Default new words to learn — chosen so they don't collide with any
# existing synonym (n/e/s/w/up/right/down/left/etc).
DEFAULT_NEW_WORDS = [
    ("ahead", "N"),
    ("back",  "S"),
]


def evaluate_primaries(bridge, n_rounds: int = 2, verbose: bool = True) -> dict:
    """Run primary-word accuracy test across N rounds.

    Returns aggregate accuracy + per-word accuracy + raw turn list.
    """
    correct_total = 0
    total_turns = 0
    correct_per_word = {w: 0 for w in DIRECTIONS}
    total_per_word = {w: 0 for w in DIRECTIONS}
    turns = []
    for round_n in range(1, n_rounds + 1):
        for word in DIRECTIONS:
            result = chat_turn(bridge, word)
            total_turns += 1
            total_per_word[word] += 1
            if result["correct"]:
                correct_total += 1
                correct_per_word[word] += 1
            turns.append({"round": round_n, **result})
            if verbose:
                marker = "[OK]" if result["correct"] else "[X] "
                d = result["delta_counts"]
                print(f"  {marker} '{word}' -> {result['predicted_direction']} "
                      f"(delta N{d['N']:+d} E{d['E']:+d} S{d['S']:+d} "
                      f"W{d['W']:+d}, x{result['confidence_ratio']:.1f})",
                      flush=True)
    return {
        "accuracy": correct_total / total_turns if total_turns else 0.0,
        "correct": correct_total,
        "total":   total_turns,
        "per_word_accuracy": {
            w: (correct_per_word[w] / total_per_word[w]
                if total_per_word[w] else 0.0)
            for w in DIRECTIONS
        },
        "turns": turns,
    }


def evaluate_new_word(bridge, word: str, target_action: str,
                      verbose: bool = True) -> dict:
    """Test whether a learned (word -> motor) binding holds at inference time.

    Mirrors chat_turn's argmax-of-delta logic but reports against the
    expected target_action rather than the standard primary-vocab match.
    """
    result = chat_turn(bridge, word)
    pred = result["predicted_action"]
    bound_ok = (pred == target_action)
    if verbose:
        marker = "[OK]" if bound_ok else "[X] "
        d = result["delta_counts"]
        print(f"  {marker} learn-test '{word}' -> motor_{pred} "
              f"(target motor_{target_action}) "
              f"(delta N{d['N']:+d} E{d['E']:+d} S{d['S']:+d} "
              f"W{d['W']:+d}, x{result['confidence_ratio']:.1f})",
              flush=True)
    return {
        "word": word,
        "target": target_action,
        "predicted": pred,
        "bound_ok": bound_ok,
        "confidence": result["confidence_ratio"],
        "delta_counts": result["delta_counts"],
    }


def run_chat_learn_demo(seed: int = 42, n_train_events: int = 200,
                         learn_n_events: int = 50,
                         new_words: list = None,
                         verbose: bool = True) -> dict:
    """Full chat-with-learn demo. Returns dict with all metrics."""
    if new_words is None:
        new_words = list(DEFAULT_NEW_WORDS)

    print(f"\n=== chat_learn_demo (seed={seed}) ===", flush=True)
    print(f"  primaries:  {DIRECTIONS}", flush=True)
    print(f"  new words:  {new_words}", flush=True)
    print(f"  train_events={n_train_events}, "
          f"learn_events={learn_n_events}\n", flush=True)

    t0 = time.time()
    bridge = train_chat_bridge(seed=seed, n_events_per_word=n_train_events,
                                verbose=verbose)
    train_sec = time.time() - t0

    # ─── Phase A: baseline primary accuracy ───
    print(f"\n[PHASE A] baseline primary accuracy", flush=True)
    phase_a = evaluate_primaries(bridge, n_rounds=2, verbose=verbose)
    print(f"  baseline primary acc: {phase_a['accuracy']:.1%} "
          f"({phase_a['correct']}/{phase_a['total']})", flush=True)

    # ─── Phase B: learn N new words ───
    print(f"\n[PHASE B] learning {len(new_words)} new words "
          f"({learn_n_events} events each)", flush=True)
    learn_log = []
    for new_word, target in new_words:
        learn_word_pairing(bridge, new_word, target,
                            n_events=learn_n_events, verbose=verbose)
        learn_log.append({"word": new_word, "target": target,
                          "n_events": learn_n_events})

    # ─── Phase C: test new-word bindings ───
    print(f"\n[PHASE C] new-word binding test", flush=True)
    learn_results = []
    for new_word, target in new_words:
        learn_results.append(
            evaluate_new_word(bridge, new_word, target, verbose=verbose))
    n_bound = sum(1 for r in learn_results if r["bound_ok"])
    binding_rate = n_bound / len(learn_results) if learn_results else 0.0
    print(f"  binding success: {n_bound}/{len(learn_results)} = "
          f"{binding_rate:.1%}", flush=True)

    # ─── Phase D: re-test primaries (catastrophic-forgetting check) ───
    print(f"\n[PHASE D] re-test primaries (forgetting check)", flush=True)
    phase_d = evaluate_primaries(bridge, n_rounds=2, verbose=verbose)
    print(f"  post-learn primary acc: {phase_d['accuracy']:.1%} "
          f"({phase_d['correct']}/{phase_d['total']})", flush=True)
    retention = (phase_d["accuracy"] / phase_a["accuracy"]
                 if phase_a["accuracy"] > 0 else 0.0)
    print(f"  retention ratio: {retention:.2f} "
          f"(post / baseline)", flush=True)

    # GO criterion (mirrors Phase 1.4 BRANCH A threshold)
    go = (binding_rate >= 0.5) and (retention >= 0.8)
    verdict = "GO" if go else "NO-GO"
    print(f"\n=== verdict: {verdict} ===", flush=True)
    print(f"  binding_rate >= 50%: "
          f"{'PASS' if binding_rate >= 0.5 else 'FAIL'} "
          f"({binding_rate:.1%})", flush=True)
    print(f"  retention   >= 80%: "
          f"{'PASS' if retention >= 0.8 else 'FAIL'} "
          f"({retention:.1%})", flush=True)
    print(f"  total wall clock: {time.time() - t0:.0f}s", flush=True)

    return {
        "seed": seed,
        "demo_kind": "chat_learn_demo",
        "n_train_events": n_train_events,
        "learn_n_events": learn_n_events,
        "new_words": [(w, a) for w, a in new_words],
        # Match chat_demo's schema so chat_demo_aggregate handles it
        "accuracy": phase_d["accuracy"],
        "correct": phase_d["correct"],
        "total":   phase_d["total"],
        "per_word_accuracy": phase_d["per_word_accuracy"],
        # Track 3-specific metrics
        "primary_baseline_accuracy": phase_a["accuracy"],
        "primary_post_learn_accuracy": phase_d["accuracy"],
        "primary_retention_ratio": retention,
        "learn_binding_rate": binding_rate,
        "learn_results": learn_results,
        "learn_log": learn_log,
        "phase_a_turns": phase_a["turns"],
        "phase_d_turns": phase_d["turns"],
        "verdict": verdict,
        "go": go,
        "wall_clock_sec": time.time() - t0,
        "train_sec": train_sec,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=200,
                    help="Events per word during initial training")
    ap.add_argument("--learn-events", type=int, default=50,
                    help="Paired co-firing events per learn command")
    ap.add_argument("--new-words", type=str, default="ahead:N,back:S",
                    help="Comma-separated word:action pairs to learn. "
                         "Example: --new-words 'ahead:N,back:S'")
    ap.add_argument("--out-stats", type=str, default=None,
                    help="JSON stats output path (matches chat_demo schema)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-turn logging")
    args = ap.parse_args()

    # Parse new_words
    new_words = []
    for pair in args.new_words.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            ap.error(f"--new-words item missing ':': {pair!r}")
        word, action = pair.split(":", 1)
        word = word.strip().lower()
        action = action.strip().upper()
        if action not in ("N", "E", "S", "W"):
            ap.error(f"--new-words action must be N/E/S/W, got {action!r}")
        new_words.append((word, action))

    if not new_words:
        ap.error("--new-words is empty")

    result = run_chat_learn_demo(
        seed=args.seed,
        n_train_events=args.train_events,
        learn_n_events=args.learn_events,
        new_words=new_words,
        verbose=not args.quiet,
    )

    if args.out_stats:
        out = Path(args.out_stats)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Strip large transcript fields for the stats JSON
        slim = {k: v for k, v in result.items()
                if k not in ("phase_a_turns", "phase_d_turns")}
        out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(f"\n[OUT] stats -> {out}", flush=True)

    return 0 if result["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
