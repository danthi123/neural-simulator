"""Chat-speak demo (Track 3 layer 4, batch A2W validation, 2026-05-09).

Inverse of chat_demo (which tests W->A: word -> motor). This runner tests
A->W (action -> word): drive motor_<action>, read language_output, decode
to a word via cosine similarity against vocab drive patterns.

Validates the chat_repl.generative_inference primitive shipped in
a675fa1 — the missing piece that made Track 3 v1 feature-complete.
End-to-end smoke: train a Tier 1 bridge, run W->A as regression
baseline, then run A->W on each of the 4 actions, emit JSON stats.

Pipeline (~10 min single seed):
  Phase A: train Tier 1 4-word bridge (~6 min)
  Phase B: regression check W->A (4 directions × N rounds)
  Phase C: A->W via generative_inference for each of N/E/S/W
  Phase D: report verdict (GO if A->W >= 50% AND W->A regression intact)

Output JSON shape (matches chat_demo's schema for chat_demo_aggregate):
  {
    "seed": 42,
    "demo_kind": "chat_speak_demo",
    "accuracy": <Phase B W->A accuracy, regression baseline>,
    "speak_accuracy": <Phase C A->W accuracy>,
    "speak_results": [
      {"target_action": "N", "predicted_word": "north", "correct": True,
       "rankings": [["north", 0.81], ["up", 0.74], ...]},
      ...
    ],
    "verdict": "GO"|"NO-GO",
    "go": True|False,
  }

Tier 1 BREAKTHROUGH validated A->W mean 45-63% (6/6 aligned). The
generative_inference primitive should reproduce that range under the
same training config.

Usage:
    python -m research.runners.chat_speak_demo --seed 42 \\
        --train-events 200 --out-stats path/to/stats.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


# Reuse Tier 1 vocab + W->A primitives
from research.runners.chat_demo import (
    DIRECTIONS, DIRECTION_TO_ACTION, ACTION_TO_DIRECTION,
    train_chat_bridge, chat_turn,
)
# Import the A->W primitive from chat_repl
from research.runners.chat_repl import generative_inference


def evaluate_w_to_a_baseline(bridge, n_rounds: int = 2, verbose: bool = True) -> dict:
    """Regression check: W->A accuracy on Tier 1 4-word vocab."""
    correct_total = 0
    total_turns = 0
    correct_per_word = {w: 0 for w in DIRECTIONS}
    total_per_word = {w: 0 for w in DIRECTIONS}
    for round_n in range(1, n_rounds + 1):
        for word in DIRECTIONS:
            result = chat_turn(bridge, word)
            total_turns += 1
            total_per_word[word] += 1
            if result["correct"]:
                correct_total += 1
                correct_per_word[word] += 1
            if verbose:
                marker = "[OK]" if result["correct"] else "[X] "
                d = result["delta_counts"]
                print(f"  {marker} W->A '{word}' -> {result['predicted_direction']} "
                      f"(delta N{d['N']:+d} E{d['E']:+d} S{d['S']:+d} "
                      f"W{d['W']:+d}, x{result['confidence_ratio']:.1f})",
                      flush=True)
    return {
        "accuracy": correct_total / total_turns if total_turns else 0.0,
        "correct": correct_total,
        "total": total_turns,
        "per_word_accuracy": {
            w: (correct_per_word[w] / total_per_word[w]
                if total_per_word[w] else 0.0)
            for w in DIRECTIONS
        },
    }


def evaluate_a_to_w(bridge, vocab_words=None, verbose: bool = True,
                     temperature: float = 0.0,
                     rng_seed: int = None) -> dict:
    """Run :speak / generative_inference on each of N/E/S/W; compare to expected.

    Args:
        temperature: 0 (default) = strict argmax for repro testing.
            >0 = softmax sampling. See generative_inference for details.
    """
    if vocab_words is None:
        vocab_words = list(DIRECTIONS)
    speak_results = []
    correct = 0
    for action in ("N", "E", "S", "W"):
        expected_word = ACTION_TO_DIRECTION[action]
        result = generative_inference(
            bridge, action, vocab_words=vocab_words,
            temperature=temperature, rng_seed=rng_seed,
        )
        pred = result["predicted_word"]
        rankings = [(w, float(s)) for w, s in result["rankings"]]
        ok = (pred == expected_word)
        if ok:
            correct += 1
        speak_results.append({
            "target_action": action,
            "expected_word": expected_word,
            "predicted_word": pred,
            "correct": ok,
            "confidence": float(result["confidence"]) if result["confidence"] != float("inf") else None,
            "rankings": rankings,
        })
        if verbose:
            mark = "[OK]" if ok else "[X] "
            top1 = rankings[0]
            print(f"  {mark} A->W motor_{action} -> '{pred}' "
                  f"(expected '{expected_word}', top-1 sim={top1[1]:.2f})",
                  flush=True)
    return {
        "accuracy": correct / 4,
        "correct": correct,
        "total": 4,
        "per_action": speak_results,
    }


def run_chat_speak_demo(seed: int = 42, n_train_events: int = 200,
                        verbose: bool = True,
                        temperature: float = 0.0) -> dict:
    # 2026-05-09: emit structured [PROGRESS] events so the webapp inflight
    # panel + 3D Brain live mode show real progress (not "0% no markers").
    # Same fix pattern shipped for continual_eval_suite tonight.
    from sim.progress import emit_progress

    print(f"\n=== chat_speak_demo (seed={seed}) ===", flush=True)
    print(f"  Tier 1 4-word vocab: {DIRECTIONS}", flush=True)
    print(f"  train_events={n_train_events}\n", flush=True)

    # 3 phases: training, W2A regression, A2W speak
    emit_progress("phase", current=0, total=3, phase="training",
                  unit="phases", label="chat_speak_demo")
    t0 = time.time()
    bridge = train_chat_bridge(seed=seed, n_events_per_word=n_train_events,
                                verbose=verbose)
    train_sec = time.time() - t0
    emit_progress("complete", current=1, total=3, phase="training",
                  unit="phases", label="chat_speak_demo",
                  wall_clock_s=int(train_sec))

    # Phase B: W->A regression baseline
    print(f"\n[PHASE B] W->A regression check (chat_demo path)", flush=True)
    emit_progress("phase", current=1, total=3, phase="W2A_regression",
                  unit="phases", label="chat_speak_demo")
    w2a = evaluate_w_to_a_baseline(bridge, n_rounds=2, verbose=verbose)
    print(f"  W->A accuracy: {w2a['accuracy']:.1%} ({w2a['correct']}/{w2a['total']})",
          flush=True)
    emit_progress("complete", current=2, total=3, phase="W2A_regression",
                  unit="phases", label="chat_speak_demo",
                  score=float(w2a['accuracy']))

    # Phase C: A->W via generative_inference (the new layer 4 primitive)
    print(f"\n[PHASE C] A->W generative decoder (:speak primitive)", flush=True)
    emit_progress("phase", current=2, total=3, phase="A2W_speak",
                  unit="phases", label="chat_speak_demo")
    a2w = evaluate_a_to_w(
        bridge, verbose=verbose,
        temperature=temperature,
        rng_seed=(seed if temperature > 0 else None),
    )
    print(f"  A->W accuracy: {a2w['accuracy']:.1%} ({a2w['correct']}/{a2w['total']})",
          flush=True)
    emit_progress("complete", current=3, total=3, phase="A2W_speak",
                  unit="phases", label="chat_speak_demo",
                  score=float(a2w['accuracy']))

    # Verdict: GO if A->W >= 50% AND W->A regression intact (>= 25% chance baseline)
    speak_pass = a2w["accuracy"] >= 0.50
    regression_intact = w2a["accuracy"] >= 0.25
    go = speak_pass and regression_intact
    verdict = "GO" if go else "NO-GO"
    print(f"\n=== verdict: {verdict} ===", flush=True)
    print(f"  W->A regression (>= 25% chance): "
          f"{'PASS' if regression_intact else 'FAIL'} ({w2a['accuracy']:.1%})",
          flush=True)
    print(f"  A->W speak (>= 50%): "
          f"{'PASS' if speak_pass else 'FAIL'} ({a2w['accuracy']:.1%})",
          flush=True)
    print(f"  total wall clock: {time.time() - t0:.0f}s", flush=True)

    return {
        "seed": seed,
        "demo_kind": "chat_speak_demo",
        "n_train_events": n_train_events,
        # Match chat_demo schema for chat_demo_aggregate
        "accuracy": w2a["accuracy"],
        "correct": w2a["correct"],
        "total": w2a["total"],
        "per_word_accuracy": w2a["per_word_accuracy"],
        # Track 3 layer 4 specific
        "speak_accuracy": a2w["accuracy"],
        "speak_correct": a2w["correct"],
        "speak_total": a2w["total"],
        "speak_results": a2w["per_action"],
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
    ap.add_argument("--out-stats", type=str, default=None,
                    help="JSON stats output path (matches chat_demo schema)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-turn logging")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Softmax sampling temperature for :speak. "
                         "0 (default) = strict argmax, matches all prior "
                         "multi-seed validations. 0.01-0.05 = sampling "
                         "with primary preference.")
    args = ap.parse_args()

    result = run_chat_speak_demo(
        seed=args.seed,
        n_train_events=args.train_events,
        verbose=not args.quiet,
        temperature=args.temperature,
    )

    if args.out_stats:
        out = Path(args.out_stats)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[OUT] stats -> {out}", flush=True)

    return 0 if result["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
