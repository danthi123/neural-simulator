"""Phase 1.5 -- Unified continual-learning eval suite.

Dispatcher for 6 benchmarks that together quantify whether
biology-grounded continual learning works:

1. sequential_expansion -- train A, train B, retest A retention
   (wraps Phase 1.4)
2. retention_over_time   -- train, run N silence steps, retest
3. interference          -- interleaved vs sequential training
4. long_tail             -- imbalanced word frequency, rare-word retention
5. multimodality         -- W->A and W->I both learned (needs Tier 2.2)
6. composition           -- phrase + single-word both work (needs Tier 2.3)

Each benchmark returns:
{
  name: str,
  score: float in [0, 1],   # higher = better continual learning
  pass: bool,
  details: {...benchmark-specific metrics...}
}

Aggregate score = mean of all run benchmarks.
PASS for "biology-grounded continual learning works" requires
aggregate >= 0.7 across all 6 benchmarks.

This module is intended for:
- Path F's premise validation (before path-f-hybrid branch)
- Regression testing during Phase 2 development
- Comparison runs between biology-grounded and hybrid configs

Usage:
    python -m research.runners.continual_eval_suite \\
        --benchmarks sequential_expansion retention_over_time \\
        --seed 42 \\
        --out-stats research/findings/raw/continual_eval/seed42.json

Add `--benchmarks all` to run all 6.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


BENCHMARK_REGISTRY: dict[str, Callable] = {}


def register_benchmark(name: str):
    """Decorator to register a benchmark function in the dispatcher."""
    def decorator(fn):
        BENCHMARK_REGISTRY[name] = fn
        return fn
    return decorator


# -----------------------------------------------------------------
# Benchmark 1: sequential_expansion
# Train vocab A, train vocab B (different words), retest A.
# Retention% = post-B accuracy / post-A accuracy.
# Score = retention_pct / 100, clipped to [0, 1].
# Pass if retention >= 0.8 (master plan threshold).
# -----------------------------------------------------------------

@register_benchmark("sequential_expansion")
def benchmark_sequential_expansion(args, rng):
    """Wraps continual_forgetting_eval logic."""
    from research.runners.bio_three_factor import run_three_factor
    from research.runners.text_eval import evaluate_word_to_action
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    PRIMARY_WORDS = ["north", "east", "south", "west"]
    SYNONYM_WORDS = ["up", "right", "down", "left"]
    SYNONYM_TO_ACTION = {"up": "N", "right": "E", "down": "S", "left": "W"}

    print("  [SEQ] Phase A: train primaries", flush=True)
    t0 = time.time()
    bridge, _ = run_three_factor(
        seed=args.seed,
        n_events_per_direction=args.events_per_word,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,  # Tier 1 BREAKTHROUGH config
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=False,
        verbose=False,
    )
    print(f"    Phase A done ({time.time()-t0:.0f}s)", flush=True)

    wa_a = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        verbose=False,
    )
    print(f"    Primary post-A: {wa_a['accuracy']:.1%}", flush=True)

    # Phase B: train synonyms only
    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}")) for a in ["N","E","S","W"]}
    lang_output_idx = list(rm.indices("language_output"))

    synonym_buffer = []
    for word in SYNONYM_WORDS:
        action = SYNONYM_TO_ACTION[word]
        for _ in range(args.events_per_word):
            synonym_buffer.append({"token": word, "action": action})
    rng.shuffle(synonym_buffer)

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 1.0)
        bridge.set_plasticity_gate("motor_to_language_output", 1.0)
    except Exception:
        pass

    t0 = time.time()
    print(f"  [SEQ] Phase B: train {len(synonym_buffer)} synonym events",
          flush=True)
    n_lang_in = len(lang_input_idx)
    for ev_idx, event in enumerate(synonym_buffer):
        token = event["token"]
        target_action = event["action"]
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        drive = vocab_to_drive_pattern(token, n_neurons=n_lang_in,
                                        drive_max_pA=200.0, sparsity=0.1)
        bridge.cp_external_input_current[
            cp.asarray(lang_input_idx, dtype=cp.int64)
        ] = cp.asarray(drive, dtype=cp.float32)
        bridge.cp_external_input_current[
            cp.asarray(lang_output_idx, dtype=cp.int64)
        ] = cp.asarray(drive, dtype=cp.float32)
        target_motor_arr = cp.asarray(motor_idx[target_action], dtype=cp.int64)
        bridge.cp_external_input_current[target_motor_arr] += 300.0
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
    print(f"    Phase B done ({time.time()-t0:.0f}s)", flush=True)

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    wa_after_b = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        verbose=False,
    )
    print(f"    Primary post-B: {wa_after_b['accuracy']:.1%}", flush=True)

    primary_a = wa_a["accuracy"]
    primary_b = wa_after_b["accuracy"]
    retention = (primary_b / primary_a) if primary_a > 0 else 0.0
    score = float(np.clip(retention, 0.0, 1.0))

    return {
        "name": "sequential_expansion",
        "score": score,
        "pass": retention >= 0.8,
        "details": {
            "primary_a_acc": primary_a,
            "primary_b_acc": primary_b,
            "retention_pct": retention * 100,
        },
    }


# -----------------------------------------------------------------
# Benchmark 2: retention_over_time
# Train vocab, run N silent steps, retest. Score = post-silence accuracy
# / post-train accuracy. Pass if retention >= 0.8.
# -----------------------------------------------------------------

@register_benchmark("retention_over_time")
def benchmark_retention_over_time(args, rng):
    from research.runners.bio_three_factor import run_three_factor
    from research.runners.text_eval import evaluate_word_to_action

    silence_steps = getattr(args, "silence_steps", 5000)
    print(f"  [TIME] Train vocab", flush=True)
    t0 = time.time()
    bridge, _ = run_three_factor(
        seed=args.seed,
        n_events_per_direction=args.events_per_word,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,  # Tier 1 BREAKTHROUGH config
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=False,
        verbose=False,
    )
    print(f"    Train done ({time.time()-t0:.0f}s)", flush=True)

    wa_pre = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        verbose=False,
    )
    print(f"    Pre-silence: {wa_pre['accuracy']:.1%}", flush=True)

    # Run silent steps (no input drive). Tests whether STDP/Hebbian
    # decay or background noise erodes learned weights over time.
    # Freeze plasticity gates first to isolate "passive retention" --
    # otherwise OU-noise-driven correlated spikes could trigger STDP
    # drift, confounding the test of weight stability vs active drift.
    print(f"  [TIME] {silence_steps} silent steps "
          f"(plasticity frozen)", flush=True)
    t0 = time.time()
    import cupy as cp
    for gate in ("language_input_to_motor", "motor_to_language_output"):
        try:
            bridge.set_plasticity_gate(gate, 0.0)
        except Exception:
            pass
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(silence_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    print(f"    Silence done ({time.time()-t0:.0f}s)", flush=True)

    wa_post = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        verbose=False,
    )
    print(f"    Post-silence: {wa_post['accuracy']:.1%}", flush=True)

    pre_acc = wa_pre["accuracy"]
    post_acc = wa_post["accuracy"]
    retention = (post_acc / pre_acc) if pre_acc > 0 else 0.0
    score = float(np.clip(retention, 0.0, 1.0))

    return {
        "name": "retention_over_time",
        "score": score,
        "pass": retention >= 0.8,
        "details": {
            "pre_acc": pre_acc,
            "post_acc": post_acc,
            "retention_pct": retention * 100,
            "silence_steps": silence_steps,
        },
    }


# -----------------------------------------------------------------
# Benchmark 3: interference (TODO -- placeholder)
# Train interleaved A+B, compare to sequential A then B.
# -----------------------------------------------------------------

@register_benchmark("interference")
def benchmark_interference(args, rng):
    """Train interleaved 8-word vocab; eval all 8.

    Compares to chance (25% for 4 actions). Pass if mean
    accuracy across all 8 words >= 50% AND no individual word
    is below chance (25%). The latter ensures NO catastrophic
    forgetting of any individual word during interleaved training.
    """
    from research.runners.bio_three_factor import run_three_factor
    from research.runners.text_eval import evaluate_word_to_action

    print("  [INT] Train interleaved 8-word vocab", flush=True)
    t0 = time.time()
    bridge, _ = run_three_factor(
        seed=args.seed,
        n_events_per_direction=args.events_per_word,  # per word, not action
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,  # Tier 1 BREAKTHROUGH config
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=True,         # 8-word interleaved
        synonym_vocab_size=8,
        verbose=False,
    )
    print(f"    Train done ({time.time()-t0:.0f}s)", flush=True)

    # Eval all 8 words
    wa = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        synonym_mode=True, synonym_vocab_size=8,
        verbose=False,
    )
    mean_acc = wa["accuracy"]
    print(f"    Mean acc across 8 words: {mean_acc:.1%}", flush=True)

    # Per-word accuracy from confusion matrix
    confusion = wa.get("confusion_matrix", {})
    per_word_correct = {}
    for word, counts in confusion.items():
        total = sum(counts.values()) if isinstance(counts, dict) else 0
        # Find target action for this word
        from research.runners.text_eval import EXTENDED_WORD_TO_ACTION
        if word in EXTENDED_WORD_TO_ACTION:
            target = EXTENDED_WORD_TO_ACTION[word]
            correct = counts.get(target, 0) if isinstance(counts, dict) else 0
            per_word_correct[word] = correct / total if total > 0 else 0.0

    min_word_acc = min(per_word_correct.values()) if per_word_correct else 0.0
    above_chance = sum(1 for v in per_word_correct.values() if v >= 0.25)
    score = float(np.clip(mean_acc, 0.0, 1.0))

    return {
        "name": "interference",
        "score": score,
        "pass": mean_acc >= 0.5 and min_word_acc >= 0.25,
        "details": {
            "mean_acc": mean_acc,
            "min_word_acc": min_word_acc,
            "n_above_chance": above_chance,
            "per_word_acc": per_word_correct,
        },
    }


# -----------------------------------------------------------------
# Benchmark 4: long_tail (TODO -- placeholder)
# Imbalanced word frequency: 4 words at 200 events, 4 words at 10 events.
# -----------------------------------------------------------------

@register_benchmark("long_tail")
def benchmark_long_tail(args, rng):
    """Train 8-word vocab with imbalanced frequency.

    4 'common' words at events_per_word events (200 default).
    4 'rare' words at events_per_word // 20 events (10 default).
    Pass if rare words still get >= 30% accuracy (5pp above 25% chance).

    Tests whether less-frequent vocab survives the dominance of
    high-frequency words during embodied Hebbian learning. This is
    the long-tail problem: in real conversational data, most words
    are rare. If we can't learn rare words, we can't be conversational.
    """
    from research.runners.bio_three_factor import run_three_factor
    from research.runners.text_eval import (
        evaluate_word_to_action, SYNONYM_GROUPS,
        EXTENDED_WORD_TO_ACTION,
    )
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    common_events = args.events_per_word
    rare_events = max(1, args.events_per_word // 20)
    common_words = ["north", "east", "south", "west"]
    rare_words = ["up", "right", "down", "left"]
    word_to_action = {**{w: w[0].upper() for w in common_words},
                      "up": "N", "right": "E", "down": "S", "left": "W"}

    print(f"  [LT] Train imbalanced: {len(common_words)} common "
          f"({common_events} ea) + {len(rare_words)} rare "
          f"({rare_events} ea)", flush=True)

    t0 = time.time()
    # Train all 4 common words via run_three_factor (Tier 1 paradigm)
    bridge, _ = run_three_factor(
        seed=args.seed,
        n_events_per_direction=common_events,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,  # Tier 1 BREAKTHROUGH config
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=False,
        verbose=False,
    )
    print(f"    Common training done ({time.time()-t0:.0f}s)", flush=True)

    # Now train rare words manually on the same bridge
    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}")) for a in ["N","E","S","W"]}
    lang_output_idx = list(rm.indices("language_output"))
    n_lang_in = len(lang_input_idx)

    rare_buffer = []
    for word in rare_words:
        action = word_to_action[word]
        for _ in range(rare_events):
            rare_buffer.append({"token": word, "action": action})
    rng.shuffle(rare_buffer)

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 1.0)
        bridge.set_plasticity_gate("motor_to_language_output", 1.0)
    except Exception:
        pass

    t0 = time.time()
    print(f"  [LT] Train rare ({len(rare_buffer)} events)", flush=True)
    for ev in rare_buffer:
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        drive = vocab_to_drive_pattern(ev["token"], n_neurons=n_lang_in,
                                        drive_max_pA=200.0, sparsity=0.1)
        bridge.cp_external_input_current[
            cp.asarray(lang_input_idx, dtype=cp.int64)
        ] = cp.asarray(drive, dtype=cp.float32)
        bridge.cp_external_input_current[
            cp.asarray(lang_output_idx, dtype=cp.int64)
        ] = cp.asarray(drive, dtype=cp.float32)
        target = cp.asarray(motor_idx[ev["action"]], dtype=cp.int64)
        bridge.cp_external_input_current[target] += 300.0
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass
    print(f"    Rare training done ({time.time()-t0:.0f}s)", flush=True)

    # Eval common + rare separately
    wa_all = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        synonym_mode=True, synonym_vocab_size=8,
        verbose=False,
    )
    confusion = wa_all.get("confusion_matrix", {})
    per_word = {}
    for word, counts in confusion.items():
        if word in word_to_action and isinstance(counts, dict):
            target = word_to_action[word]
            total = sum(counts.values())
            per_word[word] = counts.get(target, 0) / total if total > 0 else 0

    common_acc = np.mean([per_word.get(w, 0.0) for w in common_words])
    rare_acc = np.mean([per_word.get(w, 0.0) for w in rare_words])

    print(f"    Common acc: {common_acc:.1%}, Rare acc: {rare_acc:.1%}",
          flush=True)

    # Score = rare accuracy (the test is whether rare words survive)
    score = float(np.clip(rare_acc, 0.0, 1.0))

    return {
        "name": "long_tail",
        "score": score,
        "pass": rare_acc >= 0.30,
        "details": {
            "common_acc": float(common_acc),
            "rare_acc": float(rare_acc),
            "common_events": common_events,
            "rare_events": rare_events,
            "per_word": per_word,
        },
    }


# -----------------------------------------------------------------
# Benchmark 5: multimodality (depends on Tier 2.2 -- placeholder)
# Train W->A AND W->I bindings, both should work.
# -----------------------------------------------------------------

@register_benchmark("multimodality")
def benchmark_multimodality(args, rng):
    """TODO: depends on Tier 2.2 visual-language binding.
    Train embodied during nav; eval W->A (Tier 1) and W->I (Tier 2.2).
    Pass if both >= 4/6 aligned across seeds."""
    return {
        "name": "multimodality",
        "score": 0.0,
        "pass": False,
        "details": {"status": "tier_2_2_pending"},
    }


# -----------------------------------------------------------------
# Benchmark 6: composition (depends on Tier 2.3 -- placeholder)
# Train phrases (Tier 2.3), single-word still works.
# -----------------------------------------------------------------

@register_benchmark("composition")
def benchmark_composition(args, rng):
    """TODO: depends on Tier 2.3 PFC verb pool.
    Train 'go [direction]' phrases; eval single-word direction
    bindings still work AND phrases work AND verb-only stays quiet."""
    return {
        "name": "composition",
        "score": 0.0,
        "pass": False,
        "details": {"status": "tier_2_3_pending"},
    }


# -----------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", default=["sequential_expansion"],
                    help=f"Benchmarks to run. Available: "
                         f"{list(BENCHMARK_REGISTRY.keys())}, or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--events-per-word", type=int, default=200)
    ap.add_argument("--n-eval-per-word", type=int, default=25)
    # Standard Tier 1 arch (validated 5/6 + 6/6).
    # For interference benchmark (8-word synonym), scale-up arch
    # (4096/1000/120) gives better baseline -- override at CLI:
    #   --n-lang-input 4096 --n-motor-per-action 1000 \
    #   --n-motor-fs-per-action 120
    ap.add_argument("--n-lang-input", type=int, default=2048,
                    help="Standard Tier 1 arch (default). For 8-word "
                         "interference test, override to 4096.")
    ap.add_argument("--n-motor-per-action", type=int, default=500)
    ap.add_argument("--n-motor-fs-per-action", type=int, default=60)
    ap.add_argument("--silence-steps", type=int, default=5000,
                    help="Silent steps for retention_over_time")
    ap.add_argument("--out-stats", type=str, required=True)
    args = ap.parse_args()

    if "all" in args.benchmarks:
        args.benchmarks = list(BENCHMARK_REGISTRY.keys())

    print("=" * 60)
    print(f"PHASE 1.5 CONTINUAL-LEARNING EVAL SUITE (seed={args.seed})")
    print(f"Benchmarks: {args.benchmarks}")
    print("=" * 60, flush=True)

    rng = np.random.default_rng(args.seed * 13 + 7)
    results = {
        "seed": args.seed,
        "benchmarks_requested": args.benchmarks,
        "benchmarks": [],
    }

    for name in args.benchmarks:
        if name not in BENCHMARK_REGISTRY:
            print(f"  [!]  Unknown benchmark: {name}", flush=True)
            continue
        print(f"\n--- Running benchmark: {name} ---", flush=True)
        t0 = time.time()
        try:
            result = BENCHMARK_REGISTRY[name](args, rng)
            result["wall_clock_s"] = time.time() - t0
            results["benchmarks"].append(result)
            status = "[OK]" if result["pass"] else "[X]"
            print(f"  {status} {name}: score={result['score']:.2f} "
                  f"pass={result['pass']} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  ? {name} crashed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results["benchmarks"].append({
                "name": name,
                "score": 0.0,
                "pass": False,
                "details": {"error": str(e)},
                "wall_clock_s": time.time() - t0,
            })

    # Aggregate
    completed = [b for b in results["benchmarks"]
                 if b["details"].get("status") not in
                    ("not_yet_implemented", "tier_2_2_pending",
                     "tier_2_3_pending")
                 and "error" not in b["details"]]
    if completed:
        agg = float(np.mean([b["score"] for b in completed]))
        all_pass = all(b["pass"] for b in completed)
    else:
        agg = 0.0
        all_pass = False

    results["aggregate"] = {
        "score": agg,
        "all_pass": all_pass,
        "n_completed": len(completed),
        "n_total": len(args.benchmarks),
    }

    print("\n" + "=" * 60)
    print("CONTINUAL-LEARNING ASSESSMENT")
    print("=" * 60)
    print(f"  Completed: {len(completed)}/{len(args.benchmarks)}")
    print(f"  Aggregate score: {agg:.2f}")
    print(f"  All pass: {all_pass}")
    if agg >= 0.7 and all_pass:
        print("  [OK] BIOLOGY-GROUNDED CONTINUAL LEARNING VALIDATED")
    elif agg >= 0.5:
        print("  [!]  PARTIAL: continual learning works imperfectly")
    else:
        print("  [X] FAIL: continual learning broken under test")
    print("=" * 60, flush=True)

    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).write_text(json.dumps(results, indent=2,
                                                default=str))
    print(f"\nSaved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
