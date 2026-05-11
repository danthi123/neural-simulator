"""Chat-speak synonym demo — Tier 2.1 8-word :speak validation (2026-05-09).

Sister runner to `chat_speak_demo.py` (Tier 1 4-word :speak), but uses the
Tier 2.1 v4 scale-up architecture and 8-word synonym vocabulary. Tests
whether the same generative-inference primitive that works on Tier 1
(45–63% A→W in chat_speak_demo single-seed validation) also works when
each motor action has TWO bound words (synonym pair).

Sister to:
  - `chat_speak_demo.py`         — 4-word :speak (Track 3 layer 4 v1)
  - `chat_synonym_demo.py`       — 8-word W→A (Tier 2.1 v4 reception)
  - `chat_speak_synonym_demo.py` — 8-word A→W (this file, Tier 2.1 v4 production)

Pipeline (~10–13 min single seed):
  Phase A: train Tier 2.1 v4 scale-up bridge (8-word synonym, ~6–9 min)
  Phase B: W→A regression baseline on all 8 words
  Phase C: A→W via generative_inference for each of N/E/S/W,
           ranking against the full 8-word vocab
  Phase D: report verdict; "any-synonym" PASS if A→W ≥ 50%

Scoring:
  For each motor action target X (e.g. N):
    primary_correct  = (top-1 == "north")
    synonym_correct  = (top-1 == "up")
    any_correct      = (top-1 in {"north", "up"})
  any_correct is the headline number — it answers "does the network
  produce SOME word that means X when motor_X is driven?". This is
  the production-side analog of Tier 2.1's W→A reception: in W→A
  Tier 2.1 reported 5/6 aligned at A→W mean 63.7%.

Output JSON shape (extends chat_speak_demo's schema):
  {
    "seed": 42,
    "demo_kind": "chat_speak_synonym_demo",
    "vocab_size": 8,
    "accuracy": <W→A any-synonym accuracy on 8 words>,
    "speak_accuracy": <A→W any-synonym accuracy on 4 actions>,
    "speak_primary_accuracy": <A→W primary-only accuracy>,
    "speak_synonym_accuracy": <A→W synonym-only accuracy>,
    "speak_results": [
      {"target_action": "N",
       "predicted_word": "up",          # top-1 ranked
       "expected_words": ["north", "up"],
       "any_correct": True,             # top-1 IS one of the synonyms
       "primary_correct": False,        # top-1 != primary
       "synonym_correct": True,         # top-1 == synonym
       "rankings": [["up", 0.74], ["north", 0.71], ...]},
      ...
    ],
    "verdict": "GO"|"NO-GO",
  }

Usage:
    python -m research.runners.chat_speak_synonym_demo --seed 42 \\
        --train-events 400 --out-stats path/to/stats.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


# Reuse Tier 2.1 8-word vocab + scale-up training
from research.runners.chat_synonym_demo import (
    SYNONYM_GROUPS, ALL_WORDS, WORD_TO_ACTION, ACTION_TO_PRIMARY,
    train_chat_bridge, chat_turn,
)
# Import the A->W primitive from chat_repl
from research.runners.chat_repl import generative_inference


def evaluate_w_to_a_baseline_synonym(bridge, n_rounds: int = 2,
                                      verbose: bool = True) -> dict:
    """W→A regression on 8-word synonym vocab (matches chat_synonym_demo)."""
    correct_total = 0
    total_turns = 0
    correct_per_word = {w: 0 for w in ALL_WORDS}
    total_per_word = {w: 0 for w in ALL_WORDS}
    for round_n in range(1, n_rounds + 1):
        for word in ALL_WORDS:
            expected_action = WORD_TO_ACTION[word]
            result = chat_turn(bridge, word)
            predicted_action = result.get("predicted_action") or result.get(
                "predicted_direction"
            )
            ok = (predicted_action == expected_action)
            total_turns += 1
            total_per_word[word] += 1
            if ok:
                correct_total += 1
                correct_per_word[word] += 1
            if verbose:
                marker = "[OK]" if ok else "[X] "
                d = result["delta_counts"]
                print(f"  {marker} W->A '{word}' (-> {expected_action}) -> "
                      f"{predicted_action} "
                      f"(delta N{d['N']:+d} E{d['E']:+d} S{d['S']:+d} "
                      f"W{d['W']:+d})",
                      flush=True)
    return {
        "accuracy": correct_total / total_turns if total_turns else 0.0,
        "correct": correct_total,
        "total": total_turns,
        "per_word_accuracy": {
            w: (correct_per_word[w] / total_per_word[w]
                if total_per_word[w] else 0.0)
            for w in ALL_WORDS
        },
    }


def evaluate_a_to_w_synonym(bridge, verbose: bool = True,
                              temperature: float = 0.0,
                              rng_seed: int = None) -> dict:
    """A→W: drive motor_<action>, decode to one of 8 synonym words.

    Returns dict with three accuracy metrics:
      any_accuracy     — top-1 word is ANY synonym for target action
      primary_accuracy — top-1 word is the PRIMARY synonym (north/east/south/west)
      synonym_accuracy — top-1 word is the SECONDARY synonym (up/right/down/left)
    """
    speak_results = []
    any_correct = 0
    primary_correct = 0
    synonym_correct = 0
    for action in ("N", "E", "S", "W"):
        expected_synonyms = SYNONYM_GROUPS[action]
        primary_word = expected_synonyms[0]
        synonym_word = expected_synonyms[1]
        # Rank all 8 words; whichever has highest cosine to the post-drive
        # language_output delta is "what the network said". top_k=8 to
        # keep the full ranking in the JSON (default top_k=4 would truncate
        # to top-4 of 8 — top-1 is correct either way, but the full list
        # is useful for diagnosing primary-vs-synonym preference).
        # 2026-05-10: temperature plumbed for synonym-lift testing.
        # τ=0 (default) preserves deterministic argmax for repro testing.
        result = generative_inference(
            bridge, action, vocab_words=ALL_WORDS,
            top_k=8, temperature=temperature, rng_seed=rng_seed,
        )
        pred = result["predicted_word"]
        rankings = [(w, float(s)) for w, s in result["rankings"]]

        any_ok = (pred in expected_synonyms)
        primary_ok = (pred == primary_word)
        synonym_ok = (pred == synonym_word)
        if any_ok:
            any_correct += 1
        if primary_ok:
            primary_correct += 1
        if synonym_ok:
            synonym_correct += 1

        speak_results.append({
            "target_action": action,
            "expected_words": expected_synonyms,
            "primary_word": primary_word,
            "synonym_word": synonym_word,
            "predicted_word": pred,
            "any_correct": any_ok,
            "primary_correct": primary_ok,
            "synonym_correct": synonym_ok,
            "confidence": (float(result["confidence"])
                           if result["confidence"] != float("inf") else None),
            "rankings": rankings,
        })
        if verbose:
            mark = "[OK]" if any_ok else "[X] "
            top1 = rankings[0]
            tag = ("primary" if primary_ok
                   else "synonym" if synonym_ok
                   else "wrong")
            print(f"  {mark} A->W motor_{action} -> '{pred}' [{tag}] "
                  f"(expected {expected_synonyms}, top-1 sim={top1[1]:.2f})",
                  flush=True)
    return {
        "any_accuracy": any_correct / 4,
        "primary_accuracy": primary_correct / 4,
        "synonym_accuracy": synonym_correct / 4,
        "any_correct": any_correct,
        "primary_correct": primary_correct,
        "synonym_correct": synonym_correct,
        "total": 4,
        "per_action": speak_results,
    }


def run_chat_speak_synonym_demo(seed: int = 42,
                                  n_train_events: int = 400,
                                  n_lang_input: int = 4096,
                                  n_motor_per_action: int = 1000,
                                  n_motor_fs_per_action: int = 120,
                                  verbose: bool = True,
                                  temperature: float = 0.0,
                                  enable_stp: bool = False,
                                  reenable_stp_for_eval: bool = False,
                                  lineage_name: str = None,
                                  save_to_lineage: bool = False) -> dict:
    """2026-05-10: enable_stp default flipped from True to False.
    3-seed validation showed 3.28x speedup AND higher accuracy with
    STP off. See research/findings/2026-05-10-stp-default-flip.md.

    reenable_stp_for_eval (2026-05-10 user-requested test): if True,
    after training completes (STP off, fast), enable STP at runtime
    for the W2A + A2W eval phases. Tests if STP-off training is
    'reversible at inference time' — i.e., do the weights trained
    fast still work with biological STP dynamics restored at eval?
    If yes, we get best-of-both-worlds: fast training + biology at
    inference."""
    """Tier 2.1 8-word :speak demo: train scale-up bridge, then A->W."""
    # Structured progress events for live webapp + brain3d
    from sim.progress import emit_progress

    print(f"\n=== chat_speak_synonym_demo (seed={seed}) ===", flush=True)
    print(f"  Tier 2.1 8-word vocab: {SYNONYM_GROUPS}", flush=True)
    print(f"  arch: n_lang_input={n_lang_input} n_motor={n_motor_per_action} "
          f"n_motor_fs={n_motor_fs_per_action}", flush=True)
    print(f"  train_events/word={n_train_events}\n", flush=True)

    # 3 phases: training (or loading), W2A regression, A2W synonym speak
    # Lineage auto-load when requested (opt-in via lineage_name)
    lineage = None
    used_lineage_load = False
    if lineage_name:
        from sim.lineage import BridgeLineage
        lineage = BridgeLineage(lineage_name)
        if lineage.exists():
            try:
                lm = lineage.read_metadata()
                stored_mode = (lm.arch or {}).get("mode")
                if stored_mode and stored_mode != "synonym":
                    if verbose:
                        print(f"[LINEAGE] '{lineage_name}' was trained "
                              f"in mode={stored_mode}; chat_speak_synonym_demo "
                              f"is synonym. Falling back to fresh training.",
                              flush=True)
                else:
                    used_lineage_load = True
            except Exception as e:
                if verbose:
                    print(f"[LINEAGE] could not read metadata: {e}",
                          flush=True)

    if used_lineage_load:
        if verbose:
            print(f"[LINEAGE] Loading state from '{lineage_name}'",
                  flush=True)
        emit_progress("phase", current=0, total=3, phase="loading",
                      unit="phases", label="chat_speak_synonym_demo")
        t0 = time.time()
        from research.runners.bio_three_factor import run_three_factor
        bridge, _ = run_three_factor(
            seed=seed,
            n_events_per_direction=0,
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            n_motor_fs_per_action=n_motor_fs_per_action,
            biological=True,
            enable_motor_fs=True,
            enable_nmda=True,
            apply_topographic_bias=True,
            embodied_hebbian=True,
            synonym_mode=True,
            synonym_vocab_size=8,
            verbose=False,
            enable_stp=enable_stp,
        )
        bridge.load_checkpoint(str(lineage.current_path))
        try:
            bridge.set_plasticity_gate("language_input_to_motor", 0.0)
            bridge.set_plasticity_gate("motor_to_language_output", 0.0)
        except Exception:
            pass
        train_sec = time.time() - t0
        emit_progress("complete", current=1, total=3, phase="loading",
                      unit="phases", label="chat_speak_synonym_demo",
                      wall_clock_s=int(train_sec))
    else:
        emit_progress("phase", current=0, total=3, phase="training",
                      unit="phases", label="chat_speak_synonym_demo")
        t0 = time.time()
        bridge = train_chat_bridge(
            seed=seed,
            n_events_per_word=n_train_events,
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            n_motor_fs_per_action=n_motor_fs_per_action,
            verbose=verbose,
            enable_stp=enable_stp,
        )
        train_sec = time.time() - t0
        emit_progress("complete", current=1, total=3, phase="training",
                      unit="phases", label="chat_speak_synonym_demo",
                      wall_clock_s=int(train_sec))
        if save_to_lineage and lineage is not None:
            try:
                lineage.save(bridge, tier="8-word",
                              arch={"mode": "synonym",
                                    "n_neurons": int(getattr(
                                        bridge.core_sim_config,
                                        "num_neurons", 0)),
                                    "n_lang_input": n_lang_input,
                                    "n_motor_per_action": n_motor_per_action})
                meta = lineage.read_metadata()
                meta.cumulative_training_events += int(n_train_events or 0)
                meta.add_growth_event(
                    kind="init",
                    description=(
                        f"chat_speak_synonym_demo train (seed={seed}, "
                        f"n_train_events={n_train_events})"
                    ),
                    seed=seed,
                )
                lineage.write_metadata(meta)
                if verbose:
                    print(f"[LINEAGE] Saved trained state to '{lineage_name}'",
                          flush=True)
            except Exception as e:
                if verbose:
                    print(f"[LINEAGE] Save failed (non-fatal): {e}",
                          flush=True)

    # Optional: re-enable STP for eval (test reversibility of STP-off training)
    if reenable_stp_for_eval:
        if verbose:
            print(f"\n[REENABLE-STP] Activating short-term plasticity for eval "
                  f"(was {'on' if enable_stp else 'off'} during training)",
                  flush=True)
        newly = bridge.enable_stp_runtime()
        if verbose:
            print(f"  cp_stp_x newly_allocated: {newly}", flush=True)

    # Phase B: W->A regression baseline on 8 words
    print(f"\n[PHASE B] W->A regression on 8-word synonym vocab",
          flush=True)
    emit_progress("phase", current=1, total=3, phase="W2A_synonym_regression",
                  unit="phases", label="chat_speak_synonym_demo")
    w2a = evaluate_w_to_a_baseline_synonym(bridge, n_rounds=2,
                                            verbose=verbose)
    print(f"  W->A accuracy: {w2a['accuracy']:.1%} "
          f"({w2a['correct']}/{w2a['total']})", flush=True)
    emit_progress("complete", current=2, total=3, phase="W2A_synonym_regression",
                  unit="phases", label="chat_speak_synonym_demo",
                  score=float(w2a['accuracy']))

    # Phase C: A->W via generative_inference (now ranking 8 words, not 4)
    print(f"\n[PHASE C] A->W generative decoder (:speak on 8-word vocab)",
          flush=True)
    emit_progress("phase", current=2, total=3, phase="A2W_synonym_speak",
                  unit="phases", label="chat_speak_synonym_demo")
    # Pass temperature through; rng_seed=seed for reproducible sampling
    # when temperature > 0 (each seed's :speak still varies but is repeatable)
    a2w = evaluate_a_to_w_synonym(
        bridge, verbose=verbose,
        temperature=temperature,
        rng_seed=(seed if temperature > 0 else None),
    )
    print(f"  A->W any-synonym accuracy: {a2w['any_accuracy']:.1%} "
          f"({a2w['any_correct']}/{a2w['total']})", flush=True)
    print(f"  A->W primary accuracy:    {a2w['primary_accuracy']:.1%}",
          flush=True)
    print(f"  A->W synonym accuracy:    {a2w['synonym_accuracy']:.1%}",
          flush=True)
    emit_progress("complete", current=3, total=3, phase="A2W_synonym_speak",
                  unit="phases", label="chat_speak_synonym_demo",
                  score=float(a2w['any_accuracy']))

    # Verdict: GO if A->W any-synonym >= 50% AND W->A regression >= 25%
    speak_pass = a2w["any_accuracy"] >= 0.50
    regression_intact = w2a["accuracy"] >= 0.25
    go = speak_pass and regression_intact
    verdict = "GO" if go else "NO-GO"
    print(f"\n=== verdict: {verdict} ===", flush=True)
    print(f"  W->A regression (>= 25% chance, 8-word): "
          f"{'PASS' if regression_intact else 'FAIL'} ({w2a['accuracy']:.1%})",
          flush=True)
    print(f"  A->W any-synonym (>= 50%): "
          f"{'PASS' if speak_pass else 'FAIL'} ({a2w['any_accuracy']:.1%})",
          flush=True)
    print(f"  total wall clock: {time.time() - t0:.0f}s", flush=True)

    return {
        "seed": seed,
        "demo_kind": "chat_speak_synonym_demo",
        "vocab_size": 8,
        "n_train_events": n_train_events,
        "n_lang_input": n_lang_input,
        "n_motor_per_action": n_motor_per_action,
        "n_motor_fs_per_action": n_motor_fs_per_action,
        # Match chat_demo schema for chat_demo_aggregate:
        "accuracy": w2a["accuracy"],
        "correct": w2a["correct"],
        "total": w2a["total"],
        "per_word_accuracy": w2a["per_word_accuracy"],
        # Layer 4 :speak (synonym variant): three accuracy metrics
        "speak_accuracy": a2w["any_accuracy"],         # any-synonym (headline)
        "speak_primary_accuracy": a2w["primary_accuracy"],
        "speak_synonym_accuracy": a2w["synonym_accuracy"],
        "speak_correct": a2w["any_correct"],
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
    ap.add_argument("--train-events", type=int, default=400,
                    help="Events per word during initial training "
                         "(Tier 2.1 default: 400)")
    ap.add_argument("--n-lang-input", type=int, default=4096,
                    help="Language input neuron count (Tier 2.1 default: 4096)")
    ap.add_argument("--n-motor-per-action", type=int, default=1000,
                    help="Motor neurons per action (Tier 2.1 default: 1000)")
    ap.add_argument("--n-motor-fs-per-action", type=int, default=120,
                    help="Motor FS interneurons per action "
                         "(Tier 2.1 default: 120)")
    ap.add_argument("--out-stats", type=str, default=None,
                    help="JSON stats output path (matches chat_demo schema)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-turn logging")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Softmax sampling temperature for :speak. "
                         "0 (default) = deterministic argmax, "
                         "matches all prior multi-seed validations. "
                         "0.01-0.02 = 'primary dominant with synonym lift', "
                         "0.05+ = 'lots of variety, primary slightly preferred'. "
                         "Use 0 for reproducible benchmarking; >0 for "
                         "natural-feeling user-facing chat.")
    ap.add_argument("--no-stp", action="store_true",
                    help="Disable short-term plasticity. Per 2026-05-10 "
                         "perf benchmark, STP is 57%% of inner-loop step "
                         "time; disabling gives ~2.86x speedup. Default "
                         "OFF (i.e. STP enabled) until multi-seed accuracy "
                         "validated. Use this flag for the validation arc.")
    ap.add_argument("--reenable-stp-for-eval", action="store_true",
                    help="After training completes (with STP off, fast), "
                         "re-enable STP for the W->A + A->W eval phases. "
                         "Tests if STP-off training is reversible at "
                         "inference: do weights trained fast still work "
                         "with biological STP dynamics restored? If yes, "
                         "we get best-of-both-worlds: fast training + "
                         "biology at inference.")
    # Lineage (opt-in)
    ap.add_argument("--lineage", type=str, default=None,
                    help="Optional lineage NAME under bridges/lineage/. "
                         "If exists with matching synonym arch, load and "
                         "skip training. Default: None (always train).")
    ap.add_argument("--save-to-lineage", action="store_true",
                    help="After training, save trained bridge to lineage "
                         "(creates it if needed). Requires --lineage.")
    args = ap.parse_args()

    result = run_chat_speak_synonym_demo(
        seed=args.seed,
        n_train_events=args.train_events,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        verbose=not args.quiet,
        temperature=args.temperature,
        enable_stp=not args.no_stp,
        reenable_stp_for_eval=args.reenable_stp_for_eval,
        lineage_name=args.lineage,
        save_to_lineage=args.save_to_lineage,
    )

    if args.out_stats:
        out_path = Path(args.out_stats)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[STATS] {out_path}", flush=True)


if __name__ == "__main__":
    main()
