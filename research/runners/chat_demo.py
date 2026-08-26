"""Chat demo on biology-grounded Phase 1.4 BRANCH A foundation.

The first conversational artifact built on the validated
biology-grounded continual-learning architecture. Demonstrates:

- Train a 4-word vocab (north/east/south/west) via Tier 1
  embodied Hebbian co-firing
- User types a word; sim predicts the motor action via
  language_input -> motor cortex pathway
- All learning is biology-grounded (STDP + embodied co-firing,
  no backprop)

Per master plan + Phase 1.4 BRANCH A validation (5/6 PASS, mean
103% retention) and Phase 1.3 consolidation (3/3 PASS, mean 96%
hippo-OFF retention). This demo shows the END-TO-END working
capability.

Scope: 4 direction words, 1 turn per direction. Scripted
simulated user inputs; produces a transcript-style output.

Future:
- Add Tier 2.1 synonyms (8/12 word vocab)
- Add interactive (REPL) mode
- Add consolidation cycles (Phase 1.3) for "memory across sessions"

Usage:
    python -m research.runners.chat_demo \\
        --seed 42 --train-events 200 \\
        --transcript-out research/findings/2026-05-07-chat-demo.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np


DIRECTIONS = ["north", "east", "south", "west"]
DIRECTION_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}
ACTION_TO_DIRECTION = {a: d for d, a in DIRECTION_TO_ACTION.items()}


def train_chat_bridge(
    seed: int = 42,
    n_events_per_word: int = 200,
    n_lang_input: int = 2048,
    n_motor_per_action: int = 500,
    n_motor_fs_per_action: int = 60,
    verbose: bool = True,
):
    """Train Bridge with Tier 1 BREAKTHROUGH config for chat demo."""
    from research.runners.bio_three_factor import run_three_factor

    if verbose:
        print(f"[TRAINING] Tier 1 architecture (seed={seed})")
        print(f"  4-word vocab: {DIRECTIONS}")
        print(f"  n_events/word: {n_events_per_word}")
        print(f"  n_lang_input: {n_lang_input}, n_motor: {n_motor_per_action}")
        t0 = time.time()

    bridge, _ = run_three_factor(
        seed=seed,
        n_events_per_direction=n_events_per_word,
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=False,
        verbose=False,
    )

    if verbose:
        print(f"  Training complete ({time.time() - t0:.0f}s)\n")

    # Freeze plasticity for chat eval
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    return bridge


def chat_turn(
    bridge,
    user_word: str,
    stim_steps: int = 100,
    reset_steps: int = 50,
    drive_pA: float = 200.0,
    sparsity: float = 0.1,
):
    """Single chat turn: user types a word, sim predicts motor action.

    Uses BASELINE-VS-DRIVEN delta methodology (matches the validated
    Phase 1.4 BRANCH A evaluate_word_to_action):
    1. Phase A baseline: reset, run with NO input, count motor spikes
       (this is the cascade-driven background activity per pool)
    2. Phase B driven: reset, drive language_input, count motor spikes
    3. Delta = driven - baseline. argmax(delta) = predicted action.

    Without baseline subtraction, cascade asymmetry would dominate
    the prediction (e.g., motor_S has higher baseline = always wins).

    Returns dict with motor delta counts, predicted action, confidence.
    """
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}"))
                 for a in ["N", "E", "S", "W"]}
    motor_arr = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                 for a in ["N", "E", "S", "W"]}
    n_lang_in = len(lang_input_idx)
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)

    # ─── Phase A: BASELINE (no input) ───
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    baseline_counts = cp.zeros(4, dtype=cp.int32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for a_i, a in enumerate(["N", "E", "S", "W"]):
            baseline_counts[a_i] += fired[motor_arr[a]].sum()

    # ─── Phase B: DRIVEN (word input) ───
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    drive = vocab_to_drive_pattern(
        user_word, n_neurons=n_lang_in,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    bridge.cp_external_input_current[lang_input_arr] = \
        cp.asarray(drive, dtype=cp.float32)
    drive_counts = cp.zeros(4, dtype=cp.int32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for a_i, a in enumerate(["N", "E", "S", "W"]):
            drive_counts[a_i] += fired[motor_arr[a]].sum()

    # Delta = driven - baseline; argmax of delta = prediction
    bl = baseline_counts.get()
    dr = drive_counts.get()
    delta = dr - bl
    predicted_idx = int(np.argmax(delta))
    predicted_action = ["N", "E", "S", "W"][predicted_idx]
    predicted_direction = ACTION_TO_DIRECTION[predicted_action]

    # Confidence: ratio of winner delta to runner-up delta
    sorted_delta = np.sort(delta)[::-1]
    if sorted_delta[1] > 0:
        confidence = float(sorted_delta[0] / sorted_delta[1])
    else:
        confidence = float("inf") if sorted_delta[0] > 0 else 1.0

    return {
        "user_word": user_word,
        "baseline_counts": {a: int(bl[i])
                             for i, a in enumerate(["N", "E", "S", "W"])},
        "drive_counts": {a: int(dr[i])
                          for i, a in enumerate(["N", "E", "S", "W"])},
        "delta_counts": {a: int(delta[i])
                          for i, a in enumerate(["N", "E", "S", "W"])},
        "predicted_action": predicted_action,
        "predicted_direction": predicted_direction,
        "confidence_ratio": confidence,
        "correct": (predicted_direction == user_word),
    }


def run_demo(
    seed: int = 42,
    n_train_events: int = 200,
    transcript_out: str = None,
    verbose: bool = True,
    lineage_name: str = None,
    save_to_lineage: bool = False,
):
    """Full demo: train (or load from lineage), then 4-direction conversation.

    If ``lineage_name`` is given and the lineage exists with a matching
    tier1 arch, training is skipped and the lineage's saved state is
    loaded instead — useful for "demo on the evolving sim" workflow vs
    the default per-seed-from-scratch validation workflow.

    If ``save_to_lineage`` is True, the trained bridge is saved back to
    the lineage (creating it if it didn't exist). Compatible with
    cumulative metadata tracking.
    """
    # 2026-05-09: emit_progress so webapp inflight panel + 3D viz show
    # live progress (was "0% no markers")
    from sim.progress import emit_progress
    import time

    # Try lineage auto-load when requested
    lineage = None
    used_lineage_load = False
    if lineage_name:
        from sim.lineage import BridgeLineage
        lineage = BridgeLineage(lineage_name)
        if lineage.exists():
            try:
                lm = lineage.read_metadata()
                stored_mode = (lm.arch or {}).get("mode")
                if stored_mode and stored_mode != "tier1":
                    if verbose:
                        print(f"[LINEAGE] '{lineage_name}' was trained "
                              f"in mode={stored_mode}; chat_demo is tier1. "
                              f"Falling back to fresh training.", flush=True)
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
        emit_progress("phase", current=0, total=2, phase="loading",
                      unit="phases", label="chat_demo")
        t0 = time.time()
        # Rebuild the bridge skeleton (no training) then overlay weights
        from research.runners.bio_three_factor import run_three_factor
        bridge, _ = run_three_factor(
            seed=seed,
            n_events_per_direction=0,  # build only, no training
            n_lang_input=2048,
            n_motor_per_action=500,
            n_motor_fs_per_action=60,
            biological=True,
            enable_motor_fs=True,
            enable_nmda=True,
            apply_topographic_bias=True,
            embodied_hebbian=True,
            synonym_mode=False,
            verbose=False,
        )
        bridge.load_checkpoint(str(lineage.current_path))
        try:
            bridge.set_plasticity_gate("language_input_to_motor", 0.0)
            bridge.set_plasticity_gate("motor_to_language_output", 0.0)
        except Exception:
            pass
        emit_progress("complete", current=1, total=2, phase="loading",
                      unit="phases", label="chat_demo",
                      wall_clock_s=int(time.time() - t0))
    else:
        emit_progress("phase", current=0, total=2, phase="training",
                      unit="phases", label="chat_demo")
        t0 = time.time()
        bridge = train_chat_bridge(
            seed=seed, n_events_per_word=n_train_events, verbose=verbose,
        )
        emit_progress("complete", current=1, total=2, phase="training",
                      unit="phases", label="chat_demo",
                      wall_clock_s=int(time.time() - t0))
        # Save to lineage if requested (post-training)
        if save_to_lineage and lineage is not None:
            try:
                lineage.save(bridge, tier="4-word",
                              arch={"mode": "tier1",
                                    "n_neurons": int(getattr(
                                        bridge.core_sim_config,
                                        "num_neurons", 0)),
                                    "n_lang_input": 2048,
                                    "n_motor_per_action": 500})
                meta = lineage.read_metadata()
                meta.cumulative_training_events += int(n_train_events or 0)
                meta.add_growth_event(
                    kind="init",
                    description=(
                        f"chat_demo train (seed={seed}, "
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

    # Run "conversation" -- 1 turn per direction, repeated 3 times = 12 turns
    emit_progress("phase", current=1, total=2, phase="W2A_eval",
                  unit="phases", label="chat_demo")
    transcript = []
    transcript.append({"type": "header",
                        "text": "Chat demo on biology-grounded foundation"})
    transcript.append({"type": "system",
                        "text": f"Trained 4-word vocab via Tier 1 embodied "
                                f"Hebbian (seed={seed}, "
                                f"{n_train_events} events/word)."})

    correct_total = 0
    total_turns = 0
    n_total_turns = 12  # 3 rounds × 4 directions
    # Per-direction tracking (added 2026-05-07): enables aggregator to
    # surface which directions are well-bound vs not at the seed level.
    correct_per_word = {w: 0 for w in DIRECTIONS}
    total_per_word = {w: 0 for w in DIRECTIONS}
    for round_n in range(1, 4):
        transcript.append({"type": "section",
                            "text": f"Round {round_n}/3"})
        for word in DIRECTIONS:
            result = chat_turn(bridge, word)
            total_turns += 1
            total_per_word[word] += 1
            if result["correct"]:
                correct_total += 1
                correct_per_word[word] += 1
            # Per-turn eval progress (every turn so live progress is smooth)
            emit_progress("eval", current=total_turns, total=n_total_turns,
                          phase="W2A_eval", unit="turns",
                          label="chat_demo",
                          correct_pct=round(100 * correct_total / total_turns, 1))

            if verbose:
                marker = "[OK]" if result["correct"] else "[X]"
                d = result["delta_counts"]
                print(f"  {marker} You: {word} -> Sim: "
                      f"{result['predicted_direction']} "
                      f"(delta N{d['N']:+d} E{d['E']:+d} S{d['S']:+d} "
                      f"W{d['W']:+d}, confidence x{result['confidence_ratio']:.1f})")
            transcript.append({
                "type": "turn",
                "round": round_n,
                "result": result,
            })

    accuracy = correct_total / total_turns
    per_word_accuracy = {
        w: correct_per_word[w] / total_per_word[w] if total_per_word[w] else 0.0
        for w in DIRECTIONS
    }

    if verbose:
        print(f"\n[DEMO] Accuracy: {correct_total}/{total_turns} = "
              f"{accuracy:.1%}")
        print("[DEMO] Per-direction:", "  ".join(
            f"{w}: {correct_per_word[w]}/{total_per_word[w]}"
            for w in DIRECTIONS))

    transcript.append({
        "type": "summary",
        "accuracy": accuracy,
        "correct": correct_total,
        "total": total_turns,
        "per_word_accuracy": per_word_accuracy,
        "correct_per_word": correct_per_word,
        "total_per_word": total_per_word,
    })

    if transcript_out:
        write_transcript_md(transcript, transcript_out, seed=seed,
                             n_train_events=n_train_events)

    return {
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct_total,
        "total": total_turns,
        "per_word_accuracy": per_word_accuracy,
        "correct_per_word": correct_per_word,
        "total_per_word": total_per_word,
        "transcript": transcript,
    }


def write_transcript_md(transcript: List[dict], path: str,
                          seed: int, n_train_events: int):
    """Render transcript as markdown."""
    md = []
    md.append("# Chat demo on biology-grounded Phase 1.4 BRANCH A foundation\n")
    md.append(f"**Seed:** {seed}\n")
    md.append(f"**Training:** Tier 1 embodied Hebbian, "
              f"{n_train_events} events/word\n\n")
    md.append("---\n\n")
    md.append("## Conversation transcript\n\n")
    md.append("```\n")
    for entry in transcript:
        if entry["type"] == "header":
            continue
        elif entry["type"] == "system":
            md.append(f"[SYSTEM] {entry['text']}\n\n")
        elif entry["type"] == "section":
            md.append(f"--- {entry['text']} ---\n")
        elif entry["type"] == "turn":
            r = entry["result"]
            marker = "[OK]" if r["correct"] else "[X]"
            d = r["delta_counts"]
            md.append(
                f"  {marker} You: {r['user_word']:<6} -> "
                f"Sim: {r['predicted_direction']:<6} "
                f"(delta N{d['N']:+4d} E{d['E']:+4d} S{d['S']:+4d} "
                f"W{d['W']:+4d}, x{r['confidence_ratio']:.1f})\n"
            )
        elif entry["type"] == "summary":
            md.append(f"\nAccuracy: {entry['correct']}/{entry['total']} "
                      f"= {entry['accuracy']:.1%}\n")
    md.append("```\n\n")
    md.append("---\n\n")
    md.append("## What this demonstrates\n\n")
    md.append("- Tier 1 embodied Hebbian binding (Phase 1.4 architecture)\n")
    md.append("- All learning biology-grounded: STDP + co-firing teachers\n")
    md.append("- No backprop, no surrogate gradients\n")
    md.append("- 4-word vocabulary, scriptable to 8/12 with Tier 2.1 synonym mode\n")
    md.append("- Continual learning preserved (Phase 1.4 BRANCH A: 5/6 PASS, "
              "mean 103% retention)\n")
    md.append("- Memory consolidation works (Phase 1.3: 3/3 PASS, "
              "mean 96% hippo-OFF retention)\n")
    md.append("\n")
    md.append("First conversational artifact built on the validated "
              "biology-grounded continual-learning + memory consolidation "
              "foundation.\n")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=200,
                    help="Tier 1 events per word")
    ap.add_argument("--transcript-out", type=str, default=None)
    ap.add_argument("--out-stats", type=str, default=None)
    # ── Lineage (continual-learning) opt-in ────────────────────────
    # Unlike chat_repl, batch demos default to from-scratch (seed-
    # specific validation). Pass --lineage NAME to load/save state.
    ap.add_argument("--lineage", type=str, default=None,
                    help="Optional lineage NAME under bridges/lineage/. "
                         "If the lineage exists with matching arch, load "
                         "and skip training. Default: None (always train).")
    ap.add_argument("--save-to-lineage", action="store_true",
                    help="After training (and only if --lineage given), "
                         "save the trained bridge to that lineage so "
                         "future demos can load it. Creates the lineage "
                         "if it doesn't exist.")
    args = ap.parse_args()

    result = run_demo(
        seed=args.seed,
        n_train_events=args.train_events,
        transcript_out=args.transcript_out,
        verbose=True,
        lineage_name=args.lineage,
        save_to_lineage=args.save_to_lineage,
    )

    if args.out_stats:
        # Strip transcript dict objects for JSON cleanliness
        clean = {
            "seed": result["seed"],
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(clean, indent=2))
        print(f"\nSaved stats: {args.out_stats}")


if __name__ == "__main__":
    main()
