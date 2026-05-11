"""Chat demo for Tier 2.1 BREAKTHROUGH (8-word synonym binding).

Demonstrates that the biology-grounded architecture binds SYNONYMS to
the same motor action: type "north" OR "up" -> motor_N activates.
Type "east" OR "right" -> motor_E. Etc.

Built on the validated Tier 2.1 6-seed result:
  - W->A 5/6 aligned, A->W 6/6 aligned
  - A->W mean 63.7% (BEATS Tier 1 mean 45%)
  - Architecture: n_lang_input=4096, n_motor_per_action=1000,
    n_motor_fs_per_action=120 (~12K neurons, ~5M synapses, ~6GB GPU)
  - 400 events/word training (~15-20 min single seed RTX 3090; varies
    with 16-turn eval phase)

See: research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md

Scope: 8 words ({north,up}, {east,right}, {south,down}, {west,left}),
2 turns per word per round, 2 rounds = 32 turns. Uses the same
baseline-vs-driven delta methodology as chat_demo.

Usage:
    python -m research.runners.chat_synonym_demo \\
        --seed 42 --train-events 400 \\
        --transcript-out research/findings/2026-05-07-chat-synonym-demo.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np


# Tier 2.1 8-word vocab (matches research.runners.text_eval.get_synonym_groups)
SYNONYM_GROUPS = {
    "N": ["north", "up"],
    "E": ["east", "right"],
    "S": ["south", "down"],
    "W": ["west", "left"],
}
ALL_WORDS = [w for syns in SYNONYM_GROUPS.values() for w in syns]
WORD_TO_ACTION = {w: a for a, syns in SYNONYM_GROUPS.items() for w in syns}
ACTION_TO_PRIMARY = {a: syns[0] for a, syns in SYNONYM_GROUPS.items()}


def train_chat_bridge(
    seed: int = 42,
    n_events_per_word: int = 400,
    n_lang_input: int = 4096,
    n_motor_per_action: int = 1000,
    n_motor_fs_per_action: int = 120,
    verbose: bool = True,
    enable_stp: bool = False,  # 2026-05-10: flipped to False
                                  # after 3-seed validation. See
                                  # research/findings/2026-05-10-stp-default-flip.md
):
    """Train Bridge with Tier 2.1 v4 scale-up config for synonym chat demo."""
    from research.runners.bio_three_factor import run_three_factor

    if verbose:
        print(f"[TRAINING] Tier 2.1 scale-up architecture (seed={seed})")
        print(f"  8-word synonym vocab: {SYNONYM_GROUPS}")
        print(f"  n_events/word: {n_events_per_word}")
        print(f"  n_lang_input: {n_lang_input}, n_motor: {n_motor_per_action}, "
              f"n_motor_fs: {n_motor_fs_per_action}")
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
        synonym_mode=True,  # Tier 2.1 — both primary + synonym presented
        synonym_vocab_size=8,
        verbose=False,
        enable_stp=enable_stp,
    )

    if verbose:
        print(f"  Training complete ({time.time() - t0:.0f}s)\n")

    # Freeze plasticity for chat eval (matches Phase 1.4 BRANCH A protocol)
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
    """Single chat turn. Same baseline-vs-driven delta as chat_demo.

    Returns dict with motor delta counts, predicted action, predicted
    primary direction word, confidence ratio, and correctness.
    """
    # Backend-aware: cp is the active backend (cupy on CuPy, numpy on NumPy)
    from sim.backend import get_backend
    cp, _ = get_backend()
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

    # Backend-aware D->H transfer (passthrough on NumPy)
    from sim.backend import to_host as _bl_to_host
    bl = _bl_to_host(baseline_counts)
    dr = _bl_to_host(drive_counts)
    delta = dr - bl
    predicted_idx = int(np.argmax(delta))
    predicted_action = ["N", "E", "S", "W"][predicted_idx]
    predicted_direction = ACTION_TO_PRIMARY[predicted_action]

    sorted_delta = np.sort(delta)[::-1]
    if sorted_delta[1] > 0:
        confidence = float(sorted_delta[0] / sorted_delta[1])
    else:
        confidence = float("inf") if sorted_delta[0] > 0 else 1.0

    # "Correct" means the predicted action matches the user's word's action.
    # Both "north" and "up" map to action N, so either is correct on motor_N.
    expected_action = WORD_TO_ACTION[user_word]
    correct = (predicted_action == expected_action)

    return {
        "user_word": user_word,
        "expected_action": expected_action,
        "baseline_counts": {a: int(bl[i])
                             for i, a in enumerate(["N", "E", "S", "W"])},
        "drive_counts": {a: int(dr[i])
                          for i, a in enumerate(["N", "E", "S", "W"])},
        "delta_counts": {a: int(delta[i])
                          for i, a in enumerate(["N", "E", "S", "W"])},
        "predicted_action": predicted_action,
        "predicted_direction": predicted_direction,
        "confidence_ratio": confidence,
        "correct": correct,
    }


def run_demo(
    seed: int = 42,
    n_train_events: int = 400,
    transcript_out: str = None,
    verbose: bool = True,
    lineage_name: str = None,
    save_to_lineage: bool = False,
):
    """Full Tier 2.1 synonym demo: train (or load from lineage), then 8-word conversation.

    Lineage usage (opt-in via ``lineage_name``):
    - If the named lineage exists with a matching synonym arch, load
      state instead of training (skip ~6-10 min).
    - If ``save_to_lineage``, save the trained bridge back to the
      lineage (creating it if necessary).
    """
    # 2026-05-09: emit_progress for live frontend visibility
    from sim.progress import emit_progress
    import time as _time

    # Lineage auto-load when requested
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
                              f"in mode={stored_mode}; chat_synonym_demo "
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
        emit_progress("phase", current=0, total=2, phase="loading",
                      unit="phases", label="chat_synonym_demo")
        _t0 = _time.time()
        from research.runners.bio_three_factor import run_three_factor
        bridge, _ = run_three_factor(
            seed=seed,
            n_events_per_direction=0,  # build only
            n_lang_input=4096,
            n_motor_per_action=1000,
            n_motor_fs_per_action=120,
            biological=True,
            enable_motor_fs=True,
            enable_nmda=True,
            apply_topographic_bias=True,
            embodied_hebbian=True,
            synonym_mode=True,
            synonym_vocab_size=8,
            verbose=False,
            enable_stp=False,
        )
        bridge.load_checkpoint(str(lineage.current_path))
        try:
            bridge.set_plasticity_gate("language_input_to_motor", 0.0)
            bridge.set_plasticity_gate("motor_to_language_output", 0.0)
        except Exception:
            pass
        emit_progress("complete", current=1, total=2, phase="loading",
                      unit="phases", label="chat_synonym_demo",
                      wall_clock_s=int(_time.time() - _t0))
    else:
        emit_progress("phase", current=0, total=2, phase="training",
                      unit="phases", label="chat_synonym_demo")
        _t0 = _time.time()
        bridge = train_chat_bridge(
            seed=seed, n_events_per_word=n_train_events, verbose=verbose,
        )
        emit_progress("complete", current=1, total=2, phase="training",
                      unit="phases", label="chat_synonym_demo",
                      wall_clock_s=int(_time.time() - _t0))
        if save_to_lineage and lineage is not None:
            try:
                lineage.save(bridge, tier="8-word",
                              arch={"mode": "synonym",
                                    "n_neurons": int(getattr(
                                        bridge.core_sim_config,
                                        "num_neurons", 0)),
                                    "n_lang_input": 4096,
                                    "n_motor_per_action": 1000})
                meta = lineage.read_metadata()
                meta.cumulative_training_events += int(n_train_events or 0)
                meta.add_growth_event(
                    kind="init",
                    description=(
                        f"chat_synonym_demo train (seed={seed}, "
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

    emit_progress("phase", current=1, total=2, phase="W2A_synonym_eval",
                  unit="phases", label="chat_synonym_demo")
    transcript = []
    transcript.append({"type": "header",
                        "text": "Chat synonym demo on Tier 2.1 BREAKTHROUGH"})
    transcript.append({"type": "system",
                        "text": f"Trained 8-word synonym vocab via Tier 2.1 "
                                f"v4 scale-up arch (seed={seed}, "
                                f"{n_train_events} events/word)."})

    correct_per_action = {"N": 0, "E": 0, "S": 0, "W": 0}
    total_per_action = {"N": 0, "E": 0, "S": 0, "W": 0}
    primary_correct = 0
    primary_total = 0
    synonym_correct = 0
    synonym_total = 0
    _turn_idx = 0
    _n_total_turns = 2 * len(ALL_WORDS)  # 16 turns total

    for round_n in range(1, 3):
        transcript.append({"type": "section",
                            "text": f"Round {round_n}/2 (all 8 words)"})
        for word in ALL_WORDS:
            result = chat_turn(bridge, word)
            a = result["expected_action"]
            total_per_action[a] += 1
            if result["correct"]:
                correct_per_action[a] += 1
            _turn_idx += 1
            emit_progress("eval", current=_turn_idx, total=_n_total_turns,
                          phase="W2A_synonym_eval", unit="turns",
                          label="chat_synonym_demo")

            is_primary = (word == ACTION_TO_PRIMARY[a])
            if is_primary:
                primary_total += 1
                if result["correct"]:
                    primary_correct += 1
            else:
                synonym_total += 1
                if result["correct"]:
                    synonym_correct += 1

            if verbose:
                marker = "[OK]" if result["correct"] else "[X]"
                d = result["delta_counts"]
                tag = "PRI" if is_primary else "SYN"
                print(f"  {marker} You: {word:<6} ({tag}) -> Sim: motor_"
                      f"{result['predicted_action']} "
                      f"(delta N{d['N']:+d} E{d['E']:+d} S{d['S']:+d} "
                      f"W{d['W']:+d}, x{result['confidence_ratio']:.1f})")
            transcript.append({
                "type": "turn",
                "round": round_n,
                "is_primary": is_primary,
                "result": result,
            })

    total_correct = sum(correct_per_action.values())
    total_turns = sum(total_per_action.values())
    accuracy = total_correct / total_turns
    primary_acc = (primary_correct / primary_total) if primary_total else 0.0
    synonym_acc = (synonym_correct / synonym_total) if synonym_total else 0.0

    if verbose:
        print(f"\n[DEMO] Overall accuracy: {total_correct}/{total_turns} = "
              f"{accuracy:.1%}")
        print(f"  Primary words: {primary_correct}/{primary_total} = "
              f"{primary_acc:.1%}")
        print(f"  Synonym words: {synonym_correct}/{synonym_total} = "
              f"{synonym_acc:.1%}")
        print(f"  Per-action: " + "  ".join(
            f"motor_{a}: {correct_per_action[a]}/{total_per_action[a]}"
            for a in ["N", "E", "S", "W"]))

    transcript.append({
        "type": "summary",
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_turns,
        "primary_accuracy": primary_acc,
        "synonym_accuracy": synonym_acc,
        "per_action": {a: {"correct": correct_per_action[a],
                            "total": total_per_action[a]}
                       for a in ["N", "E", "S", "W"]},
    })

    if transcript_out:
        write_transcript_md(transcript, transcript_out, seed=seed,
                             n_train_events=n_train_events)

    return {
        "seed": seed,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_turns,
        "primary_accuracy": primary_acc,
        "synonym_accuracy": synonym_acc,
        "per_action_correct": correct_per_action,
        "per_action_total": total_per_action,
        "transcript": transcript,
    }


def write_transcript_md(transcript: List[dict], path: str,
                          seed: int, n_train_events: int):
    """Render synonym chat transcript as markdown."""
    md = []
    md.append("# Chat synonym demo on Tier 2.1 BREAKTHROUGH foundation\n\n")
    md.append(f"**Seed:** {seed}  \n")
    md.append(f"**Training:** Tier 2.1 v4 scale-up "
              f"(n_lang=4096, n_motor=1000, n_motor_fs=120, "
              f"{n_train_events} events/word)\n\n")
    md.append("**8-word vocab:** "
              "{north,up}, {east,right}, {south,down}, {west,left}\n\n")
    md.append("---\n\n## Conversation transcript\n\n")
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
            tag = "PRI" if entry["is_primary"] else "SYN"
            md.append(
                f"  {marker} You: {r['user_word']:<6} ({tag}) -> "
                f"Sim: motor_{r['predicted_action']:<2} "
                f"(delta N{d['N']:+4d} E{d['E']:+4d} S{d['S']:+4d} "
                f"W{d['W']:+4d}, x{r['confidence_ratio']:.1f})\n"
            )
        elif entry["type"] == "summary":
            md.append(f"\nOverall accuracy: {entry['correct']}/{entry['total']} "
                      f"= {entry['accuracy']:.1%}\n")
            md.append(f"Primary words: {entry['primary_accuracy']:.1%}\n")
            md.append(f"Synonym words: {entry['synonym_accuracy']:.1%}\n")
    md.append("```\n\n")
    md.append("---\n\n## What this demonstrates\n\n")
    md.append("- Tier 2.1 v4 scale-up architecture handles 8-word synonym vocab\n")
    md.append("- {north,up} both bind to motor_N via embodied Hebbian co-firing\n")
    md.append("- Same baseline-vs-driven delta methodology as Phase 1.4 eval\n")
    md.append("- All learning biology-grounded: STDP + co-firing teachers\n")
    md.append("- No backprop, no surrogate gradients\n\n")
    md.append("Validated 6-seed (Tier 2.1 BREAKTHROUGH 2026-05-06): "
              "**W->A 5/6 aligned, A->W 6/6 aligned**, A->W mean 63.7%. "
              "See `research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md`.\n")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=400,
                    help="Tier 2.1 events per word (default 400 = validated config)")
    ap.add_argument("--transcript-out", type=str, default=None,
                    help="Path to write a chat transcript markdown")
    ap.add_argument("--out-stats", type=str, default=None,
                    help="Path to write summary stats JSON")
    # Lineage (opt-in)
    ap.add_argument("--lineage", type=str, default=None,
                    help="Optional lineage NAME under bridges/lineage/. "
                         "If exists with matching arch, load and skip "
                         "training. Default: None (always train).")
    ap.add_argument("--save-to-lineage", action="store_true",
                    help="After training, save trained bridge to lineage "
                         "(creates it if necessary). Requires --lineage.")
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
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(
            json.dumps({k: v for k, v in result.items() if k != "transcript"},
                       indent=2),
            encoding="utf-8",
        )

    print(f"\n[DONE] seed={args.seed}: "
          f"overall {result['accuracy']:.1%}, "
          f"primary {result['primary_accuracy']:.1%}, "
          f"synonym {result['synonym_accuracy']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
