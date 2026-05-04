"""Eval sanity check — B-branch fallback if biology sweep fails.

Tests whether the word-to-action evaluation methodology works when the
network has TRIVIALLY CORRECT hand-built weights. If this gives
aligned ratio >= 4/6 across 6 seeds, the eval is sound and the issue
is that real plasticity (STDP under noisy paired-stim) cannot find
this weight pattern. If this gives aligned 0/6 even with perfect
weights, the eval methodology itself is broken — we've been chasing
a phantom signal.

Method:
  1. Build the minimal architecture (same as text_minimal_isolation).
  2. WIPE all language_input -> motor_X edges to 0.
  3. For each word w in {north, east, south, west}:
       For each src neuron in lang_active(w) (sparse code):
         For each dst neuron in motor_correct_pool(w):
           If edge (src, dst) exists in CSR: set weight = HIGH.
  4. Skip training entirely (NO STDP, NO paired-stim).
  5. Run evaluate_word_to_action and report aligned ratio.

The HIGH weight (8.0) is well above stdp_w_max=5.0, but since we
freeze plasticity for the eval, the bound doesn't matter.

Anti-cheat note: this is the OPPOSITE of cheating. We're explicitly
testing whether the eval can detect a known-correct mapping; if it
can't, no learning rule could possibly succeed under this eval.

Usage:
    python -m research.runners.eval_sanity_check \
        --seed 42 \
        --out-stats research/findings/raw/g11_bg/text_eval_sanity_check_seed42.json

    # Or all 6 seeds via the experiment runner:
    python -m research.experiment_runner experiments/eval_sanity_check.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def hand_build_perfect_weights(
    bridge,
    n_lang_input: int = 256,
    sparsity: float = 0.1,
    target_weight: float = 8.0,
    off_target_weight: float = 0.0,
    verbose: bool = True,
):
    """For each word w, set language_input -> motor_X weights to 1 (max)
    on edges leading to motor_correct(w), and 0 on edges leading to
    motor_other(w). Any pre-existing edges between language_input and
    motor pools are wiped first.

    This is the IDEAL weight pattern for the task: every active language
    neuron drives ONLY the correct motor pool. If the eval can't detect
    a winner here, no learning rule can possibly help.

    Returns a summary dict with edge counts per (word, action).
    """
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    if bridge.region_manager is None:
        raise RuntimeError("hand_build_perfect_weights: region_manager is None")

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    n_lang = len(lang_input_indices)
    if n_lang != n_lang_input:
        raise ValueError(
            f"hand_build_perfect_weights: bridge has {n_lang} language_input "
            f"neurons but caller specified {n_lang_input}"
        )

    word_to_action = {"north": "N", "east": "E", "south": "S", "west": "W"}
    actions = ["N", "E", "S", "W"]

    motor_indices = {a: set(rm.indices(f"motor_{a}")) for a in actions}
    all_motor_indices = set()
    for a in actions:
        all_motor_indices |= motor_indices[a]

    lang_input_set = set(lang_input_indices)

    indptr = bridge.cp_connections.indptr.get()
    indices = bridge.cp_connections.indices.get()
    data = bridge.cp_connections.data.get()
    n_rows = int(bridge.cp_connections.shape[0])

    # Step 1: wipe all language_input -> motor_X edges
    n_wiped = 0
    for src in lang_input_indices:
        start = int(indptr[src])
        end = int(indptr[src + 1])
        for off in range(start, end):
            dst = int(indices[off])
            if dst in all_motor_indices:
                data[off] = off_target_weight
                n_wiped += 1
    if verbose:
        print(f"[sanity-check] Wiped {n_wiped} language_input -> motor_X "
              f"edges to {off_target_weight}", flush=True)

    # Step 2: for each (word, src in active), set src -> motor_correct
    # edges to target_weight. Off-target edges stay wiped (0).
    summary = {}
    for word, target_action in word_to_action.items():
        drive = vocab_to_drive_pattern(word, n_neurons=n_lang_input,
                                        sparsity=sparsity)
        local_active = np.where(drive > 0)[0]
        global_active = [lang_input_indices[i] for i in local_active]

        target_motor = motor_indices[target_action]
        n_set = 0
        for src in global_active:
            start = int(indptr[src])
            end = int(indptr[src + 1])
            for off in range(start, end):
                dst = int(indices[off])
                if dst in target_motor:
                    data[off] = target_weight
                    n_set += 1
        summary[f"{word}->motor_{target_action}"] = {
            "n_active_src": len(global_active),
            "n_target_motor": len(target_motor),
            "edges_set": n_set,
            "weight": target_weight,
        }

    # Push back to GPU
    bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)

    if verbose:
        print(f"[sanity-check] Set perfect weights:", flush=True)
        for k, v in summary.items():
            print(f"  {k}: {v['edges_set']} edges @ w={v['weight']:.1f} "
                  f"({v['n_active_src']} src x {v['n_target_motor']} dst)",
                  flush=True)

    # Verify by counting weight buckets per word
    if verbose:
        print(f"[sanity-check] Verification:", flush=True)
        for word, target_action in word_to_action.items():
            drive = vocab_to_drive_pattern(word, n_neurons=n_lang_input,
                                            sparsity=sparsity)
            local_active = np.where(drive > 0)[0]
            global_active = [lang_input_indices[i] for i in local_active]
            for action in actions:
                tot_w = 0.0
                n_edges = 0
                for src in global_active:
                    start = int(indptr[src])
                    end = int(indptr[src + 1])
                    for off in range(start, end):
                        dst = int(indices[off])
                        if dst in motor_indices[action]:
                            tot_w += float(data[off])
                            n_edges += 1
                tag = "TARGET" if action == target_action else "off"
                print(f"  {word}->{action} [{tag}]: "
                      f"{n_edges} edges, sum_w={tot_w:.1f}", flush=True)

    return summary


def run_sanity_check(
    seed: int = 42,
    n_lang_input: int = 256,
    n_motor_per_action: int = 25,
    token_sparsity: float = 0.1,
    target_weight: float = 8.0,
    text_input_to_motor_density: float = 0.30,
    n_eval_per_word: int = 25,
    stim_steps_per_step: int = 100,
    reset_steps: int = 50,
    dt_ms: float = 1.0,
    verbose: bool = True,
):
    """Run the sanity check: build minimal arch, hand-build perfect
    weights, evaluate. Returns the eval result dict.
    """
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import build_minimal_brain_regions

    if verbose:
        print("=" * 60)
        print(f"EVAL SANITY CHECK (seed={seed})")
        print(f"  Hand-built perfect language_input -> motor_X weights")
        print(f"  NO training, NO STDP — directly evaluate ideal weights")
        print(f"  Target weight: {target_weight} (off-target: 0.0)")
        print("=" * 60, flush=True)

    regions, pathways = build_minimal_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        text_input_to_motor_density=text_input_to_motor_density,
        text_input_to_motor_weight=3.0,  # baseline; will be overwritten
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=False,  # eval-only test, no FS
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = dt_ms
    cfg.seed = seed
    cfg.enable_nmda = False
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 10.0  # well above target_weight=8.0
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Hand-build perfect weights
    weight_summary = hand_build_perfect_weights(
        bridge,
        n_lang_input=n_lang_input,
        sparsity=token_sparsity,
        target_weight=target_weight,
        off_target_weight=0.0,
        verbose=verbose,
    )

    # Freeze plasticity so the eval drive-step doesn't mutate weights
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        if verbose:
            print(f"[sanity-check] Plasticity frozen "
                  f"(language_input_to_motor gate = 0.0)", flush=True)
    except Exception as e:
        if verbose:
            print(f"[sanity-check] WARNING: could not freeze gate: {e}",
                  flush=True)

    # Run eval
    from research.runners.text_eval import evaluate_word_to_action
    if verbose:
        print(f"\n[sanity-check] Running eval ({n_eval_per_word} per word, "
              f"token_sparsity={token_sparsity})", flush=True)
    t0 = time.time()
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=n_eval_per_word,
        stim_steps_per_trial=stim_steps_per_step,
        n_reset_steps=reset_steps,
        token_sparsity=token_sparsity,
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
              f"= {wa_result['accuracy']:.1%}", flush=True)
        print(f"  Confusion: {wa_result['confusion_matrix']}", flush=True)
        print(f"  Eval time: {elapsed:.1f}s", flush=True)

    return {
        "regime": "eval_sanity_check",
        "seed": seed,
        "weight_summary": weight_summary,
        "word_to_action_eval": wa_result,
        "eval_elapsed_seconds": elapsed,
        "config": {
            "n_lang_input": n_lang_input,
            "n_motor_per_action": n_motor_per_action,
            "token_sparsity": token_sparsity,
            "target_weight": target_weight,
            "off_target_weight": 0.0,
            "n_eval_per_word": n_eval_per_word,
            "stim_steps_per_step": stim_steps_per_step,
            "reset_steps": reset_steps,
            "text_input_to_motor_density": text_input_to_motor_density,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-lang-input", type=int, default=256)
    ap.add_argument("--n-motor-per-action", type=int, default=25)
    ap.add_argument("--token-sparsity", type=float, default=0.1)
    ap.add_argument("--target-weight", type=float, default=8.0,
                    help="weight on edges from word's active src to "
                    "correct motor pool. Default 8.0 (well above default "
                    "stdp_w_max=5.0; we override stdp_w_max=10.0 to allow).")
    ap.add_argument("--text-input-to-motor-density", type=float, default=0.30,
                    help="connectivity density of the language_input -> "
                    "motor_X pathway. Default 0.30 matches text_minimal_"
                    "isolation. Use 1.0 to test eval with full connectivity.")
    ap.add_argument("--n-eval-per-word", type=int, default=25)
    ap.add_argument("--stim-steps-per-step", type=int, default=100)
    ap.add_argument("--reset-steps", type=int, default=50)
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    result = run_sanity_check(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        token_sparsity=args.token_sparsity,
        target_weight=args.target_weight,
        text_input_to_motor_density=args.text_input_to_motor_density,
        n_eval_per_word=args.n_eval_per_word,
        stim_steps_per_step=args.stim_steps_per_step,
        reset_steps=args.reset_steps,
        dt_ms=args.dt_ms,
        verbose=True,
    )

    if args.out_stats:
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(result, indent=2))
        print(f"\n  Saved: {args.out_stats}", flush=True)

    # Print verdict
    acc = result["word_to_action_eval"]["accuracy"]
    print("\n" + "=" * 60)
    if acc >= 0.5:
        print(f"VERDICT: Eval methodology appears SOUND.")
        print(f"  Hand-built perfect weights -> {acc:.1%} accuracy.")
        print(f"  If real learning gives 0/N, the issue is plasticity, not eval.")
    elif acc >= 0.3:
        print(f"VERDICT: Eval methodology PARTIAL — sub-threshold for")
        print(f"  perfect weights ({acc:.1%}). May indicate dynamics ")
        print(f"  issues (firing rate, synaptic time constants) rather")
        print(f"  than eval logic itself.")
    else:
        print(f"VERDICT: Eval methodology BROKEN. Hand-built perfect ")
        print(f"  weights give only {acc:.1%}. The eval cannot detect ")
        print(f"  the learned mapping even when perfectly encoded.")
        print(f"  Investigate: drive currents, motor pool dynamics, ")
        print(f"  measurement window, baseline subtraction.")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
