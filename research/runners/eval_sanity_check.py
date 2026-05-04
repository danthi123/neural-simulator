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
    mode: str = "perfect",
    verbose: bool = True,
):
    """Hand-build language_input -> motor_X weights for sanity-check eval.

    Mode determines the weight pattern:
      - "perfect": each word's active src -> correct motor_X = high,
        off-target = 0. Tests "can the eval detect a known-correct mapping?"
        Expected verdict: aligned >= 4/6.
      - "wrong": each word's active src -> WRONG motor_X (rotated by 1) = high,
        TRUE motor = 0. Tests "does the eval correctly reject a known-wrong
        mapping?" Expected verdict: aligned 0/6 with best-permutation = TRUE
        rotated by 1 (e.g. "ESWN" or similar).
      - "random": all language_input -> motor_X edges set to random uniform
        weight ~ U[0, target_weight]. Tests "what does the eval do under
        no learning?" Expected verdict: aligned ~chance.
      - "wipe": all language_input -> motor_X edges wiped to 0. Tests
        "does eval handle silent motor pools?" Expected: degenerate.

    Returns a summary dict with edge counts per (word, action) and the mode.
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

    # For mode="wrong", rotate target by 1 (north -> E, east -> S, etc.)
    if mode == "wrong":
        rotated = {"north": "E", "east": "S", "south": "W", "west": "N"}
        word_to_action = rotated

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

    if mode == "wipe":
        # All edges already wiped; nothing more to do.
        bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)
        if verbose:
            print(f"[sanity-check] Mode=wipe: all language->motor edges = 0",
                  flush=True)
        return {"mode": "wipe", "n_wiped": n_wiped}

    if mode == "random":
        # All language->motor edges get U[0, target_weight] random weights.
        rng = np.random.default_rng(0)
        for src in lang_input_indices:
            start = int(indptr[src])
            end = int(indptr[src + 1])
            for off in range(start, end):
                dst = int(indices[off])
                if dst in all_motor_indices:
                    data[off] = float(rng.uniform(0, target_weight))
        bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)
        if verbose:
            print(f"[sanity-check] Mode=random: all language->motor edges "
                  f"~ U[0, {target_weight}]", flush=True)
        return {"mode": "random", "target_weight_max": target_weight,
                "n_edges": n_wiped}

    # mode in ("perfect", "wrong"): for each (word, src in active),
    # set src -> motor_target edges to target_weight. Off-target edges
    # stay wiped (0).
    summary = {"mode": mode}
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
        print(f"[sanity-check] Set {mode} weights:", flush=True)
        for k, v in summary.items():
            if not isinstance(v, dict):
                # Skip non-edge entries like "mode": "perfect"
                continue
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
    mode: str = "perfect",
    biological: bool = False,
    enable_nmda: bool = False,
    ou_tau_ms: float = 15.0,
    ou_std_current_pA: float = 100.0,
    verbose: bool = True,
):
    """Run the sanity check: build minimal arch, hand-build weights per
    mode, evaluate. Returns the eval result dict.

    Modes:
      - "perfect": ideal weights (each word -> correct motor). Expected
        verdict: aligned >= 4/6.
      - "wrong": rotated weights (each word -> wrong motor). Expected:
        TRUE-mapping accuracy = 0%; best permutation = rotated.
      - "random": uniform random weights. Expected: ~chance accuracy.
      - "wipe": all language->motor edges = 0. Expected: degenerate.
    """
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_minimal_brain_regions, build_biological_brain_regions,
    )

    if verbose:
        arch_name = "BIOLOGICAL" if biological else "MINIMAL"
        print("=" * 60)
        print(f"EVAL SANITY CHECK (seed={seed}, mode={mode}, arch={arch_name})")
        print(f"  Hand-built language_input -> motor_X weights ({mode} pattern)")
        print(f"  NO training, NO STDP — directly evaluate weights")
        print(f"  Target weight: {target_weight} (off-target: 0.0)")
        print(f"  n_lang={n_lang_input}, n_motor_per_action={n_motor_per_action}")
        if biological:
            print(f"  Cortical canon ENABLED: recurrence + E/I + NMDA={enable_nmda}")
            print(f"  ou_tau_ms={ou_tau_ms}, ou_std_pA={ou_std_current_pA}")
        print("=" * 60, flush=True)

    if biological:
        regions, pathways = build_biological_brain_regions(
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            text_input_to_motor_density=text_input_to_motor_density,
            text_input_to_motor_weight=3.0,  # baseline; overwritten
            text_input_to_motor_jitter=0.5,
            enable_motor_fs=False,  # sanity-eval doesn't use FS
        )
    else:
        regions, pathways = build_minimal_brain_regions(
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            text_input_to_motor_density=text_input_to_motor_density,
            text_input_to_motor_weight=3.0,  # baseline; overwritten
            text_input_to_motor_jitter=0.5,
            enable_motor_fs=False,
        )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = dt_ms
    cfg.seed = seed
    cfg.enable_nmda = enable_nmda
    cfg.ou_tau_ms = ou_tau_ms
    cfg.ou_std_current_pA = ou_std_current_pA
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

    # Hand-build weights per mode
    weight_summary = hand_build_perfect_weights(
        bridge,
        n_lang_input=n_lang_input,
        sparsity=token_sparsity,
        target_weight=target_weight,
        off_target_weight=0.0,
        mode=mode,
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
    ap.add_argument("--mode", type=str, default="perfect",
                    choices=["perfect", "wrong", "random", "wipe"],
                    help="Weight pattern. 'perfect' = ideal mapping (expected "
                    "aligned >= 4/6); 'wrong' = rotated mapping (expected "
                    "TRUE-acc=0%%, best perm = rotated); 'random' = uniform "
                    "random (expected ~chance); 'wipe' = all zero edges "
                    "(degenerate baseline).")
    ap.add_argument("--biological", action="store_true", default=False,
                    help="use biological-scale architecture: motor pools "
                    "with recurrent E + E/I balance + NMDA bistability + "
                    "larger N. Auto-bumps lang/motor sizes if user defaults "
                    "(n_lang=2048, n_motor=500). Wang 2002 + Lefort 2009.")
    ap.add_argument("--enable-nmda", action="store_true", default=False,
                    help="enable NMDA synapses globally. Auto-on with "
                    "--biological.")
    ap.add_argument("--ou-tau-ms", type=float, default=15.0,
                    help="OU noise correlation time. Default 15ms; set "
                    "50-100ms for slower biological cortical noise.")
    ap.add_argument("--ou-std-current-pA", type=float, default=100.0,
                    help="OU noise amplitude pA. Default 100.")
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    # --biological auto-bumps sizes + enables NMDA if user used minimal defaults
    if args.biological:
        if args.n_lang_input == 256:
            args.n_lang_input = 2048
        if args.n_motor_per_action == 25:
            args.n_motor_per_action = 500
        args.enable_nmda = True

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
        mode=args.mode,
        biological=args.biological,
        enable_nmda=args.enable_nmda,
        ou_tau_ms=args.ou_tau_ms,
        ou_std_current_pA=args.ou_std_current_pA,
        verbose=True,
    )

    # Persist mode in the result for aggregation downstream
    result["mode"] = args.mode

    if args.out_stats:
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(result, indent=2))
        print(f"\n  Saved: {args.out_stats}", flush=True)

    # Print verdict per mode
    acc = result["word_to_action_eval"]["accuracy"]
    print("\n" + "=" * 60)
    if args.mode == "perfect":
        if acc >= 0.5:
            print(f"VERDICT (perfect mode): Eval methodology appears SOUND.")
            print(f"  Hand-built perfect weights -> {acc:.1%} accuracy.")
            print(f"  If real learning gives 0/N, the issue is plasticity, not eval.")
        elif acc >= 0.3:
            print(f"VERDICT (perfect mode): Eval methodology PARTIAL —")
            print(f"  sub-threshold for perfect weights ({acc:.1%}). May indicate")
            print(f"  dynamics issues (firing rate, synaptic time constants)")
            print(f"  rather than eval logic itself.")
        else:
            print(f"VERDICT (perfect mode): Eval methodology BROKEN.")
            print(f"  Hand-built perfect weights give only {acc:.1%}. The eval")
            print(f"  cannot detect the learned mapping even when perfectly encoded.")
            print(f"  Investigate: drive currents, motor pool dynamics,")
            print(f"  measurement window, baseline subtraction.")
    elif args.mode == "wrong":
        if acc <= 0.10:
            print(f"VERDICT (wrong mode): Eval correctly REJECTS wrong mapping.")
            print(f"  TRUE-mapping accuracy = {acc:.1%}.")
            print(f"  Best permutation (in result) reveals the rotated mapping.")
        else:
            print(f"VERDICT (wrong mode): UNEXPECTED — {acc:.1%} TRUE accuracy.")
            print(f"  May indicate eval is not actually responsive to weights.")
    elif args.mode == "random":
        if 0.10 <= acc <= 0.45:
            print(f"VERDICT (random mode): Eval gives near-chance ({acc:.1%}) on")
            print(f"  random weights, as expected.")
        else:
            print(f"VERDICT (random mode): UNEXPECTED — {acc:.1%} on random.")
            print(f"  Could indicate hidden bias in eval scoring.")
    elif args.mode == "wipe":
        print(f"VERDICT (wipe mode): {acc:.1%} accuracy with no edges.")
        print(f"  Pure measurement noise; should be near random or all-N (cascade default).")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
