"""B3 — supervised gradient learning on language_input -> motor_X weights.

B-branch tier-2 fallback. Fires when:
  1. The biology sweep gives 0/6 aligned across all conditions, AND
  2. eval_sanity_check (B1) confirms the eval methodology is sound
     (perfect-weight mode aligns >= 4/6).

If those two conditions both hold, the question becomes: can ANY learning
rule succeed under this architecture and eval, or is there a deeper
representational issue?

This runner replaces STDP / R-STDP with explicit supervised gradient
descent on the language_input -> motor_X weights. Gradient learning is
NOT biology-grounded — it's a probe to see what's possible. The verdict:
  - If gradient learning aligns >= 4/6: biology rules ARE the bottleneck.
    Pursue alternative learning rules (e.g. predictive coding, three-
    factor with better credit signals, evolved local rules).
  - If gradient learning aligns 2-3/6: partial; investigate dynamics.
  - If gradient learning still fails: the architecture itself cannot
    represent the mapping. Pivot to architectural rebuild.

Architecture (same as text_minimal_isolation):
  - language_input: 256 neurons (sparse code substrate)
  - motor_N, motor_E, motor_S, motor_W: 25 each (purely excitatory)
  - language_input -> motor_X pathways (4 plastic, gated to language_
    input_to_motor)
  - NO cascade, NO PFC, NO retina, NO visuomotor

Training (replaces STDP):
  For each event in synthetic balanced buffer:
    1. Compute target one-hot: rate=1.0 for correct motor pool, 0.0
       elsewhere.
    2. Drive language_input via vocab_to_drive_pattern(token).
    3. Step the bridge for stim_steps (forward propagation).
    4. Read motor_X firing per pool (sum over stim window).
    5. Normalize firing -> [0, 1] rate (divided by an empirical max).
    6. Compute error e[a] = target[a] - rate[a] per motor pool.
    7. For each (src, dst) edge in the language_input_to_motor pathway:
       Delta_w = lr * e[motor_pool(dst)] * pre_active(src, token)
       where pre_active(src, token) = 1 if src is in vocab_to_drive_pattern
       active set, else 0. Clipped to [0, stdp_w_max].
  Plasticity gate set to 0.0 so STDP doesn't compete.

Gradient formula choice:
  We use the simplest credit-assignment rule that respects causality:
  delta-rule on the post-pool error, gated by the pre's drive
  pattern (binary). No backprop through dynamics, no eligibility
  traces, no STDP. This is closer to a perceptron update than to
  proper backprop, and it's intentionally simple — the goal is to
  measure what a well-tuned credit signal can do at this scale, not
  to find the optimal learning rule.

Eval: evaluate_word_to_action with same settings as everything else.

Usage:
    python -m research.runners.b3_supervised_gradient \
        --seed 42 --n-events-per-direction 1000 \
        --learning-rate 1e-3 \
        --out-stats research/findings/raw/g11_bg/text_eval_b3_vanilla_seed42.json

    # Combined with biology fixes (positive-control variant):
    python -m research.runners.b3_supervised_gradient \
        --seed 42 --apply-topographic-bias --enable-motor-fs \
        --out-stats research/findings/raw/g11_bg/text_eval_b3_with_topo_fs_seed42.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


WORD_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}
ACTIONS = ["N", "E", "S", "W"]


def _build_pair_to_idx(bridge) -> Dict[Tuple[int, int], int]:
    """Build (pre, post) -> data offset map for the CSR. O(nnz) one-time."""
    indptr = bridge.cp_connections.indptr.get()
    indices = bridge.cp_connections.indices.get()
    pair_to_idx: Dict[Tuple[int, int], int] = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off
    return pair_to_idx


def build_supervised_gradient_step(
    bridge,
    token: str,
    target_action: str,
    lang_input_indices: List[int],
    motor_indices_by_action: Dict[str, List[int]],
    pair_to_idx: Dict[Tuple[int, int], int],
    data_np: np.ndarray,
    n_lang_input: int,
    token_sparsity: float,
    lang_input_drive_pA: float,
    stim_steps: int,
    reset_steps: int,
    learning_rate: float,
    weight_min: float,
    weight_max: float,
    expected_max_firing_per_neuron: float,
):
    """Run one supervised event: drive token, observe motor firing,
    apply delta-rule update to language_input -> motor_X weights.

    Mutates `data_np` in place. Caller is responsible for pushing
    `data_np` back to the GPU after a batch of updates.

    Returns:
        dict with per-pool target/observed/error and edge-update count.
    """
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    # Inter-trial reset
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Drive language_input only (no motor nudge — supervised mode uses
    # the gradient signal as the supervisor, not paired stim).
    in_drive = vocab_to_drive_pattern(
        token, n_neurons=n_lang_input,
        drive_max_pA=lang_input_drive_pA, sparsity=token_sparsity,
    )
    lang_input_idx_cp = cp.asarray(lang_input_indices, dtype=cp.int64)
    bridge.cp_external_input_current[lang_input_idx_cp] = cp.asarray(
        in_drive, dtype=cp.float32,
    )

    # Forward stim window: count motor firing per pool
    spike_counts: Dict[str, int] = {a: 0 for a in ACTIONS}
    motor_idx_cp = {
        a: cp.asarray(motor_indices_by_action[a], dtype=cp.int64)
        for a in ACTIONS
    }
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        firing = bridge.cp_firing_states
        for a in ACTIONS:
            spike_counts[a] += int(firing[motor_idx_cp[a]].sum().get())

    # Normalize spike counts to a [0, 1] rate by dividing by an empirical
    # maximum (n_neurons_per_pool * stim_steps * expected_max_firing).
    # This gives the gradient a calibrated scale; otherwise the error
    # term swings by orders of magnitude with stim_steps.
    n_per_pool = len(motor_indices_by_action[ACTIONS[0]])
    max_spikes = max(1.0, n_per_pool * stim_steps * expected_max_firing_per_neuron)
    rate = {a: min(1.0, spike_counts[a] / max_spikes) for a in ACTIONS}
    target = {a: (1.0 if a == target_action else 0.0) for a in ACTIONS}
    error = {a: target[a] - rate[a] for a in ACTIONS}

    # Pre-side: which language_input neurons are active for THIS token
    local_active = np.where(in_drive > 0)[0]
    global_active_pre = [lang_input_indices[i] for i in local_active]

    # Apply delta-rule update on language_input -> motor_X edges
    n_updated = 0
    for action in ACTIONS:
        e = error[action]
        if abs(e) < 1e-9:
            continue
        for src in global_active_pre:
            for dst in motor_indices_by_action[action]:
                key = (src, dst)
                idx = pair_to_idx.get(key)
                if idx is None:
                    continue
                w = float(data_np[idx])
                # Pre-active is binary 1 here; drive is constant.
                w_new = w + learning_rate * e
                if w_new < weight_min:
                    w_new = weight_min
                elif w_new > weight_max:
                    w_new = weight_max
                data_np[idx] = w_new
                n_updated += 1

    return {
        "spike_counts": spike_counts,
        "rate": rate,
        "target": target,
        "error": error,
        "n_edges_updated": n_updated,
    }


def run_supervised_gradient(
    seed: int = 42,
    n_events_per_direction: int = 1000,
    learning_rate: float = 1e-3,
    stim_steps_per_event: int = 50,
    reset_steps: int = 50,
    lang_input_drive_pA: float = 200.0,
    n_motor_per_action: int = 25,
    n_lang_input: int = 256,
    token_sparsity: float = 0.1,
    dt_ms: float = 1.0,
    text_input_to_motor_density: float = 0.30,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    weight_min: float = 0.0,
    weight_max: float = 5.0,
    apply_topographic_bias: bool = False,
    topographic_bias_factor: float = 1.5,
    off_target_bias_factor: float = 0.7,
    enable_motor_fs: bool = False,
    n_motor_fs_per_action: int = 3,
    expected_max_firing_per_neuron: float = 0.10,
    push_to_gpu_every: int = 64,
    fast_spike_reset: bool = True,
    biological: bool = False,
    enable_nmda: bool = False,
    ou_tau_ms: float = 15.0,
    ou_std_current_pA: float = 100.0,
    verbose: bool = True,
):
    """Train minimal architecture via supervised delta-rule gradient.

    Returns (bridge, training_stats) — caller runs eval afterwards.
    """
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    # Imported via module path so this runner doesn't share state with
    # text_minimal_isolation (per spec: "copy patterns, don't import-and-
    # mutate"). build_minimal_brain_regions / apply_topographic_bias are
    # pure helpers and importing them is fine — they don't carry state.
    from research.runners.text_minimal_isolation import (
        build_minimal_brain_regions, build_biological_brain_regions,
        apply_topographic_bias as _apply_topo,
    )

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 60)
        print(f"B3 SUPERVISED GRADIENT (seed={seed})")
        print(f"  n_lang_input={n_lang_input}, motor_per_action={n_motor_per_action}")
        n_total = (n_lang_input + 4 * n_motor_per_action +
                    (4 * n_motor_fs_per_action if enable_motor_fs else 0))
        print(f"  Total: {n_total} neurons")
        print(f"  {n_events_per_direction} events/dir x 4 dirs = "
              f"{4 * n_events_per_direction} events total")
        print(f"  lr={learning_rate}, stim_steps={stim_steps_per_event}, "
              f"reset_steps={reset_steps}")
        print(f"  weight bounds [{weight_min}, {weight_max}]")
        print(f"  apply_topographic_bias={apply_topographic_bias} "
              f"({topographic_bias_factor}/{off_target_bias_factor})")
        print(f"  enable_motor_fs={enable_motor_fs} (n_fs="
              f"{n_motor_fs_per_action}/pool)")
        print(f"  push_to_gpu_every={push_to_gpu_every} events")
        print("=" * 60, flush=True)

    if biological:
        regions, pathways = build_biological_brain_regions(
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            text_input_to_motor_density=text_input_to_motor_density,
            text_input_to_motor_weight=text_input_to_motor_weight,
            text_input_to_motor_jitter=text_input_to_motor_jitter,
            enable_motor_fs=enable_motor_fs,
            n_motor_fs_per_action=n_motor_fs_per_action,
        )
    else:
        regions, pathways = build_minimal_brain_regions(
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            text_input_to_motor_density=text_input_to_motor_density,
            text_input_to_motor_weight=text_input_to_motor_weight,
            text_input_to_motor_jitter=text_input_to_motor_jitter,
            enable_motor_fs=enable_motor_fs,
            n_motor_fs_per_action=n_motor_fs_per_action,
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
    # Set stdp_w_max above weight_max so soft-bound STDP is irrelevant
    # even before we freeze the gate.
    cfg.stdp_w_max = max(weight_max + 1.0, 5.0)
    cfg.fast_spike_reset = fast_spike_reset

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

    # Optional biology pre-init: topographic bias on language_input ->
    # motor weights. Same helper as text_minimal_isolation uses.
    if apply_topographic_bias:
        _apply_topo(
            bridge,
            topographic_factor=topographic_bias_factor,
            off_target_factor=off_target_bias_factor,
            n_lang_input=n_lang_input,
            sparsity=token_sparsity,
            verbose=verbose,
        )

    # Freeze STDP — gradient is the ONLY teacher.
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        if verbose:
            print(f"[b3-supervised] STDP frozen on language_input_to_motor "
                  f"(supervised gradient is sole teacher)", flush=True)
    except Exception as e:
        print(f"[b3-supervised] WARNING: could not freeze gate: {e}",
              flush=True)

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    motor_indices_by_action = {
        a: list(rm.indices(f"motor_{a}")) for a in ACTIONS
    }

    # One-time CSR pair index — O(nnz). Reused for the entire training
    # loop. data_np is the host-side weight buffer; we mutate it per
    # event and push back to the GPU every push_to_gpu_every events to
    # amortize H2D cost.
    if verbose:
        print(f"[b3-supervised] Building CSR pair index "
              f"(nnz={int(bridge.cp_connections.nnz)})...", flush=True)
    pair_to_idx = _build_pair_to_idx(bridge)
    data_np = bridge.cp_connections.data.get()

    # Build synthetic balanced buffer
    DIRECTIONS = ["north", "east", "south", "west"]
    synthetic_buffer = []
    for direction in DIRECTIONS:
        action = WORD_TO_ACTION[direction]
        for _ in range(n_events_per_direction):
            synthetic_buffer.append({"token": direction, "action": action})
    rng.shuffle(synthetic_buffer)

    if verbose:
        print(f"[b3-supervised] Synthetic buffer: {len(synthetic_buffer)} "
              f"events ({n_events_per_direction}/dir, shuffled)", flush=True)

    t_start = time.time()
    n_total_edges_updated = 0
    rolling_correct = 0
    rolling_window = 100
    rolling_history: List[bool] = []

    for event_idx, event in enumerate(synthetic_buffer):
        result = build_supervised_gradient_step(
            bridge,
            token=event["token"],
            target_action=event["action"],
            lang_input_indices=lang_input_indices,
            motor_indices_by_action=motor_indices_by_action,
            pair_to_idx=pair_to_idx,
            data_np=data_np,
            n_lang_input=n_lang_input,
            token_sparsity=token_sparsity,
            lang_input_drive_pA=lang_input_drive_pA,
            stim_steps=stim_steps_per_event,
            reset_steps=reset_steps,
            learning_rate=learning_rate,
            weight_min=weight_min,
            weight_max=weight_max,
            expected_max_firing_per_neuron=expected_max_firing_per_neuron,
        )
        n_total_edges_updated += result["n_edges_updated"]

        # Rolling accuracy proxy: did the target pool have the highest
        # rate? Independent of the gradient update; just for reporting.
        rate = result["rate"]
        pred = max(rate, key=lambda a: rate[a])
        was_correct = (pred == event["action"])
        rolling_history.append(was_correct)
        if len(rolling_history) > rolling_window:
            rolling_history.pop(0)
        if was_correct:
            rolling_correct += 0  # placeholder; we re-derive from history below

        # Push CSR weights back to GPU periodically to amortize H2D cost
        if (event_idx + 1) % push_to_gpu_every == 0:
            import cupy as cp
            bridge.cp_connections.data = cp.asarray(data_np, dtype=cp.float32)

        if verbose and (event_idx + 1) % 250 == 0:
            elapsed = time.time() - t_start
            rate_correct = (sum(rolling_history) / len(rolling_history)
                             if rolling_history else 0.0)
            print(f"  [b3-supervised] {event_idx+1}/{len(synthetic_buffer)} "
                  f"events  rolling_acc={rate_correct:.1%}  "
                  f"({elapsed:.0f}s)", flush=True)
            from sim.progress import emit_progress
            emit_progress(
                "training", event_idx + 1, len(synthetic_buffer),
                phase="supervised-gradient", unit="events",
                label="b3-supervised-gradient",
                elapsed_seconds=elapsed,
                rolling_accuracy=round(rate_correct, 4),
            )

    # Final push of any remaining edits
    import cupy as cp
    bridge.cp_connections.data = cp.asarray(data_np, dtype=cp.float32)

    elapsed = time.time() - t_start
    final_rolling = (sum(rolling_history) / len(rolling_history)
                       if rolling_history else 0.0)
    if verbose:
        print(f"\n[b3-supervised] Training complete: "
              f"{len(synthetic_buffer)} events, "
              f"{n_total_edges_updated} total edge-updates ({elapsed:.0f}s)",
              flush=True)
        print(f"  Final rolling accuracy (last {rolling_window}): "
              f"{final_rolling:.1%}", flush=True)

    training_stats = [{
        "phase": 1,
        "regime": "supervised_gradient",
        "n_total_events": len(synthetic_buffer),
        "n_per_direction": n_events_per_direction,
        "n_total_edge_updates": n_total_edges_updated,
        "elapsed_seconds": elapsed,
        "final_rolling_accuracy": final_rolling,
    }]

    return bridge, training_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events-per-direction", type=int, default=1000,
                    help="Supervised events per direction (default 1000)")
    ap.add_argument("--learning-rate", type=float, default=1e-3,
                    help="Delta-rule step size on weight (default 1e-3)")
    ap.add_argument("--stim-steps-per-event", type=int, default=50,
                    help="Forward-prop sub-steps per event (default 50). "
                    "Smaller than text_minimal_isolation's 100 because "
                    "we're doing 4000 events and each one is a full "
                    "stim window.")
    ap.add_argument("--reset-steps", type=int, default=50)
    ap.add_argument("--lang-input-drive-pA", type=float, default=200.0)
    ap.add_argument("--n-motor-per-action", type=int, default=25)
    ap.add_argument("--n-lang-input", type=int, default=256)
    ap.add_argument("--token-sparsity", type=float, default=0.1,
                    help="Fraction of language_input active per token. "
                    "Default 0.1 matches v2 baseline + biology sweep.")
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--text-input-to-motor-density", type=float, default=0.30)
    ap.add_argument("--text-input-to-motor-weight", type=float, default=3.0)
    ap.add_argument("--text-input-to-motor-jitter", type=float, default=0.5)
    ap.add_argument("--weight-min", type=float, default=0.0,
                    help="Lower clip on language_input -> motor weights")
    ap.add_argument("--weight-max", type=float, default=5.0,
                    help="Upper clip on language_input -> motor weights "
                    "(stdp_w_max is set above this).")
    ap.add_argument("--apply-topographic-bias", action="store_true",
                    default=False,
                    help="Apply biology-grounded topographic prior to "
                    "language_input -> motor weights before training. "
                    "Combined supervised + biology test.")
    ap.add_argument("--topographic-bias-factor", type=float, default=1.5,
                    help="Multiplier for word's active -> target motor "
                    "weights. Only used if --apply-topographic-bias.")
    ap.add_argument("--off-target-bias-factor", type=float, default=0.7,
                    help="Multiplier for word's active -> non-target "
                    "motor weights. Only used if --apply-topographic-bias.")
    ap.add_argument("--enable-motor-fs", action="store_true", default=False,
                    help="Add motor PV-FS interneurons (lateral inhibition). "
                    "Combined supervised + biology test.")
    ap.add_argument("--n-motor-fs-per-action", type=int, default=3)
    ap.add_argument("--expected-max-firing-per-neuron", type=float,
                    default=0.10,
                    help="Used to normalize observed firing rate to [0, 1] "
                    "for the delta-rule. Empirical at default config "
                    "(stim_steps=50, dt=1.0) ~= 0.05-0.15 firings per "
                    "neuron per step. Default 0.10.")
    ap.add_argument("--push-to-gpu-every", type=int, default=64,
                    help="Amortize H2D cost: push weights back every N "
                    "events. Smaller = more accurate but slower; larger "
                    "= delayed feedback. Default 64.")
    ap.add_argument("--no-fast-spike-reset", dest="fast_spike_reset",
                    action="store_false", default=True,
                    help="Disable cp.where masked-update spike reset.")
    ap.add_argument("--biological", action="store_true", default=False,
                    help="Use biological-scale architecture (cortical canon: "
                    "recurrent E + E/I balance + larger N + NMDA bistability). "
                    "Auto-bumps n-lang-input=2048, n-motor-per-action=500, "
                    "n-motor-fs-per-action=60. Wang 2002 + Lefort 2009.")
    ap.add_argument("--enable-nmda", action="store_true", default=False,
                    help="Enable NMDA synapses globally. Auto-on with --biological.")
    ap.add_argument("--ou-tau-ms", type=float, default=15.0)
    ap.add_argument("--ou-std-current-pA", type=float, default=100.0)
    ap.add_argument("--n-eval-per-word", type=int, default=25,
                    help="W->A eval trials per word (default 25, same "
                    "as text_minimal_isolation).")
    ap.add_argument("--eval-stim-steps", type=int, default=100,
                    help="Eval stim window (default 100, matches eval_"
                    "sanity_check / text_minimal_isolation).")
    ap.add_argument("--eval-reset-steps", type=int, default=50,
                    help="Eval inter-trial reset window (default 50).")
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    # --biological auto-bumps sizes + enables NMDA
    if args.biological:
        if args.n_lang_input == 256:
            args.n_lang_input = 2048
        if args.n_motor_per_action == 25:
            args.n_motor_per_action = 500
        if args.n_motor_fs_per_action == 3:
            args.n_motor_fs_per_action = 60
        args.enable_nmda = True

    bridge, train_stats = run_supervised_gradient(
        seed=args.seed,
        n_events_per_direction=args.n_events_per_direction,
        learning_rate=args.learning_rate,
        stim_steps_per_event=args.stim_steps_per_event,
        reset_steps=args.reset_steps,
        lang_input_drive_pA=args.lang_input_drive_pA,
        n_motor_per_action=args.n_motor_per_action,
        n_lang_input=args.n_lang_input,
        token_sparsity=args.token_sparsity,
        dt_ms=args.dt_ms,
        text_input_to_motor_density=args.text_input_to_motor_density,
        text_input_to_motor_weight=args.text_input_to_motor_weight,
        text_input_to_motor_jitter=args.text_input_to_motor_jitter,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        apply_topographic_bias=args.apply_topographic_bias,
        topographic_bias_factor=args.topographic_bias_factor,
        off_target_bias_factor=args.off_target_bias_factor,
        enable_motor_fs=args.enable_motor_fs,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        expected_max_firing_per_neuron=args.expected_max_firing_per_neuron,
        push_to_gpu_every=args.push_to_gpu_every,
        fast_spike_reset=args.fast_spike_reset,
        biological=args.biological,
        enable_nmda=args.enable_nmda,
        ou_tau_ms=args.ou_tau_ms,
        ou_std_current_pA=args.ou_std_current_pA,
        verbose=True,
    )

    # Eval
    from research.runners.text_eval import evaluate_word_to_action
    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_per_word} per word, "
          f"token_sparsity={args.token_sparsity})")
    print("=" * 60, flush=True)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=args.eval_stim_steps,
        n_reset_steps=args.eval_reset_steps,
        token_sparsity=args.token_sparsity,
    )
    acc = wa_result["accuracy"]
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {acc:.1%}", flush=True)
    print(f"  Confusion: {wa_result['confusion_matrix']}", flush=True)

    # Verdict
    print("\n" + "=" * 60)
    if acc >= 0.60:
        print(f"VERDICT (B3 supervised gradient): SUCCESS — {acc:.1%}")
        print(f"  Gradient learning succeeds where biology rules failed.")
        print(f"  -> Biology rules ARE the bottleneck, not the architecture.")
        print(f"  -> Pursue alternative learning rules (predictive coding,")
        print(f"     three-factor with better credit, evolved local rules).")
    elif acc >= 0.30:
        print(f"VERDICT (B3 supervised gradient): PARTIAL — {acc:.1%}")
        print(f"  Above chance ({acc:.1%}) but not robust. Investigate:")
        print(f"  - Learning rate / event count")
        print(f"  - Forward-pass dynamics (firing rate calibration)")
        print(f"  - Pathway density (might not have enough edges)")
    else:
        print(f"VERDICT (B3 supervised gradient): FAILED — {acc:.1%}")
        print(f"  Even gradient learning cannot align this mapping.")
        print(f"  -> The architecture itself cannot represent the task.")
        print(f"  -> Pivot to architectural rebuild (deeper hidden layer,")
        print(f"     different motor pool topology, sparse-coded readout).")
    print("=" * 60, flush=True)

    if args.out_stats:
        out = {
            "regime": "b3_supervised_gradient",
            "seed": args.seed,
            "n_events_per_direction": args.n_events_per_direction,
            "n_total_events": 4 * args.n_events_per_direction,
            "training_stats": train_stats,
            "word_to_action_eval": wa_result,
            "verdict_accuracy": acc,
            "config": {
                "n_lang_input": args.n_lang_input,
                "n_motor_per_action": args.n_motor_per_action,
                "learning_rate": args.learning_rate,
                "stim_steps_per_event": args.stim_steps_per_event,
                "reset_steps": args.reset_steps,
                "lang_input_drive_pA": args.lang_input_drive_pA,
                "token_sparsity": args.token_sparsity,
                "dt_ms": args.dt_ms,
                "text_input_to_motor_density": args.text_input_to_motor_density,
                "text_input_to_motor_weight": args.text_input_to_motor_weight,
                "weight_min": args.weight_min,
                "weight_max": args.weight_max,
                "apply_topographic_bias": args.apply_topographic_bias,
                "topographic_bias_factor": args.topographic_bias_factor,
                "off_target_bias_factor": args.off_target_bias_factor,
                "enable_motor_fs": args.enable_motor_fs,
                "n_motor_fs_per_action": args.n_motor_fs_per_action,
                "expected_max_firing_per_neuron":
                    args.expected_max_firing_per_neuron,
                "push_to_gpu_every": args.push_to_gpu_every,
            },
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
