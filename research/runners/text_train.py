"""Text training runner — supervises image-to-word and word-to-action
mappings via STDP+reward.

Three regimes:
1. IMAGE -> WORD (visual labeling): render gridworld, drive retina via
   K v2 visual cortex, clamp language_output to target word's pattern,
   reward = +1. Trains IT -> language_output to verbalize what's seen.
2. WORD -> ACTION (verbal command): set_token_drive on language_input,
   drive cortex_X for target action, reward = +1. Trains
   language_input -> cortex_X.
3. ACTION -> WORD (action verbalization): drive cortex_X, clamp
   language_output to corresponding word, reward = +1. Trains
   cortex_X -> language_output.

After training, the agent should:
- Emit the correct cardinal-direction word when shown a gridworld image
- Take the correct cardinal action when given a direction word

Design source: docs/plans/2026-05-01-text-interaction-design.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
ACTION_TO_WORD = {"N": "north", "E": "east", "S": "south", "W": "west"}


def _direction_from_positions(agent_pos, goal_pos) -> str:
    """Pick a cardinal direction ('north'/'east'/'south'/'west') that
    reduces Manhattan distance from agent to goal. Ambiguous cases pick
    the larger axis."""
    ax, ay = agent_pos
    gx, gy = goal_pos
    dx, dy = gx - ax, gy - ay
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else ("west" if dx < 0 else "north")
    else:
        return "north" if dy > 0 else ("south" if dy < 0 else "east")


def make_supervised_pair_image_word(rng, grid_size=8):
    """Generate one (image, target_word) training pair.

    Random agent + goal positions; target = direction from agent to goal."""
    from sim.visual_cortex import render_gridworld_to_image

    # Sample distinct positions
    while True:
        ax, ay = rng.integers(0, grid_size, size=2)
        gx, gy = rng.integers(0, grid_size, size=2)
        if (ax, ay) != (gx, gy):
            break
    img = render_gridworld_to_image(
        agent_pos=(int(ax), int(ay)),
        goal_pos=(int(gx), int(gy)),
        grid_size=grid_size,
        image_size=32,
    )
    target = _direction_from_positions((ax, ay), (gx, gy))
    return img, target, (int(ax), int(ay)), (int(gx), int(gy))


def run_text_training(
    out_checkpoint: str | None = None,
    out_stats: str | None = None,
    seed: int = 42,
    n_image_word_pairs: int = 500,
    n_word_action_pairs: int = 500,
    grid_size: int = 8,
    stim_steps_per_pair: int = 200,  # 100ms at dt=0.5
    drive_pA: float = 200.0,
    target_clamp_pA: float = 250.0,  # higher than drive — drive output strongly
    reward_value: float = 1.0,
    verbose: bool = True,
):
    """Train language_input ↔ cortex ↔ language_output associations.

    Returns: stats dict (used pair counts, elapsed, mean activations).
    """
    import cupy as cp

    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.visual_cortex import (
        apply_v1_gabor_weights,
        image_to_retina_drive,
    )
    from sim.text_embeddings import vocab_to_drive_pattern
    from research.runners.g11_bg_runner import build_bg_brain_regions

    rng = np.random.default_rng(seed)

    # Build minimal config: visual cortex + text I/O + cluster A/E + PFC
    regions, pathways = build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        pfc_enable_nmda=True,
        enable_visual_cortex=True,
        enable_text_io=True,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    # Disable structural plasticity — text_train repeatedly applies +1 reward,
    # which under structural plasticity rapidly grows synapses, causing
    # synapse-array shape mismatches (CSR nnz != synapse-indexed array size).
    # We only need STDP+reward weight changes for text training, not new
    # synapse formation.
    cfg.enable_structural_plasticity = False

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Apply Gabor pre-init for V1 — works because we disabled structural
    # plasticity above. Without this, V1 fires on noise and IT can't
    # learn useful representations.
    n_gabor = apply_v1_gabor_weights(
        bridge,
        n_orientations=8, n_frequencies=2, n_positions_per_dim=8,
        retina_size=32, receptive_field_radius=4, weight_scale=10.0,
    )
    if verbose:
        print(f"[text_train] Gabor weights: {n_gabor} edges installed",
              flush=True)

    # Open all gates so STDP+reward can grow weights everywhere
    for gate in [
        "visual_cortex_v1", "visual_cortex_v2", "visual_cortex_it",
        "visual_cortex_action",
        "language_input_to_cortex", "language_input_to_pfc",
        "it_to_language_output", "cortex_to_language_output",
    ]:
        try:
            bridge.set_plasticity_gate(gate, 1.0)
        except KeyError:
            pass

    # Cache region indices
    retina_idx = cp.asarray(
        list(bridge.region_manager.indices("retina")), dtype=cp.int64
    )
    lang_input_idx = cp.asarray(
        list(bridge.region_manager.indices("language_input")), dtype=cp.int64
    )
    lang_output_idx = cp.asarray(
        list(bridge.region_manager.indices("language_output")), dtype=cp.int64
    )
    cortex_idx_per_action = {
        a: cp.asarray(list(bridge.region_manager.indices(f"cortex_{a}")),
                      dtype=cp.int64)
        for a in ACTION_NAMES
    }

    n_lang_output = lang_output_idx.size

    epoch_stats = []
    t_start = time.time()

    # ─────────────── Regime 1: IMAGE → WORD ───────────────
    if verbose:
        print(f"\n[text_train] Regime 1: image -> word ({n_image_word_pairs} pairs)",
              flush=True)
    n_reset_steps = 100  # 50 ms inter-trial blank — lets NMDA decay
    for trial in range(n_image_word_pairs):
        img, target_word, ap, gp = make_supervised_pair_image_word(rng, grid_size)

        # ─── INTER-TRIAL RESET ───
        # Zero all input current and reward; run a brief blank period so
        # NMDA-mediated cortex bistability decays. Without this, activity
        # from trial N persists into trial N+1, cross-contaminating STDP.
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(n_reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # ─── TRAINING TRIAL ───
        retina_drive = image_to_retina_drive(img, drive_max_pA=drive_pA)
        bridge.cp_external_input_current[retina_idx] = cp.asarray(retina_drive, dtype=cp.float32)
        # Clamp language_output to target word's pattern (supervisor signal)
        target_drive = vocab_to_drive_pattern(
            target_word, n_neurons=int(n_lang_output),
            drive_max_pA=target_clamp_pA, sparsity=0.1,
        )
        bridge.cp_external_input_current[lang_output_idx] = cp.asarray(target_drive, dtype=cp.float32)
        bridge.core_config.current_reward_signal = float(reward_value)
        for _ in range(stim_steps_per_pair):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        if verbose and (trial + 1) % 100 == 0:
            print(f"  [I->W] {trial+1}/{n_image_word_pairs}  agent={ap} goal={gp} word={target_word}",
                  flush=True)

    # ─────────────── Regime 2: WORD → ACTION ───────────────
    if verbose:
        print(f"\n[text_train] Regime 2: word -> action ({n_word_action_pairs} pairs)",
              flush=True)
    word_choices = ["north", "east", "south", "west"]
    for trial in range(n_word_action_pairs):
        word = word_choices[trial % len(word_choices)]
        action = {"north": "N", "east": "E", "south": "S", "west": "W"}[word]

        # ─── INTER-TRIAL RESET ───
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(n_reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # ─── TRAINING TRIAL ───
        bridge.set_token_drive(word, drive_pA=drive_pA, sparsity=0.1)
        bridge.cp_external_input_current[cortex_idx_per_action[action]] = cp.float32(
            target_clamp_pA
        )
        bridge.core_config.current_reward_signal = float(reward_value)
        for _ in range(stim_steps_per_pair):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        if verbose and (trial + 1) % 100 == 0:
            print(f"  [W->A] {trial+1}/{n_word_action_pairs}  word={word} -> {action}",
                  flush=True)

    elapsed = time.time() - t_start
    epoch_stats.append({
        "regime": "training_complete",
        "n_image_word_pairs": n_image_word_pairs,
        "n_word_action_pairs": n_word_action_pairs,
        "elapsed_seconds": elapsed,
        "n_synapses": int(bridge.cp_connections.nnz) if bridge.cp_connections is not None else 0,
    })
    if verbose:
        print(f"\n[text_train] Training complete in {elapsed:.1f}s "
              f"({n_image_word_pairs + n_word_action_pairs} pairs)", flush=True)

    # ─────────────── Optional: save checkpoint ───────────────
    if out_checkpoint:
        try:
            ckpt_path = Path(out_checkpoint)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            bridge.save_checkpoint(str(ckpt_path))
            if verbose:
                print(f"[text_train] saved checkpoint: {ckpt_path}", flush=True)
        except Exception as e:
            if verbose:
                print(f"[text_train] checkpoint save failed: {e}", flush=True)

    if out_stats:
        stats_path = Path(out_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps({
            "seed": seed,
            "grid_size": grid_size,
            "n_neurons": int(cfg.num_neurons),
            "stim_steps_per_pair": stim_steps_per_pair,
            "epoch_stats": epoch_stats,
        }, indent=2))
        if verbose:
            print(f"[text_train] saved stats: {stats_path}", flush=True)

    return bridge, epoch_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-image-word", type=int, default=500)
    ap.add_argument("--n-word-action", type=int, default=500)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--out-checkpoint", type=str, default=None)
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run_text_training(
        out_checkpoint=args.out_checkpoint,
        out_stats=args.out_stats,
        seed=args.seed,
        n_image_word_pairs=args.n_image_word,
        n_word_action_pairs=args.n_word_action,
        grid_size=args.grid_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
