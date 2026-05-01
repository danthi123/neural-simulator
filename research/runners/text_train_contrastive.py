"""Contrastive text training — biology-grounded fix for the cascade
N-bias diagnosed in text_diag_cascade_bias.py.

Real cortical learning involves competing parallel pathways with mutual
inhibition (PV interneurons enforcing winner-take-all). Our standard
text_train regime drove only the CORRECT cortex_X, letting the others
fire passively from cascade defaults. This reinforced the cortex_N bias.

This regime drives CONTRASTIVE input:
- Correct cortex_X: +250 pA (excitation)
- Other cortex_X's:  -100 pA (inhibition, ~PV interneuron equivalent)

When training "east" → cortex_E:
- cortex_E receives +250 pA, fires strongly
- cortex_N/S/W receive -100 pA, silenced below normal baseline
- STDP grows east_pattern → cortex_E (active post)
- STDP does NOT grow east_pattern → cortex_N/S/W (silent post)

Result: differential learning despite cascade asymmetry.

Composes with all prior fixes (Gabor pre-init, inter-trial reset,
structural plasticity off).
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
    ax, ay = agent_pos
    gx, gy = goal_pos
    dx, dy = gx - ax, gy - ay
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else ("west" if dx < 0 else "north")
    else:
        return "north" if dy > 0 else ("south" if dy < 0 else "east")


def _make_image_word(rng, grid_size=8):
    from sim.visual_cortex import render_gridworld_to_image
    while True:
        ax, ay = rng.integers(0, grid_size, size=2)
        gx, gy = rng.integers(0, grid_size, size=2)
        if (ax, ay) != (gx, gy):
            break
    img = render_gridworld_to_image(
        agent_pos=(int(ax), int(ay)), goal_pos=(int(gx), int(gy)),
        grid_size=grid_size, image_size=32,
    )
    target = _direction_from_positions((ax, ay), (gx, gy))
    return img, target


def run_contrastive_training(
    out_stats=None,
    seed=42,
    n_image_word_pairs=300,
    n_word_action_pairs=300,
    grid_size=8,
    stim_steps=200,
    reset_steps=100,
    drive_pA=200.0,
    target_excite_pA=250.0,
    other_inhibit_pA=-100.0,
    target_clamp_pA=250.0,  # for language_output clamp in I->W regime
    reward_value=1.0,
    verbose=True,
):
    """Train text↔action associations with contrastive cortex_X supervision."""
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

    regions, pathways = build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True, pfc_enable_nmda=True,
        enable_visual_cortex=True, enable_text_io=True,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_structural_plasticity = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    n_gabor = apply_v1_gabor_weights(
        bridge,
        n_orientations=8, n_frequencies=2, n_positions_per_dim=8,
        retina_size=32, receptive_field_radius=4, weight_scale=10.0,
    )
    if verbose:
        print(f"[contrastive] Gabor: {n_gabor} edges")

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

    retina_idx = cp.asarray(
        list(bridge.region_manager.indices("retina")), dtype=cp.int64
    )
    lang_input_idx = cp.asarray(
        list(bridge.region_manager.indices("language_input")), dtype=cp.int64
    )
    lang_output_idx = cp.asarray(
        list(bridge.region_manager.indices("language_output")), dtype=cp.int64
    )
    cortex_idx = {
        a: cp.asarray(list(bridge.region_manager.indices(f"cortex_{a}")),
                      dtype=cp.int64)
        for a in ACTION_NAMES
    }
    n_lang_output = int(lang_output_idx.size)

    epoch_stats = []
    t_start = time.time()

    word_to_action = {"north": "N", "east": "E", "south": "S", "west": "W"}

    # Regime 1: image -> word with CONTRASTIVE language_output supervision
    if verbose:
        print(f"\n[contrastive] Regime 1: image -> word ({n_image_word_pairs} pairs)",
              flush=True)
    for trial in range(n_image_word_pairs):
        img, target_word = _make_image_word(rng, grid_size)

        # Inter-trial reset
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Drive retina
        bridge.cp_external_input_current[retina_idx] = cp.asarray(
            image_to_retina_drive(img, drive_max_pA=drive_pA),
            dtype=cp.float32,
        )
        # Clamp language_output to target word's pattern (excite target neurons,
        # silence others). This is a contrastive pattern: supervisor activates
        # specific 25 "north" neurons, leaves rest at 0.
        target_drive = vocab_to_drive_pattern(
            target_word, n_neurons=n_lang_output,
            drive_max_pA=target_clamp_pA, sparsity=0.1,
        )
        bridge.cp_external_input_current[lang_output_idx] = cp.asarray(
            target_drive, dtype=cp.float32,
        )
        bridge.core_config.current_reward_signal = float(reward_value)
        for _ in range(stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        if verbose and (trial + 1) % 100 == 0:
            print(f"  [I->W] {trial+1}/{n_image_word_pairs}  word={target_word}",
                  flush=True)

    # Regime 2: word -> action with CONTRASTIVE cortex_X supervision
    if verbose:
        print(f"\n[contrastive] Regime 2: word -> action ({n_word_action_pairs} pairs)",
              flush=True)
    word_choices = ["north", "east", "south", "west"]
    for trial in range(n_word_action_pairs):
        word = word_choices[trial % len(word_choices)]
        target_action = word_to_action[word]

        # Inter-trial reset
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Drive language_input
        bridge.set_token_drive(word, drive_pA=drive_pA, sparsity=0.1)

        # CONTRASTIVE cortex_X supervision: excite target, INHIBIT others
        for a in ACTION_NAMES:
            if a == target_action:
                bridge.cp_external_input_current[cortex_idx[a]] = cp.float32(
                    target_excite_pA
                )
            else:
                # Negative current = inhibitory drive (push voltage below threshold)
                bridge.cp_external_input_current[cortex_idx[a]] = cp.float32(
                    other_inhibit_pA
                )
        bridge.core_config.current_reward_signal = float(reward_value)
        for _ in range(stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        if verbose and (trial + 1) % 100 == 0:
            print(f"  [W->A contrastive] {trial+1}/{n_word_action_pairs}  "
                  f"word={word}->{target_action} (others inhibited)",
                  flush=True)

    elapsed = time.time() - t_start
    epoch_stats.append({
        "regime": "contrastive_supervision",
        "n_image_word_pairs": n_image_word_pairs,
        "n_word_action_pairs": n_word_action_pairs,
        "target_excite_pA": target_excite_pA,
        "other_inhibit_pA": other_inhibit_pA,
        "elapsed_seconds": elapsed,
    })
    if verbose:
        print(f"\n[contrastive] Training complete in {elapsed:.1f}s", flush=True)

    if out_stats:
        Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(out_stats).write_text(json.dumps({
            "seed": seed, "grid_size": grid_size,
            "epoch_stats": epoch_stats,
        }, indent=2))

    return bridge, epoch_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-image-word", type=int, default=200)
    ap.add_argument("--n-word-action", type=int, default=200)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run_contrastive_training(
        out_stats=args.out_stats,
        seed=args.seed,
        n_image_word_pairs=args.n_image_word,
        n_word_action_pairs=args.n_word_action,
        grid_size=args.grid_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
