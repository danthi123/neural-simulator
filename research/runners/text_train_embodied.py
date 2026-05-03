"""Embodied text training — biology-grounded alternative to text_train.py.

Instead of artificial supervisor clamping, the agent learns text↔action
associations as a SIDE EFFECT of doing the gridworld navigation task.
This mirrors how children learn language: not via direct supervision,
but through cross-modal binding during meaningful action (Tomasello).

During each navigation step:
1. Render image (K v2 visual cortex sees it)
2. Drive language_input with the target word (Manhattan-greedy direction)
   — like an external speaker giving the instruction
3. Drive language_output with the target word (modest current, not clamp)
   — like the agent's "inner speech" co-activating during the action
4. Run the stim window. Agent's BG cascade selects an action naturally.
5. Reward = +1 if Manhattan distance decreased (real environment reward)
6. Agent moves; repeat

STDP+reward then carves out:
- Visuomotor (retina → ... → cortex_X) — already known to work via K v2
- Language input → action (language_input → cortex_X) — strengthens when
  agent correctly acts on the spoken instruction
- Visual → language output (IT → language_output) — strengthens when
  the agent successfully reaches the goal it was looking at

Real reward signal (action-contingent) replaces tonic supervision.
Same architecture as K v2 navigation; language piggybacks on the same
reinforcement signal.

Per Kandel ch 60 (language) and Tomasello (joint attention): word↔percept
binding requires temporal contiguity AND behavioral relevance. Pure
Hebbian without behavioral grounding produces sterile associations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
WORD_FOR_ACTION = {"N": "north", "E": "east", "S": "south", "W": "west"}


def _direction_from_positions(agent_pos, goal_pos, rng=None) -> str:
    """Direction from agent to goal. Random tie-break when |dx|==|dy|
    (otherwise the >= bias systematically over-represents east/west,
    causing the 2026-05-01 confusion matrix per-word imbalance:
    east/west get 28.5% each of training trials, north/south only 21%).
    """
    ax, ay = agent_pos
    gx, gy = goal_pos
    dx, dy = gx - ax, gy - ay
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    if abs(dy) > abs(dx):
        return "north" if dy > 0 else "south"
    # Tie: random axis, deterministic per (rng) so training is reproducible
    if rng is None:
        return "east" if dx > 0 else ("west" if dx < 0 else "north")
    if rng.random() < 0.5:
        # x-axis wins
        return "east" if dx > 0 else ("west" if dx < 0 else "north")
    return "north" if dy > 0 else ("south" if dy < 0 else "east")


def _sample_balanced_start_goal(rng, grid_size: int):
    """Sample a (start, goal) pair such that the target direction is
    uniformly distributed across {north, east, south, west}. Eliminates
    the geometric bias that over-represents east/west when |dx|==|dy|.

    Strategy: pre-choose target direction uniformly; then sample agent
    position; then place goal in the chosen direction (with valid offset).
    """
    DIRECTIONS = ["north", "east", "south", "west"]
    target = DIRECTIONS[int(rng.integers(0, 4))]
    # Pre-choose offsets that strictly satisfy |dy|>|dx| (N/S) or |dx|>|dy| (E/W)
    while True:
        ax = int(rng.integers(0, grid_size))
        ay = int(rng.integers(0, grid_size))
        if target in ("east", "west"):
            # |dx| > |dy|: pick gx with required sign, gy close to ay
            sign = 1 if target == "east" else -1
            # dx must be >= 1 in sign direction. dy must satisfy |dy| < |dx|.
            for _ in range(50):
                dx_mag = int(rng.integers(1, grid_size))
                dy = int(rng.integers(-(dx_mag - 1), dx_mag))  # |dy| < |dx|
                gx = ax + sign * dx_mag
                gy = ay + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    return (ax, ay), (gx, gy), target
        else:  # north / south
            sign = 1 if target == "north" else -1
            for _ in range(50):
                dy_mag = int(rng.integers(1, grid_size))
                dx = int(rng.integers(-(dy_mag - 1), dy_mag))
                gx = ax + dx
                gy = ay + sign * dy_mag
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    return (ax, ay), (gx, gy), target
        # If we couldn't place a valid goal in 50 tries, resample agent.


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_embodied_text_training(
    out_stats: str | None = None,
    seed: int = 42,
    n_episodes: int = 50,
    steps_per_episode: int = 30,  # ~30 steps × 50 episodes = 1500 navigation steps
    grid_size: int = 8,
    # Tier 1 speedups (2026-05-01, see docs/plans/2026-05-01-training-speedups.md):
    # - stim_steps 200→100: 50ms is enough for STDP (window ~20ms, eligibility τ=500ms)
    # - reset_steps stays at 100: a previous attempt at 50ms (0.5τ NMDA) caused a
    #   regression (300-ep run 2026-05-01: I→W/W→A both dropped to 20% from 30%
    #   baseline). With per-region NMDA on PFC+cortex+motor, residual activity
    #   from one step contaminates the next step's STDP pairing window, scrambling
    #   the trained language→cortex weights. See findings doc.
    # - 2026-05-02: even partial Tier 1 (stim=100, reset=100, per-type-stp=False)
    #   regressed to chance under balanced sampling. Whether stim=100 alone or
    #   per-type-stp=False is the cause is being investigated; both now
    #   configurable.
    stim_steps_per_step: int = 100,  # 50ms at dt=0.5 (was 200/100ms — Tier 1 KEEP)
    reset_steps: int = 100,           # 50ms inter-step reset (REVERTED from 50)
    enable_per_type_stp: bool = False,  # Tier 1.5 default; pass True for revert
    # Drive levels
    retina_drive_pA: float = 200.0,
    lang_input_drive_pA: float = 200.0,
    lang_output_coactive_pA: float = 150.0,  # MODEST — doesn't clamp, just biases
    # Reward shaping (2026-05-02 fix). The default -0.5 wrong-move penalty
    # creates asymmetric LTP/LTD pressure: with cascade at ~30% correct,
    # 70% of moves get LTD (-0.5) vs 30% getting LTP (+1.0). Aggregate LTD
    # magnitude per move = 0.7*0.5 = 0.35 vs LTP magnitude = 0.3*1.0 = 0.30.
    # LTD pressure exceeds LTP, biasing some directions toward REVERSED
    # learning (observed for "south" in the PID 39408 weight diagnostic).
    # Setting wrong_move_reward=0 (no penalty) makes plasticity pressure
    # purely positive, eliminating the reversal direction. Trade-off:
    # less exploration pressure (no incentive to avoid wrong moves).
    correct_move_reward: float = 1.0,
    wrong_move_reward: float = -0.5,
    # Architecture sizing (2026-05-02). The default 10-neuron motor pools
    # may be too small to discriminate clean spike-count signals from
    # cascade noise. Real M1 has thousands of neurons per body part
    # (Penfield homunculus 1937). Bigger pools = more spike count
    # differential, less variance. Pass n_motor_per_action=30 (or higher)
    # to test architectural capacity hypothesis.
    n_motor_per_action: int = 10,
    # Language region sizing. Default 256 supports ~26 active neurons
    # per token at 0.1 sparsity. Larger regions allow more distinct
    # token patterns and richer recurrent dynamics, similar to real
    # Wernicke/Broca cortex (~10^5+ neurons each).
    text_n_input_neurons: int = 256,
    text_n_output_neurons: int = 256,
    # Motor cross-coupling (2026-05-02). Models Pulvermüller distributed
    # action-word coding (G.20 in language-mechanisms-additions.md).
    # Adds excitatory N↔E, E↔S, S↔W, W↔N coupling (adjacent directions).
    enable_motor_cross_coupling: bool = False,
    motor_cross_coupling_weight: float = 0.5,
    motor_cross_coupling_density: float = 0.3,
    verbose: bool = True,
):
    """Embodied training: navigate gridworld with language inputs/outputs
    coactive during each step. Real reward from environment.

    Returns: bridge, stats dict.
    """
    import cupy as cp

    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.visual_cortex import (
        apply_v1_gabor_weights,
        render_gridworld_to_image,
        image_to_retina_drive,
    )
    from sim.text_embeddings import vocab_to_drive_pattern
    from research.runners.g11_bg_runner import build_bg_brain_regions

    rng = np.random.default_rng(seed)

    regions, pathways = build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        pfc_enable_nmda=True,
        enable_visual_cortex=True,
        enable_text_io=True,
        n_motor_per_action=n_motor_per_action,
        text_n_input_neurons=text_n_input_neurons,
        text_n_output_neurons=text_n_output_neurons,
        enable_motor_cross_coupling=enable_motor_cross_coupling,
        motor_cross_coupling_weight=motor_cross_coupling_weight,
        motor_cross_coupling_density=motor_cross_coupling_density,
        # 2026-05-02 secondary fix: small non-zero init for readout
        # pathways. Original 0.0 init left these pathways at the
        # synaptic floor (0.01) after training — STDP couldn't grow them
        # from scratch with the weak training signal (~30% correct moves
        # giving sparse reward). Non-zero init lets STDP both strengthen
        # correct pairings (LTP) and weaken wrong ones (LTD) bidirectionally.
        text_cortex_to_output_weight=0.5,
        text_cortex_to_output_jitter=0.3,
        text_it_to_output_weight=0.5,
        text_it_to_output_jitter=0.3,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_structural_plasticity = False  # avoid CSR-grow bug
    # CRITICAL FIX (2026-05-02): disable Hebbian learning. Default is True
    # but ALL g* runners (g1-g11_bg) explicitly disable it. Without this,
    # the global hebbian_weight_decay (1e-5 per sub-step) compounds over
    # ~990K sub-steps in 100 ep training: (1-1e-5)^990000 ≈ 5e-5, driving
    # ALL weights from initial 2-3 down to hebbian_min_weight floor (0.05).
    # Weight diagnostic on text_eval_R3R6_100ep_NoT1_seed42.simstate.h5
    # confirmed: every plastic pathway at uniform 0.05 across all 4 directions
    # (no learning, just collapse). Disabling Hebbian eliminates the decay
    # while preserving STDP+reward modulation for the actual learning.
    # This was the silent cause of chance-level text I/O results since
    # 2026-05-01.
    cfg.enable_hebbian_learning = False
    # SECONDARY FIX (2026-05-02): raise stdp_w_max from default 2.0 to 5.0.
    # The PFC-bypass pathway (lang_input -> motor_X) has weight_mean=3.0
    # by design. STDP rule is soft-bound (Δw_LTP = A_plus × (w_max - w) ×
    # exp(...)); when current weight (3.0) > stdp_w_max (2.0), every
    # "LTP" event is NEGATIVE → pulls weights down to 2.0. Confirmed by
    # weight diagnostic on Hebbian-off run: PFC-bypass clipped at exactly
    # 2.0 max. CLAUDE.md documents this gotcha:
    # "set cfg.stdp_w_max above your design weights (e.g. cortex→D1 in
    # Phase B uses weight_mean=25 → set stdp_w_max=30)".
    # 5.0 leaves comfortable headroom over the 3.0 design weight.
    cfg.stdp_w_max = 5.0
    # Tier 1: per-type STP disabled by default. Only one trait pair (E→E) is
    # active in text training; per-type STP just adds a cp_synapse_conn_type
    # lookup per step with no benefit here. (Configurable as of 2026-05-02 to
    # support full Tier 1 ablation experiments.)
    cfg.enable_per_type_stp = enable_per_type_stp
    # Tested-and-reverted (2026-05-02 smoke):
    # - cfg.enable_ou_process = False    → BROKE NETWORK (correct-moves 2.4%
    #   vs 30%+ baseline). OU provides spontaneous activity that STDP needs
    #   for pre-synaptic spike events outside the explicit-input window.
    # - cfg.enable_parameter_heterogeneity = False → BROKE NETWORK (paired
    #   with the OU disable). Pure Izh parameters → pathological sync;
    #   real cortex relies on per-neuron variation to break lockstep.
    # See: 2026-05-02 smoke at correct_moves=2.4%, all-zero language_output spikes.

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    n_gabor = apply_v1_gabor_weights(
        bridge,
        n_orientations=8, n_frequencies=2, n_positions_per_dim=8,
        retina_size=32, receptive_field_radius=4, weight_scale=10.0,
    )
    if verbose:
        print(f"[embodied] Gabor: {n_gabor} edges installed")

    # Open all relevant gates
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
    n_lang_output = int(lang_output_idx.size)

    epoch_stats = []
    t_start = time.time()
    n_total_steps = 0
    n_correct_moves = 0

    for episode in range(n_episodes):
        # Balanced sampling: cycle through 4 directions so each gets equal
        # episodes. Counteracts the geometric bias where random (start, goal)
        # over-represents east/west by ~7pp due to |dx|>=|dy| tie-breaking.
        # See diagnostic in 2026-05-01-text-io-FINAL-summary.md.
        (start, goal, target_dir) = _sample_balanced_start_goal(rng, grid_size)
        x, y = start
        gx, gy = goal
        episode_target = target_dir

        for step in range(steps_per_episode):
            d_before = _manhattan((x, y), (gx, gy))
            if d_before == 0:
                # Reached goal — start fresh episode
                break

            target_word = _direction_from_positions((x, y), (gx, gy), rng=rng)

            # ─── Inter-step reset ───
            bridge.cp_external_input_current[:] = 0.0
            bridge.core_config.current_reward_signal = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            # ─── Apply embodied drive ───
            # Visual: render image, drive retina
            img = render_gridworld_to_image(
                agent_pos=(x, y), goal_pos=(gx, gy),
                grid_size=grid_size, image_size=32,
            )
            bridge.cp_external_input_current[retina_idx] = cp.asarray(
                image_to_retina_drive(img, drive_max_pA=retina_drive_pA),
                dtype=cp.float32,
            )
            # Language input: drive with target word (like external speaker)
            in_drive = vocab_to_drive_pattern(
                target_word,
                n_neurons=int(lang_input_idx.size),
                drive_max_pA=lang_input_drive_pA,
                sparsity=0.1,
            )
            bridge.cp_external_input_current[lang_input_idx] = cp.asarray(
                in_drive, dtype=cp.float32,
            )
            # Language output: MODEST coactivation (not clamp)
            # Models inner speech / motor word planning during action
            out_drive = vocab_to_drive_pattern(
                target_word,
                n_neurons=n_lang_output,
                drive_max_pA=lang_output_coactive_pA,
                sparsity=0.1,
            )
            bridge.cp_external_input_current[lang_output_idx] = cp.asarray(
                out_drive, dtype=cp.float32,
            )

            # ─── Run stim window, observe motor ───
            motor_counts = {a: 0 for a in ACTION_NAMES}
            bridge.core_config.current_reward_signal = 0.0  # reward AT END only
            for s in range(stim_steps_per_step):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                if 60 <= s < stim_steps_per_step:  # readout window
                    firing = bridge.cp_firing_states
                    for a in ACTION_NAMES:
                        motor_counts[a] += int(firing[cortex_idx_per_action[a]].sum().get())

            # ─── Action selection (argmax over cortex_X firing) ───
            chosen = max(motor_counts, key=lambda a: motor_counts[a])
            dx, dy = ACTION_DELTAS[ACTION_NAMES.index(chosen)]
            new_x = max(0, min(grid_size - 1, x + dx))
            new_y = max(0, min(grid_size - 1, y + dy))
            d_after = _manhattan((new_x, new_y), (gx, gy))

            # Real reward: did the move reduce Manhattan distance?
            reward = (correct_move_reward if d_after < d_before
                      else (wrong_move_reward if d_after > d_before else 0.0))
            bridge.core_config.current_reward_signal = float(reward)

            # Brief reward-application window (eligibility traces × reward → STDP)
            for _ in range(20):  # 10ms reward window
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            x, y = new_x, new_y
            n_total_steps += 1
            if reward > 0:
                n_correct_moves += 1

        if verbose and (episode + 1) % 10 == 0:
            print(f"  [ep {episode+1}/{n_episodes}] "
                  f"correct_moves={n_correct_moves}/{n_total_steps}="
                  f"{100*n_correct_moves/max(1,n_total_steps):.1f}%",
                  flush=True)

    elapsed = time.time() - t_start
    epoch_stats.append({
        "regime": "embodied_navigation_training",
        "n_episodes": n_episodes,
        "steps_per_episode": steps_per_episode,
        "n_total_steps": n_total_steps,
        "n_correct_moves": n_correct_moves,
        "correct_move_rate": n_correct_moves / max(1, n_total_steps),
        "elapsed_seconds": elapsed,
    })
    if verbose:
        print(f"\n[embodied] Training complete in {elapsed:.1f}s "
              f"({n_total_steps} steps, "
              f"{100*n_correct_moves/max(1,n_total_steps):.1f}% correct)",
              flush=True)

    if out_stats:
        Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(out_stats).write_text(json.dumps({
            "seed": seed,
            "grid_size": grid_size,
            "n_neurons": int(cfg.num_neurons),
            "epoch_stats": epoch_stats,
        }, indent=2))

    return bridge, epoch_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--steps-per-episode", type=int, default=30)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run_embodied_text_training(
        out_stats=args.out_stats,
        seed=args.seed,
        n_episodes=args.n_episodes,
        steps_per_episode=args.steps_per_episode,
        grid_size=args.grid_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
