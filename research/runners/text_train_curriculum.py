"""Curriculum-based embodied text training: visuomotor first, then text I/O.

Biology source: Vygotsky (1934/1978) zone of proximal development;
Piaget (1952) sensorimotor stage; Tomasello (2003) joint-attention
prerequisites for language. Real children master motor coordination
(roughly 6-12 months) BEFORE language production (roughly 12-24 months).
The motor cascade matures first, providing competent dynamics that
language learning can scaffold on.

This runner implements that two-phase development:

Phase 1: Visuomotor only (default 200 ep)
  - Language pathway plasticity FROZEN via set_plasticity_gate(0.0):
      language_input_to_cortex
      language_input_to_pfc
      language_input_to_motor (PFC bypass)
      it_to_language_output
      cortex_to_language_output
  - Language regions NOT driven (no token input/output during phase 1)
  - Visuomotor gates stay OPEN (visual_cortex_v1, _v2, _it, _action)
  - Reward signal as in standard embodied: +1 if Manhattan reduces,
    -0.5 if increases
  - Goal: cascade reaches 60%+ correct moves (clean training signal
    for phase 2)

Phase 2: Text I/O on trained cascade (default 100 ep)
  - Unfreeze all language pathway plasticity gates
  - Drive language_input + language_output as in standard R3+R6 regime
  - Visuomotor gates remain open (continued co-learning, biology-
    consistent — real cortex isn't fully closed for plasticity)
  - Same reward signal
  - STDP now operates on a CLEAN training signal because cascade
    dynamics are competent, not near-chance

Compared to v2 baseline (single-phase 100 ep):
  Expected: substantial accuracy boost on both I→W and W→A because
  language pathway STDP now gets clean target_motor firing patterns
  aligned with target words (instead of ~30% noisy ones).

Saves checkpoints at end of each phase for re-use:
  <out_stats>.phase1.simstate.h5  — visuomotor-only trained
  <out_stats>.simstate.h5          — full curriculum trained (final)

Usage:
  python -m research.runners.text_train_curriculum \\
      --phase1-episodes 200 --phase2-episodes 100 \\
      --seed 42 \\
      --out-stats research/findings/raw/g11_bg/text_eval_curriculum_seed42.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _direction_from_positions(agent_pos, goal_pos, rng=None) -> str:
    """Manhattan-greedy direction from agent to goal. Random tie-break
    when |dx|==|dy|. Same logic as text_train_embodied."""
    ax, ay = agent_pos
    gx, gy = goal_pos
    dx, dy = gx - ax, gy - ay
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    if abs(dy) > abs(dx):
        return "north" if dy > 0 else "south"
    if rng is None:
        return "east" if dx > 0 else ("west" if dx < 0 else "north")
    if rng.random() < 0.5:
        return "east" if dx > 0 else ("west" if dx < 0 else "north")
    return "north" if dy > 0 else ("south" if dy < 0 else "east")


def _sample_balanced_start_goal(rng, grid_size: int):
    """Same balanced sampling as text_train_embodied."""
    DIRECTIONS = ["north", "east", "south", "west"]
    target = DIRECTIONS[int(rng.integers(0, 4))]
    while True:
        ax = int(rng.integers(0, grid_size))
        ay = int(rng.integers(0, grid_size))
        if target in ("east", "west"):
            sign = 1 if target == "east" else -1
            for _ in range(50):
                dx_mag = int(rng.integers(1, grid_size))
                dy = int(rng.integers(-(dx_mag - 1), dx_mag))
                gx = ax + sign * dx_mag
                gy = ay + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    return (ax, ay), (gx, gy), target
        else:
            sign = 1 if target == "north" else -1
            for _ in range(50):
                dy_mag = int(rng.integers(1, grid_size))
                dx = int(rng.integers(-(dy_mag - 1), dy_mag))
                gx = ax + dx
                gy = ay + sign * dy_mag
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    return (ax, ay), (gx, gy), target


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Language pathway gates that get FROZEN during phase 1 + UNFROZEN in phase 2
LANGUAGE_GATES = [
    "language_input_to_cortex",
    "language_input_to_pfc",
    "language_input_to_motor",       # PFC-bypass
    "it_to_language_output",
    "cortex_to_language_output",
]

# Visuomotor gates that stay open through both phases
VISUOMOTOR_GATES = [
    "visual_cortex_v1",
    "visual_cortex_v2",
    "visual_cortex_it",
    "visual_cortex_action",
]


def _set_language_gates(bridge, value: float, verbose: bool = True):
    """Set all language-pathway plasticity gates to `value`.
    0.0 = frozen, 1.0 = full plasticity."""
    for gate in LANGUAGE_GATES:
        try:
            bridge.set_plasticity_gate(gate, value)
        except KeyError:
            pass  # Gate may not exist if pathway wasn't built
    if verbose:
        print(f"[curriculum] Language gates set to {value:.1f} ({len(LANGUAGE_GATES)} gates)")


def _set_visuomotor_gates(bridge, value: float, verbose: bool = True):
    """Set all visuomotor pathway plasticity gates to `value`."""
    for gate in VISUOMOTOR_GATES:
        try:
            bridge.set_plasticity_gate(gate, value)
        except KeyError:
            pass
    if verbose:
        print(f"[curriculum] Visuomotor gates set to {value:.1f} ({len(VISUOMOTOR_GATES)} gates)")


def _run_navigation_loop(
    bridge,
    cp,
    rng,
    n_episodes: int,
    steps_per_episode: int,
    grid_size: int,
    stim_steps_per_step: int,
    reset_steps: int,
    retina_drive_pA: float,
    correct_move_reward: float,
    wrong_move_reward: float,
    # Phase 2 only — language drives. Set to 0.0 for phase 1 (silent).
    lang_input_drive_pA: float = 0.0,
    lang_output_coactive_pA: float = 0.0,
    drive_language: bool = False,
    phase_label: str = "",
    # Phase 2 only — record waking experiences for Phase 3 SWR replay
    experience_buffer: list | None = None,
    # Sparse-coding sparsity for token drive (0.1 default; 0.05 = orthogonal)
    token_sparsity: float = 0.1,
    verbose: bool = True,
):
    """Inner training loop. Same as run_embodied_text_training but
    parameterized to allow language drive to be off (phase 1) or on
    (phase 2). Plasticity gating is the caller's responsibility."""
    from sim.visual_cortex import (
        render_gridworld_to_image,
        image_to_retina_drive,
    )
    from sim.text_embeddings import vocab_to_drive_pattern

    retina_idx = cp.asarray(
        list(bridge.region_manager.indices("retina")), dtype=cp.int64
    )
    if drive_language:
        lang_input_idx = cp.asarray(
            list(bridge.region_manager.indices("language_input")), dtype=cp.int64
        )
        lang_output_idx = cp.asarray(
            list(bridge.region_manager.indices("language_output")), dtype=cp.int64
        )
        n_lang_output = int(lang_output_idx.size)
        n_lang_input = int(lang_input_idx.size)
    else:
        lang_input_idx = lang_output_idx = None
        n_lang_input = n_lang_output = 0

    cortex_idx_per_action = {
        a: cp.asarray(list(bridge.region_manager.indices(f"cortex_{a}")),
                      dtype=cp.int64)
        for a in ACTION_NAMES
    }

    n_total_steps = 0
    n_correct_moves = 0

    for episode in range(n_episodes):
        (start, goal, _) = _sample_balanced_start_goal(rng, grid_size)
        x, y = start
        gx, gy = goal

        for step in range(steps_per_episode):
            d_before = _manhattan((x, y), (gx, gy))
            if d_before == 0:
                break

            target_word = _direction_from_positions((x, y), (gx, gy), rng=rng)

            # Inter-step reset
            bridge.cp_external_input_current[:] = 0.0
            bridge.core_config.current_reward_signal = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            # Apply embodied drive
            img = render_gridworld_to_image(
                agent_pos=(x, y), goal_pos=(gx, gy),
                grid_size=grid_size, image_size=32,
            )
            bridge.cp_external_input_current[retina_idx] = cp.asarray(
                image_to_retina_drive(img, drive_max_pA=retina_drive_pA),
                dtype=cp.float32,
            )
            if drive_language:
                in_drive = vocab_to_drive_pattern(
                    target_word,
                    n_neurons=n_lang_input,
                    drive_max_pA=lang_input_drive_pA,
                    sparsity=token_sparsity,
                )
                bridge.cp_external_input_current[lang_input_idx] = cp.asarray(
                    in_drive, dtype=cp.float32,
                )
                out_drive = vocab_to_drive_pattern(
                    target_word,
                    n_neurons=n_lang_output,
                    drive_max_pA=lang_output_coactive_pA,
                    sparsity=token_sparsity,
                )
                bridge.cp_external_input_current[lang_output_idx] = cp.asarray(
                    out_drive, dtype=cp.float32,
                )

            # Run stim window, observe motor
            motor_counts = {a: 0 for a in ACTION_NAMES}
            bridge.core_config.current_reward_signal = 0.0
            for s in range(stim_steps_per_step):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                if 60 <= s < stim_steps_per_step:
                    firing = bridge.cp_firing_states
                    for a in ACTION_NAMES:
                        motor_counts[a] += int(firing[cortex_idx_per_action[a]].sum().get())

            # Action selection
            chosen = max(motor_counts, key=lambda a: motor_counts[a])
            dx, dy = ACTION_DELTAS[ACTION_NAMES.index(chosen)]
            new_x = max(0, min(grid_size - 1, x + dx))
            new_y = max(0, min(grid_size - 1, y + dy))
            d_after = _manhattan((new_x, new_y), (gx, gy))

            reward = (correct_move_reward if d_after < d_before
                      else (wrong_move_reward if d_after > d_before else 0.0))
            bridge.core_config.current_reward_signal = float(reward)

            # Reward window
            for _ in range(20):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            # Record experience for Phase 3 replay (only when language is
            # active i.e. Phase 2). Records (token, action_chosen, reward,
            # was_correct) — full tuple so replay can reinforce successful
            # pairings differently from failed ones.
            if experience_buffer is not None and drive_language:
                experience_buffer.append({
                    "token": target_word,
                    "action": chosen,
                    "reward": float(reward),
                    "correct_move": (reward > 0),
                })

            x, y = new_x, new_y
            n_total_steps += 1
            if reward > 0:
                n_correct_moves += 1

        if verbose and (episode + 1) % 10 == 0:
            pct = 100 * n_correct_moves / max(1, n_total_steps)
            print(f"  [{phase_label} ep {episode+1}/{n_episodes}] "
                  f"correct_moves={n_correct_moves}/{n_total_steps}={pct:.1f}%",
                  flush=True)

    return n_total_steps, n_correct_moves


def _run_swr_replay_phase(
    bridge,
    cp,
    rng,
    experience_buffer: list,
    n_replay_events: int,
    stim_steps_per_step: int,
    reset_steps: int,
    lang_input_drive_pA: float,
    lang_output_coactive_pA: float,
    motor_replay_drive_pA: float = 50.0,
    only_correct_experiences: bool = True,
    balanced_directions: bool = False,
    token_sparsity: float = 0.1,
    verbose: bool = True,
):
    """Phase 3: SWR-style consolidation replay.

    Re-presents recent (token, action) tuples from the experience buffer
    to consolidate language-pathway weights via offline replay.
    Loosely models hippocampal sharp-wave-ripple consolidation
    (Wilson & McNaughton 1994; Buzsáki 1986+) without requiring full
    Cluster D infrastructure.

    For each replay event:
    1. Reset
    2. Drive language_input + language_output with token pattern (as in waking)
    3. Drive motor_X (or motor_pop_θ for distributed) for the correct action
    4. Run shorter stim window (compressed time, ~50ms vs waking 100ms)
    5. Apply +1 reward to reinforce STDP for correct pairings

    Optionally filters to only_correct_experiences (more stable) or replays
    all (more biology-faithful — replay includes failures too).

    2026-05-03: balanced_directions flag (H1 test). When True, the replay
    sampler draws an equal number of events from each direction word,
    instead of uniform-random over the buffer. The 2-seed v2+SWR result
    showed a consistent W->A regression (22% vs baseline 28.5%) which
    might be caused by replay-distribution bias toward N/E (the cascade's
    natural majority in training). Balanced sampling tests that hypothesis.
    See research/findings/2026-05-03-swr-2seed-result.md.

    Args:
        experience_buffer: list of dicts with token, action, correct_move
        n_replay_events: total replay events
        only_correct_experiences: if True, sample only from correct moves
            (cleaner consolidation signal). If False, replay all events
            with their original reward.
        balanced_directions: if True, draw n_replay_events / 4 from each
            of the 4 direction words (N/E/S/W). Falls back to uniform
            sampling for any direction with zero buffer events.
    """
    from sim.text_embeddings import vocab_to_drive_pattern

    if not experience_buffer:
        if verbose:
            print("[curriculum] Phase 3: empty experience buffer, skipping replay")
        return 0

    # Filter buffer per option
    if only_correct_experiences:
        buffer = [e for e in experience_buffer if e["correct_move"]]
        if not buffer:
            if verbose:
                print("[curriculum] Phase 3: no correct moves to replay, skipping")
            return 0
    else:
        buffer = list(experience_buffer)

    # 2026-05-03 (H1): if balanced_directions, build a per-direction replay
    # plan that draws an equal number of events from each of N/E/S/W.
    # We pre-compute the sample list so each direction gets exactly its
    # share regardless of buffer skew.
    direction_buffer = {"north": [], "east": [], "south": [], "west": []}
    for e in buffer:
        # Map token (which is the direction word) to a bucket. Some events
        # might use different conventions, but the buffer's token field
        # is the direction word.
        tok = e.get("token")
        if tok in direction_buffer:
            direction_buffer[tok].append(e)
    sampled_events = None
    if balanced_directions:
        per_dir = n_replay_events // 4
        # If a direction has no events, sample from the global buffer
        # to avoid wasted slots.
        sampled_events = []
        for dirname in ("north", "east", "south", "west"):
            pool = direction_buffer[dirname] if direction_buffer[dirname] else buffer
            for _ in range(per_dir):
                sampled_events.append(pool[int(rng.integers(0, len(pool)))])
        # Pad to exactly n_replay_events (rounding remainder)
        while len(sampled_events) < n_replay_events:
            sampled_events.append(buffer[int(rng.integers(0, len(buffer)))])
        # Shuffle so the directions don't replay in blocks
        rng.shuffle(sampled_events)

    if verbose:
        n_correct = sum(1 for e in experience_buffer if e["correct_move"])
        if balanced_directions:
            counts = {d: len(direction_buffer[d]) for d in direction_buffer}
            print(f"[curriculum] Phase 3 SWR BALANCED replay: {n_replay_events} events "
                  f"({n_replay_events//4}/dir) from buffer of {len(experience_buffer)} "
                  f"(correct={n_correct}); per-direction buffer sizes: {counts}",
                  flush=True)
        else:
            print(f"[curriculum] Phase 3 SWR replay: {n_replay_events} events from "
                  f"buffer of {len(experience_buffer)} (correct={n_correct}, "
                  f"replay_correct_only={only_correct_experiences})", flush=True)

    # Cache region indices
    rm = bridge.region_manager
    lang_input_idx = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    lang_output_idx = cp.asarray(list(rm.indices("language_output")), dtype=cp.int64)
    n_lang_input = int(lang_input_idx.size)
    n_lang_output = int(lang_output_idx.size)

    # Detect labeled-line vs distributed motor architecture
    distributed_motor_pop = False
    try:
        rm.indices("motor_N")
    except KeyError:
        try:
            rm.indices("motor_pop_N")
            distributed_motor_pop = True
        except KeyError:
            pass

    if distributed_motor_pop:
        # Cosine-tuned drive scheme matching the architecture's tuning
        SUBPOOL_THETA = [
            (0, "E"), (45, "NE"), (90, "N"), (135, "NW"),
            (180, "W"), (225, "SW"), (270, "S"), (315, "SE"),
        ]
        ACTION_THETA = {"N": 90, "E": 0, "S": 270, "W": 180}
        import math
        subpool_idx = {
            sfx: cp.asarray(list(rm.indices(f"motor_pop_{sfx}")), dtype=cp.int64)
            for theta, sfx in SUBPOOL_THETA
        }
        # Per-action drive pattern: cos-weighted across sub-pools
        action_subpool_drive = {}
        for action, theta_a in ACTION_THETA.items():
            action_subpool_drive[action] = {}
            for theta_p, sfx in SUBPOOL_THETA:
                d = ((theta_a - theta_p + 180) % 360) - 180
                cos_w = max(0.0, math.cos(math.radians(d)))
                action_subpool_drive[action][sfx] = cos_w
    else:
        motor_idx = {
            a: cp.asarray(list(rm.indices(f"motor_{a}")), dtype=cp.int64)
            for a in ACTION_NAMES
        }

    n_replays = 0

    for event_idx in range(n_replay_events):
        # Sample the next replay event. In default mode this is
        # uniform-random from the buffer (which inherits the cascade's
        # natural N-bias). In balanced mode, the pre-computed
        # sampled_events list draws equally from N/E/S/W and shuffles.
        if sampled_events is not None:
            event = sampled_events[event_idx]
        else:
            event = buffer[int(rng.integers(0, len(buffer)))]
        token = event["token"]
        action = event["action"]
        reward = event["reward"]

        # Inter-trial reset
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Drive language_input + language_output (replay the waking pattern)
        in_drive = vocab_to_drive_pattern(
            token, n_neurons=n_lang_input,
            drive_max_pA=lang_input_drive_pA, sparsity=token_sparsity,
        )
        bridge.cp_external_input_current[lang_input_idx] = cp.asarray(
            in_drive, dtype=cp.float32,
        )
        out_drive = vocab_to_drive_pattern(
            token, n_neurons=n_lang_output,
            drive_max_pA=lang_output_coactive_pA, sparsity=token_sparsity,
        )
        bridge.cp_external_input_current[lang_output_idx] = cp.asarray(
            out_drive, dtype=cp.float32,
        )

        # Drive motor pool for the chosen action — gives the post-synaptic
        # firing pattern that STDP needs to pair with language drive.
        # SMALL drive (~50pA) so endogenous cascade dynamics still play
        # role; this is a "nudge" toward the recorded action.
        if distributed_motor_pop:
            for sfx, weight in action_subpool_drive[action].items():
                if weight > 0.01:
                    bridge.cp_external_input_current[subpool_idx[sfx]] += (
                        weight * motor_replay_drive_pA
                    )
        else:
            bridge.cp_external_input_current[motor_idx[action]] += (
                motor_replay_drive_pA
            )

        # Compressed stim window (half of waking — biology says replay is
        # 10-20× compressed but at our scale, half is enough).
        compressed_stim = max(50, stim_steps_per_step // 2)
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(compressed_stim):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Apply reward — reinforce the (token, action) pairing
        bridge.core_config.current_reward_signal = float(reward)
        for _ in range(20):  # reward window
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        n_replays += 1

        if verbose and (event_idx + 1) % 100 == 0:
            print(f"  [P3 SWR] {event_idx+1}/{n_replay_events} replayed",
                  flush=True)

    return n_replays


def run_curriculum_training(
    out_stats: str | None = None,
    seed: int = 42,
    phase1_episodes: int = 200,
    phase2_episodes: int = 100,
    phase3_replays: int = 0,  # 0 = skip Phase 3 SWR replay (default)
    phase3_only_correct: bool = True,
    phase3_balanced_directions: bool = False,  # H1 test (2026-05-03)
    steps_per_episode: int = 30,
    grid_size: int = 8,
    # Tier 1 / config (matches v2 baseline)
    stim_steps_per_step: int = 200,  # full revert (Tier 1.1 reverted)
    reset_steps: int = 100,
    enable_per_type_stp: bool = False,
    # Drives
    retina_drive_pA: float = 200.0,
    lang_input_drive_pA: float = 200.0,
    lang_output_coactive_pA: float = 150.0,
    # Reward
    correct_move_reward: float = 1.0,
    wrong_move_reward: float = -0.5,
    # Architecture sizing
    n_motor_per_action: int = 10,
    text_n_input_neurons: int = 256,
    text_n_output_neurons: int = 256,
    # Distributed motor pool option
    enable_distributed_motor_pop: bool = False,
    n_motor_pop_per_subpool: int = 5,
    # Sparse-coding sparsity for token drive (0.1 default; 0.05 = orthogonal)
    token_sparsity: float = 0.1,
    # Plasticity overrides (default = v2 baseline values)
    # 2026-05-03: exposed for fundamentals sweep — Hebbian re-enable
    # with reduced decay tests biology-correct learning vs the v2
    # disable-it shortcut.
    enable_hebbian_learning: bool = False,
    hebbian_weight_decay: float | None = None,
    hebbian_learning_rate: float | None = None,
    stdp_w_max: float = 5.0,
    save_phase1_checkpoint: bool = True,
    verbose: bool = True,
):
    """Two-phase curriculum training."""
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from sim.visual_cortex import apply_v1_gabor_weights
    from research.runners.g11_bg_runner import build_bg_brain_regions

    rng = np.random.default_rng(seed)

    # Build same architecture as v2 baseline
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
        text_cortex_to_output_weight=0.5,
        text_cortex_to_output_jitter=0.3,
        text_it_to_output_weight=0.5,
        text_it_to_output_jitter=0.3,
        enable_distributed_motor_pop=enable_distributed_motor_pop,
        n_motor_pop_per_subpool=n_motor_pop_per_subpool,
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
    cfg.enable_per_type_stp = enable_per_type_stp
    # v2 fixes (overridable via kwargs for the fundamentals sweep)
    cfg.enable_hebbian_learning = enable_hebbian_learning
    cfg.stdp_w_max = stdp_w_max
    if hebbian_weight_decay is not None:
        cfg.hebbian_weight_decay = hebbian_weight_decay
    if hebbian_learning_rate is not None:
        cfg.hebbian_learning_rate = hebbian_learning_rate

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    apply_v1_gabor_weights(
        bridge,
        n_orientations=8, n_frequencies=2, n_positions_per_dim=8,
        retina_size=32, receptive_field_radius=4, weight_scale=10.0,
    )

    epoch_stats = []
    t_start = time.time()

    # ─────────────────────────────────────────────────────────────────
    # PHASE 1: Visuomotor only (language pathway plasticity frozen)
    # ─────────────────────────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print(f"PHASE 1: Visuomotor only — {phase1_episodes} episodes")
        print(f"  Language plasticity FROZEN, language regions NOT driven")
        print("=" * 60, flush=True)

    _set_language_gates(bridge, 0.0, verbose=verbose)
    _set_visuomotor_gates(bridge, 1.0, verbose=verbose)

    p1_steps, p1_correct = _run_navigation_loop(
        bridge, cp, rng,
        n_episodes=phase1_episodes,
        steps_per_episode=steps_per_episode,
        grid_size=grid_size,
        stim_steps_per_step=stim_steps_per_step,
        reset_steps=reset_steps,
        retina_drive_pA=retina_drive_pA,
        correct_move_reward=correct_move_reward,
        wrong_move_reward=wrong_move_reward,
        drive_language=False,
        phase_label="P1",
        verbose=verbose,
    )

    p1_elapsed = time.time() - t_start
    p1_rate = p1_correct / max(1, p1_steps)
    if verbose:
        print(f"\n[curriculum] Phase 1 complete: "
              f"{p1_correct}/{p1_steps} = {p1_rate:.1%} correct moves "
              f"({p1_elapsed:.0f}s)", flush=True)

    epoch_stats.append({
        "phase": 1,
        "regime": "visuomotor_only",
        "n_episodes": phase1_episodes,
        "n_total_steps": p1_steps,
        "n_correct_moves": p1_correct,
        "correct_move_rate": p1_rate,
        "elapsed_seconds": p1_elapsed,
    })

    # Save phase 1 checkpoint for re-use
    if save_phase1_checkpoint and out_stats:
        p1_ckpt_path = Path(out_stats).with_suffix(".phase1.simstate.h5")
        try:
            bridge.save_checkpoint(str(p1_ckpt_path))
            if verbose:
                print(f"[curriculum] Phase 1 checkpoint saved: {p1_ckpt_path}",
                      flush=True)
        except Exception as e:
            if verbose:
                print(f"[curriculum] WARNING: phase 1 checkpoint failed: {e}",
                      flush=True)

    # ─────────────────────────────────────────────────────────────────
    # PHASE 2: Text I/O on trained cascade
    # ─────────────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print(f"PHASE 2: Text I/O training — {phase2_episodes} episodes")
        print(f"  Language plasticity UNFROZEN, language regions DRIVEN")
        print("=" * 60, flush=True)

    _set_language_gates(bridge, 1.0, verbose=verbose)
    # Visuomotor gates remain open (continued co-learning)

    # Allocate experience buffer if Phase 3 is enabled
    experience_buffer = [] if phase3_replays > 0 else None

    t_phase2_start = time.time()
    p2_steps, p2_correct = _run_navigation_loop(
        bridge, cp, rng,
        n_episodes=phase2_episodes,
        steps_per_episode=steps_per_episode,
        grid_size=grid_size,
        stim_steps_per_step=stim_steps_per_step,
        reset_steps=reset_steps,
        retina_drive_pA=retina_drive_pA,
        correct_move_reward=correct_move_reward,
        wrong_move_reward=wrong_move_reward,
        drive_language=True,
        lang_input_drive_pA=lang_input_drive_pA,
        lang_output_coactive_pA=lang_output_coactive_pA,
        experience_buffer=experience_buffer,
        phase_label="P2",
        token_sparsity=token_sparsity,
        verbose=verbose,
    )

    p2_elapsed = time.time() - t_phase2_start
    p2_rate = p2_correct / max(1, p2_steps)
    if verbose:
        print(f"\n[curriculum] Phase 2 complete: "
              f"{p2_correct}/{p2_steps} = {p2_rate:.1%} correct moves "
              f"({p2_elapsed:.0f}s)", flush=True)

    # 2026-05-03: per-direction buffer composition. Tells us which direction
    # the cascade naturally over-emits during Phase 2, which directly
    # determines the SWR replay distribution. Useful for diagnosing whether
    # H1 (balanced replay) is needed: if the buffer is N/E/S/W skewed, H1
    # should help.
    buffer_per_direction = {"north": 0, "east": 0, "south": 0, "west": 0}
    if experience_buffer:
        for ev in experience_buffer:
            tok = ev.get("token")
            if tok in buffer_per_direction:
                buffer_per_direction[tok] += 1
    if verbose and experience_buffer:
        total = sum(buffer_per_direction.values())
        print(f"[curriculum] Phase 2 buffer composition: "
              f"north={buffer_per_direction['north']} ({100 * buffer_per_direction['north'] / max(1, total):.0f}%) "
              f"east={buffer_per_direction['east']} ({100 * buffer_per_direction['east'] / max(1, total):.0f}%) "
              f"south={buffer_per_direction['south']} ({100 * buffer_per_direction['south'] / max(1, total):.0f}%) "
              f"west={buffer_per_direction['west']} ({100 * buffer_per_direction['west'] / max(1, total):.0f}%)",
              flush=True)

    epoch_stats.append({
        "phase": 2,
        "regime": "text_io_on_trained_cascade",
        "n_episodes": phase2_episodes,
        "n_total_steps": p2_steps,
        "n_correct_moves": p2_correct,
        "correct_move_rate": p2_rate,
        "elapsed_seconds": p2_elapsed,
        "buffer_per_direction": buffer_per_direction,
    })

    # ─────────────────────────────────────────────────────────────────
    # PHASE 3 (optional): SWR-style consolidation replay
    # ─────────────────────────────────────────────────────────────────
    if phase3_replays > 0 and experience_buffer:
        if verbose:
            print("\n" + "=" * 60)
            print(f"PHASE 3: SWR replay consolidation — {phase3_replays} events")
            print(f"  Language gates remain UNFROZEN. Replay biases STDP "
                  f"toward recent (token, action) pairings.")
            print("=" * 60, flush=True)

        t_phase3_start = time.time()
        n_replayed = _run_swr_replay_phase(
            bridge, cp, rng,
            experience_buffer=experience_buffer,
            n_replay_events=phase3_replays,
            stim_steps_per_step=stim_steps_per_step,
            reset_steps=reset_steps,
            lang_input_drive_pA=lang_input_drive_pA,
            lang_output_coactive_pA=lang_output_coactive_pA,
            only_correct_experiences=phase3_only_correct,
            balanced_directions=phase3_balanced_directions,
            token_sparsity=token_sparsity,
            verbose=verbose,
        )
        p3_elapsed = time.time() - t_phase3_start
        if verbose:
            print(f"\n[curriculum] Phase 3 complete: {n_replayed} replays "
                  f"({p3_elapsed:.0f}s)", flush=True)

        epoch_stats.append({
            "phase": 3,
            "regime": "swr_replay_consolidation",
            "n_replay_events": n_replayed,
            "buffer_size": len(experience_buffer),
            "only_correct": phase3_only_correct,
            "balanced_directions": phase3_balanced_directions,
            "elapsed_seconds": p3_elapsed,
        })

    total_elapsed = time.time() - t_start
    if verbose:
        print(f"\n[curriculum] Total training: {total_elapsed:.0f}s "
              f"({phase1_episodes + phase2_episodes} ep + "
              f"{phase3_replays} replays)", flush=True)

    return bridge, epoch_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase1-episodes", type=int, default=200,
                    help="visuomotor-only training (default 200)")
    ap.add_argument("--phase2-episodes", type=int, default=100,
                    help="text-IO training on trained cascade (default 100)")
    ap.add_argument("--phase3-replays", type=int, default=0,
                    help="SWR consolidation replays after phase 2 (default 0 "
                    "= skip Phase 3). Try 500 for ~5x phase 2's plastic events.")
    ap.add_argument("--phase3-only-correct", action="store_true", default=True,
                    help="phase 3 replays only correct (token, action) pairs "
                    "(default True for cleaner consolidation)")
    ap.add_argument("--phase3-replay-all", dest="phase3_only_correct",
                    action="store_false",
                    help="phase 3 replays all experiences including failures")
    ap.add_argument("--phase3-balanced-directions", action="store_true",
                    default=False,
                    help="(H1, 2026-05-03) phase 3 draws equal events from "
                    "each of N/E/S/W rather than uniform-random over the "
                    "buffer. Tests whether the W->A regression observed at "
                    "n=2 with default (frequency-weighted) replay is caused "
                    "by N/E-biased replay distribution. See "
                    "research/findings/2026-05-03-swr-2seed-result.md")
    ap.add_argument("--steps-per-episode", type=int, default=30)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--n-eval-image-word", type=int, default=100)
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--retina-drive-pA", type=float, default=200.0)
    ap.add_argument("--lang-input-drive-pA", type=float, default=200.0)
    ap.add_argument("--lang-output-coactive-pA", type=float, default=150.0)
    ap.add_argument("--stim-steps-per-step", type=int, default=200)
    ap.add_argument("--reset-steps", type=int, default=100)
    ap.add_argument("--correct-move-reward", type=float, default=1.0)
    ap.add_argument("--wrong-move-reward", type=float, default=-0.5)
    ap.add_argument("--n-motor-per-action", type=int, default=10)
    ap.add_argument("--text-n-input-neurons", type=int, default=256)
    ap.add_argument("--text-n-output-neurons", type=int, default=256)
    # Distributed motor pool option (Pulvermüller G.20, 2026-05-02)
    ap.add_argument("--enable-distributed-motor-pop", action="store_true",
                    help="use 8 motor_pop sub-pools at 45deg intervals "
                    "instead of 4 labeled motor_X pools")
    ap.add_argument("--n-motor-pop-per-subpool", type=int, default=5,
                    help="neurons per sub-pool (default 5; 8x5=40)")
    ap.add_argument("--token-sparsity", type=float, default=0.1,
                    help="fraction of language_input neurons activated per "
                    "word (default 0.1, ~26 active at n=256). Try 0.05 for "
                    "near-orthogonal codes (12-13 active at n=256, "
                    "pairwise overlap ~0). Eval and training use this same "
                    "value to stay consistent.")
    ap.add_argument("--enable-hebbian", dest="enable_hebbian_learning",
                    action="store_true", default=False,
                    help="re-enable Hebbian learning (v2 default disables "
                    "it because of weight-decay collapse, but the right "
                    "biology-correct fix is to reduce decay, not disable). "
                    "Pair with --hebbian-weight-decay 1e-7 (100x lower).")
    ap.add_argument("--hebbian-weight-decay", type=float, default=None,
                    help="override Hebbian weight decay rate (sim default "
                    "1e-5). Try 1e-7 for biology-correct slow decay.")
    ap.add_argument("--hebbian-learning-rate", type=float, default=None,
                    help="override Hebbian learning rate (sim default 5e-4).")
    ap.add_argument("--stdp-w-max", type=float, default=5.0,
                    help="STDP soft-bound max weight (v2 default 5.0). "
                    "Try 10.0 to allow more weight differentiation between "
                    "word-action pairings.")
    ap.add_argument("--no-save-checkpoint", dest="save_checkpoint",
                    action="store_false")
    ap.set_defaults(save_checkpoint=True)
    args = ap.parse_args()

    print("=" * 60)
    print(f"CURRICULUM TRAINING (seed={args.seed})")
    print(f"  Phase 1: {args.phase1_episodes} ep visuomotor-only")
    print(f"  Phase 2: {args.phase2_episodes} ep text-IO on trained cascade")
    print(f"  Total: {args.phase1_episodes + args.phase2_episodes} ep "
          f"x {args.steps_per_episode} steps/ep")
    print("=" * 60, flush=True)

    bridge, train_stats = run_curriculum_training(
        out_stats=args.out_stats,
        seed=args.seed,
        phase1_episodes=args.phase1_episodes,
        phase2_episodes=args.phase2_episodes,
        phase3_replays=args.phase3_replays,
        phase3_only_correct=args.phase3_only_correct,
        phase3_balanced_directions=args.phase3_balanced_directions,
        steps_per_episode=args.steps_per_episode,
        grid_size=args.grid_size,
        stim_steps_per_step=args.stim_steps_per_step,
        reset_steps=args.reset_steps,
        retina_drive_pA=args.retina_drive_pA,
        lang_input_drive_pA=args.lang_input_drive_pA,
        lang_output_coactive_pA=args.lang_output_coactive_pA,
        correct_move_reward=args.correct_move_reward,
        wrong_move_reward=args.wrong_move_reward,
        n_motor_per_action=args.n_motor_per_action,
        text_n_input_neurons=args.text_n_input_neurons,
        text_n_output_neurons=args.text_n_output_neurons,
        enable_distributed_motor_pop=args.enable_distributed_motor_pop,
        n_motor_pop_per_subpool=args.n_motor_pop_per_subpool,
        token_sparsity=args.token_sparsity,
        enable_hebbian_learning=args.enable_hebbian_learning,
        hebbian_weight_decay=args.hebbian_weight_decay,
        hebbian_learning_rate=args.hebbian_learning_rate,
        stdp_w_max=args.stdp_w_max,
        save_phase1_checkpoint=args.save_checkpoint,
        verbose=True,
    )

    # Eval after curriculum
    from research.runners.text_eval import (
        evaluate_image_to_word, evaluate_word_to_action,
    )
    print("\n" + "=" * 60)
    print(f"EVAL: image -> word ({args.n_eval_image_word} fresh trials)")
    print("=" * 60, flush=True)
    iw_result = evaluate_image_to_word(
        bridge, n_trials=args.n_eval_image_word, grid_size=args.grid_size,
    )
    print(f"\n  Accuracy: {iw_result['correct']}/{iw_result['n_trials']} "
          f"= {iw_result['accuracy']:.1%}")
    print(f"  Confusion: {iw_result['confusion_matrix']}", flush=True)

    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_word_action} per word, "
          f"token_sparsity={args.token_sparsity})")
    print("=" * 60, flush=True)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_word_action,
        token_sparsity=args.token_sparsity,
    )
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}")
    print(f"  Confusion: {wa_result['confusion_matrix']}", flush=True)

    if args.out_stats:
        out = {
            "regime": "curriculum_embodied",
            "seed": args.seed,
            "phase1_episodes": args.phase1_episodes,
            "phase2_episodes": args.phase2_episodes,
            "steps_per_episode": args.steps_per_episode,
            "training_stats": train_stats,
            "image_to_word_eval": iw_result,
            "word_to_action_eval": wa_result,
            "config": {
                "retina_drive_pA": args.retina_drive_pA,
                "lang_input_drive_pA": args.lang_input_drive_pA,
                "lang_output_coactive_pA": args.lang_output_coactive_pA,
                "stim_steps_per_step": args.stim_steps_per_step,
                "reset_steps": args.reset_steps,
                "n_motor_per_action": args.n_motor_per_action,
                "text_n_input_neurons": args.text_n_input_neurons,
                "text_n_output_neurons": args.text_n_output_neurons,
                "token_sparsity": args.token_sparsity,
            },
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(
            json.dumps(out, indent=2, default=str)
        )
        print(f"\n  Saved: {args.out_stats}", flush=True)

        if args.save_checkpoint:
            ckpt_path = Path(args.out_stats).with_suffix(".simstate.h5")
            try:
                bridge.save_checkpoint(str(ckpt_path))
                print(f"  Saved checkpoint: {ckpt_path}", flush=True)
            except Exception as e:
                print(f"  WARNING: checkpoint save failed: {e}", flush=True)


if __name__ == "__main__":
    main()
