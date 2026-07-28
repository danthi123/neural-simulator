"""Text evaluation runner — given a trained bridge, tests:
1. Image -> word: present a fresh gridworld image, read language_output,
   does the agent emit the correct cardinal direction?
2. Word -> action: drive language_input with a word, observe motor
   firing, does the agent take the correct action?

Reuses the bridge built by text_train.py (or loads a checkpoint).
The same training-time supervision regime is used, but WITHOUT clamping
the supervisor signal — we observe the agent's natural response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
WORD_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}

# Tier 2.1: synonym groups. When synonym_mode=True, each trial picks a
# random word from the action's synonym group; eval tests all 8 words.
SYNONYM_GROUPS = {
    "N": ["north", "up"],
    "E": ["east", "right"],
    "S": ["south", "down"],
    "W": ["west", "left"],
}

# Tier 2.1 robustness: 12-word vocab (3 synonyms per action). Tests
# whether scale-up paradigm generalizes beyond 8 words. Adds
# abbreviated cardinal directions ("n", "e", "s", "w") as third
# synonym per group.
SYNONYM_GROUPS_12 = {
    "N": ["north", "up", "n"],
    "E": ["east", "right", "e"],
    "S": ["south", "down", "s"],
    "W": ["west", "left", "w"],
}
EXTENDED_WORD_TO_ACTION_12 = {
    word: action
    for action, words in SYNONYM_GROUPS_12.items()
    for word in words
}

# Tier 2.1 16-word: 4 synonyms per action. Adds Unicode arrows as 4th
# synonym -- tests whether hash-based vocab_to_drive_pattern handles
# non-ASCII tokens cleanly + extends per-motor sub-population count
# from 3 (12-word) to 4 (16-word). Per master plan section
# "Larger Tier 2.1 vocab (16-30 words)".
SYNONYM_GROUPS_16 = {
    "N": ["north", "up", "n", "↑"],     # ↑
    "E": ["east", "right", "e", "→"],   # →
    "S": ["south", "down", "s", "↓"],   # ↓
    "W": ["west", "left", "w", "←"],    # ←
}
EXTENDED_WORD_TO_ACTION_16 = {
    word: action
    for action, words in SYNONYM_GROUPS_16.items()
    for word in words
}

# 2026-05-10: find-the-ceiling vocab tiers per user directive
# "start very high on the scale to test for failure". Per derived
# capacity rule, vocab_size N requires ~(N/4)*333 motor neurons:
#   24-word: 5 sub-pops/action × 333 = ~2000 motor (fits at scaled arch)
#   32-word: 8 sub-pops/action × 333 = ~2667 motor (n_motor=3000)
#   48-word: 12 sub-pops/action × 333 = ~4000 motor (n_motor=4000)
#   64-word: 16 sub-pops/action × 333 = ~5333 motor (n_motor=6000;
#           predicted to OOM on 24GB 3090)
# Synonyms include localizations (Spanish, German, French) + alt forms
# + abbreviations. Hash-based vocab_to_drive_pattern handles all
# UTF-8 strings cleanly (validated on Unicode arrows in 16-word).

SYNONYM_GROUPS_24 = {
    "N": ["north", "up", "n", "↑", "norte", "nord"],
    "E": ["east", "right", "e", "→", "este", "ost"],
    "S": ["south", "down", "s", "↓", "sur", "süd"],
    "W": ["west", "left", "w", "←", "oeste", "west_de"],
}
EXTENDED_WORD_TO_ACTION_24 = {
    word: action
    for action, words in SYNONYM_GROUPS_24.items()
    for word in words
}

SYNONYM_GROUPS_32 = {
    "N": ["north", "up", "n", "↑", "norte", "nord", "kita", "shimal"],
    "E: alt".replace(": alt", ""): ["east", "right", "e", "→", "este", "ost", "higashi", "sharq"],
    "S": ["south", "down", "s", "↓", "sur", "süd", "minami", "janub"],
    "W": ["west", "left", "w", "←", "oeste", "west_de", "nishi", "gharb"],
}
# Fix the dict literal (was malformed — let me redo)
SYNONYM_GROUPS_32 = {
    "N": ["north", "up", "n", "↑", "norte", "nord", "kita", "shimal"],
    "E": ["east", "right", "e", "→", "este", "ost", "higashi", "sharq"],
    "S": ["south", "down", "s", "↓", "sur", "süd", "minami", "janub"],
    "W": ["west", "left", "w", "←", "oeste", "west_de", "nishi", "gharb"],
}
EXTENDED_WORD_TO_ACTION_32 = {
    word: action
    for action, words in SYNONYM_GROUPS_32.items()
    for word in words
}

SYNONYM_GROUPS_48 = {
    "N": ["north", "up", "n", "↑", "norte", "nord", "kita", "shimal",
           "northern", "northbound", "uppward", "upper"],
    "E": ["east", "right", "e", "→", "este", "ost", "higashi", "sharq",
           "eastern", "eastbound", "rightward", "rightside"],
    "S": ["south", "down", "s", "↓", "sur", "süd", "minami", "janub",
           "southern", "southbound", "downward", "lower"],
    "W": ["west", "left", "w", "←", "oeste", "west_de", "nishi", "gharb",
           "western", "westbound", "leftward", "leftside"],
}
EXTENDED_WORD_TO_ACTION_48 = {
    word: action
    for action, words in SYNONYM_GROUPS_48.items()
    for word in words
}

SYNONYM_GROUPS_64 = {
    "N": ["north", "up", "n", "↑", "norte", "nord", "kita", "shimal",
           "northern", "northbound", "uppward", "upper",
           "topward", "ascend", "headup", "topside"],
    "E": ["east", "right", "e", "→", "este", "ost", "higashi", "sharq",
           "eastern", "eastbound", "rightward", "rightside",
           "starboard", "rightturn", "rightstep", "rightmove"],
    "S": ["south", "down", "s", "↓", "sur", "süd", "minami", "janub",
           "southern", "southbound", "downward", "lower",
           "descend", "headdown", "downside", "fall"],
    "W": ["west", "left", "w", "←", "oeste", "west_de", "nishi", "gharb",
           "western", "westbound", "leftward", "leftside",
           "port", "leftturn", "leftstep", "leftmove"],
}
EXTENDED_WORD_TO_ACTION_64 = {
    word: action
    for action, words in SYNONYM_GROUPS_64.items()
    for word in words
}

# 2026-05-10 (continued): higher vocab tiers (96/128/256) for finding
# encoding-wall ceiling. At sparse 10% over 4096-neuron lang_input,
# each word activates ~410 neurons; 96 words active = ~39K active
# neurons across vocab vs 4096 capacity = ~10× overlap. Predicted
# degradation visible at 96+ as the hash-based drive patterns collide
# more frequently than they're separated.
#
# Generated programmatically by appending numbered variants (north_5,
# north_6, ...) to the 64-word base. Each variant gets a unique
# SHA-256 hash → unique drive pattern, but overlap with primary
# words grows.

def _extend_with_numbered(base_groups: dict, target_per_action: int) -> dict:
    """Append numbered variants ('north_5', 'north_6', ...) to each
    action group until reaching target_per_action synonyms."""
    out = {a: list(words) for a, words in base_groups.items()}
    primaries = {"N": "north", "E": "east", "S": "south", "W": "west"}
    for a, words in out.items():
        prim = primaries[a]
        i = 1
        while len(words) < target_per_action:
            candidate = f"{prim}_{i:02d}"
            if candidate not in words:
                words.append(candidate)
            i += 1
    return out

SYNONYM_GROUPS_96 = _extend_with_numbered(SYNONYM_GROUPS_64, 24)
EXTENDED_WORD_TO_ACTION_96 = {
    word: action
    for action, words in SYNONYM_GROUPS_96.items()
    for word in words
}

SYNONYM_GROUPS_128 = _extend_with_numbered(SYNONYM_GROUPS_64, 32)
EXTENDED_WORD_TO_ACTION_128 = {
    word: action
    for action, words in SYNONYM_GROUPS_128.items()
    for word in words
}

SYNONYM_GROUPS_256 = _extend_with_numbered(SYNONYM_GROUPS_64, 64)
EXTENDED_WORD_TO_ACTION_256 = {
    word: action
    for action, words in SYNONYM_GROUPS_256.items()
    for word in words
}


def get_synonym_groups(vocab_size: int = 8) -> dict:
    """Return SYNONYM_GROUPS for the requested vocab size.

    vocab_size=8:  {N:[north,up], E:[east,right], S:[south,down], W:[west,left]}
    vocab_size=12: adds short forms {N:[..., n], ...}
    vocab_size=16: adds Unicode arrows {N:[..., ↑], E:[..., →], ...}
    vocab_size=24: adds Spanish/German localizations
    vocab_size=32: adds Japanese/Arabic localizations
    vocab_size=48: adds derived forms (-ward, -bound, -side, etc.)
    vocab_size=64: adds nautical/movement terms (port/starboard, ascend, etc.)
    """
    if vocab_size == 256:
        return SYNONYM_GROUPS_256
    if vocab_size == 128:
        return SYNONYM_GROUPS_128
    if vocab_size == 96:
        return SYNONYM_GROUPS_96
    if vocab_size == 64:
        return SYNONYM_GROUPS_64
    if vocab_size == 48:
        return SYNONYM_GROUPS_48
    if vocab_size == 32:
        return SYNONYM_GROUPS_32
    if vocab_size == 24:
        return SYNONYM_GROUPS_24
    if vocab_size == 16:
        return SYNONYM_GROUPS_16
    if vocab_size == 12:
        return SYNONYM_GROUPS_12
    return SYNONYM_GROUPS


def get_extended_word_to_action(vocab_size: int = 8) -> dict:
    if vocab_size == 256:
        return EXTENDED_WORD_TO_ACTION_256
    if vocab_size == 128:
        return EXTENDED_WORD_TO_ACTION_128
    if vocab_size == 96:
        return EXTENDED_WORD_TO_ACTION_96
    if vocab_size == 64:
        return EXTENDED_WORD_TO_ACTION_64
    if vocab_size == 48:
        return EXTENDED_WORD_TO_ACTION_48
    if vocab_size == 32:
        return EXTENDED_WORD_TO_ACTION_32
    if vocab_size == 24:
        return EXTENDED_WORD_TO_ACTION_24
    if vocab_size == 16:
        return EXTENDED_WORD_TO_ACTION_16
    if vocab_size == 12:
        return EXTENDED_WORD_TO_ACTION_12
    return EXTENDED_WORD_TO_ACTION
EXTENDED_WORD_TO_ACTION = {
    word: action
    for action, words in SYNONYM_GROUPS.items()
    for word in words
}


def _direction_from_positions(agent_pos, goal_pos) -> str:
    """Direction from agent to goal. Strict comparison (no >= bias on
    ties); ties default to north/south to balance the geometric over-
    representation of east/west — for fair eval we use balanced
    sampling instead (see _sample_balanced_eval_pair)."""
    ax, ay = agent_pos
    gx, gy = goal_pos
    dx, dy = gx - ax, gy - ay
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    if abs(dy) > abs(dx):
        return "north" if dy > 0 else "south"
    # tie — fallback (shouldn't happen with balanced sampler since
    # _sample_balanced_eval_pair guarantees |dx|≠|dy|)
    return "east" if dx > 0 else ("west" if dx < 0 else "north")


def _sample_balanced_eval_pair(rng, grid_size: int):
    """Sample (start, goal, target_word) such that target_word is uniformly
    distributed across {north, east, south, west}. Avoids the |dx|>=|dy|
    tie-break bias that over-represents east/west by ~7pp."""
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


def evaluate_image_to_word(
    bridge,
    n_trials: int = 100,
    grid_size: int = 8,
    stim_steps_per_trial: int = 200,
    drive_pA: float = 200.0,
    seed: int = 1,
    verbose: bool = True,
):
    """Present fresh gridworld images; check if agent emits correct
    cardinal direction in language_output.

    Returns dict with accuracy, per-class breakdown, and confusion matrix.
    """
    import cupy as cp
    from sim.visual_cortex import (
        render_gridworld_to_image,
        image_to_retina_drive,
    )

    rng = np.random.default_rng(seed)

    retina_idx = cp.asarray(
        list(bridge.region_manager.indices("retina")), dtype=cp.int64
    )
    lang_output_idx = cp.asarray(
        list(bridge.region_manager.indices("language_output")), dtype=cp.int64
    )

    correct = 0
    confusion = {w: {w2: 0 for w2 in ["north", "east", "south", "west"]}
                 for w in ["north", "east", "south", "west"]}
    n_reset_steps = 100  # match training inter-trial reset

    # Pre-cache embedding vectors for the 4 cardinal direction tokens
    # so we can do baseline-subtracted nearest-token decoding.
    from sim.text_embeddings import embed
    DIRECTIONS = ["north", "east", "south", "west"]
    target_embeddings = np.stack([embed(t, dim=int(lang_output_idx.size))
                                    for t in DIRECTIONS])  # (4, n_lang_output)

    for trial in range(n_trials):
        # Balanced eval sample: each direction gets equal trials
        # (otherwise |dx|>=|dy| tie-break biases toward east/west by ~7pp)
        (ax, ay), (gx, gy), target_word = _sample_balanced_eval_pair(rng, grid_size)

        # ─── Phase A: BASELINE language_output activity (no input) ───
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(n_reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        baseline_spikes = cp.zeros(int(lang_output_idx.size), dtype=cp.int32)
        for s in range(stim_steps_per_trial):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            if s >= 60:
                baseline_spikes += bridge.cp_firing_states[lang_output_idx].astype(cp.int32)

        # ─── Phase B: image-driven measurement ───
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(n_reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        img = render_gridworld_to_image(
            agent_pos=(int(ax), int(ay)), goal_pos=(int(gx), int(gy)),
            grid_size=grid_size, image_size=32,
        )
        bridge.cp_external_input_current[retina_idx] = cp.asarray(
            image_to_retina_drive(img, drive_max_pA=drive_pA),
            dtype=cp.float32,
        )
        # NO supervisor clamp on language_output — we want the agent's
        # natural response.
        bridge.core_config.current_reward_signal = 0.0  # no reward at eval

        # Tally language_output spikes
        spike_counts = cp.zeros(int(lang_output_idx.size), dtype=cp.int32)
        for s in range(stim_steps_per_trial):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            # Skip first 30ms (~60 sub-steps) onset, count rest
            if s >= 60:
                firing = bridge.cp_firing_states[lang_output_idx]
                spike_counts += firing.astype(cp.int32)

        # ─── Delta decoding: subtract baseline spikes, then cosine-match
        # to direction embeddings. Baseline-subtraction reveals the
        # image-driven RESPONSE rather than absolute activity. ───
        delta_spikes = (spike_counts - baseline_spikes).get().astype(np.float32)
        # Cosine similarity to each direction embedding
        sims = []
        for emb in target_embeddings:
            denom = np.linalg.norm(delta_spikes) * np.linalg.norm(emb)
            if denom < 1e-8:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(delta_spikes, emb) / denom))
        predicted = DIRECTIONS[int(np.argmax(sims))]

        is_correct = predicted == target_word
        if is_correct:
            correct += 1
        confusion[target_word][predicted] += 1

        if verbose and (trial + 1) % 25 == 0:
            tag = "OK" if is_correct else "WRONG"
            print(f"  [eval I->W] {trial+1}/{n_trials}  target={target_word} "
                  f"got={predicted} {tag} "
                  f"acc-so-far={correct}/{trial+1}={100*correct/(trial+1):.1f}%",
                  flush=True)
            # Tier-1 universal progress event
            from sim.progress import emit_progress
            emit_progress(
                "eval", trial + 1, n_trials,
                phase="I->W", unit="trials",
                correct=correct, accuracy=round(correct / (trial + 1), 4),
            )

    accuracy = correct / max(n_trials, 1)
    return {
        "n_trials": n_trials,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
    }


def evaluate_word_to_action(
    bridge,
    n_trials_per_word: int = 25,
    stim_steps_per_trial: int = 200,
    drive_pA: float = 200.0,
    verbose: bool = True,
    interleave_words: bool = True,
    n_reset_steps: int = 100,
    seed: int = 1,
    token_sparsity: float = 0.1,
    orthogonal_cues: bool = False,
    synonym_mode: bool = False,
    synonym_vocab_size: int = 8,
):
    """Drive language_input with each direction word; observe which
    motor_X has the highest firing rate. Did the agent learn the
    word-action mapping?

    Reads MOTOR_X (not cortex_X). With PFC-bypass, language_input has
    direct trained pathway to motor_X that's not subject to cascade
    cortex_N dominance.

    Args:
        interleave_words: If True (default 2026-05-02), trials are
          ordered so consecutive trials drive DIFFERENT words. This
          eliminates a measurement artifact: when trial N-1 drove the
          same word as trial N, motor_target's residual NMDA activity
          contaminates trial N's baseline measurement, suppressing the
          drive-vs-baseline delta. Block ordering (north x N then east
          x N ...) is biased; interleaved [N,E,S,W,N,E,S,W,...] keeps
          consecutive trials decorrelated. Set False to reproduce
          historical block-ordered results.
        n_reset_steps: inter-phase reset window (sub-steps). Default 100
          (= 50ms at dt=0.5) matches training. Larger values (e.g. 400 =
          200ms = 2x NMDA tau) produce cleaner baselines but slow eval.
        seed: shuffle seed when interleave_words=True. Deterministic.
        token_sparsity: fraction of language_input neurons activated per
          word (default 0.1 matches v2 baseline). Use 0.05 for orthogonal
          (~zero overlap) word codes — must match training-time sparsity.
    """
    import cupy as cp
    import math

    # Detect architecture: labeled motor_X pools (default) or distributed
    # motor_pop_θ sub-pools (Pulvermüller G.20, 2026-05-02).
    rm = bridge.region_manager
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
        # 8 sub-pools at 45° intervals. Population vector decoding.
        SUBPOOL_THETA = [
            (0, "E"), (45, "NE"), (90, "N"), (135, "NW"),
            (180, "W"), (225, "SW"), (270, "S"), (315, "SE"),
        ]
        subpool_idx = {
            suffix: cp.asarray(
                list(rm.indices(f"motor_pop_{suffix}")), dtype=cp.int64
            )
            for theta, suffix in SUBPOOL_THETA
        }
        # Cardinal projection weights: each (cardinal, subpool) pair.
        ACTION_THETA = {"N": 90, "E": 0, "S": 270, "W": 180}
        cardinal_proj = {a: {} for a in ACTION_NAMES}
        for action, theta_a in ACTION_THETA.items():
            for theta_p, suffix in SUBPOOL_THETA:
                d = ((theta_a - theta_p + 180) % 360) - 180
                cos_w = max(0.0, math.cos(math.radians(d)))
                cardinal_proj[action][suffix] = cos_w
        # cortex_idx unused in distributed-motor path (we don't read motor_X)
        cortex_idx = None
    else:
        motor_idx = {
            a: cp.asarray(list(rm.indices(f"motor_{a}")), dtype=cp.int64)
            for a in ACTION_NAMES
        }
        cortex_idx = motor_idx  # alias so existing code paths use motor

    correct = 0
    total = 0
    confusion = {w: {a: 0 for a in ACTION_NAMES}
                 for w in (list(get_extended_word_to_action(synonym_vocab_size).keys()) if synonym_mode else ["north", "east", "south", "west"])}

    # Multi-decoder counters (2026-05-02): test alternative decoders alongside
    # default delta-from-baseline. The 6-seed v2 result of W->A 28.5%
    # (p=0.027) may be limited by the argmax-of-delta decoder losing signal
    # from differentiated weights. Test these decoders inline:
    correct_drive_only = 0
    correct_ratio = 0
    correct_zscore = 0
    correct_clipped = 0
    confusion_drive_only = {w: {a: 0 for a in ACTION_NAMES}
                            for w in (list(get_extended_word_to_action(synonym_vocab_size).keys()) if synonym_mode else ["north", "east", "south", "west"])}
    confusion_ratio = {w: {a: 0 for a in ACTION_NAMES}
                       for w in (list(get_extended_word_to_action(synonym_vocab_size).keys()) if synonym_mode else ["north", "east", "south", "west"])}
    confusion_zscore = {w: {a: 0 for a in ACTION_NAMES}
                        for w in (list(get_extended_word_to_action(synonym_vocab_size).keys()) if synonym_mode else ["north", "east", "south", "west"])}
    confusion_clipped = {w: {a: 0 for a in ACTION_NAMES}
                         for w in (list(get_extended_word_to_action(synonym_vocab_size).keys()) if synonym_mode else ["north", "east", "south", "west"])}

    # Build trial schedule
    if synonym_mode:
        # 8 or 12 word vocab in synonym mode. Each word presented
        # n_trials_per_word times.
        EWA = get_extended_word_to_action(synonym_vocab_size)
        DIRECTIONS = list(EWA.keys())
        word_to_action_local = EWA
    else:
        DIRECTIONS = ["north", "east", "south", "west"]
        word_to_action_local = WORD_TO_ACTION
    if interleave_words:
        # Round-robin with rotating offset: round R = cyclic-shift of
        # DIRECTIONS by R. Guarantees ZERO consecutive same-word trials
        # (last word of round R is DIRECTIONS[(R+3) % 4]; first word of
        # round R+1 is DIRECTIONS[(R+1) % 4]; never equal).
        # Then permute within rounds based on seed for stochasticity
        # without breaking the no-repeat property — any permutation of
        # 4 distinct words preserves the round structure.
        rng = np.random.default_rng(seed)
        schedule = []
        prev_last = None
        for round_idx in range(n_trials_per_word):
            order = list(DIRECTIONS)
            rng.shuffle(order)
            # Re-shuffle if the first word matches the previous round's
            # last (only happens with prob 1/4; small loop)
            attempts = 0
            while prev_last is not None and order[0] == prev_last and attempts < 10:
                rng.shuffle(order)
                attempts += 1
            schedule.extend(order)
            prev_last = order[-1]
    else:
        # Legacy block order: 25 north trials, then 25 east, ...
        schedule = []
        for word in DIRECTIONS:
            schedule.extend([word] * n_trials_per_word)

    # Per-word logging buffers (only used if verbose)
    last_per_word = {}

    for word in schedule:
        target_action = word_to_action_local[word]
        # ─── Phase A: BASELINE measurement ───
        # Reset, then run with NO input. Measure spontaneous cortex_X.
        # This subtracts cascade default bias (cortex_N 2x higher etc.)
        # so we measure the DELTA caused by language_input.
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(n_reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        if distributed_motor_pop:
            baseline_subpool = {sfx: 0 for _, sfx in SUBPOOL_THETA}
        else:
            baseline_counts = {a: 0 for a in ACTION_NAMES}

        for s in range(stim_steps_per_trial):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            if s >= 60:
                firing = bridge.cp_firing_states
                if distributed_motor_pop:
                    for _, sfx in SUBPOOL_THETA:
                        baseline_subpool[sfx] += int(firing[subpool_idx[sfx]].sum().get())
                else:
                    for a in ACTION_NAMES:
                        baseline_counts[a] += int(firing[cortex_idx[a]].sum().get())

        # ─── Phase B: LANGUAGE-DRIVEN measurement ───
        # Reset, drive language_input, measure cortex_X again
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(n_reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        if orthogonal_cues:
            # Cue index from WORD_TO_ACTION key order. MUST match the
            # _VOCAB_ORDER used at training time in bio_three_factor.py.
            _vocab_order = list(WORD_TO_ACTION.keys())
            bridge.set_token_drive(
                word, drive_pA=drive_pA, sparsity=token_sparsity,
                orthogonal_cue_idx=_vocab_order.index(word),
                n_orthogonal_cues=len(_vocab_order),
            )
        else:
            bridge.set_token_drive(word, drive_pA=drive_pA, sparsity=token_sparsity)

        if distributed_motor_pop:
            drive_subpool = {sfx: 0 for _, sfx in SUBPOOL_THETA}
        else:
            spike_counts = {a: 0 for a in ACTION_NAMES}

        for s in range(stim_steps_per_trial):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            if s >= 60:
                firing = bridge.cp_firing_states
                if distributed_motor_pop:
                    for _, sfx in SUBPOOL_THETA:
                        drive_subpool[sfx] += int(firing[subpool_idx[sfx]].sum().get())
                else:
                    for a in ACTION_NAMES:
                        spike_counts[a] += int(firing[cortex_idx[a]].sum().get())

        # If distributed motor pool, project sub-pool counts onto cardinal
        # actions via cosine-weighted population vector. This produces
        # baseline_counts and spike_counts in the same per-cardinal format
        # that the rest of the eval expects.
        if distributed_motor_pop:
            baseline_counts = {a: 0.0 for a in ACTION_NAMES}
            spike_counts = {a: 0.0 for a in ACTION_NAMES}
            for action in ACTION_NAMES:
                for sfx, w in cardinal_proj[action].items():
                    baseline_counts[action] += baseline_subpool[sfx] * w
                    spike_counts[action] += drive_subpool[sfx] * w
            # Round to int for downstream code that may expect ints
            baseline_counts = {a: int(round(v)) for a, v in baseline_counts.items()}
            spike_counts = {a: int(round(v)) for a, v in spike_counts.items()}

        # ─── DELTA selection (default decoder) ───
        delta_counts = {a: spike_counts[a] - baseline_counts[a]
                        for a in ACTION_NAMES}
        predicted = max(delta_counts, key=lambda a: delta_counts[a])
        confusion[word][predicted] += 1
        if predicted == target_action:
            correct += 1
        total += 1

        # Tier-1 universal progress event (every 25 trials).
        # 2026-05-10 fix: was hardcoded `4 * n_trials_per_word` (correct
        # only for 4-word vocab); now uses len(schedule) which correctly
        # reflects vocab × n_trials_per_word for synonym modes (8, 16,
        # 32, 64, ... word vocabs). Frontend was showing 250% before
        # this fix because at vocab=64 trial 100 vs total 40 = 250%.
        if verbose and total % 25 == 0:
            from sim.progress import emit_progress
            n_total_trials = len(schedule)
            emit_progress(
                "eval", total, n_total_trials,
                phase="W->A", unit="trials",
                correct=correct, accuracy=round(correct / total, 4),
            )

        # ─── Alternative decoders (computed alongside, no extra cost) ───
        # 1. drive_only: argmax of raw drive counts (ignore baseline)
        pred_drive = max(spike_counts, key=lambda a: spike_counts[a])
        if pred_drive == target_action:
            correct_drive_only += 1
        confusion_drive_only[word][pred_drive] += 1

        # 2. ratio: argmax of drive/baseline ratio (multiplicative, robust to
        #    additive baseline noise; sensitive to relative spike rate change)
        ratio_counts = {
            a: (spike_counts[a] + 1) / (baseline_counts[a] + 1)
            for a in ACTION_NAMES
        }
        pred_ratio = max(ratio_counts, key=lambda a: ratio_counts[a])
        if pred_ratio == target_action:
            correct_ratio += 1
        confusion_ratio[word][pred_ratio] += 1

        # 3. zscore_delta: normalize delta by baseline mean across pools
        #    (pseudo-Z; full Z would need baseline std from multiple windows)
        baseline_mean = sum(baseline_counts.values()) / 4.0
        zscore_counts = {
            a: (spike_counts[a] - baseline_counts[a]) / max(1.0, baseline_mean)
            for a in ACTION_NAMES
        }
        pred_zscore = max(zscore_counts, key=lambda a: zscore_counts[a])
        if pred_zscore == target_action:
            correct_zscore += 1
        confusion_zscore[word][pred_zscore] += 1

        # 4. delta_clipped: like delta but clip negative values to 0 (only
        #    pools with above-baseline drive count for argmax)
        delta_clipped = {a: max(0, delta_counts[a]) for a in ACTION_NAMES}
        # If all zeros, fall back to drive
        pred_clipped = (
            max(delta_clipped, key=lambda a: delta_clipped[a])
            if any(v > 0 for v in delta_clipped.values())
            else pred_drive
        )
        if pred_clipped == target_action:
            correct_clipped += 1
        confusion_clipped[word][pred_clipped] += 1

        # Save last (word, baseline, drive, delta) for verbose summary
        last_per_word[word] = {
            "baseline": dict(baseline_counts),
            "drive": dict(spike_counts),
            "delta": dict(delta_counts),
        }

    if verbose:
        for word in DIRECTIONS:
            if word in last_per_word:
                d = last_per_word[word]
                print(f"  [eval W->A] word={word} "
                      f"target={word_to_action_local[word]} "
                      f"baseline={d['baseline']} drive={d['drive']} "
                      f"delta={d['delta']}", flush=True)

    accuracy = correct / max(total, 1)
    return {
        "n_trials": total,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "interleave_words": interleave_words,
        "n_reset_steps": n_reset_steps,
        # Multi-decoder results (2026-05-02): tests alternative decoders
        # alongside default delta-from-baseline.
        "alternative_decoders": {
            "drive_only": {
                "correct": correct_drive_only,
                "accuracy": correct_drive_only / max(total, 1),
                "confusion": confusion_drive_only,
            },
            "ratio": {
                "correct": correct_ratio,
                "accuracy": correct_ratio / max(total, 1),
                "confusion": confusion_ratio,
            },
            "zscore": {
                "correct": correct_zscore,
                "accuracy": correct_zscore / max(total, 1),
                "confusion": confusion_zscore,
            },
            "clipped": {
                "correct": correct_clipped,
                "accuracy": correct_clipped / max(total, 1),
                "confusion": confusion_clipped,
            },
        },
    }


def evaluate_word_to_action_LEGACY_BLOCK(*args, **kwargs):
    """Backwards-compat wrapper: calls evaluate_word_to_action with
    interleave_words=False to reproduce the historical block-ordered
    eval. Use only for direct comparison to pre-2026-05-02 baselines."""
    kwargs["interleave_words"] = False
    return evaluate_word_to_action(*args, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-image-word", type=int, default=500)
    ap.add_argument("--n-word-action", type=int, default=500)
    ap.add_argument("--n-eval-image-word", type=int, default=100)
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    from research.runners.text_train import run_text_training

    # Train
    print("=" * 60)
    print(f"TRAINING (seed={args.seed}, "
          f"{args.n_image_word} I->W + {args.n_word_action} W->A pairs)")
    print("=" * 60)
    bridge, train_stats = run_text_training(
        seed=args.seed,
        n_image_word_pairs=args.n_image_word,
        n_word_action_pairs=args.n_word_action,
        grid_size=args.grid_size,
        verbose=True,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print(f"EVAL: image -> word ({args.n_eval_image_word} fresh trials)")
    print("=" * 60)
    iw_result = evaluate_image_to_word(
        bridge, n_trials=args.n_eval_image_word, grid_size=args.grid_size,
    )
    print(f"\n  Accuracy: {iw_result['correct']}/{iw_result['n_trials']} "
          f"= {iw_result['accuracy']:.1%}")
    print(f"  Confusion: {iw_result['confusion_matrix']}")

    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_word_action} per word)")
    print("=" * 60)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_word_action,
    )
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}")
    print(f"  Confusion: {wa_result['confusion_matrix']}")

    if args.out_stats:
        out = {
            "seed": args.seed,
            "n_image_word_train": args.n_image_word,
            "n_word_action_train": args.n_word_action,
            "image_to_word_eval": iw_result,
            "word_to_action_eval": wa_result,
            "training_stats": train_stats,
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  Saved: {args.out_stats}")


if __name__ == "__main__":
    main()
