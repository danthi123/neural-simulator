"""Interactive chat REPL on biology-grounded foundation.

The first INTERACTIVE conversational artifact built on the validated
Phase 1.4 BRANCH A continual-learning + Tier 2.1 BREAKTHROUGH synonym
binding architectures. Supports two modes:

  --mode tier1           4-word vocab (north/east/south/west)
  --mode synonym         8-word vocab (synonyms: up/right/down/left also)

User types a direction word. Sim activates the corresponding motor pool
and responds with what it predicts. Quits on "quit", "exit", or EOF.

Per master plan section "For full Path F demo": "Accept Phase 1.4
BRANCH A as the primary continual-learning result + build conversational
demo on Phase 1.4 architecture using larger Tier 1/2.1 vocab."

This is the master plan's "build conversational demo on Phase 1.4
architecture" milestone — the interactive REPL that lets a user
actually talk to the sim.

Usage:
    # Tier 1 (4-word, ~6 min training):
    python -m research.runners.chat_repl --mode tier1 --seed 43 \\
        --train-events 200

    # Tier 2.1 synonym (8-word, ~20 min training):
    python -m research.runners.chat_repl --mode synonym --seed 42 \\
        --train-events 400

    # Then interactively:
    > north
    [TIER1 seed=43] sim hears 'north', activates motor_N (delta N+205, x2.1)
    > up
    [SYNONYM seed=42] sim hears 'up', activates motor_N (delta N+87, x1.7)
    > what
    [SYNONYM] 'what' is not in vocab; tracking deltas anyway:
              motor_N+12 motor_E+45 motor_S+8 motor_W-3
              best guess: motor_E (low confidence x1.4)
    > quit
    [DONE] 8 turns total.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Bridge Lineage Manager: persistent continuous-learning state.
# Per user 2026-05-10: "is there a good way to continually work off the
# most recently trained sim state and keep improving it rather than
# settling with from-scratch training sessions?"
from sim.lineage import BridgeLineage


def _load_or_train_tier1(seed: int, n_train_events: int, verbose: bool):
    """Train Tier 1 architecture (4-word vocab) and return bridge."""
    from research.runners.bio_three_factor import run_three_factor

    if verbose:
        print(f"[TRAINING] Tier 1 architecture (seed={seed}, "
              f"n_events={n_train_events})", flush=True)
        t0 = time.time()

    bridge, _ = run_three_factor(
        seed=seed,
        n_events_per_direction=n_train_events,
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

    if verbose:
        print(f"[TRAINING] complete ({time.time() - t0:.0f}s)", flush=True)

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    return bridge


def _load_or_train_tier1_hippo(seed: int, n_train_events: int, verbose: bool):
    """Build hippocampus-enabled Tier 1 architecture and return bridge.

    Mirrors consolidation_trainer.run_consolidation_training's build path
    (Tier 1 motor pools + hippocampus regions ec/dg/dg_pv_basket/ca3/ca1)
    but supports n_train_events=0 for pure architecture build (used by
    BridgeMemory + investigate_invivo_binding_fix loading main_hippo).

    Catalog G.13 + Buzsáki 2015 + McClelland 1995 CLS theory.
    """
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    from sim.bridge import SimulationBridge
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )

    if verbose:
        print(f"[BUILD] Tier 1 + hippocampus architecture (seed={seed})",
              flush=True)
        t0 = time.time()

    regions, pathways = build_biological_brain_regions(
        n_lang_input=2048,
        n_motor_per_action=500,
        enable_motor_fs=True,
        n_motor_fs_per_action=60,
        enable_language_output=True,
        n_lang_output=2048,
        motor_to_language_output_weight=2.0,
        enable_hippocampus_consolidation=True,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 5.0
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

    if n_train_events > 0:
        # Optional: caller can request fresh training via this helper too
        from research.runners.consolidation_trainer import (
            run_consolidation_training,
        )
        bridge, _ = run_consolidation_training(
            seed=seed,
            n_awake_events_per_word=n_train_events,
            verbose=verbose,
        )

    if verbose:
        print(f"[BUILD] complete ({time.time() - t0:.0f}s)",
              flush=True)

    # Default: language plasticity gates OFF (consistent with tier1/synonym)
    for gate in ("language_input_to_motor",
                  "motor_to_language_output"):
        try:
            bridge.set_plasticity_gate(gate, 0.0)
        except Exception:
            pass

    return bridge


def _load_or_train_synonym(seed: int, n_train_events: int, verbose: bool,
                             vocab_size: int = 8,
                             n_motor_per_action: int = 1000,
                             n_motor_fs_per_action: int = 120):
    """Train Tier 2.1 scale-up synonym architecture.

    vocab_size=8: validated 3/3 GO (n_motor=1000)
    vocab_size=12: PARTIAL at default n_motor=1000, GO at n_motor=2000
                   (capacity hypothesis; per 2026-05-08 finding)
    vocab_size=16: tested only at n_motor=2000 (master plan extension)
    """
    from research.runners.bio_three_factor import run_three_factor

    if verbose:
        print(f"[TRAINING] Tier 2.1 scale-up architecture "
              f"(seed={seed}, n_events={n_train_events}, "
              f"vocab={vocab_size}, n_motor={n_motor_per_action})",
              flush=True)
        t0 = time.time()

    bridge, _ = run_three_factor(
        seed=seed,
        n_events_per_direction=n_train_events,
        n_lang_input=4096,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=True,
        synonym_vocab_size=vocab_size,
        verbose=False,
    )

    if verbose:
        print(f"[TRAINING] complete ({time.time() - t0:.0f}s)", flush=True)

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    return bridge


def chat_inference(
    bridge,
    user_word: str,
    stim_steps: int = 100,
    reset_steps: int = 50,
    drive_pA: float = 200.0,
    sparsity: float = 0.1,
):
    """Run one chat turn with baseline-vs-driven delta methodology.

    Returns dict with delta_counts, predicted_action, predicted_direction,
    confidence_ratio.
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

    # Phase A: baseline (no input)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    baseline = cp.zeros(4, dtype=cp.int32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for a_i, a in enumerate(["N", "E", "S", "W"]):
            baseline[a_i] += fired[motor_arr[a]].sum()

    # Phase B: driven (word input)
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
    bl = _bl_to_host(baseline)
    dr = _bl_to_host(drive_counts)
    delta = dr - bl
    predicted_idx = int(np.argmax(delta))
    predicted_action = ["N", "E", "S", "W"][predicted_idx]
    action_to_word = {"N": "north", "E": "east", "S": "south", "W": "west"}
    predicted_direction = action_to_word[predicted_action]

    sorted_delta = np.sort(delta)[::-1]
    if sorted_delta[1] > 0:
        confidence = float(sorted_delta[0] / sorted_delta[1])
    elif sorted_delta[0] > 0:
        confidence = float("inf")
    else:
        confidence = 1.0

    return {
        "user_word": user_word,
        "delta_counts": {a: int(delta[i])
                          for i, a in enumerate(["N", "E", "S", "W"])},
        "predicted_action": predicted_action,
        "predicted_direction": predicted_direction,
        "confidence_ratio": confidence,
    }


# ─── Dialog state (Track 3 layer 3, 2026-05-09) ───────────────────────

# Inverse-action lookup. Used by `:opposite` to flip the last predicted
# action; biologically tests whether the network has learned
# anti-correlation between opposing motor pools (motor_N vs motor_S etc.).
ACTION_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

# Canonical primary direction word per action — used to echo the inverted
# action back to the user as a word ("opposite of north was south").
ACTION_TO_PRIMARY_WORD = {"N": "north", "E": "east", "S": "south", "W": "west"}

# Recognized dialog verbs. Prefix `:` to disambiguate from vocab words.
# Includes :speak (Track 3 layer 4 generative decoder, A→W direction).
DIALOG_VERBS = {"again", "opposite", "history", "forget", "speak"}

# Reusable action-alias map for both :learn and :speak commands. Maps
# any of (N/E/S/W, full direction names, synonyms, Unicode arrows) to
# the canonical action letter.
ACTION_ALIASES = {
    "n": "N", "north": "N", "up": "N", "↑": "N",
    "e": "E", "east":  "E", "right": "E", "→": "E",
    "s": "S", "south": "S", "down": "S", "↓": "S",
    "w": "W", "west":  "W", "left": "W", "←": "W",
}


def _parse_speak_command(line: str):
    """Parse `:speak <action>` — Track 3 layer 4 generative decoder.

    Returns the canonical action letter ("N"/"E"/"S"/"W") on success,
    None otherwise. Action accepts the same aliases as :learn:
    direction letters, full direction names, synonyms, Unicode arrows.

    The :speak command drives motor_<action> and reads language_output,
    decoding the resulting spike pattern to a word via cosine similarity
    against known vocab drive patterns. Tests the A→W (action→word)
    direction validated at Tier 2.1 BREAKTHROUGH (mean A→W 63.7%).
    """
    s = line.strip()
    if not s.startswith(":"):
        return None
    body = s[1:].strip()
    parts = body.split()
    if len(parts) < 2:
        return None
    if parts[0].lower() != "speak":
        return None
    action_raw = parts[1].strip().lower()
    return ACTION_ALIASES.get(action_raw)


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two 1D numpy arrays.

    Returns 0.0 if either vector has zero norm (avoids div by zero).
    Used by the generative decoder to rank vocab words by how well
    each word's drive pattern matches the network's language_output
    spike pattern.
    """
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sample_with_temperature(rankings, temperature: float = 0.0,
                                rng_seed: int = None):
    """Sample one word from a sorted (word, similarity) list using softmax
    with temperature τ.

    Args:
        rankings: list of (word, similarity) tuples, sorted descending by sim.
        temperature: τ. 0 → return None (caller falls back to argmax).
            >0 → softmax sampling. Lower τ = sharper, more deterministic.
            Recommended τ ∈ [0.01, 0.05] for "natural feeling" synonym
            preference vs strict argmax. τ → ∞ approaches uniform random.
        rng_seed: optional seed for reproducible sampling.

    Returns:
        Sampled word (str) or None if temperature ≤ 0 / rankings empty.

    Math: probs = softmax(sims / τ). Numerically stable shift via max-subtract.
    """
    if temperature <= 0.0 or not rankings:
        return None
    words = [w for w, _ in rankings]
    sims = np.array([s for _, s in rankings], dtype=np.float64)
    # Subtract max for numerical stability (softmax is shift-invariant)
    scaled = (sims - sims.max()) / max(temperature, 1e-9)
    weights = np.exp(scaled)
    probs = weights / weights.sum()
    sampler = (
        np.random.default_rng(rng_seed) if rng_seed is not None
        else np.random.default_rng()
    )
    sampled_idx = int(sampler.choice(len(words), p=probs))
    return words[sampled_idx]


def _rank_words_by_similarity(spike_pattern, word_patterns: dict):
    """Rank vocab words by cosine similarity to the spike pattern.

    Args:
        spike_pattern: 1D numpy-like array — the network's
            language_output activity for the current motor drive.
        word_patterns: dict mapping word -> 1D drive pattern (same
            length as spike_pattern). Drive patterns produced by
            sim.text_embeddings.vocab_to_drive_pattern().

    Returns:
        list of (word, similarity) tuples, sorted descending by
        similarity. Top-1 = "spoken" word; full list = ranking.
    """
    rankings = [(w, _cosine_similarity(spike_pattern, p))
                for w, p in word_patterns.items()]
    rankings.sort(key=lambda x: -x[1])
    return rankings


def generative_inference(bridge, target_action: str,
                         vocab_words=None,
                         stim_steps: int = 100,
                         reset_steps: int = 50,
                         motor_drive_pA: float = 1500.0,
                         drive_max_pA: float = 200.0,
                         sparsity: float = 0.1,
                         top_k: int = 4,
                         temperature: float = 0.0,
                         rng_seed: int = None):
    """Generative decoder: action → word (A→W direction).

    Drives motor_<target_action> with elevated current, reads the
    resulting language_output activity (delta vs baseline), and decodes
    to a word via cosine similarity against known vocab drive patterns.

    Inverse of chat_inference. The W→A path validates "given a word,
    pick the right motor"; this A→W path validates "given an action,
    produce the right word". Both are biologically grounded — they
    travel in opposite directions through the same plastic synapses
    that embodied-Hebbian co-firing strengthened during training.

    Args:
        target_action: action letter (N/E/S/W) to drive motor pool
        vocab_words: list of words to score (default 4-word vocab)
        stim_steps: simulation steps to drive motor for
        reset_steps: simulation steps to reset between baseline + drive
        motor_drive_pA: current strength to motor pool
        drive_max_pA: drive strength for vocab pattern lookup
        sparsity: fraction of language_output neurons active per word
        top_k: how many ranked words to return
        temperature: 0.0 (default) → strict argmax, deterministic. >0 →
            softmax sampling over similarities; higher = more random.
            Recommended range: 0.01-0.05 for "natural-feeling" synonym
            preference vs strict primary win. tau=0.02 typically lifts
            secondary-synonym top-1 from 0% to ~15-30% while keeping
            primary as the dominant choice. SET TO 0 FOR REPRODUCIBLE
            TESTING. Per perf-audit 2026-05-10 + STDP-WTA-pattern
            observed in Tier 2.1 BREAKTHROUGH paper.
        rng_seed: optional seed for sampling reproducibility when
            temperature > 0. None → uses numpy default (non-reproducible).

    Returns:
        dict with:
          target_action: input action letter
          predicted_word: top-1 ranked word (or sampled when temperature > 0)
          confidence: top-1 similarity / runner-up similarity
          rankings: list of (word, similarity) sorted desc
          delta: 1D numpy array of language_output spike deltas
    """
    # Backend-aware: cp is the active backend (cupy on CuPy, numpy on NumPy)
    from sim.backend import get_backend
    cp, _ = get_backend()
    import numpy as np
    from sim.text_embeddings import vocab_to_drive_pattern

    if target_action not in ("N", "E", "S", "W"):
        raise ValueError(f"target_action must be N/E/S/W, got {target_action!r}")

    rm = bridge.region_manager
    motor_idx = list(rm.indices(f"motor_{target_action}"))
    motor_arr = cp.asarray(motor_idx, dtype=cp.int64)
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)
    n_lang_out = len(lang_out_idx)

    # Phase A: baseline (no input)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    baseline = cp.zeros(n_lang_out, dtype=cp.int32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        baseline += bridge.cp_firing_states[lang_out_arr].astype(cp.int32)

    # Phase B: drive motor_<action>, read language_output
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[motor_arr] = float(motor_drive_pA)
    drive_counts = cp.zeros(n_lang_out, dtype=cp.int32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        drive_counts += bridge.cp_firing_states[lang_out_arr].astype(cp.int32)

    # Backend-aware D->H transfer (passthrough on NumPy)
    from sim.backend import to_host as _delta_to_host
    delta = _delta_to_host(drive_counts - baseline).astype(np.float32)

    # Decode by cosine similarity to known vocab patterns
    if vocab_words is None:
        vocab_words = ["north", "east", "south", "west"]
    word_patterns = {
        w: vocab_to_drive_pattern(
            w, n_neurons=n_lang_out,
            drive_max_pA=drive_max_pA, sparsity=sparsity,
        )
        for w in vocab_words
    }
    rankings = _rank_words_by_similarity(delta, word_patterns)
    # Keep full sorted list for both top_k truncation and temperature sampling
    full_rankings = list(rankings)
    rankings = rankings[:top_k]

    # Predicted word: argmax (temperature=0) or sampled (temperature > 0).
    sampled_word = _sample_with_temperature(
        full_rankings, temperature=temperature, rng_seed=rng_seed,
    )
    predicted = sampled_word if sampled_word is not None else (
        rankings[0][0] if rankings else None
    )

    # Confidence: top-1 / top-2 ratio (Inf if top-2 is non-positive)
    if len(rankings) >= 2 and rankings[1][1] > 0:
        confidence = rankings[0][1] / rankings[1][1]
    elif rankings and rankings[0][1] > 0:
        confidence = float("inf")
    else:
        confidence = 1.0

    return {
        "target_action": target_action,
        "predicted_word": predicted,
        "argmax_word": rankings[0][0] if rankings else None,  # always argmax
        "sampled_word": sampled_word,  # None if temperature == 0
        "temperature": float(temperature),
        "confidence": confidence,
        "rankings": rankings,
        "delta": delta,
    }


def _parse_dialog_command(line: str):
    """Parse a `:verb [args]` dialog command.

    Returns a dict like {"verb": "again"} on success, None otherwise.
    `:history` may include an integer count (clamped to [1, 50]; default 5).

    Uses the `:` prefix to clearly separate dialog state commands from
    vocab inputs and the existing unprefixed `learn` command.
    """
    s = line.strip()
    if not s.startswith(":"):
        return None
    body = s[1:].strip()
    if not body:
        return None
    parts = body.split()
    verb = parts[0].lower()
    if verb not in DIALOG_VERBS:
        return None
    out = {"verb": verb}
    if verb == "history":
        # Optional count argument
        n = 5  # default
        if len(parts) >= 2:
            try:
                n = int(parts[1])
            except ValueError:
                n = 5  # fallback on junk
        out["n"] = max(1, min(50, n))
    return out


# ─── Online vocab learning (Track 3 scaffolding, 2026-05-09) ─────────


def _parse_learn_command(line: str):
    """Parse a 'learn <word> <action>' REPL command.

    Returns (word, action) on success, None on parse failure. Strips
    whitespace and lowercases word; uppercases action; validates action
    is one of N/E/S/W.

    Examples:
        'learn ahead N'         -> ('ahead', 'N')
        'learn ahead north'     -> ('ahead', 'N')   # word form ok
        'learn forward up'      -> ('forward', 'N') # synonyms accepted
        'learn  HELLO  e '      -> ('hello', 'E')   # trim + case
        'learn'                 -> None             # missing args
        'learn ahead'           -> None             # missing action
        'learn ahead nope'      -> None             # bad action
    """
    parts = line.strip().split()
    if len(parts) < 3 or parts[0].lower() != "learn":
        return None
    word = parts[1].strip().lower()
    action_raw = parts[2].strip().lower()
    # Accept N/E/S/W directly OR full direction names OR synonym words
    action_aliases = {
        "n": "N", "north": "N", "up": "N", "↑": "N",
        "e": "E", "east":  "E", "right": "E", "→": "E",
        "s": "S", "south": "S", "down": "S", "↓": "S",
        "w": "W", "west":  "W", "left": "W", "←": "W",
    }
    action = action_aliases.get(action_raw)
    if action is None:
        return None
    if not word:
        return None
    return (word, action)


def learn_word_pairing(bridge, word: str, target_action: str,
                       n_events: int = 50, stim_steps_per_event: int = 100,
                       reset_steps: int = 50, drive_pA: float = 200.0,
                       teacher_pA: float = 1500.0, sparsity: float = 0.1,
                       verbose: bool = True):
    """Online embodied-Hebbian binding of a NEW word to an existing motor pool.

    Runs ``n_events`` paired co-firing events on the already-trained bridge:
      - Drive language_input with the new word's drive pattern
      - Drive language_output with the same pattern (output teacher)
      - Drive motor_<target_action> with elevated current (action teacher)
      - Step the bridge so STDP fires on co-active synapses

    The bridge's plastic ``language_input_to_motor`` and (if present)
    ``motor_to_language_output`` gates are temporarily opened, then
    re-frozen on exit. This lets the existing population codes reach
    new bindings without disturbing inference-time stability.

    Args:
        bridge: trained SimulationBridge (post chat_repl init)
        word: new vocabulary word to bind
        target_action: one of "N", "E", "S", "W"
        n_events: number of paired events (50 is a moderate dose;
            empirically gives a ~detectable binding without dramatically
            shifting existing bindings on the same motor pool)
        stim_steps_per_event: forward-prop steps per event
        reset_steps: free-running steps between events to clear
            transient state
        drive_pA: peak drive on language input + output sites
        teacher_pA: motor-pool teacher current (must be high enough to
            drive motor_X spikes regardless of upstream)
        sparsity: fraction of language_input neurons activated by the
            word's drive pattern
        verbose: log progress every 10 events

    Returns:
        dict with summary stats (n_events_run, target_action, gates_opened)
    """
    # Backend-aware: cp is the active backend (cupy on CuPy, numpy on NumPy)
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import vocab_to_drive_pattern

    if target_action not in ("N", "E", "S", "W"):
        raise ValueError(f"target_action must be N/E/S/W, got {target_action!r}")

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = list(rm.indices(f"motor_{target_action}"))
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    motor_arr = cp.asarray(motor_idx, dtype=cp.int64)
    n_lang_in = len(lang_input_idx)

    # language_output is optional — only present if bridge was trained with
    # embodied_hebbian=True (which chat_repl always does, but defensive).
    try:
        lang_output_idx = list(rm.indices("language_output"))
        lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)
        n_lang_out = len(lang_output_idx)
        has_output = True
    except Exception:
        has_output = False
        n_lang_out = 0

    # Drive pattern for the new word — same scheme as inference path.
    drive_in = vocab_to_drive_pattern(
        word, n_neurons=n_lang_in,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_in_gpu = cp.asarray(drive_in, dtype=cp.float32)
    if has_output:
        drive_out = vocab_to_drive_pattern(
            word, n_neurons=n_lang_out,
            drive_max_pA=drive_pA, sparsity=sparsity,
        )
        drive_out_gpu = cp.asarray(drive_out, dtype=cp.float32)

    # Open plasticity gates for the duration of learning.
    gates_opened = []
    for gate_name in ("language_input_to_motor", "motor_to_language_output"):
        try:
            bridge.set_plasticity_gate(gate_name, 1.0)
            gates_opened.append(gate_name)
        except Exception:
            pass

    if verbose:
        print(f"[LEARN] '{word}' -> motor_{target_action} | "
              f"{n_events} events | gates open: {gates_opened}",
              flush=True)
        t0 = time.time()

    try:
        for ev in range(n_events):
            # Reset between events: zero drive, free-run to clear transients
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            # Drive language_input + language_output + motor_TARGET
            bridge.cp_external_input_current[lang_input_arr] = drive_in_gpu
            if has_output:
                bridge.cp_external_input_current[lang_output_arr] = drive_out_gpu
            bridge.cp_external_input_current[motor_arr] += float(teacher_pA)

            # Forward-prop — STDP fires on plastic synapses with co-active pre+post
            for _ in range(stim_steps_per_event):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            if verbose and (ev + 1) % 10 == 0:
                print(f"  [LEARN] {ev + 1}/{n_events} events", flush=True)
    finally:
        # Re-freeze gates regardless of exception
        for gate_name in gates_opened:
            try:
                bridge.set_plasticity_gate(gate_name, 0.0)
            except Exception:
                pass

    if verbose:
        print(f"[LEARN] complete ({time.time() - t0:.0f}s)", flush=True)

    return {
        "word": word,
        "target_action": target_action,
        "n_events_run": n_events,
        "gates_opened": gates_opened,
    }


# ─── REPL ─────────────────────────────────────────────────────────────

VOCAB_TIER1 = {"north", "east", "south", "west"}
VOCAB_SYNONYM = {"north", "east", "south", "west",
                  "up", "right", "down", "left"}
VOCAB_SYNONYM_12 = VOCAB_SYNONYM | {"n", "e", "s", "w"}
VOCAB_SYNONYM_16 = VOCAB_SYNONYM_12 | {"↑", "→", "↓", "←"}
# 2026-05-12: synonym24/32 extend with multi-language synonyms
# (Spanish, German, Japanese, Arabic). Sourced from text_eval's
# SYNONYM_GROUPS_24/32. Tests vocab-tier scaling on 16-word baseline.
VOCAB_SYNONYM_24 = VOCAB_SYNONYM_16 | {"norte", "nord", "este", "ost",
                                         "sur", "süd", "oeste", "west_de"}
VOCAB_SYNONYM_32 = VOCAB_SYNONYM_24 | {"kita", "shimal", "higashi", "sharq",
                                         "minami", "janub", "nishi", "gharb"}

WORD_TO_ACTION_SYNONYM = {
    "north": "N", "up": "N",
    "east": "E", "right": "E",
    "south": "S", "down": "S",
    "west": "W", "left": "W",
}
WORD_TO_ACTION_SYNONYM_12 = {**WORD_TO_ACTION_SYNONYM,
    "n": "N", "e": "E", "s": "S", "w": "W"}
WORD_TO_ACTION_SYNONYM_16 = {**WORD_TO_ACTION_SYNONYM_12,
    "↑": "N", "→": "E", "↓": "S", "←": "W"}
WORD_TO_ACTION_SYNONYM_24 = {**WORD_TO_ACTION_SYNONYM_16,
    "norte": "N", "nord": "N",
    "este": "E", "ost": "E",
    "sur": "S", "süd": "S",
    "oeste": "W", "west_de": "W"}
WORD_TO_ACTION_SYNONYM_32 = {**WORD_TO_ACTION_SYNONYM_24,
    "kita": "N", "shimal": "N",
    "higashi": "E", "sharq": "E",
    "minami": "S", "janub": "S",
    "nishi": "W", "gharb": "W"}


def _vocab_for_mode(mode: str):
    """Return (vocab_set, word_to_action_dict) for a chat_repl mode."""
    if mode == "tier1":
        return VOCAB_TIER1, {"north": "N", "east": "E",
                              "south": "S", "west": "W"}
    if mode == "synonym":
        return VOCAB_SYNONYM, WORD_TO_ACTION_SYNONYM
    if mode == "synonym12":
        return VOCAB_SYNONYM_12, WORD_TO_ACTION_SYNONYM_12
    if mode == "synonym16":
        return VOCAB_SYNONYM_16, WORD_TO_ACTION_SYNONYM_16
    if mode == "synonym24":
        return VOCAB_SYNONYM_24, WORD_TO_ACTION_SYNONYM_24
    if mode == "synonym32":
        return VOCAB_SYNONYM_32, WORD_TO_ACTION_SYNONYM_32
    raise ValueError(f"unknown mode: {mode}")


def _load_bridge_from_checkpoint(checkpoint_path: str, mode: str, seed: int,
                                   verbose: bool = True):
    """Load a previously-trained bridge from an HDF5 checkpoint.

    Reuses the standard build/init path then loads weights from disk.
    Per CLAUDE.md gotcha: save_checkpoint doesn't preserve firing
    thresholds, STP, eligibility -- but for inference (REPL chat),
    weights are what matter; dynamic state self-recovers in a few
    timesteps of free-running.
    """
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge

    if verbose:
        print(f"[LOAD] Reading bridge state from {checkpoint_path}",
              flush=True)
        t0 = time.time()

    # Re-build bridge with the same config (mode determines arch)
    if mode == "tier1":
        bridge = _load_or_train_tier1(seed, n_train_events=0, verbose=False)
    elif mode == "synonym":
        bridge = _load_or_train_synonym(seed, n_train_events=0, verbose=False)
    elif mode == "synonym12":
        # Per 2026-05-08 capacity finding: synonym12 needs n_motor=2000
        bridge = _load_or_train_synonym(seed, n_train_events=0, verbose=False,
                                          vocab_size=12,
                                          n_motor_per_action=2000,
                                          n_motor_fs_per_action=240)
    elif mode == "synonym16":
        # Master plan extension: synonym16 also uses n_motor=2000
        bridge = _load_or_train_synonym(seed, n_train_events=0, verbose=False,
                                          vocab_size=16,
                                          n_motor_per_action=2000,
                                          n_motor_fs_per_action=240)
    elif mode == "synonym24":
        # 24-word vocab. Per capacity rule (vocab_size N needs
        # ~(N/4)*333 motor neurons), 24-word needs ~2000 motor.
        bridge = _load_or_train_synonym(seed, n_train_events=0, verbose=False,
                                          vocab_size=24,
                                          n_motor_per_action=2000,
                                          n_motor_fs_per_action=240)
    elif mode == "synonym32":
        # 32-word vocab. Per capacity rule, needs ~2670 motor; use 3000.
        bridge = _load_or_train_synonym(seed, n_train_events=0, verbose=False,
                                          vocab_size=32,
                                          n_motor_per_action=3000,
                                          n_motor_fs_per_action=360)
    elif mode == "tier1_hippo":
        # Build Tier 1 architecture WITH hippocampus consolidation
        # (for lineages bootstrapped via bootstrap_hippo_lineage). No
        # training — just construct the architecture so load_checkpoint
        # can overlay the saved weights. Fixes the BridgeMemory load
        # path for hippo lineages (chat_repl previously only knew
        # "tier1" and "synonym" modes, losing the ca3/dg/ec/ca1 regions
        # during checkpoint reload).
        bridge = _load_or_train_tier1_hippo(seed, n_train_events=0,
                                              verbose=False)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Now overlay weights from the checkpoint.
    bridge.load_checkpoint(checkpoint_path)

    if verbose:
        print(f"[LOAD] complete ({time.time() - t0:.0f}s)", flush=True)

    # Re-freeze plasticity gates (load_checkpoint may have reset them)
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    return bridge


def _save_bridge_checkpoint(bridge, checkpoint_path: str, verbose: bool = True,
                              metadata: dict = None):
    """Save the trained bridge state for fast reload in future sessions.

    Writes both:
    - <checkpoint_path>: HDF5 bridge state (weights, region indices, etc.)
    - <checkpoint_path>.meta.json: sidecar metadata so the webapp's
      bridges library can list metadata without loading the HDF5.

    Sidecar fields: mode, seed, n_train_events, n_neurons, n_synapses,
    saved_at (ISO 8601), tags (optional list). Caller passes `metadata`
    to populate these. The HDF5 file is the source of truth for the
    actual state; sidecar is just for browsing.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    if verbose:
        print(f"[SAVE] Writing bridge state to {checkpoint_path}",
              flush=True)
        t0 = time.time()

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    bridge.save_checkpoint(checkpoint_path)

    # Write sidecar metadata so the webapp can browse without h5py.
    sidecar = {
        "saved_at": datetime.now().isoformat(),
        "checkpoint": Path(checkpoint_path).name,
        "n_neurons": int(getattr(bridge.core_sim_config, "num_neurons", 0)),
        "n_synapses": int(getattr(
            bridge, "actual_total_connections_n", 0
        )) if hasattr(bridge, "actual_total_connections_n") else None,
    }
    if metadata:
        sidecar.update(metadata)
    sidecar_path = Path(str(checkpoint_path) + ".meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2),
                              encoding="utf-8")

    if verbose:
        print(f"[SAVE] complete ({time.time() - t0:.0f}s) "
              f"+ sidecar {sidecar_path.name}",
              flush=True)


def _lineage_save(lineage, bridge, *, mode: str, seed: int,
                   n_train_events: int,
                   kind: str = "save",
                   description: str = "",
                   accuracy_metric: str = None,
                   accuracy_value: float = None,
                   accuracy_context: str = ""):
    """Save bridge to lineage with a tier label, arch dict, and growth event.

    Centralises the metadata-update logic so chat_repl callsites all
    record the same fields (mode, seed, training events, arch shape, etc).
    """
    # Build arch dict from bridge config. n_lang_input + n_motor_per_action
    # are the dominant scaling axes for retention/capacity.
    try:
        cfg = bridge.core_sim_config
        arch = {
            "mode": mode,
            "n_neurons": int(getattr(cfg, "num_neurons", 0)),
            "n_synapses": int(getattr(bridge, "actual_total_connections_n", 0))
                if hasattr(bridge, "actual_total_connections_n") else None,
        }
    except Exception:
        arch = {"mode": mode}

    # Pick a tier label from mode (4-word / 8-word / 12-word / 16-word).
    tier_label_map = {
        "tier1": "4-word",
        "synonym": "8-word",
        "synonym12": "12-word",
        "synonym16": "16-word",
    }
    tier = tier_label_map.get(mode, mode)

    # Persist
    lineage.save(bridge, tier=tier, arch=arch)
    # Append growth event + accuracy datapoint to metadata
    meta = lineage.read_metadata()
    meta.cumulative_training_events += int(n_train_events or 0)
    meta.add_growth_event(
        kind=kind,
        description=description,
        seed=seed,
        n_train_events=n_train_events,
    )
    if accuracy_metric is not None and accuracy_value is not None:
        meta.add_accuracy(
            metric=accuracy_metric,
            value=float(accuracy_value),
            context=accuracy_context,
        )
    lineage.write_metadata(meta)


def run_repl(mode: str, seed: int, n_train_events: int,
             transcript_out: str = None,
             load_bridge: str = None,
             save_bridge: str = None,
             scripted_words: list = None,
             allow_learn: bool = False,
             learn_n_events: int = 50,
             speak_temperature: float = 0.0,
             lineage_name: str = "main",
             from_scratch: bool = False,
             fork_lineage: str = None):
    """Train + interactive REPL loop.

    If load_bridge is given, skip training and load from checkpoint.
    If save_bridge is given (and we DID train), save the trained bridge
    for future use. Combined: training takes ~6-20 min depending on
    mode; checkpoint reload takes ~10-30 sec, making subsequent
    interactive sessions effectively instant.

    If scripted_words is given (a list of words), run those instead of
    interactive stdin -- useful for CI / regression tests / batch
    eval. Exits after processing the list.

    If ``allow_learn`` is True (per --learn), the REPL recognizes
    ``learn <word> <action>`` commands which run an online embodied-
    Hebbian binding session of ``learn_n_events`` paired events, then
    automatically test the new binding. Default OFF — learning during
    the REPL is opt-in because it can perturb existing bindings.

    Bridge Lineage:
    - lineage_name: persistent state to load/save (default 'main').
      Located under bridges/lineage/<name>/.
    - from_scratch: if True, skip lineage entirely (science mode).
    - fork_lineage: if given, after loading 'lineage_name', fork to a
      new lineage of that name and save back to the fork (not the
      original). Useful for branching experiments without disturbing
      'main'.
    """
    print("=" * 60)
    print(f"BIOLOGY-GROUNDED CHAT REPL — mode={mode}, seed={seed}")
    print(f"Type a direction word; sim activates the motor pool.")
    if allow_learn:
        print(f"Online learning: ON. Type 'learn <word> <action>' to bind a new word.")
    print(f"Quit with 'quit', 'exit', or Ctrl-D.")
    print("=" * 60, flush=True)

    # FIX 2026-07-31: bind word_to_action HERE, beside vocab, not at :1387. It was assigned only on the
    # plain-word path but READ on the :speak path (:1299). Python makes it function-local for the whole
    # of run_repl, so `:speak` before any plain word raised UnboundLocalError -- and because the read sits
    # on the right of an `or`, it evaluated ONLY when pred_word != expected_word. The crash therefore
    # fired exactly when the model was WRONG: a :speak session either printed [OK] or died, and a
    # measured incorrect answer could never be observed.
    vocab, word_to_action = _vocab_for_mode(mode)
    mode_label = mode.upper()

    # ── Lineage setup ──────────────────────────────────────────────
    # Default: continuous mode using lineage_name (default 'main').
    # Pre-existing lineage is loaded; on exit we save back.
    # Explicit --load-bridge / --save-bridge override the lineage paths.
    lineage = None
    used_lineage_load = False
    if not from_scratch:
        lineage = BridgeLineage(lineage_name)
        # Compatibility guard: if the lineage was trained at a different
        # mode/arch, loading would silently shape-mismatch. Check meta.
        if lineage.exists():
            try:
                lm = lineage.read_metadata()
                stored_mode = (lm.arch or {}).get("mode")
                if stored_mode and stored_mode != mode:
                    print(f"[LINEAGE] Lineage '{lineage_name}' was trained "
                          f"in mode={stored_mode}; current --mode={mode} "
                          f"differs. Falling back to training from scratch; "
                          f"the new run will overwrite the lineage on save.",
                          flush=True)
                    lineage = None  # do not auto-load; will retrain
            except Exception as e:
                print(f"[LINEAGE] Warning: could not read metadata "
                      f"for '{lineage_name}': {e}", flush=True)

    if load_bridge:
        # Explicit load-bridge path takes precedence over lineage auto-load.
        bridge = _load_bridge_from_checkpoint(load_bridge, mode, seed,
                                                verbose=True)
    elif lineage is not None and lineage.exists():
        # Auto-load from lineage. Build the bridge skeleton then overlay
        # weights from the lineage's current.simstate.h5.
        print(f"[LINEAGE] Loading state from lineage '{lineage_name}' "
              f"(skipping training)", flush=True)
        bridge = _load_bridge_from_checkpoint(
            str(lineage.current_path), mode, seed, verbose=True,
        )
        used_lineage_load = True
    elif mode == "tier1":
        bridge = _load_or_train_tier1(seed, n_train_events, verbose=True)
        if save_bridge:
            _save_bridge_checkpoint(
                bridge, save_bridge, verbose=True,
                metadata={
                    "mode": mode, "seed": seed,
                    "n_train_events": n_train_events,
                },
            )
    elif mode == "synonym":
        bridge = _load_or_train_synonym(seed, n_train_events, verbose=True,
                                          vocab_size=8)
        if save_bridge:
            _save_bridge_checkpoint(
                bridge, save_bridge, verbose=True,
                metadata={
                    "mode": mode, "seed": seed,
                    "n_train_events": n_train_events,
                },
            )
    elif mode == "synonym12":
        # 12-word: capacity boundary at default arch -- use scaled (n_motor=2000)
        bridge = _load_or_train_synonym(seed, n_train_events, verbose=True,
                                          vocab_size=12,
                                          n_motor_per_action=2000,
                                          n_motor_fs_per_action=240)
        if save_bridge:
            _save_bridge_checkpoint(
                bridge, save_bridge, verbose=True,
                metadata={
                    "mode": mode, "seed": seed,
                    "n_train_events": n_train_events,
                },
            )
    elif mode == "synonym16":
        # 16-word: only tested at scaled arch (n_motor=2000)
        bridge = _load_or_train_synonym(seed, n_train_events, verbose=True,
                                          vocab_size=16,
                                          n_motor_per_action=2000,
                                          n_motor_fs_per_action=240)
        if save_bridge:
            _save_bridge_checkpoint(
                bridge, save_bridge, verbose=True,
                metadata={
                    "mode": mode, "seed": seed,
                    "n_train_events": n_train_events,
                },
            )
    else:
        raise ValueError(f"unknown mode: {mode}")

    # ── Lineage: fork (if requested) and initial-save after training ──
    if not from_scratch:
        if fork_lineage:
            # Need an existing lineage to fork; if we didn't just load
            # from one, save the freshly-trained bridge to 'lineage_name'
            # first so the fork has a parent.
            if not used_lineage_load and lineage is not None:
                _lineage_save(lineage, bridge, mode=mode, seed=seed,
                              n_train_events=n_train_events,
                              kind="init",
                              description=f"Initial train (seed={seed}, "
                                            f"n_train_events={n_train_events})")
            try:
                base_lineage = lineage if lineage is not None else BridgeLineage(lineage_name)
                lineage = base_lineage.fork(fork_lineage)
                print(f"[LINEAGE] Forked '{lineage_name}' -> '{fork_lineage}'. "
                      f"Future saves go to the fork.", flush=True)
            except FileExistsError:
                print(f"[LINEAGE] Lineage '{fork_lineage}' already exists; "
                      f"refusing to overwrite. Using existing target lineage "
                      f"for saves.", flush=True)
                lineage = BridgeLineage(fork_lineage)
            except FileNotFoundError as e:
                print(f"[LINEAGE] Fork failed: {e}", flush=True)
                lineage = None
        elif lineage is not None and not used_lineage_load:
            # We just trained from scratch (lineage didn't exist) — write
            # an initial save so future sessions can load it.
            _lineage_save(lineage, bridge, mode=mode, seed=seed,
                          n_train_events=n_train_events,
                          kind="init",
                          description=f"Initial train (seed={seed}, "
                                        f"n_train_events={n_train_events})")

    print(f"\nReady. Vocab: {sorted(vocab)}")
    if scripted_words is None:
        print(f"Type a word and press Enter.\n", flush=True)
    else:
        print(f"[SCRIPTED] running {len(scripted_words)} predefined words.",
              flush=True)

    transcript = []
    n_turns = 0
    correct = 0
    scripted_iter = iter(scripted_words) if scripted_words else None

    try:
        while True:
            if scripted_iter is not None:
                try:
                    line = next(scripted_iter).strip().lower()
                    print(f"> {line}", flush=True)
                except StopIteration:
                    print("[SCRIPTED COMPLETE]", flush=True)
                    break
            else:
                try:
                    line = input("> ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n[EOF]", flush=True)
                    break

            if not line:
                continue
            if line in ("quit", "exit", "q"):
                print("[QUIT]", flush=True)
                break

            # Dialog state commands (Track 3 layer 3, 2026-05-09).
            # `:` prefix disambiguates from vocab words and the unprefixed
            # `learn` command. Always available — no flag gate.
            dialog = _parse_dialog_command(line)
            if dialog is not None:
                verb = dialog["verb"]
                if verb == "forget":
                    transcript = []
                    n_turns = 0
                    correct = 0
                    print("  [:forget] history cleared", flush=True)
                    continue
                if verb == "history":
                    n_show = min(dialog.get("n", 5), len(transcript))
                    if n_show == 0:
                        print("  [:history] no turns yet", flush=True)
                        continue
                    print(f"  [:history] last {n_show} of {len(transcript)} turns:",
                          flush=True)
                    for t in transcript[-n_show:]:
                        if t.get("is_learn_command"):
                            print(f"    learn  {t.get('learned_word')}-> "
                                  f"motor_{t.get('target_action')} "
                                  f"({'OK' if t.get('bound_correctly') else 'X'})",
                                  flush=True)
                        else:
                            mark = "OK" if t.get("correct") else \
                                ("?" if not t.get("in_vocab") else "X")
                            print(f"    [{mark}] {t.get('user_word', '?'):<8} "
                                  f"-> motor_{t.get('predicted_action', '?')}",
                                  flush=True)
                    continue
                if verb == "again":
                    # Find the last vocab/inference turn (skip learn cmds)
                    last = next((t for t in reversed(transcript)
                                  if not t.get("is_learn_command")), None)
                    if last is None:
                        print("  [:again] no prior word to repeat", flush=True)
                        continue
                    line = last["user_word"]
                    print(f"  [:again] repeating '{line}'", flush=True)
                    # fall through to the regular inference path below
                elif verb == "opposite":
                    # Find the last predicted action and invert it
                    last = next((t for t in reversed(transcript)
                                  if not t.get("is_learn_command")), None)
                    if last is None:
                        print("  [:opposite] no prior action to invert",
                              flush=True)
                        continue
                    last_action = last.get("predicted_action")
                    if last_action not in ACTION_OPPOSITE:
                        print(f"  [:opposite] last action {last_action!r} "
                              f"has no inverse", flush=True)
                        continue
                    inverse_action = ACTION_OPPOSITE[last_action]
                    line = ACTION_TO_PRIMARY_WORD[inverse_action]
                    print(f"  [:opposite] last predicted motor_{last_action}; "
                          f"asking for opposite via '{line}'", flush=True)
                    # fall through to the regular inference path below
                elif verb == "speak":
                    # Track 3 layer 4 — A→W generative decoder.
                    # The verb's action arg was already parsed via
                    # _parse_dialog_command's basic dispatch; re-parse
                    # via _parse_speak_command for the full alias map.
                    target_action = _parse_speak_command(line)
                    if target_action is None:
                        print("  [:speak] usage: :speak <action>  "
                              "(N/E/S/W or north/east/south/west or "
                              "synonyms)", flush=True)
                        continue
                    speak_result = generative_inference(
                        bridge, target_action,
                        vocab_words=sorted(vocab),
                        temperature=speak_temperature,
                    )
                    rankings = speak_result["rankings"]
                    pred_word = speak_result["predicted_word"]
                    conf = speak_result["confidence"]
                    expected_word = ACTION_TO_PRIMARY_WORD.get(target_action)
                    is_correct = (
                        pred_word == expected_word
                        or word_to_action.get(pred_word) == target_action
                    )
                    marker = "[OK]" if is_correct else "[X] "
                    rank_str = " ".join(
                        f"{w}={s:.2f}" for w, s in rankings[:4]
                    )
                    print(f"  {marker} [SPEAK] motor_{target_action} "
                          f"-> '{pred_word}' (top-1={rankings[0][1]:.2f}, "
                          f"conf x{conf:.1f})", flush=True)
                    print(f"      rankings: {rank_str}", flush=True)
                    transcript.append({
                        "turn": n_turns + 1,
                        "user_word": f":speak {target_action}",
                        "is_speak_command": True,
                        "target_action": target_action,
                        "predicted_word": pred_word,
                        "confidence": conf,
                        "rankings": [(w, float(s)) for w, s in rankings[:4]],
                        "speak_correct": is_correct,
                    })
                    n_turns += 1
                    continue

            # Online learn command (only when --learn was passed)
            if allow_learn and line.startswith("learn "):
                parsed = _parse_learn_command(line)
                if parsed is None:
                    print("  [?] usage: learn <word> <action>  "
                          "(action = N/E/S/W or north/east/south/west or "
                          "up/right/down/left)", flush=True)
                    continue
                new_word, target = parsed
                # V_SCHEMA upgrade (2026-05-12): interleave new-word
                # training with anchor-word refresh. The schema-supported
                # binding mechanism (Tse 2007) provides reproducible
                # novel-key binding when the target's anchor word is
                # well-trained in the base lineage. 2/4 demonstrated on
                # main_hippo 200ev, vs 1/4 for plain learn_word_pairing.
                anchor_word_for_target = {
                    "N": "north", "E": "east",
                    "S": "south", "W": "west",
                }[target]
                M = 20
                n_batches = max(1, learn_n_events // M)
                for _ in range(n_batches):
                    learn_word_pairing(bridge, new_word, target,
                                       n_events=M, verbose=False)
                    # Brief anchor refresh — V_SCHEMA Tse 2007 mechanism
                    learn_word_pairing(bridge, anchor_word_for_target,
                                       target, n_events=2, verbose=False)
                print(f"  [LEARN-V_SCHEMA] trained '{new_word}' -> "
                      f"motor_{target} ({n_batches}×{M} events + "
                      f"{n_batches}×2 anchor='{anchor_word_for_target}' "
                      f"refresh)", flush=True)
                test_result = chat_inference(bridge, new_word)
                td = test_result["delta_counts"]
                pred_a = test_result["predicted_action"]
                conf = test_result["confidence_ratio"]
                bound_ok = (pred_a == target)
                marker = "[OK]" if bound_ok else "[X] "
                print(f"  {marker} [LEARN-TEST] '{new_word}' -> "
                      f"motor_{pred_a} (target motor_{target}) "
                      f"(delta N{td['N']:+d} E{td['E']:+d} "
                      f"S{td['S']:+d} W{td['W']:+d}, x{conf:.1f})",
                      flush=True)
                transcript.append({
                    "turn": n_turns + 1,
                    "user_word": f"learn {new_word} {target}",
                    "is_learn_command": True,
                    "learned_word": new_word,
                    "target_action": target,
                    "predicted_action": pred_a,
                    "confidence": conf,
                    "delta": td,
                    "bound_correctly": bound_ok,
                    "n_events_run": learn_n_events,
                })
                n_turns += 1
                continue

            n_turns += 1
            result = chat_inference(bridge, line)
            d = result["delta_counts"]
            pred_action = result["predicted_action"]
            pred_word = result["predicted_direction"]
            conf = result["confidence_ratio"]

            in_vocab = line in vocab
            expected_action = word_to_action.get(line)
            is_correct = (in_vocab and pred_action == expected_action)
            if is_correct:
                correct += 1

            if in_vocab:
                marker = "[OK]" if is_correct else "[X] "
                print(f"  {marker} [{mode_label} seed={seed}] sim hears "
                      f"'{line}', activates motor_{pred_action} "
                      f"(delta N{d['N']:+d} E{d['E']:+d} "
                      f"S{d['S']:+d} W{d['W']:+d}, x{conf:.1f})",
                      flush=True)
            else:
                print(f"  [?] '{line}' is not in vocab; tracking deltas "
                      f"anyway:", flush=True)
                print(f"      delta N{d['N']:+d} E{d['E']:+d} "
                      f"S{d['S']:+d} W{d['W']:+d}", flush=True)
                print(f"      best guess: motor_{pred_action} "
                      f"(low confidence x{conf:.1f})", flush=True)

            transcript.append({
                "turn": n_turns,
                "user_word": line,
                "in_vocab": in_vocab,
                "expected_action": expected_action,
                "predicted_action": pred_action,
                "confidence": conf,
                "delta": d,
                "correct": is_correct,
            })
    finally:
        print("\n" + "=" * 60)
        print(f"[DONE] {n_turns} turns total.")
        if n_turns > 0:
            # Use .get() with default False — :speak and :learn transcript
            # records don't have an in_vocab key (they're not chat turns).
            in_vocab_turns = sum(1 for t in transcript if t.get("in_vocab"))
            if in_vocab_turns > 0:
                print(f"  In-vocab accuracy: {correct}/{in_vocab_turns} "
                      f"= {correct/in_vocab_turns:.1%}")
        print("=" * 60, flush=True)

        # ── Lineage: save back on exit (continuous mode) ──────────────
        # Skip if --from-scratch, no lineage (mode mismatch fallback), or
        # the user gave an explicit --save-bridge target (they're using
        # the legacy explicit-path workflow).
        if (not from_scratch) and lineage is not None:
            n_in_vocab = sum(1 for t in transcript if t["in_vocab"])
            session_acc = (correct / n_in_vocab) if n_in_vocab > 0 else None
            try:
                _lineage_save(
                    lineage, bridge, mode=mode, seed=seed,
                    n_train_events=n_train_events,
                    kind="repl_session_end",
                    description=(
                        f"REPL session: {n_turns} turns "
                        f"({n_in_vocab} in-vocab"
                        + (f", acc={session_acc:.1%}" if session_acc is not None else "")
                        + ")"
                    ),
                    accuracy_metric="REPL in-vocab",
                    accuracy_value=session_acc,
                    accuracy_context=f"session_n_turns={n_turns}",
                )
                # Cheap retention: keep the last 30 history snapshots.
                lineage.prune_history(keep_last=30)
                print(f"[LINEAGE] State saved to '{lineage.name}'.",
                      flush=True)
            except Exception as e:
                print(f"[LINEAGE] Save failed (non-fatal): {e}", flush=True)

        if transcript_out and transcript:
            Path(transcript_out).parent.mkdir(parents=True, exist_ok=True)
            md = []
            md.append(f"# Interactive REPL transcript (mode={mode}, "
                      f"seed={seed})\n\n")
            md.append(f"**Vocab:** {sorted(vocab)}  \n")
            md.append(f"**Training:** {n_train_events} events/word\n\n")
            md.append("## Conversation\n\n```\n")
            for t in transcript:
                marker = "[OK]" if t["correct"] else "[X] " if t["in_vocab"] else "[?] "
                d = t["delta"]
                md.append(
                    f"  {marker} You: {t['user_word']:<8} -> "
                    f"motor_{t['predicted_action']} "
                    f"(delta N{d['N']:+4d} E{d['E']:+4d} "
                    f"S{d['S']:+4d} W{d['W']:+4d}, x{t['confidence']:.1f})\n"
                )
            md.append("```\n")
            Path(transcript_out).write_text("".join(md), encoding="utf-8")
            print(f"  Transcript saved: {transcript_out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode",
                    choices=["tier1", "synonym", "synonym12", "synonym16",
                             "synonym24", "synonym32"],
                    default="tier1",
                    help="Architecture mode: "
                         "tier1=4-word (validated 5/6+6/6); "
                         "synonym=8-word (validated 5/6+6/6, 3/3 GO at "
                         "consolidation); "
                         "synonym12=12-word (PARTIAL at default arch; "
                         "REPL uses scaled n_motor=2000 per capacity "
                         "hypothesis); "
                         "synonym16=16-word (master plan extension, "
                         "Unicode arrows up/right/down/left as 4th synonym); "
                         "synonym24=24-word (multi-lang: Spanish, German); "
                         "synonym32=32-word (multi-lang + Japanese, Arabic, "
                         "uses n_motor=3000 per capacity rule)")
    ap.add_argument("--seed", type=int, default=43,
                    help="Random seed (43 is the documented best Tier 1 seed; "
                         "42 is best Tier 2.1 single-seed)")
    ap.add_argument("--train-events", type=int, default=None,
                    help="Events per word during training (default: "
                         "200 for tier1, 400 for synonym, 200 for synonym12/16)")
    ap.add_argument("--transcript-out", type=str, default=None,
                    help="Save transcript to this markdown file at exit")
    ap.add_argument("--save-bridge", type=str, default=None,
                    help="Save the trained bridge state to this HDF5 path "
                         "after training. Future sessions can reload with "
                         "--load-bridge to skip ~6 min of training.")
    ap.add_argument("--load-bridge", type=str, default=None,
                    help="Load a previously-saved bridge state instead of "
                         "training. Skips the ~6 min training phase and "
                         "starts the REPL in ~10-30 sec. Per CLAUDE.md "
                         "save_checkpoint gotcha: doesn't preserve firing "
                         "thresholds / STP / eligibility -- but for "
                         "inference (REPL chat), weights are sufficient.")
    ap.add_argument("--scripted-words", type=str, default=None,
                    help="Comma-separated word list to process instead of "
                         "interactive stdin. Useful for CI / regression / "
                         "batch eval. Example: --scripted-words 'north,up,east,right'")
    ap.add_argument("--learn", action="store_true",
                    help="Enable online vocabulary learning. The REPL will "
                         "recognize 'learn <word> <action>' commands and "
                         "run an embodied-Hebbian binding session that adds "
                         "the new word to the existing motor pool, then "
                         "auto-tests the binding. Default OFF (learning "
                         "during chat is opt-in because new bindings can "
                         "perturb existing ones).")
    ap.add_argument("--learn-events", type=int, default=50,
                    help="Number of paired co-firing events per learn "
                         "command (default 50). Higher values give a "
                         "stronger binding but risk perturbing existing "
                         "vocab on the same motor pool.")
    ap.add_argument("--speak-temperature", type=float, default=0.0,
                    help="Softmax sampling temperature for the :speak "
                         "command. 0 (default) = strict argmax (always "
                         "produces the primary word). 0.01-0.02 = primary "
                         "dominant with occasional synonym lift. 0.05+ = "
                         "more variety, primary slightly preferred. "
                         ">0 enables natural-feeling synonym selection.")
    # ── Bridge Lineage Manager (continuous-learning workflow) ──
    # Default behavior: load the 'main' lineage if it exists; save back
    # on exit. The sim "lives" between sessions. Pass --from-scratch
    # for the prior behavior (always train; never auto-save to lineage).
    ap.add_argument("--lineage", type=str, default="main",
                    help="Name of the bridge lineage to load/save state "
                         "from (default: 'main'). Lineages live under "
                         "bridges/lineage/<name>/. If the lineage exists "
                         "and matches the current --mode, training is "
                         "skipped and state is loaded. On exit the bridge "
                         "is saved back to this lineage (snapshotting the "
                         "previous state to history/). Pass --from-scratch "
                         "to disable lineage auto-load/save.")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Science mode: always train from random init, "
                         "do NOT auto-load or auto-save the lineage. Use "
                         "for multi-seed reproducibility / experiments. "
                         "Without this flag, the REPL uses the 'main' "
                         "lineage as a persistent continual-learning state.")
    ap.add_argument("--fork-lineage", type=str, default=None,
                    help="Fork the loaded lineage into a new lineage with "
                         "the given name BEFORE making any further saves. "
                         "Useful for branching experiments without "
                         "disturbing 'main'. Example: --lineage main "
                         "--fork-lineage experiment_v3.")
    # ── Auto-grow demo (Phase A2 Strategy B) ──
    ap.add_argument("--auto-grow", action="store_true",
                    help="Before starting the REPL, run a demo of the "
                         "auto-grow orchestration loop (Phase A2 Strategy "
                         "B). Uses synthetic train/transfer functions to "
                         "demonstrate tier promotion via TierPromoter, "
                         "writing growth events to the active lineage. "
                         "Does NOT run real training; use "
                         "`python -m research.runners.auto_grow_chat` for "
                         "the standalone demo. Real bio_three_factor + "
                         "weight-transfer integration (Strategy A) is "
                         "deferred pending strategic Path 1/2/3 decision.")
    ap.add_argument("--auto-grow-max-promotions", type=int, default=3,
                    help="Max promotions for --auto-grow (default 3)")
    args = ap.parse_args()

    if args.train_events is None:
        if args.mode == "tier1":
            args.train_events = 200
        elif args.mode == "synonym":
            args.train_events = 400  # Tier 2.1 BREAKTHROUGH validated config
        else:  # synonym12, synonym16
            args.train_events = 200  # Per consolidation_synonym medium

    if args.load_bridge and args.save_bridge:
        ap.error("--load-bridge and --save-bridge are mutually exclusive "
                 "(saving overwrites a checkpoint that was just loaded)")
    if args.fork_lineage and args.from_scratch:
        ap.error("--fork-lineage requires a lineage to fork from; "
                 "incompatible with --from-scratch")

    # --auto-grow demo (Phase A2 Strategy B): fires BEFORE the REPL.
    # Demonstrates the orchestration loop; records growth events to
    # the active lineage. Doesn't replace REPL — REPL still runs after
    # the demo finishes.
    if args.auto_grow:
        from research.runners.auto_grow_chat import run_auto_grow_demo
        print("\n" + "=" * 60)
        print("AUTO-GROW DEMO (Phase A2 Strategy B; synthetic train/transfer)")
        print("=" * 60, flush=True)
        # Use the lineage name from args if set, else "auto_grow_demo"
        ag_lineage = (args.lineage
                       if not args.from_scratch and args.lineage
                       else "auto_grow_demo")
        run_auto_grow_demo(
            initial_tier=4,
            threshold=0.90,
            consecutive_required=3,
            max_promotions=int(args.auto_grow_max_promotions),
            max_epochs_per_tier=20,
            lineage_name=ag_lineage,
            verbose=True,
        )
        print("=" * 60)
        print(f"Growth events written to lineage '{ag_lineage}'. "
              f"REPL will now start at tier {args.mode}.")
        print("=" * 60, flush=True)

    scripted_words = None
    if args.scripted_words:
        scripted_words = [w.strip() for w in args.scripted_words.split(",")
                          if w.strip()]
        if not scripted_words:
            ap.error("--scripted-words got an empty list")

    run_repl(
        mode=args.mode,
        seed=args.seed,
        n_train_events=args.train_events,
        transcript_out=args.transcript_out,
        load_bridge=args.load_bridge,
        save_bridge=args.save_bridge,
        scripted_words=scripted_words,
        allow_learn=args.learn,
        learn_n_events=args.learn_events,
        speak_temperature=args.speak_temperature,
        lineage_name=args.lineage,
        from_scratch=args.from_scratch,
        fork_lineage=args.fork_lineage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
