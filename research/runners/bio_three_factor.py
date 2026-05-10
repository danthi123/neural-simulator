"""Three-factor learning rule with eligibility traces — biology-plausible
gradient approximation (Fremaux & Gerstner 2016).

Tests whether a biology-grounded learning rule can match what gradient
achieves. If yes, the W->A learning is feasible under realistic biology.
If no, the field's three-factor frameworks are insufficient for this
task and we need fundamentally different learning principles.

Three-factor rule:
    deltaw[i,j] = lr × E[i,j](t) × DA[motor_pool(j)](t)

where:
    E[i,j](t) = eligibility trace: an EMA of pre × post coincidence
                that marks "this synapse fired together recently"
    DA[a](t) = dopamine signal for motor pool a:
                +1 if motor_a is the target action,
                -1 if motor_a fired but isn't target (false positive)
                0 otherwise

Comparison to gradient (B3):
    Gradient:     deltaw = lr × (target - actual_rate)[motor] × pre_active
    Three-factor: deltaw = lr × pre_post_coincidence × dopamine_sign[motor]

Both produce credit assignment, but three-factor uses ONLY local
spike-timing × global scalar, which is what real synapses can compute.
Gradient requires the synapse to know "actual rate" of its post-pool
which would require non-local information.

Biological grounding:
- Eligibility trace: NMDA-receptor-mediated calcium accumulation,
  observable in dendrites for ~1 second post-spike (Sjöström 2001)
- Dopamine modulation: Schultz 1998 RPE; sub-second pulse modulates
  STDP magnitude per Reynolds & Wickens 2002
- Combined: Fremaux & Gerstner 2016 review formalizes the framework

Architecture: SAME as B3 (biological cortical canon). Only the
credit-assignment rule differs.

Usage:
    python -m research.runners.bio_three_factor \\
        --biological --seed 42 \\
        --apply-topographic-bias --enable-motor-fs \\
        --out-stats research/findings/raw/g11_bg/text_eval_3factor_seed42.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def update_eligibility_and_weights(
    eligibility,
    weights_data,
    edge_src,
    edge_dst,
    edge_off,
    edge_action,
    lang_active_mask,
    post_active_mask,
    da_per_action,
    decay_per_step: float,
    learning_rate: float,
    weight_min: float,
    weight_max: float,
    xp,
):
    """Pure three-factor update — works with both numpy and cupy backends.

    All inputs must be the same backend (numpy OR cupy arrays). The
    function mutates `eligibility` and `weights_data` in place.

    `xp` is the backend module (numpy or cupy). The function uses
    `xp.clip` for the bounded weight update; everything else is
    array operators that both backends share.

    This factoring is critical for the GPU port: when called with
    cupy arrays, the entire update happens on GPU with no host
    round-trip. When called with numpy arrays, the same code path
    runs on CPU — making the unit test for numerical equivalence
    trivial (just call with numpy and check results).

    Args:
        eligibility: shape (n_edges,) float32. Decayed + accumulated.
        weights_data: shape (n_total_synapses,) float32. The bridge's
            full CSR data array; we only update entries at `edge_off`.
        edge_src, edge_dst: shape (n_edges,) int32. Source / dst
            neuron IDs for each lang->motor edge.
        edge_off: shape (n_edges,) int64. Each edge's offset in
            weights_data.
        edge_action: shape (n_edges,) int8. Motor action index
            (0=N, 1=E, 2=S, 3=W) for each edge's destination pool.
        lang_active_mask: shape (n_neurons,) bool. True for
            language_input neurons currently active.
        post_active_mask: shape (n_neurons,) bool. True for any
            neuron that fired during the stim window.
        da_per_action: shape (4,) float32. Dopamine signal per
            motor action: +1 target, -1 false-positive, 0 quiet.
        decay_per_step: scalar. exp(-dt_ms / eligibility_tau_ms).
        learning_rate, weight_min, weight_max: scalars.
        xp: numpy or cupy module.

    Returns:
        None. Mutates `eligibility` and `weights_data` in place.
    """
    # Decay eligibility traces (NMDA-like exponential)
    eligibility *= decay_per_step

    # Accumulate pre x post coincidence into eligibility
    active_edges_mask = lang_active_mask[edge_src] & post_active_mask[edge_dst]
    eligibility[active_edges_mask] += 1.0

    # Apply three-factor weight update where eligibility > 0 AND da != 0
    edge_da = da_per_action[edge_action]
    update_mask = (eligibility > 0) & (edge_da != 0)
    delta = learning_rate * eligibility[update_mask] * edge_da[update_mask]
    new_w = weights_data[edge_off[update_mask]] + delta
    weights_data[edge_off[update_mask]] = xp.clip(new_w, weight_min, weight_max)


def run_three_factor(
    seed: int = 42,
    n_events_per_direction: int = 1000,
    learning_rate: float = 1e-3,
    eligibility_decay_tau: float = 50.0,  # ms; eligibility decay (NMDA tau)
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
    push_to_gpu_every: int = 64,
    fast_spike_reset: bool = True,
    freeze_plasticity_during_reset: bool = False,
                                          # 2026-05-10 perf optimization #3:
                                          # zero plasticity gain during reset_steps.
                                          # Expected 1.3-1.5x speedup but unvalidated
                                          # against multi-seed accuracy. Off by
                                          # default until validated; flip to True
                                          # for benchmarking arc.
    biological: bool = False,
    enable_nmda: bool = False,
    ou_tau_ms: float = 15.0,
    ou_std_current_pA: float = 100.0,
    gpu_eligibility: bool = True,  # Phase 1: keep eligibility/edges on GPU
    fp16_synapse_state: bool = False,  # Phase 2: FP16 cp_eligibility_trace
                                          # (validated <1mV voltage drift over
                                          # 1000 steps in 2026-05-05 perf wave 2;
                                          # off by default until tomorrow's
                                          # benchmarking arc validates against
                                          # multi-seed binding accuracy on
                                          # current arch tiers)
    da_mode: str = "sign",  # "sign" (classical), "graded" (magnitude DA)
    orthogonal_cues: bool = False,  # 2026-05-05: replace random hash codes
                                    # with non-overlapping banded codes
                                    # to test input-encoding-ambiguity
                                    # hypothesis (see step 2 design doc)
    embodied_hebbian: bool = False,  # 2026-05-05: Tier 1 — embodied
                                     # Hebbian word↔motor binding via
                                     # simultaneous teacher signals on
                                     # language_input + language_output
                                     # + motor target. No scalar reward;
                                     # co-activity drives STDP.
    embodied_motor_teacher_pA: float = 300.0,  # external current to target
                                                # motor pool during training
    synonym_mode: bool = False,  # Tier 2.1: train on 8-word vocab where
                                  # each action has 2 synonyms. Tests
                                  # whether embodied Hebbian binding
                                  # generalizes — both "north" and "up"
                                  # should bind to motor_N when trained
                                  # together. Doubles vocab without
                                  # adding new actions.
    synonym_cofire_fraction: float = 0.0,  # Tier 2.1 fallback: fraction
                                            # of synonym_mode trials where
                                            # BOTH synonyms drive together
                                            # (creates shared input
                                            # representation via Hebbian
                                            # co-firing between codes).
                                            # Range [0, 1]. 0 = no co-fire
                                            # (default v1/v2). 0.2 = 20%
                                            # of trials present both.
    pre_shape_events: int = 0,  # Tier 2.1 v6 (Tomasello fast-mapping
                                 # 2023 protocol): events of solo region
                                 # drive BEFORE paired training. Each
                                 # word presented alone for N events,
                                 # each motor pool driven alone for N
                                 # events, gates open in both phases.
                                 # Cell assemblies form via internal
                                 # recurrence + Hebbian. Then standard
                                 # paired training. 0 = disabled.
    synonym_vocab_size: int = 8,  # Tier 2.1 robustness: vocab size in
                                   # synonym mode. 8 = 4 actions × 2
                                   # synonyms (default). 12 = 4 × 3.
                                   # Tests scaling beyond initial
                                   # 8-word success.
    verbose: bool = True,
):
    """Three-factor learning at language_input -> motor_X synapses.

    Architecture (when biological=True): cortical canon at bio scale,
    same as bio_proof_of_concept and B3.

    Learning loop (per training event):
      1. Reset (clear input current).
      2. Drive language_input with token's sparse pattern.
      3. Forward-step the bridge for stim_steps_per_event.
      4. Read motor pool firing rates.
      5. Compute target one-hot for the correct action.
      6. Update synapse weights via three-factor rule:
         deltaw[i,j] = lr × E[i,j] × DA[motor_pool(j)]
         where E[i,j] ≈ pre_active × post_active (binary product),
         DA[a] = +1 if a is target, -1 if a fired and isn't target,
         else 0.
      7. Clip weights to [weight_min, weight_max].

    Key difference from B3: signal is sign-of-error, not exact error.
    More biologically plausible (synapses can't compute exact magnitudes
    of remote firing rates).
    """
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_minimal_brain_regions, build_biological_brain_regions,
        apply_topographic_bias as _apply_topo,
    )
    from sim.text_embeddings import vocab_to_drive_pattern, orthogonal_drive_pattern

    # Vocab order for orthogonal-cue mode. Index = cue band assignment.
    _VOCAB_ORDER = ["north", "east", "south", "west"]

    def _drive_for_token(token: str) -> np.ndarray:
        """Encode a token into a drive vector. Branches on orthogonal_cues."""
        if orthogonal_cues:
            cue_idx = _VOCAB_ORDER.index(token)
            return orthogonal_drive_pattern(
                cue_idx=cue_idx,
                n_cues=len(_VOCAB_ORDER),
                n_neurons=n_lang_input,
                drive_max_pA=lang_input_drive_pA,
                sparsity=token_sparsity,
            )
        return vocab_to_drive_pattern(
            token, n_neurons=n_lang_input,
            drive_max_pA=lang_input_drive_pA, sparsity=token_sparsity,
        )

    rng = np.random.default_rng(seed)

    if verbose:
        arch = "BIOLOGICAL" if biological else "MINIMAL"
        print("=" * 60)
        print(f"BIO THREE-FACTOR LEARNING (seed={seed}, arch={arch})")
        print(f"  n_lang={n_lang_input}, motor_per_action={n_motor_per_action}")
        print(f"  events: {n_events_per_direction}/dir x 4 = {4 * n_events_per_direction}")
        print(f"  lr={learning_rate}, eligibility tau={eligibility_decay_tau}ms")
        print(f"  apply_topographic_bias={apply_topographic_bias}, "
              f"enable_motor_fs={enable_motor_fs}")
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
            enable_language_output=embodied_hebbian,
            n_lang_output=n_lang_input,  # match input size for symmetric encoding
            # Bumped 0.5 → 2.0 to match forward pathway strength.
            # Tier 1 smoke showed 0.5 was too weak vs language_output
            # internal noise, leaving STDP unable to differentiate
            # motor inputs above architectural N-bias floor.
            motor_to_language_output_weight=2.0,
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
    cfg.stdp_w_max = max(weight_max + 1.0, 5.0)
    cfg.fast_spike_reset = fast_spike_reset
    cfg.fp16_synapse_state = fp16_synapse_state

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

    # Topographic prior init (if requested)
    if apply_topographic_bias:
        _apply_topo(
            bridge,
            topographic_factor=topographic_bias_factor,
            off_target_factor=off_target_bias_factor,
            n_lang_input=n_lang_input,
            sparsity=token_sparsity,
            orthogonal_cues=orthogonal_cues,
            # In embodied-Hebbian mode, also apply reciprocal prior:
            # motor_X edges to "X-encoded" neurons in language_output
            # boosted, off-target reduced. Without this, A→W direction
            # has no per-action specificity to start from.
            apply_reciprocal=embodied_hebbian,
            n_lang_output=n_lang_input if embodied_hebbian else 256,
            verbose=verbose,
        )

    # Plasticity gate setup. In 3-factor mode, freeze (we apply our own
    # rule manually). In embodied-Hebbian mode, KEEP OPEN so bridge's
    # built-in STDP fires naturally on co-active pre+post.
    if not embodied_hebbian:
        try:
            bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        except Exception as e:
            if verbose:
                print(f"WARNING: could not freeze gate: {e}", flush=True)
    else:
        # Open both gates explicitly (1.0 default but be explicit)
        try:
            bridge.set_plasticity_gate("language_input_to_motor", 1.0)
            bridge.set_plasticity_gate("motor_to_language_output", 1.0)
        except Exception as e:
            if verbose:
                print(f"WARNING: could not open gate: {e}", flush=True)
        if verbose:
            print("  embodied-Hebbian mode: bridge STDP will fire on "
                  "co-active synapses; no custom 3-factor update.", flush=True)

    # GPU port: when gpu_eligibility is True, all eligibility, edge,
    # and mask arrays live on GPU and weight updates happen in-place
    # on bridge.cp_connections.data — eliminating the per-event
    # 6 MB CSR copy back and forth from CPU. Speedup ~2x measured.
    # Set gpu_eligibility=False to fall back to numpy (useful for
    # debugging numerical issues).
    xp = cp if gpu_eligibility else np

    # Build (src, dst) -> idx map for fast CSR mutation. Always pull
    # indptr/indices to CPU once (we only need them for edge enumeration,
    # not the hot path).
    indptr = bridge.cp_connections.indptr.get()
    indices = bridge.cp_connections.indices.get()
    if gpu_eligibility:
        # Hot-path uses bridge.cp_connections.data IN-PLACE on GPU.
        # `data` here is just an alias for the GPU array.
        data = bridge.cp_connections.data
    else:
        # CPU mode: pull data, mutate locally, push back periodically.
        data = bridge.cp_connections.data.get()

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}")) for a in ["N","E","S","W"]}
    # Embodied Hebbian: language_output region only exists if enabled.
    if embodied_hebbian:
        try:
            lang_output_idx = list(rm.indices("language_output"))
        except Exception:
            raise RuntimeError(
                "embodied_hebbian=True but language_output region not "
                "found. Builder must be called with enable_language_output=True."
            )
    else:
        lang_output_idx = None
    motor_idx_set = {a: set(motor_idx[a]) for a in ["N","E","S","W"]}
    all_motor_set = set()
    for a in ["N","E","S","W"]:
        all_motor_set |= motor_idx_set[a]

    # Edge list: build as parallel numpy arrays for vectorized updates.
    # CRITICAL: the naive python-loop version (1.23M edges × 4000 events
    # × O(set lookups)) takes 20+ hours/seed. Vectorized numpy is ~100x
    # faster — must run as numpy arrays, never python loops over edges.
    edge_src_l: List[int] = []
    edge_dst_l: List[int] = []
    edge_off_l: List[int] = []
    edge_action_l: List[int] = []  # 0=N, 1=E, 2=S, 3=W
    ACTION_TO_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}
    for src in lang_input_idx:
        s = int(indptr[src])
        e_end = int(indptr[src + 1])
        for off in range(s, e_end):
            dst = int(indices[off])
            for a in ["N","E","S","W"]:
                if dst in motor_idx_set[a]:
                    edge_src_l.append(src)
                    edge_dst_l.append(dst)
                    edge_off_l.append(off)
                    edge_action_l.append(ACTION_TO_IDX[a])
                    break

    # Edge arrays live on the chosen backend (GPU if gpu_eligibility=True).
    edge_src = xp.asarray(edge_src_l, dtype=xp.int32)
    edge_dst = xp.asarray(edge_dst_l, dtype=xp.int32)
    edge_off = xp.asarray(edge_off_l, dtype=xp.int64)
    edge_action = xp.asarray(edge_action_l, dtype=xp.int8)
    n_edges = int(len(edge_src_l))

    # Build neuron-id -> "is in lang_input" / "is in motor_X" boolean masks
    # for O(1) vectorized lookups. n_neurons is the bridge's full neuron
    # count (cortical canon adds inh + recurrence, so total > lang+motor).
    n_neurons = int(bridge.cp_membrane_potential_v.shape[0])
    is_lang = xp.zeros(n_neurons, dtype=bool)
    is_lang[xp.asarray(lang_input_idx, dtype=xp.int32)] = True
    motor_action_of_neuron = -xp.ones(n_neurons, dtype=xp.int8)
    for a, idx_list in motor_idx.items():
        motor_action_of_neuron[xp.asarray(idx_list, dtype=xp.int32)] = ACTION_TO_IDX[a]

    if verbose:
        backend = "GPU" if gpu_eligibility else "CPU"
        print(f"  {n_edges} language->motor edges to learn (vectorized, {backend})", flush=True)

    # Eligibility trace per edge — on chosen backend
    eligibility = xp.zeros(n_edges, dtype=xp.float32)
    decay_per_step = float(np.exp(-dt_ms / eligibility_decay_tau))

    # Build event buffer (balanced)
    if synonym_mode:
        # Tier 2.1: each trial picks a random synonym for the target
        # direction. With synonym_groups[target] = [primary, alt],
        # ~50% of trials use primary and ~50% alt. Both bind to same
        # motor via embodied Hebbian co-firing.
        # If synonym_cofire_fraction > 0, that fraction of trials
        # presents BOTH synonyms simultaneously (creates shared input
        # representation via input-input Hebbian co-firing).
        from research.runners.text_eval import get_synonym_groups
        SYN = get_synonym_groups(synonym_vocab_size)
        DIRECTIONS = ["N", "E", "S", "W"]  # actions, not words
        buffer = []
        for action in DIRECTIONS:
            synonyms = SYN[action]
            for trial_idx in range(n_events_per_direction):
                # Decide if this trial is a co-fire trial
                if (synonym_cofire_fraction > 0.0 and
                        rng.random() < synonym_cofire_fraction):
                    # Co-fire: both synonyms drive together. Token list.
                    buffer.append({
                        "token": synonyms,  # list, signals co-fire
                        "action": action,
                        "cofire": True,
                    })
                else:
                    token = synonyms[rng.integers(0, len(synonyms))]
                    buffer.append({"token": token, "action": action,
                                    "cofire": False})
        rng.shuffle(buffer)
    else:
        DIRECTIONS = ["north","east","south","west"]
        DIR_TO_ACTION = {"north":"N","east":"E","south":"S","west":"W"}
        buffer = []
        for direction in DIRECTIONS:
            a = DIR_TO_ACTION[direction]
            for _ in range(n_events_per_direction):
                buffer.append({"token": direction, "action": a})
        rng.shuffle(buffer)

    if verbose:
        print(f"\n  Synthetic buffer: {len(buffer)} events", flush=True)

    # Tier 2.1 v6: two-stage curriculum (Tomasello fast-mapping 2023).
    # Pre-shape language_input + motor pools INDEPENDENTLY before paired
    # training. Each region develops stable cell assemblies via internal
    # recurrence + Hebbian. Then a single paired event suffices to bind.
    # Plasticity gates stay OPEN during pre-shape (intra-region STDP +
    # any cross-region eligibility that fires through topographic prior).
    if pre_shape_events > 0 and embodied_hebbian:
        if verbose:
            print(f"\n  Pre-shape phase: {pre_shape_events} events solo "
                  f"per region (Tomasello two-stage protocol)", flush=True)

        # Phase 1a: drive each language_input pattern alone (no motor teacher)
        if synonym_mode:
            from research.runners.text_eval import get_synonym_groups
            _SG = get_synonym_groups(synonym_vocab_size)
            preshape_words = [w for ws in _SG.values() for w in ws]
        else:
            preshape_words = ["north", "east", "south", "west"]
        ps_buffer = []
        for w in preshape_words:
            for _ in range(pre_shape_events):
                ps_buffer.append({"token": w, "phase": "lang_solo"})
        for a in ["N", "E", "S", "W"]:
            for _ in range(pre_shape_events):
                ps_buffer.append({"token": None, "action": a, "phase": "motor_solo"})
        rng.shuffle(ps_buffer)
        if verbose:
            print(f"  Pre-shape buffer: {len(ps_buffer)} events", flush=True)

        for ps_idx, ps_event in enumerate(ps_buffer):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            if ps_event["phase"] == "lang_solo":
                # Drive language_input alone (also language_output for
                # auto-association during solo)
                drive = _drive_for_token(ps_event["token"])
                bridge.cp_external_input_current[
                    cp.asarray(lang_input_idx, dtype=cp.int64)
                ] = cp.asarray(drive, dtype=cp.float32)
                if embodied_hebbian:
                    bridge.cp_external_input_current[
                        cp.asarray(lang_output_idx, dtype=cp.int64)
                    ] = cp.asarray(drive, dtype=cp.float32)
            else:
                # Drive motor target alone (no language)
                target_motor_arr = cp.asarray(
                    motor_idx[ps_event["action"]], dtype=cp.int64)
                bridge.cp_external_input_current[target_motor_arr] = float(
                    embodied_motor_teacher_pA
                )

            for _ in range(stim_steps_per_event):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            if verbose and (ps_idx + 1) % 100 == 0:
                print(f"  [pre-shape] {ps_idx+1}/{len(ps_buffer)} solo events",
                      flush=True)

    # Per-action expected firing-rate target (one-hot, but soft thresholded)
    expected_max_per_neuron = 0.10  # per text_minimal_isolation defaults
    target_high = expected_max_per_neuron * stim_steps_per_event * 1.0   # max
    target_low = expected_max_per_neuron * stim_steps_per_event * 0.2    # quiet baseline

    t_start = time.time()
    correct_recent = 0
    n_recent = 0
    # Debug timing for first 5 events to surface bottlenecks
    DEBUG_FIRST_N = 5
    for event_idx, event in enumerate(buffer):
        ev_start = time.time() if event_idx < DEBUG_FIRST_N else 0
        token = event["token"]
        target_action = event["action"]

        # Inter-trial reset (input zeroed; plasticity continues at default
        # gain). 2026-05-10: plasticity-during-reset can be GATED OFF for
        # ~1.3-1.5x speedup on plasticity-heavy training, but the gate
        # change introduces a subtle accuracy shift (small spurious STDP
        # updates during quiet reset are zeroed). Currently OPT-IN via
        # `freeze_plasticity_during_reset=True` until validated against
        # multi-seed binding accuracy (next-day work).
        bridge.cp_external_input_current[:] = 0.0
        if freeze_plasticity_during_reset:
            bridge.set_global_plasticity_gain(0.0)
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        if freeze_plasticity_during_reset:
            bridge.set_global_plasticity_gain(1.0)

        # Drive language_input. If cofire trial (synonym_mode + cofire),
        # OR the two synonym drive vectors together — both word codes
        # active simultaneously, creating shared representation via
        # input-input Hebbian co-firing.
        if event.get("cofire", False) and isinstance(token, list):
            drives = [_drive_for_token(t) for t in token]
            drive = np.maximum.reduce(drives)  # max combine — both active
        else:
            drive = _drive_for_token(token)
        bridge.cp_external_input_current[
            cp.asarray(lang_input_idx, dtype=cp.int64)
        ] = cp.asarray(drive, dtype=cp.float32)

        # Embodied-Hebbian teachers: drive language_output (output teacher)
        # AND elevate motor target (action teacher) so all 3 sites co-fire.
        # Bridge's STDP fires automatically on plastic synapses in co-active
        # pre+post pairs. No custom rule needed.
        # In co-fire trials, language_output also gets BOTH synonym
        # patterns (max-combined) so output STDP also binds both codes.
        if embodied_hebbian:
            # language_output teacher: same drive as language_input (which
            # may be cofire-combined). Identical pattern at both sites.
            bridge.cp_external_input_current[
                cp.asarray(lang_output_idx, dtype=cp.int64)
            ] = cp.asarray(drive, dtype=cp.float32)
            # motor teacher: elevated current to target motor pool
            target_motor_arr = cp.asarray(motor_idx[target_action], dtype=cp.int64)
            bridge.cp_external_input_current[target_motor_arr] += float(
                embodied_motor_teacher_pA
            )

        # Track which lang neurons were active for this token.
        # Mask lives on chosen backend; computed once per event.
        lang_active_mask = xp.zeros(n_neurons, dtype=bool)
        active_local = np.where(drive > 0)[0]
        active_global_np = np.asarray(lang_input_idx, dtype=np.int32)[active_local]
        lang_active_mask[xp.asarray(active_global_np, dtype=xp.int32)] = True

        # Forward-prop. Accumulate spike counts on GPU (single sync at end)
        # to avoid the 50× per-step GPU->CPU stalls of the naive loop.
        motor_spike_count_gpu = cp.zeros(4, dtype=cp.int32)
        post_active_mask_gpu = cp.zeros(n_neurons, dtype=bool)
        # Pre-build motor index arrays on GPU once (per event; could move
        # outside loop but small constant)
        motor_idx_gpu = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                         for a in ["N","E","S","W"]}
        ACTION_LIST = ["N","E","S","W"]
        for _ in range(stim_steps_per_event):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            fired = bridge.cp_firing_states  # cupy bool array
            for a_i, a in enumerate(ACTION_LIST):
                cnt = fired[motor_idx_gpu[a]].sum()
                motor_spike_count_gpu[a_i] += cnt
            # Mark any neuron that fired as post-active (cumulative OR)
            post_active_mask_gpu |= fired

        # GPU-resident: post_active_mask stays on GPU when gpu_eligibility.
        # Spike counts always pulled to CPU (small, 4 ints).
        motor_spike_counts_arr = motor_spike_count_gpu.get()
        if gpu_eligibility:
            post_active_mask = post_active_mask_gpu  # stays on GPU
        else:
            post_active_mask = post_active_mask_gpu.get()  # bring to CPU

        # DA per motor pool. Two modes:
        #   "sign" (classical 3-factor, Fremaux & Gerstner 2016):
        #     +1 for target, -1 for non-target that fired (false-positive
        #     penalty), 0 otherwise. Direction-only signal.
        #   "graded" (Schultz 1998 — DA scales with deviation magnitude):
        #     da[a] = (target_rate - actual_rate) / target_high.
        #     Magnitude carries error info that scalar sign discards.
        #     Tests "is sign-only the bottleneck?" hypothesis.
        target_a = ACTION_TO_IDX[target_action]
        da_per_action = xp.zeros(4, dtype=xp.float32)
        if da_mode == "graded":
            # Target rate = full rate for target_a, target_low for others
            for a_i in range(4):
                target_rate = float(target_high if a_i == target_a else target_low)
                actual_rate = float(motor_spike_counts_arr[a_i])
                # Normalize by target_high so |da| <= ~1 in typical cases
                da_val = (target_rate - actual_rate) / max(1.0, target_high)
                # Clip to [-1, +1] to keep consistent with sign mode magnitude
                da_per_action[a_i] = max(-1.0, min(1.0, da_val))
        else:
            # Classical sign-only DA
            da_per_action[target_a] = 1.0
            for a_i in range(4):
                if a_i != target_a and motor_spike_counts_arr[a_i] > target_low:
                    da_per_action[a_i] = -1.0

        # Three-factor eligibility + weight update — SKIPPED in embodied
        # Hebbian mode (bridge's built-in STDP already fires during forward-
        # prop in the co-active synapses).
        if not embodied_hebbian:
            update_eligibility_and_weights(
                eligibility=eligibility,
                weights_data=data,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_off=edge_off,
                edge_action=edge_action,
                lang_active_mask=lang_active_mask,
                post_active_mask=post_active_mask,
                da_per_action=da_per_action,
                decay_per_step=decay_per_step,
                learning_rate=learning_rate,
                weight_min=weight_min,
                weight_max=weight_max,
                xp=xp,
            )

        # Track rolling correctness
        winner_a = int(np.argmax(motor_spike_counts_arr))
        if winner_a == target_a:
            correct_recent += 1
        n_recent += 1

        if event_idx < DEBUG_FIRST_N and verbose:
            print(f"  [3factor DEBUG] event {event_idx}: "
                  f"{time.time() - ev_start:.2f}s", flush=True)

        # Periodically push weights back to GPU (CPU mode only).
        # In GPU mode, weights live on GPU permanently — no push needed.
        if (event_idx + 1) % push_to_gpu_every == 0:
            if not gpu_eligibility:
                bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)

        # Emit [PROGRESS] every 50 events so the webapp inflight panel
        # + 3D Brain live mode show real progress regardless of `verbose`.
        # 2026-05-09 fix: this used to be nested inside the
        # push_to_gpu_every (=64) conditional, so it only fired at
        # LCM(50,64)=1600 (i.e. ONCE per typical training run). Now
        # decoupled so it actually fires every 50 events as documented.
        # Also decoupled from the `verbose` flag — the print stays
        # behind verbose, but the structured event always emits.
        if (event_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rolling_acc = correct_recent / n_recent if n_recent else 0
            if verbose:
                print(f"  [3factor] {event_idx+1}/{len(buffer)} events "
                      f"({elapsed:.0f}s) rolling_acc={rolling_acc:.1%}",
                      flush=True)
            from sim.progress import emit_progress
            emit_progress(
                "training", event_idx + 1, len(buffer),
                phase="three-factor", unit="events",
                label="bio-3factor",
                elapsed_seconds=elapsed,
                rolling_acc=rolling_acc,
            )
            correct_recent = 0
            n_recent = 0

    # Final push (CPU mode); GPU mode already has weights in place.
    if not gpu_eligibility:
        bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)
    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Training complete: {len(buffer)} events ({elapsed:.0f}s)",
              flush=True)

    # Post-training: freeze plasticity gates so eval doesn't perturb
    # learned weights. Critical-period style: training was the open
    # window; eval is the adult phase.
    if embodied_hebbian:
        try:
            bridge.set_plasticity_gate("language_input_to_motor", 0.0)
            bridge.set_plasticity_gate("motor_to_language_output", 0.0)
            if verbose:
                print("  embodied-Hebbian: plasticity gates frozen for eval.",
                      flush=True)
        except Exception as e:
            if verbose:
                print(f"WARNING: could not freeze gates post-training: {e}",
                      flush=True)

    return bridge, [{
        "regime": "bio_three_factor",
        "n_total_events": len(buffer),
        "n_per_direction": n_events_per_direction,
        "elapsed_seconds": elapsed,
    }]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events-per-direction", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--eligibility-decay-tau", type=float, default=50.0,
                    help="Eligibility trace decay (ms). Default 50ms ~ NMDA tau.")
    ap.add_argument("--stim-steps-per-event", type=int, default=50)
    ap.add_argument("--reset-steps", type=int, default=50)
    ap.add_argument("--n-motor-per-action", type=int, default=25)
    ap.add_argument("--n-lang-input", type=int, default=256)
    ap.add_argument("--n-motor-fs-per-action", type=int, default=3)
    ap.add_argument("--token-sparsity", type=float, default=0.1)
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--apply-topographic-bias", action="store_true", default=False)
    ap.add_argument("--topographic-bias-factor", type=float, default=1.5)
    ap.add_argument("--off-target-bias-factor", type=float, default=0.7)
    ap.add_argument("--enable-motor-fs", action="store_true", default=False)
    ap.add_argument("--biological", action="store_true", default=False)
    ap.add_argument("--enable-nmda", action="store_true", default=False)
    ap.add_argument("--no-gpu-eligibility", dest="gpu_eligibility",
                    action="store_false", default=True,
                    help="CPU-mode fallback: keep eligibility/edges as numpy "
                    "arrays on host. Default uses GPU (~2x faster, no host "
                    "round-trips). Use --no-gpu-eligibility for debugging.")
    ap.add_argument("--fp16-synapse-state", action="store_true", default=False,
                    help="Opt-in FP16 storage for cp_eligibility_trace (and "
                    "future synapse-side state). Voltages stay FP32. Honest "
                    "expected gain: 1.05-1.15x (we use sparse SpMV, no Tensor "
                    "Cores). Validate via tests/test_fp16_drift.py first.")
    ap.add_argument("--da-mode", type=str, default="sign",
                    choices=["sign", "graded"],
                    help="DA signal mode. 'sign' = classical 3-factor +1/-1/0 "
                    "(Fremaux & Gerstner 2016). 'graded' = magnitude-graded "
                    "Schultz 1998 — DA scales with target-actual deviation, "
                    "preserving error magnitude info. Tests if scalar-sign "
                    "DA is the bottleneck.")
    ap.add_argument("--orthogonal-cues", action="store_true", default=False,
                    help="Replace SHA-256-hashed sparse codes with "
                    "non-overlapping banded codes (cue 0 → neurons 0..n_active, "
                    "cue 1 → neurons stride..stride+n_active, etc). Tests "
                    "whether the W→A 3-factor failure is caused by input "
                    "encoding ambiguity (overlapping codes confuse scalar "
                    "DA). 2026-05-05 step 2 experiment.")
    ap.add_argument("--embodied-hebbian", action="store_true", default=False,
                    help="Tier 1 — embodied Hebbian word↔motor binding via "
                    "simultaneous teacher signals on language_input + "
                    "language_output + motor target. Bridge's built-in STDP "
                    "fires on co-active synapses; no scalar reward. Adds "
                    "language_output region + reciprocal motor → "
                    "language_output pathway. Tests Pulvermüller 2001-2012 "
                    "embodied semantics paradigm.")
    ap.add_argument("--embodied-motor-teacher-pA", type=float, default=300.0,
                    help="External current pushed to target motor pool "
                    "during embodied trial. Default 300pA = strong teacher. "
                    "Higher = faster convergence; lower = more biological "
                    "(weaker reliance on external signal).")
    ap.add_argument("--synonym-mode", action="store_true", default=False,
                    help="Tier 2.1: train on 8-word vocab (4 actions x 2 "
                    "synonyms each: {N: [north, up], E: [east, right], "
                    "S: [south, down], W: [west, left]}). Each trial picks "
                    "a random synonym; both should bind to the same motor "
                    "via embodied Hebbian co-firing. Doubles vocab without "
                    "adding new actions.")
    ap.add_argument("--synonym-cofire-fraction", type=float, default=0.0,
                    help="Tier 2.1 fallback: fraction of synonym_mode "
                    "trials where BOTH synonyms drive simultaneously. "
                    "Range [0, 1]. 0 = no co-fire (default). 0.2 = 20% "
                    "of trials present 'north' AND 'up' together. Forces "
                    "shared input representation via Hebbian co-firing "
                    "between word codes.")
    ap.add_argument("--pre-shape-events", type=int, default=0,
                    help="Tier 2.1 v6 (Tomasello fast-mapping protocol): "
                    "events of solo region drive BEFORE paired training. "
                    "Each word presented alone for N events; each motor "
                    "pool driven alone for N events. Cell assemblies "
                    "form via internal recurrence + STDP. Then standard "
                    "paired training. 0 = disabled (default). Try 100-300.")
    ap.add_argument("--synonym-vocab-size", type=int, default=8,
                    choices=[8, 12],
                    help="Tier 2.1 robustness: synonym vocab size. "
                    "8 = 4 actions x 2 synonyms (default, validated). "
                    "12 = 4 actions x 3 synonyms (adds short forms n/e/s/w). "
                    "Tests scaling beyond initial 8-word success.")
    ap.add_argument("--n-eval-per-word", type=int, default=25)
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    if args.biological:
        if args.n_lang_input == 256:
            args.n_lang_input = 2048
        if args.n_motor_per_action == 25:
            args.n_motor_per_action = 500
        if args.n_motor_fs_per_action == 3:
            args.n_motor_fs_per_action = 60
        args.enable_nmda = True

    bridge, train_stats = run_three_factor(
        seed=args.seed,
        n_events_per_direction=args.n_events_per_direction,
        learning_rate=args.learning_rate,
        eligibility_decay_tau=args.eligibility_decay_tau,
        stim_steps_per_event=args.stim_steps_per_event,
        reset_steps=args.reset_steps,
        n_motor_per_action=args.n_motor_per_action,
        n_lang_input=args.n_lang_input,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        token_sparsity=args.token_sparsity,
        dt_ms=args.dt_ms,
        apply_topographic_bias=args.apply_topographic_bias,
        topographic_bias_factor=args.topographic_bias_factor,
        off_target_bias_factor=args.off_target_bias_factor,
        enable_motor_fs=args.enable_motor_fs,
        biological=args.biological,
        enable_nmda=args.enable_nmda,
        gpu_eligibility=args.gpu_eligibility,
        fp16_synapse_state=args.fp16_synapse_state,
        da_mode=args.da_mode,
        orthogonal_cues=args.orthogonal_cues,
        embodied_hebbian=args.embodied_hebbian,
        embodied_motor_teacher_pA=args.embodied_motor_teacher_pA,
        synonym_mode=args.synonym_mode,
        synonym_cofire_fraction=args.synonym_cofire_fraction,
        pre_shape_events=args.pre_shape_events,
        synonym_vocab_size=args.synonym_vocab_size,
        verbose=True,
    )

    # Eval
    from research.runners.text_eval import evaluate_word_to_action
    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_per_word} per word)")
    print("=" * 60, flush=True)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50,
        token_sparsity=args.token_sparsity,
        orthogonal_cues=args.orthogonal_cues,
        synonym_mode=args.synonym_mode,
        synonym_vocab_size=args.synonym_vocab_size,
    )
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}", flush=True)
    print(f"  Confusion: {wa_result['confusion_matrix']}", flush=True)

    # Embodied-Hebbian Tier 1: also evaluate action → word (A→W)
    aw_result = None
    if args.embodied_hebbian:
        import cupy as cp_eval
        print("\n" + "=" * 60)
        print(f"EVAL: action -> word (Tier 1 A->W, {args.n_eval_per_word} per action)")
        print("=" * 60, flush=True)
        rm_eval = bridge.region_manager
        lang_output_idx = list(rm_eval.indices("language_output"))
        motor_idx_eval = {a: list(rm_eval.indices(f"motor_{a}"))
                          for a in ["N", "E", "S", "W"]}
        # In synonym_mode, A→W eval matches against extended vocab; correct
        # if predicted word's action == motor's action (either synonym OK).
        if args.synonym_mode:
            from research.runners.text_eval import (
                get_extended_word_to_action, get_synonym_groups,
            )
            EWA = get_extended_word_to_action(args.synonym_vocab_size)
            WORDS = list(EWA.keys())
            WORD_TO_ACTION_AW = EWA
        else:
            WORDS = ["north", "east", "south", "west"]
            WORD_TO_ACTION_AW = {"north": "N", "east": "E",
                                 "south": "S", "west": "W"}
        ACTION_TO_WORD = {"N": "north", "E": "east", "S": "south", "W": "west"}
        confusion_aw = {a: {w: 0 for w in WORDS} for a in ["N", "E", "S", "W"]}
        correct_aw = 0
        total_aw = 0
        for action in ["N", "E", "S", "W"]:
            for trial in range(args.n_eval_per_word):
                # Reset
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(50):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                # Drive motor pool
                target_arr = cp_eval.asarray(motor_idx_eval[action], dtype=cp_eval.int64)
                bridge.cp_external_input_current[target_arr] = float(
                    args.embodied_motor_teacher_pA
                )
                # Forward — accumulate language_output spikes
                lang_out_arr = cp_eval.asarray(lang_output_idx, dtype=cp_eval.int64)
                lang_out_spike_count = cp_eval.zeros(len(lang_output_idx), dtype=cp_eval.int32)
                for _ in range(100):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                    fired = bridge.cp_firing_states
                    lang_out_spike_count += fired[lang_out_arr].astype(cp_eval.int32)
                # Convert to numpy for cosine match
                activity = lang_out_spike_count.get().astype(np.float32)
                if activity.sum() > 0:
                    from sim.text_embeddings import (
                        nearest_token, vocab_to_drive_pattern,
                        orthogonal_drive_pattern,
                    )
                    # Match against word patterns. Use the SAME encoding the
                    # network was trained on.
                    best_word, best_score = None, -1
                    for w_idx, w in enumerate(WORDS):
                        if args.orthogonal_cues:
                            template = orthogonal_drive_pattern(
                                cue_idx=w_idx, n_cues=len(WORDS),
                                n_neurons=len(lang_output_idx),
                                sparsity=args.token_sparsity,
                            )
                        else:
                            template = vocab_to_drive_pattern(
                                w, n_neurons=len(lang_output_idx),
                                sparsity=args.token_sparsity,
                            )
                        # Cosine similarity (template active sites vs activity)
                        norm_t = float(np.linalg.norm(template))
                        norm_a = float(np.linalg.norm(activity))
                        if norm_t > 0 and norm_a > 0:
                            score = float(np.dot(template, activity)) / (norm_t * norm_a)
                        else:
                            score = 0.0
                        if score > best_score:
                            best_score, best_word = score, w
                    pred = best_word
                else:
                    pred = WORDS[0]  # default if silent
                confusion_aw[action][pred] += 1
                total_aw += 1
                # In synonym_mode, correct if pred's action == motor's action
                # (either synonym OK). In default mode, exact word match.
                pred_action = WORD_TO_ACTION_AW.get(pred, None)
                if pred_action == action:
                    correct_aw += 1
        aw_acc = correct_aw / max(total_aw, 1)
        print(f"\n  A->W Accuracy: {correct_aw}/{total_aw} = {aw_acc:.1%}", flush=True)
        print(f"  A->W Confusion: {confusion_aw}", flush=True)
        aw_result = {
            "n_trials": total_aw,
            "correct": correct_aw,
            "accuracy": aw_acc,
            "confusion_matrix": confusion_aw,
        }

    if args.out_stats:
        out = {
            "regime": "bio_three_factor" + ("_embodied" if args.embodied_hebbian else ""),
            "seed": args.seed,
            "training_stats": train_stats,
            "word_to_action_eval": wa_result,
            "action_to_word_eval": aw_result,
            "config": {
                "biological": args.biological,
                "n_lang_input": args.n_lang_input,
                "n_motor_per_action": args.n_motor_per_action,
                "learning_rate": args.learning_rate,
                "eligibility_decay_tau": args.eligibility_decay_tau,
                "apply_topographic_bias": args.apply_topographic_bias,
                "enable_motor_fs": args.enable_motor_fs,
                "enable_nmda": args.enable_nmda,
                "token_sparsity": args.token_sparsity,
                "embodied_hebbian": args.embodied_hebbian,
                "embodied_motor_teacher_pA": args.embodied_motor_teacher_pA,
                "orthogonal_cues": args.orthogonal_cues,
            },
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
