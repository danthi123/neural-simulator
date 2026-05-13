"""concept_pool_demo — Phase 1 of the concepts/composition/diversity arc.

User directive 2026-05-12: "those scaling axes are 100% what need to be given
our full focus currently, as the blocker for reaching conversational
capabilities... it needs concepts, composition, and diversity."

Diagnosis: every conversational blocker (P5 abstract concepts at 2/4 ceiling,
Tier 2.3 compositional grammar stuck at 34-40%, in-vivo new-vocab binding
at 2/4 fixed capacity) shares ONE root cause — only 4 motor pools.

This runner demonstrates the architectural fix: add dedicated noun pools
(APPLE/RIVER/DOG/CAT) and verb pools (GO/COME), each following the proven
Tier 1 recipe (500-neuron pool + paired teacher current + FS cross-
inhibition + reciprocal lang_output). The result is 10 distinct output
categories — 2.5x diversity over the current 4-pool ceiling.

Phase 1 (this runner): validate cross-category isolation.
    typing "north" -> motor_N fires (existing Tier 1 capability)
    typing "apple" -> noun_pool_APPLE fires (NEW)
    typing "go"    -> verb_pool_GO fires (NEW)
    typing "apple" -> noun_pool_RIVER stays silent (isolation)
    typing "apple" -> motor_N stays silent (cross-kind isolation)

Phase 2 (concept_compose_demo, future): composition.
    "go" then "north" -> verb_pool_GO + motor_N BOTH fire (NMDA bistability)
    "apple" then "north" -> noun_pool_APPLE + motor_N BOTH fire

Phase 3 (concept_speak_demo, future): A->W readout
    drive noun_pool_APPLE -> language_output produces "apple" pattern

Architecture: mirrors bio_three_factor's training loop but extended to
train all three pool kinds (motor/noun/verb) on dedicated vocabularies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Vocab ────────────────────────────────────────────────────────────
# Direction words (existing Tier 1) — bind to motor pools
DIRECTION_VOCAB: Dict[str, str] = {
    "north": "N", "east": "E", "south": "S", "west": "W",
}

# Noun words (NEW) — bind to dedicated noun pools
NOUN_VOCAB: Dict[str, str] = {
    "apple": "APPLE", "river": "RIVER", "dog": "DOG", "cat": "CAT",
}

# Verb words (NEW) — bind to dedicated verb pools.
# 2026-05-13 v2: expanded from 2 to 4 to match noun/motor pool count.
# v1 seed 42 result showed verb_pool_COME structurally dominating all
# 10 words because 2 verb pools means each verb_FS has only 1 cross-
# inhibition edge (vs 3 for 4-pool kinds). Adding STOP and LOOK gives
# verb FS the same topology as noun/motor FS (3 cross-edges per FS).
VERB_VOCAB: Dict[str, str] = {
    "go": "GO", "come": "COME", "stop": "STOP", "look": "LOOK",
}

NOUN_NAMES = list(NOUN_VOCAB.values())   # ["APPLE", "RIVER", "DOG", "CAT"]
VERB_NAMES = list(VERB_VOCAB.values())   # ["GO", "COME"]
MOTOR_NAMES = ["N", "E", "S", "W"]

# 3rd concept kind (opt-in via --enable-adjective):
# Adjective words (NEW kind) — bind to dedicated adjective pools.
ADJECTIVE_VOCAB: Dict[str, str] = {
    "big": "BIG", "small": "SMALL", "hot": "HOT", "cold": "COLD",
}
ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())   # ["BIG","SMALL","HOT","COLD"]


def build_concept_bridge(seed: int,
                          n_lang_input: int = 4096,
                          n_per_pool: int = 500,
                          n_fs_per_pool: int = 60,
                          enable_adjective: bool = False,
                          weak_dynamics: bool = False,
                          verbose: bool = True):
    """Construct a bridge with motor + noun + verb (+ optional adjective) pools.

    Mirrors bio_three_factor's biological config: NMDA bistability,
    motor FS cross-inhibition, embodied Hebbian training. Pools follow
    the Tier 1 recipe (500 neurons, exc 0.8, internal 0.10).

    enable_adjective=True adds 4 additional pools (BIG/SMALL/HOT/COLD)
    -> 14 distinct output categories (3.5x diversity over Tier 1).
    """
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import build_biological_brain_regions

    # Per v2c finding: canon dynamics (0.10/2.0/4.0) amplify structural
    # bias at biological scale with 12 pools (same as P5 iter KK). Weak
    # dynamics (iter AA recipe) prevent off-target pool self-sustaining.
    concept_internal_density = 0.05 if weak_dynamics else None
    concept_exc_weight = 0.3 if weak_dynamics else None
    concept_inh_weight = 0.8 if weak_dynamics else None

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        # NEW: enable noun + verb pools (Tier 1 recipe per kind)
        enable_noun_pools=True,
        noun_pool_names=NOUN_NAMES,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=True,
        verb_pool_names=VERB_NAMES,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        # Optional 3rd kind: adjectives
        enable_adjective_pools=enable_adjective,
        adjective_pool_names=ADJECTIVE_NAMES if enable_adjective else None,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        # Per-kind dynamics (weak when --weak-concept-dynamics)
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0  # Above design weights to avoid soft-bound collapse
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

    if verbose:
        rm = bridge.region_manager
        all_pools = (
            [f"motor_{a}" for a in MOTOR_NAMES]
            + [f"noun_pool_{n}" for n in NOUN_NAMES]
            + [f"verb_pool_{v}" for v in VERB_NAMES]
        )
        n_pool_neurons = sum(len(list(rm.indices(p))) for p in all_pools)
        n_total = int(getattr(cfg, "num_neurons", 0)) or sum(
            r.n_neurons for r in cfg.brain_regions
        )
        print(f"[BUILD] concept-pool bridge: {n_total} neurons total, "
              f"{len(all_pools)} concept pools ({n_pool_neurons} neurons in pools)",
              flush=True)

    return bridge


def apply_concept_topographic_bias(bridge,
                                     n_lang_input: int = 4096,
                                     topographic_factor: float = 2.0,
                                     off_target_factor: float = 0.5,
                                     sparsity: float = 0.1,
                                     verbose: bool = True) -> Dict:
    """Apply Pulvermüller-style topographic bias to lang_input -> {pool}.

    Extends apply_topographic_bias to cover noun_pool_X and verb_pool_X
    pathways. For each word w with active language_input neuron set A_w:
        weights[A_w -> pool_target(w)] *= topographic_factor
        weights[A_w -> pool_other]      *= off_target_factor

    With default 1.5/0.7 the ratio is ~2.1x (within Pulvermüller's
    reported 2-3x biology range). This is the proven recipe that got
    Tier 1 to 6/6 multi-seed PASS.
    """
    import numpy as np
    from sim.text_embeddings import vocab_to_drive_pattern
    from sim.backend import get_backend
    cp, _ = get_backend()

    def _to_host(arr):
        try:
            return cp.asnumpy(arr)
        except Exception:
            return np.asarray(arr)

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    if len(lang_input_indices) != n_lang_input:
        raise ValueError(
            f"apply_concept_topographic_bias: bridge has "
            f"{len(lang_input_indices)} lang_input neurons but caller "
            f"specified {n_lang_input}"
        )

    indptr = _to_host(bridge.cp_connections.indptr)
    indices = _to_host(bridge.cp_connections.indices)
    data = _to_host(bridge.cp_connections.data)

    # Pre-compute (pre, post) -> data offset for fast lookup
    pair_to_idx: Dict[Tuple[int, int], int] = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    summary: Dict[str, Dict] = {}

    # Build (word, target_pool_region, peers) tuples.
    #
    # FIX 2026-05-13 (post-v4 weight probe): peers must include ALL
    # output pools across ALL kinds, not just within kind. Otherwise
    # cross-kind edges keep their random init (~3.0) while within-kind
    # off-target edges get dampened to ~0.9. Result: cross-kind pools
    # win as max-off after training (observed in v4 weight probe).
    #
    # Now: every word dampens ALL non-target pool edges (cross-kind +
    # within-kind off-target). Only target-pool edges get the boost.
    rm_existing = bridge.region_manager
    bias_tasks: List[Tuple[str, str, List[str]]] = []

    # Discover all output pools that exist in the bridge
    all_output_pools = [f"motor_{a}" for a in MOTOR_NAMES]
    all_output_pools += [f"noun_pool_{n}" for n in NOUN_NAMES]
    all_output_pools += [f"verb_pool_{v}" for v in VERB_NAMES]
    has_adjective = False
    try:
        rm_existing.indices(f"adjective_pool_{ADJECTIVE_NAMES[0]}")
        has_adjective = True
        all_output_pools += [f"adjective_pool_{n}" for n in ADJECTIVE_NAMES]
    except Exception:
        pass

    for word, action in DIRECTION_VOCAB.items():
        target = f"motor_{action}"
        bias_tasks.append((word, target, all_output_pools))
    for word, name in NOUN_VOCAB.items():
        target = f"noun_pool_{name}"
        bias_tasks.append((word, target, all_output_pools))
    for word, name in VERB_VOCAB.items():
        target = f"verb_pool_{name}"
        bias_tasks.append((word, target, all_output_pools))
    if has_adjective:
        for word, name in ADJECTIVE_VOCAB.items():
            target = f"adjective_pool_{name}"
            bias_tasks.append((word, target, all_output_pools))

    # 2026-05-13 v7 FIX: priority-based assignment with TARGET-FIRST.
    #
    # Earlier multiplicative approach caused cumulative dampening when
    # a single edge was "target" for one word but "off-target" for 11
    # others (overlap between word drive patterns ~10%). Result with
    # 0.3 dampening factor: 3.0 boost × 0.3^11 = ~5e-6 effective.
    # Killed motor pool target firing in v6 (target_rate dropped from
    # 1.2 to 0.7).
    #
    # New approach: two-pass priority.
    #   Pass 1 (target): for each (word, target_pool), boost edges
    #          from word-active lang_input to target_pool. Track in
    #          a set so they're protected from pass 2.
    #   Pass 2 (off-target): for each (word, off-target_pool), dampen
    #          edges NOT in the target set.
    # An edge that is "target" for ANY word never gets dampened.
    # An edge that is "off-target" for AT LEAST ONE word gets dampened
    # exactly ONCE.

    # Step 1: pre-compute global_active per word
    word_to_active = {}
    for word, _, _ in bias_tasks:
        if word in word_to_active:
            continue
        drive = vocab_to_drive_pattern(
            word, n_neurons=n_lang_input, sparsity=sparsity
        )
        local_active = np.where(drive > 0)[0]
        word_to_active[word] = [lang_input_indices[i] for i in local_active]

    # Pass 1: identify and boost all target edges
    target_edges = set()
    for word, target_region, _ in bias_tasks:
        peer_neurons = list(rm.indices(target_region))
        global_active = word_to_active[word]
        n_changed = 0
        for src in global_active:
            for dst in peer_neurons:
                key = (src, dst)
                if key in pair_to_idx:
                    if key not in target_edges:
                        # First time touching this edge as target
                        idx = pair_to_idx[key]
                        data[idx] = float(data[idx]) * topographic_factor
                        target_edges.add(key)
                        n_changed += 1
        summary[f"{word}->{target_region}"] = {
            "factor": topographic_factor, "edges_modified": n_changed,
        }

    # Pass 2: dampen off-target edges (skip any in target_edges)
    dampened_edges = set()
    for word, target_region, peer_regions in bias_tasks:
        global_active = word_to_active[word]
        for peer in peer_regions:
            if peer == target_region:
                continue
            peer_neurons = list(rm.indices(peer))
            n_changed = 0
            for src in global_active:
                for dst in peer_neurons:
                    key = (src, dst)
                    if (key in pair_to_idx
                            and key not in target_edges
                            and key not in dampened_edges):
                        idx = pair_to_idx[key]
                        data[idx] = float(data[idx]) * off_target_factor
                        dampened_edges.add(key)
                        n_changed += 1
            summary[f"{word}->{peer}(off)"] = {
                "factor": off_target_factor, "edges_modified": n_changed,
            }

    bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)

    if verbose:
        print(f"[topographic-bias] Applied factor={topographic_factor:.2f}/"
              f"{off_target_factor:.2f} across motor + noun + verb pools",
              flush=True)
        # Print only target boosts for brevity (off-target counts are similar)
        for k, v in summary.items():
            if v["factor"] == topographic_factor:
                print(f"  {k}: x{v['factor']:.2f} on {v['edges_modified']} edges")

    return summary


def train_word_to_pool(bridge, word: str, target_pool_region: str,
                        n_events: int = 200,
                        stim_steps_per_event: int = 100,
                        reset_steps: int = 50,  # 25ms; NMDA tau ~150ms
                        drive_pA: float = 200.0,
                        teacher_pA: float = 1500.0,
                        sparsity: float = 0.1,
                        n_lang_input: int = 4096,
                        n_lang_output: int = 4096,
                        embodied_hebbian: bool = True,
                        verbose: bool = False) -> Dict:
    """Train a single word -> pool binding via paired teacher current.

    reset_steps gotcha (2026-05-13 design note): NMDA tau is ~150ms;
    50-step reset (25ms) doesn't fully decay NMDA activation between
    events. If you observe pool dominance unrelated to the trained
    target, try reset_steps=300 (~150ms) to let NMDA fully decay
    between events. Trade-off: 3x training wall clock.
    """
    """Train ONE word to fire ONE specific pool via embodied Hebbian.

    Tier 1's proven recipe: drive language_input with word pattern,
    drive target pool with strong teacher current, step bridge for
    STDP to fire on co-active synapses. With embodied_hebbian=True,
    also drive language_output with same pattern (Pulvermüller premotor
    co-firing for reciprocal A->W readout).

    target_pool_region: e.g. "motor_N", "noun_pool_APPLE", "verb_pool_GO"
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    pool_idx = list(rm.indices(target_pool_region))
    pool_arr = cp.asarray(pool_idx, dtype=cp.int64)

    has_output = False
    if embodied_hebbian:
        try:
            lang_output_idx = list(rm.indices("language_output"))
            lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)
            has_output = True
        except Exception:
            pass

    drive_in = vocab_to_drive_pattern(
        word, n_neurons=n_lang_input,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_in_gpu = cp.asarray(drive_in, dtype=cp.float32)
    if has_output:
        drive_out = vocab_to_drive_pattern(
            word, n_neurons=n_lang_output,
            drive_max_pA=drive_pA, sparsity=sparsity,
        )
        drive_out_gpu = cp.asarray(drive_out, dtype=cp.float32)

    # Open ONLY the target kind's gates during training. v1/v2 bug:
    # opening all 6 gates let off-target pathways (e.g., lang_input ->
    # noun_pool during direction training) accumulate STDP whenever
    # off-target pools fired by chance. Over 200 events x 12 words,
    # this added structural bias to whichever pool had random initial
    # over-firing.
    #
    # Fix: identify target kind, open only its 2 gates (in, out). All
    # other pathways stay frozen during this word's training. Each
    # word's training is now ISOLATED to its target pathway.
    if target_pool_region.startswith("motor_"):
        target_kind = "motor"
    elif target_pool_region.startswith("noun_pool_"):
        target_kind = "noun_pool"
    elif target_pool_region.startswith("verb_pool_"):
        target_kind = "verb_pool"
    elif target_pool_region.startswith("adjective_pool_"):
        target_kind = "adjective_pool"
    else:
        raise ValueError(f"unknown target pool kind: {target_pool_region}")

    gates_to_open = [
        f"language_input_to_{target_kind}",
        f"{target_kind}_to_language_output",
    ]
    gates_opened = []
    for g in gates_to_open:
        try:
            bridge.set_plasticity_gate(g, 1.0)
            gates_opened.append(g)
        except Exception:
            pass

    try:
        for ev in range(n_events):
            # Reset
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            # Drive lang_input + (optionally lang_output) + target pool
            bridge.cp_external_input_current[lang_input_arr] = drive_in_gpu
            if has_output:
                bridge.cp_external_input_current[lang_output_arr] = drive_out_gpu
            bridge.cp_external_input_current[pool_arr] += float(teacher_pA)

            for _ in range(stim_steps_per_event):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            if verbose and (ev + 1) % 50 == 0:
                print(f"  [train '{word}' -> {target_pool_region}] "
                      f"{ev+1}/{n_events}", flush=True)
    finally:
        for g in gates_opened:
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass

    return {"word": word, "target": target_pool_region,
            "n_events": n_events, "gates_opened": gates_opened}


def measure_pool_firing(bridge, word: str,
                          all_pool_regions: List[str],
                          stim_steps: int = 100,
                          reset_steps: int = 50,
                          drive_pA: float = 200.0,
                          sparsity: float = 0.1,
                          n_lang_input: int = 4096) -> Dict[str, float]:
    """Drive lang_input(word) without any teacher, measure spike counts
    across ALL listed pool regions. Returns per-pool spike count.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)

    drive_in = vocab_to_drive_pattern(
        word, n_neurons=n_lang_input,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_in_gpu = cp.asarray(drive_in, dtype=cp.float32)

    # Reset
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Drive lang_input only — no teacher current on any pool
    bridge.cp_external_input_current[lang_input_arr] = drive_in_gpu

    # Accumulate spike counts across stim window per pool
    per_pool_indices = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                        for p in all_pool_regions}
    per_pool_count = {p: 0.0 for p in all_pool_regions}

    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for p, idx_arr in per_pool_indices.items():
            count = int(fired[idx_arr].sum())
            per_pool_count[p] += count

    # Convert to per-neuron mean firing
    per_pool_rate = {}
    for p in all_pool_regions:
        n_neurons = len(list(rm.indices(p)))
        per_pool_rate[p] = per_pool_count[p] / max(n_neurons, 1)

    return per_pool_rate


def run_concept_pool_demo(seed: int = 42,
                            n_train_events: int = 200,
                            n_lang_input: int = 4096,
                            n_per_pool: int = 500,
                            n_fs_per_pool: int = 60,
                            apply_topographic: bool = True,
                            topographic_factor: float = 2.0,
                            off_target_factor: float = 0.5,
                            enable_adjective: bool = False,
                            interleaved: bool = False,
                            weak_dynamics: bool = False,
                            reset_steps: int = 50,  # 25ms (NMDA tau is ~150ms)
                            verbose: bool = True,
                            load_bridge: str = None,
                            save_bridge: str = None) -> Dict:
    """Train motor + noun + verb pools, then measure cross-category isolation.

    If load_bridge is given, skip training and load from checkpoint.
    If save_bridge is given (and we DID train), save after training so
    subsequent eval iterations don't need to retrain.
    """
    print(f"\n=== concept_pool_demo (seed={seed}) ===", flush=True)
    n_motor = 4
    n_noun = len(NOUN_VOCAB)
    n_verb = len(VERB_VOCAB)
    n_adj = len(ADJECTIVE_VOCAB) if enable_adjective else 0
    n_pools = n_motor + n_noun + n_verb + n_adj
    parts = [f"{n_motor} motor", f"{n_noun} noun", f"{n_verb} verb"]
    if enable_adjective:
        parts.append(f"{n_adj} adjective")
    print(f"  Architecture: {' + '.join(parts)} = {n_pools} pools", flush=True)
    vocab_str = (
        f"{list(DIRECTION_VOCAB)} + {list(NOUN_VOCAB)} + {list(VERB_VOCAB)}"
    )
    if enable_adjective:
        vocab_str += f" + {list(ADJECTIVE_VOCAB)}"
    print(f"  Vocab: {vocab_str}", flush=True)
    print(f"  Train events/word: {n_train_events}", flush=True)

    t0 = time.time()
    bridge = build_concept_bridge(
        seed=seed,
        n_lang_input=n_lang_input,
        n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool,
        enable_adjective=enable_adjective,
        weak_dynamics=weak_dynamics,
        verbose=verbose,
    )

    # Handle load: load checkpoint AFTER building bridge skeleton
    if load_bridge:
        print(f"\n[LOAD] loading checkpoint from {load_bridge}", flush=True)
        bridge.load_checkpoint(load_bridge)
        # Re-freeze all plasticity gates after load (load may have reopened them)
        for g in ("language_input_to_motor",
                  "language_input_to_noun_pool",
                  "language_input_to_verb_pool",
                  "motor_to_language_output",
                  "noun_pool_to_language_output",
                  "verb_pool_to_language_output"):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass

    if apply_topographic and not load_bridge:
        # Only apply topographic bias for fresh training (would overwrite
        # learned weights if applied to loaded bridge)
        apply_concept_topographic_bias(
            bridge,
            n_lang_input=n_lang_input,
            topographic_factor=topographic_factor,
            off_target_factor=off_target_factor,
            verbose=verbose,
        )

    # Build training schedule: shuffle across all (word, target_pool) pairs
    import numpy as np
    rng = np.random.default_rng(seed)
    all_targets = []  # list of (word, target_pool_region)
    for word, action in DIRECTION_VOCAB.items():
        all_targets.append((word, f"motor_{action}"))
    for word, name in NOUN_VOCAB.items():
        all_targets.append((word, f"noun_pool_{name}"))
    for word, name in VERB_VOCAB.items():
        all_targets.append((word, f"verb_pool_{name}"))
    if enable_adjective:
        for word, name in ADJECTIVE_VOCAB.items():
            all_targets.append((word, f"adjective_pool_{name}"))

    if load_bridge:
        print(f"\n[TRAIN] SKIPPED (loaded from checkpoint)", flush=True)
        train_sec = 0.0
    else:
        total_events = len(all_targets) * n_train_events
        mode_str = "interleaved" if interleaved else "sequential"
        print(f"\n[TRAIN] {len(all_targets)} (word, pool) pairs, "
              f"{n_train_events} events each = {total_events} total events "
              f"({mode_str})", flush=True)

        t_train = time.time()
        if interleaved:
            # Interleaved training: shuffle event order across all words.
            # Matches bio_three_factor's pattern. Prevents one pool from
            # dominating during long uninterrupted same-word training.
            # Per-event cost is higher (gate switching per event), so this
            # is slower than sequential but may yield cleaner discrimination
            # at biological scale.
            buffer = []
            for word, target in all_targets:
                for _ in range(n_train_events):
                    buffer.append((word, target))
            rng.shuffle(buffer)
            for ev_idx, (word, target) in enumerate(buffer):
                train_word_to_pool(
                    bridge, word, target,
                    n_events=1,
                    reset_steps=reset_steps,
                    n_lang_input=n_lang_input,
                    n_lang_output=n_lang_input,
                    verbose=False,
                )
                if (ev_idx + 1) % 100 == 0:
                    print(f"  interleaved event {ev_idx + 1}/{total_events}"
                          f" ({time.time() - t_train:.0f}s)", flush=True)
        else:
            # Sequential: train all events for word_1, then word_2, etc.
            for word, target in all_targets:
                t_word = time.time()
                train_word_to_pool(
                    bridge, word, target,
                    n_events=n_train_events,
                    reset_steps=reset_steps,
                    n_lang_input=n_lang_input,
                    n_lang_output=n_lang_input,
                    verbose=False,
                )
                print(f"  trained '{word}' -> {target} "
                      f"({time.time() - t_word:.0f}s)", flush=True)
        train_sec = time.time() - t_train
        print(f"\n[TRAIN] complete ({train_sec:.0f}s, {mode_str})",
              flush=True)

        if save_bridge:
            print(f"\n[SAVE] saving bridge to {save_bridge}", flush=True)
            from pathlib import Path
            Path(save_bridge).parent.mkdir(parents=True, exist_ok=True)
            bridge.save_checkpoint(save_bridge)
            print(f"[SAVE] checkpoint written", flush=True)

    # Measure cross-category isolation
    all_pool_regions = (
        [f"motor_{a}" for a in MOTOR_NAMES]
        + [f"noun_pool_{n}" for n in NOUN_NAMES]
        + [f"verb_pool_{v}" for v in VERB_NAMES]
    )
    if enable_adjective:
        all_pool_regions += [f"adjective_pool_{n}" for n in ADJECTIVE_NAMES]

    print(f"\n[EVAL] measuring cross-category isolation across "
          f"{len(all_pool_regions)} pools", flush=True)

    eval_results = {}
    for word, target in all_targets:
        per_pool = measure_pool_firing(
            bridge, word, all_pool_regions,
            n_lang_input=n_lang_input,
        )
        eval_results[word] = {
            "target": target,
            "per_pool": per_pool,
            "target_rate": per_pool[target],
            "max_off_target": max(v for k, v in per_pool.items() if k != target),
            "max_off_target_pool": max(
                (k for k in per_pool if k != target),
                key=lambda k: per_pool[k],
            ),
        }

    # Compute pass/fail per word — target rate must be highest
    print(f"\n[RESULTS] per-word cross-category isolation:", flush=True)
    n_pass = 0
    for word, res in eval_results.items():
        target_rate = res["target_rate"]
        max_off = res["max_off_target"]
        ratio = target_rate / max(max_off, 0.001)
        passed = target_rate > max_off
        if passed:
            n_pass += 1
        marker = "PASS" if passed else "FAIL"
        print(f"  {word:8s} -> {res['target']:20s}  "
              f"target={target_rate:.3f}  "
              f"max_off={max_off:.3f} ({res['max_off_target_pool']})  "
              f"ratio={ratio:.2f}x  [{marker}]", flush=True)

    print(f"\n[VERDICT] {n_pass}/{len(eval_results)} words have correct "
          f"target-pool dominance", flush=True)
    print(f"  Total wall clock: {time.time() - t0:.0f}s", flush=True)

    return {
        "seed": seed,
        "n_train_events": n_train_events,
        "n_lang_input": n_lang_input,
        "n_per_pool": n_per_pool,
        "n_fs_per_pool": n_fs_per_pool,
        "apply_topographic": apply_topographic,
        "n_pools": len(all_pool_regions),
        "n_pass": n_pass,
        "n_words": len(eval_results),
        "wall_clock_s": time.time() - t0,
        "results": {
            word: {
                "target": res["target"],
                "target_rate": float(res["target_rate"]),
                "max_off_target": float(res["max_off_target"]),
                "max_off_target_pool": res["max_off_target_pool"],
                "per_pool": {k: float(v) for k, v in res["per_pool"].items()},
            }
            for word, res in eval_results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 concept pool demo (concepts + diversity).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-events", type=int, default=200,
                         help="Events per word (Tier 1 default 200)")
    parser.add_argument("--n-lang-input", type=int, default=4096)
    parser.add_argument("--n-per-pool", type=int, default=500)
    parser.add_argument("--n-fs-per-pool", type=int, default=60)
    parser.add_argument("--no-topographic", action="store_true",
                         help="Skip Pulvermuller topographic bias init")
    parser.add_argument("--topographic-factor", type=float, default=2.0,
                         help="Multiplier for target pool weights "
                         "(default 2.0). v3 stronger prior: try 3.0.")
    parser.add_argument("--off-target-factor", type=float, default=0.5,
                         help="Multiplier for off-target pool weights "
                         "(default 0.5). v3 stronger prior: try 0.3.")
    parser.add_argument("--enable-adjective", action="store_true",
                         help="Add 4 adjective pools (BIG/SMALL/HOT/COLD); "
                         "14 total output categories")
    parser.add_argument("--interleaved", action="store_true",
                         help="Use interleaved training (shuffled event "
                         "order) instead of sequential word-by-word. "
                         "Matches bio_three_factor Tier 1 pattern.")
    parser.add_argument("--weak-concept-dynamics", action="store_true",
                         help="Use weak dynamics for concept pools "
                         "(0.05/0.3/0.8 instead of canon 0.10/2.0/4.0). "
                         "Per P5 iter AA + v2c finding: canon amplifies "
                         "structural bias at biological scale with many "
                         "pools. Weak prevents off-target pools from "
                         "accumulating activated states.")
    parser.add_argument("--reset-steps", type=int, default=50,
                         help="Steps to free-run between training events "
                         "(default 50 = 25ms). For v3 NMDA-decay fix: "
                         "use 300 (~150ms tau). Trade-off: 3x training time.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output JSON path (default stdout only)")
    parser.add_argument("--load-bridge", type=str, default=None,
                         help="Load bridge checkpoint instead of training "
                         "(fast iteration on eval mechanics)")
    parser.add_argument("--save-bridge", type=str, default=None,
                         help="Save trained bridge to this checkpoint for "
                         "reuse via --load-bridge")
    args = parser.parse_args()

    result = run_concept_pool_demo(
        seed=args.seed,
        n_train_events=args.n_train_events,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        apply_topographic=not args.no_topographic,
        topographic_factor=args.topographic_factor,
        off_target_factor=args.off_target_factor,
        enable_adjective=args.enable_adjective,
        interleaved=args.interleaved,
        weak_dynamics=args.weak_concept_dynamics,
        reset_steps=args.reset_steps,
        load_bridge=args.load_bridge,
        save_bridge=args.save_bridge,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n[OUT] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
