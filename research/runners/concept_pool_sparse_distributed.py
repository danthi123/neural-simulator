"""Sparse-distributed concept encoding (real catalog G.20 form).

REPLACES the contiguous-slice approximation in concept_pool_demo_shared.py
with the actual Pulvermüller / Kanerva form: each concept = sparse random
pattern in shared pool, NOT a contiguous block. Patterns OVERLAP between
concepts; Hamming distance separates them; capacity grows roughly
combinatorially.

Architecture:
  language_input (n_lang) -- plastic --> shared_concept_pool (n_pool)
                                ^
                                |  per-concept SPARSE pattern (K random
                                |  neurons of N total, K << N)
  shared_concept_pool -- plastic --> language_output (n_lang)
  shared_concept_pool <-> shared_FS (WTA)

Capacity comparison:
  Contiguous slices (current): N/K concepts max in pool of N neurons
    with K-sized slices. E.g. 1600 pool / 50 slice = 32 concepts.
    Going beyond requires bigger pool linearly.
  Sparse-distributed: C(N, K) patterns separable to Hamming distance d.
    For N=2000, K=100, d=20 (10% overlap tolerated):
    Roughly 10^150 separable patterns theoretically.
    Practical: ~500-2000 concepts before discrimination breaks.

Catalog refs:
  - G.20 Pulvermüller 1999: distributed cortical word ensembles
  - Kanerva 1988: sparse distributed memory
  - Foldiak 1990: sparse codes in cortex

Key training idea (same as concept_pool_demo_shared):
  Each concept gets a per-concept random SUBSET of n_pool neurons.
  Topographic prior boosts lang_input -> concept_subset weights.
  Training drives lang_input(N) + teacher current on concept_subset(N).
  Engram capture uses teacher-bias (validated to work).

Usage:
  python -m research.runners.concept_pool_sparse_distributed \\
      --seed 42 --n-concepts 64 --n-train-events 400 \\
      --n-lang-input 8192 --n-shared-pool 2000 \\
      --pattern-size 100 --top-k 150
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


def build_sparse_pool_bridge(
    seed: int,
    n_lang_input: int = 8192,
    n_shared_pool: int = 2000,
    n_shared_fs: int = 300,
    n_lang_output: int = 8192,
    verbose: bool = True,
):
    """Build a single shared-pool bridge (same as G.20 contiguous variant,
    just with bigger pool)."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway

    regions = [
        BrainRegion(name="language_input", n_neurons=n_lang_input,
                     exc_fraction=1.0, internal_density=0.0,
                     exc_weight_mean=0.0, inh_weight_mean=0.0,
                     weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="shared_concept_pool", n_neurons=n_shared_pool,
                     exc_fraction=0.8, internal_density=0.05,
                     exc_weight_mean=0.3, inh_weight_mean=0.8,
                     weight_jitter=0.2, plastic_internal=False),
        BrainRegion(name="shared_FS", n_neurons=n_shared_fs,
                     exc_fraction=0.0, internal_density=0.0,
                     exc_weight_mean=0.0, inh_weight_mean=0.0,
                     weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="language_output", n_neurons=n_lang_output,
                     exc_fraction=1.0, internal_density=0.0,
                     exc_weight_mean=0.0, inh_weight_mean=0.0,
                     weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [
        RegionPathway(from_region="language_input",
                       to_region="shared_concept_pool",
                       density=0.30, weight_mean=3.0, weight_jitter=0.5,
                       plastic=True,
                       plasticity_gate="language_input_to_shared"),
        RegionPathway(from_region="shared_concept_pool",
                       to_region="language_output",
                       density=0.30, weight_mean=2.0, weight_jitter=0.5,
                       plastic=True,
                       plasticity_gate="shared_to_language_output"),
        RegionPathway(from_region="shared_concept_pool",
                       to_region="shared_FS",
                       density=0.30, weight_mean=1.0, weight_jitter=0.2,
                       plastic=False),
        RegionPathway(from_region="shared_FS",
                       to_region="shared_concept_pool",
                       density=0.30, weight_mean=4.0, weight_jitter=0.2,
                       plastic=False),
    ]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = False
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 10.0
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
        print(f"[sparse-pool bridge] shared_pool={n_shared_pool}, "
              f"lang_input={n_lang_input}", flush=True)
    return bridge


def generate_sparse_patterns(n_concepts: int, n_pool: int, pattern_size: int,
                               seed: int) -> List[List[int]]:
    """Generate per-concept SPARSE RANDOM patterns in the shared pool.

    Each pattern = `pattern_size` random neurons from `n_pool`. Patterns
    overlap by chance; expected overlap between two random patterns is
    pattern_size² / n_pool ≈ 5 for K=100, N=2000.

    Uses a fixed RNG so patterns are reproducible.
    """
    rng = np.random.RandomState(seed * 17 + 19)  # stable but distinct
    patterns = []
    for _ in range(n_concepts):
        pat = sorted(rng.choice(n_pool, pattern_size, replace=False).tolist())
        patterns.append(pat)
    return patterns


def apply_sparse_topographic_prior(
    bridge, n_concepts: int, n_lang_input: int,
    sparse_patterns: List[List[int]],
    sparsity: float = 0.03,
    topographic_factor: float = 10.0,
    off_target_factor: float = 0.1,
    n_words_for_orthogonal: int = None,
    verbose: bool = True,
) -> Dict:
    """Apply topographic prior: for concept N, boost lang_input(N)'s active
    set's weights to concept N's sparse pattern, dampen to all OTHER
    concept patterns' EXCLUSIVE neurons (neurons that don't appear in
    pattern N).

    This is the sparse analog of the contiguous-slice prior."""
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    rm = bridge.region_manager

    if n_words_for_orthogonal is None:
        n_words_for_orthogonal = n_concepts

    lang_input_indices = list(rm.indices("language_input"))
    shared_indices = list(rm.indices("shared_concept_pool"))

    indptr = cp.asnumpy(bridge.cp_connections.indptr)
    indices = cp.asnumpy(bridge.cp_connections.indices)
    data = cp.asnumpy(bridge.cp_connections.data)

    # Pair lookup
    pair_to_idx: Dict[tuple, int] = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    boosted = 0
    dampened = 0
    pool_set = set(shared_indices)

    for cue_idx in range(n_concepts):
        # Active lang_input neurons for this concept
        drive = orthogonal_drive_pattern(
            cue_idx=cue_idx, n_cues=n_words_for_orthogonal,
            n_neurons=n_lang_input, drive_max_pA=1.0, sparsity=sparsity,
        )
        active_lang_local = np.where(drive > 0)[0]
        active_lang_global = [lang_input_indices[i]
                                for i in active_lang_local]
        # Target = concept's sparse pattern (mapped to global neuron ids)
        target_local = sparse_patterns[cue_idx]
        target_global = set(shared_indices[i] for i in target_local)
        # Off-target = pool neurons NOT in target pattern
        # (boost neurons that ARE in target; dampen the rest)
        off_target_global = pool_set - target_global

        for pre in active_lang_global:
            for post in target_global:
                key = (pre, post)
                if key in pair_to_idx:
                    off = pair_to_idx[key]
                    data[off] = float(data[off]) * topographic_factor
                    boosted += 1
            for post in off_target_global:
                key = (pre, post)
                if key in pair_to_idx:
                    off = pair_to_idx[key]
                    data[off] = float(data[off]) * off_target_factor
                    dampened += 1

    bridge.cp_connections.data = cp.asarray(data)
    if verbose:
        print(f"[sparse topographic prior] applied: {boosted} boosted, "
              f"{dampened} dampened on {n_concepts} sparse patterns",
              flush=True)
    return {"n_concepts": n_concepts, "boosted": boosted,
            "dampened": dampened,
            "pattern_size": len(sparse_patterns[0])}


def train_concept_sparse(
    bridge, word_idx: int, sparse_pattern: List[int],
    n_lang_input: int, n_lang_output: int, sparsity: float,
    n_words_for_orthogonal: int, teacher_pA: float = 500.0,
    lang_output_teacher_pA: float = 200.0,
):
    """Train one concept: drive lang_input + teacher on sparse_pattern
    + lang_output teacher."""
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    lang_out_arr = cp.asarray(list(rm.indices("language_output")),
                                dtype=cp.int64)
    shared_indices = list(rm.indices("shared_concept_pool"))
    pattern_global = [shared_indices[i] for i in sparse_pattern]
    pattern_arr = cp.asarray(pattern_global, dtype=cp.int64)

    drive_in = orthogonal_drive_pattern(
        cue_idx=word_idx, n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    drive_in_arr = cp.asarray(drive_in, dtype=cp.float32)
    drive_out = orthogonal_drive_pattern(
        cue_idx=word_idx, n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_output, drive_max_pA=lang_output_teacher_pA,
        sparsity=sparsity,
    )
    drive_out_arr = cp.asarray(drive_out, dtype=cp.float32)
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    ext.fill(0)
    ext[lang_arr] = drive_in_arr
    ext[lang_out_arr] = drive_out_arr
    ext[pattern_arr] = teacher_pA
    bridge.cp_external_input_current[:] = ext
    bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(10):
        bridge._run_one_simulation_step()


def eval_sparse_discrimination(
    bridge, words: List[str], sparse_patterns: List[List[int]],
    drive_pA: float = 1500.0, stim_steps: int = 100,
) -> List[Dict]:
    """For each engram-tagged concept, stim its tag and measure how much
    each concept's SPARSE PATTERN lights up. PASS if target pattern
    fires most strongly."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    shared_indices = list(rm.indices("shared_concept_pool"))
    pattern_arrs = [
        cp.asarray([shared_indices[i] for i in pat], dtype=cp.int64)
        for pat in sparse_patterns
    ]

    results = []
    for i, word in enumerate(words):
        bridge.stimulate_tag(word, drive_pA=drive_pA)
        pattern_rates = np.zeros(len(sparse_patterns), dtype=np.float32)
        for _ in range(stim_steps):
            bridge._run_one_simulation_step()
            for j, parr in enumerate(pattern_arrs):
                firing = bridge.cp_firing_states[parr]
                s = firing.sum() if hasattr(firing, 'sum') else 0
                if hasattr(s, 'item'):
                    s = s.item()
                pattern_rates[j] += float(s)
        bridge.clear_tag_drive(word)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()

        sorted_idx = np.argsort(-pattern_rates)
        rank = int(np.where(sorted_idx == i)[0][0]) + 1
        results.append({
            "word": word, "target_idx": i, "rank": rank,
            "top1": rank == 1, "top5": rank <= 5,
            "target_rate": float(pattern_rates[i]),
            "max_off_rate": float(max(pattern_rates[k]
                                       for k in range(len(words))
                                       if k != i)),
        })
    return results


# Same 60-word starter vocab as concept_pool_demo_shared
ALL_60 = [
    "apple", "river", "dog", "cat", "go", "come", "stop", "look",
    "big", "small", "hot", "cold",
    "tree", "bird", "sun", "moon", "walk", "run", "eat", "sleep",
    "red", "blue", "fast", "slow",
    "house", "road", "fire", "water", "give", "take", "find", "lose",
    "tall", "short", "wet", "dry",
    "person", "baby", "ball", "key", "open", "close", "push", "pull",
    "happy", "sad", "full", "empty",
    "food", "drink", "hand", "foot", "speak", "listen", "read", "write",
    "new", "old", "clean", "hard",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-concepts", type=int, default=64,
                    help="Number of concepts to test")
    p.add_argument("--n-train-events", type=int, default=400)
    p.add_argument("--n-lang-input", type=int, default=8192)
    p.add_argument("--n-shared-pool", type=int, default=2000,
                    help="Shared pool size for sparse patterns")
    p.add_argument("--n-shared-fs", type=int, default=300)
    p.add_argument("--pattern-size", type=int, default=100,
                    help="K (sparse pattern size per concept)")
    p.add_argument("--top-k", type=int, default=150,
                    help="Engram tag top-K size")
    p.add_argument("--sparsity", type=float, default=0.03)
    p.add_argument("--topographic-factor", type=float, default=10.0)
    p.add_argument("--off-target-factor", type=float, default=0.1)
    p.add_argument("--teacher-pA", type=float, default=500.0)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-bridge", type=str, default=None)
    p.add_argument("--vocab", type=str, default=None)
    args = p.parse_args()

    if args.vocab:
        vocab = [w.strip() for w in args.vocab.split(",") if w.strip()]
    else:
        # Use ALL_60 and replicate concepts beyond 60 with numbered suffix
        if args.n_concepts <= 60:
            vocab = ALL_60[:args.n_concepts]
        else:
            vocab = list(ALL_60)
            for i in range(60, args.n_concepts):
                vocab.append(f"concept{i}")
    if len(vocab) != args.n_concepts:
        args.n_concepts = len(vocab)

    print(f"=== concept_pool_sparse_distributed (seed={args.seed}, "
          f"n_concepts={args.n_concepts}) ===", flush=True)
    print(f"  Shared pool: {args.n_shared_pool} neurons", flush=True)
    print(f"  Pattern size: {args.pattern_size} per concept (sparse, random)",
          flush=True)
    print(f"  Substrate per concept: {args.pattern_size} neurons "
          f"(vs contiguous-slice's {args.n_shared_pool // args.n_concepts})",
          flush=True)
    print()

    # Build bridge
    t0 = time.time()
    bridge = build_sparse_pool_bridge(
        seed=args.seed, n_lang_input=args.n_lang_input,
        n_shared_pool=args.n_shared_pool,
        n_shared_fs=args.n_shared_fs, n_lang_output=args.n_lang_input,
    )
    print(f"[build] {time.time() - t0:.1f}s", flush=True)

    # Generate sparse patterns
    sparse_patterns = generate_sparse_patterns(
        n_concepts=args.n_concepts, n_pool=args.n_shared_pool,
        pattern_size=args.pattern_size, seed=args.seed,
    )
    # Estimate pairwise overlap
    if args.n_concepts > 1:
        overlap_sum = 0
        n_pairs = 0
        for i in range(min(20, args.n_concepts)):
            for j in range(i+1, min(20, args.n_concepts)):
                overlap = len(set(sparse_patterns[i]) & set(sparse_patterns[j]))
                overlap_sum += overlap
                n_pairs += 1
        mean_overlap = overlap_sum / max(n_pairs, 1)
        print(f"[sparse patterns] generated {args.n_concepts} patterns of "
              f"size {args.pattern_size}; mean pairwise overlap (first 20): "
              f"{mean_overlap:.1f} / {args.pattern_size}", flush=True)

    # Apply topographic prior
    t0 = time.time()
    prior_stats = apply_sparse_topographic_prior(
        bridge=bridge, n_concepts=args.n_concepts,
        n_lang_input=args.n_lang_input,
        sparse_patterns=sparse_patterns,
        sparsity=args.sparsity,
        topographic_factor=args.topographic_factor,
        off_target_factor=args.off_target_factor,
        n_words_for_orthogonal=args.n_concepts,
    )
    print(f"[topographic prior] {time.time() - t0:.1f}s", flush=True)

    # Open plasticity gates
    bridge.set_plasticity_gate("language_input_to_shared", 1.0)
    bridge.set_plasticity_gate("shared_to_language_output", 1.0)

    # Interleaved training
    print(f"\n[TRAIN] {args.n_concepts} x {args.n_train_events} = "
          f"{args.n_concepts * args.n_train_events} events", flush=True)
    t_train = time.time()
    np_rng = np.random.RandomState(args.seed)
    interleaved = []
    for _ in range(args.n_train_events):
        order = list(range(args.n_concepts))
        np_rng.shuffle(order)
        interleaved.extend(order)
    for evt_idx, i in enumerate(interleaved):
        train_concept_sparse(
            bridge=bridge, word_idx=i,
            sparse_pattern=sparse_patterns[i],
            n_lang_input=args.n_lang_input,
            n_lang_output=args.n_lang_input,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_concepts,
            teacher_pA=args.teacher_pA,
        )
        if (evt_idx + 1) % 500 == 0:
            print(f"  event {evt_idx + 1}/{len(interleaved)} "
                  f"({int(time.time() - t_train)}s)", flush=True)
    print(f"[TRAIN] {time.time() - t_train:.1f}s total", flush=True)

    # Freeze for capture
    bridge.set_plasticity_gate("language_input_to_shared", 0.0)
    bridge.set_plasticity_gate("shared_to_language_output", 0.0)

    # Capture engram tags with teacher-bias (the validated method)
    print(f"\n[ENGRAM] capturing {args.n_concepts} tags with teacher-bias",
          flush=True)
    t0 = time.time()
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")),
                            dtype=cp.int64)
    shared_indices = list(rm.indices("shared_concept_pool"))
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)

    for i, word in enumerate(vocab):
        drive = orthogonal_drive_pattern(
            cue_idx=i, n_cues=args.n_concepts,
            n_neurons=args.n_lang_input,
            drive_max_pA=200.0, sparsity=args.sparsity,
        )
        drive_arr = cp.asarray(drive, dtype=cp.float32)
        pattern_global = [shared_indices[k] for k in sparse_patterns[i]]
        pattern_arr = cp.asarray(pattern_global, dtype=cp.int64)

        bridge.start_engram_recording(word)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()
        for _ in range(100):  # 100-step capture
            ext.fill(0)
            ext[lang_arr] = drive_arr
            ext[pattern_arr] = 100.0  # teacher bias (validated)
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
        bridge.commit_engram_tag(
            word, top_k=args.top_k,
            region_filter=["shared_concept_pool"],
        )
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(10):
            bridge._run_one_simulation_step()
    print(f"[ENGRAM] {time.time() - t0:.1f}s", flush=True)

    # Eval
    print(f"\n[EVAL] sparse-pattern discrimination", flush=True)
    t0 = time.time()
    results = eval_sparse_discrimination(
        bridge=bridge, words=vocab,
        sparse_patterns=sparse_patterns,
        drive_pA=1500.0, stim_steps=args.drive_steps,
    )
    print(f"[EVAL] {time.time() - t0:.1f}s", flush=True)

    n_top1 = sum(1 for r in results if r["top1"])
    n_top5 = sum(1 for r in results if r["top5"])
    print(f"\n[RESULTS] {n_top1}/{args.n_concepts} top-1 "
          f"({100*n_top1/args.n_concepts:.1f}%), "
          f"{n_top5}/{args.n_concepts} top-5 "
          f"({100*n_top5/args.n_concepts:.1f}%)", flush=True)

    chance_top1 = 1.0 / args.n_concepts
    print(f"  chance top-1: {100*chance_top1:.1f}%, "
          f"observed/chance: {(n_top1/args.n_concepts)/chance_top1:.1f}x",
          flush=True)

    if args.save_bridge:
        bridge.save_checkpoint(args.save_bridge)
        print(f"[SAVE] -> {args.save_bridge}", flush=True)

    if args.out:
        out_data = {
            "seed": args.seed,
            "n_concepts": args.n_concepts,
            "n_train_events": args.n_train_events,
            "n_lang_input": args.n_lang_input,
            "n_shared_pool": args.n_shared_pool,
            "pattern_size": args.pattern_size,
            "top_k": args.top_k,
            "vocab": vocab,
            "sparse_patterns": sparse_patterns,
            "n_top1": n_top1, "n_top5": n_top5,
            "top1_pct": 100*n_top1/args.n_concepts,
            "top5_pct": 100*n_top5/args.n_concepts,
            "results": results,
            "prior_stats": prior_stats,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out_data, indent=2,
                                              default=str))
        print(f"[OUT] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
