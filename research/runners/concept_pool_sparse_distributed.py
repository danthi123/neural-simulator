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
    lang_output_wta: bool = False,
    lang_output_wta_feedforward: bool = False,
    verbose: bool = True,
):
    """Build a single shared-pool bridge (same as G.20 contiguous variant,
    just with bigger pool).

    lang_output_wta (default False = byte-identical): add a `language_output_FS` WTA pool (feedback E->I->E,
    mirroring shared_FS) so the language_output competes -> during A->W read-out TRAINING only the strongly-driven
    word band survives -> the shared_pool->language_output STDP binds each pattern to ITS band cleanly (fixes the
    non-word-specific read-out smearing, 2026-07-09-sparse-aw-speak-crosstalk-boundary). Additive; off = unchanged.

    lang_output_wta_feedforward (CYCLE-1098, default False): the E%-max version -- the language_output_FS is driven
    by the AFFERENT (shared_concept_pool), NOT by language_output's own output. So the inhibition threshold is set
    by the pattern VOLLEY before language_output can fire broadly (de Almeida-Idiart-Lisman feedforward divisive
    normalization) -> keeps the read-out output sparse DURING training -> STDP stays word-specific, fixing the a0-
    diagnosed positive-feedback read-out BROADENING (read-out grows -> pool drives broad langout -> STDP grows
    broader -> the 'more training = worse' divergence). The feedback WTA (lang_output_wta) is the saturate-or-dead
    knife-edge that over-suppressed (2026-07-09 crosstalk finding); this is the calibrated feedforward fix."""
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

    if lang_output_wta or lang_output_wta_feedforward:
        regions.append(BrainRegion(name="language_output_FS", n_neurons=max(60, n_lang_output // 20),
                                   exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0,
                                   inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
        # E%-max FEEDFORWARD: the FS is driven by the AFFERENT (shared_concept_pool) so inhibition tracks the pattern
        # VOLLEY (de Almeida-Idiart-Lisman); FEEDBACK (default): driven by language_output's own output (knife-edge).
        _fs_src = "shared_concept_pool" if lang_output_wta_feedforward else "language_output"
        pathways.append(RegionPathway(from_region=_fs_src, to_region="language_output_FS",
                                      density=0.30, weight_mean=1.0, weight_jitter=0.2, plastic=False))
        pathways.append(RegionPathway(from_region="language_output_FS", to_region="language_output",
                                      density=0.30, weight_mean=4.0, weight_jitter=0.2, plastic=False))

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
                               seed: int, composed_vocab: int = 0,
                               composed_bind: bool = False, zipf_s: float = 0.0,
                               freq_adaptive: bool = False,
                               parts_out: List[List[List[int]]] = None) -> List[List[int]]:
    """Generate per-concept SPARSE RANDOM patterns in the shared pool.

    Each pattern = `pattern_size` random neurons from `n_pool`. Patterns
    overlap by chance; expected overlap between two random patterns is
    pattern_size² / n_pool ≈ 5 for K=100, N=2000.

    Uses a fixed RNG so patterns are reproducible.
    """
    rng = np.random.RandomState(seed * 17 + 19)  # stable but distinct
    if composed_vocab and int(composed_vocab) > 0:
        # COMPOSED-FACT MODE (2026-07-29, additive; composed_vocab=0 => byte-identical to the shipped path).
        # Independent random patterns overlap only BY CHANCE (~5 neurons for K=100,N=2000). A composed FACT
        # does not: it is built from constituents SHARED with other facts (SVO triples over one vocabulary),
        # so facts overlap STRUCTURALLY and by a lot. That is the case consolidation actually needs and the
        # one the banked concept result never tested — storing 64 unrelated concepts says nothing about
        # storing 64 facts that share their words.
        V = int(composed_vocab)
        max_facts = V * (V - 1) * (V - 2) // 6
        if n_concepts > max_facts:
            raise ValueError(
                "composed_vocab=%d yields only C(%d,3)=%d distinct facts < n_concepts=%d; raise --composed-vocab"
                % (V, V, max_facts, n_concepts))
        per = max(1, pattern_size // 3)
        vocab = [np.asarray(sorted(rng.choice(n_pool, per, replace=False).tolist())) for _ in range(V)]
        # ROLE BINDING (2026-07-29, composed_bind=False => the UNION baseline, unchanged).
        # A UNION code's overlap is DEFINITIONAL: two facts sharing a word share those neurons, so
        # composed recall collapses (measured 0.583 at N=200 vs 1.000 for independent patterns). Applying
        # a fixed random PERMUTATION per ROLE before combining makes a shared word land on DIFFERENT
        # neurons depending on the role it fills, which breaks that identity -- the VSA/FHRR role-filler
        # trick the project's own composer already implements. Measured off-bridge: overlap 14.2 -> 8.9
        # (-37%) and N=200 recall 0.583 -> 0.840.
        perms = [rng.permutation(n_pool) for _ in range(3)] if composed_bind else None
        # ZIPFIAN word frequency (zipf_s=0 => uniform, the prior behaviour). Real language reuses frequent
        # words heavily, which drives up exactly the constituent sharing that causes collisions. Measured:
        # a UNIFORM gate passes at ~1.0 while the same config under classic Zipf (s=1.0) falls to 0.606
        # full / 0.393 partial at V=320/N=500 — so a uniform-sampled gate certifies a memory that degrades
        # by a third in use. FREQ_ADAPTIVE spends conjunctive budget in proportion to constituent
        # frequency (the inverse-frequency/PPMI principle): full-cue 0.597 -> 0.959 at N=500.
        if zipf_s > 0:
            _w = 1.0 / np.power(np.arange(1, V + 1), float(zipf_s))
            _w = _w / _w.sum()
        else:
            _w = None
        seen, patterns = set(), []
        _tries = 0
        while len(patterns) < n_concepts:
            _tries += 1
            if _tries > n_concepts * 400:
                raise ValueError("could not draw %d distinct facts from V=%d at zipf_s=%s "
                                 "(frequent words dominate); raise --composed-vocab or lower --zipf-s"
                                 % (n_concepts, V, zipf_s))
            tri = tuple(sorted(rng.choice(V, 3, replace=False, p=_w).tolist()))
            if tri in seen:
                continue
            seen.add(tri)
            if perms is None:
                pat = set(vocab[tri[0]].tolist()) | set(vocab[tri[1]].tolist()) | set(vocab[tri[2]].tolist())
            else:
                pat = set(perms[0][vocab[tri[0]]].tolist()) | set(perms[1][vocab[tri[1]]].tolist()) \
                    | set(perms[2][vocab[tri[2]]].tolist())
            if freq_adaptive and _w is not None:
                # spend conjunctive budget where the collisions actually are
                _f = float(np.mean([_w[i] for i in tri]) / _w[0])
                _a = float(np.clip(0.75 * np.sqrt(_f), 0.0, 0.6))
                _nc = int(round(_a * pattern_size))
                if _nc > 0:
                    _keep = sorted(pat)[:max(0, pattern_size - _nc)]
                    _cr = np.random.RandomState(abs(hash(tri)) % (2 ** 32))
                    pat = set(_keep) | set(_cr.choice(n_pool, _nc, replace=False).tolist())
            if parts_out is not None:
                # record each constituent's OWN pool neurons, so a PARTIAL cue (2 of 3) can be driven.
                # Needed because the shipped eval cues a whole engram tag and therefore only ever tests
                # FULL-cue recall — the conversationally decisive query mode was untestable on-bridge.
                if perms is None:
                    _pp = [sorted(vocab[tri[j]].tolist()) for j in range(3)]
                else:
                    _pp = [sorted(perms[j][vocab[tri[j]]].tolist()) for j in range(3)]
                parts_out.append(_pp)
            patterns.append(sorted(pat))
        return patterns
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
    n_total = int(bridge.cp_connections.shape[0])

    # GPU-VECTORIZED implementation (2026-05-15 perf fix).
    #
    # Previous version pulled the whole CSR to host, built a Python dict
    # of every (pre,post) -> offset (~8M entries), then did
    # n_concepts x active_lang x pool_size Python dict lookups
    # (73M+ ops at 256 concepts / 5000 pool). That preprocessing step
    # dominated wall-clock (CPU-bound, NOT GPU-accelerated).
    #
    # New version keeps the CSR data + indices on device. For each
    # concept it builds a per-postsynaptic-neuron multiplier array on
    # GPU (1.0 default, off_target_factor for pool neurons, then
    # topographic_factor for the concept's sparse-pattern neurons),
    # then scales each active lang row's outgoing edges via a single
    # vectorized gather+multiply on GPU. indptr is pulled to host ONCE
    # (small, ~n_total ints) so row-slice bounds are pure host
    # arithmetic with no per-row GPU sync.
    indptr_host = cp.asnumpy(bridge.cp_connections.indptr)
    indices_dev = bridge.cp_connections.indices       # stays on GPU
    data_dev = bridge.cp_connections.data             # modified in place

    shared_arr = cp.asarray(shared_indices, dtype=cp.int64)

    boosted = 0
    dampened = 0

    for cue_idx in range(n_concepts):
        drive = orthogonal_drive_pattern(
            cue_idx=cue_idx, n_cues=n_words_for_orthogonal,
            n_neurons=n_lang_input, drive_max_pA=1.0, sparsity=sparsity,
        )
        active_lang_local = np.where(drive > 0)[0]
        active_lang_global = [lang_input_indices[i]
                                for i in active_lang_local]

        # Per-concept multiplier over ALL postsynaptic neurons (GPU).
        mult = cp.ones(n_total, dtype=cp.float32)
        mult[shared_arr] = off_target_factor          # all pool dampened
        target_global = cp.asarray(
            [shared_indices[i] for i in sparse_patterns[cue_idx]],
            dtype=cp.int64)
        mult[target_global] = topographic_factor      # pattern boosted

        # For each active lang row: scale outgoing edges by mult[col].
        for pre in active_lang_global:
            s = int(indptr_host[pre])
            e = int(indptr_host[pre + 1])
            if e <= s:
                continue
            cols = indices_dev[s:e]                    # GPU slice
            data_dev[s:e] *= mult[cols]                # GPU gather+mul
        # Structural accounting (no GPU sync): each active row touches
        # pattern_size boosted edges + pool_size-pattern_size dampened
        # (upper bound; exact count would require a GPU reduction).
        n_active = len(active_lang_global)
        boosted += n_active * len(sparse_patterns[cue_idx])
        dampened += n_active * (len(shared_indices)
                                 - len(sparse_patterns[cue_idx]))

    bridge.cp_connections.data = data_dev
    if verbose:
        print(f"[sparse topographic prior] GPU-vectorized; "
              f"~{boosted} boost-targets, ~{dampened} dampen-targets "
              f"across {n_concepts} sparse patterns "
              f"(upper bounds; actual = intersection with existing edges)",
              flush=True)
    return {"n_concepts": n_concepts, "boosted_upper": boosted,
            "dampened_upper": dampened,
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


def eval_partial_cue_discrimination(
    bridge, sparse_patterns: List[List[int]], parts: List[List[List[int]]],
    drive_pA: float = 1500.0, stim_steps: int = 60, hold_roles=(0, 1), mode: str = "sum"):
    """PARTIAL-CUE recall on-bridge: drive only roles 0+1 of a fact and ask which stored fact completes.

    The shipped `eval_sparse_discrimination` stimulates a whole engram TAG, so it can only ever measure
    FULL-cue recall. Conversation queries partially ("what did the dog eat?" gives agent+action, not the
    answer), and off-substrate that mode sits at 75-100% of an information ceiling — a number this harness
    could not check at all. This drives the union of the cued roles' OWN pool neurons directly and reads
    which fact's full pattern lights up most.

    Reports `ambiguous_frac` alongside accuracy: when several stored facts share the cued roles, NO code can
    disambiguate them, so a miss there is an information limit, not a mechanism failure. Scoring without
    that split would understate the memory.
    """
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager
    shared = list(rm.indices("shared_concept_pool"))
    pat_arrs = [np.asarray([shared[i] for i in p], dtype=np.int64) for p in sparse_patterns]
    # which facts share the cued roles -> genuinely ambiguous, no code can resolve them
    keys = [tuple(tuple(parts[i][r]) for r in hold_roles) for i in range(len(parts))]
    from collections import Counter
    kc = Counter(keys)
    correct = unique_total = unique_correct = 0
    def _drive(cue_pool):
        idx = cp.asarray([shared[j] for j in cue_pool], dtype=cp.int64)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx] = float(drive_pA)
        a = np.zeros(len(sparse_patterns))
        for _ in range(int(stim_steps)):
            bridge._run_one_simulation_step()
            fs = np.asarray(to_host(bridge.cp_firing_states)).ravel()
            for f in range(len(pat_arrs)):
                a[f] += float(fs[pat_arrs[f]].sum())
        bridge.cp_external_input_current[:] = 0.0
        return a

    for i in range(len(sparse_patterns)):
        if mode == "min":
            # ROLE-CONSISTENT completion: drive each cued role SEPARATELY and require a fact to respond to
            # BOTH. Summing lets a competitor that matches ONE role strongly out-score the correct fact
            # that matches both moderately -- off-substrate this cost 0.19-0.21 accuracy on answerable
            # queries. Conjunctive gating (coincidence / dendritic AND) is a same-family primitive here.
            per_role = [_drive(sorted(set(parts[i][r]))) for r in hold_roles]
            acc = np.minimum.reduce(per_role)
        else:
            acc = _drive(sorted(set().union(*[set(parts[i][r]) for r in hold_roles])))
        hit = int(np.argmax(acc)) == i if acc.sum() > 0 else False
        correct += int(hit)
        if kc[keys[i]] == 1:
            unique_total += 1
            unique_correct += int(hit)
    n = len(sparse_patterns)
    return {"partial_cue_acc": round(correct / n, 4),
            "ambiguous_frac": round(1.0 - unique_total / n, 4),
            "acc_on_unambiguous": round(unique_correct / unique_total, 4) if unique_total else None,
            "n": n, "hold_roles": list(hold_roles), "mode": mode}


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
    p.add_argument("--composed-vocab", type=int, default=0,
                   help="COMPOSED-FACT mode: each item is an SVO-style triple over a shared vocabulary of this size, so items overlap STRUCTURALLY (0 = shipped independent-random behaviour).")
    p.add_argument("--partial-cue-mode", choices=["sum","min"], default="sum",
                   help="min = ROLE-CONSISTENT completion (drive each cued role separately, require a fact to respond to BOTH); off-substrate this gains +0.13 to +0.21 on answerable queries.")
    p.add_argument("--eval-partial-cue", action="store_true",
                   help="ALSO measure PARTIAL-cue recall on-bridge (drive 2 of 3 roles, ask which fact completes) — the conversationally decisive query mode the shipped eval cannot test.")
    p.add_argument("--zipf-s", type=float, default=0.0,
                   help="Zipfian word frequency exponent for composed facts (0 = uniform; 1.0 = classic Zipf = realistic). A uniform gate over-reports by ~a third.")
    p.add_argument("--freq-adaptive", action="store_true",
                   help="spend conjunctive code budget in proportion to constituent frequency (inverse-frequency/PPMI principle); recovers full-cue capacity under Zipf.")
    p.add_argument("--composed-bind", action="store_true",
                   help="ROLE BINDING: permute each filler by its role before combining, so a shared word lands on different neurons per role (off = UNION baseline).")
    p.add_argument("--topographic-factor", type=float, default=10.0)
    p.add_argument("--off-target-factor", type=float, default=0.1)
    p.add_argument("--teacher-pA", type=float, default=500.0)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-bridge", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None,
                   help="checkpoint to RESUME training from (accumulate more --n-train-events on top of "
                        "prior training). Loads the trained weights instead of re-applying the from-scratch "
                        "topographic prior. MUST use the same --seed and --vocab so the sparse patterns "
                        "match. Enables incremental training across breaks (each chunk a fresh fast process "
                        "-> no CuPy fragmentation from one long run).")
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
    _parts = [] if (args.composed_vocab and args.eval_partial_cue) else None
    sparse_patterns = generate_sparse_patterns(
        n_concepts=args.n_concepts, n_pool=args.n_shared_pool,
        pattern_size=args.pattern_size, seed=args.seed,
        composed_vocab=args.composed_vocab, composed_bind=args.composed_bind,
        zipf_s=args.zipf_s, freq_adaptive=args.freq_adaptive, parts_out=_parts,
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

    # Apply topographic prior (from-scratch init) OR resume from a prior checkpoint (accumulate training).
    # Resume loads the trained weights so the new --n-train-events accumulate ON TOP of prior training,
    # enabling incremental training across breaks (each chunk a fresh fast process -> no CuPy fragmentation).
    t0 = time.time()
    if args.resume_from:
        bridge.load_checkpoint(args.resume_from)
        prior_stats = {"resumed_from": args.resume_from}
        print(f"[resume] loaded {args.resume_from}; accumulating +{args.n_train_events} events/concept on top "
              f"of prior weights [{time.time() - t0:.1f}s]", flush=True)
    else:
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
    if getattr(args, "eval_partial_cue", False) and _parts:
        print("\n[EVAL] PARTIAL-cue recall (2 of 3 roles) — the query mode conversation actually uses", flush=True)
        _pc = eval_partial_cue_discrimination(bridge, sparse_patterns, _parts, mode=args.partial_cue_mode)
        print("  partial_cue_acc=%s  ambiguous_frac=%s  acc_on_unambiguous=%s"
              % (_pc["partial_cue_acc"], _pc["ambiguous_frac"], _pc["acc_on_unambiguous"]), flush=True)
        results["partial_cue"] = _pc
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
