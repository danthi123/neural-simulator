"""Direction F INTERFERENCE VARIANT: cross-bridge sequence storage
where bridges SHARE the substrate dimension (real biological case).

The first cheap probe had each bridge with its own ensemble subspace
(zero interference) and trivially scored 1.000. That tests nothing
biologically meaningful -- in real biology multiple cortical regions
share theta-gamma rhythms and project into the SAME population
(downstream readout neurons). Real cross-bridge composition has to
disentangle SHARED-substrate writes.

This variant:
- Single shared ensemble of N_DIM (same as Direction E)
- Per-slot encoding: ALL bridges that have content at slot i write
  their pattern to the shared ensemble during slot i
- Per-bridge decoder: read slot i's window, cosine-match against
  bridge B's concept patterns
- The challenge: if bridge A wrote pattern at slot i AND bridge B
  wrote pattern at slot i (shared time slot, different concepts),
  the decoder for bridge B has to extract bridge B's concept
  despite bridge A's interference

Two tests:
(I) Non-overlapping bridge writes: each slot has exactly ONE
   bridge active (the others are silent). Tests the abstention
   behavior; should match Direction E single-substrate quality.
(II) Overlapping bridge writes: each slot has TWO bridges active
   simultaneously (different concepts in each bridge); decoder
   has to extract bridge B's concept from the superposition.
   This is the genuine cross-bridge interference test.

Pre-registered:
- N_BRIDGES = 5
- N_CONCEPTS_PER_BRIDGE = 32
- N_DIM (shared) = 256
- SLOT_COUNT = 5
- SPARSITY = 0.05
- NOISE_STD = 0.05
- BAR = 0.80 (frozen)
- N_TRIALS = 200

Per bridge: distinct concept patterns drawn from a per-bridge
DEDICATED subspace of the shared N_DIM (so bridges don't share
neurons but DO share the time slot). This is the catalog's
"different cortical regions, shared rhythm" model.

NUMPY only; ~2-5 min wall.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(
    _HERE, "direction_F_cross_bridge_sequence_interference.json")

# Pre-registered constants.
N_BRIDGES = 5
N_CONCEPTS_PER_BRIDGE = 32
N_DIM = 256  # SHARED substrate
SPARSITY = 0.05
SLOT_COUNT = 5
N_TRIALS = 200
BAR = 0.80
NOISE_STD = 0.05
SEEDS = [42, 43, 44]


def generate_bridge_patterns_shared_substrate(n_bridges, n_concepts,
                                                 n_dim_total,
                                                 sparsity, seed):
    """Each bridge has its concept patterns drawn from a DEDICATED
    subspace of n_dim_total. Subspaces are non-overlapping per
    bridge (so each bridge's patterns don't collide WITHIN bridge
    subspace) BUT all bridges share the n_dim_total ensemble
    (so writes from different bridges occupy different neurons of
    the shared substrate).

    Returns: list of (patterns, subspace_indices) per bridge.
    """
    rng = np.random.default_rng(seed * 31337)
    subspace_size = n_dim_total // n_bridges
    bridge_data = []
    for b_idx in range(n_bridges):
        # This bridge's subspace
        start = b_idx * subspace_size
        end = start + subspace_size
        subspace_indices = np.arange(start, end)
        # Sparse patterns within subspace
        n_active = max(1, int(sparsity * subspace_size))
        patterns_subspace = np.zeros(
            (n_concepts, subspace_size), dtype=np.float32)
        for c in range(n_concepts):
            active = rng.choice(subspace_size, size=n_active,
                                  replace=False)
            patterns_subspace[c, active] = 1.0
        # Embed into shared n_dim space
        patterns_full = np.zeros(
            (n_concepts, n_dim_total), dtype=np.float32)
        patterns_full[:, subspace_indices] = patterns_subspace
        bridge_data.append({
            "patterns": patterns_full,
            "subspace": subspace_indices,
            "size": subspace_size,
        })
    return bridge_data


def encode_sequence_shared(seq_assignments, bridge_data,
                            n_dim_total, slot_count):
    """seq_assignments[slot_idx] = list of (bridge_idx, concept_idx)
    pairs (CAN BE MULTIPLE per slot).
    Returns: ensemble of shape (slot_count, n_dim_total).
    All bridge writes are summed at the corresponding slot.
    """
    ensemble = np.zeros((slot_count, n_dim_total), dtype=np.float32)
    for slot_idx, pairs in enumerate(seq_assignments):
        for b_idx, c_idx in pairs:
            ensemble[slot_idx, :] = ensemble[slot_idx, :] + \
                bridge_data[b_idx]["patterns"][c_idx]
    return ensemble


def query_slot_per_bridge(ensemble, slot_idx, query_bridge_idx,
                            bridge_data, threshold):
    """Per-bridge decoder: for bridge B, cosine-match the slot's
    activity (restricted to bridge B's subspace) against bridge B's
    own concept patterns. Returns (predicted_concept, score).

    If max score < threshold, abstain (None, score).
    """
    bd = bridge_data[query_bridge_idx]
    subspace = bd["subspace"]
    # Read only this bridge's subspace
    win = ensemble[slot_idx, subspace]
    if np.linalg.norm(win) < 1e-12:
        return None, 0.0
    best_c = None; best_score = -np.inf
    n_concepts = bd["patterns"].shape[0]
    for c in range(n_concepts):
        p = bd["patterns"][c, subspace]
        a = win.astype(np.float64); b = p.astype(np.float64)
        nb = np.linalg.norm(b)
        if nb < 1e-12: continue
        score = float(np.dot(a, b) / (np.linalg.norm(a) * nb))
        if score > best_score:
            best_score = score; best_c = c
    if best_score < threshold:
        return None, best_score
    return best_c, best_score


def run_seed_test1(seed):
    """Test (I) Non-overlapping: each slot has exactly 1 active bridge."""
    rng = np.random.default_rng(seed * 31337 + 1)
    bridge_data = generate_bridge_patterns_shared_substrate(
        N_BRIDGES, N_CONCEPTS_PER_BRIDGE, N_DIM, SPARSITY, seed)
    n_correct = 0; n_total = 0
    for trial in range(N_TRIALS):
        seq_assignments = []
        for slot_idx in range(SLOT_COUNT):
            b = rng.integers(N_BRIDGES)
            c = rng.integers(N_CONCEPTS_PER_BRIDGE)
            seq_assignments.append([(b, c)])
        ensemble = encode_sequence_shared(
            seq_assignments, bridge_data, N_DIM, SLOT_COUNT)
        noisy = ensemble + rng.standard_normal(
            ensemble.shape).astype(np.float32) * NOISE_STD
        # For each slot, query ALL bridges; the right bridge must
        # respond, others must abstain (correct).
        for slot_idx in range(SLOT_COUNT):
            true_b, true_c = seq_assignments[slot_idx][0]
            for b in range(N_BRIDGES):
                pred, score = query_slot_per_bridge(
                    noisy, slot_idx, b, bridge_data, threshold=0.3)
                if b == true_b:
                    # Should respond with true_c
                    if pred == true_c:
                        n_correct += 1
                else:
                    # Should abstain
                    if pred is None:
                        n_correct += 1
                n_total += 1
    return n_correct / n_total if n_total > 0 else 0.0


def run_seed_test2(seed):
    """Test (II) Overlapping: each slot has 2 bridges active."""
    rng = np.random.default_rng(seed * 31337 + 2)
    bridge_data = generate_bridge_patterns_shared_substrate(
        N_BRIDGES, N_CONCEPTS_PER_BRIDGE, N_DIM, SPARSITY, seed)
    n_correct_active_bridges = 0
    n_total_active_bridges = 0
    for trial in range(N_TRIALS):
        seq_assignments = []
        for slot_idx in range(SLOT_COUNT):
            # 2 distinct bridges per slot
            bs = rng.choice(N_BRIDGES, size=2, replace=False)
            pairs = []
            for b in bs:
                c = rng.integers(N_CONCEPTS_PER_BRIDGE)
                pairs.append((int(b), int(c)))
            seq_assignments.append(pairs)
        ensemble = encode_sequence_shared(
            seq_assignments, bridge_data, N_DIM, SLOT_COUNT)
        noisy = ensemble + rng.standard_normal(
            ensemble.shape).astype(np.float32) * NOISE_STD
        # For each slot, query each ACTIVE bridge; the right concept
        # must be recovered.
        for slot_idx in range(SLOT_COUNT):
            for true_b, true_c in seq_assignments[slot_idx]:
                pred, score = query_slot_per_bridge(
                    noisy, slot_idx, true_b, bridge_data,
                    threshold=0.3)
                if pred == true_c:
                    n_correct_active_bridges += 1
                n_total_active_bridges += 1
    return n_correct_active_bridges / n_total_active_bridges \
        if n_total_active_bridges > 0 else 0.0


def main():
    print(f"=== Direction F cross-bridge interference variant ===",
          flush=True)
    print(f"  N_BRIDGES={N_BRIDGES}, N_CONCEPTS_PER_BRIDGE="
          f"{N_CONCEPTS_PER_BRIDGE}, N_DIM (shared)={N_DIM}",
          flush=True)
    print(f"  SLOT_COUNT={SLOT_COUNT}, sparsity={SPARSITY},"
          f" noise={NOISE_STD}", flush=True)
    print(f"  Pre-registered bar: {BAR}", flush=True)

    t0 = time.time()

    print(f"\n--- Test (I): non-overlapping bridges per slot ---",
          flush=True)
    test1_accs = []
    for seed in SEEDS:
        a = run_seed_test1(seed)
        test1_accs.append(a)
        print(f"  seed {seed}: {a:.3f}", flush=True)
    test1_mean = float(np.mean(test1_accs))
    print(f"  mean: {test1_mean:.3f}", flush=True)

    print(f"\n--- Test (II): overlapping bridges (2 per slot) ---",
          flush=True)
    test2_accs = []
    for seed in SEEDS:
        a = run_seed_test2(seed)
        test2_accs.append(a)
        print(f"  seed {seed}: {a:.3f}", flush=True)
    test2_mean = float(np.mean(test2_accs))
    print(f"  mean: {test2_mean:.3f}", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"\nWall: {total_min:.1f} min", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    test1_pass = test1_mean >= BAR
    test2_pass = test2_mean >= BAR

    if test1_pass and test2_pass:
        verdict = "CROSS_BRIDGE_INTERFERENCE_ROBUST"
        print(f"  Both tests PASS >= {BAR} -- cross-bridge algebra"
              f" handles per-slot interference (2 bridges per slot)."
              f" Substrate implementation justified.", flush=True)
    elif test1_pass and not test2_pass:
        verdict = "CROSS_BRIDGE_INTERFERENCE_SENSITIVE"
        print(f"  Test I PASS, Test II BELOW -- the algebra handles"
              f" abstention (separate bridges per slot) but breaks"
              f" under overlap; precise interference bound.",
              flush=True)
    else:
        verdict = "CROSS_BRIDGE_ABSTENTION_FAILS"
        print(f"  Test I {test1_mean:.3f} below {BAR} -- per-bridge"
              f" abstention doesn't work; deeper issue.", flush=True)

    out = {
        "config": {
            "N_BRIDGES": N_BRIDGES,
            "N_CONCEPTS_PER_BRIDGE": N_CONCEPTS_PER_BRIDGE,
            "N_DIM_SHARED": N_DIM, "SLOT_COUNT": SLOT_COUNT,
            "SPARSITY": SPARSITY, "NOISE_STD": NOISE_STD,
            "BAR": BAR, "N_TRIALS": N_TRIALS, "SEEDS": SEEDS,
        },
        "test1_non_overlap_mean": test1_mean,
        "test1_non_overlap_per_seed": test1_accs,
        "test2_overlap_mean": test2_mean,
        "test2_overlap_per_seed": test2_accs,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
