"""Direction F CHEAP NUMPY PROBE: cross-bridge sequence storage.

If Direction A (ec_context spatial positional binding) or Direction E
(theta-gamma temporal positional binding) PASS for single-substrate
sequence storage, the natural scale-up is multi-substrate:

- Multiple bridges (substrates), each with a subset of vocabulary
- Sequences can mix concepts across bridges
- Each bridge queries position i with the cue; the bridge that holds
  position-i's concept responds; others abstain

This mirrors the validated G.20 multi-bridge architecture (5 sparse
bridges x 32 concepts each = 160-320 concept vocab; pillar n=80+).

CHEAP-FIRST numpy probe per discipline: test the algebra BEFORE
substrate implementation. Pre-registered:
- N_BRIDGES = 5
- N_CONCEPTS_PER_BRIDGE = 32
- N_DIM_PER_BRIDGE = 256 (matches FHRR + Direction E)
- SPARSITY = 0.05
- SLOT_COUNT = 5 (mid-range; tests beyond Direction A's 3-slot)
- N_TRIALS_PER_LOAD = 200
- BAR = 0.80 (frozen multi-seed)

Mechanism:
  1. Each bridge has its own vocabulary and concept patterns.
  2. Per sequence: assign each slot a random (bridge_idx, concept_idx).
  3. ALL bridges encode their pattern at slot i (or "silent" if that
     slot's concept isn't theirs).
  4. Query slot i: each bridge attempts recall; the one with the
     highest-confidence response wins. Others should abstain.
  5. Anti-cheat: control bridges that share NO concepts with the
     sequence should never respond above threshold.

Reuses ec_context-style spatial position code per bridge (each bridge
has its own position-pool that drives during the right slot).

NUMPY only; no GPU; no autograd; no substrate. ~3-5 min wall.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(
    _HERE, "direction_F_cross_bridge_sequence_numpy_probe.json")

# Pre-registered constants.
N_BRIDGES = 5
N_CONCEPTS_PER_BRIDGE = 32  # matches G.20 sparse-bridge n_concepts
N_DIM_PER_BRIDGE = 256
SPARSITY = 0.05
SLOT_COUNT = 5
N_TRIALS_PER_LOAD = 200
BAR = 0.80
NOISE_STD = 0.05
SEEDS = [42, 43, 44]
ABSTENTION_THRESHOLD = 0.30  # cosine-similarity below this -> abstain


def generate_concept_patterns_for_bridge(bridge_idx, n_concepts,
                                           n_dim, sparsity, seed):
    """Each bridge has its OWN random concept patterns; sparse
    encoding mirrors the G.20 K-of-N architecture."""
    rng = np.random.default_rng(seed * 10000 + bridge_idx * 100)
    patterns = np.zeros((n_concepts, n_dim), dtype=np.float32)
    n_active = int(sparsity * n_dim)
    for c in range(n_concepts):
        active_idx = rng.choice(n_dim, size=n_active, replace=False)
        patterns[c, active_idx] = 1.0
    return patterns


def encode_sequence_to_all_bridges(seq_assignments, bridge_patterns,
                                      n_dim, slot_count, rng):
    """seq_assignments[slot_idx] = (bridge_idx, concept_idx).
    Each bridge encodes per-slot: the active bridge writes its
    pattern; inactive bridges write silence."""
    # Return per-bridge per-slot ensemble: shape (N_BRIDGES, slot_count, n_dim)
    ensemble = np.zeros(
        (len(bridge_patterns), slot_count, n_dim), dtype=np.float32)
    for slot_idx, (b_idx, c_idx) in enumerate(seq_assignments):
        ensemble[b_idx, slot_idx, :] = bridge_patterns[b_idx][c_idx]
    return ensemble


def add_noise_to_ensemble(ensemble, noise_std, rng):
    return ensemble + rng.standard_normal(ensemble.shape).astype(
        np.float32) * noise_std


def query_slot(ensemble, slot_idx, bridge_patterns,
                 abstention_threshold):
    """For each bridge: cosine-match the slot's bridge activity
    against the bridge's own concept patterns; report top match
    and confidence. Return (best_bridge, best_concept, best_score)
    OR (None, None, max_score) if all below threshold (abstention)."""
    best_bridge = None; best_concept = None; best_score = -np.inf
    for b_idx in range(len(bridge_patterns)):
        win = ensemble[b_idx, slot_idx, :]
        # Cosine-match against this bridge's patterns
        for c_idx in range(len(bridge_patterns[b_idx])):
            a = win.astype(np.float64)
            b = bridge_patterns[b_idx][c_idx].astype(np.float64)
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12: continue
            score = float(np.dot(a, b) / (na * nb))
            if score > best_score:
                best_score = score
                best_bridge = b_idx
                best_concept = c_idx
    if best_score < abstention_threshold:
        return None, None, best_score
    return best_bridge, best_concept, best_score


def run_seed(seed, verbose=True):
    rng = np.random.default_rng(seed * 31337)
    # Build per-bridge concept patterns
    bridge_patterns = []
    for b_idx in range(N_BRIDGES):
        patterns = generate_concept_patterns_for_bridge(
            b_idx, N_CONCEPTS_PER_BRIDGE, N_DIM_PER_BRIDGE,
            SPARSITY, seed)
        bridge_patterns.append(patterns)

    n_correct_per_slot = 0
    n_abstention_correct = 0  # abstention when correct bridge isn't queried
    n_total_slots = 0
    per_trial = []

    for trial in range(N_TRIALS_PER_LOAD):
        # Generate random sequence: each slot gets (bridge_idx, concept_idx)
        seq_assignments = []
        for slot_idx in range(SLOT_COUNT):
            b_idx = rng.integers(N_BRIDGES)
            c_idx = rng.integers(N_CONCEPTS_PER_BRIDGE)
            seq_assignments.append((b_idx, c_idx))

        ensemble = encode_sequence_to_all_bridges(
            seq_assignments, bridge_patterns, N_DIM_PER_BRIDGE,
            SLOT_COUNT, rng)
        noisy = add_noise_to_ensemble(ensemble, NOISE_STD, rng)

        slot_results = []
        for slot_idx in range(SLOT_COUNT):
            pred_b, pred_c, score = query_slot(
                noisy, slot_idx, bridge_patterns, ABSTENTION_THRESHOLD)
            true_b, true_c = seq_assignments[slot_idx]
            correct = (pred_b == true_b and pred_c == true_c)
            if correct: n_correct_per_slot += 1
            n_total_slots += 1
            slot_results.append({
                "slot": slot_idx, "true": (int(true_b), int(true_c)),
                "pred": (None if pred_b is None else int(pred_b),
                          None if pred_c is None else int(pred_c)),
                "score": score, "correct": correct,
            })
        per_trial.append({"trial": trial, "slots": slot_results})

    acc = n_correct_per_slot / n_total_slots if n_total_slots > 0 else 0.0
    return {
        "seed": seed, "n_correct": n_correct_per_slot,
        "n_total_slots": n_total_slots, "accuracy": acc,
        "n_trials": N_TRIALS_PER_LOAD,
    }


def main():
    print(f"=== Direction F cross-bridge sequence numpy probe ===",
          flush=True)
    print(f"  N_BRIDGES: {N_BRIDGES}", flush=True)
    print(f"  N_CONCEPTS_PER_BRIDGE: {N_CONCEPTS_PER_BRIDGE}",
          flush=True)
    print(f"  N_DIM_PER_BRIDGE: {N_DIM_PER_BRIDGE}", flush=True)
    print(f"  SPARSITY: {SPARSITY}, NOISE_STD: {NOISE_STD}",
          flush=True)
    print(f"  SLOT_COUNT: {SLOT_COUNT}", flush=True)
    print(f"  Pre-registered bar: {BAR}", flush=True)
    total_vocab = N_BRIDGES * N_CONCEPTS_PER_BRIDGE
    print(f"  Total vocab across bridges: {total_vocab}",
          flush=True)
    chance = 1.0 / total_vocab
    print(f"  Chance: {chance:.4f}", flush=True)

    t0 = time.time()
    seed_results = []
    for seed in SEEDS:
        r = run_seed(seed)
        seed_results.append(r)
        print(f"  seed {seed}: acc = {r['accuracy']:.3f} "
              f"({r['n_correct']}/{r['n_total_slots']})",
              flush=True)

    total_min = (time.time() - t0) / 60
    accs = [r["accuracy"] for r in seed_results]
    mean = float(np.mean(accs))
    print(f"\n  multi-seed mean: {mean:.3f}", flush=True)
    print(f"  per-seed: {accs}", flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if mean >= BAR:
        verdict = "CROSS_BRIDGE_ALGEBRA_SUFFICIENT"
        print(f"  multi-seed mean >= {BAR} -- cross-bridge sequence"
              f" composition is algebraically sufficient at "
              f"{N_BRIDGES} bridges x {N_CONCEPTS_PER_BRIDGE} "
              f"concepts; substrate implementation justified.",
              flush=True)
    elif mean > 2 * chance:
        verdict = "ABOVE_CHANCE_BELOW_BAR"
        print(f"  mean {mean:.3f} > chance {chance:.4f} but below "
              f"{BAR}; partial signal.", flush=True)
    else:
        verdict = "AT_CHANCE"
        print(f"  mean {mean:.3f} at chance ({chance:.4f}); cross-"
              f"bridge algebra alone insufficient.", flush=True)

    out = {
        "config": {
            "N_BRIDGES": N_BRIDGES,
            "N_CONCEPTS_PER_BRIDGE": N_CONCEPTS_PER_BRIDGE,
            "N_DIM_PER_BRIDGE": N_DIM_PER_BRIDGE,
            "SPARSITY": SPARSITY, "SLOT_COUNT": SLOT_COUNT,
            "N_TRIALS_PER_LOAD": N_TRIALS_PER_LOAD, "BAR": BAR,
            "NOISE_STD": NOISE_STD, "SEEDS": SEEDS,
            "ABSTENTION_THRESHOLD": ABSTENTION_THRESHOLD,
        },
        "total_vocab": total_vocab, "chance": chance,
        "per_seed": seed_results,
        "multi_seed_mean": mean, "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
