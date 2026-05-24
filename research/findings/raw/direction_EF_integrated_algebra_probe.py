"""Direction E+F INTEGRATED algebra probe: theta-gamma + multi-bridge
+ familiarity-gate together (the algebra version of what substrate
cross-bridge sequence storage would do).

Setup:
- N_BRIDGES bridges (each with own subspace of shared N_DIM)
- N_CONCEPTS_PER_BRIDGE concepts per bridge
- Each theta cycle has N_GAMMA gamma slots
- Per-slot encoding: ALL bridges with content at gamma slot i write
  their pattern to the shared ensemble during gamma slot i
- Per-slot query: stim ec_context(slot i) AND ask each bridge for
  its top concept at that slot's gamma window
- Familiarity gate: per-bridge norm-ratio decides abstain vs respond

This is the full conversational-primitive algebra: K-slot sequence
+ cross-bridge composition + abstention + theta-gamma multiplexing.
If the algebra clears the 0.80 bar at substrate-realistic scale,
the substrate implementation has a justified target.

Pre-registered:
- N_BRIDGES = 5
- N_CONCEPTS_PER_BRIDGE = 32 (matches G.20)
- N_DIM = 256 (shared substrate dimension)
- N_GAMMA = 7 (catalog cap)
- GAMMA_PERIOD = 17 (samples per gamma slot)
- N_THETA = 119 (7 * 17 samples per theta cycle)
- SLOT_COUNT = 5 (mid-range; tests interference)
- BRIDGES_PER_SLOT = 2 (cross-bridge composition; each slot has 2
  bridges active in parallel)
- N_TRIALS = 200
- BAR = 0.80 multi-seed (frozen)
- NOISE_STD = 0.05
- THRESHOLD_NORM_RATIO = 1.5 (familiarity gate; Direction F fix)

Total vocab across bridges: 5 * 32 = 160 (the G.20 single-ensemble
vocab cap; "age-5" target in the project's standing roadmap).

NUMPY only; ~10-30 s wall.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(
    _HERE, "direction_EF_integrated_algebra_probe.json")

# Pre-registered constants.
N_BRIDGES = 5
N_CONCEPTS_PER_BRIDGE = 32
N_DIM = 256
N_GAMMA = 7
GAMMA_PERIOD = 17
N_THETA = N_GAMMA * GAMMA_PERIOD  # 119
SLOT_COUNT = 5
BRIDGES_PER_SLOT = 2
N_TRIALS = 200
BAR = 0.80
NOISE_STD = 0.05
SPARSITY = 0.05
SEEDS = [42, 43, 44]
THRESHOLD_NORM_RATIO = 1.5


def generate_bridges(seed):
    """N_BRIDGES bridges, each with N_CONCEPTS_PER_BRIDGE patterns
    drawn from a dedicated N_DIM/N_BRIDGES subspace. Per-bridge
    pattern: sparse activation of subspace_size dims, ACTIVE_FRAC
    sparsity, lasting one full gamma period."""
    rng = np.random.default_rng(seed * 31337)
    subspace_size = N_DIM // N_BRIDGES
    bridges = []
    n_active = max(1, int(SPARSITY * subspace_size))
    for b_idx in range(N_BRIDGES):
        start = b_idx * subspace_size
        subspace = np.arange(start, start + subspace_size)
        # Per-concept gamma-window pattern: (GAMMA_PERIOD, N_DIM)
        # full-dim shape with only this bridge's subspace populated.
        patterns = []
        for c in range(N_CONCEPTS_PER_BRIDGE):
            active = rng.choice(subspace_size, size=n_active,
                                  replace=False)
            window_pattern = np.zeros(
                (GAMMA_PERIOD, N_DIM), dtype=np.float32)
            for t in range(GAMMA_PERIOD):
                window_pattern[t, subspace[active]] = 1.0
            patterns.append(window_pattern)
        bridges.append({
            "patterns": patterns, "subspace": subspace,
            "subspace_size": subspace_size, "n_active": n_active,
        })
    return bridges


def encode_sequence_integrated(seq_assignments, bridges, n_theta,
                                  gamma_period, n_gamma):
    """seq_assignments[slot_idx] = list of (bridge_idx, concept_idx)
    pairs. Each pair places the bridge's concept pattern at gamma
    slot slot_idx within the theta cycle (slot_idx -> theta time
    window slot_idx * gamma_period .. (slot_idx+1) * gamma_period).
    """
    ensemble = np.zeros((n_theta, N_DIM), dtype=np.float32)
    for slot_idx, pairs in enumerate(seq_assignments):
        if slot_idx >= n_gamma: break
        t_start = slot_idx * gamma_period
        t_end = t_start + gamma_period
        for b_idx, c_idx in pairs:
            ensemble[t_start:t_end, :] = ensemble[t_start:t_end, :] + \
                bridges[b_idx]["patterns"][c_idx]
    return ensemble


def add_noise(ensemble, noise_std, rng):
    return ensemble + rng.standard_normal(ensemble.shape).astype(
        np.float32) * noise_std


def query_bridge_at_slot(noisy_ensemble, slot_idx, bridge_idx,
                            bridges, gamma_period, noise_std,
                            familiarity_threshold):
    """Per-bridge query with familiarity gate."""
    bd = bridges[bridge_idx]
    subspace = bd["subspace"]
    t_start = slot_idx * gamma_period
    t_end = t_start + gamma_period
    win = noisy_ensemble[t_start:t_end, subspace]  # gamma_period x subspace_size
    win_summed = win.sum(axis=0)
    win_norm = float(np.linalg.norm(win_summed))
    noise_floor = noise_std * np.sqrt(gamma_period * bd["subspace_size"])
    familiarity = win_norm / (noise_floor + 1e-9)

    if familiarity < familiarity_threshold:
        return None, 0.0, familiarity

    # Identify
    best_c = None; best_score = -np.inf
    for c, pattern in enumerate(bd["patterns"]):
        # pattern shape (GAMMA_PERIOD, N_DIM); restrict to subspace
        p_win_summed = pattern[:gamma_period, subspace].sum(axis=0)
        a = win_summed.astype(np.float64); b = p_win_summed.astype(
            np.float64)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12: continue
        score = float(np.dot(a, b) / (na * nb))
        if score > best_score:
            best_score = score; best_c = c
    return best_c, best_score, familiarity


def run_seed(seed, verbose=True):
    rng = np.random.default_rng(seed * 11)
    bridges = generate_bridges(seed)

    n_correct_active = 0  # correctly identified concept at active slot
    n_total_active = 0
    n_correct_inactive = 0  # correctly abstained at inactive slot
    n_total_inactive = 0

    for trial in range(N_TRIALS):
        # Per slot: pick BRIDGES_PER_SLOT distinct bridges
        seq_assignments = []
        slot_bridges = []
        for slot_idx in range(SLOT_COUNT):
            bs = rng.choice(N_BRIDGES, size=BRIDGES_PER_SLOT,
                              replace=False)
            pairs = []
            for b in bs:
                c = rng.integers(N_CONCEPTS_PER_BRIDGE)
                pairs.append((int(b), int(c)))
            seq_assignments.append(pairs)
            slot_bridges.append(set(int(b) for b in bs))

        ensemble = encode_sequence_integrated(
            seq_assignments, bridges, N_THETA, GAMMA_PERIOD, N_GAMMA)
        noisy = add_noise(ensemble, NOISE_STD, rng)

        for slot_idx in range(SLOT_COUNT):
            active_bridges_at_slot = slot_bridges[slot_idx]
            true_pairs = seq_assignments[slot_idx]
            true_by_bridge = {int(b): int(c) for b, c in true_pairs}

            for b_idx in range(N_BRIDGES):
                pred, score, fam = query_bridge_at_slot(
                    noisy, slot_idx, b_idx, bridges,
                    GAMMA_PERIOD, NOISE_STD, THRESHOLD_NORM_RATIO)

                if b_idx in active_bridges_at_slot:
                    # Should ID the right concept for this bridge
                    if pred == true_by_bridge[b_idx]:
                        n_correct_active += 1
                    n_total_active += 1
                else:
                    # Should abstain
                    if pred is None:
                        n_correct_inactive += 1
                    n_total_inactive += 1

    active_acc = (n_correct_active / n_total_active
                  if n_total_active > 0 else 0.0)
    inactive_acc = (n_correct_inactive / n_total_inactive
                    if n_total_inactive > 0 else 0.0)
    # Combined: every query (active or inactive) counted
    n_correct_total = n_correct_active + n_correct_inactive
    n_total = n_total_active + n_total_inactive
    combined_acc = n_correct_total / n_total if n_total > 0 else 0.0
    return {
        "seed": seed,
        "active_acc": active_acc, "inactive_abstain_acc": inactive_acc,
        "combined_acc": combined_acc,
        "n_active_queries": n_total_active,
        "n_inactive_queries": n_total_inactive,
    }


def main():
    print(f"=== Direction E+F INTEGRATED algebra probe ===",
          flush=True)
    print(f"  N_BRIDGES={N_BRIDGES}, "
          f"N_CONCEPTS_PER_BRIDGE={N_CONCEPTS_PER_BRIDGE}",
          flush=True)
    print(f"  N_DIM (shared)={N_DIM}, "
          f"N_THETA={N_THETA}, N_GAMMA={N_GAMMA}, "
          f"GAMMA_PERIOD={GAMMA_PERIOD}", flush=True)
    print(f"  SLOT_COUNT={SLOT_COUNT}, "
          f"BRIDGES_PER_SLOT={BRIDGES_PER_SLOT}", flush=True)
    print(f"  sparsity={SPARSITY}, noise={NOISE_STD}", flush=True)
    print(f"  THRESHOLD_NORM_RATIO={THRESHOLD_NORM_RATIO} (frozen)",
          flush=True)
    print(f"  Pre-registered bar: {BAR}", flush=True)
    total_vocab = N_BRIDGES * N_CONCEPTS_PER_BRIDGE
    print(f"  Total vocab: {total_vocab} concepts across all bridges",
          flush=True)

    t0 = time.time()
    seed_results = []
    for seed in SEEDS:
        r = run_seed(seed)
        seed_results.append(r)
        print(f"  seed {seed}: active {r['active_acc']:.3f} "
              f"({r['n_active_queries']} queries), inactive "
              f"abstain {r['inactive_abstain_acc']:.3f} "
              f"({r['n_inactive_queries']} queries), combined "
              f"{r['combined_acc']:.3f}", flush=True)

    total_min = (time.time() - t0) / 60
    active_accs = [r["active_acc"] for r in seed_results]
    inactive_accs = [r["inactive_abstain_acc"] for r in seed_results]
    combined_accs = [r["combined_acc"] for r in seed_results]
    active_mean = float(np.mean(active_accs))
    inactive_mean = float(np.mean(inactive_accs))
    combined_mean = float(np.mean(combined_accs))

    print(f"\n  multi-seed active concept-ID: {active_mean:.3f}",
          flush=True)
    print(f"  multi-seed inactive abstain: {inactive_mean:.3f}",
          flush=True)
    print(f"  multi-seed combined: {combined_mean:.3f}", flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    active_pass = active_mean >= BAR
    inactive_pass = inactive_mean >= BAR
    combined_pass = combined_mean >= BAR

    if active_pass and inactive_pass and combined_pass:
        verdict = "EF_INTEGRATED_ALGEBRA_SUFFICIENT"
        print(f"  ALL THREE metrics >= {BAR} -- the integrated "
              f"theta-gamma + cross-bridge + familiarity-gate "
              f"algebra is sufficient at {total_vocab}-concept "
              f"vocab + {SLOT_COUNT}-slot sequence + "
              f"{BRIDGES_PER_SLOT}-bridges-per-slot interference.",
              flush=True)
    elif active_pass and combined_pass and not inactive_pass:
        verdict = "EF_DISCRIM_PASS_ABSTAIN_PARTIAL"
        print(f"  Discrimination + combined PASS; abstention "
              f"partial ({inactive_mean:.3f}); precise bound.",
              flush=True)
    else:
        verdict = "EF_BELOW_BAR_BOUND_IDENTIFIED"
        print(f"  Some metric below {BAR}; precise integrated bound"
              f" identified for follow-up.", flush=True)

    out = {
        "config": {
            "N_BRIDGES": N_BRIDGES,
            "N_CONCEPTS_PER_BRIDGE": N_CONCEPTS_PER_BRIDGE,
            "N_DIM": N_DIM, "N_THETA": N_THETA,
            "N_GAMMA": N_GAMMA, "GAMMA_PERIOD": GAMMA_PERIOD,
            "SLOT_COUNT": SLOT_COUNT,
            "BRIDGES_PER_SLOT": BRIDGES_PER_SLOT,
            "SPARSITY": SPARSITY, "NOISE_STD": NOISE_STD,
            "BAR": BAR, "N_TRIALS": N_TRIALS, "SEEDS": SEEDS,
            "THRESHOLD_NORM_RATIO": THRESHOLD_NORM_RATIO,
            "total_vocab": total_vocab,
        },
        "per_seed": seed_results,
        "active_concept_id_mean": active_mean,
        "inactive_abstain_mean": inactive_mean,
        "combined_mean": combined_mean,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
