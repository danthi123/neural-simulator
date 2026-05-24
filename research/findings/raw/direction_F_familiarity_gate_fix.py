"""Direction F FAMILIARITY-GATE FIX: address the cross-bridge
abstention bound (Test I = 0.712) via a separate familiarity signal.

Per FHRR-biologization shortcut-3 RESOLVED finding: a pure attractor
clean-up CONFABULATES because every input falls into SOME basin;
abstention must be a SEPARATE familiarity/match-strength signal, not
a basin-of-attraction property OR a single cosine threshold.

The cross-bridge abstention failed (0.712 multi-seed) because per-
bridge cosine-with-threshold is the same kind of pure-attractor
mistake: when bridge B is silent at slot i, its subspace gets pure
noise but still ALWAYS has SOME concept with cosine > 0 in the
noise direction; thresholds tuned tight enough to reject these
also reject legitimate weak signals.

The fix (mirror shortcut-3 RESOLVED): use TWO separate signals per
bridge per slot:
  1. IDENTIFICATION: which concept matches best (cosine argmax)
  2. FAMILIARITY: a separate signal measuring whether the activity
     in this bridge's subspace is structured at all
The familiarity signal abstains when activity is noise-like (low L2
norm AND high entropy across subspace); identifies a concept when
activity is structured (high L2 AND concentrated).

Pre-registered:
- Same constants as direction_F_cross_bridge_sequence_interference.py
- BAR = 0.80
- Familiarity = activity L2 norm RELATIVE to noise floor; abstain
  if relative-norm < ratio threshold (pre-set, NOT tuned)
- THRESHOLD_NORM_RATIO = 1.5 (signal must be at least 1.5x the
  expected noise floor norm to claim presence)

If Test I (non-overlapping; abstention test) reaches >= 0.80 with
the familiarity gate while Test II (overlapping) stays >= 0.80 -->
the familiarity-gate fix RESOLVES the cross-bridge abstention
bound the simple-threshold version failed at. Mirrors the
FHRR-biologization shortcut-3 separated-clean-up RESOLVED pattern.

NUMPY only; ~5 s wall.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.direction_F_cross_bridge_sequence_interference import (
    N_BRIDGES, N_CONCEPTS_PER_BRIDGE, N_DIM, SPARSITY, SLOT_COUNT,
    N_TRIALS, BAR, NOISE_STD, SEEDS,
    generate_bridge_patterns_shared_substrate, encode_sequence_shared,
)

OUT_JSON = os.path.join(
    _HERE, "direction_F_familiarity_gate_fix.json")

# Pre-registered familiarity-gate parameter (frozen, not tuned).
THRESHOLD_NORM_RATIO = 1.5


def expected_noise_norm(subspace_size, noise_std):
    """Expected L2 norm of pure Gaussian noise vector of given size +
    std. E[||N(0, sigma)^n||] ~= sigma * sqrt(n) for large n."""
    return noise_std * np.sqrt(subspace_size)


def query_slot_with_familiarity(ensemble, slot_idx, query_bridge_idx,
                                   bridge_data, noise_std):
    """Two-signal query: (a) familiarity = subspace L2 norm /
    expected noise norm; (b) identification = argmax cosine if
    familiarity passes.

    Returns (predicted_concept_or_None, identify_score, familiarity).
    """
    bd = bridge_data[query_bridge_idx]
    subspace = bd["subspace"]
    win = ensemble[slot_idx, subspace]
    win_norm = float(np.linalg.norm(win))
    noise_floor = expected_noise_norm(len(subspace), noise_std)
    familiarity = win_norm / (noise_floor + 1e-9)

    if familiarity < THRESHOLD_NORM_RATIO:
        return None, 0.0, familiarity

    # Identify
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
    return best_c, best_score, familiarity


def run_seed_test1_fam(seed):
    """Test (I) Non-overlapping; familiarity gate."""
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
        for slot_idx in range(SLOT_COUNT):
            true_b, true_c = seq_assignments[slot_idx][0]
            for b in range(N_BRIDGES):
                pred, score, fam = query_slot_with_familiarity(
                    noisy, slot_idx, b, bridge_data, NOISE_STD)
                if b == true_b:
                    if pred == true_c: n_correct += 1
                else:
                    if pred is None: n_correct += 1
                n_total += 1
    return n_correct / n_total if n_total > 0 else 0.0


def run_seed_test2_fam(seed):
    """Test (II) Overlapping; familiarity gate."""
    rng = np.random.default_rng(seed * 31337 + 2)
    bridge_data = generate_bridge_patterns_shared_substrate(
        N_BRIDGES, N_CONCEPTS_PER_BRIDGE, N_DIM, SPARSITY, seed)
    n_correct = 0; n_total = 0
    for trial in range(N_TRIALS):
        seq_assignments = []
        for slot_idx in range(SLOT_COUNT):
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
        for slot_idx in range(SLOT_COUNT):
            for true_b, true_c in seq_assignments[slot_idx]:
                pred, score, fam = query_slot_with_familiarity(
                    noisy, slot_idx, true_b, bridge_data, NOISE_STD)
                if pred == true_c: n_correct += 1
                n_total += 1
    return n_correct / n_total if n_total > 0 else 0.0


def main():
    print(f"=== Direction F FAMILIARITY-GATE FIX ===", flush=True)
    print(f"  N_BRIDGES={N_BRIDGES}, "
          f"N_CONCEPTS_PER_BRIDGE={N_CONCEPTS_PER_BRIDGE}, "
          f"N_DIM={N_DIM}", flush=True)
    print(f"  SLOT_COUNT={SLOT_COUNT}, sparsity={SPARSITY},"
          f" noise={NOISE_STD}", flush=True)
    print(f"  THRESHOLD_NORM_RATIO={THRESHOLD_NORM_RATIO} (frozen)",
          flush=True)
    print(f"  Pre-registered bar: {BAR}", flush=True)
    subspace_size = N_DIM // N_BRIDGES
    expected_floor = expected_noise_norm(subspace_size, NOISE_STD)
    expected_signal_norm = np.sqrt(int(SPARSITY * subspace_size))
    print(f"  expected noise floor norm: {expected_floor:.3f}",
          flush=True)
    print(f"  expected signal norm: {expected_signal_norm:.3f}",
          flush=True)
    print(f"  signal/noise ratio: "
          f"{expected_signal_norm/expected_floor:.1f}x",
          flush=True)

    t0 = time.time()

    print(f"\n--- Test (I) WITH FAMILIARITY GATE ---", flush=True)
    test1_accs = []
    for seed in SEEDS:
        a = run_seed_test1_fam(seed)
        test1_accs.append(a)
        print(f"  seed {seed}: {a:.3f}", flush=True)
    test1_mean = float(np.mean(test1_accs))
    print(f"  mean: {test1_mean:.3f}", flush=True)

    print(f"\n--- Test (II) WITH FAMILIARITY GATE ---", flush=True)
    test2_accs = []
    for seed in SEEDS:
        a = run_seed_test2_fam(seed)
        test2_accs.append(a)
        print(f"  seed {seed}: {a:.3f}", flush=True)
    test2_mean = float(np.mean(test2_accs))
    print(f"  mean: {test2_mean:.3f}", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"\nWall: {total_min:.1f} min", flush=True)

    # Compare to baseline (cosine-threshold-only)
    print(f"\n=== COMPARISON TO BASELINE (no familiarity gate) ===",
          flush=True)
    print(f"  Test I baseline: 0.712 -> fix: {test1_mean:.3f}"
          f" (delta: {test1_mean - 0.712:+.3f})", flush=True)
    print(f"  Test II baseline: 0.996 -> fix: {test2_mean:.3f}"
          f" (delta: {test2_mean - 0.996:+.3f})", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    test1_pass = test1_mean >= BAR
    test2_pass = test2_mean >= BAR
    if test1_pass and test2_pass:
        verdict = "FAMILIARITY_GATE_RESOLVES_ABSTENTION"
        print(f"  Both tests >= {BAR} with familiarity gate. "
              f"The separate norm-ratio familiarity signal closes "
              f"the cross-bridge abstention bound; mirrors FHRR "
              f"shortcut-3 RESOLVED. Identification + familiarity "
              f"as TWO signals (not one threshold) is the right "
              f"biology-grounded fix.", flush=True)
    elif test1_pass and not test2_pass:
        verdict = "FAMILIARITY_FIXES_ABSTENTION_BREAKS_DISCRIM"
        print(f"  Familiarity gate fixes abstention but breaks "
              f"discrimination -- the threshold is too strict for "
              f"shared-slot activity.", flush=True)
    elif not test1_pass and test2_pass:
        verdict = "FAMILIARITY_INSUFFICIENT_FOR_ABSTENTION"
        print(f"  Discrimination holds but abstention still "
              f"below bar; familiarity gate insufficient alone.",
              flush=True)
    else:
        verdict = "FAMILIARITY_BREAKS_BOTH"
        print(f"  Both below bar; familiarity threshold "
              f"misconfigured or wrong mechanism.", flush=True)

    out = {
        "config": {
            "N_BRIDGES": N_BRIDGES,
            "N_CONCEPTS_PER_BRIDGE": N_CONCEPTS_PER_BRIDGE,
            "N_DIM": N_DIM, "SLOT_COUNT": SLOT_COUNT,
            "SPARSITY": SPARSITY, "NOISE_STD": NOISE_STD,
            "BAR": BAR, "THRESHOLD_NORM_RATIO": THRESHOLD_NORM_RATIO,
            "N_TRIALS": N_TRIALS, "SEEDS": SEEDS,
            "expected_noise_floor_norm": float(expected_floor),
            "expected_signal_norm": float(expected_signal_norm),
            "snr": float(expected_signal_norm / expected_floor),
        },
        "baseline_test1": 0.712, "baseline_test2": 0.996,
        "fix_test1_mean": test1_mean, "fix_test1_per_seed": test1_accs,
        "fix_test2_mean": test2_mean, "fix_test2_per_seed": test2_accs,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
