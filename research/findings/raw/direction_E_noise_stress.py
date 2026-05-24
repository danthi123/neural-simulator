"""Direction E SMELL TEST: stress the noise level to find where the
theta-gamma algebra actually breaks.

The 0.06 min wall + 1.000 at every tested load suggests the test was
too easy. Sweep PHASE_NOISE_STD to find the breaking point. Also
sweep ACTIVE_FRAC down (sparser codes harder to match) and gamma
window structure (does temporal modulation within gamma window
matter?).

NUMPY only; ~5 min wall.
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

from research.findings.raw.direction_E_theta_gamma_numpy_probe import (
    N_THETA, N_GAMMA, GAMMA_PERIOD, N_DIM, N_VOCAB,
    N_TRIALS_PER_LOAD, BAR, ACTIVE_FRAC, SEEDS,
    generate_concept_patterns, encode_sequence, add_phase_noise,
    decode_slot,
)

OUT_JSON = os.path.join(_HERE, "direction_E_noise_stress.json")

NOISE_LEVELS = [0.05, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
LOAD = 5  # mid-range fix
ACTIVE_FRACS = [0.05, 0.02, 0.01]


def run_one(seed, load, noise_std, active_frac):
    """Re-implements run_seed_load with custom noise + active_frac."""
    rng = np.random.default_rng(seed * 10007 + int(noise_std * 100)
                                  + int(active_frac * 100))
    patterns = np.zeros(
        (N_VOCAB, GAMMA_PERIOD, N_DIM), dtype=np.float32)
    n_active = int(active_frac * N_DIM)
    for c in range(N_VOCAB):
        idx = rng.choice(N_DIM, size=n_active, replace=False)
        for t in range(GAMMA_PERIOD):
            patterns[c, t, idx] = 1.0
    n_correct = 0; n_total = 0
    for trial in range(N_TRIALS_PER_LOAD):
        seq = list(rng.choice(N_VOCAB, size=load, replace=False))
        ensemble = encode_sequence(
            seq, patterns, N_THETA, GAMMA_PERIOD, N_GAMMA)
        noisy = add_phase_noise(ensemble, noise_std, rng)
        for slot_idx in range(load):
            pred, score = decode_slot(
                noisy, slot_idx, patterns, GAMMA_PERIOD, N_VOCAB)
            if pred == seq[slot_idx]:
                n_correct += 1
            n_total += 1
    return n_correct / n_total if n_total > 0 else 0.0


def main():
    print(f"=== Direction E NOISE STRESS ===", flush=True)
    print(f"  LOAD={LOAD}, vocab={N_VOCAB}", flush=True)
    print(f"  noise levels: {NOISE_LEVELS}", flush=True)
    print(f"  active fracs: {ACTIVE_FRACS}", flush=True)

    t0 = time.time()
    results = {}
    for af in ACTIVE_FRACS:
        results[af] = {}
        print(f"\n--- active_frac = {af} ---", flush=True)
        for noise in NOISE_LEVELS:
            accs = []
            for seed in SEEDS:
                accs.append(run_one(seed, LOAD, noise, af))
            mean_acc = float(np.mean(accs))
            results[af][noise] = {
                "per_seed": accs, "mean": mean_acc,
            }
            mark = "PASS" if mean_acc >= BAR else (
                "ABOVE-CHANCE" if mean_acc > 1.0 / N_VOCAB else "AT-CHANCE")
            print(f"  noise={noise:.2f}: mean {mean_acc:.3f} "
                  f"per-seed=[{', '.join(f'{a:.3f}' for a in accs)}] "
                  f"({mark})", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"\nWall: {total_min:.1f} min", flush=True)

    print(f"\n=== BREAKING POINTS ===", flush=True)
    for af in ACTIVE_FRACS:
        # Find first noise where mean drops below BAR
        breaking = None
        for noise in NOISE_LEVELS:
            if results[af][noise]["mean"] < BAR:
                breaking = noise
                break
        if breaking is None:
            print(f"  active_frac={af}: PASSes at all tested noise "
                  f"levels (up to {NOISE_LEVELS[-1]}); robust.",
                  flush=True)
        else:
            print(f"  active_frac={af}: breaks below {BAR} at "
                  f"noise={breaking}", flush=True)

    out = {
        "load": LOAD, "noise_levels": NOISE_LEVELS,
        "active_fracs": ACTIVE_FRACS, "results": results,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
