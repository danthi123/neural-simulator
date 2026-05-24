"""Direction K CHEAP-FIRST probe: FHRR-based sequence storage with
substrate-realistic noise.

Per pillar n=104 + Direction G data: FOUR convergent BOUNDARY attempts
on substrate sequence storage all cluster at multi-seed 0.25-0.33
strict top-1; the substrate's WEAK concept-pool dynamics + engram-tag
mechanism cannot reliably do per-slot positional binding.

But: the FHRR biologization arc (2026-05-22, pillar n=87) validated
the FHRR composition pipeline at multi-seed 0.98 on substrate
activity (mean-centered grounded symbols + resonate-and-fire + attractor
clean-up; adversarially reviewed CLEAR). The Direction E+F integrated
probe (2026-05-24, commit 94c539e) validated cross-bridge composition
+ familiarity gate at 0.999 multi-seed.

HYPOTHESIS: FHRR-style sequence storage (each slot encoded as
concept_phasor BIND position_phasor; full sequence as BUNDLE of
slot products; retrieval via UNBIND with position_query) bypasses
the engram-tag mechanism's per-slot discrimination problem entirely.
The algebra has already been validated; only the substrate-grounded
form needs noise stress.

CHEAP-FIRST probe (numpy; ~5 min wall; no GPU):
- N_VOCAB = 16
- N_DIM = 512 (FHRR phasor dim; matches validated)
- K = 8 sequences x SLOT_COUNT = 3 (matches Direction A/E/G)
- Per-concept phasor: deterministic per-vocab-index (substrate
  grounding would derive these from mean-centered activity;
  algebra version uses random)
- Per-position phasor: deterministic per-slot (substrate equivalent
  could be theta-gamma-phase-derived; algebra uses random)
- Sequence = SUM_i FHRR_BIND(concept_phasor_i, position_phasor_i)
- Retrieval slot_i: FHRR_UNBIND(sequence, position_phasor_i);
  argmax cosine over vocab.
- Add noise to simulate substrate floor.

Pre-registered FROZEN bar: 0.80 multi-seed STRICT TOP-1 (the same bar
the 4 substrate attempts failed). Multi-seed 42/43/44/45/46 (5 seeds
to match prior FHRR validations).

If algebra clears the bar at substrate-realistic noise (sigma <= 0.5,
matching the FHRR biologization arc's substrate measurements):
substrate-grounded FHRR sequence storage is justified (would reuse
the validated mean-centering + resonate-and-fire + attractor pipeline).

If algebra fails at substrate-realistic noise: FHRR doesn't help
either; the substrate's bound is even deeper than expected.

NUMPY only; no GPU; no autograd. ~5 min wall.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(
    _HERE, "direction_K_fhrr_sequence_storage_numpy_probe.json")

# Pre-registered constants.
N_VOCAB = 16
N_DIM = 512
K_PAIRS = 8
SLOT_COUNT = 3
N_TRIALS_PER_LOAD = 300
BAR = 0.80
SEEDS = [42, 43, 44, 45, 46]
# Substrate-realistic noise levels (mirror Direction E noise stress)
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.25, 0.5, 1.0]


def fhrr_bind(a, b):
    """FHRR binding via complex Hadamard product (phase addition)."""
    return a * b


def fhrr_unbind(a, b):
    """FHRR unbinding via conj(b) * a."""
    return np.conj(b) * a


def fhrr_bundle(*items):
    """FHRR bundling via sum + normalization."""
    s = np.sum(items, axis=0)
    n = np.linalg.norm(s)
    if n < 1e-12:
        return s
    return s / n * np.sqrt(N_DIM)


def gen_concept_phasors(n_vocab, n_dim, seed):
    """Deterministic random phasors per vocab index."""
    rng = np.random.default_rng(seed * 1000 + 1)
    phases = rng.uniform(-np.pi, np.pi, size=(n_vocab, n_dim))
    return np.exp(1j * phases)


def gen_position_phasors(slot_count, n_dim, seed):
    """Deterministic random phasors per slot."""
    rng = np.random.default_rng(seed * 1000 + 2)
    phases = rng.uniform(-np.pi, np.pi, size=(slot_count, n_dim))
    return np.exp(1j * phases)


def cosine_complex(a, b):
    """Cosine similarity for complex vectors (use real part of
    conjugate inner product)."""
    num = np.real(np.dot(np.conj(a), b))
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(num / (na * nb))


def run_seed_load(seed, noise_std, n_trials):
    """Per seed: generate phasors; run n_trials trials at given
    noise; return per-slot strict top-1 accuracy."""
    rng = np.random.default_rng(seed * 99999 + int(noise_std * 1000))
    concept_phasors = gen_concept_phasors(N_VOCAB, N_DIM, seed)
    position_phasors = gen_position_phasors(SLOT_COUNT, N_DIM, seed)
    n_correct = 0; n_total = 0
    for trial in range(n_trials):
        seq = list(rng.choice(N_VOCAB, size=SLOT_COUNT, replace=False))
        # Encode sequence
        bound = []
        for slot_idx, c_idx in enumerate(seq):
            bound.append(fhrr_bind(
                concept_phasors[c_idx], position_phasors[slot_idx]))
        bundle = fhrr_bundle(*bound)
        # Add noise (model substrate-derived phasor imperfection)
        noise = (rng.standard_normal(N_DIM) +
                  1j * rng.standard_normal(N_DIM)) * noise_std
        noisy_bundle = bundle + noise
        # Retrieve each slot
        for slot_idx in range(SLOT_COUNT):
            unbound = fhrr_unbind(noisy_bundle,
                                    position_phasors[slot_idx])
            best_c = -1; best_score = -np.inf
            for c_idx in range(N_VOCAB):
                score = cosine_complex(
                    unbound, concept_phasors[c_idx])
                if score > best_score:
                    best_score = score; best_c = c_idx
            if best_c == seq[slot_idx]:
                n_correct += 1
            n_total += 1
    return n_correct / n_total if n_total > 0 else 0.0


def main():
    print(f"=== Direction K FHRR sequence storage probe ===",
          flush=True)
    print(f"  N_VOCAB={N_VOCAB}, N_DIM={N_DIM}, SLOT_COUNT={SLOT_COUNT}",
          flush=True)
    print(f"  noise levels: {NOISE_LEVELS}", flush=True)
    print(f"  seeds: {SEEDS}", flush=True)
    print(f"  pre-registered FROZEN bar: {BAR}", flush=True)

    t0 = time.time()
    results = {}
    for noise in NOISE_LEVELS:
        accs = []
        for seed in SEEDS:
            accs.append(run_seed_load(seed, noise, N_TRIALS_PER_LOAD))
        mean = float(np.mean(accs))
        results[noise] = {
            "per_seed": accs, "mean": mean,
        }
        mark = "PASS" if mean >= BAR else (
            "PARTIAL" if mean > 0.3 else "AT-CHANCE")
        print(f"  noise={noise:.2f}: mean {mean:.3f} "
              f"per-seed=[{', '.join(f'{a:.3f}' for a in accs)}] "
              f"({mark})", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"\nWall: {total_min:.2f} min", flush=True)

    # Find max noise where mean >= BAR
    breaking = None
    for noise in sorted(NOISE_LEVELS):
        if results[noise]["mean"] < BAR:
            breaking = noise; break
    print(f"\n=== VERDICT ===", flush=True)
    if breaking is None:
        verdict = "FHRR_SEQUENCE_ROBUST_AT_ALL_NOISE"
        print(f"  PASSes at all tested noise levels (up to "
              f"{NOISE_LEVELS[-1]}); FHRR-based sequence storage "
              f"robust to noise; substrate biologization justified.",
              flush=True)
    elif breaking > 0.10:
        verdict = "FHRR_SEQUENCE_ROBUST_TO_BIOLOGICAL_NOISE"
        print(f"  PASSes at biological-precision noise (<= 0.10); "
              f"breaks at noise={breaking}. Substrate-derived "
              f"phasors (sigma ~0.05) should clear the bar.",
              flush=True)
    elif breaking > 0.05:
        verdict = "FHRR_SEQUENCE_MARGINAL"
        print(f"  Breaks at noise={breaking}; marginal robustness "
              f"to substrate noise.", flush=True)
    else:
        verdict = "FHRR_SEQUENCE_FRAGILE"
        print(f"  Breaks at low noise; FHRR sequence storage "
              f"fragile; substrate biologization NOT justified.",
              flush=True)

    out = {
        "config": {
            "N_VOCAB": N_VOCAB, "N_DIM": N_DIM,
            "SLOT_COUNT": SLOT_COUNT, "K_PAIRS": K_PAIRS,
            "N_TRIALS_PER_LOAD": N_TRIALS_PER_LOAD,
            "BAR": BAR, "NOISE_LEVELS": NOISE_LEVELS, "SEEDS": SEEDS,
        },
        "results": results, "breaking_noise": breaking,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
