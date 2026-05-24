"""Direction E CHEAP NUMPY PROBE: Lisman-Idiart theta-gamma
multiplexing positional binding (catalog N.16).

The brain encodes ordered sequences via theta-gamma multiplexing: ONE
theta cycle (~125 ms) contains ~7 gamma cycles (~17.5 ms each); each
gamma slot corresponds to one sequence position; items at position i
fire at gamma slot i within the theta cycle. The decoder reads which
gamma slot a concept's firing-phase aligns with. This is a TEMPORAL
(phase) code, distinct from ec_context's SPATIAL (per-position
neuron group) code.

Per autonomous discipline (falsify-cheaply-first): test the algebra
in numpy BEFORE committing to a spiking-substrate implementation. If
the algebra clears the frozen 0.80 bar at K=5 sequence positions
with biological noise, the principled mechanism is right and a
substrate version is justified. If it fails in the algebra, no point
spending GPU on substrate implementation.

Pre-registered:
- N_THETA = 125 (samples per theta cycle ~125 ms at 1 kHz)
- N_GAMMA = 7 (gamma slots per theta -- catalog cap; max sequence
  length without multi-theta multiplexing)
- N_DIM = 256 (concept vector dimension; matches FHRR validated)
- PHASE_NOISE_STD = 0.05 (biological-precision phase jitter; matches
  the resonate-and-fire biologization probe)
- N_VOCAB = 16 (matches concept-pool vocab)
- K = 5 (sequence length; tests beyond Direction A's 3-slot;
  N_GAMMA=7 gives capacity headroom 7)
- N_TRIALS_PER_LOAD = 300 (statistical power)
- BAR = 0.80 (frozen 0.80 multi-seed bar; never tuned)

Mechanism (in numpy):
  1. Each concept c gets a random theta-cycle firing pattern
     pattern_c(t) -> activation across N_DIM at each of N_THETA
     samples. Sparse: ~ACTIVE_FRAC of N_DIM neurons active in any
     given window.
  2. To encode "concept c at position i": shift pattern_c by
     i * GAMMA_PERIOD samples (so its activation peaks at gamma
     slot i within the theta cycle).
  3. Sequence ensemble = sum of shifted concept activations across
     all positions.
  4. To recall slot i: pick the GAMMA_PERIOD-sized window at gamma
     slot i; cosine-match the activation across that window against
     each concept's pattern at slot 0. Top match = concept at slot i.
  5. Add PHASE_NOISE_STD biological jitter per sample.

If multi-seed mean >= 0.80 at K=5 -> theta-gamma algebra is sufficient;
worth substrate implementation. If between chance and 0.80 -> BOUNDARY,
characterize what helps. If at chance -> the algebra itself is
insufficient at biologically-realistic noise.

NUMPY-only; no GPU; no autograd; no protected/frozen/moat module
touched. Reuses no project module (genuinely net-new algebra probe).
~2-5 min wall.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(_HERE, "direction_E_theta_gamma_numpy_probe.json")

# Pre-registered constants.
N_THETA = 125  # samples per theta cycle (1 kHz; ~125 ms)
N_GAMMA = 7    # gamma slots per theta cycle
GAMMA_PERIOD = N_THETA // N_GAMMA  # 17 samples per gamma slot
N_DIM = 256
PHASE_NOISE_STD = 0.05
N_VOCAB = 16
N_TRIALS_PER_LOAD = 300
BAR = 0.80
ACTIVE_FRAC = 0.05  # 5% of N_DIM active per gamma window
LOADS = [2, 3, 5, 7]  # test loads 2, 3, 5 (matches FHRR), 7 (cap)
SEEDS = [42, 43, 44]


def generate_concept_patterns(n_vocab, n_dim, gamma_period, seed):
    """Each concept = a sparse activation pattern over one gamma
    window (GAMMA_PERIOD samples x N_DIM neurons). Pattern lives
    in the FIRST gamma slot by default; shifted at encoding time."""
    rng = np.random.default_rng(seed)
    patterns = np.zeros(
        (n_vocab, gamma_period, n_dim), dtype=np.float32)
    n_active = int(ACTIVE_FRAC * n_dim)
    for c in range(n_vocab):
        active_idx = rng.choice(n_dim, size=n_active, replace=False)
        # Pattern: each active neuron fires at uniform phase across
        # the gamma window (simple model; biology has temporal
        # modulation but this is the cleanest test).
        for t in range(gamma_period):
            patterns[c, t, active_idx] = 1.0
    return patterns


def encode_sequence(seq_concept_indices, patterns, n_theta,
                     gamma_period, n_gamma):
    """Build a theta-cycle ensemble: for each (slot_idx, concept_idx),
    place patterns[concept_idx] at gamma slot slot_idx.

    Returns: ensemble of shape (n_theta, n_dim).
    """
    n_dim = patterns.shape[2]
    ensemble = np.zeros((n_theta, n_dim), dtype=np.float32)
    for slot_idx, concept_idx in enumerate(seq_concept_indices):
        if slot_idx >= n_gamma: break
        t_start = slot_idx * gamma_period
        t_end = min(t_start + gamma_period, n_theta)
        slot_len = t_end - t_start
        ensemble[t_start:t_end, :] = ensemble[t_start:t_end, :] + \
            patterns[concept_idx, :slot_len, :]
    return ensemble


def add_phase_noise(ensemble, std, rng):
    """Add Gaussian noise to model biological phase jitter."""
    return ensemble + rng.standard_normal(ensemble.shape).astype(
        np.float32) * std


def decode_slot(noisy_ensemble, slot_idx, patterns, gamma_period,
                  n_vocab):
    """Read the gamma window at slot_idx; cosine-match against each
    concept's pattern (at slot 0). Return predicted concept_idx."""
    t_start = slot_idx * gamma_period
    t_end = t_start + gamma_period
    if t_end > noisy_ensemble.shape[0]:
        t_end = noisy_ensemble.shape[0]
    window = noisy_ensemble[t_start:t_end, :]
    # Aggregate activity across the window (per-neuron sum).
    win_summed = window.sum(axis=0)  # shape (n_dim,)
    best_c = -1
    best_score = -np.inf
    for c in range(n_vocab):
        pattern_summed = patterns[c, :win_summed.shape[0] if False
                                    else patterns.shape[1], :].sum(axis=0)
        a = win_summed.astype(np.float64)
        b = pattern_summed.astype(np.float64)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12: continue
        score = float(np.dot(a, b) / (na * nb))
        if score > best_score:
            best_score = score
            best_c = c
    return best_c, best_score


def run_seed_load(seed, load, n_trials):
    rng = np.random.default_rng(seed * 1000 + load)
    patterns = generate_concept_patterns(
        N_VOCAB, N_DIM, GAMMA_PERIOD, seed=seed)
    n_correct = 0
    n_total = 0
    for trial in range(n_trials):
        seq = list(rng.choice(N_VOCAB, size=load, replace=False))
        ensemble = encode_sequence(
            seq, patterns, N_THETA, GAMMA_PERIOD, N_GAMMA)
        noisy = add_phase_noise(ensemble, PHASE_NOISE_STD, rng)
        # Decode each slot.
        for slot_idx in range(load):
            pred, score = decode_slot(
                noisy, slot_idx, patterns, GAMMA_PERIOD, N_VOCAB)
            true = seq[slot_idx]
            if pred == true:
                n_correct += 1
            n_total += 1
    return n_correct / n_total if n_total > 0 else 0.0


def main():
    print(f"=== Direction E theta-gamma numpy probe (catalog N.16) ===",
          flush=True)
    print(f"  Loads: {LOADS}, seeds: {SEEDS}, trials/load: "
          f"{N_TRIALS_PER_LOAD}", flush=True)
    print(f"  N_GAMMA={N_GAMMA} (catalog cap; sequences longer "
          f"than {N_GAMMA} need multi-theta multiplexing)",
          flush=True)
    print(f"  N_DIM={N_DIM}, phase_noise_std={PHASE_NOISE_STD}, "
          f"active_frac={ACTIVE_FRAC}", flush=True)
    print(f"  Pre-registered bar: {BAR}", flush=True)

    t0 = time.time()
    results = {}  # load -> per-seed accs
    for load in LOADS:
        accs = []
        for seed in SEEDS:
            acc = run_seed_load(seed, load, N_TRIALS_PER_LOAD)
            accs.append(acc)
        mean_acc = float(np.mean(accs))
        results[load] = {
            "per_seed_acc": accs, "mean_acc": mean_acc,
        }
        print(f"  load {load}: mean acc = {mean_acc:.3f} per-seed="
              f"[{', '.join(f'{a:.3f}' for a in accs)}]", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"\nWall: {total_min:.2f} min", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    all_pass = all(results[L]["mean_acc"] >= BAR for L in LOADS)
    if all_pass:
        verdict = "THETA_GAMMA_ALGEBRA_SUFFICIENT"
        print(f"  Multi-seed mean >= {BAR} at every load -- the "
              f"theta-gamma algebra is sufficient; biologization "
              f"to spiking substrate justified.", flush=True)
    else:
        # Find load ceiling
        max_pass_load = max((L for L in LOADS
                              if results[L]["mean_acc"] >= BAR),
                             default=None)
        worst_load = min(LOADS, key=lambda L: results[L]["mean_acc"])
        worst_acc = results[worst_load]["mean_acc"]
        if max_pass_load is not None:
            verdict = f"THETA_GAMMA_LOAD_CEILING_AT_{max_pass_load}"
            print(f"  PASSes at loads <= {max_pass_load}; collapses "
                  f"at higher loads (load {worst_load}: "
                  f"{worst_acc:.3f}).", flush=True)
        else:
            verdict = "THETA_GAMMA_ALGEBRA_INSUFFICIENT"
            print(f"  Below bar at every load (best: {worst_load} "
                  f"= {worst_acc:.3f}); algebra alone insufficient.",
                  flush=True)

    out = {
        "config": {
            "N_THETA": N_THETA, "N_GAMMA": N_GAMMA,
            "GAMMA_PERIOD": GAMMA_PERIOD, "N_DIM": N_DIM,
            "PHASE_NOISE_STD": PHASE_NOISE_STD, "N_VOCAB": N_VOCAB,
            "N_TRIALS_PER_LOAD": N_TRIALS_PER_LOAD, "BAR": BAR,
            "ACTIVE_FRAC": ACTIVE_FRAC, "LOADS": LOADS, "SEEDS": SEEDS,
        },
        "results": results, "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
