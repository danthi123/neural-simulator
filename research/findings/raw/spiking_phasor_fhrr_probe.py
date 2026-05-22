"""Spiking-phasor FHRR realization probe: does FHRR composition survive
spike-timing jitter?

Design context: docs/plans/2026-05-22-phase-coded-VSA-composition-design.md

The numpy FHRR probe established the FHRR ALGEBRA is sufficient (100%
at N=64). The next question for a biology-grounded spiking realization
is whether FHRR composition survives the spike-timing noise a real
spiking substrate has. In Orchard & Jarvis's spiking-phasor FHRR, each
vector dimension is a neuron firing once per cycle; its value is the
PHASE of that spike. A real spiking neuron's spike time is noisy --
quantized to the simulation step and jittered by membrane noise.

This probe realizes every phasor as a spike on a finite-resolution
cycle with Gaussian timing jitter, and accumulates that jitter through
the full FHRR operation chain (encode = bind + bundle; query = unbind;
clean-up), measuring whether the project's compositional task still
clears the frozen 0.80 bar.

Operations (Orchard & Jarvis 2023):
  bind   : phase addition       (phase-sum neuron)
  unbind : phase subtraction    (phase-subtraction neuron)
  bundle : phase of complex sum (phase-midpoint neuron, generalized)
  clean-up: nearest vocabulary vector by phase-similarity
Every phasor output is realized as a noisy spike (quantize to T steps
+ Gaussian jitter) -- the spiking-realization layer.

PRE-REGISTERED decision rule (fixed; never tuned):
- PASS: spiking-phasor FHRR clears the 0.80 bar at loads {2,3,5} at
  spike-timing jitter sigma <= 0.05 of a cycle (~6 ms on a 125 ms
  theta cycle -- the upper end of biological spike-timing precision).
  -> the spiking realization tolerates realistic noise; proceed to
  the biological-scale bridge integration.
- NEGATIVE: it does not clear 0.80 at sigma <= 0.05 -> the spiking
  realization is noise-fragile; routes to a noise-mitigation question
  (more dimensions; multi-cycle averaging) before any bridge build.

Standalone numpy. ENGINEERING ceiling-clarification (clearly marked,
non-load-bearing). No protected/frozen/moat module touched. No
autograd.
"""
from __future__ import annotations

import json
import sys

import numpy as np

N_CUES = 8
N_FILLERS = 8
LOADS = [2, 3, 5]
DIMS = [256, 1024]
JITTERS = [0.0, 0.01, 0.02, 0.05, 0.10]   # fraction of a cycle
T_STEPS = 1000        # simulation steps per global cycle (phase resolution)
N_TRIALS = 200
BAR = 0.80
JITTER_PASS = 0.05    # pre-registered biological-precision threshold
SEED = 42


def random_phases(n, rng):
    """A random spiking-phasor symbol: n phases in [0, 1)."""
    return rng.uniform(0.0, 1.0, size=n)


def spike_realize(phase, t_steps, jitter_sigma, rng):
    """Realize a phasor's value as a noisy spike: quantize the phase to
    a t_steps-resolution cycle and add Gaussian timing jitter. Returns
    the realized phase in [0, 1)."""
    t = phase * t_steps + rng.normal(0.0, jitter_sigma * t_steps, size=phase.shape)
    t = np.round(t)
    return np.mod(t / t_steps, 1.0)


def bind(a, b):
    """Phase-sum neuron: output phase = phi_a + phi_b."""
    return np.mod(a + b, 1.0)


def unbind(a, b):
    """Phase-subtraction neuron: output phase = phi_a - phi_b."""
    return np.mod(a - b, 1.0)


def bundle(phase_list):
    """Phase-midpoint neuron, generalized: phase of the complex sum
    (the FHRR bundle -- sum the unit-complex vectors, keep phase)."""
    z = np.sum([np.exp(2j * np.pi * p) for p in phase_list], axis=0)
    mag = np.abs(z)
    mag[mag < 1e-12] = 1e-12
    return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)


def similarity(a, b):
    """Phase-similarity: mean cosine of phase differences."""
    return float(np.mean(np.cos(2.0 * np.pi * (a - b))))


def run_cell(load, dim, jitter, n_trials, t_steps, rng):
    """One (load, dim, jitter) cell. Jitter is applied at every phasor
    output -- it accumulates through the encode + query chain."""
    n_correct = 0
    n_total = 0
    for _ in range(n_trials):
        cues = [random_phases(dim, rng) for _ in range(N_CUES)]
        fillers = [random_phases(dim, rng) for _ in range(N_FILLERS)]
        cue_idx = rng.choice(N_CUES, size=load, replace=False)
        fill_idx = rng.choice(N_FILLERS, size=load, replace=True)
        facts = list(zip(cue_idx, fill_idx))
        # ENCODE: each bind is a phase-sum neuron whose spike is noisy;
        # its inputs (cue, filler phasors) are noisy spikes too.
        bound = []
        for (c, f) in facts:
            cue_sp = spike_realize(cues[c], t_steps, jitter, rng)
            fill_sp = spike_realize(fillers[f], t_steps, jitter, rng)
            b = spike_realize(bind(cue_sp, fill_sp), t_steps, jitter, rng)
            bound.append(b)
        composite = spike_realize(bundle(bound), t_steps, jitter, rng)
        # QUERY: unbind by the cue (a fresh noisy spike of the cue).
        for (c, f) in facts:
            cue_sp = spike_realize(cues[c], t_steps, jitter, rng)
            recovered = spike_realize(unbind(composite, cue_sp), t_steps, jitter, rng)
            # CLEAN-UP: similarity vs the clean filler vocabulary.
            sims = [similarity(recovered, fillers[k]) for k in range(N_FILLERS)]
            if int(np.argmax(sims)) == f:
                n_correct += 1
            n_total += 1
    return n_correct / n_total


def main():
    print("=== Spiking-phasor FHRR realization probe ===")
    print(f"vocab {N_CUES}x{N_FILLERS}; loads={LOADS}; dims={DIMS}; "
          f"jitters={JITTERS} (fraction of cycle); T={T_STEPS} steps/cycle; "
          f"trials={N_TRIALS}; bar={BAR}")
    rng = np.random.default_rng(SEED)

    grid = {}
    for dim in DIMS:
        print(f"\n  N={dim}:")
        for load in LOADS:
            accs = []
            for jit in JITTERS:
                acc = run_cell(load, dim, jit, N_TRIALS, T_STEPS, rng)
                grid[(dim, load, jit)] = acc
                accs.append("j%.2f=%.3f" % (jit, acc))
            print(f"    L={load}:  " + "  ".join(accs))

    # Decision: at the pre-registered jitter threshold, do all loads
    # clear the bar at some dimension <= 1024?
    pass_dim = None
    for dim in DIMS:
        if all(grid[(dim, load, JITTER_PASS)] >= BAR for load in LOADS):
            pass_dim = dim
            break

    hardest = max(LOADS)
    best_at_thresh = max(grid[(dim, hardest, JITTER_PASS)] for dim in DIMS)

    print(f"\n=== VERDICT ===")
    if pass_dim is not None:
        verdict = "NOISE_TOLERANT"
        print(f"  Spiking-phasor FHRR clears the {BAR} bar at all loads "
              f"{LOADS} at biological-precision jitter sigma={JITTER_PASS} "
              f"(dimension N={pass_dim}).")
        print(f"  --> NOISE TOLERANT: the spiking realization survives "
              f"realistic spike-timing noise. Proceed to biological-scale "
              f"bridge integration.")
    else:
        verdict = "NOISE_FRAGILE"
        print(f"  Spiking-phasor FHRR does NOT clear {BAR} at all loads at "
              f"jitter sigma={JITTER_PASS}. Best at hardest load "
              f"(L={hardest}): {best_at_thresh:.3f}.")
        print(f"  --> NOISE FRAGILE: routes to noise-mitigation (more "
              f"dimensions, multi-cycle averaging) before any bridge build.")

    out = {
        "n_cues": N_CUES, "n_fillers": N_FILLERS, "loads": LOADS,
        "dims": DIMS, "jitters": JITTERS, "t_steps": T_STEPS,
        "n_trials": N_TRIALS, "bar": BAR, "jitter_pass_threshold": JITTER_PASS,
        "seed": SEED,
        "grid": {f"N{dim}_L{load}_j{jit}": grid[(dim, load, jit)]
                 for dim in DIMS for load in LOADS for jit in JITTERS},
        "pass_dim_at_threshold": pass_dim,
        "best_at_hardest_load_at_threshold": best_at_thresh,
        "verdict": verdict,
    }
    with open("research/findings/raw/spiking_phasor_fhrr_probe.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/spiking_phasor_fhrr_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
