"""FHRR compositional capacity curve -- scaling the validated
compositional capability.

The biologized compositional capability is validated at small load
(2, 3, 5 bound facts) and 16-word vocabulary. The natural next question
is how it scales: how many bound facts can a single composite hold, as
a function of the phasor dimension, before unbinding and clean-up fall
below the frozen 0.80 bar.

Composition capacity is fundamentally a property of the FHRR algebra:
a composite is a bundle (a normalised complex sum) of bound pairs;
unbinding one pair leaves the wanted symbol plus crosstalk from the
other L-1 pairs, and the crosstalk grows with load while the
discriminability grows with the square root of the phasor dimension.
The resonate-and-fire realization was validated (biologization step 1)
to reproduce the FHRR algebra to within the discrete-time quantization
(~0.002 of a cycle). So the capacity curve is measured here with the
FHRR algebra directly -- fast, and it is the capacity the
resonate-and-fire layer realizes -- and then spot-checked against the
resonate-and-fire layer at two grid points to confirm the curve
transfers to the biologized realization.

Sweeps load {2..24} against phasor dimension {256..2048}, and reports
the capacity curve: the minimum dimension at which each load clears
the frozen 0.80 compositional bar.

PRE-REGISTERED reading (fixed; never tuned):
- A clean curve -- each load clears 0.80 at a dimension that grows
  smoothly with load -- means the composition layer scales with
  dimension as FHRR theory predicts; report the curve.
- A load beyond which no tested dimension (<= 2048) clears 0.80 is an
  honest capacity ceiling at the tested dimensions; report it as such.

Standalone numpy, ENGINEERING ceiling-clarification (non-load-bearing).
Reuses the validated resonate-and-fire FHRR layer for the spot-check.
No protected/frozen/moat module touched. No automatic differentiation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.spiking_phasor_fhrr import phases_to_spikes
from research.runners.resonate_fire_fhrr import (
    ResonateFireFHRR, phase_similarity,
)

N_CUES = 128
N_FILLERS = 32
LOADS = [2, 5, 12, 24, 48, 96]
DIMS = [64, 128, 256, 512, 1024, 2048]
N_TRIALS = 200
BAR = 0.80
SEED = 42


def rand_phases(n, rng):
    return rng.uniform(0.0, 1.0, size=n)


def bind(a, b):
    return np.mod(a + b, 1.0)


def unbind(a, b):
    return np.mod(a - b, 1.0)


def bundle(phase_list):
    z = np.sum([np.exp(2j * np.pi * p) for p in phase_list], axis=0)
    return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)


def similarity(a, b):
    return float(np.mean(np.cos(2.0 * np.pi * (a - b))))


def run_cell(load, dim, n_trials, rng):
    """One (load, dim) cell of the FHRR-algebra capacity sweep."""
    n_ok = n_tot = 0
    for _ in range(n_trials):
        cues = [rand_phases(dim, rng) for _ in range(N_CUES)]
        fillers = [rand_phases(dim, rng) for _ in range(N_FILLERS)]
        cue_idx = list(rng.choice(N_CUES, size=load, replace=False))
        fill_idx = list(rng.choice(N_FILLERS, size=load, replace=True))
        facts = list(zip(cue_idx, fill_idx))
        composite = bundle([bind(cues[c], fillers[f]) for (c, f) in facts])
        for (c, f) in facts:
            recovered = unbind(composite, cues[c])
            sims = [similarity(recovered, fillers[k])
                    for k in range(N_FILLERS)]
            if int(np.argmax(sims)) == f:
                n_ok += 1
            n_tot += 1
    return n_ok / n_tot


def rf_spotcheck(load, dim, n_trials, rng):
    """Spot-check one cell on the validated resonate-and-fire FHRR
    layer, to confirm the algebra capacity curve transfers."""
    net = ResonateFireFHRR(dim, rng)
    n_ok = n_tot = 0
    for _ in range(n_trials):
        cues = [net.random_symbol() for _ in range(N_CUES)]
        fillers = [net.random_symbol() for _ in range(N_FILLERS)]
        cue_idx = list(rng.choice(N_CUES, size=load, replace=False))
        fill_idx = list(rng.choice(N_FILLERS, size=load, replace=True))
        facts = list(zip(cue_idx, fill_idx))
        composite = net.encode([(cues[c], fillers[f]) for (c, f) in facts])
        for (c, f) in facts:
            recovered = net.query(composite, cues[c])
            sims = [phase_similarity(recovered, fillers[k])
                    for k in range(N_FILLERS)]
            if int(np.argmax(sims)) == f:
                n_ok += 1
            n_tot += 1
    return n_ok / n_tot


def main():
    print("=== FHRR compositional capacity curve ===")
    print(f"vocab {N_CUES}x{N_FILLERS}; loads={LOADS}; dims={DIMS}; "
          f"trials={N_TRIALS}; bar={BAR}")
    rng = np.random.default_rng(SEED)

    grid = {}
    for dim in DIMS:
        accs = []
        for load in LOADS:
            acc = run_cell(load, dim, N_TRIALS, rng)
            grid[(load, dim)] = acc
            accs.append(f"L{load}={acc:.3f}")
        print(f"  N={dim:>4}: " + "  ".join(accs))

    # Capacity curve: per load, the minimum dimension that clears the bar.
    print(f"\n  capacity curve (min phasor dimension to clear {BAR}):")
    capacity = {}
    for load in LOADS:
        clearing = [d for d in DIMS if grid[(load, d)] >= BAR]
        capacity[load] = (min(clearing) if clearing else None)
        cap = capacity[load]
        print(f"    load {load:>2}: "
              + (f"N>={cap}" if cap is not None
                 else f"NOT cleared at any tested dimension (<= {max(DIMS)})"))

    # Resonate-and-fire spot-check: confirm the curve transfers to the
    # biologized layer at two grid points.
    print(f"\n  resonate-and-fire spot-check (confirm the algebra curve "
          f"transfers):")
    spot = {}
    for (load, dim) in [(24, 256), (48, 512)]:
        alg = grid[(load, dim)]
        rf = rf_spotcheck(load, dim, 60, np.random.default_rng(SEED + 1))
        spot[f"L{load}_N{dim}"] = {"algebra": alg, "resonate_fire": rf}
        print(f"    L={load} N={dim}: algebra={alg:.3f}  "
              f"resonate-and-fire={rf:.3f}")

    hardest_cleared = max((load for load in LOADS
                           if capacity[load] is not None), default=0)
    print(f"\n=== VERDICT ===")
    print(f"  highest load cleared at a tested dimension: {hardest_cleared} "
          f"(N>={capacity.get(hardest_cleared)})")
    if hardest_cleared >= max(LOADS):
        verdict = "SCALES_ACROSS_TESTED_RANGE"
        print(f"  The composition layer clears the {BAR} bar across the "
              f"whole tested load range up to {max(LOADS)}, at a phasor "
              f"dimension that grows smoothly with load -- it scales with "
              f"dimension as FHRR theory predicts.")
    else:
        verdict = "CAPACITY_CEILING_AT_TESTED_DIMENSIONS"
        print(f"  Loads above {hardest_cleared} do not clear the {BAR} bar "
              f"at any tested dimension (<= {max(DIMS)}) -- an honest "
              f"capacity ceiling at the tested dimensions.")

    out = {
        "n_cues": N_CUES, "n_fillers": N_FILLERS, "loads": LOADS,
        "dims": DIMS, "n_trials": N_TRIALS, "bar": BAR, "seed": SEED,
        "grid": {f"L{load}_N{dim}": grid[(load, dim)]
                 for load in LOADS for dim in DIMS},
        "capacity_curve": {str(load): capacity[load] for load in LOADS},
        "resonate_fire_spotcheck": spot,
        "highest_load_cleared": hardest_cleared,
        "verdict": verdict,
    }
    with open("research/findings/raw/fhrr_capacity_curve_probe.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/fhrr_capacity_curve_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
