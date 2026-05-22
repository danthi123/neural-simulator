"""Cheap-first FHRR reference probe (ENGINEERING ceiling-clarification).

Design: docs/plans/2026-05-22-phase-coded-VSA-composition-design.md

Standalone numpy implementation of Fourier Holographic Reduced
Representation (FHRR) per Orchard & Jarvis 2023. This is explicitly an
ENGINEERING reference test, permitted under the owner's standing rule
for clearly-marked ceiling-clarification baselines: it tells us about
engineering (is the FHRR ALGEBRA capable of the project's
compositional task), NOT about biology. It is non-load-bearing. It
touches no protected/frozen/moat module -- pure standalone numpy, no
autograd.

The decisive question: at the project's compositional loads (frozen
ladder L = 2, 3, 5) and vocabulary scale (16 symbols), against the
project's frozen 0.80 compositional bar, can FHRR bundle L facts into
one vector and recover each bound filler on query? FHRR bundling is
known-lossy (modulus discarded; loss grows with L) -- this probe
measures whether the loss stays survivable at the project's loads.

FHRR operations (Orchard & Jarvis 2023, sec 1.2):
  symbol      : N-dim vector of unit-modulus complex numbers
  bind   (x)  : elementwise complex multiply  (= phase addition)
  unbind (/)  : elementwise multiply by conjugate (= phase subtraction)
  bundle (+)  : sum, then discard modulus (keep phase only)
  similarity  : (1/N) Re(sum(a * conj(b)))  -- mean cos of phase diff
  clean-up    : argmax similarity over the vocabulary

Pre-registered decision rule (fixed; never tuned):
- FHRR clears 0.80 at L = 2, 3, 5 at a tractable dimension N <= 1024
  -> the algebra is sufficient; the next arc is the biology-grounded
  spiking-phasor implementation.
- FHRR does not clear 0.80 at L = 5 within N <= 1024 -> FHRR bundling
  capacity insufficient for the project's task; ruled out cheaply.
"""
from __future__ import annotations

import json
import sys

import numpy as np

N_CUES = 8          # project-scale: 8 cue symbols + 8 filler symbols = 16
N_FILLERS = 8
LOADS = [2, 3, 5]   # the project's frozen compositional ladder
DIMS = [64, 128, 256, 512, 1024]
N_TRIALS = 200      # random vocab draws per (load, dim) cell
BAR = 0.80          # the project's frozen compositional bar
SEED = 42


def random_symbol(n, rng):
    """A random FHRR symbol: n unit-modulus complex numbers."""
    return np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=n))


def bind(a, b):
    return a * b


def unbind(a, b):
    return a * np.conj(b)


def bundle(vecs):
    """Sum then discard modulus (keep phase). FHRR bundling."""
    s = np.sum(vecs, axis=0)
    mag = np.abs(s)
    mag[mag < 1e-12] = 1e-12  # avoid divide-by-zero on cancellation
    return s / mag


def similarity(a, b):
    """FHRR similarity: mean cosine of phase differences."""
    return float(np.real(np.sum(a * np.conj(b))) / a.size)


def run_cell(load, dim, n_trials, rng):
    """One (load, dim) cell: n_trials random vocab draws; return the
    mean per-fact recovery accuracy."""
    n_correct = 0
    n_total = 0
    for _ in range(n_trials):
        cues = [random_symbol(dim, rng) for _ in range(N_CUES)]
        fillers = [random_symbol(dim, rng) for _ in range(N_FILLERS)]
        # pick `load` facts with DISTINCT cues (a cue queried once is
        # unambiguous); fillers may repeat across facts.
        cue_idx = rng.choice(N_CUES, size=load, replace=False)
        fill_idx = rng.choice(N_FILLERS, size=load, replace=True)
        facts = list(zip(cue_idx, fill_idx))
        # encode: bundle the `load` bound (cue x filler) pairs.
        bound = [bind(cues[c], fillers[f]) for (c, f) in facts]
        composite = bundle(bound)
        # query each fact: unbind by the cue, clean up over fillers.
        for (c, f) in facts:
            recovered = unbind(composite, cues[c])
            sims = [similarity(recovered, fillers[k]) for k in range(N_FILLERS)]
            if int(np.argmax(sims)) == f:
                n_correct += 1
            n_total += 1
    return n_correct / n_total


def main():
    print("=== FHRR numpy reference probe (engineering ceiling-clarification) ===")
    print(f"vocab: {N_CUES} cues x {N_FILLERS} fillers; loads={LOADS}; "
          f"dims={DIMS}; trials={N_TRIALS}/cell; bar={BAR}")
    rng = np.random.default_rng(SEED)

    grid = {}
    for load in LOADS:
        for dim in DIMS:
            acc = run_cell(load, dim, N_TRIALS, rng)
            grid[(load, dim)] = acc
        row = "  L=%d:  " % load + "  ".join(
            "N%d=%.3f" % (d, grid[(load, d)]) for d in DIMS)
        print(row)

    # Decision: smallest dim that clears the bar at EVERY load.
    min_dim_all_loads = None
    for dim in DIMS:
        if all(grid[(load, dim)] >= BAR for load in LOADS):
            min_dim_all_loads = dim
            break

    # Hardest cell = largest load.
    hardest_load = max(LOADS)
    best_at_hardest = max(grid[(hardest_load, d)] for d in DIMS)

    print(f"\n=== VERDICT ===")
    if min_dim_all_loads is not None:
        verdict = "ALGEBRA_SUFFICIENT"
        print(f"  FHRR clears the {BAR} bar at ALL loads {LOADS} at "
              f"dimension N={min_dim_all_loads} (<= 1024).")
        print(f"  --> ALGEBRA SUFFICIENT: the FHRR algebra is capable of the "
              f"project's compositional task. The next arc is the biology-"
              f"grounded spiking-phasor implementation.")
    else:
        verdict = "ALGEBRA_INSUFFICIENT"
        print(f"  FHRR does NOT clear the {BAR} bar at all loads within "
              f"N<=1024. Best accuracy at the hardest load (L={hardest_load}): "
              f"{best_at_hardest:.3f}.")
        print(f"  --> ALGEBRA INSUFFICIENT: FHRR bundling capacity is "
              f"insufficient for the project's task; ruled out cheaply.")

    out = {
        "n_cues": N_CUES, "n_fillers": N_FILLERS, "loads": LOADS,
        "dims": DIMS, "n_trials": N_TRIALS, "bar": BAR, "seed": SEED,
        "grid": {f"L{load}_N{dim}": grid[(load, dim)]
                 for load in LOADS for dim in DIMS},
        "min_dim_clearing_all_loads": min_dim_all_loads,
        "best_at_hardest_load": best_at_hardest,
        "verdict": verdict,
    }
    with open("research/findings/raw/fhrr_numpy_probe.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/fhrr_numpy_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
