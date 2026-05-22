"""Activity-level integration probe: can the phasor symbol be DERIVED
from the substrate's population activity vector, instead of looked up
from a discrete recognized label?

Design context: docs/plans/2026-05-22-activity-level-integration-design.md

The validated two-system compositional pipeline joins the concept
substrate to the spiking-phasor composition layer at the concept-
IDENTITY level: the substrate reports one discrete recognized label,
and a fixed lookup table maps that label to a pre-assigned phasor
symbol. The substrate's graded population activity never itself enters
the composition layer.

The more biologically faithful interface would DERIVE the phasor
symbol from the population activity vector directly -- no discrete
label, no lookup table. This is activity-level integration.

The honest new failure mode: a population activity vector has trial-
to-trial variability. With a discrete-label lookup, the same label
always yields a byte-identical symbol, so the cue symbol at storage
time and at query time are exactly equal and unbinding is exact. With
an activity-derived symbol, the storage-time and query-time activity
vectors differ -- even for the same concept, even when recognition is
correct -- so the derived symbols differ and unbinding leaves a
residual phase error.

This probe models the activity-level interface and asks whether FHRR
composition (bind / unbind / bundle / clean-up) still clears the
project's frozen 0.80 compositional bar when the symbols are derived
from noisy population activity.

Model:
- Each concept has a "true" population activity centroid: high firing
  (0.2-0.8) on its active dimensions, low leakage (0.0-0.05) elsewhere
  -- the firing levels the codebase documents for a correctly
  recognized word. Two activity representations: a coarse 16-dim
  per-pool vector, and a richer 256-dim distributed population code.
- Each observation of a concept is its centroid plus zero-mean
  Gaussian noise, clipped non-negative. The noise std is swept.
- Symbol derivation: a fixed random complex projection of the
  normalized activity vector; the phase of each projected component is
  one phasor dimension. Deterministic and smooth.
- A fact is stored by binding an activity-derived cue symbol to an
  activity-derived filler symbol and bundling. It is queried by
  unbinding with a cue symbol derived from an INDEPENDENT activity
  trial of the same cue concept. Clean-up matches against a vocabulary
  of symbols derived from the stable (noise-free) centroids.
- Recognition is held correct throughout -- this isolates the one new
  variable, activity-derived vs lookup-derived symbols.

PRE-REGISTERED decision rule (fixed; never tuned):
- PASS: activity-derived FHRR clears the 0.80 bar at loads {2,3,5} at
  every activity-noise std <= 0.10 (a ~20% coefficient of variation on
  a mean firing rate near 0.5 -- within realistic cortical trial-to-
  trial rate variability) at some phasor dimension <= 1024.
  -> activity-level integration is reachable; proceed to design and
  build the real activity-level integration runner.
- NEGATIVE: it does not clear 0.80 at all loads at activity noise
  <= 0.10 -> the identity-level (discrete-label lookup) interface is
  the validated ceiling. Honest finding: the discrete-label bottleneck
  denoises, and a faithful activity-level interface would need an
  explicit denoising/averaging stage.

Standalone numpy. ENGINEERING ceiling-clarification (clearly marked,
non-load-bearing). No protected/frozen/moat module touched. No
automatic differentiation.
"""
from __future__ import annotations

import json
import sys

import numpy as np

N_CUES = 8
N_FILLERS = 8
N_CONCEPTS = N_CUES + N_FILLERS
LOADS = [2, 3, 5]
DIMS = [256, 1024]            # phasor dimension N
D_ACTS = [16, 256]            # activity-vector dimension (coarse / distributed)
SIGMAS = [0.0, 0.05, 0.10, 0.20]   # activity-noise std (firing-rate units)
N_TRIALS = 200
BAR = 0.80
SIGMA_PASS = 0.10             # pre-registered biological-precision threshold
HIGH_RATE = (0.2, 0.8)        # correctly-recognized target firing range
LEAK_RATE = (0.0, 0.05)       # off-target leakage range
SEED = 42


def make_centroids(d_act, rng):
    """V concept activity centroids. Each concept activates a random
    subset of the d_act dimensions at a high firing rate; the rest sit
    at a low leakage rate. For d_act == V this is a one-hot-per-pool
    code; for d_act >> V it is a distributed population code."""
    active_per = max(1, d_act // N_CONCEPTS)
    centroids = np.full((N_CONCEPTS, d_act), 0.0)
    for c in range(N_CONCEPTS):
        active = rng.choice(d_act, size=active_per, replace=False)
        centroids[c] = rng.uniform(LEAK_RATE[0], LEAK_RATE[1], size=d_act)
        centroids[c, active] = rng.uniform(HIGH_RATE[0], HIGH_RATE[1],
                                           size=active_per)
    return centroids


def observe(centroid, sigma, rng):
    """One noisy observation of a concept: centroid + Gaussian noise,
    clipped non-negative (firing rates cannot be negative)."""
    a = centroid + rng.normal(0.0, sigma, size=centroid.shape)
    return np.clip(a, 0.0, None)


def make_deriver(dim, d_act, rng):
    """Build the fixed activity->phasor derivation function: a fixed
    random complex projection of the normalized activity vector; the
    phase of each projected component is one phasor dimension."""
    w_re = rng.normal(0.0, 1.0, size=(dim, d_act))
    w_im = rng.normal(0.0, 1.0, size=(dim, d_act))

    def derive(activity):
        norm = np.linalg.norm(activity)
        a_hat = activity / (norm + 1e-9)
        z = w_re @ a_hat + 1j * (w_im @ a_hat)
        return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)

    return derive


def bind(a, b):
    """Phase-sum neuron: output phase = phi_a + phi_b."""
    return np.mod(a + b, 1.0)


def unbind(a, b):
    """Phase-subtraction neuron: output phase = phi_a - phi_b."""
    return np.mod(a - b, 1.0)


def bundle(phase_list):
    """Phase of the complex sum (the FHRR bundle)."""
    z = np.sum([np.exp(2j * np.pi * p) for p in phase_list], axis=0)
    return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)


def similarity(a, b):
    """Phase-similarity: mean cosine of phase differences."""
    return float(np.mean(np.cos(2.0 * np.pi * (a - b))))


def run_cell(load, dim, d_act, sigma, n_trials, rng):
    """One (load, dim, d_act, sigma) cell. The activity->phasor deriver
    is fixed for the cell; concept centroids are drawn fresh per trial;
    storage and query each draw independent activity noise."""
    derive = make_deriver(dim, d_act, rng)
    n_correct = 0
    n_total = 0
    for _ in range(n_trials):
        centroids = make_centroids(d_act, rng)
        cue_centroids = centroids[:N_CUES]
        fill_centroids = centroids[N_CUES:]
        # Clean-up vocabulary: symbols derived from the stable
        # (noise-free) centroids -- the consolidated concept identities.
        vocab = [derive(fill_centroids[k]) for k in range(N_FILLERS)]

        cue_idx = rng.choice(N_CUES, size=load, replace=False)
        fill_idx = rng.choice(N_FILLERS, size=load, replace=True)
        facts = list(zip(cue_idx, fill_idx))

        # STORE: bind an activity-derived cue symbol to an activity-
        # derived filler symbol; bundle the bound pairs.
        bound = []
        for (c, f) in facts:
            cue_sym = derive(observe(cue_centroids[c], sigma, rng))
            fill_sym = derive(observe(fill_centroids[f], sigma, rng))
            bound.append(bind(cue_sym, fill_sym))
        composite = bundle(bound)

        # QUERY: unbind with a cue symbol derived from an INDEPENDENT
        # activity trial of the same cue concept.
        for (c, f) in facts:
            cue_sym_q = derive(observe(cue_centroids[c], sigma, rng))
            recovered = unbind(composite, cue_sym_q)
            sims = [similarity(recovered, vocab[k]) for k in range(N_FILLERS)]
            if int(np.argmax(sims)) == f:
                n_correct += 1
            n_total += 1
    return n_correct / n_total


def main():
    print("=== Activity-level integration probe ===")
    print(f"vocab {N_CUES}x{N_FILLERS}; loads={LOADS}; phasor dims={DIMS}; "
          f"activity dims={D_ACTS}; activity-noise std={SIGMAS}; "
          f"trials={N_TRIALS}; bar={BAR}")
    rng = np.random.default_rng(SEED)

    grid = {}
    for d_act in D_ACTS:
        for dim in DIMS:
            print(f"\n  activity-dim D={d_act}, phasor-dim N={dim}:")
            for load in LOADS:
                accs = []
                for sigma in SIGMAS:
                    acc = run_cell(load, dim, d_act, sigma, N_TRIALS, rng)
                    grid[(d_act, dim, load, sigma)] = acc
                    accs.append("s%.2f=%.3f" % (sigma, acc))
                print(f"    L={load}:  " + "  ".join(accs))

    # Pre-registered decision: is there a (d_act, dim) for which every
    # load clears the bar at every activity-noise std <= SIGMA_PASS?
    pass_sigmas = [s for s in SIGMAS if s <= SIGMA_PASS]
    pass_cfg = None
    for d_act in D_ACTS:
        for dim in DIMS:
            if all(grid[(d_act, dim, load, s)] >= BAR
                   for load in LOADS for s in pass_sigmas):
                pass_cfg = (d_act, dim)
                break
        if pass_cfg is not None:
            break

    hardest = max(LOADS)
    best_at_thresh = max(grid[(d_act, dim, hardest, SIGMA_PASS)]
                         for d_act in D_ACTS for dim in DIMS)

    print(f"\n=== VERDICT ===")
    if pass_cfg is not None:
        verdict = "ACTIVITY_LEVEL_REACHABLE"
        print(f"  Activity-derived FHRR clears the {BAR} bar at all loads "
              f"{LOADS} at every activity-noise std <= {SIGMA_PASS} "
              f"(activity-dim D={pass_cfg[0]}, phasor-dim N={pass_cfg[1]}).")
        print(f"  --> ACTIVITY-LEVEL REACHABLE: deriving the phasor symbol "
              f"from a noisy population activity vector still supports "
              f"composition. Proceed to design and build the real "
              f"activity-level integration runner.")
    else:
        verdict = "IDENTITY_LEVEL_CEILING"
        print(f"  Activity-derived FHRR does NOT clear {BAR} at all loads at "
              f"activity-noise std <= {SIGMA_PASS}. Best at hardest load "
              f"(L={hardest}) at std {SIGMA_PASS}: {best_at_thresh:.3f}.")
        print(f"  --> IDENTITY-LEVEL CEILING: the discrete-label lookup "
              f"interface is the validated ceiling. The discrete bottleneck "
              f"denoises; a faithful activity-level interface would need an "
              f"explicit denoising/averaging stage.")

    out = {
        "n_cues": N_CUES, "n_fillers": N_FILLERS, "loads": LOADS,
        "phasor_dims": DIMS, "activity_dims": D_ACTS, "sigmas": SIGMAS,
        "n_trials": N_TRIALS, "bar": BAR, "sigma_pass_threshold": SIGMA_PASS,
        "high_rate_range": list(HIGH_RATE), "leak_rate_range": list(LEAK_RATE),
        "seed": SEED,
        "grid": {f"D{d_act}_N{dim}_L{load}_s{sigma}":
                 grid[(d_act, dim, load, sigma)]
                 for d_act in D_ACTS for dim in DIMS
                 for load in LOADS for sigma in SIGMAS},
        "pass_config": list(pass_cfg) if pass_cfg is not None else None,
        "best_at_hardest_load_at_threshold": best_at_thresh,
        "verdict": verdict,
    }
    with open("research/findings/raw/activity_level_integration_probe.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/activity_level_integration_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
