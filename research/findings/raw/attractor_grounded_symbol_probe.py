"""Attractor-grounded symbol probe -- biologization shortcut 2, deeper form.

Shortcut 2: the FHRR composition layer assigns each concept its phasor
symbol by ORACLE LOOKUP -- a fixed vector, unrelated to the substrate.
The biologized form grounds the symbol in the substrate's own activity.
The naive form -- derive the symbol directly from a single raw
substrate-activity observation -- was a decisive NEGATIVE: the
substrate's per-neuron activity has a measured trial-to-trial
coefficient of variation of about 1.6 (160%), far too noisy; the
derived symbol does not compose (the activity-level integration
decisive run scored composition-only ~0.36-0.42).

The deeper form, which the activity-level negative re-specified and
biologization shortcut 3 built the machinery for: pass the noisy
substrate activity through an attractor network whose fixed points are
the consolidated concept representations. The attractor settle DENOISES
the noisy observation toward a clean concept fixed point; that settled
fixed point is the grounded symbol. The symbol is then grounded (the
attractor's recurrent weights ARE the concept representations, not an
oracle table) and clean (the settle denoised it).

This probe asks the core question cheaply, before any real-substrate
build: can the attractor denoise an activity observation at the
measured substrate noise level well enough to recover the correct
concept? If the attractor recovers the right concept, the grounded
symbol is the clean fixed point and composes perfectly; the composition
is then recognition-bounded, exactly as the validated identity-level
integration already is.

Model:
- V concepts, each a clean phasor pattern (the consolidated
  representation = an attractor fixed point).
- A noisy observation of a concept = its clean phasor with per-
  component Gaussian phase noise, standard deviation swept.
- The attractor (the shortcut-3 ResonateFireTPAM, reused) stores the V
  clean patterns; a noisy observation is settled through it.
- Two measurements per noise level:
    recognition accuracy -- fraction of noisy observations the attractor
      settles to the correct concept (the core metric for the deeper
      form);
    un-denoised composition accuracy -- the FHRR compositional task run
      with the noisy symbols directly (no attractor). This locates the
      noise level that corresponds to the real substrate: the
      activity-level decisive run measured composition-only ~0.36-0.42,
      so the swept noise level whose un-denoised composition lands in
      that band is the measured-substrate operating point.

PRE-REGISTERED reading (fixed; never tuned):
- At the noise level where the un-denoised composition is in the
  0.36-0.42 band (the measured-substrate operating point), if the
  attractor recognition accuracy is >= 0.80 -> the deeper form is
  reachable: attractor denoising recovers a composable grounded symbol
  from substrate-noisy activity. Proceed to the real-substrate build.
- If recognition accuracy stays below 0.80 at that operating point ->
  attractor denoising cannot rescue activity at the measured noise; the
  honest ceiling is that a grounded symbol needs a cleaner substrate
  representation, which routes back to improving the recognition
  substrate itself.

Standalone numpy, ENGINEERING ceiling-clarification (non-load-bearing).
Reuses the validated/biologized subsystems by import, byte-unchanged.
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

from research.runners.spiking_phasor_fhrr import (
    phases_to_spikes, spikes_to_phases, phase_similarity,
)
from research.runners.resonate_fire_fhrr import (
    ResonateFireFHRR, ResonateFireTPAM,
    ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
)

N_CUES = 8
N_FILLERS = 8
N_CONCEPTS = N_CUES + N_FILLERS
LOADS = [2, 3, 5]
N_DIM = 256
N_TRIALS = 120
SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]   # phase-noise std (cycle fraction)
BAR = 0.80
SUBSTRATE_BAND = (0.36, 0.42)   # measured activity-level composition-only band
SEED = 42


def noisy_observation(clean_spikes, sigma, rng, t_steps):
    """A noisy observation of a concept: its clean phasor with per-
    component Gaussian phase noise."""
    phases = spikes_to_phases(clean_spikes, t_steps)
    noised = np.mod(phases + rng.normal(0.0, sigma, size=phases.shape), 1.0)
    return phases_to_spikes(noised, t_steps)


def main():
    print("=== attractor-grounded symbol probe (shortcut 2, deeper form) ===")
    print(f"concepts {N_CONCEPTS}; loads={LOADS}; N_dim={N_DIM}; "
          f"trials={N_TRIALS}; phase-noise std={SIGMAS}; bar={BAR}")
    rng = np.random.default_rng(SEED)
    net = ResonateFireFHRR(N_DIM, rng)
    t_steps = net.t_steps

    grid = {}
    for sigma in SIGMAS:
        # Recognition: can the attractor settle a noisy observation back
        # to the correct concept?
        n_rec_correct = 0
        n_rec_total = 0
        for _ in range(N_TRIALS):
            concepts = [net.random_symbol() for _ in range(N_CONCEPTS)]
            tpam = ResonateFireTPAM(concepts)
            for c in range(N_CONCEPTS):
                obs = noisy_observation(concepts[c], sigma, rng, t_steps)
                z, _ = tpam.settle_annealed(obs, ANNEAL_THETA_LOW,
                                            ANNEAL_THETA_HIGH, ANNEAL_ITERS)
                overlaps = np.abs(tpam.s.conj().T @ z)
                n_rec_correct += int(np.argmax(overlaps) == c)
                n_rec_total += 1
        rec_acc = n_rec_correct / n_rec_total

        # Un-denoised composition: the FHRR task run with noisy symbols
        # directly (no attractor) -- locates the measured-substrate
        # noise level.
        comp_correct = comp_total = 0
        for _ in range(N_TRIALS):
            cues = [net.random_symbol() for _ in range(N_CUES)]
            fillers = [net.random_symbol() for _ in range(N_FILLERS)]
            for load in LOADS:
                cue_idx = list(rng.choice(N_CUES, size=load, replace=False))
                fill_idx = list(rng.choice(N_FILLERS, size=load, replace=True))
                facts = list(zip(cue_idx, fill_idx))
                composite = net.encode([
                    (noisy_observation(cues[c], sigma, rng, t_steps),
                     noisy_observation(fillers[f], sigma, rng, t_steps))
                    for (c, f) in facts])
                for (c, f) in facts:
                    recovered = net.query(
                        composite,
                        noisy_observation(cues[c], sigma, rng, t_steps))
                    sims = [phase_similarity(recovered, fillers[k])
                            for k in range(N_FILLERS)]
                    comp_correct += int(int(np.argmax(sims)) == f)
                    comp_total += 1
        comp_acc = comp_correct / comp_total

        grid[sigma] = {"recognition_acc": rec_acc,
                       "undenoised_composition": comp_acc}
        print(f"  sigma={sigma:.2f}: attractor recognition acc={rec_acc:.4f} "
              f"| un-denoised composition={comp_acc:.4f}")

    # Locate the measured-substrate operating point: the swept sigma
    # whose un-denoised composition is closest to the centre of the
    # measured 0.36-0.42 band.
    band_mid = 0.5 * (SUBSTRATE_BAND[0] + SUBSTRATE_BAND[1])
    op_sigma = min(SIGMAS,
                   key=lambda s: abs(grid[s]["undenoised_composition"]
                                     - band_mid))
    op_rec = grid[op_sigma]["recognition_acc"]
    op_comp = grid[op_sigma]["undenoised_composition"]

    print(f"\n=== VERDICT ===")
    print(f"  measured-substrate operating point: sigma={op_sigma:.2f} "
          f"(un-denoised composition {op_comp:.3f}, in/near the measured "
          f"{SUBSTRATE_BAND} band); attractor recognition there = "
          f"{op_rec:.4f}")
    if op_rec >= BAR:
        verdict = "DEEPER_FORM_REACHABLE"
        print(f"  --> DEEPER FORM REACHABLE: at the measured-substrate "
              f"noise level the attractor recovers the correct concept "
              f">= {BAR} of the time, so the attractor-denoised grounded "
              f"symbol is composable. Proceed to the real-substrate build.")
    else:
        verdict = "ATTRACTOR_DENOISING_INSUFFICIENT"
        print(f"  --> ATTRACTOR DENOISING INSUFFICIENT: at the measured-"
              f"substrate noise level the attractor recovers the correct "
              f"concept only {op_rec:.3f} < {BAR}. Attractor denoising "
              f"cannot rescue activity at the measured noise; a grounded "
              f"symbol needs a cleaner substrate representation -- routes "
              f"to improving the recognition substrate itself.")

    out = {
        "n_concepts": N_CONCEPTS, "loads": LOADS, "n_dim": N_DIM,
        "n_trials": N_TRIALS, "sigmas": SIGMAS, "bar": BAR,
        "substrate_band": list(SUBSTRATE_BAND), "seed": SEED,
        "grid": {f"sigma_{s}": grid[s] for s in SIGMAS},
        "operating_point_sigma": op_sigma,
        "operating_point_recognition_acc": op_rec,
        "operating_point_undenoised_composition": op_comp,
        "verdict": verdict,
    }
    with open("research/findings/raw/attractor_grounded_symbol_probe.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/attractor_grounded_symbol_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
