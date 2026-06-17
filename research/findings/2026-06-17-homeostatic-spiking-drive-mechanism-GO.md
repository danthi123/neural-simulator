# Spiking homeostatic-drive mechanism — GO (3 seeds): the drive + intrinsic reward work on REAL spikes

**Date:** 2026-06-17
**Status:** **GO, 3 seeds.** The brain-based realization's low-risk half is validated on real spikes: a 2-pool
*spiking* hypothalamic drive encodes the body's energy deficit, and a neuromodulator sourced from its firing
yields the intrinsic drive-reduction reward — with no host reward formula. Next increment: the spiking BG cascade
learning the policy from this neural reward (the full loop).

## Context

The cheapest-first GO (`2026-06-17-homeostatic-drive-rl-cheap-first-GO.md`, 6 seeds) de-risked the *reward
structure* — an intrinsic drive-reduction reward is a learnable training signal — at the rate-proxy/algorithm
level (a 2-pool rate drive + tabular Q). This is the first **brain-based** increment: the drive + reward on
**real spikes**, under the brain-based-only standard (host code is the body + environment only; the drive and
reward are computed by neurons/synapses).

## What was built (`_homeostatic_spiking_drive_mechanism_derisk.py`)

A `SimulationBridge` with two Izhikevich pools — `agrp` (hunger) and `pomc` (satiety) — driven by the body's
energy state (an interoceptive current, the legitimate body→sensory boundary): `agrp ∝ deficit`, `pomc ∝ surplus`.
A `hunger` neuromodulator is sourced from the AgRP pool's firing via the **existing** `from_region_firing_signed`
production rule (the same proven path as the spiking SNc dopamine, `sim/neuromodulators.py:131`) — so the
intrinsic reward `r = −Δ(hunger concentration)` (drive REDUCTION → positive reward; Keramati & Gutkin, *eLife*
2014) is read from spikes, never a host distance/reward term.

## Result (3 seeds: 42/43/44)

| check | seed 42 | seed 43 | seed 44 | gate |
|---|---|---|---|---|
| corr(deficit, AgRP firing) | +1.00 | +1.00 | +1.00 | ≥ +0.90 — the spiking drive encodes the deficit ✓ |
| drive = AgRP − POMC monotone in deficit | True | True | True | push-pull ✓ |
| eating: hunger conc hungry → fed (r = −Δ) | 0.53→0.16 (**+0.37**) | 0.48→0.14 (**+0.33**) | 0.55→0.17 (**+0.37**) | r > 0.2 — drive-reduction reward ✓ |
| **lesion** (zero AgRP drive): rate / r | 0.004 / +0.02 | 0.003 / +0.01 | 0.004 / +0.01 | silent + r ≈ 0 ✓ |

**GO, 3/3.** A 2-pool spiking drive encodes the body's deficit; the push-pull signal is monotone; eating (the
deficit dropping) **drops the hunger modulator**, giving a positive intrinsic reward read from the drive pools'
firing; lesioning the drive (zeroing its interoceptive current) silences it and zeroes the reward. The brain-based
drive + neural reward work on real spikes.

## Debug trail (for reproducibility)

Four issues found + fixed cheaply: (1) wrong concentration accessor (`get_concentration`, not `.concentrations`);
(2) drive current too weak — Izhikevich RS needs ~hundreds of pA, not ~10 (matched the SNc probe's ~220 pA scale);
(3) modulator calibration — the per-step firing *fraction* is low (~0.08, capped by the refractory period), so the
`from_region_firing_signed` sensitivity/threshold needed raising (`sensitivity=100, threshold=0.005`) given the
concentration steady-state is `production × ~100`; (4) a state-carryover artifact — the concentration persists
across measurements, contaminating the lesion read (r_lesion +0.11) until each comparative measurement resets the
concentration to baseline first (then r_lesion ≈ 0).

## Honest scope + next

This de-risks the **mechanism** (the spiking drive encodes the deficit + the neural reward tracks drive-reduction
+ the lesion control) — the scoping's "low-risk half". It is NOT yet the full loop: the next increment is the
**spiking BG action cascade learning a policy** from this neural reward (the load-bearing learning question, here
realized in spikes rather than the rate-proxy tabular Q of the cheapest-first). An honest NEGATIVE there would pin
the exact wall (the spiking substrate learning from sparse intrinsic reward), itself a deliverable.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._homeostatic_spiking_drive_mechanism_derisk --seeds 42 43 44
```

No `sim/` edit — reuses the brain-region framework + the neuromodulator subsystem's `from_region_firing_signed`.
