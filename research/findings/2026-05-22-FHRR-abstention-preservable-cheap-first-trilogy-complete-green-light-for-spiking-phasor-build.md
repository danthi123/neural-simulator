# FHRR abstention probe = ABSTENTION PRESERVABLE: a fixed similarity threshold separates groundable from ungroundable queries at 100% accuracy at every load under biological-precision spike-timing jitter; the no-confabulation moat -- the project's distinctive trustworthy contribution -- survives FHRR composition; the cheap-first trilogy (algebra / noise / abstention) is complete and all green, a de-risked green light for the biology-grounded spiking-phasor build

## Status

Third and final cheap-first probe of the phase-coded vector-symbolic
composition arc (design:
`docs/plans/2026-05-22-phase-coded-VSA-composition-design.md`).
Standalone numpy; engineering ceiling-clarification, non-load-bearing;
no protected/frozen/moat module touched; no autograd.

## Result (pre-registered decision rule; no bar tuned)

FHRR clean-up is an argmax over the vocabulary -- it always returns a
nearest vector, which by construction is confabulation. But FHRR
exposes a natural abstention signal: the clean-up TOP-SIMILARITY. This
probe measures whether groundable and ungroundable queries separate in
that signal, under biological-precision spike-timing jitter
(sigma = 0.05 of a cycle).

```
dim 512; jitter sigma 0.05; 400 trials/load

| load | groundable sim (mean / min) | ungroundable sim (mean / max) | separation acc |
|------|-----------------------------|-------------------------------|----------------|
| L=2  | 0.472 / 0.392               | 0.045 / 0.126                 | 1.0000         |
| L=3  | 0.390 / 0.288               | 0.044 / 0.119                 | 1.0000         |
| L=5  | 0.299 / 0.191               | 0.045 / 0.122                 | 1.0000         |

Pre-registered verdict -> ABSTENTION PRESERVABLE.
```

The groundable and ungroundable top-similarity distributions are
cleanly separated at every load. Even at the hardest load (L=5) the
minimum groundable similarity (0.191) sits well above the maximum
ungroundable similarity (0.122) -- any fixed threshold in the
~0.13-0.19 gap classifies every query correctly. A composition layer
built this way emits the clean-up answer above the threshold and
abstains below it: that IS a no-confabulation moat.

## The cheap-first trilogy is complete -- all three green

The phase-coded VSA arc was staged as three pre-registered cheap-first
falsification probes before any heavy build. All three are answered,
all positive:

1. **FHRR numpy probe -> ALGEBRA SUFFICIENT.** FHRR composition
   clears the project's frozen 0.80 compositional bar at loads
   {2,3,5} at a 64-dimensional vector (100% / 100% / 99.8%).

2. **Spiking-phasor realization probe -> NOISE TOLERANT.** Realizing
   every phasor as a noisy spike (quantized + Gaussian jitter)
   accumulated through the full encode/query chain, FHRR still clears
   0.80 at all loads at biological-precision jitter (sigma = 0.05,
   ~6 ms on a 125 ms theta cycle); 100% at N=256.

3. **Abstention probe -> ABSTENTION PRESERVABLE (this finding).** A
   fixed similarity threshold separates groundable from ungroundable
   at 100% accuracy at every load under jitter -- the no-confabulation
   moat survives FHRR composition.

Together: the FHRR algebra solves the compositional task; the
spiking-phasor realization survives realistic spike-timing noise; and
the project's distinctive trustworthy property -- abstaining rather
than confabulating -- is preservable. This is a de-risked green light
for the biology-grounded spiking-phasor build.

## What is and is not established

Established (cheap-first, engineering ceiling-clarification): the
phase-coded VSA target is reachable in principle and robust to the
two things that could have killed it (spike-timing noise; loss of
abstention). The eight-architecture investigation never had this --
it kept conflating "is the target reachable" with "can the substrate
reach it". The target is now proven reachable.

NOT established: anything in the project's actual spiking substrate.
These probes are standalone numpy. The real arc -- the biology-
grounded build -- is the open work: implement Orchard & Jarvis's
spiking-phasor neuron models (phase-sum, phase-subtraction,
phase-midpoint integrators; resonate-and-fire clean-up with mutual
inhibition) as genuine time-stepped spiking units, validate they
compute the FHRR operations as real spiking neurons, then integrate
at biological scale.

## Discipline check

No bar tuned. Standalone numpy; no protected/frozen/moat module
imported or modified; no autograd. Clearly-marked engineering
ceiling-clarification per the owner's standing rule. Honest
propagation both remotes.

## Files / evidence

- Probe: `research/findings/raw/fhrr_abstention_probe.py`
- Result: `research/findings/raw/fhrr_abstention_probe.json`
- Trilogy: `fhrr_numpy_probe.{py,json}` + `spiking_phasor_fhrr_probe.{py,json}`
  + this probe.

## Next step (proceeding -- the build)

The cheap-first gates are passed. The next step is the build: a
standalone time-stepped spiking-phasor FHRR module implementing
Orchard & Jarvis's neuron models as genuine integrate-style spiking
units -- the phasor, phase-sum, phase-subtraction, phase-midpoint, and
resonate-and-fire clean-up neurons -- with a self-test that runs a
small FHRR network in true spiking simulation and verifies the
operations. This module is the foundation for, and the reference
implementation of, the subsequent biological-scale bridge
integration. Proceeding into it now.
