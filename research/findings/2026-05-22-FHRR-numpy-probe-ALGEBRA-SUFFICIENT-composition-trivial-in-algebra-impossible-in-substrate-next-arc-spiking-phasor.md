# FHRR numpy reference probe = ALGEBRA SUFFICIENT: the compositional task that 8 architectures + 4 probes + 2 substrate variants of the biology-grounded substrate could not crack is solved by the FHRR vector-symbolic algebra at 100% accuracy with a 64-dimensional vector; composition is not algebraically hard, it is hard to realize in a biology-grounded spiking substrate -- and that gap is the precise statement of the project's open problem

## Status

Cheap-first falsification probe of the phase-coded vector-symbolic
composition arc (design:
`docs/plans/2026-05-22-phase-coded-VSA-composition-design.md`, commit
`1096a11`). Explicitly an ENGINEERING ceiling-clarification test --
standalone numpy, non-load-bearing, permitted under the owner's
standing rule for clearly-marked engineering baselines. It tells us
about engineering (is the FHRR algebra capable of the project's task),
not biology. Touches no protected/frozen/moat module; no autograd.

## Result (pre-registered decision rule; no bar tuned)

A standalone numpy Fourier Holographic Reduced Representation (FHRR)
implementation per Orchard & Jarvis 2023 -- symbols are unit-modulus
complex vectors; bind = phase addition, unbind = phase subtraction,
bundle = phase midpoint, clean-up = nearest-vocabulary by similarity.
Tested on the project's compositional task: 8 cue x 8 filler
symbols (project-scale 16-symbol vocabulary), bundle L facts, query
each, against the project's frozen 0.80 compositional bar; 200 random
vocab draws per cell.

```
            N=64    N=128   N=256   N=512   N=1024
L=2 facts   1.000   1.000   1.000   1.000   1.000
L=3 facts   1.000   1.000   1.000   1.000   1.000
L=5 facts   0.998   1.000   1.000   1.000   1.000

Pre-registered verdict -> ALGEBRA SUFFICIENT.
FHRR clears the 0.80 bar at ALL loads {2,3,5} at the SMALLEST
dimension tested (N=64).
```

## What this means

The FHRR algebra solves the project's compositional retrieval task
trivially. At a 64-dimensional vector -- tiny -- bundling 5 facts and
querying each recovers the bound filler at 99.8-100%. This was the
expected outcome (FHRR is designed for exactly this; Orchard & Jarvis
bundled 6 state-transitions into a 100-dim vector at >95%). The
pre-registered probe ran it anyway, on the project's exact loads,
vocabulary scale, and frozen bar, and the algebra cleared the bar
with enormous margin.

The striking and load-bearing observation is the GAP. The same
compositional task -- cue a symbol, recall the symbol bound to it --
that the project's biology-grounded substrate could not perform
across:

- 8 architectures (gating / theta-multiplexing / disinhibition /
  per-regime monitoring / cue-suppression / generative-replay /
  aggressive-consolidation / pool-readout), all ~0.46 ceiling
- the difference-readout probe
- the storage-locus, consolidation, and ca1-variant probes
- the ACh-staged-recurrence variant (verified, dynamics-gating class
  exhausted)

-- a task that proved essentially impossible in the biology-grounded
spiking substrate (compositional readout mostly at the noise floor)
-- is solved by the FHRR algebra at 100% with a 64-dimensional
vector.

**Composition is not algebraically hard. It is hard to realize in a
biology-grounded spiking substrate.** That sentence is the precise,
honest statement of the project's open scientific problem. The
vector-symbolic-architecture literature has had the algebra for
decades (Plate 1995 HRR; FHRR; Gayler 2003). What is genuinely
unsolved -- and what is the project's actual contribution to make --
is a faithful, biology-grounded, spiking realization of that algebra
that also composes with the project's distinctive trustworthy
no-confabulation property.

## Honest framing

This probe achieves NOTHING in the project's substrate. It is an
engineering reference. Its value is precise: it cleanly separates the
two questions that the eight-arc investigation kept conflating --
"is the composition ALGEBRA capable" (yes, trivially, established
here once and for all) and "can a biology-grounded spiking substrate
REALIZE that algebra" (the real, open question). With the algebra
question settled, all subsequent effort targets the biology-grounded
realization, and no future arc needs to re-litigate whether the
algebraic target is reachable.

## Discipline check

No bar tuned. Standalone numpy; no protected/frozen/moat module
imported or modified. No autograd. The probe is clearly marked
engineering ceiling-clarification per the owner's standing rule --
its insights are about engineering, not biology, and it is
non-load-bearing. Honest propagation both remotes.

## Files / evidence

- Probe: `research/findings/raw/fhrr_numpy_probe.py`
- Result: `research/findings/raw/fhrr_numpy_probe.json`
- Design: `docs/plans/2026-05-22-phase-coded-VSA-composition-design.md`

## Next arc: biology-grounded spiking-phasor implementation

Per the pre-registered decision rule, the algebra being sufficient,
the next arc is the biology-grounded spiking realization: implement
Orchard & Jarvis's spiking-phasor neuron models -- the phase-sum
(binding), phase-subtraction (unbinding), phase-midpoint (bundling),
and resonate-and-fire clean-up populations -- in the project's
simulator, and test whether a spiking-phasor compositional layer
clears the project's frozen bar at biological scale.

This is a major arc and it carries a genuine paradigm consideration,
already surfaced honestly in the design doc: a spiking-phasor FHRR
layer is a NEW representational substrate (phase-coded), not a
variant of the validated v14/v16 rate-coded concept pools. Theta-
gamma phase coding is real biology; the phasor neuron models are
function-first engineered devices. It is biology-INSPIRED
engineering. The honest open scientific questions the arc must answer:
(1) can the project's spiking simulator host the phase-sum /
phase-subtraction / phase-midpoint integrator dynamics and the RF
clean-up; (2) does a spiking-phasor compositional layer clear the
frozen 0.80 bar at biological scale and multi-seed; (3) -- the
project-distinctive one -- does it preserve the no-confabulation moat
(abstain on ungroundable queries rather than emitting a confident
clean-up to the nearest vocabulary vector).

The arc follows the standard discipline: a pre-registered design pass
pinning the spiking neuron models + the frozen capability-verdict
module + the cheap-first scaling, then build with a dedicated
adversarial review before the decisive run, then honest propagation.
The immediate next action is that design pass.
