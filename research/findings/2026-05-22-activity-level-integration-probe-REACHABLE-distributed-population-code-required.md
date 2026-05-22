# Activity-level integration cheap-first probe = REACHABLE, with a precise design constraint: a distributed per-neuron population code works, the coarse per-pool aggregate does not

## Status

Pre-registered cheap-first probe for the activity-level integration
arc. Result: PASS (activity-level integration is reachable) -- with a
precise, biology-translatable design constraint that the probe
surfaced and that the real-substrate build must respect.

## The question

The validated two-system compositional pipeline joins the project's
concept-recognition substrate to the spiking-phasor composition layer
at the **concept-identity level**: the substrate reports one discrete
recognized label, and a fixed lookup table maps that label to a
pre-assigned phasor symbol. The substrate's graded population activity
never itself enters the composition layer.

The more biologically faithful interface -- activity-level integration
-- would **derive** the phasor symbol from the population activity
vector directly: no discrete label, no lookup table. The honest new
failure mode is that a population activity vector has trial-to-trial
variability, so the activity vector observed when a fact is stored and
when it is queried differ even for the same concept, and the derived
symbols therefore differ, leaving a residual error after unbinding.

The probe (`research/findings/raw/activity_level_integration_probe.py`,
standalone numpy, engineering ceiling-clarification, non-load-bearing)
models this: concept activity centroids, per-observation Gaussian
noise, a fixed random complex-projection derivation of activity to
phasor, FHRR store/query/clean-up, with storage and query drawing
independent activity noise.

## Pre-registered reading (fixed before the run, never tuned)

PASS if the probe clears the frozen 0.80 compositional bar at all loads
{2, 3, 5} at every activity-noise standard deviation <= 0.10 (about a
20% coefficient of variation on a mean firing rate near 0.5 -- within
realistic cortical trial-to-trial rate variability) at some phasor
dimension <= 1024.

## Result

```
activity-dim D=16  (coarse per-pool vector):
  phasor N=256:   L5  s0.00=0.860  s0.05=0.679  s0.10=0.524  s0.20=0.292
  phasor N=1024:  L5  s0.00=0.854  s0.05=0.740  s0.10=0.533  s0.20=0.351

activity-dim D=256 (distributed population code):
  phasor N=256:   L2  s0.10=0.998   L3 s0.10=0.987   L5 s0.10=0.896
  phasor N=1024:  L2  s0.10=1.000   L3 s0.10=0.995   L5 s0.10=0.979

VERDICT -> ACTIVITY-LEVEL REACHABLE (D=256, N=256)
```

The probe clears the bar at all loads at every activity-noise level up
to and including the pre-registered 0.10 threshold, at the distributed
256-dimensional activity representation. PASS.

## The design constraint the probe surfaced

The PASS is real but conditional, and the condition is the finding:

- **The coarse 16-dimensional per-pool activity vector FAILS.** Even at
  the largest phasor dimension tested (1024), the per-pool vector
  drops to 0.53 at load 5 at the 0.10 noise level -- far below the bar.
  A 16-dimensional activity vector has only 16 effective degrees of
  freedom; the noise on those 16 numbers projects into the derived
  symbol with no redundancy to average against.

- **The distributed 256-dimensional population code PASSES decisively.**
  Each concept activates roughly 16 of 256 dimensions; the fixed random
  projection that derives the symbol averages the activity noise across
  many dimensions, so the derived symbol is stable enough for binding
  and unbinding to survive. Load 5 at the 0.10 noise level is 0.90 at
  phasor dimension 256 and 0.98 at 1024.

This is a biology-translatable result. The substrate's **per-neuron**
population activity carries enough redundancy to denoise an
activity-derived symbol; the **per-pool aggregate** (which is close to
what an argmax label already captures) does not. The discrete-label
lookup of the validated interface was effectively doing a denoising
job -- and a faithful activity-level interface recovers that denoising
not by a discrete bottleneck but by using the full distributed
population code, where redundancy averages the noise out.

There is also a clean noise ceiling: at activity noise 0.20 (a 40%
coefficient of variation, beyond the realistic threshold) even the
distributed code falls to 0.52-0.69 at load 5. The pre-registered
<= 0.10 regime is the validated operating range.

## Smell test (a PASS scrutinised harder than a FAIL)

- Above chance: clean-up is over 8 fillers, chance 0.125; the result is
  0.90-1.00. Far above chance.
- Noise genuinely in the loop: accuracy degrades monotonically with the
  activity noise (1.00 -> 1.00 -> 0.98 -> 0.69 at load 5, distributed
  code) -- a graded response, not a fluke.
- Storage and query noise are genuinely independent draws (`observe()`
  is called separately for store and query, advancing the same random
  generator) -- the realistic case, verified in the code.
- The coarse-code failure has a mechanistic explanation (16 degrees of
  freedom, no redundancy) consistent with the distributed-code success
  (256 dimensions, ~16-fold redundancy). Not an artifact.
- The clean-up vocabulary is derived from noise-free centroids -- the
  consolidated, stable concept identity -- while storage and query use
  noisy in-flight activity. This asymmetry is the realistic model (a
  stable semantic dictionary vs live noisy activity) and is stated
  honestly.

## What this means for the real build

Activity-level integration is reachable, so the arc proceeds to design
and build the real activity-level integration runner under the full
standard discipline. The probe hands that build one hard requirement:
the activity vector fed to the symbol derivation must be the
substrate's **distributed per-neuron** concept-pool population
activity, not the coarse per-pool firing-rate aggregate. The project's
existing pool-firing readout returns the per-pool aggregate; the real
runner needs the per-neuron firing-rate vector of the concept-pool
population.

The real runner will, per the standard discipline: reuse the validated
recognition substrate and the validated spiking-phasor composition
subsystem by import, both byte-unchanged; capture the per-neuron
concept-pool activity; register a clean-up vocabulary by averaging the
activity over a few registration observations per concept; derive
phasor symbols from the live activity; measure against the frozen 0.80
bar with a pre-registered verdict, a dedicated adversarial review, and
honest propagation to both git remotes.

## Honest scope

This is one cheap-first numpy probe and a pre-registered decision. It
does not by itself build anything load-bearing; it de-risks the
activity-level integration arc and hands the real build a precise
design constraint. The validated identity-level integration (multi-seed
PASS, adversarial review CLEAR, capability pillar recorded) stands
regardless. The probe says only that a more biologically faithful
interface is reachable on top of it, provided the distributed
population code is used.

## Files / evidence

- Probe: `research/findings/raw/activity_level_integration_probe.py`
- Result: `research/findings/raw/activity_level_integration_probe.json`
- Design: `docs/plans/2026-05-22-activity-level-integration-design.md`
