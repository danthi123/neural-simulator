# ACh-staged recurrent excitation variant = NEGATIVE, and the negative is VERIFIED VALID by a structural-effect check (the installed recurrence genuinely transmits -- 1.41x activity spread on a supra-threshold drive -- so the verdict is NEGATIVE not VOID); the concept pools are so heavily damped that neither the ca1->concept-pool wire nor active recurrent amplification nor replay can ignite them; the dynamics-gating / wiring / amplification class of fix is now EXHAUSTED and the result converges with the prior SPEAR negative -- the compositional fix is not in network dynamics, it is in the representation

## Status

Owner-directed: work autonomously, and use external biology + open-
source research to get past the architectural blocker. The biology
research (Hasselmo SPEAR) identified acetylcholine-staged recurrent
excitation as the faithful resolution of the
trainability-vs-consolidatability tension the ca1-variant exposed.
This is the decisive test of that mechanism. Controller-only; single
seed 42; net-new code; no protected module modified; time-boxed (one
run, no iteration).

## What was tested

The ca1-variant substrate (concept pools + the 12 appended
`ca1 -> concept-pool` consolidation wires) reused with its Phase-1
checkpoint -- so Phase-1 is identical and stability is preserved by
construction. After loading, recurrent excitatory connectivity was
INSTALLED into each concept pool (30,335 edges across 12 pools;
density 0.10, weight 2.0, mirroring the validated canon motor-pool
recurrence) via `set_pathway_weights(add_missing=True)` -- the
"low-ACh release of recurrent excitation". Then: encode 4
compositional bindings, measure, run replay consolidation, measure
at 20 and 60 cycles.

## Result (pre-registered decision rule; no bar tuned)

```
| phase                       | bound-adj pool rate | selective | permuted-ctrl | mean-all-pool |
|-----------------------------|--------------------:|----------:|--------------:|--------------:|
| pre-recurrence              | 0.0024              | 2/4       | 1/4           | --            |
| post-install pre-consolidat | 0.0023              | 1/4       | 2/4           | 0.0024        |
| 20 replay cycles            | 0.0022              | 1/4       | 2/4           | 0.0018        |
| 60 replay cycles            | 0.0020              | 1/4       | 1/4           | 0.0018        |

Pre-registered verdict -> NEGATIVE.
```

Installing 30,335 weight-2.0 recurrent excitatory edges changed the
tag-stimulated pool firing by nothing (0.0024 -> 0.0023). Replay
consolidation across 0/20/60 cycles is dead flat at the noise floor.
selective 1/4 (chance), permuted control 1/4.

## The negative is VERIFIED VALID -- not a void inert-mechanism artifact

A zero-change-on-install result is ambiguous: either the recurrence
had no supra-threshold seed to amplify (valid NEGATIVE) or the
installed recurrence is functionally inert (would make the verdict
VOID). The SPEAR arc's adversarial review caught exactly this defect
class. A structural-effect check resolved it:

Drive a fixed 24% subset (48/200) of a concept pool's excitatory
neurons with a strong supra-threshold current (200 pA), measure the
WHOLE pool's firing, BEFORE vs AFTER installing the recurrence:

```
BEFORE install: whole-pool rate 0.0066
AFTER install:  whole-pool rate 0.0093   (ratio 1.41x)
-> RECURRENCE ACTIVE
```

The installed recurrence genuinely transmits: a driven subset
recurrently spreads activity to the undriven majority, a 1.41x
whole-pool rise. So the staged-recurrence variant NEGATIVE is VALID
-- the mechanism was active and the capability still did not emerge.
(`set_pathway_weights` returned n_updated = 30,335, matching the
30,335 attempted edges -- every edge installed.)

## The deeper finding: the concept pools are heavily damped

The structural check exposes the real magnitude problem. Even a
direct 200 pA drive -- the same current the validated lang_input
training uses -- on 24% of a pool's excitatory neurons yields only a
0.0066-0.0093 whole-pool firing rate. The directly-driven neurons
themselves are firing at only ~0.03. The concept pools are heavily
damped: deliberately weak region dynamics (density 0.05, exc_weight
0.3) plus FS interneuron inhibition.

The installed recurrence is active but only multiplies activity
1.41x. 1.41x of a sub-threshold signal is still sub-threshold. For a
pool to ignite into a self-sustaining attractor the recurrent loop
gain must be far higher -- which is what the canon motor-pool REGIME
provides (canon neuron parameters + canon E/I balance + canon density
+ canon weight, integrated). Adding strong recurrent EDGES onto
weak-regime NEURONS is a hybrid, not a canon pool: the edges transmit
but the weak-regime neurons do not ignite. And a true canon-regime
concept pool cannot be Phase-1-trained -- the documented "canon
amplifies bias" collapse. The trainability-vs-consolidatability
tension is in the whole pool REGIME, not just the recurrent weight;
staging the recurrent edges does not escape it.

Cranking the recurrent weight past the principled canon value (2.0)
to chase ignition would be config-cranking -- forbidden by the
discipline -- and the convergent evidence below says it is the wrong
class of fix anyway.

## Convergence with SPEAR: the dynamics-gating class is exhausted

The prior SPEAR arc tested acetylcholine phase-separation via global
synaptic-gain modulation across a theta cycle -- `full_acc = 0.00`
every rung. This staged-recurrence variant tested ACh-staged
selective recurrent excitation for consolidation -- NEGATIVE. Two
distinct ACh-gated-dynamics interventions, both honest negatives.

The compositional investigation has now tested, and exhausted, the
entire class of "fix the network dynamics" interventions:

- 8 architectures: gating, theta-multiplexing, disinhibition,
  per-regime monitoring, cue-suppression, generative replay,
  aggressive consolidation, pool-readout substitution
- difference-readout probe: the readout computation
- ca1-variant: the missing consolidation wire
- staged-recurrence variant: ACh-staged recurrent amplification

Every one operates on network dynamics / wiring / gain. Every one
hit the same wall. SPEAR's own conclusion, reached independently,
was that gating dynamics is insufficient -- composition "introduces
noise via the combination step" and the readout needs a STRUCTURED
DECODABLE object, not a sum of partially-active sub-populations.

This staged-recurrence negative is the corroborating data point. The
compositional fix is not in the network dynamics. It is in the
REPRESENTATION.

## Honest status

Compositional capability is NOT achieved. The deliverable is the
exhaustion of the dynamics-gating fix class, verified (the
structural-effect check rules out the inert-mechanism confound) and
convergent with SPEAR. The protected module set is byte-unchanged;
the no-confabulation moat is 7/7 byte-identical; no bar tuned;
reuse-by-import; no autograd. The investigation has done its job: it
has rigorously established what the fix is NOT, and points precisely
at what to try next.

## Files / evidence

- Variant runner: `research/findings/raw/ach_staged_recurrence_variant.py`
- Variant result + log: `research/findings/raw/ach_staged_recurrence_variant.{json,log}`
- Structural-effect check: `research/findings/raw/ach_recurrence_structural_check.py`
- Structural-effect result + log: `research/findings/raw/ach_recurrence_structural_check.{json,log}`
- Design: `docs/plans/2026-05-22-acetylcholine-staged-recurrence-consolidation-variant-design.md`

## Next step: phase-coded vector-symbolic composition

The genuinely-missed thread, pre-registered by the SPEAR arc itself
and never built: phase-coded vector-symbolic composition (Orchard
2023/2024 spiking-phasor / Fourier Holographic Reduced
Representations). Instead of trying to make the cortical concept
pools host a consolidated compositional attractor -- which the whole
dynamics-gating class cannot do -- the shared theta-gamma rhythm
CARRIES the composed representation as the phase of each spike within
a cycle. Bind / unbind / superposition / cleanup become operations on
a structured, decodable phase-coded object. The readout decodes a
structured object rather than reading the firing rate of a sum of
partially-active sub-populations -- which is exactly the deficiency
SPEAR localized.

This is biology-faithful: theta-gamma phase coding is observed and
well-characterized in real brains. It is the next major arc. The
immediate action is the design pass for it -- external research on
the Orchard spiking-phasor implementation and the open-source code,
then a pre-registered design doc, then build under the standard
discipline (frozen verdict module, adversarial review, decisive run,
honest propagation).
