# Biologization step 2 = NEGATIVE, terminal: the symbol cannot be grounded in the substrate's activity, because the oracle lookup's load-bearing function is supplying ORTHOGONALITY -- and the substrate's concept representations are mutually overlapping

> **CORRECTION (2026-05-22, same day): the conclusion of this document
> is OVERTURNED.** This NEGATIVE concluded the composition symbol could
> not be grounded in the substrate's activity. That conclusion was
> premature: it tested an oracle-lookup replacement and (in the
> follow-on) dentate-gyrus separation, but not the obvious transform --
> removing the common-mode. The substrate's 0.45 concept-representation
> overlap is almost entirely shared common-mode; subtracting the
> across-concept mean activity (subtractive normalisation, a recognised
> cortical computation) exposes the near-orthogonal concept-specific
> structure, and a fully-biologized grounded compositional pipeline then
> clears the frozen 0.80 bar at 0.98 multi-seed. The 0.45-overlap
> MEASUREMENT here is correct; the "cannot be grounded" CONCLUSION is
> withdrawn. See
> `2026-05-22-biologized-grounded-composition-PASS-mean-centering-closes-the-arc-and-corrects-the-premature-negatives.md`.
> This document is kept as the honest trail.

## Status

The second step of the biologization arc -- replace the composition
layer's oracle-assigned symbols with symbols grounded in the
substrate's own activity. Result: NEGATIVE, and the negative is
precise. The decisive test, on the real substrate, scored below the
un-grounded baseline; a confirmatory measurement then pinned the exact
mechanism. The finding is important and biology-translatable.

## The arc

The composition layer (a Fourier Holographic Reduced Representation,
FHRR) assigns each concept its phasor symbol by oracle lookup -- a
fixed vector, unrelated to the substrate. Grounding the symbol in the
substrate's own activity was attempted in two forms.

**Naive form (earlier).** Derive the symbol directly from a single raw
substrate-activity observation. Decisive multi-seed NEGATIVE: the
substrate's per-neuron activity has a measured trial-to-trial
coefficient of variation of about 1.6, and the derived symbol does not
compose (integrated 0.33-0.42).

**Deeper form (this step).** Pass the noisy activity-derived symbol
through an attractor network whose fixed points are the consolidated
concept representations -- the attractor settle should denoise the
symbol toward a clean concept fixed point. The attractor machinery is
the one built and validated for the clean-up in biologization step 3.

A cheap-first probe found the deeper form REACHABLE: in a model with
independent per-component phase noise, the attractor recovered the
correct concept 99.6% of the time at the measured-substrate noise
level. But that probe modelled the concept symbols as random,
near-orthogonal vectors -- which, it turns out, is exactly the
assumption that fails.

## The decisive test (real substrate)

The decisive runner reuses the real captured substrate activity (the
activity-level integration cache -- the same per-neuron observations,
with whatever real structure they have), derives a symbol from each,
settles it through an attractor over the consolidated concept
representations, and composes. Multi-seed (seeds 42/43/44, 300
trials/load, frozen 0.80 bar):

```
            integrated mean    composition-only mean
L=2         0.247              0.306
L=3         0.243              0.282
L=5         0.252              0.305

VERDICT -> NEGATIVE
```

The attractor-denoised result is not only below the bar -- it is WORSE
than the un-grounded baseline (the activity-level integration scored
integrated 0.33-0.42). The attractor denoising made the result worse,
and it is flat across load at about 0.25 -- which is chance for the
4-way clean-up. That worse-than-baseline, at-chance result triggered a
confirmatory measurement.

## The confirmatory measurement -- the precise mechanism

Three quantities measured directly on the real cached activity, all
three seeds:

```
attractor recognition of an activity-derived symbol   16/256 = 0.062
raw nearest-consolidated recognition (soft argmax)     ~189/256 = 0.74
mean pairwise similarity between consolidated symbols  ~0.45
```

These three numbers settle the mechanism.

- The attractor recognises an activity-derived symbol at 0.062 --
  exactly 1/16, chance for 16 concepts, identically at every seed. The
  attractor has collapsed: it is not 16 separable basins, it is
  effectively one dominant basin, and every input settles to it.
- The attractor collapsed because the mean pairwise similarity between
  the 16 consolidated concept symbols is 0.45. They are nowhere near
  orthogonal. An attractor weight matrix formed as the outer product
  of patterns that overlap by 0.45 does not have those patterns as
  separable fixed points -- the correlated patterns merge into one
  dominant attractor.
- Yet a raw soft nearest-match (phase similarity, no attractor)
  recognises the same activity-derived symbols at 0.74. The symbols DO
  carry real concept signal. A soft comparison tolerates the 0.45
  overlap; the attractor's hard recurrent winner-take-all does not --
  it amplifies the overlap into a collapse.

## The terminal finding

The reason the symbol cannot be grounded in the substrate's activity is
now precise, and it is not (only) trial-to-trial noise. It is
orthogonality.

FHRR -- and vector-symbolic composition in general -- requires
near-orthogonal atomic symbols. Binding is phase addition and unbinding
is phase subtraction; if two symbols are not near-orthogonal, unbinding
one leaves a large crosstalk component of the other, and composition
degrades. The oracle lookup supplies orthogonality for free: a fixed
random high-dimensional vector per concept is near-orthogonal to every
other by construction. That is the oracle's load-bearing function --
not merely "a clean symbol" but "a near-orthogonal symbol."

The substrate's own concept representations do not have this property.
The consolidated activity-derived symbols overlap by 0.45 on average,
because the substrate's per-concept population activity shares a large
common-mode structure -- different concepts drive overlapping
populations. An attractor built from these overlapping representations
is degenerate; FHRR composition over them crosstalks (the un-grounded
activity-level composition-only was 0.36-0.42, already degraded by
exactly this crosstalk).

So biologization step 2 is a terminal NEGATIVE: the oracle-assigned
symbol cannot be replaced by a substrate-grounded one, because the
substrate does not produce near-orthogonal concept representations and
the composition layer requires them.

## The cheap probe's error -- recorded honestly

The cheap-first probe said REACHABLE. It was wrong, and the reason is
worth recording. The probe modelled the concept symbols as random
near-orthogonal vectors and added independent per-component noise. Both
assumptions were optimistic: the real substrate's concept
representations are neither random nor orthogonal -- they overlap by
0.45. The probe's independent-noise model also averaged out over the
dimension in a way the real correlated activity does not. This is the
second time this session a cheap probe with an idealised noise/symbol
model was over-optimistic relative to the real substrate (the
activity-level probe did the same). The honest lesson: a cheap probe
de-risks the algebra; it cannot stand in for the real substrate's
representational geometry, which must be measured.

## What this means for the biologization arc

The arc set out to replace three engineered shortcuts. The honest
outcome:

- Shortcut 1 (function-first integrator neurons -> resonate-and-fire
  neurons): biologized, PASS.
- Shortcut 3 (argmax-over-a-list clean-up -> attractor identification +
  a separate familiarity gate for abstention): biologized, RESOLVED.
- Shortcut 2 (oracle symbols -> substrate-grounded symbols): NEGATIVE,
  terminal. The oracle's function is supplying orthogonality, and the
  substrate cannot.

So the composition layer can be biologized in its neurons and its
clean-up, but not in its symbols, on this substrate. The validated
compositional capability remains a validated engineering scaffold; the
one piece that stays engineered is now precisely named -- the
near-orthogonal atomic symbols -- and the reason is precisely
characterised: the substrate's concept representations overlap.

## The routing -- next step

This relocates the open problem with precision, and the project's own
validated biology points at the fix. The substrate's concept
representations overlap by 0.45; FHRR needs them near-orthogonal. The
brain's mechanism for turning overlapping representations into
near-orthogonal ones is pattern separation, and the project has a
validated pattern-separation result: the hippocampal dentate gyrus
(catalog D.12) orthogonalises -- a validation run measured input
cosine 0.80 reduced to dentate-gyrus cosine 0.218.

The next pre-registered step is therefore: pattern-separate the
substrate's concept representations -- via the validated dentate-gyrus
mechanism / an orthogonalising transform -- and then ground the symbol
in the separated representation. The cheap-first probe: take the
substrate's consolidated concept activity vectors, apply pattern
separation, and check whether the separated symbols are near-orthogonal
AND whether they FHRR-compose at or above the frozen 0.80 bar.

## Honest scope

Subsystem-level result. The validated identity-level integration
(which uses the oracle symbols) is unaffected and stands. This finding
characterises precisely why the symbol cannot currently be grounded,
and routes to a biology-grounded fix (pattern separation) using the
project's own validated mechanism.

## Files / evidence

- Runner: `research/findings/raw/activity_level_integration_attractor.py`
- Result: `research/findings/raw/activity_level_integration_attractor.json`
- Cheap probe: `research/findings/raw/attractor_grounded_symbol_probe.py`,
  `.json` (verdict DEEPER_FORM_REACHABLE -- the over-optimistic probe)
- Design: `docs/plans/2026-05-22-activity-level-integration-design.md`
