# Recognition-bound probe = the recognition bound is reducible: temporal averaging lifts concept recognition from 0.67 to 0.96

## Status

The cheap-first probe for the routing out of the compositional line's
convergent finding. The whole compositional line -- the validated
identity-level integration, the biologization arc, the activity-level
and pattern-separation work -- converges on one bound: the substrate's
concept-recognition accuracy, where a single 100-step activity capture
recognizes a concept word only about 0.67-0.74 of the time. This probe
asked whether that bound is reducible, cheaply, and the answer is yes:
temporal averaging lifts recognition to 0.96.

## What was tested

The probe reuses the real captured substrate activity (the
activity-level integration cache: 16 per-neuron observations of each of
16 concept words, three seeds) -- no new simulation run -- and tests
three things.

## Result (multi-seed 42/43/44)

**(a) Temporal averaging.** Averaging the per-neuron activity over K
observations of a word, before the per-pool argmax recognition:

```
K = 1    recognition 0.667
K = 2    recognition 0.795
K = 4    recognition 0.878
K = 8    recognition 0.934
K = 16   recognition 0.958
```

Recognition rises monotonically and smoothly with the number of
averaged observations. By K = 4 it is already above 0.85; by K = 16 it
is 0.958. The pre-registered target (0.85 by K = 16) is cleared with
margin.

**(b) Word fragility.** The single-observation recognition error is not
broadly concentrated -- only 2 of 16 words ("go" at 0.33, "stop" at
0.48) stay fragile. The other 14 words recognize well once averaged.
The residual ~4% gap from 1.0 at K = 16 is essentially those two
genuinely-entangled concepts.

**(c) Capture drift.** Recognition at observation index 0 versus
observation index 15 has a slope of +0.000 across the 16-observation
capture sequence. The substrate's internal state does not drift across
the capture; the per-observation variability is intrinsic trial-to-
trial noise, not an artifact of capturing many observations in
sequence.

## Why this works, and why it is legitimate

The substrate's concept-pool activity is a rate code with substantial
trial-to-trial variability (coefficient of variation about 1.6). A
rate code with zero-mean trial noise has a signal that survives
averaging while the noise falls as roughly one over the square root of
the number of samples. The observed lift -- 0.667 to 0.958 over K = 1
to 16 -- is exactly the shape that noise-averaging predicts.

Biologically this is a longer integration window: observing a concept
for longer, or sampling it across several theta cycles, and averaging
the rate estimate. It is the analogue of sustained attention. It does
not change the substrate or its representations; it changes only how
long the readout integrates. So it is a legitimate readout improvement,
not a substrate change and not a cheat.

The capture-drift result (slope 0.000) is important for honesty: it
confirms the per-observation noise is genuine intrinsic variability, so
averaging it down is a real effect, not the removal of a capture
artifact.

## What this means

The compositional capability is recognition-bounded. This probe shows
the recognition bound is itself reducible: a recognition front-end that
integrates the substrate's activity over a longer window (a modest
4-to-16-fold) recognizes concepts at 0.88-0.96 instead of 0.67. This
lifts the bound without touching the substrate.

It also reopens shortcut 2 (grounding the composition symbol)
constructively. The pattern-separation probe showed that the
dentate-gyrus mechanism orthogonalises the substrate's overlapping
concept representations into composable symbols (overlap 0.43 to 0.17,
composition 1.000), and that the only obstacle was recognition -- which
cannot be done by separating a noisy observation. The honest pipeline
is now clear: integrate the activity over a longer window to recognize
the concept (0.96), then use that concept's pattern-separated grounded
code as the symbol. Every stage is biological -- a longer-integration
rate readout, dentate-gyrus pattern separation, resonate-and-fire FHRR
composition, an attractor clean-up with a familiarity gate -- and the
pipeline is recognition-bounded at about 0.96 rather than 0.67.

## Next step

Build the fully-biologized grounded compositional pipeline on the
cached activity: longer-integration recognition, the recognized
concept's dentate-gyrus pattern-separated code as the grounded symbol,
resonate-and-fire FHRR composition, attractor clean-up. Test against
the frozen 0.80 compositional bar. This is the constructive close of
shortcut 2 -- a composition layer that is biological end to end, with
no oracle symbol table, recognition-bounded.

## Honest scope

Cheap-first probe (numpy, reuses the real activity cache). It
establishes that temporal averaging lifts recognition; it does not by
itself build the pipeline. The two fragile words ("go", "stop") remain
hard even averaged -- a small residual that is the substrate's concept
representation, consistent with the pattern-separation finding that a
few concepts are irreducibly entangled.

## Files / evidence

- Probe: `research/findings/raw/recognition_bound_probe.py`
- Result: `research/findings/raw/recognition_bound_probe.json`
