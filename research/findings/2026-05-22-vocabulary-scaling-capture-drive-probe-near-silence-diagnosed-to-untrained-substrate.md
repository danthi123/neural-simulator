# Vocabulary-scaling capture-drive probe: the 64-concept run's near-silent activity is the untrained substrate, not a weak capture drive

## Status

A diagnostic probe arc, completed 2026-05-22. It investigates why the
64-concept vocabulary-scaling decisive run was a NEGATIVE. The earlier
finding diagnosed the cause as near-silent captured activity; this arc
localises that near-silence precisely, retracts one flawed probe, and
identifies the corrective next step.

## Background

The biologized grounded-composition pipeline -- longer-integration
concept recognition, common-mode-removed grounded symbols,
resonate-and-fire phase-coded composition, an attractor clean-up with a
separate familiarity gate -- clears the project's frozen 0.80
compositional bar at multi-seed 0.98 on the validated 16-concept
substrate. The vocabulary-scaling test asked whether it still clears
that bar at 64 concepts, on the sparse-distributed concept ensemble:
each concept is a scattered pattern of 100 random neurons inside one
shared 2000-neuron pool.

The decisive run was a NEGATIVE (integrated multi-seed about 0.11). The
diagnosis: the activity captured from the 64-concept bridge was
near-silent -- roughly ten to fifteen times sparser than the activity
the pipeline was validated on -- so the grounded symbols, derived from
near-silent and therefore noise-dominated activity, did not compose.
The NEGATIVE doc noted the bridge had been built fresh and not trained,
and flagged that as a possible setup gap. This arc resolves it.

## What was run

Three cheap GPU diagnostic probes, each answering one pre-registered
question.

**Probe 1** swept the teacher current (the bias current applied to a
concept's pattern neurons during capture) over [100, 300, 600, 1000,
2000] pA on a reduced-scale bridge (1000-neuron pool) and measured the
resulting pool activity density.

**Probe 2** ran a controlled comparison at the decisive run's exact
full scale (8192 language-input neurons, a 2000-neuron pool, 300
fast-spiking interneurons). For six concepts it captured pool and
interneuron activity under three drive conditions: the teacher current
alone; the language-input drive plus the teacher current (the decisive
run's exact capture drive); and the language-input drive alone. It also
swept the teacher current under the full drive.

**Probe 3** built the same full-scale bridge, applied the validated
sparse topographic prior (which boosts each concept's language-input
connections onto its own pattern and dampens its connections onto other
neurons), and measured pool density and pattern selectivity.

## Results

The decisive run's captured density, recomputed directly from the
recording, is 0.0077 pool-fraction-nonzero (mean rate 0.00024). The
validated 16-concept substrate the pipeline passed on sits at 0.075.

```
Probe 1 (REDUCED scale, 1000-neuron pool):
  teacher 100 pA  -> pool nonzero 0.079      verdict: DRIVE_GAP_RECOVERABLE
  teacher 1000 pA -> pool nonzero 0.111

Probe 2 (FULL decisive-run scale, 2000-neuron pool):
  teacher only            -> pool nonzero 0.0026
  language-input+teacher  -> pool nonzero 0.0041   (reproduces the run)
  language-input only     -> pool nonzero 0.0040
  teacher sweep under the full drive: 100->0.0035, 600->0.033, 2000->0.052

Probe 3 (FULL scale, validated topographic prior):
  fresh,  language-input+teacher : pool 0.0041  own-pattern recruit 0.024
  primed, language-input only    : pool 0.0230  own-pattern recruit 0.062
  primed, language-input+teacher : pool 0.0189  own 0.135  off-pattern 0.018
```

## Probe 1 retracted -- a scale artifact

Probe 1 reported that a stronger teacher current recovers the activity
density and concluded the NEGATIVE was a too-weak capture drive. A
smell-test falsified this. Probe 1 recorded 0.079 pool-nonzero at the
decisive run's exact 100 pA teacher current -- but the decisive run's
recomputed density at that same 100 pA is 0.0077, fifteen times lower.
The two cannot be measuring the same system.

The difference is scale. Probe 1 used a 1000-neuron pool; the decisive
run a 2000-neuron pool. The pool's feedback-inhibition loop -- the pool
excites the fast-spiking interneurons, the interneurons inhibit the
pool -- has a loop gain that grows with pool size, because each neuron
collects more inhibitory connections in a larger pool. The full-scale
pool is therefore much harder to drive into activity. Probe 2 confirms
this directly: at full scale, teacher-only at 100 pA gives 0.0026, not
0.079. Probe 1's verdict is a scale artifact and is retracted; a
retraction notice is in the probe file.

## The near-silence is the untrained substrate

Probe 2 isolates the cause. At the decisive run's full scale, all three
drive conditions are near-silent: teacher only, language-input plus
teacher, and language-input alone all give about 0.003-0.004
pool-fraction-nonzero. The language-input drive is not a suppressor and
the teacher current is not the missing lever -- the whole freshly-built
substrate is near-silent under every capture drive.

Probe 2's teacher sweep shows a stronger teacher does raise density
(2000 pA reaches 0.052), but only by force-firing the concept's pattern
neurons directly. At that point the captured activity is essentially
the pattern itself, driven by the teacher rather than evoked by the
concept's natural input -- which makes the grounded symbol a relabelled
copy of the pattern, the same oracle shortcut the biologization arc was
built to remove. A stronger teacher is therefore not an honest fix.

Probe 3 tests the real fix. On a fresh bridge the language-input pathway
into the pool is random and non-selective: no concept's natural drive
preferentially evokes its own pattern. The validated sparse topographic
prior installs that selectivity. Applying it lifts pool density from
0.004 to about 0.02 and own-pattern recruitment from 2.4 percent to 13.5
percent, with the concept's own pattern firing 7.7 times more than other
concepts' patterns -- a real, large, and selective improvement. But the
prior-alone density (0.019) is still below the 0.04 proxy for "density
comparable to the validated substrate". That proxy was held fixed and
not moved.

The prior is only the structural half of exercising the substrate. The
validated sparse-distributed substrate is also trained: a spike-timing
encoding stage grows the language-input-to-pattern weights well beyond
the prior's static boost. Probes 1-3 tested a fresh bridge, a
prior-primed bridge, and stronger teacher currents -- they did not test
a fully trained bridge.

## What this is, and what it is not

This is a complete diagnosis of the vocabulary-scaling NEGATIVE's
near-silent activity: the cause is the untrained substrate, exactly as
the NEGATIVE doc stated. The probe arc confirmed it, retracted a flawed
scale-artifact probe, and ruled out the two cheap fixes -- a stronger
teacher current (oracle-adjacent) and the topographic prior alone
(insufficient density).

It is NOT a demonstration that the sparse-distributed substrate cannot
ground a compositional symbol. The fully trained substrate -- the one
the design doc named as the test substrate -- has not yet been tested.
The probes have de-risked that test: the prior alone already restores
selective, concept-specific activity (7.7x own-vs-other), and the
training stage only sharpens it further.

## Next step

Re-run the pre-registered 64-concept vocabulary-scaling test, capturing
from a fully trained sparse-distributed substrate. A new pre-registered
runner inserts the validated encoding -- the sparse topographic prior
plus the validated per-concept spike-timing training, both reused
unchanged from the validated substrate module -- before the activity
capture, then runs the biologized grounded-composition pipeline against
the frozen 0.80 bar, multi-seed, at composition loads {2,3,5}. The bar
is unchanged; this puts the substrate into the state the design doc
specified, correcting the diagnosed setup gap. It is not config-cranking
a NEGATIVE: the bar is frozen and the substrate is being exercised, not
the test re-tuned.

Pre-registered reading: if the trained-substrate capture composes at or
above 0.80 multi-seed at all loads, the biologized compositional
capability scales to a 64-concept vocabulary. If it does not, the honest
finding is that the activity-grounded pipeline needs a denser substrate
than the sparse pool provides even when trained, and grounding the
symbol in the sparse pattern itself is weighed honestly against whether
that is still substrate-grounded or closer to an oracle lookup.

## Honest scope

A diagnostic arc, not a capability result. No pre-registered bar was
moved; the 0.04 density figure is a diagnostic proxy and was held fixed.
One probe was retracted honestly when a smell-test caught it as a scale
artifact. No protected, frozen, or moat module was modified; the
substrate builder, the topographic prior, and the orthogonal-drive
helper were reused by import. No automatic differentiation. The
completed, twice-reviewed 16-concept FHRR-biologization arc (multi-seed
0.98) stands, unaffected.

## Files / evidence

- Probes: `research/findings/raw/g20_capture_drive_probe.py` (v1,
  retracted), `g20_capture_drive_probe_v2.py`,
  `g20_capture_drive_probe_v3.py`
- Probe outputs: the matching `.json` files
- The decisive-run NEGATIVE this arc diagnoses:
  `research/findings/2026-05-22-vocabulary-scaling-64concept-NEGATIVE-G20-sparse-activity-too-sparse-for-the-activity-grounded-pipeline.md`
- Design + plan:
  `docs/plans/2026-05-22-vocabulary-scaling-design.md`,
  `docs/plans/2026-05-22-vocabulary-scaling-implementation.md`
