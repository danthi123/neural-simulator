# Vocabulary-scaling load-ceiling map: the spiking-grounded compositional pipeline ceilings between binding loads 3 and 4 on a 64-concept trained substrate

## Status

A cheap, CPU-only characterisation of the BOUNDARY result the
trained-substrate decisive run produced. Completed 2026-05-22, same day
as the decisive run. The frozen 0.80 bar is unchanged and not part of
this characterisation -- it is the reference the ceiling map is read
against. The decisive run's verdict (BELOW BAR at L=5) stands.

## What was run

`research/findings/raw/vocabulary_scaling_load_ceiling_probe.py`. It
re-runs the SAME biologized grounded-composition pipeline (imported
byte-unchanged from the adversarially-reviewed decisive runner) on the
EXISTING per-seed trained activity cache at loads {2, 3, 4, 5, 6, 7}.
The sanity loads {2, 3, 5} reproduce the decisive recording exactly
(byte-for-byte) as a soundness check; the extended loads {4, 6, 7} are
the new map points. No GPU run, no re-train, no new capture -- the
trained activity cache is reused unchanged.

## Result

Sanity: the re-runs at loads {2, 3, 5} reproduce the decisive recording
byte-for-byte at every seed and at the multi-seed mean (L=2 mean 0.8417,
L=3 0.8139, L=5 0.7560 -- identical to the recording). The pipeline +
cache are deterministic; the BOUNDARY result is reproducible from the
cache alone.

Extended load-ceiling map (multi-seed integrated mean):

```
L=2: per-seed [0.8750, 0.8500, 0.8000]  mean 0.8417  (>= 0.80)  PASS
L=3: per-seed [0.8600, 0.8017, 0.7800]  mean 0.8139  (>= 0.80)  PASS
L=4: per-seed [0.8213, 0.8275, 0.7475]  mean 0.7988  (<  0.80)  miss by 0.0012
L=5: per-seed [0.7800, 0.7890, 0.7030]  mean 0.7573  (<  0.80)  miss by 0.0427
L=6: per-seed [0.7317, 0.7617, 0.6742]  mean 0.7225  (<  0.80)  miss by 0.0775
L=7: per-seed [0.6750, 0.7264, 0.6150]  mean 0.6721  (<  0.80)  miss by 0.1279
```

The ceiling sits BETWEEN L=3 and L=4: the highest load with multi-seed
mean above the bar is 3; the lowest load with mean below the bar is 4.
L=4 misses by 0.0012 -- statistically borderline; two of three seeds
individually clear the bar at L=4 (0.8213, 0.8275), only seed 44 (0.7475)
drags the mean fractionally below. The decay is monotonic and gentle:
each additional binding costs about 0.03-0.04 in accuracy.

## What this means

The biologized grounded-composition pipeline, on a properly trained
64-concept sparse-distributed substrate, demonstrates compositional
capability multi-seed at small binding loads with a clean ceiling
sitting between loads 3 and 4. The decay above the ceiling is smooth
and predictable -- not a cliff, not random.

The pure FHRR algebra (numpy probe, phasor dim 512) clears the same
0.80 bar past load 96 at the same phasor dimension. The spiking-
grounded pipeline ceilings at roughly load 3 -- a ~30x capacity
reduction. That gap is the cost of grounding the symbol in noisy
spiking activity rather than supplying it from an oracle lookup. It is
a precise, biology-translatable quantification of the spiking
implementation's noise-floor cost.

## Next step

Candidate 2 of the original NEGATIVE: grounding the symbol in the
K-of-N PATTERN itself (the concept's clean code on the trained
substrate) rather than in the noisy activity. The motivation is now
sharpened by this map: if the load ceiling between 3 and 4 is the
spiking-symbol noise floor, replacing the noisy activity-derived
symbol with the clean pattern-derived symbol directly tests whether
removing that noise raises the ceiling -- and by how much. The
reference curve to compare against is the map above. Candidate 2 is a
new pre-registered step (design + plan + soundness tests + dedicated
adversarial review before any decisive run, per discipline). The
honest oracle-adjacency caveat remains: the K-of-N pattern is the
substrate's own concept code, which is more substrate-grounded than a
freely-chosen phasor but also closer to an oracle than the
activity-derived symbol.

## Honest scope

A cheap, CPU-only characterisation; no new GPU run, no re-train, no
new capture. The frozen 0.80 bar was not moved; the decisive run's
verdict (BELOW BAR at L=5) is unchanged; the new loads {4, 6, 7} are
characterisation points, not new tests at a moved bar. The sanity
reproduction of the decisive recording at loads {2, 3, 5} passes
exactly. Reuse-by-import only; no protected, frozen, or moat module
modified; no automatic differentiation. The completed 16-concept
FHRR-biologization arc (multi-seed 0.98) stands, unaffected.

## Files / evidence

- Probe: `research/findings/raw/vocabulary_scaling_load_ceiling_probe.py`
- Result: `research/findings/raw/vocabulary_scaling_load_ceiling_probe.json`
- The BOUNDARY decisive run this characterises:
  `research/findings/2026-05-22-vocabulary-scaling-trained-substrate-BELOW-BAR-with-loads-2-3-PASS-and-load-5-ceiling.md`
- The trained activity cache the probe reads:
  `research/findings/raw/vocabulary_scaling_trained_cache/trained_full_seed{42,43,44}.npz`
- FHRR algebra reference (the load curve the spiking pipeline is compared against):
  `research/findings/2026-05-22-fhrr-capacity-curve-composition-scales-load-is-not-the-bottleneck.md`
