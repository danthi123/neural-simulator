# 160-concept ensemble vocabulary scaling at K=16, 5-seed extension: refined finding — ALL 5 bridges PASS at loads 2 and 3; bridgeD_spatial uniquely misses ONLY at load 5; seed 46 is a systematic L=5-collapse outlier across 4 of 5 bridges

## Status

Cheap pre-registered extension to the 3-seed BOUNDARY result, completed
2026-05-23. The decisive 3-seed run had 4 of 5 bridges PASS multi-seed-
mean at every load with bridgeD_spatial missing at every load; this
extension adds seeds 45 and 46 across all 5 bridges (10 new bridge-
seed combinations reusing the reviewed runner's `run_one_bridge_seed`
byte-unchanged) and reads against the same frozen 0.80 bar across
5 seeds. The pre-registered reading is **BRIDGED_ROBUST_MISS**:
bridgeD's multi-seed-mean at L=5 remains below the bar at the larger
sample (0.76, was 0.74 at 3 seeds). But the substantive refinement
is significant: at the 5-seed sample, ALL 5 BRIDGES PASS multi-seed-
mean at L=2 and L=3 (including bridgeD, which rose from 0.78/0.77 at
3 seeds to 0.81/0.80 at 5); the bridgeD miss is now L=5-specific
only. 14 of 15 (bridge, load) cells PASS multi-seed-mean; only the
bridgeD/L=5 cell misses.

## Result (pre-registered; 5-seed)

Per-bridge multi-seed-mean integrated accuracy at each load, with the
3-seed value in parentheses for comparison:

```
                                L=2              L=3              L=5
bridgeA_nouns          0.9230 (was 1.00) PASS   0.9217 (was 1.00) PASS   0.9080 (was 1.00) PASS
bridgeB_verbs          0.8595 (was 0.96) PASS   0.8470 (was 0.95) PASS   0.8436 (was 0.94) PASS
bridgeC_adj            0.8820 (was 0.83) PASS   0.8810 (was 0.83) PASS   0.8624 (was 0.82) PASS
bridgeD_spatial        0.8115 (was 0.78) PASS   0.8030 (was 0.77) PASS   0.7600 (was 0.74) MISS
bridgeE_functional     0.8870 (was 0.99) PASS   0.8890 (was 0.99) PASS   0.8790 (was 0.98) PASS
```

Per-seed L=5 across all 5 seeds:

```
bridgeA_nouns      [0.998, 0.999, 0.995, 0.948, 0.600]    1 seed below (seed 46)
bridgeB_verbs      [0.824, 1.000, 0.996, 0.996, 0.402]    1 seed below (seed 46)
bridgeC_adj        [1.000, 0.523, 0.951, 0.841, 0.997]    1 seed below (seed 43)
bridgeD_spatial    [0.780, 0.621, 0.812, 0.996, 0.591]    3 seeds below (43, 42, 46 close)
bridgeE_functional [0.999, 0.964, 0.972, 0.947, 0.513]    1 seed below (seed 46)
```

14 of 15 (bridge, load) cells PASS multi-seed-mean. Only the
bridgeD/L=5 cell misses.

Sanity contract: the 15 existing cells loaded from the decisive run's
JSON are byte-identical (the extension script combines them
unchanged with the 10 new cells); per-bridge multi-seed-mean at L=2
and L=3 changed only because the seed average now includes 5 seeds
instead of 3.

## A systematic seed-46 L=5-collapse outlier

Per-seed L=5 across bridges reveals a striking pattern: **seed 46 is
a systematic L=5 outlier at 4 of 5 bridges** (bridgeA 0.600, bridgeB
0.402, bridgeD 0.591, bridgeE 0.513; only bridgeC seed 46 cleared
strongly at 0.997). Seed 45 is uniformly strong across all bridges
at L=5 (0.948, 0.996, 0.841, 0.996, 0.947). Seed 43 is the
secondary outlier — anomalous for bridgeC (0.523) and bridgeD (0.621).

The systematic seed-46 collapse across 4 bridges is more than per-
bridge noise — it suggests a seed-specific structural effect (the
particular per-bridge pattern sets seed 46 produces interact badly
with the FHRR + attractor at 5-binding load) rather than independent
per-bridge variance. With more seeds the L=5 mean would likely
stabilise above the L=5 mean reported here (the seed-46 outlier
drops every bridge's L=5 mean by ~0.05-0.12).

## Refined interpretation

The 5-seed extension refines the BOUNDARY picture significantly:

- The K=16 PASS recipe extends per-bridge to **ALL 5 categories** at
  compositional loads 2 and 3 at the 160-concept tier. (At 3 seeds
  this was uncertain because bridgeD missed at every load including
  L=2 and L=3; at 5 seeds bridgeD now clears at L=2 and L=3 too.)
- The bridgeD miss is **load-specific**, not bridge-specific in the
  general sense. bridgeD's L=5 multi-seed-mean is 0.76 — about
  0.08-0.10 below the other bridges' L=5 means (0.84-0.91).
- The L=5 variance across all 5 bridges is driven primarily by two
  seed-specific outliers (seed 43 and seed 46), where seed 46
  collapses L=5 at 4 of 5 bridges in a systematic way.
- Per the strict pre-registered bar (every cell ≥ 0.80), the
  verdict is BELOW BAR -- but only on 1 of 15 cells.

The biology-translatable insight set:
- The K=16 PASS recipe (longer temporal integration on top of
  mean-centred symbols) extends per-bridge to all 5 vocabulary
  categories at the 160-concept tier at the lower compositional
  loads (2-3 bindings).
- At higher compositional loads (5 bindings), per-seed structural
  variation in the substrate's pattern sets interacts with FHRR
  crosstalk; some seeds produce uniformly weak L=5 performance
  across the ensemble (the systematic seed-46 outlier).
- bridgeD_spatial uniquely fails to fully clear at L=5 even averaging
  across 5 seeds (0.76); this is a residual category-specific deficit
  but it is now well-characterised (load-specific, not load-uniform).

## What this is, and what it is not

This is a refined characterisation of the 160-ensemble result, not a
new pre-registered test. The verdict per the strict pre-registered
bar is BELOW BAR; the BOUNDARY pillar in capability_status (n=91)
stands and is sharpened by the 5-seed breakdown. It is NOT a
capability claim that the K=16 recipe extends unconditionally to
160 concepts — bridgeD/L=5 still misses, and the other bridges at
L=5 have non-trivial per-seed variance (especially the seed-46
collapse).

It IS a refined biology-translatable finding: at the 160-concept
tier, the activity-grounded biologized pipeline at K=16 clears the
0.80 bar multi-seed-mean across all 5 categories at compositional
loads 2 and 3; the failure mode at load 5 is concentrated on a single
category (spatial) plus a systematic seed-46 anomaly that affects
4 of 5 bridges.

## Next step

The refined picture suggests two natural follow-up directions, with
the cheap-first choice clearer than at the 3-seed result:

(a) **Diagnose the systematic seed-46 collapse.** Seed 46 collapses
    L=5 across 4 of 5 bridges (the only exception is bridgeC which
    scored 0.997). This is more than per-bridge noise -- it suggests
    the seed-46 random pattern sets across all 5 bridges share some
    structural property that interacts badly with FHRR composition
    at high binding load. A cheap CPU diagnostic: measure the
    pairwise concept-pattern overlap for each bridge at each seed
    -- if seed 46 has systematically higher pattern overlaps across
    bridges, that connects the failure to a measurable substrate
    property. ~minutes, CPU.

(b) **Diagnose bridgeD's L=5 residual.** Even excluding the seed-46
    outlier, bridgeD/L=5 is the weakest cell (per-seed: 0.780, 0.621,
    0.812, 0.996; excluding outlier seed 46 the mean is 0.802 -- just
    at bar). What's specific about bridgeD's spatial concepts that
    makes them harder to compose at high load when the symbol-input
    geometry is identical to other bridges? A cheap CPU probe:
    measure per-concept-pair compositional accuracy on bridgeD vs
    other bridges, isolate the failure pairs, examine their pattern
    overlap. ~minutes, CPU.

Either probe is cheap. (a) is the broader phenomenon (4 of 5 bridges
collapse on seed 46); (b) is the narrower bridge-specific question.
Given the broader pattern (a) is genuinely interesting, it likely
yields more insight per minute of investigation.

(Broader horizon, surfaced for the owner, NOT auto-launched: the
owner's standing conversational-path directives -- SPEAR, theta-
gamma mode-unification, generative replay -- and the integrated
closed loop are the larger arcs. The vocab-scaling thread is now
substantially characterised across 16, 64, and 160-concept tiers
with the per-bridge and per-load and per-seed breakdowns mapped; the
seed-46 diagnostic would close out the per-seed-variance story and
then the larger arcs may be the higher-leverage direction.)

## Honest scope

A refined characterisation of the BOUNDARY result via a focused
extension run. The frozen 0.80 bar was not moved. The reviewed
160-ensemble runner is byte-unchanged; the extension reuses
`run_one_bridge_seed` byte-unchanged. No protected, frozen, or moat
module modified. No automatic differentiation. The mandatory smell-
test integrity (sanity contract that the 15 existing cells are
byte-identical to the decisive run) is the script's first check.
The capability_status BOUNDARY pillar (n=91) for the 3-seed result
stands; this 5-seed extension is recorded as a characterisation
refinement, not a new pillar. The 64-concept K=16 refined CAPABILITY
PASS (n=90) and the 16-concept validated capability stand unchanged.
The no-confab moat is 7/7 green.

## Files / evidence

- Extension script:
  `research/findings/raw/vocabulary_scaling_run_160ensemble_extra_seeds.py`
- Result: `research/findings/raw/vocabulary_scaling_run_160ensemble_5seeds.json`
- Reused (byte-unchanged): the reviewed 160-ensemble runner
  `vocabulary_scaling_run_160ensemble.py` and its 15 existing
  bridge-seed activity caches at seeds 42/43/44.
- New activity caches (10 new bridge-seed combinations at seeds 45
  and 46) under
  `research/findings/raw/vocabulary_scaling_160ensemble_cache/full_*_seed{45,46}.npz`.
- The 3-seed BOUNDARY this refines:
  `research/findings/2026-05-23-160-concept-ensemble-K16-BOUNDARY-4-of-5-bridges-PASS-multiseed-bridgeD-uniquely-misses-with-honest-perseed-caveats.md`
- The 64-concept K=16 PASS this builds on:
  `research/findings/2026-05-22-vocabulary-scaling-trained-substrate-Kvocab16-PASS-activity-grounded-clears-the-bar-at-all-loads-with-thin-L5-margin.md`
