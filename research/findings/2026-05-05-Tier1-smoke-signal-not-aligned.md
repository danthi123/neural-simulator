# Tier 1 embodied-Hebbian smoke — real signal, not yet aligned

**Date:** 2026-05-05 ~22:50 EDT
**Status:** Smoke test reveals real signal but doesn't pass permuted-label
control. Infrastructure works; parameter tuning needed before 6-seed
validation.

---

## What ran

```bash
python -m research.runners.bio_three_factor \
    --biological --embodied-hebbian \
    --apply-topographic-bias --enable-motor-fs \
    --n-events-per-direction 100 --seed 42
```

Configuration:
- 400 training events (100/word × 4 words)
- ~7 min wall-clock training (444s)
- Embodied teacher signals: language_input + language_output + motor target
- Bridge built-in STDP fires on co-active synapses
- Topographic prior applied to `language_input → motor_X` only (NOT
  to the new `motor_X → language_output` reciprocal pathway)

## Results

### W→A direction: 29% true, 34% best perm (WESN), NOT aligned

```
Confusion (W→A):
  north -> {N=4, E=3, S=9, W=9}    -- W predicted more than N (wrong)
  east  -> {N=6, E=10, S=4, W=5}   -- E correct most-often (40%)
  south -> {N=5, E=5, S=8, W=7}    -- S correct most-often, barely
  west  -> {N=7, E=4, S=7, W=7}    -- N predicted most-often (wrong)
```

Best permutation: WESN. True: NESW. Excess: +5pp.

**Real signal in "east" direction**: drive deltas show E went +47
(strong), N went -35, S went -15, W went -9. That's clean
differentiation. But other directions don't show this pattern.

### A→W direction: 23% true, 35% best perm (WNES), NOT aligned

```
Confusion (A→W):
  motor_N -> {north=16, east=6, south=1, west=2}    -- N→"north" 64%! 
  motor_E -> {north=18, east=5, south=1, west=1}    -- E→"north" (wrong)
  motor_S -> {north=16, east=7, south=0, west=2}    -- S→"north" (wrong)
  motor_W -> {north=12, east=3, south=8, west=2}    -- W→"north" (wrong)
```

Best permutation: WNES. True: NESW. Excess: +12pp.

**Strong N→north signal (16/25 = 64%) BUT N-bias dominates everywhere**:
every motor pool predicts "north" most-often. The N→north 64% looks
like real differentiation but isn't — the network has a structural
bias toward "north" that all motor inputs converge to.

## Diagnosis

The infrastructure WORKS:
- ✅ language_output region built correctly
- ✅ Reciprocal motor_X → language_output pathway installed
- ✅ Bridge STDP fires during training (no custom 3-factor needed)
- ✅ Plasticity gates open during training, frozen for eval
- ✅ Both W→A and A→W evals execute cleanly

But Tier 1 isn't passing:

1. **Topographic prior asymmetry.** `language_input → motor_X` has the
   biology-canon topographic prior (1.5x for target, 0.7x for off-
   target). The reciprocal `motor_X → language_output` has NO such
   prior — all edges start at weight_mean=0.5±0.3, uniform across all
   word neurons.

   Result: motor → language_output activity is non-specific. STDP
   alone in 400 trials can't differentiate the 4 outputs above
   the architectural N-bias floor.

2. **Reciprocal pathway weights too weak.** weight_mean=0.5 vs
   text_input_to_motor_weight=3.0. The reverse pathway is 6× weaker
   than the forward pathway. STDP would need many more events to
   build comparable specificity.

3. **N-bias floor.** Same architectural quirk we've documented
   throughout: "north"-coded patterns naturally dominate language_output
   activity from random init. Without strong differentiating
   pressure, all motor pools converge to firing "north" patterns.

4. **Training duration.** 100 events/word × 4 = 400 trials may be
   too short to consolidate STDP-only learning, especially with the
   weak reverse pathway.

## What to fix for Tier 1.4

Three fixes to try (in priority order):

### Fix A: Topographic prior on motor → language_output

Apply same Pulvermüller-style somatotopic prior to the reciprocal
pathway: motor_N output edges to "north"-encoded neurons in
language_output get weight × 1.5; off-target get × 0.7.

This requires extending `apply_topographic_bias` to handle the
output direction.

**Cost:** ~2 hours implementation + smoke retest.

### Fix B: Bump reciprocal pathway weight_mean

Change motor_to_language_output_weight from 0.5 → 2.0. Closer to
the forward pathway. Stronger initial drive, more STDP events
on differentiating edges.

**Cost:** 1 line config change + smoke retest.

### Fix C: More training

200-500 events/word. Lets STDP accumulate more differentiating
weight changes. ~10-25 min compute per smoke.

**Cost:** longer compute, no code change.

## Recommended next step

Implement Fix A + Fix B together (most targeted), then retest with
200 events/word. If still not aligned, try Fix C (more training).

If still not aligned after all 3, the embodied-Hebbian paradigm has
the same fundamental issue as 3-factor for arbitrary cue-action
binding at biological scale — and dendritic learning becomes the
remaining option.

## Honest takeaway

This smoke shows the infrastructure works (no crashes, both
directions evaluated cleanly) AND that Tier 1 isn't a single
experiment — it's a parameter-tuning arc. The N→north 64% suggests
the binding mechanism is real but needs more tuning to differentiate
the other directions.

Decision for tomorrow: continue Tier 1 tuning OR pivot to Tier 3
(dendritic) earlier than planned. The plan budgeted 2 weeks for
Tier 1; if 2-3 days of tuning don't yield aligned results, pivot
is justified.

## Files

- This finding
- Smoke JSON: `research/findings/raw/g11_bg/text_eval_embodied_smoke_seed42.json`
- Implementation: `research/runners/bio_three_factor.py` + `research/runners/text_minimal_isolation.py`
- 3-tier plan: `docs/plans/2026-05-05-embodied-language-3tier-design.md`
