# Direction B Probe-2 multi-seed COMPLETE: trade-off curve EMPIRICALLY CHARACTERIZED at biological scale on the unified substrate; direct binding fully SATURATES by 400ev (41/48 = 85.4% IDENTICAL to 800ev multi-seed); compositional N=3 multi-seed mean 0.405 meets the pre-registered 0.40 bar with EXTREMELY THIN +0.005 margin; positive per_regime_advantage +0.042 (only the SECOND time multi-seed compositional has been above chance in the 8+ arc series); the substrate has a non-empty dual-capability operating region but the compositional half is NOT robust enough to claim a strongly-validated sweet-spot

## Status

Multi-seed expansion of Direction B Probe-2 (commit `35a2b80`).
Trained seeds 43 and 44 at 400 events/word Phase-1 (~38 min each;
~76 min total wall-clock). Ran multi-seed direct binding (16-word
diagnostic) + multi-seed 6th arc compositional retrieval eval at
N=3 across all 3 seeds.

## Results (pre-registered; no bar change, no threshold tuning)

### Multi-seed direct binding at 400ev (16-word test)

| Seed | n_correct / n_total | Accuracy |
|------|----------------------|----------|
| 42 | 15/16 | 93.8% |
| 43 | 13/16 | 81.2% |
| 44 | 13/16 | 81.2% |
| **Aggregate** | **41/48** | **85.4%** |

**ALL 3 SEEDS individually >= 0.80 frozen bar.** Aggregate IDENTICAL
to the 800ev multi-seed result (commit `13cf569`: 41/48 = 85.4%;
seeds 42/43/44 with the SAME per-seed accuracies 15/16, 13/16,
13/16). The direct binding capability is FUNCTIONALLY IDENTICAL
between 400ev and 800ev training -- direct binding has fully
saturated by 400ev.

### Multi-seed 6th arc compositional retrieval at 400ev

```
Seeds: [42, 43, 44]; N=3; n_seeds=3
  full_acc            = 0.405   (multi-seed mean)
  uniform_ctrl_acc    = 0.363
  per_regime_advantage = +0.042  (POSITIVE; second multi-seed
                                  positive advantage in 8+ arc
                                  series, after 6th arc +0.137)
  direct_retain_acc   = 0.600   (in-runner direct retain; differs
                                  from 16-word direct binding test)
  abstain_correct     = 0.498

Runner-reported verdict: GATE = VOID (ladder prefix mismatch;
                                       ran only N=3, frozen ladder
                                       is (2,3,5))
Smell-test recompute    : GATE = VOID (matches verbatim; 16th match)
```

## Pre-registered decision rule + outcome

From AUTONOMOUS_STATE.md (commit `35a2b80`):

> "PASS iff multi-seed direct binding >= 0.80 per-seed AND multi-
> seed compositional N=3 >= 0.40 mean across 3 seeds.
> - PASS at multi-seed: 400ev is a VALIDATED dual-capability
>   sweet-spot on the unified substrate.
> - FAIL at multi-seed (multi-seed compositional N=3 mean < 0.40):
>   the substrate's dual-capability operating region was an
>   artifact of seed 42 being favorable; the operating regime is
>   seed-dependent at single-seed but does not generalize multi-
>   seed at the pre-registered bar."

Observed:
- Multi-seed direct binding: all 3 seeds >= 0.80 ✓ (aggregate 85.4%)
- Multi-seed compositional N=3: 0.405 >= 0.40 ✓ (margin **+0.005**)

**Both conditions are technically met by the pre-registered rule.**
But the compositional margin is **+0.005 absolute (~1.25%
relative).** This is so thin that honest framing matters: the
compositional half of the dual-capability claim is at the EDGE of
the pre-registered bar, not robustly above it.

## Honest framing of the trade-off curve

The capability frontier on this substrate at biological scale is
now empirically characterized across 4 training-event budgets:

| Phase-1 ev/word | Direct binding (16-word multi-seed) | Compositional N=3 multi-seed | Per-seed all >= 0.80 direct |
|-----------------|--------------------------------------|------------------------------|------------------------------|
| 100ev (single-seed seed 42) | (untested) | 0.286 (seed 42 only) | n/a |
| 200ev (6th arc multi-seed)  | 68.8% (single-seed seed 42) | **0.458 (LOCAL OPTIMUM)** | unknown multi-seed |
| **400ev (this; multi-seed)** | **85.4% aggregate (15/13/13)** | **0.405** | **YES (all 3)** |
| 800ev (multi-seed; commit 13cf569) | **85.4% aggregate (15/13/13)** | 0.143 (single-seed seed 42) | YES (all 3) |

Two empirical discoveries are durable:

1. **Direct binding capability SATURATES somewhere between 200ev
   and 400ev** (rising from 68.8% single-seed seed-42 baseline to
   85.4% multi-seed aggregate at 400ev). 400ev and 800ev produce
   IDENTICAL multi-seed direct binding results. Past 400ev,
   additional training is NOT improving direct binding; it is
   wasting compute.

2. **Compositional retrieval is MONOTONICALLY DECREASING ABOVE
   200ev**: 0.458 -> 0.405 -> 0.143 in multi-seed-where-available
   data. The 200ev sweet-spot is the LOCAL OPTIMUM and is the
   only training regime above 0.45 mean. 400ev compositional
   retains ~88% of the 200ev compositional mean (0.405 / 0.458 =
   0.884); 800ev retains only ~31% (0.143 / 0.458; seed 42 only).

The pre-registered Direction B decision rule's PASS branch is
technically met at multi-seed, but the framing should be precise:

- **Direct binding portion**: VALIDATED multi-seed at 400ev, with
  trustworthy bar (>= 0.80 per-seed) met for all 3 seeds. This is
  the SAME validation level as the 800ev pillar (commit `4739d8e`).
  The new finding here is the SATURATION POINT: 400ev is sufficient.
- **Compositional portion**: MEETS the pre-registered 0.40 bar at
  multi-seed by margin +0.005 absolute (~1.25% relative); this is
  too thin to claim "robustly validated sweet-spot". The honest
  framing is: 400ev compositional retains ~88% of the 200ev local
  optimum mean while ALSO satisfying the saturated direct binding
  trustworthy bar. The substrate has a non-empty dual-capability
  operating REGION; 400ev is INSIDE that region but at the
  COMPOSITIONAL edge.

## Biology-translatable insight #10 (REFINED multi-seed)

The earlier strong-form dissociation hypothesis (after 800ev
multi-seed) -- that direct binding and compositional retrieval have
IRRECONCILABLE training-event preferences -- is REJECTED at multi-
seed: at 400ev, BOTH capabilities are above their respective frozen
bars across all 3 seeds.

The WEAKER and more precise form holds:
- Direct binding monotonically improves with training events up to
  saturation; saturation point is ~400ev on this substrate.
- Compositional retrieval is single-peaked at ~200ev; falls off
  monotonically with both shorter (100ev) and longer (400ev,
  800ev) training.
- The two capabilities' OPTIMA are at DIFFERENT training-event
  budgets (200ev compositional optimum; 400ev direct binding
  saturation), but the substrate has a non-empty REGION where
  both capabilities are simultaneously above their trustworthy
  bars. 400ev is at the compositional EDGE of that region.

This is biology-consistent with CLS theory's complementary-but-
overlapping prediction: hippo and cortex have DIFFERENT optimal
training-event profiles, but their joint operating region is
non-empty for biologically-realistic training regimes. The
substrate empirically demonstrates this overlap; the overlap is
not a strong sweet-spot but a TRANSITIONAL REGION between the two
capabilities' optima.

## Updated insight catalog (10 durable biology-translatable insights)

1. Trustworthy abstention thresholds are SUBSTRATE-AND-PROTOCOL-
   specific (4x validated)
2. v1 half-split calibration is statistically fragile; v2 within-
   word is principled fix
3. Cue-suppression-during-RETRIEVE violates encoding-specificity
   (Tulving 1973)
4. Replay + PFC-frame augmenting is LOAD-DEPENDENT (CLS-consistent)
5. Over-consolidation is biologically harmful (sweet-spot principle)
6. Single-query diagnostic signals don't transfer to multi-pair
   encoding pipelines (methodological insight)
7. Phase-1 training has its own SWEET-SPOT for compositional;
   aggressive training improves direct binding but breaks
   compositional flexibility
8. Direct binding capability RECOVERS with cumulative training;
   longer-training compensates for added auxiliary subsystems
   (VALIDATED multi-seed 85.4% at 800ev)
9. Gentler training does NOT preserve compositional capacity below
   the substrate's MINIMUM training threshold (100ev probe rejects
   the naive CLS less-is-more prediction)
10. **REFINED multi-seed**: Direct binding capability fully
    SATURATES by 400ev (identical to 800ev multi-seed); the
    substrate's two-capability operating region is non-empty but
    has DIFFERENT optima at different training-event budgets;
    400ev is the trade-off point where direct binding has
    saturated AND compositional retrieval has dropped from 0.458
    -> 0.405 (12% relative drop; meets the 0.40 bar with thin
    +0.005 margin)

## Discipline check + propagation

NO bar change anywhere; NO threshold tuning; NO re-run. The
multi-seed direct binding script reuses `test_one_checkpoint`
byte-unchanged from `direct_binding_phase1_comparison.py`. The 6th
arc decisive runner and smell-test recompute were reused byte-
unchanged. Protected set byte-empty diff vs `e8a99a2` continues to
hold; no-confab moat 7/7 byte-identical; 4 calibrated abstention
thresholds byte-stable.

16 consecutive adversarial reviews + smell-test recomputes matching
runner-reported verdicts exactly.

## Files / evidence

- New driver script: `research/findings/raw/direct_binding_multiseed_400ev.py`
- 400ev Phase-1 multi-seed caches: `research/findings/raw/unified_per_regime/phase1_400ev/seed{42,43,44}.simstate.h5`
- Multi-seed direct binding JSON: `research/findings/raw/direct_binding_multiseed_400ev.json`
- Multi-seed compositional decisive JSON: `research/findings/raw/phase1_400ev_multiseed_decisive.json`
- Training logs: `research/findings/raw/phase1_400ev_seed{43,44}_train.log`
- Eval log: `research/findings/raw/phase1_400ev_multiseed_eval.log`

## Honest next biology-faithful direction

The HONEST scientific deliverable from this autonomous arc has now
EXPANDED to include TWO empirically-characterized capability axes
on the unified substrate at biological scale:

| Capability | Status |
|------------|--------|
| Direct binding (16-word; multi-seed; 400ev OR 800ev) | **VALIDATED 85.4%; all 3 seeds >= 0.80** (saturated by 400ev) |
| Compositional retrieval (N=3; multi-seed at 200ev) | LOCAL OPTIMUM 0.458 (6th arc) |
| Compositional retrieval (N=3; multi-seed at 400ev) | 0.405 (12% below 200ev; meets pre-registered 0.40 bar with thin +0.005 margin; positive +0.042 per_regime_advantage) |

The dual-capability operating region exists but the compositional
half at the saturated-direct-binding training budget (400ev) is
not robustly above 0.40 -- it sits at the edge. Two natural next
directions:

**Direction C** (the strongly biology-faithful direction):
characterize the substrate at the COMPOSITIONAL OPTIMUM training
budget (200ev) for DIRECT BINDING. We have the cache (it was used
for the 6th arc; commit prior to e8a99a2). Question: at 200ev,
what is the multi-seed direct binding accuracy? Predicted from
trajectory: ~70% multi-seed mean (single-seed seed 42 at 200ev =
68.8%). This would confirm whether 200ev is the COMPOSITIONAL-
favored side of the dissociation or also a viable dual-capability
point. ~10 min eval; no training needed (cache exists).

**Direction D**: refine the curve between 200ev and 400ev. The
compositional drop from 0.458 -> 0.405 is across 200ev of training.
Is the cliff at 250ev? 300ev? 350ev? A SINGLE additional cache
(e.g., 300ev seed 42 single-seed cheap-first) would distinguish
"smooth drop" from "step at some training count". ~30 min training
+ ~5 min eval.

Direction C is cheaper, faster, and has higher information content
(would confirm or reject the saturated-direct-binding claim at the
LOWER training budget, completing the trade-off curve characterization
multi-seed at BOTH 200ev AND 400ev). Choosing Direction C as the
next concrete action.
