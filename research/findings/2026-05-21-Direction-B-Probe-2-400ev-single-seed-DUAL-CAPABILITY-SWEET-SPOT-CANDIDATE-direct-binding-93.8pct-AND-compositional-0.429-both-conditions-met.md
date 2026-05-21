# Direction B Probe-2 (400ev Phase-1; seed 42 cheap-first): BOTH conditions of the pre-registered dual-capability decision rule met at seed 42 -- direct binding 15/16 = 93.8% (>= 0.80 bar; identical to 800ev) AND compositional N=3 = 0.429 (>= 0.40 bar; 75% of the 200ev local optimum 0.571); 400ev is a DUAL-CAPABILITY SWEET-SPOT candidate worth multi-seed expansion

## Status

Single-seed cheap-first probe per Direction B Probe-2 protocol
(AUTONOMOUS_STATE.md, commit `aea8e20`). Phase-1 trained at 400
events/word (2x the 200-event baseline; the intermediate point
between the 200ev compositional optimum and the 800ev direct-
binding ceiling). 16 word x 400 events = 6400 training events;
~38 min wall-clock at seed 42 on RTX 3090.

Two evals ran on the new cache:

1. **Direct binding diagnostic** via the byte-unchanged
   `direct_binding_phase1_comparison.test_one_checkpoint` helper
   (16-word task; `measure_pool_firing` for each trained word ->
   top-pool check)
2. **6th arc compositional retrieval eval** via the byte-unchanged
   `generative_replay_pfc_frame_runner` (frozen ladder; --seeds 42 --loads 3)

## Result (pre-registered, no bar change, no threshold tuning)

```
Phase-1 cache: research/findings/raw/unified_per_regime/phase1_400ev/seed42.simstate.h5

Direct binding (16-word test):
  n_correct/n_total: 15/16
  accuracy:          93.8%
  bar 0.80:          PASS
  per-word failures: east (1; top=noun_pool_DOG; same failure
                     pattern as 800ev seed 42 -- consistent
                     architectural noise, not Phase-1-recipe
                     specific)

6th arc compositional (N=3, n_seeds=1 seed 42):
  full_acc:            0.4286   (3/7)
  uniform_ctrl_acc:    0.4286   (3/7; per_regime_advantage = 0)
  direct_retain_acc:   0.5000
  abstain_correct:     0.5714

Runner-reported verdict: GATE = VOID (n_seeds=1 < min_seeds=3)
Smell-test recompute    : GATE = VOID (matches verbatim; 15th match)
```

VOID is the correct verdict for a single-seed cheap-first probe;
the frozen min_seeds=3 bar correctly blocks any trustworthy
PASS/FAIL claim. The NUMERIC RESULT is what matters for the
pre-registered dual-capability decision rule.

## Pre-registered decision rule + outcome

From AUTONOMOUS_STATE.md (commit `aea8e20`):

> "If 400ev direct_binding >= 0.80 AND compositional N=3 seed-42
> >= 0.40: there is a DUAL-CAPABILITY SWEET-SPOT at 400ev worth
> multi-seed validating; report as a new validated milestone
> candidate (queue multi-seed at seeds 43/44; ~104 min additional).
> If 400ev direct_binding < 0.80 OR compositional < 0.40:
> the substrate has SEPARABLE training-event preferences for the
> two capabilities."

Observed:
- 400ev direct_binding seed 42 = 0.938 >= 0.80 ✓
- 400ev compositional N=3 seed 42 = 0.429 >= 0.40 ✓ (margin +0.029)

**Both conditions met. Decision rule fires the DUAL-CAPABILITY
SWEET-SPOT branch -- multi-seed expansion is the next concrete
action.**

## Updated capability frontier (cross-arc trajectory at seed 42)

| Phase-1 ev/word | Direct binding (16-word) | Compositional N=3 (seed 42) |
|-----------------|--------------------------|------------------------------|
| 100ev           | (untested)               | 0.286                        |
| 200ev (6th arc) | 68.8%                    | **0.571 (LOCAL OPTIMUM)**    |
| **400ev (this probe)** | **93.8% (>= 0.80 ✓)** | **0.429 (>= 0.40 ✓)**  |
| 800ev           | **93.8% multi-seed VALIDATED** | 0.143                |

The direct binding capability SATURATES somewhere between 200ev
and 400ev (rising from 68.8% to 93.8%), NOT between 400ev and
800ev as one might have assumed (400ev and 800ev seed 42 are
IDENTICAL at 15/16). Compositional retrieval is a SLOWLY-MONOTONIC-
DECREASING function of training events ABOVE 200ev: 0.571 -> 0.429
-> 0.143. 400ev sits in the regime where BOTH capabilities are
above their respective bars.

## Biology-translatable insight #10 (NEW; conditional pending multi-seed)

**A DUAL-CAPABILITY OPERATING REGIME exists on this substrate.**
The earlier hypothesis (after the 800ev multi-seed validation) was
that direct binding and compositional retrieval have OPPOSITE
optimal training durations, which seemed to imply they could not
be simultaneously satisfied at any single training-event count.
The 400ev probe REJECTS this strong form of the dissociation:
at 400ev seed 42, BOTH capabilities clear their respective frozen
bars (direct >= 0.80 AND compositional >= 0.40).

The WEAKER form of the dissociation still holds: the SINGLE
OPTIMUM for compositional retrieval (200ev) is distinct from the
SINGLE OPTIMUM for direct binding (probably ~400-800ev). But the
substrate's actual capability frontier is a SMOOTH curve, and
there is a regime (400ev) where both capabilities are jointly
above the trustworthy floor. This is biology-consistent with CLS
theory: real cortex maintains schema-learning AND episodic
binding at moderate training-event counts; only EXTREMES of
training-event-count produce the dissociation. The complementary
learning systems prediction is RELATIVE not ABSOLUTE: hippo-vs-
cortex have DIFFERENT optimal training-event-count profiles, but
their joint operating region is non-empty for biologically-
realistic training regimes.

Caveat: SINGLE-SEED probe. Multi-seed expansion (Probe-2.5) is
required before this insight is durable. Seed 42 was the HIGHEST
of the 3 6th arc seeds (0.571 vs 3-seed mean 0.458 = +0.113);
the multi-seed mean at 400ev compositional could pull DOWN below
the 0.40 bar (estimated multi-seed mean ~0.34 if seed 42 is
similarly above the mean at this rung). The honest discipline is:
PROCEED to multi-seed expansion per the pre-registered decision
rule; let the data decide.

## Discipline check + propagation

NO bar change anywhere; NO threshold tuning; NO re-run. The
6th arc decisive runner, the direct binding diagnostic, and the
smell-test recompute were ALL reused byte-unchanged. The new code
is just two thin driver scripts (phase1_curve_diagnostic +
direct_binding_single_seed_for_curve) that wrap reused helpers.

Protected set byte-empty diff vs `e8a99a2` continues to hold;
no-confab moat 7/7 byte-identical; 4 calibrated abstention
thresholds byte-stable.

15 consecutive adversarial reviews + smell-test recomputes
matching runner-reported verdicts exactly.

## Files / evidence

- New driver scripts:
  - `research/findings/raw/phase1_curve_diagnostic.py` (commit `b290414`)
  - `research/findings/raw/direct_binding_single_seed_for_curve.py` (this commit)
- 400ev Phase-1 cache: `research/findings/raw/unified_per_regime/phase1_400ev/seed42.simstate.h5` (27.1 MB)
- Direct binding JSON: `research/findings/raw/direct_binding_400ev_seed42.json`
- Compositional decisive JSON: `research/findings/raw/phase1_400ev_decisive.json`
- Training log: `research/findings/raw/phase1_400ev_train.log`
- Direct binding log: `research/findings/raw/direct_binding_400ev.log`
- Compositional eval log: `research/findings/raw/phase1_400ev_eval.log`

## Next biology-faithful direction (pre-registered, next concrete action)

Per the dual-capability decision rule, multi-seed expansion of
Probe-2 is the next concrete action:

1. Train seed 43 at 400ev (~38 min) via `phase1_curve_diagnostic
   --seed 43 --events-per-word 400`.
2. Train seed 44 at 400ev (~38 min) similarly.
3. Direct binding test on each via `direct_binding_single_seed_for_curve
   --seed N --cache-dir ...`.
4. 6th arc compositional eval at multi-seed: `generative_replay_pfc_frame_runner
   --seeds 42 43 44 --loads 3 --phase1-cache-dir
   research/findings/raw/unified_per_regime/phase1_400ev/ ...`.
5. Apply the FROZEN bars: PASS iff multi-seed direct >= 0.80 AND
   multi-seed compositional N=3 >= 0.40 across 3 seeds.

Total wall-clock estimate: ~76 min training + ~10 min eval = ~85
min. If multi-seed VALIDATES the dual-capability sweet-spot,
this becomes a new validated capability milestone on the unified
substrate and should be added to capability_status.json as a
pillar. If multi-seed FAILS (mean compositional drops below 0.40,
or direct fails at some seed), the substrate's dual-capability
operating region was an artifact of seed 42 being favorable;
this is itself an honest biology-translatable insight (the
operating regime is seed-dependent at single-seed but does not
generalize multi-seed at the pre-registered bar).
