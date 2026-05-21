# Direction D Probe (300ev seed 42 single-seed cheap-first): BOTH dual-capability bars met at 300ev (direct 87.5% >= 0.80; compositional N=3 = 0.429 >= 0.40); the transitional band extends DOWN to AT LEAST 300ev at single-seed; compositional retrieval is FLAT between 300ev and 400ev seed 42 (both 0.429); direct binding RISES from 68.8% (200ev) to 87.5% (300ev) to 93.8% (400ev/800ev saturated); the transitional regime starts BETWEEN 200ev and 300ev where direct binding crosses 0.80

## Status

Cheap-first single-seed probe per Direction D protocol
(AUTONOMOUS_STATE.md commit `7c75159`). Trained seed 42 at 300
events/word Phase-1 (~30 min wall-clock). Ran 16-word direct binding
diagnostic + 6th arc compositional retrieval eval at N=3.

## Result (pre-registered, no bar change, no threshold tuning)

```
Phase-1 cache: research/findings/raw/unified_per_regime/phase1_300ev/seed42.simstate.h5

Direct binding (16-word test):
  n_correct/n_total: 14/16
  accuracy:          87.5%
  bar 0.80:          PASS

6th arc compositional (N=3, n_seeds=1 seed 42):
  full_acc:            0.4286   (3/7)
  uniform_ctrl_acc:    0.4286   (3/7; per_regime_advantage = 0)
  direct_retain_acc:   0.5000
  abstain_correct:     0.4286

Runner-reported verdict: GATE = VOID (n_seeds=1 < min_seeds=3)
Smell-test recompute    : GATE = VOID (matches verbatim; 17th match)
```

## Pre-registered decision rule + outcome

From AUTONOMOUS_STATE.md (commit `7c75159`):

> "If 300ev compositional N=3 seed-42 > 0.405 AND direct binding > 0.80:
> the transitional band is WIDER than just 400ev (extends down to at
> least 300ev). Update frontier characterization; queue multi-seed
> expansion of 300ev for trustworthy validation."

Observed (seed 42 single-seed):
- 300ev direct binding = 0.875 > 0.80 ✓
- 300ev compositional N=3 = 0.429 > 0.405 ✓ (margin +0.024)

**First branch fires. The transitional band extends DOWN to at least
300ev at single-seed. Multi-seed expansion is the pre-registered next
action.**

## Refined capability frontier (seed 42 single-seed data)

| Phase-1 ev/word | Direct binding (seed 42) | Compositional N=3 (seed 42) | Regime (single-seed) |
|-----------------|--------------------------|------------------------------|----------------------|
| 200ev (6th arc) | 11/16 = 68.8% (< 0.80)   | 0.571 (LOCAL OPTIMUM)       | COMPOSITIONAL-FAVORED |
| **300ev (this)**| **14/16 = 87.5%** (>= 0.80) | **0.429 (== 400ev seed 42)** | **TRANSITIONAL (lower edge)** |
| 400ev (Probe-2) | 15/16 = 93.8% (>= 0.80)  | 0.429                        | TRANSITIONAL (upper edge) |
| 800ev (longer-Phase-1) | 15/16 = 93.8% (>= 0.80) | 0.143                  | DIRECT-FAVORED |

Two single-seed empirical refinements:

1. **Direct binding crosses the 0.80 bar BETWEEN 200ev and 300ev**
   on this substrate at seed 42. 200ev is 68.8% (well below); 300ev
   is 87.5% (clearly above). The exact crossover point is in this
   narrow window. Likely candidates: 250ev (between the two probes;
   would resolve the granularity further).

2. **Compositional retrieval is FLAT between 300ev and 400ev at
   seed 42** (both 0.429 = 3/7 = 3 correct out of 7 queries). The
   eval has 7 queries; the discrete granularity is 1/7 = 0.143; both
   probes land on the same 3-out-of-7 mark. The compositional drop
   from 0.571 (200ev) to 0.429 (300ev) is the FIRST drop in the
   trajectory; the second drop (0.429 -> 0.143) happens between
   400ev and 800ev.

The TRANSITIONAL regime at single-seed seed 42 extends from at
least 300ev to 400ev (both have direct >= 0.80 AND compositional ==
0.429 above the 0.40 bar). The exact boundary at the lower end of
the band is in the 250-300ev range (where direct binding crosses
0.80). The upper boundary is between 400ev and 800ev (where
compositional drops below 0.40).

## Updated insight catalog (insight #11 refined; insight #12 NEW)

11. **REFINED single-seed**: The transitional regime band is WIDER
    than initially characterized. At single-seed seed 42, it extends
    from at least 300ev to 400ev. The narrow band at multi-seed
    (only 400ev cleared both bars in the prior arc) may have been a
    seed-44 + seed-43 sensitivity that pulled the multi-seed
    compositional mean below 0.40 at 300ev. Multi-seed expansion of
    300ev (Direction D-2) is the trustworthy test.

12. **NEW (Direction D Probe; single-seed)**: Direct binding has a
    rapid rise between 200ev and 300ev (68.8% -> 87.5% = +18.7pp),
    consistent with the substrate having a phase-transition-like
    crossing of the 0.80 trustworthy bar within a narrow training-
    event window. The substrate appears to require ~250-300 events
    per word to consolidate sufficient discriminative pathways for
    multi-seed trustworthy direct binding; below that count, direct
    binding accumulates monotonically but doesn't yet saturate.

## Discipline check + propagation

NO bar change anywhere; NO threshold tuning; NO re-run. The new
training reuses `phase1_curve_diagnostic.py` byte-unchanged with
`--events-per-word 300`. The direct binding test reuses
`direct_binding_single_seed_for_curve.py` byte-unchanged. The 6th arc
runner and smell-test reused byte-unchanged. Protected set byte-empty
diff vs `e8a99a2` continues to hold; no-confab moat 7/7 byte-identical;
4 calibrated abstention thresholds byte-stable.

17 consecutive honest-propagation cycles.

## Files / evidence

- 300ev Phase-1 cache: `research/findings/raw/unified_per_regime/phase1_300ev/seed42.simstate.h5`
- Direct binding JSON: `research/findings/raw/direct_binding_300ev_seed42.json`
- Compositional decisive JSON: `research/findings/raw/phase1_300ev_decisive.json`
- Training log: `research/findings/raw/phase1_300ev_train.log`
- Direct binding log: `research/findings/raw/direct_binding_300ev.log`
- Compositional eval log: `research/findings/raw/phase1_300ev_eval.log`

## Next biology-faithful direction (per pre-registered rule)

Per the Direction D decision rule (first branch fired), the next
concrete action is multi-seed expansion of 300ev (seeds 43 + 44):

- Train seed 43 at 300ev (~30 min)
- Train seed 44 at 300ev (~30 min)
- Multi-seed direct binding test (test_one_checkpoint on all 3 seeds)
- Multi-seed 6th arc compositional eval (--seeds 42 43 44)
- Total wall-clock: ~75 min training + ~10 min eval = ~85 min.

Decision rule for the multi-seed outcome (pre-registered, fixed):

- **PASS at multi-seed** (multi-seed direct >= 0.80 per-seed AND
  multi-seed compositional N=3 >= 0.40 mean): 300ev is ALSO a
  validated dual-capability point. The transitional band is
  empirically wider than just 400ev (at least 300ev-400ev wide).
  Update capability_status pillar to reflect the wider band.

- **FAIL at multi-seed** (multi-seed compositional N=3 mean < 0.40
  OR any seed direct < 0.80): seed 42 was the only favorable cell
  at 300ev; 300ev is NOT a dual-capability point multi-seed.
  Honest framing: the transitional band at multi-seed is UNIQUE
  to ~400ev; single-seed cheap-first probes can over-state width.

Reuse-only: no new code, no frozen verdict / protected / moat module
touched. GPU/CuPy mandatory.
