# consolidation_synonym wall-clock correction (5-9× underestimate)

**Date:** 2026-05-07 EDT
**Issue:** Multi-seed consolidation_synonym launched per master plan;
killed mid-seed-42 when realized full config takes ~6.5 hrs/seed
(not the documented "30-45 min" from the design plan).

## What happened

Per design plan `docs/plans/2026-05-07-Phase1.3-Tier2.1-combined-design.md`,
estimated wall-clock was "~30 min single seed." Smoke validation
(`--smoke`) had taken 21 min, so I extrapolated full at ~30-45 min.

Empirical observation during seed 42 of multi-seed run:
- Smoke (12 chunks of 16 awake + 50 SWR events): 21 min total
- Full (100 chunks of 16 awake + 200 SWR events): ~3.7 min/chunk
- Projection: 100 × 3.7 = 370 min = **~6.2 hours per seed**
- 3-seed multi-seed = **~19 hours**, not the ~4 hrs I planned

## Root cause

Two compounding factors I missed when scaling smoke → full:

1. **SWR events per chunk:** smoke 50 → full 200 (4× more)
2. **Chunk count:** smoke 12 → full 100 (8× more)

Naive linear scaling on awake events alone (50 → 400 = 8×) gave
21 min × 8 = ~170 min = ~3 hrs. But the actual scaling is closer to
21 min × 32 (combined effect) = ~10 hrs/seed, mitigated slightly by
GPU caching + warm-up reuse.

Lesson: when scaling between smoke and full configs that have BOTH
event count AND cycle count differences, multiply the scale factors.
SWR events × chunk count = total simulation steps for replay phases.

## Mitigation shipped

Added `--medium` mode between smoke and full:
- Smoke:  50 events/word, 50 SWR events, 12 chunks → ~21 min/seed
- Medium: 200 events/word, 100 SWR events, 50 chunks → ~80 min/seed
- Full:   400 events/word, 200 SWR events, 100 chunks → ~6.5 hrs/seed

Updated:
- `research/runners/consolidation_synonym_trainer.py`: docstring,
  `--medium` flag, mutual exclusion with `--smoke`
- `webapp/server.py`: PRESETS comment with corrected estimates;
  added `consolidation_synonym_medium` preset alongside smoke + full
- `tests/test_webapp_server.py`: regression test includes new preset

Recommendation:
- Use `--smoke` for runner validation (~21 min)
- Use `--medium` for quick multi-seed (~4 hrs / 3 seeds)
- Use full only for overnight or multi-day validation

## What was validated before kill

Seed 42 reached chunk 13/100 at 51 min elapsed before I killed it.
The architecture works; the runner trains correctly. The smoke
validation (research/findings/2026-05-07-consolidation-synonym-smoke-seed42.md)
remains the validated single-seed result:
- Pre-silence overall: 32.5%
- Hippo-OFF overall: 36.25%
- Retention ratio: 111.5%

These numbers are bug-affected (per_word_accuracy parsing fix shipped
in commit b4e269d after the smoke; future runs report correctly).

## Decision

Multi-seed consolidation_synonym deferred. The user can now launch
either:
- `bash scripts/multiseed_chat_demo.sh consolidation_synonym_medium 42 43 44` (~4 hrs)
- `bash scripts/multiseed_chat_demo.sh consolidation_synonym 42 43 44` (~19 hrs)

depending on time budget.

## Per autonomous-runs principle #6 (anti-shortcut discipline)

Documenting the miscalculation honestly. The smoke result (intriguing
111% retention) hasn't been refuted; it just needs proper multi-seed
validation at a feasible config.
