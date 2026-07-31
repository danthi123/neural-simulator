---
type: plan
status: live
date: 2026-05-09
---

# 2026-05-09 EVENING — Phase 2 resumption plan ADDENDUM

Addendum to `docs/plans/2026-05-09-Phase-2-resumption-plan.md` (drafted
earlier today). Captures what's changed in the strategic context since
the plan was written, given Phase 1.5 multi-seed completed.

## What's new since the morning plan

The resumption plan was written assuming Phase 1.5 multi-seed was still
"in flight" with possible-pass. Today's results:

1. **Phase 1.5 multi-seed FINAL = FAIL** (mean 0.629 < 0.70 threshold)
   - 3-seed across all 4 benchmarks
   - Robust pattern: 2/4 PASS (sequential 0.95 + retention 0.94),
     2/4 FAIL (interference 0.39 + long_tail 0.23)

2. **Both interference hypotheses REFUTED** (single-seed):
   - Under-training (v200 → v400): 0.34 → 0.345, +0.005
   - Capacity (n_motor 1000 → 2000): 0.345 → 0.39, +0.045
   - Neither lever crosses the 0.50 threshold
   - Per-word bimodal pattern: strong words bind, weak words near chance,
     pattern persists across both lever sweeps

3. **long_tail_relaxed in flight** (~30 min remaining at addendum time)
   - Outcome C if it passes (capacity doesn't help interleaved; dose
     helps few-shot)
   - Outcome D if it fails (neither lever helps anywhere)

In both Outcome C and Outcome D, per the 2026-05-09 decision tree,
the master plan response is identical: **demote Phase 1.5 from
milestone gate to tier report; pivot to Phase 2.2b**.

## How this changes the resumption plan

**Track 1 (Phase 2.2b 10M-param) priority: UPGRADED.** With Phase 1.5
demoted, Phase 2.2b is now the active master plan milestone (not
parallel-track). The 10M-param run becomes the next significant
experiment after the chat_speak_demo wraps.

**Track 2 (Phase 2.3b transfer test) priority: depends on 2.2b outcome.**
- If 2.2b reaches loss ~0.5-0.7 on Tiny Shakespeare (similar to Phase
  2.2's 1.016 but at 10M params): proceed to 2.3b transfer test (~3-6 hr)
- If 2.2b plateaus at ~1.0+ same as Phase 2.2 toy scale: scale was
  insufficient; Phase 2 needs even larger (cloud H100, 100M+ params)
  or fundamentally different (e.g. distillation from a real LLM)

**Track 3 (conversational scaffolding) priority: SHIPPED.** This was
parallel CPU work in the original plan. As of today's autonomous arc:
all 4 layers of Track 3 v1 ARE complete:
- layer 1: chat_repl --learn primitive (commit f6c919c)
- layer 2: chat_learn_demo runner + webapp surface (commit 20ec1ce)
- layer 3: :again / :opposite / :history / :forget (commit abbf9bf)
- layer 4: :speak generative decoder + chat_speak_demo (a675fa1, ecc185c)

GPU smoke for layer 4 is currently in the chain (after long_tail_relaxed).
Track 3 v1 is feature-complete; the documentation captures it at
`docs/plans/2026-05-09-Track-3-conversational-scaffolding-progress.md`.

## What the validated capability boundary looks like now

```
✓ VALIDATED (multi-seed):
  4-word vocab Tier 1                           6/6 (Tier 1 BREAKTHROUGH)
  8-word vocab Tier 2.1 sequential              5/6 + 6/6 (Tier 2.1 BREAKTHROUGH)
  12-word vocab Tier 2.1 scaled (n_motor=2000)  3/3 GO unanimous
  Phase 1.4 BRANCH A (no catastrophic forgetting) 5/6 PASS
  Phase 1.3 consolidation (cortex retains)      3/3 strict anti-cheat
  Phase 1.5 sequential_expansion                3/3 (mean 0.95)
  Phase 1.5 retention_over_time                 3/3 (mean 0.94)
  Track 3 v1: --learn / dialog state / :speak (4 layers shipped)

✗ ARCHITECTURAL CEILING (Phase 1.5 + lever sweep):
  Phase 1.5 interference (interleaved 8-word)   0.39 across all
                                                (events_per_word, n_motor) levers
  Phase 1.5 long_tail (few-shot 10-event rare)  0.17-0.28; relaxed-dose
                                                test pending

⏳ NOT YET TESTED:
  Phase 2.2b 10M-param Tiny Shakespeare         next major experiment
  Phase 2.3b transfer test                      after 2.2b checkpoint
  16-word vocab smoke                           parked (Phase 1.4 BRANCH A
                                                + capacity rule predicted PASS)
```

## Recommended sequencing post-Phase-1.5-demote

1. **Wait for long_tail_relaxed + chat_speak_demo to wrap** (~40 min
   remaining at addendum time). Document outcomes regardless.
2. **Phase 1.5 final demote findings doc** — pulls together all 4
   hypothesis tests + multi-seed result + decision tree resolution.
   Single comprehensive Phase 1.5 retrospective. ~30 min CPU work
   while GPU runs.
3. **Master plan update** — explicitly mark Phase 1.5 as demoted to
   tier report. Lift Phase 2.2b to active milestone. Re-anchor the
   "next 24-48 hours" planning around Phase 2.2b launch + monitoring.
4. **Phase 2.2b launch** — switch to path-f-hybrid branch, run the
   10M-param training overnight (~14 hr). Wakeup cadence: every
   ~2 hr to check loss curve.
5. **Phase 2.3b transfer test** — after 2.2b checkpoint lands.

## Why this addendum exists

Per autonomous-runs principle #6 (anti-shortcut discipline) +
principle #8 (session continuity): document strategic shifts as
addenda rather than overwrites so the original analysis stays
intact + traceable. The morning plan + this addendum together tell
the full story of why we end up at "demote Phase 1.5 + pivot to
Phase 2.2b". A reader can follow the decision logic across both
docs without losing the morning's options-on-the-table framing.

## Related

- `docs/plans/2026-05-09-Phase-2-resumption-plan.md` (the morning plan)
- `docs/plans/2026-05-09-Phase-1.5-decision-tree.md` (4-outcome strategy)
- `research/findings/2026-05-09-Phase-1.5-multi-seed-FINAL.md`
- `research/findings/2026-05-09-Phase-1.5-v400-interference-REFUTED.md`
- `research/findings/2026-05-09-Phase-1.5-n_motor_2000-interference-REFUTED.md`
- Master plan: `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`
