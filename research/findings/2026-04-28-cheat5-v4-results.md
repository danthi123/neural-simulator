# Cheat #5 v4 — Developmental Pretraining (results pending)

**Date:** 2026-04-28
**Design:** [docs/plans/2026-04-28-cheat5-v4-design.md](../../docs/plans/2026-04-28-cheat5-v4-design.md) (commit `e6ce0ce`)
**Plan:** [docs/plans/2026-04-28-cheat5-v4-implementation.md](../../docs/plans/2026-04-28-cheat5-v4-implementation.md)
**Aggregator:** [scripts/analyze_cheat5_v4.py](../../scripts/analyze_cheat5_v4.py)

> **Status:** WIP. Tier 1 wiring smoke PASSED. Tier 2 + Tier 3 pending. This file gets the verdict + numbers as they land.

## TL;DR (TBD)

Will be one of:
- **GO** — sum ≤ 4.1 with both phases ≤ 2.5; cheat #5 closed via developmental pretraining
- **GO MARGINAL** — sum ≤ 4.5; closure-without-improvement
- **PARTIAL** — sum 4.5–6.0; needs longer pretraining or more goals
- **NO-GO v4** — sum > 6.0; cross-projections off-axis even developmentally; pivot to last-resort acknowledgment

## Background

v3 lateral inhibition shipped (GO, 2026-04-28, sum 4.26 ± 0.50, no regression). v3.1 cross-projections still failed phase-2 readaptation (NO-GO, sum 8.92, P1=6.35 = 2.5× P0). Interpretation in [research/findings/2026-04-28-cheat5-v3-results.md](2026-04-28-cheat5-v3-results.md) was that adult STDP+reward on a converged BG cascade can't shape useful cross-projection structure — the refinement might be a developmental phenomenon.

v4 tests that hypothesis: pre-train cross-projections under varied tasks during a "critical period" (all plasticity gates open), then freeze them at the start of eval and run the standard 1800-step moving-goal scenario.

## Architecture

[docs/plans/2026-04-28-cheat5-v4-design.md](../../docs/plans/2026-04-28-cheat5-v4-design.md) has the full design. Summary:
- `--developmental-pretraining` runs 10 random goals × 3000 trials each (default), all gates open.
- Goal sampling: uniform random with Manhattan ≥ 3 from start, no consecutive repeats.
- After pretraining, the existing curriculum init at [g11_bg_runner.py:1220](../../research/runners/g11_bg_runner.py#L1220) naturally forces `bg_cross_projections=0.0` at eval start. No manual freeze needed.
- Eval phase is the standard flagship: phase 1 (warmup 600) cortex_to_d1 plastic + sensory frozen; phase 2 cortex_to_d1 frozen + sensory thawed; bg_cross_projections frozen throughout.

## Tier 1 — wiring smoke (PASSED)

Single seed, 1 goal × 1000 trials of pretraining + 1800 eval.

| Seed | Pretraining cross weights | Phase 0 (goal=(6,6)) | Phase 1 (goal=(1,6)) | Sum | Status |
|---|---|---|---|---|---|
| 42 | mean=10.935 std=0.499 (no NaN) | meanD=3.33 finalQ=3.85 | meanD=4.67 finalQ=3.74 | 7.59 | ✅ wiring OK |

Pass criteria all met:
- rc=0, eval ran 1800 steps to completion
- "pretraining complete: 1000 trials, 1 goal changes" line present
- cross weights summary not NaN
- Agent visited goal cell during eval (positions cycled through (1,6)/(1,7)/(2,7) area)

Sum 7.59 is not meaningful for the decision matrix — Tier 1's 1 goal × 1000 pretraining is a wiring check, not a real developmental phase. Useful signal: cross-projections grew from 0.0 to mean 10.9, confirming the helper is exercising STDP+reward on the cross-projection synapses as intended.

Wall-clock: 598s (~10 min single-process).

## Tier 2 — reduced smoke (PENDING)

3 seeds (42, 43, 44), 5 goals × 1000 trials = 5K pretraining + 1800 eval each.

| Seed | Pretraining cross weights | P0 finalQ | P1 finalQ | Sum |
|---|---|---|---|---|
| 42 | TBD | TBD | TBD | TBD |
| 43 | TBD | TBD | TBD | TBD |
| 44 | TBD | TBD | TBD | TBD |
| **mean** | — | TBD | TBD | TBD ± TBD |

**Verdict (TBD)**: ≤ 4.5 → proceed to Tier 3. 4.5–6 → review per-seed. > 6 → NO-GO v4.

## Tier 3 — full validation (PENDING)

6 seeds (42, 43, 44, 100, 101, 102), 10 goals × 3000 trials = 30K pretraining + 1800 eval each.

| Seed | Pretraining cross weights | P0 finalQ | P1 finalQ | Sum |
|---|---|---|---|---|
| 42 | TBD | TBD | TBD | TBD |
| 43 | TBD | TBD | TBD | TBD |
| 44 | TBD | TBD | TBD | TBD |
| 100 | TBD | TBD | TBD | TBD |
| 101 | TBD | TBD | TBD | TBD |
| 102 | TBD | TBD | TBD | TBD |
| **mean** | — | TBD | TBD | TBD ± TBD |

**Verdict (TBD)**: per the decision matrix above.

## Interpretation (TBD)

Will be filled in based on the final outcome:

**If GO**: cheat #5 is closed via developmental pretraining. The simulator now demonstrates two distinct learning regimes — developmental (high plasticity, varied experience) and adult (lower plasticity, task-specific) — beyond the immediate cheat #5 closure.

**If MARGINAL/PARTIAL**: developmental pretraining shapes cross-projections, but the resulting weights don't drive better behavior than the same-action-only architecture. Useful negative finding: cross-projections are a *biological constraint we can satisfy* rather than a *behavioral lever*.

**If NO-GO**: cross-projections are off-axis at this level of abstraction. Pivot to the last-resort plan — close cheat #5 *by design*, acknowledging same-action-only as the reduced-model equivalent of biological winner-take-all.

## Files

- Tier 1: `research/findings/raw/g11_bg/g11_seed42_flagship_6bcecf.json`
- Tier 2: `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_<id>.json` (TBD)
- Tier 3: `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_flagship_<id>.json` (TBD)
- Aggregator: `scripts/analyze_cheat5_v4.py`
- Tier 3 launcher: `scripts/launch_cheat5_v4_tier3.sh`

## Updates needed (when verdict lands)

- [ ] CLAUDE.md "Cheat #5 progress (2026-04-28)" — add v4 result row
- [ ] docs/SCIENCE_ROADMAP.md §4.7 — append v4 row
- [ ] research/findings/INDEX.md — link this finding doc
- [ ] CHANGELOG.md — add v4 entry to 2026-04-28
- [ ] Memory: update `project_cheat5_v3_results.md` with v4 outcome OR add `project_cheat5_v4_results.md` + line to MEMORY.md
- [ ] If GO: spawn follow-up task for pretrained-weight persistence (HDF5 save/load, deferred at design time)
- [ ] If NO-GO: spawn follow-up task for the last-resort closure-by-design plan
