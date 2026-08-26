<!--
Findings doc template for the in-flight scaled 12-word multi-seed.
When seeds 42 + 44 complete, fill in numbers below and rename to
research/findings/2026-05-08-Phase1.3-Tier2.1-12word-scaled-multi-seed-RESULT.md
-->

# Phase 1.3 + Tier 2.1 12-word scaled-up multi-seed: {VERDICT_HEADLINE}

**Date:** 2026-05-08 EDT
**Status:** {N_GO}/3 GO at scaled arch (n_motor=2000) for 12-word vocab.

## Headline result

| Seed | Pri Pre | Pri Post | Pri Ret | Syn Pre | Syn Post | Syn Ret | Verdict |
|---|---|---|---|---|---|---|---|
| 42 | {pri_42_pre}% | {pri_42_post}% | {pri_42_ret}% | {syn_42_pre}% | {syn_42_post}% | {syn_42_ret}% | {verdict_42} |
| 43 | 56.2% | 53.8% | 100.0% | 25.6% | 29.4% | 138.2% | GO |
| 44 | {pri_44_pre}% | {pri_44_post}% | {pri_44_ret}% | {syn_44_pre}% | {syn_44_post}% | {syn_44_ret}% | {verdict_44} |
| **Mean** | {pri_pre_mean}% | {pri_post_mean}% | **{pri_ret_mean}%** | {syn_pre_mean}% | {syn_post_mean}% | **{syn_ret_mean}%** | **{N_GO}/3 GO** |

Compare to 12-word default-arch (n_motor=1000) result:
- Default: pri 84.0% (2/3 GO), syn 99.6% (3/3 PASS)
- Scaled:  pri {pri_ret_mean}% ({N_PRI_PASS}/3 PASS), syn {syn_ret_mean}% ({N_SYN_PASS}/3 PASS)

## Capacity hypothesis verdict

**{CAPACITY_VERDICT}**: {CAPACITY_INTERPRETATION}

## Per-seed comparison: default vs scaled

| Seed | Default Pri Ret | Scaled Pri Ret | Default Syn Ret | Scaled Syn Ret | Lift |
|---|---|---|---|---|---|
| 42 | 86.1% | {pri_42_ret}% | 93.5% | {syn_42_ret}% | {lift_42} |
| 43 | 71.1% | 100.0% | 95.1% | 138.2% | +28.9pp pri (PARTIAL→GO) |
| 44 | 94.7% | {pri_44_ret}% | 110.3% | {syn_44_ret}% | {lift_44} |

## What this validates

{DETAILED_INTERPRETATION}

## Setup

- **Runner:** `research/runners/consolidation_synonym_trainer.py`
- **Mode:** `--medium --vocab-size 12 --n-motor-per-action 2000 --n-motor-fs-per-action 240 --n-test-per-word 20`
- **Architecture:** Tier 2.1 v4 + 2x motor pools + hippocampus consolidation
  (n_lang=4096, n_motor=2000, n_motor_fs=240, ~22.7K neurons,
  ~28M synapses, ~5 GB GPU)
- **Vocab (12-word):** {north, up, n}, {east, right, e},
  {south, down, s}, {west, left, w}
- **Wall clock:** ~3.5 hr/seed × 3 = ~6 hrs total (some seed 43 slowdown
  from earlier orphan-process contamination, since fixed)

## Output artifacts

- Per-seed: `research/findings/raw/g11_bg/g11_seed{42,43,44}_consolidation_synonym_12word_scaled_medium_*.json`
- Aggregate: `research/findings/raw/g11_bg/consolidation_synonym_12word_scaled_medium_aggregate_2026-05-08.json`

## Related

- Tier 2.1 BREAKTHROUGH (capacity hypothesis):
  `research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md`
- 8-word 3-seed GO: `research/findings/2026-05-08-Phase1.3-Tier2.1-combined-3seed-CONFIRMED.md`
- 8-word strict anti-cheat: `research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md`
- 12-word default 3-seed: `research/findings/2026-05-08-Phase1.3-Tier2.1-12word-medium-3seed-PARTIAL.md`
- 12-word scaled single-seed (seed 43): `research/findings/2026-05-08-Phase1.3-Tier2.1-12word-scaled-CAPACITY-CONFIRMED.md`
- Master plan: `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`

<!-- POST-COMPLETION NOTES:
- If 3/3 GO: this validates the capacity hypothesis at multi-seed.
  Headline: "Capacity hypothesis confirmed at multi-seed 12-word."
  Next steps: launch 16-word smoke + Phase 1.5 multi-seed.
- If 2/3 GO + 1 PARTIAL: still validates the lift (seed 43 went
  from PARTIAL to GO at scaled). Headline: "Capacity hypothesis
  partial validation; one seed still degrading at 12-word."
  Next step: investigate which seed failed, consider larger
  arch (n_motor=3000?).
- If 1/3 or worse: capacity hypothesis NOT supported at multi-seed;
  the seed 43 lift was seed-specific. Headline: "Capacity hypothesis
  NEGATIVE at multi-seed; seed 43 was outlier."
  Next step: alternative theories (synonym training interference,
  sleep replay coverage gap).
-->
