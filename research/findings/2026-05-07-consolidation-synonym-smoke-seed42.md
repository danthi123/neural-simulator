# consolidation_synonym smoke seed 42 — runner end-to-end validation

**Date:** 2026-05-07
**Run ID:** 4bb80bd65729 (launched via webapp `/api/runs/launch`)
**Wall clock:** 21.3 min total (14.5 min training + ~7 min eval)
**Mode:** `--smoke` (50 awake events/word + 50 SWR events/cycle, vs
default 400+200)
**Architecture:** Tier 2.1 v4 scale-up + hippocampus consolidation
(n_lang=4096, n_motor=1000, n_motor_fs=120, ~12.7K neurons,
~12.9M synapses, 5 hippo regions, 12 cycles awake/sleep)

## Purpose

End-to-end validation of `consolidation_synonym_trainer.py` shipped
in commit ce8a683. Confirms:
1. Tier 2.1 v4 + hippocampus framework wires correctly
2. Awake/sleep alternation runs through all 12 cycles
3. Pre-silence + hippo-OFF eval both complete on 8 synonyms
4. JSON output structure works
5. Webapp launcher dispatches the smoke preset cleanly

All 5 PASS.

## Headline result (from JSON)

```json
{
  "stats": {
    "n_awake_events": 200,
    "n_sleep_phases": 12,
    "n_total_swr_events": 600,
    "wall_clock_seconds": 869.4
  },
  "pre_silence": {"accuracy": 0.325, ...},
  "hippo_off":   {"accuracy": 0.3625, ...},
  "retention":   {"overall": 1.115, ...}
}
```

**Pre-silence overall:** 32.5% (13/40 trials, 8 words × 5 trials each)
**Hippo-OFF overall:** 36.25% (29/80, 8 words × 10 trials each)
**Retention ratio:** 111.5% (hippo-OFF is HIGHER than pre-silence)

## Caveats

**The JSON's `primary_acc` and `synonym_acc` fields show 0%/0% due to a
parsing bug in `consolidation_synonym_trainer.run_full()` that was
shipping in the version this run used.** The bug:
- `run_full` looked up `eval_result["per_word_accuracy"]` (doesn't exist)
- Should have computed from `confusion_matrix[word][action]`

Caught by this smoke run (overall 32.5% but per-word 0%). Fixed in
commit b4e269d. Future runs will report per-word correctly.

For this run, I extracted per-word last-trial deltas from the log
to estimate the split (only the FINAL trial's delta per word is
logged, not all 10):

**Pre-silence (last trial per word):**
- Primary: 4/4 correct (north, east, south, west all argmax to
  correct motor)
- Synonym: 0/4 correct (up→W, right→W, down→W, left→N)
- Pattern: synonym motor selectivity is weak; deltas are roughly
  equal across motors

**Hippo-OFF (last trial per word):**
- Primary: 1/4 correct (only north)
- Synonym: 1/4 correct (only up)
- Pattern: deltas got smaller and noisier; differentiation
  narrowed across the board

Last-trial isn't a reliable estimator of 10-trial accuracy, so these
are indicative not definitive. The headline 32.5% / 36.25% from the
full eval is the trustworthy number.

## Interpretation

**The hippo-OFF retention ratio of 111.5% is unusual.** Normally
removing a region reduces accuracy. Two plausible explanations:

1. **Pre-silence noise.** Active hippocampus during eval may
   inject noise into cortex via plastic-but-frozen pathways that
   still pass current. Removing that input reduces noise →
   slightly higher accuracy. This was the gotcha I documented in
   CLAUDE.md re: `set_plasticity_gate` freezing weight UPDATES
   only, not transmission. Even though gates are frozen, the
   pathway still injects current.

2. **Cortex-only path is sufficient.** The CLS theory prediction:
   sleep replay consolidates language_input → cortex → motor
   patterns into the cortex itself, so cortex alone is sufficient
   for inference. Removing hippo doesn't hurt because cortex
   already has the pattern.

Both predictions are consistent with the result. To distinguish,
need multi-seed + comparison to a no-consolidation baseline (run
without sleep phases).

## What this DOES validate

- **Runner works.** 12 awake/sleep cycles complete without crash.
  Hippocampus regions register, awake_gates / sleep_gates flip
  correctly, SWR replay fires.
- **Webapp launcher works.** `consolidation_synonym_smoke` preset
  routes to `consolidation_synonym_trainer` with `--out-stats`,
  no live-mode flag injection.
- **Bug surfaced and fixed.** The 0%/0% parsing bug would have
  silently propagated to all subsequent multi-seed runs. Caught
  by single smoke before invest hours in 3-seed validation.
- **Wall-clock estimate:** smoke ~21 min, full would be ~3-4×
  longer = ~60-80 min (since smoke uses 50 events/word, full uses
  400 events/word). My initial design-plan estimate of "~30-45 min
  single seed" was too low; reality is closer to 60-80 min.

## What this does NOT validate

- **The CLS hypothesis at synonym scale.** The single-seed result
  is intriguing (retention >100%) but neither GO nor NO-GO. Need
  3+ seeds with proper per-word statistics from the fixed code.
- **Synonym binding in the architecture.** Pre-silence per-word
  shows synonyms barely above chance — same pattern as the
  chat_synonym_demo seed 42 (50% primary / 0% synonym last-trial).
  This may be seed 42 specific or may indicate the smoke's
  reduced training (50 events/word vs Tier 2.1 BREAKTHROUGH's
  400 events/word) was insufficient for synonym binding.

## Next: properly validate

```bash
# Single seed full run with the fixed code (~60-80 min, predicts
# proper primary/synonym split):
python -m research.runners.consolidation_synonym_trainer \
    --seed 42 --n-awake-events-per-word 400 --n-sleep-swr-events 200 \
    --n-test-per-word 25 \
    --out-stats research/findings/raw/g11_bg/consol_syn_seed42_v2.json

# 3-seed validation (~3-4 hours total) once single seed shows
# expected 80%+ primary retention pattern:
for seed in 42 43 44; do
    bash scripts/multiseed_chat_demo.sh consolidation_synonym $seed
done
```

Both deferred to next autonomous window or user-driven run.

## Findings record

- Runner b4e269d: per-word accuracy fix (compute from confusion_matrix)
- Runner ce8a683: original consolidation_synonym_trainer + 2 webapp
  presets
- Design plan: `docs/plans/2026-05-07-Phase1.3-Tier2.1-combined-design.md`
- Wall-clock estimate corrected: 30-45 min → 60-80 min single seed
  (smoke at 21 min implies ~80 min full)

## Lesson learned

**Smoke runs catch real bugs.** Without this 21-min smoke, the parsing
bug would have shipped silently. A 3-seed × 80-min run would have
produced 4 hours of "0%/0% primary/synonym" results before anyone
noticed. The smoke phase is cheap insurance — every multi-seed
plan should include a smoke pre-flight.

Per autonomous-runs principle #6 (anti-shortcut discipline): document
the bug + fix transparently, even though it makes the result harder
to interpret.
