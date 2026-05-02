# 2026-05-02 — 100-ep R3+R6 reset-fix run: REGRESSED. Reveals prior baseline was illusory.

**Test config:** 100-ep R3+R6 seed=42, balanced sampling + reset_steps=100 (cee3403 fix kept) + remaining partial Tier 1 (stim_steps=100, per-type-stp=False).

**TL;DR:** Result **20.0% I→W / 25.0% W→A** -- bracket "20-28% (still regressed)" per the configured decision tree. Action: revert Tier 1.1 (stim 200→100). **Already launched** (PID 22124, ETA ~03:10).

**Critical reframe:** the predicted-class distribution is now BALANCED (N:10 E:9 S:11 W:10 for I→W) thanks to balanced direction sampling. The drop in headline accuracy from the prior 32.5% baseline is not a regression in CAPABILITY — it's an UNMASKING. The 32.5% was an artifact of east-prediction bias on an east-heavy eval distribution (the prior baseline file `text_eval_R3_R6_combined.json` predates the d961940 balanced-sampling fix at May-1 19:22 vs the fix at May-1 19:33). The trained network never genuinely learned the language-action mapping; it just had a tendency to predict "east" that happened to score >25% on a biased test.

## Decision-tree application

Per the established thresholds:

```
result accuracy (max of I->W, W->A): 25.0%
  >= 35%: WIN -- not us
  28-35%: matching/improving baseline -- not us
  20-28%: STILL REGRESSED -- this is our bracket
          Action: also revert Tier 1.1 (stim 200->100). Stim window may
                  also matter for STDP pre-post pairing in language pathway.
  <20%: deeper bisect needed -- not us
```

**This is exactly the action being taken now.** PID 22124 launched at 01:36 with `--stim-steps-per-step 200 --reset-steps 100` (full Tier 1 revert on stim, reset already at 100). ETA ~03:10. Saves checkpoint for downstream weight diagnostic + reeval sweep.

## Headline numbers

| Run | Date | I→W | W→A | I→W p-value | Pred dist (I→W) |
|---|---|---|---|---|---|
| R3+R6 baseline (combined.json) | May-1 19:22 | 32.5% | 30.0% | p=0.18 | N:9 E:19 S:6 W:6 |
| 300-ep + full Tier 1 (REGRESSED) | May-1 22:25 | 20.0% | 20.0% | p=0.81 | (similar bias?) |
| **100-ep + partial Tier 1 (this)** | May-2 01:34 | **20.0%** | **25.0%** | p=0.82 | N:10 E:9 S:11 W:10 |

Both p-values vs chance (n=40, 4-class) are at or above 0.5 — accuracy is statistically indistinguishable from chance.

## What changed since the prior 32.5% baseline

| | combined.json | partialT1 (this) |
|---|---|---|
| Date | May-1 19:22 | May-2 01:34 |
| Balanced direction sampling | NO | YES (d961940) |
| stim_steps_per_step | 200 | 100 (Tier 1.1) |
| reset_steps | 100 | 100 |
| enable_per_type_stp | True (default) | False (Tier 1.5) |
| eval methodology | block-ordered, n=40+10 | block-ordered, n=40+10 (process used pre-improvement code) |

So the diffs are: **balanced sampling**, **stim_steps halved**, and **per-type-stp disabled**.

## The unmasking

The prior baseline's I→W confusion matrix shows heavy east-prediction bias:

```
combined.json I->W predicted distribution: N:9 E:19 S:6 W:6
              east got 8/16 = 50% accuracy
              other words: <33% each
```

This was the artifact of the pre-balanced training distribution: when targets are sampled with `|dx|>=|dy|` tie-break bias, east/west are over-represented (~28% each vs ~21% for north/south per the diagnostic in 2026-05-01-text-io-FINAL-summary.md). The network learned "predict east often" as its only effective strategy — and 19/40 east predictions on the biased eval distribution scored 32.5%.

Balanced sampling (d961940) fixes the training distribution. The new run shows:
- Predicted dist: N:10 E:9 S:11 W:10 (balanced)
- Per-word accuracy: north 28.6%, east 20%, south 22.2%, west 14.3% — all near chance
- W→A: predicted N:11 E:9 S:9 W:11 (balanced), but only 25% accuracy

The model never genuinely learned language→action. The mid-train correct-moves (33-35%) reflects the visuomotor learning, not language.

## Cause attribution

Two confounded variables:
1. **Balanced sampling** removed the easy-east bias that the prior baseline exploited.
2. **Tier 1.1 (stim_steps=100)** halves STDP events per env step, weakening language pathway formation.

The partial-T1 run can't distinguish these because both apply. The next run (full Tier 1 revert, stim_steps=200 + balanced sampling) will isolate the effect:
- If accuracy with stim=200 is at chance: stim_steps wasn't the issue, the 32.5% baseline was purely east-bias artifact
- If accuracy with stim=200 is >30%: stim_steps reduction was the regression cause

## Implications

1. **The "32.5% / 30% baseline" cannot be trusted as a reference point.** It was an artifact of an asymmetric eval distribution masking a model that never learned the mapping.

2. **Functional textual training is FURTHER away than we thought.** We're at chance, not at "1.3x chance".

3. **Methodology infrastructure shipped tonight is essential.** Without:
   - Bigger n (100 trials): can't distinguish chance from real signal at p<0.05
   - Balanced predicted-class diagnostics: would miss the unmasking
   - Interleaved eval + checkpoint save + reeval sweep: needed to disentangle eval-side from training-side limitations

4. **Architectural changes are now justified.** Bigger language regions, longer training, stronger drives, possibly different decoding — all on the table for upcoming experiments.

## Decision and next steps

- Launched 100-ep full Tier 1 revert (stim=200, reset=100): PID 22124, ETA ~03:10. Tests stim_steps hypothesis. Saves checkpoint for downstream re-eval sweeps.
- Plan continues per `docs/plans/2026-05-02-overnight-plan.md`: aggregate full-revert result, run reeval sweep on its checkpoint, then architectural followups (stronger drive, longer training).

## Repro / files

- This result: `research/findings/raw/g11_bg/text_eval_R3_R6_100ep_partialT1.json`
  - (Internal naming used "partialT1" for the reset-fix run; this doc uses
    "reset-fix" externally per the user-specified decision-tree convention.)
- Log: `research/findings/raw/g11_bg/R3R6_100ep_partialT1.log`
- PID file (archived after completion): `research/findings/raw/g11_bg/R3R6_100ep_partialT1.pid.done`
- Analyzer module: `research/runners/text_eval_analyze.py` (commit d3f28f0)
- Comparison baseline: `research/findings/raw/g11_bg/text_eval_R3_R6_combined.json`
- In-flight followup (full Tier 1 revert, --stim-steps-per-step 200):
  `research/findings/raw/g11_bg/text_eval_R3R6_100ep_NoT1_seed42.json` (ETA ~03:10)
  PID 22124. Will save checkpoint to `text_eval_R3R6_100ep_NoT1_seed42.simstate.h5`.

## Followup pipeline (when 22124 completes)

1. Run `text_eval_analyze` on the result JSON
2. Run `text_weight_diagnostic` on the saved checkpoint -- does STDP
   actually differentiate language→motor weights for the 4 directions?
   If yes: methodology-side problem. Run `text_reeval_sweep` over
   (drive_pA, n_reset_steps) to find best readout config.
   If no: training-side problem. Architectural changes needed (stronger
   drive, longer training, bigger language regions, etc.).
3. Document, commit, push, iterate.
