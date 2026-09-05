---
type: finding
status: partial
claim_check: measured
date: 2026-09-04
mechanism: affect-coupling-neural (BRAIN_WKV_MOUTH_AFFECT_NEURAL) — default-on promotion validation
lane: scaffold-retirement (affect→neural, arc item 3) + A·Affect
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_wkv_mouth_affect_neural_promote_summary_scale1.0.json
  - research/findings/raw/_wkv_mouth_affect_neural_promote_summary_scale2.0.json
  - research/findings/raw/_wkv_mouth_affect_neural_promote_summary_scale2.5.json
  - research/findings/raw/_wkv_mouth_affect_neural_promote_summary_scale3.0.json
verdict: >
  The affect→neural coupling (a neuromodulatory gain on the spiking word-read, merged 87631edf, PARTIAL, default
  OFF) is NOT yet ready to promote to default-ON. A neg_pa_scale sweep {1.0,1.5,2.0,2.5,3.0} x 6 seeds shows the
  negative-mood direction and output naturalness are in DIRECT TENSION on this knob, so no single value clears the
  gate (>=5/6 seeds BOTH directions strict, AND negative-arm affect-word fraction under ceiling). The coupling's
  default-OFF status + its already-verified load-bearing behaviour are UNCHANGED; this is a verdict on the
  neg_pa_scale METHOD for the default-on promotion, not on the capability.
---

# affect→neural default-on: the neg_pa_scale knob can't close the negative direction cleanly

## What ran
The build-ahead-prepared validation runner (`_wkv_mouth_affect_neural_promote_validate.py`, on main) run locally
(webapp/torch deps — the mini-PC pool can't host it), CPU-forced, 6 seeds x 3 prompts per (phase, scale). Positive
direction = a +mood prime must shift the mouth's free generation toward positive words; negative = a −mood prime
toward negative words; strict per-seed = ALL 3 prompts must pass.

## The curve (the whole story is here)

<!--derived: salad fractions are the mini-sweep's rounded _max_B_neg_affect_word_frac from the cited per-scale summary artifacts-->
| neg_pa_scale | positive seeds | negative seeds | max neg affect-word frac | directional GO (>=5/6 both) |
|---|---|---|---|---|
| 1.0 (baseline, no fix) | 6/6 | **1/6** | 0.250 | no |
| 1.5 | 6/6 | 3/6 | 0.250 | no |
| 2.0 | 6/6 | 4/6 | 0.250 | no |
| 2.5 | 6/6 | **6/6** | 0.346 <!--derived--> | no (salad) |
| 3.0 | 6/6 | 6/6 | 0.442 <!--derived--> | no (salad) |

Raw per-scale summaries: `research/findings/raw/_wkv_mouth_affect_neural_promote_summary_scale2.5.json` and
`research/findings/raw/_wkv_mouth_affect_neural_promote_summary_scale3.0.json` (plus scale1.0/1.5/2.0); per-seed
directional outputs in `research/findings/raw/_wkv_mouth_affect_neural_promote_directional_scale2.5_s42.json` etc.

The positive direction is solid at every scale (6/6). The negative direction rises monotonically with the gain
(1/6 → 6/6) — but so does the affect-word fraction of the negative output (0.25 → 0.44). The negative direction
only reaches 6/6 at scale ≥2.5, exactly where the output turns repetitive.

## Why a single knob can't win
Two facts make neg_pa_scale insufficient:
1. **The objectives conflict.** Lifting the weak negative direction to ≥5/6 requires scale ≥2.5, which pushes the
   affect-word fraction to 0.346-0.442 <!--derived-->. Reading the scale-2.5 output directly: "very sad and cried and cried for
   her mom ... very sad and cried" — recognisably negative but REPETITIVE; and on one prompt it drifts back
   positive mid-sentence ("very sad and angry at the bird and the bird became best friends ... lots of fun") — the
   TinyStories positivity prior reasserts itself against the gain. This is not natural strong-negative prose.
2. **The ceiling is exceeded at baseline.** The negative affect-word fraction is already 0.25 at scale 1.0 (no
   fix), above the runner's 0.20 gate ceiling. So the gate's naturalness criterion never passes on this knob — the
   over-affective tendency is inherent to a fixed-lexicon logit/gain nudge, not introduced by the fix.

## The next mechanism (no-defer — the method is banked, the capability is not)
The residual the original affect→neural finding named is now the load-bearing one: convert the **congruence source
itself** from a host fixed-lexicon lookup into a **learned word-identity↔valence association** driving the
neuromodulator's production rule. A learned association would recruit VARIED negative words in context (not repeat
"sad/cried"), which is exactly what the naturalness ceiling is failing on; and it removes the last host component
of the coupling. A secondary lever: the mouth's TinyStories corpus is positivity-biased (it fights the negative
direction) — a less-skewed corpus would help the negative direction reach load-bearing at a LOWER gain, inside the
naturalness ceiling.

## Honest residuals
- **byte_identical phase — harness bug, not a leak:** the runner's byte_identical phase copies the original mouth
  generator to /tmp and loads the SSM checkpoint by default, then errors ("not a --recurrence linattn checkpoint"),
  so it produced no verdict here. Byte-identical-off was already verified in the landed coupling finding
  (`87631edf`); this is a checkpoint-path defect in the validation phase to fix separately, not evidence of a leak.
- **salad-ceiling calibration:** the runner's 0.20 ceiling sits BELOW the baseline (0.25) and well below the
  0.4-0.65 collapse zone the original calibration measured. A ceiling nearer the real collapse (~0.35-0.40) would
  make scale 2.5 (neg 6/6, salad 0.346 <!--derived-->) a marginal PASS — but the direct read of that output (repetitive +
  drifting) argues the strict verdict is the honest one. Worth revisiting when the learned-valence mechanism lands.
