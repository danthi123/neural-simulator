---
type: finding
status: verified
date: 2026-09-17
mechanism: the culmination default-on gate for the ONE-BRAIN 11-organ pool flip — put all 8 wired cortical organs on
  ONE shared spiking pool by default (BRAIN_ONEBRAIN_WAVE3_POOL). Two gates run: (A) the per-organ answer-preservation +
  one-brain-coherence 6-seed A/B, and (B) the integrated /api/brain-chat 38-faculty no-regression battery (ON vs OFF,
  the established bar the 2026-09-16 four-flip landing used), run CPU-numpy on AWS r7i
integration_faculty: one-brain (11-organ shared cortical pool — the default-on flip)
lane: one-brain integration (the culmination)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NOT READY for default-on — HELD default-OFF. The per-organ gate PASSES (answer-preservation + one-brain
  coherence GO 6/6: every wired organ's isolated live read is preserved ON-vs-OFF, all 8 on ONE pool, the =0 escape
  reverts). BUT the integrated 38-faculty /api/brain-chat no-regression battery FAILS (all_pass=False): 5 faculties
  flagged regressed, which collapse to ONE root cause — under the flip the da-mode ENGAGEMENT signal computes
  neutral/low_engagement where the OFF arm computes focus/engaged, and that single shift cascades into 4 identical
  answer-text diffs (the engaged-mode "worth going further here" suffix is dropped on content-selection,
  in-loop-learning, moat-verify, open-ended-generation). So the pool flip preserves each organ's isolated read but
  PERTURBS the cross-faculty engagement computation — exactly the class of regression the per-organ A/B cannot see and
  the integrated battery exists to catch. The flip stays DEFAULT-OFF (byte-identical escape intact); the residual to
  close before default-on is the engagement-signal perturbation. A wall defers a METHOD (flip-as-is), never the
  capability (one shared cortical substrate) — the fix is to make the flip engagement-preserving.
runner: research/runners/onebrain_regression_battery.py (--flag BRAIN_ONEBRAIN_WAVE3_POOL) + _onebrain_11organ_pool_flip_regression.py
artifacts:
  - research/findings/raw/_regression_battery/battery_BRAIN_ONEBRAIN_WAVE3_POOL.json
  - research/findings/raw/_onebrain_11organ_pool_flip_6seed.json
external: NO-EXTERNAL-NEEDED — an integration no-regression test of the project's own pipeline (ON-vs-OFF differential),
  no host shortcut to cognition; the established default-on bar from the 2026-09-16 four-flip landing.
builds_on:
  - research/findings/raw/_onebrain_11organ_pool_flip_6seed.json
---

# One-brain 11-organ default-on flip — integrated battery catches a da-mode engagement regression; HELD default-off

The 11-organ default-on flip (all 8 wired cortical organs onto ONE shared spiking pool) is the one-brain culmination.
It passed its per-organ gate; this is the integrated gate that decides the production default-on flip.

## The two gates
<!--derived-->
- (A) Per-organ answer-preservation + one-brain coherence (from research/findings/raw/_onebrain_11organ_pool_flip_6seed.json):
  GO 6/6 — every wired organ's live chat-handler read is preserved ON-vs-OFF, all 8 organs resolve to ONE 11-organ pool
  object, and the =0 escape reverts (no organ on the pool). Necessary but, as it turns out, not sufficient.
- (B) Integrated /api/brain-chat 38-faculty no-regression battery (from
  research/findings/raw/_regression_battery/battery_BRAIN_ONEBRAIN_WAVE3_POOL.json): all_pass=False, n_faculties=38,
  n_regressed=5, n_not_exercised=2. Regressed: content-selection, moat-verify, in-loop-learning,
  da-mode-drives-response, open-ended-generation. Not exercised: wm-binding-advanced, value-driven-choice.

## The 5 regressions are ONE root cause
<!--derived-->
The per-faculty diffs show a single mechanism, not five independent breaks:
- da-mode-drives-response: ON = mode "neutral", reason "low_engagement"; OFF = mode "focus", reason "engaged".
- content-selection / in-loop-learning / moat-verify / open-ended-generation: the ON answer is identical to OFF EXCEPT
  it drops the trailing "— worth going further here." suffix — the phrase the engaged/focus da-mode appends.
So the pool flip shifts the da-mode ENGAGEMENT computation (focus->neutral) on these probes, and that one shift ripples
into the four answer-suffix diffs. The flip preserves each organ's isolated read but changes a cross-faculty signal
(engagement = novelty + richness, fed by the pooled organs), which only the full-pipeline battery exercises.

## What it means, and the residual to close

The one-brain substrate is de-risk-validated (all 11 organs co-reside answer-preservingly, GO 6/6) but the production
default-on flip is NOT ready: it perturbs the da-mode engagement signal. It is HELD default-OFF; the byte-identical
`BRAIN_ONEBRAIN_WAVE3_POOL=0` escape stands, and the flip-prep infrastructure stays on its branch
(research/onebrain-11organ-pool-flip-prep) — NOT merged. The residual is specific and tractable: identify which pooled
organ's contribution to the engagement signal (novelty/richness) shifts under co-residence and make the flip
engagement-preserving (e.g. the same value the OFF path computes), then re-run gate (B). This is the exact value of the
integrated battery — it caught a cross-faculty regression the per-organ A/B (GO 6/6) could not. Two faculties
(wm-binding-advanced, value-driven-choice) were not exercised by the probe set and need coverage before the next
adjudication. A wall defers a METHOD (flip-as-built), never the capability (one shared cortical pool).
