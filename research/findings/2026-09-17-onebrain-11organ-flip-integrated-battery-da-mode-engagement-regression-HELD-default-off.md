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
  in-loop-learning, moat-verify, open-ended-generation). CORRECTION (same-day, verify-first — see the body): on reading
  all 19 probes the ON-vs-OFF engagement difference is a NEAR-CONSTANT ~0.16 additive offset (content-independent), and
  the novelty organ that feeds engagement is NOT one of the 11 pooled organs — so a BUILD-ORDERING / RNG-STATE confound
  (the seed trap: the merged-pool build advances the global RNG differently than separate organ builds) is the LEADING
  cause, NOT a demonstrated functional regression of the pooled cognition. So the "5-faculty regression" is DOWNGRADED
  to "an uncontrolled build-ordering offset, cause not yet isolated." The flip stays DEFAULT-OFF (correct conservative
  call, byte-identical escape intact) but is NOT confirmed-bad. DECISIVE NEXT TEST: re-run the integrated battery with
  the engagement path's RNG state CONTROLLED (seed the novelty/SNc read independently of pool-build order) — if the
  ~0.16 offset vanishes, the flip is answer-preserving and LANDABLE; if it persists, it is a real residual. A wall
  defers a METHOD, never the capability (one shared cortical substrate).
runner: research/runners/onebrain_regression_battery.py (--flag BRAIN_ONEBRAIN_WAVE3_POOL) + _onebrain_11organ_pool_flip_regression.py
artifacts:
  - research/findings/raw/_regression_battery/battery_BRAIN_ONEBRAIN_WAVE3_POOL.json
  - research/findings/raw/_onebrain_11organ_pool_flip_6seed.json
external: NO-EXTERNAL-NEEDED — an integration no-regression test of the project's own pipeline (ON-vs-OFF differential),
  no host shortcut to cognition; the established default-on bar from the 2026-09-16 four-flip landing.
builds_on:
  - research/findings/raw/_onebrain_11organ_pool_flip_6seed.json
---

# One-brain 11-organ default-on flip — a da-mode engagement OFFSET (build-ordering confound suspected); HELD default-off pending an RNG-controlled re-test

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
into the four answer-suffix diffs. The flip preserves each organ's isolated read but changes the cross-faculty
engagement signal (engagement = novelty + richness), which only the full-pipeline battery exercises.

## CORRECTION (same-day, verify-first): the offset is CONSTANT — a build-ordering/RNG confound is the leading cause, NOT a demonstrated functional regression
<!--derived-->
Reading the two arms' `da_drives.turn_engagement` across ALL 19 probe turns (arm_on/arm_off) shows the ON-vs-OFF
difference is a NEAR-CONSTANT additive offset, not a content-specific break: OFF-minus-ON is ~0.13-0.19 on every probe
(well 0.165, question 0.173, unknown 0.182, held 0.185, chase 0.133, emo 0.153, bc_b 0.160, ...). Engagement itself DOES
vary by content within each arm (0.38-0.69), so the mechanism works — the flip just subtracts a fixed ~0.16 baseline.
Two facts make a BUILD-ORDERING / RNG-STATE confound (the CLAUDE.md seed trap: each build advances the global RNG, and
a merged pool build advances it differently than separate organ builds) the leading hypothesis over a functional
regression: (1) the offset is content-independent (a baseline shift, not a per-message-novelty change); (2) the novelty
organ that feeds engagement (SpikingNoveltyHabituationOrgan) is NOT one of the 11 pooled organs, so the pool cannot
change its computation directly — only the global RNG state at its build/read point can differ between the merged-build
(ON) and separate-build (OFF) arms. So this is most likely the same class of confound that once cost the deep-credit arc
months (different neurons at the same seed from a different build order), surfacing here as a shifted engagement
baseline. It is therefore NOT yet established as a real regression of the pooled cognition.

DECISIVE NEXT TEST (before calling it either way): re-run the integrated battery with the engagement path's RNG state
CONTROLLED — e.g. seed the novelty organ / SNc-afferent read independently of the pool-build order (the same fix pattern
as tests/test_determinism.py::TestSubstrateActuallySeeded), or build the pool but read engagement from a fixed-seed
fresh substrate. If the ~0.16 offset VANISHES under RNG control, the flip is answer-preserving after all and is
LANDABLE; if it persists, it is a real residual to close. Until that test runs, the flip stays default-OFF (correct
conservative call) but the "5-faculty regression" is DOWNGRADED to "an uncontrolled build-ordering offset, cause not yet
isolated."

## What it means, and the residual to close

The one-brain substrate is de-risk-validated (all 11 organs co-reside answer-preservingly, GO 6/6). The production
default-on flip is HELD default-OFF (byte-identical `BRAIN_ONEBRAIN_WAVE3_POOL=0` escape intact; flip-prep stays on its
branch research/onebrain-11organ-pool-flip-prep, NOT merged) — but per the CORRECTION above, it is NOT confirmed to have
a real functional regression. The ordered next steps are: (1) RUN THE RNG-CONTROLLED RE-TEST — seed the engagement path
(novelty organ / SNc afferent) independently of the pool-build order and re-run gate (B); this decides whether the
~0.16 engagement offset is a build-ordering artifact (=> flip is LANDABLE) or a real residual. (2) Only if it persists:
isolate and close the real engagement perturbation. The methodological lesson stands regardless: an ON-vs-OFF
integration battery that rebuilds the substrate differently per arm must control the global-RNG build order, or a pure
seed-trap offset masquerades as a regression (this is why the per-organ A/B, which the seed trap also governs but
symmetrically, read GO while the pipeline battery flagged a diff). Two faculties
(wm-binding-advanced, value-driven-choice) were not exercised by the probe set and need coverage before the next
adjudication. A wall defers a METHOD (flip-as-built), never the capability (one shared cortical pool).
