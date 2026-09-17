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
  in-loop-learning, moat-verify, open-ended-generation). ROOT CAUSE ISOLATED (verify-first, data-confirmed — see body):
  the 5 collapse to ONE constant ~0.16 engagement offset whose entire cause is a CURIOSITY-organ CALIBRATION mismatch.
  Engagement (default-ON `shared_salience_afferent`) reads the curiosity organ, which IS pooled (min_wave=2); its
  calibration uses `per_neuron_ou_seed=True` on the pooled read path but the legacy default False on the standalone
  build path -> ~8x responsiveness split (want_novel_hz 15.45 ON vs 126.56 OFF, confirmed from the arm artifacts) ->
  the 0.165 offset (turn_engagement == shared_salience.normalized bit-for-bit). It is NOT the novelty organ (properly
  RNG-isolated) and NOT a generic seed-trap. Gate (A) was blind because its curiosity probe points (0.95, 0.0) are
  exactly the calibration anchors. So this is a NARROW, NAMED config mismatch — the flip is very likely LANDABLE once
  curiosity calibrates identically on both paths. FIX: pass `per_neuron_ou_seed=True` + matching OU config to curiosity's
  standalone build path (`curiosity_production_organ._build_one`/`build_curiosity_bridge`), then re-run the battery.
  Flip stays DEFAULT-OFF (escape intact) until that re-test reads all_pass. A wall defers a METHOD, never the capability.
runner: research/runners/onebrain_regression_battery.py (--flag BRAIN_ONEBRAIN_WAVE3_POOL) + _onebrain_11organ_pool_flip_regression.py
artifacts:
  - research/findings/raw/_regression_battery/battery_BRAIN_ONEBRAIN_WAVE3_POOL.json
  - research/findings/raw/_onebrain_11organ_pool_flip_6seed.json
external: NO-EXTERNAL-NEEDED — an integration no-regression test of the project's own pipeline (ON-vs-OFF differential),
  no host shortcut to cognition; the established default-on bar from the 2026-09-16 four-flip landing.
builds_on:
  - research/findings/raw/_onebrain_11organ_pool_flip_6seed.json
---

# One-brain 11-organ default-on flip — a curiosity-organ calibration mismatch shifts da-mode engagement; HELD default-off pending a one-line fix + re-test

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

## ROOT CAUSE ISOLATED (verify-first, data-confirmed — supersedes the two earlier hypotheses)
<!--derived-->
The ON-vs-OFF `turn_engagement` difference is a NEAR-CONSTANT additive offset (OFF-minus-ON ~0.13-0.19 on every one of
19 probes: well 0.165, question 0.173, held 0.185, chase 0.133, emo 0.153, ...), while engagement itself varies by
content within each arm (0.38-0.69) — so the mechanism works and the flip just subtracts a fixed ~0.16 baseline. My
first-pass hypotheses were BOTH wrong about the mechanism and are retired: it is neither a generic build-ordering
seed-trap nor the novelty organ (SpikingNoveltyHabituationOrgan is properly RNG-isolated via
`DaModeDrivesWorkspace._isolated()`, and is not pooled). The actual, data-confirmed cause:

- The engagement path (`webapp/da_mode_drives_chat.py:396-405`) takes the isolated novelty read, then — because
  `shared_salience_enabled()` is default-ON — OVERWRITES it with `shared_salience_afferent.read_salience(...)`, which
  reads the CURIOSITY organ (`curiosity_production_organ.get_organ`, a process-global singleton). `turn_engagement`
  equals `shared_salience.normalized` bit-for-bit on all 19 probes.
- Curiosity IS one of the 8 wave3-pooled organs (on the flip-prep branch its `get_organ` resolves through
  `get_merged_cortical_pool(min_wave=2)`). Its calibration (`want_novel_hz`/`want_familiar_hz`) uses a DIFFERENT recipe
  when pooled (`_read_want_shared`, which sets `per_neuron_ou_seed=True`) vs standalone
  (`_build_one`->`build_curiosity_bridge`, which leaves `per_neuron_ou_seed` at its legacy default False).
- Confirmed from the committed arm artifacts (no compute): ON calib {want_novel_hz 15.45, want_familiar_hz 0.0} vs OFF
  {126.56, 5.21} — an ~8x responsiveness split + a different transduction SHAPE, mapping the same raw 0.734 to
  normalized 0.483 (ON) vs 0.648 (OFF) = the exact 0.165 offset. The whole offset is this ONE calibration divergence.
- Why gate (A) was blind: `_onebrain_11organ_pool_flip_regression`'s curiosity probe uses `(0.95, 0.0)` — EXACTLY the
  calibration anchor points, where `normalized` is tautologically ~1/~0 in both arms, so a mid-range curve-shape
  difference cannot show.

So this is NOT an unfixable crowding artifact and NOT a generic seed-trap — it is a narrow, named config mismatch
(`per_neuron_ou_seed` set on the pooled read path but not the standalone build path), so the flip is very likely
LANDABLE once the two paths calibrate curiosity identically.

## What it means, and the residual to close

The one-brain substrate is de-risk-validated (all 11 organs co-reside answer-preservingly, GO 6/6). The production
default-on flip is HELD default-OFF (byte-identical `BRAIN_ONEBRAIN_WAVE3_POOL=0` escape intact; flip-prep stays on its
branch research/onebrain-11organ-pool-flip-prep, NOT merged) — but per the CORRECTION above, it is NOT confirmed to have
a real functional regression — the ROOT CAUSE section isolates it to the curiosity-organ calibration recipe. The
ordered next steps (post-gaming; the fix edits + re-test run the sim, which contends with the owner's game): (1) FIX —
pass `per_neuron_ou_seed=True` + matching OU config (`ou_std_current_pA`/`ou_mean_current_pA`/`ou_tau_ms`) to curiosity's
standalone build path so both the pooled read and the standalone build calibrate under the same OU regime; (2) CHEAP
CONFIRM (sub-second numpy) — `get_organ(seed=42).ensure_built()` under BRAIN_ONEBRAIN_WAVE3_POOL=1 vs =0 and compare
`calib` + `salience_of()` at raw 0.3/0.5/0.65/0.734/0.84: the ON/OFF curves should now agree within ~0.02; (3) DECISIVE
— re-run `onebrain_regression_battery --flag BRAIN_ONEBRAIN_WAVE3_POOL`; on all_pass the flip is LANDABLE (cherry-pick
research/onebrain-11organ-pool-flip-prep, `_WAVE3_POOL_DEFAULT_ON=True`); (4) also extend gate (A)'s curiosity probe to
assert `salience_of()` at intermediate raw points (not just the anchors) so this class of divergence can't ship again.
Two faculties (wm-binding-advanced, value-driven-choice) were not exercised by the probe set and need coverage. A wall
defers a METHOD (flip-as-built), never the capability (one shared cortical pool).
