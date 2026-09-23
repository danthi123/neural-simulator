---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
verdict: NO-GO
---

# Brain-based affect→tone NEURAL coupling is directional-NO-GO over OPEN output (sub-delta, 6-seed) — the 2nd method NO-GO → affect-expression-over-open is a WALL for external research (2026-09-23)

The §8 affect→tone Phase-2A next-method (follow-on to the additive-bias NO-GO
`2026-09-22-affect-tone-open-output-directional-6seed-positive-asymmetric-NOGO` and the decode-ceiling diagnosis
`2026-09-23-affect-tone-decode-ceiling-diagnosis-distribution-shift-INPROGRESS`) tested the BRAIN-BASED coupling
(`BRAIN_WKV_MOUTH_AFFECT_NEURAL`: the spiking affect organ's valence drives a per-pool `excitability_drive`
neuromodulator concentration → `read_window` bridge steps → argmax over `cp_firing_states` = the SPIKING competition
selects the word, NOT host logit arithmetic). It reuses the additive-bias NO-GO's exact 6-seed directional gate +
independent lexicon + all anti-cheats. Verdict: **NO-GO**, trustworthy.

## Result — 6-seed, all 9 instrument-validity preconditions CLEAN
<!--derived from research/findings/raw/_affect_tone_neural_coupling/affect_tone_neural_coupling_verdict.json -->
- **Directional: NO-GO.** pos_gap = **+0.0209** on all 6 seeds (< the PREREGISTERED delta **0.1346**, the neutral  <!--derived-->
  tone-noise band); neg_gap = **0.0** on all 6 seeds. Neither direction crosses the gate → `directional.go=false`,
  all pos/neg states `null`. The SEM-band secondary read is also NO-GO.
- **The small effect IS the coupling (not an artifact):** attribution control shows **100% of the +0.0209 effect is  <!--derived-->
  attributable to the manipulation, 0% present in the shuffled-valence control** — so the brain's affect read DOES
  reach the tone, it is just tiny.
- **All 6 seeds numerically IDENTICAL** (pos_gap +0.0209, neg_gap 0.0, tone_pos 0.0459, tone_neg 0.025) — the linattn  <!--derived-->
  mouth is seed-invariant on this prompt set (the ckpt-per-seed precondition is TRUE, so this is the known
  seed-invariance from the additive-bias finding's Honest-Residual-3, not a ckpt-fallback bug).
- **9/9 preconditions clean:** all-36-arms-present, lexicon-disjoint-from-WARRINER (overlap=0), ckpt-resolved-per-seed,
  wkv-mouth-used (not qwen), determinism lesion==lesion_rep, fluency salad<=0.16, content-identity facts+known
  identical, moat holds, attribution-control clean. So the NO-GO is a real method verdict, not UNDEFINED.
  Artifact: `research/findings/raw/_affect_tone_neural_coupling/affect_tone_neural_coupling_verdict.json` (Verdict block).

## Interpretation — the ceiling is EXPRESSION, not the mechanism
Two methods now falsified for affect→tone over the OPEN reply: additive top-margin bias (positive-asymmetric —
`pos` correct-sign 6/6 but `neg` never negative) and this brain-based neural coupling (sub-delta both directions, even
weaker). In BOTH, the affect ORGAN reads correctly (this run: attributable, correct-signed on `pos`); the wall is the
EXPRESSED tone shift being tiny — bounded by the small live valence magnitude (~0.16 clipped) and the mouth's decode
distribution, NOT by whether the coupling is host or neural. Making the coupling brain-based (the faithful move) did
NOT enlarge the expressed effect.

## WALL — this is a deep-research point, NOT another quick lever (NO-DEFER, DR-gate)
Per the NO-DEFER law a wall defers a METHOD, never the capability — affect-expression-over-open stays open. But with
3 findings in this lane in ~1 day (`gates/deep_research_at_wall` DR territory), the next step is EXTERNAL literature
research (`bash tools/deep_research.sh "affect / emotion expression in neural language generation"`), NOT another
mechanism lever. This also matches the owner steer (memory `feedback_affect_is_multimodal_ground_dont_text`): the
text-only affect boundary is EXPECTED + a FINDING — emotion is multimodal (interoception+prosody+face+context), and
the surpass is GROUNDING, not text-cleverness. So affect→tone-over-text-alone being hard is the predicted result.

Banked next methods (for after the external research): a DISTRIBUTION-SHIFTING decode coupling that can lower
far-from-margin negative-word thresholds; a LARGER live-valence signal; affect-laden prompt/training domains; the
real-Qwen prompt-steered mouth (the declared articulation scaffold — where affect steers via the prompt, expected to
be a text-boundary/partial per the affect-is-multimodal memory).

## Honesty
Functional read-out only — the tone metric tracks the spiking affect signal; no felt/phenomenal claim. No sim/ or
webapp/ edit (the coupling is the shipped default-off `BRAIN_WKV_MOUTH_AFFECT_NEURAL`; the runner is
`_lbf_affect_tone_neural_coupling_derisk`, already on main). This does NOT retire the additive-bias shortcut nor claim
D5 "feel" DONE — affect-coloring stays load-bearing on the TEMPLATED decision field (the shipped #1-metric probe); it
is the OPEN-output expression that is walled.
