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

## External research (DR-gate, 2026-09-23) — the proven mechanism is affect-CONDITIONED GENERATION, not decode-bias
The deep-research-at-wall external half (recorded in `research/queue/.external_searches.jsonl`) found the proven
class for making affect load-bearing on generated text: **Affect-LM (Ghosh, Chollet, Laksana, Morency, Scherer,
ACL 2017)** conditions the LM GENERATION with a beta-gated affect-category energy term (NOT a post-hoc decode bias);
continuous **VAD-conditioning** (Guo, Xu & Chua, arXiv:2111.04730 "Emotional Prosody Control"; EmotiCrafter
arXiv:2501.05710) decouples affect from content by conditioning generation on a continuous valence-arousal embedding.
This directly explains BOTH our NO-GOs: additive top-margin bias AND neuromodulator-at-decode are both DECODE-POINT
interventions; the proven mechanism conditions the GENERATION REPRESENTATION. So the next affect method is an
affect-CONDITIONED mouth — train-time affect conditioning of the mouth's generation, or the real-Qwen prompt-
conditioned mouth (per the affect-is-multimodal owner steer: ground it, don't decode-clever it) — NOT another decode
lever. This is a larger build with an owner-steer fork (train-time-conditioning vs Qwen-prompt-conditioning, and the
affect-laden-data direction), flagged for the owner per the autonomous charter. Sources:
[Affect-LM (ACL 2017)](https://www.researchgate.net/publication/318740920), [Emotional Prosody Control
(arXiv:2111.04730)](https://arxiv.org/html/2111.04730).
