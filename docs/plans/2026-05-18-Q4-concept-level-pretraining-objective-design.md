---
type: plan
status: live
date: 2026-05-18
---

# Q4 — Concept-level prediction objective -> word-discriminative cortex features (the cheap, genuinely-untested kernel; the heavy word-level-pretraining build is ALREADY-assessed-out-of-scope and is NOT re-ground) (design)

> Standing autonomy: documented design calls; brainstorm ->
> writing-plans -> subagent-driven-development -> pre-registered gate
> -> honest propagation EVERY outcome. No config-cranking, no
> overclaim, the no-confab moat byte-identical. CORRECTED OPERATING
> MODE: Q2R FAILed honestly -> non-stop pivot (no owner-deferral) to
> this final pivot-queue item Q4.

## Status

Pivot-queue Q4 (FINAL queued arm). Q1 VOID / Q2 FAIL / Q3 cheap-VOID /
Q2R FAIL all propagated. This design HONESTLY narrows Q4 to the only
genuinely-untested, cheaply-decisive kernel and pre-registers the cheap
falsify-first probe as the gate for whether a heavy build is even
warranted.

## The decisive grounding constraint (cheap context-exploration, done FIRST)

- Phase 2.x BPTT/tokenizer assets ARE present on `main`
  byte-UNMODIFIED-reusable: `sim/surrogate_grad.py` (80L),
  `sim/bptt_snn.py` (291L), `sim/bptt_snn_gpu.py` (173L),
  `sim/char_tokenizer.py` (168L). So Q4 is constructible from main
  (NOT a path-f-hybrid-only feasibility blocker).
- BUT `research/findings/2026-05-09-Phase-2.3b-50M-cosine-REFUTED.md`
  (written with full anti-cheat discipline) already established:
  Phase-2.3a (134K) char-level next-char features -> 22% W->A
  (NEGATIVE, < 28% random); Phase-2.3b scaled 375x (134K->50M) and
  inter-direction-word cosine got WORSE (0.72 -> 0.85) -> the
  "scale fixes transfer" thesis is REFUTED; the wrong-objective
  explanation is favored. That doc EXPLICITLY enumerated the remaining
  options and assessed:
  - **Option 1: word-level pretraining -- "much larger scope; not
    testable at our scale".**
  - Option 2: contrastive feature-separating objective -- "research
    direction; weeks of work".
  - Option 3: 1B+ cloud scale -- out of budget.
  And concluded path-f-hybrid Phase-2 is "a documented dead-end at
  single-3090 scale".

**Honest consequence (load-bearing, like Q2R's 24-prop-KB
constraint):** a HEAVY Q4 "word-level pretraining" build IS Option 1,
already honestly assessed not-testable at local scale. Grinding it
anyway to "exhaust the queue" would be going-through-the-motions /
redundant config-cranking against a prior disciplined finding --
forbidden, and dishonest. Q4 must NOT re-grind Option 1.

**What is genuinely UNTESTED and cheaply decisive:** Phase-2.3a/b
tested only the CHAR-level next-char objective. They did NOT test a
CONCEPT-level prediction objective matched to the tiny VALIDATED v16
16-concept substrate. The single open, cheaply-falsifiable question:
does a concept-level objective produce word/concept-DISCRIMINATIVE
cortex features (low inter-concept cosine) where char-level provably
produced NON-discriminative ones (cosine 0.72-0.85)? This is the exact
diagnosed failure mode; it is cheap (tiny scale, the v16 substrate is
already validated + small); it is genuinely unexplored. It is NOT the
out-of-scope Option 1 (full word-level language pretraining) -- it is
a minimal concept-prediction feature-discriminativeness probe.

## Falsify-cheaply precursor (the GATE -- runs FIRST, decides everything)

A throwaway pure-numpy/CPU probe (prefix `_`, deleted post-decision,
recorded evidence to `research/findings/raw/`), reusing the
on-`main` byte-UNMODIFIED `sim/char_tokenizer.py` +
`sim/bptt_snn.py` (the validated Phase-2.1 BPTT) for the char-level V1
baseline; net-new = ONLY a concept-level prediction objective + an
inter-concept feature-discriminativeness metric. Pre-registered
THREE-STATE (own frozen `_Q4_*`, NEVER tuned, instrument-validity
FIRST, fail-closed, VOID strictly distinct from FAIL):

- **V1 (instrument soundness):** the CHAR-level next-char objective
  trained on the same tiny corpus REPRODUCES the known
  non-discriminative regime -- mean pairwise inter-concept feature
  cosine >= a frozen high bar `_Q4_CHAR_COS_MIN` (e.g. ~0.65,
  consistent with the recorded 0.72-0.85). This proves the probe's
  feature-extraction + cosine metric can SEE the documented failure
  (so a concept-level improvement would be real signal, not a metric
  artifact).
- **Science:** the CONCEPT-level prediction objective, same tiny
  architecture/scale, yields mean pairwise inter-concept cosine <=
  a frozen `_Q4_CONCEPT_COS_MAX` (decisively word-discriminative;
  e.g. <= 0.40) AND a decisive separation margin over the char-level
  V1.
- **Controls (must fail):** `shuffled_concept_labels` (concept
  targets permuted -> must NOT become discriminative), `random_init`
  (no training -> non-discriminative), `wrongsign` if applicable.
- **Pre-registered SCALE LADDER even in the cheap probe:** concept
  vocabulary C in {4, 8, 16} (the validated v16 tiers);
  SCALE-CONFIDENT-cheap iff every rung instrument-sound + science met
  AND discriminativeness does NOT degrade with C up to a frozen tol
  AND holds at the largest rung (C=16). This directly tests the
  owner's scale-confidence question at the cheap tier.
- **Outcome rule:** GREEN (V1 sees the failure; concept-level is
  decisively discriminative; controls fail; scale-confident-cheap) ->
  the heavy build is WARRANTED -> proceed to writing-plans for the
  in-substrate Q4 (concept-pretrained cortex rewired into v16).
  NEGATIVE or VOID -> NO heavy build; propagate honestly; **the pivot
  queue is then GENUINELY EXHAUSTED and the honest terminal synthesis
  is delivered (that synthesis IS the deliverable -- it is NOT a stop
  mid-task and NOT a spin).**

If the cheap probe shows concept-level features are ALSO
non-discriminative (or a sound discriminating cheap instrument is not
constructible), that is the honest, decisive close of the final arm --
consistent with, and the terminal confirmation of, the 8x-triangulated
meta-finding.

## Architectures considered (2-3) with honest ceilings + risks

- **Q4-a (REJECTED, documented): heavy word-level language pretraining
  -> cortex adapter -> Bridge.** This IS Phase-2.3b-REFUTED's Option 1
  ("much larger scope; not testable at our scale"); re-grinding it is
  redundant config-cranking against a prior disciplined finding.
  Rejected; recorded so the rejection is auditable.
- **Q4-b (RECOMMENDED -- and its cheap probe is the gate): concept-
  level prediction objective, minimal scale, feature-
  discriminativeness tested cheaply FIRST.** Reuse the validated
  on-main BPTT byte-UNMODIFIED for the char-level V1; net-new = a
  concept-level objective + the inter-concept cosine metric + the
  pre-registered `_Q4_*` THREE-STATE/ladder. GREEN -> heavy build =
  the concept-pretrained cortex rewired into the validated v16
  concept-pool substrate (its OWN pre-registered in-substrate
  THREE-STATE + scale ladder, written by writing-plans only IF the
  cheap probe is GREEN). Honest ceiling: even a heavy PASS = a
  small-scale concept-discriminative-feature -> v16-binding
  scale-confidence PoC, explicitly NOT fluent composition / NOT an
  LLM. Risk: concept-level may ALSO fail to be discriminative at this
  scale (the cheap probe honestly catches that -> terminal synthesis).
- **Q4-c (folded into Q4-b's probe): pure feature-discriminativeness
  only, no Bridge.** This is exactly the cheap precursor of Q4-b;
  folded in, not a separate heavy arm.

**Recommendation: Q4-b**, with its cheap feature-discriminativeness
probe as the MANDATORY gate (Q3 pattern: design -> cheap probe ->
GREEN heavy / NEGATIVE-or-VOID propagate + terminal synthesis).

## Honest ceiling (stated up front, NEVER spun)

- **IS (only if cheap GREEN then heavy SCALE-CONFIDENT-PASS):** a
  concept-level-pretrained cortex yields word-discriminative features
  that, rewired into the validated v16 substrate, give a
  binding/generative capability holding/improving across a
  pre-registered local scale ladder with no architectural ceiling --
  the owner's scale-confidence deliverable, at minimal scale.
- **IS NOT (never spun; the 8x-triangulated reality stands):**
  open-ended fluent composition, an LLM, GPT-class,
  conversation-solved, or an overturning of the established honest
  finding that NO tested local architecture has met scale-confidence.
  Q4 is the FINAL queued arm; a faithful NEGATIVE/VOID exhausts the
  queue and the honest terminal synthesis is the deliverable.

## Anti-cheat plan (non-negotiable)

Cheap falsify-first FIRST (the gate; honest GREEN->heavy /
NEGATIVE-or-VOID->propagate + terminal synthesis, recorded);
pre-registered FIXED-bar `_Q4_*` THREE-STATE + scale ladder (own
frozen constants, NEVER tuned; net-new core does NOT import/mutate any
existing `*_core`; no new GLOBAL bar); reuse `sim/char_tokenizer.py` +
`sim/bptt_snn.py` byte-UNMODIFIED for the V1 baseline; if a heavy
build is warranted: dedicated ADVERSARIAL REVIEWER on load-bearing
modules BEFORE Phase B (probe: is the concept-level discriminativeness
genuine and not a metric artifact; is char-level V1 a faithful
reproduction of the documented 0.72-0.85 failure; is the v16 substrate
reused byte-UNMODIFIED; any new autograd beyond the byte-UNMODIFIED
validated BPTT; movable frozen bars); controller trust-but-verify
EVERY diff with the FULL protected set byte-empty (original protected +
constrained_decode_core/gate + q2r_core/gate + engram_bootstrap_gate +
every frozen `*_core` + the no-confab moat `abstention_gate`+test 7/7 +
grounded_decode + generator_g_core + tiny_transformer + bpe_tokenizer +
sim validated modules + text_minimal_isolation); mandatory smell-test
scrutinizing a nominal PASS HARDER than a FAIL (recompute from the
single recorded JSON; V1 genuine; controls fail; scale-confidence
recomputed; NO re-run/NO bar-tuning/NO overclaim); honest propagation
EVERY outcome (findings + capability_status pillar n=78 + schema green
+ push BOTH remotes). MONITORING DISCIPLINE: any decisive run uses the
Bash `run_in_background` parameter OR foreground -- NEVER a bare
`nohup` with a false "I will be notified" claim; completion ACTIVELY
confirmed before any result is claimed (the cheap probe is pure-numpy/
CPU and fast -> run foreground, synchronously observed).

## Build sequence (subagent-driven; anti-cheat) -- detailed by writing-plans ONLY IF cheap GREEN

Cheap falsify-first probe (controller-run, foreground, recorded) is
the gate. If GREEN: Task 0 grounding pin -> Task 1 `q4_core.py`
(frozen `_Q4_*` THREE-STATE + scale-confidence, fully specified) ->
Task 2 `q4_gate.py` (concept-pretrained cortex rewired into the
byte-UNMODIFIED v16 `build_biological_brain_regions`; kill-safe;
`--tiny` smoke) -> Task 3 dedicated adversarial reviewer BEFORE Phase
B -> Phase B no-harm -> Task 5 CONTROLLER-ONLY decisive run + smell-
test + honest propagation n=78 both remotes. If cheap NEGATIVE/VOID:
NO heavy build; propagate the finding + the honest TERMINAL SYNTHESIS
(the pivot queue genuinely exhausted; the near-exhaustively-
triangulated honest conclusion delivered to the owner) -- both remotes.
