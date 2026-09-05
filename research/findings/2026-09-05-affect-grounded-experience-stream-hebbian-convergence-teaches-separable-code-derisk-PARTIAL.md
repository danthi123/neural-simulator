---
type: finding
status: partial
claim_check: measured
date: 2026-09-05
mechanism: a grounded-experience STREAM (a per-concept body-state US delivered via COMFORT/DISCOMFORT/AROUSAL interoceptive relay pools, board #49/#84 structure, a declared world/body stand-in) + an EMERGENT rate-Hebbian competitive CONVERGENCE that TEACHES the affect concept code (Oja-stabilized, k-WTA, small-random init, never hand-set), read by the SAME validated code-separability CEILING instrument -- replacing the grounded-code PARTIAL's RAW ORACLE FUSION with a genuine synaptic convergence + a held-out generalization test
lane: affect-learned-gate-retirement (rank-7)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_grounded_experience_stream_hebbian_derisk.py
artifacts:
  - research/findings/raw/_affect_grounded_experience_stream_hebbian_6seed.json
  - research/findings/raw/_affect_grounded_experience_stream_hebbian.json
builds_on:
  - research/findings/2026-09-05-affect-gate-grounded-concept-code-lifts-ceiling-requirements-derisk-PARTIAL.md
  - research/findings/2026-09-05-affect-gate-embodied-US-necessary-not-sufficient-concept-code-must-be-grounded-BOUNDARY.md
  - research/findings/2026-08-19-embodied-affect-interoception-GO.md
  - research/findings/2026-07-02-emerge34-perception-grounded-emergence-GO.md
verdict: >
  PARTIAL / de-risk (6-seed). The prior PARTIAL de-risked the grounded-concept-code REQUIREMENTS with a RAW ORACLE
  body-state block FUSED onto the text code (hand-set axes, read directly). This runner builds the NAMED next
  mechanism it could not: an EMERGENT rate-Hebbian competitive CONVERGENCE (brain-based -- small-random init,
  k-WTA competition, Oja-stabilized, never hand-set) that TEACHES a concept code from a grounded-experience STREAM
  (a per-concept body-state US delivered into comfort/discomfort/arousal interoceptive relay pools, the board
  #49/#84 structure; the US is a declared world/body stand-in), read by the SAME validated separability CEILING.
  CORE CLAIM VALIDATED: at clean/full grounding the emergent convergence teaches a FULLY separable concept code
  where the text code cannot (learned ceiling saturates vs the text code's near-zero, all 6 seeds), grounding is
  LOAD-BEARING (the no-body-state LESION and the shuffle-binding control both stay at the text baseline), and -- the
  decisive advance the oracle fusion could not test -- the code is TAUGHT not HANDED: HELD-OUT concepts never seen by
  the Hebbian weights separate perfectly, so the convergence learned a REUSABLE text+body-state -> code map that
  generalizes, not a per-concept lookup. The instrument is validated (synthetic clean code saturates, text code
  reads below the discrimination floor). The pre-registered strict GO bar is NOT met (GO=False, only G1 fails):
  at a REALISTIC interoceptive-noise operating point the worst-case zero-FP ceiling stays near zero. The
  DECISIVE SHARPENING (differs from the oracle fusion): the binding constraint for a LEARNED code is afferent
  NOISE, not the zero-FP criterion -- the zero-noise coverage frontier clears the bar at moderate coverage, but ANY
  interoceptive noise collapses the worst-case, and (unlike the oracle fusion, which recovered under a small
  false-positive tolerance) relaxing the FP tolerance does NOT rescue the learned code, because the noise is baked
  into the assemblies at learning time. Adding the biologically-correct companion process the ungated rule proxied away -- THREE-FACTOR (US /
  neuromodulatory) gated plasticity, label-free -- lifts the noisy realistic point substantially but still does not
  clear the worst-case bar. CONCLUSION: the grounded-experience-stream + Hebbian-convergence arc is DE-RISKED and
  its core mechanism BUILT + validated; the residual is a QUANTIFIED demand on grounded-signal QUALITY (the teacher
  must deliver LOW-noise grounding, a HIGHER bar than the oracle-fusion requirements map implied) plus the named
  next rungs. A method/requirements verdict, not a capability wall (THE LAW). Additive, default-off, numpy-CPU, NO
  sim/ edit; the host _STRONG_MARGIN gate in affect_production_organ.py is UNCHANGED (this file wires nothing).
lane_wall: affect salience gate (which words may move mood) -- rank-7 / affect-learned-gate-retirement
---

# Affect salience gate: an EMERGENT Hebbian convergence over a grounded-experience stream TEACHES a separable, generalizing concept code -- the grounded-teacher mechanism is built, with afferent noise the newly-quantified binding constraint

## The question this answers
`2026-09-05-affect-gate-grounded-concept-code-lifts-ceiling-requirements-derisk-PARTIAL.md` proved (6-seed) that a
grounded body-state axis makes the affect concept code separable where text cannot, but it used a RAW ORACLE block
FUSED onto the text code -- the grounded axes were HAND-SET and read directly, so it was a REQUIREMENTS map, not a
learned code, and it could not ask whether a real synaptic mechanism can LEARN such a code or whether that code
GENERALIZES. It named the next build: a grounded-experience stream via the interoceptive relay + a HEBBIAN
convergence (the vision->concept `_genfrontier_capstone` template) that TEACHES the code. This runner builds exactly
that and asks: **does an EMERGENT Hebbian convergence, fed a per-concept body-state through interoceptive relay
pools, TEACH a concept code that (a) lifts the separability ceiling the text code cannot, (b) is load-bearing on the
grounding, and (c) GENERALIZES to concepts the Hebbian weights never saw?**

## What ran
`research/runners/_affect_grounded_experience_stream_hebbian_derisk.py` (SIM_BACKEND=numpy, 6-seed
42/43/44/100/101/102), on the SAME 164-word partition (102 affect + 62 neutral) the prior boundaries used. A
GROUNDED-EXPERIENCE STREAM delivers a signed affect current (Warriner magnitude, the world/body US stand-in) into
COMFORT (+affect), DISCOMFORT (-affect), and AROUSAL (|affect|) interoceptive relay pools (board #49/#84 structure),
at coverage rho and afferent noise sigma. A concept-assembly layer (M=48 rate neurons) learns feedforward weights by
a competitive, Oja-stabilized HEBBIAN rule (k-WTA=12, 40 epochs, small-random init -- EMERGENT, never hand-set) over
[L2(text) | L2(interoceptive relay)]. The LEARNED concept code = the divisively-normalized assembly response, read
by the reused, validated supervised ceiling probe (`code_separability_ceiling`). Arms: text-only baseline; the
grounded-taught learned code across a (rho x sigma) grid (grounded-recall); LESION (no body-state at learning);
SHUFFLE (concept<->body-state binding destroyed); HELD-OUT (convergence trained on OTHER concepts -> generalization);
TEXT-ONLY TRANSFER (grounding absent at test); a THREE-FACTOR US-gated (neuromodulatory, label-free) convergence;
the reused synthetic-clean instrument validation; and a relaxed-FP sensitivity sweep.

## Derived -- the measured numbers (all direct reads of research/findings/raw/_affect_grounded_experience_stream_hebbian_6seed.json)
<!--derived: every value below is read directly from the cited 6-seed artifact -->
- **Text-only ceiling (reproduces the BOUNDARY):** 0.059 worst-case / 0.020 mean recall@FP0. **Instrument
  validated:** synthetic clean code 1.000 worst-case; text code < the 0.2 discrimination floor (G3 pass).
- **The emergent Hebbian convergence TEACHES a separable code:** at clean/full grounding (rho=1.0, sigma=0.0) the
  learned ceiling is **1.000 worst-case, all 6 seeds** -- vs the text code's ~0. The zero-noise COVERAGE frontier
  (sigma=0.0) clears the 0.5 bar worst-case at **rho >= 0.6** (rho=0.6 -> 0.598, 0.8 -> 0.804, 1.0 -> 1.000).
- **Grounding is LOAD-BEARING (G2 pass):** the no-body-state LESION reads 0.069 worst (0.088 at the clean point) and
  the shuffle-binding control 0.059 worst -- both within margin of the text baseline; the convergence machinery /
  extra dims do not manufacture the lift.
- **The code is TAUGHT not HANDED (G2b pass) -- the advance the oracle fusion could not test:** HELD-OUT concepts
  (never in the Hebbian weight training, 30% held out) separate at **1.000 worst-case, all 6 seeds**, at clean/full
  grounding -- so the convergence learned a REUSABLE text+body-state -> code map that generalizes.
- **Pre-registered strict bar (worst-case learned ceiling @ joint-FP=0 at the realistic rho=0.6/sigma=1.0):**
  **0.010 -> GO=False (G1 fail).** The whole realistic column (sigma>=0.5) is near zero worst-case at every coverage.
- **TEXT-ONLY TRANSFER (grounding present only at learning, absent at test):** 0.010 worst / 0.023 mean -- grounding
  during learning does NOT reorganize the text->concept map so the WORD ALONE separates; the body-state must be
  present in the concept representation at recall (an honest read of the embodiment view: interoception is part of
  the concept, re-instantiated, not a residue left in the text weights).
- **THREE-FACTOR US-gated (neuromodulatory-gated, label-free) plasticity (reported):** at the realistic point,
  0.049 worst / 0.101 mean at FP0 (up from 0.010 worst ungated) and 0.108 worst / 0.208 mean at 5% FP; clean/full
  1.000; lesion 0.039. Gating helps substantially in the mean but still does not clear the worst-case bar.
- **Relaxed-FP does NOT rescue the LEARNED code (contrast with the oracle fusion):** grounded rho=0.6/sigma=1.0
  reads 0.010 -> 0.039 (5% FP) -> 0.078 (10% FP) worst-case, only marginally above the lesion/shuffle controls at
  the same FP -- whereas the oracle FUSION recovered to 0.43 worst at 5% FP. The noise corrupts the assemblies at
  learning time; a supervised ridge could down-weight noisy oracle dims, a Hebbian convergence cannot un-learn them.
- Required coverage spec by sigma: {0.0: rho>=0.6 clears; 0.5/1.0/2.0: none clears worst-case}. GO=False; failed
  gate: G1. G2 (load-bearing), G2b (generalizes), G3 (instrument) PASS.

## Reading it (no-defer)
The named next mechanism is BUILT and its core claim holds: an EMERGENT synaptic convergence -- not a hand-set
oracle block -- TEACHES a concept code that fully separates affect from register-neutral words where the text code
cannot (clean/full 1.000 vs ~0), the lift is the grounding (lesion + shuffle stay at baseline), and the taught code
GENERALIZES to held-out concepts (1.000). That last result is the substantive advance over the oracle fusion, which
by construction could not distinguish a learned generalizing map from a per-concept lookup. The strict GO bar failed
for ONE reason, now quantified: for a LEARNED code the binding constraint is afferent NOISE, not the zero-FP
criterion. The zero-noise coverage frontier clears at ~60% coverage, but any interoceptive noise collapses the
worst-case, and -- unlike the oracle fusion -- relaxing the FP tolerance does not rescue it, because the noise is
baked into the assemblies during Hebbian learning. This is the wall-reframe in action: the real system runs a
COMPANION PROCESS the ungated rule proxied away -- neuromodulatory (US-gated, three-factor) plasticity so that
noise-only concepts do not bind. Adding it (label-free, gated by the delivered arousal salience) lifts the noisy
point substantially in the mean but not past the worst-case bar. So the residual is a QUANTIFIED demand on the
grounded teacher's signal QUALITY (LOW noise, sigma<~0.5, not merely adequate coverage) -- a HIGHER bar than the
oracle-fusion requirements map implied -- plus the named next rungs. This SHARPENS the surpass; it is not a wall.

## The scoped next build (named, not deferred)
1. **Low-noise grounding** -- the teacher must deliver a clean per-concept body-state (sigma<~0.5); the zero-noise
   frontier then clears even the strict zero-FP bar at ~60% coverage.
2. **Stronger three-factor / homeostatic gating** -- the US-gated variant helps; a homeostatic threshold that
   suppresses false-grounding on neutral concepts (or reward-gated, three-factor-with-eligibility) should push the
   worst-case further, matching the affect lane's own selforg_opponent_weights discipline (the US is the third
   factor).
3. **A fully-spiking on-substrate convergence** -- reuse `_genfrontier` build_propagation_bridge (rate-Hebbian ->
   NMDA concept spikes, held-out), GPU-queued, so the concept code is read off cp_firing_states, not a numpy matrix.
4. **A real grounded world** -- the standing residual across this whole arc: a world that experiences the
   conversational vocabulary with real bodily consequences, delivering the per-concept US the interoceptive relay
   would carry.

## Honest scope + residuals
Additive, default-off, numpy-CPU, reuse-by-import, no `sim/` edit; `_STRONG_MARGIN` unchanged (nothing wired).
(1) The body-state US is a declared ORACLE STAND-IN for a grounded-perception teacher's output (the SAME stand-in
the embodied-US + grounded-code runners used); this measures whether a convergence CAN teach a separable code GIVEN
such a stream and at what signal quality -- it does NOT deliver a real grounded world (named next build). (2) GO
here would mean the grounded-TEACHER convergence arc is de-risked, NOT that the gate is retired; this is PARTIAL.
(3) rate-Hebbian (numpy-CPU) convergence; a fully-spiking on-substrate convergence is the named next rung. (4) the
ceiling is a linear supervised upper bound (the spiking opponent's mild nonlinearity was measured NOT to help by the
prior boundaries); a low ceiling is decisive, a high ceiling is an upper bound the real readout must still be built
to reach. (5) the 164-word closed partition is inherited from the prior boundaries. Biological grounding: Namburi &
Tye (2015, Nature) -- opponent valence populations bound to a real US, not lexical company; the board #49/#84
interoceptive-relay pattern; the EMERGE-34 / vision->concept convergence template.

## Reproduce
```
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_grounded_experience_stream_hebbian_derisk --smoke
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_grounded_experience_stream_hebbian_derisk \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_affect_grounded_experience_stream_hebbian_6seed.json
```
