# Q3 — Engram-prior'd laminar PREDICTIVE-CODING-INFERENCE generative composition (the durable-sound PC inference; learning = validated engram one-shot bind, NOT the PC training loop) (design)

> Standing autonomy: documented design calls; brainstorm ->
> writing-plans -> subagent-driven-development -> pre-registered gate ->
> honest propagation EVERY outcome. No config-cranking, no overclaim,
> the no-confab moat byte-identical. CORRECTED OPERATING MODE: Q2
> FAILed honestly -> immediate NON-STOP pivot (no owner-deferral) to
> this Q3.

## Status

**OUTCOME (2026-05-18): cheap falsify-first precursor (Q3c) returned
VOID-class -> heavy build NOT warranted per the pre-registered rule ->
honest propagate + autonomous NON-STOP pivot to Q4.** Run 1 was
instrument-invalid by construction (perfect-prior `no_inference`=1.000,
honestly caught, not propagated); ONE root-caused mode-agnostic fix
(degraded engram recall, faithful to validated ~87.5% stim-recall,
frozen `_Q3_*` byte-unchanged); the one allowed re-run was VOID-class
(V1 0.75-0.77<0.80 unmet AND shuffled_prior 0.71-0.86>>0.50
non-discriminating -- observation-dominance). The durable-sound PC
inference still beat prior-readout by +0.20-0.26 scale-positively
(unaffected, NOT refuted). 7th-direction triangulation. See
`research/findings/2026-05-18-Q3-laminar-PC-inference-cheap-precursor-VOID.md`
+ pillar n=76. No heavy build; no writing-plans; pivot Q4.

---

Pivot-queue Q3 (Q1 VOID, Q2 FAIL both propagated). Genuinely distinct
from every prior arm (argued below). Own pre-registered THREE-STATE +
scale ladder + honest ceiling.

## Goal (one sentence)

Test whether the **durable-SOUND predictive-coding INFERENCE** (the PC
arc's empirically-held positive: the PC per-step latent-relaxation
update tracks the backprop-gradient direction at cos ~0.995, 5 seeds --
Whittington-Bogacz; ONLY the PC training-LOOP accumulation VOIDed),
used as a laminar generative inference engine with its top-down apical
prior set by the **independently-validated Tonegawa engram one-shot
bind** (catalog D.14, reward-FREE, multi-seed-validated) -- and NO PC
training-loop and NO new local learning rule -- produces small-scale
GENERATIVE compositional output that is **scale-confident** (holds /
improves across a pre-registered local scale ladder with no
architectural ceiling).

## Why this is genuinely distinct (not a re-run of any prior arm)

- **vs the PC-learning VOID:** that VOID was specifically the PC
  *training-loop accumulation* failing to turn the cos~0.995 per-step
  signal into learning. Q3 does NOT use the PC training loop at all --
  it uses ONLY the part that empirically HELD (the sound inference
  relaxation) and gets "learning" for free from the validated engram
  one-shot bind. Different use of PC: inference-only, not the failed
  accumulation.
- **vs the dendritic / feedback-alignment BOUNDARY and the
  5x-triangulated "new local learning RULE" dead-end:** Q3 introduces
  NO new local learning rule and does not train weights by a local
  rule at all. The "learning" is the already-validated reward-free
  engram bind (a stored ensemble, not a trained weight rule). So Q3
  does not re-enter the local-rule trap.
- **vs Q1 (engram bootstrap -> temporal-credit refiner, in the spiking
  bridge):** Q1 fed an engram bootstrap into a temporal-credit
  *learner* inside `sim.bridge`; it VOIDed on the spiking
  n_rewarded=0 bootstrap. Q3 has NO temporal-credit, NO reward, NO
  spiking-bridge readout/teacher/reward loop -- it is a minimal
  laminar PC-inference relaxation with engram-set top-down priors. The
  Q1 blocker (spiking rewarded-episode bootstrap) is structurally
  absent here.
- **vs Q2 (constrained decoding of a trained Transformer):** Q2 wrapped
  a trained Transformer with a token veto. Q3 has no Transformer, no
  token decoding -- it is the laminar PC-inference substrate itself
  generating by settling.

## Cheap context grounding (done, recorded)

- PC durable positive (`research/findings/raw/pc_probe_recorded.txt`):
  V1 cos(PC update, -backprop grad) = 0.995/0.994/0.997/0.995/0.995
  (>=0.95 all 5 seeds) in the linear-Gaussian Rao-Ballard /
  Whittington-Bogacz setting; the VOID was pc_acc 0.083-0.400 ~=
  chance(0.25) ~= no_inference_acc -- the per-step direction is sound,
  the training-LOOP accumulation is the sole failure.
- `sim/dendritic_plasticity.py` ships the validated, protected
  `urbanczik_senn_update(pre_rate, soma_rate, v_basal, apical_gate,
  apical_signal=None, lr=1.0)` -- a LOCAL apical-gated two-compartment
  (soma/basal + apical) mismatch rule with FIXED-RANDOM apical
  feedback (NO weight transport). Available as a laminar substrate
  primitive if a learning variant is ever needed -- but Q3's primary
  arm deliberately uses NO weight rule (engram bind is the learning).
- Tonegawa engram API on `sim.bridge.SimulationBridge` (verified
  earlier): `start_engram_recording` / `commit_engram_tag` /
  `stimulate_tag` / `clear_tag_drive` -- reward-FREE one-shot bind,
  multi-seed-validated (87.5% stim-recall, 90% multitag).

## Architectures considered (2-3), with honest ceilings + risks

- **Q3a -- Engram-prior'd laminar PC-inference compositional generator
  (RECOMMENDED).** A minimal 2-level laminar predictive hierarchy
  (top latent z, bottom observable x; top-down generative weights W
  FIXED random, the exact linear-Gaussian setting whose per-step PC
  equivalence held). The top-down APICAL prior on z is set by a
  validated engram one-shot bind per concept (the engram tag IS the
  stored top-down code -- the reward-free validated "learning";
  NO PC-loop weight accumulation, NO new local rule). GENERATION =
  clamp the engram prior(s) on the apical/top level, run the
  durable-sound PC inference relaxation (gradient descent on prediction
  error -- the cos~0.995 mechanism) until the bottom x settles; read x
  as the generated output. COMPOSITION test = clamp engram-prior(A) +
  engram-prior(B) jointly, settle, and test whether the settled x is
  the genuine composed A&B pattern that a single-concept prior
  provably cannot produce. Honest ceiling: small-scale generative
  *composition by sound predictive inference over validated engram
  priors* -- NOT fluent language, NOT an LLM, NOT learned weights.
  Risk: the engram priors may not compose under linear PC settle
  (could collapse to one attractor) -- exactly what the cheap
  falsify-first probe tests BEFORE any heavy build. Maximally DRY,
  cheaply de-riskable in the EXACT setting whose soundness held.
- **Q3b -- laminar PC-inference + Urbanczik-Senn target-prop learning.**
  Reuse `urbanczik_senn_update` (validated, byte-UNMODIFIED) to learn
  the laminar weights, PC inference for the forward settle. Honest
  ceiling/risk: this IS a local-rule weight learner -> re-enters the
  5x-triangulated local-rule dead-end + the dendritic readout-confound
  BOUNDARY. NOT genuinely distinct enough; LOWER-recommended (kept as
  a documented alternative only).
- **Q3c -- pure inference-composition, zero learning (the cheap
  precursor of Q3a).** Strip Q3a to: validated engram one-shot priors
  + sound PC inference settle, NO learning of any kind, test only
  "does sound PC inference compose engram priors generatively". This
  IS essentially Q3a's falsify-first probe; folded in as the cheap
  precursor rather than a separate heavy arm.

**Recommendation: Q3a**, with **Q3c as its mandatory cheap
falsify-first precursor** (so Q3 -- unlike Q1 -- has a genuine cheap
de-risking gate IN THE EXACT linear-Gaussian PC setting whose
per-step soundness empirically held).

## Falsify-cheaply precursor (Q3c) -- MANDATORY, runs FIRST

Throwaway pure-numpy probe (prefix `_`, deleted post-decision,
recorded evidence to `research/findings/raw/`), NO autograd, the EXACT
linear-Gaussian 2-level Rao-Ballard/Whittington-Bogacz setting whose
per-step update == backprop direction held (cos~0.995). Engram priors
modeled as fixed sparse top-level code vectors per concept (the
validated one-shot bind = a stored top code). Pre-registered cheap
THREE-STATE (reuse the established discipline; own frozen `_Q3_*`,
NEVER tuned):
- V1 (instrument soundness): single-concept engram-prior + sound PC
  settle reconstructs that concept's bottom pattern at high fidelity
  (>= a frozen bar) AND a NO-inference control (clamp prior, NO PC
  relaxation) does NOT -- proves the sound PC inference is doing the
  generative work and the instrument can see it.
- Science: a TWO-concept joint engram-prior + sound PC settle yields
  the genuine COMPOSED pattern (>= a frozen compositional-fidelity
  bar; e.g. recovers both concepts' factored components above a
  decisive margin over the best single-concept attractor).
- Controls (must fail): `no_inference` (no PC settle), `shuffled_prior`
  (engram prior permuted off the concept), `wrongsign` (PC error sign
  flipped -- anti-settles).
- Pre-registered SCALE LADDER even in the cheap probe: C in {2,4,8}
  composed-concept count; SCALE-CONFIDENT-cheap iff every rung PASS AND
  compositional fidelity non-decreasing up to a frozen tol AND holds at
  the largest rung. GREEN -> heavy Q3a build green-lit; NEGATIVE ->
  honest propagate + autonomous pivot to Q4 (NO heavy build, NO
  config-crank). If the cheap probe shows sound PC inference cannot
  compose engram priors even in principle, that is an honest cheap
  NEGATIVE and the most decision-relevant possible outcome.

## Architecture (heavy Q3a; maximally DRY; net-new vs reused-UNMODIFIED)

**Reused UNMODIFIED (protected; byte-empty in every commit-scoped diff
AND `git diff <plan-base>..HEAD`):** the Tonegawa engram API on
`sim.bridge` (one-shot bind = the learning), `sim/dendritic_plasticity.py`
(available laminar primitive), `research/runners/text_minimal_isolation.py`
`build_biological_brain_regions` (if an in-substrate variant is built),
`sim/train_checkpoint.py` (kill-safe), the no-confab moat
(`abstention_gate` + test, 7/7), every frozen `*_core` (incl.
`compose_bridge_core`/`compose_bind_core`/`td_critic_core`/
`dendritic_fair_core`/`constrained_decode_core`/`generator_g_core`),
`sim/kernels.py`, `sim/backend.py`. NO new autograd anywhere (the
backprop reference, if any, is hand-derived numpy validity-only, as in
the PC probe).

**Net-new (load-bearing):**
1. `research/runners/laminar_pc_core.py` -- pure FIXED-bar
   THREE-STATE + scale-confidence verdict, own frozen `_LPC_*`
   (mirrors the adversarial-hardened `compose_bridge_core` /
   `constrained_decode_core` discipline EXACTLY; instrument-validity
   FIRST, fail-closed, VOID strictly distinct from FAIL, malformed/junk
   -> VOID-not-raise; does NOT import/mutate any existing `*_core`;
   imports only stdlib+typing). Holds `lpc_verdict` + a pure
   `lpc_scale_confidence` mirroring the established
   `scale_confidence`/`cdc_scale_confidence` pattern.
2. `research/runners/laminar_pc_gate.py` -- kill-safe runner: the
   minimal 2-level laminar PC-inference generative composer (the EXACT
   linear-Gaussian sound-inference relaxation), engram one-shot priors
   as the validated learning, conditions {compose, no_inference,
   shuffled_prior, wrongsign} x seeds x the pre-registered scale
   ladder; greedy noise-free compositional-fidelity readout; per-(rung,
   seed) atomic checkpoint via REUSED `sim.train_checkpoint`;
   KeyboardInterrupt-clean-exit; `--tiny` smoke whose toy verdict is
   NOT propagated; ASCII; honest-ceiling banner. NO autograd.

## Pre-registered in-sim THREE-STATE + SCALE LADDER (frozen, NEVER tuned)

- V1 (instrument soundness): single-concept engram-prior + sound PC
  settle reconstructs at >= `_LPC_V1_MIN`; the `no_inference` control
  is far below it (proves the sound PC inference, not the prior
  read-out, generates).
- Science: two-(or-more)-concept joint engram-prior + sound PC settle
  yields composed fidelity >= `_LPC_SCI_MIN` with a decisive margin
  over the best single-concept attractor.
- Controls (must fail <= `_LPC_CTRL_MAX`): `no_inference`,
  `shuffled_prior`, `wrongsign`.
- SCALE LADDER `C in {2,4,8}` composed-concept count (frozen rule
  pinned in the implementation plan; `_LPC_SCALE_TOL` frozen).
  SCALE-CONFIDENT iff every rung PASS AND composed fidelity
  non-decreasing up to tol AND the discriminating signature holds at
  the LARGEST rung. Outcome map (honest, never spun):
  SCALE-CONFIDENT-PASS / WORKS-SMALL-NO-SCALE-CONFIDENCE / FAIL / VOID
  -- all propagated, every outcome; non-PASS -> autonomous Q4 pivot.
  Frozen `_LPC_*` pre-registered in the implementation plan with
  explicit justification BEFORE any run; a sound instrument whose V1
  is unmet is an honest VOID, NOT a reason to soften a bar.

## Honest ceiling (stated up front, NEVER spun)

- **IS (only if SCALE-CONFIDENT-PASS):** the durable-sound PC
  inference, given validated reward-free engram priors and NO
  PC-training-loop / NO new local rule, generatively COMPOSES concepts
  at small capacity AND the capability holds/improves across the local
  scale ladder with no architectural ceiling -- the owner's
  scale-confidence deliverable, by inference-composition over validated
  priors.
- **IS NOT (never spun):** open-ended fluent composition, an LLM,
  GPT-class, conversation-solved, learned-weight generalization. The
  "learning" is a stored engram bind; the generation is a linear-
  Gaussian PC settle. It is a minimal mechanism-level scale-confidence
  PoC, NOT scaled language. A FAIL/VOID is the honest triangulation
  (the durable-sound PC inference does/does not compose validated
  priors), propagated never spun, and triggers the autonomous Q4
  pivot.

## Anti-cheat plan (non-negotiable)

Cheap Q3c falsify-first FIRST (honest GREEN->heavy / NEGATIVE->pivot
Q4, recorded); pre-registered FIXED-bar THREE-STATE + scale ladder in
`laminar_pc_core` (own frozen `_LPC_*` NEVER tuned; does NOT
import/mutate any existing core; no new GLOBAL bar); dedicated
ADVERSARIAL REVIEWER on BOTH net-new modules BEFORE Phase B (probe:
is the discriminating signal genuinely the sound PC INFERENCE composing
priors and not a prior-readout artifact or a degenerate attractor; is
`no_inference` a faithful control identical minus exactly the PC
settle; is the engram one-shot bind genuinely the validated reward-free
mechanism not a re-derived strawman; can a V1-broken / non-
discriminating / vacuity run be scored PASS/SCALE-CONFIDENT; are the
frozen `_LPC_*` movable by results; ANY autograd); controller
trust-but-verify EVERY diff with the PROTECTED set byte-empty;
mandatory smell-test scrutinizing a nominal PASS HARDER than a FAIL
(recompute from the single recorded JSON; V1 genuine + non-degenerate;
controls fail; scale-confidence recomputed; NO re-run, NO bar-tuning,
NO overclaim); honest propagation EVERY outcome (findings +
capability_status pillar n=76 + schema green + push BOTH remotes);
on non-SCALE-CONFIDENT-PASS the autonomous Q4 pivot (NO stop, NO
owner-deferral); the no-confab moat byte-identical + 7/7 throughout.
MONITORING DISCIPLINE: any decisive run uses the Bash run_in_background
parameter (auto-notifies on completion) or runs foreground -- NEVER a
bare nohup with a false "I will be notified" claim; completion is
actively confirmed before any result is claimed.
