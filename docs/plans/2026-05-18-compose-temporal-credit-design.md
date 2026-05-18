# Temporal-credit-bridged compositional binding — does the validated temporal-credit mechanism defeat the composition BOUNDARY? (design)

> Standing autonomy: documented design calls (NOT one-question-at-a-time);
> brainstorm -> writing-plans -> subagent-driven-development ->
> pre-registered in-sim THREE-STATE gate -> honest propagation every
> outcome. No config-cranking, no overclaim, the no-confab moat
> byte-identical.

## Goal (one sentence)

Test, in the actual sim plasticity/eligibility substrate, whether
adding the now-VALIDATED temporal-credit / eligibility mechanism lets a
compositional A->B binding bridge the temporal gap that pure
STDP-cold-start (the v16 composition BOUNDARY) structurally could not.

## Why this, given the accumulated results (strategic deliberation)

- The dendritic/feedback-alignment **spatial** credit-assignment lever
  is a hard multiply-confirmed BOUNDARY (readout-over-features confound;
  no soundly-discriminating instrument at feasible scale).
- The TD value-function critic is a genuine, anti-cheat-validated
  **PASS** -- the first clean positive of the arc. It succeeded
  precisely because it had a principled positive control + a
  discriminating signature the readout cannot rescue.
- The project's actual #1 stated blocker is **composition**. Every
  prior composition attempt failed at *spatial/pathway* credit
  (STDP cold-start never reached functional magnitude; the v16
  verb_pool->motor pathway was essentially silent) and **none had a
  temporal-credit mechanism**. Composition is fundamentally
  *temporal/sequential* (A earlier, the bound outcome later) -- exactly
  the gap the project never had a mechanism to bridge until the TD
  PASS.

So the highest-value, evidence-driven, NON-config-cranking next step is
a NEW, separately-gated arc applying the validated temporal-credit
substrate to the composition boundary -- NOT escalating the TD
increment (its PASS is its terminus), NOT re-treading the boundaried
spatial lever, NOT gambling on predictive coding re-hitting the same
readout confound.

## Cheap falsify-first gate — already GREEN, scrutinized HARDER than a FAIL (evidence)

Throwaway probe `_probe_compose_tdcredit.py` (deleted post-decision;
pure numpy, NO autograd): a minimal faithful proxy -- learn a fixed
compositional bijection A_i -> B_{pi(i)} where the reward arrives a
TEMPORAL GAP after the A-time decision (reward depends on the ordered
temporal pairing, NOT a static co-occurrence vector -> a static readout
structurally cannot shortcut it; this is exactly why the lever is
discriminable, unlike the spatial-rule case).

- **V1 (principled, analytically checkable):** the TD learner on the
  NO-GAP version reaches the analytic optimum (1.0/1.0/0.917/1.0/1.0,
  5 seeds; bar >= 0.90) -- the harness itself is sound, separate from
  the science.
- **Science:** the temporal-credit/eligibility learner on the GAPPED
  task reaches **0.917-1.000** (5 seeds; bar >= 0.90) -- it learns the
  12-way compositional binding bridging a 6-step gap.
- **The discriminating control is mechanistically ISOLATED:**
  `hebbian_no_trace` is identical to `td` in every respect EXCEPT the
  eligibility trace is zeroed each gap step (the faithful v16-analog --
  no temporal-credit to carry the A-time decision across the gap). It
  fails **at exactly chance 0.083, deterministically, all 5 seeds**.
  The ONLY causal difference is temporal-credit bridging the gap -> not
  a readout artifact (the readout is identical in both). `permuted`
  (re-randomized rule) and `wrongsign` also fail.
- **Decisive separation:** td 0.92-1.0 vs the v16-analog 0.083
  (deterministic) -- a 6-12x gap, 5 seeds.
- **Transparent STRENGTHEN-only correction (NOT bar-tuning; mirrors the
  owner-endorsed TD-critic measurement-correction precedent):** the
  first run used N=6, where the `permuted` accuracy metric
  (argmax over 6 cells vs a fixed permutation) is a high-variance
  DISCRETE Binomial(6,1/6) that produced a single-seed 3/6 by pure
  combinatorics, tripping the absolute control bar (NOT "permuted
  learned" -- it structurally cannot). Root-caused BEFORE re-run; fixed
  by N 6->12 (a strictly HARDER task: chance 0.167->0.083; control
  chance-distribution provably tight, P(permuted>0.35)~3e-4) + a
  mode-AGNOSTIC budget (8000, applied identically to all modes --
  hebbian_no_trace stays deterministically 1/N regardless, proving the
  budget cannot advantage td). Frozen pass/fail bars byte-UNCHANGED;
  pre-stated prediction confirmed.

This is the FIRST mechanistic evidence in the entire composition arc
identifying *what was missing* (temporal-credit to bridge the
bind-gap), converging with the TD PASS. Green-light a separately-gated
in-sim build.

## Architecture (maximally DRY; net-new vs reused-UNMODIFIED)

**Net-new (load-bearing):**
1. `sim/compose_temporal_bind.py` -- a minimal in-sim compositional-
   binding learner that contrasts a temporal-credit/eligibility-bridged
   update vs a no-trace update on a gapped A->B binding *in the sim's
   real eligibility substrate*. It REUSES the validated TD(lambda)
   logic from `sim/td_value_critic.py` and the eligibility kernel; the
   net-new is the compositional-binding task harness + the
   no-trace/permuted/wrongsign control modes wired to the sim
   substrate. NO autograd.
2. `research/runners/compose_bind_core.py` -- pure FIXED-bar
   THREE-STATE verdict with its OWN frozen `_CTB_*` constants
   (composition-calibrated; does NOT import/mutate `td_critic_core` /
   `dendritic_fair_core` / any existing core). Instrument-validity
   FIRST, fail-closed; strict-bool; numeric-coercion -> VOID-not-raise;
   a diverged/non-finite control = correctly-failed; VOID strictly
   distinct from FAIL; the `permuted`/control acceptance uses the
   N>=12 design so the chance distribution is provably tight (no
   absolute-bar-on-small-N artifact). Mirrors the hardened
   `td_critic_core` discipline exactly.
3. `research/runners/compose_bind_gate.py` -- kill-safe THREE-STATE
   runner; reuses `sim.train_checkpoint` + (where it composes) the
   REUSED `NeuromodulatorManager` UNMODIFIED so the temporal-credit
   delta is the phasic-DA signal. Conditions {td, hebbian_no_trace,
   permuted, wrongsign} x seeds; per-(seed) kill-safe checkpoint;
   THREE-STATE verdict; honest-ceiling banner.

**Reused UNMODIFIED (DRY; byte-empty in every commit-scoped diff):**
`sim/td_value_critic.py` (this session's VALIDATED TD(lambda) +
eligibility), `sim/kernels.py` `fused_eligibility_trace_decay`,
`sim/neuromodulators.py`, `sim/train_checkpoint.py`, `sim/backend.py`,
the no-confab moat (`abstention_gate.py` + its test, MUST stay 7/7),
every frozen `*_core` (incl. `td_critic_core`, `dendritic_fair_core`),
`sim/dendritic_plasticity.py`. NO new GLOBAL bar.

## Pre-registered FROZEN bars (in compose_bind_core.py; justified, NEVER tuned)

Chance = 1/N with N>=12 (so the control chance-distribution is
provably tight; the N=6 small-sample artifact is structurally excluded
by design):
- `_CTB_V1_ACC_MIN = 0.90` (V1: the in-sim TD harness learns the
  no-gap bijection; analytic optimum 1.0)
- `_CTB_SCIENCE_ACC_MIN = 0.90` (science: the gapped temporal-credit
  binding is learned)
- `_CTB_CONTROL_ACC_MAX = 0.35` (every control ~chance; for N>=12,
  P(control>0.35) ~ 3e-4 -- a genuine fail is unambiguous)
- `_CTB_MIN_SEEDS = 3`
THREE-STATE: instrument_valid = (V1 met all seeds) AND (every control
<= CONTROL_ACC_MAX all seeds; non-finite = correctly-failed). VOID if
not instrument_valid (V1 unmet = harness unsound; control passed =
non-discriminating). If valid: PASS iff gapped-td >= SCIENCE_ACC_MIN
all seeds; else FAIL (sound+discriminating instrument yet temporal
credit ALSO cannot compose in-sim -> strongest honest triangulation
that temporal credit is NOT the missing in-sim ingredient).

## Build sequence (subagent-driven; anti-cheat)

- **Task 0** grounding pin (end-to-end gate turns + THREE-STATE on a
  tiny config; green only after the runner task -- the gate).
- **Phase A pure-CPU-TDD**, fresh subagent per task, controller
  trust-but-verify each commit-scoped diff (protected byte-empty):
  **Task 1** `sim/compose_temporal_bind.py` (LOAD-BEARING: reuses
  `sim/td_value_critic.py` + the eligibility kernel UNMODIFIED; the
  no-trace/permuted/wrongsign modes; NO autograd; the validated
  cheap-probe contrast, now in the sim substrate); **Task 2**
  `research/runners/compose_bind_core.py` (FIXED-bar THREE-STATE, own
  frozen `_CTB_*`, hardened-`td_critic_core` discipline). BOTH ->
  **DEDICATED ADVERSARIAL REVIEWER before Phase B** (probe: fabricated
  PASS from unsound/non-discriminating; diverged-control mis-score;
  movable bars; autograd; is the discrimination genuinely isolated to
  the temporal-credit mechanism or a readout/harness artifact; is the
  no-trace control a FAITHFUL v16-analog or a strawman; byte-faithful
  reuse of the validated TD infra).
- **Phase B**: **Task 3** `compose_bind_gate.py` (import/signature
  smoke + <3-seeds->exit2 + tiny pipeline-turns + reuses
  train_checkpoint/NM/td_value_critic UNMODIFIED + no autograd) makes
  Task 0 green; **Task 4** LOAD-BEARING no-harm (protected byte-
  untouched across the whole range; moat 7/7; representative suite;
  no autograd in shipped path).
- **Task 5 (controller-only):** grounding-first tiny run (toy verdict
  NOT propagated) -> decisive kill-safe multi-seed in-sim run (>=5
  seeds; FIXED pre-registered config) -> MANDATORY anti-cheat
  smell-test scrutinizing a nominal PASS HARDER than a FAIL (recompute
  from JSON; V1 genuine+non-degenerate; the discrimination genuinely
  isolated to temporal-credit; the no-trace control genuinely the
  faithful v16-analog and genuinely failing; decisive separation; NO
  re-run/bar-tuning/overclaim) -> honest propagation EVERY outcome
  (findings doc + capability_status pillar + schema green + push BOTH
  remotes).

## Honest ceiling (stated up front, NEVER spun)

- **IS:** a *mechanism-level, in-sim* validation -- temporal-credit /
  eligibility can learn a compositional binding bridging a temporal gap
  that the faithful no-trace v16-analog structurally cannot, in the
  sim's real eligibility substrate, anti-cheat-gated.
- **IS NOT:** composition-solved. NOT compositional *language*. NOT
  scaled vocab. NOT integrated into the concept-pool / lang_input /
  chat stack. NOT a claim that the full v16 architecture now composes.
  The cheap + in-sim gates establish the missing *ingredient*; wiring
  it into the full conversational composition architecture at scale is
  a SEPARATE later gated increment (YAGNI here).
- A PASS is the honest terminus of THIS increment; a FAIL/VOID is the
  strongest honest triangulation of why composition is hard, NOT a
  license to escalate. PASS/BOUNDARY/VOID all decision-relevant +
  propagated honestly, no overclaim.

## Explicitly NOT in scope (YAGNI / honesty)

Full conversational composition; scaling to the real concept-pool
vocab; chat integration; predictive coding / laminar microcircuit.
An honest in-sim PASS/FAIL/VOID here is the terminus of THIS increment.
