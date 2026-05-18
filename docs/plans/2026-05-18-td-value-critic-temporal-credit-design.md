# TD value-function critic — biologically-canonical temporal credit assignment (design)

> Standing autonomy: documented design calls (NOT one-question-at-a-time);
> brainstorm -> writing-plans -> subagent-driven-development ->
> pre-registered THREE-STATE gate -> honest propagation every outcome.
> No config-cranking, no overclaim, the no-confab moat byte-identical.

## Goal (one sentence)

Give the simulator the single most-cited biologically-canonical
mechanism it lacks — a learned **value-function critic** doing
**temporal-difference credit assignment** (phasic DA = TD error,
Schultz) — so the sim has genuine temporal credit assignment (the
actual recurring root blocker) for the first time.

## Why this (catalog-grounded deliberation; ceilings up front)

The dendritic/feedback-alignment **spatial** credit-assignment lever is
triangulated to a hard multiply-confirmed BOUNDARY (a sound
discriminating instrument is not constructible at feasible scale; the
heavy CIFAR+conv run is the owner's eyes-open call). That does NOT
exhaust biology. Consulting the full 17-cluster
`references/feature-catalog.md`, three genuine candidates:

- **A — TD value-function critic / actor-critic (catalog C.22 /
  C.28-C.34 / O.02).** The catalog states it outright: the project
  reproduces only the *sign* of reward; it *lacks the cue-shift
  transfer because it lacks the value-function critic of an actor-critic
  architecture*. Phasic DA = TD error is the single most-validated
  computational-neuroscience result (Schultz). Distinct from the
  boundary-confirmed spatial family. **Principled, cheap,
  anti-cheat-able V1** (value provably -> true expected return; the
  Schultz cue-shift transfer is a binary classic signature) — the
  cleanly-constructible positive control the dendritic arc never had.
- **B — cortical predictive coding (Rao-Ballard/Friston).** Absent from
  the catalog; biologically central; backprop-approximating. Ceiling:
  large architectural build, high risk of the *same* readout-confound
  non-discriminability the FA arc hit; not a precisely-scoped
  catalogued gap.
- **C — canonical laminar microcircuit.** Catalog: "no module wires the
  canonical microcircuit." A substrate, not a learning principle —
  more reservoir without one; best as a later pairing.

**Recommendation: A.** It is the catalog's own #1 cited unimplemented
canonical mechanism, attacks the *actual* recurring blocker (temporal
credit assignment — permuted-label NEGATIVE, W->A global-scalar
failure, composition BOUNDARY all share this), has a principled cheap
anti-cheat-able V1, and composes with validated infra (DRY).

**Honest ceiling, no spin:** a TD critic solves *temporal* credit
assignment for *value* prediction. A scrutinized PASS = "the sim
acquires a learned predictive value function with the canonical Schultz
cue-shift transfer at feasible local scale — temporal credit assignment
it has never had." It is **NOT** conversation-solved, **NOT** grammar,
**NOT** AGI. It is a plausible *substrate* for sequential/compositional
learning; integration into the conversational stack is a **separate
later effort**, only noted. PASS/FAIL/VOID all decision-relevant and
propagated honestly.

## Cheap falsify-first gate — already GREEN (evidence)

A throwaway pure-numpy probe (`_probe_td_critic.py`, deleted
post-decision) ran the load-bearing cheap gate BEFORE any build:

- **V1 sound, robust:** TD(lambda) on the complete-serial-compound
  representation converges to the analytic exact V* at
  Vrmse 0.0012-0.0048 (3 seeds; <0.5% of non-trivial values 0.81-1.0).
  Same setup minus the bootstrap (`no_bootstrap`) diverges (Vrmse ~180)
  — proving V1 is specifically correct TD, not a trivial representation.
- **Schultz cue-shift transfer genuinely emerges:** scale-free
  transfer-fraction 0.997 (3 seeds), US-RPE decays to ~0.001 (reward
  becomes fully predicted).
- **Controls genuinely fail for mechanistically-correct reasons:**
  `no_bootstrap` 0.20 (the catalog's exact claim — no bootstrap, no
  cue power), `permuted` 0.06-0.08 (reward stays unpredicted),
  `wrongsign` diverges. PASS vs best control = 0.997 vs 0.21 (4.7x —
  not a knife-edge).
- **Measurement correction, transparently logged (NOT bar-tuning;
  STRENGTHEN-only; mirrors the endorsed dendritic gradcheck sum/mean
  HARNESS-artifact precedent):** the first transfer spec normalized by
  the early untrained dUS and set the bar 0.50; a *provably-perfect*
  critic (Vrmse 0.001, seed-invariant) deterministically yields 0.465
  on that spec -> the bar/measure was analytically mis-derived (ignored
  discounting + the irreducible correct pre-cue baseline that exists
  ONLY because of the required cue-onset-uncertainty design), i.e. a
  perfect instrument cannot meet it -> mis-spec, not science FAIL.
  Corrected to the canonical scale-free Schultz fraction
  |dCS|/(|dCS|+|dUS|) which -> 1.0 for any perfect critic by
  mathematical identity; bar 0.90 (>= original intent).

This is the FIRST lever in the whole arc with a cheaply-constructible
**sound AND discriminating** instrument where the science signature
genuinely emerges — distinct from the dendritic boundary. Green-light.

## Architecture (maximally DRY; net-new vs reused-UNMODIFIED)

**Net-new (load-bearing):**
1. `sim/td_value_critic.py` — a value-function critic. Linear value
   `V(s) = w . phi(s)` over a state representation, TD(lambda) update
   `delta = r + gamma*V(s') - V(s)`; `w += alpha*delta*e`, eligibility
   `e` via the REUSED `fused_eligibility_trace_decay`. Pure array math
   via `sim.backend` (CuPy/NumPy). **NO automatic differentiation**
   (TD needs none). Faithful mapping: `delta` IS the phasic-DA TD error
   that replaces the bare reward scalar; the critic is the missing
   "value-function critic of an actor-critic" (catalog C.30).
2. `research/runners/td_critic_core.py` — pure FIXED-bar THREE-STATE
   verdict with its OWN frozen constants (value-prediction-calibrated;
   does NOT import/mutate dendritic_fair_core / any existing core):
   - `_TDC_V1_VALUE_RMSE_MAX = 0.05` (V1: critic provably learns the
     true expected return; perfect ~0.001-0.005, no-learn ~0.3-0.8)
   - `_TDC_TRANSFER_MIN = 0.90` (canonical scale-free Schultz
     transfer-fraction; perfect ~1.0, controls <=0.21)
   - `_TDC_US_DECAY_MAX = 0.15` (reward becomes predicted; perfect
     ~0.001, controls ~0.96/diverge)
   - `_TDC_MIN_SEEDS = 3`
   - `tdc_verdict(...)` instrument-validity FIRST, fail-closed: VOID if
     NOT (finite & V1 met & every control genuinely FAILS the signature
     [diverged/non-finite => correctly failed]); only if valid: PASS iff
     transfer>=MIN & us_decay<=MAX all seeds, else FAIL. Strict `is
     True` bools; numeric-coercion -> VOID not raise; VOID strictly
     distinct from FAIL (mirror dendritic_fair_core hardened
     discipline).
3. `research/runners/td_critic_gate.py` — kill-safe runner. The
   BIOLOGICALLY-FAITHFUL sim integration: a value-readout region whose
   activity is `V(s)`; `delta` routed through the EXISTING
   neuromodulator subsystem as the phasic-DA learning signal (the
   catalog's prescribed upgrade — `current_reward_signal` becomes a
   true TD error, not a bare scalar); eligibility via
   `bridge.cp_eligibility_trace`; actor = the existing BG cascade. An
   in-sim Pavlovian schedule with jittered cue onset (canonical Schultz
   design). Per-(seed,condition) kill-safe checkpoint via REUSED
   `sim.train_checkpoint`; KeyboardInterrupt-clean-exit. Conditions:
   {td, no_bootstrap, permuted, wrongsign}. Writes JSON + ASCII verdict
   + honest-ceiling banner. The scale-free bars transfer unchanged
   (they are scale-free by construction).

**Reused UNMODIFIED (DRY; byte-empty in every commit-scoped diff):**
`sim/neuromodulators.py` (NM subsystem + `from_reward` rule +
`plasticity_rate` target), `sim/kernels.py`
`fused_eligibility_trace_decay` + `bridge.cp_eligibility_trace`,
`research/runners/g11_bg_runner.py` `build_bg_brain_regions` (actor),
`sim/train_checkpoint.py`, `sim/backend.py`, the no-confab moat
(`research/runners/abstention_gate.py` + `tests/test_abstention_gate.py`
— 7/7 green throughout), every frozen `*_core` (each owns its bars),
`sim/dendritic_plasticity.py`, `sim/bptt_snn*`, `sim/bridge.py`. NO new
GLOBAL bar.

## Build sequence (subagent-driven; anti-cheat)

- **Task 0** falsify-cheaply grounding pin: the cheap-gate principle is
  already validated (above); a tiny-synthetic end-to-end pin makes the
  gate runner turn and produce an interpretable VOID/PASS/FAIL.
- **Phase A pure-CPU-TDD**, fresh subagent per task, failing-test ->
  minimal-impl -> run -> commit, controller trust-but-verify each diff
  (protected modules byte-empty): **Task 1** `sim/td_value_critic.py`
  (LOAD-BEARING: TD(lambda) faithful; NO autograd in the shipped path;
  reuses `fused_eligibility_trace_decay` UNMODIFIED; the scale-free
  transfer + V1 measures exactly as the validated cheap probe);
  **Task 2** `research/runners/td_critic_core.py` (FIXED-bar THREE-STATE
  VOID/PASS/FAIL, instrument-validity-first fail-closed, strict-bool,
  numeric-coercion-VOID, control-divergence-correct, adversarial
  matrix). BOTH get a **DEDICATED ADVERSARIAL REVIEWER** before Phase B
  (probe: can a non-discriminating or V1-broken run be scored PASS
  instead of VOID? can a diverged control be mis-scored
  non-discriminating? any autograd in the shipped path? frozen bars
  movable by results?).
- **Phase B**: **Task 3** `td_critic_gate.py` (import/signature smoke +
  <3-seeds->exit2 + tiny-synthetic pipeline-turns + verify reuses NM
  subsystem + eligibility + train_checkpoint UNMODIFIED + no autograd);
  **Task 4** LOAD-BEARING no-harm (moat + its test + all frozen cores +
  reused modules byte-UNTOUCHED across the whole commit range;
  representative validated suite + `test_abstention_gate.py` green;
  assert NO shipped path imports torch.autograd/backward).
- **Task 5 (controller-only):** grounding-first tiny synthetic run
  (toy verdict NOT propagated) -> the decisive kill-safe multi-seed
  in-sim run (seeds 42,43,44; FIXED pre-registered config) ->
  MANDATORY anti-cheat smell-test scrutinizing a nominal PASS HARDER
  than a FAIL (recompute from recorded JSON; V1 genuinely met;
  transfer/US-decay genuine; controls genuinely fail for the
  mechanistically-correct reasons; NO re-run/NO bar-tuning/NO
  overclaim) -> honest propagation EVERY outcome (findings doc +
  capability_status pillar + schema green + push BOTH remotes).

## Anti-cheat & discipline

THREE-STATE + V1/control-validity-FIRST refuses fabricated verdicts (a
non-discriminating or V1-broken in-sim run = VOID, never PASS/FAIL).
Frozen `_TDC_*` pre-registered here, never tuned. Dedicated adversarial
reviewer for the two load-bearing modules before Phase B. Controller
trust-but-verify every subagent diff (protected byte-empty).
Controller-only decisive run + mandatory smell-test (PASS scrutinized
harder). Honest ceiling stated up front and never spun (temporal credit
assignment substrate ONLY; integration a separate later effort). The
validated no-confab moat byte-identical + 7/7 green throughout. NOT
config-cranked (cheap gate green, owner-redirect-authorized).

## Explicitly NOT in scope (YAGNI / honesty)

Integration of the critic into the conversational/composition stack;
actor-critic *policy* improvement experiments; predictive coding (B) /
laminar microcircuit (C). An honest in-sim PASS/FAIL/VOID here is the
terminus of THIS increment, not a license to escalate.
