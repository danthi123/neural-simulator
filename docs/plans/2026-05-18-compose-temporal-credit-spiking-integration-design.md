# Temporal-credit composition in the REAL spiking concept-pool substrate — does the validated mechanism survive the sim.bridge jump? (design)

> Standing autonomy: documented design calls; brainstorm ->
> writing-plans -> subagent-driven-development -> pre-registered in-sim
> THREE-STATE gate -> honest propagation every outcome. No
> config-cranking, no overclaim, the no-confab moat byte-identical.

## Goal (one sentence)

Test whether the validated temporal-credit/eligibility mechanism --
which (a) is a validated substrate [TD-critic PASS], (b) is the missing
composition ingredient on an abstract task [compose PASS], and (c)
survives the distributed-population + noisy-readout jump [pop-transfer
cheap-GREEN] -- still bridges the verb->motor compositional bind-gap
when wired into the REAL spiking `sim.bridge` concept-pool architecture
where the v16 BOUNDARY actually lived.

## Why this, given the converging evidence (deliberation)

Three converging, anti-cheat-maxed results, each with the honest
ceiling firmly stated:
- **TD value-function critic = VALIDATED PASS** -- temporal credit
  assignment is a sound validated substrate (catalog #1 unimplemented
  mechanism).
- **Compose x temporal-credit = VALIDATED PASS** -- on a minimal
  abstract 12x12 tabular task, temporal credit bridges a compositional
  bind-gap the faithful no-trace v16-analog structurally cannot.
- **Compose pop-transfer = cheap-GREEN, scrutinized harder than a
  FAIL** -- the mechanism TRANSFERS to distributed sparse-population
  codes + P x N matrix credit + population-outer-product eligibility +
  noisy decision readout (the v16 substrate's harder ingredients):
  gapped td = 1.0 all 5 seeds, the faithful no-trace v16-population-
  analog = exactly 1/12 all 5 seeds, frozen bars unchanged.

Every prior in-numpy result is mechanism-level. The single remaining
honest, decision-relevant, non-config-cranked question is the one the
compose-PASS ceiling explicitly named as "a SEPARATE later gated
increment": **does it transfer into the actual spiking sim.bridge
concept-pool architecture (the real v16 setting)?** A transfer-PASS
there would be the first genuine in-architecture dent in the
composition blocker; a boundary there would sharply localize the
remaining gap specifically to spiking-dynamics integration (not the
temporal-credit principle, not population coding) -- both
decision-relevant.

## Falsify-cheaply precursor — already satisfied (the pop-transfer cheap-GREEN)

The cheap falsify-first gate for THIS increment is the
population-transfer probe (throwaway, deleted): it added exactly the
substrate-hardening ingredients short of full spiking dynamics and the
mechanism transferred cleanly (PASS scrutinized harder than a FAIL;
faithful no-trace v16-analog deterministically at chance; frozen bars
unchanged). So the heavy in-sim build is green-lit -- but its OWN
pre-registered in-sim THREE-STATE gate decides the science, honestly,
every outcome.

## Architecture (maximally DRY; net-new vs reused-UNMODIFIED)

**Net-new (load-bearing):**
1. `research/runners/compose_bridge_core.py` -- pure FIXED-bar
   THREE-STATE verdict with its OWN frozen `_CBR_*` constants
   (in-spiking-substrate-calibrated; does NOT import/mutate
   `compose_bind_core` / `td_critic_core` / any existing core).
   Mirrors the adversarial-hardened discipline EXACTLY (strict numeric,
   malformed/non-numeric/junk -> VOID-not-raise, diverged/non-finite
   control = correctly-failed, VOID strictly distinct from FAIL,
   instrument-validity FIRST). Pre-registered bars justified in the
   plan; NEVER tuned.
2. `research/runners/compose_bridge_gate.py` -- kill-safe runner that
   wires the validated temporal-credit/eligibility mechanism into a
   MINIMAL slice of the real spiking concept-pool architecture: a small
   number of verb pools + motor pools via the REUSED
   `build_biological_brain_regions` (the v16 setting), driving the
   real `bridge.cp_eligibility_trace` + the real plasticity path; the
   TD delta routed through the REUSED `NeuromodulatorManager` as the
   phasic-DA signal (catalog C.30). Conditions {td, hebbian_no_trace,
   permuted, wrongsign} x seeds on the in-bridge verb->motor
   compositional bind-gap; per-(seed) kill-safe checkpoint via REUSED
   `sim.train_checkpoint`; THREE-STATE verdict; honest-ceiling banner.
   `hebbian_no_trace` MUST be identical to `td` in-bridge except the
   eligibility trace is suppressed across the gap (the faithful
   v16-cold-start analog -- NOT a strawman crippled elsewhere).

**Reused UNMODIFIED (DRY; byte-empty in every commit-scoped diff):**
`sim/td_value_critic.py`, `sim/compose_temporal_bind.py` (the
validated mechanism logic), `sim/kernels.py`
`fused_eligibility_trace_decay`, `sim/bridge.py`
`cp_eligibility_trace` + the real plasticity path,
`research/runners/text_minimal_isolation.py`
`build_biological_brain_regions`, `sim/neuromodulators.py`,
`sim/train_checkpoint.py`, `sim/backend.py`, the no-confab moat
(`abstention_gate.py` + its test, MUST stay 7/7), every frozen
`*_core` (incl. `compose_bind_core`, `td_critic_core`,
`dendritic_fair_core`), `sim/dendritic_plasticity.py`. NO new GLOBAL
bar. NO autograd anywhere in the shipped path.

## Pre-registered in-sim THREE-STATE gate (in compose_bridge_core.py; justified, NEVER tuned)

- V1 (instrument soundness): the in-bridge TD mechanism on a NO-GAP
  in-bridge binding reaches a justified high bar (proves the spiking
  harness itself learns the verb->motor bijection).
- Science: the in-bridge TD+eligibility on the GAPPED verb->motor bind
  reaches the science bar.
- Controls (must fail in-bridge): `hebbian_no_trace` (faithful
  v16-cold-start analog), `permuted`, `wrongsign`.
- THREE-STATE instrument-validity-FIRST fail-closed; VOID if V1 unmet
  or a control learns / is missing / non-numeric; PASS iff sound +
  discriminating + science met; else FAIL (sound+discriminating yet
  temporal credit ALSO fails in-bridge = the strongest honest
  triangulation that the remaining blocker is spiking-dynamics
  integration, not the temporal-credit principle). The exact frozen
  `_CBR_*` values + their spiking-substrate justification are
  pre-registered in the implementation plan (writing-plans step),
  calibrated to the spiking substrate's irreducible noise floor BEFORE
  any run, never tuned to a result; a sound true-TD no-gap learner
  that cannot meet V1 is an honest VOID, not a reason to soften a
  science bar.

## Build sequence (subagent-driven; anti-cheat) -- to be detailed by writing-plans

Task 0 grounding pin -> Phase A (the verdict core + the in-bridge
mechanism wiring, pure-TDD where the substrate allows; the spiking run
itself validated by the gate, project pattern) -> DEDICATED ADVERSARIAL
REVIEWER on the load-bearing modules BEFORE Phase B (explicitly probe:
is the in-bridge discrimination genuinely isolated to the
temporal-credit mechanism, not a spiking-harness artifact; is
`hebbian_no_trace` a faithful in-bridge v16-analog; is the validated
mechanism reused byte-UNMODIFIED; can a non-discriminating/V1-broken
in-bridge run be scored PASS instead of VOID; any autograd) -> Phase B
no-harm -> Task 5 controller-only decisive multi-seed in-bridge run +
MANDATORY anti-cheat smell-test (PASS scrutinized HARDER than a FAIL) +
honest propagation EVERY outcome (findings + capability_status pillar +
schema green + push BOTH remotes). Kill-safe is a HARD requirement (the
spiking in-bridge run is genuinely heavier than the cheap numpy probes;
per-(seed) atomic checkpoint + KeyboardInterrupt-clean-exit, REUSED
`sim.train_checkpoint`).

## Honest ceiling (stated up front, NEVER spun)

- **IS (if PASS):** the validated temporal-credit mechanism transfers
  into a MINIMAL slice of the real spiking sim.bridge concept-pool
  architecture -- it bridges the verb->motor compositional bind-gap
  there, where the faithful no-trace v16-cold-start analog cannot. The
  first in-architecture mechanistic dent in the composition blocker.
- **IS NOT:** composition-solved. NOT compositional *language*. NOT
  the full 16/28-pool vocab, NOT chat integration, NOT scaled. This
  is a minimal-spiking-slice MECHANISM-TRANSFER validation; the full
  conversational-composition integration at scale is a further
  SEPARATE gated increment (YAGNI here).
- A FAIL/VOID is the strongest honest triangulation that the remaining
  composition blocker is specifically the spiking-dynamics integration
  (not the temporal-credit principle, validated three times now), NOT
  a license to escalate. PASS/BOUNDARY/VOID all decision-relevant +
  propagated honestly, no overclaim.

## Explicitly NOT in scope (YAGNI / honesty)

Full conversational composition; scaled concept-pool vocab; chat_repl
integration; predictive coding / laminar microcircuit. An honest
in-sim PASS/FAIL/VOID here is the terminus of THIS increment.
