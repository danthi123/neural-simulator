---
type: plan
status: live
date: 2026-05-17
---

# Dendritic Credit Assignment — the neuron IS the credit-assignment machinery (spiking, biologically-local, self-contained) — Design (ACTIVE)

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). Continuous autonomous arc
> (user 2026-05-17, explicit: "extensively plan out and autonomously
> implement and integrate"; full standing authorization — a week alone,
> no stopping/asking, documented design calls, full architectural
> freedom, local 3090 only, biological realism maintained, public
> corpus authorized, no cheats, self-contained at RUNTIME). Addresses
> the #1 diagnosed root cause of every project negative.

## Why this is the genuinely-different, decision-relevant direction (NOT config-cranking)

The conversational-generation/realization program is a terminally
converged, multiply-pre-registered, multiply-confirmed TERMINUS (14
mechanisms + meta-terminus). The diagnosis of *why*: every project
negative traces to one root — **local biological rules (STDP /
three-factor / eligibility / Hebbian) on POINT neurons cannot do
hidden/temporal credit assignment; gradient can** (the 2026-05-05 W→A
verdict: classical-DA 1/6, graded-DA 0/6, gradient 3/3 PERFECT under
the *identical* architecture — "architecture sufficient, the
credit-assignment RULE is the bottleneck"). The deeper truth: in real
pyramidal cells the **dendritic tree itself is the credit-assignment
hardware** (Larkum 2013 apical amplification / BAC firing;
Urbanczik-Senn 2014 dendritic prediction; Guerguiev-Lillicrap-Richards
2017 segregated-dendrite spiking backprop-approximation;
Sacramento-Senn 2018 microcircuit; Payeur-Naud-Richards 2021
burst-dependent plasticity). We had point/shallow neurons with a
uniform rule stapled on. This direction is genuinely different from all
14 terminated mechanisms: it is a **substrate-level credit-assignment**
change, not a generation/realization scheme, not a decoding knob, not
order-over-an-order-blind-pool. It was the deferred 2026-05-05 design
whose own decision-gate was confirmed met that week.

## Evidence grounding (falsify-cheaply, done BEFORE designing — 2026-05-17)

A cheap throwaway probe: three learners, identical 2→H→1 net,
multi-seed (5 seeds), pre-registered FIXED bars (never tuned; the
control was STRENGTHENED to the canonical unimpeachable floor — a
single-layer delta learner — which makes the claim *harder*, the
Generator-H STRENGTHEN-only discipline):

- **baseline** (canonical non-dendritic local floor: single-layer
  delta, no hidden credit possible): OR **1.000** (fully competent
  linear learner), XOR **0.500** (chance, 5/5 — provably fails the
  nonlinear hidden-credit task).
- **dendritic** (2-compartment + FIXED-RANDOM apical feedback
  [feedback alignment: NO weight transport, NO global backprop, LOCAL
  info] + local somato-dendritic mismatch): XOR **1.000, all 5 seeds**
  = the exact-backprop oracle. OR 1.000.
- **oracle** (exact backprop, weight transport — implausible upper
  bound, correctness ceiling only): XOR 1.000.

**Decisive clean PROBE POSITIVE:** biologically-local dendritic credit
assignment genuinely solves hidden credit assignment a competent
non-dendritic local rule provably cannot — reproducing
Guerguiev-Lillicrap-Richards 2017 as a 90-second test. The #1 lever is
grounded. **Honest ceiling carried forward (no overclaim — enforced
twice this session):** this is a RATE numpy probe on XOR; it proves the
PRINCIPLE, NOT that it survives the SPIKING substrate at project scale.
That is exactly the genuinely-open question the full build's
pre-registered gate must answer.

## Thesis

A net-new **spiking two-compartment pyramidal neuron** whose morphology
*is* the credit-assignment machinery: a basal compartment integrating
bottom-up drive, an apical compartment integrating top-down feedback
delivered through a **FIXED RANDOM** projection (feedback alignment —
biologically plausible, no weight transport, not backprop), and a soma
whose Larkum-style BAC integration (basal alone → high threshold;
basal+apical coincidence → burst / lowered threshold) makes the
apical-basal relationship a LOCAL credit signal. Forward (basal)
plasticity is the LOCAL Urbanczik-Senn somato-dendritic mismatch,
apical-gated. No global error, no backprop, no non-biological gradient
at runtime. The artifact is self-contained (spiking weights only) and
local. The pre-registered question: does this move the **project's own
W→A binding task** — which point-neuron+3-factor provably failed (1/6)
— toward the gradient oracle (3/3)?

## Pre-registered gate (FIXED bars, never tuned; own module; multi-seed)

Mirror the hardened anti-cheat discipline (own frozen constants in a
net-new `dendritic_core`; do NOT import/modify gate_core /
song_g1_core / subword_lm_gate_core / generator_g_core /
generator_h_core / abstention_gate; ≥3 seeds; permuted/shuffled
control; mandatory smell-test; never tuned). LOAD-BEARING criteria:

1. **Credit-assignment moves the project-relevant trajectory:** on the
   W→A 4-word binding task (the validated point-neuron+three-factor
   architecture scores ≤1/6 aligned — the W→A verdict reference),
   the spiking two-compartment + feedback-alignment + local plasticity
   learner must reach a FIXED aligned bar (pre-registered, e.g. ≥4/6
   aligned multi-seed) — i.e. it must move from the
   global-scalar-feedback floor toward the gradient oracle. Permuted-
   label control: the same learner on shuffled labels must NOT clear
   the bar (proves it's learning the task structure, not exploiting
   architecture noise — the exact 2026-05-03 control that caught a year
   of false positives).
2. **Biologically-local by construction (the distinctive integrity
   bar):** the credit signal must be computed with LOCAL information
   only — apical feedback weights are FIXED RANDOM (no weight
   transport), no global backprop / autodiff graph at runtime
   (unit-tested: a spy that asserts no cross-neuron gradient object is
   ever formed; the shipped artifact imports no autograd). `bptt_snn`
   is reused ONLY as the correctness ORACLE in tests, never in the
   artifact.
3. **No-harm (LOAD-BEARING):** the validated no-confabulation moat
   (`abstention_gate` gate 650 + `tests/test_abstention_gate.py`) and
   all frozen cores remain byte-UNMODIFIED + green across the whole
   commit range — the distinctive contribution is never regressed.
4. **MANDATORY anti-cheat smell-test:** scrutinize a PASS HARDER than a
   FAIL (the Generator-S false-PASS lesson); read the actual learned-
   behaviour transcripts; recompute from recorded JSON; no re-run; no
   bar-tuning.

PASS (scrutinized) ⇒ the credit-assignment ROOT (#1) is genuinely
addressed in the biological substrate — a major decision-relevant
result, reported STRICTLY at that scope: it does NOT solve #3
(developmental/embodiment) and is NOT "conversation solved". FAIL ⇒
honest decision-relevant terminus (biologically-local dendritic credit
assignment does not survive the spiking substrate at feasible local
scale); the deliverable stays the validated assets. Either way
propagated (findings + capability_status + both remotes), NOT
config-cranked into endless dendrite variants.

## Architecture (net-new; validated stack reused UNMODIFIED / DRY)

Recommended **Arch A** (the direct spiking lift of the validated
principle; B/C deferred per below):

- Net-new `sim/dendritic_neuron.py` — spiking two-compartment
  pyramidal: per-neuron `(V_basal, V_apical, V_soma, recovery)`;
  Larkum BAC soma rule; basal forward synapses; apical fed by a FIXED
  RANDOM feedback projection from the teaching/target signal.
- Net-new `sim/dendritic_plasticity.py` — pure LOCAL Urbanczik-Senn
  somato-dendritic mismatch rule, apical-gated. Pure-unit-testable;
  finite-difference-checked against the `bptt_snn` oracle on a tiny
  net (equivalence within tolerance ⇒ the local rule genuinely
  approximates the gradient).
- Net-new `research/runners/dendritic_core.py` — OWN frozen
  pre-registered bars + verdict + multiseed aggregate (mirrors the
  hardened discipline; imports/modifies NO existing core).
- Net-new `research/runners/dendritic_wa_gate.py` — the pre-registered
  W→A gate runner; reuses the validated W→A task harness
  (bio_three_factor-style) for the point-neuron baseline + the gradient
  oracle; kill-safe; ASCII; <3 seeds → exit 2.
- Reused byte-UNMODIFIED: `abstention_gate` (+ its test), the validated
  W→A/bio_three_factor task definition, `sim/bptt_snn*` (oracle in
  tests only), all frozen cores, `sim/bridge.py`.

**Arch B (burst-dependent, Payeur-Naud-Richards 2021)** and **Arch C
(Sacramento-Senn microcircuit)** are deferred: B only if A's *specific*
failure mode indicates burst-multiplexing (a different mechanism
justified by A's evidence, NOT a knob escalation); C only if both A and
B fail (the deferred doc's pre-existing next-tier gating). An Arch-A
FAIL is the honest terminus, NOT an automatic license to escalate.

## Honest ceiling / risks (no overclaiming)

- The probe is RATE/XOR; spiking discretization is the genuine open
  risk — the gate exists precisely to decide it; a FAIL is a
  propagated honest terminus.
- Even a PASS addresses ONLY the credit-assignment root (#1). It does
  NOT supply the embodied developmental trajectory (#3) and is NOT
  "fluent grounded conversation solved". Reported strictly at that
  scope, never spun.
- Self-contained at RUNTIME (spiking weights only; no autograd, no
  external LLM/corpus at runtime). Local 3090 / CPU for pure tests.
- The validated no-confab moat is the distinctive primary asset and
  MUST remain byte-identical and green.

## Out of scope (YAGNI)

#3 developmental/embodiment (strategic fork the owner owns; not
closeable locally; explicitly NOT claimed). No predictive-coding
microcircuit (Arch C) unless A and B fail. No new global bar; no
modification of any validated/frozen module. No backprop/autograd in
the shipped artifact. No config-cranking any terminated mechanism.

## Scientific basis

Larkum 2013 (apical amplification, BAC firing); Urbanczik-Senn 2014
(dendritic predictive plasticity, local); Guerguiev, Lillicrap &
Richards 2017 (segregated dendrites, spiking, feedback alignment —
the principle the probe reproduced); Sacramento, Costa, Bengio & Senn
2018 (dendritic cortical microcircuits approximating backprop);
Payeur, Guerguiev, Zenke, Richards & Naud 2021 (burst-dependent
plasticity). The hardened pre-registered FIXED-bar multi-seed +
permuted-control + mandatory-smell-test discipline is the adjudicator.
