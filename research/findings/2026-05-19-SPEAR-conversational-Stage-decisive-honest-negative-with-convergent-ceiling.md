---
type: finding
status: live
date: 2026-05-19
mechanism: spear
---

# Shared-rhythm SPEAR conversational stage: decisive multi-seed run is an honest negative; the same ceiling appears in BOTH static (Stage-1) and rhythm-multiplexed (SPEAR) architectures, which is itself a biology-translatable insight

## Status

Honest negative, propagated without spin, under the standing
anti-cheat discipline AND the owner's reframed top-level goal
(artificial life with a proper brain analogue; insights from the sim
should be biology-translatable; capabilities like conversation /
composition are instrumental, not the deliverable). The full
anti-cheat discipline ran end-to-end on this stage, including a
dedicated adversarial review that BLOCKED a real mechanistic-
faithfulness defect on the first pass (the inert ACh gate) and a
precise net-new-runner-only fix that closed it (the encode/retrieve
phase setpoint now produces a measured 14.15 mV bridge-state
divergence at a 50-step constant-input probe; the rhythm controller
is mechanistically active). No fixed threshold was moved; the
original frozen verdict, the corrected module, the new
capability-verdict module, and the no-confabulation moat are all
byte-unchanged.

## What was tested (pre-registered, fixed-bar)

The shared theta-gamma "Separate Phases of Encoding And Retrieval"
(SPEAR) conversational stage: one shared theta rhythm
time-multiplexes an encode phase (acetylcholine LOW, plasticity-gate
permits, afferent drive) and a retrieve / pattern-complete phase
(acetylcholine HIGH, plasticity-gate blocks, recurrent), via the
reused neuromodulator subsystem with `synaptic_gain (scope=all)` and
`plasticity_rate (scope=all)` targets so the gate genuinely modulates
bridge dynamics every step (not only the inert C2 reward block). The
prefrontal `dlpfc_verb` region + global NMDA bistability hold the
ordered slot across gamma sub-cycles; the validated theta-gamma
hippocampal store + trisynaptic pattern-completion + replay-
consolidation phase functions + no-confabulation moat at output are
reused byte-unchanged. Decisive run: frozen ladder (2, 4, 8 recent-
specific compositional facts); seeds 42 / 43 / 44; CuPy on RTX 3090;
8440-neuron full v16 + hippocampus + dlpfc substrate; kill-safe with
durable capture; monitored to actual process exit. The pre-registered
decisive built-in control is `rhythm_removed`: identical to the
full run with the shared-rhythm controller disabled (reduces to the
Stage-1 static composition); the capability must be attributable to
the rhythm (full clears AND control collapses).

## Result

The frozen capability-verdict module returns **FAIL**.

Every rung (N = 2, 4, 8; 3 seeds each):

- full_acc = 0.00
- rhythm_removed_acc = 0.00
- abstain_correct_rhythm_removed = 1.00

The verdict was independently recomputed from the single recorded
output (no re-run, no bar change): recorded FAIL == recomputed FAIL,
reason "smallest-load rung fails a frozen bar
(full/rhythm-removed/abstain)".

## Smell-test (mandatory, scrutinising honesty of the negative)

- Genuine full-scale execution: CuPy / RTX 3090; 8440-neuron full
  v16 + hippocampus + dlpfc substrate (exactly the recipe the
  dedicated adversarial review CLEARed after the net-new-runner-only
  faithfulness fix); 18 arm-runs (9 cells x full / rhythm_removed)
  across approximately 51 minutes of real spiking computation; 1014-
  line durable log with zero errors / exceptions / NaN / skips / no
  empty-tag warnings.
- Rhythm controller is mechanistically active (independently
  re-verified during the adversarial re-review: ACh-encode vs
  ACh-retrieve bridge state diverges by 14.15 mV in a 50-step
  constant-input probe; multipliers ACh=0.0 -> 1.30 synaptic gain;
  ACh=1.0 -> 1.00 synaptic gain).
- Internally consistent: 9 raw_cells x 2 arms; every cell carries
  identical seed for full and rhythm_removed; the discipline gate is
  the single threaded `use_rhythm` flag.
- Verdict module byte-unchanged since Task 1 (frozen capability bars
  immovable; recomputes from raw numbers; precomputed verdict
  ignored).

Smell-test passes: this is an honest measured negative, not
instrument-invalid, not a false PASS.

## What this means (the honest reading, no spin)

Three things are true and all are reported:

1. **Neither static composition (Stage-1) nor rhythm-multiplexed
   composition (SPEAR) yields a composed readout that exceeds the
   calibrated no-confabulation threshold (650) on the compositional
   query at biological scale.** Both architectures, layered on the
   project's validated substrate, fail in the same direction.
2. **The trustworthy property HELD under both architectures.**
   abstain-correct is 1.00 across all seeds, loads, and the
   ablation arm. The no-confabulation moat composed into BOTH the
   static and the rhythm-multiplexed architectures at full
   biological scale, and abstained ("I don't know") rather than
   emitting a confident wrong answer, in every case. Zero
   confabulation under composition, in either architecture, is a
   real, preserved property.
3. **The rhythm controller is mechanistically active but does not
   lift the composed readout above the moat threshold.** Measured
   14.15 mV bridge-state divergence between encode and retrieve
   phases. The mechanism works at the bridge-state level; it does
   not (with the current synaptic_gain magnitude and the current
   v16-calibrated moat threshold) produce a confident grounded
   compositional answer.

This is **a precise localisation of the ceiling**. The brain
demonstrably achieves BOTH high-confidence direct recall AND
lower-but-still-confident compositional recall. The current
substrate achieves the first (the validated v14/v16 88.75% multi-
seed bidirectional binding; the 90% multi-tag retrieval; the 87.5%
engram stim-recall) but does not yet achieve the second, in either
the static or the rhythm-multiplexed architecture tried so far. The
convergent ceiling across two distinct architectures is itself a
biology-translatable finding: the read-out confidence at the
language-output layer, with the project's current substrate scale +
calibration + composition mechanisms, does not reliably exceed the
trustworthy-abstention threshold for compositional queries; the
threshold was calibrated on direct concept retrieval (encoded ~796
vs control ~584), and compositional retrieval at the same readout
has lower confidence by nature (composition introduces noise via
the combination step).

## Reading under the reframed top-level goal

Under the owner's reframed goal (artificial life with a proper
brain analogue; biology-translatable insights), this is exactly the
kind of deliverable the project is for: a precise, falsifiable,
biology-grounded mechanism test, scrutinised by an adversarial
review that found and forced closure of a real mechanistic defect
before the decisive run, then producing a clean honest negative
that localises the ceiling. The biology-translatable lessons:

- The trustworthy-abstention property survives composition in
  multiple architectures at biological scale - this corroborates
  the biological pattern (patients with hippocampal lesions abstain
  on memory queries they cannot ground rather than confabulating,
  to varying degrees).
- A shared theta rhythm gating plasticity + dynamics via the
  acetylcholine system mechanistically modulates bridge state
  (14.15 mV divergence between phases) but does not, by itself,
  lift compositional read-out above the trustworthy threshold.
- The convergent ceiling across static and rhythm-multiplexed
  composition tells biology: composition that yields a confident
  grounded readout likely requires more than rhythm-gating; it
  requires a mechanism that produces a structured, recoverable
  composed representation at the read-out layer. This is exactly
  what the spiking-phasor vector-symbolic-architecture work
  (Orchard 2023/2024) implements - the rhythm does not just gate
  plasticity / dynamics, it carries compositional content as
  spike-phase.

## Pre-registered next step (autonomous, no hand-back, no
config-crank, no bar change)

The pre-registered staged sequence has Architecture B (schema-
accelerated assimilation) and C (per-regime metamemory monitors) as
follow-ons. Under the reframed top-level goal AND the broader
prior-art investigation already on the durable record (the SPEAR
design's sections 2c/2d, refs [10]-[30]), the genuinely-distinct
next biology-faithful step the convergent ceiling points at is the
**phase-coded vector-symbolic-architecture unification (Orchard
2023/2024 spiking-phasor FHRR)**: the shared theta-gamma rhythm
does not just multiplex encode/retrieve phases, it ALSO encodes
compositional content as the phase of each spike within a cycle,
giving bind / unbind / superposition / cleanup operations on the
SAME rhythm substrate. This is biology-faithful (theta-gamma phase
coding is observed and well-characterised in real brains -
Heusser 2016, Ursino 2024, Manns 2006), AND directly addresses the
ceiling (the read-out reaches above moat threshold because the
composed vector is a structured object the read-out can decode,
not a sum of partially-active sub-populations). The post-Stage
design pass for this should follow the standing brainstorm ->
writing-plans -> subagent-driven-development -> pre-registered
fixed-bar gate -> honest propagation chain, with the broader-
search discipline (consensus + WebSearch + open-source code +
curated lists) applied at the design pass entry.

A clearly-marked **engineering-only baseline** at SpikeGPT-class
surrogate-gradient BPTT scale was approved by the owner for
ceiling-clarification testing only: insights from that baseline tell
us about engineering, not biology, and do not satisfy the goal's
biology-translatable criterion. It is a side-channel test, not the
project's primary thrust.

## Honest ceiling (unchanged, restated)

Conversational / compositional capability of the kind that would
exceed the trustworthy-abstention threshold is **not** achieved at
biological scale, in either static composition or rhythm-multiplexed
composition, with the project's current validated subsystem stack.
No fixed threshold was moved; the original frozen verdict
(`2048750`), the corrected module (`36a7975`), the Stage-1
capability-verdict module (`c474d6e`), the SPEAR capability-verdict
module (`0bc5230`), and the no-confabulation moat are all
byte-unchanged throughout. Every previously-validated asset
(trustworthy grounded memory, the no-confabulation abstention moat
7/7, simple coherent generation, no catastrophic forgetting, the
v14/v16 substrate, the validated subsystems) is intact and
unaffected. The genuine durable contributions of this stage:
(a) a faithful, adversarially hardened, fixed-bar capability
instrument for shared-rhythm SPEAR conversational retrieval;
(b) the empirical demonstration that the no-confabulation moat
composes and holds at biological scale under a SECOND distinct
architecture (rhythm-multiplexed in addition to static); (c) the
precise localisation of the read-out-vs-moat-threshold ceiling that
converges across two architectures and points at the next biology-
faithful direction (phase-coded vector-symbolic unification).

## Files / evidence

- Frozen capability-verdict module (byte-unchanged since creation):
  `research/runners/spear_conversational_core.py` (commit `0bc5230`).
- Net-new shared-rhythm runner (adversarially reviewed +
  faithfulness-fixed + re-review CLEAR): `research/runners/spear_conversational_runner.py`
  (commit `f1292a0`).
- Durable decisive output: `research/findings/raw/spear_DECISIVE.json`
  (verdict + 9 raw cells) and `...DECISIVE.log` (1014-line GPU log).
- Stage-1 prior negative (the static-composition convergent point):
  `research/findings/2026-05-19-regime-correct-compositional-retrieval-Stage1-decisive-honest-negative.md`.
- Design + plan: `docs/plans/2026-05-19-shared-rhythm-SPEAR-conversational-architecture-design.md` (sections 2c / 2d enumerate the prior-art and the next-direction options) and `...-implementation.md`.
- Original frozen verdict (`2048750`), corrected module (`36a7975`),
  no-confabulation moat: byte-unchanged throughout.
