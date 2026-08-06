---
type: research-gate
status: selected
date: 2026-08-06
lane: memory-replay
mechanism: selective-ca1-to-cortex-target-reinstatement
supersedes-method: fixed-intracortical-index-to-target-teacher (v3/v4)
---

# Selective CA1-to-cortex target reinstatement — research gate

## Wall that triggered this gate

ROADMAP Capability #4 (lived reconstructive memory) is locked at the same wall
across four banked attempts of the replay-driven cortical consolidation gate.
The v1-v4 quantities below are quoted from the cited 2026-08-03 NO-GO / UNDEFINED
findings, not re-measured here:

<!--derived-->

- v1 (`_replay_cortical_consolidation_gate`): NO-GO. Causal replay-to-cortex path
  proven (no-sleep / CA3-CA1-lesion / plasticity-off all give zero recovery) but
  weak+noisy: one seed diffuse false recall (0.358), the other near-inert.
- v2 (`..._v2`): NO-GO. Opponent fast-spiking inhibition sharpened selectivity
  (false recall 0.358 -> 0.019 on seed 212) but memory-B recovery and the
  learned-target-index / replay-order advantages did not repeat.
- v3 (`..._v3`): UNDEFINED (invalid). Added an index relay + GABA_B balance loop.
  Cortical index relayed (156/240 spikes) but cortical target under-fired
  (19/39 spikes) and cortical-target-FS fired zero, so the validity precondition
  failed.
- v4 (`..._v4`): retired pre-execution. A dendritic index->target coincidence
  plateau shunted the target (large plateau, zero target/target-FS spikes).

Every attempt fails at one point: the correct cortical target is not reinstated
strongly enough during replay to (a) recruit its opponent-FS for selectivity and
(b) let the consolidation wire potentiate.

## Sharpened diagnosis (live substrate reads this session)

Reading the v3 wiring plus running the seed-216 smoke localized two stacked
failure modes, not one (smoke numbers are from a non-scientific seed-216
diagnostic; calibration numbers are quoted from the v3 UNDEFINED artifact):

<!--derived-->

1. Smoke scale: replay reaches CA3 (30 spikes, 4 reactivated events) but sleep
   spikes read `ca1:0, cortical_cue:0, cortical_target:0, index:0`. CA1 itself is
   silent at small scale (Schaffer convergence, matching
   `2026-07-09-riii-swr-generative-replay-rung1-ca1-drive-scale`). A teacher-weight
   diagnostic sweep (index->target 34 -> 320) left `cortical_during_sleep` at 0.0
   because the deficit is upstream of the teacher. This is a scale artefact of the
   smoke, orthogonal to the mechanism.
2. Calibration scale (v3 seeds 228/229, from the UNDEFINED artifact): CA1 relays to
   the index fine, but the FIXED intracortical `cortical_index->cortical_target`
   teacher (w=34) yields only 19/39 target spikes — too few to recruit target-FS or
   to co-activate the plastic `cortical_cue->cortical_target` association wire, which
   moves 0.0. This is the circular trap named in
   `2026-07-25-consolidation-frontier-research-gate-scoping...`: the wire will not
   potentiate until the pool fires, and the pool will not fire until the wire
   potentiates.

Structural root cause: v3 fires the target during replay through a **fixed
intracortical teacher**. There is no CA1->cortical_target pathway. So target
reinstatement never depends on the learned, memory-specific hippocampal index.

## Biology (CLS) and what the record already scoped

McClelland-McNaughton-O'Reilly 1995 / Tse 2007: during sharp-wave-ripple replay the
hippocampus does not merely index — it **reinstates** the cortical target pattern
via CA1->cortex synapses potentiated at encoding; repeated co-activation trains the
intracortical association until it is recallable without the hippocampus. The
`2026-07-25` frontier gate already prescribed the two required parts: (1) co-activate
CA1 + target during replay so the wire potentiates, and (2) give the target a
selective, self-sustaining attractor that neither collapses nor becomes a single
global winner. v3/v4 attempted part (1) with a fixed intracortical teacher and never
supplied the learned hippocampal reinstatement.

## Selected next mechanism

**Learned, encoding-potentiated CA1 -> cortical_target reinstatement**, replacing
the fixed intracortical teacher:

- Add a plastic `ca1 -> cortical_target` pathway on the memory-specific pairs
  `(pat["ca1"], pat["target"])`, plasticity gate ON during wake encode. CA1 fires
  (via ca3->ca1) while the target is host-driven during encode, so Hebbian
  co-activity makes the pathway memory-specific by construction.
- Sleep: transmission ON. Uncued CA1 replay reinstates the correct target directly,
  co-activating it with the cue so `cortical_cue->cortical_target` consolidates, and
  driving target-FS so the opponent loop enforces A-vs-B selectivity.
- Retest: hippocampus disabled -> CA1 silent -> the ca1->target pathway contributes
  nothing; recall must come from the consolidated intracortical association. No
  cheat: the hippocampal reinstatement is a teaching scaffold that is removed before
  the read.
- Keep the v2 opponent-FS competition and the true replay-order control.

This is why it is "selective CA1-to-cortex reinstatement" and NOT another global
learning-rate sweep: selectivity is carried by a learned, memory-specific
hippocampal pathway rather than a uniform intracortical teacher, and the change is
structural (a new plastic pathway + its encode/sleep/retest gating), not a scalar.

## Anti-cheats (the result IS the anti-cheats)

no-sleep -> zero recovery; CA3-CA1 lesion -> zero; cortical-plasticity-off -> zero;
shuffled-target-index -> recovery collapses (the pathway is memory-specific);
shuffled-replay-order control preserved; hippocampus-disabled retest (proves
recall is intracortical); permuted-index must not beat intact by the fixed margin;
control-outperforms-real guard.

## Honest expected bounded negative + named surpass

On single-compartment point neurons the reinstated target may still fail to hold or
may latch to a single global winner (the P0.3 saturation / A1 runaway pathology the
`2026-07-25` gate predicted). If so, that is the deliverable: it names the surpass as
**spike-frequency-adaptation-driven one-of-N eviction** on the target attractor — a
simple spiking mechanism. This is explicitly NOT the v14 ion-channel / dendritic
compartment rabbit hole (P3 drift, parked); no conductance calibration is in scope.

## Compute state

Nothing new is running. The v3/v4 runners are frozen scientific artifacts with
calibration seeds locked, so a clean next verdict requires a v5 runner (adds the
plastic ca1->target pathway + encode/sleep/retest gating) and a fresh calibration
preregistration on new seeds — a deliberate act, not a run to slip in under time
pressure. The mini-PC pool is free for that v5 calibration once built. Per the
"two dead ends on one defect -> stop and report" discipline (four banked attempts
here), this session stops at the selected, spec'd mechanism.

## Evidence

`_replay_cortical_consolidation_gate_v3.py` (wiring: no ca1->target path; target
driven only by fixed index->target teacher); seed-216 smoke sleep spikes
`{ca3:30, ca1:0, cortical_target:0}`; v3 UNDEFINED + v2/v1 NO-GO findings dated
2026-08-03; scoping `2026-07-25-consolidation-frontier-research-gate-scoping...`;
CA1-drive scale `2026-07-09-riii-swr-generative-replay-rung1-ca1-drive-scale`.
