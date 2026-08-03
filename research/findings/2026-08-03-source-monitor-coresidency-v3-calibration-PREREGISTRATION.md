---
type: preregistration
status: live
lane: laneC
date: 2026-08-03
mechanism: source-monitor-coresidency-v3
---

# Source-monitor v3 local-homeostasis calibration: preregistration

**Filed before the v3 runner was implemented and before any v3 seed was run.**
V1 and v2 results are prior evidence, not tuning data for this gate. V3 uses
fresh seeds and keeps the inherited source-margin and causal-attribution floors
unchanged.

## Functional Requirement

Source monitoring serves the whole brain by preserving which physical stream
produced a remembered event: visual input, auditory input, or the brain's own
motor corollary discharge. The useful quantity is robust separation of the
weakest source representation, because one unreliable source is enough to
misattribute remembered content. Surplus separation in an already strong source
has no independent functional value once it can be exchanged for a more robust
weak source without crossing the fixed decision floor.

This gate therefore permits a **bounded tradeoff**, rather than requiring the
stabilizer to improve or exactly preserve every source independently. A strict
zero-degradation rule would reject a circuit that makes all source identities
reliably usable by redistributing excess activity. An unbounded average-benefit
rule would be equally wrong because it could hide one source falling below a
usable margin.

## Biological Mechanism

V3 keeps v2's same-bridge fast-spiking GABA-A competition and adds intrinsic
excitability homeostasis only to the three source-memory populations. Each
source neuron updates its own spike threshold from its own recent firing-rate
trace. No host process measures source accuracy, selects a winner, or applies a
source-specific gain.

The implemented rule reuses the repository's `BrainRegion.enable_homeostasis`
and `fused_homeostasis_update` mechanism. Its parameters are frozen to the
existing simulator defaults, not selected from v2 outcomes:

- target firing probability per step: `0.02`;
- firing-rate EMA coefficient: `0.0002`;
- threshold adaptation rate: `0.0005`;
- threshold bounds: `-55.0` to `-30.0` mV.

After ordinary source learning, both comparison arms receive the same balanced
rehearsal of the three single-source episode patterns with Hebbian plasticity
closed. The schedule runs for at least `5,000` simulation steps, one time
constant implied by `1 / 0.0002`. Each pattern is active for the inherited 20
steps and followed by the inherited 80-step source-free rest. The intact arm
updates source-memory thresholds; the lesion arm keeps the same initialized
thresholds fixed. Threshold updates are frozen in both arms before measurement
so recall order cannot change the result.

This is biologically justified by the primary experiment of Desai, Rutherford
and Turrigiano (1999), *Plasticity in the intrinsic excitability of cortical
pyramidal neurons*, Nature Neuroscience 2:515-520: chronic changes in activity
caused neurons to adjust intrinsic excitability in the compensating direction.
The repository already identifies this work as the basis for its region-scoped,
cell-autonomous homeostasis implementation. Source identity itself remains
grounded in corollary discharge and reafference, following the primary proposals
of Sperry (1950) and von Holst and Mittelstaedt (1950), both already cited in the
project's source-monitor record.

## Fixed Seeds And Phase Lock

- Calibration, and the only open phase: seeds `220` and `221`.
- Development, reserved and mechanically rejected: seeds `222`, `223`, and
  `318`.
- Held out and mechanically rejected: seeds `319`, `320`, and `321`.

Both calibration seeds must pass without tuning between them before a separate
development preregistration may open development. Development and held-out
seeds must not be inspected or run while this calibration gate is live.

## Fixed Acceptance Rule

The inherited source floor is `F = 0.15`. For source `s`, let `M_s` be the v3
margin and `L_s` the matched local-homeostasis-lesion margin after the same
experiences. Define:

```
loss_s = max(0, L_s - M_s)
spendable_surplus_s = max(0, L_s - F)
```

All criteria must pass on each calibration seed:

1. Every intact seen, heard, and self-generated margin is at least `F`.
2. `loss_s <= spendable_surplus_s` for every source. Thus an above-floor source
   may spend only its surplus, while a source at or below the floor may not be
   weakened at all.
3. `min_s(M_s) > min_s(L_s)`: local homeostasis must strictly improve the
   weakest source representation rather than merely change thresholds.
4. Source-memory thresholds change when local homeostasis is enabled and remain
   fixed in the matched lesion. The homeostatic region mask must contain only
   source-memory neurons.
5. All inherited v2 causal and anti-cheat controls remain required: learned
   routes start at zero; experience changes weights; all three sources win;
   source swapping follows physical afferents; mixed visual-auditory experience
   reinstates both sources; episode-to-source lesion collapses recall with at
   least `0.90` attribution; ACC lesion silences ACC while preserving source
   recall with at least `0.90` attribution; learning-off leaves zero weights and
   zero recall; an unseen episode produces zero source recall; source activity
   reaches aPFC and ACC; and the fast-spiking competition circuit is active and
   lesionable.
6. All validity preconditions pass. Any undefined validity check makes the run
   `UNDEFINED`, never a pass.

The bounded-loss formula contains no tolerance learned from v2. Its only
numerical boundary is the already-frozen `0.15` functional floor. V2's observed
`-0.0092` change is neither encoded nor used to choose any v3 parameter.

## Fixed Implementation Boundary

- One `SimulationBridge` contains episode, source-afferent, source-memory,
  competition, aPFC, and ACC populations.
- Recall accepts sparse episode activity only; no source label, proposition,
  confidence, candidate answer, or response decision enters inference.
- V1 source drive, source-afferent weight, Hebbian learning rule, v2 competition
  population sizes, and v2 competition weights stay unchanged.
- The matched homeostasis lesion uses the same seed, patterns, training order,
  learned weights, competition, rehearsal schedule, and input strengths; it
  disables only source-region threshold adaptation.
- Population spike counts, margins, and threshold summaries are host-read for
  evaluation only and never feed neural dynamics.

## Explicit Scaffolds

Caller-supplied sparse episode activity, predefined source afferents, hand-wired
competition, externally timed learning windows, and competition suppression
during source-free rest remain developmental scaffolds inherited from v2. V3
also relies on a hand-selected source-memory region mask and fixed homeostatic
set point. Separating source learning, balanced homeostatic rehearsal, and
measurement is an externally timed developmental scaffold. It does not claim
learned source pathways, natural episodic allocation, language, confidence,
truthful speech, or a complete self-model.

## Stop Rules

- Do not change seeds, the `0.15` floor, the bounded-loss formula, homeostasis
  parameters, v2 weights, or inherited controls after a calibration result.
- A failure on either calibration seed is a v3 calibration NO-GO. Record it;
  do not run development or held-out seeds.
- A runtime failure counts as failure unless shown to be infrastructure-only.
- Any successor must use new seeds and a new preregistration.
