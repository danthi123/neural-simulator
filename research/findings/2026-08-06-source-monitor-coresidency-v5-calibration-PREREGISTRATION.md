---
type: preregistration
status: live
lane: laneC
date: 2026-08-06
mechanism: source-monitor-coresidency-v5
seed_integrity: fresh-partition-before-any-evidence
---

# Source-monitor v5 calibration: preregistration

**Filed before the v5 runner produced any formal evidence and on a fresh seed partition.** V1-v4 results are prior
evidence, not tuning data. The functional criterion below is taken verbatim from the P3 functional-role spec
(2026-08-06-source-monitor-P3-functional-role-and-tradeoff-spec.md), which itself adopts the bounded-loss rule the v3
preregistration filed on 2026-08-03 before any v3 evidence. No number here is chosen from any observed v2/v3/v4 margin.

## Hypothesis

The v2 local fast-spiking GABA-A biased-competition circuit already satisfies the whole-brain-justified bounded-loss
max-min tradeoff (it was rejected only by an over-strict per-source zero-degradation control). This gate tests whether
that mechanism, UNCHANGED, meets the bounded-loss criterion robustly across fresh seeds.

## Mechanism (unchanged from v2)

Identical to `_laneC_source_monitor_coresidency_gate_v2.py`: one `SimulationBridge` holds episode, source-afferent,
source-memory, six-neuron fast-spiking competition, aPFC, and ACC populations. Each source-memory pool recruits its own
FS interneuron pool that inhibits the two rival source pools through GABA-A on the same bridge. V1 source drive,
source-afferent weight, Hebbian rule, FS population sizes, and FS weights are unchanged. No source-specific host gain,
no stronger labelled drive, no host normalization, no host confidence scalar, no host response decision. The ONLY change
from v2 is the acceptance rule and the seed partition.

## Fixed acceptance rule (bounded-loss, guard-the-floor, max-min)

Floor `F = 0.15` (inherited, unchanged). For source `s`, `M_s` is the v5 intact margin and `L_s` the matched
competition-lesion margin after identical experience. Define `loss_s = max(0, L_s - M_s)` and
`spendable_surplus_s = max(0, L_s - F)`. All must pass on EACH calibration seed:

1. **Floor held:** `min_s M_s >= F`.
2. **Only surplus spendable:** `loss_s <= spendable_surplus_s` for every source.
3. **Weakest strictly improves:** `min_s M_s > min_s L_s`.
4. **No inversion:** the correct source is dominant for seen, heard, and self-generated recall.
5. **Inherited v2 controls, all required:** learned routes start at zero; experience changes weights; source swapping
   follows physical afferents; mixed visual-auditory experience reinstates both sources; episode-to-source lesion
   collapses recall with at least `0.90` attribution; ACC lesion silences ACC while preserving source recall with at
   least `0.90` attribution; learning-off keeps zero weights and zero recall; an unseen episode produces zero source
   recall; source activity reaches aPFC and ACC; the FS competition circuit is active and lesionable; recall accepts
   sparse episode activity only, with no source label, candidate, confidence, or response entering inference.
6. **Validity:** any undefined validity precondition makes the run UNDEFINED, never a pass.

<!--derived-->
No tolerance is learned from v2's `-0.0092` (a value from the prior v2 finding, not a v5 measurement). The only
numerical boundary is the frozen `0.15` floor.

## Fixed seeds and phase lock

- Implementation smoke only, never formal evidence: seed `649`.
- Calibration, the only open formal phase: seeds `650`, `651`.
- Development, reserved and mechanically rejected: seeds `652`, `653`, `654`.
- Held out and mechanically rejected: seeds `655`, `656`, `657`.

Both calibration seeds must pass without any tuning between them before a separate development preregistration may open
development. Development and held-out seeds must not be inspected or run while this calibration gate is live. Unit or
smoke construction of dynamics may use only seed `649`.

## Stop rules

- Do not change seeds, the `0.15` floor, the bounded-loss formula, the v2 mechanism, or inherited controls after a
  calibration result.
- A failure on either calibration seed is a v5 calibration NO-GO. Record it; do not open development or held-out seeds.
- A runtime failure counts as failure unless shown to be infrastructure-only.
- Any successor must use new seeds and a new preregistration.

## Explicit scaffolds

Caller-supplied sparse episode activity, predefined source afferents, hand-wired competition, externally timed learning
windows, competition suppression during source-free rest, and host spike-count evaluation remain developmental
scaffolds inherited from v2. A pass does not claim learned source pathways, natural episodic allocation, language,
confidence, truthful speech, or a complete self-model.
