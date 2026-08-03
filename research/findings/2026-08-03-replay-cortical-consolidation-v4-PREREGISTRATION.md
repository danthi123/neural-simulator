---
type: preregistration
status: superseded
date: 2026-08-03
mechanism: replay-driven-cortical-consolidation-v4
runner: research/runners/_replay_cortical_consolidation_gate_v4.py
---

# Replay consolidation v4: dendritic target reinstatement

**Retired before scientific execution.** The fixed full-size smoke produced a
large target plateau but no target or target-FS spikes; removing the plateau
restored weak target firing. This candidate therefore failed its required
non-scientific dynamics check. Calibration seeds `451` and `457` were never
run and are not open.

**The initial v4 preregistration was filed before any v4 execution.** Seed
`216` is reserved for non-scientific construction and dynamics checks. It is
mechanically excluded from every scientific partition and cannot produce a
calibration verdict.

Smoke uses v3's unchanged full configuration so that it exercises the actual
index-to-target dynamics rather than a reduced construction-only network. Its
diagnostic checks may fail, but they cannot be converted into a scientific
verdict or used to tune the fixed v4 mechanism.

After the first reduced-size smoke proved dynamically silent downstream, the
smoke path was changed to use v3's unchanged full configuration. This changed
no mechanism, scientific criterion, or scientific seed, and occurred before
any formal execution.

## Functional requirement

Uncued hippocampal replay must cross the learned cortical index and reinstate
the associated cortical target strongly enough to recruit its local
fast-spiking inhibitory loop. The same brain must then recover both memories
after hippocampal retrieval is disabled. Recovery must still depend on learned
event identity, replay order, cortical plasticity, and the inherited local
inhibitory mechanisms.

## Mechanism under test

V3 reliably activated its learned cortical index but failed to recruit target
fast-spiking cells. V4 changes one synaptic boundary: the existing
`cortical_index` to `cortical_target` pathway also contributes to the weighted
dendritic coincidence mask. Coincident index spikes can therefore produce an
additive NMDA-like plateau in target cells. The pathway's ordinary AMPA
transmission remains active.

All v3 weights, coincidence threshold, plateau strength, timing, population
sizes, learning rules, and controls remain unchanged. The existing
CA1-to-index coincidence route also remains unchanged.

The new `target_plateau_lesion` removes only index-output synapses from the
coincidence mask during sleep. It does not close their transmission gate,
change their weights, or alter the CA1-to-index coincidence route.

## Seed and phase lock

- Non-scientific smoke only: `216`.
- Calibration, named but never opened: `451`, `457`.
- Development, locked: `461`, `463`, `467`.
- Held out, locked: `479`, `487`, `491`.

All scientific entry points now reject every seed, including `(451, 457)`.
Only smoke seed `216` remains executable so the negative dynamics boundary can
be reproduced.

## Fixed protocol

Every calibration seed runs separately initialized copies of all v3
conditions: intact, no-sleep, exact-content shuffled replay order, shuffled
learned target index, CA3-to-CA1 lesion, cortical-plasticity-off,
target-inhibition lesion, index-relay lesion, and index-balance lesion. V4 adds
the target-plateau lesion described above.

Every condition uses one bridge through wake encoding of A, wake encoding of
B, uncued sleep, and cortical retest with hippocampal retrieval disabled. The
shuffled-order condition preserves the exact event-content multiset while
changing only order. No host process chooses a replayed memory or directly
drives a cortical target during sleep.

## Fixed validity preconditions

All v3 validity preconditions remain. In addition, the result is `UNDEFINED`
unless the target-plateau lesion disables every index-output coincidence-mask
entry while leaving its AMPA transmission gain at `1.0`, and leaves every
CA1-to-index coincidence-mask entry enabled. Intact execution must have all
index-output coincidence-mask entries enabled.

## Fixed scientific criteria

All nine v3 scientific criteria remain unchanged and must pass on both
calibration seeds. V4 adds these criteria:

1. During intact sleep, target plateau peak and integrated area are both
   greater than zero.
2. During intact sleep, both cortical-target and cortical-target-FS spike
   totals are greater than zero.
3. Under target-plateau lesion, cortical-index and cortical-index-FS spike
   totals each remain greater than zero and at least `75%` of intact.
4. The target-plateau lesion reduces cortical-target spikes by at least one and
   to at most `75%` of intact.
5. The target-plateau lesion reduces cortical-target-FS spikes by at least one
   and to at most `75%` of intact.

Both calibration seeds must pass every inherited and new criterion without
tuning between them.

## Telemetry

Each sleep condition records target plateau peak and integrated area for each
memory assembly and overall, region spike totals, enabled coincidence-route
counts, and index-output transmission gain. This separates a silent plateau,
a failed target response, and a failed target-inhibitory response.

## Host boundary and scaffolds

The host defines wake episode populations, partial probes, fixed relay and
inhibitory channel membership, sleep event boundaries, and episode-agnostic
CA3 background current. It reads known assemblies for scoring and telemetry.
Fixed anatomy, the wake teacher pathway, rate-window Hebbian learning,
scheduled sleep, and the single-subunit point-neuron approximation remain
explicit scaffolds. The host may not rank memories, select a replay event,
stimulate a selected cortical target during sleep, or supply the target
plateau.

The mechanism follows the weighted, pathway-routed NMDA coincidence substrate
specified in
`research/findings/2026-06-09-coincidence-substrate-upgrade-design.md`. That
design grounds the approximation in Major, Larkum, and Schiller's review of
dendritic NMDA spikes; Poirazi, Brannon, and Mel's two-layer pyramidal-cell
model; and Branco, Clark, and Hausser's work on dendritic temporal
discrimination. V4 does not claim full dendritic compartment anatomy.

No calibration may run from this retired design. A successor needs a different
mechanism and a fresh preregistration rather than a retuned target plateau.
