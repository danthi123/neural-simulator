---
type: preregistration
status: live
date: 2026-08-03
mechanism: visual-hierarchical-part-identity
runner: research/runners/_laneD_visual_hierarchical_part_identity_gate.py
---

# Visual hierarchical part identity: calibration preregistration

**Filed before calibration seeds `503` and `509` were run.** Seed `222` is
reserved for non-scientific smoke checks and cannot support a capability claim.

## Functional requirement

The simulated brain must recognize the same object across unfamiliar changes
in position, scale, lighting, and noise. It must learn this without receiving
object labels. Local shape parts should be learned before the system tries to
bind whole-object identity across a continuous sequence of views.

## Mechanism under test

The existing fixed Gabor front end supplies orientation-sensitive V1 activity.
A 4 by 4 retinotopic V2 grid receives only local 3 by 3 spatial neighborhoods
across all orientations. Each V2 hypercolumn contains 12 excitatory part cells
and four local FS/PV inhibitory cells. V1-to-V2 permanences live in
`SimulationBridge.cp_connections.data` and change through the existing fused
local potentiation and winner-inactive-depression kernels. Nonlocal V1-to-V2
synapses do not exist.

Every V2 excitatory cell that spikes by a fixed deadline enters the V2 code.
The host does not rank V2 activity or truncate that code to a top-K or first-K
set. A sparse IT population receives the full V2 spike set. IT permanences are
trained with a presynaptic temporal trace over continuous views, following the
Földiák mechanism already implemented in
`research/runners/_emerge50_trace_rule_derisk.py`. The predecessor's
postsynaptic persistence current is removed.

This hierarchy follows the local receptive-field and ventral-stream structure
described in `docs/plans/2026-05-01-cluster-k-visual-cortex-hierarchy.md`, based
on Hubel and Wiesel, Felleman and Van Essen, Tanaka, DiCarlo and Cox, and
Riesenhuber and Poggio.

## Seed and phase lock

- Non-scientific smoke: `222`.
- Calibration, open only as the exact ordered tuple: `503`, `509`.
- Development, locked: `521`, `523`, `541`.
- Held out, locked: `547`, `557`, `563`.

Direct per-seed execution validates membership for internal use. Aggregate and
CLI execution require the complete ordered calibration tuple and reject
partial, duplicate, reordered, development, or held-out seeds. Both
calibration seeds must pass every fixed criterion with one unchanged
configuration. Otherwise calibration is a NO-GO and later phases remain
closed.

## Fixed protocol and controls

The fixed arms are intact, all learning off, V2 learning off, IT trace off,
exact-frame-multiset temporal shuffle, local V2 FS/PV lesion, IT FS lesion, and
receptive-field scramble. The receptive-field scramble preserves the exact
V1-to-V2 synapse count and each hypercolumn's fan-in while destroying spatial
locality. A separate pixel-scramble control uses the intact trained hierarchy.

Labels are available only to the scorer. They do not enter Gabor encoding,
V1/V2/IT spike selection, either learning stage, temporal traces, or inference.

## Fixed validity preconditions

A calibration result is `UNDEFINED` unless the seed partitions are disjoint;
smoke remains outside them; temporal shuffle preserves the exact frame
multiset; labels remain outside encoding and learning; intact V2 has no
nonlocal feedforward synapses; receptive-field scramble preserves synapse
count; V2 uses every cell firing by the deadline; and numeric measurements are
finite.

## Fixed scientific criteria

Every item must pass independently on both calibration seeds:

1. Intact four-way held-transform identity decoding is at least `0.60`, against
   chance `0.25`, and held-to-train cosine margin is at least `0.10`.
2. All-learning-off decoding is at least `0.20` below intact, and neither V2
   nor IT changes any permanence in that arm.
3. V2-learning-off and IT-trace-off decoding are each at least `0.10` below
   intact. Both intact learning stages must change substrate permanences.
4. Exact-frame-multiset temporal shuffle decoding is at least `0.15` below
   intact.
5. V2 local FS/PV lesion raises the V2 fired fraction by at least `0.20` and
   lowers decoding by at least `0.10`.
6. Receptive-field scramble and pixel scramble decoding are each at most
   `0.35` and at least `0.20` below intact.
7. IT FS lesion is a preregistered causal diagnostic. It must increase IT
   activity and reduce selectivity; exact thresholds will not be added after
   seeing calibration outputs. It is reported separately rather than silently
   omitted if its current measurement lacks discriminating power.

## Smoke boundary

Smoke `222` uses reduced images, a 2 by 2 V2 grid, three frames per track, and
one epoch. It verifies only local connectivity, local V2 and IT permanence
changes, zero no-learning changes, a prior-frame contribution through the IT
trace, increased V2 density after local FS lesion, absence of labels from
encoding and learning, and untruncated all-fired V2 readout. Smoke accuracy and
the scientific criteria are not evidence and produce no formal verdict.

## Host boundary and remaining scaffolds

The host still supplies fixed Gabor filters, V1 normalization and
overlap-to-current scaling, fixed V2 receptive-field topology, the IT
presynaptic trace, synthetic track boundaries, spike-deadline readout, and
labels used only for scoring. These are explicit scaffolds. This gate does not
claim natural vision, a complete ventral stream, or a finished biological
implementation.

Calibration failure must be recorded without tuning seeds `503` or `509`,
changing these criteria, or opening later partitions.
