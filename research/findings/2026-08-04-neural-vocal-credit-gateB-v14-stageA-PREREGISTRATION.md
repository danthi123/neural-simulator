---
type: preregistration
status: preregistered-not-executed
date: 2026-08-04
mechanism: neural-vocal-action-credit-v14-explicit-snr-pacemaker-stageA
---

# Gate B v14 Stage A: region-scoped SNr conductance substrate

## Purpose

Stage A tests whether the simulator can represent the complete selected SNr
channel bundle as population-scoped device state without changing default
behavior. It is an implementation gate, not a physiology or selector result.

The machine-readable authority is
`research/specs/v14_snr_conductance_stageA_implementation.json`. The detailed
biological ranges remain in
`2026-08-04-gpi-snr-autonomous-pacemaking-biophysical-fallback-RESEARCH.md` and
cannot be searched during Stage A.

## Frozen lineage and partitions

The pre-change anchor is `08b9e80cb6e1e878aa5e29d92ba74f904102474a`.
V13 remains `TONIC_OUTPUT_NO_GO`; none of its positive physiology has promotion
weight here, and its seed `1031` remains sealed.

Stage A may use only compatibility seeds `193883`, `261805`, `768106`,
`929013`, `736887`, and `366590`. Calibration seed `590297`, replication seeds
`979881`, `651019`, and `950955`, held-out seeds `312588`, `787884`, and
`625835`, and future selector seed `790326` are forbidden until later
successor-specific gates authorize them.

## Frozen implementation boundary

The default-off feature must allocate no bundle arrays. When enabled on an SNr
region in an HH bridge, the bundle contains NALCN-like, persistent sodium,
Cav2.2-like calcium, calcium-state/SK, and optional Ih currents. Maximum
conductances and all live gates are float32 device arrays. Twelve arrays set a
hard ceiling of `48 * total_neurons` persistent bytes for the first active
implementation.

The host may construct parameters, present ordinary stimuli, record outputs,
and score tests. It may not calculate or inject channel current each step.
Unsupported fused fast paths must refuse dispatch rather than omit a current.

## Required evidence

Focused tests must cover current direction, voltage dependence, finite bounded
state, calcium-to-SK coupling, steady-state initialization, region isolation,
independent lesions, default-off equivalence, strict checkpoint behavior,
same-backend continuation, cross-backend behavioral parity, and supported path
parity.

All six compatibility seeds must preserve default-off spike, voltage, recovery,
weight, and checkpoint hashes. Any mismatch is a Stage-A NO-GO.

Only after correctness passes may a seed-free RTX 3090 comparison measure
`500` warm-up and `20,000` timed steps over three repetitions. Candidate and
control run in contemporaneous randomized processes. Default-off must be at
most `1.02x` control; active execution must be at most `1.25x` matching default
HH. No per-step device synchronization is allowed.

## Stop boundary

Missing evidence is `UNDEFINED`. A valid compatibility, state, checkpoint,
backend, or performance failure is a named `NO_GO`. Stage A cannot tune
biological values, open physiology seeds, modify V13, or claim autonomous SNr
pacemaking. A Stage-A GO authorizes only a separate Stage-B preregistration.

