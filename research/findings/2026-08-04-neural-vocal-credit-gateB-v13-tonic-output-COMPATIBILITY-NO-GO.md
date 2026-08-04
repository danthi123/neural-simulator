---
type: finding
status: complete
date: 2026-08-04
verdict: COMPATIBILITY_NO_GO
mechanism: gateB-v13-tonic-output-substrate
runner: research/runners/_vocal_action_credit_gate_v13_tonic_output.py
artifacts:
  - research/findings/raw/_vocal_action_credit_gateB_v13_compatibility_earned_numpy.json
  - research/findings/raw/_vocal_action_credit_gateB_v13_compatibility_earned_numpy.json.prov.json
  - research/findings/raw/_vocal_action_credit_gateB_v13_compatibility_earned_cupy.json
  - research/findings/raw/_vocal_action_credit_gateB_v13_compatibility_earned_cupy.json.prov.json
---

# V13 tonic-output substrate stops at CuPy compatibility

<!--derived-->
**Verdict: COMPATIBILITY_NO_GO.** Source `da0ea65e6` reproduced every locked
NumPy default-off fingerprint, but the RTX 3090 run did not reproduce the
locked final CuPy membrane, recovery, or inhibitory-conductance hashes. The
CuPy spike raster, excitatory conductance, weights, external current, and
default-off intrinsic-current state were exact. The preregistration says any
compatibility mismatch blocks scientific execution until explained and
separately preregistered, so calibration, replication, held-out testing, and
performance promotion did not run.

Seed-waiver: seed `271828` is the preregistered compatibility seed, and its
failure requires an immediate stop before calibration rather than replication
on capability seeds. The first invocation omitted the mandatory artifact
precondition block and was therefore an instrument failure; the runner-only
repair was committed before this one repeated, earned measurement.

Instrument: the sealed runner reconstructed the Gate A v2 network at seed
`271828`, ran `300` fixed steps on each explicitly selected backend, and hashed
the complete spike raster and final state arrays. Automatic provenance records
the source, backend, device, command, environment, and artifacts.

Artifacts:
`research/findings/raw/_vocal_action_credit_gateB_v13_compatibility_earned_numpy.json`
and
`research/findings/raw/_vocal_action_credit_gateB_v13_compatibility_earned_cupy.json`.

## What passed

NumPy matched all seven locked hashes exactly in `0.085 s`. <!--derived--> This includes the
complete raster, `v`, `u`, `g_e`, `g_i`, weights, and external current. The new
optional intrinsic-current vector was absent, as required when every region
uses the default value.

CuPy matched the complete locked spike raster exactly:
`690867e2c44ac456ee1f3a0cb8db9addeef8448753170b587561767c6e51ec2b`.
It also matched `g_e`, weights, and external current exactly, and the optional
intrinsic-current vector was absent. The run completed on the RTX 3090 in
`0.482 s`. <!--derived-->

These passes are useful evidence that the new default-off field did not alter
topology, stimuli, immutable weights, or the observed spike decisions in this
test. They do not override the locked exact-state requirement.

## Decisive failure

The CuPy final-state hashes differed from the preregistered values:

| Array | Locked hash | Observed hash |
|---|---|---|
| `v` | `d1706d17f1a1136a57672546fb643e10f991476c32098ae1906b3b3ec88683df` | `33b46c9205febe7a51ce230a83885b085bc9793ac0ce8ec6a59b1a99acddb4ce` |
| `u` | `f319dbcfcb1d09f983ad86ddf912484820d3fce94a2b93e135d73b0219c96317` | `9e93ccd32ad453c0917033fdcff4dfd3a613e3c3ee3fb28f4b36492b511bd8a6` |
| `g_i` | `90b3c1c3825eba353c19bbf18017254a498007bdb8cf2cbab4e59acacd61f305` | `e90cf792505bfb06e51516937a76672428c204eb7a2e389bd5fe73111ac99448` |

The unchanged raster alongside divergent floating-point state is consistent
with the repository's documented nondeterministic CuPy transpose sparse-matrix
accumulation, but this finding does not treat that explanation as a waiver.
The locked gate asked for exact hashes and therefore failed.

## Decision

Stop the V13 scientific sequence before calibration. Do not consume seeds
`1013`, `1019`, or `1021`, and do not open the continuous selector.

A separate preregistration may test a deterministic sparse-matvec execution
path or replace impossible byte equality with a fixed numerical criterion that
still locks raster, topology, stimuli, weights, and the default-off feature
state exactly. That correction must establish its controls and thresholds
before new measurements; this failed result must remain visible.
