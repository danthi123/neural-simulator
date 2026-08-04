---
type: preregistration
status: locked
date: 2026-08-04
mechanism: gateB-v13-tonic-output-process-correction
supersedes_gate_only: V13 Stage-0 execution order and consumed calibration/replication seeds
does_not_supersede: V13 mechanism, ladder, physiology, lesion, checkpoint, compatibility, held-out, performance, or promotion criteria
---

# V13 Stage-0 process-order correction

This correction replaces invalid orchestration, not biology or acceptance
criteria. Calibration seed `1013` and replication seed `1019` are consumed and
must not be rerun. Their measurements remain archived but carry no promotion
weight. Held-out seed `1021` and Stage-1 seed `1031` remain untouched.

## Mechanically derived replacement seeds

Replacement seeds are derived without testing candidates. For each role, form
the exact ASCII material below, take SHA-256, interpret the first 12 hexadecimal
digits as an integer, and calculate `100000 + (integer mod 900000)`.

| role | derivation material | prefix | seed |
|---|---|---|---:|
| calibration | `V13_STAGE0_PROCESS_CORRECTION_V1\|b3d57494b7dd7d99d5e91088489da44d89a85bf3\|role=calibration\|original_seed=1013` | `496020e49afc` | `840860` |
| replication | `V13_STAGE0_PROCESS_CORRECTION_V1\|b3d57494b7dd7d99d5e91088489da44d89a85bf3\|role=replication\|original_seed=1019` | `c6ec423baa4b` | `687979` |

These are the only permitted replacements. They were fixed before either was
executed. No other seed may be substituted after observing a result.

## Frozen scientific contract

The original five-point ladder, all physiology limits, lesion, inhibitory
response, checkpoint, compatibility, topology, provenance, and performance
criteria remain byte-for-byte unchanged. The selected current must be derived
again; the diagnostic `100 pA` observation is not carried forward as a choice.

Before execution, commit the controller, its tests, this correction, and a
complete source manifest. From that source onward, no simulator, scientific
runner, spec, threshold, seed, or controller input may change during Stage 0.
Any such change makes later evidence `UNDEFINED`.

## Fail-closed controller

Registered commands may be produced only by the committed external controller.
It must never execute a later stage merely because a path exists. Every input
artifact and sidecar must be canonical, schema-valid, source-matched, and bound
by SHA-256 in a create-only manifest.

Calibration is strictly serial:

1. run NumPy at seed `840860`;
2. validate and seal its complete artifact, sidecar, source identity, and
   compatibility binding;
3. only then permit CuPy at the same seed; and
4. merge only the two validated artifacts from the identical source.

If there is no common passing ladder point, stop with `CALIBRATION_NO_GO`.

Replication uses seed `687979` and the newly sealed calibration selection.
NumPy and CuPy may run concurrently because the original protocol imposes no
order between them. Each must carry the selection digest, selected current,
source identity, compatibility digest, complete manifest, and named component
verdicts. Any failure stops Stage 0. Neither backend may be rerun after a scored
step.

Held-out seed `1021` remains serial: validate and seal CuPy first, then permit
NumPy. Both require two earned replication GO artifacts and the same selection,
source, current, and compatibility evidence. Seed `1031` remains unavailable.

## Matched controls and final evidence

Fresh brains in intact/lesion and source-on/source-off pairs must report exact
topology, immutable neuron-parameter, initial-state, weight, intrinsic-current,
and stimulus hashes. Fields intentionally changed by the intervention must be
listed explicitly. A mismatch outside that list is `UNDEFINED`, not a no-go.

The old-source performance baseline must run from exact revision `8994b5102`
plus only the already accepted deterministic instrumentation where required.
The candidate performance command must consume a manifest binding that source,
runner, GPU identity, environment, and baseline artifact digest. Baseline and
candidate GPU runs are serial.

Final merge requires the compatibility GO, corrected calibration selection,
both corrected replication GO artifacts, both held-out GO artifacts, and the
performance verdict. All dependencies must share the frozen source and carry
matching digests. Named checkpoint, lesion, inhibition, physiology, and
performance failures must remain visible rather than collapse into a generic
result.

Only a fully earned `TONIC_OUTPUT_GO` opens the separate Stage-1 selector work.
