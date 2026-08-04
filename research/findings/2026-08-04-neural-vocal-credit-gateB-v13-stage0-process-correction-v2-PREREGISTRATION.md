---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v2
spec: research/specs/v13_tonic_output_stage0_process_correction_v2.json
promotion_value: none
supersedes_process_only: gateB-v13-stage0-process-correction-v1
---

# V13 Stage-0 process correction v2

**Status:** locked before v2 execution; no v2 command has been emitted and no
v2 seed has been run

## Why v1 is undefined

Stage-0 process correction v1 is `UNDEFINED` because its evidence contract has
two independent defects:

1. The compatibility file's exact byte SHA-256 and its canonical-JSON SHA-256
   were stored under one ambiguous field. The frozen controller configuration
   recorded the byte digest, while the scientific runner recorded the
   canonical-JSON digest. CuPy readiness therefore cannot validate the sealed
   NumPy artifact under the v1 contract.
2. Artifact manifest v1 sealed the result, command envelope, and execution
   receipt, but did not seal the result's provenance sidecar. The sidecar exists
   and is consistent with the recorded run, but adding it to the original seal
   after observation would weaken the preregistered evidence chain.

This is an evidence-contract failure, not `CALIBRATION_NO_GO` and not a
rejection of the tonic-output mechanism. No CuPy calibration may be run from
the v1 chain, and no v1 result may unlock a later stage.

## Preserved diagnostic observation

The v1 NumPy artifact is preserved byte-for-byte at
`research/findings/raw/v13_tonic_output_stage0_process_correction_v1/calibration-numpy.json`.
It records that `100 pA` was the only passing point in the locked five-point
ladder for NumPy seed `840860`. The artifact remains useful for diagnosis and
controller testing, but it has no selection, promotion, or seed-choice weight
under v2. In particular, v2 does not assume that `100 pA` will pass and does not
prefer it over any other ladder point.

The original artifact, sidecar, command, receipt, and v1 manifest must remain
unchanged. Their exact digests are recorded in the v2 machine-readable spec.

## Fresh partitions without result-based selection

Seeds `1013`, `1019`, and `840860` are permanently consumed and forbidden.
The unexecuted v1 replication seed `687979` is retired and forbidden under v2,
but is not mislabeled as consumed. Held-out seed `1021` remains sealed, and
Stage-1 seed `1031` remains unavailable.

Fresh calibration and replication seeds are derived mechanically by the rule
in the v2 spec. For each role, form this exact ASCII string:

`V13_STAGE0_PROCESS_CORRECTION_V2|d091fa6692bdf8115c8073af6fd31fc9626921a8|role=<role>|prior_seed=<v1 partition seed>`

Take SHA-256, interpret the first 12 hexadecimal digits as an integer, and
calculate `100000 + (integer mod 900000)`. The source anchor is the frozen
candidate revision committed at `2026-08-04T05:45:13-04:00`, before the v1
NumPy observation was committed. The only derivation inputs are the new
namespace, this pre-observation source anchor, the role, and the corresponding
v1 partition seed. No measured current, pass/fail result, raster, state hash,
or manually tested seed is an input.

The machine-readable v2 spec is the sole seed-value authority. A controller
must recompute both values and reject substitutions, collisions with forbidden
seeds, and any different derivation material.

## Unchanged scientific contract

The calibration ladder remains exactly `75`, `100`, `125`, `150`, and `175 pA`
in ascending order. Every point must use a fresh brain. No interpolation,
extra point, changed threshold, early stop, or carry-forward of the v1 `100 pA`
observation is allowed.

V2 changes evidence handling and partitions only. It does not change the V13
tonic-output mechanism, physiology limits, topology, lesions, inhibitory
response, checkpoint gate, compatibility outcome, performance limits, held-out
criteria, or final promotion criteria in the base preregistration and spec.
The previously sealed strict 1,200-step CPU/GPU arithmetic replay v2 remains a
required prerequisite and is bound unchanged in the v2 machine-readable spec.

## Separate compatibility digests

The compatibility binding has two explicitly named and independently checked
digests:

- `file_sha256` binds the exact compatibility file bytes.
- `canonical_json_sha256` binds the parsed JSON serialized with sorted keys,
  compact separators, UTF-8 encoding, and the existing Python JSON defaults.

Both digests must be computed from the same regular-file byte buffer. The file
must be checked before and after parsing so replacement during validation fails
closed. A formatting-only change fails `file_sha256`; a semantic change fails
`canonical_json_sha256`. A path, symlink, or digest from one domain may never
substitute for the other. Scientific artifacts must record both values by
their full names.

## Artifact manifest v2

Every scientific cell must be sealed by a create-only
`v13-stage0-artifact-manifest-v2`. It must bind the exact bytes and paths of:

- the scientific artifact;
- the artifact's `.prov.json` provenance sidecar;
- the registered command envelope;
- the execution receipt;
- the frozen controller configuration and v2 process-correction spec;
- the complete candidate source manifest and source revision; and
- both compatibility digests and the named canonicalization algorithm.

The manifest must verify that artifact, sidecar, command, receipt, source,
backend, run identity, arguments, and destination agree. It must reject missing
or changed sidecars, symlinks, pre-existing outputs, unlisted files, source
drift, backend drift, and any digest mismatch. The sidecar must be sealed at the
same validation step as the artifact; it cannot be supplemented later.

V1 manifests are diagnostic inputs only and can never satisfy this v2
requirement.

## Exact execution order

The v2 controller must enforce this order without bypass or rerun:

1. Run all five calibration points on NumPy with the fresh v2 calibration seed.
2. Validate and seal the NumPy artifact and sidecar with manifest v2.
3. Only then run the same five points on CuPy with the same v2 calibration seed.
4. Validate and seal the CuPy artifact and sidecar with manifest v2.
5. Select the lowest ladder point that independently passes on both backends.
   If no common point passes, stop with `CALIBRATION_NO_GO`.
6. Run replication at only the selected point on NumPy and CuPy with the fresh
   v2 replication seed. These two cells may run concurrently only after the
   calibration selection is sealed.
7. After both replication GO artifacts are sealed, run held-out CuPy and then
   held-out NumPy with the still-sealed original held-out seed.

The old-source performance baseline and candidate performance cell remain
serial and must use their separately frozen source and environment packages.

## Stop rules and promotion boundary

Any v1 promotion attempt, forbidden or retired seed, changed derivation input,
changed ladder, digest-domain substitution, missing sidecar seal, source or
receipt mismatch, order violation, or scored rerun makes the affected evidence
`UNDEFINED` and stops progression. Readiness may inspect prerequisites but may
not emit a scientific command, create an artifact, or execute a seed.

This correction has no scientific promotion value. Only a fully validated v2
NumPy/CuPy calibration pair may select a current; only two v2 replication GO
artifacts may unlock held-out; and only the unchanged final merge may earn
`TONIC_OUTPUT_GO`. Nothing here claims action selection, learning, speech, or
conversation.
