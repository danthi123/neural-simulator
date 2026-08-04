---
type: finding
status: undefined
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v1
promotion_value: none
---

# V13 Stage-0 process correction v1 is undefined

The registered replacement NumPy calibration ran once on the frozen candidate
source and produced a complete five-point observation. The `100 pA` point
passed the NumPy physiology checks. This remains useful diagnostic evidence,
but it cannot unlock CuPy or promote the mechanism under process correction v1.

## Why the chain stopped

The runner and controller used the word `sha256` for two different, valid
digests of the same compatibility JSON:

- exact file-byte SHA-256:
  `eec19a34d4112fb6660a097ec3e24af08399e4de5ca4d0d32bc03b4c7be18971`;
- compact sorted canonical-JSON SHA-256:
  `777e1ac590847beab2bffbb71b4a9d584021f4efc7f2f13529153b6f3faa2097`.

The frozen config correctly bound the first value. The scientific runner
consistently recorded the second. The controller later compared them as if
they were the same domain and failed closed at CuPy readiness.

The evidence manifest also sealed the scientific artifact, command envelope,
and execution receipt, but not the runner-generated provenance sidecar. The
sidecar agrees with those records and is preserved byte-for-byte, but adding it
to the original create-only seal after observing the result would weaken the
registered contract.

These are evidence-contract defects, not a negative result for the tonic-output
mechanism. The NumPy artifact, receipt, command, manifest, and sidecar are
preserved under
`research/findings/raw/v13_tonic_output_stage0_process_correction_v1/`.

The measured artifact is
`research/findings/raw/v13_tonic_output_stage0_process_correction_v1/calibration-numpy.json`.
Its execution receipt is
`research/findings/raw/v13_tonic_output_stage0_process_correction_v1/receipts/calibration-numpy.json`,
and its original evidence seal is
`research/findings/raw/v13_tonic_output_stage0_process_correction_v1/manifests/calibration-numpy.json`.

## Stop decision

- Replacement calibration seed `840860` is consumed and must never be rerun.
- The v1 NumPy observation has diagnostic value only.
- No v1 CuPy command was emitted or executed.
- Held-out seed `1021` and Stage-1 seed `1031` remain sealed.
- Process correction v2 must be locked before another scientific command. It
  must use explicit byte and canonical digest fields, seal the provenance
  sidecar, and derive a fresh calibration partition mechanically.
