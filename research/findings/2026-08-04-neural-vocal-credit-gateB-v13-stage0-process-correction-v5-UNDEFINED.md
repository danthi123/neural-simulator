---
type: finding
status: undefined
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v5
promotion_value: none
seed-waiver: preregistered calibration and replication partitions; no capability GO
instrument: matched NumPy/CuPy replication arms with intact, lesion, inhibitory-response, checkpoint, and source-off controls
---

# V13 Stage-0 process correction v5 is undefined

V5 completed and sealed cross-backend calibration at `100 pA`. NumPy then
completed replication with every registered check passing. Its artifact,
receipt, provenance sidecar, and manifest are sealed.

CuPy also completed every registered measurement. It produced a scientifically
valid `REPLICATION_NO_GO` with no undefined reasons. The only failed check was
the inhibitory response: firing fell from `63.7 Hz` to `12.25 Hz`, above the
registered ceiling of ten percent of baseline (`6.37 Hz`). The source-off arm
remained at `63.625 Hz`; all other physiology, lesion, checkpoint, source-off,
immutability, and evidence checks passed.

## Why the chain stopped

The runner wrote the complete CuPy artifact and provenance sidecar, then mapped
the valid negative verdict to process exit code `1`. The execution wrapper
therefore classified the command as failed and correctly refused to create a
success receipt. Without that receipt, the negative artifact cannot receive an
evidence manifest under the locked V5 contract.

This is an evidence-contract defect. It does not erase or reverse the negative
CuPy observation, but that observation remains unsealable diagnostic evidence
and cannot support promotion or a formal cross-backend no-go.

The sealed calibration selection is
research/findings/raw/v13_tonic_output_stage0_process_correction_v5/calibration-selection.json
with artifact SHA-256
`9aefae27fc96ed6de28a551d6198337912cb8418adc1c9b132be62b7db9d8865`.
Its manifest SHA-256 is
`5ba9548e1d29eebe6b72c74c47a018600edb9d5a0cc82a8abea41efd2d00a892`.

The sealed NumPy replication artifact is
research/findings/raw/v13_tonic_output_stage0_process_correction_v5/replication-numpy.json
with artifact SHA-256
`9f6983d4a2e982df12b9ac047f380d452a62a3c5a0edfe3806cd9ae993944793`.
Its manifest SHA-256 is
`706ec4b0ed12a81fcf52d5ce3a426768382a0a396fa5adff9968b0ef9dd30179`.

The unsealed CuPy negative artifact is
research/findings/raw/v13_tonic_output_stage0_process_correction_v5/replication-cupy.json
with artifact SHA-256
`e481257a832fd990a55ee943ef20d39967661aefe4877e12cd8f658c7c0f9636`.
Its preserved provenance sidecar SHA-256 is
`64d3aaf0a821216c10059cf797c4b727ecf66c035e9e55fcb7aa23038127de30`.
No CuPy replication receipt or manifest exists.

## Disposition

- Calibration seed `216274` and replication seed `401461` are consumed.
- No held-out, performance, or final command was emitted.
- Held-out seed `1021` and Stage-1 seed `1031` remain sealed.
- V6 must exit successfully for both valid `GO` and valid `NO-GO` artifacts,
  while retaining nonzero exits for `UNDEFINED` evidence and exceptions.
- V6 must use fresh calibration and replication seeds derived without V5
  result data.

V5 has no Stage-0 promotion value.
