---
type: finding
status: undefined
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v2
promotion_value: none
---

# V13 Stage-0 process correction v2 is undefined

The registered replacement calibration ran once on NumPy and once on CuPy
against candidate source `f41575539536809158736cd62ab42ab2292cf20c`.
Both backends found `100 pA` to be the only passing calibration point. The
seed-free merge then observed `CALIBRATION_GO` and selected `100 pA`.

These observations are useful diagnostic evidence, but the merge cannot unlock
replication or promote the mechanism under process correction v2.

## Why the chain stopped

The controller's frozen merge envelope omitted `SIM_BACKEND`, while the
manifest contract and the execution receipt required
`SIM_BACKEND=numpy`. The merge command therefore executed successfully but its
artifact could not receive a valid evidence manifest. This is an orchestration
contract defect, not a negative result for the tonic-output mechanism.

The defect also existed in the unexecuted final Stage-0 merge emitter. Both
emitters now obtain their environment from the same policy used by manifest
validation, with tests asserting the environments in the actual emitted
commands.

The valid per-backend calibration evidence is preserved byte-for-byte. Its
primary NumPy artifact is
research/findings/raw/v13_tonic_output_stage0_process_correction_v2/calibration-numpy.json,
and the corresponding CuPy artifact is
research/findings/raw/v13_tonic_output_stage0_process_correction_v2/calibration-cupy.json.

- NumPy artifact SHA-256:
  `247ee5c352d4c1783c22f0aee9d3938d80cde1284a2010fc79bb414ea139adaa`;
- NumPy provenance sidecar SHA-256:
  `9e8c90fdef690d04dfece4c83ede29a2d7b2cc786bd7bd6744217c9777ef90ee`;
- NumPy manifest seal:
  `516c4c5ff7fb9ff918ad0d815f7c72c58c70a819686b9d085798a2243914c0ac`;
- CuPy artifact SHA-256:
  `0dc041797cdb23873f8153d194f486d56336937373a53ab67610e78ff12d0845`;
- CuPy provenance sidecar SHA-256:
  `fb74a211205582b22a5f685996c2fb045968aaa24a20e480b7a269371669a65b`;
- CuPy manifest seal:
  `cf9b6b2774ebeac4f51cd5fc730c562e2c11eaab0af9291a0083b5b33c7e240a`.

The unsealed merge artifact and receipt are also preserved as diagnostics. The
merge artifact SHA-256 is
`7bb6709baf49c17ac12c0068c7800a0b1c193efc64324c09a8d39d1b5dbcc8a3`.
It has no promotion value and must not be treated as a calibration-selection
manifest.

## Stop decision

- Calibration seed `645424` is consumed and must never be rerun.
- Replication seed `638726` was never executed and is retired for this chain.
- No replication or held-out command was emitted or executed.
- Held-out seed `1021` and Stage-1 seed `1031` remain sealed.
- Process correction v3 must be locked before another scientific command. It
  must derive fresh calibration and replication partitions mechanically and
  bind the actual merge envelopes to their required NumPy environment.
