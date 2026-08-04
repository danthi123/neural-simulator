---
type: finding
status: partial-physiology-go-performance-not-run
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v6
promotion_value: none
seed-waiver: preregistered calibration, replication, and held-out partitions; Stage-0 engineering and final gates remain incomplete
instrument: matched NumPy/CuPy tonic physiology, lesion, inhibitory-response, checkpoint, source-off, and immutable evidence controls
---

# V13 Stage-0 process correction v6 completes physiology

V6 completed and sealed the full preregistered physiology sequence. NumPy and
the RTX 3090 again selected `100 pA` on fresh calibration seed `754571`. Both
backends then earned replication GO on fresh seed `890220` and held-out GO on
blind seed `1021`. Every registered population, physiology, intrinsic-lesion,
inhibitory-response, checkpoint, source-off, and immutability check passed.

The sealed calibration selection is
research/findings/raw/v13_tonic_output_stage0_process_correction_v6/calibration-selection.json.
Its artifact SHA-256 is
`30818e1cb5165ba56b67ed2cb55db36fc62b8e70cc286de5eda8113a4c86e0d2`.

The replication artifacts are
research/findings/raw/v13_tonic_output_stage0_process_correction_v6/replication-numpy.json
and
research/findings/raw/v13_tonic_output_stage0_process_correction_v6/replication-cupy.json.
Their artifact SHA-256 values are
`ac7b5c3f5b3ddce3e3526d93966bd33e2b14d680351b96f18e14d06521b274e4`
and
`424a8c5d9bb07c6204fddab1cce38f1f543ae468b14b66617344b362c71d0dce`.

The held-out artifacts are
research/findings/raw/v13_tonic_output_stage0_process_correction_v6/held-out-cupy.json
and
research/findings/raw/v13_tonic_output_stage0_process_correction_v6/held-out-numpy.json.
Their artifact SHA-256 values are
`583e1424995ec14c6c5655067724cda61ded1f7873811bdb7069da87b1dd374a`
and
`ab28acb25535211f91a8b594e3e98e32e578982c8c7fbea267b6c12383fe9521`.

## Engineering boundary

The next locked step was a legacy RTX 3090 performance baseline. The frozen V6
configuration paired historical revision `8994b5102` with the current V13
measurement runner, but that runner does not exist at the historical revision.
The controller therefore rejected the source/runner pair before it could emit a
command envelope. A generic current-root receipt would also have been unsuitable
downstream, but it was not the first failure.

No performance command was emitted or executed. Running the envelope through
the current wrapper would benchmark the current source while claiming the old
revision, so progression stopped before that invalid transition.

## Disposition

- Calibration seed `754571`, replication seed `890220`, and held-out seed
  `1021` are consumed.
- The sealed physiology evidence is positive but does not by itself earn
  Stage-0 promotion.
- No performance artifact or final merge exists.
- Stage-1 seed `1031` remains sealed.
- A process-only continuation must use the audited historical source package,
  add only the already accepted measurement-runner overlay, issue a
  package-specific execution receipt, and permit the sealed V6 physiology
  evidence to unlock only the remaining engineering gates.

V6 has no Stage-0 promotion value until performance and final evidence gates
are completed.
