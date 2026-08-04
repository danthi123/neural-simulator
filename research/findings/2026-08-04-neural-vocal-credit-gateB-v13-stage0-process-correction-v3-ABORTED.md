---
type: finding
status: aborted-before-measurement
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v3
promotion_value: none
---

# V13 Stage-0 process correction v3 aborted before measurement

The registered NumPy calibration command was emitted and passed the external
controller's seed-free readiness checks. The execution wrapper started the
scientific runner, but the runner rejected the v3 process-spec path because it
still hardcoded the v2 authority.

The frozen command is preserved at
research/findings/raw/v13_tonic_output_stage0_process_correction_v3/commands/calibration-numpy.json.
The frozen controller config is
research/specs/v13_stage0_controller_config_v5.json.

The runner failed while loading the process-correction spec, before reading the
registered partition, constructing a brain, entering the current ladder, or
writing output. No scientific artifact, provenance sidecar, or success receipt
exists. Calibration seed `577995` and replication seed `578403` were therefore
never run, but both are retired from later chains to keep the attempt boundary
unambiguous. Held-out seed `1021` and Stage-1 seed `1031` remain sealed.

This is a pre-measurement orchestration failure. It says nothing about the
tonic-output mechanism and has no diagnostic or promotion value. V3 must not be
retried after changing source. V4 must bind the runner and external controller
to the same process authority before another command is emitted.
