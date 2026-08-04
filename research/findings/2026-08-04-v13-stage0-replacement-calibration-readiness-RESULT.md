---
type: diagnostic-result
status: ready-to-emit-numpy-calibration
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v1
promotion_value: none
---

# V13 Stage-0 replacement calibration readiness

The seed-free readiness path passed for the replacement NumPy calibration. It
did not create a command envelope, run the simulator, or consume a scientific
seed. This is an operations result, not evidence that the tonic-output
mechanism works.

## Frozen inputs

- Candidate source revision:
  `d091fa6692bdf8115c8073af6fd31fc9626921a8`.
- Candidate source manifest: 54 files with manifest and tree SHA-256
  `e42d9a255c68566d1e84011a265b9c148dc4eb8910816b6af44dbe8edc2edbb7`.
- Controller configuration: schema `v13-stage0-controller-config-v2`, frozen
  self-digest
  `044e0ac184f5cd6b1a9657175d68063b2a37b0756558ba96051dbd3de930556c`.
- Configuration file SHA-256:
  `13b0f8332754fada3e2fed76b0f7adf57798fdbed8496bcf2be3fd928518077d`.
- Readiness result: `ready=true`, `backend=numpy`,
  `command_emitted=false`, `execution=not_executed`.

The controller revalidated the strict arithmetic replay v2 and deterministic
compatibility prerequisites before reporting readiness.

## Independent compute checks

The focused local controller, manifest, runner, receipt, replay, lifecycle, and
pool tests passed: 160 tests on NumPy.

The same detached source revision was then deployed to fresh, revision-specific
directories on `pool40`, `pool41`, and `pool42`. Each node independently
verified:

- deployed source-manifest SHA-256
  `61300a01906258831121e93671061930e3ef1186ec892dd1a58b8b594b1f1ca8`;
- source-ancestry SHA-256
  `a9f0908caf5e73ea5841696a4b2f8c814f4ad19db4824f6c5f2d4f6f9cbf0858`;
- exact archive ancestry to the candidate revision;
- no excluded dirty source paths; and
- 100 passing read-only, seedless regression checks.

An earlier check in the legacy shared pool directory exposed stale unrelated
files and an incomplete root-document deployment. That result is not used.
Commits `75cf5a5df`, `d2fe609c3`, and `2c51900a3` corrected root-document
binding, added revision-isolated deployment, created nested destinations, and
made provisioning fail fast. The isolated rerun above is the accepted check.

## Allowed next action

The controller may create the single NumPy calibration command envelope. CuPy
calibration remains blocked until the NumPy artifact and receipt are complete,
validated, and sealed. Replication, held-out, performance, and Stage 1 remain
blocked by their preregistered prerequisites.
