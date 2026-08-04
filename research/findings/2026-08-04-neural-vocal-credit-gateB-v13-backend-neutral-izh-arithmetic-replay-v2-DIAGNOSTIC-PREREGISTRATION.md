---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-backend-neutral-izh-arithmetic-replay-v2
spec: research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2.json
promotion_value: none
artifacts:
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/comparison.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/evidence-manifest.json
---

# V13 strict-arithmetic matched-state replay v2

**Status:** protocol only; no replay or seed was run while preparing it

## Question

After correcting the opt-in GPU strict-arithmetic path so it preserves float32
subnormal values, do NumPy and CuPy remain byte-exact for `v`, `u`, and spikes
through the full 1,200-step V13 matched-state replay?

V1 found its first mismatch at step 1,114 when the GPU path flushed a subnormal
recovery-variable update to zero. V2 tests the corrected implementation. The v1
artifacts remain separate, immutable evidence of that failed replay.

## Frozen input and source

Both cells must consume the same completed NumPy-origin V13 transplant bundle
and validate its artifact, aggregate, receipts, source manifest, and locked
transplant specification against the exact digests in the v2 specification.

V2 does not hardcode a future execution revision. Immediately before execution,
the evidence tool must freeze every tracked simulator Python file and every v2
authority file from one committed revision. The NumPy cell, CuPy cell, comparison,
and final evidence manifest must all bind that same source snapshot. A checkout
with uncommitted differences in any bound source file cannot be frozen.

Bridge allocation may use random initialization only before restoration. Every
allocated simulator array and the complete CSR connection matrix must then be
overwritten and verified against the sealed bundle. From that boundary through
step 1,200, guarded random APIs must fail on any call. V2 introduces no seed.

## Locked protocol

1. Write only under
   `research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2`.
2. Run one create-only NumPy cell and one create-only CuPy cell from the same
   committed source manifest and completed transplant bundle.
3. Require `backend_neutral_izh_arithmetic=True`, preserve its default value of
   `False`, and require the incompatible step megakernel to remain disabled.
4. Replay 500 baseline steps, 200 inhibition steps, and 500 release steps with
   all runtime random processes disabled.
5. Capture all 1,200 rows of `v`, `u`, and spikes, including per-row digests.
6. Validate the exact receipted argv for both cells and for comparison.
7. Compare shape, dtype, every row digest, and complete array bytes. No numerical
   tolerance or fallback comparison is permitted.

## Acceptance

The diagnostic passes only when all 1,200 rows of all three required trajectories
are byte-exact, restoration is exact, no measured-replay random call occurs, and
all successful receipts bind the canonical commands and one source snapshot.

Any mismatch is a diagnostic failure and must identify the first trajectory,
step, and cell. V2 may not overwrite an existing artifact, receipt, command
envelope, source manifest, comparison, or evidence manifest.

## Non-claims

- This diagnostic has no scientific promotion value and no scientific verdict.
- It does not validate initialization parity or mechanisms outside this replay.
- It does not release calibration seed `840860` or any held-out seed.
- It does not tune parameters or modify completed transplant or v1 evidence.
- Passing would establish exactness only for this locked state, schedule, source,
  and the required `v`, `u`, and spike trajectories.

## Authority

- Corrected arithmetic implementation: the committed source manifest frozen at
  execution, including `sim/kernels.py`
- V1 failure result:
  `research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-izh-arithmetic-replay-v1-DIAGNOSTIC-RESULT.md`
- V1 comparison and evidence manifest:
  `research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/comparison.json`
  and
  `research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/evidence-manifest.json`
- Completed transplant result:
  `research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-state-transplant-DIAGNOSTIC-RESULT.md`
- V2 specification:
  `research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2.json`
