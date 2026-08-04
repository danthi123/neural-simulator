---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-backend-neutral-izh-arithmetic-replay
spec: research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic.json
promotion_value: none
---

# V13 strict-arithmetic matched-state replay

**Status:** protocol only; the full replay has not run

## Question

Does the opt-in strict Izhikevich arithmetic path keep NumPy and CuPy exactly
aligned for `v`, `u`, and spikes through the completed V13 schedule rather than
only at the already-localized first divergent operations?

## Frozen input

Both cells consume the completed NumPy-origin V13 transplant bundle. The new
runner must validate the bundle, its original success receipt, the completed
aggregate, and the original transplant specification against the exact paths
and SHA-256 digests locked in the new specification. It may read those files
but must not rewrite or reinterpret them as new scientific evidence.

The bridge allocation performed before restoration may initialize temporary
state with RNG. Every allocated simulator array and the full CSR connection
matrix must then be overwritten byte-exactly from the sealed bundle. The
measured replay starts only after that verification. From that boundary through
step 1,200, guarded RNG APIs must fail on any call. No replay seed is introduced.

## Locked protocol

1. Run one create-only NumPy cell and one create-only CuPy cell from the same
   committed source manifest and exact transplant bundle.
2. Restore every allocated `cp_*` array and CSR connection byte-exactly before
   the first step.
3. Confirm `backend_neutral_izh_arithmetic` is `True`, its declared default is
   still `False`, and the incompatible step megakernel remains disabled.
4. Replay the original 500-step baseline, 200-step inhibition, and 500-step
   release schedule with all runtime random processes disabled.
5. Capture all 1,200 rows of `v`, `u`, and spikes plus per-row SHA-256 digests.
6. Compare the receipted cells byte for byte. Shape or dtype disagreement is a
   failure; no numerical tolerance is allowed.

## Acceptance

The diagnostic passes only if all 1,200 `v` rows, all 1,200 `u` rows, and all
1,200 spike rows are byte-exact across NumPy and CuPy, no RNG API is called
during measured replay, restoration is exact, and both run artifacts and the
comparison are bound to the same committed source manifest by success receipts.

Any later divergence is a diagnostic failure and must report the first array,
step, and differing cell. It is not acceptable to widen a tolerance.

## Non-claims

- This diagnostic has no scientific promotion value and produces no scientific
  verdict.
- It does not validate the separate backend-neutral initialization correction.
- It does not release V13 calibration seed `840860` or any held-out seed.
- It does not prove byte parity for simulator mechanisms outside this locked
  read-only V13 path.
- No full replay or scientific seed was executed while creating this protocol.

## Authority

- Arithmetic design: `research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-arithmetic-correction-DIAGNOSTIC-PREREGISTRATION.md`
- Completed transplant result: `research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-state-transplant-DIAGNOSTIC-RESULT.md`
- Locked protocol: `research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic.json`
