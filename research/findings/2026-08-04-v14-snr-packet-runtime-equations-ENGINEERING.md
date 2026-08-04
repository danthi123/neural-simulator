---
type: research-finding
status: engineering-substrate
date: 2026-08-04
mechanism: v14-snr-authenticated-packet-runtime-equations
claim_check: synthesis
---

# V14 authenticated SNr packet runtime equations

## Boundary

The authenticated packet can now be converted into a deeply immutable runtime
record, and a separate fused kernel can evaluate the complete SNr channel
equation surface. This is an engineering result. It does not establish an
adult-mouse SNr parameter set or authorize a physiology verdict.

The source interpretation remains the one recorded in
`research/findings/2026-08-04-v14-stageB-executable-parameter-evidence-RESEARCH.md`:
preparation-matched targets, transferred channel measurements, and model priors
remain distinct evidence classes. The non-executable target catalog remains
`research/specs/v14_snr_stageB_target_packet.json`.

## Implemented

- Every one of the 69 authenticated packet leaves is parsed exactly once and
  retained in immutable raw and typed mappings.
- `nS/pF` densities are converted to `mS/cm2` through packet capacitance.
- Q10 factors are derived from each mechanism's tagged reference temperature.
- Calcium current density is converted to `uM/ms` using membrane area,
  accessible volume, current fraction, calcium valence, and Faraday's constant.
- The new fused packet kernel evaluates Phillips-form NaP kinetics, Cav2.2,
  HCN, physical calcium decay/influx, and direction-dependent SK kinetics.
- A single full-population parameter matrix supports packet, legacy SNr, and
  control regions in one fused launch. Packet regions also receive their
  authenticated fast-HH values, reversals, initial state, and per-gate Q10
  factors after any legacy heterogeneity is applied.
- Standard and CuPy direct-output bridge paths consume the same packet arrays.
- Packet checkpoint schema v2 stores only the seven dynamic channel states.
  Conductance maxima and the 36-value equation matrix are regenerated from
  live reauthenticated artifacts. Packet-owned slices of otherwise mixed HH
  parameter arrays are zeroed in HDF5 and regenerated after manifest checking.
- The existing legacy SNr kernel remains unchanged.

## Verification

The final combined focused suite passed `81` NumPy tests with three expected
CuPy-only skips and `84` CuPy tests on the RTX 3090. This includes independent
equation-oracle checks, mixed packet/legacy ownership, legacy-equation
differential checks, exact standard/direct-output GPU equivalence, reset,
dynamic-only checkpoint continuation, and fail-closed tamper cases for dtype,
domain, missing state, injected immutable arrays, and exposed packet HH values.

## Next action

Construct versioned executable candidate packets without collapsing measured
targets, transferred channel evidence, and model priors into one authority.
Bind those candidates and their held-out perturbations into a preregistered,
multi-seed adaptive Stage B campaign. The completed runtime gates authorize
that experiment; they do not establish adult-mouse SNr physiology.
