---
type: research-finding
status: stageA-go
date: 2026-08-04
mechanism: neural-vocal-action-credit-v14-explicit-snr-pacemaker-stageA
---

# Gate B v14 Stage A: engineering GO

## Verdict boundary

The default-off, region-scoped SNr conductance substrate passes its equation,
state, isolation, checkpoint, backend, memory, and default-off compatibility
checks at commit `a3d72420fc1a1d4c35db73a121c32b1e647c8ae0`.

The prospective V3 RTX 3090 matrix subsequently passed every fixed performance
and dispersion threshold at candidate `2c88daaac`. Stage A is therefore an
engineering GO. The timing receipt is
`research/findings/raw/v14_stageA_performance_v3_2c88daaac.json`.

This authorizes writing and source-sealing a separate Stage B calibration
specification. It does not itself open calibration seed `590297`, replication,
held-out, future selector seed `790326`, or V13 seed `1031`.

## Implemented substrate

An HH region may now request maximum conductances for a NALCN-like current,
persistent sodium, Cav2.2-like calcium, calcium-coupled SK, and optional Ih.
The bridge holds five immutable maximum-conductance arrays and seven dynamic
gate/calcium arrays, advances them in one fused device operation, and subtracts
their ionic current inside HH membrane dynamics. No per-step channel value is
calculated by the host.

All bundle arrays remain `None` when no region requests the feature. The active
state is exactly twelve float32 arrays, or `48 * total_neurons` bytes. State is
initialized at voltage-dependent equilibrium, included in recording/playback,
cleared with the bridge, and saved under a fail-closed checkpoint schema.

## Evidence

- Focused and adjacent local NumPy suite: `136 passed`, `3` expected GPU-only
  skips. The existing HH epsilon test emitted its known divide warning.
- Focused CuPy suite on the RTX 3090: `12 passed`.
- Six preregistered default-off compatibility seeds matched the committed
  pre-change anchor `6c9034991` byte for byte for trajectory hashes. Every
  uninterrupted/restored continuation hash matched, and neither source
  allocated a bundle array.
- `pool40`: `80 passed` for equations and configuration.
- `pool41`: `16 passed` for bridge, checkpoint, and old intrinsic-current
  compatibility.
- `pool42`: `40 passed`, `3` expected GPU-only skips for kernel and fast-path
  regressions.
- All pool nodes used the same read-only archived commit, source manifest
  `8570a394c8db556c90c02ae87aeeaef905494cab8694723d66758b0255d0ed43`,
  and ancestry attestation
  `4f980fdd088e0bd8ab3dcd60b912803f829d893db06de6b6de986a336f505f01`.

Raw candidate/control compatibility cells are in
`research/findings/raw/v14_snr_conductance_stageA/`.

## Honest scope

Stage A validates executable equations and engine lifecycle behavior, not an
adult SNr parameter set or autonomous tonic firing. The initial kinetics are
evidence-center equation seeds; calcium conversion and coupling remain model
parameters because no located primary source supplies a complete matched adult
SNr density and calcium-nanodomain vector.

V13's constant-current physiology remains engineering context only and its
`TONIC_OUTPUT_NO_GO` is unchanged. The scaffold ledger row cannot be closed
until a later preparation-matched ensemble passes the required causal lesions
and then works inside a continuous selector.

## Performance closure

<!--derived-->
The sealed V3 matrix completed all twelve observations. Its median paired host
ratios were `0.9975` for default-off behavior, `0.9407` for active execution,
and `0.7493` for direct output relative to unfused output. These pass the fixed
ceilings of `1.02`, `1.25`, and `0.85`. All telemetry and dispersion checks
passed. This evidence affects engineering feasibility only and makes no
physiology or behavior claim.

## Next exact action

Write a preparation-matched Stage B calibration specification around the
adult SNr pacemaking role, causal channel lesions, and the already reserved
seed sequence. Hash and commit that protocol before opening calibration seed
`590297`.
