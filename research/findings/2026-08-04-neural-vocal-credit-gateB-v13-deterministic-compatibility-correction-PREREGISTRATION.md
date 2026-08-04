---
type: preregistration
status: locked
date: 2026-08-04
mechanism: gateB-v13-deterministic-default-off-compatibility
spec: research/specs/v13_tonic_output_deterministic_compatibility.json
supersedes_gate_only: V13 Stage-0 CuPy exact-state compatibility criterion
does_not_supersede: V13 physiology, lesion, checkpoint, provenance, performance, or promotion criteria
---

# V13 deterministic compatibility correction

This protocol corrects one invalid measurement method in the V13 tonic-output
preregistration. It does not revise the tonic-output mechanism, physiology
criteria, seed partitions, or promotion bar.

The first registered compatibility run remains
`COMPATIBILITY_NO_GO`: NumPy reproduced every default-off fingerprint, while
CuPy reproduced the complete spike raster and immutable state but not exact
final `v`, `u`, and `g_i` hashes. The simulator already documents why exact
CuPy state hashes are not a valid fingerprint under the default transpose
sparse-matrix path: its atomic scatter has nondeterministic floating-point
summation order.

The correction is to remove that nondeterminism for this audit, not to weaken
the equality requirement after seeing the result.

## Pre-execution seed-schema clarification

Before any formal correction seed was executed, an adversarial audit found
that the machine-readable spec mislabeled the six documented compatibility
seeds as `calibration` and also contained an undocumented held-out seed
`271859`. The spec now records only one `compatibility` partition containing
the six seeds locked below. Seed `271859` is removed and must not be executed
or interpreted as evidence for this correction. This clarification changes no
documented seed, measurement, acceptance threshold, or scientific partition.

## Locked implementation correction

When `deterministic_transpose_matvec=True`, the E/I split must:

1. materialize the transposed connection matrix as CSR;
2. run one one-dimensional CSR matrix-vector product for excitatory sources;
3. run a separate one-dimensional CSR matrix-vector product for inhibitory
   sources; and
4. preserve the existing two-column path byte-for-byte when the flag is false.

A two-column CSR matrix-matrix multiply is not accepted as the deterministic
path. The correction must have focused tests showing repeated exact output for
both one-dimensional products on the RTX 3090 and a frozen unregistered NumPy
trajectory showing unchanged default-off behavior against the pre-correction
source.

This is simulator instrumentation. It does not add intrinsic current, change a
weight, alter a stimulus, or choose an action.

## Sealed source twins

After the deterministic correction is committed, create two clean, separately
identified worktrees:

- **baseline twin:** source `8994b5102`, plus only the deterministic matvec
  correction commit and the compatibility executor required to record it;
- **candidate twin:** current V13 source, including the identical deterministic
  correction commit.

Record the base revision, deterministic patch digest, complete execution-file
manifest, environment, CUDA/CuPy versions, GPU identity, and runner digest for
both twins. The baseline is invalid if it contains the intrinsic-current field
or any other candidate simulator change. The candidate is invalid if any
execution file is dirty.

Generate and seal all baseline artifacts before executing a candidate twin.
Candidate execution must read the baseline fingerprint artifact rather than a
hash copied into source code.

## Fresh partitions

The correction uses six fresh compatibility seeds:

`271829, 271831, 271837, 271843, 271849, 271853`.

These are instrumentation/regression seeds. They do not replace V13
calibration seed `1013`, replication seed `1019`, held-out seed `1021`, or
Stage-1 seed `1031`.

For every source twin, seed, and backend, launch three separate Python
processes. Each process reconstructs the Gate A v2 network, sets
`deterministic_transpose_matvec=True`, keeps intrinsic current at its default,
uses the locked `250 pA` shared practice drive, disables commit NMDA, and runs
the same `300` steps as the original compatibility gate.

## Preconditions

Every artifact must carry an earned `tools.verdict.Verdict` precondition block
covering:

- exact seed, backend, source manifest, runner, and deterministic patch;
- RTX 3090 identity for CuPy;
- deterministic flag read back as true;
- intrinsic vector observed as `None`;
- complete `300 x 600` raster capture;
- exact topology, weight, and external-current audit; and
- no dirty execution/config input.

A missing or failed precondition makes the result `UNDEFINED`, never a no-go.

## Exact acceptance gate

For each source twin, seed, and backend, all three separate-process runs must
have identical hashes for:

- complete spike raster;
- final `v`, `u`, `g_e`, and `g_i`;
- complete weights; and
- final external current.

The candidate must then match its corresponding baseline twin exactly for all
seven hashes on all six seeds and both backends. Both twins must report
`cp_intrinsic_current_pA is None`.

No numerical tolerance, majority vote, raster-only substitution, excluded
startup window, or seed replacement is allowed. Any repeat mismatch is
`DETERMINISM_NO_GO`. Any cross-twin mismatch is `COMPATIBILITY_NO_GO`.

## Stop and promotion rule

Only `DETERMINISTIC_COMPATIBILITY_GO` resolves the original compatibility
block and permits V13 Stage 0 to begin calibration at seed `1013`. It does not
itself establish autonomous tonic output, inhibition/recovery, lesion causality,
checkpoint continuation, selector behavior, learning, or performance.

The first failed compatibility finding remains in the record. Do not edit its
artifacts or the original preregistration. This correction artifact must be an
explicit prerequisite of later V13 merge and held-out stages.
