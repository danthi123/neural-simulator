---
type: finding
status: complete
date: 2026-08-04
verdict: DETERMINISTIC_COMPATIBILITY_GO
mechanism: gateB-v13-tonic-output-substrate
runner: research/runners/_v13_deterministic_compatibility.py
artifacts:
  - research/findings/raw/v13_deterministic_compatibility/bundle-baseline_8994_plus_deterministic_patch.json
  - research/findings/raw/v13_deterministic_compatibility/bundle-candidate_v13.json
  - research/findings/raw/v13_deterministic_compatibility/comparison-baseline-vs-candidate.json
---

# V13 deterministic compatibility correction earns GO

**Verdict: DETERMINISTIC_COMPATIBILITY_GO.** The preregistered correction ran
six compatibility seeds (`271829`, `271831`, `271837`, `271843`, `271849`,
`271853`) on NumPy and CuPy in three separate processes per seed/backend: an
exact `6 x 2 x 3` matrix for each source twin, or 72 runs overall. The sealed
baseline matrix was generated before the candidate matrix, as required.

Both `baseline_8994_plus_deterministic_patch` and `candidate_v13` earned
within-twin determinism GO. Across the two twins, topology and all seven locked
hashes matched exactly for every seed and backend: the complete spike raster,
final `v`, `u`, `g_e`, and `g_i`, complete weights, and final external current.
Both twins used stable deterministic patch ID
`18bd23624a3247cb0f205795081b7a540c15ed89`, the same runner, locked spec, and
preregistration. All comparison preconditions passed with no undefined reason.

The evidence is sealed in:

- `research/findings/raw/v13_deterministic_compatibility/bundle-baseline_8994_plus_deterministic_patch.json`
- `research/findings/raw/v13_deterministic_compatibility/bundle-candidate_v13.json`
- `research/findings/raw/v13_deterministic_compatibility/comparison-baseline-vs-candidate.json`

This resolves the deterministic compatibility block and opens V13 Stage 0
calibration at seed `1013`. It does not establish autonomous tonic output,
inhibition or recovery, lesion causality, checkpoint continuation, selector
behavior, learning, or performance. It does not erase the original failed
compatibility finding or promote replication, held-out, or Stage 1 seeds; those
remain governed by their own preregistered gates.

## Audit limitations

The sealed baseline finished before candidate execution began, with a measured
gap of 137.85 seconds. Candidate cell invocations did not themselves read the
baseline bundle, so that ordering was operational rather than an executor-level
dependency. The Stage-0 runner now verifies that the candidate source commit
already contained the byte-identical sealed baseline bundle, and the comparison
runner loaded and hashed that bundle before issuing this verdict. Future
compatibility executors should make the baseline bundle digest a required
candidate-cell input instead of relying on orchestration order.

The preregistration describes the baseline both as lacking the new intrinsic
field and, elsewhere, as reporting that field as `None`. Those statements cannot
both be literal. The executed source-twin contract uses the correction's stated
purpose: all 36 baseline cells prove the field is absent, while all 36 candidate
cells prove it exists with a default value of `None`. The Stage-0 runner now
rejects the compatibility evidence unless that distinction is intact.

All 72 run sidecars exist, but the shared `runs.jsonl` ledger contains only 18
of the 72 cell records. The bundles bind every cell artifact by digest and the
commits seal the sidecars, so this does not change the numerical verdict. The
create-only `provenance-manifest-cells.json` now also binds all 72 artifact and
sidecar byte hashes and records the historical source limitations: 66 sidecars
reported a dirty checkout and none carried a source-manifest digest. This
closes artifact enumeration, not the original provenance completeness defect;
the finding does not claim the historical shared ledger or source seals were
complete.
