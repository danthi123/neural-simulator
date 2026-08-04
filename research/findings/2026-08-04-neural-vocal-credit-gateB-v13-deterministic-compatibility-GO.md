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
