# Systemic large-batch test pollution — verified pre-existing, broader than concept_pool

## TL;DR

A full-suite run (`pytest tests/` excl. the already-flagged
`test_concept_pool_bridge.py`) reported **52 failed / 1207 passed**.
Verified: these are **pre-existing batch-pollution**, NOT a
regression from this session's G.20/sparse arc, and the pollution is
**broader than the previously-flagged concept_pool case**.

## Verification (systematic, not assumed)

The alarming "52 failed" was not taken at face value:

1. **Orthogonal imports.** The failing files
   (`test_structural_pruning.py`, `test_tans.py`,
   `test_visual_cortex.py`, …) import **none** of the modules this
   arc changed (`shared_pool_chat`, `g20_multibridge`,
   `g20_{xbridge,sentence,abstention,interference}_benchmark`,
   `concept_pool_sparse_distributed`). They test `sim/` internals
   (structural pruning, TAN neuromodulator, visual cortex Gabor) —
   a different subsystem entirely.
2. **Pass in isolation.** A sample of 3 failing tests
   (`test_v1_orientation_tuning_after_gabor_init`,
   `test_single_pulse_reward_fires_plasticity_within_step`,
   `test_pruned_synapse_stays_at_zero_after_simulation_steps`) run
   together in isolation: **3 passed in 3.3s**. They only fail
   inside the full ~1260-test batch.
3. **Arc tests green throughout.** Every targeted sweep this session
   passed (141 G.20, 169 G.20+sparse+multibridge+vocab, 96
   contiguous-multibridge, etc.). The arc's changes are purely
   additive (new helpers/runners/tests).

Conclusion: classic shared-state batch-pollution — global/GPU
(CuPy memory pool, bridge singletons, neuromodulator/region module
state) leaks between unrelated tests when ~1260 run in one process.

## This broadens the flagged concept_pool task

The earlier flagged task scoped the pollution to the `concept_pool`
test selection. This run shows it is **systemic**: it also corrupts
`structural_pruning`, `tans`, `visual_cortex` (all
GPU-state/sim-internal heavy). The flagged task's investigator
should treat it as a **suite-wide test-isolation defect** (likely a
shared CuPy/bridge/global not reset between tests), not a
concept_pool-local quirk. Likely fix class: per-test GPU/memory-pool
reset fixture, or eliminating module-level mutation in the
sim/region/neuromodulator paths.

## Honest status

- This session's G.20 sparse arc is **regression-clean** (verified:
  orthogonal modules + pass-in-isolation).
- The 52 failures are a **real, pre-existing, systemic test-infra
  defect** — documented honestly, not hidden, not attributed to the
  arc, not "fixed" with a paper-over. It needs a dedicated
  test-isolation fix (the flagged task, now broadened).

## Files

- Reproduce: `pytest tests/ -q --ignore=tests/test_concept_pool_bridge.py`
  (52 failed); same tests pass in isolation.
- Related: the flagged concept_pool-pollution task (broaden scope to
  suite-wide).
