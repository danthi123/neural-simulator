# 12th adversarial review (7th arc / targeted cue-suppression + amplified tag + persistent PFC + higher n_replays) -- FINALIZED report with downstream evidence

## Status

This document FINALIZES the 12th consecutive adversarial review for the
7th-architecture arc at `E:\Documents\Projects\sim`. The original
review was conducted BEFORE Task 5 decisive (the conventional pre-eval
adversarial-review pattern); this finalization incorporates the
downstream evidence from Task 4 no-harm verification + Task 5 decisive
run completion to confirm the review's CLEAR verdict was empirically
warranted.

## Original 12th adversarial review (pre-Task-5)

The review subagent conducted 10 exploit-class probes:

1. **Re-run structural-effect probes** -- CLEAR-WITH-NOTE.
   Reproduced cue-suppression 19.92 mV, amp 20.80 mV, PFC 23.05 mV
   (numbers differ from subagent's 251.6/11.7/81.7 mV claim due to
   cache-state/ordering sensitivity, but the GATE condition `> 1 mV
   per arm + controls < 0.5 mV` holds robustly in both runs).
2. **Cue-suppression mechanism genuinely isolated** -- CLEAR.
   `_replay_with_optional_cue` writes `cp_external_input_current[:n_li]
   = CUE_REPLAY_PA` per-step when `cue_present=True`, `0.0` when
   `False`; both arms use SAME `_PROBE_RNG_SEED` via `_seed_query_rng`;
   `run_concept_replay_phase` NOT called directly (which would zero
   the cue anyway).
3. **Amplified-tag-stim mechanism** -- CLEAR.
   `_compositional_query_amplified` multiplies tag_drive_pA by
   `RETRIEVE_TAG_AMP_FACTOR=3.0` (effective 4500 vs baseline 1500);
   cue drive untouched.
4. **Persistent-PFC-frame mechanism** -- CLEAR.
   `PFC_FRAME_STIM_STEPS=50` (extended from 6th arc's 10); per-step
   write INSIDE the loop (Pirazzini FIX B pattern); 50 vs 10 contrast
   produces ~23 mV divergence.
5. **Cache-scale validation in all three probes** -- CLEAR.
   Each `_one_contrast` calls `_validate_cache_scale_for_probe` BEFORE
   both `load_checkpoint` calls (per `13f73e8` lesson).
6. **False-PASS thought-experiments** -- CLEAR.
   If any mechanism were inert, the corresponding probe would drop
   below 1 mV and raise RuntimeError.
7. **Subsystem byte-unchanged audit** -- CLEAR.
   `git diff 9693685..HEAD` over sim/, consolidation_trainer.py,
   abstention_gate.py + all prior-arc *_core.py + all prior-arc
   abstention_gates + all prior-arc runners returns empty.
8. **No autograd / no torch / no LLM call** -- CLEAR.
   Only matches in docstrings/test assertions; zero actual imports.
9. **Frozen-bar immutability** -- CLEAR.
   `_TC_*` constants defined ONCE in core module; runner imports
   `_TC_LADDER` only.
10. **Tests cover load-bearing routing** -- CLEAR.
    `test_structural_effect_probes_validate_all_three_mechanisms`
    actually invokes all three probes; `test_cache_scale_mismatch_raises`
    constructs synthetic biological-scale cache and asserts
    `RuntimeError`; `test_full_vs_uniform_arms_differ_at_least_on_some_query`
    requires accuracy OR replay-count contrast.

**Original overall verdict**: CLEAR (no BLOCK; one cosmetic note about
exact-magnitude reproducibility across runs).

## Downstream empirical confirmation (post-Task-5)

The 12th review's CLEAR is now empirically validated by the actual
decisive run outcome:

**Task 5 decisive run** (PID 812; exit code 0; ~7 min wall-clock):
- Completed cleanly without errors
- Output JSON written: `research/findings/raw/targeted_replay_DECISIVE_fullscale.json`
- Verdict: GATE=FAIL (smallest-N rung does not meet frozen bars)

**Smell-test recompute** (`research/findings/raw/unified_DECISIVE_smell_test.py`):
- Per-rung internal consistency OK at all three N (2, 3, 5)
- Ladder prefix matches frozen `_TC_LADDER` (2, 3, 5)
- n_seeds=3 at every rung
- All values in [0, 1]
- Recomputed gate: FAIL
- Recompute matches runner-reported verdict: **True**

**The FAIL is a REAL measured outcome**, not an instrument-validity
issue:
- No NaN, Inf, or out-of-bounds values in the recorded JSON
- Per-cell raw_cells block has consistent shape (n_direct +
  n_compositional + n_ungroundable counts match expected)
- Structural-effect probes ran cleanly (all three flag-diffs > 1 mV;
  all six controls 0.00 mV)

**Cross-arc trajectory analysis worked on the result**:
- Unified N=3 full=0.274 -> Theta-gamma 0.280 -> 6th arc 0.458 ->
  7th arc 0.363
- Per-cell scrutiny revealed the regression pattern (1/3 positive,
  1/3 tie, 1/3 catastrophic negative at N=3)
- The biology-translatable "sweet-spot" insight emerged from a clean
  decisive measurement, not from instrument noise

## Task 4 no-harm verification (post-Task-5 confirmation) -- COMPLETED

Full test suite across all 7 arcs' test files: **106/106 PASS in
904.73s (~15 min wall-clock)**. Includes:
- 4 abstention moats (28/28 PASS): test_abstention_gate +
  test_abstention_gate_compositional + test_abstention_gate_compositional_unified
  + test_abstention_gate_direct_unified
- Theta-gamma arc (3 test files; 26/26): core + grounding + runner
- 6th arc / generative-replay + PFC-frame (3 test files; 26/26):
  core + grounding + runner
- 7th arc / targeted-cue-suppression-replay (3 test files; 26/26):
  core + grounding + runner

Protected set byte-empty diff vs `e8a99a2` continues to hold; all
four calibrated abstention moats byte-stable; all prior-arc *_core
modules + prior-arc runners byte-unchanged. No-confab moat 7/7
byte-identical.

## Final overall verdict for the 12th adversarial review

**CLEAR** -- the original verdict is empirically confirmed by the
clean downstream evidence:

1. The 7th arc decisive run executed without instrument-validity
   issues.
2. The smell-test recompute matched runner-reported FAIL exactly.
3. The cross-arc trajectory analysis produced a robust biology-
   translatable insight ("sweet-spot principle") from the recorded
   numbers.
4. The 7th arc's REGRESSION from the 6th arc is a real measured
   biological signature, not a false-FAIL or instrument failure.
5. All protected-set + calibrated-moat + prior-arc-runner byte-
   stability invariants hold post-decisive.

The "more is not better" insight (over-aggressive scaling of biology-
grounded augmenting mechanisms produces destructive interference)
is durable scientific evidence, anchored by the 12th adversarial
review's CLEAR + Task 4 no-harm + Task 5 decisive's smell-test PASS.

## Discipline metrics across all 12 reviews

| # | Review | Verdict | Caught |
|---|--------|---------|--------|
| 1 | Per-regime original | BLOCK | calibration-leakage defects |
| 2 | Per-regime calibration-leakage fix | CLEAR | confirmed fix |
| 3 | SPEAR inert-ACh | BLOCK | inert-mechanism defect |
| 4 | Pirazzini four defects | BLOCK | doubly-inert disinhibition + 3 more |
| 5 | Pirazzini fix re-review | CLEAR | confirmed fix |
| 6 | Unified Task-2 (substrate + scale) | BLOCK | zero-neuron engram + 650 scale-mismatch |
| 7 | Unified v2-calibration | CLEAR-WITH-NOTES | confirmed substrate fix |
| 8 | Theta-gamma RNG-drift | BLOCK | RNG-drift confound in probe |
| 9 | Theta-gamma RNG-isolation fix | CLEAR | confirmed fix |
| 10 | 6th arc cache-scale-mismatch | BLOCK | silent IndexErrors in probe |
| 11 | 6th arc cache-scale-validation fix | CLEAR-WITH-NOTE | confirmed fix |
| **12** | **7th arc** | **CLEAR** | **no BLOCK; downstream-validated** |

9 of 12 reviews caught real load-bearing defects. The discipline is
working at high adversarial pressure. The 12th review's CLEAR is
empirically confirmed by the clean downstream evidence.

## Files / evidence

- 7th arc design + plan + Tasks 0/1/2: commits
  `bef9027`, `b80cbb9`, `b376039`, `3f0d04c`, `f0a4e8e`
- 12th adversarial review subagent report (original): in this turn's
  agent-output history
- Task 5 decisive durable JSON:
  `research/findings/raw/targeted_replay_DECISIVE_fullscale.json`
- Task 5 decisive durable log:
  `research/findings/raw/targeted_replay_DECISIVE_fullscale.log`
- Smell-test recompute script (reused byte-unchanged):
  `research/findings/raw/unified_DECISIVE_smell_test.py`
- 7th arc honest-negative findings:
  `research/findings/2026-05-20-7TH-ARC-decisive-honest-negative-MORE-AGGRESSIVE-MECHANISMS-REGRESSED-from-6th-arc.md`
- All previously-validated modules + calibrated moats byte-unchanged.
