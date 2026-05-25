# Direction 3 V=32 production adversarial reviewer report (2026-05-25)

## Inputs reviewed

- `research/findings/raw/direction_3_v32_production.json` (primary result)
- `research/findings/raw/direction_3_v32_production.log` (per-seed trace + wall clock)
- `research/findings/raw/direction_3_v32_smoke.json` (sanity comparator)
- `research/findings/raw/direction_3_verdict.py` (frozen verdict module)
- `research/findings/raw/direction_3_v32_runner.py` (runner)
- `research/findings/raw/direction_3_bridge_builder.py` (V=32 wrapper)
- `research/findings/raw/direction_3_vocab_spec.py` (frozen vocab spec)
- `research/findings/raw/cross_bridge_mode_unification_probe.py` (parallel-matching primitive, pillar n=95 source, commit cd30fc6)
- `research/findings/raw/biologized_spiking_mode_unification_parallel_matching_runner.py` (DERIV_SEED source)
- `research/findings/raw/biologized_spiking_mode_unification_helpers.py` (`gamma_slot_positions`)
- `research/findings/raw/vocabulary_scaling_run.py` (`BAR`, `N_DIM`, `N_TRIALS`)
- `research/findings/raw/pattern_separation_grounding_probe.py` (`make_deriver`)
- `research/runners/resonate_fire_fhrr.py` (ResonateFireFHRR class)
- `research/runners/spiking_phasor_fhrr.py` (`phases_to_spikes`)
- `research/runners/concept_pool_demo.py` (`train_word_to_pool`)

Production produced at HEAD commit `3ffae15`. Adversarial verification performed without modifying any module.

## Per-item scrutiny

### Item 1: Multi-seed reproducibility (3/3 seeds at every cell)

**PASS.** The JSON's `per_seed` contains exactly 3 entries for seeds [42, 43, 44].
For each seed, `per_load` contains entries for loads {2, 3, 5} with `n_trials=200`
(NOT the 50-trial smoke). Every cell at every seed has OB >= 0.80 AND OI >= 0.80:

- seed 42: L=2 OB=1.000 OI=1.000; L=3 OB=1.000 OI=1.000; L=5 OB=1.000 OI=0.995 — SEED PASS
- seed 43: L=2 OB=1.000 OI=1.000; L=3 OB=1.000 OI=1.000; L=5 OB=1.000 OI=0.995 — SEED PASS
- seed 44: L=2 OB=1.000 OI=1.000; L=3 OB=1.000 OI=1.000; L=5 OB=1.000 OI=0.990 — SEED PASS

Lowest cell value is 0.990 — comfortably above the 0.80 bar; no tied-threshold edge case.
`verdict_entry` per seed agrees with `per_load` to < 1e-12 tolerance.

### Item 2: Smell-test recomputation from raw per-seed data

**PASS.** Independent recompute (without invoking the verdict module) from JSON's
`per_load[*][order_*_accuracy]` reproduces the JSON's `verdict_entry` exactly and
yields PASS at every cell. Multi-seed means independently recomputed:

- L=2: OB_mean=1.000000 (agg=1.000000); OI_mean=1.000000 (agg=1.000000)
- L=3: OB_mean=1.000000 (agg=1.000000); OI_mean=1.000000 (agg=1.000000)
- L=5: OB_mean=1.000000 (agg=1.000000); OI_mean=0.993333 (agg=0.993333)

All means match recorded `aggregate` block to < 1e-9 tolerance. No discrepancy.
Multi-seed verdict independently computed: DIRECTION_3_V32_PASS — matches recorded
verdict and matches the log's printed "DIRECTION_3_V32_PASS" line verbatim.

### Item 3: V=16 vs V=32 genuineness (does V=32 add genuine new info?)

**PASS.** Verified:

- `V` field in JSON equals 32 (top-level and per-seed).
- `substrate` equals `bio_brain_regions_v14v16_recipe_V32` (design-doc-specified label).
- `d_act` is 6400 for all 3 production seeds (= 32 pools x 200 neurons), consistent;
  smoke had `d_act=3200` (32 pools x 100 neurons), so the scaling matches production
  parameters.
- Vocab spec exposes 32 unique concept identifiers (4 motor + 12 noun + 12 verb +
  4 adjective). `DIRECTION_3_V32_TARGET_POOL` maps to 32 distinct pools.
- V=16 baseline (16 words: 4+4+4+4) is fully contained in V=32, plus 16 genuinely
  new extension words (tree, bird, sun, moon, book, chair, house, wheel, walk, run,
  eat, sleep, sit, stand, jump, climb).
- Runner code at lines 543-548, 554 uses `V = len(words)` for both the distractor
  pool choice (`qrng.choice(V, size=load, replace=False)`) and the decoder ranking
  (`scores_oi_gpu = xp.zeros(V)`; `batched_phase_similarity(unbinds[k], vocab_phase_matrix, xp)`
  returns vector of length V=32). Genuine V=32 scaling, not a V=16 subset.

### Item 4: Frozen verdict module output matches `compute_verdict` from the JSON

**PASS.** Imported `research.findings.raw.direction_3_verdict.compute_verdict`,
passed `[s['verdict_entry'] for s in per_seed]` from the production JSON:

- compute_verdict(...) returned: `DIRECTION_3_V32_PASS`
- JSON recorded verdict:        `DIRECTION_3_V32_PASS`
- Match: True

Adversarial corner cases (all behave correctly, no crashes):
- `None`           -> `DIRECTION_3_V32_VOID_MALFORMED`
- `[]`             -> `DIRECTION_3_V32_VOID_MALFORMED`
- 1-seed input     -> `DIRECTION_3_V32_VOID_MALFORMED`
- 2-seed input     -> `DIRECTION_3_V32_VOID_MALFORMED`
- Forced OI(L=5)=0 on all 3 seeds (rest unchanged) -> `DIRECTION_3_V32_PARTIAL`
- All cells zero    -> `DIRECTION_3_V32_NEGATIVE`

PARTIAL/NEGATIVE/VOID branches functional; runner did not tamper with the verdict.

### Item 5: Parallel-matching primitive byte-unchanged

**PASS.** All five reused primitives are byte-unchanged since their pillar commits:

- `cross_bridge_mode_unification_probe.py` (`batched_phase_similarity`,
  `verify_batched_equivalent_to_scalar`): diff vs cd30fc6 (pillar n=95
  commit) = **0 lines**.
- `biologized_spiking_mode_unification_parallel_matching_runner.py` (`DERIV_SEED`):
  diff vs first commit (0738c4f) = **0 lines**.
- `biologized_spiking_mode_unification_helpers.py` (`gamma_slot_positions`):
  diff vs first commit (0503859) = **0 lines**.
- `vocabulary_scaling_run.py` (`BAR`, `N_DIM`, `N_TRIALS`): diff vs first commit
  (e771c3c) = **0 lines**.
- `pattern_separation_grounding_probe.py` (`make_deriver`): diff vs first commit
  (6afab33) = **0 lines**.
- `resonate_fire_fhrr.py` (ResonateFireFHRR): diff vs cd30fc6 = **0 lines**.
- `spiking_phasor_fhrr.py` (`phases_to_spikes`): diff vs cd30fc6 = **0 lines**.
- `concept_pool_demo.py` (`train_word_to_pool`): diff vs cd30fc6 = **0 lines**.

Runner imports the primitives at module top (lines 73-91); no copy-paste local
re-implementation. `git status --porcelain` shows no protected modules modified in
the working tree.

### Item 6: Score-tuning/threshold-tampering check

**PASS.** Verified:

- `bar_ob` in JSON = 0.8 (exact); `bar_oi` in JSON = 0.8 (exact).
- `min_seeds` in JSON = 3.
- `seeds` in JSON = [42, 43, 44] (canonical multi-seed set).
- Frozen verdict module constants:
  `_DIRECTION_3_V32_OB_MIN = 0.8`, `_DIRECTION_3_V32_OI_MIN = 0.8`,
  `_DIRECTION_3_V32_LOADS = (2, 3, 5)`, `_DIRECTION_3_V32_MIN_SEEDS = 3`.
  All match design-doc values.
- Grep on runner module for `np.clip|cp.clip|epsilon|attempt|softer|relax|fallback|override`
  patterns: only legitimate matches ("V=32 scale parameters (frozen for the runner")
  and similar comment-only hits; no score-adjustment code.
- Grep on verdict module for same patterns: only documentation strings asserting
  the constants ARE frozen; no setter/override code.
- Grep on production result JSON for `attempt_1|softer|relax|fallback_verdict|_alt|_v2|_old`:
  **No matches found.** No commented-out / extra fields suggesting a permissive
  verdict was attempted before the final one.

### Item 7: Load-ceiling map V=16 reference applicability

**PASS.** Verified against the V=16 reference values in the prompt:

- V=16 cross-bridge OI (reference, prompt): 1.000 / 1.000 / 0.790 at loads {2,3,5}
  (global_mean).
- V=16 per-bridge OI (reference, prompt): uniformly 1.000.
- V=32 production single-bridge OI multi-seed: 1.000 / 1.000 / 0.993 at loads {2,3,5}.

All V=32 cells >= 0.80; no BOUNDARY/regression label needed. V=32 single-bridge
OI at L=5 (0.993) is ABOVE the V=16 cross-bridge ceiling (0.790), so no regression
worth surfacing as a regression-class warning per the prompt's criterion. Per the
prompt's option (a), V=32 OI at single-bridge matches the V=16 per-bridge pattern
(close to uniform 1.000), demonstrating the substrate scales the OI capability
cleanly to V=32 on a single bridge.

## Additional structural checks (informational)

- Wall clock printed in log = 146.8 min (~2.45 hr). ETA was ~5-6 hr; production
  is below ETA but per-seed breakdown shows all 3 seeds trained from scratch
  (seed 42: 57.8 min, seed 43: 40.7 min, seed 44: 41.0 min; plus capture+probe
  ~7 min total). No cached-bridge short-circuit. 32 words x 200 events = 6400
  events per seed actually ran. Faster than ETA because GPU is RTX 3090 and the
  per-seed cost was lower than the planning estimate; no reduction in trials
  or seeds.
- Per-seed bridge cache files exist for production seeds (`bridge_full_seed42.simstate.h5`,
  `bridge_full_seed43.simstate.h5`, `bridge_full_seed44.simstate.h5`) and activity
  caches likewise; smoke caches also present (separate `_smoke_` tag).
- `batched_vs_scalar_max_diff` per seed: seed 42 = 2.78e-17, seed 43 = 1.39e-17,
  seed 44 = 2.08e-17. All near machine precision (~2e-17), confirming the
  GPU-batched primitive equivalence holds under V=32.

## Verdict

CLEAR

All 7 scrutiny items PASS. The Direction 3 V=32 multi-seed production result is
a genuine biology-grounded scaling demonstration on the bio_brain_regions
substrate. The validated parallel-matching mode-unification (pillars n=93/n=94/n=95)
extends cleanly from V=16 to V=32 (18/18 cells PASS; multi-seed mean L=5 OI 0.993).
Pillar n=105 candidate is APPROVED for promotion.
