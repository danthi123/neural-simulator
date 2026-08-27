---
type: finding
status: live
lane: onebrain-merge
date: 2026-08-27
---

# One-brain merge FRAMEWORK — folding the 4 bespoke pool organs into the registry (rung #2 of the migration plan)

Status: GO for the SUBSTRATE-INIT + ORGAN-READ co-residence-invariance MIGRATION gate, for all 4 bespoke
pool organs (surprise, world-model, metacog, pragmatic), 6/6 seeds. This closes rung #2 named by
`2026-08-27-onebrain-merge-framework-groupA-batch-migration.md`'s "Next rungs": *"Fold the 4 already-merged
pool organs (surprise/world-model/metacog/pragmatic) into the SAME registry, retiring
`MergedSubstrate`/`MergedSubstrate2` to a thin `merge_organs([...])` call."* The fold is done and verified;
the literal class-body retirement is NOT done in this pass (see "What is NOT done" below — an honest scope
boundary, not a hidden gap).

**Method note (external grounding):** the round-trip-vs-shipped verification technique this session leans on
(`_smoke`/`_smoke2` — run the SAME inputs against the OLD bespoke implementation and the NEW declarative one,
require byte-identical output, before any retirement is even considered) is Michael Feathers' Characterization
/ Golden-Master Test (Feathers, *Working Effectively with Legacy Code*, 2004;
<https://en.wikipedia.org/wiki/Characterization_test>) — the standard technique for safely refactoring
undertested legacy code by pinning its observed behavior before changing its implementation. This is a
software-engineering migration task (registering existing, already-verified biology in a data-driven registry),
not a fresh biological mechanism decision, so the external check that applies is engineering methodology, not
neuroscience literature.

## What changed in `research/runners/onebrain_merge_framework.py`

Surprise + world-model (pool #1) were ALREADY registered as `OrganDescriptor`s by the framework's original
prototype skeleton (2026-08-27 DESIGN), verified only by a SEED-42 init-array smoke against the shipped
`MergedSubstrate`. This session:

1. **Registered METACOG + PRAGMATIC** (pool #2's organs) as `OrganDescriptor`s — spec_fn reused BY CALLING
   the shipped `MergedSubstrate2._metacog_specs`/`._pragmatic_specs` on a throwaway instance (one definition
   of the geometry, no copy to drift), config mirroring `MergedSubstrate2.ensure_built`'s cfg block, an
   `explicit_wiring_fn` for metacog's K dense self-recurrent assembly loops (`_build_assembly_loop_population`,
   reused-by-import), and `organ_cls`/`read_fn`/`answer_fn` wired to the REAL shipped production classes
   (`MetacogProductionOrgan`, `PragmaticProductionOrgan`) — both ALREADY carry a `shared=` kwarg in production
   (unlike most Group-A organs, which needed one added). Two new methods on `MergedPool`
   (`metacog_idx`/`pragmatic_item_dev`) dispatch to the descriptors' `idx_fn`, mirroring the existing
   `surprise_idx_map` pattern, so the shipped organ classes run UNMODIFIED against the registry pool.
2. **Extended the pool-1 smoke to 6 seeds** (`_smoke`, was seed-42-only) and **added a pool-2 round-trip
   smoke** (`_smoke2`) that goes PAST init-array comparison to run the REAL shipped organs' read pipelines
   (`judge()`/`interpret()`) against both the registry pool and the shipped `MergedSubstrate2`, plus a
   **build-twice determinism check** (`_determinism2`) hashing per-neuron init arrays + wired connection
   weights.
3. `REGISTRY` now holds all 4 pool organs + the 7 Group-A organs (11 total).

## A real bug the round-trip caught (why the extra verification step earned its cost)

The first `--smoke2` run FAILED: `cp_izh_d_increment` on the "workspace" region differed by 111 between the
registry pool and the shipped `MergedSubstrate2` at IDENTICAL seed/config. Cause: `MergedSubstrate2` sets
`cfg.per_region_parameter_heterogeneity=True` (the name-keyed, co-residence-invariant Izhikevich-jitter seam);
the framework only auto-sets that flag when a descriptor declares `param_het=True` (its own masked-reconciliation
path), which METACOG/PRAGMATIC deliberately do NOT use (pool #2 sets parameter-heterogeneity GLOBALLY — no
competing organ in a 2-organ pool wants it off, exactly matching the shipped class). Missing the explicit
`per_region_parameter_heterogeneity=True` in the descriptor config silently fell back to the LEGACY
whole-pool, position-dependent heterogeneity draw. Fixed by adding the flag explicitly to both descriptors'
`config` dicts (see the code comment at its use site). Re-run: byte-identical, 6/6.

## Results (6 seeds: 42, 43, 44, 100, 101, 102; numpy CPU, bit-exact)

**Pool #2 (metacog + pragmatic), the newly-folded organs — TWO independent verifications:**

| check | artifact | result |
|---|---|---|
| Internal: merged-vs-coresident-on-superset-config (the framework's own migration gate, matching the 7 Group-A organs' bar) — substrate-init AND organ-read (real `judge()`/`interpret()` reads + answer preservation) | `research/findings/raw/_onebrain_merge_pool2_6seed.json` | 6/6 GO both organs, both gates; legacy (seams-off) discriminator diverges 6/6 (non-vacuous) |
| Round-trip vs the SHIPPED `MergedSubstrate2` — init arrays AND the real production organs' reads run against each pool | `research/findings/raw/_onebrain_merge_framework_smoke2_pool2_6seed.json` | 6/6 PASS, `init_delta=0.0` / `read_delta=0.0` every seed |
| Build-twice-at-one-seed hash determinism (the `cfg.seed` reproducibility gotcha) | `research/findings/raw/_onebrain_merge_pool2_determinism_6seed.json` | 6/6 identical hashes |

**Pool #1 (surprise + world-model) — the pre-existing registration, verification extended from 1 to 6 seeds:**

| check | artifact | result |
|---|---|---|
| Round-trip vs the SHIPPED `MergedSubstrate` (init arrays; the smoke's original scope) | `research/findings/raw/_onebrain_merge_framework_smoke_pool1_6seed.json` | 6/6 PASS, `max_init_delta=0.0` every seed |

Pool #1's organ-read round-trip (real `judge()`/`expectation()` reads, not just init) is NOT re-verified here
— that bar was already met by the pre-existing `_onebrain_merge_rung1_verify.py` (6/6 GO, cited in
`onebrain_merge_production.py`'s docstring); this session only strengthened the FRAMEWORK-registry side
(`_smoke`) to 6 seeds, since surprise/worldmodel were not re-registered, only re-verified.

**Term check (`docs/TERMS.md`):** "byte-identical" is asserted from the data above (exact hash / max-delta
compare across every cited artifact), never inferred from reading the code.

## Name-collision honesty (why metacog/pragmatic are NOT in the "all"/Group-A batch)

Metacog's "workspace"/"workspace_fs" region names collide with self_schema's (Group-A) — the identical
collision class the DESIGN doc names for affect vs metacog (every merge seam keys its name-invariant RNG on
the region NAME; a rename would break byte-identity to the standalone organ). So metacog+pragmatic are
verified as THEIR OWN pair (`--keys metacog,pragmatic`), exactly how pool #1 (surprise/worldmodel) was
already excluded from "all" for the same reason (surprise's "cue" collides with curiosity's "cue"). This is
not a new limitation — it is the SAME structural fact the framework's `_resolve_keys` docstring already
states, now true of 4 registry members instead of 2.

## What is NOT done — the retirement question, answered honestly

**`MergedSubstrate` (pool #1) blocks the family-wide retirement.** Reading `onebrain_merge_production.py` in
full (required by this session's research-first mandate) shows it is NOT a clean 2-organ merge in production:
`get_merged_substrate()` ALSO optionally carries the RF-phasor recall composer region, a phase->spike
transducer cleanup region, and — when `composer_merge_enabled()`/`parser_merge_enabled()` are on (both
default-ON in production, `_COMPOSER_IN_POOL1_DEFAULT_ON` / `_PARSER_IN_POOL1_DEFAULT_ON`) — the FULL
production `OneBrainComposer` layout (`onebrain_composer` region sized to its complete RF span) with its
Hebbian PARSER transplanted onto the pool via `_bind_parser_onto_pool` (weight transplant + a permanent
per-synapse gain-0 freeze + isolation-wrapped stepping). None of that composer/parser production wiring is
represented in the framework's `SURPRISE`/`WORLDMODEL` descriptors (which only cover the two organs' own
regions) and folding it in is a substantially larger, separate piece of work — out of this session's scope
(a single sonnet-tier pass under a hard memory-safety constraint with a concurrent GPU run). So: **pool #1's
`MergedSubstrate` class stays as-is; the family-wide "MergedSubstrate\* can be retired" bar is NOT met.**

**`MergedSubstrate2` (pool #2) is retirement-READY but not flipped in this pass.** Unlike pool #1, it is
genuinely clean — exactly the 2 organs, no extra production wiring — and both organs are now cleanly folded
+ verified byte-identical (both internally and against the shipped class, both init AND real-organ-read, both
directions of the round-trip, plus build-twice determinism). A thin-wrapper refactor of
`MergedSubstrate2.ensure_built()` to delegate to `merge_organs([METACOG, PRAGMATIC], seed, wire=True)`
internally (preserving its external API — `.bridge`/`.cfg`/`.xp`/`.snap`/`.metacog_idx()`/
`.pragmatic_item_dev()`/`.ensure_built()`, so `MetacogProductionOrgan`/`PragmaticProductionOrgan` and
`get_merged_substrate2()` are unaffected) is now well-de-risked. It was deliberately NOT done here: the class
also has single-organ constructor callers outside the production entry point
(`_metacog_robust_confidence_derisk.py`'s `MergedSubstrate2(organs=("metacog",))` /
`(("pragmatic",))`, and `_onebrain_production_flip2_verify.py`'s identical pattern) whose exact behavior a
refactor must also preserve, and `metacog`/`pragmatic` are DEFAULT-ON in live production chat right now,
concurrently with a heavy GPU run elsewhere — a live-production-path edit deserves its own narrowly-scoped
commit + a rerun of `_onebrain_production_flip2_verify.py` as a dedicated regression gate, not a rider on this
verification pass. **This is the well-scoped next rung, not a discovered blocker.**

## Next rungs (unchanged from the groupA-batch-migration finding, minus the item this closes)

1. The pool-#2 thin-wrapper retirement above (de-risked; needs the dedicated regression pass).
2. Fold pool #1's composer/parser production wiring into the framework (or accept it as a permanent,
   documented exception the registry does not model) before the family-wide `MergedSubstrate*` retirement
   claim can be made.
3. Group-B seams (STP / OU+neuromod / apical-NMDA-slow / multi-bridge) for the 6 organs
   `GROUP_A_DEFERRED` names.
4. INTEGRATION phase (DESIGN §4): cross-region edges under the F-gate. Byte-identity is the safety gate
   beneath this, not the goal itself — a pool with zero cross-edges (all 11 registry members, today) is
   MIGRATED, not INTEGRATED.
