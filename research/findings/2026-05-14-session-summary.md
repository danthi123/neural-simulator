---
type: finding
status: corrected
date: 2026-05-14
---

# Session 2026-05-14: bug retraction + validated semantic conversation

## TL;DR

Today's autonomous arc accomplished a major correction-and-validation cycle:

1. **Discovered + retracted critical architecture-mismatch bug** in
   `compose_concept_engram` that had been silently inflating concept-
   concept measurements since 2026-05-13.
2. **Re-validated 87.5% multi-seed engram-tag stim-recall** (catalog
   D.14 Tonegawa) with corrected architecture + strong encoding recipe.
3. **Designed + validated 90% FULL / 100% PARTIAL multi-seed multi-tag
   cue retrieval** — the genuine concept-concept conversational
   capability the user wanted.
4. **Scaled tests**: 2-assoc (90% FULL), 3-assoc (72.5% FULL), v17 28-
   word (0% FULL — architectural limit identified).
5. **Webapp integration**: 2 new presets surfaced, tests passing.

## Timeline

### Phase 1: bug discovery (morning)

Resuming from yesterday's v18 NEGATIVE finding (cross-pool plastic
pathways didn't break the 25% top-1 plateau), I tried a v19 fix:
freeze the `cross_pool_concept` gate during Phase 1, open only during
encoding. Trained v19 seed 42 (~20 min), ran strict eval — same 2/8
result as v18 (and v16).

The exact same per-pair top-1 outputs across v16/v18/v19 was the red
flag. Investigation revealed that `compose_engram_demo_v2.py` at
module-load time monkey-patches `concept_pool_demo`'s `NOUN_VOCAB` /
`VERB_VOCAB` / `ADJECTIVE_VOCAB` to v17 (28-word) values. This patch
was firing transitively whenever `compose_concept_engram` was
imported.

Result: every concept_concept eval was building 28-pool bridges
(10368 neurons) and loading 16-pool v16/v18/v19 checkpoints into
them. Architecture mismatch silently corrupted the weights. The
"25% top-1 ceiling" was the corrupted firing pattern coincidentally
scoring above chance.

### Phase 2: retraction (mid-morning)

Removed the module-level v2 patch import. Added explicit
`--enable-cross-pool-concept-pathways` flag to `compose_concept_*`
runners.

Re-tested v16, v18, v19 strict on seed 42 with corrected architecture
(7680 neurons, matching v16 bridges):
- v16 strict: 0/8 top-1, 3/8 top-3
- v18 strict: 0/8 top-1, 2/8 top-3
- v19 strict: 0/8 top-1, 2/8 top-3
- v16 chain: 1/4 PASS (was claimed 90% multi-seed)

All previous "25% / 65% / 90%" claims invalidated. Documented in
[`2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`](2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md).

### Phase 3: re-validation (mid-day)

Tested compose_concept_engram (which uses Tonegawa-style engram
tagging) with corrected architecture. Got 4/8 stim-recall on v16
seed 42 with default settings (200 events, no teacher).

Strengthened encoding: 500 events + `--balanced-teacher-pA 500.0`.
Result on v16 seed 42: 7/8 stim-recall, 2/8 assoc-recall.

Multi-seed test (5 seeds × 8 pairs):
- seed 42: 7/8 stim, 2/8 assoc
- seed 43: 6/8 stim, 3/8 assoc
- seed 44: 8/8 stim, 2/8 assoc
- seed 45: 8/8 stim, 3/8 assoc
- seed 46: 6/8 stim, 1/8 assoc
- **Total: 35/40 = 87.5% stim-recall, 11/40 = 27.5% assoc-recall**

Stim-recall is real (chance 8.3%, observed 87.5%). Assoc-recall is
barely above chance (20%).

Documented in [`2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`](2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md).

### Phase 4: multitag mechanism (afternoon)

Insight: the engram-stim-recall mechanism is per-tag. To get cue-
driven retrieval (user types word → system retrieves associates),
we can stim every engram tag containing the cue and aggregate
`lang_output` cosines across all matching tags.

Implemented in `compose_concept_chat.py` as `handle_multitag()`.
Built a standalone evaluator `multitag_eval.py`.

Multi-seed test (5 seeds × 8 cues with 2 associates each):
- seed 42: 7/8 FULL, 8/8 PARTIAL
- seed 43: 7/8 FULL, 8/8 PARTIAL
- seed 44: 8/8 FULL, 8/8 PARTIAL
- seed 45: 8/8 FULL, 8/8 PARTIAL
- seed 46: 6/8 FULL, 8/8 PARTIAL
- **Total: 36/40 = 90% FULL, 40/40 = 100% PARTIAL**

Chance for FULL (top-2 of 15 covering 2 specific words): ~0.95%.
**Result is ~95× chance.**

Documented in [`2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md`](2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md).

### Phase 5: scaling (afternoon → evening)

Tested 3-associate cues (12 pairs encoded, every cue has 3 associates):
- 5 seeds × 8 cues at top-N=3
- FULL: 29/40 = 72.5% (chance ~3.6%, so 20× chance)
- PARTIAL: 40/40 = 100%

Some degradation from 90% → 72.5% at higher graph density but
PARTIAL stays perfect.

Tested v17 28-word vocab (4 seeds × 9 cues, 12 pairs):
- FULL: 0/36 = 0% (chance 0.31%)
- PARTIAL: 15/36 = 41.7% (chance 7.7%)

v17 28-word fails. Phase 1 weakness (50% PASS vs v16's 81%) is
the bottleneck. Stronger encoding (1000 events + teacher 1000 pA)
doesn't fix it.

### Phase 6: webapp integration (evening)

Added `engram_stim_recall` + `multitag_cue_recall` presets to
`webapp/server.py` PRESETS dict + PRESET_RUNNERS routing. Added
UI options to `webapp/static/index.html` in new "SEMANTIC MEMORY"
group. Added category to `webapp/static/ui.js` `categorizeExperiment`.
Added RETRACTED status to `tests/test_webapp_server.py` valid_statuses.

All 58/58 webapp tests pass.

## What's validated (production capabilities)

| Capability | Result | Recipe |
|---|---|---|
| Engram-tag stim-recall | 87.5% multi-seed | v16 + teacher 500 + enc 500 |
| Multi-tag cue retrieval (2 assoc) | 90% FULL / 100% PARTIAL multi-seed | v16 + teacher 500 + enc 500 |
| Multi-tag cue retrieval (3 assoc) | 72.5% FULL / 100% PARTIAL multi-seed | v16 + teacher 500 + enc 500 |
| Chat REPL with /stim and multitag modes | Operational, 90% multi-seed default | compose_concept_chat.py |

Plus prior validated capabilities (unaffected by bug):
- Tier 1 4-word direction: 6/6 BIDIR multi-seed
- Tier 2.1 8-word synonym: 6/6 BIDIR multi-seed
- Synonym32 (32-word multi-language) chat_speak: 100% A→W seed 42
- Phase 1.3 hippocampus consolidation: 3/3 strict anti-cheat multi-seed
- P5 ventral semantic: 6/6 multi-seed
- Encoding-axis 64-word: 3/3 GO unanimous

## What was retracted

- "Semantic Memory + Transitive Inference: 65% direct + 90% chained"
- "TRANSITIVE SEMANTIC INFERENCE - 90% multi-seed"
- "Concept-Concept Pool-Firing Readout - 65% multi-seed"
- "v18 25% top-1 architectural plateau" (was always 0/8 at corrected arch)
- "v19 gate-frozen Phase 1 superiority" (also 0/8 at corrected arch)

## Open boundaries

1. **v17 28-word vocab scaling**: 0% FULL. Need stronger Phase 1
   training (target 80% PASS). Retrain in flight at 400 events.

2. **Higher-density association graph**: 90% FULL at 2 assoc, 72.5%
   at 3 assoc. Likely drops further at 4+ associates. Capacity
   ceiling not fully mapped.

3. **Hippocampus consolidation of engrams**: catalog D.13 pattern
   completion + D.14 ensemble tagging combined. Not yet tested for
   multitag. Could stabilize tags across longer time spans.

4. **Multi-turn conversation**: chat REPL currently per-message.
   Multi-turn state (the system remembers prior exchanges) needs
   additional infrastructure.

## Lessons learned

1. **Module-level side-effect imports are dangerous.** A patch at
   module-load time silently corrupts every transitive importer.
   Patches should be opt-in via explicit function calls or wrapper
   scripts, never imported for side effects.

2. **`load_checkpoint` accepts architecture mismatches silently.**
   Weights are loaded into wrong positions without raising. A
   sanity check on `n_neurons == loaded["num_neurons"]` would have
   caught this earlier.

3. **Cross-pool plastic pathways (v18/v19) don't help.** Even with
   isolated STDP (gate frozen during Phase 1, opened during
   encoding) and strong teacher current, 500-event encoding doesn't
   grow cross-pool weights to functional magnitude.

4. **Tonegawa engram tagging is the strong substrate.** The catalog
   D.14 ensemble-binding mechanism is robust at 87.5%. Layered with
   tag-name indexing (multitag), it delivers cue-driven retrieval.

5. **Cue-only "associative recall" via raw pool firing fails at the
   corrected architecture.** 27.5% multi-seed is barely above
   chance 20%. The previous "65% pool-firing readout" was bug-
   driven.

## Files added today

- `research/runners/multitag_eval.py` — new multi-seed evaluator
- `research/runners/multitag_eval_v17.py` — v17 vocab wrapper
- `research/runners/compose_concept_chat.py` — three-mode chat REPL
- `research/runners/v19_multiseed.ps1` — multi-seed launcher (kept
  for future runs even though v19 architecture didn't help)
- `research/findings/2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`
- `research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`
- `research/findings/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md`
- `research/findings/2026-05-14-session-summary.md` (this file)
- ~20 result JSONs in `research/findings/raw/g11_bg/{compose_concept_strict,multitag_eval}/`

## Commits today

- `cc19cf6` CRITICAL fix: compose_concept_* architecture-mismatch retraction
- `e52ea6b` feat(engram-stim-recall): 87.5% multi-seed semantic memory VALIDATED
- `13c9b3b` feat(chat-repl): add /stim tag-recall mode (87.5% validated capability)
- `4ff9d38` feat(multitag-cue-retrieval): 90% FULL / 100% PARTIAL multi-seed semantic conversation
- `5350866` test(multitag): 3-associate scaling - 72.5% FULL / 100% PARTIAL multi-seed
- `76b2bc6` feat(webapp): surface multitag_cue_recall + engram_stim_recall presets
- `7d56e8a` test(multitag-v17): 28-word vocab scaling - 0% FULL, 41.7% PARTIAL multi-seed
- `3d7bcbe` test(multitag-v17): stronger encoding doesn't fix 28-word scaling

## Next direction

v17 retrain with 400 events in flight (~45-60 min). If Phase 1
reaches 80%+, retest multitag at 28-word. If still ~50%, architecture
needs deeper rework (hippocampus consolidation, larger n_lang_input,
different pool dynamics).
