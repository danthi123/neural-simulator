# CRITICAL: compose_concept_* architecture-mismatch bug (2026-05-14)

## TL;DR

The "real semantic memory" claims from 2026-05-13/14 (25% top-1 strict,
65% top-3 / pool-firing readout, 90% transitive inference multi-seed)
were **measurement artifacts caused by an architecture-mismatch bug**.

When measured with corrected (matching) bridge architecture:
- **Strict top-1 (v16): 0/8 (chance ~6%)** — was claimed 25%
- **Chain transitive (v16 seed 42): 1/4 (25%)** — was claimed 90% multi-seed
- **v18 cross-pool pathways: 0/8 (vs corrected v16's 0/8)** — no improvement

The Tier 1 / Tier 2.1 / Phase 1.3 / P5 results documented in CLAUDE.md
were validated via DIFFERENT pipelines (bio_three_factor,
validate_ventral_semantic, consolidation_trainer) which do NOT have
this bug. Those remain real and unaffected.

## The bug

`research/runners/compose_engram_demo_v2.py` was created to support
v17 (28-word vocab) experiments. At module import it monkey-patches
the `NOUN_VOCAB`, `VERB_VOCAB`, `ADJECTIVE_VOCAB` dicts in
`concept_pool_demo` to v17's extended vocab:

```python
cpd_v1.NOUN_VOCAB = NOUN_VOCAB     # 8 nouns instead of 4
cpd_v1.VERB_VOCAB = VERB_VOCAB     # 8 verbs instead of 4
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB  # 8 adjectives instead of 4
```

In `compose_concept_engram.py` at line 27 (pre-fix), this v2 module
was imported at the TOP-LEVEL purely for the side effect:

```python
# Patch v17 vocab for extended-vocab compatibility
import research.runners.compose_engram_demo_v2  # noqa: F401
```

Any other runner that imported `encode_concept_pair` from
`compose_concept_engram` transitively triggered the v17 patch — even
when the bridge under test was a v16 (16-pool) bridge.

Affected eval pipelines (all imported `compose_concept_engram`):
- `compose_concept_strict.py`
- `compose_concept_increment.py`
- `compose_concept_pool_readout.py`
- `compose_concept_chain_test.py`
- `compose_concept_chat.py`

When these scripts called `cpd.build_concept_bridge(...)`, the bridge
skeleton was built with 28 pools (10368 neurons, 10.3M synapses)
instead of v16's 16 pools (7680 neurons, 5M synapses). Then
`bridge.load_checkpoint(<v16-bridge>)` loaded the 16-pool weights
into the 28-pool skeleton, with most weights silently mismatched.

The mismatched weights produced a pseudo-random firing pattern that
happened to score 25% top-1 / 65% top-3 — but the signal was
inflated by **the random pool-mapping introduced by the mismatch**,
not by genuine learned cross-pool weights.

## Evidence

### Strict top-1 on seed 42

| Bridge | Old (mismatched arch) | Fixed (matched arch) |
|---|---|---|
| v16 (no cross-pool pathways) | 2/8 (25%) | 0/8 (0%) — chance ~6% |
| v18 (cross-pool pathways) | 2/8 (25%) | 0/8 (0%) |
| v19 (gate-frozen Phase 1) | 2/8 (25%) | 0/8 (0%) |

All three produce **the same exact per-pair top-1 outputs** with
corrected architecture, indicating the cross-pool weights aren't
firing strongly enough to differentiate bridges with different
Phase 1 training. The cross-pool architecture isn't doing anything
measurable.

### Chain (transitive) on seed 42

| Bridge | Old (mismatched arch) | Fixed (matched arch) |
|---|---|---|
| v16 | 90% (claimed multi-seed) | 1/4 (25%) on seed 42 |

The previous "90% transitive inference" result was the bug operating
on multiple seeds in the same direction.

## Fix

Removed the `import research.runners.compose_engram_demo_v2` from
`compose_concept_engram.py`. The v17 vocab patch is now opt-in via
the explicit v17 wrapper scripts (`compose_engram_demo_v2` itself,
or future v17-specific runners). Default 16-pool architecture is
preserved for everything that imports `encode_concept_pair`.

Also added `--enable-cross-pool-concept-pathways` flag to
`compose_concept_strict.py`, `compose_concept_increment.py`, and
`compose_concept_engram.py` so they build bridges with matching
architecture when loading v18/v19 checkpoints.

## What's still real

The compose_concept_* pipeline measurements are now corrected. The
following Tier 1+ results are NOT affected by this bug because they
use independent pipelines:

- **Tier 1 (4-word direction): 6/6 BIDIR multi-seed** —
  `bio_three_factor` runner
- **Tier 2.1 (8-word synonym): 6/6 BIDIR multi-seed** — same runner
- **Synonym32 chat_speak: 100% A→W single seed** —
  `chat_speak_synonym_demo`
- **P5 ventral semantic comprehension: 6/6 multi-seed PASS** —
  `validate_ventral_semantic`
- **Phase 1.3 hippocampus consolidation: 3/3 strict anti-cheat
  multi-seed** — `consolidation_trainer` + `consolidation_eval`

These pipelines use bridges built by different runners
(`bio_three_factor`, `consolidation_trainer`, `validate_ventral_semantic`)
that do not import `compose_concept_engram`. They are independent and
remain valid.

## What's no longer claimed

- "Real semantic memory" 65% pool-firing readout (compose_concept)
- "Transitive inference" 90% multi-seed (chain_test)
- "v18 25% top-1 architectural plateau" — was always 0/8 fixed
- v19 "gate-frozen Phase 1" superiority — also 0/8 fixed

The architectural exploration v18 → v19 still has merit as
**runtime-gate-management infrastructure** for future experiments,
but does not produce measurable semantic association at 200-event
encoding scale with current cross-pool density (0.10).

## Lessons

1. **Module-level side-effect imports are dangerous.** A patch at
   module-load time silently corrupts every transitive importer.
   Patches should be opt-in via explicit function calls or wrapper
   scripts, never imported for side effects.

2. **Load checkpoint silently accepts architecture mismatches.** When
   the bridge skeleton has more pools/synapses than the checkpoint,
   weights are loaded into wrong positions without error. The
   architecture sanity check `n_neurons == loaded["num_neurons"]`
   exists but doesn't prevent partial-load.

3. **The previous summary's "user pivot to concept-concept work"
   was correct in spirit but measured with corrupted instrument.**
   The compose_concept_* line genuinely tested a different
   architectural hypothesis (concept→concept bindings beyond
   word→motor routing), but the measurements were bug artifacts.

4. **The capability_status.json and CLAUDE.md "ARCHITECTURAL PIVOT"
   section need correction.** The Pool-Firing Readout 65% and
   Transitive Inference 90% claims should be retracted pending
   re-verification with corrected pipeline.

## Next steps

1. Revert capability_status.json + CLAUDE.md to reflect the bug fix.
2. Push the bug-fix commit (compose_concept_engram no longer imports
   v2 patch, eval runners gain --enable-cross-pool flag).
3. Decide whether to invest further in compose_concept-line or
   pivot back to bio_three_factor + chat_speak_synonym_demo which
   has the validated multi-seed BIDIR capability.

## Files changed

- `research/runners/compose_concept_engram.py` — removed v2 patch
  import; added --enable-cross-pool-concept-pathways flag
- `research/runners/compose_concept_strict.py` — added flag
- `research/runners/compose_concept_increment.py` — added flag
- `research/runners/concept_pool_demo.py` — added Phase 1 gate close
  for cross_pool_concept (v19 logic, retained for future use)
- `research/findings/raw/g11_bg/concept_pool_demo/seed42_v19.simstate.h5`
- `research/findings/raw/g11_bg/compose_concept_strict/seed42_v16_strict_fixed.json`
- `research/findings/raw/g11_bg/compose_concept_strict/seed42_v18_strict_fixed.json`
- `research/findings/raw/g11_bg/compose_concept_strict/seed42_v19_strict_fixed.json`
- `research/findings/raw/g11_bg/compose_concept_strict/seed42_v16_chain_fixed.json`
