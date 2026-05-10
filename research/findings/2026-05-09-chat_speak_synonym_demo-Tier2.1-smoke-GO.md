# chat_speak_synonym_demo seed 42 smoke — GO at threshold

**Date:** 2026-05-09 22:03 EDT
**Status:** ✅ GO (marginal — 50% A2W any-synonym, exactly at threshold)
**Run ID:** a7647c1afb58
**Wall clock:** 1662s (27.7 min)
**Result file:** `research/findings/raw/g11_bg/g11_seed42_chat_speak_synonym_demo_a7647c.json`

## Architecture

Tier 2.1 v4 scale-up (matches `chat_synonym_demo`):
- n_lang_input = 4096
- n_motor_per_action = 1000
- n_motor_fs_per_action = 120
- biological + embodied-Hebbian + topographic-prior + motor_FS + NMDA
- 8-word synonym vocab: {N: [north, up], E: [east, right], S: [south, down], W: [west, left]}
- 400 events/direction (1600 total, ~50/50 primary/synonym)

## Result — single-seed smoke

```
W→A regression (8-word):  25.0% (4/16)  PASS (>= 25% chance)
A→W any-synonym:          50.0% (2/4)   PASS (>= 50% threshold)
A→W primary-only:         50.0% (2/4)
A→W synonym-only:           0% (0/4)
Verdict:                  GO  (smoke threshold met)
```

### Per-direction A→W

| Action | Expected | Predicted | Tag | Sim |
|--------|----------|-----------|-----|-----|
| N | [north, up] | down | wrong | 0.08 |
| E | [east, right] | east | ✅ primary | 0.05 |
| S | [south, down] | south | ✅ primary | 0.11 |
| W | [west, left] | right | wrong | 0.07 |

### Per-word W→A

| Word (action) | Trials | Correct | Acc |
|---------------|--------|---------|-----|
| north (N) | 2 | 0 | 0% |
| up (N) | 2 | 0 | 0% |
| east (E) | 2 | 1 | 50% |
| right (E) | 2 | 0 | 0% |
| south (S) | 2 | 1 | 50% |
| down (S) | 2 | 0 | 0% |
| west (W) | 2 | 2 | 100% |
| left (W) | 2 | 0 | 0% |

**Primary mean: 50%, synonym mean: 0%.** Synonyms completely fail
both W→A and A→W on this seed.

## Observations

1. **Synonym binding failed entirely on seed 42** in BOTH directions.
   The architecture learned primary→action and action→primary, but
   no synonym→action or action→synonym pairing transferred.

2. **Cross-action synonym confusion** in 2 of 4 A→W trials:
   - motor_N produced "down" (south's synonym)
   - motor_W produced "right" (east's synonym)

   Both wrong predictions are SYNONYMS for OTHER actions, not
   primaries for other actions. Suggests the network reads motor
   activity as some kind of distributed representation that overlaps
   with secondary-synonym drive patterns more than primary patterns
   when the binding fails.

3. **Confidence is very low across the board** (sim 0.05-0.11). Top-1
   wins are not strong wins. Multi-seed will tell us if this seed is
   an outlier or representative.

4. **W ("west") was the strongest** primary in W→A (100%) but its
   motor pool's A→W output was wrong. Asymmetric — reading west the
   word produces motor_W reliably, but driving motor_W reads back to
   "right" (east's synonym), not "west" or "left". Production-side
   does NOT mirror reception-side at this seed.

5. **Tier 2.1 BREAKTHROUGH paper claimed 6/6 aligned at A→W mean
   63.7%** on the same architecture. Our seed 42 hits only 50%
   any-synonym. Either:
   - Single-seed variability (consistent with Tier 1 chat_speak_demo
     where seed 42 hit 75% but seed 102 hit 25% — same arch, same
     training)
   - Some configuration drift between Tier 2.1 BREAKTHROUGH (2026-05-06)
     and now (2026-05-09)
   - The chat_speak_synonym_demo's verdict threshold (any-synonym
     ≥ 50%) is already passed at exactly 50%, and Tier 2.1
     BREAKTHROUGH measured 6/6 alignment at a different rate

## Decision: launch 6-seed multi-seed

Single-seed at threshold doesn't strongly validate or refute. Multi-seed
will:
- Confirm whether Tier 2.1 BREAKTHROUGH's 63.7% mean reproduces in
  the production-side A→W direction, OR
- Confirm seed 42 is the outlier and the typical case is 60-80% A→W
- Surface whether the synonym (vs primary) collapse is seed-specific

Wrapper: `scripts/multiseed_chat_speak_synonym_demo.sh` (seeds 42, 43,
44, 100, 101, 102; ETA ~3 hrs). Uses `chat_demo_aggregate` chat_speak
branch.

After multi-seed: 16-word smoke (`consolidation_synonym_16word_scaled_smoke`,
~35 min) tests capacity rule extension at the next vocab tier.

## Next steps in chain

1. ✅ chat_speak_synonym_demo seed 42 smoke (GO)
2. ⏳ 6-seed multi-seed via `scripts/multiseed_chat_speak_synonym_demo.sh`
3. ⏳ 16-word smoke `consolidation_synonym_16word_scaled_smoke`
4. ⏳ Decision: 16-word medium (if smoke GO) or scale further (if NO-GO)
