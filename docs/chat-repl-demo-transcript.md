# Concept Chat REPL — End-to-end demo

Full conversational interaction with the neural simulator's semantic
memory system. Validated 90% FULL / 96.7% accuracy multi-seed on 2026-05-14.

Configuration: v16 16-pool bridge, seed 44, no pre-loaded pairs (user
teaches everything from scratch).

## Transcript

```
============================================================
CONCEPT CHAT
Vocab: ['apple', 'river', 'dog', 'cat', 'go', 'come', 'stop', 'look',
        'big', 'small', 'hot', 'cold']
Learned associations: []
Encoded tags: []
============================================================

> vocab
  vocab: ['apple', 'river', 'dog', 'cat', 'go', 'come', 'stop', 'look',
          'big', 'small', 'hot', 'cold']

# Teach 7 associations
> remember apple is big
  [remembered: apple_big]
> remember apple is hot
  [remembered: apple_hot]
> remember cat is big
  [remembered: cat_big]
> remember cat is hot
  [remembered: cat_hot]
> remember dog is small
  [remembered: dog_small]
> remember dog is cold
  [remembered: dog_cold]
> remember river is cold
  [remembered: river_cold]

> tags
  tags: ['apple_big', 'apple_hot', 'cat_big', 'cat_hot',
         'dog_small', 'dog_cold', 'river_cold']

# Query — retrieve both associates of apple
> what is apple
  [multitag, leverages 87.5% stim-recall per tag]
  matched 2 tag(s): ['apple_big', 'apple_hot']
  top-5 associates:
    hot   = 0.259 via apple_hot  **    # 0.10+ = above noise
    big   = 0.204 via apple_big  **
    cat   = 0.059 via apple_big
    go    = 0.059 via apple_big
    stop  = 0.059 via apple_big

# Cat has the same trained associates as apple (clustered concepts)
> what is cat
  matched 2 tag(s): ['cat_big', 'cat_hot']
  top-5: [hot=0.33, big=0.13, ...]    # both correct

# Dog has different associates (small, cold)
> what is dog
  matched 2 tag(s): ['dog_small', 'dog_cold']
  top-5: [small=0.32, cold=0.23, ...] # both correct

# Yes/no questions
> is apple big
  YES: 'apple' is bound to 'big' (tag apple_big, both in lang_output top-5)

> is apple cold        # Never trained — should be NO
  NO: no tag binding 'apple' and 'cold' (checked apple_cold, cold_apple)

> is cat hot
  YES: 'cat' is bound to 'hot' (tag cat_hot, both in lang_output top-5)

# Compositional reasoning — what's both apple AND cat?
> what is apple and cat
  [intersection] cue=(apple AND cat)
    hot   = min(0.26, 0.34) = 0.26      # shared trained association
    big   = min(0.20, 0.16) = 0.16      # shared trained association
    stop  = min(0.06, 0.05) = 0.05      # noise floor

> what is dog and river
  [intersection] cue=(dog AND river)
    cold  = min(0.26, 0.09) = 0.09      # the only shared association
    big   = min(0.06, 0.05) = 0.05
    stop  = min(0.06, 0.05) = 0.05

# Forget (unlearn) an association
> forget apple_hot
  [forgot: apple_hot]

# After forgetting, "apple" only retrieves big
> what is apple
  matched 1 tag(s): ['apple_big']
  top-5: [big=0.21, go=0.07, ...]      # hot no longer above noise

# Verify the forget worked
> is apple hot
  NO: no tag binding 'apple' and 'hot' (checked apple_hot, hot_apple)
```

## What was demonstrated

The REPL exercises the full conversational capability:

1. **Vocabulary introspection** (`vocab`)
2. **Runtime encoding** (`remember a is b`) — 7 pairs learned
3. **Tag listing** (`tags`) — see what's stored
4. **Direct retrieval** (`what is X`) — 90% FULL multi-seed:
   - apple → big + hot (both trained associates returned)
   - cat → big + hot (same cluster)
   - dog → small + cold (different cluster)
5. **Yes/no verification** (`is a b?`) — 96.7% multi-seed accuracy:
   - is apple big → YES
   - is apple cold → NO (correctly identifies unlearned binding)
   - is cat hot → YES
6. **Compositional intersection** (`what is a and b`) — 90% multi-seed:
   - apple AND cat → hot, big (both shared associations)
   - dog AND river → cold (their only shared association)
7. **Runtime deletion** (`forget tag`) — unlearns:
   - forget apple_hot → tag deleted
   - subsequent `what is apple` no longer includes hot
   - subsequent `is apple hot?` → NO

## Performance

- Loading bridge: ~5s (one-time)
- `remember a is b`: ~5s per pair (500 STDP events + teacher current)
- `what is X` / multitag: ~1s per query (one stim per matching tag)
- `is a b?` / yes/no: ~0.5s (tag check + optional confidence stim)
- `what is a and b` / intersection: ~2s (two multitags then intersect)
- `forget tag`: instant

## Reliability summary (multi-seed validated)

| Operation | Multi-seed reliability | Chance baseline |
|---|---|---|
| Engram stim-recall (per-tag) | 87.5% | ~8% |
| Multitag cue retrieval (FULL) | 90% | ~1% |
| Compositional intersection (FULL) | 90% | <1% |
| Yes/no questions (accuracy) | 96.7% | 50% |
| Capacity precision @ 20 pairs | 98.8% | ~25% |

## How to run

```bash
# 1. Train v16 bridge (one-time, ~18 min)
python -m research.runners.concept_pool_demo --seed 44 \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --save-bridge bridges/v16/seed44.simstate.h5

# 2. Launch interactive chat (or scripted via --scripted)
python -m research.runners.compose_concept_chat \
    --load-bridge bridges/v16/seed44.simstate.h5 \
    --seed 44 \
    --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 \
    --n-words-for-orthogonal 16 --encoding-steps 500 --sparsity 0.05 \
    --balanced-teacher-pA 500.0 \
    --pairs ""    # empty = teach from scratch
```

## Architectural foundation

- **Engram tagging** (Tonegawa, catalog D.14): bind co-firing neurons
  across concept pools into named ensembles
- **Multi-tag aggregation**: leverage per-tag 87.5% stim-recall by
  combining results across all tags containing the cue
- **Tag-existence index**: O(1) lookup for yes/no queries
- **STDP + teacher current**: 500 events with 500 pA bilateral teacher
  during encoding produces stable engrams

All operations work on the standard v16 16-pool architecture (4 motor
+ 4 noun + 4 verb + 4 adjective). No special pretraining required
beyond the standard concept_pool_demo Phase 1 training.

## Boundary

The v17 28-pool architecture has Phase 1 weakness that limits
multitag to ~42% PARTIAL / 0% FULL. v16 16-word vocab is the
current proven ceiling for high-reliability conversation. See
[`CLAUDE.md`](../CLAUDE.md) "v17 28-word scaling LIMIT" section.
