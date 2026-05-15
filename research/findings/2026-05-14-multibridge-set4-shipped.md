# Set 4 trained — 48-word multi-bridge vocab milestone

## TL;DR

Set 4 finished training 2026-05-14 23:35 EST. **13/16 PASS (81%)** at
seed 42. Wall clock ~18.5 min. Set 5 auto-launched.

Combined with sets 1-3, the multi-bridge chat REPL now supports
**48 unique concept words**.

## Per-word verdicts (seed 42, set 4)

| Word | Pool | Target | Max off | Ratio | Verdict |
|------|------|--------|---------|-------|---------|
| north | motor_N | 1.355 | 1.025 (noun_KEY) | 1.32x | PASS |
| east | motor_E | 1.045 | 0.945 (adj_FULL) | 1.11x | PASS |
| south | motor_S | 1.260 | 0.905 (adj_SAD) | 1.39x | PASS |
| west | motor_W | 0.680 | 0.525 (adj_HAPPY) | 1.30x | PASS |
| person | noun_PERSON | 0.860 | 0.825 (motor_E) | 1.04x | PASS |
| baby | noun_BABY | 1.090 | 0.800 (verb_PUSH) | 1.36x | PASS |
| ball | noun_BALL | 1.300 | 0.925 (motor_E) | 1.41x | PASS |
| key | noun_KEY | 1.530 | 1.000 (adj_EMPTY) | 1.53x | PASS |
| open | verb_OPEN | 1.120 | 1.120 (noun_KEY) | 1.00x | FAIL |
| close | verb_CLOSE | 0.740 | 0.745 (adj_EMPTY) | 0.99x | FAIL |
| push | verb_PUSH | 1.460 | 1.065 (verb_CLOSE) | 1.37x | PASS |
| pull | verb_PULL | 1.010 | 0.755 (verb_CLOSE) | 1.34x | PASS |
| happy | adj_HAPPY | 0.725 | 0.815 (motor_N) | 0.89x | FAIL |
| sad | adj_SAD | 1.120 | 1.065 (adj_HAPPY) | 1.05x | PASS |
| full | adj_FULL | 1.295 | 0.915 (motor_S) | 1.42x | PASS |
| empty | adj_EMPTY | 1.015 | 0.910 (motor_N) | 1.12x | PASS |

Failures:
- `open` collides with `noun_KEY` (1.00x ratio — coincidence given
  thematic relationship key/open)
- `close` collides with `adjective_pool_EMPTY` (0.99x — similar
  thematic clue)
- `happy` loses to `motor_N` (anti-target bias, seen across all 4
  trained bridges as the dominant failure mode)

## Per-bridge PASS ladder so far

| Bridge | Vocab | PASS | Notes |
|--------|-------|------|-------|
| Set 1 (v16) | apple/river/dog/cat/... | 11-12/16 | Validated 90% multi-seed multitag |
| Set 2 | tree/bird/sun/moon/... | 11/16 | Phase 1 baseline |
| Set 3 | house/road/fire/water/... | 14/16 | Best per-bridge so far |
| Set 4 | person/baby/ball/key/... | 13/16 | Verb pool (open/close) weaker |
| **Combined** | **48 unique words** | **~50/64** | **78% effective coverage** |

Note: multitag aggregation at retrieval (90% multi-seed FULL on
intra-set associates) compensates for per-word Phase 1 weakness —
the user-visible conversational capability is much higher than
the 50/64 raw Phase 1 PASS rate suggests.

## Now in flight

- Set 5: started 23:35, ~18 min ETA. Brings total to **60 unique
  concept words** across 5 bridges.

## Conversational features shipped tonight

Multi-bridge chat REPL extended with parsing-level features (no
neural retraining required):

1. N-word sentence encoding (3-5 word tags, cross-set partial encoding)
2. Role queries: 'who ate apple?' / 'what did dog ate?'
3. Negation: 'remember the dog is not big' -> NOT_dog_big
4. Yes/no questions with 3-valued logic (YES/NO/UNKNOWN)
5. Conjunctions: 'dog is big and cat is small'
6. Possessives: "apple's color is red" -> color_of_apple_red
7. Pronoun coreference: it/he/she/they -> last_subject
8. Memory management: about X, forget, save
9. Relational queries: 'what is the color of apple?'

83 unit tests pass in ~1.2s, all CPU-only.

## File map

```
research/findings/raw/g11_bg/concept_pool_demo/
  seed42_set2.simstate.h5     (set 2, 11/16, 2026-05-14 PM)
  seed42_set3.simstate.h5     (set 3, 14/16, 2026-05-14 23:17)
  seed42_set4.simstate.h5     (set 4, 13/16, 2026-05-14 23:35)
  seed42_set5.simstate.h5     (set 5, in flight)
```

## Status

- Sets 1, 2, 3, 4: SHIPPED
- Set 5: TRAINING (~18 min ETA)
- Total when complete: 60 unique concept words + 4 motors
- Operational chat REPL: 83 unit tests + 9 conversational features
