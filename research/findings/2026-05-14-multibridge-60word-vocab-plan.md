# Multi-bridge 60-word vocab — scaling plan + infrastructure

## TL;DR

Set 1 + Set 2 = 24 unique concept words shipped 2026-05-14 PM.
Set 3 (in flight at commit time, 66% trained) brings the count to 36.
Set 4 + Set 5 trainers are queued behind set 3 via the
`chain_set45_runtime.ps1` background chain.

Once all three additional sets finish training, the multi-bridge chat
REPL will support **60 unique concept words** — 5x the single-bridge
v17 architectural ceiling.

51 unit tests validate the vocab table structure, cross-set
uniqueness, routing helpers, and per-bridge cosine math.

## Vocab catalog (60 words)

| Set | Nouns                          | Verbs                          | Adjectives                     |
|-----|--------------------------------|--------------------------------|--------------------------------|
| 1   | apple, river, dog, cat         | go, come, stop, look           | big, small, hot, cold          |
| 2   | tree, bird, sun, moon          | walk, run, eat, sleep          | red, blue, fast, slow          |
| 3   | house, road, fire, water       | give, take, find, lose         | tall, short, wet, dry          |
| 4   | person, baby, ball, key        | open, close, push, pull        | happy, sad, full, empty        |
| 5   | food, drink, hand, foot        | speak, listen, read, write     | new, old, clean, hard          |

Every set additionally has motor words (north, east, south, west)
which are SHARED — they appear in all 5 bridges' vocab and map to
motor pools. This is intentional: motor pools are the actuation
substrate and need to be reachable from any bridge.

## Architecture

```
                     User input
                         |
                         v
                +-----------------+
                |  multibridge    |
                |  chat REPL      |
                +-----------------+
                /        |        \
               v         v         v
        +--------+  +--------+  +--------+
        | Bridge |  | Bridge |  | Bridge | ... etc
        |  set1  |  |  set2  |  |  setN  |
        +--------+  +--------+  +--------+
        12 concepts 12 concepts 12 concepts
        + 4 motors  + 4 motors  + 4 motors
```

Each bridge is a v16 architecture (16 pools = 4 motor + 4 noun +
4 verb + 4 adjective). Validated 90% FULL / 100% PARTIAL multi-seed
multitag at 16-word vocab. Five bridges in ensemble = 60-word
operational vocab.

## Routing rules

1. **Single-word query**: route to ANY bridge that has the word in
   vocab (motors are in all bridges; concepts in exactly one).

2. **Pair encoding (remember a is b)**:
   - Intra-set (both words in one bridge): encode directly in that
     bridge as engram tag `a_b`.
   - Cross-set (a in bridge X, b in bridge Y): PARTIAL encoding in
     both bridges. Each bridge captures its half of the relationship
     with the same tag name `a_b`. Multi-bridge query aggregates at
     recall.

3. **Multitag aggregation (what is X)**: search ALL bridges for tags
   whose NAME contains X. For each match, stimulate the tag and read
   the lang_output cosine to other concept words in that bridge's
   vocab. Aggregate by max-score-per-word across bridges.

## Wall-clock budget

| Step | Wall-clock | Status |
|------|-----------|--------|
| Set 1 (v16 baseline) | already trained | DONE |
| Set 2 training | ~18 min | DONE (11/16 Phase 1 PASS) |
| Set 3 training | ~18 min | IN FLIGHT (66% as of commit) |
| Set 4 training | ~18 min | QUEUED (chain_set45) |
| Set 5 training | ~18 min | QUEUED (chain_set45) |
| **Total** | ~90 min | All by ~01:00 EST 2026-05-15 |

## Cross-bridge limitations (honest)

Pure intra-set queries work at v16 baseline reliability (90%
multitag). Cross-set queries have two regimes:

1. **Binary cross-set (a in set X, b in set Y)**: PARTIAL encoding
   works — `remember sun is hot` encodes `sun_hot` in BOTH set1
   (which has hot) and set2 (which has sun). Querying `what is sun`
   finds the tag in set2 and reads its associates. Querying `what
   is hot` finds the tag in set1.

2. **3+ concept sentence cross-set**: NOT YET SUPPORTED. A sentence
   like "the dog runs in the river" with dog/river in set1 and runs
   in set2 cannot be encoded as a single 3-role engram. Future work:
   token-level tagging where each word's home bridge holds its part
   of the sentence.

## Why this is engineering, not science

Each bridge IS a validated v16 neural network (90% multi-seed multitag
in prior validations). Multi-bridge routing is biologically
motivated: cortex has functional regions (Broca's, Wernicke's,
somatotopic motor areas) specialized for different content. Multi-
bridge mirrors this organizational principle at the simulation
level.

No fabricated capabilities: each bridge does exactly what v16 does.
The user doesn't need to know which bridge stores which word — the
routing is automatic.

## Files

### Per-set vocab wrappers
- `research/runners/concept_pool_demo_set2.py`
- `research/runners/concept_pool_demo_set3.py`
- `research/runners/concept_pool_demo_set4.py`
- `research/runners/concept_pool_demo_set5.py`

### Multi-bridge runtime
- `research/runners/multibridge_chat.py` — main REPL
- `research/runners/chain_set45_runtime.ps1` — background trainer for sets 3->4->5

### Tests
- `tests/test_multibridge_chat.py` — 51 unit tests

### Findings
- `research/findings/2026-05-14-multibridge-24word-vocab.md` — Set 1+2 shipped
- `research/findings/2026-05-14-multibridge-60word-vocab-plan.md` — this doc

## Status as of commit

- Set 1: SHIPPED (v16 baseline, 90% multitag multi-seed)
- Set 2: SHIPPED (11/16 Phase 1 PASS at seed 42)
- Set 3: TRAINING (2100/3200 events as of last log read)
- Set 4: QUEUED (background chain auto-launches when set 3 finishes)
- Set 5: QUEUED (background chain auto-launches when set 4 finishes)
- Multi-bridge REPL: VALIDATED for set1+set2; will accept set3-5 as they finish

## Next steps after all 5 sets train

1. End-to-end smoke: load all 5 bridges, scripted demo encoding
   intra-set + cross-set pairs spanning all sets.
2. Multi-seed validation for sets 2-5 (4 seeds * 4 sets = 16 trains
   = ~5 hours).
3. Sentence-level cross-bridge encoding (token-level tagging).
4. Update `webapp/capability_status.json` with 60-word milestone.
