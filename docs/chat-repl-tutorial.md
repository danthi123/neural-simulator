# Concept Chat REPL Tutorial

The `compose_concept_chat.py` runner provides an interactive REPL for
genuine concept-concept semantic conversation with the simulator's
neural network. Validated 90% FULL / 100% PARTIAL multi-seed retrieval
(2026-05-14).

## Quick start

1. **Train a v16 concept-pool bridge** (one-time, ~18 min):

```bash
python -m research.runners.concept_pool_demo --seed 42 \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --save-bridge bridges/v16/seed42.simstate.h5
```

2. **Launch the chat REPL**:

```bash
python -m research.runners.compose_concept_chat \
    --load-bridge bridges/v16/seed42.simstate.h5 --seed 42 \
    --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 \
    --n-words-for-orthogonal 16 --encoding-steps 500 --sparsity 0.05 \
    --balanced-teacher-pA 500.0 \
    --pairs "apple:big,dog:small"
```

The `--pairs` flag pre-loads some initial associations. You can also
omit it and add associations interactively via `remember`.

## Commands

### Querying

| Command | Behavior |
|---|---|
| `<word>` | Multi-tag retrieval (recommended). Auto-stim all tags containing the word, return top-5 associates by cosine. **90% FULL multi-seed.** |
| `what is <word>` | Same as `<word>` — natural-language alias. |
| `tell me about <word>` | Same as `<word>`. |
| `what is <a> and <b>` | **Compositional intersection** — words associated with BOTH a and b. **90% FULL multi-seed.** |
| `<a> and <b>` | Shortcut for intersection query. |
| `is <a> <b>?` | **Yes/no question** — does the system know (a, b) is bound? |
| `tell me more` | Show next-best associates of the last query (multi-turn drill-down). |
| `/stim <tag>` | Direct stimulation of a specific tag. Useful when you know the tag name. **87.5% multi-seed per-tag.** |
| `/cue <word>` | Raw pool firing rank (experimental, ~28% reliability). |

### Encoding

| Command | Behavior |
|---|---|
| `remember <a> is <b>` | Encode a new association at runtime. Uses 500 events + teacher 500 pA. Takes ~5s. |
| `remember <a> <b>` | Space-separated form. |
| `forget <tag>` | Delete an engram tag at runtime. |
| `save [path]` | **Persist bridge + tags to checkpoint** for cross-session memory. |

### Introspection

| Command | Behavior |
|---|---|
| `/tags` or `tags` | List all encoded engram tags. |
| `/vocab` or `vocab` | List available concept words. |
| `quit` or `exit` | Exit REPL. |

## Vocabulary

The v16 architecture supports 16 concept words:

- **Motors**: north, east, south, west
- **Nouns**: apple, river, dog, cat
- **Verbs**: go, come, stop, look
- **Adjectives**: big, small, hot, cold

`remember` and `what is` operate on the non-motor concept words
(motor pools are excluded from engram region_filter — they're for
the navigation pipeline, not semantic memory).

## Example session

```
> vocab
  vocab: ['apple', 'river', 'dog', 'cat', 'go', 'come', 'stop', 'look', 'big', 'small', 'hot', 'cold']

> remember apple is big
  [remembered: apple_big]

> remember apple is hot
  [remembered: apple_hot]

> remember cat is big
  [remembered: cat_big]

> remember cat is hot
  [remembered: cat_hot]

> is apple big
  YES: 'apple' is bound to 'big' (tag apple_big, both in lang_output top-5)

> is apple cold
  NO: no tag binding 'apple' and 'cold' (checked apple_cold, cold_apple)

> what is apple
  matched 2 tag(s): ['apple_big', 'apple_hot']
  top-5: [big=0.20, hot=0.16, cat=0.11, ...]

> tell me more
  [more for 'apple']
  stop = 0.07 via apple_big

> what is apple and cat
  [intersection]
  hot     = min(0.20, 0.31) = 0.20
  big     = min(0.20, 0.16) = 0.16

> forget apple_big
  [forgot: apple_big]

> what is apple
  matched 1 tag(s): ['apple_hot']
  top-5: [hot=0.27, big=0.07, ...]

> remember apple is cat
  [remembered: apple_cat]

> what is apple
  matched 2 tag(s): ['apple_big', 'apple_cat']
  top-5 associates:
    big      = 0.20 via apple_big    **
    cat      = 0.17 via apple_cat    **
    stop     = 0.06 via apple_cat
    go       = 0.06 via apple_big
    come     = 0.06 via apple_big

> remember dog is small
  [remembered: dog_small]

> remember dog is river
  [remembered: dog_river]

> tell me about dog
  matched 2 tag(s): ['dog_small', 'dog_river']
  top-5 associates:
    river    = 0.43 via dog_river    **
    small    = 0.34 via dog_small    **
    cat      = 0.06 via dog_small
    big      = 0.06 via dog_small
    look     = 0.05 via dog_small

> quit
  Done.
```

In this session the user teaches the system 4 associations and the
system retrieves them with high confidence. The `**` markers indicate
which associates are above the noise floor (cosine > 0.10).

## Performance notes

- **Initial bridge load**: ~5-10s (one-time per REPL session).
- **`remember`**: ~5s per pair (encoding takes 500 simulation steps).
- **`what is`** or plain word: ~0.7s per query (one stim per
  matching tag).
- **`/stim`**: ~0.4s.

## Mechanism

`remember a is b` calls `encode_concept_pair(bridge, a, b, "a_b")`:

1. Drive `lang_input(a)` AND `lang_input(b)` simultaneously for 500
   simulation steps.
2. Apply teacher current (500 pA) to both `concept_pool_A` and
   `concept_pool_B` so they fire reliably during co-firing.
3. Engram-tag the top-100 most-active neurons in concept pools
   (excluding motor pools via region_filter).

`what is a` calls `handle_multitag(a)`:

1. Find all encoded tags containing `a`.
2. For each, `stimulate_tag(tag, drive_pA=1500)` and read
   `lang_output` cosine to all 16 vocab words.
3. Aggregate (max score per associate across all matching tags).
4. Return top-5 by score.

The multi-tag aggregation exploits the per-tag 87.5% stim-recall
reliability. With 2 matching tags, both associates appear in
`lang_output` top-5 with 90% probability.

## Cross-session persistence

The chat REPL supports true long-term conversational memory via the
`save` command. The bridge's engram-tag state is persisted to the
HDF5 checkpoint:

```
# Session 1
> remember apple is big
> remember cat is hot
> save              # writes to current bridge path
  [saved bridge + 2 engram tag(s) to bridges/v16/seed44.simstate.h5]

# Exit, come back later
> python -m research.runners.compose_concept_chat \
    --load-bridge bridges/v16/seed44.simstate.h5 ...
  [restored 2 engram tag(s) from checkpoint: ['apple_big', 'cat_hot']]

> what is apple        # Still knows! 90% retrieval
  matched 1 tag(s): ['apple_big']
  top-5: [big=0.21, ...]
```

This means the user can:
- Teach the system new associations
- Save before exiting
- Return tomorrow / next week and continue
- The system's "memory" persists biologically in the bridge

## Limitations (2026-05-14)

- **16-word vocab**: validated. 28-word (v17) doesn't yet retrieve
  reliably because Phase 1 is weaker at higher pool count.
- **Per-cue capacity**: at 8 pairs (2 associates per cue), FULL
  retrieval is 90%. At 12 pairs (3 associates), drops to 72.5%.
  At 16 pairs (4-5 associates), 65%. At 20 pairs (5-6 associates),
  98.8% PRECISION (every returned word is real) but FULL drops to
  ~12% because top-N can't hold all associates.
- **Multi-turn within-session**: `tell me more` extends a query to
  next-best associates. Beyond that, queries are mostly independent.
- **Cross-session via save**: tags persist between REPL sessions.
- **Tag overlap**: encoding `apple_big` and `apple_cat` creates two
  separate tags with overlapping neurons. Heavy overlap may
  eventually cause tag interference.
- **Soft transitive inference**: 2nd-degree neighbors appear at
  lower cosine scores (45% multi-seed at top-5, chance 33%).

## Further reading

- Multi-seed validation: [`research/findings/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md`](../research/findings/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md)
- Stim-recall baseline: [`research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`](../research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md)
- Bug retraction history: [`research/findings/2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`](../research/findings/2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md)
- Biological basis (catalog D.14): Tonegawa engram tagging
