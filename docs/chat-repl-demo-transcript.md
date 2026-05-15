# Concept Chat REPL — Full conversational demo (2026-05-14)

Genuine semantic conversation with neural memory. 7 validated multi-seed
metrics: 87.5% engram stim-recall, 90% multitag, 90% intersection,
96.7% yes/no, 98.8% precision at 20-pair capacity, **100% 3-word
sentence role queries**, **100% 4-word sentence role queries**. Plus
5-word sentence support, cross-session persistence, natural-language I/O.

## Full conversation transcript

```
============================================================
CONCEPT CHAT
Vocab: ['apple', 'river', 'dog', 'cat', 'go', 'come', 'stop', 'look',
        'big', 'small', 'hot', 'cold']
Learned associations: []
Encoded tags: []
============================================================

# Vocab introspection
> vocab
  vocab: ['apple', 'river', 'dog', 'cat', 'go', 'come', 'stop', 'look',
          'big', 'small', 'hot', 'cold']

# Natural assertions — NO 'remember' prefix needed
> apple is big
  Got it, I'll remember: apple_big
> apple is hot
  Got it, I'll remember: apple_hot
> cat is big
  Got it, I'll remember: cat_big
> dog is small
  Got it, I'll remember: dog_small

# 4-word sentences
> dog look hot cold
  Got it, I'll remember: dog_look_hot_cold
> cat stop big apple
  Got it, I'll remember: cat_stop_big_apple

# 5-word sentence
> apple stop big hot cat
  Got it, I'll remember: apple_stop_big_hot_cat

# Introspection
> tags
  tags: ['apple_big', 'apple_hot', 'cat_big', 'dog_small',
         'dog_look_hot_cold', 'cat_stop_big_apple',
         'apple_stop_big_hot_cat']

# Multi-tag retrieval (90% multi-seed)
> what is apple
  matched 2 tag(s): ['apple_big', 'apple_hot']
  top-5 associates:
    hot   = 0.250 via apple_hot    **
    big   = 0.199 via apple_big    **
    cold  = 0.084 via apple_big
    cat   = 0.065 via apple_big
    go    = 0.065 via apple_big

# Natural-language synthesis
> describe apple
  apple is hot and big.

# Yes/no questions (96.7% multi-seed)
> is apple big
  YES: 'apple' is bound to 'big' (tag apple_big, both in lang_output top-5)
> is apple small
  NO: no tag binding 'apple' and 'small' (checked apple_small, small_apple)

# Compositional intersection (90% multi-seed)
> what is apple and cat
  [intersection] cue=(apple AND cat)
    big   = min(0.22, 0.14) = 0.14    # the shared trained association

# 4-word role queries (100% multi-seed)
> who look hot cold
  Who look hot cold? dog
> what did cat stop big
  What did cat stop big? apple

# 5-word role query (also works)
> what did apple stop big hot
  What did apple stop big hot? cat

# Cross-session persistence
> save
  [saved bridge + 7 engram tag(s) to bridges/v16/seed44.simstate.h5]

> quit
```

## What's working

**Six capabilities validated at multi-seed:**

| Capability | Reliability | Mechanism |
|---|---|---|
| Engram stim-recall | 87.5% | Tonegawa tagging (catalog D.14) |
| Multi-tag cue retrieval | 90% FULL | Stim all matching tags, aggregate |
| Compositional intersection | 90% FULL | Multi-tag intersection by min-cosine |
| Yes/no questions | 96.7% | Tag existence + neural verify |
| Capacity precision (20 pairs) | 98.8% | Top-N includes only valid associates |
| **3-word sentence role queries** | **100%** | Tag-name pattern + neural verify |
| **4-word sentence role queries** | **100%** | Tag-name pattern + neural verify |
| 5-word sentences | 100% role, ~95% recall | Same mechanism, denser encoding |

**Plus UX features:**
- Cross-session persistence (`save` writes engram tags to HDF5; load restores)
- Natural-language input (article stripping, "is" connector)
- Natural-language output (`describe` synthesizes "X is Y and Z")
- Multi-turn drill-down (`tell me more`)
- Runtime CRUD (remember/forget)

## What's NOT working (honest)

- **16-word vocab is the ceiling.** v17 28-word ceiling exhaustively
  explored across 8 single-parameter hypotheses — all failed. Requires
  multi-day architectural rework (concept-only architecture or
  hippocampus consolidation).
- **5 words/sentence is the practical sentence limit.** 6+ word
  sentences would saturate concept-pool firing.
- **Order info lives in tag-name strings**, not in temporal firing
  pattern. Position-conditioned recall via STDP-grown ec_context→pool
  weights validated at only 50% peak — concept-pool architecture lacks
  true temporal binding. The tag-name index provides order
  discrimination as a UX layer.
- **No coreference**: "alice is happy, how is she?" — no pronoun
  resolution.
- **No inference beyond direct retrieval**: "alice has a red apple,
  what color is alice's fruit?" — no compositional reasoning.

## How to run

```bash
# 1. Train v16 bridge (one-time, ~18 min)
python -m research.runners.concept_pool_demo --seed 44 \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --save-bridge bridges/v16/seed44.simstate.h5

# 2. Launch chat REPL
python -m research.runners.compose_concept_chat \
    --load-bridge bridges/v16/seed44.simstate.h5 \
    --seed 44 \
    --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 \
    --n-words-for-orthogonal 16 --encoding-steps 500 --sparsity 0.05 \
    --balanced-teacher-pA 500.0 \
    --pairs ""   # empty = teach from scratch
```

## Architectural foundation

- **Tonegawa engram tagging** (catalog D.14): each sentence captures
  100 co-firing neurons in concept pools as one named ensemble
- **Multi-tag aggregation**: per-tag 87.5% stim-recall reliability,
  union across matching tags for cue queries
- **Tag-name indexing**: order information in tag-name preserves
  word order ('a_v_b' ≠ 'b_v_a'); pattern-match on tag names for
  role queries
- **Neural verification**: every retrieval re-stims the matched tag
  and confirms target words appear in lang_output top-K

This is the chat REPL for the validated multi-seed conversational
capability at v16 16-word vocab. It's genuinely usable for semantic
conversation with cross-session memory.
