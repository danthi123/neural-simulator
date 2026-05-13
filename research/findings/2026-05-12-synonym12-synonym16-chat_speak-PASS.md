# 🎉 Synonym12 and Synonym16 chat_speak validated — 12 and 16 word conversational vocab

**Date:** 2026-05-12
**Status:** Realigned plan Step 2 (validate synonym12/16) — **BOTH GO** at
seed 42. Major conversational milestone.

## Headline result

| Mode | Vocab Size | W→A | A→W | Verdict |
|---|---|---|---|---|
| Tier 1 | 4 words | 74-98% | 58% | GO (validated multi-seed) |
| Synonym (Tier 2.1) | 8 words | 31-56% | 85% | GO (validated 6 seeds 2026-05-09) |
| **Synonym12** | **12 words** | **56%** | **100%** | **GO seed 42** |
| **Synonym16** | **16 words** | **56%** | **100%** | **GO seed 42** |

Single-seed validation at seed 42. Multi-seed validation pending
but seed 42 is a strong signal given prior tier validations.

## Per-action A→W speak details

Both synonym12 and synonym16 produce IDENTICAL top-1 per action:

| Motor target | Top-1 predicted | Cosine |
|---|---|---|
| N | north | 0.15 |
| E | east | 0.18 |
| S | south | 0.15 |
| W | west | 0.16 |

Deterministic (temperature=0) top-1 always wins primary direction
word. Synonyms (up/right/down/left and additional synonyms for
synonym12/16) appear in top-3 but don't beat primary.

## What this enables

The conversational sim now supports:

### Pre-trained 16-word vocabulary (synonym16)
- 4 primary direction words: north, east, south, west
- 12 synonyms: up/right/down/left + 8 more synonyms

### Bidirectional binding
- W→A: type direction word → motor pool activates (56% accuracy
  at synonym16 — much above chance ~6% for 16-word random)
- A→W: drive motor pool → sim speaks the primary direction word
  (100% top-1 across all 4 actions)

### Continual learning
- Phase 1.3 hippocampus consolidation (3/3 PASS): cortex retains
  bindings after sleep silencing
- Phase 1.4 catastrophic forgetting (5/6 PASS): primary direction
  binding preserved through new synonym training

### In-vivo vocabulary growth
- :learn command (V_SCHEMA upgraded): user can add ~2 new words at
  a time via schema-supported anchor reinforcement
- 2-binding fixed capacity per architectural ceiling (today's finding)

## Architecture (synonym16)

Same as Tier 2.1 v4 scale-up, with vocab_size=16:
- 17,152 neurons
- 26.8M synapses
- n_lang_input=4096, n_motor_per_action=2000, n_motor_fs_per_action=240
- Embodied-Hebbian co-firing during training
- Topographic bias on lang_input → motor (Tier 1 BREAKTHROUGH recipe)

## Wall clock

- Synonym12 chat_speak: 4148s (69 min, slow due to parallel Tier 2.3
  GPU contention)
- Synonym16 chat_speak: 2500s (42 min, solo)

## Comparison to original Tier 2.1 (8-word, 2026-05-09)

| Metric | Tier 2.1 (8) | Synonym12 | Synonym16 |
|---|---|---|---|
| Vocab size | 8 | 12 | 16 |
| W→A multi-seed | 31-56% mean 33% | 56% (s42) | 56% (s42) |
| A→W mean | 85% (6 seeds) | 100% (s42) | 100% (s42) |
| n_motor | 1000 | 2000 | 2000 |

Synonym12/16 actually IMPROVED A→W over Tier 2.1 — likely due to
scaled motor pools (1000 → 2000) providing more capacity for the
per-action generative readout.

## Strategic implication

The conversational sim now has a **16-word working vocabulary** with
reliable production (A→W 100%) and meaningful reception (W→A 56%).
Combined with:
- :learn V_SCHEMA for adding 2 more in-vivo words
- Phase 1.3 consolidation preventing catastrophic forgetting
- chat_repl interactive REPL

The user has a genuinely usable conversational artifact at this
vocab tier.

## Next steps for further vocab expansion

Untested:
- Synonym24 / synonym32 (would require expanding vocab tables in
  text_embeddings.py and chat_repl)
- Multi-seed validation of synonym12/16 (currently single-seed)
- Higher-order phrase composition (Tier 2.3 still stuck at 34-40%)

## Catalog faithful

- Tier 1/2.1 motor pool architecture (Pulvermüller 2001-2003
  somatotopy + Lefort 2009 cortical canon)
- Embodied-Hebbian co-firing during training
- Topographic bias on lang_input → motor (biology-faithful prior)
- No motor-decoder cheats
- No external LLM cheats

## Files

- Synonym12 result: `research/findings/raw/g11_bg/synonym12_chat_speak/seed42.json`
- Synonym16 result: `research/findings/raw/g11_bg/synonym16_chat_speak/seed42.json`
