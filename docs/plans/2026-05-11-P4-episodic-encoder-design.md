# P4 — Episodic encoder + relational binding design

**Date:** 2026-05-11
**Phase:** P4 of realigned plan v3
**Catalog entries:** D.01 (episodic memory), D.02 (Eichenbaum–Cohen
relational binding)
**Citations:** Kandel 6e Ch 52 pp 1296–1302; Eichenbaum/Cohen 2014
**Roadmap:** sits between roadmap T1.A-C (P1-P3) and the language-
clusters work (P5+)

## What "episodic memory" actually means

From catalog D.01 (Kandel pp 1296–1302):

> Binds multimodal items into **events** and events into **episodes**
> via temporal/spatial context (Tulving 1972; Eichenbaum/Cohen
> relational memory).

From D.02:

> Stores events as **items-in-context**, episodes as temporal
> sequences of events, and networks via overlapping events allowing
> flexible inference (e.g., transitive). Distinguishes overlapping
> episodes that share elements (same restaurant, different visits)
> without interference.

For the conversational-sim goal: an "item" is a tagged concept
(P2 engram); an "episode" is a sentence; binding "items in context"
is the operation that makes "alice ate the apple" different from
"the apple ate alice" — same concepts, different relations.

## What the existing architecture provides

The trisynaptic loop (P1) + engram tagging (P2) give:
- **Items:** tagged CA3 ensembles (P2 `engram_tags`)
- **Pattern separation:** DG orthogonalizes overlapping inputs (P1 D.12 PASS)
- **Pattern completion:** CA3 reconstructs from partial cues (P1 D.13 partial pass)
- **Persistence:** tags survive save/load (P2 commit a3acb9c)

What's MISSING for episodic binding:
- **Context representation:** no current way to mark "position" or
  "time" within an utterance/episode
- **Item-in-context binding:** no mechanism that makes
  CA3-cells-for-(apple,position-1) distinct from
  CA3-cells-for-(apple,position-2) while still treating both as
  related to "apple"
- **Sequential episode structure:** no linkage between successive
  events in a sequence (e.g., "alice" → "ate" → "apple")

## P4.1 — Positional / temporal context (concrete plan)

### Add a context region to the EC input layer

Currently EC receives only the lang_input pattern (word identity).
For episodic encoding, EC also needs a CONTEXT signal.

Options for context:
- **Positional code** (simplest): add a `ec_position` region whose
  activity pattern depends on the position index within the current
  sentence (1st word, 2nd word, ...). Implementation: a learned or
  hand-coded positional embedding similar to the word embedding.
- **Time-cell code** (catalog D.11): activate sequentially over time
  during a "delay period". Pastalkova 2008 time cells fire at
  specific times after an event start. Each time cell tiles a window
  of the episode.
- **Grid-cell code** (catalog D.07): grid cells provide a discrete
  metric. Could index "position in episode" the same way they index
  "position in space."

Recommendation: **positional code first** (cheapest), then time-cell
code as a refinement.

### Mechanism

New region: `ec_context` (200 neurons, sparse 10% per position).
For position k in {1, 2, ..., N}, drive a stable sparse pattern
`context_k` over `ec_context`. The pattern is fixed per position
(deterministic, like the existing `vocab_to_drive_pattern` but for
positions).

Pathway: `ec_context → dg` (alongside the existing `ec → dg`).

When a word is presented at position k, BOTH `ec` (word pattern) AND
`ec_context` (position pattern) drive DG. DG's expansion recoding
produces a SPARSE ENSEMBLE that's specific to (word, position) — the
item-in-context.

CA3 trained on this DG output stores `(apple, pos_1)` as a distinct
attractor from `(apple, pos_2)`, even though both share the word
"apple."

### Test

```python
# Encode "alice ate apple" via 3 (word, position) bindings
for pos, word in enumerate(["alice", "ate", "apple"], start=1):
    drive_ec_word(word)
    drive_ec_context(pos)
    bridge.start_engram_recording(f"sentence1_pos{pos}")
    for _ in range(100):
        bridge._run_one_simulation_step()
    bridge.commit_engram_tag(f"sentence1_pos{pos}", top_k=50,
                              region_filter=["ca3"])

# Encode "apple ate alice" via different (word, position) bindings
for pos, word in enumerate(["apple", "ate", "alice"], start=1):
    drive_ec_word(word)
    drive_ec_context(pos)
    bridge.start_engram_recording(f"sentence2_pos{pos}")
    for _ in range(100):
        bridge._run_one_simulation_step()
    bridge.commit_engram_tag(f"sentence2_pos{pos}", top_k=50,
                              region_filter=["ca3"])

# Test: are the (apple, pos_3) and (apple, pos_1) tags distinct?
tag_apple_pos1 = bridge.get_engram_tag_indices("sentence2_pos1")  # apple-first
tag_apple_pos3 = bridge.get_engram_tag_indices("sentence1_pos3")  # apple-last
cos = cosine_similarity_indices(tag_apple_pos1, tag_apple_pos3, n)
# PASS: cos < 0.3 (distinct positions get distinct ensembles)

# Test: do they share some apple-related neurons?
tag_alice_pos1 = bridge.get_engram_tag_indices("sentence1_pos1")  # alice-first
cos_apple_apple = cosine_similarity_indices(tag_apple_pos1, tag_apple_pos3, n)
cos_apple_alice = cosine_similarity_indices(tag_apple_pos1, tag_alice_pos1, n)
# PASS: cos_apple_apple > cos_apple_alice (same word > different word
# despite same position)
```

### Validation criteria

1. **Position discriminability:** (apple, pos_1) tag overlap with
   (apple, pos_3) tag < 0.3 (DG separates them despite same word).
2. **Word similarity preserved:** (apple, pos_1) shares more with
   (apple, pos_3) than with (alice, pos_1) — word identity
   contributes more to ensemble overlap than position identity.
3. **Sentence-level discrimination:** the sum/union of all tags for
   "alice ate apple" is distinct from the sum/union of tags for
   "apple ate alice."

## P4.2 — Sequential episode binding (deferred)

The next layer: tag the SEQUENCE of CA3 ensembles as an episode. This
needs theta-paced sequential CA3 activity (O&N supplemental in
catalog D.05) and is more complex.

For the immediate primary path (conversation), P4.1 (item-in-context)
is enough to distinguish "alice ate apple" from "apple ate alice."
Sequence binding can come later.

## Open questions

1. **How many context positions to support?** Realistically: 8-16 for
   a sentence-level system; 128+ for paragraph-level. Each position
   needs a distinct sparse pattern in `ec_context`. Sparse codes with
   ~10% activity each: 200 ec_context neurons × 10% = 20 active per
   position → can support ~10 distinct positions before overlap
   becomes problematic.

2. **Should context come from a separate region OR be part of EC?**
   Adding `ec_context` keeps things clean (separation of concerns)
   but increases neuron count. Embedding context into EC itself
   (e.g. position bits in the word pattern) is more compact.

3. **Is position the right context?** Real biology uses spatial
   place (where the event happened), time (when), and semantic
   relations (causal, agent/patient). Position is the cheapest
   approximation for sentence-level binding. Real conversational sim
   eventually needs richer context.

## Sequencing

```
P1 (trisynaptic) — single-seed PASS, multi-seed borderline
P2 (engram tagging) — SHIPPED
Two-concept discrimination test — in flight (relative criterion)
    ↓
If two-concept PASSES: proceed to P3 concept replay
If FAILS: more P1 tuning needed first
    ↓
P3 (concept replay during NREM)
    ↓
P4.1 (item-in-context — this design)
    ↓
P4.2 (sequence binding — deferred until needed)
    ↓
P5 (ventral semantic stream)
```

## Effort estimate

- ec_context region + positional embedding: ~100 LOC
- Pathway wiring (`ec_context → dg`): ~10 LOC
- Test runner: ~200 LOC
- Multi-seed validation: 1-2 days wall clock
- **Total: ~3-5 days implementation + 1 week validation**

## Why this matters

Without item-in-context binding, the sim cannot represent meaning
that depends on WORD ORDER. "alice ate apple" and "apple ate alice"
become indistinguishable. That's a hard ceiling for any
conversational behavior.

P4.1 unlocks the basic grammatical distinction. P5 (ventral semantic
stream) then maps these distinct (word, position) ensembles to
abstract concepts (alice-agent, eat-action, apple-patient). P6
(Broca's) adds compositional generation.

This is the path from "concepts as tagged ensembles" to "sentences
as tagged episode ensembles."
