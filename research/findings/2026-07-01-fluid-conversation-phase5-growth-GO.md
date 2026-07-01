# Fluid conversation — Phase 5 GO: growth through conversation (learn new facts from dialogue, generalize to novel entities)

**2026-07-01 (autonomous night; owner's fluid-conversation priority).** Phases 0–4 gave a fluent, grounded, focused,
multi-turn conversation over a FIXED taught curriculum. Phase 5 closes the **growth** axis — the owner's *"still being
able to grow through these experiences"*: a NEW fact stated in conversation is learned by the brain and immediately
usable, old facts are retained, and the no-confab moat holds. Reuse-by-import (brain store + the Phase-2/3 RA-QA
pipeline); **NO `sim/` edit**.

## Result — GO (3 seeds)
`_fluidconv_phase5_growth_derisk.py`. Teach three NEW facts mid-conversation, then answer each; check retention, moat,
durability.

| gate | result (3 seeds) |
|---|---|
| **LEARN** — each new fact, once heard, is answered grounded (RA-rendered) | **3/3** (all facts) |
| **NOVEL-GENERALIZE** — the generator renders a subject UNSEEN in its fine-tune vocab from the provided fact | **True** |
| **RETENTION** — base facts still recalled after growth (no catastrophic forgetting at the store level) | **True→True** |
| **MOAT** — a still-untaught vocab cue ("otter") → abstain (gate-first, model not invoked) | **0-FA** |
| **DURABILITY** — the first-learned new fact still recalled after all later growth + queries | **True** |

Transcript (learning three facts live):
```
taught 'wolf eat rabbit'  -> "the wolf eats rabbit."       (wolf IS in the fine-tune vocab)
taught 'camel eat grass'  -> "the camel eats grass, yes."  (camel NOVEL to the generator)
taught 'zebra like hay'   -> "the zebra likes hay."        (zebra NOVEL to the generator)
```

## The standout — RA-generalization to novel entities
The generator was fine-tuned (Phase 2) on `wolf/dog/cat/…` (NOT `camel`/`zebra`), yet it renders *"the camel eats
grass"* and *"the zebra likes hay"* correctly from the PROVIDED fact. This confirms the fine-tune taught the **format**
("use the provided facts to answer") — it did NOT memorize a fixed fact table. That generalization is exactly what
makes breadth tractable: as the brain learns new entities from conversation, the retrieval-augmented generator renders
them fluently without any retraining.

## What this + Phases 0–4 establish (the arc, complete on its core axes)
The minimized (~21M, 15–25× < Qwen-0.5B), brain-trained, brain-gated conversational stack now demonstrates every
core axis the owner named:
- **fluent** (Phase 0: a ~21M TinyStories generator, SCALE-CONFIDENT grounded);
- **grounded** (Phase 1: prompt-condition + free-gen + post-hoc VERIFY);
- **focused conversational Q&A** (Phase 2: the RA render/QA "brain-train" fine-tune — answers, not story rambles);
- **the full single-turn** (Phase 3: comprehend → gate → answer → verify);
- **context / multi-turn** (Phase 4: a pronoun resolves to the held referent on the spiking WM loop);
- **grows through experience** (Phase 5: learns new facts from conversation, generalizes to novel entities, retains).
The BRAIN supplies comprehension + knowledge + grounding + the moat; the minimized generator supplies fluency. Moat
preserved throughout (gate-first). **NO `sim/` edit anywhere in Phases 0–5.**

## Honest scope + next
- **Growth is over PRE-ALLOCATED concept codes** (the composer's vocab is fixed at build — as in the develop loop);
  learning brand-new concept CODES is the separate dendritic/allocation frontier. Full **cross-session persistence**
  is validated in the develop loop (Tier-3 live-and-remember); here a within-session durability check stands in.
- **Breadth** ("almost any topic") remains the honest scale wall — partly addressed by the RA-generalization shown
  here (a broader learned KB → the generator renders it), bounded by the composer's FHRR capacity (∝√D; raise D for
  more concepts, validated to 320) and the abstention moat as the truthful "I don't know" boundary.
- **Tracked shortcuts / deferred:** the generator runs as an ANN (the spiking-forward conversion is deferred until the
  KV-cache speed lever lands — a validated-mechanism reuse, not a new capability); the interrogative parse is a
  rule-based scaffold (→ a neural interrogative parser). Both flagged for burn-down per the end-state-fully-spiking
  standard.

**Artifacts:** `research/runners/_fluidconv_phase5_growth_derisk.py`; result
`research/findings/raw/_fluidconv_phase5_growth.json`.
