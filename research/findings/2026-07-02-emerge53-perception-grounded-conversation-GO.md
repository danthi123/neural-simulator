# EMERGE-53 — the PERCEPTION-GROUNDED conversational console: SEE an object → discover its category → talk about it — GO (3-seed)

**Date:** 2026-07-02
**Runner:** `research/runners/_emerge53_perception_grounded_conversation.py`
**Tests:** `tests/test_emerge53_perception_grounded_conversation.py` (4 CPU/numpy, offline, ~4.3 s)
**Raw:** `research/findings/raw/_emerge53_perception_grounded_conversation.json`
**Verdict:** **GO** — reuse-by-import, **NO `sim/` edit**.

## What this closes (the master-directive "grounded in the brain's OWN EXPERIENCES" clause, for CONVERSATION)

EMERGE-51/52 made the pooler-discovered categories **conversationally queryable**, but the categories were discovered
from **abstract feature tokens** ("a robin has wings feathers small"). EMERGE-53 grounds the same conversation in **real
PERCEPTION**: the brain **SEES** an object through the project's real Gabor/V1 front end, the competitive self-organizing
pooler **discovers its category from the VISUAL similarity**, and the user **talks about it in plain language** — inherit
/ cancel / abstain over the visually-discovered codes, with the intrinsic no-confab moat.

> **SEE an object → discover its category → talk about it.**

It is a **composition of validated pieces** (EMERGE-34/36 perception-grounded emergence + EMERGE-42/51 on-bridge
inheritance/cancellation + the EMERGE-29/51 natural-language console), **not a new mechanism**.

## Mechanism (no inference engine, no transformer; NO `sim/` edit)

- **Perception (environment renders the sensory input):** a `PerceptualWorld` renders 2 visual categories × 12 exemplars
  as oriented-bar bird/fish shapes (EMERGE-34's `build_shape_set`) and encodes each through the **real retina→V1 Gabor
  receptive-field bank** (`sim.visual_cortex.build_v1_simple_weights`, reused via `_genfrontier_optionB`'s `encode_v1`).
  Each rendered exemplar is a **named object** (robin, sparrow, … / trout, salmon, …). `see(name)` returns that object's
  **top-T (=20) active V1 cells** among the NF=512 most-active global V1 cells — the perceived object's feature vector.
- **Discovery (the brain):** the competitive HTM Spatial Pooler (EMERGE-38/42: winners potentiate active inputs + depress
  inactive + homeostatic boosting, k-WTA) **self-organizes a codon per perceived object** from those V1 features.
  Same-category shapes → overlapping V1 features → **overlapping codons** = the emergent VISUAL categories.
- **Talk (the brain, on the spiking `SimulationBridge`):** teaching "a `<exemplar>` can P" potentiates the exemplar's
  **discovered codon → P** coincidence pool via the committed `sim/` three-term kernel (Bouhadjar-Diesmann); co-seen
  members **INHERIT** P via the shared codon. A member exception "a `<member>` P" potentiates a **member-identity ensemble
  → P**, a stronger direct fact that out-drives the inherited default (**cancellation**). A graded apical read over the
  discovered codes answers "can a X P?". A visually-degenerate / never-seen percept drives no shared codon → the **moat
  abstains** (intrinsic).

## De-risk gates (3-seed 42/43/44) — all met

| seed | held-out PERCEIVED inherit | cancel | moat abstains (unknown) | moat false-accepts | **per-image SCRAMBLE inherit** | RSA (intact) | RSA (scrambled) |
|------|---------------------------|--------|-------------------------|--------------------|-------------------------------|--------------|-----------------|
| 42   | 1.00 | 1.00 | yes | 0 | 0.00 | 0.87 | 0.11 |
| 43   | 0.75 | 1.00 | yes | 0 | 0.00 | 0.77 | −0.04 |
| 44   | 1.00 | 1.00 | yes | 0 | 0.00 | 0.84 | 0.05 |
| **mean** | **0.92** | **1.00** | **all** | **0** | **0.00** | **0.83** | **0.04** |

- **Held-out PERCEIVED-object inheritance 0.92** (gate ≥ 0.75, chance 0.50): a novel object (owl/wren for bird, minnow/gar
  for fish) — SEEN through Gabor/V1 but **never named in a fact** — inherits its category's property via the
  **visually-discovered codon**. This is generalization, not retrieval: the held-out members are excluded from the class
  teaching loop.
- **Cancellation 1.00**: the exception members (penguin → walks, pike → lurks) answer their **own specific fact**, not the
  class default.
- **Moat 0 false-accepts**: every never-seen token (zzz/qqq/wobble) abstains; the moat never invents a category.
- **PER-IMAGE PIXEL SCRAMBLE → inheritance 0.00 (the load-bearing perception control):** scrambling each object's pixels
  destroys the within-category **visual** similarity → the pooler can't discover the categories → held-out inheritance
  collapses (a **0.92 gap**). This isolates **real perception** as the cause, exactly as in EMERGE-34.
- **RSA pixel-provenance 0.83 vs scrambled 0.04 (label-free):** the discovered structure tracks the raw-**pixel** visual
  similarity, not any injected label; scrambling collapses it.

## Demo transcript (seed 42, `--demo`)

```
--- SEE objects through the real retina/V1 Gabor front end (the pooler DISCOVERS the visual categories) ---
you> see robin        brain> ok -- I've seen a robin.
...   (9 bird-shapes + 9 fish-shapes SEEN; the competitive pooler self-organizes the two VISUAL categories)
--- bind the perceived objects to their class name ---
you> a robin is a bird     brain> ok -- a robin is a bird.
...
--- TEACH the class property via a few perceived exemplars + member-specific EXCEPTIONS ---
you> a robin can fly       brain> ok -- a robin can fly.        (class property, via a perceived bird exemplar)
you> a trout can swim      brain> ok -- a trout can swim.       (class property, via a perceived fish exemplar)
you> a penguin walks       brain> ok -- a penguin walks.        (member-specific EXCEPTION -- cancellation)
you> a pike lurks          brain> ok -- a pike lurks.           (member-specific EXCEPTION -- cancellation)
--- ASK in natural language (answered by the on-substrate inference over the VISUALLY-discovered codes) ---
you> can a owl fly?        brain> Yes, an owl can fly.          (INHERIT -- never told; via the VISUALLY-discovered bird category)
you> can a wren fly?       brain> Yes, a wren can fly.          (INHERIT)
you> can a minnow swim?    brain> Yes, a minnow can swim.       (INHERIT -- via the VISUALLY-discovered fish category)
you> can a gar swim?       brain> Yes, a gar can swim.          (INHERIT)
you> can a penguin fly?    brain> No, a penguin walks.          (CANCEL -- the penguin's own exception)
you> can a pike swim?      brain> No, a pike lurks.             (CANCEL -- the pike's own exception)
you> can a zzz fly?        brain> I don't know what a zzz is.   (MOAT -- a never-seen / visually-novel percept)
```

The interactive console also supports the pronoun form: `see owl` then `can it fly?` → "Yes, an owl can fly." (`it`
resolves to the last perceived object).

## Honest scope

- **Perception is the rate-reference sensory front end.** The Gabor/V1 encode + the competitive pooler are the
  representation steps (a rate reference for the fully-spiking versions — EMERGE-35/36 already have the spiking
  Marr-codon pooler). The **inheritance/cancellation run on the real spiking `SimulationBridge`** over the discovered
  codes. Coupling the fully-spiking V1/pooler into this console is the follow-on.
- **Teaching protocol (EMERGE-42-standard):** the class property is taught via several perceived exemplars (6 of 9
  per-category members named in "a X can P"); the **2 held-out members per category** are SEEN but never named in a
  can/exception sentence, so they inherit **only** via the shared visually-discovered codon.
- **Vocabulary:** 2 visual categories (oriented-bar bird/fish shapes, the EMERGE-34 set). Richer objects, a spiking
  V1/pooler in the console, and multi-level perceptual taxonomy in natural language are follow-ons.
- **Reuse-by-import; NO `sim/` edit** — the console composes the committed kernels + the validated perception pooler +
  the validated NL console.

## Files

- `research/runners/_emerge53_perception_grounded_conversation.py` — the console (`--demo` / `--script` / interactive /
  `--derisk --seeds 42 43 44`).
- `tests/test_emerge53_perception_grounded_conversation.py` — 4 CI tests (held-out inheritance + cancellation + moat;
  NL replies; per-image-scramble collapse + RSA collapse; moat on an unseen object).
- `research/findings/raw/_emerge53_perception_grounded_conversation.json` — the raw 3-seed de-risk record.
