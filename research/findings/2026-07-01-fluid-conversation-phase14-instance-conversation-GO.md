# Fluid conversation — Phase 14 GO: instance-rep in a MULTI-TURN flow (discourse-instance tracking)

**2026-07-01 (autonomous; owner steer = grow grounded knowledge / all levers in parallel; the "which dog?" nuance,
now conversational).** Phase-13 proved the kind/instance MECHANISM (own facts + isa-inheritance + definite/generic).
This consolidates it into a running CONVERSATIONAL flow: an instance is minted when first mentioned, tracked as a
discourse referent, attributed across turns, and queried — with a second instance kept distinct. Reuse-by-import
(Phase-13 `_resolve` + the brain); **NO `sim/` edit**; CPU (brain-only).

## Result — GO (3 seeds: 42, 43, 44)
`_fluidconv_phase14_instance_conversation_derisk.py` — the owner's example as a live multi-turn transcript:
```
  you>   i saw a dog            brain> ok, a dog.              (mint dog_1 isa dog; discourse-current)
  you>   the dog was brown      brain> ok, the dog is brown.  (store dog_1 is brown)
  you>   what is the dog?       brain> the dog is brown.       (the INSTANCE's own fact — not the kind's "mammal")
  you>   what does the dog eat? brain> the dog eats meat.       (INHERITED from the kind via the isa link)
  you>   what do dogs eat?      brain> dogs eats meat.          (the GENERIC kind)
  you>   i saw a cat            brain> ok, a cat.               (mint cat_1; discourse focus now the cat)
  you>   what is the dog?       brain> the dog is brown.        (dog_1 STILL distinct + retrievable)
  you>   what does the wolf eat? brain> I don't know.           (moat — "wolf" never introduced)
```
All 5 gates hold every seed: **own** (definite → the instance's own fact) · **inherit** (isa-inheritance) ·
**generic** ("dogs" → the kind) · **distinct-persist** (a second instance keeps the first retrievable) · **moat**.

## The two fixes vs the Phase-14 v1 HONEST/PARTIAL
1. **Plural normalization in kind-detection** — "dogs" wasn't matched to the kind "dog" (fell to "ok."). `_kind_of`
   now returns `(kind, is_plural)`: a bare kind token → `(kind, False)`; a token whose singular is a kind (`dogs` →
   `dog`) → `(kind, True)`. Generic questions ("what do dogs eat?") route to the kind; definite ("the dog") to the
   instance.
2. **Per-kind instance tracking** — a single `self._cur` discourse pointer broke distinct-persist: after minting a
   cat, "the dog" fell through to the generic kind. Replaced with `self._last_inst = {kind: token}` (the last instance
   of EACH kind), so "the dog" resolves to the most-recent DOG instance regardless of the current discourse focus.
   (`self._cur` is retained only for bare-pronoun "it" attribution.)

## Honest ceiling
- A lightweight **rule-based** mint/attribute/query router over the validated Phase-13 mechanism + per-kind
  most-recent-instance pointers (object-files). The interrogative/mint PARSE is a scaffold; the neural interrogative
  parser (Phase-7) is the brain-based path — a bounded follow-on to wire in.
- Multiple SIMULTANEOUS same-kind instances ("the first dog vs the second dog") → the already-mapped biased-competition
  WTA drop-in (not reopened here).
- A PERCEIVED/consolidated episodic instance ("the specific dog I saw on my walk") → the engram-tag/hippocampal path
  (composes with the Tier-3 live-and-remember loop) — a follow-on.

## Where this sits (the grounded-growth conversational path, now multi-turn)
- **Phase-12 (GO):** knowledge-acquisition pipeline (learn a real-fact corpus, staged cumulatively). [breadth]
- **Phase-13 (GO):** kind vs instance — the MECHANISM. **Phase-14 (this, GO):** the same, in a CONVERSATIONAL flow.
- Together with Phase-10 (open-ended discussion) + Phase-11 (richness scales with the KB): the brain learns real
  grounded knowledge, distinguishes "the dog" (a referent) from "dogs" (the kind), and discusses it across turns —
  grounded, hedged, moat-safe.

**Artifacts:** `research/runners/_fluidconv_phase14_instance_conversation_derisk.py`; result
`research/findings/raw/_fluidconv_phase14_instance_conversation.json`.
