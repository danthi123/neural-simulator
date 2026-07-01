# Fluid conversation — Phase 10 GO: open-ended grounded DISCUSSION (discuss ideas/concepts, not just fact-lookup)

**2026-07-01 (autonomous; owner morning steer — "talk in depth about ideas/concepts the brain has RELEVANT info on
but no explicit ANSWER to; extrapolate").** Research-gated (scoping `2026-07-01-open-ended-grounded-discussion-scoping.md`:
the ask is ~90% built; the gap is integration + a synthesis step). This closes it: the brain now DISCUSSES a topic
using its relevant grounded knowledge, instead of one-fact-lookup + abstain. Reuse-by-import; **NO `sim/` edit**.

## The gap (measured on the current console)
one-fact-lookup + abstain: *"how are dogs and cats different?" → "I don't know."*; *"tell me about predators" → "I
don't know."* No cross-fact synthesis, no concept-level extrapolation.

## The mechanism (#1 cheap-first from the scoping, with the empirically-forced correction)
`_fluidconv_phase10_discussion_derisk.py` (`Discussant`): RETRIEVE the topic's grounded **neighbourhood** (the
association-graph facts-about/mentioning + category members — the same adjacency the GO `DiscursiveTurn` gathers) →
render each fact → **VERIFY each** → concatenate the verified sentences into a multi-fact grounded discussion.
- **Empirical correction (important):** conditioning the RA generator on the MULTIPLE facts AT ONCE (the scoping's
  first idea) makes the 21M **confabulate by mixing entities** (*"the dog eats fish"* — dog + fish from different
  facts → ungrounded). The fix is the GO `DiscursiveTurn`'s proven approach: render each fact **separately** (the
  validated FAITHFUL single-fact render) + per-sentence VERIFY + concatenate. No confabulation; ≥2 grounded facts.
- Moat reframed (per `feedback_moat_not_hard`): from hard-abstain to **grounded-elaboration-with-hedging** — discuss
  what it relates; hedge/"I don't know much about that" on an empty neighbourhood; VERIFY drops any ungrounded claim.

## Result — GO (3 seeds)
- **CONCEPT** *"tell me about predators"* → *"Here's what I know about the predator: the dog eats meat. the dog chases
  cat. the cat eats fish. the cat chases mouse."* (**4 grounded facts** — the brain has no "predators" answer, so it
  extrapolates from the members it knows are predators).
- **RICH TOPIC** *"tell me about the dog"* → 2 grounded facts.
- **LESION** (empty neighbourhood, "dragons") → honest hedge (no fabrication).
- **PERMUTED** (retrieve the wrong topic) → the discussion is about the wrong thing (retrieval is load-bearing).
- **CONFAB-PROBE** (inject a false fact into the render set) → VERIFY DROPS it (0 ungrounded claims emitted).

## Honest ceiling
- The discussion is **per-fact faithful render + concatenation**, NOT free abstractive synthesis — a genuinely fluent
  single-pass synthesis over multiple facts confabulates on this 21M (the field's abstractive-hallucination wall).
  The extract→render-each→verify→concatenate pipeline is the honest, grounded, traceable shape (the spiking store +
  no-confab VERIFY are the distinctive extract+verify halves most RAG lacks).
- **Copula/"is" (category-membership) facts don't render** via the transitive-verb-tuned generator (VERIFY drops
  them); the transitive member-facts carry the concept discussion. A vocab-coverage limit (the RA fine-tune didn't
  include the copula), not a mechanism failure — a bounded follow-on (a copula render or a small fine-tune addition).
- **Free open-world inference BEYOND the stored+adjacent neighbourhood remains the field wall** — the honest hedge
  ("discuss what it relates; say where its knowledge ends") is the deliverable at that boundary.

**NEXT:** wire the `Discussant` into the fluid console (route open/concept/compare questions through it → discussion
instead of abstain). Then compare/gist relational framing (#2), and the copula-render follow-on.

**Artifacts:** `research/runners/_fluidconv_phase10_discussion_derisk.py`; result
`research/findings/raw/_fluidconv_phase10_discussion.json`.
