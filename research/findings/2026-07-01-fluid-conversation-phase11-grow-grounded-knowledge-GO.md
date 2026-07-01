# Fluid conversation — Phase 11 GO: grow grounded knowledge → richer grounded discussion (the owner's chosen path)

**2026-07-01 (autonomous; owner steer = GROW GROUNDED KNOWLEDGE, keep the no-confab thesis).** The owner's "tell me
about dogs" example showed a rich LLM reply that is ~95% the transformer's UNVERIFIED parametric world knowledge —
exactly what this project avoids. The owner chose to keep the thesis: the BRAIN learns richer REAL knowledge
(grounded/verified), discussions get richer as the KB grows (thin-but-true, richer over time). This de-risks the core
claim + honestly maps the bottlenecks. Reuse-by-import; **NO `sim/` edit**.

## Result — GO (3 seeds)
`_fluidconv_phase11_grow_knowledge_derisk.py`: teach a RICHER grounded KB (real simple SVO facts about dogs, the
offline-textbook-author pattern — all TRUE, grounded), then discuss vs the toy KB.
- **RICHNESS SCALES:** toy KB → **3** grounded facts cited; rich KB → **7** (all 3 seeds). The rich reply:
  *"Here's what I know about the dog: the dog eats meat. the dog chases cat. the dog likes bone. the dog sees human.
  the dog finds toy. the dog catches ball. the dog chases wolf."* — a genuinely richer grounded discussion.
- **GROUNDED:** 0 ungrounded fact-claims (VERIFY). **GENERIC:** "dogs" → the kind ('dog'). **MOAT:** unknown concept
  ("dragons") → honest hedge.
- **⇒ the bottleneck is the KB size, NOT the discussion mechanism** (Phase-10 GO). Grow the grounded KB → richer
  discussion, staying grounded.

## The two knowledge-growth bottlenecks (honestly mapped)
1. **The brain's KB** — grows FREELY via parse+store (ANY verb). The scaling arc: a **real-corpus knowledge-
   acquisition pipeline** — parse simple factual sentences (a simple-Wikipedia-style fact corpus) → `composer.store`,
   staged cumulatively (per `project_deep_knowledge_brain_fluency_build`). This is the path to encyclopedic grounded
   breadth over time.
2. **The RA generator's render vocab** — only its ~18 fine-tune verbs render fluently. **9 of the 12** rich dog facts
   use in-fine-tune verbs (renderable); **3** (guard/help/herd) use out-of-vocab verbs — the brain KNOWS them, but
   the RA generator can't render them (VERIFY drops them). The lever: a **broader render fine-tune** (more verbs), or
   the **brain's own neural serial-order render** as a grounded (less-fluent) fallback for any verb. This is a fluency-
   coverage limit, not a knowledge limit.

## Honest ceiling
- Discussion richness scales with the grounded KB — but the discussion still LISTS facts (the abstractive-synthesis
  wall stands; genuine single-pass fluent synthesis over multiple facts confabulates on this 21M).
- The RA generator renders only its trained verbs (a fine-tune lever); the brain's KB is the deeper scaling arc.
- Free open-world inference beyond the stored facts remains the field wall — the honest hedge is the deliverable.

## Next (the owner's grounded-growth path)
1. **Generic/definite** in the console: "dogs"/"a dog"/generic → the kind; "the dog" → a held referent, else the kind
   (clarify "which dog?" a refinement). [plural normalization already wired; the referent-vs-kind nuance next.]
2. **The real-corpus knowledge-acquisition pipeline** (the scaling arc): ingest a simple factual corpus → parse →
   grounded facts → grow the KB over "days" (the develop-loop), so discussions get encyclopedically richer, grounded.
3. **A broader render fine-tune** (more verbs) or the brain's-own-render fallback → more of the KB renders fluently.

**Artifacts:** `research/runners/_fluidconv_phase11_grow_knowledge_derisk.py`; result
`research/findings/raw/_fluidconv_phase11_grow_knowledge.json`.
