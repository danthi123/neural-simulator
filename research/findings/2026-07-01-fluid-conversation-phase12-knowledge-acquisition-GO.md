# Fluid conversation — Phase 12 GO: the knowledge-acquisition pipeline (learn a real-fact corpus, staged cumulatively)

**2026-07-01 (autonomous; owner steer = grow grounded knowledge; parallel track).** Phase-11 mapped the bottleneck to
the KB SIZE. This builds + de-risks the **scaling mechanism**: the brain LEARNS real facts from a corpus, staged
cumulatively (the develop-loop / McClelland-CLS pattern), grounded + retained. Reuse-by-import (the validated
parse+store path); NO `sim/` edit; CPU (brain-only).

## Result — GO (3 seeds)
`_fluidconv_phase12_knowledge_acquisition_derisk.py`: a real-knowledge mini-encyclopedia (30 TRUE facts across 15
concepts, simplified to 3-word SVO) INGESTED day-by-day (parse each sentence → `composer.store`), composer D=256.
- **ACQUISITION:** recall **30/30** (all ingested facts recalled) — every seed.
- **STAGED-CUMULATIVE + RETENTION:** day-1 facts recalled **10/10 after every day** (day 1, 2, 3) — **no catastrophic
  forgetting** as days 2–3 grow the KB.
- **BREADTH:** ends knowing **15 concepts**. **MOAT:** a never-ingested cue ("dragon") → abstain (0-FA).
- ⇒ the brain grows grounded knowledge from a corpus, staged, retained, moat-safe — the scaling mechanism for the
  owner's "grow grounded knowledge" path.

## Where this sits (the grounded-growth path, now end-to-end)
- **Phase-12 (this):** the ACQUISITION pipeline — learn a real-fact corpus, staged, retained. (the KB-size lever)
- **Phase-11:** richer grounded KB → richer grounded discussion (richness scales with the KB).
- **Phase-10:** open-ended grounded discussion (retrieve neighbourhood → render each → VERIFY → concatenate).
- **The broader render fine-tune (parallel track, in flight):** more verbs render → more of the learned facts render
  fluently. Together: learn a real corpus → discuss it richly, grounded, hedged.

## Honest ceiling
- The PIPELINE (parse→store→grow, staged, retained) is the deliverable; the DATA SOURCE here is a real-knowledge
  mini-encyclopedia (offline-textbook-author), **swappable for a downloaded fact corpus** (ConceptNet-style triples /
  simplified-Wikipedia) — a data-source upgrade, the mechanism is source-agnostic.
- The parser handles **simple 3-word SVO**; ingesting raw complex prose needs a **fact-extraction front-end** (a
  bounded follow-on — simplify sentences to SVO triples before parsing).
- Composer FHRR capacity (~√D) bounds facts-per-brain (D=256 comfortably held 30); larger KBs raise D (validated to
  320 concepts) or shard.
- Free open-world inference beyond the learned facts remains the field wall (the honest hedge is the deliverable).

**Artifacts:** `research/runners/_fluidconv_phase12_knowledge_acquisition_derisk.py`; result
`research/findings/raw/_fluidconv_phase12_knowledge_acquisition.json`.
