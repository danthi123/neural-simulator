# Fluid conversation — Phase 15 GO: REAL grounded-knowledge breadth from Wikidata (the ConceptNet-down alternate)

**2026-07-01 (autonomous; owner steer = grow grounded knowledge / you-choose-keep-going).** Phase-12 built the
acquisition PIPELINE and noted the DATA SOURCE is swappable for a real fact corpus. ConceptNet's REST API returned 502
for multiple cycles (their server down). This delivers the breadth lever via an alternate **real, curated,
encyclopedic** source — **Wikidata** (a different server, up): its triples are (entity, property, value) = SVO-ready.
Reuse-by-import (the validated parse+store + Phase-13 isa-chase); **NO `sim/` edit**; CPU (brain-only).

## Result — GO (3 seeds: 42, 43, 44)
`_fluidconv_phase15_wikidata_breadth_derisk.py`: fetch a bounded set of REAL facts for common concepts via curated
clean properties (**P279 subclass-of → `isa`**, **P527 has-part → `has`**), simplify each value to a clean single head
token, **cache to JSON** (fetch-once → reproducible + offline for multi-seed), ingest via the validated parse+store, and
converse. **24 real Wikidata facts** (e.g. *dog isa mammal · cat has tooth/ear/eye · tree isa plant · tree has
root/bark/trunk · plant has root/stem/layer · river has water*):
- **ACQUISITION:** recall **24/24** every seed (all real facts learned + recalled).
- **REAL TRANSITIVE-ISA INHERITANCE:** the isa link is chased hop-by-hop → **dog → mammal → vertebrata → chordata** — a
  real MULTI-LEVEL taxonomic chain, every edge a Wikidata subclass assertion. A dog inherits membership in its higher
  categories (Collins-Quillian). This is exactly the inheritance Wikidata's subclass chain encodes.
- **STAGED RETENTION:** ingested in 2 batches; batch-1 **14/14** still recalled after batch-2 (no catastrophic
  forgetting).
- **MOAT:** a never-fetched concept ("dragon") → abstain (0 false-accepts).

## The key data-model finding (why the inheritance test changed)
Wikidata annotates **has-parts on species** (`cat has tooth`), **not on the class** (`mammal` has no P527 has-part). So
class-level part-inheritance ("dog isa mammal, therefore dog has <mammal's parts>") is **not in the data**. The
inheritance that IS in the data — and is genuinely rich — is the **transitive subclass (isa) chain**: dog → mammal →
vertebrata → chordata. That is real Collins-Quillian taxonomic inheritance, and it chases cleanly on the composer.

## Honest ceiling
- The Wikidata→SVO front-end is **host-side data-prep** (supplying grounded facts for the brain to LEARN — legitimate
  "environment", per BRAIN-BASED-ONLY), NOT a brain mechanism; the brain still learns via the validated
  `composer.store`. The simplification is single head-token of the value label + curated clean properties (P279/P527).
- Richer relations (diet, habitat, capable-of) + multi-word values need a fuller extraction pass; some head-token
  reductions are imperfect ("river isa body" ← "river water body"). Bounded follow-on.
- Composer FHRR capacity (~√D) bounds facts-per-brain — D=256 comfortably held 24 (Phase-12 held 30); larger KBs raise
  D (validated to 320) or shard.
- Free open-world inference beyond the fetched facts remains the field wall (the honest hedge is the deliverable).

## Where this sits (the grounded-growth path, now on REAL encyclopedic data)
- **Phase-12 (GO):** the acquisition pipeline (staged, retained). **Phase-15 (this, GO):** a REAL data source (Wikidata)
  feeding it — encyclopedic grounded breadth, live.
- **Phase-13/14 (GO):** kind vs instance ("the dog" vs "dogs") + the multi-turn console — now applicable to the real
  concepts Wikidata supplies.
- ⇒ the brain learns REAL, verified, encyclopedic knowledge and converses over it (recall · transitive-isa taxonomy ·
  retention · moat), grounded + hedged. ConceptNet returning would add capable-of/desires action facts; the pipeline is
  source-agnostic.

**Artifacts:** `research/runners/_fluidconv_phase15_wikidata_breadth_derisk.py`; result
`research/findings/raw/_fluidconv_phase15_wikidata_breadth.json`; cached facts
`research/findings/raw/_fluidconv_phase15_wikidata_facts.json`.
