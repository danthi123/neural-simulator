# Taxonomy data-acquisition research gate: the multi-level is-a taxonomy is CHEAPLY UNLOCKABLE via data acquisition — the ready EMERGE-44/45 stacked pooler just needs a clean is-a-structured corpus. PRIMARY = Simple English Wikipedia first-sentence definitions ("A dog is a mammal", emergent-defensible: learn is-a from definitional TEXT via the existing copular miner); FALLBACK = Wikidata P279 subclass-of (text-converted). Reuse-existing-code de-risk. Read-only research; NO `sim/` edit.

**Date:** 2026-07-08
**Type:** deep-research gate (read-only) for the taxonomy data-gate (triply-confirmed CYCLE 1038).

## The verdict
The multi-level taxonomy is a DATA gate, not a mechanism gap (the EMERGE-44/45 stacked pooler is validated on synthetic is-a data). The path forward is data acquisition of a genuinely is-a-structured corpus. Ranked candidates:

| Rank | Resource | Clean is-a? | License / access | Emergent-defensible? |
|---|---|---|---|---|
| **1 (primary)** | **Simple English Wikipedia first-sentence definitions** | ✅ ~clean genus-differentia ("X is a Y") | CC-BY-SA, free dumps | ✅ STRONG — brain reads definitions (experience) → copular miner extracts is-a → stacked pooler discovers the hierarchy |
| 1B (fallback) | Wikidata P279 (subclass-of) | ✅ curated, 100% precision | CC0 public domain | ⚠️ borderline — a curated graph; text-convert ("dog subclass-of mammal" → "a dog is a mammal") to recover the emergent path |
| 2 | WebIsA (400M Hearst hypernyms) | ✅ (precision ~0.44 "is a") | CC-BY-NC-SA (blocks deploy) | ⚠️ research-only |
| 3 | ConceptNet IsA | ✅ crowdsourced | CC-BY-SA | ⚠️ hand-built edges; text-convert needed |
| — | Dictionary genus (Webster's 1913) | ✗ noisy | — | ALREADY TESTED + REJECTED |

## The emergent-directive judgment
Simple English Wikipedia is the most defensible: the brain LEARNS is-a from clean definitional SENTENCES (legitimate encyclopedic experience, like a student reading definitions), via the SAME copular miner + stacked-pooler pipeline the project already has. The encyclopedia is the *environment*, not a hand-supplied structure. Wikidata P279 is acceptable only text-converted (else it's injecting a curated graph = a shortcut).

## The cheapest de-risk (reuse-existing-code, 2-3 days scoped)
1. **Acquire** Simple English Wikipedia first-sentence definitions (Wikimedia dump), OR (lighter) fetch a Wikidata P279 taxonomic subset via the project's existing Wikidata fetcher (the fluid-conversation P15 arc already fetches P279 subclass→isa triples) and text-convert.
2. **Extract** is-a pairs with the EXISTING `_realcorpus_copular_isa_miner_derisk.py` (zero changes — NP-head "X is a Y", hub superordinates ≥3 children). Gate: `len(hubs)>=8 and len(hub_pairs)>=40` (expected PASS on clean definitions, vs the WikiText NEGATIVE).
3. **Consume** the mined is-a pairs with the EMERGE-44/45 stacked pooler (`_emerge44_stacked_pooler_derisk.py`); test held-out-member inheritance vs a permuted-super control (metrics from EMERGE-44: held-out ≥0.80, permuted collapse ≥0.25, L2-grouping ≥0.15), 6-seed.

Reusable code (no `sim/` edit): `corpus_stream.py`, `_realcorpus_copular_isa_miner_derisk.py`, `_emerge44_stacked_pooler_derisk.py`, `_emergent_vocab_breadth_scale_derisk.py`.

## What this establishes
The taxonomy data-gate is converted from a blocker into a concrete, actionable acquisition plan: a clean is-a-structured corpus (Simple Wikipedia definitions / Wikidata P279 text-converted) fed through the project's existing copular miner + the validated stacked pooler. The next step is the data-acquisition de-risk. Single-level reasoning (inherit + cancel) stands; this is the path to MULTI-level.

## Files
Research gate (read-only). De-risk reuses `_realcorpus_copular_isa_miner_derisk.py` + `_emerge44_stacked_pooler_derisk.py`. Prior: the triply-confirmed data-gate `2026-07-08-copular-isa-extraction-NEGATIVE-taxonomy-data-gate-confirmed-3rd-approach.md`; the ready mechanism `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`.
