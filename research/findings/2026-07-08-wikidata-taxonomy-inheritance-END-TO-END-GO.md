# The multi-level taxonomy is UNLOCKED END-TO-END (GO, 6-seed): with a clean is-a resource (Wikidata P279), the brain learns concept codes from the is-a DEFINITIONAL stream ("the dog is a mammal" — same-super members share their super as co-occurrence context) and a HELD-OUT member inherits its super's property (0.995), while the super-DERANGEMENT control COLLAPSES (0.000) — the grouping is load-bearing. The deep, triply-confirmed taxonomy data-gate is surpassed via data acquisition + the ready emergent mechanism. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_wikidata_inheritance_derisk.py` (reuses `learn_stream_codes` + the multilevel `_teach_test`; consumes the CYCLE-1040 Wikidata is-a graph). numpy. NO `sim/` edit.
**Verdict:** GO (6-seed) — real is-a data + the ready emergent mechanism → held-out multi-level inheritance, derangement collapses.

## Why this ran (closing the taxonomy unlock end-to-end)
The taxonomy data-gate was triply-confirmed (distributional + copular extraction all NEGATIVE on natural corpora); CYCLE 1040 UNLOCKED it with real Wikidata P279 (a clean multi-level is-a graph). This closes the loop: does the READY emergent mechanism (code-learning + the associative-memory inheritance read) consume the real is-a data for held-out inheritance?

## The emergent-path mechanism
- **Learn codes** from the is-a as a DEFINITIONAL stream: "the <child> is a <parent>" for each Wikidata is-a pair, shuffled/repeated. Same-super members thus share their PARENT (super) token as co-occurrence context (the EMERGE-30 shared-context grouping) → same-super members get similar codes. (`learn_stream_codes`, members=vocab, supers=hubs.)
- **Inherit** via the associative-memory read (`_teach_test`): teach a super's property to HALF its members; a HELD-OUT member (the other half) inherits the super whose taught members its code most resembles.
- **Derangement control**: teach each super's property with a DIFFERENT (permuted) super's members while querying held-out members at their TRUE super → a mismatch → inheritance must collapse if the grouping/codes are load-bearing.

## The result — 6-seed (42/43/44/100/101/102), Wikidata is-a graph (8 supers, 71 held-out queries)
```
held-out inheritance = 0.995 (min 0.986)   -- a held-out member inherits its super's property
super-DERANGEMENT    = 0.000               -- collapses completely (the grouping/codes are load-bearing)
chance               = 0.125 (1/8 supers)
VERDICT: GO (beats chance + derangement, all seeds)
```

## An adversarial-control-bug caught + fixed (the discipline working)
The first pass gave held-out=1.000 AND derangement=1.000 (PARTIAL) — the "derangement" applied the same permutation to BOTH the taught set and the queries (a CONSISTENT RELABEL = the same test), so it validated nothing. Rather than commit the false result, the bug was diagnosed and the control fixed to a genuine mismatch (teach each super with a different super's members; query at the TRUE super) → derangement now collapses to 0.000 while inheritance stays 0.995. The GO stands on the valid control.

## Honest scope (emergent-defensibility)
The is-a STRUCTURE is Wikidata-curated (a legitimate encyclopedic resource — the CYCLE-1039 judgment: an encyclopedia is legitimate "experience"); the brain LEARNS the codes + inherits from the DEFINITIONAL TEXT derived from it (not by injecting the graph). This is the data-acquisition path: DISCOVERY of the taxonomy from raw distributional experience remains gated (natural corpora lack the signal — the honest NEGATIVE); a clean is-a resource + the emergent code-learning + inheritance mechanism surpasses it. The single-level reasoning (inherit + cancel) stands; this adds validated MULTI-level inheritance over real is-a data.

## What this establishes
The multi-level taxonomy — a deep, triply-confirmed data-gate — is now UNLOCKED end-to-end: real is-a data (Wikidata P279 definitional stream) + the ready emergent mechanism → held-out multi-level inheritance (0.995), with a valid collapsing derangement control (0.000), 6-seed. Follow-on: wire multi-level inheritance into the conversational console (ask "can a robin breathe?" → inherit via the discovered is-a); the 2-level grandparent chains (dog→mammal→animal); Simple Wikipedia definitions as a fully-natural-text source.

## Files
`research/runners/_realcorpus_wikidata_inheritance_derisk.py`. Reuses `_emergent_vocab_breadth_scale_derisk.learn_stream_codes`, `_realcorpus_inheritance_multilevel_derisk._teach_test`, the CYCLE-1040 Wikidata is-a graph. Prior: the data unlock `2026-07-08-wikidata-p279-clean-multilevel-isa-taxonomy-data-gate-UNLOCKED.md`; the acquisition research `2026-07-08-taxonomy-acquisition-research-gate-simple-wiki-definitions-path.md`; the ready mechanism EMERGE-44/EMERGE-30.
