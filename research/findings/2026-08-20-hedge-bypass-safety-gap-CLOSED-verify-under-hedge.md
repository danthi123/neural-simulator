---
type: finding
status: live
date: 2026-08-20
mechanism: open-text-moat-verifier
lane: D-pragmatics
seeds: [42]
seed-waiver: A deterministic ROUTING fix + its before/after check — does a hedged confabulation now get verified-and-suppressed instead of bypassed, while genuine taste content still bypasses. The evidence is per-item keep/suppress correctness on a fixed labelled set, not a stochastic effect size; single seed is the substrate build seed.
instrument: research/runners/_hedged_assertion_verify_derisk.py — route_clause() 3-way router in front of the UNCHANGED extract (NPHeadBinder) + entail (FactStore) pipeline
runner: research/runners/_hedged_assertion_verify_derisk.py
artifacts:
  - research/findings/raw/_hedged_assertion_verify_derisk.json
external: NO-EXTERNAL-NEEDED — this is an in-repo routing fix for a gap our own de-risk found (2026-08-20-fluent-paraphrase-verify-suppress); a subjective-vs-factual lexical split, same host-preprocessing category as the existing is_opinion/segment_clause rules. No capability wall or paradigm claim.
---
# CLOSED: the hedge-bypass moat safety gap — a hedged assertion is now VERIFIED, not waved through

Artifact: research/findings/raw/_hedged_assertion_verify_derisk.json

**One line.** The fluency de-risk (2026-08-20-fluent-paraphrase-verify-suppress-hedge-bypass-safety-gap) found the #1
risk to widening Qwen: `is_opinion()` routed ANY hedged clause ("I believe X") to an UNCHECKED opinion bypass, so a
CONFABULATION wrapped in a hedge sailed through kept (precision 0/2 on hedge-confab) — a fluent Qwen could evade the
moat just by hedging. This lands the fix: a hedge is a STANCE marker on a factual assertion, so strip it and run the
SAME extract+entail; only genuinely SUBJECTIVE content (taste/preference, no truth-value) legitimately bypasses. The
gap is closed with real before/after numbers, extraction + entailment imported UNCHANGED.

## The fix: `route_clause()` (a 3-way router in front of the unchanged pipeline)
Priority order: (1) a SUBJECTIVE predicate present (a small declared `SUBJECTIVE_VERBS`/`SUBJECTIVE_ADJS` lexicon —
like/love/hate/prefer/enjoy/adore; delicious/beautiful/boring/best/worst/…) → the ONLY legitimate opinion bypass,
checked FIRST so taste bypasses even without a hedge ("I like cats") and even under a hedge (subjectivity wins the
race); (2) a HEDGES phrase present → STRIP the hedge (+ trailing "that") and run the SAME `extract_svo_npbind` +
`classify_claim` on the remainder exactly as a bare assertion; (3) else → unchanged. The spiking extraction
(NPHeadBinder + BridgeParser) and the FactStore entailment are imported and used byte-unchanged — only the
pre-extraction routing decision is new (the same host-preprocessing category as the existing `is_opinion`/
`segment_clause` lexical rules).

## Before / after (numpy, seed 42; independently re-run + reproduced)
| metric | before | after |
|---|---|---|
| precision-on-hedge-confab | 0/2 (0.0) | **2/2 (1.0)** — the confab now caught |
| recall-on-hedge-faithful | 2/2 (1.0) | 2/2 (1.0) — true hedged facts still kept |
| subjective-bypass-correctness (6 genuine taste items) | 2/6 | **6/6 (1.0)** |
| regression (plain/passive/synonym/embedded styles) | — | byte-identical before/after |

`"I believe Darwin proposed gravity"` now routes to `assertion_ungrounded → SUPPRESS` instead of sailing through.
The before-pass also exposed a smaller real brittleness the fix cured: 4/6 genuine taste claims ("I love the ocean")
were previously MIS-suppressed as false facts because they carried no hedge word — now they correctly bypass.

## Scope / what remains
This closes the ONE safety hole. The other two fluency residuals are UNCHANGED (confirmed byte-identical), as scoped:
synonym-verb brittleness (fails safe, over-redacts true rewordings) and reporting-clause segmentation. Order still
holds: hedge-bypass (DONE here) → synonym expansion → reporting-clause stripping; after synonym expansion, Qwen
widening is fluent as well as safe. This fix lives in a de-risk runner; wiring the corrected routing into the LIVE
open-text verifier path is the integration step (tracked with the Qwen-widening arc, #99). (Agent-built, verified.)
