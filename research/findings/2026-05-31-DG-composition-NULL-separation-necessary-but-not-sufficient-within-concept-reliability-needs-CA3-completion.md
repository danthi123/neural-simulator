# DG-composition decisive test: NULL. DG pattern-separation is NECESSARY-BUT-NOT-SUFFICIENT for composition. The sparse k-WTA DG code SEPARATES concepts (between-concept cosine 0.18/0.10, excellent) but is UNRELIABLE WITHIN-concept -- silent on one observation-half for ~1/3-1/2 of words, and storage vs query DG of the SAME concept are near-disjoint -- so unbind cannot recover the filler. This is the classic separation-vs-reliability tension, and it is EXACTLY what the next trisynaptic region, CA3 PATTERN COMPLETION, exists to resolve. The convergent prescription refines from "DG separation" to "the full trisynaptic loop: DG separates, CA3 completes/stabilizes."

**Date:** 2026-05-31
**Status:** Honest NULL for DG-separation-ALONE as the composition-symbol carrier. Multi-seed (42/43/44), two sparse regimes, apples-to-apples vs the pool baseline. The scientific deliverable: a precisely-characterized necessary-but-not-sufficient boundary that points to the validated next mechanism (CA3 completion, P1 D.13). The DG gate's separation PASS stands; this test isolates what separation alone cannot do.

## The test + result

The DG gate confirmed DG orthogonalizes the overlapping concept activity (0.806 -> 0.30/0.17). This test asked the end-to-end question: do composition symbols DERIVED from the DG-separated activity clear the 0.80 bar at {2,3,5}, where the raw concept-pool symbols failed (L2 0.834, L3 0.694, L5 0.575)?

Probe (`research/findings/raw/_dg_composition_test.py`, throwaway): the gate's hippocampus bridge (byte-faithful, DG=800, FFi present), driving the genuine trained-substrate concept activity (denoise64 caches) into DG via the gate's fixed sparse afferent, ec held silent. Per concept, DISJOINT observation halves (storage = mean obs[:32], query = mean obs[32:64]) each drive DG -> a DG-sized deriver (N_DIM=512, dg_dim=800) -> storage/query phasor symbol. FHRR compose + argmax cleanup over the 4-filler DG vocab, 60 trials/load. Two regimes (headline sparsity ~0.05, anchor ~0.02), 3 seeds.

| Load | Pool baseline (raw activity) | DG headline (sparsity 0.05) | DG anchor (sparsity 0.02) |
|---|---|---|---|
| L=2 | 0.834 | 0.409 | 0.361 |
| L=3 | 0.694 | 0.374 | 0.381 |
| L=5 | 0.575 | 0.330 | 0.323 |

DG-symbol composition is BELOW the pool baseline at every load in both regimes, only marginally above the 0.25 chance floor. **PRE-REGISTERED VERDICT: NULL.**

(The probe also printed a `no-silent` column -- DISREGARD it; the controller and subagent agree it is a vocab-collapse artifact: excluding silent words shrinks the 4-filler vocab to ~1 element on several seeds -> trivial 100% over a 1-vocab, chance no longer 0.25, not comparable. The all-words column is the honest comparison.)

## The mechanism (the precise boundary)

The failure is NOT separation -- it is WITHIN-CONCEPT RELIABILITY:
- Between-concept DG-symbol cosine: 0.18 (headline) / 0.10 (anchor) -- separation is excellent, confirming the gate.
- SILENT-DG words (zero firing on >=1 half): ~4-5 of 12 (headline), ~5-7 of 12 (anchor). A silent half -> a degenerate symbol.
- Even for non-silent words, storage and query DG of the SAME concept are near-disjoint (e.g. seed 42 "hot": store sparsity 0.000, query 0.229). The sparse k-WTA picks DIFFERENT winners for the two observation halves of the same concept -> the query DG symbol does not match the stored DG symbol -> FHRR unbind recovers noise.

This is the classic separation-vs-reliability (separation-vs-completion) tension, also visible in the gate's dose-response: sparse DG separates (cosine low) but is unstable within-concept; dense DG is stable but does NOT separate (cosine climbs to 0.5-0.8). No single DG operating point gives both separation AND within-concept reliability at this bridge scale.

## The resolution (the biology already prescribes it): CA3 pattern completion

The trisynaptic loop is DG -> CA3 by design precisely because DG separation alone is unstable. CA3 is a recurrent ATTRACTOR autoassociator: it takes a sparse/partial/noisy DG pattern and COMPLETES it to a stable stored ensemble (pattern completion). That is exactly the within-concept reliability the DG code lacks -- both the storage-half and query-half sparse DG codes of one concept should settle to the SAME stable CA3 ensemble, while DISTINCT concepts settle to distinct ensembles (separation preserved). P1 validated CA3 completion (D.13: 50% of a stored CA3 ensemble recalls the full pattern at cosine 0.748). So the symbol should be derived from the COMPLETED CA3 code, not the raw separated-but-unstable DG code.

So the convergent prescription refines: NOT "DG pattern-separation" alone, but "the full hippocampal trisynaptic loop -- DG separates (this test confirms 0.82 -> 0.18), CA3 completes/stabilizes (the next test)." Both halves are individually P1-validated; this arc is testing whether they COMPOSE into a composable, reliable, separated concept symbol.

## Next + honest risk

Refined next test: route concept -> DG -> CA3 (train CA3 ensembles per concept via the D.13 direct-CA3 methodology: co-fire the full pattern + open the ca3_swr_burst gate to store; recall by the partial/noisy DG drive), derive the composition symbol from the CA3 (completed, stable) code, re-test composition at {2,3,5}. HONEST RISK: P1's D.13 was seed-variable (direct-CA3 passed at 0.748; EC-driven FAILED), so CA3 completion's reliability on the DG-separated concept activity is uncertain. If CA3 gives both separation + reliability -> composition clears the bar -> the trisynaptic loop biologizes the oracle lookup (the artificial-life milestone). If CA3 cannot simultaneously separate AND complete the concept activity at this scale -> a deeper, honest boundary (the substrate's hippocampus cannot serve as the compositional symbol source). Either is a real biology-translatable result.

## Discipline

Throwaway probe only; no tracked .py modified (gate probe, denoiser probe, sim/, activity_level_integration, spiking_phasor_fhrr, text_minimal_isolation reused by import byte-unchanged). No bars moved. No autograd. The NULL was scrutinized (the no-silent artifact identified + disregarded; the mechanism -- within-concept instability, not separation -- confirmed in the per-word data; between-concept separation genuinely ~0.18). Honest NULL = the deliverable; it refines the convergent hypothesis to the full trisynaptic loop and points to the validated next mechanism (CA3 completion).
