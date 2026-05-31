# Foldiak learned-decorrelation cheap probe = BOUNDARY (over-sparsifies). A biology-grounded LEARNED decorrelation (Foldiak 1990 local anti-Hebbian sparse coding) DOES push the substrate concept codes toward near-orthogonal (between-concept cosine 0.299, beating the fixed-random-projection floor of 0.488) -- but it gets there by OVER-SPARSIFYING: within-concept reliability collapses to 0.484 AND 7 of 16 concepts produce all-zero (dead) codes, so the apparent near-orthogonality is partly a dead-code artifact. It hits the SAME separation-vs-reliability frontier as the spiking DG and the fixed random projection. THREE independent coding methods (spiking DG, fixed random projection, learned anti-Hebbian decorrelation) now fail to reach near-orthogonal AND reliable AND all-alive on this substrate activity -> the near-orthogonality bar for clean VSA binding is a GENERAL property of the substrate activity's structure, NOT method-specific. The oracle near-orthogonal code is irreducible-from-this-activity. Definitively converges on accepting the oracle as an engineering component + advancing the validated P4 stack.

**Date:** 2026-05-31
**Status:** Cheap-first CPU/numpy probe of the LEARNED-decorrelation mechanism (Foldiak 1990) for the near-orthogonality bar -- a genuinely-new, biology-grounded, check-existing-first'd direction (only Albus cerebellar anti-Hebbian LTD existed in-project, unrelated; no Foldiak/decorrelation-learning anywhere in code/plans/catalog). Honest BOUNDARY. Completes the near-orthogonality boundary characterization with a third independent method.

## Why this probe (the genuinely-new mechanism)

The modular-coding clarification established: the substrate concept activity is ID-separable (within 0.896 > between 0.768) but the unmet bar is NEAR-ORTHOGONALITY (between -> ~0) for clean VSA binding; a FIXED random-projection k-WTA floors at between ~0.45 (it cannot ACTIVELY decorrelate). Foldiak's lateral anti-Hebbian weights ACTIVELY decorrelate the output units (push correlated concepts apart) with STABLE learned forward features -> a genuinely different mechanism than the fixed projection (which floored) or the spiking DG (which lost reliability via near-threshold instability). Reasoned hypothesis: learned decorrelation could reach near-orthogonal AND stable where neither could. Cheap-first numpy test before any spiking anti-Hebbian build.

## Result (multi-seed 42/43/44; controls built in)

| method | between (separation) | within (reliability) | note |
|---|---|---|---|
| RAW activity (no transform) | 0.762 | 0.896 | the input; ID-separable, not near-ortho |
| FIXED random-projection k-WTA | 0.488 | 0.656 | the floor: keeps reliability, can't reach near-ortho |
| FOLDIAK learned decorrelation | **0.299** | **0.484** | reaches near-ortho BUT loses reliability + dead=7/16 |

Instrument valid (random control reproduced ~0.49 floor; raw ~0.76).

VERDICT: BOUNDARY. Foldiak DOES beat the random floor on separation (0.299 < 0.488 -> it genuinely actively decorrelates, unlike the fixed projection) -- but by OVER-SPARSIFYING: within collapses to 0.484 (< the 0.60 reliability bar) AND 7 of 16 concepts go DEAD (all-zero codes). The apparent near-orthogonality (0.299) is partly a dead-code artifact: zero-vectors are trivially orthogonal, so the between-mean over all 16 (including ~7 dead -> cos 0 pairs) is deflated; the genuine live-concept separation is worse than 0.299.

## Scrutiny (scrutinize-a-PASS-and-a-FAIL; the auto-message was misleading)

- The probe's auto-verdict BOUNDARY MESSAGE originally said "does NOT beat the random floor" -- FACTUALLY BACKWARDS (0.299 < 0.488 = it DID beat on separation). Corrected the probe message to the true mechanism: OVER-SPARSIFICATION. The verdict (BOUNDARY) is correct (within 0.484 < 0.60); only its stated reason was wrong.
- The dead=7/16 is the load-bearing confound: the anti-Hebbian + threshold-homeostasis drove half the concepts to silence. A gentler param point (lower anti-Hebbian beta / homeostasis gamma / more units) would reduce dead codes + raise within -- but then between would rise back toward the random floor (less decorrelation). That is MOVING ALONG the separation-vs-reliability frontier, not OFF it (same lesson as the DG size-lever's tradeoff curve + the graded-sparse model's "no sweet spot"). So a param sweep would trace the frontier, not thread it.

## The general conclusion (three methods converge)

Near-orthogonal (between -> ~0) AND reliable (within high) AND all-alive concept codes are NOT achievable from this substrate activity by ANY single coding stage tested:
- spiking DG (k-WTA via spikes): between ~0.66, within collapses at sparse k (near-threshold instability).
- fixed random-projection k-WTA: between floors at ~0.45, within ok (~0.66) -- can't reach near-ortho.
- learned anti-Hebbian decorrelation (Foldiak): reaches near-ortho (~0.30) but within collapses (0.48) + kills 7/16 codes.
Each sits on the SAME separation-vs-reliability frontier; none reaches the near-ortho + reliable corner. So the near-orthogonality bar is a GENERAL property of the substrate activity's intrinsic structure (16 concepts, between 0.768, within 0.896), NOT a method-specific artifact. This SUPERSEDES the earlier "spiking-dynamics-specific" framing for the NEAR-ORTHO bar (that framing holds only for the within-COLLAPSE / identifiability, which clean methods avoid; the near-ortho bar is general).

## Disposition

The biologize-the-VSA-near-orthogonal-symbol line is now DEFINITIVELY boundary-characterized (3 independent methods converge on the same frontier). The oracle near-orthogonal code (G.20 Kanerva-SDM external patterns) is genuinely irreducible from the substrate activity -- it is an engineering component, not a biology shortcut that a cleverer code removes. CONVERGES decisively on night-synthesis P3(c): accept the oracle, advance the validated P4 retrieval stack (multitag 90%, directional multi-hop shipped, 160/320 concepts, hierarchy, yes/no, tokenization). The honest, well-tested BOUNDARY is the biology-translatable scientific deliverable. A genuinely-new biological escape would require a different SUBSTRATE (not a new readout of the same activity) -- e.g. learning concept representations that are near-orthogonal BY CONSTRUCTION during acquisition (a months-scale predictive-coding/pretraining effort), which is a strategic owner decision, not a cheap probe.

## Discipline

Throwaway CPU probe only; stdlib+numpy + cached activity; no protected/frozen/moat/sim/runner module touched. Check-existing-first done thoroughly (code+plans+catalog) before building -- the mechanism was genuinely unexplored. Frozen three-state bar set before the run; instrument-validity gate applied (random + raw controls reproduced). The PASS-ish near-ortho number (0.299) was scrutinized and found CONFOUNDED (dead codes) -> honest BOUNDARY, not an over-read RESOLVES. The auto-verdict's misleading message was corrected in the artifact. No spiking anti-Hebbian build started.
