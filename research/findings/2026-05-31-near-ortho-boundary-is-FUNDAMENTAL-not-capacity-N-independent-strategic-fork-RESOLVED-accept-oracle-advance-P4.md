# Near-ortho boundary is FUNDAMENTAL (per-pair-overlap-set), NOT capacity-limited -> strategic fork RESOLVED with evidence: accept the oracle near-ortho code + advance P4; the only biological escape is months-scale richer TRAINING (intrinsically less-overlapping concepts), NOT a readout change or a "richer substrate" with the same per-pair overlap. The random-projection between-concept floor is FLAT at ~0.48 from N=4 to N=16 concepts (delta +0.002) -- near-ortho (<0.30) is unreachable even at 4 concepts. So the floor is set by the substrate activity's intrinsic per-pair overlap (~0.75), independent of concept count; more dims/concepts at the same overlap would not help. Combined with the 3 convergent coding-method results (spiking DG, random projection, learned decorrelation all hit the separation-vs-reliability frontier), this DEFINITIVELY closes the cheap-first biological VSA-symbol-grounding investigation.

**Date:** 2026-05-31
**Status:** Decision-relevant capstone to the biological-symbol-grounding investigation. Resolves the strategic fork (accept-oracle-advance-P4 vs commit-to-richer-substrate) with evidence. Cheap numpy probe; no protected import.

## The decisive measurement (random-projection between vs concept-count)

| N concepts | raw between | random-proj between | random-proj within |
|---|---|---|---|
| 4 | 0.755 | 0.486 | 0.620 |
| 8 | 0.741 | 0.478 | 0.635 |
| 12 | 0.756 | 0.473 | 0.651 |
| 16 | 0.762 | 0.488 | 0.656 |

The random-projection between-concept floor is FLAT at ~0.48 across N=4..16 (delta +0.002). Near-orthogonality (between < 0.30) is NOT reachable even at 4 concepts.

## Interpretation (decision-relevant)

- The near-ortho floor (~0.48) is set by the substrate activity's intrinsic PER-PAIR overlap (~0.75), NOT by concept count. Any two concepts' activity projects to ~0.48 cosine regardless of how many concepts exist. This is Johnson-Lindenstrauss behaviour: a random projection approximately preserves cosines, so a 0.75-overlapping input floors at ~0.48 after sparsification.
- Therefore a "richer substrate" in the sense of MORE DIMENSIONS / MORE CONCEPTS at the SAME per-pair overlap would NOT push toward near-ortho (the probe shows N-independence). That fork option is LOW-merit.
- The ONLY way to reach near-ortho is to REDUCE the intrinsic per-pair overlap -- i.e. concepts whose ACTIVITY is less-overlapping by construction. That requires richer TRAINING (more data / a bigger model -> intrinsically more-distinct concept representations), the months-scale pretraining direction. And even then, the Foldiak result warns that pushing learned reps toward near-ortho risks the over-sparsification frontier.

## Strategic fork RESOLVED (evidenced recommendation)

The biological VSA-near-orthogonal-symbol-grounding investigation is COMPLETE and the boundary is robust:
- 3 convergent coding methods (spiking DG 0.66/within-collapse; random projection 0.45-0.48 floor/reliable; learned decorrelation 0.30/over-sparsified+dead) all hit the same separation-vs-reliability frontier.
- The floor is per-pair-overlap-fundamental, N-independent (this finding).

RECOMMENDATION (for the owner): ACCEPT the oracle near-ortho code (G.20 Kanerva-SDM external patterns) as an irreducible ENGINEERING component, and advance the validated P4 retrieval stack as the deliverable (multitag 90%, directional multi-hop "trace" shipped this session, 160/320 concepts, hierarchy, yes/no, tokenization). The honest, well-tested BOUNDARY is the biology-translatable scientific deliverable.

The ONLY biological escape -- months-scale richer TRAINING to learn intrinsically-less-overlapping concept representations (predictive-coding / bigger-model pretraining; the Phase-2 BPTT direction, previously toy-scale falsified at ~134K params) -- is HIGH-COST and UNCERTAIN (the Phase-2 toy-scale result + the Foldiak over-sparsification both caution). It should be an explicit OWNER strategic decision, not an autonomous launch. Pending owner direction, advance P4.

## Discipline

Throwaway CPU probe; stdlib+numpy + cached activity; no protected/frozen/moat/sim module touched. The capacity-vs-fundamental question was the right decision-relevant cheap test before recommending months-scale work; it returned FUNDAMENTAL decisively (flat floor across N), which makes the accept-oracle recommendation evidence-based, not a giving-up. The investigation arc (DG boundary -> modular instrument-invalid + ID-separability clarification -> Foldiak over-sparsification -> N-independence) is a coherent, honest, biology-translatable characterization of exactly what this substrate's activity can and cannot ground.
