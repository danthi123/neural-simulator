---
type: finding
status: go
date: 2026-08-17
mechanism: wave1-banking
---
BURNDOWN 3E — brain-owns-generation wiring de-risk: GO (6/6 seeds).

Result: the b2 generative-replay proposer, wired into the conversational turn as a GENERATE channel alongside retrieval, produces novel, grounded, moat-verified propositions on all 6 seeds. Both runs emit OVERALL VERDICT: GO — seeds 42/43/44 (3seed.log) and 100/101/102 (json + s100-102.log). Headlines: novel-composition mean 0.570/0.652 (range 0.511-0.689) vs the measured retrieval baseline 0.0; plausibility advantage over random recombination min 14.5x (up to 24.2x); shuffled-graph collapses the plausibility to ~the random floor on every seed; 0 hypothesis->known-fact moat leaks and 0 negated facts re-proposed across all 6 seeds; untaught-cue abstention 1.000 / min 0.95 (bar 0.95); lesioning the plausibility gate floods nonsense (13-18% plausible vs 30-43% gated). <!--derived--> (means/ranges over seeds; values live in the cited per-run artifacts)

What makes it load-bearing: four independent anti-cheats, not a falling metric — a random-recombination floor, an edge-shuffle control that destroys neighborhoods while preserving the marginal edge distribution, a gate lesion, and a re-check that the no-confab moat still abstains on every generated hypothesis.

Honest residual: this is a WIRING de-risk at toy CPU scale (8x8 taxonomy, 24 affirmed + 12 negated facts), not production-default and not a fully-spiking loop. Only the single generative DRAW is spiking (soft-WTA over an Izhikevich bank, confirmed by SIM_BRIDGE Izhikevich init in the logs); the plausibility/likelihood is a host-computed PPMI co-occurrence matrix, the store + moat is the RF phasor composer (host numpy), and the novelty/plausibility/non-contradiction gates are host code. So 'brain owns generation' = the LEARNED structure (not a host template) drives plausibility and the draw is neural. The corpus/PPMI graph is built with a fixed seed (tau=0.477, universe=45 identical across all 6 seeds), so seeds vary fact-sampling + draw RNG only, not the cortex. GO gates the DECISION to wire the emerge generate channel; it does not itself demonstrate production-default integration.

Banked artifacts (this branch): the 100/101/102 GO JSON is banked as `research/findings/raw/_burndown_3E_brain_owns_generation_s100-102.json` (+.prov.json) to avoid overwriting main's earlier 42/43/44 GO JSON at `research/findings/raw/_burndown_3E_brain_owns_generation.json` (novel_mean 0.711); full stdout evidence for both seed-triples is in `research/findings/raw/emerge/burndown_3E_brain_owns_generation_3seed.log` and `..._s100-102.log`. <!--derived--> (0.711 is main's earlier 42/43/44 record, cross-referenced here)

Note: the seeds 100/101/102 raw JSON is a historical pre-gate artifact (no `preconditions` block) retained at its origin path (cited above); the committed `.log` files and this finding carry the verdict evidence.
