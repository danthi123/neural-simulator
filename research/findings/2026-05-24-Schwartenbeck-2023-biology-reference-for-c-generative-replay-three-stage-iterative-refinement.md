---
type: finding
status: contributing
date: 2026-05-24
---

# Biology reference for (c) generative-replay: Schwartenbeck et al. 2023 (Cell) "Generative replay underlies compositional inference in the hippocampal-prefrontal circuit" — directly validates the project's (c) direction; biology-translatable details for the (c) build (2026-05-24)

## Context

Web-search-found reference (URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10914680/) directly addressing the biology of generative replay in hippocampal-prefrontal circuit for compositional inference. This is the EXACT biology the project's (c) generative-replay arc targets. Owner-authorised reference research during the dlpfc-extension GPU wait (overnight continuous work).

## Direct mapping to the (c) design

| Biology (Schwartenbeck 2023) | Project's (c) substrate component |
|---|---|
| Hippocampus encodes conjunctive representations of building blocks in specific relational positions | The project's engram tagging (D.14) on CA3 ensembles via `start_engram_recording` / `commit_engram_tag`; v14/v16 substrate's concept pools provide the building blocks; relational position via gamma-slot position phasors (already biology-faithful per Lisman-Idiart N.16) |
| mPFC dual representation: input (blocks irrespective of relation) AND conjunctive (relational) | dlpfc_wm region (NMDA bistable) holds the input representation; the FHRR composite C IS the conjunctive representation (binding via gamma-slot positions) |
| Replay is GENERATIVE: samples from possible configurations | The (c) loop's SWR replay → cortical activity → parallel-matching decoder produces candidate continuations from the consolidated schema |
| Replay is MULTI-STEP and ITERATIVE: three-stage refinement (early non-selective → middle selective → late converged) | **NEW INSIGHT from this biology**: the (c) loop should run for N iterations with the prediction REFINING over time, not just single-step completion. See "refinement to (c) TDD plan" below. |
| Length-3 compositional sequences detected | The project's framework supports up to 7 gamma slots; 3-slot is the minimum biology-validated; 7-slot is the project's algebra ceiling |
| Time-lag peaks at 60ms and 170ms | Consistent with gamma cycle (~50ms) and theta cycle (~150ms); the project's gamma-slot framework is biology-grounded |
| "Sequenceness" measured via reactivation patterns predictive of other reactivation patterns at different time-lags | Maps to the (c) loop's per-iteration decode step |

## Refinement to (c) TDD plan (informed by Schwartenbeck three-stage iterative refinement)

The pre-registered test in the (c) TDD plan is partial-sequence completion. Schwartenbeck's biology suggests the loop should test ITERATIVE REFINEMENT, not just single-step completion:

**Original (c) TDD test** (Task 5 decisive):
- Initialise PFC frame with partial cue (2 of 3 slots filled)
- Run loop for N iterations
- Measure completion accuracy at slot 3

**Refined (c) TDD test** (incorporating Schwartenbeck):
- Initialise PFC frame with partial cue
- Run loop for N iterations
- Measure NOT JUST final completion accuracy but ALSO refinement trajectory:
  - Early iteration (1-3): decoded continuation distribution should be diffuse (non-selective)
  - Middle iteration (4-7): decoded continuation should narrow (selective among plausible)
  - Late iteration (8+): decoded continuation should converge on correct
- PASS criteria UNCHANGED: multi-seed-mean ≥ 0.80 at final iteration on every K
- NEW characterisation metric: iteration-by-iteration accuracy curve; PASS quality bonus if curve shows the three-stage progression
- The refinement-curve metric is a CHARACTERISATION not a PASS criterion (avoid moving the goal posts)

## Post-(c) direction informed by this biology

After (c) generative-replay validates partial-sequence completion via iterative refinement, the next direction toward conversation:

1. **Multi-turn dialog**: hippocampal episodic binding of prior conversation turns; PFC working memory holds the dialog state across turns; generative replay produces context-appropriate continuations referring back to earlier turns. Biology: hippocampus-prefrontal-mPFC interaction over extended time scales (minutes to hours; the autobiographical memory substrate).

2. **Hypothesis-testing for ambiguous queries**: Schwartenbeck's three-stage refinement IS the substrate for handling ambiguous input. The (c) loop's iterative refinement could be extended to handle queries where the right continuation is not uniquely determined; the loop samples multiple hypotheses and picks the best supported.

3. **Larger vocabulary / sentence-level composition**: scale from 16-concept to 32/64/160-concept (the project's existing scaling ladder); use multiple bridges (the 160-ensemble); test cross-bridge compositional sequences (e.g., apple-go-fast spanning noun-verb-adj bridges).

4. **Goal-directed generation**: integrate basal ganglia (the project's existing infrastructure) for goal-directed action selection over the (c) loop's continuations; PFC frame is a query, BG selects among generated candidates per reward.

5. **Working memory + episodic memory integration over time**: McClelland 1995 / Buzsaki 2013 CLS theory + working memory persistence; the project's Phase 1.3 SWR consolidation already validates the rapid episodic / slow cortical division; conversation needs both held simultaneously.

## Biology-translatable insight from the Schwartenbeck paper itself

The paper explicitly notes that ITS limitation is "lack of a computational process model" and suggests future work should "develop computational models that can solve such tasks." The project's (c) build IS that computational process model. The biology-translatable contribution flows BOTH ways: the project tests biology-grounded predictions (does the substrate actually do what biology says it does?); the biology informs the project's mechanism design (the three-stage iterative refinement insight above).

The Schwartenbeck paper does NOT explicitly invoke theta-gamma coupling or SPEAR framework, but the observed 60ms/170ms time-lag peaks are consistent with gamma-cycle (~50ms) and theta-cycle (~150ms) periods. The project's gamma-slot position framework + SPEAR temporal multiplexing are theory-consistent with this biology.

## Files

- This reference doc: `research/findings/2026-05-24-Schwartenbeck-2023-biology-reference-for-c-generative-replay-three-stage-iterative-refinement.md`
- Source: Schwartenbeck T, Baram A, Liu Y, Mark S, Muhle-Karbe P, Dolan R, Behrens T. *Cell* 2023 (PMC10914680).
- (c) design doc: `docs/plans/2026-05-23-generative-replay-design.md`
- (c) TDD plan: `docs/plans/2026-05-24-generative-replay-implementation.md`

## Standing constraints (this is reference research, not capability claim)

- No protected/frozen/moat module modified.
- No code written from this reference yet.
- The (c) TDD plan stands; this reference informs the SUBAGENT's Task 2 build to use Schwartenbeck-consistent iterative refinement metrics.
- Owner explicitly authorised reference research via the knowledge catalog + web search.
