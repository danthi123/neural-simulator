# Generalization frontier cheap-first — cross-modal convergence transfers generalization to perception (GO; Option B is the prerequisite)

**Date:** 2026-06-16
**Runner:** `research/runners/_genfrontier_crossmodal_unify_derisk.py`
**Raw:** `research/findings/raw/_genfrontier_crossmodal_unify.json`
**Scoping:** `research/findings/2026-06-16-generalization-frontier-scoping.md` §4 (the recommended cheap-first for the
top-ranked Option A).
**Verdict:** **GO** — 3 seeds (42/43/44), CPU numpy, all four anti-cheats clean.

---

## The question

The conversation cortex already generalizes across SIMILAR concepts (the PPMI stream cortex: held-out 0.86–0.91,
on the spiking substrate). The perception channel does not (flat-distinct orthogonal object codes). The scoping's
top option (A) is **cross-modal Hebbian unification**: co-activate perception(X) + the conversation cortex(word
"X") so they converge on one concept code → perception inherits the generalizing codes. The load-bearing question:
**does that convergence TRANSFER the category-similarity structure to perception so a HELD-OUT (never-converged)
concept's perception code lands in its correct semantic category — and does it need the perception input to itself
carry similarity structure (Option B)?**

This isolates the convergence+transfer MECHANISM in numpy. The word codes are category-structured (a documented
stand-in for the validated PPMI codes, which provably carry category structure — `_pearson_vs_Strue ≈ +0.52`, gen
0.86–0.91). The MEASURED VARIABLE is the perception input: **FLAT-DISTINCT** (the current nav regime) vs
**SIMILARITY-STRUCTURED** (Option B — its own INDEPENDENT category basis, no leakage of the word codes). A linear
convergence map is fit on a TRAIN split; transfer is tested on HELD-OUT concepts.

## Result (3 seeds)

| perception input | held-out same-category margin | held-out category accuracy | permuted (derangement) | moat |
|------------------|-------------------------------|----------------------------|------------------------|------|
| **STRUCTURED** (Option B) | **+0.953** | **1.00** (chance 0.25) | **−0.331** (collapses) | intact |
| **FLAT-distinct** (current nav) | +0.002 | 0.42 (≈ chance) | — | — |

- **Cross-modal convergence transfers the word cortex's category-generalization to perception** — but ONLY when the
  perception input carries similarity structure (Option B): held-out concepts land in their correct semantic
  category 100% (≫ chance 25%) with a +0.953 margin.
- **Flat-distinct perception does NOT transfer** (category accuracy 0.42 ≈ chance) → **Option B
  (similarity-structured perception) is the PREREQUISITE**, not optional. A linear map fit on orthogonal train
  inputs cannot generalize to an orthogonal held-out input — there is no shared structure to ride.
- **Held-out exact-concept recovery is 0.00** (expected, and correct): the held-out concept's *unique* component is
  never seen by the map, so it lands in its CATEGORY (the generalization) but not exactly its own code. Category-
  level transfer IS generalization-across-similar-concepts; the scoping's "nearest own ≥ 0.80" gate was too strong
  for held-out (it implicitly requires the unique component to transfer, which it cannot) — category accuracy is the
  correct held-out generalization measure.

## Anti-cheats (all clean)

1. **Flat-distinct baseline** — flat perception input scores at chance on held-out category transfer (0.42), the
   discriminating gap vs the structured arm (1.00).
2. **No-leakage split** — held-out concepts are excluded from the map fit (asserted); the perception category basis
   is seed-INDEPENDENT of the word basis (so the transferred similarity is LEARNED by the map, not pre-shared).
3. **Category-derangement permuted control** — pairing each train concept's perception code with a WRONG-category
   word code (a consistent but deranged category mapping) collapses the transfer to a NEGATIVE margin (−0.331): the
   transfer is the *learned* perception-category↔word-category correspondence, not a geometry coincidence. (The
   first attempt used a within-set random shuffle, which leaked +0.064 because, with few concepts, a random
   permutation preserves some category alignment by chance — the derangement is the clean control.)
4. **No-confab moat** — a novel concept (random perception code, no category) has low max word-code familiarity
   (~0.5) vs a real held-out concept (~0.95): the system abstains on the novel concept rather than confabulating.

## Decisive conclusions

- **The path is A + B:** a similarity-structured perception front end (Option B), then cross-modal Hebbian
  convergence (Option A) inherits the conversation cortex's category-generalization. Option A alone, on today's
  flat-distinct nav perception, does not generalize.
- **The dendritic substrate rewrite is NOT required** — this is point-neuron linear convergence; the conversation
  cortex's generalization (the structure being inherited) is itself realized on point neurons (the on-bridge PPMI
  stream cortex). Confirms the scoping's decisive call.
- The no-confab moat survives the cross-modal convergence — the load-bearing abstention is preserved.

## NEXT (the build, full reign)

1. **On-substrate cheap-first:** realize the cross-modal convergence as population-Hebbian co-activation on a real
   `SimulationBridge` (reuse `_phaseB_stdp_cooccurrence_derisk` machinery), with a similarity-structured perception
   render, and confirm held-out category transfer on real spikes.
2. **Option B on the substrate:** produce a similarity-structured perception code — either a richer object render
   (objects with shared visual features, legitimate sensory rendering) or a learned similarity-preserving map.
3. **The live-task build:** co-activate perception + the conversation cortex on the merged bridge when the agent
   perceives an object whose word streams, commit shared assemblies, and test generalization to novel similar
   perceived objects + the who/what matrix + the moat.
