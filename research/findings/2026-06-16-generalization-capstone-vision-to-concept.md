# Generalization capstone — STAGE 1 (vision → concept): Option B → A closed end-to-end on the spiking substrate — GO

**Date:** 2026-06-16
**Runner:** `research/runners/_genfrontier_capstone_vision_to_concept_derisk.py`
**Raw:** `research/findings/raw/_genfrontier_capstone_vision_to_concept.json`
**Verdict:** **GO** (3 seeds 42/43/44, GPU `SIM_BACKEND=cupy`, RTX 3090, ~199 s total). **NO `sim/` edit.**

## What this closes

The generalization mechanism was de-risked in four pieces (all 2026-06-16). The two that this capstone
integrates:

- **Option B** (`_genfrontier_optionB_visual_similarity_derisk`, GO): object SHAPES with shared visual
  features, encoded through the project's REAL Gabor/V1 front end (`sim.visual_cortex.build_v1_simple_weights`),
  produce a similarity-STRUCTURED perception code (within-cat cos 0.86 / between-cat 0.08, RSA-to-pixels 0.99).
- **Graded propagation** (`_genfrontier_graded_propagation_derisk`, GO): a perception→concept(NMDA) bridge where
  rate-Hebbian co-activation LEARNS the convergence and the NMDA concept assembly SPIKES (real
  `cp_firing_states`) category-correctly for a held-out cue.

The prior convergence de-risks drove a **synthetic** structured perception ensemble (a shared per-category core +
a per-concept unique tail — the same-category overlap was *manufactured by construction*, a controlled given).
**This capstone replaces that synthetic input with the REAL Option-B vision-derived perception code**, so the
convergence + the NMDA concept-spiking run on genuine perception. That closes Option B → A:
**perceive a NOVEL object through real vision → its concept neurons fire (spike) for the right category.**

## (1) The conversion: Gabor/V1 code → a structure-preserving perception drive

Each shape → render → the real Gabor/V1 front end → a **V1-complex code** (dim
`N_V1_COMPLEX = 8 orient × 16 × 16 pos = 2048`, non-negative real rates; Option B's `pool_v1_to_complex`). The
convergence bridge's **perception region is set to exactly those 2048 V1-complex cells** — one perception neuron
per V1-complex feature, a faithful feature/retinotopic map, **no relabeling, no injected category**.

The conversion of the real-valued code into the bridge's sparse index-addressed perception interface
(`perc_sets[j]` = a set of active perception indices) is **top-K**: the perception drive for a shape = the
indices of its **top-K most-active V1-complex features** (`vision_to_perception_sets`, K=60, strictly-positive
only — no zero-padding that would create a shared background common-mode). Same-category shapes excite the same
orientation/position columns → their top-K active sets overlap → same-category VISUAL overlap becomes
same-category PERCEPTION-ENSEMBLE overlap; different categories excite different columns → disjoint. The
conversion consults **only each image's own code**, never the labels, so it can only *preserve* the pixel-derived
structure, not inject it.

**The structure-preservation assert** (`active_set_overlap_margin`, within-vs-between cosine of the binary
active-set vectors), per seed:

| seed | Gabor V1-complex code margin (within / between) | top-60 active-SET margin (within / between) | preserved |
|---|---|---|---|
| 42 | +0.755 (0.901 / 0.146) | **+0.528 (0.528 / 0.000)** | yes |
| 43 | +0.771 (0.930 / 0.159) | **+0.553 (0.553 / 0.001)** | yes |
| 44 | +0.816 (0.956 / 0.140) | **+0.647 (0.647 / 0.000)** | yes |

Within-category active-set overlap 0.53–0.65; between-category ~0.000 every seed. The between-cat ~0 confirms
**no spurious common-mode** — the conversion is even cleaner between categories than the synthetic ensembles
(which shared a category core but had a non-zero floor). The structure survives the conversion with a wide
margin above the 0.05 assert threshold.

## (2) The result — held-out vision-derived concept-spike category accuracy

For a HELD-OUT shape (its concept block NEVER co-activated during training): render → Gabor/V1 → top-60
perception drive → run the bridge → does the NMDA concept assembly SPIKE in the correct semantic category?

| seed | VISION concept-spike cat-acc (chance 0.25) | concept margin | FLAT baseline | PERMUTED (derangement) margin | moat | concept spikes/cue |
|---|---|---|---|---|---|---|
| 42 | 0.75 | +0.066 | 0.25 | −0.022 | OK (ho 0.12 / novel 0.05) | 80 |
| 43 | 0.50 | +0.093 | 0.25 | −0.020 | OK (ho 0.23 / novel 0.15) | 171 |
| 44 | 1.00 | +0.179 | 0.00 | −0.024 | OK (ho 0.24 / novel 0.10) | 138 |
| **mean** | **0.75** | **+0.113** | **0.17** | **−0.022** | **INTACT (3/3)** | **130** |

Every seed strictly above chance (0.75 / 0.50 / 1.00) with a positive concept margin; mean 0.75 = **3× chance**.

**Anti-cheats all pass:**
- **Flat-distinct baseline at chance** (mean 0.17, same set sizes but disjoint/orthogonal, no visual structure) →
  the VISUAL structure is load-bearing, not the bridge wiring.
- **Category-derangement collapses** (margin −0.022 vs structured +0.113; co-activating each train shape's vision
  perception with a WRONG-category concept block lands the held-out cue in the wrong category) → the transfer is
  the LEARNED vision-category ↔ concept-category correspondence.
- **No-confab moat survives all 3 seeds** — a visually-novel NO-CATEGORY shape (rendered at an unseen
  orientation/position basis, distinct from the 4 trained category bases, then Gabor-encoded) drives a
  best-category familiarity well below the held-out concepts (held-out > novel × 1.5 every seed), so the system
  does not confidently confabulate a category for a never-seen visual basis.

## (3) Did the Gabor structure survive end-to-end?

Yes. The chain is intact at every hop: Gabor code margin (min +0.755) → top-60 active-set margin (min +0.528,
structure PRESERVED all seeds) → the converged concept assembly SPIKES (130 spikes/cue, real `cp_firing_states`,
not membrane potential) → category-correct concept spikes (0.75 mean, margin +0.113). The downstream NMDA
read-out region also spikes (244/cue), confirming the response propagates a synapse further (the readout's own
category read is noisier at 0.33 — the concept assembly's spike code is the clean, load-bearing signal, as in the
graded-propagation GO; the readout propagation is a bonus, not the gate).

## (4) `sim/` edits

**None.** Reuse-by-import only: `sim.visual_cortex` (Gabor/V1 front end) + Option B's shape
construction/encoding helpers + the graded-propagation bridge/training/NMDA-spike-read/anti-cheat helpers +
the convergence runner's `N_CAT/N_PER_CAT/F` constants. `git status --porcelain sim/` and `git diff --stat sim/`
are both empty.

## Honest scope / next

- 4 categories × 4 exemplars = 16 concepts; oriented-bar shapes (the canonical Gabor stimulus); K=60 top-K
  conversion. The mechanism is what is de-risked; richer shapes / more categories are a scaling follow-on.
- seed 43 at 0.50 is the noisiest (2/4 held-out correct) but still 2× chance with positive margin and intact
  moat; the population-code levers (more `n-concept-per`, more epochs, larger `top-K`) are available if a tighter
  per-seed floor is wanted.
- The readout-region's own category read (0.33) is below the concept assembly's — consistent with the
  graded-prop finding that the concept spike code is the clean signal. If a downstream stage needs the readout
  region itself to be category-clean, that is a localized read-out tuning, not a mechanism gap.

**This is capstone STAGE 1 (vision → spiking concept). STAGE 2** is the who/what + no-confab pipeline running on
the generalized concept (the live-task conversational read-out on a vision-perceived novel object).
