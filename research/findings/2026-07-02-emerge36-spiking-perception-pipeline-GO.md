# EMERGE-36 / toward-semantics — GO (6/6 seeds): the FULLY-SPIKING PERCEPTION→POOLER→INFERENCE pipeline (the capstone of the fully-spiking emergent-structure arc). Objects SEEN through the real Gabor/V1 front end → a spiking sparse-expansion codon (EMERGE-35, no numpy kWTA) → a held-out PERCEIVED object inherits its visual category's property — all SPIKING end-to-end. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge36_spiking_perception_pipeline_derisk.py`; CI guard `tests/test_emerge36_spiking_perception_pipeline.py` (2 tests). Composes EMERGE-34 (perception) + EMERGE-35 (spiking pooler); reuse-by-import (`_genfrontier_optionB` V1 + `_emerge14` + `_emerge12`); NO `sim/` edit; CPU numpy-backend; 6-seed.

## The claim (6/6 seeds)
The whole pipeline runs with NO numpy kWTA anywhere: object shapes → pixels → the real `sim.visual_cortex` Gabor/V1 responses → the top-active V1 cells (feature layer) → a spiking SPARSE-EXPANSION column codon (250 columns, each sampling 3 decorrelated V1 features, firing if ≥2 active via `coincidence_weighted_drive` — EMERGE-35's Marr-Albus codon) → a property taught on training objects' codons (the committed `sim/` three-term kernel) → **a held-out PERCEIVED object inherits its visual category's property (held-out inheritance 1.00 on EVERY seed)**. SEE an object, discover what a category is, reason about a novel one — spiking end-to-end.

## Anti-cheats (6-seed) — gate on the input-destruction perception control
- **PER-IMAGE PIXEL SCRAMBLE** (each object's pixels shuffled independently → within-category visual similarity destroyed) — the LOAD-BEARING perception control: collapses to **0.56 mean** vs the intact 1.00 (margin 0.44). Per-seed noisy (0.33–0.83, small-setup coarseness) so the GO keys on the multi-seed mean; the single-seed CI uses the deterministic lesion.
- **dAP-LESION** (coincidence off → no codon): deterministically collapses to **0.00**.
- 6-seed unanimous held-out 1.00.

## Significance — the fully-spiking emergent-structure pipeline is complete
This replaces EMERGE-34's NUMPY competitive pooler with EMERGE-35's spiking codon, making the ENTIRE perception-grounded emergence spiking end-to-end: pixels → real Gabor/V1 receptive fields → a spiking cerebellar-granule-codon pooler → on-bridge inference. The brain SEES an object, discovers its category from visual similarity (spiking), and reasons about a held-out perceived object — all on the spiking substrate, biology-grounded (Gabor/V1 + Marr-Albus codon), NO `sim/` edit. Combined with EMERGE-30..35, the emergent-structure-from-experience arc is spiking from perception to inference.

## Honest scope + next
- The Gabor/V1 encode is the rate-reference sensory front end (the retina/V1 receptive-field bank); the pooler codon + inheritance run on the spiking bridge — no numpy kWTA. The sparse-expansion codon is FIXED + decorrelated (the Marr codon), not competitively learned; the LEARNED self-organizing pooler (per the research gate) is the further refinement.
- Next: the competitive self-organizing pooler (LEARNED feat→col); couple perception-grounded emergence into the EMERGE-31 experiential console; multi-category perceptual taxonomy; a fully-spiking V1 front end.

## Artifacts
`research/runners/_emerge36_spiking_perception_pipeline_derisk.py`, `tests/test_emerge36_spiking_perception_pipeline.py`, `research/findings/raw/_emerge36_spiking_perception_pipeline.json`. Prior: `2026-07-02-emerge35-spiking-pooler-GO.md`, `2026-07-02-emerge34-perception-grounded-emergence-GO.md`.
