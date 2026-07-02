# EMERGE-34 / toward-semantics — GO (6/6 seeds): PERCEPTION-GROUNDED EMERGENCE (the deepest master-directive step). The brain forms categories from REAL SENSORY EXPERIENCE, not symbolic tokens: objects SEEN through the project's real Gabor/V1 visual front end, categories DISCOVERED by a competitive pooler, and a property taught on some perceived objects INHERITED by a HELD-OUT PERCEIVED object — on the spiking bridge. NO `sim/` edit.

**2026-07-02 (autonomous; the perception-grounded next frontier).** Runner `research/runners/_emerge34_perception_grounded_emergence_derisk.py`; CI guard `tests/test_emerge34_perception_grounded_emergence.py` (3 tests). Reuse-by-import (`_genfrontier_optionB` real Gabor/V1 front end + `_emerge14` + `_emerge12`); NO `sim/` edit; CPU numpy-backend; 6-seed.

## Why it is the deepest step
Every prior emergence result (EMERGE-30/32/33) fed the brain SYMBOLIC feature/context tokens (hand-chosen). EMERGE-34's input is PERCEPTION: object shapes rendered to pixels and encoded through the real retina→V1 Gabor receptive-field bank (`sim.visual_cortex.build_v1_simple_weights`). Same-category objects (similar shapes) overlap in V1 features (within-category ~0.25, cross-category ~0.00 — the perception PRESERVES the visual similarity); the pooler self-organizes those into a shared column block; the on-bridge inheritance rides it. The brain LEARNS what a category IS by LOOKING, then reasons about it.

## The claim (6/6 seeds)
Objects (2 visual categories × 12 exemplars) are seen through the Gabor/V1 front end → top-T V1 features → a competitive pooler forms each object's column code; a property is taught on the TRAINING objects' codes (the committed `sim/` three-term kernel on the spiking bridge); **3 objects per category are HELD OUT** (a finer accuracy metric than 1/category); then:
- **Held-out perceived-object inheritance 0.97** (5/6 seeds at 1.00, one at 0.83): held-out objects (property never taught) inherit their visual category's property — learned by looking, never told.
- **Moat 1.00:** a code disjoint from every block abstains.

## Anti-cheats (6-seed) — the LOAD-BEARING control is the per-image scramble
- **PER-IMAGE PIXEL SCRAMBLE** (each object's pixels shuffled independently → within-category visual similarity destroyed) — the LOAD-BEARING control: held-out inheritance collapses to **0.53 mean (chance)** vs the intact 0.97 (a clean 0.44 margin, well past the 0.30 gate), isolating the VISUAL shape as the cause of the category (a consistent single-permutation does NOT collapse — the control targets within-category similarity, not absolute pixel identity). Per-seed it is noisy (0.33–0.83), so the GO keys on the multi-seed mean; the single-seed CI uses the deterministic dAP-lesion instead.
- **dAP-LESION** (bridge coincidence off): deterministically collapses to **0.00** — the clean mechanism-ablation control.
- **RANDOM-CODES** (no pooler → SEED-DEPENDENT random codes): 0.47 mean, reported but NOT gated on. (Methodology: a FIXED, seed-INDEPENDENT random-code control is UNRELIABLE — over a small 80-column space a fixed held-out code can COINCIDENTALLY inherit [the first run read 1.00]; the GO keys on the reliable input-destruction scramble + the deterministic lesion, per `2026-07-02-anti-cheat-control-validity-methodology.md`.)
- **MOAT 1.00.** 6-seed.

## Significance
This connects REAL PERCEPTION to the emergent semantics arc: the brain forms categories from what it SEES (real Gabor/V1 receptive fields), discovers the category structure unsupervised, and infers a property about a novel perceived object — the master-directive "learn from experience" direction, realized end-to-end from pixels to inference, on the spiking bridge, transformer-free, NO `sim/` edit. It composes the project's validated visual front end (the genfrontier Option-B shape/V1 machinery, RSA pixel-provenance r=0.99) with the EMERGE-33 pooler + on-bridge inheritance.

## Honest scope + next
- The visual front end (Gabor/V1) + the pooler are the perception + representation steps (a rate reference for the fully-spiking versions); the INHERITANCE runs on the real spiking bridge. The shapes are simple oriented bars (2 visual categories).
- Next: richer objects + more categories; a fully-spiking V1 + lateral-inhibition pooler; couple perception-grounded emergence into the experiential console (see an object → name/learn it → answer inference questions); multi-level perceptual taxonomy.

## Artifacts
`research/runners/_emerge34_perception_grounded_emergence_derisk.py`, `tests/test_emerge34_perception_grounded_emergence.py`, `research/findings/raw/_emerge34_perception_grounded_emergence.json`. Prior: `2026-07-02-emerge33-spatial-pooler-emergence-GO.md`, `2026-07-02-emerge30-emergent-superordinate-GO.md`, the genfrontier Option-B visual-similarity findings.
