# EMERGENCE-BAR step 3 (6-seed, positive — the master-directive close): the reservoir-generation ladder's category structure emerges from REAL PERCEPTION — objects SEEN through the Gabor/V1 front end — with no hand label, no designed feature pool, no symbolic token; destroying the visual shape (pixel-scramble) collapses it on every seed

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_perception_grounded_codes_derisk.py` (reuse-by-import: EMERGE-34's shape/Gabor/V1 machinery + the Rung-3 grammar, reservoir, and one-step-local-delta read-out; NO `sim/` edit, NO BPTT, NO deep credit).
**Verdict:** **A genuine positive (moderate strength) — the strongest, most-grounded of the three emergence closes.** The step-1 residual (the category came from a designed feature structure) is closed by grounding the category in PERCEPTION: each animal is rendered as an object SHAPE and SEEN through the real retina→V1 Gabor bank; the category emerges from VISUAL similarity, the ladder's Rung-3 generalization rides it, and a per-image PIXEL-SCRAMBLE (destroy the visual shape) collapses it on all 6 seeds. main 0.769 vs pixel-scramble 0.426 (margin 0.22–0.45 per seed). Not overclaimed as a clean 6/6 (the realistic soft signal + the tiny 9-animals/category vocab make the absolute accuracy moderate/per-seed-noisy).

## The mechanism (category from seeing, not designed structure)
Each animal is rendered as an object shape (`build_shape_set`, 2 visual categories, N_EX=9 exemplars each), SEEN through the project's real retina→V1 Gabor receptive-field bank (`encode_v1`), and its perception feature = the top-T=20 active V1 cells. Same-category shapes overlap in V1 features (EMERGE-34: within-cat ~0.25, cross-cat ~0.00 — the perception PRESERVES the visual similarity), so a FIXED random codon (F.12; step 1 showed a fixed codon, not a learned pooler, is the honest surfacer) turns them into a shared-category SDR. That SDR replaces the ladder's hand class-bit as the animal's category component; the reservoir + one-step-local-delta read-out are byte-identical. "The brain LEARNS what a category IS by looking, then reasons about it."

## Result — 6-seed (dev 42/43/44 + blind 100/101/102), `heldagent_cat_acc` (2-way floor ≈ 0.5 from `onehot`)
| Arm | heldagent | role |
|---|---|---|
| **main** — animals SEEN via Gabor/V1 → visual-similarity category code | **0.769** (per-seed 0.61–1.00) | the ladder rides the perceived category |
| **scramble** — per-image PIXEL SCRAMBLE (destroys within-category visual similarity) | **0.426** (collapses) | **load-bearing control: the VISUAL SHAPE is the cause** |
| onehot — content bit only, no category block | 0.472 | floor |
| untrained — frozen read-out | 0.000 | floor |

**main beats the pixel-scramble on all 6 seeds** (per-seed margin 0.22, 0.28, 0.28, 0.39, 0.44, 0.45). ⇒ the ladder's Rung-3 generalization rides a category the brain DISCOVERED by SEEING; scrambling the pixels (destroying the visual shape, so V1 sees no category structure) collapses it — isolating the VISUAL shape, not any symbolic or designed structure, as the cause. This is a **stronger signal than the corpus-distributional toy** (step 2: main ~0.5 on overlapping contexts) — real perceptual similarity beats tiny-vocab distributional statistics, as predicted.

## Honest scope
- **Genuine positive, moderate strength.** main > scramble on every seed (the mechanism is robustly load-bearing), but the absolute generalization is moderate (0.77) and per-seed-noisy (3/6 clear a strict `main ≥ 0.75` threshold; all 6 clear a `margin ≥ 0.2` threshold). The softness is the realistic signal (visual within-cat similarity ~0.25, not a hard partition) + the tiny 9-animals/category read-out. **Tested + honest:** scaling the CODON dimensionality (T_ACTIVE 20→40, N_COL 80→160, K 12→20) does NOT firm it (6-seed main 0.64 ≤ the default 0.77) — the default params are near-optimal and the bottleneck is the tiny animal VOCAB (fixed at 9/category by the Rung-3 grammar) + the soft realistic visual signal, NOT the codon. Firming it therefore needs a LARGER animal vocabulary (a grammar/vocab extension), not a bigger codon.
- **What is now emergent vs given.** The category LABEL, the feature pool, and the symbolic token are all GONE — the category comes from the pixels of a SEEN shape through the real V1 front end, and the pixel-scramble control proves the visual shape is the cause. The remaining given is the world's structure (same-category animals look similar) — which is legitimate host per BRAIN-BASED-ONLY (the environment renders the senses; the brain does the categorization). The MEETS token and ACTION-category structure remain fixed distinct codes (a follow-on).

## ⇒ significance
The reservoir-generation ladder's content-generalization now rides a category the brain DISCOVERS BY LOOKING — grounded in real Gabor/V1 perception, with a pixel-scramble control that isolates the visual shape as the cause, on all 6 seeds. Across the three emergence steps: (1) fixed codon of a designed feature partition [clean but designed], (2) corpus distributional co-occurrence [scale-limited on the toy], (3) **real perception [the genuine, master-directive-aligned close, positive at moderate strength].** The category structure the ladder depends on is no longer hand-installed; it emerges from experience — from seeing. NEXT: scale the perceptual vocabulary to firm the accuracy; the on-substrate spiking codon (EMERGE-35/38–41) so the surfacing is itself spiking; then self-organize the remaining fixed structure (MEETS/action).

## Files
`_emerge_reservoir_lm_perception_grounded_codes_derisk.py`; 6-seed raw `research/findings/raw/_perc/s{...}.json`; reuses EMERGE-34 (`build_shape_set`/`build_gabor_response_matrix`/`encode_v1`) + `sim.visual_cortex`; follows `2026-07-11-EMERGENT-category-codes-drive-the-generation-ladder-6seed.md` (step 1) + `-corpus-cooccurrence-scale-boundary.md` (step 2).
```
python -m research.runners._emerge_reservoir_lm_perception_grounded_codes_derisk --seeds 42
```
