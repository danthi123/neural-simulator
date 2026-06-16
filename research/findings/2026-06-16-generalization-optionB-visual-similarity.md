# Option B de-risk — shared visual features through the Gabor/V1 front end DO produce similarity-structured perception codes (GO)

**Date:** 2026-06-16
**Runner:** `research/runners/_genfrontier_optionB_visual_similarity_derisk.py`
**Raw:** `research/findings/raw/_genfrontier_optionB_visual_similarity.json` (default 4×4),
`research/findings/raw/_genfrontier_optionB_visual_similarity_6x6.json` (harder 6×6 robustness)
**Verdict:** **GO** — 3 seeds (42/43/44), CPU numpy, all anti-cheats clean, holds at a harder 6-category setting.

---

## The question

The cross-modal cheap-first (`2026-06-16-generalization-crossmodal-unify-cheap-first.md`) went GO but with a
prerequisite: cross-modal Hebbian convergence transfers the conversation cortex's category-generalization to
perception **only when the perception input itself carries similarity structure** (flat-distinct nav perception does
not transfer; "Option B is the PREREQUISITE"). This probe de-risks **Option B's load-bearing claim**: does
**legitimate sensory rendering** — object SHAPES that share visual features within a category, passed through the
project's existing Gabor/V1 front end — produce perception codes where same-category objects are SIMILAR and
different-category objects are DISSIMILAR (the way IT does for visually-similar objects; Op de Beeck/Kriegeskorte
RSA, Kiani 2007)? The similarity must come from PIXELS (shared visual features), not from an injected semantic label.

## (1) How the shapes were built + which front end

**Shapes (similarity in pixels only).** Each CATEGORY is a distinct VISUAL basis: an oriented **bar** (a line
segment — the canonical Gabor-activating stimulus, Hubel-Wiesel) at a base orientation `theta_c` spread across
`[0, pi)` and a base centre `(cx_c, cy_c)` spread on a ring around the 32×32 image. Each EXEMPLAR within a category
= the category bar + small per-exemplar VISUAL jitter (angle wobble ±7°, centre shift ±3% of the image, length ±8%,
thickness ±10%, plus low-amplitude per-pixel noise). The bar is rendered into a `(2, 32, 32)` ON/OFF image (ON =
soft-edged line intensity; OFF = the line's gradient-magnitude edge at ~0.3 amplitude — matching the gridworld
render's ON/OFF convention). **Nothing about the category is injected as a code** — the category identity lives
*only* in the rendered pixels (which orientation columns + retinotopic positions the bar drives). Construction is
documented per-image in the JSON (`sample_meta`).

**Front end — the project's REAL Gabor RF bank, reuse-by-import (no `sim/` edit).** The public API
(`render_gridworld_to_image`) only renders the gridworld scene, so — as the task permits — I drove the Gabor RF bank
directly on the shape images:
- `sim.visual_cortex.build_v1_simple_weights(...)` → the exact retina→V1-simple sparse Gabor weights the deployed
  nav stack installs (`apply_v1_gabor_weights`, `g11_bg_runner` ~line 2546). Densified into a `(8192, 2048)` matrix;
  V1-simple response = `relu(W @ retina_drive)` (rectified — V1 rates are non-negative).
- V1-simple → V1-complex pooling = the runner's fixed phase/frequency pooling (sum over the 4 frequencies within
  each orientation×position; complex index `orient*(n_pos²)+pos_y*n_pos+pos_x`, `g11_bg_runner` ~line 2561) → the
  `(2048,)` "IT-like" pooled code. **These are exactly the V1→complex layers the deployed visual hierarchy uses.**

The headline perception code is the IT-like pooled code; the raw V1-simple code is reported too (≈ identical margin).

## (2) Result — within vs between, vs flat baseline (3 seeds)

| perception code | within-cat cos | between-cat cos | **margin** | flat-distinct baseline margin |
|---|---|---|---|---|
| **IT-like pooled** (default 4 cat × 4 ex) | 0.86 | 0.08 | **+0.781** (min 0.755) | **0.000** |
| V1-simple (default) | — | — | +0.782 | — |
| **IT-like pooled** (harder 6 cat × 6 ex) | — | — | **+0.734** (min 0.705) | **0.000** |

- The Gabor/V1-encoded perception codes show **within-category cosine exceeding between-category by +0.78** (default)
  / **+0.73** (harder 6-category), **all 3 seeds GO** (margin ≥ 0.15 gate), ≫ the flat-distinct baseline ≈ 0.
- **Flat-distinct baseline = 0.000** exactly (the current nav `orthogonal_drive_pattern` regime — each object its own
  non-overlapping band → between-code cosine = 0). This is the discriminating gap: shared visual features buy the
  similarity structure that orthogonal codes structurally cannot have.

**Verdict: GO.** Legitimate sensory rendering (shared visual features → Gabor/V1) produces similarity-structured
perception codes, the IT/RSA signature. Option B's load-bearing claim holds.

**Honest scope on the magnitude.** The *absolute* margin (~0.78) is high because oriented bars at well-separated
orientations are near-orthogonal across categories yet near-identical within — real IT same/different-category
margins are more modest. The load-bearing result is **qualitative + relative**: the front end produces
similarity-structure (margin ≫ 0.15) where the flat regime produces none (≈ 0), and it tracks the pixels (below).
The harder 6-category arm (closer orientations, more exemplars, more noise) still GO at +0.73 with *higher* cluster
purity (0.83) — so this is not a 4-category toy artifact.

## (3) Anti-cheats — does the structure follow VISUAL FEATURES, not labels? (all clean)

1. **RSA pixel-provenance (LABEL-FREE, the strongest form): PASS.** Correlate the off-diagonal of the raw-PIXEL
   cosine matrix with the off-diagonal of the perception-code cosine matrix — never touching labels. **r = 0.99**
   (default) / **0.98** (6×6): code similarity tracks pixel similarity almost perfectly → the code's structure comes
   from the visual features, full stop.
2. **Cluster purity (structure recovers the pixel groups): PASS.** Spherical k-means on the codes' own cosine
   geometry recovers the pixel-defined groups at **0.75** (default) / **0.83** (6×6) — the high-margin grouping IS
   the pixel grouping, found without labels.
3. **Random-partition label-independence null: PASS.** The within/between margin is a function of the *grouping*. If
   the structure followed an injected LABEL, re-grouping objects by a random partition (same group sizes, ignoring
   pixels) would keep the margin; since it lives in the PIXELS, mixing same-pixel objects into different groups drops
   it. The true margin sits **9.3 SDs** (default) / **21.5 SDs** (6×6) above the 500-draw null mean, gap **+0.76**.
   *(Methodology note: a single within-set random shuffle leaks a small spurious margin — ~0.22 here — because, with
   few items, a random permutation co-locates a few same-pixel objects by chance, the documented small-N artifact
   from the predecessor finding; averaging over 500 draws quantifies + controls it. A category-block "derangement"
   is a no-op for this margin — it only renames blocks, leaving the partition unchanged — so the random-partition
   null is the correct control, not a derangement.)*
4. **Flat-distinct baseline = 0.000** (the current nav regime) — the discriminating gap.
5. **No pre-seeded semantics:** the category basis is a RENDERED pixel pattern (an oriented bar at `theta_c`, `p_c`),
   never a hand-set code vector added to the perception code.

→ The similarity structure **follows the visual features, not the labels** (RSA r=0.99; clusters recover the pixel
groups; the margin collapses under random re-grouping). Option B does not smuggle in the category via a label.

## (4) `sim/` edits

**NONE.** Reuse-by-import of `sim.visual_cortex.build_v1_simple_weights` (the real Gabor RF bank) +
`sim.text_embeddings.orthogonal_drive_pattern` (the flat baseline) only. `git status` shows no `sim/` change.

## Decisive conclusion + localized next step

- **Option B is achievable from legitimate sensory rendering**: shared visual features through the *existing*
  Gabor/V1 front end give similarity-structured perception codes (margin +0.78 ≫ flat 0), and the structure provably
  tracks the pixels (RSA r=0.99). **No learned similarity-preserving projection is required for this** — the raw
  visual hierarchy already supplies it. This unblocks the cross-modal A+B path: a similarity-structured perception
  front end (B) + cross-modal Hebbian convergence (A, already GO) → perception inherits the conversation cortex's
  category-generalization.
- **NEXT (the build):** (1) realize this on the substrate — render category-structured object shapes, drive them
  through the on-bridge `retina → cortex_v1_simple (Gabor) → v1_complex → v2 → it` hierarchy, and confirm the V2/IT
  *spiking* codes carry the same within>between margin (the rate-pooled IT code here is the numpy ceiling; the
  on-bridge check is whether IT firing preserves it). (2) Co-activate that IT perception code + the conversation
  cortex(word) on the merged bridge (the A+B live-task build) and test held-out category transfer + the who/what
  matrix + the no-confab moat on novel similar *perceived* objects.
- **HONEST residual:** the absolute margin overstates real IT (the bar basis is deliberately orientation-separable);
  the on-substrate spiking re-test at realistic object richness is the load-bearing confirmation, and the
  generalization payoff is only demonstrated once A+B run together on held-out perceived objects (this probe
  established the *prerequisite*, not the end-to-end generalization).
