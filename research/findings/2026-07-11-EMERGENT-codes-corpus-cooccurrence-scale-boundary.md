# EMERGENCE-BAR step 2 (honest boundary) — deriving the ladder's category from OVERLAPPING distributional co-occurrence hits a SCALE limit on the toy: a fixed codon AND a learned pooler both fail to extract the category for held-out generalization when the category is not a hard partition; only the disjoint case (= step 1) works. The corpus close needs SCALE or perception-grounding

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_corpus_cooccurrence_codes_derisk.py` (reuse-by-import; NO `sim/` edit, NO BPTT). Follows `2026-07-11-EMERGENT-category-codes-drive-the-generation-ladder-6seed.md` (step 1).
**Verdict:** **Honest boundary (informative) — the step-1 residual (the category came from a DISJOINT hand-partitioned feature pool) is NOT cheaply closed by moving to OVERLAPPING distributional co-occurrence at this toy scale.** The category signal from overlapping distributions over a 9-animals-per-category / 10-context vocabulary is too weak for held-out generalization. The genuine close needs SCALE (a larger corpus/vocabulary — Redington-Chater-Finch distributional induction is a large-corpus phenomenon) or a stronger similarity source (real perception, EMERGE-34).

## What was tested
Step 1 removed the hand category-LABEL but the feature→category correlation used DISJOINT pools (PRED features 0–3, PREY 4–7). Step 2 replaces those with OVERLAPPING distributional contexts: each animal's sparse distributional signature draws `N_ACTIVE=3` context tokens from its category's distribution, where the two categories' context distributions share mass (`own_mass < 1.0` → overlap, not a hard partition) — so the category must emerge from co-occurrence STATISTICS. The signature is surfaced by a fixed random codon (F.12) OR a competitive HTM Spatial Pooler (the `learned` arm) into the reservoir input; the reservoir + one-step-local-delta read-out are byte-identical.

## Result (seed-swept; `heldagent_cat_acc`, 2-category floor ≈ 0.5 from the `onehot` arm)
| `own_mass` (↓ = more overlap) | main (fixed codon) | learned (competitive SP) | scramble (uniform contexts) |
|---|---|---|---|
| 1.0 (DISJOINT = step 1) | **0.778** | — | 0.333 |
| 0.95 (light overlap) | 0.63 | 0.76 | 0.50 |
| 0.9 | 0.63 | 0.67 | 0.50 |
| 0.85 | 0.50 | 0.61 | 0.33 |
| 0.75 | 0.556 | 0.389 | 0.333 |

- **Disjoint (own_mass 1.0): clean** — main 0.778 vs scramble 0.333 (this is the step-1 result re-derived via a co-occurrence framing).
- **Any real overlap: the signal collapses** — main drops to ~0.5–0.63 (barely above the `onehot`/2-way floor ~0.5), and the margin over `scramble` falls to ~0.13–0.26. The competitive SP (`learned`) is only marginally better and not robust (0.39–0.76 across settings; `learned_scramble` = 0.556, the SP overfits the uniform-context noise rather than collapsing). Neither a fixed codon nor a learned pooler extracts a reliable held-out category from overlapping distributions at this scale.

## Why (the honest diagnosis) + the next lever
The toy has only 9 animals/category and 10 contexts with 3 sparse active — so overlapping distributions give category-mates only ~1–2 shared contexts, a signal-to-noise too low for a held-out animal to land in its category's code region. This is a SCALE limit, not a mechanism failure: distributional category induction (Redington-Chater-Finch 1998) is empirically a LARGE-corpus phenomenon. The refined NEXT levers to close the step-1 residual (the feature→category correlation must come from experience, not a designed pool):
1. **SCALE the corpus** — many more animals/contexts/co-occurrences so overlapping distributions become separable (the honest amount of data distributional induction needs).
2. **PERCEPTION-grounding (EMERGE-34, the stronger route)** — objects SEEN through the real Gabor/V1 front end give a category signal from VISUAL similarity that is far stronger than tiny-vocab distributional contexts (EMERGE-34: within-cat 0.86 vs between-cat 0.08, RSA pixel-provenance r=0.99). Feeding EMERGE-34's visual-similarity codes to the ladder is the likely-clean close.

## ⇒ significance
An honest negative that MAPS the emergence path precisely: the step-1 result (a fixed codon surfaces a category from a hard-partition feature structure) does NOT extend cheaply to soft/overlapping distributional structure at toy scale — the corpus-co-occurrence close is real but needs SCALE, and the more tractable close is PERCEPTION-grounding (EMERGE-34), where the category signal is strong. The runner is a reusable `own_mass`-parametrized tool for the scaled version.

## Files
`_emerge_reservoir_lm_corpus_cooccurrence_codes_derisk.py` (`--own-mass`; arms main/learned/scramble/learned_scramble/onehot/untrained).
```
python -m research.runners._emerge_reservoir_lm_corpus_cooccurrence_codes_derisk --seeds 42 --own-mass 1.0   # clean (disjoint)
python -m research.runners._emerge_reservoir_lm_corpus_cooccurrence_codes_derisk --seeds 42 --own-mass 0.85  # boundary (overlap)
```
