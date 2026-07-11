# EMERGENCE-BAR de-risk (6-seed, reproducible) — the reservoir-generation ladder's content-generalization needs NO hand category-LABEL: the category comes from feature co-occurrence STATISTICS, surfaced by a FIXED random Marr-Albus codon; a competitive learned pooler is NOT needed, and destroying the feature statistics collapses it

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_emergent_category_codes_derisk.py` (reuse-by-import: the Rung-3/Rung-4 grammar, reservoir, and one-step-local-delta read-out + the EMERGE-33 pooler / an F.12 fixed codon; NO `sim/` edit, NO BPTT, NO deep credit).
**Verdict:** **A step toward the emergence bar (honest, scoped) — the ladder's Rung-3 content-generalization rides a category structure DISCOVERED from feature statistics, NOT a hand class-label; but the feature→category correlation is still a designed "environment," so this is one rung short of perception/corpus grounding.** Two concurrent adversarial skeptics returned **SURVIVES-WITH-SCOPE-FIX** and drove three corrections folded in below. GO 6/6 on the load-bearing control (`nolearn` fixed codon vs `scramble`).

## What this shows (and the honest mechanism, after the adversarial verify sharpened it)
The rungs coded each animal as `[content one-hot] ⊕ [a class bit set from the hand label ANIMAL_CAT]`. This de-risk removes that hand class-bit: each animal is streamed as a **feature vector** (3 active features drawn from its category's feature pool), and its reservoir-input category component is a **sparse projection of those features** into columns. The reservoir + one-step-local-delta read-out are **byte-identical**; only the category component changes from HAND-GIVEN to DERIVED-FROM-FEATURES. Result — **6-seed, reproducible (PYTHONHASHSEED-independent after a determinism fix, below), heldagent_cat_acc, random-over-vocab floor 3/25 = 0.12:**

| Arm | heldagent | role |
|---|---|---|
| **nolearn** — a **FIXED RANDOM sparse projection** (Marr-Albus codon, F.12) of the category-structured features | **0.963** (per-seed 0.89–1.00) | **the honest mechanism** |
| main — a competitive HTM Spatial Pooler (800 epochs) of the same features | 0.731 | *the SP learning is NOT needed — WORSE than the fixed codon* |
| **scramble** — features drawn from a MIXED pool (no category co-occurrence structure) | **0.417** (collapses every seed) | **load-bearing control** |
| nopooler — random per-animal code (similarity-blind) | 0.574 | collapse (noisy control) |
| onehot — content bit only, no category block | 0.472 | collapse |
| untrained — frozen read-out | 0.000 | floor |

**`nolearn` 0.963 vs `scramble` 0.417 — margin 0.55, and per-seed `nolearn` (0.89–1.00) beats `scramble` (0.11–0.56) by ≥0.33 on all 6 seeds.** ⇒ the generalization rides the **category structure present in the feature co-occurrence statistics** (destroying it — `scramble` — collapses it), surfaced by a **fixed biological codon**; a similarity-blind code (`nopooler`) or no block (`onehot`) also collapses; and a trained read-out is required (`untrained` 0). **The category LABEL is not hand-given** (no PRED/PREY bit installed) and **no learning is needed** to surface it (the fixed codon beats the learned pooler — consistent with EMERGE-35 and catalog F.12 Marr-Albus).

## The adversarial verify — 3 corrections it forced (the discipline working)
Two concurrent skeptics (leakage/inductive + control-validity), both **SURVIVES-WITH-SCOPE-FIX**:
1. **"Self-organizing pooler discovers the category" was an OVERCLAIM.** Skeptic A prescribed a `POOL_EPOCHS=0` (fixed-random-projection) arm to isolate whether the competitive LEARNING is load-bearing. It is NOT — `nolearn` (fixed codon) 0.963 ≥ `main` (SP) 0.731. So the honest mechanism is a **fixed codon of the feature statistics**, not a learned pooler. The finding is reframed accordingly.
2. **A reproducibility bug I caught while running the `nolearn` arm:** the pooler seeded feature assignments with `hash(animal)`, which is **process-salted** (PYTHONHASHSEED) → non-deterministic across runs (it made `scramble` drift 0.34→0.53). Fixed to a deterministic per-animal index (`r3.WORD_IDX[a]`); now bit-identical across PYTHONHASHSEED 0 vs 777.
3. **`heldagent` is the clean metric; `reversal` is confounded** (skeptic B): the Rung-4 reversal sub-metric's `onehot` control (0.537) exceeds `main` on some seeds because the trajectory read carries word order on twins with a TRAINED agent (memorizable via the content bit). So the gate + headline are on **heldagent generalization** (genuinely held animals, order-washed cum feature, non-memorizable); reversal is reported only as secondary corroboration (`nolearn` 0.926 vs `scramble` 0.370).

## Honest scope (the residual, stated plainly — both skeptics' core point)
The category is **not given as a label**, but it IS structurally baked into the hand-assigned feature stream (PRED = features 0–3, PREY = features 4–7, disjoint pools). What EMERGES is the CATEGORY from those feature statistics (unsupervised; `scramble` proves it is load-bearing); the feature→category correlation itself is a **designed environment** (as sensory input is). So this **relocates** the hand-specification from a category *label* to a *feature correlation* + a fixed codon — genuinely one level more emergent (the label is gone), but **one rung short of features grounded in real perception or corpus co-occurrence.** An INDUCTIVE variant (`--inductive`, pooler never sees held animals) confirms the generalization is not transductive leakage (main 0.926). The MEETS token and the ACTION-category structure also remain fixed distinct codes.

## ⇒ significance + the honest NEXT step
The ladder's content-generalization does not require a hand category-LABEL or a learned pooler — a fixed Marr-Albus codon of the feature co-occurrence statistics suffices, and destroying those statistics collapses it. This de-risks the discovered-code → reservoir plumbing and pins the exact residual: **the feature→category correlation must itself come from experience.** The genuine emergence-bar close is therefore the immediate NEXT build — feed the ladder codes whose category structure comes from **real perception** (EMERGE-34 Gabor/V1 visual similarity; per-image scramble collapses, RSA r=0.99) OR **corpus co-occurrence** (EMERGE-30/62: category-mates that recur in the same contexts develop shared codes — the unified "one stream the generator reads" answer), then re-run Rung-3 generalization. That removes the last designed structure.

## Files
`_emerge_reservoir_lm_emergent_category_codes_derisk.py` (`--inductive` flag; arms main/nolearn/scramble/nopooler/onehot/untrained); 6-seed raw `research/findings/raw/_emcodes/s{...}.json`; builds on `2026-07-11-RUNG3-*.md`, `2026-07-11-RUNG4-*.md`, the EMERGE-33 pooler, and catalog F.12 (Marr-Albus codon / random-feature reservoir).
```
python -m research.runners._emerge_reservoir_lm_emergent_category_codes_derisk --seeds 42
```
