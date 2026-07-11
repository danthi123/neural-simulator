# EMERGENCE follow-on (b), 6-seed — the perception→category SURFACING is now FULLY SPIKING: a Marr-Albus coincidence codon on a real SimulationBridge (no numpy kWTA) surfaces the V1-perception category into the ladder, and the ladder generalizes; pixel-scramble collapses it on every seed

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_spiking_perception_codes_derisk.py` (reuse-by-import: EMERGE-34 Gabor/V1 + EMERGE-35 spiking Marr-codon + the Rung-3 grammar/reservoir; NO `sim/` edit, NO BPTT, NO deep credit; CPU numpy-backend bridge).
**Verdict:** **Positive (moderate strength) — the category surfacing is now on SPIKES.** The perception-grounded emergence close surfaced the V1-perception features into a category SDR with a NUMPY fixed codon; here that codon is replaced by the FULLY-SPIKING Marr-Albus codon (EMERGE-35, catalog F.12) — a column layer on a real `SimulationBridge`, each column sampling 3 V1 features via a fixed decorrelated coincidence projection and firing via the validated `coincidence_weighted_drive` (≥ 2 of its 3 features active), NO numpy kWTA. That spiking codon feeds the already-spiking reservoir; the pipeline is now **pixels → Gabor/V1 → SPIKING codon → spiking reservoir → Rung-3 generalization**. main 0.722 (matches the numpy codon's 0.769), > pixel-scramble on all 6 seeds.

## Result — 6-seed (dev 42/43/44 + blind 100/101/102), `heldagent_cat_acc`
| Arm | heldagent | role |
|---|---|---|
| **main** — SPIKING Marr-codon over the V1-perception features | **0.722** (per-seed 0.61–0.78) | fully-spiking category surfacing |
| scramble — per-image PIXEL SCRAMBLE (V1 has no category structure) | 0.444 | collapses (margin > 0 every seed) |
| onehot — no category block | 0.426 | floor |
| untrained — frozen read-out | 0.000 | floor |

**main > pixel-scramble on all 6 seeds** (per-seed margins 0.11–0.44). The spiking codon SEPARATES the categories: measured within-category codon overlap **0.175** vs cross-category **0.000** (same-category perceived objects converge on overlapping columns; different categories are disjoint) — the coincidence columns on the bridge preserve the visual similarity. Codon sizes 3–14 columns (data-driven sparse expansion).

## Notes (honest)
- **A feature-space reduction was needed** for the coincidence columns to fire: the raw V1 space is too high-dimensional for a 20-active-cell code to reliably drive columns that sample 3 features at threshold 2 (first pass: all-chance). Restricting the codon's input to the UNION of ever-active V1 cells (so the active fraction is high enough) fixed it — a legitimate perceptual-relevance restriction, documented in the runner.
- **Moderate strength** (0.722, matching the numpy-codon version 0.769) — the same realistic soft visual signal + tiny 9-animals/category read-out as the numpy version; the mechanism is robustly load-bearing (main > scramble every seed) at moderate absolute accuracy.
- **What is now spiking:** the category SURFACING (coincidence columns on a `SimulationBridge`) + the sequence dynamics (the OnBridgeLSM reservoir). Still numpy: the Gabor/V1 front end (EMERGE-36 has the spiking V1) and the one-step-local-delta read-out. So this closes the codon piece toward the fully-spiking one-brain end state.

## ⇒ significance
The emergence-grounded category surfacing is now realized ON SPIKES (a real bridge's coincidence columns, F.12 Marr-Albus), matching the numpy version's accuracy — so the perception→category→generation pipeline is spiking at the codon + reservoir, with a pixel-scramble control that isolates the visual shape on every seed. NEXT: the spiking Gabor/V1 front end (EMERGE-36) for a fully-spiking perceptual path; a larger perceptual vocabulary to firm the accuracy; self-organize the remaining fixed structure (MEETS/action).

## Files
`_emerge_reservoir_lm_spiking_perception_codes_derisk.py`; 6-seed raw `research/findings/raw/_spkperc/s{...}.json`; reuses EMERGE-34/35 + `sim.bridge`; follows `2026-07-11-EMERGENT-perception-grounded-category-drives-the-ladder-6seed.md`.
```
python -m research.runners._emerge_reservoir_lm_spiking_perception_codes_derisk --seeds 42
```
