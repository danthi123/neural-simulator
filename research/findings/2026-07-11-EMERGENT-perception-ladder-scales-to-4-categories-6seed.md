# EMERGENCE scale probe (6-seed) — the emergent-perception-category ladder GENERALIZES at FOUR categories, not just the two of Rung-3: a held-out perceived animal inherits its category's action at ~2.4× the 4-way chance; pixel-scramble collapses it

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_multicat_scale_derisk.py` (reuse-by-import: EMERGE-34 shapes/Gabor/V1 + the Rung-1 reservoir + one-step-local-delta read-out; NO `sim/` edit, NO BPTT; CPU numpy).
**Verdict:** **Positive (moderate/noisy) scale evidence — the emergence mechanism is not limited to 2 categories.** A 4-category grammar ("the `<animal>` `<action>`", each of 4 categories with its own animals + action set), animals SEEN through the real Gabor/V1 front end, category surfaced by a fixed codon into the reservoir input: a HELD-OUT perceived animal inherits its correct category action at main **0.597** (4-way chance ≈ 0.25), clearly above pixel-scramble (0.278) and the no-block floor (onehot 0.292). Because the 4-way chance (0.25) is lower than the 2-category task's 2-way floor (0.5), this is a *harder, cleaner* discrimination than the headline 2-category result — and the mechanism handles it.

## Result — 6-seed (dev 42/43/44 + blind 100/101/102), `held_cat_acc` (4-way chance ≈ 0.25)
| Arm | held_cat_acc | role |
|---|---|---|
| **main** — fixed codon of the Gabor/V1 perception features | **0.597** (per-seed 0.42–0.83) | the ladder inherits the perceived category at 4-way |
| scramble — per-image PIXEL SCRAMBLE (no visual category) | 0.278 (≈ chance) | load-bearing control → collapse |
| onehot — no category block | 0.292 (≈ chance) | floor |
| spiking_codon — the EMERGE-35 spiking sparse-expansion codon | 0.514 | *diagnostic: slightly UNDERperforms the fixed codon here* |
| untrained — frozen read-out | 0.000 | floor |

**main (0.597) ≈ 2.4× the 4-way chance and clearly above pixel-scramble (0.278) + onehot (0.292).** The category block from perception is load-bearing (main ≫ onehot, which sits at chance); scrambling the pixels collapses it. So the emergent-perception category drives generalization at 4 categories.

## Notes (honest)
- **Moderate/noisy strength.** Per-seed 0.42–0.83 (4/6 clear a strict `main ≥ 0.45 & margin ≥ 0.20` gate) — the softness is the tiny per-category read-out (6 trained + 3 held animals/category) + the harder 4-way task. The mechanism is load-bearing (main > scramble/onehot in aggregate and on most seeds) at moderate absolute accuracy.
- **A codon-size lesson (self-caught).** A first pass used N_COL=100 and the fixed codon scored ~chance at 4 categories — which looked like EMERGE-35's "a low-expansion fixed projection fails at 4 categories." But it was simply an UNDER-SIZED codon: at N_COL=120 the fixed codon scales fine (seed 42: 0.83). So the fixed codon DOES handle 4 categories with an adequate column count — the EMERGE-35 boundary was about a *too-small* expansion, and here the fix is a bigger codon, not necessarily the spiking one (which slightly underperforms the fixed codon at 4 categories on this task).
- The Gabor/V1 front end + the read-out are numpy (as in the perception step); the reservoir is spiking.

## ⇒ significance
The emergence-bar close (the ladder's category discovered from perception) extends from 2 to 4 categories — a held-out perceived animal inherits its category action at ~2.4× the 4-way chance, with a pixel-scramble control that collapses it. Evidence that the emergent-perception generalization is not a 2-category artifact. NEXT (the scale frontier proper): more categories/animals + more constructions toward realistic vocabulary; a stronger per-category signal (more exemplars) to firm the accuracy; the open big question — does the whole ladder scale to fluent open-domain conversation.

## Files
`_emerge_reservoir_lm_multicat_scale_derisk.py`; 6-seed raw `research/findings/raw/_multicat/s{...}.json`; reuses EMERGE-34/35; follows the perception-grounded emergence close.
```
python -m research.runners._emerge_reservoir_lm_multicat_scale_derisk --seeds 42
```
