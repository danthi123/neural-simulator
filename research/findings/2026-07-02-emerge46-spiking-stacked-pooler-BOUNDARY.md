# EMERGE-46 / toward-semantics — BOUNDARY (3/3 seeds, honest): the FULLY-SPIKING stacked pooler. BOTH pooler layers' LEARNING (L1 features→sub-category codons + L2 L1-codons→superordinate codons) is now genuinely on-substrate — the permanences live in `cp_connections.data` and are moved by the committed `sim/` kernels, NO numpy pooler. BUT the strict held-out-sub-category inheritance GO does NOT hold on-substrate at this scale: the on-substrate L2 competitive pooler's held-out within-super L2 overlap is ~0.01 vs the numpy reference's ~0.12 on identical inputs (a ~12× deficit in the exact routing quantity). NO NEW `sim/` edit (only the already-committed winner-inactive kernel from EMERGE-40).

**2026-07-02 (autonomous).** Runner `research/runners/_emerge46_spiking_stacked_pooler_derisk.py`; CI guard `tests/test_emerge46_spiking_stacked_pooler.py` (3 tests, pinning the on-substrate mechanism facts — NOT the inheritance GO). Reuse-by-import (`_emerge14` committed kernel + `_emerge12` prime; the EMERGE-40 `OnSubstratePooler` mechanism generalized to arbitrary `(n_in, n_col)`); CPU numpy-backend; 3-seed (42/43/44). Prior: `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md` (the numpy-pooler version that GO'd), `2026-07-02-emerge40-spiking-htm-sp-kernel-GO.md` (the single-layer fully-spiking pooler that GO'd).

## What IS fully-spiking now (the genuine deliverable)
EMERGE-44 discovered the same 2-level taxonomy but used a NUMPY `_competitive_pool` (a rate-reference) for BOTH pooler layers' LEARNING. EMERGE-46 replaces BOTH layers' learning with the EMERGE-40 on-substrate mechanism (`OnSubstratePooler`):
- The feat→col coincidence **permanences LIVE in the real `SimulationBridge`'s `cp_connections.data`** (a dense plastic coincidence projection, small random init) — for BOTH L1 (`n_in=NF=21`, `n_col=200`) and L2 (`n_in=NCOL1=200`, `n_col=120`).
- **LEARNING is the two committed `sim/` fused kernels**: potentiation via `fused_htm_permanence_update` (ld=0) applied through `apply_kernel_update`, + winner-inactive depression via the committed `fused_htm_winner_inactive_depression` — over `cp_connections.data`, + homeostatic boosting. NO numpy pooler weights anywhere in the learning.
- The inheritance (L2-codon → superordinate property) runs on a THIRD spiking bridge via the committed three-term kernel + the coincidence-plateau read (== EMERGE-44).
- Winner **SELECTION** is a host top-k over the on-substrate drive (EMERGE-41 de-risked the spiking FS-WTA selection separately; the NEW thing here is that the LEARNING is fully-`sim/`-kernel for BOTH layers). Flagged honestly.

The 3 CI tests pass: both pooler layers demonstrably learn on the bridge (permanences in `cp_connections`, kernels applied, L2 codons non-empty read from the on-substrate drive); L2 discovers a POSITIVE superordinate grouping (within-super minus cross-super L2 overlap **+0.08 mean**); the depended-on kernel is the already-committed one.

## The BOUNDARY (3/3 seeds) — the strict held-out-sub-category inheritance GO does NOT hold on-substrate
On the EMERGE-44 task (6 sub-categories → 2 superordinates; hold out ENTIRE sub-categories {2,5} from super-property teaching; the held-out sub-category can inherit ONLY via the L2-discovered grouping), 3-seed (42/43/44), NCOL2=120 (proper columns):

| arm | super-acc (mean) | per-seed |
|---|---|---|
| **stacked (on-substrate)** | **0.03** | 0.00 / 0.00 / 0.08 |
| permuted-co-occurrence | 0.08 | 0.08 / 0.00 / 0.17 |
| L1→L2 lesion (fixed-random) | 0.53 | 0.50 / 0.58 / 0.50 |
| dAP-lesion | 0.00 | 0.00 / 0.00 / 0.00 |
| L2-grouping (within−cross) | +0.08 | +0.08 / +0.09 / +0.08 |

Stacked super-acc (0.03) is **below chance (0.50)** and does not beat permuted — the held-out sub-category's L2 code doesn't route the property. Gate misses: super-acc 0.03 < 0.80; L2-grouping +0.08 < 0.15; not ≥ permuted+0.25; not ≥ dAP-lesion+0.30. **Honest BOUNDARY.**

## The ISOLATED residual (the surpass-round, per CLAUDE.md — do not accept a boundary without pinning + quantifying it)
Fed IDENTICAL good (numpy) L1 codons at NCOL2=120, the two L2 poolers diverge sharply on the EXACT quantity that routes the held-out inheritance — the held-out sub-category's L2-codon overlap with the trained same-super members' L2 codons:

| L2 pooler (on identical numpy-L1 codons) | held-out within-super overlap | cross-super | diff |
|---|---|---|---|
| **numpy `_competitive_pool` (EMERGE-44)** | **0.119** | 0.000 | **+0.119 → routes** |
| on-substrate (EMERGE-46, with selectivity) | **0.010** | 0.002 | +0.008 → fails |
| on-substrate (selectivity OFF) | 0.071 | 0.056 | +0.015 → within rises 7× but cross rises too → no discrimination |

**The precise residual: the on-substrate competitive pooler produces ~12× LESS held-out same-super overlap than the numpy reference on identical inputs.** The mechanisms are mathematically equivalent per step (potentiation active→winner; depression winner-inactive; homeostatic boosting; a top-k connected-drive read) — the divergence is in the LEARNED generalization to HELD-OUT sub-categories, which is sensitive to the SGD-like trajectory (train-order RNG + float32 permanence accumulation) and the on-substrate pooler consistently lands in a code regime that tightly tunes to seen members but does NOT extend the shared columns to the held-out sub-category.

## Levers swept (none is a cheap fix)
- **L1 quality (N_PER 6→9):** lifts on-substrate L1 within-sub-cat codon overlap 0.25→0.45 (= numpy-parity 0.42), but the held-out routing is STILL ~0.00 (stacked super-acc 0.00 at N_PER=9, NCOL2=120). So the residual is the L2 pooler, not L1 quality.
- **L2 column count (NCOL2 120→40):** forcing collision raises held-out within-super overlap (0.22 at NCOL2=40) BUT raises cross-super overlap equally → PERMUTED and L1→L2-lesion also route (permuted super-acc 0.44 ≥ stacked 0.39 at N_PER=9/NCOL2=40) → the anti-cheat breaks. Not valid.
- **Selectivity kernel (winner-inactive depression):** OFF raises within-super held-out 0.01→0.07 (7×) but cross 0.00→0.06 too → discrimination diff stays +0.015. Not the sole cause; over-sparsification is part of it but not the fix.
- **L2 selectivity rate / L2 epochs:** flat (grouping 0.082→0.086 across `ld_wi` 0.02–0.08 and epochs 400/800).

## Honest verdict + why this is the right stopping point (not a forced GO)
Per the master directive + the anti-cheat control-validity methodology: BOTH pooler layers' **learning** is genuinely on-substrate (the deliverable the task asked for), and the single-layer fully-spiking pooler already GO'd (EMERGE-40). But the STACKED **held-out-sub-category generalization** — inheriting a superordinate property through a sub-category the pooler never saw grouped — does NOT reproduce on-substrate what numpy achieves, and the surpass round shows this is not a cheap knob (L1 quality, column count, selectivity all fail or break the anti-cheat). Forcing it via NCOL2 collision would defeat the anti-cheat (a shortcut, not a fix). This is an honest characterized boundary of the point-neuron competitive pooler's GENERALIZATION, and the honest negative IS the scientific deliverable (it maps what the on-substrate pooler can/can't do).

## Next-rung research (the genuine path past this boundary)
The residual is that the on-substrate L2 pooler tunes to SEEN members without extending shared columns to a held-out sub-category. The literature-grounded next mechanism is a **cross-sub-category L2-input decorrelation / a stronger competitive-learning generalization rule** (the same family as the PPMI/local-normalization reframe that unlocked the conversation cortex's generalization, and the Cui-Ahmad-Hawkins HTM-SP boosting + SDR-overlap tuning). Concretely: (a) an L2 input representation that preserves cross-sub-category feature overlap (so same-super members share L2-input structure the pooler can group), or (b) a competitive-learning rule whose winners tune to the SHARED (superordinate) features rather than the discriminative (sub-category) ones at L2. This is next-rung research, not a tuning knob — matching EMERGE-40's own honest scope note that the fully-spiking pooler is "a single competitive layer on a controlled task; hierarchical pooling is a follow-on."

## Honest scope
- **3 seeds (42/43/44)**; the boundary is consistent across all three (stacked ≤ 0.08 every seed) so 6-seed is not needed to establish the negative.
- Winner SELECTION is a host top-k over the on-substrate drive (EMERGE-41 has the spiking FS-WTA version).
- `l2lesion` (L1→L2 fixed-random lesion) is a REPORTED secondary (not a gate term), per the anti-cheat control-validity methodology — here it is ~0.53, above stacked, a further symptom that the on-substrate stacked routing is not working (a random L2 pooler in this small space coincidentally routes as often as the "trained" on-substrate one — exactly the boundary).
- Two levels; a 3-level fully-spiking stacked hierarchy is a further follow-on (moot until the 2-level on-substrate generalization is solved).

## Artifacts
`research/runners/_emerge46_spiking_stacked_pooler_derisk.py` (the `OnSubstratePooler` + `SpikingStackedPoolerProbe`), `tests/test_emerge46_spiking_stacked_pooler.py` (3 tests), `research/findings/raw/_emerge46_spiking_stacked_pooler.json`. Prior: `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`, `2026-07-02-emerge40-spiking-htm-sp-kernel-GO.md`, `2026-07-02-anti-cheat-control-validity-methodology.md`.
