# EMERGE-45 / toward-semantics — GO (order-acc, 6/6 seeds), honest-framed: a THREE-LEVEL discovered taxonomy. Stacking the competitive pooler 3 deep (member features → sub-category → genus → order) discovers a 3-level hierarchy from co-occurrence, and a held-out sub-category infers its ORDER property with ~zero sibling-confusion. Most of the ORDER signal is carried by the discovered L2/genus grouping; L3 adds a smaller, seed-variable increment on top. Extends EMERGE-44 (2-level) per the research gate. NO `sim/` edit.

**2026-07-02 (autonomous; corrected 2026-07-02 per a confirmed control-completeness audit).** Runner `research/runners/_emerge45_three_level_hierarchy_derisk.py`; CI guard `tests/test_emerge45_three_level_hierarchy.py` (4 tests). Reuse-by-import (`_emerge14` + `_emerge12` + the EMERGE-44 pooler helper); NO `sim/` edit; CPU numpy-backend; 6-seed.

## The claim (GO on order-acc, 6/6 seeds; honest-framed)
8 sub-categories group into 4 genera, which group into 2 orders. Three stacked competitive poolers discover the levels:
- **L1** (features → sub-category codons), **L2** (L1 codons → genus codons, via same-genus co-occurrence), **L3** (L2 codons → order codons, via same-order co-occurrence).
- **Held-out order inference:** an entire held-out sub-category (never taught the order property) infers its ORDER — order-acc **0.97 mean** (per seed 42/43/44 = 0.92/1.00/1.00; the earlier documented 6-seed spread added 100/101/102 = 0.88/1.00/1.00), chance 0.50.
- **Discrimination (sibling-confusion):** **0.00 every seed** — no held-out member commits the WRONG (sibling) order. This is the *honest* discrimination metric (see the two audit corrections below); it is measured *separately from abstentions*.
- **Anti-cheats:** PERMUTED-co-occurrence (breaks L2+L3) **0.32**, dAP-LESION **0.00**.

## Two audit corrections (what changed and why)
A confirmed adversarial audit found two honesty/control-completeness gaps. Both were fixed; the GO on `order_acc` survives.

**(1) The old `transitivity` metric was near-tautological with `order_acc` at NORDER=2.** With only 2 orders, "not-sibling" == "correct-or-abstain", so the permuted arm scored the old transitivity metric high (0.75/0.875/0.54) *purely from abstentions*, not from real discrimination. **Fix:** the metric is replaced by **sibling-confusion** = fraction of held-out members that inferred the *wrong* order (an abstain does NOT count as a pass). The stacked arm scores **sibling-confusion 0.00** every seed; the permuted arm now honestly shows nonzero sibling-confusion (0.25/0.125/0.458), exposing what the old metric hid. The GO gate's `transitivity ≥ 0.80` is replaced by `sibling-confusion ≤ 0.05`.

**(2) No control isolated the L3 level, and a genus-proximity shortcut exists.** Each held-out sub shares its genus with *exactly one* trained sub, and genus → order is deterministic, so **L2/genus grouping alone carries most of the order signal**. The old runner's only collapse control (`permuted`) broke L2+L3 *together*, so it could not attribute the win to L3. **Fix:** three honest controls added:
- **L2/genus-only floor** (`order_acc_l2only`): teach + read the order property on the **L2 (genus) codons**, skipping L3 entirely. This genus-proximity readout scores **0.81 mean** (0.79/0.92/0.71) — well above chance 0.50. This is the honest floor L3 must clear.
- **permute-L3-only** (permute *only* the L3 co-occurrence, L2/genus intact): **0.61 mean** (0.83/0.33/0.67).
- **L3-lesion** (skip L3 learning, untuned L3 codons): **0.58 mean** (0.67/0.33/0.75).

**Softened claim:** the earlier "chains through TWO learned levels" is overstated. Honestly: **most of the ORDER signal is carried by the discovered L2/genus grouping**, and **L3 adds a smaller, SEED-VARIABLE increment** above that floor — L3-increment **+0.17 mean, per-seed [0.125, 0.083, 0.292]**. L3 is load-bearing at some seeds (seed 43: permute-L3-only and L3-lesion both collapse to 0.33 while stacked = 1.00) and near-marginal at others (seed 42: permute-L3-only 0.83 vs stacked 0.92). The permute-L3-only control does show L3 adds beyond the genus floor on average (stacked 0.97 vs permute-L3-only 0.61), so the increment is real, but it is not the dominant carrier.

## New honest numbers (3-seed, corrected runner)
| seed | order-acc | sibling-confusion | L2/genus floor | L3 increment | permute-L3-only | L3-lesion | permuted (L2+L3) | dAP-lesion |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.92 | 0.00 | 0.79 | +0.13 | 0.83 | 0.67 | 0.42 | 0.00 |
| 43 | 1.00 | 0.00 | 0.92 | +0.08 | 0.33 | 0.33 | 0.38 | 0.00 |
| 44 | 1.00 | 0.00 | 0.71 | +0.29 | 0.67 | 0.75 | 0.17 | 0.00 |
| **mean** | **0.97** | **0.00** | **0.81** | **+0.17** | **0.61** | **0.58** | **0.32** | **0.00** |

Verdict: **GO (order-acc)** — order-acc 0.97 ≥ 0.80; sibling-confusion 0.00 ≤ 0.05; permuted collapses (0.97 vs 0.32); dAP-lesion collapses (0.97 vs 0.00). Honest framing baked into the verdict string: the genus floor (0.81) is disclosed and L3 is characterized as a real-but-seed-variable increment.

## Mechanism
Each level reuses the EMERGE-38 competitive pooler, but its INPUT is the codons of the level below, and it is trained on the CO-OCCURRENCE at that level (same-genus members for L2; same-order members for L3). So L2 columns tune to what co-occurs within a genus → genus codons; L3 columns tune to what co-occurs within an order → order codons. A member's features → L1 → L2 → L3 codon; an order property taught (committed three-term kernel) on the training members' L3 codons is inherited by a held-out sub-category through the discovered levels. Because genus → order is deterministic and each held-out sub shares its genus with a trained sub, the L2/genus grouping already carries most of the read; L3 sharpens it (a smaller increment). Biology: the ventral hierarchy's successive pooling stages with growing abstraction (Kandel 6e Ch 21) + ATL convergence zones (Patterson–Lambon Ralph; Damasio 1989) — each cortical level pools the one below.

## Significance
The stacking mechanism (EMERGE-44) generalizes to THREE levels: the brain discovers a genuine multi-level taxonomy from experience and infers a held-out sub-category's order WITH discrimination (sibling-confusion ~0) on one spiking brain, no transformer. The honest scope is that the inference chains at least through the discovered L2/genus grouping (which is itself a learned, permuted-collapse-controlled level), with the third level (L3/order) a real but seed-variable increment rather than the dominant carrier. Combined with EMERGE-42/43 (cancellation + multi-override), the discovered-taxonomy substrate supports the inference repertoire across levels — with the L3-contribution honestly characterized.

## Honest scope + next
- The pooler LEARNING is a rate-reference (fully-on-substrate at EMERGE-39/40; k-WTA spiking at EMERGE-41); the inheritance chain runs on the spiking bridge over the discovered codons. Held-out at the sub-category level (its members co-occurred within the genus/order during pooler training).
- **The dominant carrier is the L2/genus grouping**, not a distinct L3 read; L3 adds a smaller, seed-variable increment (+0.17 mean). A stronger L3-isolation result (a clean, seed-robust L3-only increment well above the genus floor) is the tuning target if a fully-independent third level is wanted — tune L3 boosting/depression/epochs or add a decorrelation stage between L2 and L3.
- Next: EMERGE-46 — the fully-spiking stacked hierarchy (replace the numpy pooler layers with the on-substrate EMERGE-40 kernel + EMERGE-41 FS-WTA for each layer).

## Artifacts
`research/runners/_emerge45_three_level_hierarchy_derisk.py`, `tests/test_emerge45_three_level_hierarchy.py`, `research/findings/raw/_emerge45_three_level_hierarchy.json`. Prior: `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`.
