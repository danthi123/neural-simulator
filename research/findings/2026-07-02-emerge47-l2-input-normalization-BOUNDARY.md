# EMERGE-47 / toward-semantics — BOUNDARY (honest, 3-seed): L2-INPUT LOCAL NORMALIZATION (the PPMI/divisive-norm family) gives a REAL, CONSISTENT, DATA-DRIVEN lift on the EMERGE-46 stacked-pooler generalization residual — but it does NOT reach the GO threshold ON ITS OWN. The dominant lever is the over-selective winner-inactive DEPRESSION (soft/union pooling = the next rung); normalization is an additive secondary lift on top of it. Surpass-round mechanism value MEASURED, not forced. NO NEW `sim/` edit.

**2026-07-02 (autonomous, ultracode).** Runner `research/runners/_emerge47_l2_input_normalization_derisk.py`; CI guard `tests/test_emerge47_l2_input_normalization.py` (3 tests, pinning the MECHANISM facts — data-driven weights, OFF==EMERGE-44 identity, the directional lift — NOT a GO). Reuse-by-import (`_emerge44` numpy pooler + `_emerge46` on-substrate pooler + `_emerge14`/`_emerge12` kernels); CPU numpy-backend; 3-seed (42/43/44). Launched by the EMERGE-46 SURPASS research gate (`2026-07-02-emerge46-boundary-surpass-research-gate.md`) which ranked L2-input LOCAL NORMALIZATION as the cheapest rung. Prior: `2026-07-02-emerge46-spiking-stacked-pooler-BOUNDARY.md`, `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`.

## The decisive cheap experiment first (a finding in itself): the numpy pooler CANNOT be degraded by epochs — the boundary is the DEPRESSION
The task asked to reproduce a FAILING regime where the L2 pooler does not generalize. **Degrading L2 epochs does NOT work**: the EMERGE-44 numpy `_competitive_pool` (winner-inactive depression `POOL_LD=0.02`) generalizes at ANY epoch count — even **1 epoch** gives held-out within-super overlap 0.117 / super-acc 0.97 (swept 1→400). So the on-substrate EMERGE-46 boundary is NOT an under-training artifact.

Root-causing what DOES reproduce it (per EMERGE-46's own note "over-sparsification is part of it"): the **winner-inactive DEPRESSION rate**. At `ld ≥ 0.10` the numpy pooler collapses held-out within-super overlap to ~0.001 (matching the on-substrate ~0.01) while cross stays 0.000 — the pooler tunes tightly to SEEN members' discriminative features and does NOT extend the shared columns to a held-out sub-category. **THIS is the faithful failing regime** (over-selective depression), and it is exactly what the on-substrate pooler's float32/kernel dynamics land in even at its committed `ld_wi=0.02`. EMERGE-47 uses `POOL_LD_STRONG=0.15` as the numpy stand-in for the on-substrate regime.

## The mechanism (concrete form)
The L2 pooler's input is the L1 CODONS — sparse binary index-sets over `[0, NCOL1)`. Before the L2 competitive pooler learns, LOCAL-NORMALIZE the L1 drive by each L1-column's MARGINAL firing frequency across the L2 co-occurrence corpus: an IDF / smoothed-PPMI-marginal weight `w[j] = log((1+N)/(1+df[j]))` (`df[j]` = # co-occurrence samples in which L1 column j is active). Down-weights ubiquitous L1 columns (present in most members → uninformative), up-weights informative SHARED columns. The competitive winner score becomes `(connected_perms @ (x * w))` instead of `(connected_perms @ x)`. The weights are DATA-DRIVEN (computed from the corpus marginals). Biology: divisive normalization (Carandini-Heeger 1999; Kandel 6e Ch 28) + the EMERGE-19 PPMI reframe. `OFF (in_weights=None)` is byte-identical to the EMERGE-44 pooler (a clean A/B; CI-pinned).

## NUMPY DIAGNOSTIC (3-seed, the faithful over-selective failing regime `ld=0.15`)
| arm | held-within | held-cross | super-acc | L2-group |
|---|---|---|---|---|
| **OFF (== EMERGE-46 boundary)** | 0.003 | 0.000 | 0.06 | +0.09 |
| **ON (L2-input normalization)** | **0.009** | 0.000 | **0.11** | +0.09 |
| ON_permuted_cooc | 0.000 | 0.000 | 0.00 | +0.08 |
| ON_permuted_stats (shuffled column ids) | 0.000 | 0.000 | 0.00 | +0.09 |
| ON_dap_lesion | 0.009 | 0.000 | 0.00 | +0.09 |

**Normalization LIFTS the exact routing quantity ~3× (held-within 0.003→0.009) WITHOUT raising cross-super (0.000→0.000)**, and the lift is DATA-DRIVEN: the permuted-stats control (normalization weights computed from SHUFFLED L1-column identities) collapses it back to 0.000 / super-acc 0.00. Permuted-co-occurrence and dAP-lesion also collapse. So the mechanism is REAL and correctly-controlled — but the super-acc lift (0.06→0.11) is well below the 0.80 GO.

Seed 42 alone is more favorable (OFF within 0.005 / acc 0.08 → ON within 0.028 / acc 0.33, a 5.6× lift); seeds 43/44 are harder, so the 3-seed mean is the modest 0.003→0.009 / 0.06→0.11.

## ON-SUBSTRATE PORT (EMERGE-46's `OnSubstratePooler` + L2-input normalization, 3-seed)
The numpy lift TRANSFERS to the real substrate (seed 42: OFF within 0.003 / super-acc 0.00 → ON within 0.010 / super-acc **0.33** — a real lift from EMERGE-46's 0.00, cross stays 0.000). 3-seed aggregate:

| on-substrate arm | held-within | super-acc (per-seed) | L2-group |
|---|---|---|---|
| **stacked_norm** | 0.011 | **0.25** (0.33/0.25/0.17) | +0.07 |
| permuted_cooc | 0.006 | 0.08 (0.00/0.08/0.17) | +0.07 |
| permuted_stats (shuffled column ids) | 0.005 | **0.03** (0.00/0.00/0.08) | +0.08 |
| dap_lesion | 0.011 | 0.00 (0.00/0.00/0.00) | +0.07 |

**Normalization lifts the on-substrate super-acc from EMERGE-46's 0.03 to 0.25 (~8×)** — a real, consistent (all 3 seeds up: 0.33/0.25/0.17), on-substrate lift, and DATA-DRIVEN: the permuted-stats control (normalization weights from SHUFFLED L1-column identities) collapses it back to 0.03, permuted-cooc to 0.08, dAP-lesion to 0.00. **BUT it does NOT reach the gated 0.80 GO** (and stacked 0.25 is not ≥ permuted-cooc 0.08 + 0.25, nor ≥ dAP-lesion + 0.30 — the lift, while real and better-than-controls, is too small to clear the strict gate). Interestingly the held-within routing overlap barely moves (0.010→0.011) while super-acc rises 8×: the normalization sharpens the RELATIVE drive toward the correct superordinate property cell even when the absolute L2-codon overlap stays small.

## The KEY sweep — normalization is a SECONDARY lift; the DOMINANT lever is soft/union pooling (rung 2)
Sweeping the depression `ld` (over-selective → soft) × normalize ON/OFF (numpy, 3-seed):

| ld | OFF super-acc | ON super-acc | ON held-within | ON held-cross |
|---|---|---|---|---|
| 0.15 (over-selective) | 0.06 | 0.11 | 0.009 | 0.000 |
| 0.08 | 0.03 | 0.08 | 0.015 | 0.000 |
| 0.05 | 0.31 | **0.56** | 0.034 | 0.000 |
| 0.03 | 0.78 | 0.72 | 0.068 | 0.000 |
| 0.02 (default soft) | 0.97 | 1.00 | 0.113 | 0.000 |

Two clean facts: (1) **softening the depression alone recovers the regime** (OFF 0.06 at ld=0.15 → 0.97 at ld=0.02) — the boundary was PRIMARILY over-selective depression, i.e. the fix family is soft/union pooling. (2) **normalization is an additive secondary lift** on top: it helps most in the mid regime (ld=0.05: 0.31→0.56, +0.25) and never hurts by much, but it does not, on its own, carry an over-selective pooler to GO. This precisely matches the surpass gate's ranking: rung 1 (L2-input normalization) is a real but partial contributor; **rung 2 (soft/union pooling — HTM temporal pooler / HMAX soft-max; relax the winner-inactive depression so multiple columns strengthen on similar inputs) is the dominant lever**.

## Honest verdict — BOUNDARY for the *isolated* hypothesis, with the precise next rung identified
**L2-input local normalization is a REAL, consistent, data-driven mechanism that lifts the held-out generalization residual (~3-5× the exact routing overlap; on-substrate super-acc 0.00→0.33 at seed 42), correctly controlled (permuted-cooc + permuted-stats + dAP-lesion all collapse) — but it does NOT surpass the EMERGE-46 boundary to a GO on its own.** The dominant cause of the boundary is over-selective winner-inactive depression, and the dominant fix is SOFT/UNION pooling (rung 2). Per the master directive + the anti-cheat control-validity methodology, this is an honest characterized boundary of the *isolated* normalization hypothesis, and the honest negative IS the deliverable — it maps that normalization *contributes* but soft-pooling is the load-bearing lever. NOT forced to a GO (super-acc 0.11-0.33 ≪ 0.80).

## Next rung (the genuine surpass path)
Per the surpass gate: **combine L2-input normalization (rung 1, this de-risk — keep it, it's an additive lift) with SOFT/UNION pooling (rung 2)** — relax the winner-inactive depression so multiple columns strengthen on similar inputs (HTM temporal pooler, Hawkins-Ahmad 2016; HMAX soft-max, Serre 2005; the selectivity `ld_wi` becomes a soft threshold, ~10-line kernel edit) — OR the Földiák (1991) trace / temporal-continuity rule (a slow eligibility trace pooling features that co-occur in TIME, needs grouped/curriculum presentation). The mid-regime numpy evidence (ld=0.05: normalization lifts 0.31→0.56) suggests normalization + a modest softening together may reach GO — that is the EMERGE-48 candidate.

## Anti-cheats (all correctly-behaving)
- **Held out ENTIRE sub-categories {2,5}** (as EMERGE-44) — a held-out member can inherit ONLY via the L2-discovered grouping.
- **PERMUTED-co-occurrence** collapses (super-acc 0.00) — no superordinate structure to normalize toward.
- **PERMUTED-STATS** (normalization weights from SHUFFLED L1-column identities) collapses the lift (within 0.009→0.000, super-acc 0.11→0.00) — proves the normalization statistics are LEARNED from the data, not hard-wired to the task.
- **dAP-lesion** collapses (super-acc 0.00) — the coincidence-plateau read is load-bearing.
- Cross-super overlap stays 0.000 in the ON arm — the lift does NOT come from indiscriminate collision (which would break the anti-cheat, as EMERGE-46's NCOL2-collision shortcut did).
- Gate is super-acc ≥ 0.80 on permuted + dAP-lesion (NOT on l2lesion, per the audit).

## Honest scope
- **3 seeds (42/43/44)**; the boundary (normalization helps but doesn't reach GO alone) is consistent across all three.
- The numpy failing regime uses strong depression (`ld=0.15`) as a faithful stand-in for the on-substrate over-sparsification; the on-substrate port uses EMERGE-46's committed `OnSubstratePooler` (its intrinsic dynamics land in the failing regime at the committed `ld_wi=0.02`).
- The normalization steers WHICH columns win (the drive is `x·w`); the Hebbian potentiation target is the unchanged binary active mask, so OFF == EMERGE-44 exactly.
- Winner SELECTION is a host top-k over the on-substrate drive (EMERGE-41 has the spiking FS-WTA version).
- Two levels; the soft/union-pooling combination (EMERGE-48) is the genuine next rung.

## Artifacts
`research/runners/_emerge47_l2_input_normalization_derisk.py` (`compute_idf_weights`, `_competitive_pool_normalized`, `NormalizedStackedPoolerProbe`, the on-substrate `NormalizedOnSubstratePooler`/`NormalizedSpikingStackedPoolerProbe`; `--demo` / `--numpy-diagnostic` / `--onsubstrate`), `tests/test_emerge47_l2_input_normalization.py` (3 tests), `research/findings/raw/_emerge47_l2_input_normalization.json`. Prior: `2026-07-02-emerge46-boundary-surpass-research-gate.md`, `2026-07-02-emerge46-spiking-stacked-pooler-BOUNDARY.md`, `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`, `2026-07-02-anti-cheat-control-validity-methodology.md`.
