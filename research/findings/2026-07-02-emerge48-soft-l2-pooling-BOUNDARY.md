# EMERGE-48 / toward-semantics — BOUNDARY (honest, 3-seed): SOFT / UNION L2 POOLING (lowering the winner-inactive depression `ld_wi`) is the DOMINANT lever in NUMPY (super-acc 0.06 → 1.00) but DOES NOT TRANSFER to the on-substrate competitive pooler — the on-substrate pooler has NO soft-pooling WINDOW: mild-soft `ld_wi` stays over-selective (held-out within ~0.01, super-acc 0.03 = the EMERGE-46 boundary) and `ld_wi=0` collapses to INDISCRIMINATE COLLISION (held-out within ≈ cross ≈ 0.07, super-acc 0.53 ≈ chance). The genuine residual is precisely ISOLATED (not L1 codons, not the depression rate) to the on-substrate competitive pooler's lack of graded soft-pooling dynamics. The honest negative IS the deliverable; the next rung is the Földiák (1991) trace / temporal-continuity rule. NO NEW `sim/` edit.

**2026-07-02 (autonomous, ultracode).** Runner `research/runners/_emerge48_soft_l2_pooling_derisk.py`; CI guard `tests/test_emerge48_soft_l2_pooling.py` (3 tests, pinning the NUMPY MECHANISM facts — over-selective fails, soft recovers WITHOUT collision, soft keeps cross near-zero — NOT an on-substrate GO). Reuse-by-import (`_emerge44` task + numpy pooler, `_emerge46` `OnSubstratePooler` + bridge, `_emerge47` `compute_idf_weights`/`_competitive_pool_normalized`, `_emerge14`/`_emerge12` kernels); CPU numpy-backend; 3-seed (42/43/44). Launched by the EMERGE-47 BOUNDARY's identified next rung (soft/union pooling, the dominant lever). Prior: `2026-07-02-emerge47-l2-input-normalization-BOUNDARY.md`, `2026-07-02-emerge46-spiking-stacked-pooler-BOUNDARY.md`, `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`.

## The mechanism (concrete form)
The EMERGE-46 on-substrate L2 competitive pooler is OVER-SELECTIVE (EMERGE-47's root-cause): its winner-inactive DEPRESSION (`ld_wi`, the committed `fused_htm_winner_inactive_depression`'s `lam_dep_wi`) tunes each L2 column TIGHTLY to the SEEN members' discriminative features, so a HELD-OUT sub-category's L1 codon does not drive the shared L2 columns and inheritance collapses. SOFT/UNION POOLING (HTM temporal pooler, Hawkins-Ahmad 2016; HMAX soft-max, Serre-Poggio 2005; Kandel Ch 17 V1 complex cells) says: LOWER `ld_wi` so multiple L2 columns strengthen on similar inputs → same-superordinate members (incl. a held-out sub-category) SHARE L2 columns → inheritance routes. L1 stays at the normal discriminative `ld_wi`. The winner-inactive kernel ALREADY takes `lam_dep_wi` as a soft rate (0 = off), so a lower L2 rate needs NO NEW `sim/` edit.

## NUMPY SWEEP — softening `ld_wi` IS the dominant lever (confirms EMERGE-47), a CLEAN window
Sweeping the L2 winner-inactive depression `ld` (over-selective → soft/union), L1 at the normal rate, held-out ENTIRE sub-categories {2,5} (numpy, 3-seed 42/43/44):

| L2 `ld` | held-within | held-cross | super-acc | L2-group |
|---|---|---|---|---|
| 0.15 (over-selective = EMERGE-46 regime) | 0.003 | 0.000 | 0.06 | +0.09 |
| 0.05 | 0.037 | 0.000 | 0.31 | +0.12 |
| **0.02** | 0.123 | 0.000 | **0.97** | +0.19 |
| **0.01** | 0.259 | 0.000 | **1.00** | +0.34 |
| **0.005** | **0.474** | **0.000** | **1.00** | +0.51 |
| 0.00 (fully off) | 0.760 | 0.076 | 1.00 | +0.73 |

**NUMPY GO with a CLEAN window: at `ld` = 0.005–0.02 the held-out within-super overlap is 0.12–0.47 while cross-super stays 0.000 — GENERALIZATION, not collision — and super-acc = 0.97–1.00.** Only at `ld=0.0` does cross-super start to rise (0.076), i.e. the numpy pooler has a wide soft-pooling window where discrimination is preserved. This is the dominant lever EMERGE-47 identified; the numpy mechanism is REAL and clean.

## ON-SUBSTRATE PORT — the numpy recovery DOES NOT TRANSFER (the boundary)
Porting the soft/union `ld_wi` to EMERGE-46's `OnSubstratePooler` (both pooler layers' learning in `cp_connections`, the committed `sim/` kernels), 3-seed:

| on-substrate arm | held-within | held-cross | super-acc (per-seed) | L2-group |
|---|---|---|---|---|
| **stacked_soft (ld_wi=0.005)** | 0.007 | 0.002 | **0.03** (0.08/0.00/0.00) | +0.08 |
| permuted_cooc | 0.006 | 0.008 | 0.08 (0.08/0.00/0.17) | +0.08 |
| dAP-lesion | 0.007 | 0.002 | 0.00 | +0.08 |
| l2lesion (untrained random L2, reported-not-gated) | 0.078 | 0.065 | 0.53 (0.50/0.58/0.50) | +0.05 |

Direct sweep of the on-substrate L2 `ld_wi` (separate diagnostic, 3-seed):

| on-substrate L2 `ld_wi` | held-within | held-cross | super-acc |
|---|---|---|---|
| 0.02 | 0.010 | 0.002 | 0.03 |
| 0.01 | 0.009 | 0.002 | 0.03 |
| 0.005 | 0.013 | 0.000 | 0.03 |
| **0.00 (fully off)** | **0.078** | **0.065** | **0.53** |
| 0.00 + normalization | 0.062 | 0.044 | 0.47 |

**There is NO soft-pooling window on-substrate.** Mild-soft `ld_wi` (0.005–0.02) stays in the OVER-SELECTIVE regime (held-out within ~0.01, super-acc 0.03 — identical to the EMERGE-46 boundary; lowering `ld_wi` barely moves it). `ld_wi=0` (fully off) DOES raise the held-out within-super overlap (0.01 → 0.078), but it raises **cross-super EQUALLY (0.065)** — super-acc 0.53 ≈ chance 0.50, within ≈ cross → **INDISCRIMINATE COLLISION, not generalization** (the exact shortcut the de-risk guarded against, and it matches the untrained-random `l2lesion` collision signature within 0.078 / cross 0.065 / acc 0.53). Adding EMERGE-47 normalization on top does not help (still collision). So the on-substrate competitive pooler jumps STRAIGHT from over-selective (no pooling) to collapsed (indiscriminate collision), skipping numpy's clean soft-pooling window.

## The genuine residual, PRECISELY ISOLATED (the deliverable)
An L1-vs-L2 isolation diagnostic (feed the on-substrate L2 pooler the GOOD numpy L1 codons; cross the L2-pooler backend):

| L1 source (within-subcat overlap) | numpy L2, ld=0.005 (held-within/cross) | on-substrate L2, ld=0.005 (held-within/cross) |
|---|---|---|
| numpy-L1 (0.422) | **0.483 / 0.000** | 0.012 / 0.000 |
| on-substrate-L1 (0.254) | **0.351 / 0.000** | 0.013 / 0.000 |

**The L1 codons are NOT the bottleneck** — fed the SAME good numpy L1 codons, the numpy L2 recovers (held-within 0.483, cross 0.000) while the on-substrate L2 does NOT (held-within 0.012). **The residual is the on-substrate competitive pooler's LEARNED REPRESENTATION**, not the depression rate (swept 0.0–0.05) and not the L1 codon quality (numpy-parity codons fail identically). The on-substrate pooler (potentiate active→winner via `fused_htm_permanence_update` lp=0.05 over 400 epochs + hard `perm>0.5` connected-threshold read) lacks the graded real-valued permanence dynamics that give numpy's `(W>0.5)@x` its soft-pooling window: on-substrate, potentiation drives winner permanences to a sharp connected/not-connected split, so softening the (separate) depression term cannot create shared-but-discriminative column tuning — either the columns stay discriminative (over-selective) or the connectivity floods (collision).

## Honest verdict — BOUNDARY (the numpy dominant lever does NOT transfer on-substrate)
**Soft/union pooling (lowering `ld_wi`) is confirmed the DOMINANT lever IN NUMPY (super-acc 0.06 → 1.00, a clean generalization window with cross-super 0.000), but it DOES NOT SURPASS the EMERGE-46 boundary ON-SUBSTRATE**: the on-substrate competitive pooler has no soft-pooling window — mild-soft stays over-selective (super-acc 0.03 = the boundary) and fully-off collapses to indiscriminate collision (within ≈ cross, super-acc 0.53 ≈ chance). GATE misses: super-acc 0.03 < 0.80; L2-group +0.08 < 0.15; not ≥ permuted (0.08) + 0.25; not ≥ dAP-lesion (0.00) + 0.30; NO within>cross discrimination (0.007 vs 0.002 = collision). Per the master directive + the anti-cheat control-validity methodology, this is an honest characterized boundary — **the honest negative IS the deliverable**: it maps that the on-substrate point-neuron competitive pooler's generalization residual is NOT a depression-rate knob (the numpy dominant lever) but a deeper representation-dynamics limit (no graded soft-pooling window). NOT forced to a GO (super-acc 0.03 ≪ 0.80; the only arm that raises overlap breaks discrimination).

## Next rung (the genuine surpass path)
The on-substrate pooler needs a soft-pooling mechanism that raises WITHIN-super overlap WITHOUT flooding cross-super — which lowering the depression rate cannot do. Per the surpass gate, the next rung is the **Földiák (1991) trace / temporal-continuity rule**: a slow eligibility trace that pools features co-occurring in TIME (present same-superordinate members in temporally-contiguous bouts; the trace binds their L1 codons into shared L2 columns), which creates the shared-but-discriminative tuning structurally rather than by relaxing selectivity. This needs GROUPED / curriculum presentation (a training-protocol change, not just a rate knob) and is the EMERGE-49 candidate. A parallel option: a GRADED on-substrate read (soften the `perm>0.5` connected-threshold to a graded contribution) so sub-threshold shared permanences contribute, reproducing numpy's graded window — but that touches the read, worth a scoped check.

## Anti-cheats (all correctly-behaving)
- **Held out ENTIRE sub-categories {2,5}** (as EMERGE-44/46/47) — a held-out member can inherit ONLY via the L2-discovered grouping.
- **PERMUTED-co-occurrence** on-substrate super-acc 0.08 (no superordinate structure) — the stacked arm does not clear it (0.03 vs 0.08), correctly reporting the boundary.
- **dAP-lesion** super-acc 0.00 — the coincidence-plateau read is load-bearing.
- **The shortcut guard FIRED CORRECTLY**: at `ld_wi=0` the held-out overlap rises but within ≈ cross (0.078 vs 0.065), so the verdict logic reports COLLISION (not generalization) and refuses the GO — it did NOT let indiscriminate collision masquerade as inheritance.
- **l2lesion** (untrained random L2) within 0.078 / cross 0.065 / super-acc 0.53 — the collision-at-chance signature; REPORTED-not-gated (a fixed-random control, per the anti-cheat control-validity methodology).

## Honest scope
- **3 seeds (42/43/44)**; the boundary (numpy recovers, on-substrate does not) is consistent across all three (on-substrate stacked super-acc 0.08/0.00/0.00).
- The numpy mechanism (the CI-pinned facts) is REAL and clean — this de-risk's negative is specifically the ON-SUBSTRATE TRANSFER of the soft/union lever, isolated to the competitive pooler's representation dynamics.
- Winner SELECTION is a host top-k over the on-substrate drive (EMERGE-41 has the spiking FS-WTA version).
- NO NEW `sim/` edit (the L2 `ld_wi` is a LOWER value of the already-committed kernel's `lam_dep_wi` argument; every existing `sim/` path byte-unchanged).

## Artifacts
`research/runners/_emerge48_soft_l2_pooling_derisk.py` (`SoftL2NumpyProbe`, `_numpy_sweep`, the on-substrate `SoftNormalizedOnSubstratePooler`/`SoftL2SpikingStackedPoolerProbe` via `_build_onsubstrate_probe`, `_onsubstrate_run`; `--demo` / `--numpy-sweep` / `--onsubstrate` / `--l2-ld` / `--normalize`), `tests/test_emerge48_soft_l2_pooling.py` (3 tests), `research/findings/raw/_emerge48_soft_l2_pooling.json`. Prior: `2026-07-02-emerge47-l2-input-normalization-BOUNDARY.md`, `2026-07-02-emerge46-spiking-stacked-pooler-BOUNDARY.md`, `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`, `2026-07-02-anti-cheat-control-validity-methodology.md`.
